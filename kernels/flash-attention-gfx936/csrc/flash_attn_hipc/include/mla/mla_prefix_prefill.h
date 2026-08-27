#include "numeric_types.h"
#include "intrinsic.h"
#include "wait.h"
#include "flash.h"
using namespace flash;



template<int WARP_M, int kHeadDimVSplit, typename ElementAccum>
__forceinline__ __device__ void mla_prefix_prefill_initialize(
    ElementAccum scores_max[WARP_M / 16],
    ElementAccum scores_sum[WARP_M / 16],
    vec4_Accum<ElementAccum> acc_o[WARP_M / 16][kHeadDimVSplit / 16]
) {
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        scores_max[m_idx] = -INFINITY;
        scores_sum[m_idx] = 0.f;
    }
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        #pragma unroll
        for (int pv_tile = 0; pv_tile < kHeadDimVSplit / 16; ++pv_tile) {
            acc_o[m_idx][pv_tile].b64[0] = 0x0;
            acc_o[m_idx][pv_tile].b64[1] = 0x0;
        }
    }
}



template<int kBlockM, int WARP_M, int WARP_NUM, typename Element>
__forceinline__ __device__ void mla_prefix_prefill_fetch_q_to_vgpr(
    union_vec4_f16x2<Element> qv_regs[WARP_M / 16][8],
    union_vec4_f16x2<Element> q_regs[WARP_M / 16],
    Element* qv_ptr,
    Element* q_ptr,
    int m_block,
    int warp_id_row,
    int warp_id_col,
    int lane_id,
    int qv_row_stride,
    int q_row_stride,
    int actual_seqlen_q
) {
    constexpr bool IS_8_WAVES = WARP_NUM == 8;
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        #pragma unroll
        for (int load_loop = 0; load_loop < 8; ++load_loop) {
            int qv_row = min(actual_seqlen_q - 1 - m_block * kBlockM, m_idx * (IS_8_WAVES ? 64: WARP_M) + warp_id_row * 16 + (lane_id & 15));
            int qv_col = (lane_id >> 4) * 8 + warp_id_col * 32 + load_loop * 64;
            int qv_buffer_offset = qv_row * qv_row_stride + qv_col;
            qv_regs[m_idx][load_loop] = *(union_vec4_f16x2<Element>*)(qv_ptr + qv_buffer_offset);
        }
    }
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        int q_row = min(actual_seqlen_q - 1 - m_block * kBlockM, m_idx * (IS_8_WAVES ? 64: WARP_M) + warp_id_row * 16 + (lane_id & 15));
        int q_col = (lane_id >> 4) * 8 + warp_id_col * 32;
        int q_buffer_offset = q_row * q_row_stride + q_col;
        q_regs[m_idx] = *(union_vec4_f16x2<Element>*)(q_ptr + q_buffer_offset);
    }
}


template<int kBlockN, int WARP_NUM, typename Element>
__forceinline__ __device__ void mla_prefix_prefill_prefetch_k_rope_to_lds(
    Element* k_rope_lds,
    vec4_uint k_buffer,
    int warp_id,
    int lane_id,
    int k_row_stride,
    int seqlen_kv_limit
) {
    if constexpr (WARP_NUM == 8) {
        int warp_id_row = warp_id & 3;
        int warp_id_col = warp_id >> 2;
        #pragma unroll
        for (int load_loop = 0; load_loop < 2; ++load_loop) {
            int k_row = min(seqlen_kv_limit - 1, load_loop * 64 + warp_id_row * 16 + (lane_id >> 2));
            int k_col = warp_id_col * 32 + (lane_id & 3) * 8;
            int k_buffer_offset = k_row * k_row_stride + k_col;
            int lds_write_offset = load_loop * WARP_NUM * 16 * 32 + warp_id * 16 * 32; // 8 * 4 * 16 * 32 * sizeof(fp16) = 32KB
            safe_inline_buffer_load_dwordx4_lds<Element, 1>(k_rope_lds, k_buffer, lds_write_offset, 0, k_buffer_offset);
        }
    } else if constexpr (WARP_NUM == 4) {
        constexpr int K_LOAD_REQUESTS = kBlockN / (16 * 2);
        int warp_id_row = warp_id >> 1;
        int warp_id_col = warp_id & 1;
        #pragma unroll
        for (int load_loop = 0; load_loop < K_LOAD_REQUESTS; ++load_loop) {
            int k_row = min(seqlen_kv_limit - 1, load_loop * 32 + warp_id_row * 16 + (lane_id >> 2));
            int k_col = warp_id_col * 32 + (lane_id & 3) * 8;
            int k_buffer_offset = k_row * k_row_stride + k_col;
            int lds_write_offset = load_loop * WARP_NUM * 16 * 32 + warp_id * 16 * 32; // 4 * 4 * 16 * 32 * sizeof(fp16) = 16KB
            safe_inline_buffer_load_dwordx4_lds<Element, 1>(k_rope_lds, k_buffer, lds_write_offset, 0, k_buffer_offset);
        }
    }
}


template<int kBlockN, int WARP_M, int WARP_N, int WARP_NUM, typename Element, typename ElementAccum>
__forceinline__ __device__ void mla_prefix_prefill_compute_fwd_qk_rope(
        vec4_Accum<ElementAccum> s_reg[WARP_M / 16][(kBlockN / 16)],
        union_vec4_f16x2<Element> q_regs[WARP_M / 16],
        vec4_uint k_buffer,
        Element* k_rope_lds,
        int warp_id,
        int lane_id,
        int k_row_stride,
        int seqlen_kv_limit) {
    if constexpr (WARP_NUM == 8) {
        // mla_prefetch_k_rope_to_lds<kBlockN, Element>(k_rope_lds, k_buffer, warp_id, lane_id, k_row_stride, seqlen_kv_limit);
        wait_buffer_data_arrived<true>(0);
        int warp_id_col = warp_id >> 2;
        union_vec4_f16x2<Element> k_regs[kBlockN / 16];
        #pragma unroll
        for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
            int lds_wave_offset = (n_loop >> 2) * 8 * 16 * 32 + (n_loop & 3) * 16 * 32 + warp_id_col * 4 * 16 * 32;
            int lds_tx_offset = (lane_id & 15) * 32 + (lane_id >> 4) * 8;
            inlineasm_ds_read_b128(reinterpret_cast<size_t>(k_rope_lds + lds_wave_offset + lds_tx_offset), k_regs[n_loop]);
        }
        #pragma unroll
        for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
            wait_lds_data_arrived<false/*sync*/>(kBlockN / 16 - n_loop - 1);
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                s_reg[m_idx][n_loop].f32 = mmac<Element, ElementAccum>(q_regs[m_idx].f16x4[0], k_regs[n_loop].f16x4[0], s_reg[m_idx][n_loop].f32);
                s_reg[m_idx][n_loop].f32 = mmac<Element, ElementAccum>(q_regs[m_idx].f16x4[1], k_regs[n_loop].f16x4[1], s_reg[m_idx][n_loop].f32);
            }
        }
        __syncthreads();
    } else if constexpr (WARP_NUM == 4) {
        // mla_prefetch_k_rope_to_lds<kBlockN, Element>(k_rope_lds, k_buffer, warp_id, lane_id, k_row_stride, seqlen_kv_limit);
        wait_buffer_data_arrived<true>(0);
        int warp_id_col = warp_id & 1;
        union_vec4_f16x2<Element> k_regs[kBlockN / 16];
        #pragma unroll
        for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
            int lds_wave_offset = n_loop * 2 * 16 * 32 + warp_id_col * 16 * 32;
            int lds_tx_offset = (lane_id & 15) * 32 + (lane_id >> 4) * 8;
            inlineasm_ds_read_b128(reinterpret_cast<size_t>(k_rope_lds + lds_wave_offset + lds_tx_offset), k_regs[n_loop]);
        }
        #pragma unroll
        for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
            wait_lds_data_arrived<false/*sync*/>(kBlockN / 16 - n_loop - 1);
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                s_reg[m_idx][n_loop].f32 = mmac<Element, ElementAccum>(q_regs[m_idx].f16x4[0], k_regs[n_loop].f16x4[0], s_reg[m_idx][n_loop].f32);
                s_reg[m_idx][n_loop].f32 = mmac<Element, ElementAccum>(q_regs[m_idx].f16x4[1], k_regs[n_loop].f16x4[1], s_reg[m_idx][n_loop].f32);
            }
        }
        __syncthreads();
    }
}



template<int kBlockN, int WARP_NUM, typename Element>
__forceinline__ __device__ void mla_prefix_prefill_prefetch_k_nope_to_lds(
    Element* v_lds,
    vec4_uint v_buffer,
    int warp_id,
    int lane_id,
    int v_row_stride,
    int seqlen_kv_limit
) {
    if constexpr (WARP_NUM == 8) {
        constexpr int PREFETCH_K_BLOCKS = 2;
        constexpr int K_LOAD_REQUESTS   = kBlockN / (16 * 4); // 16 * 4 = 64
        int warp_id_row = warp_id & 3;
        int warp_id_col = warp_id >> 2;
        #pragma unroll
        for (int load_id = 0; load_id < PREFETCH_K_BLOCKS; load_id += 2) {
            #pragma unroll
            for (int depth = 0; depth < 2; ++depth) {
                #pragma unroll
                for (int load_loop = 0; load_loop < K_LOAD_REQUESTS; ++load_loop) {
                    int k_row = min(seqlen_kv_limit - 1, load_loop * 64 + warp_id_row * 16 + (lane_id >> 2));
                    int k_col = (load_id + depth) * 64 +  warp_id_col * 32 + (lane_id & 3) * 8;
                    int k_buffer_offset = k_row * v_row_stride + k_col;
                    int lds_write_offset = depth * K_LOAD_REQUESTS * WARP_NUM * 16 * 32 + load_loop * WARP_NUM * 16 * 32 + warp_id * 16 * 32; // 2 * 2 * 8 * 16 * 32 * sizeof(fp16) = 32KB
                    safe_inline_buffer_load_dwordx4_lds<Element, 1>(v_lds, v_buffer, lds_write_offset, 0, k_buffer_offset);
                }
            }
        }
    } else if constexpr (WARP_NUM == 4) {
        __syncthreads();
        constexpr int K_LOAD_REQUESTS = kBlockN / (16 * 2);
        int warp_id_row = warp_id >> 1;
        int warp_id_col = warp_id & 1;
        int stage_id = 0;
        constexpr int load_id = 0;
        #pragma unroll
        for (int load_loop = 0; load_loop < K_LOAD_REQUESTS; ++load_loop) {
            int k_row = min(seqlen_kv_limit - 1, load_loop * 32 + warp_id_row * 16 + (lane_id >> 2));
            int k_col = load_id * 64 +  warp_id_col * 32 + (lane_id & 3) * 8;
            int k_buffer_offset = k_row * v_row_stride + k_col;
            int lds_write_offset = stage_id * K_LOAD_REQUESTS * WARP_NUM * 16 * 32 + load_loop * WARP_NUM * 16 * 32 + warp_id * 16 * 32; // 4 * 4 * 16 * 32 * sizeof(fp16) = 16KB
            safe_inline_buffer_load_dwordx4_lds<Element, 1>(v_lds, v_buffer, lds_write_offset, 0, k_buffer_offset);
        }
    }
}




template<int kBlockN, int WARP_M, int WARP_N, int WARP_NUM, typename Element, typename ElementAccum>
__forceinline__ __device__ void mla_prefix_prefill_compute_fwd_qk_nope(
        vec4_Accum<ElementAccum> s_reg[WARP_M / 16][(kBlockN / 16)],
        union_vec4_f16x2<Element> qv_regs[WARP_M / 16][8],
        vec4_uint v_buffer,
        Element* v_lds,
        vec4_uint k_buffer,
        Element* k_rope_lds,
        int warp_id,
        int lane_id,
        int v_row_stride,
        int k_row_stride,
        int seqlen_kv_limit) {
    if constexpr (WARP_NUM == 8) {
        constexpr int PREFETCH_K_BLOCKS = 2;
        constexpr int K_LOAD_REQUESTS   = kBlockN / (16 * 4);
        int warp_id_row = warp_id & 3;
        int warp_id_col = warp_id >> 2;
        // prefetch_k_nope_to_lds<Element>(v_lds, v_buffer, warp_id, lane_id, v_row_stride, seqlen_kv_limit);
        #pragma unroll
        for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
            #pragma unroll
            for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
                inline_vgpr4_init_zero(s_reg[m_idx][n_loop]);
            }
        }
        #pragma unroll
        for (int load_id = 0; load_id < PREFETCH_K_BLOCKS; load_id += 2) {
            #pragma unroll
            for (int depth = 0; depth < 2; ++depth) {
                wait_buffer_data_arrived<true>((2 - depth - 1) * K_LOAD_REQUESTS);
                union_vec4_f16x2<Element> k_regs[kBlockN / 16];
                #pragma unroll
                for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
                    int lds_wave_offset = depth * K_LOAD_REQUESTS * WARP_NUM * 16 * 32 + (n_loop >> 2) * WARP_NUM * 16 * 32 + (n_loop & 3) * 16 * 32 + warp_id_col * 4 * 16 * 32;
                    int lds_tx_offset = (lane_id & 15) * 32 + (lane_id >> 4) * 8;
                    inlineasm_ds_read_b128(reinterpret_cast<size_t>(v_lds + lds_wave_offset + lds_tx_offset), k_regs[n_loop]);
                }
                #pragma unroll
                for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
                    wait_lds_data_arrived<false>(kBlockN / 16 - n_loop - 1);
                    #pragma unroll
                    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                        s_reg[m_idx][n_loop].f32 = mmac<Element, ElementAccum>(qv_regs[m_idx][load_id + depth].f16x4[0], k_regs[n_loop].f16x4[0], s_reg[m_idx][n_loop].f32);
                        s_reg[m_idx][n_loop].f32 = mmac<Element, ElementAccum>(qv_regs[m_idx][load_id + depth].f16x4[1], k_regs[n_loop].f16x4[1], s_reg[m_idx][n_loop].f32);
                    }
                }
            }
        }
        asm volatile("s_barrier\n"); // 上面在读 lds, 下面在写 lds, 有数据冲突的隐患

        // 提前预取 k_rope 部分的数据, 注意 lds 部分重叠
        mla_prefix_prefill_prefetch_k_rope_to_lds<kBlockN, WARP_NUM, Element>(k_rope_lds, k_buffer, warp_id, lane_id, k_row_stride, seqlen_kv_limit);

        // 接着做剩下的内容
        if constexpr (true) {
            int stage_id = 0;
            {
                #pragma unroll
                for (int load_loop = 0; load_loop < K_LOAD_REQUESTS; ++load_loop) {
                    int k_row = min(seqlen_kv_limit - 1, load_loop * 64 + warp_id_row * 16 + (lane_id >> 2));
                    int k_col = PREFETCH_K_BLOCKS * 64 +  warp_id_col * 32 + (lane_id & 3) * 8;
                    int k_buffer_offset = k_row * v_row_stride + k_col;
                    int lds_write_offset = stage_id * K_LOAD_REQUESTS * WARP_NUM * 16 * 32 + load_loop * WARP_NUM * 16 * 32 + warp_id * 16 * 32;
                    safe_inline_buffer_load_dwordx4_lds<Element, 1>(v_lds, v_buffer, lds_write_offset, 0, k_buffer_offset);
                }
            }

            stage_id ^= 1;
            #pragma unroll
            for (int load_id = PREFETCH_K_BLOCKS + 1; load_id < 8; load_id += 1) {
                #pragma unroll
                for (int load_loop = 0; load_loop < K_LOAD_REQUESTS; ++load_loop) {
                    int k_row = min(seqlen_kv_limit - 1, load_loop * 64 + warp_id_row * 16 + (lane_id >> 2));
                    int k_col = load_id * 64 +  warp_id_col * 32 + (lane_id & 3) * 8;
                    int k_buffer_offset = k_row * v_row_stride + k_col;
                    int lds_write_offset = stage_id * K_LOAD_REQUESTS * WARP_NUM * 16 * 32 + load_loop * WARP_NUM * 16 * 32 + warp_id * 16 * 32;
                    safe_inline_buffer_load_dwordx4_lds<Element, 1>(v_lds, v_buffer, lds_write_offset, 0, k_buffer_offset);
                }
                wait_buffer_data_arrived<true>(K_LOAD_REQUESTS);
                stage_id ^= 1;
                union_vec4_f16x2<Element> k_regs[kBlockN / 16];
                #pragma unroll
                for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
                    int lds_wave_offset = stage_id * K_LOAD_REQUESTS * WARP_NUM * 16 * 32 + (n_loop >> 2) * WARP_NUM * 16 * 32 + (n_loop & 3) * 16 * 32 + warp_id_col * 4 * 16 * 32;
                    int lds_tx_offset = (lane_id & 15) * 32 + (lane_id >> 4) * 8;
                    inlineasm_ds_read_b128(reinterpret_cast<size_t>(v_lds + lds_wave_offset + lds_tx_offset), k_regs[n_loop]);
                }
                #pragma unroll
                for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
                    wait_lds_data_arrived<false>(kBlockN / 16 - n_loop - 1);
                    #pragma unroll
                    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                        s_reg[m_idx][n_loop].f32 = mmac<Element, ElementAccum>(qv_regs[m_idx][load_id - 1].f16x4[0], k_regs[n_loop].f16x4[0], s_reg[m_idx][n_loop].f32);
                        s_reg[m_idx][n_loop].f32 = mmac<Element, ElementAccum>(qv_regs[m_idx][load_id - 1].f16x4[1], k_regs[n_loop].f16x4[1], s_reg[m_idx][n_loop].f32);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                __syncthreads();
                __builtin_amdgcn_sched_barrier(0);
            }
            // rest
            {
                constexpr int load_id = 8;
                wait_buffer_data_arrived<true>(0);
                stage_id ^= 1;
                union_vec4_f16x2<Element> k_regs[kBlockN / 16];
                #pragma unroll
                for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
                    int lds_wave_offset = stage_id * K_LOAD_REQUESTS * 8 * 16 * 32 + (n_loop >> 2) * 8 * 16 * 32 + (n_loop & 3) * 16 * 32 + warp_id_col * 4 * 16 * 32;
                    int lds_tx_offset = (lane_id & 15) * 32 + (lane_id >> 4) * 8;
                    inlineasm_ds_read_b128(reinterpret_cast<size_t>(v_lds + lds_wave_offset + lds_tx_offset), k_regs[n_loop]);
                }
                #pragma unroll
                for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
                    wait_lds_data_arrived<false>(kBlockN / 16 - n_loop - 1);
                    #pragma unroll
                    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                        s_reg[m_idx][n_loop].f32 = mmac<Element, ElementAccum>(qv_regs[m_idx][load_id - 1].f16x4[0], k_regs[n_loop].f16x4[0], s_reg[m_idx][n_loop].f32);
                        s_reg[m_idx][n_loop].f32 = mmac<Element, ElementAccum>(qv_regs[m_idx][load_id - 1].f16x4[1], k_regs[n_loop].f16x4[1], s_reg[m_idx][n_loop].f32);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                __syncthreads();
                __builtin_amdgcn_sched_barrier(0);
            }
        }
    } else if constexpr (WARP_NUM == 4) {
        constexpr int K_LOAD_REQUESTS = kBlockN / (16 * 2);
        int warp_id_row = warp_id >> 1;
        int warp_id_col = warp_id & 1;
        int stage_id = 0;
        // mla_prefetch_k_nope_to_lds<kBlockN, Element>(v_lds, v_buffer, warp_id, lane_id, v_row_stride, seqlen_kv_limit);
        #pragma unroll
        for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
            #pragma unroll
            for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
                inline_vgpr4_init_zero(s_reg[m_idx][n_loop]);
            }
        }
        stage_id ^= 1;
        #pragma unroll
        for (int load_id = 1; load_id < 8; ++load_id) {
            #pragma unroll
            for (int load_loop = 0; load_loop < K_LOAD_REQUESTS; ++load_loop) {
                int k_row = min(seqlen_kv_limit - 1, load_loop * 32 + warp_id_row * 16 + (lane_id >> 2));
                int k_col = load_id * 64 +  warp_id_col * 32 + (lane_id & 3) * 8;
                int k_buffer_offset = k_row * v_row_stride + k_col;
                int lds_write_offset = stage_id * K_LOAD_REQUESTS * WARP_NUM * 16 * 32 + load_loop * WARP_NUM * 16 * 32 + warp_id * 16 * 32; // 4 * 4 * 16 * 32 * sizeof(fp16) = 16KB
                safe_inline_buffer_load_dwordx4_lds<Element, 1>(v_lds, v_buffer, lds_write_offset, 0, k_buffer_offset);
            }
            wait_buffer_data_arrived<true>(K_LOAD_REQUESTS);
            stage_id ^= 1;
            union_vec4_f16x2<Element> k_regs[kBlockN / 16];
            #pragma unroll
            for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
                int lds_wave_offset = stage_id * K_LOAD_REQUESTS * WARP_NUM * 16 * 32 + n_loop * 2 * 16 * 32 + warp_id_col * 16 * 32;
                int lds_tx_offset = (lane_id & 15) * 32 + (lane_id >> 4) * 8;
                inlineasm_ds_read_b128(reinterpret_cast<size_t>(v_lds + lds_wave_offset + lds_tx_offset), k_regs[n_loop]);
            }
            #pragma unroll
            for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
                wait_lds_data_arrived<false>(kBlockN / 16 - n_loop - 1);
                #pragma unroll
                for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                    s_reg[m_idx][n_loop].f32 = mmac<Element, ElementAccum>(qv_regs[m_idx][load_id - 1].f16x4[0], k_regs[n_loop].f16x4[0], s_reg[m_idx][n_loop].f32);
                    s_reg[m_idx][n_loop].f32 = mmac<Element, ElementAccum>(qv_regs[m_idx][load_id - 1].f16x4[1], k_regs[n_loop].f16x4[1], s_reg[m_idx][n_loop].f32);
                }
            }
            __syncthreads();
        }
        // 预取 rope 部分的 K 数据, 注意 k_rope_lds 和 k_lds 的重叠关系
        mla_prefix_prefill_prefetch_k_rope_to_lds<kBlockN, WARP_NUM, Element>(k_rope_lds, k_buffer, warp_id, lane_id, k_row_stride, seqlen_kv_limit);
        {
            int load_id = 8;
            wait_buffer_data_arrived<true>(K_LOAD_REQUESTS);
            stage_id ^= 1;
            union_vec4_f16x2<Element> k_regs[kBlockN / 16];
            #pragma unroll
            for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
                int lds_wave_offset = stage_id * K_LOAD_REQUESTS * WARP_NUM * 16 * 32 + n_loop * 2 * 16 * 32 + warp_id_col * 16 * 32;
                int lds_tx_offset = (lane_id & 15) * 32 + (lane_id >> 4) * 8;
                inlineasm_ds_read_b128(reinterpret_cast<size_t>(v_lds + lds_wave_offset + lds_tx_offset), k_regs[n_loop]);
            }
            #pragma unroll
            for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
                wait_lds_data_arrived<false>(kBlockN / 16 - n_loop - 1);
                // 准备做 mmac
                #pragma unroll
                for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                    s_reg[m_idx][n_loop].f32 = mmac<Element, ElementAccum>(qv_regs[m_idx][load_id - 1].f16x4[0], k_regs[n_loop].f16x4[0], s_reg[m_idx][n_loop].f32);
                    s_reg[m_idx][n_loop].f32 = mmac<Element, ElementAccum>(qv_regs[m_idx][load_id - 1].f16x4[1], k_regs[n_loop].f16x4[1], s_reg[m_idx][n_loop].f32);
                }
            }
            __syncthreads();
        }
    }
}



template<int kBlockN, int WARP_M, int WARP_NUM, typename ElementAccum>
__forceinline__ __device__ void mla_prefix_prefill_combine_s_reg_of_2waves(vec4_Accum<ElementAccum> s_reg[WARP_M / 16][(kBlockN / 16)], ElementAccum* s_reg_lds, int warp_id, int lane_id) {
    constexpr bool IS_8_WAVES = WARP_NUM == 8;
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
            int lds_write_offset = n_loop * WARP_NUM * (64 * 4) + warp_id * 64 * 4 + lane_id * 4;
            *(vec4_fp32*)(s_reg_lds + lds_write_offset) = s_reg[m_idx][n_loop].f32;
        }
        __syncthreads();
        #pragma unroll
        for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
            int warp_id_symmetry = IS_8_WAVES
                ? ((warp_id >= 4) ? warp_id - 4: warp_id + 4)
                : ((warp_id & 1)  ? warp_id - 1: warp_id + 1);
            int lds_load_offset = n_loop * WARP_NUM * (64 * 4) + warp_id_symmetry * 64 * 4 + lane_id * 4;
            vec4_Accum<ElementAccum> symmetry_data = *(vec4_Accum<ElementAccum>*)(s_reg_lds + lds_load_offset);
            s_reg[m_idx][n_loop].u64[0] = __builtin_hcu_pk_add_f32(s_reg[m_idx][n_loop].u64[0], symmetry_data.u64[0]);
            s_reg[m_idx][n_loop].u64[1] = __builtin_hcu_pk_add_f32(s_reg[m_idx][n_loop].u64[1], symmetry_data.u64[1]);
        }
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);
    }
}



template<int kBlockN, int WARP_M, int kHeadDimVSplit, typename ElementAccum>
__forceinline__ __device__ void mla_prefix_prefill_compute_fwd_softmax(
        vec4_Accum<ElementAccum> s_reg[WARP_M / 16][(kBlockN / 16)],
        ElementAccum scores_max[WARP_M / 16],
        ElementAccum scores_sum[WARP_M / 16],
        ElementAccum scale_softmax_log2,
        vec4_Accum<ElementAccum> acc_o[WARP_M / 16][kHeadDimVSplit / 16]) {
    ElementAccum scores_max_cur[WARP_M / 16];
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        scores_max_cur[m_idx] = scores_max[m_idx];
        #pragma unroll
        for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
            #pragma unroll
            for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                scores_max_cur[m_idx] = max(scores_max_cur[m_idx], s_reg[m_idx][n_loop].f32[vec_idx]);
            }
        }
        scores_max_cur[m_idx] = max(scores_max_cur[m_idx], __shfl_xor_tmp(scores_max_cur[m_idx], 32));
        scores_max_cur[m_idx] = max(scores_max_cur[m_idx], __shfl_xor_tmp(scores_max_cur[m_idx], 16));
    }
    // 做 softmax
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        __float2 max_scaled;
        max_scaled[0] = scores_max_cur[m_idx] == -INFINITY ? 0.f: -scores_max_cur[m_idx] * scale_softmax_log2;
        max_scaled[1] = max_scaled[0];
        __float2 scale_softmax_log2_pair;
        scale_softmax_log2_pair[0] = scale_softmax_log2;
        scale_softmax_log2_pair[1] = scale_softmax_log2;
        #pragma unroll
        for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
            s_reg[m_idx][n_loop].u64[0] = __builtin_hcu_pk_fma_f32(s_reg[m_idx][n_loop].u64[0], scale_softmax_log2_pair, max_scaled);
            s_reg[m_idx][n_loop].u64[1] = __builtin_hcu_pk_fma_f32(s_reg[m_idx][n_loop].u64[1], scale_softmax_log2_pair, max_scaled);
            s_reg[m_idx][n_loop].f32[0] = __llvm_exp2_f32(s_reg[m_idx][n_loop].f32[0]);
            s_reg[m_idx][n_loop].f32[1] = __llvm_exp2_f32(s_reg[m_idx][n_loop].f32[1]);
            s_reg[m_idx][n_loop].f32[2] = __llvm_exp2_f32(s_reg[m_idx][n_loop].f32[2]);
            s_reg[m_idx][n_loop].f32[3] = __llvm_exp2_f32(s_reg[m_idx][n_loop].f32[3]);
        }
    }
    // 求和
    ElementAccum scores_sum_cur[WARP_M / 16];
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        __float2 scores_sum_pair;
        scores_sum_pair[0] = 0;
        scores_sum_pair[1] = 0;
        #pragma unroll
        for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
            scores_sum_pair = __builtin_hcu_pk_add_f32(scores_sum_pair, s_reg[m_idx][n_loop].u64[0]);
            scores_sum_pair = __builtin_hcu_pk_add_f32(scores_sum_pair, s_reg[m_idx][n_loop].u64[1]);
        }
        scores_sum_cur[m_idx] = scores_sum_pair[0] + scores_sum_pair[1];
        scores_sum_cur[m_idx] = scores_sum_cur[m_idx] + __shfl_xor(scores_sum_cur[m_idx], 32);
        scores_sum_cur[m_idx] = scores_sum_cur[m_idx] + __shfl_xor(scores_sum_cur[m_idx], 16);
    }
    // 放缩
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        __float2 scores_scale;
        scores_scale[0] = __llvm_exp2_f32(__llvm_fma_f32(scores_max[m_idx], scale_softmax_log2, /*max_scaled[0]*/-scores_max_cur[m_idx] * scale_softmax_log2));
        scores_scale[1] = scores_scale[0];
        scores_sum[m_idx] *= scores_scale[0];
        #pragma unroll
        for (int pv_tile = 0; pv_tile < kHeadDimVSplit; ++pv_tile) {
            acc_o[m_idx][pv_tile].u64[0] = __builtin_hcu_pk_mul_f32(acc_o[m_idx][pv_tile].u64[0], scores_scale);
            acc_o[m_idx][pv_tile].u64[1] = __builtin_hcu_pk_mul_f32(acc_o[m_idx][pv_tile].u64[1], scores_scale);
        }
    }
    // update max/sum
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        scores_sum[m_idx] += scores_sum_cur[m_idx];
        scores_max[m_idx] = scores_max_cur[m_idx];
    }
}



template<int kBlockN, int WARP_M, int WARP_NUM, typename ElementAccum>
__forceinline__ __device__ void mla_prefix_prefill_apply_mask(
        vec4_Accum<ElementAccum> s_reg[WARP_M / 16][(kBlockN / 16)],
        int lane_id,
        const int col_idx_offset_,
        const int max_seqlen_k,
        const int row_idx_offset_,
        const int max_seqlen_q) {
    constexpr bool IS_8_WAVES = WARP_NUM == 8;
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        int col_idx_limit_right = max_seqlen_k;
        #pragma unroll
        for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
            #pragma unroll
            for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                const int col_idx = col_idx_offset_ + n_loop * 16 + vec_idx * 4 + (lane_id >> 4);
                s_reg[m_idx][n_loop].f32[vec_idx] = (col_idx >= col_idx_limit_right) ? -INFINITY: s_reg[m_idx][n_loop].f32[vec_idx];
            }
        }
    }
}



template<int kBlockN, int WARP_M, int WARP_NUM, typename ElementAccum>
__forceinline__ __device__ void mla_prefix_prefill_apply_mtp_mask(
        vec4_Accum<ElementAccum> s_reg[WARP_M / 16][(kBlockN / 16)],
        int lane_id,
        const int col_idx_offset_,
        const int max_seqlen_k,
        const int row_idx_offset_,
        const int max_seqlen_q,
        const int ngroups,
        const int mtp) {
    constexpr bool IS_8_WAVES = WARP_NUM == 8;
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        int row_idx = row_idx_offset_ + (IS_8_WAVES ? m_idx * 64: m_idx * WARP_M) + (lane_id & 15);
        int row_in_mtp = row_idx / ngroups;
        int col_idx_limit_right = min(max_seqlen_k, row_in_mtp + max_seqlen_k - mtp);
        #pragma unroll
        for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
            #pragma unroll
            for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                const int col_idx = col_idx_offset_ + n_loop * 16 + vec_idx * 4 + (lane_id >> 4);
                s_reg[m_idx][n_loop].f32[vec_idx] = (col_idx > col_idx_limit_right) ? -INFINITY: s_reg[m_idx][n_loop].f32[vec_idx];
            }
        }
    }
}



template<int kBlockN, int WARP_M, int WARP_NUM, typename ElementAccum>
__forceinline__ __device__ void mla_prefix_prefill_apply_causal_mask(
        vec4_Accum<ElementAccum> s_reg[WARP_M / 16][(kBlockN / 16)],
        int lane_id,
        const int col_idx_offset_,
        const int max_seqlen_k,
        const int row_idx_offset_,
        const int max_seqlen_q) {
    constexpr bool IS_8_WAVES = WARP_NUM == 8;
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        int row_idx = row_idx_offset_ + (IS_8_WAVES ? m_idx * 64: m_idx * WARP_M) + (lane_id & 15);
        int col_idx_limit_right = min(max_seqlen_k, row_idx + max_seqlen_k - max_seqlen_q);
        #pragma unroll
        for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
            #pragma unroll
            for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                const int col_idx = col_idx_offset_ + n_loop * 16 + vec_idx * 4 + (lane_id >> 4);
                s_reg[m_idx][n_loop].f32[vec_idx] = (col_idx > col_idx_limit_right) ? -INFINITY: s_reg[m_idx][n_loop].f32[vec_idx];
            }
        }
    }
}



template<int kBlockN, int WARP_M, typename ElementAccum, typename Element>
__forceinline__ __device__ void mla_prefix_prefill_cvt_dtype(
        vec4_Accum<ElementAccum> s_reg[WARP_M / 16][kBlockN / 16],
        union_vec2_f16x2<Element> p_reg[WARP_M / 16][kBlockN / 16]) {
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        #pragma unroll
        for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
            #if defined(__gfx938__) || defined(__gfx946__)
                #pragma unroll
                for (int vec_idx = 0; vec_idx < 2; ++vec_idx) {
                    p_reg[m_idx][n_loop].f16x2[vec_idx] = DownCastPair<ElementAccum, Element>(s_reg[m_idx][n_loop].f32x2[vec_idx]);
                }
            #else
                #pragma unroll
                for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                    p_reg[m_idx][n_loop].f16[vec_idx] = DownCast<ElementAccum, Element, false>(s_reg[m_idx][n_loop].f32[vec_idx]);
                }
            #endif
        }
    }
}


template<int PREFETCH_V_BLOCKS, int WARP_NUM, typename Element>
__forceinline__ __device__ void mla_prefix_prefill_prefetch_v_to_lds(
    vec4_uint v_buffer,
    Element* v_lds,
    int v_row_stride,
    int warp_id,
    int lane_id,
    int seqlen_kv_limit
) {
    if constexpr (WARP_NUM == 8) {
        #pragma unroll
        for (int n_loop = 0; n_loop < PREFETCH_V_BLOCKS; n_loop += 2) {
            #pragma unroll
            for (int depth = 0; depth < 2; ++depth) {
                #pragma unroll
                for (int load_loop = 0; load_loop < 2; ++load_loop) {
                    int v_row = min(seqlen_kv_limit - 1, (n_loop + depth) * 16 + (lane_id >> 2));
                    int v_col = load_loop * 8 * 32 + warp_id * 32 + (lane_id & 3) * 8;
                    int v_buffer_offset = v_row * v_row_stride + v_col;
                    int lds_write_offset = depth * 2 * WARP_NUM * 512 + load_loop * WARP_NUM * 512 + warp_id * 512; // 2 * 2 * 8 * 512 * sizeof(half) = 32KB
                    safe_inline_buffer_load_dwordx4_lds<Element, 1>(v_lds, v_buffer, lds_write_offset, 0, v_buffer_offset);
                }
            }
        }
    } else if constexpr (WARP_NUM == 4) {
        __syncthreads();
        constexpr int V_LOAD_REQUESTS = PREFETCH_V_BLOCKS; // union
        int warp_id_col = warp_id & 1;
        int stage_id = 0;
        constexpr int n_loop = 0;
        #pragma unroll
        for (int load_loop = 0; load_loop < V_LOAD_REQUESTS; ++load_loop) {
            int v_row = min(seqlen_kv_limit - 1, n_loop * 16 + (lane_id >> 2));
            int v_col = load_loop * 4 * 32 + warp_id * 32 + (lane_id & 3) * 8;
            int v_buffer_offset = v_row * v_row_stride + v_col;
            int lds_write_offset = stage_id * V_LOAD_REQUESTS * WARP_NUM * 512 + load_loop * WARP_NUM * 512 + warp_id * 512; // 4 * 4 * 512 * sizeof(fp16) = 16KB
            safe_inline_buffer_load_dwordx4_lds<Element, 1>(v_lds, v_buffer, lds_write_offset, 0, v_buffer_offset);
        }
    }
}


template<bool PREFETCH_K, int PREFETCH_V_BLOCKS, int kBlockN, int WARP_M, int WARP_NUM, int kHeadDimVSplit, typename Element, typename ElementAccum>
__forceinline__ __device__ void mla_prefix_prefill_compute_fwd_pv(
        vec4_Accum<ElementAccum> acc_o[WARP_M / 16][kHeadDimVSplit / 16],
        union_vec2_f16x2<Element> p_reg[WARP_M / 16][kBlockN / 16],
        vec4_uint v_buffer,
        Element* v_lds,
        int warp_id,
        int lane_id,
        int v_row_stride,
        int seqlen_kv_limit,
        int v_buffer_offset) {
    if constexpr (WARP_NUM == 8) {
        wait_buffer_data_arrived<true>(0);
        constexpr int V_LOAD_REQUESTS = 2;
        int warp_id_col = warp_id >> 2;
        #pragma unroll
        for (int n_loop = 0; n_loop < PREFETCH_V_BLOCKS; n_loop += 2) {
            #pragma unroll
            for (int depth = 0; depth < 2; ++depth) {
                // lds -> vgprs
                union_vec4_f16x2<Element> v_regs[kHeadDimVSplit / 32];
                #pragma unroll
                for (int v_tile = 0; v_tile < kHeadDimVSplit / 32; ++v_tile) {
                    int v_load_base_offset = depth * V_LOAD_REQUESTS * WARP_NUM * 512 + warp_id_col * 8 * 512 + v_tile * 512;
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        int v_load_offset = v_load_base_offset + i * 8 * 32 + (lane_id >> 4) * 32 + (lane_id & 15) * 2;
                        inline_ds_read2_b32_no_wait_bytes(reinterpret_cast<size_t>(v_lds + v_load_offset), v_regs[v_tile].f16x4[i], 64);
                    }
                }
                // pv mmac
                #pragma unroll
                for (int v_tile = 0; v_tile < kHeadDimVSplit / 32; ++v_tile) {
                    wait_lds_data_arrived<false>((kHeadDimVSplit / 32 - v_tile - 1) * 2);
                    // v interleave into vgprs
                    union_vec4_f16x2<Element> v_composed;
                    v_composed.f16x4[0] = make_vec4_f16<Element>(v_regs[v_tile].f16[0 * 2 + 0], v_regs[v_tile].f16[1 * 2 + 0], v_regs[v_tile].f16[2 * 2 + 0], v_regs[v_tile].f16[3 * 2 + 0]);
                    v_composed.f16x4[1] = make_vec4_f16<Element>(v_regs[v_tile].f16[0 * 2 + 1], v_regs[v_tile].f16[1 * 2 + 1], v_regs[v_tile].f16[2 * 2 + 1], v_regs[v_tile].f16[3 * 2 + 1]);
                    // pv mmac
                    #pragma unroll
                    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                        acc_o[m_idx][v_tile * 2 + 0].f32 = mmac<Element, ElementAccum>(p_reg[m_idx][n_loop + depth].f16x4, v_composed.f16x4[0], acc_o[m_idx][v_tile * 2 + 0].f32);
                        acc_o[m_idx][v_tile * 2 + 1].f32 = mmac<Element, ElementAccum>(p_reg[m_idx][n_loop + depth].f16x4, v_composed.f16x4[1], acc_o[m_idx][v_tile * 2 + 1].f32);
                    }
                }
            }
        }

        asm volatile("s_barrier\n"); // 上面在读 lds, 下面在写 lds, 有数据冲突的隐患

        // 做没预取的部分, 还需要重新取数据
        if constexpr (true) {
            int stage_id = 0;
            {
                #pragma unroll
                for (int load_loop = 0; load_loop < V_LOAD_REQUESTS; ++load_loop) {
                    int v_row = min(seqlen_kv_limit - 1, PREFETCH_V_BLOCKS * 16 + (lane_id >> 2));
                    int v_col = load_loop * 8 * 32 + warp_id * 32 + (lane_id & 3) * 8;
                    int v_buffer_offset = v_row * v_row_stride + v_col;
                    int lds_write_offset = stage_id * V_LOAD_REQUESTS * WARP_NUM * 512 + load_loop * WARP_NUM * 512 + warp_id * 512; // 2 * 2 * 8 * 512 * sizeof(half) = 32KB
                    safe_inline_buffer_load_dwordx4_lds<Element, 1>(v_lds, v_buffer, lds_write_offset, 0, v_buffer_offset);
                }
            }
            stage_id ^= 1;
            #pragma unroll
            for (int n_loop = PREFETCH_V_BLOCKS + 1; n_loop < kBlockN / 16; n_loop += 1) {
                #pragma unroll
                for (int load_loop = 0; load_loop < V_LOAD_REQUESTS; ++load_loop) {
                    int v_row = min(seqlen_kv_limit - 1, n_loop * 16 + (lane_id >> 2));
                    int v_col = load_loop * 8 * 32 + warp_id * 32 + (lane_id & 3) * 8;
                    int v_buffer_offset = v_row * v_row_stride + v_col;
                    int lds_write_offset = stage_id * V_LOAD_REQUESTS * WARP_NUM * 512 + load_loop * WARP_NUM * 512 + warp_id * 512; // 2 * 2 * 8 * 512 * sizeof(half) = 32KB
                    safe_inline_buffer_load_dwordx4_lds<Element, 1>(v_lds, v_buffer, lds_write_offset, 0, v_buffer_offset);
                }
                stage_id ^= 1;
                wait_buffer_data_arrived<true>(V_LOAD_REQUESTS);
                // lds -> vgprs
                union_vec4_f16x2<Element> v_regs[kHeadDimVSplit / 32];
                #pragma unroll
                for (int v_tile = 0; v_tile < kHeadDimVSplit / 32; ++v_tile) {
                    int v_load_base_offset = stage_id * V_LOAD_REQUESTS * WARP_NUM * 512 + warp_id_col * 8 * 512 + v_tile * 512;
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        int v_load_offset = v_load_base_offset + i * 8 * 32 + (lane_id >> 4) * 32 + (lane_id & 15) * 2;
                        inline_ds_read2_b32_no_wait_bytes(reinterpret_cast<size_t>(v_lds + v_load_offset), v_regs[v_tile].f16x4[i], 64);
                    }
                }
                // pv mmac
                #pragma unroll
                for (int v_tile = 0; v_tile < kHeadDimVSplit / 32; ++v_tile) {
                    wait_lds_data_arrived<false>((kHeadDimVSplit / 32 - v_tile - 1) * 2);
                    // v interleave into vgprs
                    union_vec4_f16x2<Element> v_composed;
                    v_composed.f16x4[0] = make_vec4_f16<Element>(v_regs[v_tile].f16[0 * 2 + 0], v_regs[v_tile].f16[1 * 2 + 0], v_regs[v_tile].f16[2 * 2 + 0], v_regs[v_tile].f16[3 * 2 + 0]);
                    v_composed.f16x4[1] = make_vec4_f16<Element>(v_regs[v_tile].f16[0 * 2 + 1], v_regs[v_tile].f16[1 * 2 + 1], v_regs[v_tile].f16[2 * 2 + 1], v_regs[v_tile].f16[3 * 2 + 1]);
                    // pv mmac
                    #pragma unroll
                    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                        acc_o[m_idx][v_tile * 2 + 0].f32 = mmac<Element, ElementAccum>(p_reg[m_idx][n_loop - 1].f16x4, v_composed.f16x4[0], acc_o[m_idx][v_tile * 2 + 0].f32);
                        acc_o[m_idx][v_tile * 2 + 1].f32 = mmac<Element, ElementAccum>(p_reg[m_idx][n_loop - 1].f16x4, v_composed.f16x4[1], acc_o[m_idx][v_tile * 2 + 1].f32);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                __syncthreads();
                __builtin_amdgcn_sched_barrier(0);
            }
            // rest
            {
                constexpr int n_loop = kBlockN / 16;
                stage_id ^= 1;
                wait_buffer_data_arrived<true>(0);
                // lds -> vgprs
                union_vec4_f16x2<Element> v_regs[kHeadDimVSplit / 32];
                #pragma unroll
                for (int v_tile = 0; v_tile < kHeadDimVSplit / 32; ++v_tile) {
                    int v_load_base_offset = stage_id * V_LOAD_REQUESTS * WARP_NUM * 512 + warp_id_col * 8 * 512 + v_tile * 512;
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        int v_load_offset = v_load_base_offset + i * 8 * 32 + (lane_id >> 4) * 32 + (lane_id & 15) * 2;
                        inline_ds_read2_b32_no_wait_bytes(reinterpret_cast<size_t>(v_lds + v_load_offset), v_regs[v_tile].f16x4[i], 64);
                    }
                }
                // pv mmac
                #pragma unroll
                for (int v_tile = 0; v_tile < kHeadDimVSplit / 32; ++v_tile) {
                    wait_lds_data_arrived<false>((kHeadDimVSplit / 32 - v_tile - 1) * 2);
                    // v interleave into vgprs
                    union_vec4_f16x2<Element> v_composed;
                    v_composed.f16x4[0] = make_vec4_f16<Element>(v_regs[v_tile].f16[0 * 2 + 0], v_regs[v_tile].f16[1 * 2 + 0], v_regs[v_tile].f16[2 * 2 + 0], v_regs[v_tile].f16[3 * 2 + 0]);
                    v_composed.f16x4[1] = make_vec4_f16<Element>(v_regs[v_tile].f16[0 * 2 + 1], v_regs[v_tile].f16[1 * 2 + 1], v_regs[v_tile].f16[2 * 2 + 1], v_regs[v_tile].f16[3 * 2 + 1]);
                    // pv mmac
                    #pragma unroll
                    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                        acc_o[m_idx][v_tile * 2 + 0].f32 = mmac<Element, ElementAccum>(p_reg[m_idx][n_loop - 1].f16x4, v_composed.f16x4[0], acc_o[m_idx][v_tile * 2 + 0].f32);
                        acc_o[m_idx][v_tile * 2 + 1].f32 = mmac<Element, ElementAccum>(p_reg[m_idx][n_loop - 1].f16x4, v_composed.f16x4[1], acc_o[m_idx][v_tile * 2 + 1].f32);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                __syncthreads();
                __builtin_amdgcn_sched_barrier(0);
            }
        }
    } else if constexpr (WARP_NUM == 4) {
        constexpr int V_LOAD_REQUESTS = 4;
        // mla_prefetch_v_to_lds<V_LOAD_REQUESTS, Element>(v_buffer, v_lds, v_row_stride, warp_id, lane_id, seqlen_kv_limit);
        int stage_id = 1;
        int warp_id_col = warp_id & 1;
        #pragma unroll
        for (int n_loop = 1; n_loop < kBlockN / 16; ++n_loop) {
            #pragma unroll
            for (int load_loop = 0; load_loop < V_LOAD_REQUESTS; ++load_loop) {
                int v_row = min(seqlen_kv_limit - 1, n_loop * 16 + (lane_id >> 2));
                int v_col = load_loop * 4 * 32 + warp_id * 32 + (lane_id & 3) * 8;
                int v_buffer_offset = v_row * v_row_stride + v_col;
                int lds_write_offset = stage_id * 4 * 4 * 512 + load_loop * 4 * 512 + warp_id * 512; // 4 * 4 * 512 * sizeof(fp16) = 16KB
                safe_inline_buffer_load_dwordx4_lds<Element, 1>(v_lds, v_buffer, lds_write_offset, 0, v_buffer_offset);
            }
            wait_buffer_data_arrived<true>(V_LOAD_REQUESTS);
            stage_id ^= 1;
            #pragma unroll
            for (int v_tile = 0; v_tile < kHeadDimVSplit / 32; ++v_tile) {
                // lds -> vgprs
                union_vec4_f16x2<Element> v_regs;
                int v_load_base_offset = stage_id * 4 * 4 * 512 + warp_id_col * 8 * 512 + v_tile * 512;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int v_load_offset = v_load_base_offset + i * 4 * 32 + (lane_id >> 4) * 32 + (lane_id & 15) * 2;
                    v_regs.f16x2[i] = *(vec2_Element<Element>*)(v_lds + v_load_offset);
                }
                // v regs interleave
                union_vec4_f16x2<Element> v_composed;
                v_composed.f16x4[0] = make_vec4_f16<Element>(v_regs.f16[0 * 2 + 0], v_regs.f16[1 * 2 + 0], v_regs.f16[2 * 2 + 0], v_regs.f16[3 * 2 + 0]);
                v_composed.f16x4[1] = make_vec4_f16<Element>(v_regs.f16[0 * 2 + 1], v_regs.f16[1 * 2 + 1], v_regs.f16[2 * 2 + 1], v_regs.f16[3 * 2 + 1]);
                // pv mmac
                #pragma unroll
                for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                    acc_o[m_idx][v_tile * 2 + 0].f32 = mmac<Element, ElementAccum>(p_reg[m_idx][n_loop - 1].f16x4, v_composed.f16x4[0], acc_o[m_idx][v_tile * 2 + 0].f32);
                    acc_o[m_idx][v_tile * 2 + 1].f32 = mmac<Element, ElementAccum>(p_reg[m_idx][n_loop - 1].f16x4, v_composed.f16x4[1], acc_o[m_idx][v_tile * 2 + 1].f32);
                }
            }
            __syncthreads();
        }
        {
            if constexpr (PREFETCH_K) {
                vec4_uint k_rope_buffer = v_buffer;
                *(int64_t*)&k_rope_buffer += v_buffer_offset;
                mla_prefix_prefill_prefetch_k_nope_to_lds<kBlockN, WARP_NUM, Element>(v_lds, k_rope_buffer, warp_id, lane_id, v_row_stride, seqlen_kv_limit - kBlockN);
                wait_buffer_data_arrived<true>(kBlockN / (16 * 2));
            } else {
                wait_buffer_data_arrived<true>(0);
            }
            constexpr int n_loop = kBlockN / 16;
            stage_id ^= 1;
            #pragma unroll
            for (int v_tile = 0; v_tile < kHeadDimVSplit / 32; ++v_tile) {
                // lds -> vgprs
                union_vec4_f16x2<Element> v_regs;
                int v_load_base_offset = stage_id * 4 * 4 * 512 + warp_id_col * 8 * 512 + v_tile * 512;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int v_load_offset = v_load_base_offset + i * 4 * 32 + (lane_id >> 4) * 32 + (lane_id & 15) * 2;
                    v_regs.f16x2[i] = *(vec2_Element<Element>*)(v_lds + v_load_offset);
                }
                // v vgpr interleave
                union_vec4_f16x2<Element> v_composed;
                v_composed.f16x4[0] = make_vec4_f16<Element>(v_regs.f16[0 * 2 + 0], v_regs.f16[1 * 2 + 0], v_regs.f16[2 * 2 + 0], v_regs.f16[3 * 2 + 0]);
                v_composed.f16x4[1] = make_vec4_f16<Element>(v_regs.f16[0 * 2 + 1], v_regs.f16[1 * 2 + 1], v_regs.f16[2 * 2 + 1], v_regs.f16[3 * 2 + 1]);
                // pv mmac
                #pragma unroll
                for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                    acc_o[m_idx][v_tile * 2 + 0].f32 = mmac<Element, ElementAccum>(p_reg[m_idx][n_loop - 1].f16x4, v_composed.f16x4[0], acc_o[m_idx][v_tile * 2 + 0].f32);
                    acc_o[m_idx][v_tile * 2 + 1].f32 = mmac<Element, ElementAccum>(p_reg[m_idx][n_loop - 1].f16x4, v_composed.f16x4[1], acc_o[m_idx][v_tile * 2 + 1].f32);
                }
            }
            __syncthreads();
        }
    }
}




template<int kBlockM, int WARP_M, int WARP_NUM, int kHeadDimVSplit, typename ElementAccum>
__forceinline__ __device__ void mla_prefix_prefill_rescale_acc_o(
    vec4_Accum<ElementAccum> acc_o[WARP_M / 16][kHeadDimVSplit / 16],
    ElementAccum* scores_max_ptr,
    ElementAccum* scores_sum_ptr,
    ElementAccum* softmax_lse_ptr,
    ElementAccum scores_max[WARP_M / 16],
    ElementAccum scores_sum[WARP_M / 16],
    ElementAccum scale_softmax,
    int64_t row_offset_lse,
    int m_block,
    int warp_id,
    int warp_id_row,
    int lane_id,
    int actual_seqlen_q
) {
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        ElementAccum sum = scores_sum[m_idx];
        ElementAccum lse = (sum == 0.f || sum != sum) ? INFINITY: __llvm_fma_f32(scores_max[m_idx], scale_softmax, __logf(sum));
        if constexpr (WARP_NUM == 8) {
            if (lane_id < 16 and warp_id < 4) {
                int lse_offset = warp_id * 16 + lane_id;
                if (lse_offset < actual_seqlen_q - m_block * kBlockM) {
                    scores_max_ptr[row_offset_lse + lse_offset]  = scores_max[m_idx] * scale_softmax;
                    scores_sum_ptr[row_offset_lse + lse_offset]  = scores_sum[m_idx];
                    softmax_lse_ptr[row_offset_lse + lse_offset] = lse;
                }
            }
        } else if constexpr (WARP_NUM == 4) {
            if (lane_id < 16 and ((warp_id & 1) == 0)) {
                int lse_offset = m_idx * WARP_M + warp_id_row * 16 + lane_id;
                if (lse_offset < actual_seqlen_q - m_block * kBlockM) {
                    scores_max_ptr[row_offset_lse + lse_offset]  = scores_max[m_idx] * scale_softmax;
                    scores_sum_ptr[row_offset_lse + lse_offset]  = scores_sum[m_idx];
                    softmax_lse_ptr[row_offset_lse + lse_offset] = lse;
                }
            }
        }
        // 放缩 acc_o
        __float2 inv_sum;
        inv_sum[0] = 1.0f / sum;
        inv_sum[1] = inv_sum[0];
        #pragma unroll
        for (int pv_tile = 0; pv_tile < kHeadDimVSplit / 16; ++pv_tile) {
            acc_o[m_idx][pv_tile].u64[0] = __builtin_hcu_pk_mul_f32(acc_o[m_idx][pv_tile].u64[0], inv_sum);
            acc_o[m_idx][pv_tile].u64[1] = __builtin_hcu_pk_mul_f32(acc_o[m_idx][pv_tile].u64[1], inv_sum);
        }
    }
}



template<int kBlockM, int WARP_M, int WARP_NUM, int kHeadDimVSplit, typename Element, typename ElementAccum>
__forceinline__ __device__ void mla_prefix_prefill_store_output(
    vec4_Accum<ElementAccum> acc_o[WARP_M / 16][kHeadDimVSplit / 16],
    void* __restrict__ o_raw_ptr,
    int64_t row_offset_o,
    int m_block,
    int warp_id_row,
    int warp_id_col,
    int lane_id,
    int o_row_stride,
    int actual_seqlen_q
) {
    Element *o_ptr = reinterpret_cast<Element*>(o_raw_ptr) + row_offset_o;
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        #pragma unroll
        for (int v_tile = 0; v_tile < kHeadDimVSplit / 32; ++v_tile) {
            int row_idx = (WARP_NUM == 8 ? m_idx * 64: m_idx * WARP_M) + warp_id_row * 16 + (lane_id & 15);
            if (m_block * kBlockM + row_idx < actual_seqlen_q) {
                #pragma unroll
                for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                    vec2_Element<Element> data;
                    #if defined(__gfx938__) || defined(__gfx946__)
                        #pragma unroll
                        for (int mmac_id = 0; mmac_id < 2; ++mmac_id) {
                            data[mmac_id] = DownCast<ElementAccum, Element, true>(acc_o[m_idx][v_tile * 2 + mmac_id].f32[vec_idx]);
                        }
                    #else
                        data = DownCastPairNoPack<ElementAccum, Element>(acc_o[m_idx][v_tile * 2 + 0].f32[vec_idx], acc_o[m_idx][v_tile * 2 + 1].f32[vec_idx]);
                    #endif
                    int col_idx = warp_id_col * 256 + v_tile * 32 + vec_idx * 8 + (lane_id >> 4) * 2;
                    int64_t write_offset = row_idx * int64_t(o_row_stride) + col_idx;
                    *(vec2_Element<Element>*)(o_ptr + write_offset) = data;
                }
            }
        }
    }
}