#pragma once
#include "kvcache_qk_gemm_prefetch_v_3stage.h"

#define USE_DS_OVERLAP_MMAC


template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int WARP_NUM, int STAGES, int M_MMAC_COUNT, typename Element, typename ElementAccum>
__forceinline__ __device__ void  kvcache_qk_gemm_prefetch_v(
        vec4_uint gQ,
        vec4_uint gK,
        vec4_uint gV,
        Element* q_lds,
        Element* k_lds,
        Element* v_lds,
        union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2],
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (WARP_N / 32)][4],
        int WARP_ID,
        int kcache_seqlen_stride,
        int vcache_seqlen_stride,
        int max_seq_k_offset=0) {

    static_assert (kBlockK == 32 and "To simplify, only kBlockK = 32 is supported! otherwise, restore q_warp_buffer_load_k_id and so on");

    union_vec4_f16x2<Element> k_reg[STAGES * (WARP_N * kBlockK) / (32 * 32) * 2];

    int lane_id = threadIdx.x & 63; // lane id, 0-63
    int laneid_shfl_4 = lane_id >> 4;
    int laneid_and_15 = lane_id & 15;
    int qk_lane_m_idx = (laneid_shfl_4 & 1) * 2 + (laneid_shfl_4 >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int qk_lane_head_dim_idx = laneid_and_15;

    int k_warp_n_id = WARP_ID & (WARP_N / WARP_N - 1);
    int k_ds_read_offset = k_warp_n_id * (WARP_N / 32) * (32 * 17) + (lane_id & 1) * 16 + (laneid_and_15 >> 1) * 65 + (laneid_shfl_4 & 1) * 8 + (lane_id / 32);

    constexpr int k_lds_load_num  = WARP_N * kBlockK / (4 * 32);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num;
    constexpr int ELEMENT_BYTES   = sizeof(Element);

    int stage_id = 0;

    // load 指令发下去之后, 先做一些初始化运算
    #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
        if constexpr (M_MMAC_COUNT == 1) {
            inline_vgpr4_init_zero_1x2x4(s_reg);
        } else {
            inline_vgpr4_init_zero_1x4x4(s_reg);
        }
        __builtin_amdgcn_sched_barrier(0);
    #else
        uint64_t pk_zero = 0;
        #pragma unroll
        for (int i = 0; i < (WARP_N / WARP_N) * (WARP_M / 32); ++i) {
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    s_reg[i][min_tile_n * 2 + min_tile_m].u64[0] = pk_zero;
                    s_reg[i][min_tile_n * 2 + min_tile_m].u64[1] = pk_zero;
                }
            }
        }
    #endif

    constexpr int K_LOOP_START = (STAGES == 2) ? 1: 0;
    if constexpr (STAGES == 2) stage_id ^= 1;
    for(int k_loop = K_LOOP_START; k_loop < (kHeadDim / kBlockK); ++k_loop) {

        if constexpr (true) {
            int k_block_buffer_load_global_offset = k_loop * kBlockK + WARP_ID * WARP_N * kcache_seqlen_stride;
            int k_lds_stage_offset = WARP_ID * (WARP_N / 32) *  (kBlockK / 32) * (32 * 34) + stage_id * WARP_NUM * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
            for(int load = 0; load < K_LOAD_REQUESTS; ++load) {
                int __load = (load + WARP_ID) % K_LOAD_REQUESTS;
                int padding = (__load & 7) * 2;
                int k_warp_buffer_load_n_id = __load & (WARP_N / 4 - 1);
                int k_warp_buffer_load_lds_offset = k_lds_stage_offset + (k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32);
                int s_offset = k_block_buffer_load_global_offset / ELEMENT_BYTES;
                int v_offset = min(k_warp_buffer_load_n_id * 4 + qk_lane_m_idx, max_seq_k_offset - 1) * kcache_seqlen_stride / ELEMENT_BYTES + qk_lane_head_dim_idx;
                int lds_offset = (k_warp_buffer_load_lds_offset + padding) / ELEMENT_BYTES;
                inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
            }
        }

        // 在 wait 之前提前计算这部分偏移量
        if constexpr (STAGES == 2) stage_id ^=1;

        int precompute_k_lds_offset[2 * 2];
        int k_lds_stage_offset = WARP_ID * (WARP_N / 32) *  (kBlockK / 32) * (32 * 17) + stage_id * WARP_NUM * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
        vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
        for(int i = 0; i < 2; ++i) {
            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                    for(int j = 0; j < 2; ++j) {
                        precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * (WARP_N * 17) + n_idx * (32 * 17) + j * 4 + i * 32 + k_ds_read_offset) * 4;
                    }
                }
            }
        }

        __builtin_amdgcn_sched_barrier(0);
        #ifdef USE_PINGPANG_BUFFER
            if constexpr (STAGES == 2) {
                buffer_load_lds_dwordx1_wait_nosync<K_LOAD_REQUESTS>();
            } else if constexpr (STAGES == 1) {
                buffer_load_lds_dwordx1_wait_nosync<0>();
            }
        #else
            buffer_load_lds_dwordx1_wait_nosync<0>();
        #endif
        __builtin_amdgcn_sched_barrier(0);

        if constexpr (true) {
            #pragma unroll
            for(int i = 0; i < 2; ++i) {
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int j = 0; j < 2; ++j) {
                            int lds_offset = precompute_k_lds_offset[i * 2 + j];
                            inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[stage_id * WARP_N * kBlockK / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
                        }
                    }
                }
            }
        }
        #ifdef USE_DS_OVERLAP_MMAC
            asm volatile("s_waitcnt lgkmcnt(2)");
        #else
            asm volatile("s_waitcnt lgkmcnt(0)");
        #endif
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_setprio 1");

        {
            int min_tile_n = 0;
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m<M_MMAC_COUNT; ++min_tile_m) {
                                int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                                int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id  = stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    q_reg[q_tile_id].f16x4[min_tile_k],
                                    k_reg[k_tile_id].f16x4[min_tile_k],
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
        }
        #ifdef USE_DS_OVERLAP_MMAC
            asm volatile("s_waitcnt lgkmcnt(0)");
        #endif
        __builtin_amdgcn_sched_barrier(0);
        {
            int min_tile_n = 1;
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m<M_MMAC_COUNT; ++min_tile_m) {
                                int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                                int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id  = stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    q_reg[q_tile_id].f16x4[min_tile_k],
                                    k_reg[k_tile_id].f16x4[min_tile_k],
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
        }
        asm volatile("s_setprio 0");
        // asm volatile("s_barrier ; sync before load in the coming round");
    }

    // 保留第 1 阶段最后一波数据实际的 stage_id
    int last_stage_id = stage_id ^ 1;

    // 等待第 1 阶段最后一波数据返回做计算
    if constexpr (STAGES == 2) {
        // stage_id ^= 1;
        // 在 wait 之前提前计算好 lds load 的下标
        int precompute_k_lds_offset[2 * 2];
        if constexpr (true) {
            vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
            int k_lds_stage_offset = WARP_ID * (WARP_N / 32) *  (kBlockK / 32) * (32 * 17) + last_stage_id * WARP_NUM * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
            for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    for(int i = 0; i < 2; ++i) {
                        for(int j = 0; j < 2; ++j) {
                            precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * (WARP_N * 17) + n_idx * (32 * 17) + j * 4 + i * 32 + k_ds_read_offset) * 4;
                        }
                    }
                }
            }
        }
        // 等待最后一波数据的返回
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (kBlockN >= (WARP_N * 2)) {
            buffer_load_lds_dwordx1_wait_nosync<K_LOAD_REQUESTS>();
        } else {
            buffer_load_lds_dwordx1_wait_nosync<0>();
        }
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (true) {
            #pragma unroll
            for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    for(int i = 0; i < 2; ++i) {
                        for(int j = 0; j < 2; ++j) {
                            int lds_offset = precompute_k_lds_offset[i * 2 + j];
                            inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[last_stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
                        }
                    }
                }
            }
        }

        #ifdef USE_DS_OVERLAP_MMAC
            asm volatile("s_waitcnt lgkmcnt(2)");
        #else
            asm volatile("s_waitcnt lgkmcnt(0)");
        #endif
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_setprio 1");

        {
            int min_tile_n = 0;
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m<M_MMAC_COUNT; ++min_tile_m) {
                                int q_tile_id  = ((kHeadDim / kBlockK) - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id  = last_stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    q_reg[q_tile_id].f16x4[min_tile_k],
                                    k_reg[k_tile_id].f16x4[min_tile_k],
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
        }
        #ifdef USE_DS_OVERLAP_MMAC
            asm volatile("s_waitcnt lgkmcnt(0)");
        #endif
        __builtin_amdgcn_sched_barrier(0);
        {
            int min_tile_n = 1;
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m<M_MMAC_COUNT; ++min_tile_m) {
                                int q_tile_id  = ((kHeadDim / kBlockK) - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id  = last_stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    q_reg[q_tile_id].f16x4[min_tile_k],
                                    k_reg[k_tile_id].f16x4[min_tile_k],
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
        }
        asm volatile("s_setprio 0");
        // asm volatile("s_barrier ; sync before load in the coming round");
    }

    // need to reduce results on scores_max and prefetch V, and thus sync
    __syncthreads();

    // qk gemm 等待最后一次计算需要的数据之前, 可以先把需要的 V load 指令发下去;
    if constexpr (STAGES > 1) {
        kvcache_prefetch_v_to_lds<kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, 32/*WARP_K*/, 0, WARP_NUM, Element, STAGES>(gV, v_lds, WARP_ID, vcache_seqlen_stride, max_seq_k_offset);
    }

} // qk_gemm

