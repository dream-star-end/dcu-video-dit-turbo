#pragma once
#include "mla_qk_gemm_utils_tile16x32.h"

#define USE_DS_OVERLAP_MMAC


template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int WARP_NUM, int STAGES, int M_MMAC_COUNT, typename Element, typename ElementAccum>
__forceinline__ __device__ void mla_qk_gemm_prefetch_v_tile16x32(
        vec4_uint q_addr,
        vec4_uint k_addr,
        vec4_uint v_addr,
        Element* q_lds,
        Element* k_lds,
        Element* v_lds,
        union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2],
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (WARP_N / 32)][4],
        int warp_id,
        int kvcache_seqlen_stride,
        int max_seq_k_offset=-1) {

    static_assert(kBlockK == 32 and "To simplify, only kBlockK = 32 is supported! otherwise, restore q_warp_buffer_load_k_id and so on");

    union_vec4_f16x2<Element> k_reg[1 * (WARP_N * kBlockK) / (32 * 32) * 2];

    // 预先计算一些表达式
    int lane_id       = threadIdx.x & 63; // lane id, 0-63
    int laneid_shfl_4 = lane_id >> 4;
    int laneid_and_15 = lane_id & 15;

#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) // >= bmz
    int qk_lane_m_idx             = lane_id >> 2;
    int qk_lane_head_dim_idx      = (lane_id & 3) << 2;
    auto BUFFER_LOAD_FUNC         = &inline_buffer_load_dwordx4_lds<Element, 2>;
    constexpr int READ_ONCE_LINES = 16;
#else // zd
    int qk_lane_m_idx             = laneid_shfl_4;
    int qk_lane_head_dim_idx      = laneid_and_15;
    auto BUFFER_LOAD_FUNC         = &inline_buffer_load_dword_lds<Element, 2>;
    constexpr int READ_ONCE_LINES = 4;
#endif
    constexpr int k_lds_load_num  = (WARP_N * kBlockK) / (READ_ONCE_LINES * 32);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num;

    int k_warp_n_id = (warp_id & (WARP_N / WARP_N - 1));
    // 0,0,32,32,0,0,32,32 | 0,0,32,32,0,0,32,32 | 16,16,48,48,16,16,48,48 | 16,16,48,48,16,16,48,48
    // (lane_id & 1) * 16: in seqlen direction, [0,1,0,1,2,3,2,3], odd threads need skip 32 Halfs, 16 dword
    // (laneid_and_15 >> 1) * 64: threads 0,1 occupy 4 lines, 4x32 Halfs, 64 dword.... 2,3 and 4,5 and 6,7 is the same
    // laneid_and_15 >> 1, padding
    // (laneid_shfl_4 & 1) * 8: threads 0,32 is even times of 16, thus 0,32; threads 16,48 is odd times of 16, thus 0,32,16,48; 0->16 need skip 16 Halfs, 8 dword
    // (lane_id / 32): 0,0,32,32,0,0,32,32, 0->32, skip 2 Halfs, 1 dword
    int k_ds_read_offset = k_warp_n_id * (WARP_N / 32) * (32 * 16) + laneid_and_15 * 16 + (laneid_shfl_4 & 1) * 8 + (lane_id / 32) * 4;

    // 初始化 s
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

    int stage_id = 0;

    #ifdef MLA_K_LOAD_32x64_BLOCKS
        constexpr int K_LOAD_BLOCKS = 2;
    #else
        constexpr int K_LOAD_BLOCKS = 1;
    #endif

    constexpr int K_LOOP_START = (STAGES == 2) ? K_LOAD_BLOCKS: 0;
    if constexpr (STAGES == 2) stage_id ^= 1;
    for (int k_loop = K_LOOP_START; k_loop < (kHeadDim / kBlockK); k_loop += K_LOAD_BLOCKS) {

        if constexpr (true) {
            int k_block_buffer_load_global_offset = k_loop * kBlockK/*offset in headdim direction*/;
            int k_lds_stage_offset = (warp_id * STAGES * K_LOAD_BLOCKS + stage_id * K_LOAD_BLOCKS) * (WARP_N / 32) * (kBlockK / 32) * (32 * 32);
            #pragma unroll
            for (int load = 0; load < K_LOAD_REQUESTS; ++load) {
                int k_warp_buffer_load_n_id = load & (WARP_N / READ_ONCE_LINES - 1);
                int k_warp_buffer_load_lds_offset = k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 32) + (k_warp_buffer_load_n_id & 7) * (READ_ONCE_LINES * 32));
                int lds_offset = k_warp_buffer_load_lds_offset / 2;
                int offset_s   = k_block_buffer_load_global_offset / 2;
                int offset_v   = min(k_warp_buffer_load_n_id * READ_ONCE_LINES + qk_lane_m_idx + warp_id * WARP_N, max_seq_k_offset - 1) * kvcache_seqlen_stride / 2 + qk_lane_head_dim_idx;
                BUFFER_LOAD_FUNC(k_lds, k_addr, lds_offset, offset_s, offset_v);
            }
        }
        #ifdef MLA_K_LOAD_32x64_BLOCKS
        if constexpr (true) {
            int k_block_buffer_load_global_offset = (k_loop + 1) * kBlockK/*offset in headdim direction*/;
            int k_lds_stage_offset = (warp_id * STAGES * K_LOAD_BLOCKS + stage_id * K_LOAD_BLOCKS + 1) * (WARP_N / 32) * (kBlockK / 32) * (32 * 32);
            #pragma unroll
            for (int load = 0; load < K_LOAD_REQUESTS; ++load) {
                int k_warp_buffer_load_n_id = load & (WARP_N / READ_ONCE_LINES - 1);
                int k_warp_buffer_load_lds_offset = k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 32) + (k_warp_buffer_load_n_id & 7) * (READ_ONCE_LINES * 32));
                int lds_offset = k_warp_buffer_load_lds_offset / 2;
                int offset_s   = k_block_buffer_load_global_offset / 2;
                int offset_v   = min(k_warp_buffer_load_n_id * READ_ONCE_LINES + qk_lane_m_idx + warp_id * WARP_N, max_seq_k_offset - 1) * kvcache_seqlen_stride / 2 + qk_lane_head_dim_idx;
                BUFFER_LOAD_FUNC(k_lds, k_addr, lds_offset, offset_s, offset_v);
            }
        }
        #endif

        // 在 wait 之前提前计算这部分偏移量
        if constexpr (STAGES == 2) stage_id ^= 1;

        int precompute_k_lds_offset[2];
        int k_lds_stage_offset = (warp_id * STAGES * K_LOAD_BLOCKS + stage_id * K_LOAD_BLOCKS) * (WARP_N / 32) * (kBlockK / 32) * (32 * 16);
        vec2_Element<Element> *k_lds_v2fp16 = (vec2_Element<Element> *)(k_lds);
        for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
            for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                for (int i = 0; i < 2; ++i) {
                    precompute_k_lds_offset[i] = k_lds_stage_offset + head_dim_idx * WARP_N * 16 + n_idx * 32 * 16 + i * 16 * 16 + k_ds_read_offset;
                }
            }
        }

        __builtin_amdgcn_sched_barrier(0);
        #ifdef USE_PINGPANG_BUFFER
            if constexpr (STAGES == 2) {
                #ifdef MLA_K_LOAD_32x64_BLOCKS
                    buffer_load_lds_dwordx1_wait_nosync<3 * K_LOAD_REQUESTS>();
                #else
                    buffer_load_lds_dwordx1_wait_nosync<1 * K_LOAD_REQUESTS>();
                #endif
            } else if constexpr (STAGES == 1) {
                buffer_load_lds_dwordx1_wait_nosync<0>();
            }
        #else
            buffer_load_lds_dwordx1_wait_nosync<0>();
        #endif
        __builtin_amdgcn_sched_barrier(0);

        #pragma unroll
        for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
            #pragma unroll
            for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    k_reg[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].f32 = *(vec4_fp32*)(k_lds_v2fp16 + precompute_k_lds_offset[i]);
                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_setprio 1");

        {
            int min_tile_n = 0;
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                    #pragma unroll
                    for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                        #pragma unroll
                        for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                int k_loop_idx = (STAGES == 2) ? k_loop - K_LOAD_BLOCKS: k_loop;
                                int q_tile_id = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    vec4_Element<Element>{q_reg[q_tile_id].f16x2[2 * min_tile_k][0],
                                                          q_reg[q_tile_id].f16x2[2 * min_tile_k][1],
                                                          q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                          q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                    vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        {
            int min_tile_n = 1;
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                    #pragma unroll
                    for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                        #pragma unroll
                        for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                int k_loop_idx = (STAGES == 2) ? k_loop - K_LOAD_BLOCKS: k_loop;
                                int q_tile_id = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    vec4_Element<Element>{q_reg[q_tile_id].f16x2[2 * min_tile_k][0],
                                                          q_reg[q_tile_id].f16x2[2 * min_tile_k][1],
                                                          q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                          q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                    vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
        }
        asm volatile("s_setprio 0");
        // asm volatile("s_barrier ; sync before load in the coming round");

        #ifdef MLA_K_LOAD_32x64_BLOCKS
        {
            int precompute_k_lds_offset[2];
            int k_lds_stage_offset = (warp_id * STAGES * K_LOAD_BLOCKS + stage_id * K_LOAD_BLOCKS + 1) * (WARP_N / 32) * (kBlockK / 32) * (32 * 16);
            vec2_Element<Element> *k_lds_v2fp16 = (vec2_Element<Element> *)(k_lds);
            for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                    for (int i = 0; i < 2; ++i) {
                        precompute_k_lds_offset[i] = k_lds_stage_offset + head_dim_idx * WARP_N * 16 + n_idx * 32 * 16 + i * 16 * 16 + k_ds_read_offset;
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            #ifdef USE_PINGPANG_BUFFER
                if constexpr (STAGES == 2) {
                    buffer_load_lds_dwordx1_wait_nosync<2 * K_LOAD_REQUESTS>();
                } else if constexpr (STAGES == 1) {
                    buffer_load_lds_dwordx1_wait_nosync<0>();
                }
            #else
                buffer_load_lds_dwordx1_wait_nosync<0>();
            #endif
            __builtin_amdgcn_sched_barrier(0);

            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                #pragma unroll
                for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        k_reg[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].f32 = *(vec4_fp32*)(k_lds_v2fp16 + precompute_k_lds_offset[i]);
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 1");

            {
                int min_tile_n = 0;
                #pragma unroll
                for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
                    #pragma unroll
                    for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                        #pragma unroll
                        for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                            #pragma unroll
                            for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                                #pragma unroll
                                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                    int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                                    int q_tile_id = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id].f16x2[2 * min_tile_k][0],
                                                            q_reg[q_tile_id].f16x2[2 * min_tile_k][1],
                                                            q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            {
                int min_tile_n = 1;
                #pragma unroll
                for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
                    #pragma unroll
                    for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                        #pragma unroll
                        for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                            #pragma unroll
                            for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                                #pragma unroll
                                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                    int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                                    int q_tile_id = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id].f16x2[2 * min_tile_k][0],
                                                            q_reg[q_tile_id].f16x2[2 * min_tile_k][1],
                                                            q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
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
        #endif
    }

    // 保留第 1 阶段最后一波数据实际的 stage_id
    int last_stage_id = stage_id ^ 1;

    // 等待第 1 阶段最后一波数据返回做计算
    if constexpr (STAGES == 2) {
        constexpr int k_loop = kHeadDim / kBlockK;
        // 在 wait 之前提前计算好 lds load 的下标
        int precompute_k_lds_offset[2];
        vec2_Element<Element> *k_lds_v2fp16 = (vec2_Element<Element> *)(k_lds);
        int k_lds_stage_offset = (warp_id * STAGES * K_LOAD_BLOCKS + last_stage_id * K_LOAD_BLOCKS) * (WARP_N / 32) * (kBlockK / 32) * (32 * 16);
        for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
            for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                for (int i = 0; i < 2; ++i) {
                    precompute_k_lds_offset[i] = k_lds_stage_offset + head_dim_idx * (WARP_N * 16) + n_idx * (32 * 16) + i * 16 * 16 + k_ds_read_offset;
                }
            }
        }
        // 等待最后一波数据的返回
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (kBlockN >= (WARP_N * 2)) {
            #ifdef MLA_K_LOAD_32x64_BLOCKS
                buffer_load_lds_dwordx1_wait_nosync<1 * K_LOAD_REQUESTS>();
            #else
                buffer_load_lds_dwordx1_wait_nosync<0 * K_LOAD_REQUESTS>();
            #endif
        } else {
            buffer_load_lds_dwordx1_wait_nosync<0>();
        }
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                for (int i = 0; i < 2; ++i) {
                    k_reg[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].f32 = *(vec4_fp32*)(k_lds_v2fp16 + precompute_k_lds_offset[i]);
                }
            }
        }

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_setprio 1");

        {
            int min_tile_n = 0;
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                    #pragma unroll
                    for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                        #pragma unroll
                        for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                int q_tile_id = (k_loop - K_LOAD_BLOCKS) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    vec4_Element<Element>{q_reg[q_tile_id].f16x2[2 * min_tile_k][0],
                                                          q_reg[q_tile_id].f16x2[2 * min_tile_k][1],
                                                          q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                          q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                    vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        {
            int min_tile_n = 1;
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                    #pragma unroll
                    for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                        #pragma unroll
                        for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                int q_tile_id = (k_loop - K_LOAD_BLOCKS) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    vec4_Element<Element>{q_reg[q_tile_id].f16x2[2 * min_tile_k][0],
                                                          q_reg[q_tile_id].f16x2[2 * min_tile_k][1],
                                                          q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                          q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                    vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
        }
        asm volatile("s_setprio 0");
        // asm volatile("s_barrier ; sync before load in the coming round");
        #ifdef MLA_K_LOAD_32x64_BLOCKS
        {
            int precompute_k_lds_offset[2];
            vec2_Element<Element> *k_lds_v2fp16 = (vec2_Element<Element> *)(k_lds);
            int k_lds_stage_offset = (warp_id * STAGES * K_LOAD_BLOCKS + last_stage_id * K_LOAD_BLOCKS + 1) * (WARP_N / 32) * (kBlockK / 32) * (32 * 16);
            for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                    for (int i = 0; i < 2; ++i) {
                        precompute_k_lds_offset[i] = k_lds_stage_offset + head_dim_idx * (WARP_N * 16) + n_idx * (32 * 16) + i * 16 * 16 + k_ds_read_offset;
                    }
                }
            }
            // 等待最后一波数据的返回
            __builtin_amdgcn_sched_barrier(0);
            buffer_load_lds_dwordx1_wait_nosync<0 * K_LOAD_REQUESTS>();
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                    for (int i = 0; i < 2; ++i) {
                        k_reg[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].f32 = *(vec4_fp32*)(k_lds_v2fp16 + precompute_k_lds_offset[i]);
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 1");

            {
                int min_tile_n = 0;
                #pragma unroll
                for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
                    #pragma unroll
                    for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                        #pragma unroll
                        for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                            #pragma unroll
                            for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                                #pragma unroll
                                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                    int q_tile_id = (k_loop - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id].f16x2[2 * min_tile_k][0],
                                                            q_reg[q_tile_id].f16x2[2 * min_tile_k][1],
                                                            q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            {
                int min_tile_n = 1;
                #pragma unroll
                for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
                    #pragma unroll
                    for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                        #pragma unroll
                        for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                            #pragma unroll
                            for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                                #pragma unroll
                                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                    int q_tile_id = (k_loop - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id].f16x2[2 * min_tile_k][0],
                                                            q_reg[q_tile_id].f16x2[2 * min_tile_k][1],
                                                            q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
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
        #endif
    }

    // need to reduce results on scores_max and prefetch V, and thus sync
    __syncthreads();

    // qk gemm 等待最后一次计算需要的数据之前, 可以先把需要的 V load 指令发下去;
    if constexpr (STAGES > 1) {
        mla_prefetch_v_to_lds_tile16x32<kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, 32/*WARP_K*/, 0, WARP_NUM, Element, STAGES>(v_addr, v_lds, warp_id, kvcache_seqlen_stride, max_seq_k_offset);
    }

} // qk_gemm

