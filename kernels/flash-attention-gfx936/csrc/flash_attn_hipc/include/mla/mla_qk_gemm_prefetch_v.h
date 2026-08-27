#pragma once
#include "mla_qk_gemm_prefetch_v_qinlds.h"

#define USE_DS_OVERLAP_MMAC


template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int WARP_NUM, int STAGES, int M_MMAC_COUNT, typename Element, typename ElementAccum>
__forceinline__ __device__ void mla_qk_gemm_prefetch_v(
        vec4_uint q_addr,
        vec4_uint k_addr,
        vec4_uint v_addr,
        Element* q_lds,
        Element* k_lds,
        Element* v_lds,
        vec2_Element<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2][4],
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (WARP_N / 32)][4],
        int warp_id,
        int kvcache_seqlen_stride,
        int max_seq_k_offset=-1) {

    static_assert(kBlockK == 32 and "To simplify, only kBlockK = 32 is supported! otherwise, restore q_warp_buffer_load_k_id and so on");
    constexpr int k_lds_load_num  = (WARP_N * kBlockK) / (4 * 32);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num;

    union_vec4_f16x2<Element> k_reg[STAGES * (WARP_N * kBlockK) / (32 * 32) * 2];

    // 预先计算一些表达式
    int lane_id       = threadIdx.x & 63; // lane id, 0-63
    int laneid_shfl_4 = lane_id >> 4;
    int laneid_and_15 = lane_id & 15;
    int qk_lane_m_idx = (laneid_shfl_4 & 1) * 2 + (laneid_shfl_4 >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int qk_lane_head_dim_idx = laneid_and_15;
    int k_warp_n_id   = (warp_id & (WARP_N / WARP_N - 1));
    int k_ds_read_offset = k_warp_n_id * (WARP_N / 32) * (32 * 17) + (lane_id & 1) * 16 + (laneid_and_15 >> 1) * 64 + (laneid_and_15 >> 1) + (laneid_shfl_4 & 1) * 8 + (lane_id / 32);

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

    constexpr int K_LOOP_START = (STAGES == 2) ? 1: 0;
    if constexpr (STAGES == 2) stage_id ^= 1;
    for (int k_loop = K_LOOP_START; k_loop < (kHeadDim / kBlockK); k_loop++) {

        if constexpr (true) {
            int k_block_buffer_load_global_offset = k_loop * kBlockK + warp_id * WARP_N * kvcache_seqlen_stride;
            int k_lds_stage_offset = (stage_id * WARP_NUM + warp_id) * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
            for (int load = 0; load < K_LOAD_REQUESTS; ++load) {
                int __load     = (load + warp_id) % K_LOAD_REQUESTS;
                int padding    = (__load & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
                int k_warp_buffer_load_n_id = __load & (WARP_N / 4 - 1);
                int k_warp_buffer_load_lds_offset = k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32));
                int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
                int offset_s   = k_block_buffer_load_global_offset / 2;
                int offset_v   = min(k_warp_buffer_load_n_id * 4 + qk_lane_m_idx, max_seq_k_offset - 1) * kvcache_seqlen_stride / 2 + qk_lane_head_dim_idx;
                inline_buffer_load_dword_lds(k_lds, k_addr, lds_offset, offset_s, offset_v);
            }
        }

        // 在 wait 之前提前计算这部分偏移量
        if constexpr (STAGES == 2) stage_id ^= 1;

        int precompute_k_lds_offset[2 * 2];
        int k_lds_stage_offset = (stage_id * WARP_NUM + warp_id) * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
        vec2_Element<Element> *k_lds_v2fp16 = (vec2_Element<Element> *)(k_lds);
        for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
            for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * WARP_N * 17 + n_idx * 32 * 17 + j * 4 + i * 32 + k_ds_read_offset) * 4/*4 bytes per dword*/;
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

        #pragma unroll
        for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
            #pragma unroll
            for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        int lds_offset = precompute_k_lds_offset[i * 2 + j];
                        inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
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
                                int k_tile_id = stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                          q_reg[q_tile_id][2 * min_tile_k][1],
                                                          q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                          q_reg[q_tile_id][2 * min_tile_k + 1][1]},
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
        #ifdef USE_DS_OVERLAP_MMAC
            asm volatile("s_waitcnt lgkmcnt(0)");
        #endif
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
                                int k_tile_id = stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                          q_reg[q_tile_id][2 * min_tile_k][1],
                                                          q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                          q_reg[q_tile_id][2 * min_tile_k + 1][1]},
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

    // 保留第 1 阶段最后一波数据实际的 stage_id
    int last_stage_id = stage_id ^ 1;

    // 等待第 1 阶段最后一波数据返回做计算
    if constexpr (STAGES == 2) {
        // stage_id ^= 1;
        // 在 wait 之前提前计算好 lds load 的下标
        int precompute_k_lds_offset[2 * 2];
        vec2_Element<Element> *k_lds_v2fp16 = (vec2_Element<Element> *)(k_lds);
        int k_lds_stage_offset = warp_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 17) + last_stage_id * WARP_NUM * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
        for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
            for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * (WARP_N * 17) + n_idx * (32 * 17) + j * 4 + i * 32 + k_ds_read_offset) * 4;
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
        #pragma unroll
        for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        int lds_offset = precompute_k_lds_offset[i * 2 + j];
                        inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[last_stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
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
            for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < WARP_N / 32; ++n_idx) {
                    #pragma unroll
                    for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                        #pragma unroll
                        for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                int q_tile_id = ((kHeadDim / kBlockK) - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id = last_stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                          q_reg[q_tile_id][2 * min_tile_k][1],
                                                          q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                          q_reg[q_tile_id][2 * min_tile_k + 1][1]},
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
        #ifdef USE_DS_OVERLAP_MMAC
            asm volatile("s_waitcnt lgkmcnt(0)");
        #endif
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
                                int q_tile_id = ((kHeadDim / kBlockK) - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id = last_stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                          q_reg[q_tile_id][2 * min_tile_k][1],
                                                          q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                          q_reg[q_tile_id][2 * min_tile_k + 1][1]},
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

    // need to reduce results on scores_max and prefetch V, and thus sync
    __syncthreads();

    // qk gemm 等待最后一次计算需要的数据之前, 可以先把需要的 V load 指令发下去;
    if constexpr (STAGES > 1) {
        mla_prefetch_v_to_lds<kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, 32/*WARP_K*/, 0, WARP_NUM, Element, STAGES>(v_addr, v_lds, warp_id, kvcache_seqlen_stride, max_seq_k_offset);
    }

} // qk_gemm

