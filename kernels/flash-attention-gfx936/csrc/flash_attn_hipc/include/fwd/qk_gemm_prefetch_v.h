#pragma once
#include "qk_gemm_prefetch_v_headdim128.h"

#define USE_DS_OVERLAP_MMAC

namespace flash {


template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum, bool Is_even_MN>
__forceinline__ __device__ void  qk_gemm_prefetch_v(
        vec4_uint gQ,
        vec4_uint gK,
        vec4_uint gV,
        Element* q_lds,
        Element* k_lds,
        Element* v_lds,
        vec2_Element<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2][4],
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (kBlockN / 32)][4],
        int WARP_ID,
        int seqlen_k_stride,
        int seqlen_v_stride,
        int max_seq_k_offset= - 1) {

    static_assert(kBlockK == 32 and "To simplify, only kBlockK = 32 is supported! otherwise, restore q_warp_buffer_load_k_id and so on");

    union_vec4_f16x2<Element> k_reg[STAGES * (WARP_N * kBlockK) / (32 * 32) * 2];

    constexpr int WARP_NUM        = kBlockM / WARP_M;
    constexpr int q_lds_load_num  = kBlockM * kBlockK / (4 * 32);
    constexpr int Q_LOAD_REQUESTS = q_lds_load_num / WARP_NUM;
    constexpr int k_lds_load_num  = WARP_N * kBlockK / (4 * 32);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num / WARP_NUM;
    constexpr int QK_LOOP_COUNT   = kHeadDim / kBlockK;

    int lane_id       = threadIdx.x & 63; // lane id, 0-63
    int laneid_shfl_4 = lane_id >> 4;
    int laneid_and_15 = lane_id & 15;
    int qk_lane_m_idx = (laneid_shfl_4 & 1) * 2 + (laneid_shfl_4 >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int qk_lane_head_dim_idx = laneid_and_15;
    int q_warp_m_id   = WARP_ID & ((kBlockM / WARP_M) - 1);
    int k_warp_n_id   = WARP_ID & (WARP_N / WARP_N - 1);
    int q_ds_read_offset = WARP_ID * (WARP_M / 32) * (32 * 17) + (lane_id & 1) * 16 + (laneid_and_15 >> 1) * 65 + (laneid_shfl_4 & 1) * 8 + (lane_id / 32);
    int k_ds_read_offset = k_warp_n_id * (WARP_N / 32) * (32 * 17) + (lane_id & 1) * 16 + (laneid_and_15 >> 1) * 65 + (laneid_shfl_4 & 1) * 8 + (lane_id / 32);

    int stage_id = 0;

    #pragma unroll
    for (int i = 0; i < (kBlockN / WARP_N) * (WARP_M / 32); ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            inline_vgpr4_init_zero(s_reg[i][j]);
        }
    }

    constexpr int K_LOOP_START = (STAGES == 2) ? 1: 0;
    if constexpr (STAGES == 2) stage_id ^= 1;
    for(int k_loop = K_LOOP_START; k_loop < QK_LOOP_COUNT; ++k_loop) {

        if constexpr (true) {
            int k_block_buffer_load_global_offset = k_loop * kBlockK;
            int k_lds_stage_offset = stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
            for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
                int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
                int k_warp_buffer_load_lds_offset = k_lds_stage_offset + (k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32);
                int s_offset = k_block_buffer_load_global_offset / 2;
                int seqlen_pos = k_warp_buffer_load_n_id * 4 + qk_lane_m_idx;
                if constexpr (not Is_even_MN) {
                    seqlen_pos = min(seqlen_pos, max_seq_k_offset - 1);
                }
                int v_offset = seqlen_pos * seqlen_k_stride / 2 + qk_lane_head_dim_idx;
                int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
                inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
            }
        }

        // 在 wait 之前提前计算这部分偏移量
        if constexpr (STAGES == 2) stage_id ^= 1;

        int precompute_k_lds_offset[2 * 2];
        int k_lds_stage_offset = stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
        vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
        for(int i = 0; i < 2; ++i) {
            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                    for(int j = 0; j < 2; ++j) {
                        precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * (WARP_N  * 17) + n_idx * (32 * 17) + j * 4 + i * 32 + k_ds_read_offset) * 4;
                    }
                }
            }
        }

        __builtin_amdgcn_sched_barrier(0);
        #ifdef USE_PINGPANG_BUFFER
            if constexpr (STAGES == 2) {
                buffer_load_lds_dwordx1_wait<K_LOAD_REQUESTS>();
            } else if constexpr (STAGES == 1) {
                buffer_load_lds_dwordx1_wait<0>();
            }
        #else
            buffer_load_lds_dwordx1_wait<0>();
        #endif
        __builtin_amdgcn_sched_barrier(0);

        if constexpr (true) {
            // lds -> vgpr use ds_read_m; right matrix
            #pragma unroll
            for(int i = 0; i < 2; ++i) {
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int j = 0; j < 2; ++j) {
                            int lds_offset = precompute_k_lds_offset[i * 2 + j];
                            inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
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
                #pragma unroll
                for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                                int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id  = stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
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
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                #pragma unroll
                for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                                int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id  = stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
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
        asm volatile("s_barrier ; sync before load in the coming round");
    }

    // 保留第 1 阶段最后一波数据实际的 stage_id
    int last_stage_id = stage_id ^ 1;


    // 先把第 2 阶段的 load 指令先发出去
    if constexpr (kBlockN >= (WARP_N * 2)) {
        // stage_id = 0;
        if constexpr (STAGES == 2) {
            int k_loop = 0;
            if constexpr (true) {
                int k_block_buffer_load_global_offset = k_loop * kBlockK;
                int k_lds_stage_offset = stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
                for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                    int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
                    int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
                    int k_warp_buffer_load_lds_offset = k_lds_stage_offset + (k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32);
                    int s_offset = k_block_buffer_load_global_offset / 2;
                    int seqlen_pos = k_warp_buffer_load_n_id * 4 + qk_lane_m_idx + WARP_N;
                    if constexpr (not Is_even_MN) {
                        seqlen_pos = min(seqlen_pos, max_seq_k_offset - 1);
                    }
                    int v_offset = seqlen_pos * seqlen_k_stride / 2 + qk_lane_head_dim_idx;
                    int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
                    inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
                }
            }
        }
    }


    // 等待第 1 阶段最后一波数据返回做计算
    if constexpr (STAGES == 2) {
        // stage_id ^= 1;
        // 在 wait 之前提前计算好 lds load 的下标
        int precompute_k_lds_offset[2 * 2];
        if constexpr (true) {
            vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
            int k_lds_stage_offset = last_stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
            for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    for(int i = 0; i < 2; ++i) {
                        for(int j = 0; j < 2; ++j) {
                            precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * (WARP_N  * 17) + n_idx * (32 * 17) + j * 4 + i * 32 + k_ds_read_offset) * 4;
                        }
                    }
                }
            }
        }
        // 等待最后一波数据的返回
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (kBlockN >= (WARP_N * 2)) {
            buffer_load_lds_dwordx1_wait<K_LOAD_REQUESTS>();
        } else {
            buffer_load_lds_dwordx1_wait<0>();
        }
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (true) {
            // lds -> vgpr use ds_read_m; right matrix
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
                #pragma unroll
                for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                int q_tile_id  = (QK_LOOP_COUNT - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id  = last_stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
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
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                #pragma unroll
                for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                int q_tile_id  = (QK_LOOP_COUNT - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id  = last_stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
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
        asm volatile("s_barrier ; sync before load in the coming round");
    }


    // 第 2 阶段主循环
    if constexpr (kBlockN >= (WARP_N * 2)) {

        if constexpr (STAGES == 2) stage_id ^= 1;
        constexpr int K_LOOP_START = (STAGES == 2) ? 1: 0;
        for(int k_loop = K_LOOP_START; k_loop<QK_LOOP_COUNT; ++k_loop) {

            if constexpr (true) {
                int k_block_buffer_load_global_offset = k_loop * kBlockK;
                int k_lds_stage_offset = stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
                for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                    int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
                    int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
                    int k_warp_buffer_load_lds_offset = k_lds_stage_offset + (k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32);
                    int s_offset = k_block_buffer_load_global_offset / 2;
                    int seqlen_pos = k_warp_buffer_load_n_id * 4 + qk_lane_m_idx + WARP_N;
                    if constexpr (not Is_even_MN) {
                        seqlen_pos = min(seqlen_pos, max_seq_k_offset - 1);
                    }
                    int v_offset = seqlen_pos * seqlen_k_stride / 2 + qk_lane_head_dim_idx;
                    int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
                    inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
                }
            }

            if constexpr (STAGES == 2) stage_id ^= 1;

            int precompute_k_lds_offset[2 * 2];
            int k_lds_stage_offset = stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
            vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
            for(int i = 0; i < 2; ++i) {
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        for(int j = 0; j < 2; ++j) {
                            precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * (WARP_N  * 17) + n_idx * (32 * 17) + j * 4 + i * 32 + k_ds_read_offset) * 4;
                        }
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            #ifdef USE_PINGPANG_BUFFER
                if constexpr (STAGES == 2) {
                    buffer_load_lds_dwordx1_wait<K_LOAD_REQUESTS>();
                } else if constexpr (STAGES == 1) {
                    buffer_load_lds_dwordx1_wait<0>();
                }
            #else
                buffer_load_lds_dwordx1_wait<0>();
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
                                inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
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
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/1 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/1 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
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
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/1 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/1 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("s_setprio 0");
            asm volatile("s_barrier ; sync before load in the coming round");
        }
    }

    // 保留第 2 阶段最后一波数据实际的 stage_id
    last_stage_id = stage_id ^ 1;


    // 先把第 3 阶段的 load 指令先发出去
    if constexpr (kBlockN >= (WARP_N * 3)) {
        // stage_id = 0;
        if constexpr (STAGES == 2) {
            int k_loop = 0;

            if constexpr (true) {
                int k_block_buffer_load_global_offset = k_loop * kBlockK;
                int k_lds_stage_offset = stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
                for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                    int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
                    int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
                    int k_warp_buffer_load_lds_offset = k_lds_stage_offset + (k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32);
                    int s_offset = k_block_buffer_load_global_offset / 2;
                    int seqlen_pos = k_warp_buffer_load_n_id * 4 + qk_lane_m_idx + 2 * WARP_N;
                    if constexpr (not Is_even_MN) {
                        seqlen_pos = min(seqlen_pos, max_seq_k_offset - 1);
                    }
                    int v_offset = seqlen_pos * seqlen_k_stride / 2 + qk_lane_head_dim_idx;
                    int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
                    inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
                    }
            }
        }
    }

    // 等待第 2 阶段最后一波数据返回做计算
    if constexpr (kBlockN >= (WARP_N * 2)) {
        if constexpr (STAGES == 2) {
            // stage_id ^= 1;
            // 在 wait 之前提前计算好 lds load 的下标
            int precompute_k_lds_offset[2 * 2];
            if constexpr (true) {
                vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
                int k_lds_stage_offset = last_stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
                for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                        for(int i = 0; i < 2; ++i) {
                            for(int j = 0; j < 2; ++j) {
                                precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * (WARP_N  * 17) + n_idx * (32 * 17) + j * 4 + i * 32 + k_ds_read_offset) * 4;
                            }
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            // buffer_load_lds_dwordx1_wait<V_LOAD_REQUESTS>(); // when use prefetch V
            if constexpr (kBlockN >= (WARP_N * 3)) {
                buffer_load_lds_dwordx1_wait<K_LOAD_REQUESTS>();
            } else {
                buffer_load_lds_dwordx1_wait<0>();
            }
            __builtin_amdgcn_sched_barrier(0);

            if constexpr (true) {
                // lds -> vgpr use ds_read_m; right matrix
                #pragma unroll
                for(int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
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
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int q_tile_id  = (QK_LOOP_COUNT - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = last_stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/1 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/1 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
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
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int q_tile_id  = (QK_LOOP_COUNT - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = last_stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/1 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/1 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("s_setprio 0");
            asm volatile("s_barrier ; sync before load in the coming round");
        }
    }


    // 第 3 阶段主循环
    if constexpr (kBlockN >= (WARP_N * 3)) {
        if constexpr (STAGES == 2) stage_id ^= 1;
        constexpr int K_LOOP_START = (STAGES == 2) ? 1: 0;
        for(int k_loop = K_LOOP_START; k_loop<QK_LOOP_COUNT; ++k_loop) {

            if constexpr (true) {
                int k_block_buffer_load_global_offset = k_loop * kBlockK;
                int k_lds_stage_offset = stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
                for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                    int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
                    int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
                    int k_warp_buffer_load_lds_offset = k_lds_stage_offset + (k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32);
                    int s_offset = k_block_buffer_load_global_offset / 2;
                    int seqlen_pos = k_warp_buffer_load_n_id * 4 + qk_lane_m_idx + 2 * WARP_N;
                    if constexpr (not Is_even_MN) {
                        seqlen_pos = min(seqlen_pos, max_seq_k_offset - 1);
                    }
                    int v_offset = seqlen_pos * seqlen_k_stride / 2 + qk_lane_head_dim_idx;
                    int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
                    inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
                }
            }


            if constexpr (STAGES == 2) stage_id ^= 1;

            int precompute_k_lds_offset[2 * 2];
            int k_lds_stage_offset = stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
            vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
            for(int i = 0; i < 2; ++i) {
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        for(int j = 0; j < 2; ++j) {
                            precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * (WARP_N  * 17) + n_idx * (32 * 17) + j * 4 + i * 32 + k_ds_read_offset) * 4;
                        }
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            #ifdef USE_PINGPANG_BUFFER
                if constexpr (STAGES == 2) {
                    buffer_load_lds_dwordx1_wait<K_LOAD_REQUESTS>();
                } else if constexpr (STAGES == 1) {
                    buffer_load_lds_dwordx1_wait<0>();
                }
            #else
                buffer_load_lds_dwordx1_wait<0>();
            #endif
            __builtin_amdgcn_sched_barrier(0);

            if constexpr (true) {
                // lds -> vgpr use ds_read_m; right matrix
                #pragma unroll
                for(int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                        #pragma unroll
                        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                            #pragma unroll
                            for(int j = 0; j < 2; ++j) {
                                int lds_offset = precompute_k_lds_offset[i * 2 + j];
                                inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
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
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/2 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/2 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
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
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/2 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/2 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("s_setprio 0");
            asm volatile("s_barrier ; sync before load in the coming round");
        }
    }


    // 保留第 3 阶段最后一波数据实际的 stage_id
    last_stage_id = stage_id ^ 1;


    // 先把第 4 阶段的 load 指令先发出去
    if constexpr (kBlockN >= (WARP_N * 4)) {
        // stage_id = 0;
        if constexpr (STAGES == 2) {
            int k_loop = 0;

            if constexpr (true) {
                int k_block_buffer_load_global_offset = k_loop * kBlockK;
                int k_lds_stage_offset = stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
                for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                    int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
                    int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
                    int k_warp_buffer_load_lds_offset = k_lds_stage_offset + (k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32);
                    int s_offset = k_block_buffer_load_global_offset / 2;
                    int seqlen_pos = k_warp_buffer_load_n_id * 4 + qk_lane_m_idx + 3 * WARP_N;
                    if constexpr (not Is_even_MN) {
                        seqlen_pos = min(seqlen_pos, max_seq_k_offset - 1);
                    }
                    int v_offset = seqlen_pos * seqlen_k_stride / 2 + qk_lane_head_dim_idx;
                    int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
                    inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
                }
            }
        }
    }


    // 等待第 3 阶段最后一波数据返回做计算
    if constexpr (kBlockN >= (WARP_N * 3)) {
        if constexpr (STAGES == 2) {
            // stage_id ^= 1;
            // 在 wait 之前提前计算好 lds load 的下标
            int precompute_k_lds_offset[2 * 2];
            if constexpr (true) {
                vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
                int k_lds_stage_offset = last_stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
                for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                        for(int i = 0; i < 2; ++i) {
                            for(int j = 0; j < 2; ++j) {
                                precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * (WARP_N  * 17) + n_idx * (32 * 17) + j * 4 + i * 32 + k_ds_read_offset) * 4;
                            }
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            // buffer_load_lds_dwordx1_wait<V_LOAD_REQUESTS>(); // when use prefetch V
            if constexpr (kBlockN >= (WARP_N * 4)) {
                buffer_load_lds_dwordx1_wait<K_LOAD_REQUESTS>();
            } else {
                buffer_load_lds_dwordx1_wait<0>();
            }
            __builtin_amdgcn_sched_barrier(0);

            if constexpr (true) {
                #pragma unroll
                for(int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                        #pragma unroll
                        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
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
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int q_tile_id  = (QK_LOOP_COUNT - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = last_stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/2 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/2 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
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
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int q_tile_id  = (QK_LOOP_COUNT - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = last_stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/2 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/2 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("s_setprio 0");
            asm volatile("s_barrier ; sync before load in the coming round");
        }
    }

    
    // 第 4 阶段主循环
    if constexpr (kBlockN >= (WARP_N * 4)) {

        if constexpr (STAGES == 2) stage_id ^= 1;
        constexpr int K_LOOP_START = (STAGES == 2) ? 1: 0;
        for(int k_loop = K_LOOP_START; k_loop<QK_LOOP_COUNT; ++k_loop) {

            if constexpr (true) {
                int k_block_buffer_load_global_offset = k_loop * kBlockK;
                int k_lds_stage_offset = stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
                for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                    int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
                    int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
                    int k_warp_buffer_load_lds_offset = k_lds_stage_offset + (k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32);
                    int s_offset = k_block_buffer_load_global_offset / 2;
                    int seqlen_pos = k_warp_buffer_load_n_id * 4 + qk_lane_m_idx + 3 * WARP_N;
                    if constexpr (not Is_even_MN) {
                        seqlen_pos = min(seqlen_pos, max_seq_k_offset - 1);
                    }
                    int v_offset = seqlen_pos * seqlen_k_stride / 2 + qk_lane_head_dim_idx;
                    int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
                    inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
                }
            }

            if constexpr (STAGES == 2) stage_id ^= 1;

            int precompute_k_lds_offset[2 * 2];
            int k_lds_stage_offset = stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
            vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
            for(int i = 0; i < 2; ++i) {
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        for(int j = 0; j < 2; ++j) {
                            precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * (WARP_N  * 17) + n_idx * (32 * 17) + j * 4 + i * 32 + k_ds_read_offset) * 4;
                        }
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            #ifdef USE_PINGPANG_BUFFER
                if constexpr (STAGES == 2) {
                    buffer_load_lds_dwordx1_wait<K_LOAD_REQUESTS>();
                } else if constexpr (STAGES == 1) {
                    buffer_load_lds_dwordx1_wait<0>();
                }
            #else
                buffer_load_lds_dwordx1_wait<0>();
            #endif
            __builtin_amdgcn_sched_barrier(0);

            if constexpr (true) {
                // lds -> vgpr use ds_read_m; right matrix
                #pragma unroll
                for(int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                        // a warp load min size is (row, col) = (32,16) float
                        #pragma unroll
                        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                            #pragma unroll
                            for(int j = 0; j < 2; ++j) {
                                int lds_offset = precompute_k_lds_offset[i * 2 + j];
                                inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
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
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/3 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/3 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
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
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/3 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/3 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("s_setprio 0");
            asm volatile("s_barrier ; sync before load in the coming round");
        }
    }

    // 等待第 4 阶段最后一波数据返回做计算
    if constexpr (kBlockN >= (WARP_N * 4)) {
        if constexpr (STAGES == 2) {
            stage_id ^= 1;
            // 在 wait 之前提前计算好 lds load 的下标
            int precompute_k_lds_offset[2 * 2];
            if constexpr (true) {
                vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
                int k_lds_stage_offset = stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
                for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                        for(int i = 0; i < 2; ++i) {
                            for(int j = 0; j < 2; ++j) {
                                precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * (WARP_N  * 17) + n_idx * (32 * 17) + j * 4 + i * 32 + k_ds_read_offset) * 4;
                            }
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            // buffer_load_lds_dwordx1_wait<V_LOAD_REQUESTS>(); // when use prefetch V
            buffer_load_lds_dwordx1_wait<0>();
            __builtin_amdgcn_sched_barrier(0);

            if constexpr (true) {
                // lds -> vgpr use ds_read_m; right matrix
                #pragma unroll
                for(int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                        #pragma unroll
                        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                            #pragma unroll
                            for(int j = 0; j < 2; ++j) {
                                int lds_offset = precompute_k_lds_offset[i * 2 + j];
                                inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
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
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int q_tile_id  = (QK_LOOP_COUNT - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/3 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/3 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
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
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx< (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int q_tile_id  = (QK_LOOP_COUNT - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = stage_id * (WARP_N * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/3 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/3 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("s_setprio 0");
            asm volatile("s_barrier ; sync before load in the coming round");
        }
    }



    // qk gemm 等待最后一次计算需要的数据之前, 可以先把需要的 V load 指令发下去;
    constexpr int V_LOAD_REQUESTS = (WARP_M * kBlockK) / (4 * 32) / WARP_NUM;
    if constexpr (STAGES == 2) {
        if constexpr (Is_even_MN)
            prefetch_v_to_lds<kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, 0, Element, Is_even_MN>(gV, v_lds, WARP_ID, seqlen_v_stride);
        else
            prefetch_v_to_lds<kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, 0, Element, Is_even_MN>(gV, v_lds, WARP_ID, seqlen_v_stride, max_seq_k_offset);
    }

} // qk_gemm

} // namespace flash
