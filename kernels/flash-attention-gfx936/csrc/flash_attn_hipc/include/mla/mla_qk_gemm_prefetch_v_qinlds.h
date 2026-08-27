#pragma once
#include "mla_qk_gemm_utils.h"

#define USE_DS_OVERLAP_MMAC


template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int M_WARP_COUNT, int N_WARP_COUNT, int WARP_NUM, int STAGES, int M_MMAC_COUNT, typename Element, typename ElementAccum>
__forceinline__ __device__ void mla_qk_gemm_prefetch_v_qinlds(
        vec4_uint q_addr,
        vec4_uint k_addr,
        vec4_uint v_addr,
        Element* q_lds,
        Element* k_lds,
        Element* v_lds,
        vec2_Element<Element> q_reg[1 * M_WARP_COUNT * (kBlockK / 32) * 2][4],
        vec4_Accum<ElementAccum> s_reg[M_WARP_COUNT * N_WARP_COUNT][4],
        int warp_id,
        int kvcache_seqlen_stride,
        int max_seq_k_offset=-1) {

    constexpr int WARP_M = M_WARP_COUNT * 32;
    constexpr int WARP_N = N_WARP_COUNT * 32;
    constexpr int K_WARP_COUNT = kBlockK / 32;
    static_assert(kBlockK == 32 and "To simplify, only kBlockK = 32 is supported! otherwise, restore q_warp_buffer_load_k_id and so on");
    static_assert(STAGES == 1 and "For mla_qk_gemm_prefetch_v_qinlds, only depth 1 is supported");

    union_vec4_f16x2<Element> k_reg[STAGES * N_WARP_COUNT * K_WARP_COUNT * 2];

    int lane_id       = threadIdx.x & 63; // lane id, 0-63
    int laneid_shfl_4 = lane_id >> 4;
    int laneid_and_15 = lane_id & 15;
    int qk_lane_m_idx = (laneid_shfl_4 & 1) * 2 + (laneid_shfl_4 >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int qk_lane_head_dim_idx = laneid_and_15;
    int k_warp_n_id   = (warp_id & (kBlockN / WARP_N - 1));
    int k_ds_read_offset = k_warp_n_id * N_WARP_COUNT * (32 * 17) + (lane_id & 1) * 16 + (laneid_and_15 >> 1) * 65 + (laneid_shfl_4 & 1) * 8 + (lane_id / 32);

    constexpr int k_lds_load_num = (WARP_N * kBlockK) / (4 * 32);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num;

    // initial qk gemm results as 0
    __builtin_amdgcn_sched_barrier(0);
    uint64_t pk_zero = 0;
    #pragma unroll
    for (int i = 0; i < (kBlockN / WARP_N) * M_WARP_COUNT; ++i) {
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            #pragma unroll
            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                s_reg[i][min_tile_n * 2 + min_tile_m].u64[0] = pk_zero;
                s_reg[i][min_tile_n * 2 + min_tile_m].u64[1] = pk_zero;
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);

    // k loop across kblockN
    int stage_id = 0;
    constexpr int K_LOOP_START = 0;
    for (int k_loop = K_LOOP_START; k_loop < (kHeadDim / kBlockK); k_loop++) {

        {
            int k_block_buffer_load_global_offset = k_loop * kBlockK + warp_id * WARP_N * kvcache_seqlen_stride;
            int k_lds_stage_offset = (stage_id * WARP_NUM + warp_id) * N_WARP_COUNT * K_WARP_COUNT * (32 * 34);
            for (int load = 0; load < K_LOAD_REQUESTS; ++load) {
                int __load     = (load + warp_id) % K_LOAD_REQUESTS;
                int padding    = (__load & 7) * 2;
                int k_warp_buffer_load_n_id = __load & (WARP_N / 4 - 1);
                int k_warp_buffer_load_lds_offset = k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32));
                int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
                int offset_s   = k_block_buffer_load_global_offset / 2;
                int offset_v   = (min(k_warp_buffer_load_n_id * 4 + qk_lane_m_idx, max_seq_k_offset - 1) * kvcache_seqlen_stride) / 2 + qk_lane_head_dim_idx;
                inline_buffer_load_dword_lds(k_lds, k_addr, lds_offset, offset_s, offset_v);
            }
        }

        int q_lds_stage_offset = k_loop * (kBlockM / 32) * K_WARP_COUNT * (32 * 17) >> 1;
        vec2_Element<Element> *q_lds_v2fp16 = (vec2_Element<Element> *)(q_lds);
        #pragma unroll
        for (int head_dim_idx = 0; head_dim_idx < K_WARP_COUNT; ++head_dim_idx) {
            #pragma unroll
            for (int m_idx = 0; m_idx < M_WARP_COUNT; m_idx++) {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int lds_offset = q_lds_stage_offset + head_dim_idx * (kBlockM >> 1) * 17 + j * 2 + i * 32 + (lane_id & 1) * 16 + (laneid_and_15 >> 1) * 64 + (laneid_and_15 >> 1/*padding*/) + (laneid_shfl_4 & 1) * 8 + (lane_id / 32);
                        int is_useless = (lane_id >> 3) & 1;
                        lds_offset = (is_useless) ? 0: lds_offset;
                        inline_ds_read_b32_wait(q_lds_v2fp16, lds_offset, q_reg[0 * M_WARP_COUNT * K_WARP_COUNT * 2 + (head_dim_idx * M_WARP_COUNT + m_idx) * 2 + i][j]);
                    }
                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);

        int precompute_k_lds_offset[2 * 2];
        int k_lds_stage_offset = warp_id * N_WARP_COUNT *  K_WARP_COUNT * (32 * 17) + stage_id * WARP_NUM * N_WARP_COUNT * K_WARP_COUNT * (32 * 17);
        vec2_Element<Element> *k_lds_v2fp16 = (vec2_Element<Element> *)(k_lds);
        for (int n_idx = 0; n_idx < N_WARP_COUNT; ++n_idx) {
            for (int head_dim_idx = 0; head_dim_idx < K_WARP_COUNT; ++head_dim_idx) {
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * (WARP_N * 17) + n_idx * (32 * 17) + j * 4 + i * 32 + k_ds_read_offset) * 4;
                    }
                }
            }
        }

        __builtin_amdgcn_sched_barrier(0);
        #ifdef USE_PINGPANG_BUFFER
            buffer_load_lds_dwordx1_wait_nosync<0>();
        #else
            buffer_load_lds_dwordx1_wait_nosync<0>();
        #endif
        __builtin_amdgcn_sched_barrier(0);

        // lds -> vgpr use ds_read_m; right matrix
        #pragma unroll
        for (int n_idx = 0; n_idx < N_WARP_COUNT; ++n_idx) {
            #pragma unroll
            for (int head_dim_idx = 0; head_dim_idx < K_WARP_COUNT; ++head_dim_idx) {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        int lds_offset = precompute_k_lds_offset[i * 2 + j];
                        inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[stage_id * N_WARP_COUNT * K_WARP_COUNT * 2 + (head_dim_idx * N_WARP_COUNT + n_idx) * 2 + i].u64[j], 2);
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
            for (int m_idx = 0; m_idx < M_WARP_COUNT; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < N_WARP_COUNT; ++n_idx) {
                    #pragma unroll
                    for (int head_dim_idx = 0; head_dim_idx < K_WARP_COUNT; ++head_dim_idx) {
                        #pragma unroll
                        for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                int q_tile_id = 0 *  M_WARP_COUNT * K_WARP_COUNT * 2 + (head_dim_idx * M_WARP_COUNT + m_idx) * 2 + min_tile_m;
                                int k_tile_id = stage_id * N_WARP_COUNT * K_WARP_COUNT * 2 + (head_dim_idx * N_WARP_COUNT + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * M_WARP_COUNT + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                          q_reg[q_tile_id][2 * min_tile_k][1],
                                                          q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                          q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                    vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                    s_reg[n_idx * M_WARP_COUNT + m_idx][min_tile_n * 2 + min_tile_m].f32);
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
            for (int m_idx = 0; m_idx < M_WARP_COUNT; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < N_WARP_COUNT; ++n_idx) {
                    #pragma unroll
                    for (int head_dim_idx = 0; head_dim_idx < K_WARP_COUNT; ++head_dim_idx) {
                        #pragma unroll
                        for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                int q_tile_id = 0 *  M_WARP_COUNT * K_WARP_COUNT * 2 + (head_dim_idx * M_WARP_COUNT + m_idx) * 2 + min_tile_m;
                                int k_tile_id = stage_id * N_WARP_COUNT * K_WARP_COUNT * 2 + (head_dim_idx * N_WARP_COUNT + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * M_WARP_COUNT + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                          q_reg[q_tile_id][2 * min_tile_k][1],
                                                          q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                          q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                    vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                          k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                    s_reg[n_idx * M_WARP_COUNT + m_idx][min_tile_n * 2 + min_tile_m].f32);
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


} // qk_gemm

