#pragma once
#include "qk_gemm_utils.h"


namespace flash {

// #define USE_SCHEDULE_0_INIT

template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum, bool Is_even_MN>
__forceinline__ __device__ void  qk_gemm_prefetch_v_headdim128(
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
        int max_seq_k_offset=-1) {

    if constexpr (kHeadDim == 128) {
        static_assert(STAGES == 2 and "For double prefetch in headdim 128/64, only STAGES=2 is supported!\n");
        static_assert(kBlockN >= 64 and "For double prefetch in headdim 128/64, only BLOCK_N >= 64 is supported!\n");
    }
    static_assert(kBlockK == 32 and "To simplify, only kBlockK = 32 is supported! otherwise, restore q_warp_buffer_load_k_id and so on");
    constexpr int QK_LOOP_COUNT = kHeadDim / kBlockK;

    union_vec4_f16x2<Element> k_reg[(WARP_N * kBlockK) / (32 * 32) * 2];
    union_vec4_f16x2<Element> k_reg_tmp[(WARP_N * kBlockK) / (32 * 32) * 2];

    int lane_id              = threadIdx.x & 63; // lane id, 0-63
    int laneid_shfl_4        = lane_id >> 4;
    int laneid_and_15        = lane_id & 15;
    int qk_lane_m_idx        = (laneid_shfl_4 & 1) * 2 + (laneid_shfl_4 >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int qk_lane_head_dim_idx = laneid_and_15;
    int k_warp_n_id          = WARP_ID & (WARP_N / WARP_N - 1);
    int q_ds_read_offset     = WARP_ID * (WARP_M / 32) * (32 * 17) + (lane_id & 1) * 16 + (laneid_and_15>>1) * 65 + (laneid_shfl_4 & 1) * 8 + (lane_id / 32);
    int k_ds_read_offset     = k_warp_n_id * (WARP_N / 32) * (32 * 17) + (lane_id & 1) * 16 + (laneid_and_15>>1) * 65 + (laneid_shfl_4 & 1) * 8 + (lane_id / 32);

    constexpr int q_lds_load_num  = kBlockM * kBlockK / (4 * 32);
    constexpr int WARP_NUM        = kBlockM / WARP_M;
    constexpr int Q_LOAD_REQUESTS = q_lds_load_num / WARP_NUM;
    constexpr int k_lds_load_num  = WARP_N * kBlockK / (4 * 32);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num / WARP_NUM;

    #pragma unroll
    for (int i = 0; i < (kBlockN / WARP_N) * (WARP_M / 32); ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            inline_vgpr4_init_zero(s_reg[i][j]);
        }
    }

    // 准备 load 下一轮数据
    int stage_id = 0;
    stage_id ^= 1;

    // 第 1, 2 阶段主循环
    constexpr int K_LOOP_START = 1;
    for(int k_loop = K_LOOP_START; k_loop < QK_LOOP_COUNT; ++k_loop) {

        // load 下一轮的第 1 阶段数据
        {
            int k_block_buffer_load_global_offset = k_loop * kBlockK;
            int k_lds_stage_offset = (stage_id * STAGES) * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
            for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
                int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
                int k_warp_buffer_load_lds_offset =  k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32)) ; ;
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

        // 提前 load 下一轮的第 2 阶段数据, 让 load 指令飞得更久一点
        {
            int k_block_buffer_load_global_offset = k_loop * kBlockK;
            int k_lds_stage_offset = (stage_id * STAGES + 1) * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
            for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
                int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
                int k_warp_buffer_load_lds_offset =  k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32)) ; ;
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

        // 切到计算的轮次
        stage_id ^= 1;

        // 在 wait 之前提前计算这部分 lds load 的偏移量
        int precompute_k_lds_offset[2 * 2];
        int k_lds_stage_offset = (stage_id * STAGES) * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
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
        // 看 threadtrace 的话, 最前面发出去 2 + 2 笔请求, 刚才又发出去 2 + 2 笔请求, 现在等第一个 2 笔请求, 需要 wait vmcnt(6)
        buffer_load_lds_dwordx1_wait<3 * K_LOAD_REQUESTS>();
        __builtin_amdgcn_sched_barrier(0);

        // 第一笔的数据 lds -> vgpr
        {
            #pragma unroll
            for(int i = 0; i < 2; ++i) {
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int j = 0; j < 2; ++j) {
                            int lds_offset = precompute_k_lds_offset[i * 2 + j];
                            inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
                        }
                    }
                }
            }
        }
        // 预取第二阶段的第一笔数据, 先计算需要的 lds offset
        if constexpr (kBlockN >= (WARP_N * 2)) {
            for(int i = 0; i < 2; ++i) {
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        for(int j = 0; j < 2; ++j) {
                            precompute_k_lds_offset[i * 2 + j] += 1 * (WARP_N / 32) * (kBlockK / 32) * (32 * 17) * 4;
                        }
                    }
                }
            }
            // 预取第二阶段的第一笔数据, overlap 上面的 ds 读取的时延, 把 ds 指令提前发出去
            __builtin_amdgcn_sched_barrier(0);
            // 发出去 4 笔 K_LOAD_REQUESTS 的请求, 这里等待第 2 笔数据
            buffer_load_lds_dwordx1_wait < 2 * K_LOAD_REQUESTS>();
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
                                inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg_tmp[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
                            }
                        }
                    }
                }
            }
        }
        // 发下去 4 笔 ds_read2_b32 请求, 先等前两笔结果的返回, 但由于预发了 4 笔 ds_read2_b32 请求, 所以 2 + 4 = 6
        asm volatile("s_waitcnt lgkmcnt(6)");
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_setprio 1");
        {
            int min_tile_n = 0;
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                #pragma unroll
                for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                int k_loop_idx = k_loop - 1;
                                int q_tile_id = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
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
        // 等后两笔 ds_read2_b32 的结果
        asm volatile("s_waitcnt lgkmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        {
            int min_tile_n = 1;
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                #pragma unroll
                for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                int k_loop_idx = k_loop - 1;
                                int q_tile_id = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
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

        // 第 2 笔数据 lds -> vgpr
        if constexpr (kBlockN >= (WARP_N * 2)) {
            asm volatile("s_waitcnt lgkmcnt(2)");
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 1");
            {
                int min_tile_n = 0;
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = k_loop - 1;
                                    int q_tile_id = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/1 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/1 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            {
                int min_tile_n = 1;
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = k_loop - 1;
                                    int q_tile_id = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/1 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
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

    // 保留第 1,2 阶段最后一波数据需要计算的 id
    int last_stage_id = stage_id ^ 1;

    // 这里做了预取
    // 先把第 3 阶段的 load 指令先发出去
    if constexpr (kBlockN >= (WARP_N * 3)) {
        int k_loop = 0;
        int k_block_buffer_load_global_offset = k_loop * kBlockK;
        int k_lds_stage_offset = (stage_id * STAGES) * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
        for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
            int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
            int k_warp_buffer_load_lds_offset = k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32));
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

    // 这里做了预取
    // 先把第 4 阶段的 load 指令先发出去
    if constexpr (kBlockN >= (WARP_N * 4)) {
        int k_loop = 0;
        int k_block_buffer_load_global_offset = k_loop * kBlockK;
        int k_lds_stage_offset = (stage_id * STAGES + 1) * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
        for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
            int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
            int k_warp_buffer_load_lds_offset = k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32));
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


    // 等待第 1,2 阶段最后一波数据返回做计算
    {
        // 在 wait 之前提前计算好 lds load 的下标
        int precompute_k_lds_offset[2 * 2];
        vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
        int k_lds_stage_offset = (last_stage_id * STAGES) * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                for(int i = 0; i < 2; ++i) {
                    for(int j = 0; j < 2; ++j) {
                        precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * (WARP_N * 17) + n_idx * (32 * 17) + j * 4 + i * 32 + k_ds_read_offset) * 4;
                    }
                }
            }
        }
        // 原来发了 4 笔 K_LOAD_REQUESTS, 已经算了 2 笔, 这里在等第 3 笔, 但是预取了 3,4 阶段的数据, 所以 1 + 2 = 3 个 K_LOAD_REQUESTS
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (kBlockN >= (WARP_N * 3)) {
            buffer_load_lds_dwordx1_wait<3 * K_LOAD_REQUESTS>();
        } else {
            buffer_load_lds_dwordx1_wait<1 * K_LOAD_REQUESTS>();
        }
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
            #pragma unroll
            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                for(int i = 0; i < 2; ++i) {
                    for(int j = 0; j < 2; ++j) {
                        int lds_offset = precompute_k_lds_offset[i * 2 + j];
                        inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
                    }
                }
            }
        }
        if constexpr (kBlockN >= (WARP_N * 2)) {
            for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    for(int i = 0; i < 2; ++i) {
                        for(int j = 0; j < 2; ++j) {
                            precompute_k_lds_offset[i * 2 + j] += 1 * (WARP_N / 32) * (kBlockK / 32) * (32 * 17) * 4;
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            // 等待最后 1 笔数据的返回, 但是预取了 3, 4 阶段的数据
            if constexpr (kBlockN >= (WARP_N * 3)) {
                buffer_load_lds_dwordx1_wait < 2 * K_LOAD_REQUESTS>();
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
                            #pragma unroll
                            for(int j = 0; j < 2; ++j) {
                                int lds_offset = precompute_k_lds_offset[i * 2 + j];
                                inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg_tmp[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
                            }
                        }
                    }
                }
            }
        }
        asm volatile("s_waitcnt lgkmcnt(6)");
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_setprio 1");
        {
            int min_tile_n = 0;
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                #pragma unroll
                for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                int k_loop_idx = QK_LOOP_COUNT - 1;
                                int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
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
        asm volatile("s_waitcnt lgkmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        {
            int min_tile_n = 1;
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                #pragma unroll
                for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                int k_loop_idx = QK_LOOP_COUNT - 1;
                                int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
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

        // 等第 2 阶段的最后一波数据回来计算
        if constexpr (kBlockN >= (WARP_N * 2)) {
            asm volatile("s_waitcnt lgkmcnt(2)");
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 1");
            {
                int min_tile_n = 0;
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = QK_LOOP_COUNT - 1;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/1 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/1 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            {
                int min_tile_n = 1;
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = QK_LOOP_COUNT - 1;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/1 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
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


    // 第 3, 4 阶段主循环
    if constexpr (kBlockN >= (WARP_N * 3)) {
        // 切到 load 数据的轮次
        stage_id ^= 1;
        for(int k_loop = K_LOOP_START; k_loop < QK_LOOP_COUNT; ++k_loop) {
            // 发第 3 阶段的下一个轮次的 load 请求
            {
                int k_block_buffer_load_global_offset = k_loop * kBlockK;
                int k_lds_stage_offset = (stage_id * STAGES) * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
                for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                    int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
                    int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
                    int k_warp_buffer_load_lds_offset =  k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32)) ; ;
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
            
            // 提前把第 4 阶段的 load 发下去, 放 load 指令飞的更久一点
            if constexpr (kBlockN >= (WARP_N * 4)) {
                int k_block_buffer_load_global_offset = k_loop * kBlockK;
                int k_lds_stage_offset = (stage_id * STAGES + 1) * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
                for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                    int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
                    int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
                    int k_warp_buffer_load_lds_offset =  k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32)) ; ;
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

            // 切到计算的轮次
            stage_id ^= 1;

            // 在 wait 前先计算好 lds offset
            int precompute_k_lds_offset[2 * 2];
            int k_lds_stage_offset = (stage_id * STAGES) * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
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
           buffer_load_lds_dwordx1_wait<3 * K_LOAD_REQUESTS>();
            __builtin_amdgcn_sched_barrier(0);

            // lds -> vgpr
            #pragma unroll
            for(int i = 0; i < 2; ++i) {
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int j = 0; j < 2; ++j) {
                            int lds_offset = precompute_k_lds_offset[i * 2 + j];
                            inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
                        }
                    }
                }
            }
            if constexpr (kBlockN >= (WARP_N * 4)) {
                for(int i = 0; i < 2; ++i) {
                    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                            for(int j = 0; j < 2; ++j) {
                                precompute_k_lds_offset[i * 2 + j] += 1 * (WARP_N / 32) * (kBlockK / 32) * (32 * 17) * 4;
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                buffer_load_lds_dwordx1_wait < 2 * K_LOAD_REQUESTS>();
                __builtin_amdgcn_sched_barrier(0);

                #pragma unroll
                for(int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                        #pragma unroll
                        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                            #pragma unroll
                            for(int j = 0; j < 2; ++j) {
                                int lds_offset = precompute_k_lds_offset[i * 2 + j];
                                inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg_tmp[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
                            }
                        }
                    }
                }
            }
            asm volatile("s_waitcnt lgkmcnt(6)");
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 1");
            {
                int min_tile_n = 0;
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = k_loop - 1;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
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
            asm volatile("s_waitcnt lgkmcnt(4)");
            __builtin_amdgcn_sched_barrier(0);
            {
                int min_tile_n = 1;
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = k_loop - 1;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
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

            // 等待第 4 阶段的数据
            if constexpr (kBlockN >= (WARP_N * 4)) {
                asm volatile("s_waitcnt lgkmcnt(2)");
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 1");
                {
                    int min_tile_n = 0;
                    #pragma unroll
                    for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                        #pragma unroll
                        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                            #pragma unroll
                            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    #pragma unroll
                                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                        int k_loop_idx = k_loop - 1;
                                        int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                        int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                        s_reg[(/*n_loop*/3 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                            vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                                q_reg[q_tile_id][2 * min_tile_k][1],
                                                                q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                                q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                            vec4_Element<Element>{k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][0],
                                                                k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][1],
                                                                k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                                k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                            s_reg[(/*n_loop*/3 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                    }
                                }
                            }
                        }
                    }
                }
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_sched_barrier(0);
                {
                    int min_tile_n = 1;
                    #pragma unroll
                    for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                        #pragma unroll
                        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                            #pragma unroll
                            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    #pragma unroll
                                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                        int k_loop_idx = k_loop - 1;
                                        int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                        int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                        s_reg[(/*n_loop*/3 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                            vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                                q_reg[q_tile_id][2 * min_tile_k][1],
                                                                q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                                q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                            vec4_Element<Element>{k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][0],
                                                                k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][1],
                                                                k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                                k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
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
    }

    // 保留第 3, 4 阶段最后一波数据实际的 stage_id
    last_stage_id = stage_id ^ 1;

    // 先把第 5 阶段的 load 指令先发出去
    if constexpr (kBlockN >= (WARP_N * 5)) {
        int k_loop = 0;
        int k_block_buffer_load_global_offset = k_loop * kBlockK;
        int k_lds_stage_offset = (stage_id * STAGES) * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
        for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
            int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
            int k_warp_buffer_load_lds_offset = k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32));
            int s_offset = k_block_buffer_load_global_offset / 2;
            int seqlen_pos = k_warp_buffer_load_n_id * 4 + qk_lane_m_idx + 4 * WARP_N;
            if constexpr (not Is_even_MN) {
                seqlen_pos = min(seqlen_pos, max_seq_k_offset - 1);
            }
            int v_offset = seqlen_pos * seqlen_k_stride / 2 + qk_lane_head_dim_idx;
            int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
            inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
        }
    }

    // 先把第 6 阶段的 load 指令先发出去
    if constexpr (kBlockN >= (WARP_N * 6)) {
        int k_loop = 0;
        int k_block_buffer_load_global_offset = k_loop * kBlockK;
        int k_lds_stage_offset = (stage_id * STAGES + 1) * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
        for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
            int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
            int k_warp_buffer_load_lds_offset = k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32));
            int s_offset = k_block_buffer_load_global_offset / 2;
            int seqlen_pos = k_warp_buffer_load_n_id * 4 + qk_lane_m_idx + 4 * WARP_N;
            if constexpr (not Is_even_MN) {
                seqlen_pos = min(seqlen_pos, max_seq_k_offset - 1);
            }
            int v_offset = seqlen_pos * seqlen_k_stride / 2 + qk_lane_head_dim_idx;
            int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
            inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
        }
    }

    // 等待第 3, 4 阶段最后一波数据返回做计算
    if constexpr (kBlockN >= (WARP_N * 3)) {
        // 在 wait 之前提前计算好 lds load 的下标
        int precompute_k_lds_offset[2 * 2];
        vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
        int k_lds_stage_offset = (last_stage_id * STAGES) * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                for(int i = 0; i < 2; ++i) {
                    for(int j = 0; j < 2; ++j) {
                        precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * (WARP_N * 17) + n_idx * (32 * 17) + j * 4 + i * 32 + k_ds_read_offset) * 4;
                    }
                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (kBlockN >= (WARP_N * 5)) {
            buffer_load_lds_dwordx1_wait<3 * K_LOAD_REQUESTS>();
        } else {
            buffer_load_lds_dwordx1_wait<1 * K_LOAD_REQUESTS>();
        }
        __builtin_amdgcn_sched_barrier(0);

        // lds -> vgpr
        #pragma unroll
        for(int i = 0; i < 2; ++i) {
            #pragma unroll
            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                    for(int j = 0; j < 2; ++j) {
                        int lds_offset = precompute_k_lds_offset[i * 2 + j];
                        inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
                    }
                }
            }
        }
        if constexpr (kBlockN >= (WARP_N * 4)) {
            for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    for(int i = 0; i < 2; ++i) {
                        for(int j = 0; j < 2; ++j) {
                            precompute_k_lds_offset[i * 2 + j] += 1 * (WARP_N / 32) * (kBlockK / 32) * (32 * 17) * 4;
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            if constexpr (kBlockN >= (WARP_N * 5)) {
                buffer_load_lds_dwordx1_wait < 2 * K_LOAD_REQUESTS>();
            } else {
                buffer_load_lds_dwordx1_wait<0 * K_LOAD_REQUESTS>();
            }
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for(int i = 0; i < 2; ++i) {
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int j = 0; j < 2; ++j) {
                            int lds_offset = precompute_k_lds_offset[i * 2 + j];
                            inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg_tmp[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
                        }
                    }
                }
            }
        }
        asm volatile("s_waitcnt lgkmcnt(6)");
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_setprio 1");
        {
            int min_tile_n = 0;
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                #pragma unroll
                for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                int k_loop_idx = QK_LOOP_COUNT - 1;
                                int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
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
        asm volatile("s_waitcnt lgkmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        {
            int min_tile_n = 1;
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                #pragma unroll
                for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                int k_loop_idx = QK_LOOP_COUNT - 1;
                                int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
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

        // 等第 4 阶段 load 指令的最后一波数据回来
        if constexpr (kBlockN >= (WARP_N * 4)) {
            asm volatile("s_waitcnt lgkmcnt(2)");
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 1");
            {
                int min_tile_n = 0;
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = QK_LOOP_COUNT - 1;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/3 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/3 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            {
                int min_tile_n = 1;
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = QK_LOOP_COUNT - 1;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/3 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
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

    // 第 5, 6 阶段主循环
    if constexpr (kBlockN >= (WARP_N * 5)) {
        // 切到 load 数据的轮次
        stage_id ^= 1;
        for(int k_loop = K_LOOP_START; k_loop < QK_LOOP_COUNT; ++k_loop) {
            // 发第 5 阶段的下一个轮次的 load 请求
            {
                int k_block_buffer_load_global_offset = k_loop * kBlockK;
                int k_lds_stage_offset = (stage_id * STAGES) * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
                for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                    int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
                    int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
                    int k_warp_buffer_load_lds_offset =  k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32)) ; ;
                    int s_offset = k_block_buffer_load_global_offset / 2;
                    int seqlen_pos = k_warp_buffer_load_n_id * 4 + qk_lane_m_idx + 4 * WARP_N;
                    if constexpr (not Is_even_MN) {
                        seqlen_pos = min(seqlen_pos, max_seq_k_offset - 1);
                    }
                    int v_offset = seqlen_pos * seqlen_k_stride / 2 + qk_lane_head_dim_idx;
                    int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
                    inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
                }
            }
            
            // 提前把第 6 阶段的 load 发下去, 放 load 指令飞的更久一点
            if constexpr (kBlockN >= (WARP_N * 6)) {
                int k_block_buffer_load_global_offset = k_loop * kBlockK;
                int k_lds_stage_offset = (stage_id * STAGES + 1) * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
                for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                    int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
                    int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
                    int k_warp_buffer_load_lds_offset =  k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32)) ; ;
                    int s_offset = k_block_buffer_load_global_offset / 2;
                    int seqlen_pos = k_warp_buffer_load_n_id * 4 + qk_lane_m_idx + 4 * WARP_N;
                    if constexpr (not Is_even_MN) {
                        seqlen_pos = min(seqlen_pos, max_seq_k_offset - 1);
                    }
                    int v_offset = seqlen_pos * seqlen_k_stride / 2 + qk_lane_head_dim_idx;
                    int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
                    inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
                }
            }

            // 切到计算的轮次
            stage_id ^= 1;

            // 在 wait 前先计算好 lds offset
            int precompute_k_lds_offset[2 * 2];
            int k_lds_stage_offset = (stage_id * STAGES) * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
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
            buffer_load_lds_dwordx1_wait<3 * K_LOAD_REQUESTS>();
            __builtin_amdgcn_sched_barrier(0);

            // lds -> vgpr
            #pragma unroll
            for(int i = 0; i < 2; ++i) {
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int j = 0; j < 2; ++j) {
                            int lds_offset = precompute_k_lds_offset[i * 2 + j];
                            inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
                        }
                    }
                }
            }
            if constexpr (kBlockN >= (WARP_N * 4)) {
                for(int i = 0; i < 2; ++i) {
                    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                            for(int j = 0; j < 2; ++j) {
                                precompute_k_lds_offset[i * 2 + j] += 1 * (WARP_N / 32) * (kBlockK / 32) * (32 * 17) * 4;
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                buffer_load_lds_dwordx1_wait < 2 * K_LOAD_REQUESTS>();
                __builtin_amdgcn_sched_barrier(0);

                #pragma unroll
                for(int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                        #pragma unroll
                        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                            #pragma unroll
                            for(int j = 0; j < 2; ++j) {
                                int lds_offset = precompute_k_lds_offset[i * 2 + j];
                                inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg_tmp[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
                            }
                        }
                    }
                }
            }
            asm volatile("s_waitcnt lgkmcnt(6)");
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 1");
            {
                int min_tile_n = 0;
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = k_loop - 1;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/4 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/4 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("s_waitcnt lgkmcnt(4)");
            __builtin_amdgcn_sched_barrier(0);
            {
                int min_tile_n = 1;
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = k_loop - 1;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/4 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/4 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("s_setprio 0");

            // 等待第 6 阶段的数据
            if constexpr (kBlockN >= (WARP_N * 6)) {
                asm volatile("s_waitcnt lgkmcnt(2)");
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 1");
                {
                    int min_tile_n = 0;
                    #pragma unroll
                    for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                        #pragma unroll
                        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                            #pragma unroll
                            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    #pragma unroll
                                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                        int k_loop_idx = k_loop - 1;
                                        int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                        int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                        s_reg[(/*n_loop*/5 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                            vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                                q_reg[q_tile_id][2 * min_tile_k][1],
                                                                q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                                q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                            vec4_Element<Element>{k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][0],
                                                                k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][1],
                                                                k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                                k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                            s_reg[(/*n_loop*/5 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                    }
                                }
                            }
                        }
                    }
                }
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_sched_barrier(0);
                {
                    int min_tile_n = 1;
                    #pragma unroll
                    for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                        #pragma unroll
                        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                            #pragma unroll
                            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    #pragma unroll
                                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                        int k_loop_idx = k_loop - 1;
                                        int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                        int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                        s_reg[(/*n_loop*/5 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                            vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                                q_reg[q_tile_id][2 * min_tile_k][1],
                                                                q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                                q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                            vec4_Element<Element>{k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][0],
                                                                k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][1],
                                                                k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                                k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                            s_reg[(/*n_loop*/5 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
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
    }

    // 保留第 5, 6 阶段最后一波数据实际的 stage_id
    last_stage_id = stage_id ^ 1;


    // 等待第 5, 6 阶段最后一波数据返回做计算
    if constexpr (kBlockN >= (WARP_N * 5)) {
        // 在 wait 之前提前计算好 lds load 的下标
        int precompute_k_lds_offset[2 * 2];
        vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
        int k_lds_stage_offset = (last_stage_id * STAGES) * (WARP_N / 32) * (kBlockK / 32) * (32 * 17);
        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                for(int i = 0; i < 2; ++i) {
                    for(int j = 0; j < 2; ++j) {
                        precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v2fp16) + (k_lds_stage_offset + head_dim_idx * (WARP_N * 17) + n_idx * (32 * 17) + j * 4 + i * 32 + k_ds_read_offset) * 4;
                    }
                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        buffer_load_lds_dwordx1_wait<1 * K_LOAD_REQUESTS>();
        __builtin_amdgcn_sched_barrier(0);

        // lds -> vgpr
        #pragma unroll
        for(int i = 0; i < 2; ++i) {
            #pragma unroll
            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                #pragma unroll
                for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int j = 0; j < 2; ++j) {
                        int lds_offset = precompute_k_lds_offset[i * 2 + j];
                        inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
                    }
                }
            }
        }
        if constexpr (kBlockN >= (WARP_N * 6)) {
            for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    for(int i = 0; i < 2; ++i) {
                        for(int j = 0; j < 2; ++j) {
                            precompute_k_lds_offset[i * 2 + j] += 1 * (WARP_N / 32) * (kBlockK / 32) * (32 * 17) * 4;
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            buffer_load_lds_dwordx1_wait<0 * K_LOAD_REQUESTS>();
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for(int i = 0; i < 2; ++i) {
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        for(int j = 0; j < 2; ++j) {
                            int lds_offset = precompute_k_lds_offset[i * 2 + j];
                            inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg_tmp[(head_dim_idx * (WARP_N / 32) + n_idx) * 2 + i].u64[j], 2);
                        }
                    }
                }
            }
        }
        asm volatile("s_waitcnt lgkmcnt(6)");
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_setprio 1");
        {
            int min_tile_n = 0;
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                #pragma unroll
                for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                int k_loop_idx = QK_LOOP_COUNT - 1;
                                int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[(/*n_loop*/4 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                        q_reg[q_tile_id][2 * min_tile_k][1],
                                                        q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                        q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                    vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                        k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                        k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                        k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                    s_reg[(/*n_loop*/4 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
        }
        asm volatile("s_waitcnt lgkmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        {
            int min_tile_n = 1;
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                #pragma unroll
                for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                    #pragma unroll
                    for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                int k_loop_idx = QK_LOOP_COUNT - 1;
                                int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[(/*n_loop*/4 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                    vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                        q_reg[q_tile_id][2 * min_tile_k][1],
                                                        q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                        q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                    vec4_Element<Element>{k_reg[k_tile_id].f16x2[2 * min_tile_k][0],
                                                        k_reg[k_tile_id].f16x2[2 * min_tile_k][1],
                                                        k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                        k_reg[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                    s_reg[(/*n_loop*/4 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
        }
        asm volatile("s_setprio 0");

        // 等第 6 阶段 load 指令的最后一波数据回来
        if constexpr (kBlockN >= (WARP_N * 6)) {
            asm volatile("s_waitcnt lgkmcnt(2)");
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 1");
            {
                int min_tile_n = 0;
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = QK_LOOP_COUNT - 1;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/5 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/5 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            {
                int min_tile_n = 1;
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
                        #pragma unroll
                        for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                #pragma unroll
                                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                    int k_loop_idx = QK_LOOP_COUNT - 1;
                                    int q_tile_id  = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id  = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[(/*n_loop*/5 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{q_reg[q_tile_id][2 * min_tile_k][0],
                                                            q_reg[q_tile_id][2 * min_tile_k][1],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][0],
                                                            q_reg[q_tile_id][2 * min_tile_k + 1][1]},
                                        vec4_Element<Element>{k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k][1],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][0],
                                                            k_reg_tmp[k_tile_id].f16x2[2 * min_tile_k + 1][1]},
                                        s_reg[(/*n_loop*/5 * (WARP_N / 32) + n_idx) * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
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
    __builtin_amdgcn_sched_barrier(0);

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
