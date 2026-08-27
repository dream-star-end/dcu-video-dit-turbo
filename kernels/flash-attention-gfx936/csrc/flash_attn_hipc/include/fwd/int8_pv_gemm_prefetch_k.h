#include "int8_pv_gemm_prefetch_k_headdim128.h"


template<bool PREFETCH_K, int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int STAGES, typename Element, typename Element_k, typename ElementAccum, bool Is_even_MN>
__forceinline__ __device__ void  int8_pv_gemm_prefetch_k(
        vec4_uint gV,
        vec4_uint gK,
        Element_k* v_lds,
        Element_k* k_lds,
        float scales_v[kBlockK / WARP_N][2][4],
        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockK / 32)][4],
        vec4_Accum<ElementAccum> pv_reg[(kHeadDimV / kBlockN) * (WARP_M / 32) * (kBlockN / 32)][4],
        int WARP_ID,
        int seqlen_k_stride,
        int seqlen_v_stride,
        int max_seq_kv_offset=-1) {

    static_assert(kBlockK >= 32, "Error: pv gemm kBlockK must be equal or greater than 32");
    static_assert(kBlockM >= WARP_M, "Error: pv gemm kBlockM must be equal or greater than WARP_M");
    static_assert(kBlockN == WARP_N, "Error: pv gemm kBlockN must be equal to WARP_N");
    constexpr int  WARP_NUM       = (kBlockM * kBlockN) / (WARP_M * WARP_N);
    constexpr bool IS_HEADDIM_128 = (kHeadDim == 128 and kHeadDimV == 128) or (kHeadDim == 64 and kHeadDimV == 64) or (kHeadDim == 192 and kHeadDimV == 192);

    union_vec2_f16x2<Element> v_reg[STAGES * (32 * WARP_N) / (32 * 32)][4];

    // 预先计算一些公共表达式
    int lane_id       = threadIdx.x & 63;
    int laneid_shfl_2 = lane_id >> 2; // 0 ~ 15, 4 个线程读取一行
    int laneid_shfl_3 = lane_id >> 3; // 0 ~ 7,  8 个线程读取一行
    int laneid_shfl_4 = lane_id >> 4; // 0 ~ 3,  16 个线程读取一行
    int laneid_shfl_5 = lane_id >> 5; // 0 ~ 1,  lds 读取时, 8x32的数据按照线程 [0, 16, 0, 16, 32, 48, 32, 48] 来读取, 每 32 个线程读取一个 4x32
    constexpr int NEXT_DWORD_OFFSET = 32; // 8x32 的数据, 一个 wave 每个线程读 4 个 half, 即 2 个 dword, 使用 ds_read2_b32 指令, 按照上面的读取方式, 第二个 dword 偏移 32 个 dword

#if defined(USE_BUFFER_LOAD_DWORDX4)
    // 对于 headdim 128 而言, 2 组 32x32 可以写成 4 个 warp 分别读取 8 个 half, 即 4x64x8, 可以使用 buffer_load_dwordx4
    // 对于其他 headdim 而言, 暂不做那么激进的优化, 只有 1 组 32x32, 最多用 buffer_load_dwordx2
    constexpr int WARP_K             = 32;
    constexpr int READ_ONCE_LINES    = IS_HEADDIM_128 ? 16: 8;                    // 一个 warp 一次读几行数据, loadx2, 每行 32 个元素需要 32 / (4) = 8 个线程
    constexpr int READ_ONCE_COUNT    = READ_ONCE_LINES * 32;                      // 一个 warp 一次 load 多少数据
    constexpr int V_LDS_LOAD_NUM     = IS_HEADDIM_128 ? (1 * kBlockN * WARP_K/*对于非 headdim128 结尾的选项这里需要填 1, 一次只取 1 个 32x32 块, pv_gemm_utils.h 也要跟着改*/) / READ_ONCE_COUNT: (kBlockN * WARP_K) / READ_ONCE_COUNT; // 整个 workgroup 要发多少 load 指令
    constexpr int V_LOAD_REQUESTS    = V_LDS_LOAD_NUM / WARP_NUM;                 // 每个 warp 要发多少 load 指令
    constexpr int READ_ELEMENT_COUNT = IS_HEADDIM_128 ? 8: 4;                     // 每个线程一次读取几个 half
    int v_lane_headdim_n_idx         = IS_HEADDIM_128 ? lane_id & 3: lane_id & 7; // 当前 lane 负责这个 warp 的第几个 dwordx4 或者 dwordx2
    // 为了解决 ds_read2_b32 的 bank 冲突, 需要交换 0-15 线程的一些读取地址, 4 个线程为一组 (0, 1, 2, 3 ---> 0, 2, 1, 3 的写入位置, 从而满足 ds_read2_b32 offset32 的要求)
    // 非 headdim 的话, 则交换 0-7 线程的一些读取地址, 4 个线程为一组
    int base                         = IS_HEADDIM_128 ? (laneid_shfl_2 & 0xc): (laneid_shfl_3 & 0x4); // 第几个4线程组的最小id
    int tail                         = IS_HEADDIM_128 ? (laneid_shfl_2 & 0x3): (laneid_shfl_3 & 0x3); // 线程组中的第几个线程
    int v_lane_seq_k_idx             = base + (tail & 1) * 2 + (tail >> 1);
    int v_ds_read_offset             = (laneid_shfl_5 * 4 + (laneid_shfl_4 & 1)) * 32 + (lane_id & 15) * 2; // 一次读写 8x32, 0-31 线程读取前面 4x32, 32-63 读取后面 4x32, 4x32按照线程 [0, 16, 0, 16] 这种方式来读取
    auto BUFFER_LOAD_FUNC            = IS_HEADDIM_128 ? &inline_buffer_load_dwordx4_lds<Element_k, 2>: &inline_buffer_load_dwordx2_lds<Element_k, 2>;
#else
    constexpr int WARP_K             = 32;
    constexpr int READ_ONCE_LINES    = 4;
    constexpr int READ_ONCE_COUNT    = READ_ONCE_LINES * 32;
    constexpr int V_LDS_LOAD_NUM     = (kBlockN * WARP_K) / READ_ONCE_COUNT;
    constexpr int V_LOAD_REQUESTS    = V_LDS_LOAD_NUM / WARP_NUM;
    constexpr int READ_ELEMENT_COUNT = 2;
    int v_lane_headdim_n_idx         = lane_id & 15;
    int v_lane_seq_k_idx             = (laneid_shfl_4 & 1) * 2 + laneid_shfl_5; // 0, 1, 2, 3 ---> 0, 2, 1, 3
    int v_ds_read_offset             = (laneid_shfl_5 * 4 + (laneid_shfl_4 & 1)) * 32 + (lane_id & 15) * 2; // 一次读写 8x32, 0-31 线程读取前面 4x32, 32-63 读取后面 4x32, 4x32按照线程 [0, 16, 0, 16] 这种方式来读取
    auto BUFFER_LOAD_FUNC            = &inline_buffer_load_dword_lds<Element_k, 2>;
#endif

    int stage_id = 0;

    {
        if constexpr (STAGES == 2) stage_id ^= 1;
        const int N_LOOP_START = (STAGES == 2) ? 1: 0;
        for(int n_loop = N_LOOP_START; n_loop < (kHeadDimV / kBlockN); n_loop++) {

            int v_block_buffer_load_global_offset = n_loop * kBlockN;

            for(int load = 0, warp_loop = WARP_ID; load < V_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                int v_warp_buffer_load_k_id = warp_loop / (kBlockN / 32);
                int s_offset = v_block_buffer_load_global_offset / 2;
                int seqlen_pos = v_lane_seq_k_idx + v_warp_buffer_load_k_id * READ_ONCE_LINES;
                if constexpr (not Is_even_MN) {
                    seqlen_pos = min(seqlen_pos, max_seq_kv_offset - 1);
                }
                int v_offset = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + seqlen_pos * seqlen_v_stride) / 2;
                int v_lds_offset = v_warp_buffer_load_k_id * READ_ONCE_COUNT / 2;
                BUFFER_LOAD_FUNC(v_lds + stage_id * WARP_K * kBlockN, gV, v_lds_offset, s_offset, v_offset);
            }

            // 把 ds_read 之前的一些计算挪到 wait 之前, 等待数据返回
            if constexpr (STAGES == 2) stage_id ^= 1;
            int precompute_v_lds_offset[4];
            vec2_Element<Element> *v_lds_v2fp16 =  (vec2_Element<Element> *)(v_lds);
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                    for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                        precompute_v_lds_offset[vec_idx] = reinterpret_cast<size_t>(v_lds_v2fp16) + ((stage_id * WARP_K * kBlockN + seq_idx * 32 * kBlockN + head_dim_idx * 32 * 32 + vec_idx * 8 * 32 + v_ds_read_offset) / 2) * 4;
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            #ifdef USE_PINGPANG_BUFFER
                if constexpr (STAGES == 2) {
                    buffer_load_lds_dwordx1_wait<V_LOAD_REQUESTS>();
                } else if constexpr (STAGES == 1) {
                    buffer_load_lds_dwordx1_wait<0>();
                }
            #else
                buffer_load_lds_dwordx1_wait<0>();
            #endif
            __builtin_amdgcn_sched_barrier(0);

            #pragma unroll
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                #pragma unroll
                for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                        inline_ds_read2_b32_no_wait_bytes(precompute_v_lds_offset[vec_idx], v_reg[stage_id * WARP_N * WARP_K / (32 * 32) + head_dim_idx * (WARP_K / 32) + seq_idx][vec_idx].u64, NEXT_DWORD_OFFSET);
                    }
                }
            }

            // 拆成两段, 一共发出去 8 条 ds_read 指令, 前 4 条指令 perm, 用来做 mmac，然后再等待后面 4 条指令的返回, 类似
            vec4_Element<Element> v_vgprs[(32 * WARP_N) / (32 * 32)][2];
            {
                constexpr int min_tile_k = 0;
                // 先把 p 寄存器需要的数据 v_pack 在一起
                vec4_Element<Element> p_vgprs[2];
                for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    p_vgprs[min_tile_m] = make_vec4_f16(
                        p_reg[0][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[0][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[0][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                        p_reg[0][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                   );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(2)");
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            int v_tile_id = stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                        #pragma unroll
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    int n_loop_idx = (STAGES == 2) ? n_loop - 1: n_loop;
                                    int pv_tile_id = n_loop_idx * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
                asm volatile("s_setprio 0");
            }
            // ds 和 vgpr 之间的 ping-pang buffer
            {
                constexpr int min_tile_k = 1;
                vec4_Element<Element> p_vgprs[2];
                for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    p_vgprs[min_tile_m] = make_vec4_f16(
                        p_reg[0][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[0][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[0][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                        p_reg[0][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                   );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            int v_tile_id = stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                        #pragma unroll
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    int n_loop_idx = (STAGES == 2) ? n_loop - 1: n_loop;
                                    int pv_tile_id = n_loop_idx * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
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

    int last_stage_id = stage_id ^ 1;


    // 先把第 2 阶段的 load 指令发下去
    if constexpr (kBlockK >= 64) {
        // stage_id = 0;
        if constexpr (STAGES == 2) {
            int n_loop = 0;
            int v_block_buffer_load_global_offset = n_loop * kBlockN;
            for(int load = 0, warp_loop = WARP_ID; load < V_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                int v_warp_buffer_load_k_id = warp_loop / (kBlockN / 32);
                int s_offset = v_block_buffer_load_global_offset / 2;
                int seqlen_pos = v_lane_seq_k_idx + v_warp_buffer_load_k_id * READ_ONCE_LINES + WARP_K;
                if constexpr (not Is_even_MN) {
                    seqlen_pos = min(seqlen_pos, max_seq_kv_offset - 1);
                }
                int v_offset = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + seqlen_pos * seqlen_v_stride) / 2;
                int v_lds_offset = v_warp_buffer_load_k_id * READ_ONCE_COUNT / 2;
                BUFFER_LOAD_FUNC(v_lds + stage_id * WARP_K * kBlockN, gV, v_lds_offset, s_offset, v_offset);
            }
        }
    }


    // 等待第 1 阶段的最后一波数据回来做计算
    if constexpr (STAGES == 2) {
        // stage_id ^= 1;
        int precompute_v_lds_offset[4];
        vec2_Element<Element> *v_lds_v2fp16 =  (vec2_Element<Element> *)(v_lds);
        for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
            for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                    precompute_v_lds_offset[vec_idx] = reinterpret_cast<size_t>(v_lds_v2fp16) + (last_stage_id * WARP_K * kBlockN + seq_idx * 32 * kBlockN + head_dim_idx * 32 * 32 + vec_idx * 8 * 32 + v_ds_read_offset) * 2;
                }
            }
        }
        // buffer_load_lds_dwordx1_wait<1, 1, 1>();
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (kBlockK >= 64) {
            buffer_load_lds_dwordx1_wait<V_LOAD_REQUESTS>();
        } else {
            buffer_load_lds_dwordx1_wait<0>();
        }
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
            #pragma unroll
            for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                #pragma unroll
                for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                    inline_ds_read2_b32_no_wait_bytes(precompute_v_lds_offset[vec_idx], v_reg[last_stage_id * WARP_N * WARP_K / (32 * 32) + head_dim_idx * (WARP_K / 32) + seq_idx][vec_idx].u64, NEXT_DWORD_OFFSET);
                }
            }
        }

        vec4_Element<Element> v_vgprs[(32 * WARP_N) / (32 * 32)][2];
        {
            constexpr int min_tile_k = 0;
            vec4_Element<Element> p_vgprs[2];
            for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                p_vgprs[min_tile_m] = make_vec4_f16(
                    p_reg[0][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                    p_reg[0][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                    p_reg[0][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                    p_reg[0][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
               );
            }
            asm volatile("s_setprio 1");
            asm volatile("s_waitcnt lgkmcnt(2)");
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                        int v_tile_id = last_stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                        v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                            v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                            v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                            v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                            v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                        };
                    }
                }
            }
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                int pv_tile_id = (kHeadDimV / kBlockN - 1) * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                    flash::mmac<Element, ElementAccum>(
                                        p_vgprs[min_tile_m],
                                        v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
            asm volatile("s_setprio 0");
        }
        {
            constexpr int min_tile_k = 1;
            vec4_Element<Element> p_vgprs[2];
            for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                p_vgprs[min_tile_m] = make_vec4_f16(
                    p_reg[0][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                    p_reg[0][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                    p_reg[0][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                    p_reg[0][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
               );
            }
            asm volatile("s_setprio 1");
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                        int v_tile_id = last_stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                        v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                            v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                            v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                            v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                            v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                        };
                    }
                }
            }
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                            #pragma unroll
                            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                int pv_tile_id = (kHeadDimV / kBlockN - 1) * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                    flash::mmac<Element, ElementAccum>(
                                        p_vgprs[min_tile_m],
                                        v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
            asm volatile("s_setprio 0");
            asm volatile("s_barrier ; sync before load in the coming round");
        }
    }


    // 第 2 阶段的主循环
    if constexpr (kBlockK >= 64) {

        if constexpr (STAGES == 2) stage_id ^= 1;
        constexpr int N_LOOP_START = (STAGES == 2) ? 1: 0;
        for(int n_loop = N_LOOP_START; n_loop < (kHeadDimV / kBlockN); n_loop++) {

            int v_block_buffer_load_global_offset = n_loop * kBlockN;

            for(int load = 0, warp_loop = WARP_ID; load < V_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                int v_warp_buffer_load_k_id = warp_loop / (kBlockN / 32);
                int s_offset = v_block_buffer_load_global_offset / 2;
                int seqlen_pos = v_lane_seq_k_idx + v_warp_buffer_load_k_id * READ_ONCE_LINES + WARP_K;
                if constexpr (not Is_even_MN) {
                    seqlen_pos = min(seqlen_pos, max_seq_kv_offset - 1);
                }
                int v_offset = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + seqlen_pos * seqlen_v_stride) / 2;
                int v_lds_offset = v_warp_buffer_load_k_id * READ_ONCE_COUNT / 2;
                BUFFER_LOAD_FUNC(v_lds + stage_id * WARP_K * kBlockN, gV, v_lds_offset, s_offset, v_offset);
            }

            if constexpr (STAGES == 2) stage_id ^= 1;

            int precompute_v_lds_offset[4];
            vec2_Element<Element> *v_lds_v2fp16 =  (vec2_Element<Element> *)(v_lds);
            // lds -> vgpr use ds_read_m; right matrix
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                    for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                        precompute_v_lds_offset[vec_idx] = reinterpret_cast<size_t>(v_lds_v2fp16) + (stage_id * WARP_K * kBlockN + seq_idx * 32 * kBlockN + head_dim_idx * 32 * 32 + vec_idx * 8 * 32 + v_ds_read_offset) * 2;
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            #ifdef USE_PINGPANG_BUFFER
                if constexpr (STAGES == 2) {
                    buffer_load_lds_dwordx1_wait<V_LOAD_REQUESTS>();
                } else if constexpr (STAGES == 1) {
                    buffer_load_lds_dwordx1_wait<0>();
                }
            #else
                buffer_load_lds_dwordx1_wait<0>();
            #endif
            __builtin_amdgcn_sched_barrier(0);

            #pragma unroll
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                #pragma unroll
                for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                        inline_ds_read2_b32_no_wait_bytes(precompute_v_lds_offset[vec_idx], v_reg[stage_id * WARP_N * WARP_K / (32 * 32) + head_dim_idx * (WARP_K / 32) + seq_idx][vec_idx].u64, NEXT_DWORD_OFFSET);
                    }
                }
            }

            vec4_Element<Element> v_vgprs[(32 * WARP_N) / (32 * 32)][2];
            {
                constexpr int min_tile_k = 0;
                vec4_Element<Element> p_vgprs[2];
                for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    p_vgprs[min_tile_m] = make_vec4_f16(
                        p_reg[1][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[1][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[1][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                        p_reg[1][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                   );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(2)");
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            int v_tile_id = stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                        #pragma unroll
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    int n_loop_idx = (STAGES == 2) ? n_loop - 1: n_loop;
                                    int pv_tile_id = n_loop_idx * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
                asm volatile("s_setprio 0");
            }
            {
                constexpr int min_tile_k = 1;
                vec4_Element<Element> p_vgprs[2];
                for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    p_vgprs[min_tile_m] = make_vec4_f16(
                        p_reg[1][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[1][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[1][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                        p_reg[1][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                   );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            int v_tile_id = stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                        #pragma unroll
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    int n_loop_idx = (STAGES == 2) ? n_loop - 1: n_loop;
                                    int pv_tile_id = n_loop_idx * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
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


    last_stage_id = stage_id ^ 1;


    // 先把第 3 阶段的 load 指令发下去
    if constexpr (kBlockK >= 96) {
        // stage_id = 0;
        if constexpr (STAGES == 2) {
            int n_loop = 0;
            int v_block_buffer_load_global_offset = n_loop * kBlockN;
            for(int load = 0, warp_loop = WARP_ID; load < V_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                int v_warp_buffer_load_k_id = warp_loop / (kBlockN / 32);
                int s_offset = v_block_buffer_load_global_offset / 2;
                int seqlen_pos = v_lane_seq_k_idx + v_warp_buffer_load_k_id * READ_ONCE_LINES + 2 * WARP_K;
                if constexpr (not Is_even_MN) {
                    seqlen_pos = min(seqlen_pos, max_seq_kv_offset - 1);
                }
                int v_offset = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + seqlen_pos * seqlen_v_stride) / 2;
                int v_lds_offset = v_warp_buffer_load_k_id * READ_ONCE_COUNT / 2;
                BUFFER_LOAD_FUNC(v_lds + stage_id * WARP_K * kBlockN, gV, v_lds_offset, s_offset, v_offset);
            }
        }
    }


    // 等待第 2 阶段的最后一波数据回来做计算
    if constexpr (kBlockK >= 64) {
        if constexpr (STAGES == 2) {
            // 在 wait 之前计算一些 ds_read 需要的下标
            // stage_id ^= 1;
            int precompute_v_lds_offset[4];
            vec2_Element<Element> *v_lds_v2fp16 =  (vec2_Element<Element> *)(v_lds);
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                    for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                        precompute_v_lds_offset[vec_idx] = reinterpret_cast<size_t>(v_lds_v2fp16) + (last_stage_id * WARP_K * kBlockN + seq_idx * 32 * kBlockN + head_dim_idx * 32 * 32 + vec_idx * 8 * 32 + v_ds_read_offset) * 2;
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            if constexpr (kBlockK >= 96) {
                buffer_load_lds_dwordx1_wait<V_LOAD_REQUESTS>();
            } else {
                buffer_load_lds_dwordx1_wait<0>();
            }
            __builtin_amdgcn_sched_barrier(0);

            #pragma unroll
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                #pragma unroll
                for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                        inline_ds_read2_b32_no_wait_bytes(precompute_v_lds_offset[vec_idx], v_reg[last_stage_id * WARP_N * WARP_K / (32 * 32) + head_dim_idx * (WARP_K / 32) + seq_idx][vec_idx].u64, NEXT_DWORD_OFFSET);
                    }
                }
            }

            vec4_Element<Element> v_vgprs[(32 * WARP_N) / (32 * 32)][2];
            {
                constexpr int min_tile_k = 0;
                vec4_Element<Element> p_vgprs[2];
                for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    p_vgprs[min_tile_m] = make_vec4_f16(
                        p_reg[1][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[1][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[1][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                        p_reg[1][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                   );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(2)");
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            int v_tile_id = last_stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                        #pragma unroll
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    int pv_tile_id = (kHeadDimV / kBlockN - 1) * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
                asm volatile("s_setprio 0");
            }
            // ping-pang buffer
            {
                constexpr int min_tile_k = 1;
                vec4_Element<Element> p_vgprs[2];
                for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    p_vgprs[min_tile_m] = make_vec4_f16(
                        p_reg[1][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[1][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[1][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                        p_reg[1][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                   );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            int v_tile_id = last_stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                        #pragma unroll
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    int pv_tile_id = (kHeadDimV / kBlockN - 1) * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
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

    // 第 3 阶段的主循环
    if constexpr (kBlockK >= 96) {

        if constexpr (STAGES == 2) stage_id ^= 1;
        constexpr int N_LOOP_START = (STAGES == 2) ? 1: 0;
        for(int n_loop = N_LOOP_START; n_loop < (kHeadDimV / kBlockN); n_loop++) {

            int v_block_buffer_load_global_offset = n_loop * kBlockN;
            for(int load = 0, warp_loop = WARP_ID; load < V_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                int v_warp_buffer_load_k_id = warp_loop / (kBlockN / 32);
                int s_offset = v_block_buffer_load_global_offset / 2;
                int seqlen_pos = v_lane_seq_k_idx + v_warp_buffer_load_k_id * READ_ONCE_LINES + 2 * WARP_K;
                if constexpr (not Is_even_MN) {
                    seqlen_pos = min(seqlen_pos, max_seq_kv_offset - 1);
                }
                int v_offset = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + seqlen_pos * seqlen_v_stride) / 2;
                int v_lds_offset = v_warp_buffer_load_k_id * READ_ONCE_COUNT / 2;
                BUFFER_LOAD_FUNC(v_lds + stage_id * WARP_K * kBlockN, gV, v_lds_offset, s_offset, v_offset);
            }

            if constexpr (STAGES == 2) stage_id ^= 1;

            int precompute_v_lds_offset[4];
            vec2_Element<Element> *v_lds_v2fp16 =  (vec2_Element<Element> *)(v_lds);
            // lds -> vgpr use ds_read_m; right matrix
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                    for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                        precompute_v_lds_offset[vec_idx] = reinterpret_cast<size_t>(v_lds_v2fp16) + (stage_id * WARP_K * kBlockN + seq_idx * 32 * kBlockN + head_dim_idx * 32 * 32 + vec_idx * 8 * 32 + v_ds_read_offset) * 2;
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            #ifdef USE_PINGPANG_BUFFER
                if constexpr (STAGES == 2) {
                    buffer_load_lds_dwordx1_wait<V_LOAD_REQUESTS>();
                } else if constexpr (STAGES == 1) {
                    buffer_load_lds_dwordx1_wait<0>();
                }
            #else
                buffer_load_lds_dwordx1_wait<0>();
            #endif
            __builtin_amdgcn_sched_barrier(0);

            #pragma unroll
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                #pragma unroll
                for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                        inline_ds_read2_b32_no_wait_bytes(precompute_v_lds_offset[vec_idx], v_reg[stage_id * WARP_N * WARP_K / (32 * 32) + head_dim_idx * (WARP_K / 32) + seq_idx][vec_idx].u64, NEXT_DWORD_OFFSET);
                    }
                }
            }

            vec4_Element<Element> v_vgprs[(32 * WARP_N) / (32 * 32)][2];
            {
                constexpr int min_tile_k = 0;
                vec4_Element<Element> p_vgprs[2];
                for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    p_vgprs[min_tile_m] = make_vec4_f16(
                        p_reg[2][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[2][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[2][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                        p_reg[2][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                   );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(2)");
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            int v_tile_id = stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                        #pragma unroll
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    int n_loop_idx = (STAGES == 2) ? n_loop - 1: n_loop;
                                    int pv_tile_id = n_loop_idx * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
                asm volatile("s_setprio 0");
            }
            {
                constexpr int min_tile_k = 1;
                vec4_Element<Element> p_vgprs[2];
                for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    p_vgprs[min_tile_m] = make_vec4_f16(
                        p_reg[2][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[2][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[2][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                        p_reg[2][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                   );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            int v_tile_id = stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                        #pragma unroll
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    int n_loop_idx = (STAGES == 2) ? n_loop - 1: n_loop;
                                    int pv_tile_id = n_loop_idx * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
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


    last_stage_id = stage_id ^ 1;

    // 先把第 4 阶段的 load 指令发下去
    if constexpr (kBlockK >= 128) {
        // stage_id = 0;
        if constexpr (STAGES == 2) {
            int n_loop = 0;
            int v_block_buffer_load_global_offset = n_loop * kBlockN;
            for(int load = 0, warp_loop = WARP_ID; load < V_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                int v_warp_buffer_load_k_id = warp_loop / (kBlockN / 32);
                int s_offset = v_block_buffer_load_global_offset / 2;
                int seqlen_pos = v_lane_seq_k_idx + v_warp_buffer_load_k_id * READ_ONCE_LINES + 3 * WARP_K;
                if constexpr (not Is_even_MN) {
                    seqlen_pos = min(seqlen_pos, max_seq_kv_offset - 1);
                }
                int v_offset = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + seqlen_pos * seqlen_v_stride) / 2;
                int v_lds_offset = v_warp_buffer_load_k_id * READ_ONCE_COUNT / 2;
                BUFFER_LOAD_FUNC(v_lds + stage_id * WARP_K * kBlockN, gV, v_lds_offset, s_offset, v_offset);
            }
        }
    }

    // 等待第 3 阶段的最后一波数据回来做计算
    if constexpr (kBlockK >= 96) {
        if constexpr (STAGES == 2) {
            // 在 wait 之前计算一些 ds_read 需要的下标
            // stage_id ^= 1;
            int precompute_v_lds_offset[4];
            vec2_Element<Element> *v_lds_v2fp16 =  (vec2_Element<Element> *)(v_lds);
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                    for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                        precompute_v_lds_offset[vec_idx] = reinterpret_cast<size_t>(v_lds_v2fp16) + (last_stage_id * WARP_K * kBlockN + seq_idx * 32 * kBlockN + head_dim_idx * 32 * 32 + vec_idx * 8 * 32 + v_ds_read_offset) * 2;
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            if constexpr (kBlockK >= 128) {
                buffer_load_lds_dwordx1_wait<V_LOAD_REQUESTS>();
            } else {
                buffer_load_lds_dwordx1_wait<0>();
            }
            __builtin_amdgcn_sched_barrier(0);

            #pragma unroll
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                #pragma unroll
                for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                        inline_ds_read2_b32_no_wait_bytes(precompute_v_lds_offset[vec_idx], v_reg[last_stage_id * WARP_N * WARP_K / (32 * 32) + head_dim_idx * (WARP_K / 32) + seq_idx][vec_idx].u64, NEXT_DWORD_OFFSET);
                    }
                }
            }

            vec4_Element<Element> v_vgprs[(32 * WARP_N) / (32 * 32)][2];
            {
                constexpr int min_tile_k = 0;
                vec4_Element<Element> p_vgprs[2];
                for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    p_vgprs[min_tile_m] = make_vec4_f16(
                        p_reg[2][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[2][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[2][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                        p_reg[2][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                   );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(2)");
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            int v_tile_id = last_stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                        #pragma unroll
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    int pv_tile_id = (kHeadDimV / kBlockN - 1) * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
                asm volatile("s_setprio 0");
            }
            // ping-pang buffer
            {
                constexpr int min_tile_k = 1;
                vec4_Element<Element> p_vgprs[2];
                for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    p_vgprs[min_tile_m] = make_vec4_f16(
                        p_reg[2][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[2][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[2][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                        p_reg[2][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                   );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            int v_tile_id = last_stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                        #pragma unroll
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    int pv_tile_id = (kHeadDimV / kBlockN - 1) * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
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

    // 第 4 阶段的主循环
    if constexpr (kBlockK >= 128) {

        if constexpr (STAGES == 2) stage_id ^= 1;
        constexpr int N_LOOP_START = (STAGES == 2) ? 1: 0;
        for(int n_loop = N_LOOP_START; n_loop < (kHeadDimV / kBlockN); n_loop++) {

            int v_block_buffer_load_global_offset = n_loop * kBlockN;

            for(int load = 0, warp_loop = WARP_ID; load < V_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                int v_warp_buffer_load_k_id = warp_loop / (kBlockN / 32);
                int s_offset = v_block_buffer_load_global_offset / 2;
                int seqlen_pos = v_lane_seq_k_idx + v_warp_buffer_load_k_id * READ_ONCE_LINES + 3 * WARP_K;
                if constexpr (not Is_even_MN) {
                    seqlen_pos = min(seqlen_pos, max_seq_kv_offset - 1);
                }
                int v_offset = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + seqlen_pos * seqlen_v_stride) / 2;
                int v_lds_offset = v_warp_buffer_load_k_id * READ_ONCE_COUNT / 2;
                BUFFER_LOAD_FUNC(v_lds + stage_id * WARP_K * kBlockN, gV, v_lds_offset, s_offset, v_offset);
            }

            if constexpr (STAGES == 2) stage_id ^= 1;

            int precompute_v_lds_offset[4];
            vec2_Element<Element> *v_lds_v2fp16 =  (vec2_Element<Element> *)(v_lds);
            // lds -> vgpr use ds_read_m; right matrix
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                    for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                        precompute_v_lds_offset[vec_idx] = reinterpret_cast<size_t>(v_lds_v2fp16) + (stage_id * WARP_K * kBlockN + seq_idx * 32 * kBlockN + head_dim_idx * 32 * 32 + vec_idx * 8 * 32 + v_ds_read_offset) * 2;
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            #ifdef USE_PINGPANG_BUFFER
                if constexpr (STAGES == 2) {
                    buffer_load_lds_dwordx1_wait<V_LOAD_REQUESTS>();
                } else if constexpr (STAGES == 1) {
                    buffer_load_lds_dwordx1_wait<0>();
                }
            #else
                buffer_load_lds_dwordx1_wait<0>();
            #endif
            __builtin_amdgcn_sched_barrier(0);


            #pragma unroll
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                #pragma unroll
                for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                        inline_ds_read2_b32_no_wait_bytes(precompute_v_lds_offset[vec_idx], v_reg[stage_id * WARP_N * WARP_K / (32 * 32) + head_dim_idx * (WARP_K / 32) + seq_idx][vec_idx].u64, NEXT_DWORD_OFFSET);
                    }
                }
            }

            vec4_Element<Element> v_vgprs[(32 * WARP_N) / (32 * 32)][2];
            {
                constexpr int min_tile_k = 0;
                vec4_Element<Element> p_vgprs[2];
                for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    p_vgprs[min_tile_m] = make_vec4_f16(
                        p_reg[3][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[3][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[3][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                        p_reg[3][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                   );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(2)");
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            int v_tile_id = stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                        #pragma unroll
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    int n_loop_idx = (STAGES == 2) ? n_loop - 1: n_loop;
                                    int pv_tile_id = n_loop_idx * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
                asm volatile("s_setprio 0");
            }
            {
                constexpr int min_tile_k = 1;
                vec4_Element<Element> p_vgprs[2];
                for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    p_vgprs[min_tile_m] = make_vec4_f16(
                        p_reg[3][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[3][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[3][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                        p_reg[3][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                   );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            int v_tile_id = stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                        #pragma unroll
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    int n_loop_idx = (STAGES == 2) ? n_loop - 1: n_loop;
                                    int pv_tile_id = n_loop_idx * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
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

    // 先把 load K 数据的指令发下去
    if constexpr (PREFETCH_K) {
        Is_even_MN
        ? int8_prefetch_k_to_lds<kHeadDim, kBlockK, kBlockN, WARP_NUM, WARP_N, Element_k, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride)
        : int8_prefetch_k_to_lds<kHeadDim, kBlockK, kBlockN, WARP_NUM, WARP_N, Element_k, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride, max_seq_kv_offset);
    }

    // 等待第 4 阶段的最后一波数据回来做计算
    if constexpr (kBlockK >= 128) {
        if constexpr (STAGES == 2) {
            // 在 wait 之前计算一些 ds_read 需要的下标
            stage_id ^= 1;
            int precompute_v_lds_offset[4];
            vec2_Element<Element> *v_lds_v2fp16 =  (vec2_Element<Element> *)(v_lds);
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                    for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                        precompute_v_lds_offset[vec_idx] = reinterpret_cast<size_t>(v_lds_v2fp16) + (stage_id * WARP_K * kBlockN + seq_idx * 32 * kBlockN + head_dim_idx * 32 * 32 + vec_idx * 8 * 32 + v_ds_read_offset) * 2;
                    }
                }
            }
            if constexpr (PREFETCH_K) {
                constexpr int k_lds_load_num  = (WARP_N * kBlockN) / (4 * 64);
                constexpr int K_LOAD_REQUESTS = k_lds_load_num / WARP_NUM;
                buffer_load_lds_dwordx1_wait<K_LOAD_REQUESTS>();
            } else {
                buffer_load_lds_dwordx1_wait<0>();
            }

            #pragma unroll
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                #pragma unroll
                for(int seq_idx = 0; seq_idx < (WARP_K / 32); ++seq_idx) {
                    #pragma unroll
                    for(int head_dim_idx = 0; head_dim_idx < (WARP_N / 32); ++head_dim_idx) {
                        inline_ds_read2_b32_no_wait_bytes(precompute_v_lds_offset[vec_idx], v_reg[stage_id * WARP_N * WARP_K / (32 * 32) + head_dim_idx * (WARP_K / 32) + seq_idx][vec_idx].u64, NEXT_DWORD_OFFSET);
                    }
                }
            }

            vec4_Element<Element> v_vgprs[(32 * WARP_N) / (32 * 32)][2];
            {
                constexpr int min_tile_k = 0;
                vec4_Element<Element> p_vgprs[2];
                for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    p_vgprs[min_tile_m] = make_vec4_f16(
                        p_reg[3][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[3][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[3][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                        p_reg[3][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                   );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(2)");
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            int v_tile_id = stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                        #pragma unroll
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    int pv_tile_id = (kHeadDimV / kBlockN - 1) * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
                asm volatile("s_setprio 0");
            }
            // ping-pang buffer
            {
                constexpr int min_tile_k = 1;
                vec4_Element<Element> p_vgprs[2];
                for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    p_vgprs[min_tile_m] = make_vec4_f16(
                        p_reg[3][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[3][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                        p_reg[3][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                        p_reg[3][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                   );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                            int v_tile_id = stage_id * WARP_N * WARP_K / (32 * 32) + n_idx * (WARP_K / 32) + k_idx;
                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                                v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                    #pragma unroll
                    for(int k_idx = 0; k_idx < (WARP_K / 32); ++k_idx) {
                        #pragma unroll
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                                #pragma unroll
                                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                                    int pv_tile_id = (kHeadDimV / kBlockN - 1) * (WARP_M / 32) * (kBlockN / 32) + n_idx * (WARP_M / 32) + m_idx;
                                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx * (WARP_K / 32) + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
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
}
