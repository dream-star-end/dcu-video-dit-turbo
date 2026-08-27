#include "mla_pv_gemm_utils.h"


template<int K_LOOP_COUNT, int kBlockM, int kBlockN, int kBlockK, int M_WARP_COUNT, int PV_N_WARP_COUNT, int PV_K_WARP_COUNT, int STAGES, int WARP_NUM, int M_MMAC_COUNT, typename Element, typename ElementAccum>
__forceinline__ __device__ void mla_pv_gemm_prefetch_k(
        vec4_uint v_addr,
        vec4_uint k_addr,
        Element* v_lds,
        Element* k_lds,
        union_vec2_f16x2<Element> p_reg[M_WARP_COUNT * PV_K_WARP_COUNT][4],
        vec4_Accum<ElementAccum> pv_reg[K_LOOP_COUNT * M_WARP_COUNT * (kBlockN / 32)][4],
        int warp_id,
        int kvcache_seqlen_stride,
        int max_seq_kv_offset=-1) {

    constexpr int WARP_K = PV_K_WARP_COUNT * 32;
    static_assert(kBlockK >= 32, "Error: pv gemm kBlockK must be equal or greater than 32");
    static_assert(kBlockN == PV_N_WARP_COUNT * 32, "Error: kBlockN in mla_pv_gemm_prefetch_k must be WARP_N * 32");

    union_vec2_f16x2<Element> v_reg[STAGES * PV_K_WARP_COUNT * PV_N_WARP_COUNT][4];

    // 预先计算一些公共表达式
    int lane_id       = threadIdx.x & 63;
    int laneid_shfl_2 = lane_id >> 2; // 0 ~ 15, 4 个线程读取一行
    int laneid_shfl_3 = lane_id >> 3; // 0 ~ 7,  8 个线程读取一行
    int laneid_shfl_4 = lane_id >> 4; // 0 ~ 3,  16 个线程读取一行
    int laneid_shfl_5 = lane_id >> 5; // 0 ~ 1,  lds 读取时, 8x32的数据按照线程 [0, 16, 0, 16, 32, 48, 32, 48] 来读取, 每 32 个线程读取一个 4x32
    constexpr int NEXT_DWORD_OFFSET = 32; // 8x32 的数据, 一个 wave 每个线程读 4 个 half, 即 2 个 dword, 使用 ds_read2_b32 指令, 按照上面的读取方式, 第二个 dword 偏移 32 个 dword

#if defined(USE_BUFFER_LOAD_DWORDX4)
    constexpr int READ_ONCE_LINES    = 16;                                 // 一个 warp 每次读几行数据, loadx4, 每个线程读取 8 个 Half, 每行 32 个 Half 需要 32 / 8 = 4 个线程, 所以一个 wave 64 线程会读取 16 行
    constexpr int READ_ONCE_COUNT    = READ_ONCE_LINES * 32;               // 一个 warp 每次 load 多少数据, 16x32
    constexpr int V_LDS_LOAD_NUM     = kBlockN * WARP_K / READ_ONCE_COUNT; // 一个 warp 一共要发几次读取请求
    constexpr int V_LOAD_REQUESTS    = V_LDS_LOAD_NUM;                     // 一个 warp 一共要发几次读取请求
    constexpr int READ_ELEMENT_COUNT = 8;                                  // 每个线程一次读取几个 Half
    int v_lane_headdim_n_idx         = lane_id & 3;                        // 当前 lane 负责这个 warp 的第几个 dwordx2
    int base                         = (laneid_shfl_2 & 0xc);              // 第几个 4 线程组的最小id
    int tail                         = (laneid_shfl_2 & 0x3);              // 4 线程组中的第几个线程
    int v_lane_seq_k_idx             = base + (tail & 1) * 2 + (tail >> 1);// global -> lds, seqlen 方向的坐标
    int v_ds_read_offset             = (laneid_shfl_5 * 4 + (laneid_shfl_4 & 1)) * 32 + (lane_id & 15) * 2; // 一次读写 8x32, 0-31 线程读取前面 4x32, 32-63 读取后面 4x32, 4x32按照线程 [0, 16, 0, 16] 这种方式来读取
    auto BUFFER_LOAD_FUNC            = &inline_buffer_load_dwordx4_lds<Element, 2>;
#else
    constexpr int READ_ONCE_LINES    = 4;
    constexpr int READ_ONCE_COUNT    = READ_ONCE_LINES * 32;
    constexpr int V_LDS_LOAD_NUM     = (kBlockN * WARP_K) / READ_ONCE_COUNT;
    constexpr int V_LOAD_REQUESTS    = V_LDS_LOAD_NUM;
    constexpr int READ_ELEMENT_COUNT = 2;
    int v_lane_headdim_n_idx         = lane_id & 15;
    int v_lane_seq_k_idx             = (laneid_shfl_4 & 1) * 2 + laneid_shfl_5; // 0, 1, 2, 3 ---> 0, 2, 1, 3
    int v_ds_read_offset             = (laneid_shfl_5 * 4 + (laneid_shfl_4 & 1)) * 32 + (lane_id & 15) * 2; // 一次读写 8x32, 0-31 线程读取前面 4x32, 32-63 读取后面 4x32, 4x32按照线程 [0, 16, 0, 16] 这种方式来读取
    auto BUFFER_LOAD_FUNC            = &inline_buffer_load_dword_lds<Element, 2>;
#endif

    // each wave need 2 32x32 lds space
    v_lds = v_lds + warp_id * STAGES * WARP_K * kBlockN;

    int stage_id = (STAGES == 2) ? 1: 0;

    constexpr int N_LOOP_START = (STAGES == 2) ? 1: 0;
    for (int n_loop = N_LOOP_START; n_loop < K_LOOP_COUNT; ++n_loop) {

        int v_block_buffer_load_global_offset = warp_id * WARP_K * kvcache_seqlen_stride + n_loop * kBlockN;
        for (int load = 0; load < V_LOAD_REQUESTS; ++load) {
            int v_warp_buffer_load_k_id = (load + warp_id) % V_LOAD_REQUESTS;
            int v_warp_buffer_load_lds_offset = load * READ_ONCE_COUNT;
            int v_gvoffset_s = v_block_buffer_load_global_offset / 2;
            int v_gvoffset_v = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + min(v_lane_seq_k_idx + load * READ_ONCE_LINES, max_seq_kv_offset - 1) * kvcache_seqlen_stride) / 2;
            int v_lds_offset = v_warp_buffer_load_lds_offset / 2;
            BUFFER_LOAD_FUNC(v_lds + stage_id * WARP_K * kBlockN, v_addr, v_lds_offset, v_gvoffset_s, v_gvoffset_v);
        }

        if constexpr (STAGES == 2) stage_id ^= 1;

        // 把 ds_read 之前的一些计算挪到 wait 之前, 等待数据返回
        int precompute_v_lds_offset[4];
        vec2_Element<Element> *v_lds_v2fp16 = (vec2_Element<Element> *)(v_lds);
        for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
            for (int seq_idx = 0; seq_idx < PV_K_WARP_COUNT; ++seq_idx) {
                for (int head_dim_idx = 0; head_dim_idx < PV_N_WARP_COUNT; ++head_dim_idx) {
                    precompute_v_lds_offset[vec_idx] = reinterpret_cast<size_t>(v_lds_v2fp16) + ((stage_id * WARP_K * kBlockN + seq_idx * 32 * kBlockN + head_dim_idx * 32 * 32 + vec_idx * 8 * 32 + v_ds_read_offset) / 2) * 4/*4 bytes per dword*/;
                }
            }
        }

        __builtin_amdgcn_sched_barrier(0);
        #ifdef USE_PINGPANG_BUFFER
            if constexpr (STAGES == 2) {
                buffer_load_lds_dwordx1_wait_nosync<V_LOAD_REQUESTS>();
            } else if constexpr (STAGES == 1) {
                buffer_load_lds_dwordx1_wait_nosync<0>();
            }
        #else
            buffer_load_lds_dwordx1_wait_nosync<0>();
        #endif
        __builtin_amdgcn_sched_barrier(0);

        #pragma unroll
        for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
            #pragma unroll
            for (int seq_idx = 0; seq_idx < PV_K_WARP_COUNT; ++seq_idx) {
                #pragma unroll
                for (int head_dim_idx = 0; head_dim_idx < PV_N_WARP_COUNT; ++head_dim_idx) {
                    inline_ds_read2_b32_no_wait_bytes(precompute_v_lds_offset[vec_idx], v_reg[stage_id * PV_K_WARP_COUNT * PV_N_WARP_COUNT + (head_dim_idx * PV_K_WARP_COUNT + seq_idx)][vec_idx].u64, NEXT_DWORD_OFFSET);
                }
            }
        }

        // 拆成两段, 一共发出去 8 条 ds_read 指令, 前 4 条指令 perm, 用来做 mmac，然后再等待后面 4 条指令的返回, 类似
        vec4_Element<Element> v_vgprs[PV_K_WARP_COUNT * PV_N_WARP_COUNT][2];
        {
            constexpr int min_tile_k = 0;
            // 先把 p 寄存器需要的数据 v_pack 在一起
            vec4_Element<Element> p_vgprs[2];
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                p_vgprs[min_tile_m] = make_vec4_f16(
                    p_reg[0][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0/*vec_idx*/],
                    p_reg[0][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                    p_reg[0][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                    p_reg[0][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                );
            }
            asm volatile("s_setprio 1");
            asm volatile("s_waitcnt lgkmcnt(2)");
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for (int k_idx = 0; k_idx < PV_K_WARP_COUNT; ++k_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < PV_N_WARP_COUNT; ++n_idx) {
                    int v_tile_id = stage_id * PV_K_WARP_COUNT * PV_N_WARP_COUNT + n_idx * PV_K_WARP_COUNT + k_idx;
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        v_vgprs[n_idx * PV_K_WARP_COUNT + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                            v_reg[v_tile_id][0 + min_tile_k * 2/*vec_idx*/].f16x2[0][min_tile_n],
                            v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                            v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                            v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                        };
                    }
                }
            }
            #pragma unroll
            for (int m_idx = 0; m_idx < M_WARP_COUNT; ++m_idx) {
                #pragma unroll
                for (int k_idx = 0; k_idx < PV_K_WARP_COUNT; ++k_idx) {
                    #pragma unroll
                    for (int n_idx = 0; n_idx < PV_N_WARP_COUNT; ++n_idx) {
                        int n_loop_idx = (STAGES == 2) ? n_loop - 1: n_loop;
                        int pv_tile_id = n_loop_idx * M_WARP_COUNT * PV_N_WARP_COUNT + n_idx * M_WARP_COUNT + m_idx;
                        #pragma unroll
                        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = flash::mmac<Element, ElementAccum>(
                                    p_vgprs[min_tile_m],
                                    v_vgprs[n_idx * PV_K_WARP_COUNT + k_idx][/*min_tile_k * 2 + */min_tile_n],
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
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
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
            for (int k_idx = 0; k_idx < PV_K_WARP_COUNT; ++k_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < PV_N_WARP_COUNT; ++n_idx) {
                    int v_tile_id = stage_id * PV_K_WARP_COUNT * PV_N_WARP_COUNT + n_idx * PV_K_WARP_COUNT + k_idx;
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        v_vgprs[n_idx * PV_K_WARP_COUNT + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                            v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[0][min_tile_n],
                            v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                            v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                            v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                        };
                    }
                }
            }
            #pragma unroll
            for (int m_idx = 0; m_idx < M_WARP_COUNT; ++m_idx) {
                #pragma unroll
                for (int k_idx = 0; k_idx < PV_K_WARP_COUNT; ++k_idx) {
                    #pragma unroll
                    for (int n_idx = 0; n_idx < PV_N_WARP_COUNT; ++n_idx) {
                        int n_loop_idx = (STAGES == 2) ? n_loop - 1: n_loop;
                        int pv_tile_id = n_loop_idx * M_WARP_COUNT * PV_N_WARP_COUNT + n_idx * M_WARP_COUNT + m_idx;
                        #pragma unroll
                        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 =
                                    flash::mmac<Element, ElementAccum>(
                                        p_vgprs[min_tile_m],
                                        v_vgprs[n_idx * PV_K_WARP_COUNT + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
            asm volatile("s_setprio 0");
            // asm volatile("s_barrier ; sync before load in the coming round");
        }
    }

    if constexpr (STAGES == 2) {
        int n_loop = K_LOOP_COUNT - 1;
        stage_id ^= 1;
        // 把 ds_read 之前的一些计算挪到 wait 之前, 等待数据返回
        int precompute_v_lds_offset[4];
        vec2_Element<Element> *v_lds_v2fp16 = (vec2_Element<Element> *)(v_lds);
        for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
            for (int seq_idx = 0; seq_idx < PV_K_WARP_COUNT; ++seq_idx) {
                for (int head_dim_idx = 0; head_dim_idx < PV_N_WARP_COUNT; ++head_dim_idx) {
                    precompute_v_lds_offset[vec_idx] = reinterpret_cast<size_t>(v_lds_v2fp16) + ((stage_id * WARP_K * kBlockN + (seq_idx * 32 * kBlockN) + head_dim_idx * 32 * 32 + vec_idx * 8 * 32 + v_ds_read_offset) / 2) * 4;
                }
            }
        }

        __builtin_amdgcn_sched_barrier(0);
        #ifdef USE_PINGPANG_BUFFER
            if constexpr (STAGES == 2) {
                buffer_load_lds_dwordx1_wait_nosync<0>();
            } else if constexpr (STAGES == 1) {
                buffer_load_lds_dwordx1_wait_nosync<0>();
            }
        #else
            buffer_load_lds_dwordx1_wait_nosync<0>();
        #endif
        __builtin_amdgcn_sched_barrier(0);

        #pragma unroll
        for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
            #pragma unroll
            for (int seq_idx = 0; seq_idx < PV_K_WARP_COUNT; ++seq_idx) {
                #pragma unroll
                for (int head_dim_idx = 0; head_dim_idx < PV_N_WARP_COUNT; ++head_dim_idx) {
                    inline_ds_read2_b32_no_wait_bytes(precompute_v_lds_offset[vec_idx], v_reg[stage_id * PV_K_WARP_COUNT * PV_N_WARP_COUNT + (head_dim_idx * PV_K_WARP_COUNT + seq_idx)][vec_idx].u64, NEXT_DWORD_OFFSET);
                }
            }
        }

        // 拆成两段, 一共发出去 8 条 ds_read 指令, 前 4 条指令 perm, 用来做 mmac，然后再等待后面 4 条指令的返回, 类似
        vec4_Element<Element> v_vgprs[PV_K_WARP_COUNT * PV_N_WARP_COUNT][2];
        {
            constexpr int min_tile_k = 0;
            // 先把 p 寄存器需要的数据 v_pack 在一起
            vec4_Element<Element> p_vgprs[2];
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                p_vgprs[min_tile_m] = make_vec4_f16(
                    p_reg[0][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0/*vec_idx*/],
                    p_reg[0][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                    p_reg[0][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                    p_reg[0][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                );
            }
            asm volatile("s_setprio 1");
            asm volatile("s_waitcnt lgkmcnt(2)");
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for (int k_idx = 0; k_idx < PV_K_WARP_COUNT; ++k_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < PV_N_WARP_COUNT; ++n_idx) {
                    int v_tile_id = stage_id * PV_K_WARP_COUNT * PV_N_WARP_COUNT + n_idx * PV_K_WARP_COUNT + k_idx;
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        v_vgprs[n_idx * PV_K_WARP_COUNT + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                            v_reg[v_tile_id][0 + min_tile_k * 2/*vec_idx*/].f16x2[0][min_tile_n],
                            v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                            v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                            v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                        };
                    }
                }
            }
            #pragma unroll
            for (int m_idx = 0; m_idx < M_WARP_COUNT; ++m_idx) {
                #pragma unroll
                for (int k_idx = 0; k_idx < PV_K_WARP_COUNT; ++k_idx) {
                    #pragma unroll
                    for (int n_idx = 0; n_idx < PV_N_WARP_COUNT; ++n_idx) {
                        int n_loop_idx = n_loop;
                        int pv_tile_id = n_loop_idx * M_WARP_COUNT * PV_N_WARP_COUNT + n_idx * M_WARP_COUNT + m_idx;
                        #pragma unroll
                        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = flash::mmac<Element, ElementAccum>(
                                    p_vgprs[min_tile_m],
                                    v_vgprs[n_idx * PV_K_WARP_COUNT + k_idx][/*min_tile_k * 2 + */min_tile_n],
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
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                p_vgprs[min_tile_m] = make_vec4_f16(
                    p_reg[0][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0/*vec_idx*/],
                    p_reg[0][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0],
                    p_reg[0][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1],
                    p_reg[0][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1]
                );
            }
            asm volatile("s_setprio 1");
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for (int k_idx = 0; k_idx < PV_K_WARP_COUNT; ++k_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < PV_N_WARP_COUNT; ++n_idx) {
                    int v_tile_id = stage_id * PV_K_WARP_COUNT * PV_N_WARP_COUNT + n_idx * PV_K_WARP_COUNT + k_idx;
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        v_vgprs[n_idx * PV_K_WARP_COUNT + k_idx][/*min_tile_k * 2 + */min_tile_n] = vec4_Element<Element>{
                            v_reg[v_tile_id][0 + min_tile_k * 2/*vec_idx*/].f16x2[0][min_tile_n],
                            v_reg[v_tile_id][0 + min_tile_k * 2].f16x2[1][min_tile_n],
                            v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[0][min_tile_n],
                            v_reg[v_tile_id][1 + min_tile_k * 2].f16x2[1][min_tile_n]
                        };
                    }
                }
            }
            #pragma unroll
            for (int m_idx = 0; m_idx < M_WARP_COUNT; ++m_idx) {
                #pragma unroll
                for (int k_idx = 0; k_idx < PV_K_WARP_COUNT; ++k_idx) {
                    #pragma unroll
                    for (int n_idx = 0; n_idx < PV_N_WARP_COUNT; ++n_idx) {
                        int n_loop_idx = n_loop;
                        int pv_tile_id = n_loop_idx * M_WARP_COUNT * PV_N_WARP_COUNT + n_idx * M_WARP_COUNT + m_idx;
                        #pragma unroll
                        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            #pragma unroll
                            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = flash::mmac<Element, ElementAccum>(
                                    p_vgprs[min_tile_m],
                                    v_vgprs[n_idx * PV_K_WARP_COUNT + k_idx][/*min_tile_k * 2 + */min_tile_n],
                                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
            asm volatile("s_setprio 0");
            // asm volatile("s_barrier ; sync before load in the coming round");
        }
    }

    __syncthreads(); // here, K/V use more lds, and thus reuse togather, need sync
}
