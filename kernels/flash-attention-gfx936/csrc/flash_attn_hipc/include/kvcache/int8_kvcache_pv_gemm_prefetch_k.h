#include "int8_kvcache_pv_gemm_prefetch_k_3stage.h"


template<bool PREFETCH_K, int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int WARP_K, int STAGES, int WARP_NUM, int M_MMAC_COUNT, typename Element, typename Element_k, typename ElementAccum>
__forceinline__ __device__ void  int8_kvcache_pv_gemm_prefetch_k(
        vec4_uint gV,
        vec4_uint gK,
        Element_k* v_lds,
        Element_k* k_lds,
        float scales_v[2][4],
        union_vec2_f16x2<Element> p_reg[(WARP_M/32)*(WARP_K/32)][4],
        vec4_Accum<ElementAccum> pv_reg[(kHeadDimV/kBlockN)*(WARP_M/32)*(kBlockN/32)][4],
        int WARP_ID,
        int vcache_seqlen_stride,
        int max_seq_kv_offset = -1) {

    static_assert(kBlockK>=32, "Error: pv gemm kBlockK must be equal or greater than 32");
    static_assert(kBlockM>=WARP_M, "Error: pv gemm kBlockM must be equal or greater than WARP_M");
    static_assert(kBlockN==WARP_N, "Error: pv gemm kBlockN must be equal to WARP_N");

    union_vec2_f16x2<Element> v_reg[STAGES*((32*WARP_N)/(32*32))][4];

    // 预先计算一些公共表达式
    int lane_id       = threadIdx.x & 63;
    int laneid_shfl_2 = lane_id >> 2; // 0 ~ 15, 4 个线程读取一行
    int laneid_shfl_3 = lane_id >> 3; // 0 ~ 7,  8 个线程读取一行
    int laneid_shfl_4 = lane_id >> 4; // 0 ~ 3,  16 个线程读取一行
    int laneid_shfl_5 = lane_id >> 5; // 0 ~ 1,  lds 读取时, 8x32的数据按照线程 [0, 16, 0, 16, 32, 48, 32, 48] 来读取, 每 32 个线程读取一个 4x32
    constexpr int NEXT_DWORD_OFFSET = 32; // 8x32 的数据, 一个 wave 每个线程读 4 个 half, 即 2 个 dword, 使用 ds_read2_b32 指令, 按照上面的读取方式, 第二个 dword 偏移 32 个 dword

#if defined(USE_BUFFER_LOAD_DWORDX4)
    // constexpr int WARP_K             = 32;
    constexpr int READ_ONCE_LINES    = 16;                                 // 一个 warp 一次读几行数据, loadx2, 每行 32 个元素需要 32 / (4) = 8 个线程
    constexpr int READ_ONCE_COUNT    = READ_ONCE_LINES * 32;               // 一个 warp 一次 load 多少数据
    constexpr int V_LDS_LOAD_NUM     = kBlockN * WARP_K / READ_ONCE_COUNT; // 整个 workgroup 要发多少 load 指令
    constexpr int V_LOAD_REQUESTS    = V_LDS_LOAD_NUM;                     // 每个 warp 要发多少 load 指令
    constexpr int READ_ELEMENT_COUNT = 8;                                  // 每个线程一次读取几个 half
    int v_lane_headdim_n_idx         = lane_id & 3;                        // 当前 lane 负责这个 warp 的第几个 dwordx2
    int base                         = (laneid_shfl_2 & 0xc);              // 第几个4线程组的最小id
    int tail                         = (laneid_shfl_2 & 0x3);              // 线程组中的第几个线程
    int v_lane_seq_k_idx             = base + (tail & 1) * 2 + (tail >> 1);// 每个线程负责读取第几行数据的 4 个 dword
    int v_ds_read_offset             = (laneid_shfl_5 * 4 + (laneid_shfl_4 & 1)) * 32 + (lane_id & 15) * 2; // 一次读写 8x32, 0-31 线程读取前面 4x32, 32-63 读取后面 4x32, 4x32按照线程 [0, 16, 0, 16] 这种方式来读取
    auto BUFFER_LOAD_FUNC            = &inline_buffer_load_dwordx4_lds<Element_k, 2>;
#else
    // constexpr int WARP_K             = 32;
    constexpr int READ_ONCE_LINES    = 4;
    constexpr int READ_ONCE_COUNT    = READ_ONCE_LINES * 32;
    constexpr int V_LDS_LOAD_NUM     = (kBlockN * WARP_K) / READ_ONCE_COUNT;
    constexpr int V_LOAD_REQUESTS    = V_LDS_LOAD_NUM;
    constexpr int READ_ELEMENT_COUNT = 2;
    int v_lane_headdim_n_idx         = lane_id & 15;
    int v_lane_seq_k_idx             = (laneid_shfl_4 & 1) * 2 + laneid_shfl_5; // 0, 1, 2, 3 ---> 0, 2, 1, 3
    int v_ds_read_offset             = (laneid_shfl_5 * 4 + (laneid_shfl_4 & 1)) * 32 + (lane_id & 15) * 2; // 一次读写 8x32, 0-31 线程读取前面 4x32, 32-63 读取后面 4x32, 4x32按照线程 [0, 16, 0, 16] 这种方式来读取
    auto BUFFER_LOAD_FUNC            = &inline_buffer_load_dword_lds<Element_k, 2>;
#endif

    // each wave need 2 32x32 lds space
    v_lds = v_lds + WARP_ID * STAGES * WARP_K * kBlockN;

    constexpr int N_LOOP_START = (STAGES == 2) ? 1: 0;

    int stage_id = (STAGES == 2) ? 1: 0;

    for (int n_loop = N_LOOP_START; n_loop < (kHeadDimV / kBlockN); ++n_loop) {
        {
            // int v_block_buffer_load_global_offset = n_loop*kBlockN;
            int v_block_buffer_load_global_offset = WARP_ID * kHeadDimV * WARP_K + n_loop * kBlockN;

            for(int load = 0; load < V_LOAD_REQUESTS; ++load) {
                // global->lds, right matrix
                int v_warp_buffer_load_k_id = (load + WARP_ID) % V_LOAD_REQUESTS; // (load / (kBlockN/32));
                // int v_warp_buffer_load_n_id = (warp_loop & (kBlockN/32 - 1));
                // int v_warp_buffer_load_global_offset = (v_warp_buffer_load_n_id * 32);
                int v_warp_buffer_load_lds_offset    = /*(v_warp_buffer_load_n_id * 32) +*/  (load * READ_ONCE_COUNT);
                // int v_gvoffset = (v_block_buffer_load_global_offset + v_warp_buffer_load_globhalf_tal_offset + /*(k_idx*16*M) + (m_idx*32) +*/ (v_lane_n_idx * 2 + v_lane_k_idx * kHeadDim)) / 2;
                int v_gvoffset_s = (v_block_buffer_load_global_offset/* + v_warp_buffer_load_global_offset*/) / 2;
                int v_gvoffset_v = ((v_lane_headdim_n_idx * READ_ELEMENT_COUNT + min(v_lane_seq_k_idx + load*READ_ONCE_LINES, max_seq_kv_offset - 1) * kHeadDimV)) / 2;
                int v_lds_offset = (v_warp_buffer_load_lds_offset) / 2;
                BUFFER_LOAD_FUNC(v_lds + (stage_id)*WARP_K*kBlockN, gV, v_lds_offset, v_gvoffset_s, v_gvoffset_v);
            }
        }

        if constexpr (STAGES == 2) stage_id ^= 1;

        // 把 ds_read 之前的一些计算挪到 wait 之前, 等待数据返回
        int precompute_v_lds_offset[4];
        vec2_Element<Element> *v_lds_v2fp16 = (vec2_Element<Element> *)(v_lds);
        // lds -> vgpr use ds_read_m; right matrix
        for(int vec_idx=0; vec_idx<4; vec_idx++) {
            for(int seq_idx=0; seq_idx<(WARP_K/32); seq_idx++) {
                for(int head_dim_idx=0; head_dim_idx<(WARP_N/32); head_dim_idx++) {
                    precompute_v_lds_offset[vec_idx] = reinterpret_cast<size_t>(v_lds_v2fp16) + ((stage_id*WARP_K*kBlockN + (seq_idx*32*kBlockN) + head_dim_idx*32*32 + vec_idx*8*32 + v_ds_read_offset)/2) * 4;
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

        // lds -> vgpr use ds_read_m; left matrix
        // int v_lane_head_dim_idx = lane_id % 16;
        // int v_lane_seq_idx = lane_id >> 4;
        // vec2_Element<Element> *v_lds_v2fp16 = (vec2_Element<Element> *)(v_lds);
        // lds -> vgpr use ds_read_m; right matrix
        #pragma unroll
        for(int vec_idx=0; vec_idx<4; vec_idx++) {
            #pragma unroll
            for(int seq_idx=0; seq_idx<(WARP_K/32); seq_idx++) {
                #pragma unroll
                for(int head_dim_idx=0; head_dim_idx<(WARP_N/32); head_dim_idx++) {
                    // #pragma unroll
                    inline_ds_read2_b32_no_wait_bytes(precompute_v_lds_offset[vec_idx], v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (head_dim_idx*(WARP_K/32) + seq_idx)][vec_idx].u64, NEXT_DWORD_OFFSET);
                }
            }
        }

        // 拆成两段, 一共发出去 8 条 ds_read 指令, 前 4 条指令 perm, 用来做 mmac，然后再等待后面 4 条指令的返回, 类似
        vec4_Element<Element> v_vgprs[(32*WARP_N)/(32*32)][2];
        {
            constexpr int min_tile_k = 0;
            // 先把 p 寄存器需要的数据 v_pack 在一起 (p_vgprs 格式暂且简化, 不考虑下面那个复杂的 m_idx 跟 k_idx)
            vec4_Element<Element> p_vgprs[2];
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                p_vgprs[min_tile_m] = make_vec4_f16(
                    p_reg[0][(0*2 + min_tile_m)].f16[min_tile_k*2 + 0],
                    p_reg[0][(1*2 + min_tile_m)].f16[min_tile_k*2 + 0],
                    p_reg[0][(0*2 + min_tile_m)].f16[min_tile_k*2 + 1],
                    p_reg[0][(1*2 + min_tile_m)].f16[min_tile_k*2 + 1]
                );
            }
            asm volatile("s_setprio 1");
            asm volatile("s_waitcnt lgkmcnt(0)");  // 这里暂时先写死是 2, 传编译期参数进去会导致性能略微下降
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for(int k_idx=0; k_idx<(WARP_K/32); k_idx++) {
                #pragma unroll
                for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                    #pragma unroll
                    for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                        v_vgprs[n_idx* (WARP_K/32) + k_idx][/*min_tile_k*2 + */min_tile_n] = vec4_Element<Element>{
                            v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][0 + min_tile_k*2].f16x2[0][min_tile_n],
                            v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][0 + min_tile_k*2].f16x2[1][min_tile_n],
                            v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][1 + min_tile_k*2].f16x2[0][min_tile_n],
                            v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][1 + min_tile_k*2].f16x2[1][min_tile_n]
                        };
                    }
                }
            }
            #pragma unroll
            for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
                #pragma unroll
                for(int k_idx=0; k_idx<(WARP_K/32); k_idx++) {
                    #pragma unroll
                    for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                        #pragma unroll
                        for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                            #pragma unroll
                            for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                                int n_loop_idx = (STAGES == 2) ? n_loop - 1: n_loop;
                                pv_reg[n_loop_idx * ((WARP_M/32)*(kBlockN/32)) + (n_idx*(WARP_M/32) + m_idx)][min_tile_n*2 + min_tile_m].f32 =
                                    flash::mmac<Element, ElementAccum>(
                                        p_vgprs[min_tile_m],
                                        v_vgprs[n_idx* (WARP_K/32) + k_idx][/*min_tile_k*2 + */min_tile_n],
                                        pv_reg[n_loop_idx * ((WARP_M/32)*(kBlockN/32)) + (n_idx*(WARP_M/32) + m_idx)][min_tile_n*2 + min_tile_m].f32);
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
                    p_reg[0][(0*2 + min_tile_m)].f16[min_tile_k*2 + 0],
                    p_reg[0][(1*2 + min_tile_m)].f16[min_tile_k*2 + 0],
                    p_reg[0][(0*2 + min_tile_m)].f16[min_tile_k*2 + 1],
                    p_reg[0][(1*2 + min_tile_m)].f16[min_tile_k*2 + 1]
                );
            }
            asm volatile("s_setprio 1");
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for(int k_idx=0; k_idx<(WARP_K/32); k_idx++) {
                #pragma unroll
                for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                    #pragma unroll
                    for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                        v_vgprs[n_idx* (WARP_K/32) + k_idx][/*min_tile_k*2 + */min_tile_n] = vec4_Element<Element>{
                            v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][0 + min_tile_k*2].f16x2[0][min_tile_n],
                            v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][0 + min_tile_k*2].f16x2[1][min_tile_n],
                            v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][1 + min_tile_k*2].f16x2[0][min_tile_n],
                            v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][1 + min_tile_k*2].f16x2[1][min_tile_n]
                        };
                    }
                }
            }
            #pragma unroll
            for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
                #pragma unroll
                for(int k_idx=0; k_idx<(WARP_K/32); k_idx++) {
                    #pragma unroll
                    for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                        #pragma unroll
                        for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                            #pragma unroll
                            for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                                int n_loop_idx = (STAGES == 2) ? n_loop - 1: n_loop;
                                pv_reg[n_loop_idx * ((WARP_M/32)*(kBlockN/32)) + (n_idx*(WARP_M/32) + m_idx)][min_tile_n*2 + min_tile_m].f32 =
                                    flash::mmac<Element, ElementAccum>(
                                        p_vgprs[min_tile_m],
                                        v_vgprs[n_idx* (WARP_K/32) + k_idx][/*min_tile_k*2 + */min_tile_n],
                                        pv_reg[n_loop_idx * ((WARP_M/32)*(kBlockN/32)) + (n_idx*(WARP_M/32) + m_idx)][min_tile_n*2 + min_tile_m].f32);
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
        int n_loop = (kHeadDimV / kBlockN) - 1;
        stage_id  ^= 1;
        {
            // 把 ds_read 之前的一些计算挪到 wait 之前, 等待数据返回
            int precompute_v_lds_offset[4];
            vec2_Element<Element> *v_lds_v2fp16 = (vec2_Element<Element> *)(v_lds);
            // lds -> vgpr use ds_read_m; right matrix
            for(int vec_idx=0; vec_idx<4; vec_idx++) {
                for(int seq_idx=0; seq_idx<(WARP_K/32); seq_idx++) {
                    for(int head_dim_idx=0; head_dim_idx<(WARP_N/32); head_dim_idx++) {
                        precompute_v_lds_offset[vec_idx] = reinterpret_cast<size_t>(v_lds_v2fp16) + ((stage_id*WARP_K*kBlockN + (seq_idx*32*kBlockN) + head_dim_idx*32*32 + vec_idx*8*32 + v_ds_read_offset)/2) * 4;
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

            // lds -> vgpr use ds_read_m; left matrix
            // int v_lane_head_dim_idx = lane_id % 16;
            // int v_lane_seq_idx = lane_id >> 4;
            // vec2_Element<Element> *v_lds_v2fp16 = (vec2_Element<Element> *)(v_lds);
            // lds -> vgpr use ds_read_m; right matrix
            #pragma unroll
            for(int vec_idx=0; vec_idx<4; vec_idx++) {
                #pragma unroll
                for(int seq_idx=0; seq_idx<(WARP_K/32); seq_idx++) {
                    #pragma unroll
                    for(int head_dim_idx=0; head_dim_idx<(WARP_N/32); head_dim_idx++) {
                        // #pragma unroll
                        inline_ds_read2_b32_no_wait_bytes(precompute_v_lds_offset[vec_idx], v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (head_dim_idx*(WARP_K/32) + seq_idx)][vec_idx].u64, NEXT_DWORD_OFFSET);
                    }
                }
            }

            // 拆成两段, 一共发出去 8 条 ds_read 指令, 前 4 条指令 perm, 用来做 mmac，然后再等待后面 4 条指令的返回, 类似
            vec4_Element<Element> v_vgprs[(32*WARP_N)/(32*32)][2];
            {
                constexpr int min_tile_k = 0;
                // 先把 p 寄存器需要的数据 v_pack 在一起 (p_vgprs 格式暂且简化, 不考虑下面那个复杂的 m_idx 跟 k_idx)
                vec4_Element<Element> p_vgprs[2];
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    p_vgprs[min_tile_m] = make_vec4_f16(
                        p_reg[0][(0*2 + min_tile_m)].f16[min_tile_k*2 + 0],
                        p_reg[0][(1*2 + min_tile_m)].f16[min_tile_k*2 + 0],
                        p_reg[0][(0*2 + min_tile_m)].f16[min_tile_k*2 + 1],
                        p_reg[0][(1*2 + min_tile_m)].f16[min_tile_k*2 + 1]
                    );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(0)");  // 这里暂时先写死是 2, 传编译期参数进去会导致性能略微下降
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx=0; k_idx<(WARP_K/32); k_idx++) {
                    #pragma unroll
                    for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                        #pragma unroll
                        for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                            v_vgprs[n_idx* (WARP_K/32) + k_idx][/*min_tile_k*2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][0 + min_tile_k*2].f16x2[0][min_tile_n],
                                v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][0 + min_tile_k*2].f16x2[1][min_tile_n],
                                v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][1 + min_tile_k*2].f16x2[0][min_tile_n],
                                v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][1 + min_tile_k*2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
                    #pragma unroll
                    for(int k_idx=0; k_idx<(WARP_K/32); k_idx++) {
                        #pragma unroll
                        for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                            #pragma unroll
                            for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                                #pragma unroll
                                for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                                    int n_loop_idx = n_loop;
                                    pv_reg[n_loop_idx * ((WARP_M/32)*(kBlockN/32)) + (n_idx*(WARP_M/32) + m_idx)][min_tile_n*2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx* (WARP_K/32) + k_idx][/*min_tile_k*2 + */min_tile_n],
                                            pv_reg[n_loop_idx * ((WARP_M/32)*(kBlockN/32)) + (n_idx*(WARP_M/32) + m_idx)][min_tile_n*2 + min_tile_m].f32);
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
                        p_reg[0][(0*2 + min_tile_m)].f16[min_tile_k*2 + 0],
                        p_reg[0][(1*2 + min_tile_m)].f16[min_tile_k*2 + 0],
                        p_reg[0][(0*2 + min_tile_m)].f16[min_tile_k*2 + 1],
                        p_reg[0][(1*2 + min_tile_m)].f16[min_tile_k*2 + 1]
                    );
                }
                asm volatile("s_setprio 1");
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int k_idx=0; k_idx<(WARP_K/32); k_idx++) {
                    #pragma unroll
                    for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                        #pragma unroll
                        for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                            v_vgprs[n_idx* (WARP_K/32) + k_idx][/*min_tile_k*2 + */min_tile_n] = vec4_Element<Element>{
                                v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][0 + min_tile_k*2].f16x2[0][min_tile_n],
                                v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][0 + min_tile_k*2].f16x2[1][min_tile_n],
                                v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][1 + min_tile_k*2].f16x2[0][min_tile_n],
                                v_reg[stage_id*((WARP_N*WARP_K)/(32*32)) + (n_idx* (WARP_K/32) + k_idx)][1 + min_tile_k*2].f16x2[1][min_tile_n]
                            };
                        }
                    }
                }
                #pragma unroll
                for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
                    #pragma unroll
                    for(int k_idx=0; k_idx<(WARP_K/32); k_idx++) {
                        #pragma unroll
                        for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                            #pragma unroll
                            for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                                #pragma unroll
                                for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                                    int n_loop_idx = n_loop;
                                    pv_reg[n_loop_idx * ((WARP_M/32)*(kBlockN/32)) + (n_idx*(WARP_M/32) + m_idx)][min_tile_n*2 + min_tile_m].f32 =
                                        flash::mmac<Element, ElementAccum>(
                                            p_vgprs[min_tile_m],
                                            v_vgprs[n_idx* (WARP_K/32) + k_idx][/*min_tile_k*2 + */min_tile_n],
                                            pv_reg[n_loop_idx * ((WARP_M/32)*(kBlockN/32)) + (n_idx*(WARP_M/32) + m_idx)][min_tile_n*2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
                asm volatile("s_setprio 0");
                // asm volatile("s_barrier ; sync before load in the coming round");
            }
        }
    }



    __syncthreads(); // here, K/V use more lds, and thus reuse togather, need sync
}
