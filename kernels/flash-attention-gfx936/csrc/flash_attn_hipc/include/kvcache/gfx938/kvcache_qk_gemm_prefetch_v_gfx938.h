#pragma once
#include "kvcache_qk_gemm_utils_gfx938.h"

#define USE_DS_OVERLAP_MMAC


template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int WARP_NUM, int STAGES, int M_MMAC_COUNT, typename Element, typename ElementAccum>
__forceinline__ __device__ void kvcache_qk_gemm_prefetch_v_gfx938(
        vec4_uint q_addr,
        vec4_uint k_addr,
        vec4_uint v_addr,
        Element* q_lds,
        Element* k_lds,
        Element* v_lds,
        union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2],
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (WARP_N / 32)][4],
        int warp_id,
        int kcache_seqlen_stride,
        int vcache_seqlen_stride,
        int max_seq_k_offset=0) {

    static_assert(kBlockK == 32 and "To simplify, only kBlockK = 32 is supported! otherwise, restore q_warp_buffer_load_k_id and so on");
    constexpr int K_LOAD_REQUESTS = (WARP_N / 32) * (kBlockK / 32);

    // 分配 k 计算 mmac 需要的寄存器资源
    // 一次加载 32x32 个 half, 每个线程持有 16 个 half
    union_vec4_f16x2<Element> k_reg[1 * (WARP_N * kBlockK) / (32 * 32) * 2];

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

    // 准备 MLS resource 寄存器
    vec4_uint k_srsrc;
    k_srsrc[1] = k_addr[1];
    k_srsrc[2] = kcache_seqlen_stride;

    int stage_id = 0;

    constexpr int K_LOOP_START = (STAGES == 2) ? 2: 0;
    if constexpr (STAGES == 2) stage_id ^= 1;
    for (int k_loop = K_LOOP_START; k_loop < (kHeadDim / kBlockK); k_loop += 2) {

        #pragma unroll
        for (int prefetch_id = 0; prefetch_id < 2; ++prefetch_id) {
            // 计算当前 wave 写到 lds 的起始地址
            int k_lds_stage_offset = (warp_id * STAGES * 2 + stage_id * 2 + prefetch_id) * K_LOAD_REQUESTS * (32 * 32);

            // 计算当前 wave 沿着 kHeadDim 方向循环读取的起始地址, 读到第几个 32x32 块了
            int k_mls_loop_global_offset = (k_loop + prefetch_id) * kBlockK * sizeof(Element);

            // 计算当前 wave 从 global 读取数据的起始地址
            int k_mls_warp_global_offset; // = warp_id * WARP_N * kcache_seqlen_stride * sizeof(Element);

            if constexpr (true) {
                int nm_filter_max = warp_id * WARP_N + 32 - max_seq_k_offset; // 判断是否有 warp 取空数据
                int real_mls_warp_id = nm_filter_max >= 32 ? 0: warp_id;       // 如果取空数据, 938 不支持, 退化到取 warp 0 的数据
                k_mls_warp_global_offset = real_mls_warp_id * WARP_N * kcache_seqlen_stride * sizeof(Element);
                int nm_filter = inline_min_max<0, 32>(real_mls_warp_id * WARP_N + 32 - max_seq_k_offset); // 如果取空数据, 使用 warp 0 的 nm_filter 值
                k_srsrc[3] = nm_filter << 8;
            }
            // 根据偏移计算 global load 的字节偏移数
            // k_srsrc[0] = k_addr[0] + k_mls_loop_global_offset + k_mls_warp_global_offset;
            *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_addr + k_mls_loop_global_offset + k_mls_warp_global_offset);
            int lds_offset_bytes = k_lds_stage_offset * 2/*half -> bytes*/;
            inline_matrix_load_32x32_b16_lds_trans<0, 0>(k_lds, k_srsrc, lds_offset_bytes, 0);
        }

        // 等待 MLS 数据回来
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (STAGES == 2) {
            buffer_load_lds_dwordx1_wait_nosync<3 * K_LOAD_REQUESTS>();
        } else {
            buffer_load_lds_dwordx1_wait_nosync<0>();
        }
        __builtin_amdgcn_sched_barrier(0);

        // __builtin_amdgcn_sched_barrier(0);
        if constexpr (STAGES == 2) stage_id ^= 1;

        // 加载上一次 MLS 写到 lds 的数据到寄存器
        int lds_load_offset = reinterpret_cast<size_t>(k_lds) + (warp_id * STAGES * 2 + stage_id * 2) * K_LOAD_REQUESTS * (32 * 32) * sizeof(Element)/*half -> bytes*/;
        DS_READ_MATRIX_32X32_B16(lds_load_offset, k_reg[0].f16, k_reg[1].f16, true);

        #pragma unroll
        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(%0)\n" :: "B"(2 - min_tile_n - 1));
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 1");
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
                                int k_loop_idx = (STAGES == 2) ? k_loop - 2: k_loop;
                                int q_tile_id = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                                    q_reg[q_tile_id].f16x4[min_tile_k],
                                    k_reg[k_tile_id].f16x4[min_tile_k],
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 0");
        }
        // =============================================================================================================
        {
            __builtin_amdgcn_sched_barrier(0);
            if constexpr (STAGES == 2) {
                buffer_load_lds_dwordx1_wait_nosync<2 * K_LOAD_REQUESTS>();
            } else {
                buffer_load_lds_dwordx1_wait_nosync<0>();
            }
            __builtin_amdgcn_sched_barrier(0);

            // 加载上一次 MLS 写到 lds 的数据到寄存器
            int lds_load_offset = reinterpret_cast<size_t>(k_lds) + (warp_id * STAGES * 2 + stage_id * 2 + 1) * K_LOAD_REQUESTS * (32 * 32) * sizeof(Element)/*half -> bytes*/;
            DS_READ_MATRIX_32X32_B16(lds_load_offset, k_reg[0].f16, k_reg[1].f16, true);

            #pragma unroll
            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_waitcnt lgkmcnt(%0)\n" :: "B"(2 - min_tile_n - 1));
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 1");
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
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                                        q_reg[q_tile_id].f16x4[min_tile_k],
                                        k_reg[k_tile_id].f16x4[min_tile_k],
                                        s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 0");
            }
        }
    }

    if constexpr (STAGES == 2) {
        // 等待 MLS 数据回来
        __builtin_amdgcn_sched_barrier(0);
        buffer_load_lds_dwordx1_wait_nosync<1>();
        __builtin_amdgcn_sched_barrier(0);

        // 切换到上一次 lds 被写入的轮次
        stage_id ^= 1;

        // 从 lds 加载最后一部分数据
        int lds_load_offset = reinterpret_cast<size_t>(k_lds) + (warp_id * STAGES * 2 + stage_id * 2) * K_LOAD_REQUESTS * (32 * 32) * sizeof(Element)/*half -> bytes*/;
        DS_READ_MATRIX_32X32_B16(lds_load_offset, k_reg[0].f16, k_reg[1].f16, true);

        #pragma unroll
        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(%0)\n" :: "B"(2 - min_tile_n - 1));
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 1");
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
                                int k_loop_idx = kHeadDim / kBlockK - 2;
                                int q_tile_id = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                int k_tile_id = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                                    q_reg[q_tile_id].f16x4[min_tile_k],
                                    k_reg[k_tile_id].f16x4[min_tile_k],
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                            }
                        }
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 0");
        }
        // ==========================================================================
        {
            __builtin_amdgcn_sched_barrier(0);
            buffer_load_lds_dwordx1_wait_nosync<0>();
            __builtin_amdgcn_sched_barrier(0);

            // 从 lds 加载最后一部分数据
            int lds_load_offset = reinterpret_cast<size_t>(k_lds) + (warp_id * STAGES * 2 + stage_id * 2 + 1) * K_LOAD_REQUESTS * (32 * 32) * sizeof(Element)/*half -> bytes*/;
            DS_READ_MATRIX_32X32_B16(lds_load_offset, k_reg[0].f16, k_reg[1].f16, true);

            #pragma unroll
            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_waitcnt lgkmcnt(%0)\n" :: "B"(2 - min_tile_n - 1));
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 1");
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
                                    int k_loop_idx = kHeadDim / kBlockK - 1;
                                    int q_tile_id = k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + min_tile_m;
                                    int k_tile_id = (head_dim_idx * (WARP_N / 32) + n_idx) * 2 + min_tile_n;
                                    s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                                        q_reg[q_tile_id].f16x4[min_tile_k],
                                        k_reg[k_tile_id].f16x4[min_tile_k],
                                        s_reg[n_idx * (WARP_M / 32) + m_idx][min_tile_n * 2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 0");
            }
        }
    }


    // need to reduce results on scores_max and prefetch V, and thus sync
    __syncthreads();

    // qk gemm 等待最后一次计算需要的数据之前, 可以先把需要的 V load 指令发下去;
    if constexpr (STAGES > 1) {
        kvcache_prefetch_v_to_lds_gfx938<kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, 32/*WARP_K*/, 0, WARP_NUM, Element, STAGES>(v_addr, v_lds, warp_id, vcache_seqlen_stride, max_seq_k_offset);
    }

} // qk_gemm

