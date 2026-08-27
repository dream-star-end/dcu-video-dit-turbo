#pragma once
#include "f16_mla_tp8_pv_gemm_utils_gfx938.h"


template<int K_LOOP_COUNT, int kBlockM, int kBlockN, int kBlockK, int M_WARP_COUNT, int PV_N_WARP_COUNT, int PV_K_WARP_COUNT, int STAGES, int WARP_NUM, int M_MMAC_COUNT, typename Element, typename ElementAccum>
__forceinline__ __device__ void f16_mla_tp8_pv_gemm_gfx938(
        vec4_uint v_addr,
        vec4_uint k_addr,
        Element* v_lds,
        Element* k_lds,
        union_vec2_f16x2<Element> p_reg[M_WARP_COUNT * PV_K_WARP_COUNT][4],
        vec4_Accum<ElementAccum> pv_reg[K_LOOP_COUNT * M_WARP_COUNT * (kBlockN / 32)][4],
        int warp_id,
        int kvcache_seqlen_stride,
        int max_seq_kv_offset=0) {

    constexpr int WARP_K = PV_K_WARP_COUNT * 32;
    static_assert (kBlockK >= 32, "Error: pv gemm kBlockK must be equal or greater than 32");
    static_assert (kBlockN == PV_N_WARP_COUNT * 32, "Error: kBlockN in kvcache_pv_gemm_prefetch_k must be WARP_N * 32");
    static_assert (M_WARP_COUNT == 1, "for gfx938, only WARP_M = 32 is supported yet!");
    static_assert (PV_N_WARP_COUNT == 1, "for gfx938, only WARP_N = 32 is supported yet!");
    static_assert (PV_K_WARP_COUNT == 1, "for gfx938, only WARP_K = 32 is supported yet!");

    constexpr int V_LOAD_REQUESTS = (WARP_K * kBlockN) / (32 * 32);

    // 准备寄存器, 每次加载 32x32 的 half 用于 mmac 计算, 每个线程持有 16 个 half, 因此是 8 * 2, 一列有 8 个 half, 有两列
    union_vec4_f16x2<Element> v_reg[1 * PV_K_WARP_COUNT * PV_N_WARP_COUNT * 2];

    // 准备 MLS 的 resource 寄存器
    vec4_uint v_srsrc;
    v_srsrc[1] = v_addr[1];
    v_srsrc[2] = kvcache_seqlen_stride; // stride

    // 防止与多 wave reduce max 需要的 lds 冲突
    __syncthreads();

    int stage_id = (STAGES == 2) ? 1: 0;

    // 一次加载多批数据
    constexpr int N_LOOP_STEP  = (STAGES == 2) ? 2: 1;
    constexpr int N_LOOP_START = (STAGES == 2) ? K_LOOP_COUNT - N_LOOP_STEP * 2: K_LOOP_COUNT - 1;
    constexpr int N_LOOP_END   = 0;
    for (int n_loop = N_LOOP_START; n_loop >= N_LOOP_END; n_loop -= N_LOOP_STEP) {

        #pragma unroll
        for (int prefetch_id = 0; prefetch_id < N_LOOP_STEP; ++prefetch_id) {

            // 计算当前 wave 当前加载的 32x32 block 的偏移字节数
            int v_mls_warp_global_offset = (n_loop + prefetch_id) * kBlockN * sizeof(Element);

            // 计算当前 wave 写入 lds 的偏移地址(注意 v_lds 相较于 smem 的偏移量)
            int v_mls_lds_warp_offset = (warp_id * STAGES * 2 + stage_id * 2 + prefetch_id) * (V_LOAD_REQUESTS * 32 * 32) * sizeof(Element);

            // 计算当前 wave 读取数据的起始偏移字节数
            int v_mls_loop_global_offset; // = warp_id * WARP_K * kvcache_seqlen_stride * sizeof(Element);

            // 计算 MLS 读取数据的 global 地址, 判断边界
            if constexpr (true) {
                int nm_filter_max = warp_id * WARP_K + 32 - max_seq_kv_offset; // 判断是否有 warp 取空数据
                int real_mls_warp_id = nm_filter_max >= 32 ? 0: warp_id;       // 如果取空数据, 938 不支持, 退化到取 warp 0 的数据
                v_mls_loop_global_offset = real_mls_warp_id * WARP_K * kvcache_seqlen_stride * sizeof(Element);
                int nm_filter = inline_min_max<0, 32>(real_mls_warp_id * WARP_K + 32 - max_seq_kv_offset); // 如果取空数据, 使用 warp 0 的 nm_filter 值
                v_srsrc[3] = max_seq_kv_offset % kBlockN == 0 ? 0: nm_filter << 8;
                v_srsrc[3] += 0x20000;
            }
            // v_srsrc[0] = v_addr[0] + v_mls_loop_global_offset + v_mls_warp_global_offset;
            *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_addr + v_mls_loop_global_offset + v_mls_warp_global_offset);
            __builtin_amdgcn_sched_barrier(0);
            inline_matrix_load_32x32_b16_lds<0, 1>(v_lds, v_srsrc, v_mls_lds_warp_offset, 0);
            __builtin_amdgcn_sched_barrier(0);
        }

        // 等待 MLS 数据回来
        if constexpr (N_LOOP_STEP == 2) {
            buffer_load_lds_dwordx1_wait_nosync<3 * V_LOAD_REQUESTS>();
        } else if constexpr (N_LOOP_STEP == 1 and STAGES == 2) {
            buffer_load_lds_dwordx1_wait_nosync<V_LOAD_REQUESTS>();
        } else if constexpr (N_LOOP_STEP == 1 and STAGES == 1) {
            buffer_load_lds_dwordx1_wait_nosync<0>();
        }
        __builtin_amdgcn_sched_barrier(0);

        // 切换到 load 轮次
        if constexpr (STAGES == 2) {
            stage_id ^= 1;
        }
        int lds_load_offset = reinterpret_cast<size_t>(v_lds) + (warp_id * STAGES * 2 + stage_id * 2) * (V_LOAD_REQUESTS * 32 * 32) * 2/*bytes*/;
        DS_READ_MATRIX_32X32_B16_ALT2(lds_load_offset, v_reg[0].f16, v_reg[1].f16, false); // hint: multiple prefetching can be applied here

        // 拆成两段, 一共发出去 8 条 ds_read 指令, 前 4 条指令 perm, 用来做 mmac，然后再等待后面 4 条指令的返回, 类似
        #pragma unroll
        for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(%0)" :: "B"(2 - min_tile_k - 1));
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 2");
            int pv_tile_id = (STAGES == 2) ? n_loop + 2: n_loop;
            #pragma unroll
            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = flash::mmac_4interleave<Element, ElementAccum>(
                        p_reg[0][min_tile_k * 2 + min_tile_m].f16x4,
                        v_reg[min_tile_k].f16x4[min_tile_n],
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                }
            }
            asm volatile("s_setprio 0");
        }
        // ============================================================================================================
        // 处理预取的第二段数据
        if constexpr (N_LOOP_STEP == 2) {
            __builtin_amdgcn_sched_barrier(0);
            buffer_load_lds_dwordx1_wait_nosync<2 * V_LOAD_REQUESTS>();
            __builtin_amdgcn_sched_barrier(0);
            int lds_load_offset = reinterpret_cast<size_t>(v_lds) + (warp_id * STAGES * 2 + stage_id * 2 + 1/*prefetch_id*/) * (V_LOAD_REQUESTS * 32 * 32) * 2/*bytes*/;
            __builtin_amdgcn_sched_barrier(0);
            DS_READ_MATRIX_32X32_B16_ALT2(lds_load_offset, v_reg[0].f16, v_reg[1].f16, false);

            // 拆成两段, 一共发出去 8 条 ds_read 指令, 前 4 条指令 perm, 用来做 mmac，然后再等待后面 4 条指令的返回, 类似
            #pragma unroll
            for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_waitcnt lgkmcnt(%0)" :: "B"(2 - min_tile_k - 1));
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 2");
                int pv_tile_id = (STAGES == 2) ? n_loop + 3: n_loop;
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = flash::mmac_4interleave<Element, ElementAccum>(
                            p_reg[0][min_tile_k * 2 + min_tile_m].f16x4,
                            v_reg[min_tile_k].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
                asm volatile("s_setprio 0");
            }
        }
    }

    if constexpr (STAGES == 2) {
        // 等待 MLS 数据回来
        __builtin_amdgcn_sched_barrier(0);
        buffer_load_lds_dwordx1_wait_nosync<1 * V_LOAD_REQUESTS>();
        __builtin_amdgcn_sched_barrier(0);

        int n_loop = N_LOOP_END - N_LOOP_STEP;

        // 切换
        stage_id ^= 1;
        int lds_load_offset = reinterpret_cast<size_t>(v_lds) + (warp_id * STAGES * 2 + stage_id * 2) * (V_LOAD_REQUESTS * 32 * 32) * 2/*bytes*/;
        DS_READ_MATRIX_32X32_B16_ALT2(lds_load_offset, v_reg[0].f16, v_reg[1].f16, false);

        // 拆成两段, 一共发出去 8 条 ds_read 指令, 前 4 条指令 perm, 用来做 mmac，然后再等待后面 4 条指令的返回, 类似
        #pragma unroll
        for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(%0)" :: "B"(2 - min_tile_k - 1));
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 2");
            int pv_tile_id = n_loop + N_LOOP_STEP;
            #pragma unroll
            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = flash::mmac_4interleave<Element, ElementAccum>(
                        p_reg[0][min_tile_k * 2 + min_tile_m].f16x4,
                        v_reg[min_tile_k].f16x4[min_tile_n],
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                }
            }
            asm volatile("s_setprio 0");
        }
        // ============================================================================================================
        // 处理预取的第二段数据
        if constexpr (N_LOOP_STEP == 2) {
            __builtin_amdgcn_sched_barrier(0);
            buffer_load_lds_dwordx1_wait_nosync<0 * V_LOAD_REQUESTS>();
            __builtin_amdgcn_sched_barrier(0);
            int lds_load_offset = reinterpret_cast<size_t>(v_lds) + (warp_id * STAGES * 2 + stage_id * 2 + 1/*prefetch_id*/) * (V_LOAD_REQUESTS * 32 * 32) * 2/*bytes*/;
            __builtin_amdgcn_sched_barrier(0);
            DS_READ_MATRIX_32X32_B16_ALT2(lds_load_offset, v_reg[0].f16, v_reg[1].f16, false);

            // 拆成两段, 一共发出去 8 条 ds_read 指令, 前 4 条指令 perm, 用来做 mmac，然后再等待后面 4 条指令的返回, 类似
            #pragma unroll
            for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_waitcnt lgkmcnt(%0)" :: "B"(2 - min_tile_k - 1));
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 2");
                int pv_tile_id = n_loop + N_LOOP_STEP + 1;
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = flash::mmac_4interleave<Element, ElementAccum>(
                            p_reg[0][min_tile_k * 2 + min_tile_m].f16x4,
                            v_reg[min_tile_k].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
                asm volatile("s_setprio 0");
            }
        }
    }

    __syncthreads(); // here, K/V use more lds, and thus reuse togather, need sync
}
