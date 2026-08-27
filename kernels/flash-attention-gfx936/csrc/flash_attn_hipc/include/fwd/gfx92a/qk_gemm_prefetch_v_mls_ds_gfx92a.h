#pragma once
#include "fwd/gfx938/pv_gemm_utils_mls_ds.h"
#include "fwd/gfx92a/qk_gemm_utils_mls_ds_gfx92a.h"



template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum, bool Is_even_MN>
__forceinline__ __device__ void qk_gemm_prefetch_v_mls_ds_gfx92a_2TG(
        vec4_uint k_ptr,
        vec4_uint v_ptr,
        Element* k_lds,
        Element* v_lds,
        union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2],
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (kBlockN / 32)][4],
        int warp_id,
        int seqlen_k_stride,
        int seqlen_v_stride,
        int max_seq_k_offset=0) {

    // Simplify
    static_assert (kBlockK == 32 and "To simplify, only kBlockK = 32 is supported!");
    static_assert (WARP_M == 32 and "To simplify, only WARP_M = 32 is supported!");
    static_assert (WARP_N == 32 and "To simplify, only WARP_N = 32 is supported!");
    constexpr int WARP_NUM = kBlockM / WARP_M;
    constexpr int k_lds_load_num  = WARP_N * kHeadDim / (32 * 32);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num / WARP_NUM;
    constexpr int ELEMENT_BYTES   = sizeof(Element);

    // 准备 K 寄存器
    union_vec4_f16x2<Element> k_reg[STAGES * (WARP_N * kBlockK) / (32 * 32) * 2];

    // 计算 K lds 起始偏移量
    int k_lds_base = reinterpret_cast<size_t>(k_lds);

    // here, v_mov_b64 can be applied
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (kBlockN == 128) {
        inline_vgpr4_init_zero_4x4x4(s_reg);
    } else {
        for (int i = 0; i < (WARP_M / 32) * (kBlockN / 32); ++i) { // for kBlockN = 64, only wave 0 get the right QK gemm results
            for (int j = 0; j < 4; ++j) {
                s_reg[i][j].u64[0] = 0;
                s_reg[i][j].u64[1] = 0;
            }
        }
    }
    flash::lower_priority();

    // MLS
    vec4_uint k_srsrc;
    k_srsrc[2] = seqlen_k_stride;  // stride
    k_srsrc[3] = 0;

    #pragma unroll
    for(int n_loop = 0; n_loop < (kBlockN / WARP_N); ++n_loop) {

        // Wait global data
        flash::wait_buffer_data_arrived<true>(kBlockN / WARP_N - n_loop - 1);

        // DS
        int stage_id = 0;
        {
            constexpr int k_loop = 0;
            int lds_load_offset = k_lds_base + (n_loop * WARP_N * kHeadDim + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16_GFX946(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);
        }
        stage_id ^= 1;
        #pragma unroll
        for(int k_loop = 1; k_loop < (kHeadDim / kBlockK); ++k_loop) {
            // DS
            int lds_load_offset = k_lds_base + (n_loop * WARP_N * kHeadDim + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16_GFX946(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);

            flash::wait_lds_data_arrived<false>(3);
            flash::raise_priority();
            // MMAC
            stage_id ^= 1;
            {
                int min_tile_n = 0;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                        int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[q_tile_id].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
            flash::wait_lds_data_arrived<false>(2);
            {
                int min_tile_n = 1;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                        int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[q_tile_id].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
            flash::lower_priority();
        }
        stage_id ^= 1;
        // Wait DS
        flash::wait_lds_data_arrived<false>(1);
        flash::raise_priority();
        // last mmac
        {
            int min_tile_n = 0;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = kHeadDim / kBlockK - 1;
                    int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[q_tile_id].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
        flash::wait_lds_data_arrived<false>(0);
        {
            int min_tile_n = 1;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = kHeadDim / kBlockK - 1;
                    int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[q_tile_id].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[n_loop][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
        flash::lower_priority();
    }

    if constexpr (STAGES == 2) {
    #if defined(__gfx938__) || defined(__gfx946__)
        prefetch_v_to_lds_mls_ds<kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, 2, Element, Is_even_MN>(v_ptr, v_lds, warp_id, seqlen_v_stride, max_seq_k_offset);
    #else

    #endif
    }

} // qk_gemm




















template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum, bool Is_even_MN>
__forceinline__ __device__ void qk_gemm_prefetch_v_mls_ds_gfx92a(
        vec4_uint k_ptr,
        vec4_uint v_ptr,
        Element* k_lds,
        Element* v_lds,
        union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2],
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (kBlockN / 32)][4],
        int warp_id,
        int seqlen_k_stride,
        int seqlen_v_stride,
        int max_seq_k_offset=0) {

    // Simplify
    static_assert (kBlockK == 32 and "To simplify, only kBlockK = 32 is supported!");
    static_assert (WARP_M == 32 and "To simplify, only WARP_M = 32 is supported!");
    static_assert (WARP_N == 32 and "To simplify, only WARP_N = 32 is supported!");
    constexpr int WARP_NUM = kBlockM / WARP_M;
    constexpr int k_lds_load_num  = WARP_N * kHeadDim / (32 * 32);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num / WARP_NUM;
    constexpr int ELEMENT_BYTES   = sizeof(Element);

    // Prepare regs for k
    union_vec4_f16x2<Element> k_reg[STAGES * (WARP_N * kBlockK) / (32 * 32) * 2];

    // Zero-initialize s_reg
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (kBlockN == 128) {
        inline_vgpr4_init_zero_4x4x4(s_reg);
    } else {
        for (int i = 0; i < (WARP_M / 32) * (kBlockN / 32); ++i) { // for kBlockN = 64, only wave 0 get the right QK gemm results
            for (int j = 0; j < 4; ++j) {
                s_reg[i][j].u64[0] = 0;
                s_reg[i][j].u64[1] = 0;
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);

    // Prepare MLS buffer resource sregs
    vec4_uint k_srsrc;
    k_srsrc[2] = seqlen_k_stride;  // stride
    k_srsrc[3] = 0;
    int n_stage_id = 1;

    #pragma unroll
    for(int n_loop = 1; n_loop < (kBlockN / WARP_N); ++n_loop) {
        // MLS dispatch
        const bool has_tail = max_seq_k_offset % kBlockN != 0;
        const int nm_filter_max = n_loop * WARP_N + 32 - max_seq_k_offset;
        const int k_load_loop = has_tail && nm_filter_max >= 32 ? 0 : n_loop;
        const int nm_filter = inline_min_max<0, 31>(k_load_loop * WARP_N + 32 - max_seq_k_offset);
        const int __nm_filter = __builtin_amdgcn_readfirstlane(nm_filter);
        *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_ptr + (k_load_loop * WARP_N * seqlen_k_stride + warp_id * 32) * ELEMENT_BYTES);
        k_srsrc[3] = has_tail ? __nm_filter << 8 : 0; // set only once
        int lds_write_offset = (n_stage_id * WARP_N * kHeadDim + warp_id * 32 * 32) * ELEMENT_BYTES;
        flash::wait_all_warp_arrived(); // sync lds usage when ping-pong
        inline_matrix_load_32x32_b16_lds_trans<0, 0>(k_lds, k_srsrc, lds_write_offset, 0);

        // Wait MLS
        n_stage_id ^= 1;
        int stage_id = 0;
        flash::wait_buffer_data_arrived<true>(K_LOAD_REQUESTS);

        // DS dispatch
        {
            constexpr int k_loop = 0;
            int lds_load_offset = (n_stage_id * WARP_N * kHeadDim + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16_GFX946(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);
        }
        stage_id ^= 1;
        for(int k_loop = 1; k_loop < (kHeadDim / kBlockK); ++k_loop) {
            // DS dispatch
            int lds_load_offset = (n_stage_id * WARP_N * kHeadDim + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16_GFX946(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);

            // Wait DS
            flash::wait_lds_data_arrived<false>(3);

            // MMAC
            asm volatile("s_setprio 2");
            stage_id ^= 1;
            {
                int min_tile_n = 0;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                        int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[q_tile_id].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
            flash::wait_lds_data_arrived<false>(2);
            {
                int min_tile_n = 1;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                        int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[q_tile_id].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
            asm volatile("s_setprio 0");
        }
        stage_id ^= 1;
        // Wait DS
        flash::wait_lds_data_arrived<false>(1);
        asm volatile("s_setprio 2");
        // MMAC
        {
            int min_tile_n = 0;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = kHeadDim / kBlockK - 1;
                    int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[q_tile_id].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
        flash::wait_lds_data_arrived<false>(0);
        {
            int min_tile_n = 1;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = kHeadDim / kBlockK - 1;
                    int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[q_tile_id].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
        asm volatile("s_setprio 0");
    }

    {
        // Wait MLS
        constexpr int n_loop = 4;
        n_stage_id ^= 1;
        int stage_id = 0;
        flash::wait_buffer_data_arrived<true>(0);

        // DS dispatch
        {
            int k_loop = 0;
            int lds_load_offset = (n_stage_id * WARP_N * kHeadDim + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16_GFX946(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);
        }
        stage_id ^= 1;
        for(int k_loop = 1; k_loop < (kHeadDim / kBlockK); ++k_loop) {
            // DS dispatch
            int lds_load_offset = (n_stage_id * WARP_N * kHeadDim + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16_GFX946(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);

            // Wait DS
            flash::wait_lds_data_arrived<false>(3);

            // MMAC
            asm volatile("s_setprio 2");
            stage_id ^= 1;
            {
                int min_tile_n = 0;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                        int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[q_tile_id].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
            flash::wait_lds_data_arrived<false>(2);
            {
                int min_tile_n = 1;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                        int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[q_tile_id].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
            asm volatile("s_setprio 0");
        }
        stage_id ^= 1;
        flash::wait_lds_data_arrived<false>(1);
        // MMAC
        asm volatile("s_setprio 2"); // flash::raise_priority 性能下降严重, 157.8 -> 148.2 tflops, strange, 需要看汇编
        {                            // 对比汇编, 差异就在于单独的 s_setprio 会被胡乱调度到 mmac 中间, 但这样跑出来却性能更高; 强行加 scheduled barrier 跑出来性能更低;
            int min_tile_n = 0;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = kHeadDim / kBlockK - 1;
                    int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[q_tile_id].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
        flash::wait_lds_data_arrived<false>(0);
        {
            int min_tile_n = 1;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = kHeadDim / kBlockK - 1;
                    int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[q_tile_id].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
        asm volatile("s_setprio 0"); // flash::lower_priority 性能下降 0.6 tflops 左右
    }

    if constexpr (STAGES == 2) {
        prefetch_v_to_lds_mls_ds<kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, 2, Element, Is_even_MN>(v_ptr, v_lds, warp_id, seqlen_v_stride, max_seq_k_offset);
    }

} // qk_gemm
