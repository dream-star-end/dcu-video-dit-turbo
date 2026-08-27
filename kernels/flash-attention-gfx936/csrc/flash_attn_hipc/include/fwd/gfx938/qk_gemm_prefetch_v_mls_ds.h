#pragma once
#include "pv_gemm_utils_mls_ds.h"



template<int kHeadDim, int kHeadDimV, int computeHeadDim, int TailTile16, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum, bool Is_even_MN>
__forceinline__ __device__ void qk_gemm_prefetch_v_mls_ds(
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
    constexpr int WARP_NUM        = kBlockM / WARP_M;
    constexpr int kHeadDim_       = (kHeadDim == 192) ? 128 : kHeadDim;
    constexpr int K_LDS_LOAD_NUM  = WARP_N * kHeadDim_ / (32 * 32);
    constexpr int K_LOAD_REQUESTS = K_LDS_LOAD_NUM / WARP_NUM;
    constexpr int ELEMENT_BYTES   = sizeof(Element);

    // sync V lds usage
    flash::wait_all_warp_arrived();

    // here, v_mov_b64 can be applied
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (kBlockN == 128) {
        #pragma unroll
        for (int i = 0; i < (WARP_M / 32) * (kBlockN / 32); ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                asm volatile("v_mov_b64 %0, 0x0"
                    : "=v"(s_reg[i][j].u64[0])
                    :);
                asm volatile("v_mov_b64 %0, 0x0"
                    : "=v"(s_reg[i][j].u64[1])
                    :);
            }
        }
    } else {
        for (int i = 0; i < (WARP_M / 32) * (kBlockN / 32); ++i) {
            for (int j = 0; j < 4; ++j) {
                s_reg[i][j].u64[0] = 0;
                s_reg[i][j].u64[1] = 0;
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);

    // 准备 k 寄存器
    union_vec4_f16x2<Element> k_reg[STAGES * (WARP_N * kBlockK) / (32 * 32) * 2];

    // 计算 k lds 的起始偏移量
    int k_lds_base = reinterpret_cast<size_t>(k_lds);

    // MLS
    vec4_uint k_srsrc;
    k_srsrc[2] = seqlen_k_stride;  // stride
    k_srsrc[3] = 0;
    int n_stage_id = 0;

    n_stage_id ^= 1;
    // n_loop = 0 第一阶段，prefetch_k_to_lds 已 load 128 * 32 K矩阵
    #pragma unroll
    for(int n_loop = 1; n_loop < (kBlockN / WARP_N); ++n_loop) {
        // MLS
        if constexpr (kHeadDim == 128) {
            *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_ptr + (n_loop * WARP_N * seqlen_k_stride + warp_id * 32) * ELEMENT_BYTES);
            if constexpr (true) {
                int nm_filter = inline_min_max<0, 32>(n_loop * WARP_N + 32 - max_seq_k_offset);
                k_srsrc[3] = max_seq_k_offset % kBlockN == 0x0 ? 0: nm_filter << 8; // set only once
            }
            int lds_offset = (n_stage_id * WARP_N * kHeadDim_ + warp_id * 32 * 32) * ELEMENT_BYTES;
            flash::wait_all_warp_arrived();
            inline_matrix_load_32x32_b16_lds_trans<0, 0>(k_lds, k_srsrc, lds_offset, 0);
        } else if constexpr (kHeadDim == 192) {
            int warp_id_m = warp_id / 2;
            int warp_id_n = warp_id % 2;
            int k_load = 1;
            int n_loop_ = n_loop - 1; // Add this to support headdim>128, we have to deal with previous n_loop
            *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_ptr + (n_loop_ * WARP_N * seqlen_k_stride + warp_id_m * 32 + warp_id_n * 16 * seqlen_k_stride + k_load * 32 * WARP_NUM) * ELEMENT_BYTES);
            if constexpr (true) {
                int nm_filter = inline_min_max<0, 16>(n_loop_ * WARP_N + warp_id_n * 16 + 16 - max_seq_k_offset);
                k_srsrc[3] = max_seq_k_offset % kBlockN == 0x0 ? 0: nm_filter << 8;
            }
            int lds_offset = (n_stage_id * WARP_N * kHeadDim_ + warp_id * 32 * 16) * ELEMENT_BYTES;
            flash::wait_all_warp_arrived();
            inline_matrix_load_32x16_b16_lds_trans<0, 0>(k_lds, k_srsrc, lds_offset, 0);
        }

        // Wait MLS
        n_stage_id ^= 1;
        int stage_id = 0;
        flash::wait_buffer_data_arrived<true>(K_LOAD_REQUESTS);

        // DS
        {
            constexpr int k_loop = 0;
            int lds_load_offset = k_lds_base + (n_stage_id * WARP_N * kHeadDim_ + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);
        }
        stage_id ^= 1;
        for(int k_loop = 1; k_loop < (computeHeadDim / kBlockK); ++k_loop) {

            // Wait for special headdim
            if constexpr (kHeadDim == 192) {
                if ((k_loop & 3) == 0x0) {
                    flash::wait_buffer_data_arrived<true>(0);   
                }
            }

            // DS
            int lds_load_offset = k_lds_base + (n_stage_id * WARP_N * kHeadDim_ + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);

            // Wait DS
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
            flash::lower_priority();
            // MLS for special headdim
            if constexpr (kHeadDim == 192) { // Once finish 128x32, we need issue next MLS instruction, and get now we need use data via wait
                if ((k_loop & 3) == 0x0) {
                    // n_stage_id ^=1;
                    int k_load = k_loop / 4;
                    int n_loop_ = ((kHeadDim / kBlockK) - k_loop) < 4 ? (k_load = 0, n_loop): n_loop_;   // if finish kHeadDim*WarpN prefetch, we prefetch next n_loop data
                    *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_ptr + (n_loop_ * WARP_N * seqlen_k_stride + warp_id * 32 + k_load * 32 * WARP_NUM) * ELEMENT_BYTES);
                    if constexpr (true) {
                        int nm_filter = inline_min_max<0, 32>(n_loop_ * WARP_N + 32 - max_seq_k_offset);
                        k_srsrc[3] = max_seq_k_offset % kBlockN == 0x0 ? 0: nm_filter << 8;
                    }
                    int lds_offset = (n_stage_id * WARP_N * kHeadDim_ + warp_id * 32 * 32) * ELEMENT_BYTES;
                    flash::wait_all_warp_arrived();
                    inline_matrix_load_32x32_b16_lds_trans<0, 0>(k_lds, k_srsrc, lds_offset, 0);
                }
            }
        }
        stage_id ^= 1;
        // Wait DS
        flash::wait_lds_data_arrived<false>(1);
        flash::raise_priority();
        // MMAC
        {
            int min_tile_n = 0;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < TailTile16; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = computeHeadDim / kBlockK - 1;
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
            for(int min_tile_k = 0; min_tile_k < TailTile16; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = computeHeadDim / kBlockK - 1;
                    int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[q_tile_id].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
        flash::lower_priority();
    }

    {
        constexpr int n_loop = kBlockN / WARP_N;
        // MLS for special headdim
        if constexpr (kHeadDim == 192) {
            int warp_id_m = warp_id / 2;
            int warp_id_n = warp_id % 2;
            constexpr int n_loop_ = n_loop - 1;
            int k_load = 1;
            n_stage_id ^= 1;                
            *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_ptr + (n_loop_ * WARP_N * seqlen_k_stride + warp_id_m * 32 + warp_id_n * 16 * seqlen_k_stride + k_load * 32 * WARP_NUM) * ELEMENT_BYTES);
            if constexpr (true) {
                int nm_filter = inline_min_max<0, 16>(n_loop_ * WARP_N + warp_id_n * 16 + 16 - max_seq_k_offset);
                k_srsrc[3] = max_seq_k_offset % kBlockN == 0x0 ? 0: nm_filter << 8;
            }
            int lds_offset = (n_stage_id * WARP_N * kHeadDim_ + warp_id * 32 * 16) * ELEMENT_BYTES;
            flash::wait_all_warp_arrived();
            inline_matrix_load_32x16_b16_lds_trans<0, 0>(k_lds, k_srsrc, lds_offset, 0);
        }
        
        // Wait MLS
        n_stage_id ^= 1;
        int stage_id = 0;
        if constexpr (kHeadDim == 128) {
            flash::wait_buffer_data_arrived<true>(0);
        } else if constexpr (kHeadDim == 192) {
            flash::wait_buffer_data_arrived<true>(K_LOAD_REQUESTS);
        }

        // DS
        {
            int k_loop = 0;
            int lds_load_offset = k_lds_base + (n_stage_id * WARP_N * kHeadDim_ + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);
        }
        stage_id ^= 1;
        for(int k_loop = 1; k_loop < (computeHeadDim / kBlockK); ++k_loop) {

            // Wait for special headdim
            if constexpr (kHeadDim == 192) {
                if ((k_loop & 3) == 0x0) {
                    flash::wait_buffer_data_arrived<true>(0);
                }
            }

            // DS
            int lds_load_offset = k_lds_base + (n_stage_id * WARP_N * kHeadDim_ + k_loop * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16(lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);

            // Wait DS
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
            flash::lower_priority();
        }
        stage_id ^= 1;
        // MMAC
        flash::wait_lds_data_arrived<false>(1);
        flash::raise_priority();
        {
            int min_tile_n = 0;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < TailTile16; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = computeHeadDim / kBlockK - 1;
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
            for(int min_tile_k = 0; min_tile_k < TailTile16; ++min_tile_k) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int k_loop_idx = computeHeadDim / kBlockK - 1;
                    int q_tile_id  = k_loop_idx * 2 + min_tile_m;
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[q_tile_id].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[n_loop - 1][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
        flash::lower_priority();
    }

    // sync V lds usage
    flash::wait_all_warp_arrived();

    if constexpr (STAGES == 2) {
    #if defined(__gfx938__) || defined(__gfx946__) || (defined(__gfx92a__) && defined(YY_USE_MPERMUTE)) // 有的 prefetch v 写到了 mha 主 kernel 代码里
        prefetch_v_to_lds_mls_ds<kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, TailTile16, Element, Is_even_MN>(v_ptr, v_lds, warp_id, seqlen_v_stride, max_seq_k_offset);
    #else

    #endif
    }

} // qk_gemm

