#pragma once
#include "mla_pv_gemm_utils_mls_ds.h"



template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum, bool Is_even_MN, bool Is_FlashMLA>
__forceinline__ __device__ void qk_gemm_prefetch_v_mls_ds_576_512(
        vec4_uint qv_ptr,
        vec4_uint q_ptr,
        vec4_uint k_ptr,
        vec4_uint v_ptr,
        Element* q_lds,
        Element* k_lds,
        Element* v_lds,
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 16) * (kBlockN / 32)][2],
        int warp_id,
        int seqlen_qv_stride,
        int __seqlen_q_stride,
        int seqlen_k_stride,
        int seqlen_v_stride,
        int max_seq_q_offset=0,
        int max_seq_k_offset=0) {

    // Simplify
    static_assert (kBlockK == 32 and "To simplify, only kBlockK = 32 is supported!");
    static_assert (WARP_M == 16 and "To simplify, only WARP_M = 16 is supported!");
    static_assert (WARP_N == 64 and "To simplify, only WARP_N = 64 is supported!");
    constexpr int WARP_NUM        = kBlockM / WARP_M;
    constexpr int kHeadDim_OPT       = (kHeadDim == 576) ? 64 : kHeadDim;
    constexpr int Q_LDS_LOAD_NUM  = (kBlockM * kBlockK) / (16 * 32);
    constexpr int Q_LOAD_REQUESTS = Q_LDS_LOAD_NUM / WARP_NUM;
    constexpr int K_LDS_LOAD_NUM  = (kHeadDim_OPT * WARP_N) / (32 * 16);
    constexpr int K_LOAD_REQUESTS = K_LDS_LOAD_NUM / WARP_NUM;
    constexpr int ELEMENT_BYTES   = sizeof(Element);
    constexpr int WARP_NUM_M = 2; 
    constexpr int WARP_NUM_N = 4;
    int warp_id_m = warp_id / WARP_NUM_N;
    int warp_id_n = warp_id % WARP_NUM_N;

    __builtin_amdgcn_sched_barrier(0);
    if constexpr (kBlockN == 128) {
        inline_vgpr4_init_zero_4x4x4(s_reg);
    } else {
        for (int i = 0; i < (WARP_M / 16) * (kBlockN / 32); ++i) {
            for (int j = 0; j < 2; ++j) {
                s_reg[i][j].u64[0] = 0.0f;
                s_reg[i][j].u64[1] = 0.0f;
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);

    // 准备 q,k 寄存器
    union_vec4_f16x2<Element> q_reg[(WARP_M * kBlockK) / (16 * 32)];
    union_vec4_f16x2<Element> k_reg[STAGES * (32 * kBlockK) / (32 * 32) * 2];

    // 计算 q_lds,k_lds 的起始偏移量
    int q_lds_base = reinterpret_cast<size_t>(q_lds);
    int k_lds_base = reinterpret_cast<size_t>(k_lds);

    // MLS
    vec4_uint q_srsrc;
    vec4_uint k_srsrc;

    q_srsrc[2] = __seqlen_q_stride;
    if constexpr (Is_FlashMLA) {
        k_srsrc[2] = seqlen_k_stride;
    } else {
        k_srsrc[2] = seqlen_v_stride;
    }
    
    q_srsrc[3] = 0; 
    k_srsrc[3] = 0;

    int q_stage_id = 0;
    int k_stage_id = 0;

    if constexpr (STAGES == 2) {
        q_stage_id ^= 1;
        k_stage_id ^= 1;
    }

    {   
        for(int k_loop = 1; k_loop < (kHeadDim / kBlockK); ++k_loop) {
            // k预取的标志位
            int k_even = ((k_loop & 1) == 0x0) ? 1 : 0;
            
            {
                uint64_t q_base_addr;
                int seqlen_q_stride;
                int kloop_true;
                if constexpr (Is_FlashMLA) {
                    q_srsrc[2] = __seqlen_q_stride;
                    q_base_addr = *(uint64_t*)&q_ptr;
                    seqlen_q_stride = __seqlen_q_stride;
                    kloop_true = k_loop;
                } else {
                    q_srsrc[2] = (k_loop >= 2) ? seqlen_qv_stride : __seqlen_q_stride;
                    q_base_addr = (k_loop >= 2) ? *(uint64_t*)&qv_ptr : *(uint64_t*)&q_ptr;
                    seqlen_q_stride = (k_loop >= 2) ? seqlen_qv_stride : __seqlen_q_stride;
                    kloop_true = (k_loop >= 2) ? (k_loop - 2) : (k_loop);
                }

                *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(q_base_addr + (kloop_true * kBlockK + warp_id * 16 * seqlen_q_stride) * ELEMENT_BYTES);
                int nm_filter = inline_min_max<0,16>(16 * warp_id + 16 - max_seq_q_offset);
                q_srsrc[3] = max_seq_q_offset % kBlockM == 0 ? 0: nm_filter << 8;
                int lds_offset = (q_stage_id * kBlockM * kBlockK + warp_id * 16 * 32) * ELEMENT_BYTES;
                flash::wait_all_warp_arrived();
                inline_matrix_load_32x16_b16_lds_trans<0, 1>(q_lds, q_srsrc, lds_offset, 0);
                
                if (k_even) {
                    k_stage_id ^= 1;
                    int nm_filter =  inline_min_max<0,16>(16 * warp_id_n + 16 - max_seq_k_offset);
                    if constexpr (Is_FlashMLA) {
                        *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_ptr + (warp_id_m * 32 + warp_id_n * 16 * seqlen_k_stride + ((k_loop) / 2) * kHeadDim_OPT) * ELEMENT_BYTES);
                    } else {
                        *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_ptr + (warp_id_m * 32 + warp_id_n * 16 * seqlen_v_stride + ((k_loop - 2) / 2) * kHeadDim_OPT) * ELEMENT_BYTES);
                    }
                    k_srsrc[3] = (max_seq_k_offset % kBlockN == 0x0 ? 0: nm_filter) << 8;
                    int lds_offset = (k_stage_id * WARP_N * kHeadDim_OPT + warp_id * 32 * 16) * ELEMENT_BYTES;
                    flash::wait_all_warp_arrived();
                    inline_matrix_load_32x16_b16_lds_trans<0, 0>(k_lds, k_srsrc, lds_offset, 0);
                }
            }

            // 不对称MLS指令
            if (k_even) {
                flash::wait_buffer_data_arrived<true>(Q_LOAD_REQUESTS + K_LOAD_REQUESTS);
            } else {
                flash::wait_buffer_data_arrived<true>(Q_LOAD_REQUESTS);
            }
            
            q_stage_id ^= 1;

            // Q DS
            {
                int q_lds_load_offset = q_lds_base + (q_stage_id * kBlockM * kBlockK + warp_id * 16 * 32) * ELEMENT_BYTES;
                DS_READ_MATRIX_32X16_B16(q_lds_load_offset, q_reg[0].f16, true);  
            }

            k_stage_id ^= 1;
            int stage_id = 0;

            // K DS  
            {
                int k_lds_load_offset = k_lds_base + (k_stage_id * WARP_N * kHeadDim_OPT + k_even * 32 * 64 + 0 * 32 * 32) * ELEMENT_BYTES;
                DS_READ_MATRIX_32X32_B16(k_lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);
            }

            // K DS PRE
            stage_id ^= 1;
            {
                int k_lds_load_offset = k_lds_base + (k_stage_id * WARP_N * kHeadDim_OPT + k_even * 32 * 64 + 1 * 32 * 32) * ELEMENT_BYTES;
                DS_READ_MATRIX_32X32_B16(k_lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);
            }

            // Wait DS
            flash::wait_lds_data_arrived<false>(2);
            flash::raise_priority();

            // MMAC 
            stage_id ^= 1;
            { 
                int min_tile_n = 0;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[stage_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[0].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[stage_id][min_tile_n].f32);
                }
            }

            { 
                int min_tile_n = 1;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[stage_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[0].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[stage_id][min_tile_n].f32);
                }
            }

            flash::lower_priority();
            flash::wait_lds_data_arrived<false>(0);
            flash::raise_priority();
            stage_id ^= 1;
            { 
                int min_tile_n = 0;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[stage_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[0].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[stage_id][min_tile_n].f32);
                }
            }

            { 
                int min_tile_n = 1;
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                        int k_tile_id  = stage_id * 2 + min_tile_n;
                        s_reg[stage_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                            q_reg[0].f16x4[min_tile_k],
                            k_reg[k_tile_id].f16x4[min_tile_k],
                            s_reg[stage_id][min_tile_n].f32);
                }
            }

            flash::lower_priority();
            
        }
        
        constexpr int k_loop = kHeadDim / kBlockK; 
        constexpr int k_even = ((k_loop & 1) == 0x0) ? 1 : 0; 

        if constexpr (k_even) {
            k_stage_id ^= 1;
        }

        // 等回最后的q_panel
        flash::wait_buffer_data_arrived<true>(0);

        // Q DS
        q_stage_id ^= 1;
        {
            int q_lds_load_offset = q_lds_base + (q_stage_id * kBlockM * kBlockK + warp_id * 16 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X16_B16(q_lds_load_offset, q_reg[0].f16, true);  
        }

        // K DS  
        k_stage_id ^= 1;
        int stage_id = 0;
        {   
            int k_lds_load_offset = k_lds_base + (k_stage_id * WARP_N * kHeadDim_OPT + k_even * 32 * 64 + 0 * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16(k_lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);
        }

        // K DS PRE
        stage_id ^= 1;
        {
            int k_lds_load_offset = k_lds_base + (k_stage_id * WARP_N * kHeadDim_OPT + k_even * 32 * 64 + 1 * 32 * 32) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16(k_lds_load_offset, k_reg[stage_id * 2].f16, k_reg[stage_id * 2 + 1].f16, true);
        }

        // Wait DS
        flash::wait_lds_data_arrived<false>(2);
        flash::raise_priority();

        // MMAC 
        stage_id ^= 1;
        { 
            int min_tile_n = 0;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[stage_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[0].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[stage_id][min_tile_n].f32);
            }
        }

        { 
            int min_tile_n = 1;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[stage_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[0].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[stage_id][min_tile_n].f32);
            }
        }

        flash::lower_priority();
        flash::wait_lds_data_arrived<false>(0);
        flash::raise_priority();
        stage_id ^= 1;
        { 
            int min_tile_n = 0;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[stage_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[0].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[stage_id][min_tile_n].f32);
            }
        }

        { 
            int min_tile_n = 1;
            #pragma unroll
            for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    int k_tile_id  = stage_id * 2 + min_tile_n;
                    s_reg[stage_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[0].f16x4[min_tile_k],
                        k_reg[k_tile_id].f16x4[min_tile_k],
                        s_reg[stage_id][min_tile_n].f32);
            }
        }

        flash::lower_priority();

    }

    if constexpr (STAGES == 2) {
    #if defined(__gfx938__) || defined(__gfx946__) || (defined(__gfx92a__) && defined(YY_USE_MPERMUTE))
        prefetch_v_to_lds_mls_ds_576_512<kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, Element, Is_even_MN>(v_ptr, v_lds, warp_id, seqlen_v_stride, max_seq_k_offset);
    #else

    #endif
    }

} // qk_gemm