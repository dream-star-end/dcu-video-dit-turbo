#include "mla_qk_gemm_utils_mls_ds.h"
#include "static_switch.h"


template<bool PREFETCH_K, int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum, bool Is_even_MN>
__forceinline__ __device__ void pv_gemm_prefetch_k_mls_ds_576_512(
        vec4_uint q_ptr,
        vec4_uint k_ptr,
        vec4_uint v_ptr,
        Element* q_lds,
        Element* k_lds,
        Element* v_lds,
        union_vec2_f16x2<Element> p_reg[(WARP_M / 16) * (kBlockK / 32)][2],
        vec4_Accum<ElementAccum> pv_reg[(kHeadDimV / kBlockN) * (WARP_M / 16) * (kBlockN / 32)][2],
        int warp_id,
        int seqlen_q_stride,
        int seqlen_k_stride,
        int seqlen_v_stride,
        int max_seq_q_offset=0,
        int max_seq_kv_offset=0) {

    constexpr int WARP_NUM = kBlockM * kBlockN / (WARP_M * WARP_N);
    constexpr int WARP_K   = 32;
    constexpr int READ_ONCE_COUNT = 32 * 32;
    constexpr int kHeadDimV_OPT = 256;          // lds 32x32x8x2B == 16KB
    constexpr int V_LDS_LOAD_NUM  = (kHeadDimV_OPT * WARP_K) / READ_ONCE_COUNT;
    constexpr int V_LOAD_REQUESTS = V_LDS_LOAD_NUM / WARP_NUM;
    constexpr int ELEMENT_BYTES   = sizeof(Element);
    static_assert (kBlockK >= 32, "Error: pv gemm kBlockK must be equal or greater than 32");
    static_assert (kBlockM >= WARP_M, "Error: pv gemm kBlockM must be equal or greater than WARP_M");
    static_assert (kBlockN == WARP_N, "Error: pv gemm kBlockN must be equal to WARP_N");
    static_assert (WARP_K == 32 and "Error: To simplify, only WARP_K = 32 is supported!");
    static_assert (WARP_M == 16 and "Error: To simplify, only WARP_M = 16 is supported!");
    static_assert (WARP_N == 32 and "Error: To simplify, only WARP_N = 32 is supported!");

    // 计算 V lds 起始偏移量
    int v_lds_base = reinterpret_cast<size_t>(v_lds);

    // 准备 V 寄存器
    union_vec4_f16x2<Element> v_reg[STAGES * (32 * WARP_N) / (32 * 32) * 2];

    // MLS
    vec4_uint v_srsrc;
    v_srsrc[0] = v_ptr[0];
    v_srsrc[1] = v_ptr[1];
    v_srsrc[2] = seqlen_v_stride; // stride
    v_srsrc[3] = 0;

    int lds_stage_id = 1;
    for (int n_loop = 1; n_loop < (kBlockK / WARP_K); ++n_loop) {
        // prefetch same warpk, next 32x256 G2S
        {   
            int n_load = 1;
            int n_loop_ = n_loop - 1;
            *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_ptr + (n_loop_ * WARP_K * seqlen_v_stride + warp_id * 32 + n_load * WARP_NUM * 32) * ELEMENT_BYTES);
            if constexpr (true) {
                int nm_filter     = inline_min_max<0, 32>(n_loop_ * WARP_K + 32 - max_seq_kv_offset);
                v_srsrc[3]        = max_seq_kv_offset % kBlockK == 0 ? 0: nm_filter << 8;
            }
            int lds_offset = (lds_stage_id * WARP_K * kHeadDimV_OPT + warp_id * 32 * 32) * ELEMENT_BYTES;
            flash::wait_all_warp_arrived();
            inline_matrix_load_32x32_b16_lds<0, 1>(v_lds, v_srsrc, lds_offset, 0);
        }
        
        // DS
        lds_stage_id ^= 1;
        int stage_id = 0;
        flash::wait_buffer_data_arrived<true>(V_LOAD_REQUESTS);

        int lds_load_offset = v_lds_base + (0/*k_loop*/ * 32 * 32 + lds_stage_id * WARP_K * kHeadDimV_OPT) * ELEMENT_BYTES;
        DS_READ_MATRIX_32X32_B16(lds_load_offset, v_reg[stage_id * 2 + 0].f16, v_reg[stage_id * 2 + 1].f16, false/*transpose*/);

        stage_id ^= 1;
        for (int k_loop = 1; k_loop < (kHeadDimV / kBlockN); ++k_loop) {

            // Wait for special headdim
            if ((k_loop & 7) == 0x0) {
                flash::wait_buffer_data_arrived<true>(0);   
            }

            int lds_load_offset = v_lds_base + (k_loop * 32 * 32 + lds_stage_id * WARP_K * kHeadDimV_OPT) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16(lds_load_offset, v_reg[stage_id * 2 + 0].f16, v_reg[stage_id * 2 + 1].f16, false/*transpose*/);
            flash::wait_lds_data_arrived<false>(3);

            // MMAC
            flash::raise_priority();
            stage_id ^= 1;
            {
                constexpr int min_tile_k = 0;
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 1; ++min_tile_m) {
                        int pv_tile_id = (STAGES == 2) ? k_loop - 1: k_loop;
                        int v_tile_id = stage_id * 2 + min_tile_k;
                        pv_reg[pv_tile_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                            p_reg[n_loop - 1][min_tile_k].f16x4,
                            v_reg[v_tile_id].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n].f32);
                    }
                }
            }
            flash::wait_lds_data_arrived<false>(2);
            {
                constexpr int min_tile_k = 1;
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 1; ++min_tile_m) {
                        int pv_tile_id = (STAGES == 2) ? k_loop - 1: k_loop;
                        int v_tile_id = stage_id * 2 + min_tile_k;
                        pv_reg[pv_tile_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                            p_reg[n_loop - 1][min_tile_k].f16x4,
                            v_reg[v_tile_id].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n].f32);
                    }
                }
            }
            flash::lower_priority();

            // MLS for special headdimV
            if ((k_loop & 7) == 0x0) {
                int n_loop_ = n_loop;
                    
                if constexpr (Is_even_MN) {
                    *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_ptr + (n_loop_ * WARP_K * seqlen_v_stride + warp_id * 32 + 0 * 32 * WARP_NUM) * ELEMENT_BYTES);
                } else {
                    int nm_filter_max = n_loop_ * WARP_K + 32 - max_seq_kv_offset;
                    int real_mls_loop = nm_filter_max >= 32 ? 0: n_loop_; // 如果全越界了, 则只访问 n_loop = 0 的那波数据
                    int nm_filter     = inline_min_max<0, 32>(real_mls_loop * WARP_K + 32 - max_seq_kv_offset); // 重新计算 nm_filter
                    v_srsrc[3]        = nm_filter << 8;
                    *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_ptr + (real_mls_loop * WARP_K * seqlen_v_stride + warp_id * 32 + 0 * 32 * WARP_NUM) * ELEMENT_BYTES);
                }
                int lds_offset = (lds_stage_id * WARP_K * kHeadDimV_OPT + warp_id * 32 * 32) * ELEMENT_BYTES;
                flash::wait_all_warp_arrived(); // 预防有 warp 还没算完7,还在读 v lds, 若是此时写 v lds,则 data cover
                inline_matrix_load_32x32_b16_lds<0, 1>(v_lds, v_srsrc, lds_offset, 0);
            }
        }
        stage_id ^= 1;
        // Wait DS
        flash::wait_lds_data_arrived<false>(1);
        // last mmac
        flash::raise_priority();
        {
            constexpr int min_tile_k = 0;
            #pragma unroll
            for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 1; ++min_tile_m) {
                    int pv_tile_id = (kHeadDimV / kBlockN) - 1;
                    int v_tile_id = stage_id * 2 + min_tile_k;
                    pv_reg[pv_tile_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                        p_reg[n_loop - 1][min_tile_k].f16x4,
                        v_reg[v_tile_id].f16x4[min_tile_n],
                        pv_reg[pv_tile_id][min_tile_n].f32);
                }
            }
        }
        flash::wait_lds_data_arrived<false>(0);
        {
            constexpr int min_tile_k = 1;
            #pragma unroll
            for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 1; ++min_tile_m) {
                    int pv_tile_id = (kHeadDimV / kBlockN) - 1;
                    int v_tile_id = stage_id * 2 + min_tile_k;
                    pv_reg[pv_tile_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                        p_reg[n_loop - 1][min_tile_k].f16x4,
                        v_reg[v_tile_id].f16x4[min_tile_n],
                        pv_reg[pv_tile_id][min_tile_n].f32);
                }
            }
        }
        flash::lower_priority();
    }

    {
        constexpr int n_loop = kBlockK / WARP_K;
        // MLS for special headdimV
        {
            constexpr int n_loop_ = n_loop - 1;
            int n_load = 1;
            lds_stage_id ^= 1;
            
            if constexpr (Is_even_MN) {
                *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_ptr + (n_loop_ * WARP_K * seqlen_v_stride + warp_id * 32 + n_load * WARP_NUM * 32) * ELEMENT_BYTES);
            } else {
                int nm_filter_max = n_loop_ * WARP_K + 32 - max_seq_kv_offset;
                int real_mls_loop = nm_filter_max >= 32 ? 0: n_loop_; // 如果全越界了, 则只访问 n_loop = 0 的那波数据
                int nm_filter     = inline_min_max<0, 32>(real_mls_loop * WARP_K + 32 - max_seq_kv_offset); // 重新计算 nm_filter
                v_srsrc[3]        = nm_filter << 8;
                *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_ptr + (real_mls_loop * WARP_K * seqlen_v_stride + warp_id * 32 + n_load * WARP_NUM * 32) * ELEMENT_BYTES);
            }
            int lds_offset = (lds_stage_id * WARP_K * kHeadDimV_OPT + warp_id * 32 * 32) * ELEMENT_BYTES;
            flash::wait_all_warp_arrived();
            inline_matrix_load_32x32_b16_lds<0, 1>(v_lds, v_srsrc, lds_offset, 0);
        }

        lds_stage_id ^= 1;
        int stage_id = 0;

        flash::wait_buffer_data_arrived<true>(V_LOAD_REQUESTS); // [TODO]更早的预取

        // DS
        int lds_load_offset = v_lds_base + (0/*k_loop*/ * 32 * 32 + lds_stage_id * WARP_K * kHeadDimV_OPT) * ELEMENT_BYTES;
        DS_READ_MATRIX_32X32_B16(lds_load_offset, v_reg[stage_id * 2 + 0].f16, v_reg[stage_id * 2 + 1].f16, false/*transpose*/);

        stage_id ^= 1;
        for (int k_loop = 1; k_loop < (kHeadDimV / kBlockN); ++k_loop) {

            // Wait for special headdim
            if ((k_loop & 7) == 0x0) {
                flash::wait_buffer_data_arrived<true>(0);   
            }

            // DS
            int lds_load_offset = v_lds_base + (k_loop * 32 * 32 + lds_stage_id * WARP_K * kHeadDimV_OPT) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16(lds_load_offset, v_reg[stage_id * 2 + 0].f16, v_reg[stage_id * 2 + 1].f16, false/*transpose*/);
            flash::wait_lds_data_arrived<false>(3);

            // MMAC
            flash::raise_priority();
            stage_id ^= 1;
            {
                constexpr int min_tile_k = 0;
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 1; ++min_tile_m) {
                        int pv_tile_id = (STAGES == 2) ? k_loop - 1: k_loop;
                        int v_tile_id = stage_id * 2 + min_tile_k;
                        pv_reg[pv_tile_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                            p_reg[n_loop - 1][min_tile_k].f16x4,
                            v_reg[v_tile_id].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n].f32);
                    }
                }
            }
            flash::wait_lds_data_arrived<false>(2);
            {
                constexpr int min_tile_k = 1;
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 1; ++min_tile_m) {
                        int pv_tile_id = (STAGES == 2) ? k_loop - 1: k_loop;
                        int v_tile_id = stage_id * 2 + min_tile_k;
                        pv_reg[pv_tile_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                            p_reg[n_loop - 1][min_tile_k].f16x4,
                            v_reg[v_tile_id].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n].f32);
                    }
                }
            }
            flash::lower_priority();
        }

        stage_id ^= 1;
        flash::wait_lds_data_arrived<false>(1);
        // last mmac
        flash::raise_priority();
        {
            constexpr int min_tile_k = 0;
            #pragma unroll
            for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 1; ++min_tile_m) {
                    int pv_tile_id = (kHeadDimV / kBlockN) - 1;
                    int v_tile_id = stage_id * 2 + min_tile_k;
                    pv_reg[pv_tile_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                        p_reg[n_loop - 1][min_tile_k].f16x4,
                        v_reg[v_tile_id].f16x4[min_tile_n],
                        pv_reg[pv_tile_id][min_tile_n].f32);
                }
            }
        }
        flash::wait_lds_data_arrived<false>(0);
        {
            constexpr int min_tile_k = 1;
            #pragma unroll
            for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 1; ++min_tile_m) {
                    int pv_tile_id = (kHeadDimV / kBlockN) - 1;
                    int v_tile_id = stage_id * 2 + min_tile_k;
                    pv_reg[pv_tile_id][min_tile_n].f32 = mmac_4interleave<Element, ElementAccum>(
                        p_reg[n_loop - 1][min_tile_k].f16x4,
                        v_reg[v_tile_id].f16x4[min_tile_n],
                        pv_reg[pv_tile_id][min_tile_n].f32);
                }
            }
        }
        flash::lower_priority();
    }

    
    // 预取Q K
    if constexpr (PREFETCH_K) {
        prefetch_q_to_lds_mls_ds_576_512<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(q_ptr, q_lds, warp_id, seqlen_q_stride, max_seq_q_offset);
        prefetch_k_to_lds_mls_ds_576_512<kHeadDim, kBlockK, kBlockN, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, max_seq_kv_offset - kBlockK);
    }
}