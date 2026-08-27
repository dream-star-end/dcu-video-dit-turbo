#include "fwd/gfx92a/qk_gemm_utils_mls_ds_gfx92a.h"
#include "static_switch.h"


template<bool PREFETCH_K, int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum, bool Is_even_MN>
__forceinline__ __device__ void pv_gemm_prefetch_k_mls_ds_gfx92a(
        vec4_uint v_ptr,
        vec4_uint k_ptr,
        Element* v_lds,
        Element* k_lds,
        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockK / 32)][4],
        vec4_Accum<ElementAccum> pv_reg[(kHeadDimV / kBlockN) * (WARP_M / 32) * (kBlockN / 32)][4],
        int warp_id,
        int seqlen_k_stride,
        int seqlen_v_stride,
        int max_seq_kv_offset=0) {

    constexpr int WARP_NUM = kBlockM * kBlockN / (WARP_M * WARP_N);
    constexpr int WARP_K   = 32;
    constexpr int READ_ONCE_COUNT = 32 * 32;
    constexpr int V_LDS_LOAD_NUM  = (kHeadDimV * WARP_K) / READ_ONCE_COUNT;
    constexpr int V_LOAD_REQUESTS = V_LDS_LOAD_NUM / WARP_NUM;
    constexpr int ELEMENT_BYTES   = sizeof(Element);
    static_assert (kBlockK >= 32, "Error: pv gemm kBlockK must be equal or greater than 32");
    static_assert (kBlockM >= WARP_M, "Error: pv gemm kBlockM must be equal or greater than WARP_M");
    static_assert (kBlockN == WARP_N, "Error: pv gemm kBlockN must be equal to WARP_N");
    static_assert (WARP_K == 32 and "Error: To simplify, only WARP_K = 32 is supported!");
    static_assert (WARP_M == 32 and "Error: To simplify, only WARP_M = 32 is supported!");
    static_assert (WARP_N == 32 and "Error: To simplify, only WARP_N = 32 is supported!");

    // Prepare V regs
    union_vec4_f16x2<Element> v_reg[STAGES * (32 * WARP_N) / (32 * 32) * 2];

    // Prepare V lds offset
    int v_lds_base = 0; // reinterpret_cast<size_t>(v_lds); // ===> 性能下降 ?

    // Prepare MLS buffer resource sregs
    vec4_uint v_srsrc;
    v_srsrc[0] = v_ptr[0];
    v_srsrc[1] = v_ptr[1];
    v_srsrc[2] = seqlen_v_stride; // stride
    v_srsrc[3] = 0;

    int lds_stage_id = 1;

    // Main loop across blockN(128) among seqlenkv
    for (int n_loop = 1; n_loop < (kBlockK / WARP_K); ++n_loop) {

        // Do k-dim interleave for next mmac
        #if defined(__gfx92a__)
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 0 + 0].f16x4);
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 0 + 1].f16x4);
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 1 + 0].f16x4);
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 1 + 1].f16x4);
        #endif

        // MLS dispatch
        if constexpr (Is_even_MN) {
            *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_ptr + (n_loop * WARP_K * seqlen_v_stride + warp_id * 32) * ELEMENT_BYTES);
            v_srsrc[3] = 0x20000;
        } else {
            int nm_filter_max = n_loop * WARP_K + 32 - max_seq_kv_offset;
            int real_mls_loop = nm_filter_max >= 32 ? 0 : n_loop;
            int nm_filter = inline_min_max<0, 32>(real_mls_loop * WARP_K + 32 - max_seq_kv_offset);
            v_srsrc[3] = (nm_filter << 8) + 0x20000;
            *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_ptr + (real_mls_loop * WARP_K * seqlen_v_stride + warp_id * 32) * ELEMENT_BYTES);
        }

        int lds_write_offset = (lds_stage_id * WARP_K * kHeadDimV + warp_id * 32 * 32) * ELEMENT_BYTES;
        flash::wait_all_warp_arrived();
        inline_matrix_load_32x32_b16_lds<0, 1>(v_lds, v_srsrc, lds_write_offset, 0);

        // Wait buffer
        lds_stage_id ^= 1;
        int stage_id = 0;
        flash::wait_buffer_data_arrived<true>(V_LOAD_REQUESTS);

        // DS dispatch
        int lds_load_offset = (0/*k_loop*/ * 32 * 32 + lds_stage_id * WARP_K * kHeadDimV) * ELEMENT_BYTES;
        DS_READ_MATRIX_32X32_B16_ALT2(lds_load_offset, v_reg[stage_id * 2 + 0].f16, v_reg[stage_id * 2 + 1].f16, false/*transpose*/);

        // Wait ds_mpermute
        #if defined(__gfx92a__)
            flash::wait_lds_data_arrived<false>(2);
        #endif

        stage_id ^= 1;
        for (int k_loop = 1; k_loop < (kHeadDimV / kBlockN); ++k_loop) {

            // DS dispatch
            int lds_load_offset = (k_loop * 32 * 32 + lds_stage_id * WARP_K * kHeadDimV) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16_ALT2(lds_load_offset, v_reg[stage_id * 2 + 0].f16, v_reg[stage_id * 2 + 1].f16, false/*transpose*/);

            // Wait DS
            flash::wait_lds_data_arrived<false>(3);

            // MMAC
            stage_id ^= 1;
            {
                constexpr int min_tile_k = 0;
                flash::raise_priority(1);
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int pv_tile_id = (STAGES == 2) ? k_loop - 1: k_loop;
                        int v_tile_id = stage_id * 2 + min_tile_k;
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                            v_reg[v_tile_id].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
                flash::lower_priority();
            }
            flash::wait_lds_data_arrived<false>(2);
            {
                constexpr int min_tile_k = 1;
                flash::raise_priority(1);
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int pv_tile_id = (STAGES == 2) ? k_loop - 1: k_loop;
                        int v_tile_id = stage_id * 2 + min_tile_k;
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                            v_reg[v_tile_id].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
                flash::lower_priority();
            }
        }
        stage_id ^= 1;
        // Wait DS
        flash::wait_lds_data_arrived<false>(1);
        // last mmac
        {
            constexpr int min_tile_k = 0;
            flash::raise_priority(1);
            #pragma unroll
            for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int pv_tile_id = (kHeadDimV / kBlockN) - 1;
                    int v_tile_id = stage_id * 2 + min_tile_k;
                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                        v_reg[v_tile_id].f16x4[min_tile_n],
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                }
            }
            flash::lower_priority();
        }
        flash::wait_lds_data_arrived<false>(0);
        {
            constexpr int min_tile_k = 1;
            flash::raise_priority(1);
            #pragma unroll
            for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int pv_tile_id = (kHeadDimV / kBlockN) - 1;
                    int v_tile_id = stage_id * 2 + min_tile_k;
                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                        v_reg[v_tile_id].f16x4[min_tile_n],
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                }
            }
            flash::lower_priority();
        }
    }

    // Prefetch K
    if constexpr (PREFETCH_K) {
        prefetch_k_to_lds_mls_ds<kHeadDim, kBlockK, kBlockN, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, max_seq_kv_offset);
    }

    {
        constexpr int n_loop = 4;

        // Do k-dim interleave for next mmac
        #if defined(__gfx92a__)
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 0 + 0].f16x4);
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 0 + 1].f16x4);
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 1 + 0].f16x4);
            ds_mpermute_kdim_for_mmac(p_reg[n_loop - 1][2 * 1 + 1].f16x4);
        #endif
        lds_stage_id ^= 1;
        int stage_id = 0;

        // Wait buffer
        if constexpr (PREFETCH_K) {
            flash::wait_buffer_data_arrived<true>(V_LOAD_REQUESTS);
        } else {
            flash::wait_buffer_data_arrived<true>(0);
        }

        // DS dispatch
        int lds_load_offset = (0/*k_loop*/ * 32 * 32 + lds_stage_id * WARP_K * kHeadDimV) * ELEMENT_BYTES;
        DS_READ_MATRIX_32X32_B16_ALT2(lds_load_offset, v_reg[stage_id * 2 + 0].f16, v_reg[stage_id * 2 + 1].f16, false/*transpose*/);

        // Wait ds_mpermute
        #if defined(__gfx92a__)
            flash::wait_lds_data_arrived<false>(2);
        #endif

        stage_id ^= 1;
        for (int k_loop = 1; k_loop < (kHeadDimV / kBlockN); ++k_loop) {

            // DS dispatch
            int lds_load_offset = (k_loop * 32 * 32 + lds_stage_id * WARP_K * kHeadDimV) * ELEMENT_BYTES;
            DS_READ_MATRIX_32X32_B16_ALT2(lds_load_offset, v_reg[stage_id * 2 + 0].f16, v_reg[stage_id * 2 + 1].f16, false/*transpose*/);

            // Wait DS
            flash::wait_lds_data_arrived<false>(3);

            // MMAC
            stage_id ^= 1;
            {
                constexpr int min_tile_k = 0;
                flash::raise_priority(1);
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int pv_tile_id = (STAGES == 2) ? k_loop - 1: k_loop;
                        int v_tile_id = stage_id * 2 + min_tile_k;
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                            v_reg[v_tile_id].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
                flash::lower_priority();
            }
            flash::wait_lds_data_arrived<false>(2);
            {
                constexpr int min_tile_k = 1;
                flash::raise_priority(1);
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int pv_tile_id = (STAGES == 2) ? k_loop - 1: k_loop;
                        int v_tile_id = stage_id * 2 + min_tile_k;
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                            p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                            v_reg[v_tile_id].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
                flash::lower_priority();
            }
        }

        stage_id ^= 1;
        flash::wait_lds_data_arrived<false>(1);
        // last mmac
        {
            constexpr int min_tile_k = 0;
            flash::raise_priority(1);
            #pragma unroll
            for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int pv_tile_id = (kHeadDimV / kBlockN) - 1;
                    int v_tile_id = stage_id * 2 + min_tile_k;
                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                        v_reg[v_tile_id].f16x4[min_tile_n],
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                }
            }
            flash::lower_priority();
        }
        flash::wait_lds_data_arrived<false>(0);
        {
            constexpr int min_tile_k = 1;
            flash::raise_priority(1);
            #pragma unroll
            for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    int pv_tile_id = (kHeadDimV / kBlockN) - 1;
                    int v_tile_id = stage_id * 2 + min_tile_k;
                    pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        p_reg[n_loop - 1][2 * min_tile_k + min_tile_m].f16x4,
                        v_reg[v_tile_id].f16x4[min_tile_n],
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                }
            }
            flash::lower_priority();
        }
    }
}
