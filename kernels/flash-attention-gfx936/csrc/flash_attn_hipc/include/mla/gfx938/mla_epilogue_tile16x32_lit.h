#include "numeric_types.h"
#include "intrinsic.h"


// DataType: {vec2_Accum<ElementAccum>, vec_Accum<ElementAccum>}
template<int WARP_M, int kBlockK, int kHeadDimV, bool Is_dropout, typename ElementAccum, typename DataType=union_vec2_fp32/* vec2_Accum<ElementAccum> */, int M_MMAC_COUNT=2>
__forceinline__ __device__ void prefill_mla_epilugue_rescale_acco(
    vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / (16 * M_MMAC_COUNT)) * (kBlockK / 32)][2 * M_MMAC_COUNT],
    DataType lse[WARP_M / (16 * M_MMAC_COUNT)],
    DataType scores_max[WARP_M / (16 * M_MMAC_COUNT)],
    DataType scores_sum[WARP_M / (16 * M_MMAC_COUNT)],
    const ElementAccum scale_softmax,
    const ElementAccum rp_dropout) {
    // Epilogue
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / (16 * M_MMAC_COUNT)); ++mi) {
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            ElementAccum sum = scores_sum[mi].f32[min_tile_m];
            ElementAccum inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            lse[mi].f32[min_tile_m] = (sum == 0.f || sum != sum) ? INFINITY : scores_max[mi].f32[min_tile_m] * scale_softmax + __logf(sum);
            ElementAccum scale = Is_dropout ? inv_sum * rp_dropout: inv_sum;
            __float2 scale_pair = {scale, scale};
            #pragma unroll
            for (int ni = 0; ni < (kBlockK / 32); ++ni) {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    int mmac_id;
                    if constexpr (M_MMAC_COUNT == 2) {
                        mmac_id = min_tile_n * 2 + min_tile_m;
                    } else {
                        mmac_id = min_tile_n;
                    }
                    #pragma unroll
                    for(int pv_n_loop = 0; pv_n_loop < (kHeadDimV / kBlockK); ++pv_n_loop) {
                        const int pv_tile_id = pv_n_loop * (WARP_M / (16 * M_MMAC_COUNT)) * (kBlockK / 32) + ni * (WARP_M / (16 * M_MMAC_COUNT)) + mi;
                        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                            for(int vec_id = 0; vec_id < 2; ++vec_id) {
                                acc_o[pv_tile_id][mmac_id].u64[vec_id] = __builtin_hcu_pk_mul_f32(
                                    acc_o[pv_tile_id][mmac_id].u64[vec_id],
                                    scale_pair
                                );
                            }
                        #else
                            for(int vec_id = 0; vec_id < 4; ++vec_id) {
                                acc_o[pv_tile_id][mmac_id].f32[vec_id] *= scale;
                            }
                        #endif
                    }
                }
            }
        }
    }
}



template<int WARP_M, bool Is_even_MN, bool SplitD, bool Is_Interleaved, typename ElementAccum, typename DataType=union_vec2_fp32/* vec2_Accum<ElementAccum> */, int M_MMAC_COUNT=2>
__forceinline__ __device__ void prefill_mla_epilogue_store_lse(
    DataType lse[WARP_M / (16 * M_MMAC_COUNT)],
    void *softmax_lse_ptr,
    int row_offset_lse,
    int warp_id,
    int lane_id,
    int headdim_split_id,
    int seqlen_q_limit) {

    ElementAccum * gLSE = reinterpret_cast<ElementAccum*>(softmax_lse_ptr) + row_offset_lse;
    #if (DEBUG_LEVEL >= 1)
        ElementAccum * scores_sum_ptr = reinterpret_cast<ElementAccum*>(scores_sum_ptr) + row_offset_lse;
        ElementAccum * scores_max_ptr = reinterpret_cast<ElementAccum*>(scores_max_ptr) + row_offset_lse;
    #endif
    const bool write_lse = SplitD > 1 ? (lane_id >> 4) == 0 and headdim_split_id == 0: (lane_id >> 4) == 0;
    if (write_lse) {
        #pragma unroll
        for (int mi = 0; mi < (WARP_M / (16 * M_MMAC_COUNT)); ++mi) {
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                const int row = Is_Interleaved
                    ? warp_id * WARP_M + mi * (16 * M_MMAC_COUNT) + (lane_id & 15) + min_tile_m * 16
                    : warp_id * WARP_M + mi * (16 * M_MMAC_COUNT) + (lane_id & 15) * 2 + min_tile_m;
                if constexpr (Is_even_MN) {
                    gLSE[row] = lse[mi].f32[min_tile_m];
                    #if (DEBUG_LEVEL >= 1)
                        scores_sum_ptr[row] = scores_sum[mi].f32[min_tile_m];
                        scores_max_ptr[row] = scores_max[mi].f32[min_tile_m];
                    #endif
                } else {
                    if (row < seqlen_q_limit) {
                        gLSE[row] = lse[mi].f32[min_tile_m];
                        #if (DEBUG_LEVEL >= 1)
                            scores_sum_ptr[row] = scores_sum[mi].f32[min_tile_m];
                            scores_max_ptr[row] = scores_max[mi].f32[min_tile_m];
                        #endif
                    }
                }
            }
        }
    }
}



template<int kHeadDimV, int kBlockM, int kBlockK, int WARP_M, bool Is_even_MN, bool Is_Interleaved, bool TcpSwizzle, typename Element, typename ElementAccum, int M_MMAC_COUNT=2>
__forceinline__ __device__ void prefill_mla_epilogue_store_output(
        Element *o_ptr,
        vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / (16 * M_MMAC_COUNT)) * (kBlockK / 32)][2 * M_MMAC_COUNT],
        int m_block,
        int warp_id,
        int lane_id,
        int seqlen_o_stride,
        int seqlen_q_limit) {

    int pv_lane_seq_idx      = lane_id & 15;
    int pv_lane_head_dim_idx = lane_id >> 4;

    if constexpr (Is_Interleaved) {
        #if defined(__gfx92a__) && defined(YY_USE_MPERMUTE)
            union_vec2_f16x2<Element> acc_o_fp16[(kHeadDimV / kBlockK) * (WARP_M / (16 * M_MMAC_COUNT)) * (kBlockK / 32)][2 * M_MMAC_COUNT];
            #pragma unroll
            for (int k_loop = 0; k_loop < (kHeadDimV / kBlockK); ++k_loop) {
                #pragma unroll 2
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    #pragma unroll 2
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        int mmac_id;
                        if constexpr (M_MMAC_COUNT == 2)
                            mmac_id    = min_tile_m + min_tile_n * 2;
                        else
                            mmac_id    = min_tile_n;
                        #pragma unroll
                        for (int vec_index = 0; vec_index < 2; ++vec_index) {
                            // convert float -> bf16/fp16
                            acc_o_fp16[k_loop][mmac_id].f16x2[vec_index] = DownCastPair<ElementAccum, Element>(acc_o[k_loop][mmac_id].f32x2[vec_index]);
                        }
                        ds_mpermute_kdim_for_mmac(acc_o_fp16[k_loop][mmac_id].f32);
                    }
                }
            }
        #endif

        #pragma unroll
        for (int k_loop = 0; k_loop < (kHeadDimV / kBlockK); ++k_loop) {
            #if defined(__gfx92a__) && defined(YY_USE_MPERMUTE)
                flash::wait_lds_data_arrived<false>((kHeadDimV / kBlockK - k_loop - 1) * 2 * 2);
            #endif
            #pragma unroll
            for (int warp_m_idx = 0; warp_m_idx < (WARP_M / (16 * M_MMAC_COUNT)); ++warp_m_idx) {
                #pragma unroll
                for (int k_tile_idx = 0; k_tile_idx < (kBlockK / 32); ++k_tile_idx) {
                    #pragma unroll 2
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        #pragma unroll 2
                        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            const int pv_tile_id = k_loop * (WARP_M / (16 * M_MMAC_COUNT)) * (kBlockK / 32) + warp_m_idx * (kBlockK / 32) + k_tile_idx;
                            int mmac_id;                            
                            if constexpr (M_MMAC_COUNT == 2) {
                                mmac_id    = min_tile_m + min_tile_n * 2;
                            } else {
                                mmac_id    = min_tile_n;
                            }
                            
                            int seqlen_q_offset  = warp_id * WARP_M + warp_m_idx * (16 * M_MMAC_COUNT) + min_tile_m * 16 + pv_lane_seq_idx;
                            // prepare for store
                            int s_offset = k_tile_idx * 32 + min_tile_n * 16;
                            int v_offset = seqlen_q_offset * seqlen_o_stride + k_loop * kBlockK + pv_lane_head_dim_idx * 4;
                            #if defined(__gfx92a__) && defined(YY_USE_MPERMUTE)
                                if constexpr (not Is_even_MN) {
                                    if (m_block * kBlockM + seqlen_q_offset < seqlen_q_limit) {
                                        *(union_vec2_f16x2<Element>*)(o_ptr + v_offset + s_offset) = acc_o_fp16[k_loop][mmac_id];
                                    }
                                } else {
                                    *(union_vec2_f16x2<Element>*)(o_ptr + v_offset + s_offset) = acc_o_fp16[k_loop][mmac_id];
                                }
                            #else
                                union_vec2_f16x2<Element> v_data;
                                #pragma unroll
                                for (int vec_index = 0; vec_index < 2; ++vec_index) {
                                    // convert float -> bf16/fp16
                                    v_data.f16x2[vec_index] = DownCastPair<ElementAccum, Element>(acc_o[pv_tile_id][mmac_id].f32x2[vec_index]);
                                }
                                if constexpr (not Is_even_MN) {
                                    if (m_block * kBlockM + seqlen_q_offset < seqlen_q_limit) {
                                        *(union_vec2_f16x2<Element>*)(o_ptr + v_offset + s_offset) = v_data;
                                    }
                                } else {
                                    *(union_vec2_f16x2<Element>*)(o_ptr + v_offset + s_offset) = v_data;
                                }
                            #endif
                        }
                    }
                }
            }
        } // brace, to control vgpr usage
    } else {    // 仅支持LIT的部分
        auto gO = prepare_for_buffer_load<kHeadDimV, Element>(o_ptr);
        #pragma unroll
        for(int k_loop = 0; k_loop < (kHeadDimV / kBlockK); ++k_loop) {
            #pragma unroll
            for(int warp_m_idx = 0; warp_m_idx < (WARP_M / (16 * M_MMAC_COUNT)); ++warp_m_idx) {
                #pragma unroll
                for(int k_tile_idx = 0; k_tile_idx < (kBlockK / 32); ++k_tile_idx) {
                    #pragma unroll 2
                    for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        #pragma unroll
                        for(int vec_index = 0; vec_index < 4; ++vec_index) {
                            if constexpr (not Is_even_MN) {
                                #pragma unroll
                                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                                    const int seqlen_q_offset = warp_id * WARP_M + warp_m_idx * (16 * M_MMAC_COUNT) + pv_lane_seq_idx + min_tile_m * 16; /*算的是 1 个 kBlockM 内在 seqlen_q 方向上的位置*/
                                    int pv_global_addr = seqlen_q_offset * seqlen_o_stride + /*headdim 方向上的偏移*/k_loop * kBlockK + k_tile_idx * 32 + vec_index * 8 + pv_lane_head_dim_idx * 2 + min_tile_n;
                                    if(m_block * kBlockM + seqlen_q_offset < seqlen_q_limit) {
                                        if constexpr (M_MMAC_COUNT == 2)
                                            o_ptr[pv_global_addr] = DownCast<ElementAccum, Element>(acc_o[k_loop * (WARP_M / (16 * M_MMAC_COUNT)) * (kBlockK / 32) + warp_m_idx * (kBlockK / 32) + k_tile_idx][min_tile_m + min_tile_n * 2].f32[vec_index]);
                                        else
                                            o_ptr[pv_global_addr] = DownCast<ElementAccum, Element>(acc_o[k_loop * (WARP_M / (16 * M_MMAC_COUNT)) * (kBlockK / 32) + warp_m_idx * (kBlockK / 32) + k_tile_idx][min_tile_n].f32[vec_index]);
                                    }
                                }
                            }
                            else {
                                int tile32x32_id = k_loop * (WARP_M / (16 * M_MMAC_COUNT)) * (kBlockK / 32) + warp_m_idx * (kBlockK / 32) + k_tile_idx;
                                int s_offset = k_loop * kBlockK;
                                int s_offset_constexpr = k_tile_idx * 32 + vec_index * 8; /*overflow for s_offset_constexpr*/
                                int v_offset = (warp_id * WARP_M + warp_m_idx * (16 * M_MMAC_COUNT) + pv_lane_seq_idx + min_tile_m * 16) * seqlen_o_stride + pv_lane_head_dim_idx * 2;
                                vec2_Element<Element> v_data;
                                // convert float -> bf16/fp16
                                if constexpr (std::is_same<Element, bhalf_t>::value) {
                                #if 1
                                    v_data[0] = DownCast<ElementAccum, Element, true>(acc_o[tile32x32_id][min_tile_m + 0 * 2].f32[vec_index]);
                                    v_data[1] = DownCast<ElementAccum, Element, true>(acc_o[tile32x32_id][min_tile_m + 1 * 2].f32[vec_index]);
                                #else
                                    v_data[0] = inlineasm_float2bfloat16_ushort_nonan(acc_o[tile32x32_id][min_tile_m + 0 * 2].f32[vec_index]);
                                    v_data[1] = inlineasm_float2bfloat16_ushort_nonan(acc_o[tile32x32_id][min_tile_m + 1 * 2].f32[vec_index]);
                                #endif
                                }
                                else if constexpr (std::is_same<Element, half_t>::value) {
                                #ifdef USE_CVT_PKRTZ_FP16_FP32
                                        *(vec2_Element<Element>*)&v_data = DownCastPair<ElementAccum, Element>(
                                        acc_o[tile32x32_id][min_tile_m + 0 * 2].f32[vec_index],
                                        acc_o[tile32x32_id][min_tile_m + 1 * 2].f32[vec_index]
                                    );
                                #else
                                    v_data[0] = DownCast<ElementAccum, Element>(acc_o[tile32x32_id][min_tile_m + 0 * 2].f32[vec_index]);
                                    v_data[1] = DownCast<ElementAccum, Element>(acc_o[tile32x32_id][min_tile_m + 1 * 2].f32[vec_index]);
                                #endif
                                }
                                // write to global memory
                                inline_buffer_store_dword<vec2_Element<Element>, 1>(v_data, v_offset, gO, s_offset, /* immediate integer */s_offset_constexpr);
                            }
                        }
                    }
                }
            }
        } // brace, to control vgpr usage
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
    }

}