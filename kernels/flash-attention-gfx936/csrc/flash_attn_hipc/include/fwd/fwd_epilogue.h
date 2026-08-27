#include "numeric_types.h"
#include "intrinsic.h"

__forceinline__ __device__ float fwd_attention_sink_load(
    const void *s_aux_ptr,
    int s_aux_type,
    int head_idx) {
    if (s_aux_type == 1) {
        return reinterpret_cast<const float *>(s_aux_ptr)[head_idx];
    } else if (s_aux_type == 2) {
        return UpCast<half_t, float>(
            reinterpret_cast<const half_t *>(s_aux_ptr)[head_idx]);
    } else {
        return UpCast<BFloat16, float>(
            reinterpret_cast<const BFloat16 *>(s_aux_ptr)[head_idx]);
    }
}

template<int WARP_M, int kBlockK, int kHeadDimV, typename ElementAccum>
__forceinline__ __device__ void fwd_apply_attention_sink(
    vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4],
    vec2_Accum<ElementAccum> scores_max[WARP_M / 32],
    vec2_Accum<ElementAccum> scores_sum[WARP_M / 32],
    const ElementAccum scale_softmax,
    const float sink_value) {
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
            const ElementAccum old_scaled_max = scores_max[mi].f32[min_tile_m] * scale_softmax;
            const ElementAccum new_scaled_max = max(old_scaled_max, ElementAccum(sink_value));
            const ElementAccum old_rescale = __expf(old_scaled_max - new_scaled_max);
            scores_sum[mi].f32[min_tile_m] = scores_sum[mi].f32[min_tile_m] * old_rescale + __expf(ElementAccum(sink_value) - new_scaled_max);
            scores_max[mi].f32[min_tile_m] = new_scaled_max / scale_softmax;

            __float2 old_rescale_pair = {old_rescale, old_rescale};
            #pragma unroll
            for (int ni = 0; ni < (kBlockK / 32); ++ni) {
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int mmac_id = min_tile_n * 2 + min_tile_m;
                    #pragma unroll
                    for (int pv_n_loop = 0; pv_n_loop < (kHeadDimV / kBlockK); ++pv_n_loop) {
                        const int pv_tile_id = pv_n_loop * (WARP_M / 32) * (kBlockK / 32) + ni * (WARP_M / 32) + mi;
                        #if defined(__gfx936__) || defined(__gfx938__)
                            #pragma unroll
                            for (int vec_id = 0; vec_id < 2; ++vec_id) {
                                acc_o[pv_tile_id][mmac_id].u64[vec_id] =
                                    __builtin_hcu_pk_mul_f32(acc_o[pv_tile_id][mmac_id].u64[vec_id], old_rescale_pair);
                            }
                        #else
                            #pragma unroll
                            for (int vec_id = 0; vec_id < 4; ++vec_id) {
                                acc_o[pv_tile_id][mmac_id].f32[vec_id] *= old_rescale;
                            }
                        #endif
                    }
                }
            }
        }
    }
}

template<int WARP_M, int kBlockK, int kHeadDimV, bool Is_dropout, typename ElementAccum, bool StoreLSE = true>
__forceinline__ __device__ void fwd_epilugue_rescale_acco(
    vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4],
    vec2_Accum<ElementAccum> lse[WARP_M / 32],
    vec2_Accum<ElementAccum> scores_max[WARP_M / 32],
    vec2_Accum<ElementAccum> scores_sum[WARP_M / 32],
    const ElementAccum scale_softmax,
    const ElementAccum rp_dropout) {
    // Epilogue
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
            ElementAccum sum = scores_sum[mi].f32[min_tile_m];
            ElementAccum inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            if constexpr (StoreLSE) {
                lse[mi].f32[min_tile_m] = (sum == 0.f || sum != sum) ? INFINITY : scores_max[mi].f32[min_tile_m] * scale_softmax + __logf(sum);
            }
            ElementAccum scale = Is_dropout ? inv_sum * rp_dropout: inv_sum;
            __float2 scale_pair = {scale, scale};
            #pragma unroll
            for (int ni = 0; ni < (kBlockK / 32); ++ni) {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int mmac_id = min_tile_n * 2 + min_tile_m;
                    #pragma unroll
                    for(int pv_n_loop = 0; pv_n_loop < (kHeadDimV / kBlockK); ++pv_n_loop) {
                        const int pv_tile_id = pv_n_loop * (WARP_M / 32) * (kBlockK / 32) + ni * (WARP_M / 32) + mi;
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



template<int WARP_M, bool Is_even_MN, bool SplitD, bool Is_Interleaved, typename ElementAccum>
__forceinline__ __device__ void fwd_epilogue_store_lse(
    vec2_Accum<ElementAccum> lse[WARP_M / 32],
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
        for (int mi = 0; mi < (WARP_M / 32); ++mi) {
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                const int row = Is_Interleaved
                    ? warp_id * WARP_M + mi * 32 + (lane_id & 15) + min_tile_m * 16
                    : warp_id * WARP_M + mi * 32 + (lane_id & 15) * 2 + min_tile_m;
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



template<int kHeadDimV, int kBlockM, int kBlockK, int WARP_M, bool Is_even_MN, bool Is_Interleaved, bool TcpSwizzle, typename Element, typename ElementAccum>
__forceinline__ __device__ void fwd_epilogue_store_output(
        Element *o_ptr,
        vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4],
        int m_block,
        int warp_id,
        int lane_id,
        int seqlen_o_stride,
        int seqlen_q_limit) {

    int pv_lane_seq_idx      = lane_id & 15;
    int pv_lane_head_dim_idx = lane_id >> 4;

    #if defined(__gfx938__) || defined(__gfx946__)
        constexpr bool Is_Interleaved_ = Is_Interleaved and kHeadDimV == 128;
    #else
        constexpr bool Is_Interleaved_ = Is_Interleaved;
    #endif

    // gfx938/946: hdim>=128 generic epilogue and non-even hdim<128 direct stores
    // have an incorrect lane-to-output mapping. Fall back to LDS interleave store.
    if constexpr (kHeadDimV >= 128 || (not Is_even_MN && kHeadDimV < 128)) {
    #if defined(__gfx938__) || defined(__gfx946__)
        #pragma unroll
        for(int k_loop = 0; k_loop < (kHeadDimV / kBlockK); ++k_loop) {
            #pragma unroll
            for(int warp_m_idx = 0; warp_m_idx < (WARP_M / 32); ++warp_m_idx) {
                #pragma unroll
                for(int k_tile_idx = 0; k_tile_idx < (kBlockK / 32); ++k_tile_idx) {
                    #pragma unroll 2
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int tile32x32_id = k_loop * (WARP_M / 32) * (kBlockK / 32) + warp_m_idx * (kBlockK / 32) + k_tile_idx;
                        int s_offset = k_loop * kBlockK;
                        int seqlen_q_offset = (warp_id * WARP_M + warp_m_idx * 32 + pv_lane_seq_idx * 2 + min_tile_m);
                        int v_offset = seqlen_q_offset * seqlen_o_stride + pv_lane_head_dim_idx * 8;
                        union_vec4_f16x2<Element> v_data;
                        #pragma unroll
                        for(int vec_index = 0; vec_index < 4; ++vec_index) {
                            constexpr bool is_bf16 = std::is_same<Element, bhalf_t>::value;
                            v_data.f16x2[vec_index][0] = DownCast<ElementAccum, Element, is_bf16>(acc_o[tile32x32_id][min_tile_m + 0 * 2].f32[vec_index]);
                            v_data.f16x2[vec_index][1] = DownCast<ElementAccum, Element, is_bf16>(acc_o[tile32x32_id][min_tile_m + 1 * 2].f32[vec_index]);
                        }
                        auto lds = (__attribute__((address_space(3))) float*)(0);
                        int lds_write_offset = (warp_id * 512 + pv_lane_seq_idx * 16 + pv_lane_head_dim_idx * 4 + pv_lane_seq_idx * 4) * 4;
                        __builtin_amdgcn_sched_barrier(0);
                        inlineasm_ds_write_b128(lds_write_offset, v_data.f32);
                        flash::wait_lds_data_arrived<false>(0);
                        #pragma unroll
                        for(int vec_index = 0; vec_index < 2; ++vec_index) {
                            int lds_load_offset = (warp_id * 512 + pv_lane_seq_idx * 16 + vec_index * 8 + pv_lane_head_dim_idx + pv_lane_seq_idx * 4) * 4;
                            asm volatile("ds_read2_b32 %0, %1 offset0:0 offset1:%2\n":: "v"(v_data.data[vec_index]), "v"(lds_load_offset), "B"(4));
                        }
                        flash::wait_lds_data_arrived<false>(0);
                        if(m_block * kBlockM + seqlen_q_offset < seqlen_q_limit) {
                            *(vec4_fp32*)(o_ptr + v_offset + s_offset + k_tile_idx * 32) = v_data.f32;
                        }
                    }
                }
            }
        }
        return;
    #endif
    }

    if constexpr (Is_Interleaved_) {
    #if defined(__gfx938__) || defined(__gfx946__)
        #pragma unroll
        for (int k_loop = 0; k_loop < (kHeadDimV / kBlockK); ++k_loop) {
            #pragma unroll
            for (int warp_m_idx = 0; warp_m_idx < (WARP_M / 32); ++warp_m_idx) {
                #pragma unroll
                for (int k_tile_idx = 0; k_tile_idx < (kBlockK / 32); ++k_tile_idx) {
                    #pragma unroll 2
                    for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        #pragma unroll 2
                        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            const int pv_tile_id = k_loop * (WARP_M / 32) * (kBlockK / 32) + warp_m_idx * (kBlockK / 32) + k_tile_idx;
                            const int mmac_id    = min_tile_m + min_tile_n * 2;
                            int seqlen_q_offset  = warp_id * WARP_M + warp_m_idx * 32 + min_tile_m * 16 + pv_lane_seq_idx;
                            // prepare for store
                            int s_offset = k_tile_idx * 32 + min_tile_n * 16;
                            int v_offset = seqlen_q_offset * seqlen_o_stride + k_loop * kBlockK + pv_lane_head_dim_idx * 4;
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
                        }
                    }
                }
            }
        } // brace, to control vgpr usage
    #else
        // simulate mmac-4interleave via lds
        // todo: lds bank conflicts, vgpr spills
        #pragma unroll
        for(int k_loop = 0; k_loop < (kHeadDimV / kBlockK); ++k_loop) {
            #pragma unroll
            for(int warp_m_idx = 0; warp_m_idx < (WARP_M / 32); ++warp_m_idx) {
                #pragma unroll
                for(int k_tile_idx = 0; k_tile_idx < (kBlockK / 32); ++k_tile_idx) {
                    #pragma unroll 2
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int tile32x32_id = k_loop * (WARP_M / 32) * (kBlockK / 32) + warp_m_idx * (kBlockK / 32) + k_tile_idx;
                        int s_offset = k_loop * kBlockK;
                        int seqlen_q_offset = (warp_id * WARP_M + warp_m_idx * 32 + pv_lane_seq_idx * 2 + min_tile_m);
                        int v_offset = seqlen_q_offset * seqlen_o_stride + pv_lane_head_dim_idx * 8;
                        // prepare vgprs
                        union_vec4_f16x2<Element> v_data;
                        #pragma unroll
                        for(int vec_index = 0; vec_index < 4; ++vec_index) {
                            // convert float -> bf16/fp16
                            constexpr bool is_bf16 = std::is_same<Element, bhalf_t>::value;
                            v_data.f16x2[vec_index][0] = DownCast<ElementAccum, Element, is_bf16>(acc_o[tile32x32_id][min_tile_m + 0 * 2].f32[vec_index]);
                            v_data.f16x2[vec_index][1] = DownCast<ElementAccum, Element, is_bf16>(acc_o[tile32x32_id][min_tile_m + 1 * 2].f32[vec_index]);
                        }

                        // try interleave
                        auto lds = (__attribute__((address_space(3))) float*)(0);
                        int lds_write_offset = (warp_id * 512 + pv_lane_seq_idx * 16 + pv_lane_head_dim_idx * 4 + pv_lane_seq_idx * 4) * 4;
                        __builtin_amdgcn_sched_barrier(0);
                        inlineasm_ds_write_b128(lds_write_offset, v_data.f32);
                        flash::wait_lds_data_arrived<false>(0);
                        #pragma unroll
                        for(int vec_index = 0; vec_index < 2; ++vec_index) {
                            int lds_load_offset = (warp_id * 512 + pv_lane_seq_idx * 16 + vec_index * 8 + pv_lane_head_dim_idx + pv_lane_seq_idx * 4) * 4;
                            asm volatile("ds_read2_b32 %0, %1 offset0:0 offset1:%2\n":: "v"(v_data.data[vec_index]), "v"(lds_load_offset), "B"(4));
                        }
                        flash::wait_lds_data_arrived<false>(0);

                        // write to global memory
                        if constexpr (Is_even_MN) {
                            *(vec4_fp32*)(o_ptr + v_offset + s_offset + k_tile_idx * 32) = v_data.f32;
                        } else {
                            if(m_block * kBlockM + seqlen_q_offset < seqlen_q_limit) {
                                *(vec4_fp32*)(o_ptr + v_offset + s_offset + k_tile_idx * 32) = v_data.f32;
                            }
                        }
                    }
                }
            }
        }
    #endif
    } else {
        auto o_resource = prepare_for_buffer_load<kHeadDimV, Element, TcpSwizzle>(o_ptr);
        #pragma unroll
        for(int k_loop = 0; k_loop < (kHeadDimV / kBlockK); ++k_loop) {
            #pragma unroll
            for(int warp_m_idx = 0; warp_m_idx < (WARP_M / 32); ++warp_m_idx) {
                #pragma unroll
                for(int k_tile_idx = 0; k_tile_idx < (kBlockK / 32); ++k_tile_idx) {
                    #pragma unroll 2
                    for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        #pragma unroll
                        for(int vec_index = 0; vec_index < 4; ++vec_index) {
                            int tile32x32_id = k_loop * (WARP_M / 32) * (kBlockK / 32) + warp_m_idx * (kBlockK / 32) + k_tile_idx;
                            int s_offset = k_loop * kBlockK;
                            int s_offset_constexpr = k_tile_idx * 32 + vec_index * 8; /*overflow for s_offset_constexpr*/
                            int seqlen_q_offset = (warp_id * WARP_M + warp_m_idx * 32 + pv_lane_seq_idx * 2 + min_tile_m);
                            int v_offset = seqlen_q_offset * seqlen_o_stride + pv_lane_head_dim_idx * 2;
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
                            if constexpr (Is_even_MN) {
                                inline_buffer_store_dword<vec2_Element<Element>, 1>(v_data, v_offset, o_resource, s_offset, /* immediate integer */s_offset_constexpr);
                            } else {
                                if(m_block * kBlockM + seqlen_q_offset < seqlen_q_limit) {
                                    *(vec2_Element<Element>*)(o_ptr + v_offset + s_offset + s_offset_constexpr) = v_data;
                                }
                            }
                        }
                    }
                }
            }
        } // brace, to control vgpr usage
        if constexpr (Is_even_MN) {
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
        }
    }

}

namespace flash {
template<int kHeadDimV, typename ElementAccum>
__forceinline__ __device__ void prefix_prefill_hdim512_16x64_apply_attention_sink(
        vec4_Accum<ElementAccum> acc_o[kHeadDimV / 16],
        ElementAccum &scores_max,
        ElementAccum &scores_sum,
        const ElementAccum scale_softmax,
        const float sink_value) {
    const ElementAccum old_scaled_max = scores_max * scale_softmax;
    const ElementAccum new_scaled_max = max(old_scaled_max, ElementAccum(sink_value));
    const ElementAccum old_rescale = __expf(old_scaled_max - new_scaled_max);
    scores_sum = scores_sum * old_rescale + __expf(ElementAccum(sink_value) - new_scaled_max);
    scores_max = new_scaled_max / scale_softmax;

    __float2 old_rescale_pair = {old_rescale, old_rescale};
    #pragma unroll
    for (int i = 0; i < kHeadDimV / 16; ++i) {
        acc_o[i].u64[0] = __builtin_hcu_pk_mul_f32(acc_o[i].u64[0], old_rescale_pair);
        acc_o[i].u64[1] = __builtin_hcu_pk_mul_f32(acc_o[i].u64[1], old_rescale_pair);
    }
}

template<int kHeadDimV, typename ElementAccum>
__forceinline__ __device__ ElementAccum prefix_prefill_hdim512_16x64_rescale_acc_o(
        vec4_Accum<ElementAccum> acc_o[kHeadDimV / 16],
        const ElementAccum scores_max,
        const ElementAccum scores_sum,
        const ElementAccum scale_softmax) {
    const ElementAccum inv_sum = (scores_sum == 0.f || scores_sum != scores_sum) ? 1.f : 1.f / scores_sum;
    __float2 inv_sum_pair = {inv_sum, inv_sum};
    #pragma unroll
    for (int i = 0; i < kHeadDimV / 16; ++i) {
        acc_o[i].u64[0] = __builtin_hcu_pk_mul_f32(acc_o[i].u64[0], inv_sum_pair);
        acc_o[i].u64[1] = __builtin_hcu_pk_mul_f32(acc_o[i].u64[1], inv_sum_pair);
    }
    return (scores_sum == 0.f || scores_sum != scores_sum) ? INFINITY : scores_max * scale_softmax + __logf(scores_sum);
}

template<int WARP_M, typename ElementAccum>
__forceinline__ __device__ void prefix_prefill_hdim512_16x64_store_lse(
        ElementAccum lse,
        void* softmax_lse_ptr,
        const int64_t row_offset_lse,
        const int warp_id,
        const int lane_id,
        const int seqlen_q_limit) {
    static_assert(WARP_M == 16);
    if (softmax_lse_ptr != nullptr && (lane_id >> 4) == 0) {
        const int row = warp_id * WARP_M + (lane_id & 15);
        if (row < seqlen_q_limit) {
            reinterpret_cast<ElementAccum*>(softmax_lse_ptr)[row_offset_lse + row] = lse;
        }
    }
}

template<int kHeadDimV, int WARP_M, typename Element, typename ElementAccum>
__forceinline__ __device__ void prefix_prefill_hdim512_16x64_store_output(
        vec4_Accum<ElementAccum> acc_o[kHeadDimV / 16],
        Element* o_ptr,
        const int warp_id,
        const int lane_id,
        const int o_row_stride,
        const int seqlen_q_limit) {
    static_assert(kHeadDimV == 512);
    static_assert(WARP_M == 16);
    const int row = warp_id * WARP_M + (lane_id & 15);
    if (row < seqlen_q_limit) {
        #pragma unroll
        for (int d_loop = 0; d_loop < kHeadDimV / 32; ++d_loop) {
            #pragma unroll
            for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                vec2_Element<Element> data = DownCastPairNoPack<ElementAccum, Element>(
                    acc_o[d_loop * 2 + 0].f32[vec_idx],
                    acc_o[d_loop * 2 + 1].f32[vec_idx]);
                const int col = d_loop * 32 + vec_idx * 8 + (lane_id >> 4) * 2;
                *reinterpret_cast<vec2_Element<Element>*>(o_ptr + row * int64_t(o_row_stride) + col) = data;
            }
        }
    }
}

} // namespace flash
