#include "numeric_types.h"
#include "intrinsic.h"

__forceinline__ __device__ float fp8_attention_sink_load(const void *s_aux_ptr, int s_aux_type, int head_idx) {
    if (s_aux_type == 1) {
        return reinterpret_cast<const float *>(s_aux_ptr)[head_idx];
    } else if (s_aux_type == 2) {
        return UpCast<half_t, float>(reinterpret_cast<const half_t *>(s_aux_ptr)[head_idx]);
    } else {
        return UpCast<BFloat16, float>(reinterpret_cast<const BFloat16 *>(s_aux_ptr)[head_idx]);
    }
}

template<int kHeadDim, int WARP_M, int WARP_N, typename ElementAccum>
__forceinline__ __device__ void fp8_attention_sink_apply(
    vec4_Accum<ElementAccum> acc_o[kHeadDim / 32][WARP_M / 16][WARP_N / 16],
    ElementAccum scores_max[WARP_M / 16],
    ElementAccum scores_sum[WARP_M / 16],
    ElementAccum softmax_scale,
    float sink_value
) {
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        const ElementAccum old_scaled_max = scores_max[m_idx] * softmax_scale;
        const ElementAccum new_scaled_max = max(old_scaled_max, ElementAccum(sink_value));
        const ElementAccum old_rescale = __expf(old_scaled_max - new_scaled_max);
        scores_sum[m_idx] = scores_sum[m_idx] * old_rescale + __expf(ElementAccum(sink_value) - new_scaled_max);
        scores_max[m_idx] = new_scaled_max / softmax_scale;

        __float2 old_rescale_pair;
        old_rescale_pair[0] = old_rescale;
        old_rescale_pair[1] = old_rescale;
        #pragma unroll
        for (int k_loop = 0; k_loop < kHeadDim / 32; ++k_loop) {
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                acc_o[k_loop][m_idx][n_idx].u64[0] = __builtin_hcu_pk_mul_f32(acc_o[k_loop][m_idx][n_idx].u64[0], old_rescale_pair);
                acc_o[k_loop][m_idx][n_idx].u64[1] = __builtin_hcu_pk_mul_f32(acc_o[k_loop][m_idx][n_idx].u64[1], old_rescale_pair);
            }
        }
    }
}

template<bool AssumeValidRows, int kHeadDim, int WARP_M, int WARP_N, bool StoreLSE, typename ElementAccum>
__forceinline__ __device__ void fp8_epilogue_rescale_acc_o(
    vec4_Accum<ElementAccum> acc_o[kHeadDim / 32][WARP_M / 16][WARP_N / 16],
    ElementAccum scores_max[WARP_M / 16],
    ElementAccum scores_sum[WARP_M / 16],
    ElementAccum lse[WARP_M / 16],
    ElementAccum softmax_scale,
    ElementAccum v_descale
) {
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        ElementAccum sum = scores_sum[m_idx];
        if constexpr (StoreLSE) {
            lse[m_idx] = (sum == 0.f || sum != sum) ? INFINITY : __llvm_fma_f32(scores_max[m_idx], softmax_scale, __logf(sum));
        }
        ElementAccum total_rescale;
        if constexpr (AssumeValidRows) {
            total_rescale = v_descale / sum;
        } else {
            total_rescale = (sum == 0.f || sum != sum) ? 0.f : v_descale / sum;
        }
        __float2 total_scale_pair;
        total_scale_pair[0] = total_rescale;
        total_scale_pair[1] = total_rescale;
        // __float2 inv_sum_pair;
        // inv_sum_pair[0] = 1.0f / sum;
        // inv_sum_pair[1] = inv_sum_pair[0];
        #pragma unroll
        for (int k_loop = 0; k_loop < kHeadDim / 32; ++k_loop) {
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                acc_o[k_loop][m_idx][n_idx].u64[0] = __builtin_hcu_pk_mul_f32(acc_o[k_loop][m_idx][n_idx].u64[0], total_scale_pair);
                acc_o[k_loop][m_idx][n_idx].u64[1] = __builtin_hcu_pk_mul_f32(acc_o[k_loop][m_idx][n_idx].u64[1], total_scale_pair);
            }
        }
    }
}




template<bool Is_even_MN, int WARP_M, typename ElementAccum>
__forceinline__ __device__ void fp8_epilogue_store_lse(
    // ElementAccum* scores_max_ptr,
    // ElementAccum* scores_sum_ptr,
    ElementAccum* softmax_lse_ptr,
    ElementAccum scores_max[WARP_M / 16],
    ElementAccum scores_sum[WARP_M / 16],
    ElementAccum lse[WARP_M / 16],
    int row_offset_lse, /*(bidb * h + bidh) * actual_seqlen_q*/
    int actual_seqlen_q,
    int wave_row_offset, /*m_block * kBlockM + warp_id * WARP_M*/
    int lane_id
) {
    if (lane_id < 16) {
        #pragma unroll
        for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
            int lse_row_id = row_offset_lse + wave_row_offset + ((lane_id & 15) >> 2) * 8 + m_idx * 4 + (lane_id & 3);;
            // scores_max_ptr[lse_row_id] = scores_max[m_idx];
            // scores_sum_ptr[lse_row_id] = scores_sum[m_idx];
            if (lse_row_id-row_offset_lse < actual_seqlen_q){
                softmax_lse_ptr[lse_row_id] = lse[m_idx];
            }
        }
    }
}



template<bool Is_even_MN, int kBlockM, int kHeadDim, int WARP_M, int WARP_N, typename Element, typename ElementAccum>
__forceinline__ __device__ void fp8_epilogue_store_output(
    Element* acc_o_ptr,
    vec4_Accum<ElementAccum> acc_o[kHeadDim / 32][WARP_M / 16][WARP_N / 16],
    int m_block,
    int warp_id,
    int lane_id,
    int o_row_stride,
    int actual_seqlen_q
) {
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        #pragma unroll
        for (int pv_loop = 0; pv_loop < kHeadDim / 32; ++pv_loop) {
            #pragma unroll
            for (int mmac_id = 0; mmac_id < 2; ++mmac_id) {
                int row_idx = warp_id * WARP_M + ((lane_id & 15) >> 2) * 8 + m_idx * 4 + (lane_id & 3);
                int col_idx = pv_loop * 32 + mmac_id * 16 + (lane_id >> 4) * 4;
                int offset = row_idx * o_row_stride + col_idx;
                union_vec2_f16x2<Element> v_data;
                #pragma unroll
                for (int vec_index = 0; vec_index < 2; ++vec_index) {
                    v_data.f16x2[vec_index] = DownCastPair<ElementAccum, Element>(acc_o[pv_loop][m_idx][mmac_id].f32x2[vec_index]);
                    // v_data = __builtin_hcu_cvt_pk_f16_f32(acc_o[pv_loop][m_idx][mmac_id].f32[vec_index*2], acc_o[pv_loop][m_idx][mmac_id].f32[vec_index*2+1], false/*clamp*/, 0/*omod*/);
                    // v_data[0] = DownCast<ElementAccum, Float16>(acc_o[pv_loop][m_idx][mmac_id].f32[vec_index*2]);
                    // v_data[1] = DownCast<ElementAccum, Float16>(acc_o[pv_loop][m_idx][mmac_id].f32[vec_index*2+1]);
                }
                if constexpr (Is_even_MN) {
                    *(union_vec2_f16x2<Element>*)(acc_o_ptr + offset) = v_data;
                } else if (m_block * kBlockM + row_idx < actual_seqlen_q) {
                    *(union_vec2_f16x2<Element>*)(acc_o_ptr + offset) = v_data;
                }
                // *(vec4_fp32*)(acc_o_ptr + offset) = acc_o[pv_loop][m_idx][mmac_id].f32;
            }
        }
    }
}
