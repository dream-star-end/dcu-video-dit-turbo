#include "numeric_types.h"
#include "intrinsic.h"


template<int kHeadDimV, int kBlockM, int kBlockK, int WARP_M, bool Is_even_MN, bool Is_Interleaved, bool TcpSwizzle, typename Element, typename ElementAccum>
__forceinline__ __device__ void fwd_epilogue_store_output_mls_gfx92a(
        Element *o_ptr,
        vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4],
        int m_block,
        int warp_id,
        int lane_id,
        int seqlen_o_stride,
        int seqlen_q_limit) {

    int pv_lane_seq_idx      = lane_id & 15;
    int pv_lane_head_dim_idx = lane_id >> 4;

    // MLS gfx92a PV accumulators are laid out as 4-interleaved rows. Keep
    // this store path private to the MLS gfx92a kernels so the generic fwd
    // epilogue can continue to serve the legacy FA_FWD_NO_MLS path unchanged.
    if constexpr (false) {
    } else {
        auto gO = prepare_for_buffer_load<kHeadDimV, Element, TcpSwizzle>(o_ptr);
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
                            int s_offset_constexpr = k_tile_idx * 32 + vec_index * 8;
                            int seqlen_q_offset = warp_id * WARP_M + warp_m_idx * 32 + pv_lane_seq_idx + min_tile_m * 16;
                            int v_offset = seqlen_q_offset * seqlen_o_stride + pv_lane_head_dim_idx * 2;
                            vec2_Element<Element> v_data;
                            if constexpr (std::is_same<Element, bhalf_t>::value) {
                                *(vec2_Element<Element>*)&v_data = DownCastPairNoPack<ElementAccum, Element>(
                                    acc_o[tile32x32_id][min_tile_m + 0 * 2].f32[vec_index],
                                    acc_o[tile32x32_id][min_tile_m + 1 * 2].f32[vec_index]
                                );
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
                            if constexpr (Is_even_MN) {
                                inline_buffer_store_dword<vec2_Element<Element>, 1>(v_data, v_offset, gO, s_offset, s_offset_constexpr);
                            } else {
                                if (m_block * kBlockM + seqlen_q_offset < seqlen_q_limit) {
                                    inline_buffer_store_dword<vec2_Element<Element>, 1>(v_data, v_offset, gO, s_offset, s_offset_constexpr);
                                }
                            }
                        }
                    }
                }
            }
        }
        flash::wait_buffer_data_arrived<true>(0);
    }
}


template<int kHeadDimV, int kBlockM, int kBlockK, int WARP_M, bool Is_even_MN, bool Is_Interleaved, bool TcpSwizzle, typename Element, typename ElementAccum>
__forceinline__ __device__ void fwd_epilogue_store_output_gfx92a(
        Element *o_ptr,
        vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4],
        int m_block,
        int warp_id,
        int lane_id,
        int seqlen_o_stride,
        int seqlen_q_limit) {
    fwd_epilogue_store_output_mls_gfx92a<kHeadDimV, kBlockM, kBlockK, WARP_M, Is_even_MN, Is_Interleaved, TcpSwizzle, Element, ElementAccum>(
        o_ptr, acc_o, m_block, warp_id, lane_id, seqlen_o_stride, seqlen_q_limit);
}
