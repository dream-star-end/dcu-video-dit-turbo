#include "numeric_types.h"
#include "intrinsic.h"



template<int kHeadDimV, int kBlockM, int kBlockK, int WARP_M, int TailTile16, bool Is_even_MN, bool Is_Interleaved, bool TcpSwizzle, typename Element, typename ElementAccum>
__forceinline__ __device__ void fwd_epilogue_store_output_gfx938(
        Element *o_ptr,
        vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4],
        int m_block,
        int warp_id,
        int lane_id,
        int seqlen_o_stride,
        int seqlen_q_limit) {

    static_assert (Is_Interleaved and "For fwd_epilogue_store_output_gfx938, mmac must be 4interleave");

    int pv_lane_seq_idx      = lane_id & 15;
    int pv_lane_head_dim_idx = lane_id >> 4;

    if constexpr (TailTile16 == 2) {
        #pragma unroll
        for (int k_loop = 0; k_loop < (kHeadDimV / kBlockK); ++k_loop) {
            #pragma unroll
            for (int warp_m_idx = 0; warp_m_idx < (WARP_M / 32); ++warp_m_idx) {
                #pragma unroll
                for (int k_tile_idx = 0; k_tile_idx < (kBlockK / 32); ++k_tile_idx) {
                    #pragma unroll 2
                    for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        int seqlen_q_offset  = warp_id * WARP_M + warp_m_idx * 32 + min_tile_m * 16 + pv_lane_seq_idx;
                        const int pv_tile_id = k_loop * (WARP_M / 32) * (kBlockK / 32) + warp_m_idx * (kBlockK / 32) + k_tile_idx;
                        union_vec4_f16x2<Element> v_data;
                        v_data.f16x2[0 + 0 * 2] = DownCastPairNoPack<ElementAccum, Element>(acc_o[pv_tile_id][min_tile_m + 0 * 2].f32[0], acc_o[pv_tile_id][min_tile_m + 1 * 2].f32[0]);
                        v_data.f16x2[1 + 0 * 2] = DownCastPairNoPack<ElementAccum, Element>(acc_o[pv_tile_id][min_tile_m + 0 * 2].f32[1], acc_o[pv_tile_id][min_tile_m + 1 * 2].f32[1]);
                        v_data.f16x2[0 + 1 * 2] = DownCastPairNoPack<ElementAccum, Element>(acc_o[pv_tile_id][min_tile_m + 0 * 2].f32[2], acc_o[pv_tile_id][min_tile_m + 1 * 2].f32[2]);
                        v_data.f16x2[1 + 1 * 2] = DownCastPairNoPack<ElementAccum, Element>(acc_o[pv_tile_id][min_tile_m + 0 * 2].f32[3], acc_o[pv_tile_id][min_tile_m + 1 * 2].f32[3]);
                        int s_offset = k_tile_idx * 32;
                        int v_offset = seqlen_q_offset * seqlen_o_stride + k_loop * kBlockK + pv_lane_head_dim_idx * 8;
                        if constexpr (not Is_even_MN) {
                            if (m_block * kBlockM + seqlen_q_offset < seqlen_q_limit) {
                                *(vec4_fp32*)(o_ptr + v_offset + s_offset) = v_data.f32;
                            }
                        } else {
                            *(vec4_fp32*)(o_ptr + v_offset + s_offset) = v_data.f32;
                        }
                    }
                }
            }
        } // brace, to control vgpr usage
    } else {
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
    }
    __builtin_amdgcn_sched_barrier(0);
}