#pragma once
#include "numeric_types.h"


template<typename Params, int kHeadDimV, int kHeadDimVSplit, bool Interleave2, bool Split, typename SplitkvAccumType, typename ElementAccum, int kBlockM, int kBlockK, int WARP_NUM, int K_LOOP_COUNT, int M_WARP_COUNT, int K_WARP_COUNT, int M_MMAC_COUNT>
__forceinline__ __device__ void mla_tp8_epilogue_store_output_gfx938(
        vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4],
        Params params,
        int bidb,
        int bidh,
        int m_block,
        int split_id,
        int headdim_split_id,
        int warp_id,
        int lane_id) {
    int o_row_stride = params.o_row_stride;
    const int64_t row_offset_o = bidb * int64_t(params.o_batch_stride) + bidh * params.o_head_stride + headdim_split_id * kHeadDimVSplit;
    SplitkvAccumType* o_ptr  = Split
                             ? reinterpret_cast<SplitkvAccumType *>(params.oaccum_ptr) + row_offset_o + /*which split*/ split_id * params.b * params.o_batch_stride
                             : reinterpret_cast<SplitkvAccumType *>(params.o_ptr) + row_offset_o;
    int pv_lane_seq_idx      = lane_id & 15;
    int pv_lane_head_dim_idx = lane_id >> 4;
    #pragma unroll
    for (int k_loop = 0; k_loop < K_LOOP_COUNT; k_loop += 4) {
        #pragma unroll
        for (int warp_m_idx = 0; warp_m_idx < M_WARP_COUNT; ++warp_m_idx) {
            #pragma unroll
            for (int k_tile_idx = 0; k_tile_idx < K_WARP_COUNT; ++k_tile_idx) {
                // which 32x32 tile
                int tile_32x32_id = k_loop * M_WARP_COUNT * K_WARP_COUNT + warp_m_idx * K_WARP_COUNT + k_tile_idx;
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    // index along seqlen_q dimension
                    int seqlen_q_idx = m_block * kBlockM + warp_m_idx * 32 + pv_lane_seq_idx + min_tile_m * 16;
                    if (seqlen_q_idx < params.seqlen_q) {
                        if constexpr (Interleave2) {
                            /*contiguous 64 bytes storation*/
                            union_vec4_f16x2<SplitkvAccumType> v_data;
                            v_data.f16x2[0 + 0 * 2] = DownCastPairNoPack<ElementAccum, SplitkvAccumType>(acc_o[tile_32x32_id][min_tile_m + 0 * 2].f32[0], acc_o[tile_32x32_id][min_tile_m + 1 * 2].f32[0]);
                            v_data.f16x2[1 + 0 * 2] = DownCastPairNoPack<ElementAccum, SplitkvAccumType>(acc_o[tile_32x32_id][min_tile_m + 0 * 2].f32[1], acc_o[tile_32x32_id][min_tile_m + 1 * 2].f32[1]);
                            v_data.f16x2[0 + 1 * 2] = DownCastPairNoPack<ElementAccum, SplitkvAccumType>(acc_o[tile_32x32_id][min_tile_m + 0 * 2].f32[2], acc_o[tile_32x32_id][min_tile_m + 1 * 2].f32[2]);
                            v_data.f16x2[1 + 1 * 2] = DownCastPairNoPack<ElementAccum, SplitkvAccumType>(acc_o[tile_32x32_id][min_tile_m + 0 * 2].f32[3], acc_o[tile_32x32_id][min_tile_m + 1 * 2].f32[3]);
                            int pv_global_addr = seqlen_q_idx * o_row_stride + (k_loop + warp_id) * kBlockK + k_tile_idx * 32 + pv_lane_head_dim_idx * 8;
                            *(vec4_fp32*)(o_ptr + pv_global_addr) = v_data.f32;
                        } else {
                            #pragma unroll
                            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                                union_vec2_f16x2<SplitkvAccumType> data;
                                int mmac_id = min_tile_m + min_tile_n * 2;
                                #pragma unroll
                                for (int vec_index = 0; vec_index < 2; ++vec_index) {
                                    data.f16x2[vec_index] = DownCastPair<ElementAccum, SplitkvAccumType>(acc_o[tile_32x32_id][mmac_id].f32x2[vec_index]);
                                }
                                int pv_global_addr = seqlen_q_idx * o_row_stride + (k_loop + warp_id) * kBlockK + k_tile_idx * 32 + pv_lane_head_dim_idx * 4 + min_tile_n * 16;
                                *(union_vec2_f16x2<SplitkvAccumType>*)(o_ptr + pv_global_addr) = data;
                            }
                        }
                    }
                }
            }
        }
    }
}