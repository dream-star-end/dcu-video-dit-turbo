#pragma once

#include "fwd/utils.h"

using namespace flash;


template <typename DataType, int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT>
inline __device__ void mla_apply_mask_tile16x32(DataType tensor[M_WARP_COUNT * N_WARP_COUNT][4], const int max_seqlen_k,
                                  const int col_idx_offset_ = 0) {
    const int lane_id = threadIdx.x & 63; // lane id, 0-63
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4);
    #pragma unroll
    for (int ni = 0; ni < N_WARP_COUNT; ++ni)  {
        #pragma unroll
        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n * 16;
            #pragma unroll
            for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                const int col_idx = col_idx_base + vec_idx * 4;
                if (col_idx >= max_seqlen_k) {
                    #pragma unroll
                    for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
                        #pragma unroll
                        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                            tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                        }
                    }
                }
            }
        }
    }
}



template <typename DataType, int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT, int MTP_REGROUP_COUNT, int REUSE_KV_TIMES>
inline __device__ void mla_apply_mask_causal_tile16x32(DataType tensor[M_WARP_COUNT * N_WARP_COUNT][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q, const int mtp, const int layout) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + lane_id & 15;
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4);
    #pragma unroll
    for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m * 16;
            int col_idx_limit_right;
            if constexpr (REUSE_KV_TIMES == 0) {
                col_idx_limit_right = std::min(max_seqlen_k, row_idx + max_seqlen_k - max_seqlen_q);
            } else {
                const int row_in_mtp = layout == 0 ? (row_idx % mtp): (row_idx / MTP_REGROUP_COUNT);
                col_idx_limit_right = std::min(max_seqlen_k, row_in_mtp + max_seqlen_k - mtp);
            }
            #pragma unroll
            for (int ni = 0; ni < N_WARP_COUNT; ++ni)  {
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n * 16;
                    #pragma unroll
                    for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx * 4;
                        tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx] = (col_idx > col_idx_limit_right) ? -INFINITY: tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                    }
                }
            }
        }
    }
}
