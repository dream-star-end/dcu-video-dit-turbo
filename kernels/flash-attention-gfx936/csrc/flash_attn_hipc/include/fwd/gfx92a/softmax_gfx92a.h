#pragma once

#include "philox.cuh"
#include "../utils.h"

using namespace flash;

template <typename DataType, int WARP_M, int WARP_N>
inline __device__ void apply_mask_gfx92a(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int max_seqlen_k,
                                  const int col_idx_offset_ = 0) {
    const int lane_id = threadIdx.x & 63;
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4);
    #pragma unroll
    for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
        #pragma unroll
        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n * 16;
            #pragma unroll
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                const int col_idx = col_idx_base + vec_idx * 4;
                if (col_idx >= max_seqlen_k) {
                    #pragma unroll
                    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
                        #pragma unroll
                        for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                            tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                        }
                    }
                }
            }
        }
    }
}



template <typename DataType, int WARP_M, int WARP_N>
inline __device__ void apply_mask_causal_gfx92a(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15);
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4);
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m * 16;
            const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + max_seqlen_k - max_seqlen_q);
            #pragma unroll
            for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n * 16;
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx * 4;
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = (col_idx > col_idx_limit_right) ? -INFINITY: tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                    }
                }
            }
        }
    }
}




template <bool HasWSLeft=true, typename DataType, int WARP_M, int WARP_N>
inline __device__ void apply_mask_local_gfx92a(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q,
                                        const int window_size_left, const int window_size_right) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15);
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4);

    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m * 16;
            const int col_idx_limit_left = std::max(0, row_idx + 1 + max_seqlen_k - max_seqlen_q - window_size_left);
            const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + max_seqlen_k - max_seqlen_q + window_size_right);
            #pragma unroll
            for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n * 16;
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx * 4;
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = (col_idx > col_idx_limit_right || (HasWSLeft && col_idx < (col_idx_limit_left - 1))) ?
                            -INFINITY: tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                    }
                }
            }
        }
    }
}


template <typename DataType, int WARP_M, int WARP_N>
inline __device__ void apply_alibi_gfx92a(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q, float g_alibi) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15);
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4);
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m * 16;
            #pragma unroll
            for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n * 16;
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx * 4;
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] += g_alibi * (col_idx - row_idx);
                    }
                }
            }
        }
    }
}