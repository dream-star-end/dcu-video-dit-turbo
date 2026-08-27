#pragma once

#include "fwd/utils.h"

using namespace flash;


template <typename DataType, int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT>
inline __device__ void fp8_mla_apply_mask_gfx938(DataType tensor[M_WARP_COUNT * N_WARP_COUNT][4], const int max_seqlen_k,
                                  const int col_idx_offset_ = 0) {
    const int lane_id = threadIdx.x & 63; // lane id, 0-63
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 8;
    #pragma unroll
    for (int ni = 0; ni < N_WARP_COUNT; ++ni) {
        #pragma unroll
        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n * 4;
            #pragma unroll
            for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                const int col_idx = col_idx_base + vec_idx;
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


template <typename DataType, int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT>
inline __device__ void fp8_mla_apply_mask_causal_gfx938_mtp(DataType tensor[M_WARP_COUNT * N_WARP_COUNT][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q, const int mtp, const int layout) {
    const int MTP_REGROUP_COUNT = max_seqlen_q / mtp;
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15);
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 8;
    #pragma unroll
    for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m * 16;
            const int row_in_mtp = layout == 0 ? (row_idx % mtp): (row_idx / MTP_REGROUP_COUNT);
            const int col_idx_limit_right = std::min(max_seqlen_k, row_in_mtp + max_seqlen_k - mtp);
            #pragma unroll
            for (int ni = 0; ni < N_WARP_COUNT; ++ni)  {
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n * 4;
                    #pragma unroll
                    for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx;
                        tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx] = (col_idx > col_idx_limit_right) ? -INFINITY: tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                    }
                }
            }
        }
    }
}



template <typename DataType, int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT>
inline __device__ void fp8_mla_apply_descale_gfx938(DataType tensor[M_WARP_COUNT * N_WARP_COUNT][4], const __float2 qk_descale) {
    #pragma unroll
    for (int i = 0; i < M_WARP_COUNT * N_WARP_COUNT; ++i) {
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            #pragma unroll
            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                tensor[i][min_tile_n * 2 + min_tile_m].u64[0] = __builtin_hcu_pk_mul_f32(tensor[i][min_tile_n * 2 + min_tile_m].u64[0], qk_descale);
                tensor[i][min_tile_n * 2 + min_tile_m].u64[1] = __builtin_hcu_pk_mul_f32(tensor[i][min_tile_n * 2 + min_tile_m].u64[1], qk_descale);
            }
        }
    }
}