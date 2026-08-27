#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_causal>
struct Alibi {

    const float alibi_slope;
    const int max_seqlen_k, max_seqlen_q;

    __forceinline__ __device__ Alibi(const float alibi_slope, const int max_seqlen_k, const int max_seqlen_q)
        : alibi_slope(alibi_slope)
        , max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q) {
    };

    template <typename Engine, typename Layout>
    __forceinline__ __device__ void apply_alibi(Tensor<Engine, Layout> &tensor,
                                      const int col_idx_offset_,
                                      const int row_idx_offset,
                                      const int warp_row_stride) {
        // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
        static_assert(Layout::rank == 2, "Only support 2D Tensor");
        const int lane_id = threadIdx.x % 64;
        const int col_idx_offset = col_idx_offset_ + lane_id / 16;
        const int stride_between_each_repeat = 16;
        const int stride_between_each_thread = 4;

        if constexpr (Is_causal) {  // Simpler, we add the same bias vector to all rows
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * stride_between_each_repeat;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j * stride_between_each_thread;
                    #pragma unroll
                    for (int mi = 0; mi < size<0>(tensor); ++mi) {
                        tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                    }
                }
            }
        } else {  // Bias depends on both row_idx and col_idx
            #pragma unroll
            for (int mi = 0; mi < size<0>(tensor); ++mi) {
                const int row_idx = row_idx_offset + mi * warp_row_stride;
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * stride_between_each_repeat;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j * stride_between_each_thread;
                        tensor(mi, make_coord(j, nj)) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                    }
                }
            }
        }
    }

    template <typename Engine, typename Layout>
    __forceinline__ __device__ void apply_alibi_continuous(Tensor<Engine, Layout> &tensor,
                                      const int col_idx_offset_,
                                      const int row_idx_offset,
                                      const int warp_row_stride) {
        // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
        static_assert(Layout::rank == 2, "Only support 2D Tensor");
        const int lane_id = threadIdx.x % 64;
        const int col_idx_offset = col_idx_offset_ + (lane_id / 16) * 4;
        const int stride_between_each_repeat = 16;
        const int stride_between_each_thread = 1;

        if constexpr (Is_causal) {  // Simpler, we add the same bias vector to all rows
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * stride_between_each_repeat;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j * stride_between_each_thread;
                    #pragma unroll
                    for (int mi = 0; mi < size<0>(tensor); ++mi) {
                        tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                    }
                }
            }
        } else {  // Bias depends on both row_idx and col_idx
            #pragma unroll
            for (int mi = 0; mi < size<0>(tensor); ++mi) {
                const int row_idx = row_idx_offset + mi * warp_row_stride;
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * stride_between_each_repeat;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j * stride_between_each_thread;
                        tensor(mi, make_coord(j, nj)) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                    }
                }
            }
        }
    }

    template <typename Engine, typename Layout>
    __forceinline__ __device__ void apply_alibi_trans(Tensor<Engine, Layout> &tensor,
                                      const int col_idx_offset_,
                                      const int row_idx_offset,
                                      const int warp_row_stride) {
        // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
        static_assert(Layout::rank == 2, "Only support 2D Tensor");
        const int lane_id = threadIdx.x % 64;
        const int col_idx_offset = col_idx_offset_ + (lane_id / 16) * 4;
        const int stride_between_each_repeat = 16;
        const int stride_between_each_thread = 1;

        if constexpr (Is_causal) {  // Simpler, we add the same bias vector to all rows
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * stride_between_each_repeat;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j * stride_between_each_thread;
                    #pragma unroll
                    for (int mi = 0; mi < size<0>(tensor); ++mi) {
                        const int row_idx = row_idx_offset + mi * warp_row_stride;
                        tensor(mi, make_coord(j, nj)) += alibi_slope * row_idx;
                    }
                }
            }
        } else {  // Bias depends on both row_idx and col_idx
            #pragma unroll
            for (int mi = 0; mi < size<0>(tensor); ++mi) {
                const int row_idx = row_idx_offset + mi * warp_row_stride;
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * stride_between_each_repeat;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j * stride_between_each_thread;
                        tensor(mi, make_coord(j, nj)) -= alibi_slope * abs(col_idx + max_seqlen_k - max_seqlen_q - row_idx);
                    }
                }
            }
        }
    }
};

}  // namespace flash
