#pragma once

#include "philox.cuh"
#include "../utils.h"

using namespace flash;

template<typename RegType, typename StoreType, int WARP_M, int WARP_N, int kBlockM, int kBlockN, bool Is_even_MN>
__forceinline__ __device__ void store_s_dmask_gfx938(
    void *p_ptr,
    RegType s_reg[(WARP_M / 32) * (WARP_N / 32)][4],
    int bidb,
    int bidh,
    int m_block,
    int n_block,
    int warp_id,
    int lane_id,
    int num_heads,
    int seqlen_q_rounded,
    int seqlen_k_rounded,
    int actual_seqlen_q,
    int actual_seqlen_k
) {
    if (p_ptr == nullptr) return;
    StoreType *p = reinterpret_cast<StoreType *>(p_ptr);
    const int logical_m_block = actual_seqlen_q <= kBlockM ? 0 : ((actual_seqlen_q - 1) / kBlockM - m_block);
    const int row_offset_p = ((bidb * num_heads + bidh) * seqlen_q_rounded + logical_m_block * kBlockM) * seqlen_k_rounded + n_block * kBlockN;
    const int lane_row = lane_id & 15;
    const int lane_col_group = lane_id >> 4;

    #pragma unroll
    for (int ni = 0; ni < (WARP_N / 32); ++ni) {
        #pragma unroll
        for (int mi = 0; mi < (WARP_M / 32); ++mi) {
            #pragma unroll
            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                    #pragma unroll
                    for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int q_row_in_tile = warp_id * WARP_M + mi * 32 + lane_row + min_tile_m * 16;
                        const int k_col_in_tile = ni * 32 + lane_col_group * 4 + min_tile_n * 16 + vec_idx;
                        if constexpr (Is_even_MN) {
                            const int offset = row_offset_p + q_row_in_tile * seqlen_k_rounded + k_col_in_tile;
                            p[offset] = static_cast<StoreType>(
                                s_reg[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]
                            );
                        } else {
                            const int q_row = m_block * kBlockM + q_row_in_tile;
                            const int k_col = n_block * kBlockN + k_col_in_tile;
                            if (q_row < actual_seqlen_q && k_col < actual_seqlen_k) {
                                const int offset = row_offset_p + q_row_in_tile * seqlen_k_rounded + k_col_in_tile;
                                p[offset] = static_cast<StoreType>(
                                    s_reg[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename DataType, int WARP_M, int WARP_N>
inline __device__ void apply_mask_gfx938(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int max_seqlen_k,
                                  const int col_idx_offset_ = 0) {
    const int lane_id = threadIdx.x & 63;
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 4;
    #pragma unroll
    for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
        #pragma unroll
        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n * 16;
            #pragma unroll
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                const int col_idx = col_idx_base + vec_idx;
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
inline __device__ void apply_mask_causal_gfx938(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15);
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 4;
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
                        const int col_idx = col_idx_base + vec_idx;
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = (col_idx > col_idx_limit_right) ? -INFINITY: tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                    }
                }
            }
        }
    }
}




template <bool HasWSLeft=true, typename DataType, int WARP_M, int WARP_N>
inline __device__ void apply_mask_local_gfx938(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q,
                                        const int window_size_left, const int window_size_right) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15);
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 4;

    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m * 16;
            const int col_idx_limit_left = std::max(0, row_idx + 1 + max_seqlen_k - max_seqlen_q - window_size_left);
            const int col_idx_limit_right = std::min(max_seqlen_k - 1, row_idx + max_seqlen_k - max_seqlen_q + window_size_right);
            #pragma unroll
            for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n * 16;
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx;
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = (col_idx > col_idx_limit_right || (HasWSLeft && col_idx < (col_idx_limit_left - 1))) ?
                            -INFINITY: tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                    }
                }
            }
        }
    }
}


template <typename DataType, int WARP_M, int WARP_N>
inline __device__ void apply_alibi_gfx938(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q, float g_alibi) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15);
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 4;
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
                        const int col_idx = col_idx_base + vec_idx;
                        const int relative_pos = row_idx + max_seqlen_k - max_seqlen_q - col_idx;
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] +=
                            -g_alibi * static_cast<float>(relative_pos >= 0 ? relative_pos : -relative_pos);
                    }
                }
            }
        }
    }
}


template<bool Is_first, bool Check_inf=false, typename DataType0, typename DataType1, int K/*head_dim*/, int kBlockK, int WARP_M, int WARP_N>
inline __device__ void softmax_rescale_o_gfx938(DataType0 scores[(WARP_N / 32) * (WARP_M / 32)][4], DataType1 *scores_max, DataType1 *scores_sum,
                                         DataType0 acc_o[(K / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4], float softmax_scale_log2) {
    if constexpr (Is_first) {
        reduce_max</*zero_init=*/true, DataType0, DataType1, WARP_M, WARP_N>(scores, scores_max);
        scale_apply_exp2<true, DataType0, DataType1, WARP_M, WARP_N>(scores, scores_max, softmax_scale_log2);
        reduce_sum<true, DataType0, DataType1, WARP_M, WARP_N>(scores, scores_sum);
    } else {
        DataType1 scores_max_cur[(WARP_M / 32)];
        reduce_max</*zero_init=*/false, DataType0, DataType1, WARP_M, WARP_N>(scores, scores_max, scores_max_cur); // scores_max is prev scores max

        for (int mi = 0; mi < (WARP_M / 32); ++mi) {
            // If max is -inf, then all elements must have been -inf (possibly due to masking).
            // We don't want (-inf - (-inf)) since that would give NaN.
            // If we don't have float around M_LOG2E the multiplication is done in fp64.
            // #pragma unroll
            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                float scores_max_cur_reg = !Check_inf
                        ? scores_max_cur[mi * 2].f32[min_tile_m]
                        : (scores_max_cur[mi * 2].f32[min_tile_m] == -INFINITY ? 0.0f : scores_max_cur[mi * 2].f32[min_tile_m]);

                // optimization from flash-attention-4
                if (scores_max[mi * 2].f32[min_tile_m] < scores_max_cur_reg) {
                    float scores_scale = __llvm_exp2_f32((scores_max[mi * 2].f32[min_tile_m] - scores_max_cur_reg) * softmax_scale_log2);
                    scores_sum[mi * 2].f32[min_tile_m] *= scores_scale;

                    __float2 scores_scale_pair = {scores_scale, scores_scale};

                    #pragma unroll
                    for(int pv_n_loop = 0; pv_n_loop < (K / kBlockK); pv_n_loop++)  {
                        #pragma unroll
                        for (int ni = 0; ni < (kBlockK / 32); ++ni)  {
                            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                            // max * log_2(e)) This allows the compiler to use the ffma
                            // instruction instead of fadd and fmul separately.
                            // min tile is 32*32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
                            for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                                int pv_tile_id = pv_n_loop * (WARP_M / 32) * (kBlockK / 32) + mi + ni * (WARP_M / 32);
                                int mmac_id    = min_tile_n * 2 + min_tile_m;
                            #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                                #pragma unroll
                                for(int vec_idx = 0; vec_idx < 2; ++vec_idx) {
                                    acc_o[pv_tile_id][min_tile_n * 2 + min_tile_m].u64[vec_idx] = __builtin_hcu_pk_mul_f32(
                                        acc_o[pv_tile_id][mmac_id].u64[vec_idx],
                                        scores_scale_pair
                                    );
                                }
                            #else
                                #pragma unroll
                                for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                                    acc_o[pv_tile_id][mmac_id].f32[vec_idx] *= scores_scale;
                                }
                            #endif
                            }
                        }
                    }
                }
            }
        }
        scale_apply_exp2<true, DataType0, DataType1, WARP_M, WARP_N>(scores, scores_max_cur, softmax_scale_log2);

        DataType1 scores_sum_cur[(WARP_M / 32)];
        for (int mi = 0; mi < (WARP_M / 32); ++mi) {
            scores_sum_cur[mi].u64 = 0x0;
        }
        reduce_sum<true, DataType0, DataType1, WARP_M, WARP_N>(scores, scores_sum_cur);

        for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
            scores_sum[mi].u64 = __builtin_hcu_pk_add_f32(
                scores_sum[mi].u64,
                scores_sum_cur[mi].u64
            );
        #else // for perf-model, add listed below will be optimized as v_fmac_f32, leading to incorrect results
            scores_sum[mi].f32[0] += scores_sum_cur[mi].f32[0];
            scores_sum[mi].f32[1] += scores_sum_cur[mi].f32[1];
        #endif

        #if defined(USE_V_MOV_B64) && (defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__))
            inlineasm_fa_v_mov_b64(
                scores_max[mi].u64,
                scores_max_cur[mi].u64
            );
        #else
            scores_max[mi].f32[0] = scores_max_cur[mi].f32[0];
            scores_max[mi].f32[1] = scores_max_cur[mi].f32[1];
        #endif
        }
    }
};
