#pragma once

#include "philox.cuh"
#include "fwd/utils.h"

using namespace flash;

template <typename DataType, int WARP_M, int WARP_N, int M_MMAC_COUNT>
inline __device__ void kvcache_apply_mask(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int max_seqlen_k,
                                  const int col_idx_offset_ = 0) {
    const int lane_id = threadIdx.x & 63;
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 2;
    #pragma unroll
    for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
        #pragma unroll
        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n;
            #pragma unroll
            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                const int col_idx = col_idx_base + vec_idx * 8;
                if (col_idx >= max_seqlen_k) {
                    #pragma unroll
                    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
                        #pragma unroll
                        for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                            tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                        }
                    }
                }
            }
        }
    }
}

template <typename DataType, int WARP_M, int WARP_N, int BLOCK_ROW_STRIDE, int M_MMAC_COUNT>
inline __device__ void kvcache_apply_dropout(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int max_seqlen_k, const int col_idx_offset_,
                                     unsigned long long seed, unsigned long long offset, uint32_t p_dropout_in_8bits_value,
                                     union_vec2_uint rowcol, uint32_t* dropout_debug_count) {
    // static_assert(WARP_M == 32 and "For Dropout, only WARP_M=32 is supported yet!");
    const int lane_id = threadIdx.x & 63; // lane id, 0-63
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 2;
    // prepare 4 uint for 16 uint8
    union_vec4_uint random_uint4;
    for (int mi = 0; mi < (WARP_M / 32); ++mi, rowcol.u32.x += BLOCK_ROW_STRIDE) { // when WARP_M > 32, attention, block_row_idx is computed by BLOCK_M / 32 rather than BLOCK_M / WARP_M
        #pragma unroll
        for (uint32_t ni = 0; ni < (WARP_N / 32); ++ni, ++rowcol.u32.y)  {
            // for each 16 elements, generate 16 int8 -> 4 u32
            random_uint4.u32 = flash::philox(seed, rowcol.u64, offset);
            int cnt = 0;
            #pragma unroll
            for(uint32_t min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n;
                #pragma unroll
                for(uint32_t vec_idx = 0; vec_idx < 4; ++vec_idx) {
                    const int col_idx = col_idx_base + vec_idx * 8;
                    if (col_idx < max_seqlen_k) {
                        #pragma unroll
                        for(uint32_t min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                            uint32_t cur_pos = (min_tile_n * 2 + min_tile_m) * 4 + vec_idx;
                            uint32_t cur_rand = random_uint4.u8[cur_pos] & 0xffffffff; // uint8 -> u32, since hcu has no compare instructions with 8/16 bits
                            if (cur_rand >= p_dropout_in_8bits_value) {
                                tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = 0x0;
                                ++cnt;
                            }
                        }
                    }
                }
            }
            #if 0
            atomicAdd(dropout_debug_count, cnt);
            if (threadIdx.x == 0) atomicAdd(dropout_debug_count + 1, 1);
            #endif
        }
    }
}


template <typename DataType, int WARP_M, int WARP_N, int M_MMAC_COUNT>
inline __device__ void kvcache_apply_mask_causal(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q, const int ngroups) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15) * 2;
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m;
            const int col_idx_limit_right = std::min(max_seqlen_k, (row_idx / ngroups)/*only for layout 1: bshd*/ + max_seqlen_k - (max_seqlen_q / ngroups));
            #pragma unroll
            for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n;
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx * 8;
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = (col_idx > col_idx_limit_right) ? -INFINITY: tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                    }
                }
            }
        }
    }
}


template <bool HasWSLeft=true, typename DataType, int WARP_M, int WARP_N, int M_MMAC_COUNT>
inline __device__ void kvcache_apply_mask_local(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q, 
                                        const int window_size_left, const int window_size_right) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15) * 2;
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 2;

    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m;
            const int col_idx_limit_left = std::max(0, row_idx + 1 + max_seqlen_k - max_seqlen_q - window_size_left);
            const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + max_seqlen_k - max_seqlen_q + window_size_right);
            #pragma unroll
            for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n;
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx * 8;
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = (col_idx > col_idx_limit_right || (HasWSLeft && col_idx < (col_idx_limit_left - 1))) ?
                            -INFINITY: tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                    }
                }
            }
        }
    }
}


template <typename DataType, int WARP_M, int WARP_N, int M_MMAC_COUNT>
inline __device__ void kvcache_apply_alibi(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q, float g_alibi) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15) * 2;
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m;
            #pragma unroll
            for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n;
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx * 8;
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] += g_alibi * (col_idx - row_idx);
                    }
                }
            }
        }
    }
}

template<bool zero_init=true, typename Operator, int OpType, typename DataType0, typename DataType1, int WARP_M, int WARP_N, int M_MMAC_COUNT>
__device__ inline void kvcache_thread_reduce_max(const DataType0 tensor[(WARP_M / 32) * (WARP_N / 32)][4], DataType1 *summary, Operator &op, DataType1 *summary_cur=nullptr) {
    if(zero_init == true) {
        #pragma unroll
        for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                summary[m_idx * 2].f32[min_tile_m] = -INFINITY;  // OpType：0 is sum operator, 1 is max operator
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                            summary[m_idx * 2].f32[min_tile_m] = op(summary[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                        }
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                summary_cur[m_idx * 2].f32[min_tile_m] = summary[m_idx * 2].f32[min_tile_m];
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                            summary_cur[m_idx * 2].f32[min_tile_m] = op(summary_cur[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                        }
                    }
                }
            }
        }
    }
}



template<bool zero_init=true, typename Operator, int OpType, typename DataType0, typename DataType1, int WARP_M, int WARP_N, int M_MMAC_COUNT>
__device__ inline void kvcache_thread_reduce_sum(const DataType0 tensor[(WARP_M / 32) * (WARP_N / 32)][4], DataType1 *summary, Operator &op, DataType1 *summary_cur=nullptr) {
    if(zero_init == true) {
        #pragma unroll
        for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
            summary[m_idx * 2].u64 = 0x0;
            #pragma unroll
            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        __float2 additem_pair = {tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2].f32[vec_idx], tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2 + 1].f32[vec_idx]};
                        summary[m_idx * 2].u64 = __builtin_hcu_pk_add_f32(
                            summary[m_idx * 2].u64,
                            additem_pair
                        );
                    }
                }
            }
        #else
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                summary[m_idx * 2].f32[min_tile_m] = 0;  // OpType：0 is sum operator, 1 is max operator
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                            summary[m_idx * 2].f32[min_tile_m] = op(summary[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                        }
                    }
                }
            }
        #endif
        }
    } else {
        #pragma unroll
        for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
            summary_cur[m_idx * 2].u64 = summary[m_idx * 2].u64;
            #pragma unroll
            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        __float2 additem_pair = {tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2].f32[vec_idx], tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2 + 1].f32[vec_idx]};
                        summary_cur[m_idx * 2].u64 = __builtin_hcu_pk_add_f32(
                            summary_cur[m_idx * 2].u64,
                            additem_pair
                        );
                    }
                }
            }
        #else
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                summary_cur[m_idx * 2].f32[min_tile_m] = summary[m_idx * 2].f32[min_tile_m];
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                            summary_cur[m_idx * 2].f32[min_tile_m] = op(summary_cur[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                        }
                    }
                }
            }
        #endif
        }
    }
}


template<typename Operator, typename DataType, int WARP_M>
__device__ inline void kvcache_quad_allreduce_(DataType *dst, DataType *src, Operator &op) {
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); mi++) {
        dst[mi] = Allreduce<64>::run(src[mi], op);
    }
}

template<bool zero_init=true, typename Operator, int OpType, typename DataType0, typename DataType1, int WARP_M, int WARP_N, int M_MMAC_COUNT>
__device__ inline void kvcache_reduce_(const DataType0 tensor[(WARP_M / 32) * (WARP_N / 32)][4], DataType1 *summary, Operator &op, DataType1 *summary_cur=nullptr) {
    if constexpr (OpType == 0) { // sum
        if constexpr (zero_init == true) {
            kvcache_thread_reduce_sum<true, Operator, 0, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, summary, op);
            kvcache_quad_allreduce_<Operator, DataType1, WARP_M>(summary, summary, op);
        } else {
            kvcache_thread_reduce_sum<false, Operator, 0, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, summary, op, summary_cur);
            kvcache_quad_allreduce_<Operator, DataType1, WARP_M>(summary_cur, summary_cur, op);
        }
    } else if constexpr (OpType == 1) { // max
        if constexpr (zero_init == true) {
            kvcache_thread_reduce_max<true, Operator, 1, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, summary, op);
            kvcache_quad_allreduce_<Operator, DataType1, WARP_M>(summary, summary, op);
        } else {
            kvcache_thread_reduce_max<false, Operator, 1, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, summary, op, summary_cur);
            kvcache_quad_allreduce_<Operator, DataType1, WARP_M>(summary_cur, summary_cur, op);
        }
    }
}


template<bool zero_init=true, typename DataType0, typename DataType1, int WARP_M, int WARP_N, int M_MMAC_COUNT>
__device__ inline void kvcache_reduce_max(const DataType0 tensor[(WARP_M / 32) * (WARP_N / 32)][4], DataType1 *max , DataType1 *max_cur=nullptr) {
    MaxOp<float> max_op;
    if constexpr (zero_init == true) {
        kvcache_reduce_<true, MaxOp<float>, 1, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, max, max_op);
    } else {
        kvcache_reduce_<false, MaxOp<float>, 1, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, max, max_op, max_cur);
    }
}

template<bool zero_init=true, typename DataType0, typename DataType1, int WARP_M, int WARP_N, int M_MMAC_COUNT>
__device__ inline void kvcache_reduce_sum(DataType0 tensor[(WARP_M / 32) * (WARP_N / 32)][4], DataType1 *sum,  DataType1 *sum_cur=nullptr){
    SumOp<float> sum_op;
    if constexpr (zero_init == true) {
        kvcache_reduce_<true, SumOp<float>, 0, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, sum, sum_op);
    } else {
        kvcache_reduce_<false, SumOp<float>, 0, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, sum, sum_op, sum_cur);
    }
}



// Apply the exp to all the elements.
template <bool Scale_max=true, typename DataType0, typename DataType1, int WARP_M, int WARP_N, int M_MMAC_COUNT>
inline __device__ void kvcache_scale_apply_exp2(DataType0 tensor[(WARP_M / 32) * (WARP_N / 32)][4], const DataType1 *max, const float scale) {
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            const float max_scaled = (max[mi * 2].f32[min_tile_m] == -INFINITY) ? 0.f : (max[mi * 2].f32[min_tile_m] * (Scale_max ? scale : float(M_LOG2E)));
            __float2 neg_max_scaled_pair = {-max_scaled, -max_scaled};
            __float2 scale_pair = {scale, scale};
            #pragma unroll
            for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
                // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                // max * log_2(e)) This allows the compiler to use the ffma
                // instruction instead of fadd and fmul separately.
                // min tile is 32 * 32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 2; ++vec_idx) {
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].u64[vec_idx] = __builtin_hcu_pk_fma_f32(
                            tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].u64[vec_idx],
                            scale_pair,
                            neg_max_scaled_pair
                        );
                    }
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = __llvm_exp2_f32(tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                    }
                #else
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = __llvm_exp2_f32(tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] * scale - max_scaled);
                    }
                #endif
                }
            }
        }
    }
}



template<bool Is_first, bool Check_inf=false, typename DataType0, typename DataType1, typename DataType2, int K/*head_dim*/, int kBlockK, int WARP_M, int WARP_N, int WARP_NUM, int M_MMAC_COUNT>
inline __device__ void kvcache_softmax_rescale_o(DataType0 scores[(WARP_N / 32) * (WARP_M / 32)][4], DataType1 *scores_max, DataType1 *scores_sum,
                                         DataType0 acc_o[(K / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4], DataType2* max_lds, int warp_id, float softmax_scale_log2) {
    if constexpr (Is_first) {
        kvcache_reduce_max</*zero_init=*/true, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(scores, scores_max);
        kvcache_scale_apply_exp2<true, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(scores, scores_max, softmax_scale_log2);
        kvcache_reduce_sum<true, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(scores, scores_sum);
    } else {
        DataType1 scores_max_cur[(WARP_M / 32)];
        kvcache_reduce_max</*zero_init=*/false, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(scores, scores_max, scores_max_cur); // scores_max is prev scores max

        int lane_id = threadIdx.x & 63;
        if constexpr (WARP_NUM > 1) {
            int dword_offset_base = (lane_id & 15);
            if (lane_id < 16) {
                if (warp_id == 0) {
                    for (int m_loop = 0; m_loop < M_MMAC_COUNT; ++m_loop) {
                        max_lds[dword_offset_base + m_loop * 32] = -INFINITY;
                    }
                }
                __syncthreads();
                for (int m_loop = 0; m_loop < M_MMAC_COUNT; ++m_loop) {
                    __builtin_amdgcn_ds_fmaxf((__attribute__((address_space(3))) float *)max_lds + dword_offset_base + m_loop * 32, scores_max_cur[0].f32[m_loop], 0, 0, false);
                }
            }
            __syncthreads();
            for (int m_loop = 0; m_loop < M_MMAC_COUNT; ++m_loop) {
                scores_max_cur[0].f32[m_loop] = max_lds[dword_offset_base + m_loop * 32];
            }
        }

        #pragma unroll
        for (int mi = 0; mi < (WARP_M / 32); ++mi) {
            // If max is -inf, then all elements must have been -inf (possibly due to masking).
            // We don't want (-inf - (-inf)) since that would give NaN.
            // If we don't have float around M_LOG2E the multiplication is done in fp64.
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                float scores_max_cur_reg = !Check_inf
                        ? scores_max_cur[mi * 2].f32[min_tile_m]
                        : (scores_max_cur[mi * 2].f32[min_tile_m] == -INFINITY ? 0.0f : scores_max_cur[mi * 2].f32[min_tile_m]);

                float scores_scale = __llvm_exp2_f32((scores_max[mi * 2].f32[min_tile_m] - scores_max_cur_reg) * softmax_scale_log2);
                scores_sum[mi * 2].f32[min_tile_m] *= scores_scale;

                __float2 scores_scale_pair = {scores_scale, scores_scale};

                #pragma unroll
                for(int pv_n_loop = 0; pv_n_loop < (K / kBlockK); ++pv_n_loop)  {
                    #pragma unroll
                    for (int ni = 0; ni < (kBlockK / 32); ++ni)  {
                        // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                        // max * log_2(e)) This allows the compiler to use the ffma
                        // instruction instead of fadd and fmul separately.
                        // min tile is 32 * 32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
                        #pragma unroll
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            // 936 及之后的架构有 pk_mul 指令
                        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                            #pragma unroll
                            for(int vec_idx = 0; vec_idx < 2; ++vec_idx) {
                                acc_o[pv_n_loop * (WARP_M / 32) * (kBlockK / 32) + (mi + ni * (WARP_M / 32))][min_tile_n * 2 + min_tile_m].u64[vec_idx] = __builtin_hcu_pk_mul_f32(
                                    acc_o[pv_n_loop * (WARP_M / 32) * (kBlockK / 32) + (mi + ni * (WARP_M / 32))][min_tile_n * 2 + min_tile_m].u64[vec_idx],
                                    scores_scale_pair
                                );
                            }
                        #else
                            // 928 及之前的架构没 pk_mul 指令
                            #pragma unroll
                            for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                                acc_o[pv_n_loop * (WARP_M / 32) * (kBlockK / 32) + (mi + ni * (WARP_M / 32))][min_tile_n * 2 + min_tile_m].f32[vec_idx] *= scores_scale;
                            }
                        #endif
                        }
                    }
                }
            }
        }
        kvcache_scale_apply_exp2<true, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(scores, scores_max_cur, softmax_scale_log2);

        DataType1 scores_sum_cur[(WARP_M / 32)];
        for (int mi = 0; mi < (WARP_M / 32); ++mi) {
            scores_sum_cur[mi].u64 = 0x0;
        }
        kvcache_reduce_sum<true, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(scores, scores_sum_cur);

        if constexpr (WARP_NUM > 1) {
            // 重新求多个 wave 的归一化和
            DataType2* sum_lds = max_lds + 64;
            if(lane_id < 16) {
                // 每个 wave 的归一化和写到 lds
                for (int mi = 0; mi < (WARP_M / 32); ++mi) {
                    if constexpr (M_MMAC_COUNT == 1) {
                        sum_lds[warp_id * WARP_M + mi * 32 + lane_id * 2] = scores_sum_cur[mi].f32[0];
                    } else {
                        *(__float2*)(sum_lds + warp_id * WARP_M + mi * 32 + lane_id * 2) = scores_sum_cur[mi].u64; // M_MMAC_COUNT doesn't exceed 2
                    }
                }
                __syncthreads();
                // 0 号 wave reduce 其他 wave 的归一化和
                if (warp_id == 0) {
                    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
                        if constexpr (M_MMAC_COUNT == 1) {
                            float tmp = sum_lds[mi * 32 + lane_id * 2];
                            for(int warp_loop = 1; warp_loop < WARP_NUM; warp_loop++) {
                                tmp += sum_lds[warp_loop * WARP_M + mi * 32 + lane_id * 2];
                            }
                            sum_lds[mi * 32 + lane_id * 2] = tmp;
                        } else {
                            __float2 cur_wave_sum = *(__float2*)(sum_lds + mi * 32 + lane_id * 2);
                            #pragma unroll
                            for(int warp_loop = 1; warp_loop < WARP_NUM; warp_loop++) {
                                __float2 other_warp_sum = *(__float2*)(sum_lds + warp_loop * WARP_M + mi * 32 + lane_id * 2);
                                #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                                    cur_wave_sum = __builtin_hcu_pk_add_f32(cur_wave_sum, other_warp_sum);
                                #else
                                    cur_wave_sum[0] += other_warp_sum[0];
                                    cur_wave_sum[1] += other_warp_sum[1];
                                #endif
                            }
                            *(__float2*)(sum_lds + mi * 32 + lane_id * 2) = cur_wave_sum;
                        }
                    }
                }
            }
            __syncthreads();
            // 4 个 wave 从 lds 中读取最后 reduce 的归一化和
            for (int mi = 0; mi < (WARP_M / 32); ++mi) {
                if constexpr (M_MMAC_COUNT == 1) {
                    scores_sum_cur[mi * 2].f32[0] = sum_lds[mi * 32 + (lane_id & 15) * 2];
                } else {
                    scores_sum_cur[mi * 2].u64 = *(__float2*)(sum_lds + mi * 32 + (lane_id & 15) * 2);
                }
            }
            __syncthreads(); // 以免后续的 buffer_load_to_lds 调度到这之前
        }

        for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
            scores_sum[mi].u64 = __builtin_hcu_pk_add_f32(
                scores_sum[mi].u64,
                scores_sum_cur[mi].u64
            );
            // #######################################################
            scores_max[mi].u64 = scores_max_cur[mi].u64;
        #else
            scores_sum[mi].f32[0] += scores_sum_cur[mi].f32[0];
            scores_sum[mi].f32[1] += scores_sum_cur[mi].f32[1];
            // #######################################################
            scores_max[mi].f32[0] = scores_max_cur[mi].f32[0];
            scores_max[mi].f32[1] = scores_max_cur[mi].f32[1];
        #endif
        }
    }
};



template <int WARP_M, int WARP_N, int M_MMAC_COUNT, typename Element, typename ElementAccum>
inline __device__ void kvcache_convert_pk_type(union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (WARP_N / 32)][4], union_vec4_fp32 s_reg[(WARP_M / 32) * (WARP_N / 32)][4]) {
    #pragma unroll
    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
        #pragma unroll
        for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__) || defined(__gfx92a__)
                        p_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f16x2[min_tile_k] = DownCastPair<float, Element>(
                        s_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f32x2[min_tile_k]);
                        p_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f16x2[min_tile_k] = DownCastPair<float, Element>(
                        s_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f32x2[min_tile_k]);
                    #else
                        p_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0] = DownCast<float, Element, false>(
                        s_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f32[min_tile_k * 2 + 0]);
                        p_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0] = DownCast<float, Element, false>(
                        s_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f32[min_tile_k * 2 + 0]);
                        p_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1] = DownCast<float, Element, false>(
                        s_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f32[min_tile_k * 2 + 1]);
                        p_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1] = DownCast<float, Element, false>(
                        s_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f32[min_tile_k * 2 + 1]);
                    #endif
                }
            }
        }
    }
}
