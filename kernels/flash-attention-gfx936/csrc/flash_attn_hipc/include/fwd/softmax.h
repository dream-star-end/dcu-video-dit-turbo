#pragma once

#include "philox.cuh"
#include "utils.h"

using namespace flash;

template<typename RegType, typename StoreType, int WARP_M, int WARP_N, int kBlockM, int kBlockN, bool Is_even_MN>
__forceinline__ __device__ void store_s_dmask_gfx936(
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
                        const int q_row = m_block * kBlockM + warp_id * WARP_M + mi * 32 + (lane_id & 15) * 2 + min_tile_m;
                        const int k_col = n_block * kBlockN + ni * 32 + (lane_id >> 4) * 2 + min_tile_n + vec_idx * 8;
                        if constexpr (Is_even_MN) {
                            const int offset = ((bidb * num_heads + bidh) * seqlen_q_rounded + q_row) * seqlen_k_rounded + k_col;
                            p[offset] = static_cast<StoreType>(s_reg[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                        } else {
                            if (q_row < seqlen_q_rounded && k_col < seqlen_k_rounded) {
                                const int offset = ((bidb * num_heads + bidh) * seqlen_q_rounded + q_row) * seqlen_k_rounded + k_col;
                                if (q_row < actual_seqlen_q && k_col < actual_seqlen_k) {
                                    p[offset] = static_cast<StoreType>(s_reg[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                                } else {
                                    p[offset] = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename DataType, int WARP_M, int WARP_N>
inline __device__ void apply_mask(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int max_seqlen_k,
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
                        for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                            tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                        }
                    }
                }
            }
        }
    }
}

template <bool Encode_dropout_in_sign_bit=false, typename DataType, int WARP_M, int WARP_N, int BLOCK_ROW_STRIDE, bool Is_even_MN>
inline __device__ void apply_dropout(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int max_seqlen_q, const int max_seqlen_k,
                                     const int row_idx_offset_, const int col_idx_offset_, const int batch_head_idx,
                                     unsigned long long seed, unsigned long long offset, uint32_t p_dropout_in_8bits_value,
                                     union_vec2_uint rowcol, uint32_t* dropout_debug_count) {
    const int lane_id = threadIdx.x & 63; // lane id, 0-63
    const int lane_row = lane_id & 15;
    const int lane_col_group = lane_id >> 4;
    const int row_idx_offset = row_idx_offset_ + lane_row * 2;
    const int col_idx_offset = col_idx_offset_ + lane_col_group * 2;
    const uint32_t p_dropout_threshold = min(255u, p_dropout_in_8bits_value + 1u);
    const unsigned long long offset_base = offset - static_cast<unsigned long long>(lane_id);
    (void)rowcol;
    (void)BLOCK_ROW_STRIDE;
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        #pragma unroll
        for (uint32_t ni = 0; ni < (WARP_N / 32); ++ni) {
            #pragma unroll
            for(uint32_t vec_idx = 0; vec_idx < 4; ++vec_idx) {
                const int abs_q_base = row_idx_offset + mi * 32;
                const int abs_k_base = col_idx_offset + ni * 32 + vec_idx * 8;
                const uint32_t rand_pack = flash::dropout_rand_2x2_u32(seed, offset_base, batch_head_idx, abs_q_base, abs_k_base);
                #pragma unroll
                for(uint32_t min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int abs_k = abs_k_base + min_tile_n;
                    #pragma unroll
                    for(uint32_t min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                        const int abs_q = abs_q_base + min_tile_m;
                        if constexpr (Is_even_MN) {
                            uint32_t cur_rand = static_cast<uint32_t>(flash::dropout_rand_2x2_byte(rand_pack, abs_q, abs_k));
                            const bool drop = cur_rand > p_dropout_threshold;
                            if constexpr (Encode_dropout_in_sign_bit) {
                                const float value = tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                                tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] =
                                    drop ? flash::encode_dropout_sign_bit(value == 0.0f ? 1.0f : value)
                                         : flash::clear_dropout_sign_bit(value);
                            } else {
                            #if defined(__gfx936__)
                                const float keep = drop ? 0.0f : 1.0f;
                                tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] *= keep;
                            #else
                                if (drop) { tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] *= 0.0f; }
                            #endif
                            }
                            #ifdef FA_DEBUG
                            if (drop) {
                                atomicAdd(dropout_debug_count, 1);
                            }
                            #endif
                        } else if constexpr (not Is_even_MN) {
                            if (abs_q < max_seqlen_q && abs_k < max_seqlen_k) {
                                uint32_t cur_rand = static_cast<uint32_t>(flash::dropout_rand_2x2_byte(rand_pack, abs_q, abs_k));
                                const bool drop = cur_rand > p_dropout_threshold;
                                if constexpr (Encode_dropout_in_sign_bit) {
                                    const float value = tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                                    tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] =
                                        drop ? flash::encode_dropout_sign_bit(value == 0.0f ? 1.0f : value)
                                             : flash::clear_dropout_sign_bit(value);
                                } else {
                                #if defined(__gfx936__)
                                    const float keep = drop ? 0.0f : 1.0f;
                                    tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] *= keep;
                                #else
                                    if (drop) { tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] *= 0.0f; }
                                #endif
                                }
                                #ifdef FA_DEBUG
                                if (drop) {
                                    atomicAdd(dropout_debug_count, 1);
                                }
                                #endif
                            }
                        }
                    }
                }
            }
        }
    }
}


template <typename DataType, int WARP_M, int WARP_N>
inline __device__ void apply_mask_causal(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15) * 2;
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m;
            const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + max_seqlen_k - max_seqlen_q); // attention, when max_seqlen_k == max_seqlen_q, vgpr can be reduced again
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


template <bool HasWSLeft=true, typename DataType, int WARP_M, int WARP_N>
inline __device__ void apply_mask_local(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
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
        for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m;
            const int col_idx_limit_left = std::max(0, row_idx + 1 + max_seqlen_k - max_seqlen_q - window_size_left);
            const int col_idx_limit_right = std::min(max_seqlen_k - 1, row_idx + max_seqlen_k - max_seqlen_q + window_size_right);
            #pragma unroll
            for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n;
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx * 8;
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = (row_idx >= max_seqlen_q || col_idx > col_idx_limit_right || (HasWSLeft && col_idx < (col_idx_limit_left - 1))) ?
                            -INFINITY: tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                    }
                }
            }
        }
    }
}


template <typename DataType, int WARP_M, int WARP_N>
inline __device__ void apply_alibi(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q, float gAlibi) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15) * 2;
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m;
            #pragma unroll
            for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n;
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx * 8;
                        const int relative_pos = row_idx + max_seqlen_k - max_seqlen_q - col_idx;
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] +=
                            -gAlibi * static_cast<float>(relative_pos >= 0 ? relative_pos : -relative_pos);
                    }
                }
            }
        }
    }
}

template<bool zero_init=true, typename Operator, int OpType, typename DataType0, typename DataType1, int WARP_M, int WARP_N>
__device__ inline void thread_reduce_max(const DataType0 tensor[(WARP_M / 32) * (WARP_N / 32)][4], DataType1 *summary, Operator &op, DataType1 *summary_cur=nullptr) {
    if(zero_init == true) {
        #pragma unroll
        for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                summary[m_idx * 2].f32[min_tile_m] = -INFINITY;  // OpType：0 is sum operator, 1 is max operator
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        for(int vec_idx = 0; vec_idx < 4; ++vec_idx) { // mmac min_tile is 16*16, a warp is 64 thread
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
            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                summary_cur[m_idx * 2].f32[min_tile_m] = summary[m_idx * 2].f32[min_tile_m];
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        for(int vec_idx = 0; vec_idx < 4; ++vec_idx) { // mmac min_tile is 16*16, a warp is 64 thread
                            summary_cur[m_idx * 2].f32[min_tile_m] = op(summary_cur[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                        }
                    }
                }
            }
        }
    }
}



template<bool zero_init=true, typename Operator, int OpType, typename DataType0, typename DataType1, int WARP_M, int WARP_N>
__device__ inline void thread_reduce_sum(const DataType0 tensor[(WARP_M / 32) * (WARP_N / 32)][4], DataType1 *summary, Operator &op, DataType1 *summary_cur=nullptr) {
    if(zero_init == true) {
        #pragma unroll
        for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
        // 对于 gfx936 及以上的架构, 可以使用 v_pk_add_f32
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
            summary[m_idx * 2].u64 = 0x0;
            #pragma unroll
            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
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
            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                summary[m_idx * 2].f32[min_tile_m] = 0;  // OpType：0 is sum operator, 1 is max operator
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        for(int vec_idx = 0; vec_idx < 4; ++vec_idx) { // mmac min_tile is 16*16, a warp is 64 thread
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
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
            summary_cur[m_idx * 2].u64 = summary[m_idx * 2].u64;
            #pragma unroll
            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) { // mmac min_tile is 16*16, a warp is 64 thread
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
            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                summary_cur[m_idx * 2].f32[min_tile_m] = summary[m_idx * 2].f32[min_tile_m];
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        for(int vec_idx = 0; vec_idx < 4; ++vec_idx) { // mmac min_tile is 16*16, a warp is 64 thread
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
__device__ inline void quad_allreduce_(DataType *dst, DataType *src, Operator &op) {
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); mi++) {
        dst[mi] = Allreduce<64>::run(src[mi], op);
    }
}

template<bool zero_init=true, typename Operator, int OpType, typename DataType0, typename DataType1, int WARP_M, int WARP_N>
__device__ inline void reduce_(const DataType0 tensor[(WARP_M / 32) * (WARP_N / 32)][4], DataType1 *summary, Operator &op, DataType1 *summary_cur=nullptr) {
    if constexpr (OpType == 0) { // sum
        if constexpr (zero_init == true) {
            thread_reduce_sum<true, Operator, 0, DataType0, DataType1, WARP_M, WARP_N>(tensor, summary, op);
            quad_allreduce_<Operator, DataType1, WARP_M>(summary, summary, op);
        } else {
            thread_reduce_sum<false, Operator, 0, DataType0, DataType1, WARP_M, WARP_N>(tensor, summary, op, summary_cur);
            quad_allreduce_<Operator, DataType1, WARP_M>(summary_cur, summary_cur, op);
        }
    } else if constexpr (OpType == 1) { // max
        if constexpr (zero_init == true) {
            thread_reduce_max<true, Operator, 1, DataType0, DataType1, WARP_M, WARP_N>(tensor, summary, op);
            quad_allreduce_<Operator, DataType1, WARP_M>(summary, summary, op);
        } else {
            thread_reduce_max<false, Operator, 1, DataType0, DataType1, WARP_M, WARP_N>(tensor, summary, op, summary_cur);
            quad_allreduce_<Operator, DataType1, WARP_M>(summary_cur, summary_cur, op);
        }
    }
}

// zero_init==true, max is current max_score, max_cur=nullptr
// zero_init==true, max is prev max_score, max_cur!=nullptr
template<bool zero_init=true, typename DataType0, typename DataType1, int WARP_M, int WARP_N>
__device__ inline void reduce_max(const DataType0 tensor[(WARP_M / 32) * (WARP_N / 32)][4], DataType1 *max , DataType1 *max_cur=nullptr) {
    MaxOp<float> max_op;
    if constexpr (zero_init == true) {
        reduce_<true, MaxOp<float>, 1, DataType0, DataType1, WARP_M, WARP_N>(tensor, max, max_op);
    } else {
        reduce_<false, MaxOp<float>, 1, DataType0, DataType1, WARP_M, WARP_N>(tensor, max, max_op, max_cur);
    }
}

template<bool zero_init=true, typename DataType0, typename DataType1, int WARP_M, int WARP_N>
__device__ inline void reduce_sum(DataType0 tensor[(WARP_M / 32) * (WARP_N / 32)][4], DataType1 *sum,  DataType1 *sum_cur=nullptr){
    SumOp<float> sum_op;
    if constexpr (zero_init == true) {
        reduce_<true, SumOp<float>, 0, DataType0, DataType1, WARP_M, WARP_N>(tensor, sum, sum_op);
    } else {
        reduce_<false, SumOp<float>, 0, DataType0, DataType1, WARP_M, WARP_N>(tensor, sum, sum_op, sum_cur);
    }
}



// Apply the exp to all the elements.
template <bool Scale_max=true, typename DataType0, typename DataType1, int WARP_M, int WARP_N>
inline __device__ void scale_apply_exp2(DataType0 tensor[(WARP_M / 32) * (WARP_N / 32)][4], const DataType1 *max, const float scale) {
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / 32); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
            const float max_scaled = (max[mi * 2].f32[min_tile_m] == -INFINITY) ? 0.f : (max[mi * 2].f32[min_tile_m] * (Scale_max ? scale : float(M_LOG2E)));
            __float2 neg_max_scaled_pair = {-max_scaled, -max_scaled};
            __float2 scale_pair = {scale, scale};
            #pragma unroll
            for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
                // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                // max * log_2(e)) This allows the compiler to use the ffma
                // instruction instead of fadd and fmul separately.
                // min tile is 32*32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    int mmac_id    = min_tile_n * 2 + min_tile_m;
                    int qk_tile_id = mi + ni * (WARP_M / 32);
                #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
                    for(int vec_idx = 0; vec_idx < 2; ++vec_idx) {
                        tensor[qk_tile_id][mmac_id].u64[vec_idx] = __builtin_hcu_pk_fma_f32(
                            tensor[qk_tile_id][mmac_id].u64[vec_idx],
                            scale_pair,
                            neg_max_scaled_pair
                        );
                    }
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        tensor[qk_tile_id][mmac_id].f32[vec_idx] = __llvm_exp2_f32(tensor[qk_tile_id][mmac_id].f32[vec_idx]);
                    }
                #else
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        tensor[qk_tile_id][mmac_id].f32[vec_idx] = __llvm_exp2_f32(tensor[qk_tile_id][mmac_id].f32[vec_idx] * scale - max_scaled);
                    }
                #endif
                }
            }
        }
    }
}



template<bool Is_first, bool Check_inf, typename DataType0, typename DataType1, int K/*head_dim*/, int kBlockK, int WARP_M, int WARP_N, bool IsInference=true>
inline __device__ void softmax_rescale_o(DataType0 scores[(WARP_N / 32) * (WARP_M / 32)][4], DataType1 *scores_max, DataType1 *scores_sum,
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
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                float scores_max_cur_reg = !Check_inf
                        ? scores_max_cur[mi * 2].f32[min_tile_m]
                        : (scores_max_cur[mi * 2].f32[min_tile_m] == -INFINITY ? 0.0f : scores_max_cur[mi * 2].f32[min_tile_m]);

                // optimization from flash-attention-4
                if (IsInference or scores_max[mi * 2].f32[min_tile_m] < scores_max_cur_reg) {
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
                            #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
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
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
            scores_sum[mi].u64 = __builtin_hcu_pk_add_f32(
                scores_sum[mi].u64,
                scores_sum_cur[mi].u64
            );
        #else
            scores_sum[mi].f32[0] += scores_sum_cur[mi].f32[0];
            scores_sum[mi].f32[1] += scores_sum_cur[mi].f32[1];
        #endif
            scores_max[mi].f32[0] = scores_max_cur[mi].f32[0];
            scores_max[mi].f32[1] = scores_max_cur[mi].f32[1];
        }
    }
};




template <int WARP_M, int WARP_N, typename Element, typename ElementAccum, bool IsInference=false>
inline __device__ void convert_pk_type(union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (WARP_N / 32)][4], union_vec4_fp32 s_reg[(WARP_M / 32) * (WARP_N / 32)][4]) {
    #pragma unroll
    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
        #pragma unroll
        for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m < 2; ++min_tile_m) {
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                        p_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f16x2[min_tile_k] = DownCastPair<float, Element>(
                        s_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f32x2[min_tile_k]);
                        p_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f16x2[min_tile_k] = DownCastPair<float, Element>(
                        s_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f32x2[min_tile_k]);
                    #else
                        if constexpr (IsInference) {
                            p_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f16x2[min_tile_k] = DownCastPairNoPack<float, Element>(
                                s_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f32[min_tile_k * 2 + 0],
                                s_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f32[min_tile_k * 2 + 1]
                            );
                            p_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f16x2[min_tile_k] = DownCastPairNoPack<float, Element>(
                                s_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f32[min_tile_k * 2 + 0],
                                s_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f32[min_tile_k * 2 + 1]
                            );
                        } else {
                            // For training, higher precision is needed
                            p_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0] = DownCast<float, Element, false>(
                            s_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f32[min_tile_k * 2 + 0]);
                            p_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0] = DownCast<float, Element, false>(
                            s_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f32[min_tile_k * 2 + 0]);
                            p_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1] = DownCast<float, Element, false>(
                            s_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f32[min_tile_k * 2 + 1]);
                            p_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1] = DownCast<float, Element, false>(
                            s_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f32[min_tile_k * 2 + 1]);
                        }
                    #endif
                }
            }
        }
    }
}

namespace flash {
template<int kBlockN, int WARP_M, typename ElementAccum>
__forceinline__ __device__ void prefix_prefill_hdim512_16x64_apply_mask(
        vec4_Accum<ElementAccum> s_reg[kBlockN / 16],
        const int lane_id,
        const int col_idx_offset,
        const int max_seqlen_k) {
    static_assert(kBlockN == 64);
    static_assert(WARP_M == 16);
    #pragma unroll
    for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
        #pragma unroll
        for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
            const int col_idx = col_idx_offset + n_loop * 16 + vec_idx * 4 + (lane_id >> 4);
            s_reg[n_loop].f32[vec_idx] = (col_idx >= max_seqlen_k) ? -INFINITY : s_reg[n_loop].f32[vec_idx];
        }
    }
}

template<int kBlockN, int WARP_M, typename ElementAccum>
__forceinline__ __device__ void prefix_prefill_hdim512_16x64_apply_causal_mask(
        vec4_Accum<ElementAccum> s_reg[kBlockN / 16],
        const int lane_id,
        const int col_idx_offset,
        const int max_seqlen_k,
        const int row_idx_offset,
        const int max_seqlen_q) {
    static_assert(kBlockN == 64);
    static_assert(WARP_M == 16);
    const int row_idx = row_idx_offset + (lane_id & 15);
    const int col_idx_limit_right = min(max_seqlen_k, row_idx + max_seqlen_k - max_seqlen_q);
    #pragma unroll
    for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
        #pragma unroll
        for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
            const int col_idx = col_idx_offset + n_loop * 16 + vec_idx * 4 + (lane_id >> 4);
            s_reg[n_loop].f32[vec_idx] = (col_idx > col_idx_limit_right) ? -INFINITY : s_reg[n_loop].f32[vec_idx];
        }
    }
}

template<bool HasWSLeft, int kBlockN, int WARP_M, typename ElementAccum>
__forceinline__ __device__ void prefix_prefill_hdim512_16x64_apply_local_mask(
        vec4_Accum<ElementAccum> s_reg[kBlockN / 16],
        const int lane_id,
        const int col_idx_offset,
        const int max_seqlen_k,
        const int row_idx_offset,
        const int max_seqlen_q,
        const int window_size_left,
        const int window_size_right) {
    static_assert(kBlockN == 64);
    static_assert(WARP_M == 16);
    const int row_idx = row_idx_offset + (lane_id & 15);
    const int col_idx_limit_left = max(0, row_idx + 1 + max_seqlen_k - max_seqlen_q - window_size_left);
    const int col_idx_limit_right = min(max_seqlen_k, row_idx + max_seqlen_k - max_seqlen_q + window_size_right);
    #pragma unroll
    for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
        #pragma unroll
        for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
            const int col_idx = col_idx_offset + n_loop * 16 + vec_idx * 4 + (lane_id >> 4);
            const bool masked = col_idx > col_idx_limit_right || (HasWSLeft && col_idx < (col_idx_limit_left - 1));
            s_reg[n_loop].f32[vec_idx] = masked ? -INFINITY : s_reg[n_loop].f32[vec_idx];
        }
    }
}

template<int kBlockN, int WARP_M, typename ElementAccum>
__forceinline__ __device__ void prefix_prefill_hdim512_16x64_apply_alibi(
        vec4_Accum<ElementAccum> s_reg[kBlockN / 16],
        const int lane_id,
        const int col_idx_offset,
        const int row_idx_offset,
        const float alibi) {
    static_assert(kBlockN == 64);
    static_assert(WARP_M == 16);
    const int row_idx = row_idx_offset + (lane_id & 15);
    #pragma unroll
    for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
        #pragma unroll
        for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
            const int col_idx = col_idx_offset + n_loop * 16 + vec_idx * 4 + (lane_id >> 4);
            s_reg[n_loop].f32[vec_idx] += alibi * (col_idx - row_idx);
        }
    }
}

template<int kBlockN, int kHeadDimV, typename ElementAccum>
__forceinline__ __device__ void prefix_prefill_hdim512_16x64_softmax_rescale_o(
        vec4_Accum<ElementAccum> s_reg[kBlockN / 16],
        ElementAccum &scores_max,
        ElementAccum &scores_sum,
        vec4_Accum<ElementAccum> acc_o[kHeadDimV / 16],
        const ElementAccum scale_softmax_log2) {
    ElementAccum scores_max_cur = scores_max;
    #pragma unroll
    for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
        #pragma unroll
        for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
            scores_max_cur = max(scores_max_cur, s_reg[n_loop].f32[vec_idx]);
        }
    }
    scores_max_cur = max(scores_max_cur, __shfl_xor_tmp(scores_max_cur, 32));
    scores_max_cur = max(scores_max_cur, __shfl_xor_tmp(scores_max_cur, 16));

    const ElementAccum max_scaled = scores_max_cur == -INFINITY ? 0.f : -scores_max_cur * scale_softmax_log2;
    __float2 scale_softmax_log2_pair = {scale_softmax_log2, scale_softmax_log2};
    __float2 max_scaled_pair = {max_scaled, max_scaled};
    #pragma unroll
    for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
        s_reg[n_loop].u64[0] = __builtin_hcu_pk_fma_f32(s_reg[n_loop].u64[0], scale_softmax_log2_pair, max_scaled_pair);
        s_reg[n_loop].u64[1] = __builtin_hcu_pk_fma_f32(s_reg[n_loop].u64[1], scale_softmax_log2_pair, max_scaled_pair);
        s_reg[n_loop].f32[0] = __llvm_exp2_f32(s_reg[n_loop].f32[0]);
        s_reg[n_loop].f32[1] = __llvm_exp2_f32(s_reg[n_loop].f32[1]);
        s_reg[n_loop].f32[2] = __llvm_exp2_f32(s_reg[n_loop].f32[2]);
        s_reg[n_loop].f32[3] = __llvm_exp2_f32(s_reg[n_loop].f32[3]);
    }

    __float2 sum_pair = {0.f, 0.f};
    #pragma unroll
    for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
        sum_pair = __builtin_hcu_pk_add_f32(sum_pair, s_reg[n_loop].u64[0]);
        sum_pair = __builtin_hcu_pk_add_f32(sum_pair, s_reg[n_loop].u64[1]);
    }
    ElementAccum scores_sum_cur = sum_pair[0] + sum_pair[1];
    scores_sum_cur += __shfl_xor(scores_sum_cur, 32);
    scores_sum_cur += __shfl_xor(scores_sum_cur, 16);

    const ElementAccum old_rescale = (scores_max_cur == -INFINITY)
        ? 1.f
        : __llvm_exp2_f32(
            __llvm_fma_f32(scores_max, scale_softmax_log2, -scores_max_cur * scale_softmax_log2));
    __float2 old_rescale_pair = {old_rescale, old_rescale};
    scores_sum *= old_rescale;
    #pragma unroll
    for (int i = 0; i < kHeadDimV / 16; ++i) {
        acc_o[i].u64[0] = __builtin_hcu_pk_mul_f32(acc_o[i].u64[0], old_rescale_pair);
        acc_o[i].u64[1] = __builtin_hcu_pk_mul_f32(acc_o[i].u64[1], old_rescale_pair);
    }
    scores_sum += scores_sum_cur;
    scores_max = scores_max_cur;
}

template<int kBlockN, typename Element, typename ElementAccum>
__forceinline__ __device__ void prefix_prefill_hdim512_16x64_convert_p(
        union_vec2_f16x2<Element> p_reg[kBlockN / 16],
        vec4_Accum<ElementAccum> s_reg[kBlockN / 16]) {
    #pragma unroll
    for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
        p_reg[n_loop].f16x2[0] = DownCastPairNoPack<ElementAccum, Element>(
            s_reg[n_loop].f32[0], s_reg[n_loop].f32[1]);
        p_reg[n_loop].f16x2[1] = DownCastPairNoPack<ElementAccum, Element>(
            s_reg[n_loop].f32[2], s_reg[n_loop].f32[3]);
    }
}

} // namespace flash
