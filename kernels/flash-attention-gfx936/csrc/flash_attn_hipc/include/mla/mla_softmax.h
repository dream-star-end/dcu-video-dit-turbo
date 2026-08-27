#pragma once

#include "philox.cuh"
#include "fwd/utils.h"

using namespace flash;

template <typename DataType, int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT>
inline __device__ void mla_apply_mask(DataType tensor[M_WARP_COUNT * N_WARP_COUNT][4], const int max_seqlen_k,
                                  const int col_idx_offset_ = 0) {
    const int lane_id = threadIdx.x & 63; // lane id, 0-63
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 2;
    #pragma unroll
    for (int ni = 0; ni < N_WARP_COUNT; ++ni) {
        #pragma unroll
        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n;
            #pragma unroll
            for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                const int col_idx = col_idx_base + vec_idx * 8;
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
inline __device__ void mla_apply_mask_causal(DataType tensor[M_WARP_COUNT * N_WARP_COUNT][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q, const int mtp, const int layout) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15) * 2;
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m;
            int col_idx_limit_right;
            if constexpr (REUSE_KV_TIMES == 0) {
                col_idx_limit_right = std::min(max_seqlen_k, row_idx + max_seqlen_k - max_seqlen_q);
            } else {
                const int row_in_mtp = layout == 0 ? (row_idx % mtp): (row_idx / MTP_REGROUP_COUNT);
                col_idx_limit_right = std::min(max_seqlen_k, row_in_mtp + max_seqlen_k - mtp);
            }
            #pragma unroll
            for (int ni = 0; ni < N_WARP_COUNT; ++ni) {
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n;
                    #pragma unroll
                    for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx * 8;
                        tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx] = (col_idx > col_idx_limit_right) ? -INFINITY: tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                    }
                }
            }
        }
    }
}



template<bool zero_init=true, typename Operator, int OpType, typename DataType0, typename DataType1, int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT>
__device__ inline void mla_thread_reduce_max(const DataType0 tensor[M_WARP_COUNT * N_WARP_COUNT][4], DataType1 *summary, Operator &op, DataType1 *summary_cur=nullptr) {
    if(zero_init == true) {
        #pragma unroll
        for (int m_idx = 0; m_idx < M_WARP_COUNT; ++m_idx) {
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                summary[m_idx * 2].f32[min_tile_m] = -INFINITY;
                #pragma unroll
                for (int n_idx = 0; n_idx < N_WARP_COUNT; ++n_idx) {
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                            summary[m_idx * 2].f32[min_tile_m] = op(summary[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                        }
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for (int m_idx = 0; m_idx < M_WARP_COUNT; ++m_idx) {
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                summary_cur[m_idx * 2].f32[min_tile_m] = summary[m_idx * 2].f32[min_tile_m];
                #pragma unroll
                for (int n_idx = 0; n_idx < N_WARP_COUNT; ++n_idx) {
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                            summary_cur[m_idx * 2].f32[min_tile_m] = op(summary_cur[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                        }
                    }
                }
            }
        }
    }
}



template<bool zero_init=true, typename Operator, int OpType, typename DataType0, typename DataType1, int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT>
__device__ inline void mla_thread_reduce_sum(const DataType0 tensor[M_WARP_COUNT * N_WARP_COUNT][4], DataType1 *summary, Operator &op, DataType1 *summary_cur=nullptr) {
    if(zero_init == true) {
        #pragma unroll
        for (int m_idx = 0; m_idx < M_WARP_COUNT; ++m_idx) {
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
            summary[m_idx * 2].u64 = 0x0;
            #pragma unroll
            for (int n_idx = 0; n_idx < N_WARP_COUNT; ++n_idx) {
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        __float2 additem_pair = {tensor[m_idx + n_idx * M_WARP_COUNT][min_tile_n * 2].f32[vec_idx], tensor[m_idx + n_idx * M_WARP_COUNT][min_tile_n * 2 + 1].f32[vec_idx]};
                        summary[m_idx * 2].u64 = __builtin_hcu_pk_add_f32(
                            summary[m_idx * 2].u64,
                            additem_pair
                        );
                    }
                }
            }
        #else
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                summary[m_idx * 2].f32[min_tile_m] = 0;
                #pragma unroll
                for (int n_idx = 0; n_idx < N_WARP_COUNT; ++n_idx) {
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                            summary[m_idx * 2].f32[min_tile_m] = op(summary[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                        }
                    }
                }
            }
        #endif
        }
    } else {
        #pragma unroll
        for (int m_idx = 0; m_idx < M_WARP_COUNT; ++m_idx) {
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
            summary_cur[m_idx * 2].u64 = summary[m_idx * 2].u64;
            #pragma unroll
            for (int n_idx = 0; n_idx < N_WARP_COUNT; ++n_idx) {
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        __float2 additem_pair = {tensor[m_idx + n_idx * M_WARP_COUNT][min_tile_n * 2].f32[vec_idx], tensor[m_idx + n_idx * M_WARP_COUNT][min_tile_n * 2 + 1].f32[vec_idx]};
                        summary_cur[m_idx * 2].u64 = __builtin_hcu_pk_add_f32(
                            summary_cur[m_idx * 2].u64,
                            additem_pair
                        );
                    }
                }
            }
        #else
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                summary_cur[m_idx * 2].f32[min_tile_m] = summary[m_idx * 2].f32[min_tile_m];
                #pragma unroll
                for (int n_idx = 0; n_idx < N_WARP_COUNT; ++n_idx) {
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                            summary_cur[m_idx * 2].f32[min_tile_m] = op(summary_cur[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                        }
                    }
                }
            }
        #endif
        }
    }
}


template<typename Operator, typename DataType, int M_WARP_COUNT>
__device__ inline void mla_quad_allreduce_(DataType *dst, DataType *src, Operator &op) {
    #pragma unroll
    for (int mi = 0; mi < M_WARP_COUNT; mi++) {
        dst[mi] = Allreduce<64>::run(src[mi], op);
    }
}

template<bool zero_init=true, typename Operator, int OpType, typename DataType0, typename DataType1, int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT>
__device__ inline void mla_reduce_(const DataType0 tensor[M_WARP_COUNT * N_WARP_COUNT][4], DataType1 *summary, Operator &op, DataType1 *summary_cur=nullptr) {
    if constexpr (OpType == 0) { // sum
        if constexpr (zero_init == true) {
            mla_thread_reduce_sum<true, Operator, 0, DataType0, DataType1, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(tensor, summary, op);
            mla_quad_allreduce_<Operator, DataType1, M_WARP_COUNT>(summary, summary, op);
        } else {
            mla_thread_reduce_sum<false, Operator, 0, DataType0, DataType1, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(tensor, summary, op, summary_cur);
            mla_quad_allreduce_<Operator, DataType1, M_WARP_COUNT>(summary_cur, summary_cur, op);
        }
    } else if constexpr (OpType == 1) { // max
        if constexpr (zero_init == true) {
            mla_thread_reduce_max<true, Operator, 1, DataType0, DataType1, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(tensor, summary, op);
            mla_quad_allreduce_<Operator, DataType1, M_WARP_COUNT>(summary, summary, op);
        } else {
            mla_thread_reduce_max<false, Operator, 1, DataType0, DataType1, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(tensor, summary, op, summary_cur);
            mla_quad_allreduce_<Operator, DataType1, M_WARP_COUNT>(summary_cur, summary_cur, op);
        }
    }
}


template<bool zero_init=true, typename DataType0, typename DataType1, int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT>
__device__ inline void mla_reduce_max(const DataType0 tensor[M_WARP_COUNT * N_WARP_COUNT][4], DataType1 *max , DataType1 *max_cur=nullptr) {
    MaxOp<float> max_op;
    if constexpr (zero_init == true) {
        mla_reduce_<true, MaxOp<float>, 1, DataType0, DataType1, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(tensor, max, max_op);
    } else {
        mla_reduce_<false, MaxOp<float>, 1, DataType0, DataType1, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(tensor, max, max_op, max_cur);
    }
}

template<bool zero_init=true, typename DataType0, typename DataType1, int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT>
__device__ inline void mla_reduce_sum(DataType0 tensor[M_WARP_COUNT * N_WARP_COUNT][4], DataType1 *sum,  DataType1 *sum_cur=nullptr) {
    SumOp<float> sum_op;
    if constexpr (zero_init == true) {
        mla_reduce_<true, SumOp<float>, 0, DataType0, DataType1, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(tensor, sum, sum_op);
    } else {
        mla_reduce_<false, SumOp<float>, 0, DataType0, DataType1, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(tensor, sum, sum_op, sum_cur);
    }
}



// Apply the exp to all the elements.
template <bool Scale_max=true, typename DataType0, typename DataType1, int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT>
inline __device__ void mla_scale_apply_exp2(DataType0 tensor[M_WARP_COUNT * N_WARP_COUNT][4], const DataType1 *max, const float scale) {
    #pragma unroll
    for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            const float max_scaled = (max[mi].f32[min_tile_m] == -INFINITY) ? 0.f : (max[mi].f32[min_tile_m] * (Scale_max ? scale : float(M_LOG2E)));
            __float2 neg_max_scaled_pair = {-max_scaled, -max_scaled};
            __float2 scale_pair = {scale, scale};
            #pragma unroll
            for (int ni = 0; ni < N_WARP_COUNT; ++ni) {
                // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                // max * log_2(e)) This allows the compiler to use the ffma
                // instruction instead of fadd and fmul separately.
                // min tile is 32*32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                    for (int vec_idx = 0; vec_idx < 2; vec_idx++) {
                        tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].u64[vec_idx] = __builtin_hcu_pk_fma_f32(
                            tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].u64[vec_idx],
                            scale_pair,
                            neg_max_scaled_pair
                        );
                    }
                    for (int vec_idx = 0; vec_idx < 4; vec_idx++) {
                        tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx] = __llvm_exp2_f32(tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                    }
                #else
                    for (int vec_idx = 0; vec_idx < 4; vec_idx++) {
                        tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx] = __llvm_exp2_f32(tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx] * scale - max_scaled);
                    }
                #endif
                }
            }
        }
    }
}



template<bool Check_inf=false, typename softmaxType, int K_LOOP_COUNT, int K_WARP_COUNT, int M_WARP_COUNT, int N_WARP_COUNT, int WARP_NUM, int M_MMAC_COUNT>
inline __device__ void mla_softmax_rescale_o(
        vec4_Accum<softmaxType> scores[N_WARP_COUNT * M_WARP_COUNT][4],
        vec2_Accum<softmaxType> *scores_max,
        vec2_Accum<softmaxType> *scores_sum,
        vec4_Accum<softmaxType> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4],
        softmaxType* max_lds,
        int warp_id,
        float softmax_scale_log2) {

    static_assert (std::is_same<softmaxType, float>::value and "For softmax after QK gemm, only float32 is supported!");

    // 求当前 32x32 的最大值, 以及和前面计算得到的最大值
    vec2_Accum<softmaxType> scores_max_cur[M_WARP_COUNT];
    mla_reduce_max</*zero_init=*/false, vec4_Accum<softmaxType>, vec2_Accum<softmaxType>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(scores, scores_max, scores_max_cur); // scores_max is prev scores max

    int lane_id = threadIdx.x & 63;
    constexpr int WARP_M = M_WARP_COUNT * 32;
    if constexpr (WARP_NUM > 1) {
        static_assert (M_WARP_COUNT == 1);
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

    for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            float scores_max_cur_reg = !Check_inf
                    ? scores_max_cur[mi].f32[min_tile_m]
                    : (scores_max_cur[mi].f32[min_tile_m] == -INFINITY ? 0.0f : scores_max_cur[mi].f32[min_tile_m]);

            float scores_scale = __llvm_exp2_f32((scores_max[mi].f32[min_tile_m] - scores_max_cur_reg) * softmax_scale_log2);
            scores_sum[mi].f32[min_tile_m] *= scores_scale;

            __float2 scores_scale_pair = {scores_scale, scores_scale};

            #pragma unroll
            for (int pv_n_loop = 0; pv_n_loop < K_LOOP_COUNT; ++pv_n_loop) {
                #pragma unroll
                for (int ni = 0; ni < K_WARP_COUNT; ++ni) {
                    // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                    // max * log_2(e)) This allows the compiler to use the ffma
                    // instruction instead of fadd and fmul separately.
                    // min tile is 32*32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
                    int loop_id = (pv_n_loop * K_WARP_COUNT + ni) * M_WARP_COUNT + mi;
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        // 936 及之后的架构有 pk_mul 指令
                    #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                        #pragma unroll
                        for (int vec_idx = 0; vec_idx < 2; vec_idx++) {
                            acc_o[loop_id][min_tile_n * 2 + min_tile_m].u64[vec_idx] = __builtin_hcu_pk_mul_f32(
                                acc_o[loop_id][min_tile_n * 2 + min_tile_m].u64[vec_idx],
                                scores_scale_pair
                            );
                        }
                    #else
                        #pragma unroll
                        for (int vec_idx = 0; vec_idx < 4; vec_idx++) {
                            acc_o[loop_id][min_tile_n * 2 + min_tile_m].f32[vec_idx] *= scores_scale;
                        }
                    #endif
                    }
                }
            }
        }
    }
    mla_scale_apply_exp2</*zero_init=*/true, vec4_Accum<softmaxType>, vec2_Accum<softmaxType>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(scores, scores_max_cur, softmax_scale_log2);

    vec2_Accum<softmaxType> scores_sum_cur[M_WARP_COUNT];
    for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
        scores_sum_cur[mi].u64 = 0x0;
    }
    mla_reduce_sum</*zero_init=*/true, vec4_Accum<softmaxType>, vec2_Accum<softmaxType>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(scores, scores_sum_cur);

    if constexpr (WARP_NUM > 1) {
        // sum 无法用 ds_atomic_add_f32, 因为 non-desterminstic
        softmaxType* sum_lds = max_lds + 64;
        if(lane_id < 16) {
            // 每个 wave 的归一化和写到 lds
            for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
                if constexpr (M_MMAC_COUNT == 1) {
                    sum_lds[warp_id * WARP_M + mi * 32 + lane_id * 2] = scores_sum_cur[mi].f32[0];
                } else {
                    *(__float2*)(sum_lds + warp_id * WARP_M + mi * 32 + lane_id * 2) = scores_sum_cur[mi].u64;
                }
            }
            __syncthreads();
            // 0 号 wave reduce 其他 wave 的归一化和
            if (warp_id == 0) {
                for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
                    if constexpr (M_MMAC_COUNT == 1) {
                        float tmp = sum_lds[mi * 32 + lane_id * 2];
                        for (int warp_loop = 1; warp_loop < WARP_NUM; ++warp_loop) {
                            tmp += sum_lds[warp_loop * WARP_M + mi * 32 + lane_id * 2];
                        }
                        sum_lds[mi * 32 + lane_id * 2] = tmp;
                    } else {
                        __float2 cur_wave_sum = *(__float2*)(sum_lds + mi * 32 + lane_id * 2);
                        #pragma unroll
                        for (int warp_loop = 1; warp_loop < WARP_NUM; ++warp_loop) {
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
        for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
            if constexpr (M_MMAC_COUNT == 1) {
                scores_sum_cur[mi].f32[0] = sum_lds[mi * 32 + (lane_id & 15) * 2];
            } else {
                scores_sum_cur[mi].u64 = *(__float2*)(sum_lds + mi * 32 + (lane_id & 15) * 2);
            }
        }
    }

    for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
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
};



template <int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT, typename Element, typename ElementAccum>
inline __device__ void mla_convert_pk_type(union_vec2_f16x2<Element> p_reg[M_WARP_COUNT * N_WARP_COUNT][4], union_vec4_fp32 s_reg[M_WARP_COUNT * N_WARP_COUNT][4]) {
    #pragma unroll
    for (int n_idx = 0; n_idx < N_WARP_COUNT; ++n_idx) {
        #pragma unroll
        for (int m_idx = 0; m_idx < M_WARP_COUNT; ++m_idx) {
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                #pragma unroll
                for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__) || defined(__gfx92a__)
                        p_reg[n_idx * M_WARP_COUNT + m_idx][0 * 2 + min_tile_m].f16x2[min_tile_k] = DownCastPair<float, Element>(
                        s_reg[n_idx * M_WARP_COUNT + m_idx][0 * 2 + min_tile_m].f32x2[min_tile_k]);
                        p_reg[n_idx * M_WARP_COUNT + m_idx][1 * 2 + min_tile_m].f16x2[min_tile_k] = DownCastPair<float, Element>(
                        s_reg[n_idx * M_WARP_COUNT + m_idx][1 * 2 + min_tile_m].f32x2[min_tile_k]);
                    #else
                        p_reg[n_idx * M_WARP_COUNT + m_idx][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 0] = DownCast<float, Element, false>(
                        s_reg[n_idx * M_WARP_COUNT + m_idx][0 * 2 + min_tile_m].f32[min_tile_k * 2 + 0]);
                        p_reg[n_idx * M_WARP_COUNT + m_idx][0 * 2 + min_tile_m].f16[min_tile_k * 2 + 1] = DownCast<float, Element, false>(
                        s_reg[n_idx * M_WARP_COUNT + m_idx][0 * 2 + min_tile_m].f32[min_tile_k * 2 + 1]);
                        p_reg[n_idx * M_WARP_COUNT + m_idx][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 0] = DownCast<float, Element, false>(
                        s_reg[n_idx * M_WARP_COUNT + m_idx][1 * 2 + min_tile_m].f32[min_tile_k * 2 + 0]);
                        p_reg[n_idx * M_WARP_COUNT + m_idx][1 * 2 + min_tile_m].f16[min_tile_k * 2 + 1] = DownCast<float, Element, false>(
                        s_reg[n_idx * M_WARP_COUNT + m_idx][1 * 2 + min_tile_m].f32[min_tile_k * 2 + 1]);
                    #endif
                }
            }
        }
    }
}
