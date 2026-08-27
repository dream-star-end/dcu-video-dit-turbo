#pragma once

#include "philox.cuh"
#include "fwd/utils.h"

using namespace flash;

template<int THREADS, typename DataType=union_vec2_fp32>
struct PrefillMlaAllreduce {
    static_assert(THREADS == 64);
    template<typename Operator>
    static __device__ inline DataType run(DataType x, Operator &op) {
        DataType res;
        if constexpr (std::is_same<DataType, union_vec2_fp32>::value) {
            if constexpr (std::is_same<Operator, SumOp<float> >::value) {
            #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                res.f32[0] = __shfl_xor_tmp(x.f32[0], 32);
                res.f32[1] = __shfl_xor_tmp(x.f32[1], 32);
                x.u64 = __builtin_hcu_pk_add_f32(x.u64, res.u64);
                res.f32[0] = __shfl_xor_tmp(x.f32[0], 16);
                res.f32[1] = __shfl_xor_tmp(x.f32[1], 16);
                res.u64 = __builtin_hcu_pk_add_f32(res.u64, x.u64);
            #else
                x.f32[0] = x.f32[0] + __shfl_xor_tmp(x.f32[0], 32);
                x.f32[1] = x.f32[1] + __shfl_xor_tmp(x.f32[1], 32);
                res.f32[0] = x.f32[0] + __shfl_xor_tmp(x.f32[0], 16);
                res.f32[1] = x.f32[1] + __shfl_xor_tmp(x.f32[1], 16);
            #endif
            }
            else if constexpr (std::is_same<Operator, MaxOp<float> >::value) {
                x.f32[0] = op(x.f32[0], __shfl_xor_tmp(x.f32[0], 32));
                x.f32[1] = op(x.f32[1], __shfl_xor_tmp(x.f32[1], 32));
                res.f32[0] = op(x.f32[0], __shfl_xor_tmp(x.f32[0], 16));
                res.f32[1] = op(x.f32[1], __shfl_xor_tmp(x.f32[1], 16));
            }
        } else { // union_vec_fp32 f32
            if constexpr (std::is_same<Operator, SumOp<float> >::value) {
                x.f32[0] = x.f32[0] + __shfl_xor_tmp(x.f32[0], 32);
                res.f32[0] = x.f32[0] + __shfl_xor_tmp(x.f32[0], 16);
            }
            else if constexpr (std::is_same<Operator, MaxOp<float> >::value) {
                x.f32[0] = op(x.f32[0], __shfl_xor_tmp(x.f32[0], 32));
                res.f32[0] = op(x.f32[0], __shfl_xor_tmp(x.f32[0], 16));
            }
        }
        return res;
    }
};

template<bool zero_init=true, typename Operator, int OpType, typename DataType0, typename DataType1, int WARP_M, int WARP_N, int M_MMAC_COUNT=2>
__device__ inline void prefill_mla_thread_reduce_max(const DataType0 tensor[(WARP_M / (16 * M_MMAC_COUNT)) * (WARP_N / 32)][2 * M_MMAC_COUNT], DataType1 *summary, Operator &op, DataType1 *summary_cur=nullptr) {
    if(zero_init == true) {
        #pragma unroll
        for(int m_idx = 0; m_idx < (WARP_M / (16 * M_MMAC_COUNT)); ++m_idx) {
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                summary[m_idx * 2].f32[min_tile_m] = -INFINITY;  // OpType：0 is sum operator, 1 is max operator
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        for(int vec_idx = 0; vec_idx < 4; ++vec_idx) { // mmac min_tile is 16*16, a warp is 64 thread
                            if constexpr (M_MMAC_COUNT == 2)
                                summary[m_idx * 2].f32[min_tile_m] = op(summary[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                            else
                                summary[m_idx * 2].f32[min_tile_m] = op(summary[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * (WARP_M / 16)][min_tile_n].f32[vec_idx]);
                        }
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for(int m_idx = 0; m_idx < (WARP_M / (16 * M_MMAC_COUNT)); ++m_idx) {
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                summary_cur[m_idx * 2].f32[min_tile_m] = summary[m_idx * 2].f32[min_tile_m];
                #pragma unroll
                for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                    #pragma unroll
                    for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        for(int vec_idx = 0; vec_idx < 4; ++vec_idx) { // mmac min_tile is 16*16, a warp is 64 thread
                            if constexpr (M_MMAC_COUNT == 2)
                                summary_cur[m_idx * 2].f32[min_tile_m] = op(summary_cur[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                            else
                                summary_cur[m_idx * 2].f32[min_tile_m] = op(summary_cur[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * (WARP_M / 16)][min_tile_n].f32[vec_idx]);
                        }
                    }
                }
            }
        }
    }
}



template<bool zero_init=true, typename Operator, int OpType, typename DataType0, typename DataType1, int WARP_M, int WARP_N, int M_MMAC_COUNT=2>
__device__ inline void prefill_mla_thread_reduce_sum(const DataType0 tensor[(WARP_M / (16 * M_MMAC_COUNT)) * (WARP_N / 32)][2 * M_MMAC_COUNT], DataType1 *summary, Operator &op, DataType1 *summary_cur=nullptr) {
    if(zero_init == true) {
        #pragma unroll
        for(int m_idx = 0; m_idx < (WARP_M / (16 * M_MMAC_COUNT)); ++m_idx) {
        // 对于 gfx936 及以上的架构, 可以使用 v_pk_add_f32
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
            if constexpr (M_MMAC_COUNT == 2) {
                summary[m_idx * 2].u64    = 0x0;
            } else {
                summary[m_idx * 2].f32[0] = 0x0;
            } 
            #pragma unroll
            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        if constexpr (M_MMAC_COUNT == 2){
                            __float2 additem_pair = {tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2].f32[vec_idx], tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2 + 1].f32[vec_idx]};
                            summary[m_idx * 2].u64 = __builtin_hcu_pk_add_f32(
                                summary[m_idx * 2].u64,
                                additem_pair
                            );
                        } else {
                            summary[m_idx * 2].f32[0] = summary[m_idx * 2].f32[0] + tensor[m_idx + n_idx * (WARP_M / 16)][min_tile_n].f32[vec_idx];
                        }
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
                        for(int vec_idx = 0; vec_idx < 4; ++vec_idx) { // mmac min_tile is 16*16, a warp is 64 thread
                            if constexpr (M_MMAC_COUNT == 2) {
                                summary[m_idx * 2].f32[min_tile_m] = op(summary[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                            } else {
                                summary[m_idx * 2].f32[min_tile_m] = op(summary[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * (WARP_M / 16)][min_tile_n].f32[vec_idx]);
                            }
                        }
                    }
                }
            }
        #endif
        }
    } else {
        #pragma unroll
        for(int m_idx = 0; m_idx < (WARP_M / (16 * M_MMAC_COUNT)); ++m_idx) {
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
            if constexpr (M_MMAC_COUNT == 2) {
                summary_cur[m_idx * 2].u64 = summary[m_idx * 2].u64;
            } else {
                summary_cur[m_idx * 2].f32[0] = summary[m_idx * 2].f32[0];
            }
            #pragma unroll
            for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) { // mmac min_tile is 16*16, a warp is 64 thread
                        if constexpr (M_MMAC_COUNT == 2) {
                            __float2 additem_pair = {tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2].f32[vec_idx], tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2 + 1].f32[vec_idx]};
                            summary_cur[m_idx * 2].u64 = __builtin_hcu_pk_add_f32(
                                summary_cur[m_idx * 2].u64,
                                additem_pair
                            );
                        } else {
                            summary_cur[m_idx * 2].f32[0] = summary_cur[m_idx * 2].f32[0] + tensor[m_idx + n_idx * (WARP_M / 16)][min_tile_n].f32[vec_idx];
                        }
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
                        for(int vec_idx = 0; vec_idx < 4; ++vec_idx) { // mmac min_tile is 16*16, a warp is 64 thread
                            if constexpr (M_MMAC_COUNT == 2) {
                                summary_cur[m_idx * 2].f32[min_tile_m] = op(summary_cur[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx]);
                            } else {
                                summary_cur[m_idx * 2].f32[min_tile_m] = op(summary_cur[m_idx * 2].f32[min_tile_m], tensor[m_idx + n_idx * (WARP_M / 16)][min_tile_n].f32[vec_idx]);
                            }
                        }
                    }
                }
            }
        #endif
        }
    }
}


template<typename Operator, typename DataType, int WARP_M, int M_MMAC_COUNT=2>
__device__ inline void prefill_mla_quad_allreduce_(DataType *dst, DataType *src, Operator &op) {
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / (16 * M_MMAC_COUNT)); mi++) {
        dst[mi] = PrefillMlaAllreduce<64, DataType>::run(src[mi], op);
    }
}

template<bool zero_init=true, typename Operator, int OpType, typename DataType0, typename DataType1, int WARP_M, int WARP_N, int M_MMAC_COUNT=2>
__device__ inline void prefill_mla_reduce_(const DataType0 tensor[(WARP_M / (16 * M_MMAC_COUNT)) * (WARP_N / 32)][2 * M_MMAC_COUNT], DataType1 *summary, Operator &op, DataType1 *summary_cur=nullptr) {
    if constexpr (OpType == 0) { // sum
        if constexpr (zero_init == true) {
            prefill_mla_thread_reduce_sum<true, Operator, 0, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, summary, op);
            prefill_mla_quad_allreduce_<Operator, DataType1, WARP_M, M_MMAC_COUNT>(summary, summary, op);
        } else {
            prefill_mla_thread_reduce_sum<false, Operator, 0, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, summary, op, summary_cur);
            prefill_mla_quad_allreduce_<Operator, DataType1, WARP_M, M_MMAC_COUNT>(summary_cur, summary_cur, op);
        }
    } else if constexpr (OpType == 1) { // max
        if constexpr (zero_init == true) {
            prefill_mla_thread_reduce_max<true, Operator, 1, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, summary, op);
            prefill_mla_quad_allreduce_<Operator, DataType1, WARP_M, M_MMAC_COUNT>(summary, summary, op);
        } else {
            prefill_mla_thread_reduce_max<false, Operator, 1, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, summary, op, summary_cur);
            prefill_mla_quad_allreduce_<Operator, DataType1, WARP_M, M_MMAC_COUNT>(summary_cur, summary_cur, op);
        }
    }
}

// zero_init==true, max is current max_score, max_cur=nullptr
// zero_init==false, max is prev max_score, max_cur!=nullptr
template<bool zero_init=true, typename DataType0, typename DataType1, int WARP_M, int WARP_N, int M_MMAC_COUNT=2>
__device__ inline void reduce_max(const DataType0 tensor[(WARP_M / (16 * M_MMAC_COUNT)) * (WARP_N / 32)][2 * M_MMAC_COUNT], DataType1 *max , DataType1 *max_cur=nullptr) {
    MaxOp<float> max_op;
    if constexpr (zero_init == true) {
        prefill_mla_reduce_<true, MaxOp<float>, 1, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, max, max_op);
    } else {
        prefill_mla_reduce_<false, MaxOp<float>, 1, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, max, max_op, max_cur);
    }
}

template<bool zero_init=true, typename DataType0, typename DataType1, int WARP_M, int WARP_N, int M_MMAC_COUNT=2>
__device__ inline void reduce_sum(DataType0 tensor[(WARP_M / (16 * M_MMAC_COUNT)) * (WARP_N / 32)][2 * M_MMAC_COUNT], DataType1 *sum,  DataType1 *sum_cur=nullptr){
    SumOp<float> sum_op;
    if constexpr (zero_init == true) {
        prefill_mla_reduce_<true, SumOp<float>, 0, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, sum, sum_op);
    } else {
        prefill_mla_reduce_<false, SumOp<float>, 0, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(tensor, sum, sum_op, sum_cur);
    }
}



// Apply the exp to all the elements.
template <bool Scale_max=true, typename DataType0, typename DataType1, int WARP_M, int WARP_N, int M_MMAC_COUNT=2>
inline __device__ void scale_apply_exp2(DataType0 tensor[(WARP_M / (16 * M_MMAC_COUNT)) * (WARP_N / 32)][2 * M_MMAC_COUNT], const DataType1 *max, const float scale) {
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / (16 * M_MMAC_COUNT)); ++mi) {
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
                // min tile is 32*32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    int mmac_id;
                    if constexpr (M_MMAC_COUNT == 2) {
                        mmac_id    = min_tile_n * 2 + min_tile_m;
                    } else {
                        mmac_id    = min_tile_n;
                    }
                    int qk_tile_id = mi + ni * (WARP_M / (16 * M_MMAC_COUNT));
                #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
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



template<bool Is_first, bool Check_inf=false, typename DataType0, typename DataType1, int K/*head_dim_v*/, int kBlockK, int WARP_M, int WARP_N, int M_MMAC_COUNT=2>
inline __device__ void prefill_mla_softmax_rescale_o(DataType0 scores[(WARP_N / 32) * (WARP_M / (16 * M_MMAC_COUNT))][2 * M_MMAC_COUNT], DataType1 *scores_max, DataType1 *scores_sum,
                                         DataType0 acc_o[(K / kBlockK) * (WARP_M / (16 * M_MMAC_COUNT)) * (kBlockK / 32)][2 * M_MMAC_COUNT], float softmax_scale_log2) {
    if constexpr (Is_first) {
        reduce_max</*zero_init=*/true, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(scores, scores_max);
        scale_apply_exp2<true, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(scores, scores_max, softmax_scale_log2);
        reduce_sum<true, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(scores, scores_sum);
    } else {
        DataType1 scores_max_cur[(WARP_M / (16 * M_MMAC_COUNT))];
        reduce_max</*zero_init=*/false, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(scores, scores_max, scores_max_cur); // scores_max is prev scores max

        for (int mi = 0; mi < (WARP_M / (16 * M_MMAC_COUNT)); ++mi) {
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
                for(int pv_n_loop = 0; pv_n_loop < (K / kBlockK); pv_n_loop++)  {
                    #pragma unroll
                    for (int ni = 0; ni < (kBlockK / 32); ++ni)  {
                        // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                        // max * log_2(e)) This allows the compiler to use the ffma
                        // instruction instead of fadd and fmul separately.
                        // min tile is 32*32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
                        for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            int pv_tile_id = pv_n_loop * (WARP_M / (16 * M_MMAC_COUNT)) * (kBlockK / 32) + mi + ni * (WARP_M / (16 * M_MMAC_COUNT));
                            int mmac_id;
                            if constexpr (M_MMAC_COUNT == 2) {
                                mmac_id    = min_tile_n * 2 + min_tile_m;
                            } else {
                                mmac_id    = min_tile_n;
                            }
                        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                            #pragma unroll
                            for(int vec_idx = 0; vec_idx < 2; ++vec_idx) {
                                acc_o[pv_tile_id][mmac_id].u64[vec_idx] = __builtin_hcu_pk_mul_f32(
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
        scale_apply_exp2<true, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(scores, scores_max_cur, softmax_scale_log2);

        DataType1 scores_sum_cur[(WARP_M / (16 * M_MMAC_COUNT))];
        for (int mi = 0; mi < (WARP_M / (16 * M_MMAC_COUNT)); ++mi) {
            if constexpr (M_MMAC_COUNT == 2) {
                scores_sum_cur[mi].u64 = 0x0;
            } else {
                scores_sum_cur[mi].f32[0] = 0x0;
            }
        }
        reduce_sum<true, DataType0, DataType1, WARP_M, WARP_N, M_MMAC_COUNT>(scores, scores_sum_cur);

        for (int mi = 0; mi < (WARP_M / (16 * M_MMAC_COUNT)); ++mi) {
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
            if constexpr (M_MMAC_COUNT == 2) {
                scores_sum[mi].u64 = __builtin_hcu_pk_add_f32(
                    scores_sum[mi].u64,
                    scores_sum_cur[mi].u64
                );
            } else {
                scores_sum[mi].f32[0] = scores_sum[mi].f32[0] + scores_sum_cur[mi].f32[0];
            }
        #else // for perf-model, add listed below will be optimized as v_fmac_f32, leading to incorrect results
            if constexpr (M_MMAC_COUNT == 2) {
                scores_sum[mi].f32[0] += scores_sum_cur[mi].f32[0];
                scores_sum[mi].f32[1] += scores_sum_cur[mi].f32[1];
            } else {
                scores_sum[mi].f32[0] += scores_sum_cur[mi].f32[0];
            }
        #endif

        #if defined(USE_V_MOV_B64) && (defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__))
            if constexpr (M_MMAC_COUNT == 2) {
                inlineasm_fa_v_mov_b64(
                    scores_max[mi].u64,
                    scores_max_cur[mi].u64
                );
            } else {
                scores_max[mi].f32[0] = scores_max_cur[mi].f32[0];
            }
        #else
            if constexpr (M_MMAC_COUNT == 2) {
                scores_max[mi].f32[0] = scores_max_cur[mi].f32[0];
                scores_max[mi].f32[1] = scores_max_cur[mi].f32[1];
            } else {
                scores_max[mi].f32[0] = scores_max_cur[mi].f32[0];
            }
        #endif
        }
    }
};


// #define USE_CVT_PKRTZ_FP16_FP32
template <int WARP_M, int WARP_N, typename Element, typename ElementAccum, int M_MMAC_COUNT=2>
inline __device__ void prefill_mla_convert_pk_type(union_vec2_f16x2<Element> p_reg[(WARP_M / (16 * M_MMAC_COUNT)) * (WARP_N / 32)][2 * M_MMAC_COUNT], union_vec4_fp32 s_reg[(WARP_M / (16 * M_MMAC_COUNT)) * (WARP_N / 32)][2 * M_MMAC_COUNT]) {
    #pragma unroll
    for(int n_idx = 0; n_idx < (WARP_N / 32); ++n_idx) {
        #pragma unroll
        for(int m_idx = 0; m_idx < (WARP_M / (16 * M_MMAC_COUNT)); ++m_idx) {
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                #pragma unroll
                for(int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    #if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                        if constexpr (M_MMAC_COUNT == 2) {
                            p_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f16x2[min_tile_k] = DownCastPair<float, Element>(
                            s_reg[n_idx * (WARP_M / 32) + m_idx][0 * 2 + min_tile_m].f32x2[min_tile_k]);
                            p_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f16x2[min_tile_k] = DownCastPair<float, Element>(
                            s_reg[n_idx * (WARP_M / 32) + m_idx][1 * 2 + min_tile_m].f32x2[min_tile_k]);
                        } else {
                            p_reg[n_idx * (WARP_M / 16) + m_idx][0].f16x2[min_tile_k] = DownCastPair<float, Element>(
                            s_reg[n_idx * (WARP_M / 16) + m_idx][0].f32x2[min_tile_k]);
                            p_reg[n_idx * (WARP_M / 16) + m_idx][1].f16x2[min_tile_k] = DownCastPair<float, Element>(
                            s_reg[n_idx * (WARP_M / 16) + m_idx][1].f32x2[min_tile_k]);
                        }
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

template <typename DataType, int WARP_M, int WARP_N, int M_MMAC_COUNT=2>
inline __device__ void prefill_mla_apply_mask_gfx938(DataType tensor[(WARP_M / (16 * M_MMAC_COUNT)) * (WARP_N / 32)][2 * M_MMAC_COUNT], const int max_seqlen_k,
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
                    for (int mi = 0; mi < (WARP_M / (16 * M_MMAC_COUNT)); ++mi) {
                        #pragma unroll
                        for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                            if constexpr (M_MMAC_COUNT == 2) {
                                tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                            } else {
                                tensor[mi + ni * (WARP_M / 16)][min_tile_n].f32[vec_idx] = -INFINITY;
                            }
                        }
                    }
                }
            }
        }
    }
}



template <typename DataType, int WARP_M, int WARP_N, int M_MMAC_COUNT=2>
__forceinline__ __device__ void prefill_mla_apply_mask_causal_gfx938(DataType tensor[(WARP_M / (16 * M_MMAC_COUNT)) * (WARP_N / 32)][2 * M_MMAC_COUNT], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15);
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 4;
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / (16 * M_MMAC_COUNT)); ++mi) {
        const int row_idx_base = row_idx_offset + mi * (16 * M_MMAC_COUNT);
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
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
                        if constexpr (M_MMAC_COUNT == 2) {
                            tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = (col_idx > col_idx_limit_right) ? -INFINITY: tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                        } else {
                            tensor[mi + ni * (WARP_M / 16)][min_tile_n].f32[vec_idx] = (col_idx > col_idx_limit_right) ? -INFINITY: tensor[mi + ni * (WARP_M / 16)][min_tile_n].f32[vec_idx];
                        }
                    }
                }
            }
        }
    }
}

template <typename DataType, int WARP_M, int WARP_N, int M_MMAC_COUNT=2>
__forceinline__ __device__ void prefill_mla_apply_mtp_mask_causal_gfx938(DataType tensor[(WARP_M / (16 * M_MMAC_COUNT)) * (WARP_N / 32)][2 * M_MMAC_COUNT], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15) + max_seqlen_k - max_seqlen_q;
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 4;
    #pragma unroll
    for (int mi = 0; mi < (WARP_M / (16 * M_MMAC_COUNT)); ++mi) {
        const int row_idx_base = row_idx_offset + mi * (16 * M_MMAC_COUNT);
        #pragma unroll
        for(int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m * 16;
            #pragma unroll
            for (int ni = 0; ni < (WARP_N / 32); ++ni)  {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n * 16;
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx;
                        if constexpr (M_MMAC_COUNT == 2) {
                            tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] = (col_idx > row_idx) ? -INFINITY: tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                        } else {
                            tensor[mi + ni * (WARP_M / 16)][min_tile_n].f32[vec_idx] = (col_idx > row_idx) ? -INFINITY: tensor[mi + ni * (WARP_M / 16)][min_tile_n].f32[vec_idx];
                        }
                    }
                }
            }
        }
    }
}

template<typename DataType, int kBlockN, int WARP_M, int WARP_NUM>
__forceinline__ __device__ void flashmla_apply_mtp_mask_causal_gfx938(
        DataType s_reg[(WARP_M / 16) * (kBlockN / 32)][2],
        const int col_idx_offset_,
        const int max_seqlen_k,
        const int row_idx_offset_,
        const int max_seqlen_q,
        const int ngroups,
        const int mtp) {
    const int lane_id = threadIdx.x & 63;
    constexpr int mi = 0;
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        int row_idx = row_idx_offset_ + (lane_id & 15);
        int row_in_mtp = row_idx / ngroups;
        int col_idx_limit_right = min(max_seqlen_k, row_in_mtp + max_seqlen_k - mtp);
        #pragma unroll
        for (int ni = 0; ni < kBlockN / 32; ++ni) {
            #pragma unroll
            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                    const int col_idx = col_idx_offset_ + ni * 32 + min_tile_n * 16 + (lane_id >> 4) * 4 + vec_idx; /*BMZ vec_idx * 4 + (lane_id >> 4) */
                    s_reg[mi + ni * (WARP_M / 16)][min_tile_n].f32[vec_idx] = (col_idx > col_idx_limit_right) ? -INFINITY: s_reg[mi + ni * (WARP_M / 16)][min_tile_n].f32[vec_idx];
                }
            }
        }
    }
}

template <bool HasWSLeft=true, typename DataType, int WARP_M, int WARP_N>
inline __device__ void prefill_mla_apply_mask_local_gfx938(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
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
            const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + max_seqlen_k - max_seqlen_q + window_size_right);
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
inline __device__ void prefill_mla_apply_alibi_gfx938(DataType tensor[(WARP_M / 32) * (WARP_N / 32)][4], const int col_idx_offset_,
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
                        tensor[mi + ni * (WARP_M / 32)][min_tile_n * 2 + min_tile_m].f32[vec_idx] += g_alibi * (col_idx - row_idx);
                    }
                }
            }
        }
    }
}