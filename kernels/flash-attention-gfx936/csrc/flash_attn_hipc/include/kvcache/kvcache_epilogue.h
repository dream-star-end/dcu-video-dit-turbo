#pragma once
#include "numeric_types.h"

__forceinline__ __device__ float kvcache_attention_sink_load(
    const void *s_aux_ptr,
    int s_aux_type,
    int head_idx) {
    if (s_aux_type == 1) {
        return reinterpret_cast<const float *>(s_aux_ptr)[head_idx];
    } else if (s_aux_type == 2) {
        return UpCast<half_t, float>(
            reinterpret_cast<const half_t *>(s_aux_ptr)[head_idx]);
    } else {
        return UpCast<BFloat16, float>(
            reinterpret_cast<const BFloat16 *>(s_aux_ptr)[head_idx]);
    }
}

template<int K_LOOP_COUNT, int M_WARP_COUNT, int K_WARP_COUNT, int M_MMAC_COUNT, typename ElementAccum>
__forceinline__ __device__ void kvcache_apply_attention_sink(
    vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4],
    vec2_Accum<ElementAccum> scores_max[M_WARP_COUNT],
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT],
    const void *s_aux_ptr,
    int s_aux_type,
    int bidh,
    int ngroups,
    int m_block,
    int kBlockM,
    int lane_id,
    ElementAccum scale_softmax) {
    #pragma unroll
    for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            const int row = m_block * kBlockM + mi * 32 + (lane_id & 15) + min_tile_m * 16;
            const int group_id = row % ngroups;
            const int sink_head = bidh * ngroups + group_id;
            const ElementAccum sink_value = kvcache_attention_sink_load(
                s_aux_ptr, s_aux_type, sink_head);
            const ElementAccum old_scaled_max = scores_max[mi].f32[min_tile_m] * scale_softmax;
            const ElementAccum new_scaled_max = max(old_scaled_max, sink_value);
            const ElementAccum old_rescale = __expf(old_scaled_max - new_scaled_max);
            scores_sum[mi].f32[min_tile_m] =
                scores_sum[mi].f32[min_tile_m] * old_rescale +
                __expf(sink_value - new_scaled_max);
            scores_max[mi].f32[min_tile_m] = new_scaled_max / scale_softmax;

            __float2 old_rescale_pair = {old_rescale, old_rescale};
            #pragma unroll
            for (int pv_n_loop = 0; pv_n_loop < K_LOOP_COUNT; ++pv_n_loop) {
                #pragma unroll
                for (int ni = 0; ni < K_WARP_COUNT; ++ni) {
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        const int mmac_id = min_tile_n * 2 + min_tile_m;
                        const int tile_32x32_id =
                            pv_n_loop * M_WARP_COUNT * K_WARP_COUNT +
                            (ni * M_WARP_COUNT + mi);
                        #pragma unroll
                        for (int vec_id = 0; vec_id < 2; ++vec_id) {
                            acc_o[tile_32x32_id][mmac_id].u64[vec_id] =
                                __builtin_hcu_pk_mul_f32(
                                    acc_o[tile_32x32_id][mmac_id].u64[vec_id],
                                    old_rescale_pair);
                        }
                    }
                }
            }
        }
    }
}


template<int K_LOOP_COUNT, int M_WARP_COUNT, int K_WARP_COUNT, int M_MMAC_COUNT, typename ElementAccum>
__forceinline__ __device__ void kvcache_epilugue_rescale_acco(
    vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4],
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT]) {
    #pragma unroll
    for (int pv_n_loop = 0; pv_n_loop < K_LOOP_COUNT; ++pv_n_loop) {
        #pragma unroll
        for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < K_WARP_COUNT; ++ni) {
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    ElementAccum sum     = scores_sum[mi].f32[min_tile_m];
                    ElementAccum inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
                    __float2 scale_pair  = {inv_sum, inv_sum};
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        int mmac_id       = min_tile_n * 2 + min_tile_m;
                        int tile_32x32_id = pv_n_loop * M_WARP_COUNT * K_WARP_COUNT + (ni * M_WARP_COUNT + mi);
                    #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                        for (int vec_id = 0; vec_id < 2; ++vec_id) {
                            acc_o[tile_32x32_id][mmac_id].u64[vec_id] = __builtin_hcu_pk_mul_f32(
                                acc_o[tile_32x32_id][mmac_id].u64[vec_id],
                                scale_pair
                            );
                        }
                    #else
                        for (int vec_id = 0; vec_id < 4; ++vec_id) {
                            acc_o[tile_32x32_id][mmac_id].f32[vec_id] *= inv_sum;
                        }
                    #endif
                    }
                }
            }
        }
    }
}


template<bool Split, bool Is_16x32, int M_WARP_COUNT, int M_MMAC_COUNT, typename ElementAccum>
__forceinline__ __device__ void kvcache_epilogue_store_max_sum(
    vec2_Accum<ElementAccum> scores_max[M_WARP_COUNT],
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT],
    ElementAccum *scores_max_ptr,
    ElementAccum *scores_sum_ptr,
    ElementAccum scale_softmax,
    int warp_id,
    int thread_id,
    int lane_id,
    int headdim_split_id,
    int seqlen_q_limit
) {
    if constexpr (Split) {
        bool write_ok = Is_16x32 ? (thread_id < 16 and headdim_split_id == 0): thread_id < 16;
        if (write_ok) { // 0-15 号线程储存有 max/sum 的数据, 16~31/32~47/48~63 号线程也含有, 但只需要写一次即可
            #pragma unroll
            for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    const int row = Is_16x32
                        ? /*warp_id * WARP_M + */mi * 32 + lane_id/*equal to lane_id & 15*/ + min_tile_m * 16
                        : warp_id * M_WARP_COUNT * 32 + mi * 32 + thread_id * 2 + min_tile_m;
                    if (row < seqlen_q_limit) {
                        scores_sum_ptr[row] = scores_sum[mi].f32[min_tile_m];
                        scores_max_ptr[row] = scores_max[mi].f32[min_tile_m] * scale_softmax;
                    }
                }
            }
        }
    }
}



template<bool Split, bool Is_16x32, int M_WARP_COUNT, int M_MMAC_COUNT, typename ElementAccum>
__forceinline__ __device__ void kvcache_varlen_epilogue_store_max_sum(
    vec2_Accum<ElementAccum> scores_max[M_WARP_COUNT],
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT],
    ElementAccum *scores_max_ptr,
    ElementAccum *scores_sum_ptr,
    ElementAccum scale_softmax,
    int warp_id,
    int thread_id,
    int lane_id,
    int headdim_split_id,
    int seqlen_q_limit,
    int total_q,
    int ngroups
) {
    if constexpr (Split) {
        bool write_ok = Is_16x32 ? (thread_id < 16 and headdim_split_id == 0): thread_id < 16;
        if (write_ok) { // 0-15 号线程储存有 max/sum 的数据, 16~31/32~47/48~63 号线程也含有, 但只需要写一次即可
            #pragma unroll
            for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    const int row = Is_16x32
                        ? /*warp_id * WARP_M + */mi * 32 + lane_id/*equal to lane_id & 15*/ + min_tile_m * 16
                        : warp_id * M_WARP_COUNT * 32 + mi * 32 + thread_id * 2 + min_tile_m;
                    if (row < seqlen_q_limit) {
                        int row_target = (row / ngroups) + (row % ngroups) * total_q;
                        scores_sum_ptr[row_target] = scores_sum[mi].f32[min_tile_m];
                        scores_max_ptr[row_target] = scores_max[mi].f32[min_tile_m] * scale_softmax;
                    }
                }
            }
        }
    }
}



template<bool Is_Varlen, bool Is_16x32, int M_WARP_COUNT, int M_MMAC_COUNT, typename ElementAccum>
__forceinline__ __device__ void kvcache_epilogue_store_softmax_lse(
    vec2_Accum<ElementAccum> scores_max[M_WARP_COUNT],
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT],
    ElementAccum *softmax_lse_ptr,
    ElementAccum scale_softmax,
    int warp_id,
    int thread_id,
    int lane_id,
    int headdim_split_id,
    int seqlen_q_limit,
    int total_q,
    int ngroups
) {
    bool write_ok = Is_16x32 ? (thread_id < 16 and headdim_split_id == 0): thread_id < 16;
    if (write_ok) {
        #pragma unroll
        for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                ElementAccum lse = scores_max[mi].f32[min_tile_m] * scale_softmax + __logf(scores_sum[mi].f32[min_tile_m]);
                const int row = Is_16x32
                    ? /*warp_id * WARP_M + */mi * 32 + lane_id/*equal to lane_id & 15*/ + min_tile_m * 16
                    : warp_id * M_WARP_COUNT * 32 + mi * 32 + thread_id * 2 + min_tile_m;
                if (row < seqlen_q_limit) {
                    int row_target = Is_Varlen ? (row / ngroups) + (row % ngroups) * total_q: row;
                    softmax_lse_ptr[row_target] = lse;
                }
            }
        }
    }
}



template<typename Params, int kHeadDimV, int kHeadDimVSplit, bool Split, bool Is_16x32, typename SplitkvAccumType, typename ElementAccum, int kBlockM, int kBlockK, int WARP_NUM, int K_LOOP_COUNT, int M_WARP_COUNT, int K_WARP_COUNT, int M_MMAC_COUNT>
__forceinline__ __device__ void kvcache_epilogue_store_output(
        vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4],
        Params params,
        int bidb,
        int bidh,
        int m_block,
        int split_id,
        int headdim_split_id,
        int warp_id,
        int lane_id) {
    int output_seqlen_stride = params.o_row_stride;
    const int64_t row_offset_o   = bidb * int64_t(params.o_batch_stride) + bidh * params.o_head_stride + headdim_split_id * kHeadDimVSplit;
    SplitkvAccumType* o_ptr  = Split
                             ? reinterpret_cast<SplitkvAccumType *>(params.oaccum_ptr) + row_offset_o + /*which split*/ split_id * params.b * params.o_batch_stride
                             : reinterpret_cast<SplitkvAccumType *>(params.o_ptr) + row_offset_o;
    int pv_lane_seq_idx      = lane_id & 15;
    int pv_lane_head_dim_idx = lane_id >> 4;
    #pragma unroll
    for (int k_loop = 0; k_loop < K_LOOP_COUNT; ++k_loop) {
        #pragma unroll
        for (int warp_m_idx = 0; warp_m_idx < M_WARP_COUNT; ++warp_m_idx) {
            #pragma unroll
            for (int k_tile_idx = 0; k_tile_idx < K_WARP_COUNT; ++k_tile_idx) {
                // acquire tile 32x32 id
                int tile_32x32_id = k_loop * M_WARP_COUNT * K_WARP_COUNT + warp_m_idx * K_WARP_COUNT + k_tile_idx;
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    // seqlen_q offset
                    int seqlen_q_idx = m_block * kBlockM + warp_m_idx * 32 + (Is_16x32 ? pv_lane_seq_idx + min_tile_m * 16: pv_lane_seq_idx * 2 + min_tile_m);
                    if (seqlen_q_idx < params.seqlen_q) {
                        if constexpr (WARP_NUM == 4) { // for 4 waves, storation can be done togather, performance 4%
                            #if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                                int vec_index      = warp_id;
                                int64_t pv_global_addr = seqlen_q_idx * output_seqlen_stride + k_loop * kBlockK + k_tile_idx * 32 + vec_index * 8 + pv_lane_head_dim_idx * 2;
                                vec2_Element<SplitkvAccumType> result = DownCastPairNoPack<ElementAccum, SplitkvAccumType>(acc_o[tile_32x32_id][min_tile_m + 0 * 2].f32[vec_index], acc_o[tile_32x32_id][min_tile_m + 1 * 2].f32[vec_index]);
                                *(vec2_Element<SplitkvAccumType>*)(o_ptr + pv_global_addr) = result;
                            #else
                                #pragma unroll 2
                                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                                    int mmac_id        = min_tile_m + min_tile_n * 2;
                                    int vec_index      = warp_id;
                                    int64_t pv_global_addr = seqlen_q_idx * output_seqlen_stride + k_loop * kBlockK + k_tile_idx * 32 + vec_index * 8 + pv_lane_head_dim_idx * 2 + min_tile_n;
                                    ElementAccum data  = acc_o[tile_32x32_id][mmac_id].f32[vec_index];
                                    o_ptr[pv_global_addr] = DownCast<ElementAccum, SplitkvAccumType>(data);
                                }
                            #endif
                        } else { // non-4-waves should use this, but lead to performance drop when 4 waves per SIMD
                            #pragma unroll
                            for (int vec_index = 0;  vec_index < 4; ++vec_index) {
                                #pragma unroll 2
                                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                                    // 当前 32x32 tile 的第几个 mmac
                                    int mmac_id = min_tile_m + min_tile_n * 2;
                                    int64_t pv_global_addr = seqlen_q_idx * output_seqlen_stride + k_loop * kBlockK + k_tile_idx * 32 + vec_index * 8 + pv_lane_head_dim_idx * 2 + min_tile_n;
                                    ElementAccum data  = acc_o[tile_32x32_id][mmac_id].f32[vec_index];
                                    o_ptr[pv_global_addr] = DownCast<ElementAccum, SplitkvAccumType>(data);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)");
}



template<typename Params, int kHeadDimV, int kHeadDimVSplit, bool Split, bool Is_16x32, typename SplitkvAccumType, typename ElementAccum, int kBlockM, int kBlockK, int WARP_NUM, int K_LOOP_COUNT, int M_WARP_COUNT, int K_WARP_COUNT, int M_MMAC_COUNT>
__forceinline__ __device__ void kvcache_varlen_epilogue_store_output(
        vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4],
        Params params,
        int64_t row_offset_o,
        int actual_seqlen_q,
        int bidb,
        int bidh,
        int m_block,
        int split_id,
        int headdim_split_id,
        int warp_id,
        int lane_id) {
    int output_seqlen_stride = params.o_row_stride;
    const int64_t row_offset_split = params.ngroups * int64_t(params.total_q) * params.o_row_stride;
    SplitkvAccumType* o_ptr  = Split
                             ? reinterpret_cast<SplitkvAccumType *>(params.oaccum_ptr) + row_offset_o + /*which split*/ split_id * row_offset_split
                             : reinterpret_cast<SplitkvAccumType *>(params.o_ptr) + row_offset_o;
    int pv_lane_seq_idx      = lane_id & 15;
    int pv_lane_head_dim_idx = lane_id >> 4;
    #pragma unroll
    for (int k_loop = 0; k_loop < K_LOOP_COUNT; ++k_loop) {
        #pragma unroll
        for (int warp_m_idx = 0; warp_m_idx < M_WARP_COUNT; ++warp_m_idx) {
            #pragma unroll
            for (int k_tile_idx = 0; k_tile_idx < K_WARP_COUNT; ++k_tile_idx) {
                // acquire tile 32x32 id
                int tile_32x32_id = k_loop * M_WARP_COUNT * K_WARP_COUNT + warp_m_idx * K_WARP_COUNT + k_tile_idx;
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    // seqlen_q offset
                    int seqlen_q_idx = m_block * kBlockM + warp_m_idx * 32 + (Is_16x32 ? pv_lane_seq_idx + min_tile_m * 16: pv_lane_seq_idx * 2 + min_tile_m);
                    if (seqlen_q_idx < actual_seqlen_q) {
                        if constexpr (WARP_NUM == 4) { // for 4 waves, storation can be done togather, performance 4%
                            #if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                                int vec_index      = warp_id;
                                int true_seqlen_q = seqlen_q_idx / params.ngroups;
                                int true_group_id = seqlen_q_idx % params.ngroups;
                                int64_t pv_global_addr = true_seqlen_q * params.ngroups * output_seqlen_stride + true_group_id * params.o_head_stride + k_loop * kBlockK + k_tile_idx * 32 + vec_index * 8 + pv_lane_head_dim_idx * 2;
                                vec2_Element<SplitkvAccumType> result = DownCastPairNoPack<ElementAccum, SplitkvAccumType>(acc_o[tile_32x32_id][min_tile_m + 0 * 2].f32[vec_index], acc_o[tile_32x32_id][min_tile_m + 1 * 2].f32[vec_index]);
                                *(vec2_Element<SplitkvAccumType>*)(o_ptr + pv_global_addr) = result;
                            #else
                                #pragma unroll 2
                                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                                    int mmac_id        = min_tile_m + min_tile_n * 2;
                                    int vec_index      = warp_id;
                                    int true_seqlen_q = seqlen_q_idx / params.ngroups;
                                    int true_group_id = seqlen_q_idx % params.ngroups;
                                    int64_t pv_global_addr = true_seqlen_q * params.ngroups * output_seqlen_stride + true_group_id * params.o_head_stride + k_loop * kBlockK + k_tile_idx * 32 + vec_index * 8 + pv_lane_head_dim_idx * 2 + min_tile_n;
                                    ElementAccum data  = acc_o[tile_32x32_id][mmac_id].f32[vec_index];
                                    o_ptr[pv_global_addr] = DownCast<ElementAccum, SplitkvAccumType>(data);
                                }
                            #endif
                        } else { // non-4-waves should use this, but lead to performance drop when 4 waves per SIMD
                            #pragma unroll
                            for (int vec_index = 0;  vec_index < 4; ++vec_index) {
                                #pragma unroll 2
                                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                                    // 当前 32x32 tile 的第几个 mmac
                                    int mmac_id = min_tile_m + min_tile_n * 2;
                                    int true_seqlen_q = seqlen_q_idx / params.ngroups;
                                    int true_group_id = seqlen_q_idx % params.ngroups;
                                    int64_t pv_global_addr = true_seqlen_q * params.ngroups * output_seqlen_stride + true_group_id * params.o_head_stride + k_loop * kBlockK + k_tile_idx * 32 + vec_index * 8 + pv_lane_head_dim_idx * 2 + min_tile_n;
                                    ElementAccum data  = acc_o[tile_32x32_id][mmac_id].f32[vec_index];
                                    o_ptr[pv_global_addr] = DownCast<ElementAccum, SplitkvAccumType>(data);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)");
}
