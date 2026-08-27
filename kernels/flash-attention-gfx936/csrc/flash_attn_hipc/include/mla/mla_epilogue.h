#include "numeric_types.h"


template<int K_LOOP_COUNT, int M_WARP_COUNT, int K_WARP_COUNT, int M_MMAC_COUNT, typename ElementAccum>
__forceinline__ __device__ void mla_epilugue_rescale_acco(
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
                    #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
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
__forceinline__ __device__ void mla_tp8_epilogue_store_softmax_lse(
    vec2_Accum<ElementAccum> scores_max[M_WARP_COUNT],
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT],
    ElementAccum *softmax_lse_ptr,
    ElementAccum scale_softmax,
    int warp_id,
    int thread_id,
    int lane_id,
    int headdim_split_id,
    int seqlen_q_limit
) {
    if constexpr (Split) {
        bool write_ok = Is_16x32 ? (thread_id < 16 and headdim_split_id == 0): thread_id < 16;
        if (write_ok) {
            #pragma unroll
            for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    const int row = Is_16x32
                        ? mi * 32 + lane_id/*equal to lane_id & 15*/ + min_tile_m * 16
                        : warp_id * M_WARP_COUNT * 32 + mi * 32 + thread_id * 2 + min_tile_m;
                    if (row < seqlen_q_limit) {
                        softmax_lse_ptr[row] = scores_max[mi].f32[min_tile_m] * scale_softmax + __logf(scores_sum[mi].f32[min_tile_m]);
                    }
                }
            }
        }
    }
}


template<typename Params, int kHeadDimV, int kHeadDimVSplit, bool Split, typename SplitkvAccumType, typename ElementAccum, int kBlockM, int kBlockK, int WARP_NUM, int K_LOOP_COUNT, int M_WARP_COUNT, int K_WARP_COUNT, int M_MMAC_COUNT>
__forceinline__ __device__ void mla_epilogue_store_output(
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
    const int64_t row_offset_o = bidb * int64_t(params.o_batch_stride) + bidh * params.o_head_stride + headdim_split_id * kHeadDimVSplit;
    SplitkvAccumType* o_ptr  = Split
                             ? reinterpret_cast<SplitkvAccumType *>(params.oaccum_ptr) + row_offset_o + /*which split*/ split_id * params.b * params.o_batch_stride
                             : reinterpret_cast<SplitkvAccumType *>(params.o_ptr) + row_offset_o;
    auto gO = prepare_for_buffer_load<kHeadDimV, SplitkvAccumType, false/*USE_CACHE_SWIZZLE*/>(o_ptr);
    int pv_lane_seq_idx       = (lane_id & 15);
    int pv_lane_head_dim_idx  = (lane_id >> 4);
    #pragma unroll
    for (int k_loop = 0; k_loop < K_LOOP_COUNT; ++k_loop) {
        #pragma unroll
        for (int warp_m_idx = 0; warp_m_idx < M_WARP_COUNT; ++warp_m_idx) {
            #pragma unroll
            for (int k_tile_idx = 0; k_tile_idx < K_WARP_COUNT; ++k_tile_idx) {
                // 获取第几个 32x32 tile
                int tile_32x32_id = k_loop * M_WARP_COUNT * K_WARP_COUNT + warp_m_idx * K_WARP_COUNT + k_tile_idx;
                #pragma unroll 2
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        // 当前 32x32 tile 的第几个 mmac
                        int mmac_id = min_tile_m + min_tile_n * 2;
                        // seqlen_q 方向上的坐标
                        int seqlen_q_idx = m_block * kBlockM + warp_m_idx * 32 + pv_lane_seq_idx * 2 + min_tile_m;
                        if constexpr (WARP_NUM == 4) { // for 4 waves, storation can be done togather, performance 4%
                            int vec_index      = warp_id;
                            int64_t pv_global_addr = seqlen_q_idx * output_seqlen_stride + k_loop * kBlockK + k_tile_idx * 32 + vec_index * 8 + pv_lane_head_dim_idx * 2 + min_tile_n;
                            ElementAccum data  = acc_o[tile_32x32_id][mmac_id].f32[vec_index];
                            if (seqlen_q_idx < params.seqlen_q) {
                                o_ptr[pv_global_addr] = DownCast<ElementAccum, SplitkvAccumType>(data);
                            }
                        } else { // non-4-waves should use this, but lead to performance drop when 4 waves per SIMD
                            #pragma unroll
                            for (int vec_index = 0;  vec_index < 4; ++vec_index) {
                                int64_t pv_global_addr = seqlen_q_idx * output_seqlen_stride + k_loop * kBlockK + k_tile_idx * 32 + vec_index * 8 + pv_lane_head_dim_idx * 2 + min_tile_n;
                                ElementAccum data  = acc_o[tile_32x32_id][mmac_id].f32[vec_index];
                                if (seqlen_q_idx < params.seqlen_q) {
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