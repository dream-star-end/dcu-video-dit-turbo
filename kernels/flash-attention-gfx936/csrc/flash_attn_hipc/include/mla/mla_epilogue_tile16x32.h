#include "numeric_types.h"



template<bool Split, int M_WARP_COUNT, int M_MMAC_COUNT, typename ElementAccum>
__forceinline__ __device__ void mla_epilogue_store_max_sum_tile16x32(
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
        if (headdim_split_id == 0) { // 因为 split-D 使用同样的 QK, 计算得到同样的 scores_sum/scores_max 会写多遍, 可能会有数据冲突, 所以强制只写一遍
            if (thread_id < 16) { // 0-15 号线程储存有 max/sum 的数据, 16~31/32~47/48~63 号线程也含有, 但只需要写一次即可
                #pragma unroll
                for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
                    #pragma unroll
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        const int row = /*warp_id * WARP_M + */mi * 32 + lane_id/*equal to lane_id & 15*/ + min_tile_m * 16;
                        if (row < seqlen_q_limit) {
                            scores_sum_ptr[row] = scores_sum[mi].f32[min_tile_m];
                            scores_max_ptr[row] = scores_max[mi].f32[min_tile_m] * scale_softmax;
                        }
                    }
                }
            }
        }
    }
}



template<typename Params, int kHeadDimV, int kHeadDimVSplit, bool Split, typename SplitkvAccumType, typename ElementAccum, int kBlockM, int kBlockK, int WARP_NUM, int K_LOOP_COUNT, int M_WARP_COUNT, int K_WARP_COUNT, int M_MMAC_COUNT>
__forceinline__ __device__ void mla_epilogue_store_output_tile16x32(
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
    int pv_lane_seq_idx      = (lane_id & 15);
    int pv_lane_head_dim_idx = (lane_id >> 4);
    #pragma unroll
    for (int k_loop = 0; k_loop < K_LOOP_COUNT; k_loop += WARP_NUM) {
        #pragma unroll
        for (int warp_m_idx = 0; warp_m_idx < M_WARP_COUNT; ++warp_m_idx) {
            #pragma unroll
            for (int k_tile_idx = 0; k_tile_idx < K_WARP_COUNT; ++k_tile_idx) {
                int tile_32x32_id = k_loop * M_WARP_COUNT * K_WARP_COUNT + warp_m_idx * K_WARP_COUNT + k_tile_idx;
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    int seqlen_q_idx = m_block * kBlockM + warp_m_idx * 32 + pv_lane_seq_idx + min_tile_m * 16;
                    if (seqlen_q_idx < params.seqlen_q) {
                        #pragma unroll
                        for (int vec_index = 0; vec_index < 4; ++vec_index) {
                            vec2_Element<SplitkvAccumType> data;
                            #pragma unroll
                            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                                int mmac_id = min_tile_m + min_tile_n * 2;
                                data[min_tile_n] = DownCast<ElementAccum, SplitkvAccumType, true>(acc_o[tile_32x32_id][mmac_id].f32[vec_index]);
                            }
                            int64_t pv_global_addr = seqlen_q_idx * output_seqlen_stride + (k_loop + warp_id) * kBlockK + k_tile_idx * 32 + vec_index * 8 + pv_lane_head_dim_idx * 2;
                            *(vec2_Element<SplitkvAccumType>*)(o_ptr + pv_global_addr) = data;
                        }
                    }
                }
            }
        }
    }
}
