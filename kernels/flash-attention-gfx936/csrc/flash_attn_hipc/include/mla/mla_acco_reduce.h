#include "numeric_types.h"


template<int REUSE_KV_TIMES, int K_LOOP_COUNT, int K_WARP_COUNT, int M_WARP_COUNT, int M_MMAC_COUNT, int WARP_NUM, typename ElementAccum>
__forceinline__ __device__ void mla_acco_reduce(
        vec4_Accum < ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4],
        ElementAccum* acc_o_lds,
        int seqlen_q,
        int warp_id,
        int lane_id) {
    constexpr int kBlockK = K_WARP_COUNT * 32;

    // when REUSE_KV not in templated, compute max reuse times
    int EVEN_REUSE_KV_TIMES = (REUSE_KV_TIMES > 0) ? ((REUSE_KV_TIMES + 1) / 2) * 2: ((seqlen_q + 1) / 2) * 2;
    int HALF_REUSE_KV_TIMES = EVEN_REUSE_KV_TIMES >> 1;

    int q_seq_idx = (lane_id & 15);
    if (q_seq_idx < HALF_REUSE_KV_TIMES) { // 除以 2, 是因为每个线程都会储存两行的数据, seq 方向上是 0,0,1,1,2,2,3,3,4,4,....,15,15
        for (int h_idx = 0; h_idx < K_LOOP_COUNT; ++h_idx) {
            // ####################################################################################################################################################
            // 4 个 wave 分别把自己负责的 acc_o 计算结果写到 LDS 中
            for (int k_idx = 0; k_idx < K_WARP_COUNT; ++k_idx) {
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        // 一个 wave 共同持有 seqlen_q x kHeadDim 个 Half, 但为了节省 lds 用量, 每次只 reduce seqlen_q x kBlockK 个 Half
                        int lds_offset = (warp_id * EVEN_REUSE_KV_TIMES + q_seq_idx * 2 + min_tile_m) * kBlockK + k_idx * 32 + min_tile_n * 16 + (lane_id >> 4/*0~3*/) * 4/*0~15*/;
                        *(vec4_fp32*)(acc_o_lds + lds_offset) = acc_o[h_idx * (K_WARP_COUNT + k_idx) * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32;
                    }
                }
            }
            __syncthreads();
            // ####################################################################################################################################################
            // 在 lds 中求和, 把 4 个 wave 写的 acc_o 的数据加起来
            // 如果恰好是 4 个 wave, 则 4 个 wave 一起参与到 lds 操作, 每个 wave 操作 4 个元素中的一个
            if constexpr (WARP_NUM == 4) {
                for (int k_idx = 0; k_idx < K_WARP_COUNT; ++k_idx) {
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            int lds_offset = (q_seq_idx * 2 + min_tile_m) * kBlockK + k_idx * 32 + min_tile_n * 16 + (lane_id >> 4) * 4 + warp_id; // 之前是一次性写了 4 个 Half 到 lds, 现在 4 个 wave 分别处理这 4 个位置的 acc_o reduce
                            float acc_tmp_wave0 = acc_o_lds[lds_offset];
                            for (int loop = 1; loop < WARP_NUM; ++loop) {
                                acc_tmp_wave0 += acc_o_lds[lds_offset + loop * EVEN_REUSE_KV_TIMES * kBlockK];
                            }
                            acc_o_lds[lds_offset] = acc_tmp_wave0;
                        }
                    }
                }
            }
            // 不是恰好 4 个 wave, 则把 wave 0 单独拎出来做 lds reduce 操作
            else if constexpr (WARP_NUM > 1) {
                if (warp_id == 0) {
                    for (int k_idx = 0; k_idx < K_WARP_COUNT; ++k_idx) {
                        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                                for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                                    int lds_offset = (q_seq_idx * 2 + min_tile_m) * kBlockK + k_idx * 32 + min_tile_n * 16 + (lane_id >> 4) * 4 + vec_idx;
                                    float acc_tmp_wave0 = acc_o_lds[lds_offset];
                                    for (int loop = 1; loop < WARP_NUM; ++loop) {
                                        acc_tmp_wave0 += acc_o_lds[lds_offset + loop * EVEN_REUSE_KV_TIMES * kBlockK];
                                    }
                                    acc_o_lds[lds_offset] = acc_tmp_wave0;
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
            // ####################################################################################################################################################
            // 每个 wave 都从 LDS 获取最终的求和结果
            for (int k_idx = 0; k_idx < K_WARP_COUNT; ++k_idx) {
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        int lds_offset = (q_seq_idx * 2 + min_tile_m) * kBlockK + k_idx * 32 + min_tile_n * 16 + (lane_id >> 4) * 4;
                        acc_o[h_idx * (K_WARP_COUNT + k_idx) * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32 = *(vec4_fp32*)(acc_o_lds + lds_offset);
                    }
                }
            }
            __syncthreads();
        }
    }
}