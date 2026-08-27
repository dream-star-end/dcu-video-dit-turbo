#include "numeric_types.h"


template<int REUSE_KV_TIMES, int kHeadDim, int kBlockK, int WARP_M, int M_MMAC_COUNT, int WARP_NUM, typename ElementAccum>
__forceinline__ __device__ void kvcache_acco_reduce(
        vec4_Accum<ElementAccum> acc_o[(kHeadDim / kBlockK) * ((WARP_M / 32) * (kBlockK / 32))][4],
        ElementAccum* acc_o_lds,
        int seqlen_q,
        int WARP_ID,
        int lane_id) {

    // when REUSE_KV not in templated, compute max reuse times
    int EVEN_REUSE_KV_TIMES = (REUSE_KV_TIMES > 0) ? ((REUSE_KV_TIMES + 1) / 2) * 2: ((seqlen_q + 1) / 2) * 2;
    int HALF_REUSE_KV_TIMES = EVEN_REUSE_KV_TIMES >> 1;

    int q_seq_idx = (lane_id & 15);
    constexpr int __kHeadDim = (REUSE_KV_TIMES >= 16 or kHeadDim == 512) ? kHeadDim: kHeadDim + 4/*<=15 can use misalign to reduce bank conflicts, but >16 may lead to lds>32KB, less waves per SIMD*/;
    if (q_seq_idx < HALF_REUSE_KV_TIMES) {
        // ####################################################################################################################################################
        // 4 个 wave 分别把自己负责的 acc_o 计算结果写到 LDS 中
        for (int h_idx = 0; h_idx < (kHeadDim / kBlockK); h_idx++) {
            for (int k_idx = 0; k_idx < (kBlockK / 32); k_idx++) {
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; min_tile_m++) {
                    for (int min_tile_k = 0; min_tile_k < 2; min_tile_k++) {
                        int lds_offset = WARP_ID * EVEN_REUSE_KV_TIMES * __kHeadDim + q_seq_idx * 2 * __kHeadDim + min_tile_m * __kHeadDim + h_idx * kBlockK + k_idx * 32 + min_tile_k * 16 + (lane_id >> 4) * 4;
                        *(vec4_fp32*)(acc_o_lds + lds_offset) = acc_o[h_idx * ((WARP_M / 32) * (kBlockK / 32)) + k_idx * (WARP_M / 32)][min_tile_k * 2 + min_tile_m].f32;
                    }
                }
            }
        }
        __syncthreads();
        // ####################################################################################################################################################
        // 4 个 wave 共同参与 acc_o 在 LDS 中的相加
        if constexpr (WARP_NUM == 4) {
            for (int h_idx = 0; h_idx < (kHeadDim / kBlockK); h_idx++) {
                for (int k_idx = 0; k_idx < (kBlockK / 32); k_idx++) {
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; min_tile_m++) {
                        union_vec2_fp32 acc_tmp;
                        int lds_offset0 = min_tile_m * __kHeadDim + q_seq_idx * 2 * __kHeadDim + h_idx * kBlockK + k_idx * 32 + 0 * 16 + (lane_id >> 4) * 4 + WARP_ID;
                        int lds_offset1 = min_tile_m * __kHeadDim + q_seq_idx * 2 * __kHeadDim + h_idx * kBlockK + k_idx * 32 + 1 * 16 + (lane_id >> 4) * 4 + WARP_ID;
                        acc_tmp.u64 = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float *)acc_o_lds + lds_offset0, 0, 16, false);
                        union_vec2_fp32 acc_tmp_wave1;
                        acc_tmp_wave1.u64 = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float *)acc_o_lds + lds_offset0 + 1 * EVEN_REUSE_KV_TIMES * __kHeadDim, 0, 16, false);
                        acc_tmp.f32[0] += acc_tmp_wave1.f32[0];
                        acc_tmp.f32[1] += acc_tmp_wave1.f32[1];
                        union_vec2_fp32 acc_tmp_wave2;
                        acc_tmp_wave2.u64 = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float *)acc_o_lds + lds_offset0 + 2 * EVEN_REUSE_KV_TIMES * __kHeadDim, 0, 16, false);
                        acc_tmp.f32[0] += acc_tmp_wave2.f32[0];
                        acc_tmp.f32[1] += acc_tmp_wave2.f32[1];
                        union_vec2_fp32 acc_tmp_wave3;
                        acc_tmp_wave3.u64 = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float *)acc_o_lds + lds_offset0 + 3 * EVEN_REUSE_KV_TIMES * __kHeadDim, 0, 16, false);
                        acc_tmp.f32[0] += acc_tmp_wave3.f32[0];
                        acc_tmp.f32[1] += acc_tmp_wave3.f32[1];
                        // ds_write2_b32
                        acc_o_lds[lds_offset0] = acc_tmp.f32[0];
                        acc_o_lds[lds_offset1] = acc_tmp.f32[1];
                    }
                }
            }
            __syncthreads();
        } else if constexpr (WARP_NUM > 1) {
            if (WARP_ID == 0) {
                for (int h_idx = 0; h_idx < (kHeadDim / kBlockK); h_idx++) {
                    for (int k_idx = 0; k_idx < (kBlockK / 32); k_idx++) {
                        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; min_tile_m++) {
                            for (int min_tile_k = 0; min_tile_k < 2; min_tile_k++) {
                                for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                                    int lds_offset = min_tile_m * __kHeadDim + q_seq_idx * 2 * __kHeadDim + h_idx * kBlockK + k_idx * 32 + min_tile_k * 16 + (lane_id >> 4) * 4 + vec_idx;
                                    float acc_tmp_wave0  = acc_o_lds[lds_offset];
                                    for (int loop = 1; loop < WARP_NUM; loop++) {
                                        acc_tmp_wave0 += acc_o_lds[lds_offset + loop * EVEN_REUSE_KV_TIMES * __kHeadDim];
                                    }
                                    acc_o_lds[lds_offset] = acc_tmp_wave0;
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        // ####################################################################################################################################################
        // 每个 wave 都从 LDS 获取最终的求和结果
        for (int h_idx = 0; h_idx < (kHeadDim / kBlockK); h_idx++) {
            for (int k_idx = 0; k_idx < (kBlockK / 32); k_idx++) {
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; min_tile_m++) {
                    for (int min_tile_k = 0; min_tile_k < 2; min_tile_k++) {
                        int lds_offset = q_seq_idx * 2 * __kHeadDim + min_tile_m * __kHeadDim + h_idx * kBlockK + k_idx * 32 + min_tile_k * 16 + (lane_id >> 4) * 4;
                        acc_o[h_idx * ((WARP_M / 32)*(kBlockK / 32)) + k_idx * (WARP_M / 32)][min_tile_k * 2 + min_tile_m].f32 = *(vec4_fp32*)(acc_o_lds + lds_offset);
                    }
                }
            }
        }
    }
}