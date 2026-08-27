#pragma once
#include "numeric_types.h"


template<int K_LOOP_COUNT, int K_WARP_COUNT, int M_WARP_COUNT, int M_MMAC_COUNT, int WARP_NUM, int Padding, typename ElementAccum>
__forceinline__ __device__ void kvcache_acco_reduce_tile16x32(
        vec4_Accum < ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4],
        ElementAccum* acc_o_lds,
        int seqlen_q,
        int warp_id,
        int lane_id) {
    #if defined(__gfx938__) || defined(__gfx946__)
        constexpr int OPT_FOR_HDIM128 = bool(WARP_NUM == 4 and M_MMAC_COUNT == 1 and Padding == 0 and K_LOOP_COUNT == WARP_NUM and K_WARP_COUNT == 1 and M_WARP_COUNT == 1); // Specialized optimization for headdim 128
    #else
        constexpr int OPT_FOR_HDIM128 = false; // keep same as origin for archs <= gfx936
    #endif
    if constexpr (OPT_FOR_HDIM128) {
        // #######################################################################################################################################
        //                                                    bank-conflicts free path, higher performance
        // #######################################################################################################################################
        constexpr int PREFETCH = WARP_NUM;
        #pragma unroll
        for (int k_loop = 0; k_loop < K_LOOP_COUNT; k_loop += PREFETCH) {
            #pragma unroll
            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #pragma unroll
                for (int prefetch = 0; prefetch < PREFETCH; ++prefetch) {
                    vec4_fp32 f32x4 = acc_o[k_loop + prefetch][min_tile_n * 2].f32;
                    int lds_write_offset = warp_id * 2048 + prefetch * 2 * 16 * 16 + min_tile_n * 16 * 16;
                    lds_write_offset = reinterpret_cast<size_t>(acc_o_lds + lds_write_offset + lane_id * 4);
                    inlineasm_ds_write_b128(lds_write_offset, f32x4);
                }
            }
            union_vec4_fp32 data[2][WARP_NUM];
            constexpr int ds_bursts = PREFETCH;
            {
                constexpr int min_tile_n = 0;
                flash::wait_lds_data_arrived<true>((1 - min_tile_n) * PREFETCH);
                #pragma unroll
                for (int neighbor = 0; neighbor < PREFETCH; ++neighbor) {
                    int lds_read_offset = reinterpret_cast<size_t>(acc_o_lds + neighbor * 2048 + warp_id * 2 * 16 * 16 + min_tile_n * 16 * 16 + lane_id * 4);
                    inlineasm_ds_read_b128(lds_read_offset, data[min_tile_n][neighbor].f32);
                }
                inline_vgpr2_init_zero(acc_o[k_loop + 0][min_tile_n * 2].b64[0]);
                inline_vgpr2_init_zero(acc_o[k_loop + 0][min_tile_n * 2].b64[1]);
            }
            {
                constexpr int min_tile_n = 1;
                flash::wait_lds_data_arrived<true>((1 - min_tile_n) * PREFETCH + ds_bursts);
                #pragma unroll
                for (int neighbor = 0; neighbor < PREFETCH; ++neighbor) {
                    int lds_read_offset = reinterpret_cast<size_t>(acc_o_lds + neighbor * 2048 + warp_id * 2 * 16 * 16 + min_tile_n * 16 * 16 + lane_id * 4);
                    inlineasm_ds_read_b128(lds_read_offset, data[min_tile_n][neighbor].f32);
                }
                inline_vgpr2_init_zero(acc_o[k_loop + 0][min_tile_n * 2].b64[0]);
                inline_vgpr2_init_zero(acc_o[k_loop + 0][min_tile_n * 2].b64[1]);
            }
            {
                constexpr int min_tile_n = 0;
                #pragma unroll
                for (int neighbor = 0; neighbor < PREFETCH; ++neighbor) {
                    flash::wait_lds_data_arrived<false>(ds_bursts - 1 - neighbor + ds_bursts);
                    inline_v_pk_add_f32(acc_o[k_loop + 0][min_tile_n * 2].u64[0], acc_o[k_loop + 0][min_tile_n * 2].u64[0], data[min_tile_n][neighbor].u64[0]);
                    inline_v_pk_add_f32(acc_o[k_loop + 0][min_tile_n * 2].u64[1], acc_o[k_loop + 0][min_tile_n * 2].u64[1], data[min_tile_n][neighbor].u64[1]);
                }
            }
            {
                constexpr int min_tile_n = 1;
                #pragma unroll
                for (int neighbor = 0; neighbor < PREFETCH; ++neighbor) {
                    flash::wait_lds_data_arrived<false>(ds_bursts - 1 - neighbor);
                    inline_v_pk_add_f32(acc_o[k_loop + 0][min_tile_n * 2].u64[0], acc_o[k_loop + 0][min_tile_n * 2].u64[0], data[min_tile_n][neighbor].u64[0]);
                    inline_v_pk_add_f32(acc_o[k_loop + 0][min_tile_n * 2].u64[1], acc_o[k_loop + 0][min_tile_n * 2].u64[1], data[min_tile_n][neighbor].u64[1]);
                }
            }
            flash::wait_all_warp_arrived();
        }
    } else {
        constexpr int kBlockK = K_WARP_COUNT * 32 + Padding;

        int EVEN_REUSE_KV_TIMES = ((seqlen_q + 1) / 2) * 2;

        int q_seq_idx = (lane_id & 15);
        if (q_seq_idx < EVEN_REUSE_KV_TIMES) {
            for (int h_idx = 0; h_idx < K_LOOP_COUNT; ++h_idx) {
            // ####################################################################################################################################################
            for (int k_idx = 0; k_idx < K_WARP_COUNT; ++k_idx) {
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        // 一个 wave 共同持有 seqlen_q x kHeadDim 个 Half, 但为了节省 lds 用量, 每次只 reduce seqlen_q x kBlockK 个 Half
                        int lds_offset = (warp_id * EVEN_REUSE_KV_TIMES * M_MMAC_COUNT + q_seq_idx + min_tile_m * 16) * kBlockK + k_idx * 32 + min_tile_n * 16 + (lane_id >> 4/*0~3*/) * 4/*0~15*/;
                        int tile_32x32_id = h_idx * M_WARP_COUNT * K_WARP_COUNT + k_idx * M_WARP_COUNT;
                        *(vec4_fp32*)(acc_o_lds + lds_offset) = acc_o[tile_32x32_id][min_tile_n * 2 + min_tile_m].f32;
                    }
                }
            }
            __syncthreads();
            // 在 lds 中求和, 把 4 个 wave 写的 acc_o 的数据加起来
            if constexpr (WARP_NUM == 4) {
                for (int k_idx = 0; k_idx < K_WARP_COUNT; ++k_idx) {
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            int lds_offset = (q_seq_idx + min_tile_m * 16) * kBlockK + k_idx * 32 + min_tile_n * 16 + (lane_id >> 4) * 4 + warp_id; // 之前是一次性写了 4 个 Half 到 lds, 现在 4 个 wave 分别处理这 4 个位置的 acc_o reduce
                            float acc_tmp_wave0 = acc_o_lds[lds_offset];
                            for (int loop = 1; loop < WARP_NUM; ++loop) {
                                acc_tmp_wave0 += acc_o_lds[lds_offset + loop * EVEN_REUSE_KV_TIMES * M_MMAC_COUNT * kBlockK];
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
                                    int lds_offset = (q_seq_idx + min_tile_m * 16) * kBlockK + k_idx * 32 + min_tile_n * 16 + (lane_id >> 4) * 4 + vec_idx;
                                    float acc_tmp_wave0 = acc_o_lds[lds_offset];
                                    for (int loop = 1; loop < WARP_NUM; ++loop) {
                                        acc_tmp_wave0 += acc_o_lds[lds_offset + loop * EVEN_REUSE_KV_TIMES * M_MMAC_COUNT * kBlockK];
                                    }
                                    acc_o_lds[lds_offset] = acc_tmp_wave0;
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
            // 每个 wave 都从 LDS 获取最终的求和结果
            for (int k_idx = 0; k_idx < K_WARP_COUNT; ++k_idx) {
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        int lds_offset = (q_seq_idx + min_tile_m * 16) * kBlockK + k_idx * 32 + min_tile_n * 16 + (lane_id >> 4) * 4;
                        int tile_32x32_id = h_idx * M_WARP_COUNT * K_WARP_COUNT + k_idx * M_WARP_COUNT;
                        acc_o[tile_32x32_id][min_tile_n * 2 + min_tile_m].f32 = *(vec4_fp32*)(acc_o_lds + lds_offset);
                    }
                }
            }
            __syncthreads();
        }
    }
    }
}
