#include "numeric_types.h"


template<int REUSE_KV_TIMES, int K_LOOP_COUNT, int K_WARP_COUNT, int M_WARP_COUNT, int M_MMAC_COUNT, int WARP_NUM, int Padding, typename ElementAccum>
__forceinline__ __device__ void mla_acco_reduce_tile16x32(
        vec4_Accum < ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4],
        ElementAccum* acc_o_lds,
        int seqlen_q,
        int warp_id,
        int lane_id) {
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
}