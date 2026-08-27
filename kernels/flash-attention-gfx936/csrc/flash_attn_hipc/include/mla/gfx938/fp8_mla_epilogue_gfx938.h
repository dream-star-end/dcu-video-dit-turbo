#pragma once
#include "numeric_types.h"


template<int K_LOOP_COUNT, int M_WARP_COUNT, int K_WARP_COUNT, int M_MMAC_COUNT, typename ElementAccum>
__forceinline__ __device__ void fp8_mla_epilugue_rescale_acco_gfx938(
    vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4],
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT],
    ElementAccum v_descale) {
    #pragma unroll
    for (int pv_n_loop = 0; pv_n_loop < K_LOOP_COUNT; ++pv_n_loop) {
        #pragma unroll
        for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < K_WARP_COUNT; ++ni) {
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    ElementAccum sum     = scores_sum[mi].f32[min_tile_m];
                    ElementAccum inv_sum = (sum == 0.f || sum != sum) ? v_descale : v_descale / sum;
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