#pragma once // prepare for prefetch V in qk gemm
#include "intrinsic_mls_ds.h"
template<bool Is_even_MN, int kBlockN, int kHeadDim, int WARP_N, typename Element>
__forceinline__ __device__ void fp8_prefetch_v_to_lds(
    Element* v_ptr,
    int8_t* v_lds,
    int warp_id,
    int v_row_stride,
    int max_seq_kv_offset
) {
    static_assert(kHeadDim == 128 || kHeadDim == 256);
    // 准备 MLS 寄存器, 填充 stride
    vec4_uint v_root = prepare_for_matrix_load<kHeadDim, Element>(v_ptr);
    vec4_uint v_srsrc;
    v_srsrc[0] = v_root[0];
    v_srsrc[1] = v_root[1];
    v_srsrc[2] = v_row_stride; // stride
    v_srsrc[3] = 0x00000;

    // 4 个 wave 直接全量预取
    int v_lds_write_bytes = warp_id * WARP_N * kHeadDim * sizeof(Element);
    // 每次读取 32x128 的数据

    // tile1: 行 [warp_id*32, warp_id*32+16)
    // 整个 tile 被 filter 时保留一行合法 V，避免 0 * NaN 污染 PV。
    int nm_filter_warp0_tile1 = inline_min_max<0, 16>(16 - max_seq_kv_offset);
    int nm_filter = inline_min_max<0, 16>(32 * warp_id + 16 - max_seq_kv_offset);
    v_srsrc[0] = (nm_filter == 16) ? v_root[0] : v_root[0] + (warp_id * 2) * 16 * v_row_stride * sizeof(Element);
    nm_filter = (nm_filter == 16) ? min(nm_filter_warp0_tile1, 15) : nm_filter;
    v_srsrc[3] = v_srsrc[3] + ((max_seq_kv_offset % 128 == 0) ? 0: (nm_filter << 8));
    flash::wait_all_warp_arrived();
    __builtin_hcu_matrix_load_128X16_b8(v_srsrc, v_lds+v_lds_write_bytes, 0, true, false, false, false, false);
    if constexpr (kHeadDim == 256) {
        v_srsrc[0] += 128 * sizeof(Element);
        __builtin_hcu_matrix_load_128X16_b8(v_srsrc, v_lds + v_lds_write_bytes + 4096, 0, true, false, false, false, false);
    }

    // tile2: 行 [warp_id*32+16, warp_id*32+32)
    int nm_filter_warp0_tile2 = inline_min_max<0, 16>(32 - max_seq_kv_offset);
    nm_filter = inline_min_max<0, 16>(32 * warp_id + 32 - max_seq_kv_offset);
    v_srsrc[0] = (nm_filter == 16) ? v_root[0] : v_root[0] + (warp_id * 2 + 1) * 16 * v_row_stride * sizeof(Element);
    nm_filter = (nm_filter == 16) ? min(nm_filter_warp0_tile2, 15) : nm_filter;
    v_srsrc[3] = ((max_seq_kv_offset % 128 == 0) ? 0: (nm_filter << 8));
    __builtin_hcu_matrix_load_128X16_b8(v_srsrc, v_lds+v_lds_write_bytes + (128 * 16 >> 1), 0, true, false, false, false, false);
    if constexpr (kHeadDim == 256) {
        v_srsrc[0] += 128 * sizeof(Element);
        __builtin_hcu_matrix_load_128X16_b8(v_srsrc, v_lds + v_lds_write_bytes + 4096 + (128 * 16 >> 1), 0, true, false, false, false, false);
    }
}
