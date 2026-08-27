#pragma once
#include "fp8_mla_tp8_pv_gemm_prefetch_k_gfx938.h"


template<int K_LOOP_COUNT, int kBlockN, int WARP_NUM, typename V_Element>
__forceinline__ __device__ void fp8_mla_tp8_prefetch_v_gfx938(
        vec4_uint v_addr,
        V_Element* v_lds,
        int warp_id,
        int v_row_stride,
        int max_seq_v_offset=0) {
    static_assert (K_LOOP_COUNT % 2 == 0);
    constexpr int K_LOOP_COUNT_ = K_LOOP_COUNT / (64 / kBlockN);
    constexpr int PREFETCH = 2;

    // 防止与多 wave reduce max 需要的 lds 冲突
    flash::wait_lds_data_arrived<true/*sync*/>(0);

    // 准备 MLS 的 resource 寄存器
    vec4_uint v_srsrc;
    v_srsrc[1] = v_addr[1];
    v_srsrc[2] = v_row_stride;

    // pingpong
    int stage_id = 0;

    {
        int k_loop = K_LOOP_COUNT_ - 1;
        #pragma unroll
        for (int load_id = 0; load_id < PREFETCH; ++load_id) {
            // 准备读取 V 32x64 个 fp8
            // lds 的写入地址
            int warp_lds_write_bytes = stage_id * 16384 + (WARP_NUM * load_id + warp_id) * 32 * 64 * sizeof(V_Element);

            // global 随着 warp 的地址偏移
            int warp_global_bytes; // = warp_id * 32 * v_row_stride * sizeof(V_Element);

            // global 随着 k_loop 的地址偏移
            int v_loop_global_bytes = (k_loop - load_id) * 64 * sizeof(V_Element);

            // 计算边界
            if constexpr (true) {
                int nm_filter_max = warp_id * 32 + 32 - max_seq_v_offset; // 判断是否有 warp 取空数据
                int real_mls_warp_id = nm_filter_max >= 32 ? 0: warp_id;  // 如果取空数据, 938 不支持, 退化到取 warp 0 的数据
                warp_global_bytes = real_mls_warp_id * 32 * v_row_stride * sizeof(V_Element);
                int nm_filter = inline_min_max<0, 32>(real_mls_warp_id * 32 + 32 - max_seq_v_offset); // 如果取空数据, 使用 warp 0 的 nm_filter 值
                v_srsrc[3] = nm_filter << 8;
                v_srsrc[3] += 0x20000;
            }
            *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_addr + warp_global_bytes + v_loop_global_bytes);
            inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(v_lds, v_srsrc, warp_lds_write_bytes, 0);
        }
    }
}
