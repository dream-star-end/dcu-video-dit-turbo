#pragma once
#include "intrinsic.h"
#include "fwd/utils.h"
#include "intrinsic_mls_ds.h"


template<int kBlockK, int WARP_N, typename Element, int STAGES, int WARP_NUM>
__forceinline__ __device__ void f16_mla_tp8_prefetch_k_to_lds_gfx938(
        vec4_uint k_addr,
        Element* k_lds,
        int warp_id,
        int kvcache_seqlen_stride,
        int max_seq_k_offset=0) {

    // 准备 MLS 寄存器
    vec4_uint k_srsrc;
    k_srsrc[1] = k_addr[1];
    k_srsrc[2] = kvcache_seqlen_stride;

    // pingpong buffer 的第一阶段
    int stage_id = 0;

    // kHeadDim 方向上的第几个 32x32 块
    int k_loop = 0;

    #pragma unroll
    for (int prefetch_id = 0; prefetch_id < 2; ++prefetch_id) {
        // 计算当前 wave 写到 lds 的起始地址
        int k_lds_stage_offset = (warp_id * STAGES * 2 + stage_id * 2 + prefetch_id) * (WARP_N / 32) * (kBlockK / 32) * (32 * 32);

        // 计算当前 wave 沿着 kHeadDim 方向循环读取的起始地址, 读到第几个 32x32 块了
        int k_mls_loop_global_offset = (k_loop + prefetch_id) * kBlockK * sizeof(Element);

        // 计算当前 wave 从 global 读取数据的起始地址
        int k_mls_warp_global_offset; // = warp_id * WARP_N * kvcache_seqlen_stride;

        if constexpr (true) {
            int nm_filter_max = warp_id * WARP_N + 32 - max_seq_k_offset; // 判断是否有 warp 取空数据
            int real_mls_warp_id = nm_filter_max >= 32 ? 0: warp_id;       // 如果取空数据, 938 不支持, 退化到取 warp 0 的数据
            k_mls_warp_global_offset = real_mls_warp_id * WARP_N * kvcache_seqlen_stride * sizeof(Element);
            int nm_filter = inline_min_max<0, 32>(real_mls_warp_id * WARP_N + 32 - max_seq_k_offset); // 如果取空数据, 使用 warp 0 的 nm_filter 值
            k_srsrc[3] = nm_filter << 8;
        }
        // 根据偏移计算 global load 的字节偏移数
        // k_srsrc[0] = k_addr[0] + k_mls_loop_global_offset + k_mls_warp_global_offset;
        *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_addr + k_mls_loop_global_offset + k_mls_warp_global_offset);
        int lds_offset_bytes = k_lds_stage_offset * 2/*half -> bytes*/;
        inline_matrix_load_32x32_b16_lds_trans<0, 0>(k_lds, k_srsrc, lds_offset_bytes, 0);
        __builtin_amdgcn_sched_barrier(0);
    }

}