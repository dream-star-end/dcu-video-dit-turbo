#pragma once // prepare for prefetch V in qk gemm
#include "intrinsic.h"
#include "fwd/utils.h"
#include "intrinsic_mls_ds.h"


template<int kHeadDim, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int WARP_K, int stage_id, int WARP_NUM, typename Element, int STAGES>
__forceinline__ __device__ void f16_mla_tp8_prefetch_v_to_lds_gfx938(
        vec4_uint v_addr,
        Element* v_lds,
        int warp_id,
        int kvcache_seqlen_stride,
        int max_seq_kv_offset=0) {

    constexpr int V_LOAD_REQUESTS = (WARP_K * kBlockN) / (32 * 32);
    constexpr int N_LOOP_STEP     = 2;

    // 准备 MLS 的 resource 寄存器
    vec4_uint v_srsrc;
    v_srsrc[1] = v_addr[1];
    v_srsrc[2] = kvcache_seqlen_stride; // stride

    // 从倒数第 2 个 block 开始读取
    int n_loop = kHeadDim / kBlockN - N_LOOP_STEP;

    #pragma unroll
    for (int prefetch_id = 0; prefetch_id < N_LOOP_STEP; ++prefetch_id) {

        // 计算当前 wave 当前加载的 32x32 block 的偏移字节数
        int v_mls_warp_global_offset = (n_loop + prefetch_id) * kBlockN * sizeof(Element);

        // 计算当前 wave 写入 lds 的偏移地址(注意 v_lds 相较于 smem 的偏移量)
        int v_mls_lds_warp_offset = (warp_id * STAGES * 2 + stage_id * 2 + prefetch_id) * (V_LOAD_REQUESTS * 32 * 32) * sizeof(Element);

        // 计算当前 wave 读取数据的起始偏移字节数
        int v_mls_loop_global_offset;// = warp_id * WARP_K * kvcache_seqlen_stride * sizeof(Element);

        // 计算 MLS 读取数据的 global 地址, 判断边界
        if constexpr (true) {
            int nm_filter_max = warp_id * WARP_K + 32 - max_seq_kv_offset; // 判断是否有 warp 取空数据
            int real_mls_warp_id = nm_filter_max >= 32 ? 0: warp_id;       // 如果取空数据, 938 不支持, 退化到取 warp 0 的数据
            v_mls_loop_global_offset = real_mls_warp_id * WARP_K * kvcache_seqlen_stride * sizeof(Element);
            int nm_filter = inline_min_max<0, 32>(real_mls_warp_id * WARP_K + 32 - max_seq_kv_offset); // 如果取空数据, 使用 warp 0 的 nm_filter 值
            v_srsrc[3] = max_seq_kv_offset % kBlockN == 0 ? 0: nm_filter << 8;
            v_srsrc[3] += 0x20000;
        }
        // v_srsrc[0] = v_addr[0] + v_mls_loop_global_offset + v_mls_warp_global_offset;
        *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_addr + v_mls_loop_global_offset + v_mls_warp_global_offset);
        __builtin_amdgcn_sched_barrier(0);
        inline_matrix_load_32x32_b16_lds<0, 1>(v_lds, v_srsrc, v_mls_lds_warp_offset, 0);
        __builtin_amdgcn_sched_barrier(0);
    }
}