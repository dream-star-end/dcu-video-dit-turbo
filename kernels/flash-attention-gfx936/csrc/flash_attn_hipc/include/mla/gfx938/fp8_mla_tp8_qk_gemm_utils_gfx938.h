#pragma once
#include "intrinsic.h"
#include "fwd/utils.h"
#include "intrinsic_mls_ds_b8.h"


template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockK, int WARP_M, int WARP_NUM, typename Element, typename ElementAccum, int STAGES, int M_MMAC_COUNT>
__forceinline__ __device__ void fp8_mla_tp8_prefetch_q_to_vgpr_gfx938_with_initialization(
        vec4_uint q_addr,
        Element* q_lds,
        union_vec16_fp8 q_reg[M_MMAC_COUNT][kHeadDim / 64],
        int warp_id,
        int q_row_stride,
        int max_seq_q_offset,
        vec2_Accum<ElementAccum> scores_max[WARP_M / 32],
        vec2_Accum<ElementAccum> scores_sum[WARP_M / 32],
        vec4_Accum<ElementAccum> acc_o[kHeadDimV / kBlockK][4]) {

    // 准备 MLS 寄存器
    vec4_uint q_srsrc;
    q_srsrc[0] = q_addr[0];
    q_srsrc[1] = q_addr[1];
    q_srsrc[2] = q_row_stride;
    q_srsrc[3] = 0;

    // 计算 lds 写入地址
    int q_lds_write_bytes = warp_id * 16 * 128 * sizeof(Element);

    // 计算 global 读取地址
    int q_mls_warp_global_offset = warp_id * 128 * sizeof(Element);
    *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_addr + q_mls_warp_global_offset);

    // mls 读取 16x128 bytes
    if constexpr (true) {
        int nm_filter = inline_min_max<0, 16>(16 - max_seq_q_offset);
        q_srsrc[3] = nm_filter << 8;
    }
    inline_matrix_load_128x16_b8_lds_trans<0, 1>(q_lds, q_srsrc, q_lds_write_bytes, 0);

    // add alu between def-use
    attention_initialize<kHeadDimV / kBlockK, WARP_M / 32, 1, M_MMAC_COUNT, ElementAccum>(scores_max, scores_sum, acc_o);

    // 等待 4 个 warp 数据写入 lds 完毕
    flash::wait_buffer_data_arrived<true/*sync*/>(0);

    // 从 lds 读取数据
    #pragma unroll
    for (int i = 0; i < WARP_NUM; ++i) {
        int q_lds_load_offset = reinterpret_cast<size_t>(q_lds) + (i * 16 * 128) * sizeof(Element);
        DS_READ_MATRIX_64x16_B8(q_lds_load_offset,        q_reg[0][i * 2].i32x4, true/*transpose*/)
        DS_READ_MATRIX_64x16_B8(q_lds_load_offset + 1024, q_reg[0][i * 2 + 1].i32x4, true/*transpose*/)
    }
    __builtin_amdgcn_sched_barrier(0);

    // 接着读取剩下的 16x64
    // =====================================================================================================
    if (warp_id == 0) {
        // [RTL bug] MLS 128B 请求指令使用 m_filter 需要限制起始地址和 stride 都是 128B 对齐, 否则在访问矩阵最后一行末尾时, 若地址跨越 64B, 一定概率跨越了页表, 导致 invalid address
        *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_addr + ((WARP_NUM - 1) * 128 + 64) * sizeof(Element));
        inline_matrix_load_128x16_b8_lds_trans<0, 1>(q_lds + 16384, q_srsrc, q_lds_write_bytes, 0);
        // 等待数据写到 lds
        flash::wait_buffer_data_arrived<false/*sync*/>(0);
    }
    // sync
    flash::wait_all_warp_arrived();
    // 每个 warp 读取 16x64 的内容
    int q_lds_load_offset = reinterpret_cast<size_t>(q_lds + 16384) * sizeof(Element);
    DS_READ_MATRIX_64x16_B8(q_lds_load_offset + 1024, q_reg[0][8].i32x4, true/*transpose*/)

    // 同步, 等待数据写到寄存器, 同时防止 lds 被新的 mls 指令写入
    flash::wait_lds_data_arrived<true/*sync*/>(0);

}


template<int WARP_NUM, typename Element>
__forceinline__ __device__ void fp8_mla_tp8_prefetch_k_gfx938(
        vec4_uint k_addr,
        Element* k_lds,
        int warp_id,
        int k_row_stride,
        int max_seq_k_offset=0) {

    int stage_id = 0;

    // 准备 MLS resource 寄存器
    vec4_uint k_srsrc;
    k_srsrc[1] = k_addr[1];
    k_srsrc[2] = k_row_stride;

    {
        constexpr int k_loop = 0;

        // lds 的写入地址
        int warp_lds_write_bytes = (stage_id * WARP_NUM + warp_id) * 32 * 64 * sizeof(Element);

        // global 随着 warp 的地址偏移
        int warp_global_bytes; // = warp_id * 32 * k_row_stride * sizeof(Element);

        // global 随着 k_loop 的地址偏移
        int k_loop_global_bytes = k_loop * 64 * sizeof(Element);

        // 计算边界
        if constexpr (true) {
            int nm_filter_max = warp_id * 32 + 32 - max_seq_k_offset; // 判断是否有 warp 取空数据
            int real_mls_warp_id = nm_filter_max >= 32 ? 0: warp_id;  // 如果取空数据, 938 不支持, 退化到取 warp 0 的数据
            warp_global_bytes = real_mls_warp_id * 32 * k_row_stride * sizeof(Element);
            int nm_filter = inline_min_max<0, 32>(real_mls_warp_id * 32 + 32 - max_seq_k_offset); // 如果取空数据, 使用 warp 0 的 nm_filter 值
            k_srsrc[3] = nm_filter << 8;
            k_srsrc[3] += 0x40000;
        }
        *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_addr + warp_global_bytes + k_loop_global_bytes);
        inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(k_lds, k_srsrc, warp_lds_write_bytes, 0);
        __builtin_amdgcn_sched_barrier(0);
    }
}