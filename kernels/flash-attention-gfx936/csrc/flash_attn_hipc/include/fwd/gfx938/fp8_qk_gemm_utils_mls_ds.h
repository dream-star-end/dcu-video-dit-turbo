#pragma once
#include "intrinsic_mls_ds.h"
#include "intrinsic_mls_ds_b8.h"
#define USE_MLS_128B_REQUEST

template<bool Is_even_MN, int kHeadDim, int WARP_M, typename Element>
__forceinline__ __device__ void fp8_prefetch_q_to_lds(
    Element* q_ptr,
    int8_t* q_lds,
    int warp_id,
    int q_row_stride,
    int max_seq_q_offset
) {
    static_assert(kHeadDim == 128 || kHeadDim == 192 || kHeadDim == 256);
    // 准备 MLS 寄存器, 填充 stride
    vec4_uint q_root = prepare_for_matrix_load<128, Element>(q_ptr);
    vec4_uint q_srsrc;
    q_srsrc[0] = q_root[0];
    q_srsrc[1] = q_root[1];
    q_srsrc[2] = q_row_stride; // stride
    q_srsrc[3] = 0x40000; // [17: 18], interleave 4

    // 计算 lds 写入地址
    constexpr int kLdsHeadDimStride = kHeadDim == 192 ? 256 : kHeadDim;
    int q_lds_offset = warp_id * WARP_M * kLdsHeadDimStride * sizeof(Element);
    int q_lds_write_bytes = reinterpret_cast<size_t>(q_lds) + q_lds_offset;

    const bool q_need_filter = max_seq_q_offset > 0 && max_seq_q_offset < 128;

    // 启动 mls 读取
    #ifdef USE_MLS_128B_REQUEST
        // inline_matrix_load_128x32_b8_lds_rearrange<0, 1>(q_lds, q_srsrc, q_lds_offset/*lds bytes*/, 0/*matrix_offset, 0 or 16*/);
        int nm_filter = inline_min_max<0, 16>(32 * warp_id + 16 - max_seq_q_offset);
        *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_root + ((q_need_filter && nm_filter == 16)
            ? 0
            : int64_t(warp_id) * WARP_M * int64_t(q_row_stride) * sizeof(Element)));
        q_srsrc[3] = 0x40000 + (q_need_filter ? (nm_filter << 8) : 0);
        __builtin_hcu_matrix_load_128X16_b8(q_srsrc, q_lds+q_lds_offset, 0, true, false, false, false, false);
        nm_filter = inline_min_max<0, 16>(32 * warp_id + 32 - max_seq_q_offset);
        *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_root + ((q_need_filter && nm_filter == 16)
            ? 0
            : int64_t(warp_id) * WARP_M * int64_t(q_row_stride) * sizeof(Element)));
        q_srsrc[3] = 0x40000 + (q_need_filter ? (nm_filter << 8) : 0);
        __builtin_hcu_matrix_load_128X16_b8(q_srsrc, q_lds+q_lds_offset+512, 16, true, false, false, false, false);
        if constexpr (kHeadDim > 128) {
            constexpr int kSecondMlsLoadHeadOffset = kHeadDim == 192 ? 64 : 128;
            nm_filter = inline_min_max<0, 16>(32 * warp_id + 16 - max_seq_q_offset);
            *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_root + (((q_need_filter && nm_filter == 16)
                ? 0
                : int64_t(warp_id) * WARP_M * int64_t(q_row_stride) * sizeof(Element))
                + int64_t(kSecondMlsLoadHeadOffset) * sizeof(Element)));
            q_srsrc[3] = 0x40000 + (q_need_filter ? (nm_filter << 8) : 0);
            __builtin_hcu_matrix_load_128X16_b8(q_srsrc, q_lds+q_lds_offset+4096, 0, true, false, false, false, false);
            nm_filter = inline_min_max<0, 16>(32 * warp_id + 32 - max_seq_q_offset);
            *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_root + (((q_need_filter && nm_filter == 16)
                ? 0
                : int64_t(warp_id) * WARP_M * int64_t(q_row_stride) * sizeof(Element))
                + int64_t(kSecondMlsLoadHeadOffset) * sizeof(Element)));
            q_srsrc[3] = 0x40000 + (q_need_filter ? (nm_filter << 8) : 0);
            __builtin_hcu_matrix_load_128X16_b8(q_srsrc, q_lds+q_lds_offset+4608, 16, true, false, false, false, false);
        }
    #else
        inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(q_lds, q_srsrc, q_lds_write_bytes/*lds bytes*/, 0/*matrix_offset, 0 or 16*/);
        *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_root + 64 * sizeof(Element));
        inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(q_lds, q_srsrc, q_lds_write_bytes + 2048/*lds bytes*/, 0/*matrix_offset, 0 or 16*/); // Q 部分可以考虑 128x16 或者非 4-interleave 形式
    #endif
    __builtin_amdgcn_sched_barrier(0);
}



// #define USE_DS_READ_B128_FOR_INTERLEAVE4

template<int kHeadDim, int WARP_M, typename Element>
__forceinline__ __device__ void load_q_from_lds_to_vgpr(
    union_vec16_fp8 q_regs[WARP_M / 16][kHeadDim / 64],
    int8_t* q_lds,
    int warp_id,
    int lane_id
) {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)\n");
    __builtin_amdgcn_sched_barrier(0);
    static_assert(kHeadDim == 128 || kHeadDim == 192 || kHeadDim == 256);
    constexpr int kLdsHeadDimStride = kHeadDim == 192 ? 256 : kHeadDim;
    // lds 写到两个地方去了, 注意是 rearrange, 所以跳 1K; transpose 跳 2K
    // MLS0: [0: 512) 和 [1024, 1536)
    // MLS1: [512: 1024) 和 [1536, 2048)
    // 分 4 次读取寄存器, 第一次是 [0, 1024), 即 16x64 的内容, 每个线程读取 16 个 fp8
    #ifdef USE_DS_READ_B128_FOR_INTERLEAVE4
        int row = (lane_id & 15) >> 1;
        int col = lane_id >> 4;
        int col_swizzle = (row + col) & 3;
        int lds_load_offset = row * 128 + col_swizzle * 16 + (lane_id & 1) * 64 + warp_id * WARP_M * kHeadDim;
        q_regs[0][0].i32x4 = *(vec4_int32*)(q_lds + lds_load_offset + 0);
        q_regs[1][0].i32x4 = *(vec4_int32*)(q_lds + lds_load_offset + 1024/*ds fmt 0, dmft1 */);
        q_regs[0][1].i32x4 = *(vec4_int32*)(q_lds + lds_load_offset + 2048/*ds fmt 0, dmft1 */);
        q_regs[1][1].i32x4 = *(vec4_int32*)(q_lds + lds_load_offset + 3072/*ds fmt 0, dmft1 */);
    #else
        #pragma unroll
        for (int h_idx = 0; h_idx < kHeadDim / 64; ++h_idx) {
            const int h_offset = (kHeadDim == 192 && h_idx == 2) ? 3 * 2048 : h_idx * 2048;
            q_regs[0][h_idx].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(q_lds + h_offset + 0 + warp_id * WARP_M * kLdsHeadDimStride, 0, 3, 1, 0);
            q_regs[1][h_idx].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(q_lds + h_offset + 1024 + warp_id * WARP_M * kLdsHeadDimStride, 0, 3, 1, 0);
        }
    #endif
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();
    __builtin_amdgcn_sched_barrier(0);
}



template<bool Is_even_MN, int kHeadDim, int WARP_N, typename Element>
__forceinline__ __device__ void fp8_prefetch_k_to_lds(
    Element* k_ptr,
    int8_t* k_lds,
    int warp_id,
    int k_row_stride,
    int max_seq_kv_offset
) {
    static_assert(kHeadDim == 128 || kHeadDim == 192 || kHeadDim == 256);
    // 准备 MLS 寄存器, 填充 stride
    vec4_uint k_root = prepare_for_matrix_load<kHeadDim, Element>(k_ptr);
    vec4_uint k_srsrc;
    k_srsrc[0] = k_root[0];
    k_srsrc[1] = k_root[1];
    k_srsrc[2] = k_row_stride; // stride
    k_srsrc[3] = 0x40000; // [17: 18], interleave 4

    // 计算 lds 写入地址
    constexpr int kLdsHeadDimStride = kHeadDim == 192 ? 256 : kHeadDim;
    int k_lds_offset = warp_id * WARP_N * kLdsHeadDimStride * sizeof(Element);
    int k_lds_write_bytes = reinterpret_cast<size_t>(k_lds) + k_lds_offset;

    const bool k_need_filter = max_seq_kv_offset > 0 && max_seq_kv_offset < 128;

    // 同步所有warp，确保srsrc参数准备完毕后再发起MLS load
    flash::wait_all_warp_arrived();

    // 启动 mls 读取
    #ifdef USE_MLS_128B_REQUEST
        int nm_filter = inline_min_max<0, 16>(32 * warp_id + 16 - max_seq_kv_offset);
        *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_root + ((k_need_filter && nm_filter == 16)
            ? 0
            : int64_t(warp_id) * 32 * int64_t(k_row_stride) * sizeof(Element)));
        k_srsrc[3] = 0x40000 + (k_need_filter ? (nm_filter << 8) : 0);
        __builtin_hcu_matrix_load_128X16_b8(k_srsrc, k_lds+k_lds_offset, 0, true, false, false, false, false);
        nm_filter = inline_min_max<0, 16>(32 * warp_id + 32 - max_seq_kv_offset);
        *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_root + ((k_need_filter && nm_filter == 16)
            ? 0
            : int64_t(warp_id) * 32 * int64_t(k_row_stride) * sizeof(Element)));
        k_srsrc[3] = 0x40000 + (k_need_filter ? (nm_filter << 8) : 0);
        __builtin_hcu_matrix_load_128X16_b8(k_srsrc, k_lds+k_lds_offset+512, 16, true, false, false, false, false);
        if constexpr (kHeadDim > 128) {
            constexpr int kSecondMlsLoadHeadOffset = kHeadDim == 192 ? 64 : 128;
            nm_filter = inline_min_max<0, 16>(32 * warp_id + 16 - max_seq_kv_offset);
            *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_root + (((k_need_filter && nm_filter == 16)
                ? 0
                : int64_t(warp_id) * 32 * int64_t(k_row_stride) * sizeof(Element))
                + int64_t(kSecondMlsLoadHeadOffset) * sizeof(Element)));
            k_srsrc[3] = 0x40000 + (k_need_filter ? (nm_filter << 8) : 0);
            __builtin_hcu_matrix_load_128X16_b8(k_srsrc, k_lds+k_lds_offset+4096, 0, true, false, false, false, false);
            nm_filter = inline_min_max<0, 16>(32 * warp_id + 32 - max_seq_kv_offset);
            *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_root + (((k_need_filter && nm_filter == 16)
                ? 0
                : int64_t(warp_id) * 32 * int64_t(k_row_stride) * sizeof(Element))
                + int64_t(kSecondMlsLoadHeadOffset) * sizeof(Element)));
            k_srsrc[3] = 0x40000 + (k_need_filter ? (nm_filter << 8) : 0);
            __builtin_hcu_matrix_load_128X16_b8(k_srsrc, k_lds+k_lds_offset+4608, 16, true, false, false, false, false);
        }
    #else
        inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(k_lds, k_srsrc, k_lds_write_bytes/*lds bytes*/, 0/*matrix_offset, 0 or 16*/);
        *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_root + 64 * sizeof(Element));
        inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(k_lds, k_srsrc, k_lds_write_bytes + 2048/*lds bytes*/, 0/*matrix_offset, 0 or 16*/);
    #endif
}
