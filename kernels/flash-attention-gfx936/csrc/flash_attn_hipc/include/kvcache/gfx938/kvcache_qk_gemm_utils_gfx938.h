#pragma once
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "static_switch.h"
#include "kvcache_pv_gemm_utils_gfx938.h"
#include "intrinsic_mls_ds_b8.h"


template<int kHeadDim, int kBlockM, int kBlockK, int WARP_M, int WARP_NUM, typename Element, int STAGES, int M_MMAC_COUNT>
__forceinline__ __device__ void kvcache_prefetch_q_to_vgpr_gfx938(
        vec4_uint q_addr,
        Element* q_lds,
        union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2],
        int warp_id,
        int query_seqlen_stride,
        int max_seq_q_offset=0) {

    if constexpr (kHeadDim == 128 and WARP_NUM == 4) {
        // 准备 MLS 寄存器
        vec4_uint q_srsrc;
        q_srsrc[1] = q_addr[1];
        q_srsrc[2] = query_seqlen_stride;

        // kHeadDim 方向上的第几个 32x32 块
        int q_loop = 0;

        // 计算当前 wave 写到 lds 的起始地址
        int k_lds_stage_offset = warp_id * (WARP_M / 32) * (kBlockK / 32) * (32 * 32);

        // 计算当前 wave 从 global 读取数据的起始地址
        int k_mls_warp_global_offset = warp_id * kBlockK;

        // 计算当前 wave 沿着 kHeadDim 方向循环读取的起始地址, 读到第几个 32x32 块了
        int k_mls_loop_global_offset = q_loop * kBlockK;

        // 根据偏移计算 global load 的字节偏移数
        q_srsrc[0] = q_addr[0] + (k_mls_loop_global_offset + k_mls_warp_global_offset ) * 2;
        if constexpr (true) {
            int nm_filter = inline_min_max<0, 32>(32 - max_seq_q_offset);
            q_srsrc[3] = max_seq_q_offset % 32 == 0 ? 0: nm_filter << 8; // set only once
        }
        int lds_offset_bytes = k_lds_stage_offset * 2/*half -> bytes*/;
        flash::wait_lds_data_arrived<true>(0);
        inline_matrix_load_32x32_b16_lds_trans<0, 0>(q_lds, q_srsrc, lds_offset_bytes, 0);

        flash::wait_buffer_data_arrived<true>(0);

        // 开始读取数据
        __builtin_amdgcn_sched_barrier(0);
        // 注意 M_MMAC_COUNT = 1 的时候只需要读一次
        if constexpr (M_MMAC_COUNT == 1) {
            DS_READ_MATRIX_32X16_B16(0 * 32 * 32 * 2, q_reg[0 * 2].f16, true);
            DS_READ_MATRIX_32X16_B16(1 * 32 * 32 * 2, q_reg[1 * 2].f16, true);
            DS_READ_MATRIX_32X16_B16(2 * 32 * 32 * 2, q_reg[2 * 2].f16, true);
            DS_READ_MATRIX_32X16_B16(3 * 32 * 32 * 2, q_reg[3 * 2].f16, true);
        } else {
            DS_READ_MATRIX_32X32_B16(0 * 32 * 32 * 2, q_reg[0 * 2].f16, q_reg[0 * 2 + 1].f16, true);
            DS_READ_MATRIX_32X32_B16(1 * 32 * 32 * 2, q_reg[1 * 2].f16, q_reg[1 * 2 + 1].f16, true);
            DS_READ_MATRIX_32X32_B16(2 * 32 * 32 * 2, q_reg[2 * 2].f16, q_reg[2 * 2 + 1].f16, true);
            DS_READ_MATRIX_32X32_B16(3 * 32 * 32 * 2, q_reg[3 * 2].f16, q_reg[3 * 2 + 1].f16, true);
        }
        flash::wait_lds_data_arrived<true>(0);
    }
    else {
        constexpr int Q_LOAD_REQUESTS = (kBlockM * kBlockK >> 1/*16x32 tile*/) * M_MMAC_COUNT / (4 * 32 * WARP_NUM);
        constexpr int SEQUENCE_READ   = M_MMAC_COUNT;
        constexpr int READ_ONCE_LINES = 4;
        auto BUFFER_LOAD_FUNC         = &builtin_buffer_load_dword_lds<Element, float, 1>; // buffer_load_dwordx4 can also be applied if necessary

        int lane_id             = threadIdx.x & 63; // lane id, 0-63
        int q_lane_m_idx        = lane_id >> 4;
        int q_lane_head_dim_idx = lane_id & 15;
        int laneid_shfl_4       = lane_id >> 4;
        int laneid_and_15       = lane_id & 15;
        int q_ds_read_offset    = laneid_and_15 * 16 + laneid_shfl_4 * 2;

        int stage_id = 0;
        if constexpr (STAGES > 1) {
            int k_loop = 0;
            int q_block_buffer_load_global_offset = k_loop * kBlockK;
            int q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 32);
            for (int load = 0, warp_loop = warp_id; load < Q_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                int q_warp_buffer_load_m_id = warp_loop & (kBlockM / READ_ONCE_LINES - 1);
                int q_warp_buffer_load_lds_offset = q_lds_stage_offset + (q_warp_buffer_load_m_id >> 3) * (32 * 32) + (q_warp_buffer_load_m_id & 7) * (READ_ONCE_LINES * 32);
                int offset_s = q_block_buffer_load_global_offset / 2;
                int offset_v = q_warp_buffer_load_m_id * READ_ONCE_LINES + q_lane_m_idx;
                int lds_offset = q_warp_buffer_load_lds_offset / 2;
                offset_v = (min(offset_v, max_seq_q_offset - 1) * query_seqlen_stride) / 2 + q_lane_head_dim_idx;
                BUFFER_LOAD_FUNC(q_lds, q_addr, lds_offset, offset_s, offset_v);
            }
        }
        if constexpr (STAGES > 1) stage_id ^= 1;
        constexpr int K_LOOP_START = (STAGES > 1) ? 1: 0;
        for (int k_loop = K_LOOP_START; k_loop < (kHeadDim / kBlockK); ++k_loop) {
            int q_block_buffer_load_global_offset = k_loop * kBlockK;
            int q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 32);
            for (int load = 0, warp_loop = warp_id; load < Q_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                int q_warp_buffer_load_m_id = warp_loop & (kBlockM / READ_ONCE_LINES - 1);
                int q_warp_buffer_load_lds_offset = q_lds_stage_offset + (q_warp_buffer_load_m_id >> 3) * (32 * 32) + (q_warp_buffer_load_m_id & 7) * (READ_ONCE_LINES * 32);
                int offset_s = q_block_buffer_load_global_offset / 2;
                int offset_v = q_warp_buffer_load_m_id * READ_ONCE_LINES + q_lane_m_idx;
                int lds_offset = q_warp_buffer_load_lds_offset / 2;
                offset_v = (min(offset_v, max_seq_q_offset - 1) * query_seqlen_stride) / 2 + q_lane_head_dim_idx;
                BUFFER_LOAD_FUNC(q_lds, q_addr, lds_offset, offset_s, offset_v);
            }

            __builtin_amdgcn_s_waitcnt(0);
            __syncthreads();
            if constexpr (STAGES > 1) stage_id ^= 1;
            q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 16);

            vec2_Element<Element> *q_lds_v2fp16 = (vec2_Element<Element> *)(q_lds);
            #pragma unroll
            for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                #pragma unroll
                for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
                    #pragma unroll
                    for (int i = 0; i < SEQUENCE_READ; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            int lds_offset = q_lds_stage_offset + head_dim_idx * kBlockM * 16 + i * 16 * 16 + q_ds_read_offset + j * 8;
                            int k_loop_idx = (STAGES > 1) ? k_loop - 1: k_loop;
                            q_reg[k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i].u64[j] = *(__float2*)(q_lds_v2fp16 + lds_offset);
                        }
                    }
                }
            }
            __syncthreads();
            // __builtin_amdgcn_sched_barrier(0);
        }

        if constexpr (STAGES > 1) {
            __builtin_amdgcn_s_waitcnt(0);
            stage_id ^= 1;
            int q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 16);
            vec2_Element<Element> *q_lds_v2fp16 = (vec2_Element<Element> *)(q_lds);
            #pragma unroll
            for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
                #pragma unroll
                for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
                    #pragma unroll
                    for (int i = 0; i < SEQUENCE_READ; ++i) {
                        #pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            int lds_offset = q_lds_stage_offset + head_dim_idx * kBlockM * 16 + i * 16 * 16 + q_ds_read_offset + j * 8;
                            q_reg[((kHeadDim / kBlockK) - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i].u64[j] = *(__float2*)(q_lds_v2fp16 + lds_offset);
                        }
                    }
                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
    }
}


template<int kBlockK, int WARP_N, typename Element, int STAGES, int WARP_NUM>
__forceinline__ __device__ void kvcache_prefetch_k_to_lds_gfx938(
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

////////////////////////////////////////////////////////////////////////////////////////////////////
// FP8 MLS Paged Attention helpers, >= gfx938
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int WARP_NUM, typename Element>
__forceinline__ __device__ void fp8_kvcache_prefetch_k_gfx938(
        vec4_uint k_addr,
        Element* k_lds,
        int warp_id,
        int k_row_stride,
        int max_seq_k_offset) {
    int stage_id = 0;
    vec4_uint k_srsrc;
    k_srsrc[1] = k_addr[1];
    k_srsrc[2] = k_row_stride;

    constexpr int k_loop = 0;
    int warp_lds_write_bytes = (stage_id * WARP_NUM + warp_id) * 32 * 64 * sizeof(Element);
    int warp_global_bytes;
    int k_loop_global_bytes = k_loop * 64 * sizeof(Element);

    int nm_filter_max = warp_id * 32 + 32 - max_seq_k_offset;
    int real_mls_warp_id = nm_filter_max >= 32 ? 0 : warp_id;
    warp_global_bytes = real_mls_warp_id * 32 * k_row_stride * sizeof(Element);
    int nm_filter = inline_min_max<0, 32>(real_mls_warp_id * 32 + 32 - max_seq_k_offset);
    k_srsrc[3] = (nm_filter << 8) + 0x40000;

    *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_addr + warp_global_bytes + k_loop_global_bytes);
    inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(k_lds, k_srsrc, warp_lds_write_bytes, 0);
    __builtin_amdgcn_sched_barrier(0);
}

template<int K_LOOP_COUNT, int kBlockK, int WARP_NUM, typename Element, int load_id>
__forceinline__ __device__ void fp8_kvcache_prefetch_v_one_gfx938(
        vec4_uint v_addr,
        Element* v_lds,
        int warp_id,
        int v_row_stride,
        int max_seq_v_offset);

template<bool PrefetchVInQK, int K_LOOP_COUNT, int kHeadDim, int kBlockK, int WARP_M, int WARP_N, int WARP_NUM, int M_MMAC_COUNT, typename Element, typename ElementAccum>
__forceinline__ __device__ void fp8_kvcache_qk_gemm_gfx938(
        vec4_uint k_addr,
        vec4_uint v_addr,
        Element* k_lds,
        Element* v_lds,
        union_vec16_fp8 q_reg[M_MMAC_COUNT][kHeadDim / 64],
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (WARP_N / 32)][4],
        int warp_id,
        int k_row_stride,
        int v_row_stride,
        int max_seq_k_offset = 0) {
    static_assert(!PrefetchVInQK || (kHeadDim == 128 && K_LOOP_COUNT == 2));
    int stage_id = 0;
    vec4_uint k_srsrc;
    k_srsrc[1] = k_addr[1];
    k_srsrc[2] = k_row_stride;

    #pragma unroll
    for (int i = 0; i < (WARP_N / WARP_N) * (WARP_M / 32); ++i) {
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            #pragma unroll
            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                asm volatile(
                    "v_mov_b64 %0, 0x0\n\t"
                    "v_mov_b64 %1, 0x0\n\t"
                    : "=v"(s_reg[i][min_tile_n * 2 + min_tile_m].u64[0]),
                      "=v"(s_reg[i][min_tile_n * 2 + min_tile_m].u64[1])
                    :);
            }
        }
    }

    stage_id ^= 1;
    #pragma unroll
    for (int k_loop = 1; k_loop < kHeadDim / 64; ++k_loop) {
        int warp_lds_write_bytes = (stage_id * WARP_NUM + warp_id) * 32 * 64 * sizeof(Element);
        int warp_global_bytes;
        int k_loop_global_bytes = k_loop * 64 * sizeof(Element);

        int nm_filter_max = warp_id * 32 + 32 - max_seq_k_offset;
        int real_mls_warp_id = nm_filter_max >= 32 ? 0 : warp_id;
        warp_global_bytes = real_mls_warp_id * 32 * k_row_stride * sizeof(Element);
        int nm_filter = inline_min_max<0, 32>(real_mls_warp_id * 32 + 32 - max_seq_k_offset);
        k_srsrc[3] = (nm_filter << 8) + 0x40000;

        *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_addr + warp_global_bytes + k_loop_global_bytes);
        inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(k_lds, k_srsrc, warp_lds_write_bytes, 0);
        flash::wait_buffer_data_arrived<false/*sync*/>(1);
        stage_id ^= 1;

        union_vec16_fp8 k_regs[WARP_N / 16];
        int lds_load_bytes = (stage_id * WARP_NUM + warp_id) * 32 * 64 * sizeof(Element);
        DS_READ_MATRIX_64x16_B8(lds_load_bytes,        k_regs[0].i32x4, true/*transpose*/)
        DS_READ_MATRIX_64x16_B8(lds_load_bytes + 1024, k_regs[1].i32x4, true/*transpose*/)

        #pragma unroll
        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            flash::wait_lds_data_arrived<false/*sync*/>(1 - min_tile_n);
            if constexpr (PrefetchVInQK) {
                if (min_tile_n == 1) {
                    fp8_kvcache_prefetch_v_one_gfx938<K_LOOP_COUNT, kBlockK, WARP_NUM, Element, 0>(
                        v_addr, v_lds, warp_id, v_row_stride, max_seq_k_offset);
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 1");
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                #pragma unroll
                for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    s_reg[0][min_tile_n * 2 + min_tile_m].f32 =
                        mmac_4interleave_b8<int8_t, ElementAccum>(
                            q_reg[min_tile_m][k_loop - 1].i8x8[min_tile_k],
                            k_regs[min_tile_n].i8x8[min_tile_k],
                            s_reg[0][min_tile_n * 2 + min_tile_m].f32);
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 0");
        }
    }

    {
        constexpr int k_loop = kHeadDim / 64;
        if constexpr (PrefetchVInQK) {
            flash::wait_buffer_data_arrived<false/*sync*/>(1);
        } else {
            flash::wait_buffer_data_arrived<false/*sync*/>(0);
        }
        stage_id ^= 1;

        union_vec16_fp8 k_regs[WARP_N / 16];
        int lds_load_bytes = (stage_id * WARP_NUM + warp_id) * 32 * 64 * sizeof(Element);
        DS_READ_MATRIX_64x16_B8(lds_load_bytes,        k_regs[0].i32x4, true/*transpose*/)
        DS_READ_MATRIX_64x16_B8(lds_load_bytes + 1024, k_regs[1].i32x4, true/*transpose*/)

        #pragma unroll
        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            flash::wait_lds_data_arrived<false/*sync*/>(1 - min_tile_n);
            if constexpr (PrefetchVInQK) {
                if (min_tile_n == 1) {
                    fp8_kvcache_prefetch_v_one_gfx938<K_LOOP_COUNT, kBlockK, WARP_NUM, Element, 1>(
                        v_addr, v_lds, warp_id, v_row_stride, max_seq_k_offset);
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 1");
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                #pragma unroll
                for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    s_reg[0][min_tile_n * 2 + min_tile_m].f32 =
                        mmac_4interleave_b8<int8_t, ElementAccum>(
                            q_reg[min_tile_m][k_loop - 1].i8x8[min_tile_k],
                            k_regs[min_tile_n].i8x8[min_tile_k],
                            s_reg[0][min_tile_n * 2 + min_tile_m].f32);
                }
            }
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_setprio 0");
        }
    }

}
