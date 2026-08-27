#pragma once
#include "fp8_mla_tp8_qk_gemm_utils_gfx938.h"



template<int kHeadDim, int kBlockK, int WARP_M, int WARP_N, int WARP_NUM, int M_MMAC_COUNT, typename Element, typename ElementAccum>
__forceinline__ __device__ void fp8_mla_tp8_qk_gemm_gfx938(
        vec4_uint k_addr,
        Element* k_lds,
        union_vec16_fp8 q_reg[M_MMAC_COUNT][kHeadDim / 64],
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (WARP_N / 32)][4],
        int warp_id,
        int k_row_stride,
        int max_seq_k_offset=0) {

    int stage_id = 0;

    // 准备 MLS resource 寄存器
    vec4_uint k_srsrc;
    k_srsrc[1] = k_addr[1];
    k_srsrc[2] = k_row_stride;

    // 初始化 s
    #pragma unroll
    for (int i = 0; i < (WARP_N / WARP_N) * (WARP_M / 32); ++i) {
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            #pragma unroll
            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                asm volatile(
                    "v_mov_b64 %0, 0x0\n\t"
                    "v_mov_b64 %1, 0x0\n\t"
                    : "=v"(s_reg[i][min_tile_n * 2 + min_tile_m].u64[0]), "=v"(s_reg[i][min_tile_n * 2 + min_tile_m].u64[1])
                    :);
            }
        }
    }

    // round
    stage_id ^= 1;

    #pragma unroll
    for (int k_loop = 1; k_loop < kHeadDim / 64; ++k_loop) {
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

        // 等待 4 个 warp 数据写入 lds 完毕, 各 warp 之间数据不共享, 可以尝试不 sync
        flash::wait_buffer_data_arrived<false/*sync*/>(1);

        // round
        stage_id ^= 1;

        // 分配 k 计算 mmac 需要的寄存器资源
        union_vec16_fp8 k_regs[WARP_N / 16];
        // 从 lds 读取数据到寄存器
        int lds_load_bytes = (stage_id * WARP_NUM + warp_id) * 32 * 64 * sizeof(Element);
        DS_READ_MATRIX_64x16_B8(lds_load_bytes,        k_regs[0].i32x4, true/*transpose*/)
        DS_READ_MATRIX_64x16_B8(lds_load_bytes + 1024, k_regs[1].i32x4, true/*transpose*/)

        // mmac
        #pragma unroll
        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            // 等待数据写到寄存器
            flash::wait_lds_data_arrived<false/*sync*/>(1 - min_tile_n);
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                #pragma unroll
                for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    s_reg[0][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(
                        q_reg[min_tile_m][k_loop - 1].i8x8[min_tile_k],
                        k_regs[min_tile_n].i8x8[min_tile_k],
                        s_reg[0][min_tile_n * 2 + min_tile_m].f32
                    );
                }
            }
        }
    }

    {
        constexpr int k_loop = kHeadDim / 64;

        // 等待 4 个 warp 数据写入 lds 完毕, 各 warp 之间数据不共享, 可以尝试不 sync
        flash::wait_buffer_data_arrived<false/*sync*/>(0);

        stage_id ^= 1;

        // 分配 k 计算 mmac 需要的寄存器资源
        union_vec16_fp8 k_regs[WARP_N / 16];
        // 从 lds 读取数据到寄存器
        int lds_load_bytes = (stage_id * WARP_NUM + warp_id) * 32 * 64 * sizeof(Element);
        DS_READ_MATRIX_64x16_B8(lds_load_bytes,        k_regs[0].i32x4, true/*transpose*/)
        DS_READ_MATRIX_64x16_B8(lds_load_bytes + 1024, k_regs[1].i32x4, true/*transpose*/)

        // mmac
        #pragma unroll
        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            // 等待数据写到寄存器
            flash::wait_lds_data_arrived<false/*sync*/>(1 - min_tile_n);
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                #pragma unroll
                for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    s_reg[0][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(
                        q_reg[min_tile_m][k_loop - 1].i8x8[min_tile_k],
                        k_regs[min_tile_n].i8x8[min_tile_k],
                        s_reg[0][min_tile_n * 2 + min_tile_m].f32
                    );
                }
            }
        }
    }

    // need to reduce results on scores_max and prefetch V, and thus sync
    flash::wait_lds_data_arrived<true/*sync*/>(0);

} // qk_gemm

