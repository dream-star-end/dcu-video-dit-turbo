#pragma once
#include "fp8_mla_tp8_pv_gemm_utils_gfx938.h"
#include "fp8_mla_tp8_qk_gemm_utils_gfx938.h"


template<bool PrefetchK, int K_LOOP_COUNT, int kBlockN, int kBlockK, int M_WARP_COUNT, int PV_K_WARP_COUNT, int WARP_NUM, int M_MMAC_COUNT, typename V_Element, typename P_Element, typename ElementAccum>
__forceinline__ __device__ void fp8_mla_tp8_pv_gemm_prefetch_k_gfx938(
        vec4_uint v_addr,
        vec4_uint& k_addr,
        V_Element* v_lds,
        V_Element* k_lds,
        union_vec2_f16x2<P_Element> p_reg[M_WARP_COUNT * PV_K_WARP_COUNT][4],
        vec4_Accum<ElementAccum> pv_reg[K_LOOP_COUNT * M_WARP_COUNT * (kBlockN / 32)][4],
        int warp_id,
        int k_row_stride,
        int v_row_stride,
        int max_seq_v_offset,
        int64_t k_addr_offset) {

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
    int stage_id = 1;

    #pragma unroll
    for (int k_loop = K_LOOP_COUNT_ - 1 - PREFETCH; k_loop >= 1; k_loop -= PREFETCH) {

        #pragma unroll
        for (int load_id = 0; load_id < PREFETCH; ++load_id) {
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

        // 等待 4 个 warp 数据写入 lds 完毕, 各 warp 之间数据不共享, 可以尝试不 sync
        flash::wait_buffer_data_arrived<false/*sync*/>(PREFETCH);

        stage_id ^= 1;

        #pragma unroll
        for (int load_id = 0; load_id < PREFETCH; ++load_id) {
            // 分配 v 计算 mmac 需要的寄存器资源
            union_vec16_fp8 v_regs[2];
            // 从 lds 读取数据到寄存器
            int lds_load_bytes = stage_id * 16384 + (WARP_NUM * load_id + warp_id) * 32 * 64 * sizeof(V_Element);
            DS_READ_MATRIX_32x32_B8_ALT2(lds_load_bytes,      v_regs[0].i32x4, false/*transpose*/)
            DS_READ_MATRIX_32x32_B8_ALT2(lds_load_bytes + 32, v_regs[1].i32x4, false/*transpose*/)

            // mmac
            // P, fp16, 半精度
            // V, fp8
            int k_loop_inner = k_loop - load_id + PREFETCH;
            #pragma unroll
            for (int tile32x32_id = 0; tile32x32_id < 2; ++tile32x32_id) {
                // wait data written to registers
                flash::wait_lds_data_arrived<false/*sync*/>(1 - tile32x32_id);
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    // 16 fp8 for ds32x32_b8
                    #pragma unroll
                    for (int min_tile_dim = 0; min_tile_dim < 2; ++min_tile_dim) {
                        // fp8 -> f32
                        vec2_fp32 v_f32x2[4]; // 8 fp8 -> 8 f32, for 1 mmac
                        v_f32x2[0] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 0], false/*word_sel*/);
                        v_f32x2[1] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 0], true/*word_sel*/);
                        v_f32x2[2] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 1], false/*word_sel*/);
                        v_f32x2[3] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 1], true/*word_sel*/);
                        // f32 -> fp16
                        union_vec4_f16x2<P_Element> v_f16x8;
                        v_f16x8.f16x2[0] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[0][0], v_f32x2[0][1], false/*clamp*/, 0/*o_modifier*/);
                        v_f16x8.f16x2[1] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[1][0], v_f32x2[1][1], false/*clamp*/, 0/*o_modifier*/);
                        v_f16x8.f16x2[2] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[2][0], v_f32x2[2][1], false/*clamp*/, 0/*o_modifier*/);
                        v_f16x8.f16x2[3] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[3][0], v_f32x2[3][1], false/*clamp*/, 0/*o_modifier*/);
                        // mmac_16x16x16, 4 fp16
                        #pragma unroll
                        for (int mmac_id = 0; mmac_id < 2; ++mmac_id) {
                            pv_reg[k_loop_inner * 2 + tile32x32_id][min_tile_dim * 2 + min_tile_m].f32 = mmac_4interleave<P_Element, ElementAccum>(
                                p_reg[0][mmac_id * 2 + min_tile_m].f16x4,
                                v_f16x8.f16x4[mmac_id],
                                pv_reg[k_loop_inner * 2 + tile32x32_id][min_tile_dim * 2 + min_tile_m].f32
                            );
                        }
                    }
                }
            }
        }
    }

    // 处理 K
    *(int64_t*)&k_addr += k_addr_offset;
    if constexpr (PrefetchK) {
        fp8_mla_tp8_prefetch_k_gfx938<WARP_NUM, V_Element>(k_addr, k_lds, warp_id, k_row_stride, max_seq_v_offset - kBlockK);
        flash::wait_buffer_data_arrived<false/*sync*/>(1);
    } else {
        flash::wait_buffer_data_arrived<false/*sync*/>(0);
    }

    {
        constexpr int k_loop = 1 - PREFETCH;

        stage_id ^= 1;

        #pragma unroll
        for (int load_id = 0; load_id < PREFETCH; ++load_id) {
            // 分配 v 计算 mmac 需要的寄存器资源
            union_vec16_fp8 v_regs[2];
            // 从 lds 读取数据到寄存器
            int lds_load_bytes = stage_id * 16384 + (WARP_NUM * load_id + warp_id) * 32 * 64 * sizeof(V_Element);
            DS_READ_MATRIX_32x32_B8_ALT2(lds_load_bytes,      v_regs[0].i32x4, false/*transpose*/)
            DS_READ_MATRIX_32x32_B8_ALT2(lds_load_bytes + 32, v_regs[1].i32x4, false/*transpose*/)

            // mmac
            // P, fp16, 半精度
            // V, fp8
            int k_loop_inner = k_loop - load_id + PREFETCH;
            #pragma unroll
            for (int tile32x32_id = 0; tile32x32_id < 2; ++tile32x32_id) {
                // wait data written to registers
                flash::wait_lds_data_arrived<false/*sync*/>(1 - tile32x32_id);
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    // 16 fp8 for ds32x32_b8
                    #pragma unroll
                    for (int min_tile_dim = 0; min_tile_dim < 2; ++min_tile_dim) {
                        // fp8 -> f32
                        vec2_fp32 v_f32x2[4]; // 8 fp8 -> 8 f32, for 1 mmac
                        v_f32x2[0] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 0], false/*word_sel*/);
                        v_f32x2[1] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 0], true/*word_sel*/);
                        v_f32x2[2] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 1], false/*word_sel*/);
                        v_f32x2[3] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 1], true/*word_sel*/);
                        // f32 -> fp16
                        union_vec4_f16x2<P_Element> v_f16x8;
                        v_f16x8.f16x2[0] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[0][0], v_f32x2[0][1], false/*clamp*/, 0/*o_modifier*/);
                        v_f16x8.f16x2[1] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[1][0], v_f32x2[1][1], false/*clamp*/, 0/*o_modifier*/);
                        v_f16x8.f16x2[2] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[2][0], v_f32x2[2][1], false/*clamp*/, 0/*o_modifier*/);
                        v_f16x8.f16x2[3] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[3][0], v_f32x2[3][1], false/*clamp*/, 0/*o_modifier*/);
                        // mmac_16x16x16, 4 fp16
                        #pragma unroll
                        for (int mmac_id = 0; mmac_id < 2; ++mmac_id) {
                            pv_reg[k_loop_inner * 2 + tile32x32_id][min_tile_dim * 2 + min_tile_m].f32 = mmac_4interleave<P_Element, ElementAccum>(
                                p_reg[0][mmac_id * 2 + min_tile_m].f16x4,
                                v_f16x8.f16x4[mmac_id],
                                pv_reg[k_loop_inner * 2 + tile32x32_id][min_tile_dim * 2 + min_tile_m].f32
                            );
                        }
                    }
                }
            }
        }
    }

    flash::wait_lds_data_arrived<true/*sync*/>(0); // here, K/V use more lds, and thus reuse togather, need sync
}
