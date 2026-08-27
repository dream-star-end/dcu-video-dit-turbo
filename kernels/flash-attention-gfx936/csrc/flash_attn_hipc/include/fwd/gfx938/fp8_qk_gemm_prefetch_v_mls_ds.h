#pragma once
#include "fp8_pv_gemm_utils_mls_ds.h"
// #define USE_DS_READ_B128_FOR_INTERLEAVE4
template<int kBlockN, int kHeadDim, int WARP_M, int WARP_N, typename Element, typename ElementAccum>
__forceinline__ __device__ void fp8_qk_gemm(
    vec4_Accum<ElementAccum> s_reg[kBlockN / WARP_N][WARP_M / 16][WARP_N / 16],
    union_vec16_fp8 q_regs[WARP_M / 16][kHeadDim / 64],
    int8_t* k_lds
) {
    static_assert(kHeadDim == 128 || kHeadDim == 192 || kHeadDim == 256);
    constexpr int kLdsHeadDimStride = kHeadDim == 192 ? 256 : kHeadDim;
    int tx = threadIdx.x;
    int lane_id = tx & 63;
    int row = (lane_id & 15) >> 1;
    int col = lane_id >> 4;
    int col_swizzle = (row + col) & 3;

    // 等待 K 的数据都写到 lds 了, 4 wave 同步
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0)\ns_barrier\n"); // hint: pv mmac 太短, 不够隐藏这一段时延, 可以考虑把 vmcnt 拆细一点, 先等一部分数据回来计算也可以
    __builtin_amdgcn_sched_barrier(0);
    // __syncthreads();

    if constexpr (true) {
        // 直接从 lds 读数据, 看看 lds 的数据排布
        union_vec16_fp8 k_regs[kBlockN / WARP_N][WARP_N / 16][kHeadDim / 64];
        #pragma unroll
        for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
            // 分两次读取寄存器, 第一次是 [0, 1024), 即 16x64 的内容, 每个线程读取 16 个 fp8
             #ifdef USE_DS_READ_B128_FOR_INTERLEAVE4
                int k_lds_load_offset = row * 128 + col_swizzle * 16 + (lane_id & 1) * 64 + k_loop * WARP_N * kHeadDim;
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 0,    k_regs[k_loop][0][0].i32x4);
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 1024, k_regs[k_loop][1][0].i32x4);
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 2048, k_regs[k_loop][0][1].i32x4);
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 3072, k_regs[k_loop][1][1].i32x4);
            #else
                int k_lds_load_offset = k_loop * WARP_N * kLdsHeadDimStride;
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 0,    k_regs[k_loop][0][0].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 1024, k_regs[k_loop][1][0].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 2048, k_regs[k_loop][0][1].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 3072, k_regs[k_loop][1][1].i32x4, true/*transpose*/)
                #pragma unroll
                for (int h_idx = 0; h_idx < kHeadDim / 64; ++h_idx) {
                    const int h_offset = (kHeadDim == 192 && h_idx == 2) ? 3 * 2048 : h_idx * 2048;
                    k_regs[k_loop][0][h_idx].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + h_offset + 0, 0, 3, 1, 0);
                    k_regs[k_loop][1][h_idx].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + h_offset + 1024, 0, 3, 1, 0);
                }
            #endif
        }
        // init s_reg
        #pragma unroll
        for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                    inline_vgpr4_init_zero(s_reg[k_loop][m_idx][n_idx]);
                }
            }
        }

        #pragma unroll
        for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(%0)\n":: "B"((kBlockN / WARP_N - k_loop - 1) * 4));
            // asm volatile("s_waitcnt lgkmcnt(0)\n");
            __builtin_amdgcn_sched_barrier(0);
            // ======================================================== QK mmac ======================================================================
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                    #pragma unroll
                    for (int h_idx = 0; h_idx < kHeadDim / 64; ++h_idx) {
                        s_reg[k_loop][m_idx][n_idx].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[m_idx][h_idx].i8x8[0], k_regs[k_loop][n_idx][h_idx].i8x8[0], s_reg[k_loop][m_idx][n_idx].f32);
                        s_reg[k_loop][m_idx][n_idx].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[m_idx][h_idx].i8x8[1], k_regs[k_loop][n_idx][h_idx].i8x8[1], s_reg[k_loop][m_idx][n_idx].f32);
                    }
                }
            }
        }
    } else if constexpr (false and WARP_M == 32 and WARP_N == 32 and kBlockN == 128 and kHeadDim == 128) {
        union_vec16_fp8 k_regs[WARP_N / 16][kHeadDim / 64];
        {
            constexpr int k_loop = 0;
            // 分两次读取寄存器, 第一次是 [0, 1024), 即 16x64 的内容, 每个线程读取 16 个 fp8
            #ifdef USE_DS_READ_B128_FOR_INTERLEAVE4
                int k_lds_load_offset = row * 128 + col_swizzle * 16 + (lane_id & 1) * 64 + k_loop * WARP_N * kHeadDim;
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 0,    k_regs[0][0].i32x4);
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 1024, k_regs[1][0].i32x4);
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 2048, k_regs[0][1].i32x4);
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 3072, k_regs[1][1].i32x4);
            #else
                int k_lds_load_offset = k_loop * WARP_N * kHeadDim;
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 0,    k_regs[0][0].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 1024, k_regs[1][0].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 2048, k_regs[0][1].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 3072, k_regs[1][1].i32x4, true/*transpose*/)
                k_regs[0][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 0,    0, 3, 1, 0);
                k_regs[1][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 1024, 0, 3, 1, 0);
                k_regs[0][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 2048, 0, 3, 1, 0);
                k_regs[1][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 3072, 0, 3, 1, 0);
            #endif

            // init s_reg
            #pragma unroll
            for (int m_idx = 0; m_idx < 2; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < 2; ++n_idx) {
                    inline_vgpr4_init_zero(s_reg[k_loop][m_idx][n_idx]);
                }
            }

            // ======================================================== QK mmac ======================================================================
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(2)\n");
            // asm volatile("s_waitcnt lgkmcnt(0)\n");
            __builtin_amdgcn_sched_barrier(0);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[0], k_regs[0][0].i8x8[0], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[1], k_regs[0][0].i8x8[1], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[0], k_regs[0][0].i8x8[0], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[1], k_regs[0][0].i8x8[1], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[0], k_regs[1][0].i8x8[0], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[1], k_regs[1][0].i8x8[1], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[0], k_regs[1][0].i8x8[0], s_reg[k_loop][1][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[1], k_regs[1][0].i8x8[1], s_reg[k_loop][1][1].f32);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)\n");
            __builtin_amdgcn_sched_barrier(0);
            // 寄存器不够全量预取, 则预取一部分
            #ifdef USE_DS_READ_B128_FOR_INTERLEAVE4
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 0 + WARP_N * kHeadDim,    k_regs[0][0].i32x4);
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 1024 + WARP_N * kHeadDim, k_regs[1][0].i32x4);
            #else
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 0 + WARP_N * kHeadDim,    k_regs[0][0].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 1024 + WARP_N * kHeadDim, k_regs[1][0].i32x4, true/*transpose*/)
                k_regs[0][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 0,    0, 3, 1, 0);
                k_regs[1][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 1024, 0, 3, 1, 0);
            #endif
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[0], k_regs[0][1].i8x8[0], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[1], k_regs[0][1].i8x8[1], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[0], k_regs[0][1].i8x8[0], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[1], k_regs[0][1].i8x8[1], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[0], k_regs[1][1].i8x8[0], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[1], k_regs[1][1].i8x8[1], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[0], k_regs[1][1].i8x8[0], s_reg[k_loop][1][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[1], k_regs[1][1].i8x8[1], s_reg[k_loop][1][1].f32);
        }
        {
            constexpr int k_loop = 1;
            // 分两次读取寄存器, 第一次是 [0, 1024), 即 16x64 的内容, 每个线程读取 16 个 fp8
            #ifdef USE_DS_READ_B128_FOR_INTERLEAVE4
                int k_lds_load_offset = row * 128 + col_swizzle * 16 + (lane_id & 1) * 64 + k_loop * WARP_N * kHeadDim;
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 2048, k_regs[0][1].i32x4);
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 3072, k_regs[1][1].i32x4);
            #else
                int k_lds_load_offset = k_loop * WARP_N * kHeadDim;
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 2048, k_regs[0][1].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 3072, k_regs[1][1].i32x4, true/*transpose*/)
                k_regs[0][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 2048, 0, 3, 1, 0);
                k_regs[1][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 3072, 0, 3, 1, 0);
            #endif

            // init s_reg
            #pragma unroll
            for (int m_idx = 0; m_idx < 2; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < 2; ++n_idx) {
                    inline_vgpr4_init_zero(s_reg[k_loop][m_idx][n_idx]);
                }
            }

            // ======================================================== QK mmac ======================================================================
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(2)\n");
            // asm volatile("s_waitcnt lgkmcnt(0)\n");
            __builtin_amdgcn_sched_barrier(0);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[0], k_regs[0][0].i8x8[0], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[1], k_regs[0][0].i8x8[1], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[0], k_regs[0][0].i8x8[0], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[1], k_regs[0][0].i8x8[1], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[0], k_regs[1][0].i8x8[0], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[1], k_regs[1][0].i8x8[1], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[0], k_regs[1][0].i8x8[0], s_reg[k_loop][1][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[1], k_regs[1][0].i8x8[1], s_reg[k_loop][1][1].f32);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)\n");
            __builtin_amdgcn_sched_barrier(0);
            // 寄存器不够全量预取, 则预取一部分
            #ifdef USE_DS_READ_B128_FOR_INTERLEAVE4
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 0 + WARP_N * kHeadDim,    k_regs[0][0].i32x4);
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 1024 + WARP_N * kHeadDim, k_regs[1][0].i32x4);
            #else
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 0 + WARP_N * kHeadDim,    k_regs[0][0].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 1024 + WARP_N * kHeadDim, k_regs[1][0].i32x4, true/*transpose*/)
                k_regs[0][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 0,    0, 3, 1, 0);
                k_regs[1][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 1024, 0, 3, 1, 0);
            #endif
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[0], k_regs[0][1].i8x8[0], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[1], k_regs[0][1].i8x8[1], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[0], k_regs[0][1].i8x8[0], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[1], k_regs[0][1].i8x8[1], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[0], k_regs[1][1].i8x8[0], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[1], k_regs[1][1].i8x8[1], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[0], k_regs[1][1].i8x8[0], s_reg[k_loop][1][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[1], k_regs[1][1].i8x8[1], s_reg[k_loop][1][1].f32);
        }
        {
            constexpr int k_loop = 2;
            // 分两次读取寄存器, 第一次是 [0, 1024), 即 16x64 的内容, 每个线程读取 16 个 fp8
            #ifdef USE_DS_READ_B128_FOR_INTERLEAVE4
                int k_lds_load_offset = row * 128 + col_swizzle * 16 + (lane_id & 1) * 64 + k_loop * WARP_N * kHeadDim;
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 2048, k_regs[0][1].i32x4);
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 3072, k_regs[1][1].i32x4);
            #else
                int k_lds_load_offset = k_loop * WARP_N * kHeadDim;
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 2048, k_regs[0][1].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 3072, k_regs[1][1].i32x4, true/*transpose*/)
                k_regs[0][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 2048, 0, 3, 1, 0);
                k_regs[1][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 3072, 0, 3, 1, 0);
            #endif

            // init s_reg
            #pragma unroll
            for (int m_idx = 0; m_idx < 2; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < 2; ++n_idx) {
                    inline_vgpr4_init_zero(s_reg[k_loop][m_idx][n_idx]);
                }
            }

            // ======================================================== QK mmac ======================================================================
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(2)\n");
            // asm volatile("s_waitcnt lgkmcnt(0)\n");
            __builtin_amdgcn_sched_barrier(0);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[0], k_regs[0][0].i8x8[0], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[1], k_regs[0][0].i8x8[1], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[0], k_regs[0][0].i8x8[0], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[1], k_regs[0][0].i8x8[1], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[0], k_regs[1][0].i8x8[0], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[1], k_regs[1][0].i8x8[1], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[0], k_regs[1][0].i8x8[0], s_reg[k_loop][1][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[1], k_regs[1][0].i8x8[1], s_reg[k_loop][1][1].f32);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)\n");
            __builtin_amdgcn_sched_barrier(0);
            // 寄存器不够全量预取, 则预取一部分
            #ifdef USE_DS_READ_B128_FOR_INTERLEAVE4
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 0 + WARP_N * kHeadDim,    k_regs[0][0].i32x4);
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 1024 + WARP_N * kHeadDim, k_regs[1][0].i32x4);
            #else
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 0 + WARP_N * kHeadDim,    k_regs[0][0].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 1024 + WARP_N * kHeadDim, k_regs[1][0].i32x4, true/*transpose*/)
                k_regs[0][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 0,    0, 3, 1, 0);
                k_regs[1][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 1024, 0, 3, 1, 0);
            #endif
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[0], k_regs[0][1].i8x8[0], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[1], k_regs[0][1].i8x8[1], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[0], k_regs[0][1].i8x8[0], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[1], k_regs[0][1].i8x8[1], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[0], k_regs[1][1].i8x8[0], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[1], k_regs[1][1].i8x8[1], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[0], k_regs[1][1].i8x8[0], s_reg[k_loop][1][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[1], k_regs[1][1].i8x8[1], s_reg[k_loop][1][1].f32);
        }
        {
            constexpr int k_loop = 3;
            // 分两次读取寄存器, 第一次是 [0, 1024), 即 16x64 的内容, 每个线程读取 16 个 fp8
            #ifdef USE_DS_READ_B128_FOR_INTERLEAVE4
                int k_lds_load_offset = row * 128 + col_swizzle * 16 + (lane_id & 1) * 64 + k_loop * WARP_N * kHeadDim;
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 2048, k_regs[0][1].i32x4);
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 3072, k_regs[1][1].i32x4);
            #else
                int k_lds_load_offset = k_loop * WARP_N * kHeadDim;
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 2048, k_regs[0][1].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 3072, k_regs[1][1].i32x4, true/*transpose*/)
                k_regs[0][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 2048, 0, 3, 1, 0);
                k_regs[1][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 3072, 0, 3, 1, 0);
            #endif

            // init s_reg
            #pragma unroll
            for (int m_idx = 0; m_idx < 2; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < 2; ++n_idx) {
                    inline_vgpr4_init_zero(s_reg[k_loop][m_idx][n_idx]);
                }
            }

            // ======================================================== QK mmac ======================================================================
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(2)\n");
            // asm volatile("s_waitcnt lgkmcnt(0)\n");
            __builtin_amdgcn_sched_barrier(0);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[0], k_regs[0][0].i8x8[0], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[1], k_regs[0][0].i8x8[1], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[0], k_regs[0][0].i8x8[0], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[1], k_regs[0][0].i8x8[1], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[0], k_regs[1][0].i8x8[0], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[1], k_regs[1][0].i8x8[1], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[0], k_regs[1][0].i8x8[0], s_reg[k_loop][1][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[1], k_regs[1][0].i8x8[1], s_reg[k_loop][1][1].f32);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)\n");
            __builtin_amdgcn_sched_barrier(0);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[0], k_regs[0][1].i8x8[0], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[1], k_regs[0][1].i8x8[1], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[0], k_regs[0][1].i8x8[0], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[1], k_regs[0][1].i8x8[1], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[0], k_regs[1][1].i8x8[0], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[1], k_regs[1][1].i8x8[1], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[0], k_regs[1][1].i8x8[0], s_reg[k_loop][1][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[1], k_regs[1][1].i8x8[1], s_reg[k_loop][1][1].f32);
        }
    } else if constexpr (false) {
        #pragma unroll
        for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {

            // 直接从 lds 读数据, 看看 lds 的数据排布
            union_vec16_fp8 k_regs[WARP_N / 16][kHeadDim / 64];
            // 分两次读取寄存器, 第一次是 [0, 1024), 即 16x64 的内容, 每个线程读取 16 个 fp8
            #ifdef USE_DS_READ_B128_FOR_INTERLEAVE4
                int k_lds_load_offset = row * 128 + col_swizzle * 16 + (lane_id & 1) * 64 + k_loop * WARP_N * kHeadDim;
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 0,    k_regs[0][0].i32x4);
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 1024, k_regs[1][0].i32x4);
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 2048, k_regs[0][1].i32x4);
                inline_ds_read_b128_no_wait_bytes(reinterpret_cast<size_t>(k_lds) + k_lds_load_offset + 3072, k_regs[1][1].i32x4);
            #else
                int k_lds_load_offset = k_loop * WARP_N * kHeadDim;
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 0,    k_regs[0][0].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 1024, k_regs[1][0].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 2048, k_regs[0][1].i32x4, true/*transpose*/)
                // DS_READ_MATRIX_64x16_B8(k_lds_load_offset + 3072, k_regs[1][1].i32x4, true/*transpose*/)
                k_regs[0][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 0,    0, 3, 1, 0);
                k_regs[1][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 1024, 0, 3, 1, 0);
                k_regs[0][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 2048, 0, 3, 1, 0);
                k_regs[1][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 3072, 0, 3, 1, 0);
            #endif

            // init s_reg
            #pragma unroll
            for (int m_idx = 0; m_idx < 2; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < 2; ++n_idx) {
                    inline_vgpr4_init_zero(s_reg[k_loop][m_idx][n_idx]);
                }
            }

            // ======================================================== QK mmac ======================================================================
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(2)\n");
            // asm volatile("s_waitcnt lgkmcnt(0)\n");
            __builtin_amdgcn_sched_barrier(0);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[0], k_regs[0][0].i8x8[0], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[1], k_regs[0][0].i8x8[1], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[0], k_regs[0][0].i8x8[0], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[1], k_regs[0][0].i8x8[1], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[0], k_regs[1][0].i8x8[0], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][0].i8x8[1], k_regs[1][0].i8x8[1], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[0], k_regs[1][0].i8x8[0], s_reg[k_loop][1][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][0].i8x8[1], k_regs[1][0].i8x8[1], s_reg[k_loop][1][1].f32);
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)\n");
            __builtin_amdgcn_sched_barrier(0);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[0], k_regs[0][1].i8x8[0], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][0][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[1], k_regs[0][1].i8x8[1], s_reg[k_loop][0][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[0], k_regs[0][1].i8x8[0], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][1][0].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[1], k_regs[0][1].i8x8[1], s_reg[k_loop][1][0].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[0], k_regs[1][1].i8x8[0], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][0][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[0][1].i8x8[1], k_regs[1][1].i8x8[1], s_reg[k_loop][0][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[0], k_regs[1][1].i8x8[0], s_reg[k_loop][1][1].f32);
            s_reg[k_loop][1][1].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[1][1].i8x8[1], k_regs[1][1].i8x8[1], s_reg[k_loop][1][1].f32);
        }
    } else { // 寄存器占用更少的写法
        #pragma unroll
        for (int k_loop = 0; k_loop < kBlockN / WARP_N; ++k_loop) {
            // init s_reg
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                    s_reg[k_loop][m_idx][n_idx].f32[0] = 0;
                    s_reg[k_loop][m_idx][n_idx].f32[1] = 0;
                    s_reg[k_loop][m_idx][n_idx].f32[2] = 0;
                    s_reg[k_loop][m_idx][n_idx].f32[3] = 0;
                }
            }

            // 直接从 lds 读数据, 看看 lds 的数据排布
            union_vec16_fp8 k_regs[WARP_N / 16][kHeadDim / 64];
            // 分两次读取寄存器, 第一次是 [0, 1024), 即 16x64 的内容, 每个线程读取 16 个 fp8
            #ifdef USE_DS_READ_B128_FOR_INTERLEAVE4
                int k_lds_load_offset = row * 128 + col_swizzle * 16 + (lane_id & 1) * 64 + k_loop * WARP_N * kHeadDim;
                k_regs[0][0].i32x4 = *(vec4_int32*)(k_lds + k_lds_load_offset + 0);
                k_regs[1][0].i32x4 = *(vec4_int32*)(k_lds + k_lds_load_offset + 1024/*ds fmt 0, dmft1 */);
                k_regs[0][1].i32x4 = *(vec4_int32*)(k_lds + k_lds_load_offset + 2048);
                k_regs[1][1].i32x4 = *(vec4_int32*)(k_lds + k_lds_load_offset + 3072);
            #else
                int k_lds_load_offset = k_loop * WARP_N * kHeadDim;
                k_regs[0][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 0,    0, 3, 1, 0);
                k_regs[1][0].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 1024, 0, 3, 1, 0);
                k_regs[0][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 2048, 0, 3, 1, 0);
                k_regs[1][1].i32x4 = __builtin_hcu_ds_read_matrix_trans_format_u8(k_lds + k_lds_load_offset + 3072, 0, 3, 1, 0);
            #endif

            // ======================================================== QK mmac ======================================================================
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                    s_reg[k_loop][m_idx][n_idx].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[m_idx][0].i8x8[0], k_regs[n_idx][0].i8x8[0], s_reg[k_loop][m_idx][n_idx].f32);
                    s_reg[k_loop][m_idx][n_idx].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[m_idx][0].i8x8[1], k_regs[n_idx][0].i8x8[1], s_reg[k_loop][m_idx][n_idx].f32);
                    s_reg[k_loop][m_idx][n_idx].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[m_idx][1].i8x8[0], k_regs[n_idx][1].i8x8[0], s_reg[k_loop][m_idx][n_idx].f32);
                    s_reg[k_loop][m_idx][n_idx].f32 = mmac_4interleave_b8<int8_t, ElementAccum>(q_regs[m_idx][1].i8x8[1], k_regs[n_idx][1].i8x8[1], s_reg[k_loop][m_idx][n_idx].f32);
                }
            }
        }
    }
}
