#pragma once // prepare for prefetch V in qk gemm
#include "intrinsic.h"
#include "fwd/utils.h"
#include "intrinsic_mls_ds.h"
#include "intrinsic_mls_ds_b8.h"


template<int kHeadDim, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int WARP_K, int stage_id, int WARP_NUM, typename Element, int STAGES>
__forceinline__ __device__ void kvcache_prefetch_v_to_lds_gfx938(
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

////////////////////////////////////////////////////////////////////////////////////////////////////
// FP8 MLS Paged Attention PV helpers, >= gfx938
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int WARP_NUM, typename Element>
__forceinline__ __device__ void fp8_kvcache_prefetch_k_gfx938(
        vec4_uint k_addr,
        Element* k_lds,
        int warp_id,
        int k_row_stride,
        int max_seq_k_offset);

template<int K_LOOP_COUNT, int kBlockK, int WARP_NUM, typename Element>
__forceinline__ __device__ void fp8_kvcache_prefetch_v_gfx938(
        vec4_uint v_addr,
        Element* v_lds,
        int warp_id,
        int v_row_stride,
        int max_seq_v_offset) {
    static_assert(K_LOOP_COUNT % 2 == 0);
    constexpr int PREFETCH = 2;

    vec4_uint v_srsrc;
    v_srsrc[1] = v_addr[1];
    v_srsrc[2] = v_row_stride;

    int stage_id = 0;
    constexpr int k_loop = K_LOOP_COUNT - 1;
    #pragma unroll
    for (int load_id = 0; load_id < PREFETCH; ++load_id) {
        int warp_lds_write_bytes = stage_id * 16384 + (WARP_NUM * load_id + warp_id) * 32 * 64 * sizeof(Element);
        int warp_global_bytes;
        int v_loop_global_bytes = (k_loop - load_id) * 64 * sizeof(Element);

        int nm_filter_max = warp_id * 32 + 32 - max_seq_v_offset;
        int real_mls_warp_id = nm_filter_max >= 32 ? 0 : warp_id;
        warp_global_bytes = real_mls_warp_id * 32 * v_row_stride * sizeof(Element);
        int nm_filter = inline_min_max<0, 32>(real_mls_warp_id * 32 + 32 - max_seq_v_offset);
        v_srsrc[3] = (nm_filter << 8) + 0x20000;

        *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_addr + warp_global_bytes + v_loop_global_bytes);
        inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(v_lds, v_srsrc, warp_lds_write_bytes, 0);
    }
}

template<int K_LOOP_COUNT, int kBlockK, int WARP_NUM, typename Element, int load_id>
__forceinline__ __device__ void fp8_kvcache_prefetch_v_one_gfx938(
        vec4_uint v_addr,
        Element* v_lds,
        int warp_id,
        int v_row_stride,
        int max_seq_v_offset) {
    static_assert(K_LOOP_COUNT == 2);
    static_assert(load_id == 0 || load_id == 1);

    vec4_uint v_srsrc;
    v_srsrc[1] = v_addr[1];
    v_srsrc[2] = v_row_stride;

    constexpr int stage_id = 0;
    constexpr int k_loop = K_LOOP_COUNT - 1;
    const int warp_lds_write_bytes = stage_id * 16384 + (WARP_NUM * load_id + warp_id) * 32 * 64 * sizeof(Element);
    const int v_loop_global_bytes = (k_loop - load_id) * 64 * sizeof(Element);

    const int nm_filter_max = warp_id * 32 + 32 - max_seq_v_offset;
    const int real_mls_warp_id = nm_filter_max >= 32 ? 0 : warp_id;
    const int warp_global_bytes = real_mls_warp_id * 32 * v_row_stride * sizeof(Element);
    const int nm_filter = inline_min_max<0, 32>(real_mls_warp_id * 32 + 32 - max_seq_v_offset);
    v_srsrc[3] = (nm_filter << 8) + 0x20000;

    *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_addr + warp_global_bytes + v_loop_global_bytes);
    inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(v_lds, v_srsrc, warp_lds_write_bytes, 0);
}


template<bool PrefetchK, int K_LOOP_COUNT, int kBlockK, int kBlockN, int M_WARP_COUNT, int K_WARP_COUNT, int WARP_NUM, int M_MMAC_COUNT, typename V_Element, typename P_Element, typename ElementAccum>
__forceinline__ __device__ void fp8_kvcache_pv_gemm_prefetch_k_gfx938(
        vec4_uint v_addr,
        vec4_uint& k_addr,
        V_Element* v_lds,
        V_Element* k_lds,
        union_vec2_f16x2<P_Element> p_reg[M_WARP_COUNT * K_WARP_COUNT][4],
        vec4_Accum<ElementAccum> pv_reg[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4],
        int warp_id,
        int k_row_stride,
        int v_row_stride,
        int max_seq_v_offset,
        int64_t k_addr_offset) {
    static_assert(K_LOOP_COUNT % 2 == 0);
    constexpr int PREFETCH = 2;

    flash::wait_lds_data_arrived<true/*sync*/>(0);

    vec4_uint v_srsrc;
    v_srsrc[1] = v_addr[1];
    v_srsrc[2] = v_row_stride;

    int stage_id = 1;
    #pragma unroll
    for (int k_loop = K_LOOP_COUNT - 1 - PREFETCH; k_loop >= 1; k_loop -= PREFETCH) {
        #pragma unroll
        for (int load_id = 0; load_id < PREFETCH; ++load_id) {
            int warp_lds_write_bytes = stage_id * 16384 + (WARP_NUM * load_id + warp_id) * 32 * 64 * sizeof(V_Element);
            int warp_global_bytes;
            int v_loop_global_bytes = (k_loop - load_id) * 64 * sizeof(V_Element);

            int nm_filter_max = warp_id * 32 + 32 - max_seq_v_offset;
            int real_mls_warp_id = nm_filter_max >= 32 ? 0 : warp_id;
            warp_global_bytes = real_mls_warp_id * 32 * v_row_stride * sizeof(V_Element);
            int nm_filter = inline_min_max<0, 32>(real_mls_warp_id * 32 + 32 - max_seq_v_offset);
            v_srsrc[3] = (nm_filter << 8) + 0x20000;

            *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_addr + warp_global_bytes + v_loop_global_bytes);
            inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(v_lds, v_srsrc, warp_lds_write_bytes, 0);
        }

        flash::wait_buffer_data_arrived<false/*sync*/>(PREFETCH);
        stage_id ^= 1;

        #pragma unroll
        for (int load_id = 0; load_id < PREFETCH; ++load_id) {
            union_vec16_fp8 v_regs[2];
            int lds_load_bytes = stage_id * 16384 + (WARP_NUM * load_id + warp_id) * 32 * 64 * sizeof(V_Element);
            DS_READ_MATRIX_32x32_B8_ALT2(lds_load_bytes,      v_regs[0].i32x4, false/*transpose*/)
            DS_READ_MATRIX_32x32_B8_ALT2(lds_load_bytes + 32, v_regs[1].i32x4, false/*transpose*/)

            int k_loop_inner = k_loop - load_id + PREFETCH;
            #pragma unroll
            for (int tile32x32_id = 0; tile32x32_id < 2; ++tile32x32_id) {
                flash::wait_lds_data_arrived<false/*sync*/>(1 - tile32x32_id);
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 2");
                #pragma unroll
                for (int min_tile_dim = 0; min_tile_dim < 2; ++min_tile_dim) {
                    vec2_fp32 v_f32x2[4];
                    v_f32x2[0] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 0], false/*word_sel*/);
                    v_f32x2[1] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 0], true/*word_sel*/);
                    v_f32x2[2] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 1], false/*word_sel*/);
                    v_f32x2[3] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 1], true/*word_sel*/);
                    union_vec4_f16x2<P_Element> v_f16x8;
                    v_f16x8.f16x2[0] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[0][0], v_f32x2[0][1], false/*clamp*/, 0/*o_modifier*/);
                    v_f16x8.f16x2[1] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[1][0], v_f32x2[1][1], false/*clamp*/, 0/*o_modifier*/);
                    v_f16x8.f16x2[2] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[2][0], v_f32x2[2][1], false/*clamp*/, 0/*o_modifier*/);
                    v_f16x8.f16x2[3] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[3][0], v_f32x2[3][1], false/*clamp*/, 0/*o_modifier*/);

                    #pragma unroll
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        #pragma unroll
                        for (int mmac_id = 0; mmac_id < 2; ++mmac_id) {
                            pv_reg[k_loop_inner * 2 + tile32x32_id][min_tile_dim * 2 + min_tile_m].f32 =
                                mmac_4interleave<P_Element, ElementAccum>(
                                    p_reg[0][mmac_id * 2 + min_tile_m].f16x4,
                                    v_f16x8.f16x4[mmac_id],
                                    pv_reg[k_loop_inner * 2 + tile32x32_id][min_tile_dim * 2 + min_tile_m].f32);
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 0");
            }
        }
    }

    flash::wait_buffer_data_arrived<false/*sync*/>(0);

    constexpr bool PrefetchKInPV = PrefetchK && K_LOOP_COUNT == 2;

    {
        constexpr int k_loop = 1 - PREFETCH;
        stage_id ^= 1;

        #pragma unroll
        for (int load_id = 0; load_id < PREFETCH; ++load_id) {
            union_vec16_fp8 v_regs[2];
            int lds_load_bytes = stage_id * 16384 + (WARP_NUM * load_id + warp_id) * 32 * 64 * sizeof(V_Element);
            DS_READ_MATRIX_32x32_B8_ALT2(lds_load_bytes,      v_regs[0].i32x4, false/*transpose*/)
            DS_READ_MATRIX_32x32_B8_ALT2(lds_load_bytes + 32, v_regs[1].i32x4, false/*transpose*/)

            int k_loop_inner = k_loop - load_id + PREFETCH;
            #pragma unroll
            for (int tile32x32_id = 0; tile32x32_id < 2; ++tile32x32_id) {
                flash::wait_lds_data_arrived<false/*sync*/>(1 - tile32x32_id);
                if constexpr (PrefetchKInPV) {
                    if (load_id == 0 && tile32x32_id == 1) {
                        *(int64_t*)&k_addr += k_addr_offset;
                        fp8_kvcache_prefetch_k_gfx938<WARP_NUM, V_Element>(k_addr, k_lds, warp_id, k_row_stride, max_seq_v_offset - kBlockN);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 2");
                #pragma unroll
                for (int min_tile_dim = 0; min_tile_dim < 2; ++min_tile_dim) {
                    vec2_fp32 v_f32x2[4];
                    v_f32x2[0] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 0], false/*word_sel*/);
                    v_f32x2[1] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 0], true/*word_sel*/);
                    v_f32x2[2] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 1], false/*word_sel*/);
                    v_f32x2[3] = __builtin_hcu_cvt_pk_f32_fp8(v_regs[tile32x32_id].i32[min_tile_dim * 2 + 1], true/*word_sel*/);
                    union_vec4_f16x2<P_Element> v_f16x8;
                    v_f16x8.f16x2[0] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[0][0], v_f32x2[0][1], false/*clamp*/, 0/*o_modifier*/);
                    v_f16x8.f16x2[1] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[1][0], v_f32x2[1][1], false/*clamp*/, 0/*o_modifier*/);
                    v_f16x8.f16x2[2] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[2][0], v_f32x2[2][1], false/*clamp*/, 0/*o_modifier*/);
                    v_f16x8.f16x2[3] = __builtin_hcu_cvt_pk_f16_f32(v_f32x2[3][0], v_f32x2[3][1], false/*clamp*/, 0/*o_modifier*/);

                    #pragma unroll
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        #pragma unroll
                        for (int mmac_id = 0; mmac_id < 2; ++mmac_id) {
                            pv_reg[k_loop_inner * 2 + tile32x32_id][min_tile_dim * 2 + min_tile_m].f32 =
                                mmac_4interleave<P_Element, ElementAccum>(
                                    p_reg[0][mmac_id * 2 + min_tile_m].f16x4,
                                    v_f16x8.f16x4[mmac_id],
                                    pv_reg[k_loop_inner * 2 + tile32x32_id][min_tile_dim * 2 + min_tile_m].f32);
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 0");
            }
        }
    }

    if constexpr (PrefetchK && !PrefetchKInPV) {
        *(int64_t*)&k_addr += k_addr_offset;
        fp8_kvcache_prefetch_k_gfx938<WARP_NUM, V_Element>(k_addr, k_lds, warp_id, k_row_stride, max_seq_v_offset - kBlockN);
    }

    flash::wait_lds_data_arrived<true/*sync*/>(0);
}


template<bool PrefetchK, int K_LOOP_COUNT, int kBlockK, int kBlockN, int M_WARP_COUNT, int K_WARP_COUNT, int WARP_NUM, int M_MMAC_COUNT, typename V_Element, typename ElementAccum>
__forceinline__ __device__ void fp8_kvcache_pv_gemm_fp8_prefetch_k_gfx938(
        vec4_uint v_addr,
        vec4_uint& k_addr,
        V_Element* v_lds,
        V_Element* k_lds,
        union_vec32_fp8 p_reg[M_MMAC_COUNT],
        vec4_Accum<ElementAccum> pv_reg[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4],
        int warp_id,
        int k_row_stride,
        int v_row_stride,
        int max_seq_v_offset,
        int64_t k_addr_offset) {
    static_assert(K_LOOP_COUNT % 2 == 0);
    static_assert(M_WARP_COUNT == 1);
    static_assert(K_WARP_COUNT == 2);
    constexpr int PREFETCH = 2;

    flash::wait_lds_data_arrived<true/*sync*/>(0);

    vec4_uint v_srsrc;
    v_srsrc[1] = v_addr[1];
    v_srsrc[2] = v_row_stride;

    int stage_id = 1;
    #pragma unroll
    for (int k_loop = K_LOOP_COUNT - 1 - PREFETCH; k_loop >= 1; k_loop -= PREFETCH) {
        #pragma unroll
        for (int load_id = 0; load_id < PREFETCH; ++load_id) {
            int warp_lds_write_bytes = stage_id * 16384 + (WARP_NUM * load_id + warp_id) * 32 * 64 * sizeof(V_Element);
            int warp_global_bytes;
            int v_loop_global_bytes = (k_loop - load_id) * 64 * sizeof(V_Element);

            int nm_filter_max = warp_id * 32 + 32 - max_seq_v_offset;
            int real_mls_warp_id = nm_filter_max >= 32 ? 0 : warp_id;
            warp_global_bytes = real_mls_warp_id * 32 * v_row_stride * sizeof(V_Element);
            int nm_filter = inline_min_max<0, 32>(real_mls_warp_id * 32 + 32 - max_seq_v_offset);
            v_srsrc[3] = (nm_filter << 8) + 0x20000;

            *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_addr + warp_global_bytes + v_loop_global_bytes);
            inline_matrix_load_64x32_b8_lds_rearrange<0, 1>(v_lds, v_srsrc, warp_lds_write_bytes, 0);
        }

        flash::wait_buffer_data_arrived<false/*sync*/>(PREFETCH);
        stage_id ^= 1;

        #pragma unroll
        for (int load_id = 0; load_id < PREFETCH; ++load_id) {
            union_vec16_fp8 v_regs[2];
            int lds_load_bytes = stage_id * 16384 + (WARP_NUM * load_id + warp_id) * 32 * 64 * sizeof(V_Element);
            DS_READ_MATRIX_32x32_B8_ALT2(lds_load_bytes,      v_regs[0].i32x4, false/*transpose*/)
            DS_READ_MATRIX_32x32_B8_ALT2(lds_load_bytes + 32, v_regs[1].i32x4, false/*transpose*/)

            int k_loop_inner = k_loop - load_id + PREFETCH;
            #pragma unroll
            for (int tile32x32_id = 0; tile32x32_id < 2; ++tile32x32_id) {
                flash::wait_lds_data_arrived<false/*sync*/>(1 - tile32x32_id);
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 2");
                #pragma unroll
                for (int min_tile_dim = 0; min_tile_dim < 2; ++min_tile_dim) {
                    #pragma unroll
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        pv_reg[k_loop_inner * 2 + tile32x32_id][min_tile_dim * 2 + min_tile_m].f32 =
                            mmac_4interleave_b8<int8_t, ElementAccum>(
                                p_reg[min_tile_m].i8x8[0],
                                v_regs[tile32x32_id].i8x8[min_tile_dim],
                                pv_reg[k_loop_inner * 2 + tile32x32_id][min_tile_dim * 2 + min_tile_m].f32);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 0");
            }
        }
    }

    flash::wait_buffer_data_arrived<false/*sync*/>(0);

    constexpr bool PrefetchKInPV = PrefetchK && K_LOOP_COUNT == 2;

    {
        constexpr int k_loop = 1 - PREFETCH;
        stage_id ^= 1;

        #pragma unroll
        for (int load_id = 0; load_id < PREFETCH; ++load_id) {
            union_vec16_fp8 v_regs[2];
            int lds_load_bytes = stage_id * 16384 + (WARP_NUM * load_id + warp_id) * 32 * 64 * sizeof(V_Element);
            DS_READ_MATRIX_32x32_B8_ALT2(lds_load_bytes,      v_regs[0].i32x4, false/*transpose*/)
            DS_READ_MATRIX_32x32_B8_ALT2(lds_load_bytes + 32, v_regs[1].i32x4, false/*transpose*/)

            int k_loop_inner = k_loop - load_id + PREFETCH;
            #pragma unroll
            for (int tile32x32_id = 0; tile32x32_id < 2; ++tile32x32_id) {
                flash::wait_lds_data_arrived<false/*sync*/>(1 - tile32x32_id);
                if constexpr (PrefetchKInPV) {
                    if (load_id == 0 && tile32x32_id == 1) {
                        *(int64_t*)&k_addr += k_addr_offset;
                        fp8_kvcache_prefetch_k_gfx938<WARP_NUM, V_Element>(k_addr, k_lds, warp_id, k_row_stride, max_seq_v_offset - kBlockN);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 2");
                #pragma unroll
                for (int min_tile_dim = 0; min_tile_dim < 2; ++min_tile_dim) {
                    #pragma unroll
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        pv_reg[k_loop_inner * 2 + tile32x32_id][min_tile_dim * 2 + min_tile_m].f32 =
                            mmac_4interleave_b8<int8_t, ElementAccum>(
                                p_reg[min_tile_m].i8x8[0],
                                v_regs[tile32x32_id].i8x8[min_tile_dim],
                                pv_reg[k_loop_inner * 2 + tile32x32_id][min_tile_dim * 2 + min_tile_m].f32);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_setprio 0");
            }
        }
    }

    if constexpr (PrefetchK && !PrefetchKInPV) {
        *(int64_t*)&k_addr += k_addr_offset;
        fp8_kvcache_prefetch_k_gfx938<WARP_NUM, V_Element>(k_addr, k_lds, warp_id, k_row_stride, max_seq_v_offset - kBlockN);
    }

    flash::wait_lds_data_arrived<true/*sync*/>(0);
}

template <int M_MMAC_COUNT, typename Element, typename ElementAccum>
inline __device__ void fp8_kvcache_cvt_f32_to_fp8_gfx938(
        union_vec32_fp8 p_reg[M_MMAC_COUNT],
        vec4_Accum<ElementAccum> s_reg[1][4]) {
    #pragma unroll
    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
        __builtin_hcu_cvt_pk4_fp8_f32<Element>(s_reg[0][0 * 2 + min_tile_m].f32, p_reg[min_tile_m].i32[0]);
        __builtin_hcu_cvt_pk4_fp8_f32<Element>(s_reg[0][1 * 2 + min_tile_m].f32, p_reg[min_tile_m].i32[1]);
    }
}
