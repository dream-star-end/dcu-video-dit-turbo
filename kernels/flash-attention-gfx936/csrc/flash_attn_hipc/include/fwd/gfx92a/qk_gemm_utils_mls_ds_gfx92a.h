#pragma once

#include "fwd/gfx938/qk_gemm_utils_mls_ds.h"


template<int kHeadDim, int kBlockM, int kBlockK, int WARP_M, typename Element, bool Is_even_MN>
__forceinline__ __device__ void prefetch_q_to_vgpr_mls_ds_gfx92a(
        vec4_uint q_ptr,
        Element* q_lds,
        union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2],
        int warp_id,
        int seqlen_q_stride,
        int max_seq_q_offset=0) {

    constexpr int WARP_NUM        = kBlockM / WARP_M;
    constexpr int Q_LDS_LOAD_NUM  = kBlockM * kBlockK / (32 * 32);
    constexpr int Q_LOAD_REQUESTS = Q_LDS_LOAD_NUM / WARP_NUM;
    constexpr int ELEMENT_BYTES   = sizeof(Element);

    int q_lds_base = reinterpret_cast<size_t>(q_lds);
    flash::wait_lds_data_arrived<true>(0);

    vec4_uint q_srsrc;
    q_srsrc[2] = seqlen_q_stride;

    const int q_row = warp_id * 32;
    const int nm_filter = q_row + 32 - max_seq_q_offset;
    const bool has_tail = max_seq_q_offset % kBlockM != 0;
    const int q_load_row = has_tail && nm_filter >= 32 ? 0 : q_row;
    // gfx92a has a 5-bit MLS filter field, so never encode 32.
    const int q_filter = inline_min_max<0, 31>(q_load_row + 32 - max_seq_q_offset);
    q_srsrc[3] = has_tail ? q_filter << 8 : 0;

    int stage_id = 0;
    {
        int k_loop = 0;
        *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_ptr + (k_loop * kBlockK + q_load_row * seqlen_q_stride) * ELEMENT_BYTES);
        int lds_offset = (stage_id * kBlockM * kBlockK + warp_id * 32 * 32) * ELEMENT_BYTES;
        inline_matrix_load_32x32_b16_lds_trans<0, 1>(q_lds, q_srsrc, lds_offset, 0);
    }

    stage_id ^= 1;
    #pragma unroll
    for(int k_loop = 1; k_loop < (kHeadDim / kBlockK); ++k_loop) {
        *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_ptr + (k_loop * kBlockK + q_load_row * seqlen_q_stride) * ELEMENT_BYTES);
        int lds_offset = (stage_id * kBlockM * kBlockK + warp_id * 32 * 32) * ELEMENT_BYTES;
        inline_matrix_load_32x32_b16_lds_trans<0, 1>(q_lds, q_srsrc, lds_offset, 0);

        stage_id ^= 1;
        buffer_load_lds_dwordx1_wait<Q_LOAD_REQUESTS>();
        __builtin_amdgcn_sched_barrier(0);

        int lds_load_offset = q_lds_base + (stage_id * kBlockM * kBlockK + warp_id * 32 * 32) * ELEMENT_BYTES;
        DS_READ_MATRIX_32X32_B16_GFX946(lds_load_offset, q_reg[(k_loop - 1) * 2].f16, q_reg[(k_loop - 1) * 2 + 1].f16, true);
        flash::wait_lds_data_arrived<true>(0);
    }

    {
        stage_id ^= 1;
        buffer_load_lds_dwordx1_wait<0>();
        __builtin_amdgcn_sched_barrier(0);
        constexpr int k_loop = kHeadDim / kBlockK - 1;
        int lds_load_offset = q_lds_base + (stage_id * kBlockM * kBlockK + warp_id * 32 * 32) * ELEMENT_BYTES;
        DS_READ_MATRIX_32X32_B16_GFX946(lds_load_offset, q_reg[k_loop * 2].f16, q_reg[k_loop * 2 + 1].f16, true);
    }
    __builtin_amdgcn_s_waitcnt(0);
    flash::wait_lds_data_arrived<true>(0);
}


template<int kHeadDim, int kBlockN, int kBlockK, int WARP_NUM, int WARP_N, typename Element, bool Is_even_MN>
__forceinline__ __device__ void prefetch_k_to_lds_mls_ds_gfx92a(
        vec4_uint k_ptr,
        Element* k_lds,
        int warp_id,
        int seqlen_k_stride,
        int max_seq_k_offset=0) {

    flash::wait_all_warp_arrived();

    constexpr int ELEMENT_BYTES = sizeof(Element);
    #pragma unroll
    for (int n_loop = 0; n_loop < kBlockN / WARP_N; ++n_loop) {
        vec4_uint k_srsrc;
        k_srsrc[2] = seqlen_k_stride;
        const bool has_tail = max_seq_k_offset % kBlockN != 0;
        const int nm_filter_max = n_loop * WARP_N + 32 - max_seq_k_offset;
        const int k_load_loop = has_tail && nm_filter_max >= 32 ? 0 : n_loop;
        const int nm_filter = inline_min_max<0, 31>(k_load_loop * WARP_N + 32 - max_seq_k_offset);
        *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_ptr + (k_load_loop * WARP_N * seqlen_k_stride + warp_id * 32) * ELEMENT_BYTES);
        k_srsrc[3] = has_tail ? nm_filter << 8 : 0;
        int lds_offset = (n_loop * WARP_N * kHeadDim + warp_id * 32 * 32) * ELEMENT_BYTES;
        inline_matrix_load_32x32_b16_lds_trans<0, 0>(k_lds, k_srsrc, lds_offset, 0);
    }
    __builtin_amdgcn_sched_barrier(0);
}
