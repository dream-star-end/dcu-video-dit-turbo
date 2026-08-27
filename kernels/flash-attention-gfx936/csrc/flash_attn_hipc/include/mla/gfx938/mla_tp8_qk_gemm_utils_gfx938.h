#pragma once
#include "intrinsic_mls_ds.h"


template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockK, int WARP_M, int WARP_NUM, int M_MMAC_COUNT, typename Element, typename ElementAccum>
__forceinline__ __device__ void mla_prefetch_q_to_vgpr_gfx938_with_initialization(
        vec4_uint q_addr,
        Element* q_lds,
        union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2],
        int warp_id,
        int query_seqlen_stride,
        int max_seq_q_offset,
        vec2_Accum<ElementAccum> scores_max[WARP_M / 32],
        vec2_Accum<ElementAccum> scores_sum[WARP_M / 32],
        vec4_Accum<ElementAccum> acc_o[kHeadDimV / kBlockK][4]) {

    flash::wait_all_warp_arrived();
    // prepare mls buffer resource registers
    vec4_uint q_srsrc;
    q_srsrc[2] = query_seqlen_stride;
    q_srsrc[3] = 0;
    // total 16x576 f16s
    // 16x128 f16s per wave first
    constexpr int LOAD = 4;
    constexpr int block32x16_bytes = 32 * 16 * sizeof(Element);
    #pragma unroll
    for (int load_id = 0; load_id < LOAD; ++load_id) {
        // lds address
        int lds_offset_bytes = (load_id * WARP_NUM + warp_id) * block32x16_bytes;
        // global offset
        int q_warp_offset = (load_id * WARP_NUM + warp_id) * 32;
        // compute global address
        *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_addr + q_warp_offset * sizeof(Element));
        // matrix load
        __builtin_amdgcn_sched_barrier(0);
        inline_matrix_load_32x16_b16_lds_trans<0, 1>(q_lds, q_srsrc, lds_offset_bytes, 0);
        __builtin_amdgcn_sched_barrier(0);
    }

    // insert valus in def-use
    attention_initialize<kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, M_MMAC_COUNT, ElementAccum>(scores_max, scores_sum, acc_o);

    // fetch data from lds, from MID-th blocks
    const int MID = 1;
    #pragma unroll
    for (int load_id = 0; load_id < MID; ++load_id) {
        // wait global data written to lds
        flash::wait_buffer_data_arrived<true/*sync*/>(LOAD - load_id - 1);
        #pragma unroll
        for (int i = 0; i < WARP_NUM; ++i) {
            DS_READ_MATRIX_32X16_B16(load_id * WARP_NUM * block32x16_bytes + i * block32x16_bytes, q_reg[(load_id * 4 + i) * 2].f16, true);
        }
    }

    // -------------------------------------------------------------------
    // prefetch rest 16x64 loads
    // 16x32 f16s 0-1 wave later
    int lds_offset_bytes = (LOAD * WARP_NUM + warp_id) * block32x16_bytes;
    int real_warp_id = warp_id >= 2 ? 0: warp_id;
    int q_warp_offset = (LOAD * WARP_NUM + real_warp_id) * 32;
    *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_addr + q_warp_offset * sizeof(Element));
    __builtin_amdgcn_sched_barrier(0);
    inline_matrix_load_32x16_b16_lds_trans<0, 1>(q_lds, q_srsrc, lds_offset_bytes, 0);
    __builtin_amdgcn_sched_barrier(0);

    // continue from MID
    #pragma unroll
    for (int load_id = MID; load_id < LOAD; ++load_id) {
        // wait global data written to lds
        flash::wait_buffer_data_arrived<true/*sync*/>(LOAD - load_id - 1 + MID);
        #pragma unroll
        for (int i = 0; i < WARP_NUM; ++i) {
            DS_READ_MATRIX_32X16_B16(load_id * WARP_NUM * block32x16_bytes + i * block32x16_bytes, q_reg[(load_id * 4 + i) * 2].f16, true);
        }
    }

    // wait global data written to lds
    flash::wait_buffer_data_arrived<true/*sync*/>(0);

    // write last data into registers
    DS_READ_MATRIX_32X16_B16((LOAD * WARP_NUM + 0) * block32x16_bytes, q_reg[(16 + 0) * 2].f16, true);
    DS_READ_MATRIX_32X16_B16((LOAD * WARP_NUM + 1) * block32x16_bytes, q_reg[(16 + 1) * 2].f16, true);

    // wait all data written to registers
    flash::wait_lds_data_arrived<true/*sync*/>(0);
}