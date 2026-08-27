#pragma once
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "static_switch.h"
#include "pv_gemm_utils.h"


template<int kHeadDim, int kBlockM, int kBlockK, int WARP_M, typename Element, bool Is_even_MN>
__forceinline__ __device__ void  prefetch_q_to_vgpr(
        vec4_uint gQ,
        Element* q_lds,
        vec2_Element<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2][4],
        int WARP_ID,
        int seqlen_q_stride,
        int max_seq_q_offset=-1) {

    constexpr int WARP_NUM = kBlockM / WARP_M;
    constexpr int q_lds_load_num = kBlockM * kBlockK / (4 * 32);
    constexpr int Q_LOAD_REQUESTS = q_lds_load_num / WARP_NUM;

    int lane_id             = threadIdx.x & 63; // lane id, 0-63
    int q_lane_m_idx        = ((lane_id >> 4) & 1) * 2 + ((lane_id >> 4) >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int q_lane_head_dim_idx = lane_id & 15;

    int stage_id = 0;
    {
        int k_loop = 0;
        // global->lds, left matrix
        int q_block_buffer_load_global_offset = k_loop * kBlockK;
        const int q_lds_load_num = kBlockM * kBlockK / (4 * 32);
        int q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 34);

        for(int load = 0,warp_loop = WARP_ID; load < Q_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
            int q_warp_buffer_load_m_id = warp_loop & (kBlockM / 4 - 1);
            int q_warp_buffer_load_lds_offset = q_lds_stage_offset + (q_warp_buffer_load_m_id >> 3) * (32 * 34) + (q_warp_buffer_load_m_id & 7) * (4 * 32);
            int s_offset = q_block_buffer_load_global_offset / 2;
            int seqlen_pos = q_warp_buffer_load_m_id * 4 + q_lane_m_idx;
            if constexpr (not Is_even_MN) {
                seqlen_pos = min(seqlen_pos, max_seq_q_offset - 1);
            }
            int v_offset = seqlen_pos * seqlen_q_stride / 2 + q_lane_head_dim_idx;
            int lds_offset = (q_warp_buffer_load_lds_offset + padding) / 2;
            builtin_buffer_load_dword_lds(q_lds, gQ, lds_offset, s_offset, v_offset);
        }
    }
    stage_id ^= 1;
    for(int k_loop = 1; k_loop < (kHeadDim / kBlockK); ++k_loop) {
        // global->lds, left matrix
        int q_block_buffer_load_global_offset = k_loop * kBlockK;
        int q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 34);
        for(int load = 0,warp_loop = WARP_ID; load < Q_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
            int q_warp_buffer_load_m_id = warp_loop & (kBlockM / 4 - 1);
            int q_warp_buffer_load_lds_offset = q_lds_stage_offset + (q_warp_buffer_load_m_id >> 3) * (32 * 34) + (q_warp_buffer_load_m_id & 7) * (4 * 32);
            int s_offset = q_block_buffer_load_global_offset / 2;
            int seqlen_pos = q_warp_buffer_load_m_id * 4 + q_lane_m_idx;
            if constexpr (not Is_even_MN) {
                seqlen_pos = min(seqlen_pos, max_seq_q_offset - 1);
            }
            int v_offset = seqlen_pos * seqlen_q_stride / 2 + q_lane_head_dim_idx;
            int lds_offset = (q_warp_buffer_load_lds_offset + padding) / 2;
            builtin_buffer_load_dword_lds(q_lds, gQ, lds_offset, s_offset, v_offset);
        }

        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        stage_id ^= 1;
        q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 17);

        vec2_Element<Element> *q_lds_v2fp16 = (vec2_Element<Element> *)(q_lds);
        #pragma unroll
        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                #pragma unroll
                for(int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for(int j = 0; j < 4; ++j) {
                        int lds_offset = q_lds_stage_offset + head_dim_idx * kBlockM * 17 + (WARP_ID * (WARP_M / 32) + m_idx) * (32 * 17) + j * 2 + i * 32 + (lane_id & 1) * 16 + ((lane_id & 15) >> 1) * 64 + /*padding*/ ((lane_id & 15) >> 1) + ((lane_id / 16) & 1) * 8 + (lane_id / 32);
                        inline_ds_read_b32_wait(q_lds_v2fp16, lds_offset, q_reg[(k_loop - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i][j]);
                    }
                }
            }
        }
        __syncthreads();
        // __builtin_amdgcn_sched_barrier(0);
    }

    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    stage_id ^= 1;
    int q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 17);
    vec2_Element<Element> *q_lds_v2fp16 = (vec2_Element<Element> *)(q_lds);
    #pragma unroll
    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 32); ++head_dim_idx) {
        #pragma unroll
        for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
            #pragma unroll
            for(int i = 0; i < 2; ++i) {
                #pragma unroll
                for(int j = 0; j < 4; ++j) {
                    int lds_offset = q_lds_stage_offset + head_dim_idx * kBlockM * 17 + (WARP_ID * (WARP_M / 32) + m_idx) * (32 * 17) + j * 2 + i * 32 + (lane_id & 1) * 16 + ((lane_id & 15) >> 1) * 64 + /*padding*/ ((lane_id & 15) >> 1) + ((lane_id / 16) & 1) * 8 + (lane_id / 32);
                    inline_ds_read_b32_wait(q_lds_v2fp16, lds_offset, q_reg[((kHeadDim / kBlockK) - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i][j]);
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    __builtin_amdgcn_sched_barrier(0);
}




template<int kHeadDim, int kBlockN, int kBlockK, int WARP_NUM, int WARP_N, typename Element, bool Is_even_MN, int STAGES=2>
__forceinline__ __device__ void prefetch_k_to_lds(
        vec4_uint gK,
        Element* k_lds,
        int WARP_ID,
        int seqlen_k_stride,
        int max_seq_k_offset=-1) {

    // constexpr int WARP_NUM = kBlockN / WARP_N;
    constexpr int k_lds_load_num  = (WARP_N * kBlockK) / (4 * 32);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num / WARP_NUM;

    int lane_id = threadIdx.x & 63; // lane id, 0-63
    int laneid_shfl_4 = lane_id >> 4;
    int laneid_and_15 = lane_id & 15;
    int qk_lane_m_idx = (laneid_shfl_4 & 1) * 2 + (laneid_shfl_4 >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int qk_lane_head_dim_idx = laneid_and_15;

    int stage_id = 0;
    int k_loop   = 0;
    int k_block_buffer_load_global_offset = k_loop * kBlockK;
    int k_lds_stage_offset = stage_id * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
    for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
        int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
        int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
        int k_warp_buffer_load_lds_offset = k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32));
        int s_offset = k_block_buffer_load_global_offset / 2;
        int seqlen_pos = k_warp_buffer_load_n_id * 4 + qk_lane_m_idx;
        if constexpr (not Is_even_MN) {
            seqlen_pos = min(seqlen_pos, max_seq_k_offset - 1);
        }
        int v_offset = seqlen_pos * seqlen_k_stride / 2 + qk_lane_head_dim_idx;
        int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
        inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
    }

    __builtin_amdgcn_sched_barrier(0);
    if constexpr (kHeadDim == 128 or kHeadDim == 64) {
        int k_block_buffer_load_global_offset = k_loop * kBlockK;
        int k_lds_stage_offset = (stage_id * STAGES + 1) * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
        for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
            int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
            int k_warp_buffer_load_lds_offset = k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32));
            int s_offset = k_block_buffer_load_global_offset / 2;
            int seqlen_pos = k_warp_buffer_load_n_id * 4 + qk_lane_m_idx + WARP_N;
            if constexpr (not Is_even_MN) {
                seqlen_pos = min(seqlen_pos, max_seq_k_offset - 1);
            }
            int v_offset = seqlen_pos * seqlen_k_stride / 2 + qk_lane_head_dim_idx;
            int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
            inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}
namespace flash {
template<int kHeadDim, int kBlockK, int WARP_M, typename Element, bool Is_even_MN>
__forceinline__ __device__ void prefix_prefill_hdim512_16x64_fetch_q_to_vgpr(
        const Element* q_ptr,
        union_vec4_f16x2<Element> q_regs[kHeadDim / kBlockK],
        const int warp_id,
        const int lane_id,
        const int q_row_stride,
        const int max_seq_q_offset) {
    static_assert(kHeadDim == 512);
    static_assert(kBlockK == 32);
    static_assert(WARP_M == 16);

    const int row_in_block_raw = warp_id * WARP_M + (lane_id & 15);
    const int row_in_block = Is_even_MN ? row_in_block_raw : min(row_in_block_raw, max_seq_q_offset - 1);
    const int col_in_vec = (lane_id >> 4) * 8;
    #pragma unroll
    for (int k_loop = 0; k_loop < kHeadDim / kBlockK; ++k_loop) {
        q_regs[k_loop] = *reinterpret_cast<const union_vec4_f16x2<Element>*>(
            q_ptr + row_in_block * int64_t(q_row_stride) + k_loop * kBlockK + col_in_vec);
    }
}

template<int kBlockN, int kBlockK, int WARP_M, typename Element, bool Is_even_MN>
__forceinline__ __device__ void prefix_prefill_hdim512_16x64_prefetch_k_to_lds(
        vec4_uint gK,
        Element* k_lds,
        const int warp_id,
        const int lane_id,
        const int k_row_stride,
        const int max_seq_k_offset,
        const int k_loop) {
    static_assert(kBlockN == 64);
    static_assert(kBlockK == 32);
    static_assert(WARP_M == 16);

    const int row_raw = warp_id * WARP_M + (lane_id >> 2);
    const int row = Is_even_MN ? row_raw : min(row_raw, max_seq_k_offset - 1);
    const int col = k_loop * kBlockK + (lane_id & 3) * 8;
    const int lds_offset = row_raw * kBlockK + (lane_id & 3) * 8;
    safe_inline_buffer_load_dwordx4_lds<Element, 1>(
        k_lds, gK, lds_offset, 0, row * k_row_stride + col);
}

template<int kHeadDim, int kBlockN, int kBlockK, int WARP_M, typename Element, typename ElementAccum, bool Is_even_MN>
__forceinline__ __device__ void prefix_prefill_hdim512_16x64_qk(
        vec4_uint gK,
        Element* k_lds,
        union_vec4_f16x2<Element> q_regs[kHeadDim / kBlockK],
        vec4_Accum<ElementAccum> s_reg[kBlockN / 16],
        const int warp_id,
        const int lane_id,
        const int k_row_stride,
        const int max_seq_k_offset) {
    static_assert(kHeadDim == 512);
    static_assert(kBlockN == 64);
    static_assert(kBlockK == 32);
    static_assert(WARP_M == 16);

    #pragma unroll
    for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
        inline_vgpr4_init_zero(s_reg[n_loop]);
    }

    #pragma unroll
    for (int k_loop = 0; k_loop < kHeadDim / kBlockK; ++k_loop) {
        prefix_prefill_hdim512_16x64_prefetch_k_to_lds<kBlockN, kBlockK, WARP_M, Element, Is_even_MN>(
            gK, k_lds, warp_id, lane_id, k_row_stride, max_seq_k_offset, k_loop);
        wait_buffer_data_arrived<true>(0);

        union_vec4_f16x2<Element> k_regs[kBlockN / 16];
        #pragma unroll
        for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
            const int lds_offset = n_loop * 16 * kBlockK + (lane_id & 15) * kBlockK + (lane_id >> 4) * 8;
            inlineasm_ds_read_b128(reinterpret_cast<size_t>(k_lds + lds_offset), k_regs[n_loop]);
        }
        wait_lds_data_arrived<false>(0);

        #pragma unroll
        for (int n_loop = 0; n_loop < kBlockN / 16; ++n_loop) {
            #pragma unroll
            for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                s_reg[n_loop].f32 = mmac<Element, ElementAccum>(
                    q_regs[k_loop].f16x4[min_tile_k],
                    k_regs[n_loop].f16x4[min_tile_k],
                    s_reg[n_loop].f32);
            }
        }
        wait_all_warp_arrived();
    }
}

} // namespace flash
