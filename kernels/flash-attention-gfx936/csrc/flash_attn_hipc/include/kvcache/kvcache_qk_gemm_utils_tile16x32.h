#pragma once
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "static_switch.h"
#include "kvcache_pv_gemm_utils_tile16x32.h"


template<int kHeadDim, int kBlockM, int kBlockK, int WARP_M, int WARP_NUM, typename Element, int STAGES, int M_MMAC_COUNT>
__forceinline__ __device__ void kvcache_prefetch_q_to_vgpr_tile16x32(
        vec4_uint q_addr,
        Element* q_lds,
        union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * ((WARP_M * kBlockK) / (32 * 32)) * 2],
        int warp_id,
        int query_seqlen_stride,
        int max_seq_q_offset=-1) {

    constexpr int Q_LOAD_REQUESTS = (kBlockM * kBlockK >> 1/*16x32 tile*/) * M_MMAC_COUNT / (4 * 32 * WARP_NUM);
    constexpr int SEQUENCE_READ   = M_MMAC_COUNT;
    constexpr int READ_ONCE_LINES = 4;
    auto BUFFER_LOAD_FUNC         = &builtin_buffer_load_dword_lds<Element, float, 1>; // buffer_load_dwordx4 can also be applied if necessary

    int lane_id             = threadIdx.x & 63; // lane id, 0-63
    int q_lane_m_idx        = lane_id >> 4;
    int q_lane_head_dim_idx = lane_id & 15;
    int laneid_shfl_4       = lane_id >> 4;
    int laneid_and_15       = lane_id & 15;
    int q_ds_read_offset    = laneid_and_15 * 16 + (laneid_shfl_4 & 1) * 8 + (lane_id / 32) * 4;

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
            int q_warp_buffer_load_lds_offset = q_lds_stage_offset + ((q_warp_buffer_load_m_id >> 3) * (32 * 32) + (q_warp_buffer_load_m_id & 7) * (READ_ONCE_LINES * 32));
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
                    int lds_offset = q_lds_stage_offset + head_dim_idx * kBlockM * 16 + i * 16 * 16 + q_ds_read_offset;
                    int k_loop_idx = (STAGES > 1) ? k_loop - 1: k_loop;
                    q_reg[k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i].f32 = *(vec4_fp32*)(q_lds_v2fp16 + lds_offset);
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
                    int lds_offset = q_lds_stage_offset + head_dim_idx * kBlockM * 16 + i * 16 * 16 + q_ds_read_offset;
                    q_reg[((kHeadDim / kBlockK) - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i].f32 = *(vec4_fp32*)(q_lds_v2fp16 + lds_offset);
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
}



template<int kBlockK, int WARP_N, typename Element, int STAGES, int WARP_NUM>
__forceinline__ __device__ void kvcache_prefetch_k_to_lds_tile16x32(
        vec4_uint k_addr,
        Element* k_lds,
        int warp_id,
        int kvcache_seqlen_stride,
        int max_seq_k_offset=-1) {

    // 预先计算一些表达式
    int lane_id       = threadIdx.x & 63; // lane id, 0-63
    int laneid_shfl_4 = lane_id >> 4;
    int laneid_and_15 = lane_id & 15;

#if defined(__gfx936__) || defined(__gfx938__)  // >= bmz
    int qk_lane_m_idx             = lane_id >> 2;
    int qk_lane_head_dim_idx      = (lane_id & 3) << 2;
    auto BUFFER_LOAD_FUNC         = &inline_buffer_load_dwordx4_lds<Element, 2>;
    constexpr int READ_ONCE_LINES = 16;
#else // zd
    int qk_lane_m_idx             = laneid_shfl_4;
    int qk_lane_head_dim_idx      = laneid_and_15;
    auto BUFFER_LOAD_FUNC         = &inline_buffer_load_dword_lds<Element, 2>;
    constexpr int READ_ONCE_LINES = 4;
#endif
    constexpr int k_lds_load_num  = (WARP_N * kBlockK) / (READ_ONCE_LINES * 32);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num;

    int stage_id = 0;
    int k_loop   = 0;
    if constexpr (STAGES > 1) {
        int k_block_buffer_load_global_offset = k_loop * kBlockK;
        int k_lds_stage_offset = (warp_id * STAGES * 2 + stage_id * 2) * (WARP_N / 32) * (kBlockK / 32) * (32 * 32);
        #pragma unroll
        for (int load = 0; load < K_LOAD_REQUESTS; ++load) {
            int k_warp_buffer_load_n_id = load & (WARP_N / READ_ONCE_LINES - 1);
            int k_warp_buffer_load_lds_offset = k_lds_stage_offset + (k_warp_buffer_load_n_id >> 3) * (32 * 32) + (k_warp_buffer_load_n_id & 7) * (READ_ONCE_LINES * 32);
            int lds_offset = k_warp_buffer_load_lds_offset / 2;
            int offset_s   = k_block_buffer_load_global_offset / 2;
            int offset_v   = min(k_warp_buffer_load_n_id * READ_ONCE_LINES + qk_lane_m_idx + warp_id * WARP_N, max_seq_k_offset - 1) * kvcache_seqlen_stride / 2 + qk_lane_head_dim_idx;
            BUFFER_LOAD_FUNC(k_lds, k_addr, lds_offset, offset_s, offset_v);
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (STAGES > 1) {
        int k_block_buffer_load_global_offset = (k_loop + 1) * kBlockK;
        int k_lds_stage_offset = (warp_id * STAGES * 2 + stage_id * 2 + 1) * (WARP_N / 32) * (kBlockK / 32) * (32 * 32);
        #pragma unroll
        for (int load = 0; load < K_LOAD_REQUESTS; ++load) {
            int k_warp_buffer_load_n_id = load & (WARP_N / READ_ONCE_LINES - 1);
            int k_warp_buffer_load_lds_offset = k_lds_stage_offset + (k_warp_buffer_load_n_id >> 3) * (32 * 32) + (k_warp_buffer_load_n_id & 7) * (READ_ONCE_LINES * 32);
            int lds_offset = k_warp_buffer_load_lds_offset / 2;
            int offset_s   = k_block_buffer_load_global_offset / 2;
            int offset_v   = min(k_warp_buffer_load_n_id * READ_ONCE_LINES + qk_lane_m_idx + warp_id * WARP_N, max_seq_k_offset - 1) * kvcache_seqlen_stride / 2 + qk_lane_head_dim_idx;
            BUFFER_LOAD_FUNC(k_lds, k_addr, lds_offset, offset_s, offset_v);
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}