#pragma once
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "static_switch.h"
#include "mla_pv_gemm_utils.h"


template<int kHeadDim, int kBlockM, int kBlockK, int WARP_M, int WARP_NUM, typename Element, int STAGES, int REUSE_KV_TIMES, int M_MMAC_COUNT>
__forceinline__ __device__ void mla_prefetch_q_to_vgpr(
        vec4_uint q_addr,
        Element* q_lds,
        vec2_Element<Element> q_reg[(kHeadDim / kBlockK) * ((WARP_M * kBlockK) / (32 * 32)) * 2][4],
        int warp_id,
        int query_seqlen_stride,
        int max_seq_q_offset = -1) {

    constexpr bool MMAC_32x32     = M_MMAC_COUNT > 1;
    constexpr int Q_LOAD_REQUESTS = (REUSE_KV_TIMES == 0) ? (kBlockM * kBlockK) / (4 * 32 * WARP_NUM): MMAC_32x32 ? ((REUSE_KV_TIMES + 1) >> 1) << 2 / WARP_NUM: 1/*MHA only need the first token*/;
    constexpr int SEQUENCE_READ   = MMAC_32x32 ? 2: 1;

    int lane_id             = threadIdx.x & 63; // lane id, 0-63
    int q_lane_m_idx        = ((lane_id >> 4) & 1) * 2 + ((lane_id >> 4) >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int q_lane_head_dim_idx = lane_id & 15;
    int laneid_shfl_4       = lane_id >> 4;
    int laneid_and_15       = lane_id & 15;
    int q_ds_read_offset    = (lane_id & 1) * 16 + (laneid_and_15 >> 1) * 64 + (laneid_and_15 >> 1) + (laneid_shfl_4 & 1) * 8 + (lane_id / 32);

    int stage_id = 0;
    if constexpr (STAGES > 1) {
        int k_loop = 0;
        int q_block_buffer_load_global_offset = k_loop * kBlockK;
        int q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 34);
        for (int load = 0, warp_loop = warp_id; load < Q_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
            int q_warp_buffer_load_m_id = warp_loop & (kBlockM / 4 - 1);
            int q_warp_buffer_load_lds_offset = q_lds_stage_offset + (q_warp_buffer_load_m_id >> 3) * (32 * 34) + (q_warp_buffer_load_m_id & 7) * (4 * 32);
            int offset_s = q_block_buffer_load_global_offset / 2;
            int offset_v = q_warp_buffer_load_m_id * 4 + q_lane_m_idx;
            if (offset_v < max_seq_q_offset) {
                int lds_offset = (q_warp_buffer_load_lds_offset + padding) / 2;
                offset_v = (offset_v * query_seqlen_stride) / 2 + q_lane_head_dim_idx;
                builtin_buffer_load_dword_lds(q_lds, q_addr, lds_offset, offset_s, offset_v);
            }
        }
    }
    if constexpr (STAGES > 1) stage_id ^= 1;
    constexpr int K_LOOP_START = (STAGES > 1) ? 1: 0;
    for (int k_loop = K_LOOP_START; k_loop < (kHeadDim / kBlockK); ++k_loop) {
        int q_block_buffer_load_global_offset = k_loop * kBlockK;
        int q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 34);
        for (int load = 0, warp_loop = warp_id; load < Q_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int padding = (warp_loop & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
            int q_warp_buffer_load_m_id = warp_loop & (kBlockM / 4 - 1);
            int q_warp_buffer_load_lds_offset = q_lds_stage_offset + ((q_warp_buffer_load_m_id >> 3) * (32 * 34) + (q_warp_buffer_load_m_id & 7) * (4 * 32));
            int offset_s = q_block_buffer_load_global_offset / 2;
            int offset_v = q_warp_buffer_load_m_id * 4 + q_lane_m_idx;
            if (offset_v < max_seq_q_offset) {
                int lds_offset = (q_warp_buffer_load_lds_offset + padding) / 2;
                offset_v = (offset_v * query_seqlen_stride) / 2 + q_lane_head_dim_idx;
                builtin_buffer_load_dword_lds(q_lds, q_addr, lds_offset, offset_s, offset_v);
            }
        }

        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        if constexpr (STAGES > 1) stage_id ^= 1;
        q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 17);

        vec2_Element<Element> *q_lds_v2fp16 = (vec2_Element<Element> *)(q_lds);
        #pragma unroll
        for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
                #pragma unroll
                for (int i = 0; i < SEQUENCE_READ; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int lds_offset = q_lds_stage_offset + head_dim_idx * kBlockM * 17 + j * 2 + i * 32 + q_ds_read_offset;
                        int k_loop_idx = (STAGES > 1) ? k_loop - 1: k_loop;
                        inline_ds_read_b32_wait(q_lds_v2fp16, lds_offset, q_reg[k_loop_idx * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i][j]);
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
        int q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 17);
        vec2_Element<Element> *q_lds_v2fp16 = (vec2_Element<Element> *)(q_lds);
        #pragma unroll
        for (int head_dim_idx = 0; head_dim_idx < kBlockK / 32; ++head_dim_idx) {
            #pragma unroll
            for (int m_idx = 0; m_idx < WARP_M / 32; ++m_idx) {
                #pragma unroll
                for (int i = 0; i < SEQUENCE_READ; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int lds_offset = q_lds_stage_offset + head_dim_idx * kBlockM * 17 + j * 2 + i * 32 + q_ds_read_offset;
                        inline_ds_read_b32_wait(q_lds_v2fp16, lds_offset, q_reg[((kHeadDim / kBlockK) - 1) * (WARP_M * kBlockK) / (32 * 32) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i][j]);
                    }
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
}




template<int kHeadDim, int kBlockM, int kBlockK, int WARP_NUM, typename Element, int STAGES, int REUSE_KV_TIMES, int M_MMAC_COUNT>
__forceinline__ __device__ void mla_prefetch_q_to_lds_stage1(
        vec4_uint q_addr,
        Element* q_lds,
        int warp_id,
        int query_seqlen_stride,
        int max_seq_q_offset = -1) {
    constexpr bool MMAC_32x32    = M_MMAC_COUNT > 1 and REUSE_KV_TIMES > 0; /*mtp <= 16 的时候, 也是走这条路径, 只需要加载 16x576 个 Half 到 lds*/
    constexpr int Q_LOAD_REQUESTS = (REUSE_KV_TIMES == 0) ? (kBlockM * kBlockK) / (4 * 32 * WARP_NUM): MMAC_32x32 ? ((REUSE_KV_TIMES + 1) >> 1) << 2 / WARP_NUM: 1/*MHA only need the first token*/;

    int lane_id             = threadIdx.x & 63; // lane id, 0-63
    int q_lane_m_idx        = ((lane_id >> 4) & 1) * 2 + ((lane_id >> 4) >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int q_lane_head_dim_idx = lane_id & 15;

    constexpr int K_LOOP_START = 0;
    for (int k_loop = K_LOOP_START; k_loop < (kHeadDim / kBlockK); k_loop++) {
        // global->lds, left matrix
        int q_block_buffer_load_global_offset = k_loop * kBlockK;
        int q_lds_stage_offset = k_loop * (kBlockM / 32) * (kBlockK / 32) * (32 * 17) >> 1;
        for (int load = 0, warp_loop = warp_id; load < Q_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int q_warp_buffer_load_m_id = warp_loop & (kBlockM / 4 - 1);
            int offset_s = q_block_buffer_load_global_offset / 2;
            int offset_v = q_warp_buffer_load_m_id * 4 + q_lane_m_idx;
            if (offset_v < max_seq_q_offset) {
                int q_warp_buffer_load_lds_offset = (q_warp_buffer_load_m_id >> 3) * (32 * 17) + (q_warp_buffer_load_m_id & 7) * (4 * 32);
                int lds_offset = q_lds_stage_offset + (q_warp_buffer_load_lds_offset >> 1) + (warp_loop & 7/*padding*/);
                offset_v = (offset_v * query_seqlen_stride) / 2 + q_lane_head_dim_idx;
                builtin_buffer_load_dword_lds(q_lds, q_addr, lds_offset, offset_s, offset_v);
            }
        }
    }

    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
}



template<int kBlockK, int WARP_N, typename Element, int STAGES, int WARP_NUM>
__forceinline__ __device__ void mla_prefetch_k_to_lds(
        vec4_uint k_addr,
        Element* k_lds,
        int warp_id,
        int kvcache_seqlen_stride,
        int max_seq_k_offset=-1) {

    constexpr int k_lds_load_num  = (WARP_N * kBlockK) / (4 * 32);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num;

    // compute offset pattern
    int lane_id       = threadIdx.x & 63; // lane id, 0-63
    int laneid_shfl_4 = lane_id >> 4;
    int laneid_and_15 = lane_id & 15;
    int qk_lane_m_idx = (laneid_shfl_4 & 1) * 2 + (laneid_shfl_4 >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int qk_lane_head_dim_idx = laneid_and_15;

    // decide buffer load func
    auto BUFFER_LOAD_FUNC = &inline_buffer_load_dword_lds<Element, 2>;

    int stage_id = 0;
    int k_loop   = 0;
    if constexpr (STAGES > 1) {
        int k_block_buffer_load_global_offset = k_loop * kBlockK + warp_id * WARP_N * kvcache_seqlen_stride;
        int k_lds_stage_offset = (stage_id * WARP_NUM + warp_id) * (WARP_N / 32) * (kBlockK / 32) * (32 * 34);
        for (int load = 0; load < K_LOAD_REQUESTS; ++load) {
            int __load     = (load + warp_id) % K_LOAD_REQUESTS;
            int padding    = (__load & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
            int k_warp_buffer_load_n_id = __load & (WARP_N / 4 - 1);
            int k_warp_buffer_load_lds_offset = k_lds_stage_offset + (k_warp_buffer_load_n_id >> 3) * (32 * 34) + (k_warp_buffer_load_n_id & 7) * (4 * 32);
            int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
            int offset_s   = k_block_buffer_load_global_offset / 2;
            int offset_v   = min(k_warp_buffer_load_n_id * 4 + qk_lane_m_idx, max_seq_k_offset - 1) * kvcache_seqlen_stride / 2 + qk_lane_head_dim_idx;
            BUFFER_LOAD_FUNC(k_lds, k_addr, lds_offset, offset_s, offset_v);
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}