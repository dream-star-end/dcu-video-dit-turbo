#pragma once
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "static_switch.h"
#include "int8_pv_gemm_utils.h"


template<int kHeadDim, int kBlockM, int kBlockK, int WARP_M, typename Element_k, bool Is_even_MN>
__forceinline__ __device__ void  int8_prefetch_q_to_vgpr(
        vec4_uint gQ,
        Element_k* q_lds,
        // vec2_Element<Element_k> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2][4],
        vec4_int8 q_reg[(kHeadDim/kBlockK)*((WARP_M*kBlockK)/(32*kBlockK))*2][4],
        int WARP_ID,
        int seqlen_q_stride,
        int max_seq_q_offset=-1) {

    constexpr int WARP_NUM = kBlockM / WARP_M;  
    constexpr int q_lds_load_num = kBlockM * kBlockK / (4 * 64);
    constexpr int Q_LOAD_REQUESTS = q_lds_load_num / WARP_NUM;

    int lane_id             = threadIdx.x & 63; // lane id, 0-63
    int q_lane_m_idx        = ((lane_id >> 4) & 1) * 2 + ((lane_id >> 4) >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int q_lane_head_dim_idx = lane_id & 15;

    int stage_id = 0;
    {
        int k_loop = 0;
        // global->lds, left matrix
        int q_block_buffer_load_global_offset = k_loop * kBlockK;
        const int q_lds_load_num = kBlockM * kBlockK / (4 * 64);
        int q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 34);

        for(int load = 0,warp_loop = WARP_ID; load < Q_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int padding = (warp_loop & 7) * 4; // padding size in shared memory per buffer load, to avoid bank conflict
            int q_warp_buffer_load_m_id = warp_loop & (kBlockM / 4 - 1);
            int q_warp_buffer_load_lds_offset = q_lds_stage_offset + (q_warp_buffer_load_m_id >> 3) * (32 * 68) + (q_warp_buffer_load_m_id & 7) * (4 * 64);
            int s_offset = q_block_buffer_load_global_offset / 4;
            int seqlen_pos = q_warp_buffer_load_m_id * 4 + q_lane_m_idx;
            if constexpr (not Is_even_MN) {
                seqlen_pos = min(seqlen_pos, max_seq_q_offset - 1);
            }
            int v_offset = seqlen_pos * seqlen_q_stride / 4 + q_lane_head_dim_idx;
            int lds_offset = (q_warp_buffer_load_lds_offset + padding) / 4;
            builtin_buffer_load_dword_lds(q_lds, gQ, lds_offset, s_offset, v_offset);
        }
    }
    stage_id ^= 1;
    for(int k_loop = 1; k_loop < (kHeadDim / kBlockK); ++k_loop) {
        // global->lds, left matrix
        int q_block_buffer_load_global_offset = k_loop * kBlockK;
        int q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 32) * (32 * 34);
        for(int load = 0,warp_loop = WARP_ID; load < Q_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int padding = (warp_loop & 7) * 4; // padding size in shared memory per buffer load, to avoid bank conflict
            int q_warp_buffer_load_m_id = warp_loop & (kBlockM / 4 - 1);
            int q_warp_buffer_load_lds_offset = q_lds_stage_offset + (q_warp_buffer_load_m_id >> 3) * (32 * 68) + (q_warp_buffer_load_m_id & 7) * (4 * 64);
            int s_offset = q_block_buffer_load_global_offset / 4;
            int seqlen_pos = q_warp_buffer_load_m_id * 4 + q_lane_m_idx;
            if constexpr (not Is_even_MN) {
                seqlen_pos = min(seqlen_pos, max_seq_q_offset - 1);
            }
            int v_offset = seqlen_pos * seqlen_q_stride / 4 + q_lane_head_dim_idx;
            int lds_offset = (q_warp_buffer_load_lds_offset + padding) / 4;
            builtin_buffer_load_dword_lds(q_lds, gQ, lds_offset, s_offset, v_offset);
        }

        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        stage_id ^= 1;
        q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 64) * (32 * 17);

        // vec2_Element<Element> *q_lds_v2fp16 = (vec2_Element<Element> *)(q_lds);
        vec4_int8 *q_lds_v4int8 =  (vec4_int8 *)(q_lds);
        #pragma unroll
        for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 64); ++head_dim_idx) {
            #pragma unroll
            for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
                #pragma unroll
                for(int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for(int j = 0; j < 4; ++j) {
                        int lds_offset = q_lds_stage_offset + head_dim_idx * kBlockM * 17 + (WARP_ID * (WARP_M / 32) + m_idx) * (32 * 17) + j * 2 + i * 32 + (lane_id & 1) * 16 + ((lane_id & 15) >> 1) * 64 + /*padding*/ ((lane_id & 15) >> 1) + ((lane_id / 16) & 1) * 8 + (lane_id / 32);
                        inline_ds_read_b32_wait(q_lds_v4int8, lds_offset, q_reg[(k_loop - 1) * (WARP_M * kBlockK) / (32 * 64) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i][j]);
                        // printf("lds_offset is %d, q_reg is %d\n", lds_offset*4, q_reg[(k_loop - 1) * (WARP_M * kBlockK) / (32 * 64) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i][j][0]);
                        // printf("lds_offset is %d, q_reg is %d\n", lds_offset*4+1, q_reg[(k_loop - 1) * (WARP_M * kBlockK) / (32 * 64) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i][j][1]);
                        // printf("lds_offset is %d, q_reg is %d\n", lds_offset*4+2, q_reg[(k_loop - 1) * (WARP_M * kBlockK) / (32 * 64) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i][j][2]);
                        // printf("lds_offset is %d, q_reg is %d\n", lds_offset*4+3, q_reg[(k_loop - 1) * (WARP_M * kBlockK) / (32 * 64) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i][j][3]);
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
    int q_lds_stage_offset = stage_id * (kBlockM / 32) * (kBlockK / 64) * (32 * 17);
    // vec2_Element<Element> *q_lds_v2fp16 = (vec2_Element<Element> *)(q_lds);
    vec4_int8 *q_lds_v4int8 =  (vec4_int8 *)(q_lds);
    #pragma unroll
    for(int head_dim_idx = 0; head_dim_idx < (kBlockK / 64); ++head_dim_idx) {
        #pragma unroll
        for(int m_idx = 0; m_idx < (WARP_M / 32); ++m_idx) {
            #pragma unroll
            for(int i = 0; i < 2; ++i) {
                #pragma unroll
                for(int j = 0; j < 4; ++j) {
                    int lds_offset = q_lds_stage_offset + head_dim_idx * kBlockM * 17 + (WARP_ID * (WARP_M / 32) + m_idx) * (32 * 17) + j * 2 + i * 32 + (lane_id & 1) * 16 + ((lane_id & 15) >> 1) * 64 + /*padding*/ ((lane_id & 15) >> 1) + ((lane_id / 16) & 1) * 8 + (lane_id / 32);
                    inline_ds_read_b32_wait(q_lds_v4int8, lds_offset, q_reg[((kHeadDim / kBlockK) - 1) * (WARP_M * kBlockK) / (32 * 64) * 2 + (head_dim_idx * (WARP_M / 32) + m_idx) * 2 + i][j]);
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    __builtin_amdgcn_sched_barrier(0);
}




template<int kHeadDim, int kBlockN, int kBlockK, int WARP_NUM, int WARP_N, typename Element_k, bool Is_even_MN, int STAGES=2>
__forceinline__ __device__ void int8_prefetch_k_to_lds(
        vec4_uint gK,
        Element_k* k_lds,
        int WARP_ID,
        int seqlen_k_stride,
        int max_seq_k_offset=-1
        /*Element_k* k_ptr*/) {

    // constexpr int WARP_NUM = kBlockN / WARP_N;
    constexpr int k_lds_load_num  = (WARP_N * kBlockK) / (4 * 64);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num / WARP_NUM;

    int lane_id = threadIdx.x & 63; // lane id, 0-63
    int laneid_shfl_4 = lane_id >> 4;
    int laneid_and_15 = lane_id & 15;
    int qk_lane_m_idx = (laneid_shfl_4 & 1) * 2 + (laneid_shfl_4 >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int qk_lane_head_dim_idx = laneid_and_15;

    int stage_id = 0;
    int k_loop   = 0;
    int k_block_buffer_load_global_offset = k_loop * kBlockK;
    int k_lds_stage_offset = stage_id * (WARP_N / 32) * (kBlockK / 64) * (32 * 68);

    for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
        int padding = (warp_loop & 7) * 4; // padding size in shared memory per buffer load, to avoid bank conflict
        int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
        int k_warp_buffer_load_lds_offset = k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 68) + (k_warp_buffer_load_n_id & 7) * (4 * 64));
        int s_offset = k_block_buffer_load_global_offset / 4;
        int seqlen_pos = k_warp_buffer_load_n_id * 4 + qk_lane_m_idx;
        if constexpr (not Is_even_MN) {
            seqlen_pos = min(seqlen_pos, max_seq_k_offset - 1);
        }
        int v_offset = seqlen_pos * seqlen_k_stride / 4 + qk_lane_head_dim_idx;
        int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 4;
        // printf("WARP_ID is %d, seqlen_k_stride is %d, max_seq_k_offset is %d, lds_offset is %d\n", WARP_ID, seqlen_k_stride, max_seq_k_offset, lds_offset*4);
        inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
        // __builtin_amdgcn_sched_barrier(0);
        // buffer_load_lds_dwordx1_wait<0>();
        // __builtin_amdgcn_sched_barrier(0);
        // int global_offset = v_offset*4+k_block_buffer_load_global_offset;
        // printf("warp_loop is %d, WARP_ID is %d, lane_id is %d, seqlen_pos is %d, seqlen_k_stride is %d, max_seq_k_offset is %d, global offset is %d, lds_offset is %d, value is %d\n", warp_loop, WARP_ID, lane_id, seqlen_pos, seqlen_k_stride, max_seq_k_offset, global_offset, lds_offset*4 + lane_id*4, k_lds[lds_offset*4 + lane_id*4]);
        // printf("warp_loop is %d, WARP_ID is %d, lane_id is %d, seqlen_pos is %d, seqlen_k_stride is %d, max_seq_k_offset is %d, global offset is %d, lds_offset is %d, value is %d\n", warp_loop, WARP_ID, lane_id, seqlen_pos, seqlen_k_stride, max_seq_k_offset, global_offset+1, lds_offset*4 + lane_id*4+1, k_lds[lds_offset*4 + lane_id*4+1]);
        // printf("warp_loop is %d, WARP_ID is %d, lane_id is %d, seqlen_pos is %d, seqlen_k_stride is %d, max_seq_k_offset is %d, global offset is %d, lds_offset is %d, value is %d\n", warp_loop, WARP_ID, lane_id, seqlen_pos, seqlen_k_stride, max_seq_k_offset, global_offset+2, lds_offset*4 + lane_id*4+2, k_lds[lds_offset*4 + lane_id*4+2]);
        // printf("warp_loop is %d, WARP_ID is %d, lane_id is %d, seqlen_pos is %d, seqlen_k_stride is %d, max_seq_k_offset is %d, global offset is %d, lds_offset is %d, value is %d\n", warp_loop, WARP_ID, lane_id, seqlen_pos, seqlen_k_stride, max_seq_k_offset, global_offset+3, lds_offset*4 + lane_id*4+3, k_lds[lds_offset*4 + lane_id*4+3]);
    }

    __builtin_amdgcn_sched_barrier(0);
    if constexpr (kHeadDim == 128 or kHeadDim == 64 or kHeadDim == 192) {
        int k_block_buffer_load_global_offset = k_loop * kBlockK;
        int k_lds_stage_offset = (stage_id * STAGES + 1) * (WARP_N / 32) * (kBlockK / 64) * (32 * 68);
        for(int load = 0, warp_loop = WARP_ID; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int padding = (warp_loop & 7) * 4; // padding size in shared memory per buffer load, to avoid bank conflict
            int k_warp_buffer_load_n_id = warp_loop & (WARP_N / 4 - 1);
            int k_warp_buffer_load_lds_offset = k_lds_stage_offset + ((k_warp_buffer_load_n_id >> 3) * (32 * 68) + (k_warp_buffer_load_n_id & 7) * (4 * 64));
            int s_offset = k_block_buffer_load_global_offset / 4;
            int seqlen_pos = k_warp_buffer_load_n_id * 4 + qk_lane_m_idx + WARP_N;
            if constexpr (not Is_even_MN) {
                seqlen_pos = min(seqlen_pos, max_seq_k_offset - 1);
            }
            int v_offset = seqlen_pos * seqlen_k_stride / 4 + qk_lane_head_dim_idx;
            // printf("global offset is %d\n", v_offset*4+k_block_buffer_load_global_offset);
            int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 4;
            // printf("WARP_ID is %d, seqlen_k_stride is %d, max_seq_k_offset is %d, lds_offset is %d\n", WARP_ID, seqlen_k_stride, max_seq_k_offset, lds_offset*4);
            inline_buffer_load_dword_lds(k_lds, gK, lds_offset, s_offset, v_offset);
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}