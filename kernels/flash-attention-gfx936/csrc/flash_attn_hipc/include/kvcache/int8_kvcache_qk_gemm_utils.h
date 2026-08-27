#pragma once
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "static_switch.h"

template<int kHeadDim, int kBlockM, int kBlockK, int WARP_M, int WARP_NUM, typename Element, typename Element_q, int STAGES, int REUSE_KV_TIMES, int M_MMAC_COUNT>
__forceinline__ __device__ void int8_kvcache_prefetch_q_to_vgpr(
        vec4_uint gQ,
        Element_q* q_lds,
        vec4_int8 q_reg[(kHeadDim/kBlockK)*((WARP_M*kBlockK)/(32*kBlockK))*2][4],
        int WARP_ID,
        int max_seq_q_offset = -1) {
    //计算常量，每个元素的字节数
    const int bytes_per_Element = 1;
    //计算常量，每个dword中包含的元素个数
    const int Element_per_dword = 4/bytes_per_Element;

    // const int WARP_NUM = (kBlockM) / (WARP_M);
    // const int q_lds_load_num = (kBlockM * kBlockK) / (4 * 32); // 32 * 32 / 4 * 32 = 8
    // const int Q_LOAD_REQUESTS = q_lds_load_num / WARP_NUM; // 8 / 4 = 2
    // static_assert(REUSE_KV_TIMES <= 16 and WARP_NUM == 4);
    constexpr bool Is_GQA         = M_MMAC_COUNT > 1;
    constexpr int Q_LOAD_REQUESTS = Is_GQA ? ((REUSE_KV_TIMES + 1) >> 1) << 2 / WARP_NUM: 1/*MHA only need the first token*/;
    constexpr int SEQUENCE_READ   = Is_GQA ? 2: 1;

    int lane_id             = threadIdx.x & 63; // lane id, 0-63
    int q_lane_m_idx        = ((lane_id >> 4) & 1) * 2 + ((lane_id >> 4) >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int q_lane_head_dim_idx = lane_id & 15;

    int stage_id = 0;
    //to_be_modified
    if constexpr (STAGES > 1) {
        int k_loop = 0;
        // global->lds, left matrix
        int q_block_buffer_load_global_offset = k_loop * kBlockK;
        int q_lds_stage_offset = stage_id * (kBlockM/32) * (kBlockK/32)*(32*34);

        for(int load = 0, warp_loop = WARP_ID; load < Q_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int padding = (warp_loop & 7)*4; // padding size in shared memory per buffer load, to avoid bank conflict
            int q_warp_buffer_load_m_id = (warp_loop & (kBlockM/4 - 1));
            int q_warp_buffer_load_lds_offset     =  q_lds_stage_offset/* + (q_warp_buffer_load_k_id * kBlockM * 34)*/ + ((q_warp_buffer_load_m_id >> 3)*(32*68) + (q_warp_buffer_load_m_id & 7)*(4*64));
            int gvOffset_s = (q_block_buffer_load_global_offset/* + q_warp_buffer_load_global_offset*/) / Element_per_dword;
            int gvOffset_v = q_warp_buffer_load_m_id * 4 + q_lane_m_idx;
            if (gvOffset_v < max_seq_q_offset) {
                int lds_offset = (q_warp_buffer_load_lds_offset + padding) / Element_per_dword;
                // int lds_offset = (q_lds_stage_offset) / Element_per_dword + padding + (q_warp_buffer_load_m_id & 7)*64;
                gvOffset_v = (gvOffset_v * kHeadDim) / Element_per_dword + q_lane_head_dim_idx;
                builtin_buffer_load_dword_lds(q_lds, gQ, lds_offset, gvOffset_s, gvOffset_v);
            }
        }
    }
    if constexpr (STAGES > 1) stage_id ^= 1;
    constexpr int K_LOOP_START = (STAGES > 1) ? 1: 0;
    //to_be_modified
    for(int k_loop = K_LOOP_START; k_loop<(kHeadDim/kBlockK); k_loop++) {
        // global->lds, left matrix
        int q_block_buffer_load_global_offset = k_loop * kBlockK;
        int q_lds_stage_offset = stage_id * (kBlockM/32) * (kBlockK/32)*(32*34);
        for(int load = 0, warp_loop = WARP_ID; load < Q_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int padding = (warp_loop & 7)*4; // padding size in shared memory per buffer load, to avoid bank conflict
            int q_warp_buffer_load_m_id = (warp_loop & (kBlockM/4 - 1));
            int q_warp_buffer_load_lds_offset     =  q_lds_stage_offset/* + (q_warp_buffer_load_k_id * kBlockM * 34)*/ + ((q_warp_buffer_load_m_id >> 3)*(32*68) + (q_warp_buffer_load_m_id & 7)*(4*64));
            int gvOffset_s = (q_block_buffer_load_global_offset/* + q_warp_buffer_load_global_offset*/) / Element_per_dword;
            int gvOffset_v = q_warp_buffer_load_m_id * 4 + q_lane_m_idx;
            if (gvOffset_v < max_seq_q_offset) {
                // int lds_offset = (q_lds_stage_offset) / Element_per_dword + padding + (q_warp_buffer_load_m_id & 7)*64;
                int lds_offset = (q_warp_buffer_load_lds_offset + padding) / Element_per_dword;
                gvOffset_v = (gvOffset_v * kHeadDim) / Element_per_dword + q_lane_head_dim_idx;
                builtin_buffer_load_dword_lds(q_lds, gQ, lds_offset, gvOffset_s, gvOffset_v);
            }
        }
        //to_be_modified
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        if constexpr (STAGES > 1) stage_id ^= 1;
        q_lds_stage_offset = stage_id * (kBlockM/32) * (kBlockK*bytes_per_Element/2/32)*(32*17);
        // vec2_Element<Element> *q_lds_v2fp16 = (vec2_Element<Element> *)(q_lds);
        vec4_int8 *q_lds_v4int8 =  (vec4_int8 *)(q_lds);
        #pragma unroll
        for(int head_dim_idx=0; head_dim_idx<(kBlockK*bytes_per_Element/2/32); head_dim_idx++) {
            #pragma unroll
            for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
                #pragma unroll
                for(int i=0; i<SEQUENCE_READ; i++) {
                    #pragma unroll
                    for(int j=0; j<4; j++) {
                        int lds_offset = q_lds_stage_offset + head_dim_idx*kBlockM*17 + j*2 + i*32 + (lane_id & 1)*16 + ((lane_id & 15)>>1)*64 + /*padding*/ ((lane_id & 15)>>1) + ((lane_id/16) &1)*8 + (lane_id/32);
                        int k_loop_idx = (STAGES > 1) ? k_loop - 1: k_loop;
                        // for mha, only 0/32/16/48 need read, and thus if (lane_id % 16 == 0), but (land_id & 15 == 0) will lead to errors
                        inline_ds_read_b32_wait(q_lds_v4int8, lds_offset, q_reg[k_loop_idx*((WARP_M*kBlockK)/(32*64))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + i][j]);
                    }
                }
            }
        }
        __syncthreads();
    }
    //to_be_modified
    if constexpr (STAGES > 1) {
        __builtin_amdgcn_s_waitcnt(0);
        stage_id ^= 1;
        int q_lds_stage_offset = stage_id * (kBlockM/32) * (kBlockK*bytes_per_Element/2/32)*(32*17);
        // vec2_Element<Element> *q_lds_v2fp16 = (vec2_Element<Element> *)(q_lds);
        vec4_int8 *q_lds_v4int8 =  (vec4_int8 *)(q_lds);
        #pragma unroll
        for(int head_dim_idx=0; head_dim_idx<(kBlockK*bytes_per_Element/2/32); head_dim_idx++) {
            #pragma unroll
            for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
                #pragma unroll
                for(int i=0; i<SEQUENCE_READ; i++) {
                    #pragma unroll
                    for(int j=0; j<4; j++) {
                        int lds_offset = q_lds_stage_offset + head_dim_idx*kBlockM*17 + j*2 + i*32 + (lane_id & 1)*16 + ((lane_id & 15)>>1)*64 + /*padding*/ ((lane_id & 15)>>1) + ((lane_id/16) &1)*8 + (lane_id/32);
                        // for mha, only 0/32/16/48 need read, and thus if (lane_id % 16 == 0), but (land_id & 15 == 0) will lead to errors             
                        inline_ds_read_b32_wait(q_lds_v4int8, lds_offset, q_reg[((kHeadDim/kBlockK) - 1)*((WARP_M*kBlockK)/(32*64))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + i][j]);
                    }
                }
            }
        }
    }
}




template<int kHeadDim, int kBlockM, int kBlockK, int WARP_M, int WARP_N, typename Element, typename Element_k, int STAGES, int WARP_NUM>
__forceinline__ __device__ void int8_kvcache_prefetch_k_to_lds(
        vec4_uint k_ptr,
        Element_k* k_lds,
        int WARP_ID,
        int kcache_seqlen_stride,
        int max_seq_k_offset = -1) {
    //计算常量，每个元素的字节数
    const int bytes_per_Element = 1;
    //计算常量，每个dword中包含的元素个数
    int Element_per_dword = 4/bytes_per_Element;
    
    // const int WARP_NUM = (kBlockM)/(WARP_M);
    int lane_id = threadIdx.x & 63; // lane id, 0-63
    int laneid_shfl_4 = lane_id >> 4;
    int laneid_and_15 = lane_id & 15;
    int qk_lane_m_idx = (laneid_shfl_4 & 1)*2 + (laneid_shfl_4 >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int qk_lane_head_dim_idx = laneid_and_15;

    constexpr int k_lds_load_num  = (WARP_N*kBlockK) / (4*64);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num;

    auto BUFFER_LOAD_FUNC = &inline_buffer_load_dword_lds<Element_k, 2>;

    int stage_id = 0;
    int k_loop   = 0;

    //to_be_modified
    if constexpr (STAGES > 1) {
        int k_block_buffer_load_global_offset = k_loop * kBlockK + WARP_ID * WARP_N * kcache_seqlen_stride;
        int k_lds_stage_offset = WARP_ID * (WARP_N/32)* (kBlockK/32)*(32*34) + stage_id * WARP_NUM * (WARP_N/32) * (kBlockK/32)*(32*34);
        for(int load = 0; load < K_LOAD_REQUESTS; ++load) {
            // int __load = (load + WARP_ID) % K_LOAD_REQUESTS;
            int padding = (load & 7); // padding size in shared memory per buffer load, to avoid bank conflict
            int k_warp_buffer_load_n_id = (load & (WARP_N/4 - 1));
            int gvOffset_s = (k_block_buffer_load_global_offset/* + k_warp_buffer_load_global_offset*/) / Element_per_dword; 
            int gvOffset_v = ((min(k_warp_buffer_load_n_id * 4 + qk_lane_m_idx, max_seq_k_offset - 1)) * kcache_seqlen_stride) / Element_per_dword + qk_lane_head_dim_idx;
            int lds_offset = (k_lds_stage_offset) / Element_per_dword + padding + ((k_warp_buffer_load_n_id >> 3)*(32*17) + (k_warp_buffer_load_n_id & 7)*64);
            BUFFER_LOAD_FUNC(k_lds, k_ptr, lds_offset, gvOffset_s, gvOffset_v);
        }
    }
}
