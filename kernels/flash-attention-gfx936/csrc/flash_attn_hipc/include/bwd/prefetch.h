#pragma once
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "utils.h"
#include "static_switch.h"
#include "numeric_types.h"
#include "intrinsic_mls_ds.h"
template<int K, int BLOCK_M, int BLOCK_K, int WARP_M,  typename Element, typename ElementAccum, bool Is_even_MN>
inline __device__ void  prefetch_to_vgpr(
        vec4_uint k_ptr,
        Element* k_lds,
        union_vec2_f16x2<Element> k_reg[(K/BLOCK_K)*((WARP_M*BLOCK_K)/(32*32))*2][2],
        int max_seq_k_offset,
        int row_stride) {
    const int WARP_NUM = (BLOCK_M)/(WARP_M);
    const int k_lds_load_num = (BLOCK_M * BLOCK_K) / (4*32);
    const int K_LOAD_REQUESTS = k_lds_load_num / WARP_NUM;

    int warp_id =0;
    int warp_id_vec = threadIdx.x / 64; //warp id in a block

    warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);


    int k_warp_m_id         = (warp_id & ((BLOCK_M/WARP_M) - 1));
    int lane_id             = threadIdx.x & 63; //lane id, 0-63
    int k_lane_m_idx        = ((lane_id >> 4) & 1)*2 + ((lane_id >> 4) >> 1); //(0, 1, 2, 3) --> (0, 2, 1, 3)
    int k_lane_head_dim_idx = lane_id & 15;

    // int lds_offset = row * 8 + col * 32;
    int stage_id = 0;
    
    // MLS
    vec4_uint k_srsrc;
    k_srsrc[2] = row_stride;  // stride
    k_srsrc[3] = 0;

    #pragma unroll
    for(int k_loop = 0; k_loop<K/BLOCK_K; k_loop++) {
        {
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_waitcnt(0);
            __syncthreads();
            __builtin_amdgcn_sched_barrier(0);
            //global->lds, left matrix
            int q_block_buffer_load_global_offset = k_loop * BLOCK_K ;//+ block_id_m * BLOCK_M * K;
            // k_ptr buffer load mini size is 4*32, (BLOCK_M * BLOCK_K) mini size is (32*32)
            int k_lds_stage_offset = stage_id * (BLOCK_M/32) * (BLOCK_K/32)*(32*34);
            for(int load = 0,warp_loop = warp_id; load < K_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
                int padding = (warp_loop & 7)*2; // padding size in shared memory per buffer load, to avoid bank conflict
                int k_warp_buffer_load_m_id = (warp_loop & (BLOCK_M/4 - 1)); //这样子对L1和utlc1有啥影响呢？
                    // int q_warp_buffer_load_k_id = (warp_loop / (BLOCK_M/4));
                int q_warp_buffer_load_lds_offset     =  k_lds_stage_offset/* + (q_warp_buffer_load_k_id * BLOCK_M * 34)*/ + ((k_warp_buffer_load_m_id >> 3)*(32*34) + (k_warp_buffer_load_m_id & 7)*(4*32));
                // int q_warp_buffer_load_global_offset  =  (q_warp_buffer_load_k_id * 32);

                int gvOffset_s = (q_block_buffer_load_global_offset/* + q_warp_buffer_load_global_offset*/) / 2;
                int gvOffset_v;
                if constexpr (not Is_even_MN) {
                    gvOffset_v = ((min(k_warp_buffer_load_m_id * 4 + k_lane_m_idx, max_seq_k_offset - 1)) * row_stride) / 2 + k_lane_head_dim_idx;
                } else {
                    gvOffset_v = ((k_warp_buffer_load_m_id * 4 + k_lane_m_idx) * row_stride) / 2 + k_lane_head_dim_idx;
                }
                int lds_offset = (q_warp_buffer_load_lds_offset + padding) / 2; // +  lane_id;

                builtin_buffer_load_dword_lds_bypass_glc_slc(k_lds, k_ptr, lds_offset, gvOffset_s, gvOffset_v);
            }

            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_waitcnt(0);
            __syncthreads();
            __builtin_amdgcn_sched_barrier(0);
                    
            // k_lds_stage_offset = stage_id * (BLOCK_M/32) * (BLOCK_K/32)*(32*17);

            vec2_Element<Element> *k_lds_v2fp16 = (vec2_Element<Element> *)(k_lds);
            ds_read_tile_pad(WARP_M, BLOCK_K, WARP_NUM, Element, k_lds_v2fp16, k_lds_stage_offset, k_reg, k_loop, warp_id, lane_id);
        }
    }
}

//matrix_load单位：32 * 32
//ds_read_matrix单位：32 * 16
//M = 128, N = 128
template<bool trans, int M, int N,  typename Element, typename ElementAccum, bool Is_even_MN>
inline __device__ void  prefetch_to_vgpr_gfx938(
        vec4_uint ptr,
        Element* lds,
        union_vec4_f16x2<Element> reg[M * N / (64 * 8)],//vec4_fp16x2有8个element，64个线程
        int max_column_offset,
        int warp_id) {
    constexpr int ELEMENT_BYTES   = sizeof(Element);
    const int stages = 2;
    const int WARP_NUM = 4;
    int row_stride = ptr[2];
    vec4_uint srsrc;
    srsrc[2] = row_stride;
    srsrc[3] = 0;

    //计算LDS地址，每个warp使用一个32*32
    int lds_offset = (warp_id * 32 * 32);
    size_t lds_load_offset = reinterpret_cast<size_t>(lds) + lds_offset * ELEMENT_BYTES;
    
    int stages_id = 0;
    if(stages == 2) {
        int m_loop = 0;
        int n_loop = 0;
        int global_offset = (warp_id * row_stride * 32 + n_loop * 32);
        int lds_offset_stage = (lds_offset + stages_id * (WARP_NUM * 32 * 32)) * ELEMENT_BYTES;
        if constexpr (!Is_even_MN) {
            //对M方向进行边界判断，看需要pad多少0
            int nm_filter_max = (m_loop * 128 + (warp_id + 1) * 32) - max_column_offset;
            int nm_filter = max(0, (m_loop * 128 + (warp_id + 1) * 32) - max_column_offset);
            if(nm_filter_max >= 32) {
                global_offset = (0 * row_stride * 32 + n_loop * 32);
                nm_filter = max(0, (m_loop * 128 + 0 * 32) - max_column_offset);
            }
            srsrc[3] = nm_filter << 8; // set only once
        }
        *(uint64_t*)&srsrc = VA_LIMIT_BITS(*(uint64_t*)&ptr + global_offset * ELEMENT_BYTES);
        if(trans) {
            inline_matrix_load_32x32_b16_lds_trans<0, 0>(lds, srsrc, lds_offset_stage, 0);
        } else {
            inline_matrix_load_32x32_b16_lds<0, 0>(lds, srsrc, lds_offset_stage, 0);
        }
    }
    for(int m_loop = 0; m_loop < M / 128; ++m_loop) {
        for(int n_loop = stages - 1; n_loop < N / 32 + stages - 1; ++n_loop) {
            if(stages == 2) {
                stages_id ^= 1;
            }
            //更新global地址
            int global_offset = (warp_id * row_stride * 32 + n_loop * 32);
            int lds_offset_stage = (lds_offset + stages_id * (WARP_NUM * 32 * 32)) * ELEMENT_BYTES;
            // size_t lds_load_offset_stage = reinterpret_cast<size_t>(lds) + (stages == 2 ? (stages_id ^ 1) : stages_id) * (WARP_NUM * 32 * 32) * ELEMENT_BYTES + lds_offset * ELEMENT_BYTES;
            if constexpr (!Is_even_MN) {
                //对M方向进行边界判断，看需要pad多少0
                int nm_filter_max = (m_loop * 128 + (warp_id + 1) * 32) - max_column_offset;
                int nm_filter = max(0, (m_loop * 128 + (warp_id + 1) * 32) - max_column_offset);
                if(nm_filter_max >= 32) {
                    global_offset = (0 * row_stride * 32 + n_loop * 32);
                    nm_filter = max(0, (m_loop * 128 + 0 * 32) - max_column_offset);
                }
                srsrc[3] = nm_filter << 8; // set only once
            }
            *(uint64_t*)&srsrc = VA_LIMIT_BITS(*(uint64_t*)&ptr + global_offset * ELEMENT_BYTES);
            if(n_loop < N / 32) {
                if(trans) {
                    inline_matrix_load_32x32_b16_lds_trans<0, 0>(lds, srsrc, lds_offset_stage, 0);
                } else {
                    inline_matrix_load_32x32_b16_lds<0, 0>(lds, srsrc, lds_offset_stage, 0);
                }
            }
        
            if(stages == 2 && n_loop < N /32) {
                vmcnt_wait_nosync(1);
            } else {
                vmcnt_wait_nosync(0);
            }
            // __builtin_amdgcn_s_waitcnt(0);
            // __syncthreads();
            if(trans){
                // DS_READ_MATRIX_32X32_B16(ds_offset_cast(lds_load_offset_stage), reg[(stages == 2 ? (n_loop - 1) : n_loop) * 2].f16, reg[(stages == 2 ? (n_loop - 1) : n_loop) * 2 + 1].f16, true);
                if constexpr (std::is_same_v<Element, half_t>) {
                    reg[(stages == 2 ? (n_loop - 1) : n_loop) * 2].f16x8 =  __builtin_hcu_ds_read_matrix_trans_format_f16(lds + (stages == 2 ? (stages_id ^ 1) : stages_id) * (WARP_NUM * 32 * 32) + lds_offset, 0, 2, 1, 0);
                    reg[(stages == 2 ? (n_loop - 1) : n_loop) * 2 + 1].f16x8 =  __builtin_hcu_ds_read_matrix_trans_format_f16(lds + (stages == 2 ? (stages_id ^ 1) : stages_id) * (WARP_NUM * 32 * 32) + lds_offset, 1024, 2, 1, 0);
                } else {
                    reg[(stages == 2 ? (n_loop - 1) : n_loop) * 2].f16x8 =  __builtin_hcu_ds_read_matrix_trans_format_bf16(lds + (stages == 2 ? (stages_id ^ 1) : stages_id) * (WARP_NUM * 32 * 32) + lds_offset, 0, 2, 1, 0);
                    reg[(stages == 2 ? (n_loop - 1) : n_loop) * 2 + 1].f16x8 =  __builtin_hcu_ds_read_matrix_trans_format_bf16(lds + (stages == 2 ? (stages_id ^ 1) : stages_id) * (WARP_NUM * 32 * 32) + lds_offset, 1024, 2, 1, 0);
                }
            } else {
                // DS_READ_MATRIX_32X32_B16(ds_offset_cast(lds_load_offset_stage), reg[(stages == 2 ? (n_loop - 1) : n_loop) * 2].f16, reg[(stages == 2 ? (n_loop - 1) : n_loop) * 2 + 1].f16, false);
                if constexpr (std::is_same_v<Element, half_t>) {
                    reg[(stages == 2 ? (n_loop - 1) : n_loop) * 2].f16x8 =  __builtin_hcu_ds_read_matrix_format_f16(lds + (stages == 2 ? (stages_id ^ 1) : stages_id) * (WARP_NUM * 32 * 32) + lds_offset, 0, 2, 1, 0);
                    reg[(stages == 2 ? (n_loop - 1) : n_loop) * 2 + 1].f16x8 =  __builtin_hcu_ds_read_matrix_format_f16(lds + (stages == 2 ? (stages_id ^ 1) : stages_id) * (WARP_NUM * 32 * 32) + lds_offset, 1024, 2, 1, 0);
                } else {
                    reg[(stages == 2 ? (n_loop - 1) : n_loop) * 2].f16x8 =  __builtin_hcu_ds_read_matrix_format_bf16(lds + (stages == 2 ? (stages_id ^ 1) : stages_id) * (WARP_NUM * 32 * 32) + lds_offset, 0, 2, 1, 0);
                    reg[(stages == 2 ? (n_loop - 1) : n_loop) * 2 + 1].f16x8 =  __builtin_hcu_ds_read_matrix_format_bf16(lds + (stages == 2 ? (stages_id ^ 1) : stages_id) * (WARP_NUM * 32 * 32) + lds_offset, 1024, 2, 1, 0);
                }
            }
            lgkmcnt_wait(0);
            // __builtin_amdgcn_s_waitcnt(0);
            // __syncthreads();
        }
    }
}

//matrix_load单位：32 * 32
//ds_read_matrix单位：32 * 16
//M = 32, N = 128
template<bool trans, int M, int N,  typename Element, typename ElementAccum, bool Is_even_MN, int WARP_NUM = 4>
inline __device__ void  prefetch_to_lds_gfx938(
        vec4_uint ptr,
        int global_start_offset,
        Element* lds,
        int max_column_offset,
        int warp_id) {
    const int ELEMENT_BYTES   = sizeof(Element);
    const int LOAD_NUM = M * N / (32 * 32);
    int row_stride = ptr[2];
    vec4_uint srsrc;
    srsrc[2] = row_stride;
    srsrc[3] = 0;
    // __builtin_amdgcn_s_waitcnt(0);
    // __syncthreads();
    //直接拉通M * N，看有多少个 32*32 的矩阵需要load
    for(int loop = 0; loop < (LOAD_NUM + WARP_NUM - 1) / WARP_NUM; loop++) {
        int loop_warp = loop * WARP_NUM + warp_id;
        if (loop_warp < LOAD_NUM) {
            int m_loop = loop_warp / (N / 32);
            int n_loop = loop_warp % (N / 32);
            //更新global地址
            int global_offset = (global_start_offset + m_loop * row_stride + n_loop * 32) * ELEMENT_BYTES;
            if constexpr (!Is_even_MN) {
                //对M方向进行边界判断，看需要pad多少0
                int nm_filter_max = (m_loop + 1) * 32 - max_column_offset;
                int nm_filter = nm_filter_max;
                if(nm_filter_max >= 32) {
                    global_offset = (global_start_offset + 0 * row_stride + n_loop * 32) * ELEMENT_BYTES;
                    nm_filter = (0 + 1) * 32 - max_column_offset;
                }
                nm_filter = max(0, nm_filter);
                srsrc[3] = nm_filter << 8; // set only once
            }
            *(uint64_t*)&srsrc = VA_LIMIT_BITS(*(uint64_t*)&ptr + global_offset);
            //计算LDS地址，每个warp使用一个32*32；下一个loop重复利用
            int lds_offset = (loop_warp * 32 * 32) * ELEMENT_BYTES;
            int lds_load_offset = reinterpret_cast<size_t>(lds) + lds_offset;
            if (trans) {
                inline_matrix_load_32x32_b16_lds_trans<0, 0>(lds, srsrc, lds_offset, 0);
            } else {
                inline_matrix_load_32x32_b16_lds<0, 0>(lds, srsrc, lds_offset, 0);
            }
        }
    }
    // __builtin_amdgcn_s_waitcnt(0);
    // __syncthreads();
}

template<bool Is_even_MN, int K/*head_dim*/, int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, typename Element>
__forceinline__ __device__ void  prefetch_to_tmp_lds_wait(vec4_uint B_ptr, Element* B_lds, int max_n_len_offset, int warp_id, int row_stride)
{
    const int WARP_NUM = BLOCK_M/WARP_M;
    int lane_id = threadIdx.x & 63; //lane id, 0-63
    for(int n_loop = 0 ; n_loop < BLOCK_N/WARP_N; n_loop++){
        for(int k_loop = 0; k_loop < K/BLOCK_K; k_loop++) {
            const int lgkmcnt = (BLOCK_N/WARP_N * K/BLOCK_K - 1) - (n_loop * K/BLOCK_K + k_loop);
            lgkmcnt_wait(lgkmcnt);
            int B_block_buffer_load_global_offset = k_loop * BLOCK_K + n_loop * WARP_N * K;
            // headdim=256时的LDS用量为 256/32 * 32 * 34 * 2byte= 17 KB，如果同时读Q和dO到LDS，就会超过32KB
            // headdim=224时的LDS用量为 224/32 * 32 * 34 * 2byte= 14.875 KB，如果同时读Q和dO到LDS，不会超32KB
            int B_lds_stage_offset = k_loop * (WARP_N/32) * (BLOCK_K/32)*(32*34) + n_loop * (K/32) * (WARP_N/32)*(32*34);
            buffer_load_lds_tile_pad(Is_even_MN, WARP_NUM, row_stride, WARP_N, BLOCK_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset, max_n_len_offset, warp_id, lane_id);
        }
    }
}