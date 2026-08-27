#pragma once
#include "int8_kvcache_qk_gemm_prefetch_v_3stage.h"

#define USE_DS_OVERLAP_MMAC


/*
 * gQ: Transposed 32x16 matrix
 * gK: Non-transposed 32x16 matrix
 * s_ptr: Non-transposed 32x32 matrix
 */
template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int kBlockK_v, int WARP_M, int WARP_N, int WARP_NUM, int STAGES, int M_MMAC_COUNT, typename Element, typename Element_k, typename ElementAccum>
__forceinline__ __device__ void  int8_kvcache_qk_gemm_prefetch_v(
        vec4_uint gQ,
        vec4_uint gK,
        vec4_uint gV,
        Element_k* q_lds,
        Element_k* k_lds,
        Element_k* v_lds,
        vec4_int8 q_reg[(kHeadDim/kBlockK)*((WARP_M*kBlockK)/(32*kBlockK))*2][4],
        vec4_int32 s_reg[(WARP_M/32)*(WARP_N/32)][4],
        int WARP_ID,
        int kcache_seqlen_stride,
        int vcache_seqlen_stride,
        int max_seq_k_offset = -1) {

    // static_assert(kBlockK == 32 and "To simplify, only kBlockK = 32 is supported! otherwise, restore q_warp_buffer_load_k_id and so on");

    // union_vec4_f16x2<Element> k_reg[STAGES*((WARP_N*kBlockK)/(32*32))*2];
    vec4_int8 k_reg[STAGES*((WARP_N*kBlockK)/(32*32))*4][4];

    int lane_id = threadIdx.x & 63; // lane id, 0-63

    int laneid_shfl_4 = lane_id >> 4;
    int laneid_and_15 = lane_id & 15;

    int qk_lane_m_idx = (laneid_shfl_4 & 1)*2 + (laneid_shfl_4 >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int qk_lane_head_dim_idx = laneid_and_15;

    int k_warp_n_id = (WARP_ID & (WARP_N/WARP_N - 1));

    int k_ds_read_offset = k_warp_n_id*(WARP_N/32)*(32*17) + (lane_id & 1)*16 + (laneid_and_15>>1)*65 + (laneid_shfl_4 & 1)*8 + (lane_id/32);

    constexpr int k_lds_load_num = (WARP_N*kBlockK) / (4*32);
    constexpr int K_LOAD_REQUESTS = k_lds_load_num;

    int stage_id = 0;

    constexpr int K_LOOP_START = (STAGES == 2) ? 1: 0;
    if constexpr (STAGES == 2) stage_id ^= 1;
    //to_be_modified
    for(int k_loop = K_LOOP_START; k_loop<(kHeadDim/kBlockK); k_loop++) {

        if constexpr (true) {
            // int k_lane_seq_idx = (laneid_shfl_4);
            // neighbour sequence is in the same thread --->(seq0, seq1) in thread0, (seq2, seq3) in thread1...
            // int k_lane_seq_idx = ((laneid_shfl_4) & 1)*2 + ((laneid_shfl_4) >> 1); //(0, 1, 2, 3) --> (0, 2, 1, 3)
            // int k_lane_head_dim_idx = laneid_and_15;
            int k_block_buffer_load_global_offset = k_loop * kBlockK + WARP_ID * WARP_N * kcache_seqlen_stride;
            int k_lds_stage_offset = WARP_ID * (WARP_N/32)* (kBlockK/32)*(32*34) + stage_id * WARP_NUM * (WARP_N/32) * (kBlockK/32)*(32*34);
            for(int load = 0; load < K_LOAD_REQUESTS; ++load) {
                int __load = (load + WARP_ID) % K_LOAD_REQUESTS;
                int padding = (__load & 7) * 2; // padding size in shared memory per buffer load, to avoid bank conflict
                int k_warp_buffer_load_n_id = (__load & (WARP_N/4 - 1));
                // int k_warp_buffer_load_k_id = (warp_loop / (WARP_N/4));
                int k_warp_buffer_load_lds_offset =  k_lds_stage_offset/* + (k_warp_buffer_load_k_id * WARP_N * 34)*/ + ((k_warp_buffer_load_n_id >> 3)*(32*34) + (k_warp_buffer_load_n_id & 7)*(4*32)) ; ;
                // int k_warp_buffer_load_global_offset  =  (k_warp_buffer_load_k_id * 32);
                int gvOffset_s = (k_block_buffer_load_global_offset/* + k_warp_buffer_load_global_offset*/) / 2; 
                int gvOffset_v = ((min(k_warp_buffer_load_n_id * 4 + qk_lane_m_idx, max_seq_k_offset - 1)) * kcache_seqlen_stride) / 2 + qk_lane_head_dim_idx;
                int lds_offset = (k_warp_buffer_load_lds_offset + padding) / 2;
                inline_buffer_load_dword_lds(k_lds, gK, lds_offset, gvOffset_s, gvOffset_v);
            }
        }
        //to_be_modified
        // 在 wait 之前提前计算这部分偏移量
        if constexpr (STAGES == 2) stage_id ^=1;

        int precompute_k_lds_offset[2*2];
        int k_lds_stage_offset = WARP_ID * (WARP_N/32)* (kBlockK/32)*(32*17) + stage_id * WARP_NUM * (WARP_N/32) * (kBlockK/32)*(32*17);
        // vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
        vec4_int8 *k_lds_v4int8 =  (vec4_int8 *)(k_lds);
        for(int i=0; i<2; i++) {
            for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                for(int head_dim_idx=0; head_dim_idx<(kBlockK/32); head_dim_idx++) {
                    for(int j=0; j<2; j++) {
                        precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v4int8) + (k_lds_stage_offset + head_dim_idx*(WARP_N*17) + n_idx*(32*17) + j*4 + i*32 + k_ds_read_offset) * 4;
                    }
                }
            }
        }

        __builtin_amdgcn_sched_barrier(0);
        #ifdef USE_PINGPANG_BUFFER
            if constexpr (STAGES == 2) {
                buffer_load_lds_dwordx1_wait_nosync<K_LOAD_REQUESTS>(); // 对于当前的写法, 每个 wave 处理自己的数据, 不需要 wave 同步; 直到计算最大值
            } else if constexpr (STAGES == 1) {
                buffer_load_lds_dwordx1_wait_nosync<0>();
            }
        #else
            buffer_load_lds_dwordx1_wait_nosync<0>();
        #endif
        __builtin_amdgcn_sched_barrier(0);
        //to_be_modified
        if constexpr (true) {
            // lds -> vgpr use ds_read_m; right matrix
            // int k_warp_n_id = (WARP_ID & (WARP_N/WARP_N - 1));
            // int k_lds_stage_offset = stage_id * (WARP_N/32) * (kBlockK/32)*(32*17);
            // vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
            #pragma unroll
            for(int i=0; i<2; i++) {
                #pragma unroll
                for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                    #pragma unroll
                    for(int head_dim_idx=0; head_dim_idx<(kBlockK/32); head_dim_idx++) {
                        #pragma unroll
                        for(int j=0; j<2; j++) {
                            int lds_offset = precompute_k_lds_offset[i * 2 + j];
                            // inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + i].u64[j], 2);
                            // inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + i][j], 2);
                            inline_ds_read_b32_wait(k_lds_v4int8, lds_offset, k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + i][j]);
                        }
                    }
                }
            }
        }
        #ifdef USE_DS_OVERLAP_MMAC
            asm volatile("s_waitcnt lgkmcnt(2)");
        #else
            asm volatile("s_waitcnt lgkmcnt(0)");
        #endif
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_setprio 1");
        //to_be_modified
        {
            int min_tile_n = 0;
            #pragma unroll
            for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
                for(int head_dim_idx=0; head_dim_idx< (kBlockK/32); head_dim_idx++) {
                    #pragma unroll
                    for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                        #pragma unroll
                        for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                            #pragma unroll
                            for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                                int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                                // s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                //     vec4_Element<Element>{q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][0],
                                //                         q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][1],
                                //                         q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][0],
                                //                         q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][1]},
                                //     vec4_Element<Element>{k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k][0],
                                //                         k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k][1],
                                //                         k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k + 1][0],
                                //                         k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k + 1][1]},
                                //     s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32);
                                s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m] = __builtin_hcu_mmac_i32_16x16x32_i8(vec8_int8{q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][0],
                                                                                                                                                    q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][1],
                                                                                                                                                    q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][2],
                                                                                                                                                    q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][3],
                                                                                                                                                    q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][0],
                                                                                                                                                    q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][1],
                                                                                                                                                    q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][2],
                                                                                                                                                    q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][3]},
                                                                                                                                           vec8_int8{k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][0],
                                                                                                                                                     k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][1],
                                                                                                                                                     k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][2],
                                                                                                                                                     k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][3],
                                                                                                                                                     k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][0],
                                                                                                                                                     k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][1],
                                                                                                                                                     k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][2],
                                                                                                                                                     k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][3]},
                                                                                                                                        s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m]);
                                                                                                                                           
                            }   
                        }
                    }
                }
            }
        }
        #ifdef USE_DS_OVERLAP_MMAC
            asm volatile("s_waitcnt lgkmcnt(0)");
        #endif
        __builtin_amdgcn_sched_barrier(0);
        //to_be_modified
        {
            int min_tile_n = 1;
            #pragma unroll
            for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
                for(int head_dim_idx=0; head_dim_idx< (kBlockK/32); head_dim_idx++) {
                    #pragma unroll
                    for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                        #pragma unroll
                        for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                            #pragma unroll
                            for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                                int k_loop_idx = (STAGES == 2) ? k_loop - 1: k_loop;
                                // s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                //     vec4_Element<Element>{q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][0],
                                //                         q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][1],
                                //                         q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][0],
                                //                         q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][1]},
                                //     vec4_Element<Element>{k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k][0],
                                //                         k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k][1],
                                //                         k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k + 1][0],
                                //                         k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k + 1][1]},
                                //     s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32);
                                s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m] = __builtin_hcu_mmac_i32_16x16x32_i8(vec8_int8{q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][0],
                                                                                                                                                    q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][1],
                                                                                                                                                    q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][2],
                                                                                                                                                    q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][3],
                                                                                                                                                    q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][0],
                                                                                                                                                    q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][1],
                                                                                                                                                    q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][2],
                                                                                                                                                    q_reg[(k_loop_idx)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][3]},
                                                                                                                                           vec8_int8{k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][0],
                                                                                                                                                     k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][1],
                                                                                                                                                     k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][2],
                                                                                                                                                     k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][3],
                                                                                                                                                     k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][0],
                                                                                                                                                     k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][1],
                                                                                                                                                     k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][2],
                                                                                                                                                     k_reg[stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][3]},
                                                                                                                                        s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m]);
                            }
                        }
                    }
                }
            }
        }
        asm volatile("s_setprio 0");
        // asm volatile("s_barrier ; sync before load in the coming round");
    }

    // 保留第 1 阶段最后一波数据实际的 stage_id
    int last_stage_id = stage_id ^ 1;
    //to_be_modified
    // 等待第 1 阶段最后一波数据返回做计算
    if constexpr (STAGES == 2) {
        // stage_id ^= 1;
        // 在 wait 之前提前计算好 lds load 的下标
        int precompute_k_lds_offset[2 * 2];
        if constexpr (true) {
            // vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
            vec4_int8 *k_lds_v4int8 =  (vec4_int8 *)(k_lds);
            int k_lds_stage_offset = WARP_ID * (WARP_N/32)* (kBlockK/32)*(32*17) + last_stage_id * WARP_NUM * (WARP_N/32) * (kBlockK/32)*(32*17);
            for(int head_dim_idx=0; head_dim_idx<(kBlockK/32); head_dim_idx++) {
                for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                    for(int i=0; i<2; i++) {
                        for(int j=0; j<2; j++) {
                            precompute_k_lds_offset[i * 2 + j] = reinterpret_cast<size_t>(k_lds_v4int8) + (k_lds_stage_offset + head_dim_idx*(WARP_N*17) + n_idx*(32*17) + j*4 + i*32 + k_ds_read_offset) * 4;
                        }
                    }
                }
            }
        }
        // 等待最后一波数据的返回
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (kBlockN >= (WARP_N * 2)) {
            buffer_load_lds_dwordx1_wait_nosync<K_LOAD_REQUESTS>();
        } else {
            buffer_load_lds_dwordx1_wait_nosync<0>();
        }
        __builtin_amdgcn_sched_barrier(0);
        //to_be_modified
        if constexpr (true) {
            // lds -> vgpr use ds_read_m; right matrix
            // int k_warp_n_id = (WARP_ID & (WARP_N/WARP_N - 1));
            // int k_lds_stage_offset = stage_id * (WARP_N/32) * (kBlockK/32)*(32*17);
            // vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
            vec4_int8 *k_lds_v4int8 =  (vec4_int8 *)(k_lds);
            #pragma unroll
            for(int head_dim_idx=0; head_dim_idx<(kBlockK/32); head_dim_idx++) {
                #pragma unroll
                for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                    for(int i=0; i<2; i++) {
                        for(int j=0; j<2; j++) {
                            int lds_offset = precompute_k_lds_offset[i * 2 + j];
                            // inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + i][j], 2);
                            inline_ds_read_b32_wait(k_lds_v4int8, lds_offset, k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + i][j]);
                        }
                    }
                }
            }
        }

        #ifdef USE_DS_OVERLAP_MMAC
            asm volatile("s_waitcnt lgkmcnt(2)");
        #else
            asm volatile("s_waitcnt lgkmcnt(0)");
        #endif
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_setprio 1");
        //to_be_modified
        {
            int min_tile_n = 0;
            #pragma unroll
            for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
                for(int head_dim_idx=0; head_dim_idx< (kBlockK/32); head_dim_idx++) {
                    #pragma unroll
                    for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                        #pragma unroll
                        for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                            #pragma unroll
                            for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                                // s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                //     vec4_Element<Element>{q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][0],
                                //                         q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][1],
                                //                         q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][0],
                                //                         q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][1]},
                                //     vec4_Element<Element>{k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k][0],
                                //                         k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k][1],
                                //                         k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k + 1][0],
                                //                         k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k + 1][1]},
                                //     s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32);
                                s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m] = __builtin_hcu_mmac_i32_16x16x32_i8(vec8_int8{q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][0],
                                                                                                                    q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][1],
                                                                                                                    q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][2],
                                                                                                                    q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][3],
                                                                                                                    q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][0],
                                                                                                                    q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][1],
                                                                                                                    q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][2],
                                                                                                                    q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][3]},
                                                                                                            vec8_int8{k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][0],
                                                                                                                        k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][1],
                                                                                                                        k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][2],
                                                                                                                        k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][3],
                                                                                                                        k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][0],
                                                                                                                        k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][1],
                                                                                                                        k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][2],
                                                                                                                        k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][3]},
                                                                                                        s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m]);

                            }
                        }
                    }
                }
            }
        }
        #ifdef USE_DS_OVERLAP_MMAC
            asm volatile("s_waitcnt lgkmcnt(0)");
        #endif
        __builtin_amdgcn_sched_barrier(0);
        //to_be_modified
        {
            int min_tile_n = 1;
            #pragma unroll
            for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
                for(int head_dim_idx=0; head_dim_idx< (kBlockK/32); head_dim_idx++) {
                    #pragma unroll
                    for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                        #pragma unroll
                        for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                            #pragma unroll
                            for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                                // s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                //     vec4_Element<Element>{q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][0],
                                //                         q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][1],
                                //                         q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][0],
                                //                         q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][1]},
                                //     vec4_Element<Element>{k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k][0],
                                //                         k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k][1],
                                //                         k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k + 1][0],
                                //                         k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n].f16x2[2*min_tile_k + 1][1]},
                                //     s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32);
                                s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m] = __builtin_hcu_mmac_i32_16x16x32_i8(vec8_int8{q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][0],
                                                                                                                    q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][1],
                                                                                                                    q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][2],
                                                                                                                    q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k][3],
                                                                                                                    q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][0],
                                                                                                                    q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][1],
                                                                                                                    q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][2],
                                                                                                                    q_reg[((kHeadDim/kBlockK)-1)*((WARP_M*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][2*min_tile_k + 1][3]},
                                                                                                            vec8_int8{k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][0],
                                                                                                                        k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][1],
                                                                                                                        k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][2],
                                                                                                                        k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k][3],
                                                                                                                        k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][0],
                                                                                                                        k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][1],
                                                                                                                        k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][2],
                                                                                                                        k_reg[last_stage_id*((WARP_N*kBlockK)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + min_tile_n][2*min_tile_k + 1][3]},
                                                                                                        s_reg[(n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m]);

                            }
                        }
                    }
                }
            }
        }
        asm volatile("s_setprio 0");
        // asm volatile("s_barrier ; sync before load in the coming round");
    }

    // need to reduce results on scores_max and prefetch V, and thus sync
    __syncthreads();

    // qk gemm 等待最后一次计算需要的数据之前, 可以先把需要的 V load 指令发下去;
    if constexpr (STAGES > 1) {
        int8_kvcache_prefetch_v_to_lds<kHeadDimV, kBlockM, kBlockK_v, kBlockN, WARP_M, kBlockK_v, 32/*WARP_K*/, 0, WARP_NUM, Element_k, STAGES>(gV, v_lds, WARP_ID, vcache_seqlen_stride, max_seq_k_offset);
    }

} // qk_gemm

