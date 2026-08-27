#pragma once
#include "int8_kvcache_qk_gemm_utils.h"
#include "int8_kvcache_pv_gemm_utils.h"

#define mmac() \
if (bytes_per_Element == 1){\
    s_reg[0][min_tile_n*2 + min_tile_m].int32 = __builtin_hcu_mmac_i32_16x16x32_i8(vec8_int8{q_reg[q_idx][2*min_tile_k][0],\
        q_reg[q_idx][2*min_tile_k][1],\
        q_reg[q_idx][2*min_tile_k][2],\
        q_reg[q_idx][2*min_tile_k][3],\
        q_reg[q_idx][2*min_tile_k + 1][0],\
        q_reg[q_idx][2*min_tile_k + 1][1],\
        q_reg[q_idx][2*min_tile_k + 1][2],\
        q_reg[q_idx][2*min_tile_k + 1][3]},\
    vec8_int8{k_reg[k_idx].f8x4[2*min_tile_k][0],\
         k_reg[k_idx].f8x4[2*min_tile_k][1],\
         k_reg[k_idx].f8x4[2*min_tile_k][2],\
         k_reg[k_idx].f8x4[2*min_tile_k][3],\
         k_reg[k_idx].f8x4[2*min_tile_k + 1][0],\
         k_reg[k_idx].f8x4[2*min_tile_k + 1][1],\
         k_reg[k_idx].f8x4[2*min_tile_k + 1][2],\
         k_reg[k_idx].f8x4[2*min_tile_k + 1][3]},\
    s_reg[0][min_tile_n*2 + min_tile_m].int32);\
}

/*
 * 3 阶段的 pingpang buffer, 写法跟之前的 2 阶段差异较大, 因此没统一在一起
 * 至于 4 阶段, 暂不考虑, 因为 max_lds 跟 V 不能复用, 4 个 wave、4 倍的 LDS 用量, 直接干到 32KB 了, 放不下 max_lds, 因此暂时不适用 4 阶段的 pingpang buffer
 */
template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int kBlockK_v, int WARP_M, int WARP_N, int WARP_NUM, int STAGES, int M_MMAC_COUNT, typename Element, typename Element_q, typename ElementAccum>
__forceinline__ __device__ void  int8_kvcache_qk_gemm_prefetch_v_3stage(
        vec4_uint gQ,
        vec4_uint gK,
        vec4_uint gV,
        Element_q* q_lds,
        Element_q* k_lds,
        Element_q* v_lds,
        vec4_int8 q_reg[(kHeadDim/kBlockK)*((WARP_M*kBlockK)/(32*kBlockK))*2][4],
        union_vec4_int32 s_reg[(WARP_M/32)*(WARP_N/32)][4],
        /*vec4_Accum<ElementAccum> s_reg[(WARP_M/32)*(WARP_N/32)][4],*/
        int WARP_ID,
        int kcache_seqlen_stride,
        int vcache_seqlen_stride,
        int max_seq_k_offset = -1) {

    //计算常量，每个元素的字节数
    const int bytes_per_Element = 1;
    //计算常量，每个dword中包含的元素个数
    int Element_per_dword = 4/bytes_per_Element;
    // vec4_int8 k_reg[STAGES*((WARP_N*kBlockK*bytes_per_Element/2)/(32*32))*4][4];

    union_vec4_f16x2<int8_t> k_reg[STAGES*((WARP_N*kBlockK*bytes_per_Element/2)/(32*32))*2];

    int lane_id = threadIdx.x & 63; // lane id, 0-63

    int laneid_shfl_4 = lane_id >> 4;
    int laneid_and_15 = lane_id & 15;

    int qk_lane_m_idx = (laneid_shfl_4 & 1)*2 + (laneid_shfl_4 >> 1); // (0, 1, 2, 3) --> (0, 2, 1, 3)
    int qk_lane_head_dim_idx = laneid_and_15;

    int k_warp_n_id = (WARP_ID & (WARP_N/WARP_N - 1));

    int k_ds_read_offset = k_warp_n_id*(WARP_N/32)*(32*17) + (lane_id & 1)*16 + (laneid_and_15>>1)*65 + (laneid_shfl_4 & 1)*8 + (lane_id/32);

    int k_lds_load_num = (WARP_N*kBlockK) / (4*64);
    int K_LOAD_REQUESTS = k_lds_load_num;

    auto BUFFER_LOAD_FUNC = &inline_buffer_load_dword_lds<Element_q, 2>;

    // load 指令发下去之后, 先做一些初始化运算
    #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
        if constexpr (M_MMAC_COUNT == 1) {
            inline_vgpr4_init_zero_1x2x4(s_reg);
        } else {
            inline_vgpr4_init_zero_1x4x4(s_reg);
        }
        __builtin_amdgcn_sched_barrier(0);
    #else
        uint64_t pk_zero = 0;
        #pragma unroll
        for (int i = 0; i < (WARP_N / WARP_N) * (WARP_M / 32); ++i) {
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    s_reg[i][min_tile_n * 2 + min_tile_m].u64[0] = pk_zero;
                    s_reg[i][min_tile_n * 2 + min_tile_m].u64[1] = pk_zero;
                }
            }
        }
    #endif

    int k_loop   = 0;
    int stage_id = 0;
    // for(int k_loop = K_LOOP_START; k_loop<(kHeadDim/kBlockK); k_loop++) 
    {
        stage_id = 0;
        //当使用int8，kBlockK由32变为64，所以下面的计算会发生变化，需要重新计算，to_be_modified
        // 在 wait 之前提前计算这部分偏移量

        int precompute_k_lds_offset[2*2];
        int k_lds_stage_offset = WARP_ID * (WARP_N/32)* (kBlockK*bytes_per_Element/2/32)*(32*17) + stage_id * WARP_NUM * (WARP_N/32) * (kBlockK*bytes_per_Element/2/32)*(32*17);
        // vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
        vec4_int8 *k_lds_v4int8 =  (vec4_int8 *)(k_lds);
        for(int i=0; i<2; i++) {
            for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                for(int head_dim_idx=0; head_dim_idx<(kBlockK*bytes_per_Element/2/32); head_dim_idx++) {
                    for(int j=0; j<2; j++) {
                        precompute_k_lds_offset[i * 2 + j] = (k_lds_stage_offset + head_dim_idx*(WARP_N*17) + n_idx*(32*17) + j*4 + i*32 + k_ds_read_offset);
                    }
                }
            }
        }

        vmcnt_wait(K_LOAD_REQUESTS);
        // vmcnt_wait(0);

        //to_be_modified
        if constexpr (true) {
            // lds -> vgpr use ds_read_m; right matrix
            // int k_warp_n_id = (WARP_ID & (WARP_N/WARP_N - 1));
            // int k_lds_stage_offset = stage_id * (WARP_N/32) * (kBlockK*bytes_per_Element/2/32)*(32*17);
            // vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(k_lds);
            #pragma unroll
            for(int i=0; i<2; i++) {
                #pragma unroll
                for(int head_dim_idx=0; head_dim_idx<(kBlockK*bytes_per_Element/64); head_dim_idx++) {
                    #pragma unroll
                    for(int j=0; j<2; j++) {
                        int lds_offset = precompute_k_lds_offset[i * 2 + j];
                        int reg_offset = stage_id * 2 + i;
                        // inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[stage_id*((WARP_N*kBlockK*bytes_per_Element/2)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + i].u64[j], 2);
                        k_reg[reg_offset].f8x4[j*2][0] = k_lds_v4int8[lds_offset][0];
                        k_reg[reg_offset].f8x4[j*2][1] = k_lds_v4int8[lds_offset][1];
                        k_reg[reg_offset].f8x4[j*2][2] = k_lds_v4int8[lds_offset][2];
                        k_reg[reg_offset].f8x4[j*2][3] = k_lds_v4int8[lds_offset][3];
                        k_reg[reg_offset].f8x4[j*2+1][0] = k_lds_v4int8[lds_offset+2][0];
                        k_reg[reg_offset].f8x4[j*2+1][1] = k_lds_v4int8[lds_offset+2][1];
                        k_reg[reg_offset].f8x4[j*2+1][2] = k_lds_v4int8[lds_offset+2][2];
                        k_reg[reg_offset].f8x4[j*2+1][3] = k_lds_v4int8[lds_offset+2][3];
                    }
                }
            }
        }
        // #ifdef USE_DS_OVERLAP_MMAC
        //     asm volatile("s_waitcnt lgkmcnt(2)");
        // #else
        //     asm volatile("s_waitcnt lgkmcnt(0)");
        // #endif
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_setprio 1");
        //block_k增大为64，循环体可能需要修改，to_be_modified
        {
            int min_tile_n = 0;
            for(int head_dim_idx=0; head_dim_idx< (kBlockK*bytes_per_Element/2/32); head_dim_idx++) {
                #pragma unroll
                for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                    #pragma unroll
                    for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                        int k_loop_idx = k_loop;
                        int q_idx = k_loop_idx*2 + (head_dim_idx*(WARP_M/32))*2 + min_tile_m;
                        int k_idx = stage_id*2 + head_dim_idx*(WARP_N/32)*2 + min_tile_n;
                        mmac();                                
                    }
                }
            }
        }
        // #ifdef USE_DS_OVERLAP_MMAC
        //     asm volatile("s_waitcnt lgkmcnt(0)");
        // #endif
        __builtin_amdgcn_sched_barrier(0);
        //to_be_modified
        {
            int min_tile_n = 1;
            for(int head_dim_idx=0; head_dim_idx< (kBlockK*bytes_per_Element/2/32); head_dim_idx++) {
                #pragma unroll
                for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                    #pragma unroll
                    for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                        int k_loop_idx = k_loop;
                        int q_idx = k_loop_idx*2 + (head_dim_idx*(WARP_M/32))*2 + min_tile_m;
                        int k_idx = stage_id*2 + head_dim_idx*(WARP_N/32)*2 + min_tile_n;
                        mmac();
                    }
                }
            }
        }
        asm volatile("s_setprio 0");
        // asm volatile("s_barrier ; sync before load in the coming round");
    }

    k_loop   = 1;
    //to_be_modified
    {
        {
            stage_id = 1;
            // int k_lane_seq_idx = (laneid_shfl_4);
            // neighbour sequence is in the same thread --->(seq0, seq1) in thread0, (seq2, seq3) in thread1...
            // int k_lane_seq_idx = ((laneid_shfl_4) & 1)*2 + ((laneid_shfl_4) >> 1); //(0, 1, 2, 3) --> (0, 2, 1, 3)
            // int k_lane_head_dim_idx = laneid_and_15;
            int k_block_buffer_load_global_offset = (k_loop) * kBlockK + WARP_ID * WARP_N * kcache_seqlen_stride;
            int k_lds_stage_offset = WARP_ID * (WARP_N/32)* (kBlockK/32)*(32*34) + stage_id * WARP_NUM * (WARP_N/32) * (kBlockK/32)*(32*34);
            for(int load = 0; load < K_LOAD_REQUESTS; ++load) {
                // int __load = (load + WARP_ID) % K_LOAD_REQUESTS;
                int padding = (load & 7); // padding size in shared memory per buffer load, to avoid bank conflict
                int k_warp_buffer_load_n_id = (load & (WARP_N/4 - 1));
                int gvOffset_s = (k_block_buffer_load_global_offset/* + k_warp_buffer_load_global_offset*/) / Element_per_dword; 
                int gvOffset_v = ((min(k_warp_buffer_load_n_id * 4 + qk_lane_m_idx, max_seq_k_offset - 1)) * kcache_seqlen_stride) / Element_per_dword + qk_lane_head_dim_idx;
                int lds_offset = (k_lds_stage_offset) / Element_per_dword + padding + ((k_warp_buffer_load_n_id >> 3)*(32*17) + (k_warp_buffer_load_n_id & 7)*64);
                BUFFER_LOAD_FUNC(k_lds, gK, lds_offset, gvOffset_s, gvOffset_v);
            }  
        }

        stage_id = 1;
        //to_be_modified
        // 在 wait 之前提前计算这部分偏移量

        int precompute_k_lds_offset[2*2];
        int k_lds_stage_offset = WARP_ID * (WARP_N/32)* (kBlockK*bytes_per_Element/2/32)*(32*17) + stage_id * WARP_NUM * (WARP_N/32) * (kBlockK*bytes_per_Element/2/32)*(32*17);
        vec4_int8 *k_lds_v4int8 =  (vec4_int8 *)(k_lds);
        for(int i=0; i<2; i++) {
            for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                for(int head_dim_idx=0; head_dim_idx<(kBlockK*bytes_per_Element/2/32); head_dim_idx++) {
                    for(int j=0; j<2; j++) {
                        precompute_k_lds_offset[i * 2 + j] = (k_lds_stage_offset + head_dim_idx*(WARP_N*17) + n_idx*(32*17) + j*4 + i*32 + k_ds_read_offset);
                    }
                }
            }
        }

        vmcnt_wait(0);

        //to_be_modified
        if constexpr (true) {
            // lds -> vgpr use ds_read_m; right matrix
            #pragma unroll
            for(int i=0; i<2; i++) {
                #pragma unroll
                for(int head_dim_idx=0; head_dim_idx<(kBlockK*bytes_per_Element/2/32); head_dim_idx++) {
                    #pragma unroll
                    for(int j=0; j<2; j++) {
                        int lds_offset = precompute_k_lds_offset[i * 2 + j];
                        int reg_offset = stage_id * 2 + i;
                        // inline_ds_read2_b32_no_wait_bytes(lds_offset, k_reg[stage_id*((WARP_N*kBlockK*bytes_per_Element/2)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx)*2 + i].u64[j], 2);
                        k_reg[reg_offset].f8x4[j*2][0] = k_lds_v4int8[lds_offset][0];
                        k_reg[reg_offset].f8x4[j*2][1] = k_lds_v4int8[lds_offset][1];
                        k_reg[reg_offset].f8x4[j*2][2] = k_lds_v4int8[lds_offset][2];
                        k_reg[reg_offset].f8x4[j*2][3] = k_lds_v4int8[lds_offset][3];
                        k_reg[reg_offset].f8x4[j*2+1][0] = k_lds_v4int8[lds_offset+2][0];
                        k_reg[reg_offset].f8x4[j*2+1][1] = k_lds_v4int8[lds_offset+2][1];
                        k_reg[reg_offset].f8x4[j*2+1][2] = k_lds_v4int8[lds_offset+2][2];
                        k_reg[reg_offset].f8x4[j*2+1][3] = k_lds_v4int8[lds_offset+2][3];
                    }
                }
            }
        }
        // #ifdef USE_DS_OVERLAP_MMAC
        //     asm volatile("s_waitcnt lgkmcnt(2)");
        // #else
        //     asm volatile("s_waitcnt lgkmcnt(0)");
        // #endif
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_setprio 1");
        //to_be_modified
        {
            int min_tile_n = 0;
            for(int head_dim_idx=0; head_dim_idx< (kBlockK*bytes_per_Element/2/32); head_dim_idx++) {
                #pragma unroll
                for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                    #pragma unroll
                    for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                        int k_loop_idx = k_loop;
                        int q_idx = k_loop_idx*2 + (head_dim_idx*(WARP_M/32))*2 + min_tile_m;
                        int k_idx = stage_id*2 + head_dim_idx*(WARP_N/32)*2 + min_tile_n;
                        mmac();
                    }
                }
            }
        }
        // #ifdef USE_DS_OVERLAP_MMAC
        //     asm volatile("s_waitcnt lgkmcnt(0)");
        // #endif
        __builtin_amdgcn_sched_barrier(0);
        //to_be_modified
        {
            int min_tile_n = 1;
            for(int head_dim_idx=0; head_dim_idx< (kBlockK*bytes_per_Element/2/32); head_dim_idx++) {
                #pragma unroll
                for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                    #pragma unroll
                    for(int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                        int k_loop_idx = k_loop;
                        int q_idx = k_loop_idx*2 + (head_dim_idx*(WARP_M/32))*2 + min_tile_m;
                        int k_idx = stage_id*2 + head_dim_idx*(WARP_N/32)*2 + min_tile_n;
                        mmac();
                    }
                }
            }
        }
        asm volatile("s_setprio 0");
        // asm volatile("s_barrier ; sync before load in the coming round");
    }

    // need to reduce results on scores_max and prefetch V, and thus sync
    // __syncthreads();
    // 可以先把需要的 V load 指令发下去;
    int8_kvcache_prefetch_v_to_lds<kHeadDimV, kBlockM, kBlockK_v, kBlockN, WARP_M, kBlockK_v, 32/*WARP_K*/, 0, WARP_NUM, Element_q, STAGES>(gV, v_lds, WARP_ID, vcache_seqlen_stride, max_seq_k_offset);

} // qk_gemm

