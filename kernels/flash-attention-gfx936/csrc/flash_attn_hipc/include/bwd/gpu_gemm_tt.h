#pragma once
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "intrinsic.h"
#include "numeric_types.h"
#include "intrinsic_mls_ds.h"
#include "prefetch.h"
// 无预取：prefetch_level = 0;   预取到LDS:prefetch_level = 1;    预取到寄存器:prefetch_level = 2;
template<bool Is_store_B, bool Is_preload_C, bool Is_even_MN, int A_prefetch_level, int B_prefetch_level, int K, int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum = float>                                                                                                               
__forceinline__ __device__ void  gemm_tt_kq(vec4_uint A_ptr, vec4_uint B_ptr, Element* A_lds, Element* B_lds, int max_m_len_offset, int max_n_len_offset, union_vec2_f16x2<Element> A_reg[(K/BLOCK_K)*((WARP_M*BLOCK_K)/(32*32))*2/((A_prefetch_level == 3)? 1 : 2)][2], union_vec2_f16x2<Element> B_reg[STAGES*((WARP_N*BLOCK_K)/(32*32))*2][2], union_vec4_fp32 C_reg[(WARP_M/32)*(BLOCK_N/32)][4], int warp_id, int seqlen_A_stride, int seqlen_B_stride) {
    const int WARP_NUM = BLOCK_M/WARP_M;

    //wave size should be defined in launch file. Here use 64 threads
    int lane_id = threadIdx.x & 63; //lane id, 0-63

    int row = lane_id % 4;
    int col = lane_id / 4;
#if 1
    for(int n_loop = 0 ; n_loop < BLOCK_N/WARP_N; n_loop++)
    {
    int stage_id = 0;
    int stage_id_reg = 0;
     {  int k_loop = 0;
        if(STAGES > 1) {
            if(A_prefetch_level == 0) {
                int A_block_buffer_load_global_offset = k_loop * BLOCK_K;
                int A_lds_stage_offset = stage_id * (BLOCK_M/32) * (BLOCK_K/32)*(32*34);
                buffer_load_lds_tile_pad(Is_even_MN, WARP_NUM, seqlen_A_stride, BLOCK_M, BLOCK_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset, K, warp_id, lane_id);
            }

            if(B_prefetch_level == 0) {
                int B_block_buffer_load_global_offset = k_loop * BLOCK_K + n_loop * WARP_N * K;
                int B_lds_stage_offset = stage_id * (WARP_N/32) * (BLOCK_K/32)*(32*34);
                if constexpr (Is_store_B){
                    B_lds_stage_offset += n_loop * (K/32) * (WARP_N/32)*(32*34);
                }
                buffer_load_lds_tile_pad_1(Is_even_MN, WARP_NUM, seqlen_B_stride, WARP_N, BLOCK_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset, K, warp_id, lane_id);
            }
        }
     }
    
    // int lds_offset = row * 8 + col * 32;
    for(int k_loop = 1; k_loop<(K/BLOCK_K) + 1; k_loop++) {
        if constexpr (STAGES > 1) {
            if constexpr (Is_store_B || B_prefetch_level == 1){
                stage_id++;
            } else {
                stage_id ^= 1;
            }            
        }

        if(STAGES == 1) {
            k_loop--;
        }
        if(k_loop < (K/BLOCK_K)){
            if(A_prefetch_level == 0 || (A_prefetch_level == 1 && k_loop >= (K/BLOCK_K)/2)) {
                int A_block_buffer_load_global_offset = k_loop * BLOCK_K;
                int A_lds_stage_offset = stage_id * (BLOCK_M/32) * (BLOCK_K/32)*(32*34);
                buffer_load_lds_tile_pad(Is_even_MN, WARP_NUM, seqlen_A_stride, BLOCK_M, BLOCK_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset, K, warp_id, lane_id);
            }

            if(B_prefetch_level == 0) {
                int B_block_buffer_load_global_offset = k_loop * BLOCK_K + n_loop * WARP_N * K;
                int B_lds_stage_offset = stage_id * (WARP_N/32) * (BLOCK_K/32)*(32*34);
                if constexpr (Is_store_B || B_prefetch_level == 1){
                    B_lds_stage_offset += n_loop * (K/32) * (WARP_N/32)*(32*34);
                }
                buffer_load_lds_tile_pad_1(Is_even_MN, WARP_NUM, seqlen_B_stride, WARP_N, BLOCK_K, Element, B_ptr, B_lds, B_block_buffer_load_global_offset, B_lds_stage_offset, K, warp_id, lane_id);
            }
        } 
        else if (B_prefetch_level==0){
            vmcnt_wait(0);
        }


        int precompute_B_lds_offset[2*2];
        //lds -> vgpr use ds_read_m; right matrix
        int k_warp_n_id = (warp_id & (WARP_N/WARP_N - 1));

        int k_lds_stage_offset = STAGES == 1 ? 0 : ( (Is_store_B || B_prefetch_level == 1) ? (stage_id - 1)  * (WARP_N/32) * (BLOCK_K/32)*(32*17) : (stage_id ^ 1)  * (WARP_N/32) * (BLOCK_K/32)*(32*17));
        vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(B_lds);

        //a warp load min size is (row, col) = (32,16) float
        #pragma unroll
        for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
            #pragma unroll
            for(int head_dim_idx=0; head_dim_idx<(BLOCK_K/32); head_dim_idx++) { //32 half in col direction
                #pragma unroll
                for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) { 
                    #pragma unroll
                    for(int vec_idx = 0; vec_idx < 2; vec_idx ++) {
                        int lds_offset = k_lds_stage_offset + head_dim_idx*(WARP_N*17) + (k_warp_n_id*(WARP_N/32) + n_idx)*(32*17) + vec_idx * 4 + min_tile_n*32 + (lane_id & 1)*16 + ((lane_id & 15)>>1)*64 + /*padding*/ ((lane_id & 15)>>1) + ((lane_id/16) & 1)*8 + (lane_id/32);
                        precompute_B_lds_offset[min_tile_n * 2 + vec_idx] = lds_offset;
                        if constexpr (Is_store_B || B_prefetch_level == 1){
                            precompute_B_lds_offset[min_tile_n * 2 + vec_idx] += n_loop * (WARP_N/32) * (K/32)*(32*17);
                        }
                    }
                }
            }
        }
        
        if(STAGES > 1) {
            if constexpr(B_prefetch_level==1){
                if constexpr (std::is_same<Element,Float8_e4m3_t>::value){
                    vmcnt_wait(0);
                } else {
                    vmcnt_wait(((BLOCK_N/WARP_N * K/BLOCK_K)*(Is_preload_C ? 2 : 1) - (n_loop * (K/BLOCK_K) + k_loop)) * (WARP_N*BLOCK_K) / (4*32)/WARP_NUM);
                }
            } else {
                if(k_loop < (K/BLOCK_K)){
                    if(A_prefetch_level == 0 && B_prefetch_level != 0) {
                        buffer_load_lds_dwordx1_wait<(BLOCK_M * BLOCK_K) / (4*32)/WARP_NUM>();
                    } else if(A_prefetch_level != 0 && B_prefetch_level == 0) {
                        buffer_load_lds_dwordx1_wait<(WARP_N*BLOCK_K) / (4*32)/WARP_NUM>();
                    } else if(A_prefetch_level == 0 && B_prefetch_level == 0) {
                        buffer_load_lds_dwordx1_wait<(BLOCK_M * BLOCK_K) / (4*32)/WARP_NUM + (WARP_N*BLOCK_K) / (4*32)/WARP_NUM>();
                    } 
                }
            }
        } else {
            vmcnt_wait(0);
        }



        if constexpr (STAGES > 1) {
            if constexpr (Is_store_B || B_prefetch_level == 1){
                stage_id--;
            } else {
                stage_id ^= 1;
            }   
        }

        union_vec2_f16x2<Element> A_reg_tmp[2][2];
        if (A_prefetch_level == 0 || (A_prefetch_level != 3 && k_loop >= (K/BLOCK_K)/2 )) {
            //lds -> vgpr use ds_read_m; left matrix
            int A_warp_m_id = (warp_id & ((BLOCK_M/WARP_M) - 1));
            
            int A_lds_stage_offset = stage_id  * (BLOCK_M/32) * (BLOCK_K/32)*(32*17);

            vec2_Element<Element> *A_lds_v2fp16 = (vec2_Element<Element> *)(A_lds);
            asm volatile("s_setprio 1");
            // #pragma unroll
            // for(int head_dim_idx=0; head_dim_idx<(BLOCK_K/32); head_dim_idx++) { //32 half in col direction
            //     #pragma unroll
            //     for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
            //         //a warp load min size is (row, col) = (32,16) float
            //         #pragma unroll
            //         for(int i=0; i<2; i++) {       //sequence direction
            //             #pragma unroll
            //             for(int j=0; j<2; j++) {   //head dim direction
            //                 int lds_offset = A_lds_stage_offset + head_dim_idx*BLOCK_M*17 + (warp_id*(WARP_M/32) + m_idx)*(32*17) + j*4 + i*32 + (lane_id & 1)*16 + ((lane_id & 15)>>1)*64 + /*padding*/ ((lane_id & 15)>>1) + ((lane_id/16) &1)*8 + (lane_id/32);
            //                 inline_ds_read2_b32_no_wait(A_lds_v2fp16, lds_offset, A_reg_tmp[(head_dim_idx*(WARP_M/32) + m_idx)*2 + i][j].f32, 2); 
            //             }
            //             // #pragma unroll
            //             // for(int j=0; j<4; j++) {   //head dim direction
            //             //     int lds_offset = A_lds_stage_offset + head_dim_idx*BLOCK_M*17 + (warp_id*(WARP_M/32) + m_idx)*(32*17) + j*2 + i*32 + (lane_id & 1)*16 + ((lane_id & 15)>>1)*64 + /*padding*/ ((lane_id & 15)>>1) + ((lane_id/16) &1)*8 + (lane_id/32);
            //             //     inline_ds_read_b32_wait(A_lds_v2fp16, lds_offset, A_reg_tmp[(head_dim_idx*(WARP_M/32) + m_idx)*2 + i][j/2].f16x2[j%2]);
            //             // }
            //         }
            //     }
            // }

            ds_read_tile_pad<WARP_M, BLOCK_K, WARP_NUM, Element>(A_lds_v2fp16, A_lds_stage_offset, A_reg_tmp, 0, warp_id, lane_id);
            asm volatile("s_setprio 0");
        }

        // int k_warp_n_id = (warp_id & (WARP_N/WARP_N - 1));
        // int k_lds_stage_offset = STAGES == 1 ? 0 : (stage_id )  * (WARP_N/32) * (BLOCK_K/32)*(32*17);
        // if constexpr (Is_store_B || B_prefetch_level == 1){
        //     k_lds_stage_offset += n_loop * (WARP_N/32) * (K/32)*(32*17);
        // }
        // vec2_Element<Element> *k_lds_v2fp16 =  (vec2_Element<Element> *)(B_lds);
        ds_read2_tile_pad_no_wait(WARP_M, BLOCK_K, WARP_NUM, Element, k_lds_v2fp16, precompute_B_lds_offset, B_reg, stage_id_reg, k_warp_n_id, lane_id);
        // ds_read2_tile_pad_no_wait<WARP_M, BLOCK_K, WARP_NUM, Element>(k_lds_v2fp16, k_lds_stage_offset, B_reg, stage_id_reg, k_warp_n_id, lane_id);
        // ds_read_tile_pad<WARP_M, BLOCK_K, WARP_NUM, Element>(k_lds_v2fp16, k_lds_stage_offset, B_reg, stage_id_reg, k_warp_n_id, lane_id);

        if constexpr (STAGES == 1){
            k_loop++;
        }
        //min tile is 32*32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
        asm volatile("s_setprio 1");
        #pragma unroll
        for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
            const int lgkmcnt = 2 - min_tile_n*2;
            lgkmcnt_wait(lgkmcnt);
            #pragma unroll
            for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                #pragma unroll
                for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
                    for(int head_dim_idx=0; head_dim_idx< (BLOCK_K/32); head_dim_idx++) {
                        #pragma unroll
                        for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                            #pragma unroll
                            for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                                if constexpr (std::is_same<Element,Float8_e4m3_t>::value){
                                    if (A_prefetch_level == 0 || (A_prefetch_level != 3 && k_loop >= (K/BLOCK_K)/2 + 1 )){
                                        C_reg[n_loop*(WARP_N/32*WARP_M/32) + (n_idx) * (WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32 =   flash::mmac<half_t, ElementAccum>(
                                        vec4_Element<half_t>{
                                            UpCast<Element, half_t, true>(A_reg_tmp[(head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[0][0]),
                                            UpCast<Element, half_t, true>(A_reg_tmp[(head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[0][1]),
                                            UpCast<Element, half_t, true>(A_reg_tmp[(head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[1][0]),
                                            UpCast<Element, half_t, true>(A_reg_tmp[(head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[1][1])}, 
                                        vec4_Element<half_t>{
                                            UpCast<Element, half_t, true>(B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[0][0]),
                                            UpCast<Element, half_t, true>(B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[0][1]),
                                            UpCast<Element, half_t, true>(B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[1][0]),
                                            UpCast<Element, half_t, true>(B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[1][1])}, 
                                        C_reg[n_loop*(WARP_N/32*WARP_M/32) + (n_idx) * (WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32);
                                    } else {
                                        C_reg[n_loop*(WARP_N/32*WARP_M/32) + (n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32 =   flash::mmac<half_t, ElementAccum>(
                                        vec4_Element<half_t>{
                                            UpCast<Element, half_t, true>(A_reg[(k_loop-1)*((WARP_M*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[0][0]),
                                            UpCast<Element, half_t, true>(A_reg[(k_loop-1)*((WARP_M*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[0][1]),
                                            UpCast<Element, half_t, true>(A_reg[(k_loop-1)*((WARP_M*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[1][0]),
                                            UpCast<Element, half_t, true>(A_reg[(k_loop-1)*((WARP_M*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[1][1])}, 
                                        vec4_Element<half_t>{
                                            UpCast<Element, half_t, true>(B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[0][0]),
                                            UpCast<Element, half_t, true>(B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[0][1]),
                                            UpCast<Element, half_t, true>(B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[1][0]),
                                            UpCast<Element, half_t, true>(B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[1][1])}, 
                                        C_reg[n_loop*(WARP_N/32*WARP_M/32) + (n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32);
                                    }
                                } else {
                                    if (A_prefetch_level == 0 || (A_prefetch_level != 3 && k_loop >= (K/BLOCK_K)/2 + 1 )){
                                        C_reg[n_loop*(WARP_N/32*WARP_M/32) + (n_idx) * (WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32 =   flash::mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{
                                            A_reg_tmp[(head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[0][0],
                                            A_reg_tmp[(head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[0][1],
                                            A_reg_tmp[(head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[1][0],
                                            A_reg_tmp[(head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[1][1]}, 
                                        vec4_Element<Element>{
                                            B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[0][0],
                                            B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[0][1],
                                            B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[1][0],
                                            B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[1][1]}, 
                                        C_reg[n_loop*(WARP_N/32*WARP_M/32) + (n_idx) * (WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32);
                                    } else {
                                        C_reg[n_loop*(WARP_N/32*WARP_M/32) + (n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32 =   flash::mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{
                                            A_reg[(k_loop-1)*((WARP_M*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[0][0],
                                            A_reg[(k_loop-1)*((WARP_M*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[0][1],
                                            A_reg[(k_loop-1)*((WARP_M*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[1][0],
                                            A_reg[(k_loop-1)*((WARP_M*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_M/32) + m_idx)*2 + min_tile_m][min_tile_k].f16x2[1][1]}, 
                                        vec4_Element<Element>{
                                            B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[0][0],
                                            B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[0][1],
                                            B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[1][0],
                                            B_reg[(stage_id_reg *((WARP_N*BLOCK_K)/(32*32))*2 + (head_dim_idx*(WARP_N/32) + n_idx))*2 + min_tile_n][min_tile_k].f16x2[1][1]}, 
                                        C_reg[n_loop*(WARP_N/32*WARP_M/32) + (n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        asm volatile("s_setprio 0");
        if constexpr (STAGES > 1){
            if constexpr (!Is_store_B && B_prefetch_level !=1) {
                stage_id ^= 1;
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __syncthreads();
                __builtin_amdgcn_sched_barrier(0);
            } else{
                stage_id++;
            } 
        } else {
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __syncthreads();
            __builtin_amdgcn_sched_barrier(0);            
        }
    }
    }
#endif
}


// 无预取：prefetch_level = 0;   预取到LDS:prefetch_level = 1;    预取到寄存器:prefetch_level = 2;
template<bool Is_store_B, bool Is_preload_C, bool Is_even_MN, int A_prefetch_level, int B_prefetch_level, int K, int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum = float>                                                                                                               
__forceinline__ __device__ void  gemm_tt_kq_gfx938(
        vec4_uint A_ptr, 
        vec4_uint B_ptr, 
        Element* A_lds, 
        Element* B_lds, 
        int max_m_len_offset, 
        int max_n_len_offset, 
        union_vec4_f16x2<Element> A_reg[(K/BLOCK_K)*((WARP_M*BLOCK_K)/(32*32))*2/((A_prefetch_level == 3)? 1 : 2)], 
        union_vec4_f16x2<Element> B_reg[STAGES*((WARP_N*BLOCK_K)/(32*32))*2], 
        union_vec4_fp32 C_reg[(WARP_M/32)*(BLOCK_N/32)][4], 
        int warp_id, 
        int seqlen_A_stride, 
        int seqlen_B_stride) {
    const int ELEMENT_BYTES   = sizeof(Element);
    const int WARP_NUM = BLOCK_M/WARP_M;

    //wave size should be defined in launch file. Here use 64 threads
    int lane_id = threadIdx.x & 63; //lane id, 0-63

    int row = lane_id % 4;
    int col = lane_id / 4;

#if 1
    for(int n_loop = 0 ; n_loop < BLOCK_N/WARP_N; n_loop++)
    {
    int stage_id = 0;
    int stage_id_reg = 0;
     {  int k_loop = 0;
        if(STAGES > 1) {
            if(A_prefetch_level == 0) {
                int A_block_buffer_load_global_offset = k_loop * BLOCK_K;
                int A_lds_stage_offset = stage_id * BLOCK_M* BLOCK_K;
                prefetch_to_lds_gfx938<true, BLOCK_M, BLOCK_K, Element, ElementAccum, Is_even_MN>(A_ptr, A_block_buffer_load_global_offset, A_lds + A_lds_stage_offset, seqlen_A_stride, warp_id);
            }

            if(B_prefetch_level == 0) {
                int B_block_buffer_load_global_offset = k_loop * BLOCK_K + n_loop * WARP_N * K;
                int B_lds_stage_offset = stage_id * WARP_N * BLOCK_K;
                if constexpr (Is_store_B){
                    B_lds_stage_offset += n_loop * K * WARP_N;
                }
                prefetch_to_lds_gfx938<true, WARP_N, BLOCK_K, Element, ElementAccum, Is_even_MN>(B_ptr, B_block_buffer_load_global_offset, B_lds + B_lds_stage_offset, seqlen_B_stride, warp_id);
            }
        }
     }
    
    // int lds_offset = row * 8 + col * 32;
    for(int k_loop = 1; k_loop<(K/BLOCK_K) + 1; k_loop++) {
        if constexpr (STAGES > 1) {
            if constexpr (Is_store_B || B_prefetch_level == 1){
                stage_id++;
            } else {
                stage_id ^= 1;
            }            
        }

        if(STAGES == 1) {
            k_loop--;
        }
        if(k_loop < (K/BLOCK_K)){
            if(A_prefetch_level == 0 || (A_prefetch_level == 1 && k_loop >= (K/BLOCK_K)/2)) {
                int A_block_buffer_load_global_offset = k_loop * BLOCK_K;
                int A_lds_stage_offset = stage_id * BLOCK_M* BLOCK_K;
                prefetch_to_lds_gfx938<true, BLOCK_M, BLOCK_K, Element, ElementAccum, Is_even_MN>(A_ptr, A_block_buffer_load_global_offset, A_lds + A_lds_stage_offset, seqlen_A_stride, warp_id);
            }

            if(B_prefetch_level == 0) {
                int B_block_buffer_load_global_offset = k_loop * BLOCK_K + n_loop * WARP_N * K;
                int B_lds_stage_offset = stage_id * WARP_N * BLOCK_K;
                if constexpr (Is_store_B){
                    B_lds_stage_offset += n_loop * K * WARP_N;
                }
                prefetch_to_lds_gfx938<true, WARP_N, BLOCK_K, Element, ElementAccum, Is_even_MN>(B_ptr, B_block_buffer_load_global_offset, B_lds + B_lds_stage_offset, seqlen_B_stride, warp_id);
            }
        }
        else if (B_prefetch_level==0){
            vmcnt_wait_nosync(0);
        }
        
        //MLS可以一条指令读32*32，而buffer_load_dword一条指令读2*64的数据，所以waitcnt需要进行修改
        //且MLS是一个warp读32*32，4个warp读128*32，需要判断warp_id来wait
        if(STAGES > 1) {
            if constexpr(B_prefetch_level==1){
                if((k_loop - 1) % WARP_NUM == warp_id) 
                {
                    if(Is_preload_C) {
                        vmcnt_wait_nosync(1);
                    } else {
                        vmcnt_wait_nosync(0);
                    }
                }
            } else {
                if(k_loop < (K/BLOCK_K)){
                    if(A_prefetch_level == 0 && B_prefetch_level != 0) {
                        //BM = 128
                        vmcnt_wait_nosync((BLOCK_M * BLOCK_K) / (32*32)/WARP_NUM);
                    } else if(A_prefetch_level != 0 && B_prefetch_level == 0) {
                        //BN = 32
                        if(warp_id == 0) {
                            vmcnt_wait_nosync(1);
                        }
                    } else if(A_prefetch_level == 0 && B_prefetch_level == 0) {
                        //BM= 128 & BN = 32
                        if(warp_id == 0) {
                            vmcnt_wait_nosync((BLOCK_M * BLOCK_K) / (32*32)/WARP_NUM + 1);
                        } else {
                            vmcnt_wait_nosync(1);
                        }
                    }
                }
            }
        } else {
            vmcnt_wait_nosync(0);
        }
        __syncthreads();

        if constexpr (STAGES > 1) {
            if constexpr (Is_store_B || B_prefetch_level == 1){
                stage_id--;
            } else {
                stage_id ^= 1;
            }   
        }

        union_vec4_f16x2<Element> A_reg_tmp[2];
        if (A_prefetch_level == 0 || (A_prefetch_level != 3 && k_loop >= (K/BLOCK_K)/2 )) {
            int A_lds_stage_offset = stage_id  * BLOCK_M * BLOCK_K;
            // DS_READ_MATRIX_32X32_B16(ds_offset_cast(A_lds + A_lds_stage_offset), A_reg_tmp[0].f16, A_reg_tmp[1].f16, true);
            if constexpr (std::is_same_v<Element, half_t>) {
                A_reg_tmp[0].f16x8 =  __builtin_hcu_ds_read_matrix_trans_format_f16(A_lds + A_lds_stage_offset, 0, 2, 1, 0);
                A_reg_tmp[1].f16x8 =  __builtin_hcu_ds_read_matrix_trans_format_f16(A_lds + A_lds_stage_offset, 1024, 2, 1, 0);
            } else {
                A_reg_tmp[0].f16x8 =  __builtin_hcu_ds_read_matrix_trans_format_bf16(A_lds + A_lds_stage_offset, 0, 2, 1, 0);
                A_reg_tmp[1].f16x8 =  __builtin_hcu_ds_read_matrix_trans_format_bf16(A_lds + A_lds_stage_offset, 1024, 2, 1, 0);
            }
        }
        int B_lds_stage_offset = stage_id  * WARP_N * BLOCK_K;
        DS_READ_MATRIX_32X32_B16(ds_offset_cast(B_lds + B_lds_stage_offset), B_reg[0].f16, B_reg[1].f16, true);
    

        if constexpr (STAGES == 1){
            k_loop++;
        }
        //min tile is 32*32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
        // asm volatile("s_setprio 1");
        lgkmcnt_wait(0);
        #pragma unroll
        for(int min_tile_n=0; min_tile_n<2; min_tile_n++) { 
            #pragma unroll
            for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
                #pragma unroll
                for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
                    for(int head_dim_idx=0; head_dim_idx< (BLOCK_K/32); head_dim_idx++) {
                        #pragma unroll
                        for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                            #pragma unroll
                            for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                                if constexpr (std::is_same<Element,Float8_e4m3_t>::value){
                                } else {
                                    if (A_prefetch_level == 0 || (A_prefetch_level != 3 && k_loop >= (K/BLOCK_K)/2 + 1 )){
                                        C_reg[n_loop*(WARP_N/32*WARP_M/32) + (n_idx) * (WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32 =   mmac_4interleave<Element, ElementAccum>(
                                        A_reg_tmp[min_tile_m].f16x4[min_tile_k],
                                        B_reg[min_tile_n].f16x4[min_tile_k],
                                        C_reg[n_loop*(WARP_N/32*WARP_M/32) + (n_idx) * (WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32);
                                    } else {
                                        C_reg[n_loop*(WARP_N/32*WARP_M/32) + (n_idx) * (WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32 =   mmac_4interleave<Element, ElementAccum>(
                                        A_reg[(k_loop - 1) * 2 + min_tile_m].f16x4[min_tile_k],
                                        B_reg[min_tile_n].f16x4[min_tile_k],
                                        C_reg[n_loop*(WARP_N/32*WARP_M/32) + (n_idx)*(WARP_M/32) + m_idx][min_tile_n*2 + min_tile_m].f32);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // asm volatile("s_setprio 0");
        if constexpr (STAGES > 1){
            if constexpr (!Is_store_B && B_prefetch_level !=1) {
                stage_id ^= 1;
                __builtin_amdgcn_sched_barrier(0);
                // asm volatile("s_waitcnt lgkmcnt(0)");
                __syncthreads();
                __builtin_amdgcn_sched_barrier(0);
            } else{
                stage_id++;
            } 
        } else {
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __syncthreads();
            __builtin_amdgcn_sched_barrier(0);            
        }
    }
    }
#endif
}