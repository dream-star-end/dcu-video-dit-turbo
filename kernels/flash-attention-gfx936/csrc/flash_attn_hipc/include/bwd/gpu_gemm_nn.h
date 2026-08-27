#pragma once
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "intrinsic.h"
#include "prefetch.h"
//                                                                                   K                  BLOCK_K      BLOCK_N      BLOCK_M      BLOCK_K     WARP_N
template<bool Is_preload_A, bool Is_store_A, bool Is_preload_C, bool Is_even_MN, int M/*head_dim*/, int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum = float>
__forceinline__ __device__ void  gpu_gemm_B_in_reg(vec4_uint A_ptr, vec4_uint C_ptr, Element* A_lds, union_vec2_f16x2<Element> B_reg[(WARP_M/32)*(BLOCK_K/32)][4], union_vec4_fp32 C_reg[(M/BLOCK_M)*(WARP_M/32)*(WARP_N/32)][4], int N/*seq_kv*/, int K/*seq_q*/, int warp_id, int seqlen_A_stride)
{
    #if 1
    const int WARP_NUM = (BLOCK_M*BLOCK_N)/(WARP_M*WARP_N);
    const int A_lds_load_num = (BLOCK_M*BLOCK_K) / (4*32);
    static_assert(BLOCK_K>=32, "Error: gpu_gemm_B_in_reg gemm BLOCK_K must be equal or greater than 32");
    static_assert(BLOCK_N>=WARP_N, "Error: gpu_gemm_B_in_reg gemm BLOCK_N must be equal or greater than WARP_N");
    static_assert(BLOCK_M==WARP_M, "Error: gpu_gemm_B_in_reg gemm BLOCK_M must be equal to WARP_M");

    union_vec2_f16x2<Element> A_reg[((WARP_M*BLOCK_K)/(32*32))*2][2];
    //c mini tile is 32*32
    // vec4_fp32 o[(WARP_M/32)*(WARP_N/32)][4]={0};

    // __shared__ Element A_lds[STAGES*BLOCK_N * BLOCK_K];

    //wave size should be defined in launch file. Here use 64 threads

    int lane_id = threadIdx.x & 63; //lane id, 0-63

    int row = lane_id % 4;
    int col = lane_id / 4;

    int stage_id = 0;

    if(STAGES > 1 && (!Is_preload_A)) {
        int m_loop = 0;

        int A_block_buffer_load_global_offset = m_loop*BLOCK_M ; //+ k_loop * BLOCK_K * N;

        // A_ptr buffer load mini size is 32*32, buffer_load_dword mini size is 4*32
        int A_lane_m_idx = lane_id % 16;
        // int A_lane_k_idx = lane_id / 16;
        int A_lane_k_idx = ((lane_id >> 4) & 1)*2 + ((lane_id >> 4) >> 1); //(0, 1, 2, 3) --> (0, 2, 1, 3)
        for(int warp_loop=warp_id; warp_loop<A_lds_load_num; warp_loop+=WARP_NUM) {
        // for(int warp_loop_tmp = 0; warp_loop_tmp < A_lds_load_num / WARP_NUM; warp_loop_tmp++){
        //     int warp_loop = warp_loop_tmp * WARP_NUM + warp_id;
            //global->lds, right matrix
            int A_warp_buffer_load_k_id = (warp_loop / (BLOCK_M/32));  //seq_len
            int A_warp_buffer_load_m_id = (warp_loop % (BLOCK_M/32));  //head_dim

            {
                int A_warp_buffer_load_global_offset = (A_warp_buffer_load_m_id * 32);
                int A_warp_buffer_load_lds_offset    = (A_warp_buffer_load_m_id * 32) + (A_warp_buffer_load_k_id * 4 * BLOCK_M);
                if(Is_store_A){
                    A_warp_buffer_load_lds_offset    = (A_warp_buffer_load_m_id * 32) + (A_warp_buffer_load_k_id * (4 * BLOCK_M + 2));
                }
                int A_gsoffset = (A_block_buffer_load_global_offset + A_warp_buffer_load_global_offset)/2 ;
                int A_gvoffset;
                if constexpr (Is_even_MN){
                    A_gvoffset = ((A_lane_m_idx * 2 + (A_lane_k_idx + A_warp_buffer_load_k_id*4)* seqlen_A_stride))/2 ;
                } else {
                    A_gvoffset = ((A_lane_m_idx * 2 + min(A_lane_k_idx + A_warp_buffer_load_k_id*4, K-1)* seqlen_A_stride))/2 ;
                }
                
                // int gvOffset = (64*8 + lane_id*8)/2;
                int A_lds_offset = ((stage_id)*BLOCK_K*BLOCK_M + A_warp_buffer_load_lds_offset)/2;
                if(Is_store_A){
                    A_lds_offset = ((stage_id)*(BLOCK_K/32)*(BLOCK_M/32)*32*34 + A_warp_buffer_load_lds_offset)/2;
                }
                builtin_buffer_load_dword_lds(A_lds , A_ptr, A_lds_offset, A_gsoffset, A_gvoffset);
            }
        }
    }

    #if 1   
    // int lds_offset = row * 8 + col * 32;
    for(int m_loop = 1; m_loop<(M/BLOCK_M) + 1; m_loop++) {
        if(STAGES > 1) {
            if constexpr(Is_preload_A || Is_store_A){
                stage_id ++;
            } else {
                stage_id = stage_id ^ 1;
            }
        }

        if(STAGES == 1) {
            m_loop--;
        }

        if((!Is_preload_A)&& m_loop < (M/BLOCK_M)) {
            int A_block_buffer_load_global_offset = m_loop*BLOCK_M;
            if(Is_store_A){
                int A_lds_stage_offset = (stage_id)*(BLOCK_K/32)*(BLOCK_M/32)*32*34;
                buffer_load_lds_tile_pad(Is_even_MN, WARP_NUM, seqlen_A_stride, BLOCK_M, BLOCK_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset, K, warp_id, lane_id);
            } else {
                int A_lds_stage_offset = (stage_id)*BLOCK_K*BLOCK_M;
                buffer_load_lds_tile(Is_even_MN, WARP_NUM, seqlen_A_stride, BLOCK_M, BLOCK_K, Element, A_ptr, A_lds, A_block_buffer_load_global_offset, A_lds_stage_offset, K, warp_id, lane_id);
            }
        }

        if(!Is_preload_A){
            if(STAGES > 1) {
                if(m_loop < (M/BLOCK_M)){
                    // if constexpr(Is_preload_A){
                    //     vmcnt_wait((M/BLOCK_M - m_loop) * (BLOCK_K*BLOCK_M) / (4*32)/WARP_NUM);
                    // } else {
                        vmcnt_wait((BLOCK_K*BLOCK_M) / (4*32)/WARP_NUM);
                    // }
                } else {
                    vmcnt_wait(0);
                }
            } else {
                vmcnt_wait(0);
            }
        }

        if constexpr (STAGES > 1) {
            if constexpr(Is_preload_A || Is_store_A){
                stage_id --;
            } else {
                stage_id = stage_id ^ 1;
            }
        }
        if (Is_preload_C && m_loop > 1) {
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __syncthreads();
            __builtin_amdgcn_sched_barrier(0);
            int C_block_buffer_load_global_offset = (m_loop - 2)*BLOCK_M;
            // A_ptr buffer load mini size is 32*32, buffer_load_dword mini size is 4*32
            int C_lane_m_idx = lane_id % 16;
            // int A_lane_k_idx = lane_id / 16;
            int C_lane_k_idx = ((lane_id >> 4) & 1)*2 + ((lane_id >> 4) >> 1); //(0, 1, 2, 3) --> (0, 2, 1, 3)
            for(int warp_loop_temp=0; warp_loop_temp< A_lds_load_num/WARP_NUM; warp_loop_temp++) {
                int warp_loop = warp_loop_temp * WARP_NUM + warp_id;
                //global->lds, right matrix
                int C_warp_buffer_load_k_id = (warp_loop / (BLOCK_K/32));  //seq_len
                int C_warp_buffer_load_m_id = (warp_loop % (BLOCK_M/32));  //head_dim
                {
                    int C_warp_buffer_load_global_offset = (C_warp_buffer_load_m_id * 32);
                    int C_warp_buffer_load_lds_offset    = (C_warp_buffer_load_m_id * 32) + (C_warp_buffer_load_k_id * 4 * BLOCK_M);
                    if(Is_store_A){
                        C_warp_buffer_load_lds_offset    = (C_warp_buffer_load_m_id * 32) + (C_warp_buffer_load_k_id * (4 * BLOCK_M + 2));
                    }
                    int C_gsoffset = (C_block_buffer_load_global_offset + C_warp_buffer_load_global_offset)/2 ;
                    int C_gvoffset;
                    if constexpr (Is_even_MN){
                        C_gvoffset = ((C_lane_m_idx * 2 + (C_lane_k_idx + C_warp_buffer_load_k_id*4)* M))/2 ;
                    } else {
                        C_gvoffset = ((C_lane_m_idx * 2 + min(C_lane_k_idx + C_warp_buffer_load_k_id*4, K-1)* M))/2 ;
                    }
                    
                    // int gvOffset = (64*8 + lane_id*8)/2;
                    int A_lds_offset = ((m_loop - 2)*BLOCK_K*BLOCK_M + C_warp_buffer_load_lds_offset)/2;
                    if(Is_store_A){
                        A_lds_offset = ((m_loop - 2)*(BLOCK_K/32)*(BLOCK_M/32)*32*34 + C_warp_buffer_load_lds_offset)/2;
                    }
                    builtin_buffer_load_dword_lds(A_lds , C_ptr, A_lds_offset, C_gsoffset, C_gvoffset);
                }
            }
        }
        //lds -> vgpr use ds_read_m; left matrix        

        int A_lane_head_dim_idx = lane_id % 16;
        int A_lane_seq_idx = lane_id / 16;

        // __builtin_amdgcn_s_waitcnt(4080 + ((M/BLOCK_M) - m_loop)*(A_lds_load_num/WARP_NUM));
        // vmcnt_wait_no_barrier(((M/BLOCK_M) - m_loop)*(A_lds_load_num/WARP_NUM));

        vec2_Element<Element> *A_lds_v2fp16 =  (vec2_Element<Element> *)(A_lds );
        //lds -> vgpr use ds_read_m; right matrix        
        #pragma unroll
        for(int head_dim_idx=0; head_dim_idx<(WARP_M/32); head_dim_idx++) {
            #pragma unroll
            for(int seq_idx=0; seq_idx<(BLOCK_K/32); seq_idx++) {
                #pragma unroll
                for(int seq_min_tile_idx=0; seq_min_tile_idx<2; seq_min_tile_idx++) { // min k tile
                    // __builtin_amdgcn_s_waitcnt(4082);
                    #pragma unroll
                    for(int vec_idx=0; vec_idx<4; vec_idx++) //16*32 half need 4 ds_read_b32
                    {
                        // int lds_offset = (stage_id*BLOCK_K*BLOCK_M + (seq_idx*32*BLOCK_M) + head_dim_idx*32 * 32 + A_lane_seq_idx/2*4*32 + A_lane_seq_idx%2*32 + (seq_min_tile_idx*32*2)   + vec_idx*8*32 + A_lane_head_dim_idx*2)/2;
                        int lds_offset = stage_id * BLOCK_K * BLOCK_M / 2 + seq_idx * BLOCK_M * 16 + head_dim_idx * 512 + A_lane_seq_idx/2 * 64 + A_lane_seq_idx % 2 * 16 + seq_min_tile_idx * 32 + vec_idx * 128 + A_lane_head_dim_idx;
                        if constexpr(Is_preload_A || Is_store_A){
                            // lds_offset = (stage_id*(BLOCK_K/32)*(BLOCK_M/32)*32*34 + (seq_idx*34*BLOCK_M) + head_dim_idx*32 * 34 + A_lane_seq_idx/2*(4*32 + 2) + A_lane_seq_idx%2*32 + (seq_min_tile_idx*32*2)   + vec_idx*(8*32+4) + A_lane_head_dim_idx*2)/2;
                            // lds_offset += (stage_id*(BLOCK_K/32)*(BLOCK_M/32)*32*2 + 2*seq_idx*BLOCK_M + head_dim_idx * 32 * 2 + A_lane_seq_idx/2*2 + vec_idx*4)/2;
                            lds_offset += stage_id * BLOCK_K * BLOCK_M / 32 + seq_idx * BLOCK_M + head_dim_idx * 32 + A_lane_seq_idx / 2 + vec_idx * 2;
                        }
                        inline_ds_read_b32_wait(A_lds_v2fp16, lds_offset, A_reg[(head_dim_idx*(BLOCK_K/32) + seq_idx)*2 + seq_min_tile_idx][vec_idx/2].f16x2[vec_idx%2]); 
                    }

                    // #pragma unroll
                    // for(int vec_idx=0; vec_idx<2; vec_idx++) //16*32 half need 4 ds_read_b32
                    // {
                    //     int lds_offset = (stage_id*BLOCK_K*BLOCK_M + (seq_idx*32*BLOCK_M) + head_dim_idx*32 * 32 + A_lane_seq_idx/2*4*32 + A_lane_seq_idx%2*32 + (seq_min_tile_idx*32*2)   + vec_idx*16*32 + A_lane_head_dim_idx*2)/2;
                    //     if constexpr(Is_preload_A || Is_store_A){
                    //         lds_offset = (stage_id*(BLOCK_K/32)*(BLOCK_M/32)*32*34 + (seq_idx*34*BLOCK_M) + head_dim_idx*32 * 34 + A_lane_seq_idx/2*(4*32 + 2) + A_lane_seq_idx%2*32 + (seq_min_tile_idx*32*2)   + vec_idx*(16*32+8) + A_lane_head_dim_idx*2)/2;
                    //     }
                    //     inline_ds_read2_b32_no_wait(A_lds_v2fp16, lds_offset, A_reg[(head_dim_idx*(BLOCK_K/32) + seq_idx)*2 + seq_min_tile_idx][vec_idx].f32, 4*32);
                    // }
                }
            }
        }
        // asm volatile("s_waitcnt lgkmcnt(0)");
        // __builtin_amdgcn_sched_barrier(0); 
        if constexpr (STAGES == 1){
            m_loop++;
        }
        asm volatile("s_setprio 1");
        #pragma unroll
        for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
            #pragma unroll
            for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
                #pragma unroll
                for(int k_idx=0; k_idx<(BLOCK_K/32); k_idx++) {  //BLOCK_K mini size is 32
                    //min tile is 32*32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
                    #pragma unroll
                    for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                        #pragma unroll
                        for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                            #pragma unroll
                            for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                                if constexpr (std::is_same<Element,Float8_e4m3_t>::value){
                                    C_reg[(m_loop-1) * ((WARP_M/32)*(WARP_N/32)) + (n_idx*(WARP_M/32) + m_idx)][min_tile_n*2 + min_tile_m].f32 =  flash::mmac<half_t, ElementAccum>(
                                        vec4_Element<half_t>{
                                            UpCast<Element, half_t, true>(A_reg[(m_idx* (BLOCK_K/32) + k_idx)*2 + 0][min_tile_k].f16x2[0][min_tile_m]), 
                                            UpCast<Element, half_t, true>(A_reg[(m_idx* (BLOCK_K/32) + k_idx)*2 + 1][min_tile_k].f16x2[0][min_tile_m]), 
                                            UpCast<Element, half_t, true>(A_reg[(m_idx* (BLOCK_K/32) + k_idx)*2 + 0][min_tile_k].f16x2[1][min_tile_m]), 
                                            UpCast<Element, half_t, true>(A_reg[(m_idx* (BLOCK_K/32) + k_idx)*2 + 1][min_tile_k].f16x2[1][min_tile_m])}, 
                                        vec4_Element<half_t>{
                                            UpCast<Element, half_t, true>(B_reg[(k_idx)*(WARP_N/32) + n_idx][0*2 + min_tile_n].f16x2[min_tile_k][0]), 
                                            UpCast<Element, half_t, true>(B_reg[(k_idx)*(WARP_N/32) + n_idx][1*2 + min_tile_n].f16x2[min_tile_k][0]), 
                                            UpCast<Element, half_t, true>(B_reg[(k_idx)*(WARP_N/32) + n_idx][0*2 + min_tile_n].f16x2[min_tile_k][1]), 
                                            UpCast<Element, half_t, true>(B_reg[(k_idx)*(WARP_N/32) + n_idx][1*2 + min_tile_n].f16x2[min_tile_k][1])},  
                                        C_reg[(m_loop-1) * ((WARP_M/32)*(WARP_N/32)) + (n_idx*(WARP_M/32) + m_idx)][min_tile_n*2 + min_tile_m].f32);
                                } else {
                                    C_reg[(m_loop-1) * ((WARP_M/32)*(WARP_N/32)) + (n_idx*(WARP_M/32) + m_idx)][min_tile_n*2 + min_tile_m].f32 =  flash::mmac<Element, ElementAccum>(
                                        vec4_Element<Element>{
                                            A_reg[(m_idx* (BLOCK_K/32) + k_idx)*2 + 0][min_tile_k].f16x2[0][min_tile_m], 
                                            A_reg[(m_idx* (BLOCK_K/32) + k_idx)*2 + 1][min_tile_k].f16x2[0][min_tile_m], 
                                            A_reg[(m_idx* (BLOCK_K/32) + k_idx)*2 + 0][min_tile_k].f16x2[1][min_tile_m], 
                                            A_reg[(m_idx* (BLOCK_K/32) + k_idx)*2 + 1][min_tile_k].f16x2[1][min_tile_m]}, 
                                        vec4_Element<Element>{
                                            B_reg[(k_idx)*(WARP_N/32) + n_idx][0*2 + min_tile_n].f16x2[min_tile_k][0], 
                                            B_reg[(k_idx)*(WARP_N/32) + n_idx][1*2 + min_tile_n].f16x2[min_tile_k][0], 
                                            B_reg[(k_idx)*(WARP_N/32) + n_idx][0*2 + min_tile_n].f16x2[min_tile_k][1], 
                                            B_reg[(k_idx)*(WARP_N/32) + n_idx][1*2 + min_tile_n].f16x2[min_tile_k][1]},
                                        C_reg[(m_loop-1) * ((WARP_M/32)*(WARP_N/32)) + (n_idx*(WARP_M/32) + m_idx)][min_tile_n*2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
        }
        asm volatile("s_setprio 0");
        if(STAGES > 1) {
            if constexpr(Is_preload_A || Is_store_A){
                stage_id ++;
            } else {
                stage_id ^=1;
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __syncthreads();
                __builtin_amdgcn_sched_barrier(0);
            }
        } else {
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __syncthreads();
            __builtin_amdgcn_sched_barrier(0);
        }
    }

    if constexpr (Is_preload_C) {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);
        int C_block_buffer_load_global_offset = 3*BLOCK_M;
        // A_ptr buffer load mini size is 32*32, buffer_load_dword mini size is 4*32
        int C_lane_m_idx = lane_id % 16;
        // int A_lane_k_idx = lane_id / 16;
        int C_lane_k_idx = ((lane_id >> 4) & 1)*2 + ((lane_id >> 4) >> 1); //(0, 1, 2, 3) --> (0, 2, 1, 3)
        const int C_lds_load_num = (BLOCK_M*BLOCK_K) / (4*32);
        for(int warp_loop_temp=0; warp_loop_temp< C_lds_load_num/WARP_NUM; warp_loop_temp++) {
            int warp_loop = warp_loop_temp * WARP_NUM + warp_id;
            //global->lds, right matrix
            int C_warp_buffer_load_k_id = (warp_loop / (BLOCK_K/32));  //seq_len
            int C_warp_buffer_load_m_id = (warp_loop % (BLOCK_M/32));  //head_dim
            {
                int C_warp_buffer_load_global_offset = (C_warp_buffer_load_m_id * 32);
                int C_warp_buffer_load_lds_offset    = (C_warp_buffer_load_m_id * 32) + (C_warp_buffer_load_k_id * 4 * BLOCK_M);
                if(Is_store_A){
                    C_warp_buffer_load_lds_offset    = (C_warp_buffer_load_m_id * 32) + (C_warp_buffer_load_k_id * (4 * BLOCK_M + 2));
                }
                int C_gsoffset = (C_block_buffer_load_global_offset + C_warp_buffer_load_global_offset)/2 ;
                int C_gvoffset;
                if constexpr (Is_even_MN){
                    C_gvoffset = ((C_lane_m_idx * 2 + (C_lane_k_idx + C_warp_buffer_load_k_id*4)* M))/2 ;
                } else {
                    C_gvoffset = ((C_lane_m_idx * 2 + min(C_lane_k_idx + C_warp_buffer_load_k_id*4, K-1)* M))/2 ;
                }
                
                // int gvOffset = (64*8 + lane_id*8)/2;
                int A_lds_offset = (3*BLOCK_K*BLOCK_M + C_warp_buffer_load_lds_offset)/2;
                if(Is_store_A){
                    A_lds_offset = (3*(BLOCK_K/32)*(BLOCK_M/32)*32*34 + C_warp_buffer_load_lds_offset)/2;
                }
                builtin_buffer_load_dword_lds(A_lds , C_ptr, A_lds_offset, C_gsoffset, C_gvoffset);
            }
        }
    }
    #endif
#endif
}




//                                                                                   K                  BLOCK_K      BLOCK_N      BLOCK_M      BLOCK_K     WARP_N
template<bool Is_preload_A, bool Is_store_A, bool Is_even_MN, int M/*head_dim*/, int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum = float>
__forceinline__ __device__ void  gpu_gemm_B_in_reg_gfx938(
    vec4_uint A_ptr, 
    vec4_uint C_ptr, 
    Element* A_lds, 
    union_vec4_f16x2<Element> B_reg[(WARP_M/32)*(BLOCK_K/32)*2], 
    union_vec4_fp32 C_reg[(M/BLOCK_M)*(WARP_M/32)*(WARP_N/32)][4], 
    int N/*seq_kv*/, 
    int K/*seq_q*/, 
    int warp_id, 
    int seqlen_A_stride) {
    #if 1
    const int WARP_NUM = (BLOCK_M*BLOCK_N)/(WARP_M*WARP_N);
    const int A_lds_load_num = (BLOCK_M*BLOCK_K) / (4*32);
    static_assert(BLOCK_K>=32, "Error: gpu_gemm_B_in_reg gemm BLOCK_K must be equal or greater than 32");
    static_assert(BLOCK_N>=WARP_N, "Error: gpu_gemm_B_in_reg gemm BLOCK_N must be equal or greater than WARP_N");
    static_assert(BLOCK_M==WARP_M, "Error: gpu_gemm_B_in_reg gemm BLOCK_M must be equal to WARP_M");

    union_vec4_f16x2<Element> A_reg[((WARP_M*BLOCK_K)/(32*32))*2];
    //c mini tile is 32*32
    // vec4_fp32 o[(WARP_M/32)*(WARP_N/32)][4]={0};

    // __shared__ Element A_lds[STAGES*BLOCK_N * BLOCK_K];

    //wave size should be defined in launch file. Here use 64 threads

    int lane_id = threadIdx.x & 63; //lane id, 0-63

    int row = lane_id % 4;
    int col = lane_id / 4;

    int stage_id = 0;

    if(STAGES > 1 && (!Is_preload_A)) {
        int m_loop = 0;
        int A_block_buffer_load_global_offset = m_loop * BLOCK_M;
        int A_lds_stage_offset = stage_id * BLOCK_M * BLOCK_K;
        prefetch_to_lds_gfx938<false, BLOCK_M, BLOCK_K, Element, ElementAccum, Is_even_MN>(A_ptr, A_block_buffer_load_global_offset, A_lds + A_lds_stage_offset, seqlen_A_stride, warp_id);
    }

    #if 1   
    // int lds_offset = row * 8 + col * 32;
    for(int m_loop = 1; m_loop<(M/BLOCK_M) + 1; m_loop++) {
        if(STAGES > 1) {
            if constexpr(Is_preload_A || Is_store_A){
                stage_id ++;
            } else {
                stage_id = stage_id ^ 1;
            }
        }

        if(STAGES == 1) {
            m_loop--;
        }

        if((!Is_preload_A)&& m_loop < (M/BLOCK_M)) {
            int A_block_buffer_load_global_offset = m_loop*BLOCK_M;
            int A_lds_stage_offset = (stage_id)*BLOCK_K*BLOCK_M;
            prefetch_to_lds_gfx938<false, BLOCK_M, BLOCK_K, Element, ElementAccum, Is_even_MN>(A_ptr, A_block_buffer_load_global_offset, A_lds + A_lds_stage_offset, seqlen_A_stride, warp_id);
        }

        //BM = 32, BK = 32
        if(warp_id == 0) {
            if(!Is_preload_A){
                if(STAGES > 1 && m_loop < (M/BLOCK_M)) {
                    vmcnt_wait(1);
                } else {
                    vmcnt_wait(0);
                }
            }
        }

        if constexpr (STAGES > 1) {
            if constexpr(Is_preload_A || Is_store_A){
                stage_id --;
            } else {
                stage_id = stage_id ^ 1;
            }
        }
        //lds -> vgpr use ds_read_m; left matrix        

        if(!Is_preload_A) {
            int A_lds_stage_offset = stage_id * BLOCK_K * BLOCK_M;
            // DS_READ_MATRIX_32X32_B16(ds_offset_cast(A_lds + A_lds_stage_offset), A_reg[0].f16, A_reg[1].f16, false);
            if constexpr (std::is_same_v<Element, half_t>) {
                A_reg[0].f16x8 =  __builtin_hcu_ds_read_matrix_format_f16(A_lds + A_lds_stage_offset, 0, 2, 1, 0);
                A_reg[1].f16x8 =  __builtin_hcu_ds_read_matrix_format_f16(A_lds + A_lds_stage_offset, 1024, 2, 1, 0);
            } else {
                A_reg[0].f16x8 =  __builtin_hcu_ds_read_matrix_format_bf16(A_lds + A_lds_stage_offset, 0, 2, 1, 0);
                A_reg[1].f16x8 =  __builtin_hcu_ds_read_matrix_format_bf16(A_lds + A_lds_stage_offset, 1024, 2, 1, 0);
            }
        } else {
            // gfx938 m_ab = 0的gemm想要复用m_ab = 1的LDS数据
            int A_lane_head_dim_idx = lane_id % 16;
            int A_lane_seq_idx = lane_id / 16;
            vec2_Element<Element> *A_lds_v2fp16 =  (vec2_Element<Element> *)(A_lds);
            for(int min_tile_k = 0; min_tile_k < 2; min_tile_k++) {
                for(int vec_idx = 0; vec_idx < 4; vec_idx++) {
                    //dword为单位
                    int lds_offset = stage_id * BLOCK_K * BLOCK_M / 2 + A_lane_seq_idx * 4 * 16 + vec_idx * 16 + min_tile_k * 16 * 16;
                    lds_offset += (A_lane_head_dim_idx + vec_idx / 2 * 4 + (A_lane_seq_idx % 2) * 8) % 16;

                    // int lds_offset = stage_id * BLOCK_K * BLOCK_M / 2 + A_lane_seq_idx/2 * 64 + A_lane_seq_idx % 2 * 16 + min_tile_k * 32 + vec_idx * 128 + A_lane_head_dim_idx;

                    inline_ds_read_b32_wait(A_lds_v2fp16, lds_offset, A_reg[min_tile_k].f16x2[vec_idx]);
                }
            }
        }

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0); 

        if constexpr (STAGES == 1){
            m_loop++;
        }
        // asm volatile("s_setprio 1");
        #pragma unroll
        for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
            #pragma unroll
            for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
                #pragma unroll
                for(int k_idx=0; k_idx<(BLOCK_K/32); k_idx++) {  //BLOCK_K mini size is 32
                    //min tile is 32*32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
                    #pragma unroll
                    for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                        #pragma unroll
                        for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                            #pragma unroll
                            for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                                if constexpr (std::is_same<Element,Float8_e4m3_t>::value){
                                } else {
                                    //A采用ds_read后对应的mmac
                                    C_reg[m_loop-1][min_tile_n*2 + min_tile_m].f32 = mmac<Element, ElementAccum>(
                                        //BN = 32, BK = 32
                                        // vec4_Element<Element>{A_reg[min_tile_k].f16[0*2 + min_tile_m], A_reg[min_tile_k].f16[1*2 + min_tile_m], A_reg[min_tile_k].f16[2*2 + min_tile_m], A_reg[min_tile_k].f16[3*2 + min_tile_m]},
                                        vec4_Element<Element>{A_reg[min_tile_k].f16x2[0][min_tile_m], A_reg[min_tile_k].f16x2[1][min_tile_m], A_reg[min_tile_k].f16x2[2][min_tile_m], A_reg[min_tile_k].f16x2[3][min_tile_m]},
                                        B_reg[min_tile_k].f16x4[min_tile_n],
                                        C_reg[m_loop-1][min_tile_n*2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
        }

        // asm volatile("s_setprio 0");
        if(STAGES > 1) {
            if constexpr(Is_preload_A || Is_store_A){
                stage_id ++;
            } else {
                stage_id ^=1;
                __builtin_amdgcn_sched_barrier(0);
                // asm volatile("s_waitcnt lgkmcnt(0)");
                __syncthreads();
                __builtin_amdgcn_sched_barrier(0);
            }
        } else {
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __syncthreads();
            __builtin_amdgcn_sched_barrier(0);
        }
    }
    #endif
#endif
}







//                                                                                   K                  BLOCK_K      BLOCK_N      BLOCK_M      BLOCK_K     WARP_N
template<bool Is_preload_A, bool Is_store_A, bool Is_even_MN, int M/*head_dim*/, int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int STAGES, typename Element, typename ElementAccum = float>
__forceinline__ __device__ void  gpu_gemm_B_in_reg_gfx946(
    vec4_uint A_ptr, 
    vec4_uint C_ptr, 
    Element* A_lds, 
    union_vec4_f16x2<Element> B_reg[(WARP_M/32)*(BLOCK_K/32)*2], 
    union_vec4_fp32 C_reg[(M/BLOCK_M)*(WARP_M/32)*(WARP_N/32)][4], 
    int N/*seq_kv*/, 
    int K/*seq_q*/, 
    int warp_id, 
    int seqlen_A_stride) {
    #if 1
    const int WARP_NUM = (BLOCK_M*BLOCK_N)/(WARP_M*WARP_N);
    const int A_lds_load_num = (BLOCK_M*BLOCK_K) / (4*32);
    static_assert(BLOCK_K>=32, "Error: gpu_gemm_B_in_reg gemm BLOCK_K must be equal or greater than 32");
    static_assert(BLOCK_N>=WARP_N, "Error: gpu_gemm_B_in_reg gemm BLOCK_N must be equal or greater than WARP_N");
    static_assert(BLOCK_M==WARP_M, "Error: gpu_gemm_B_in_reg gemm BLOCK_M must be equal to WARP_M");

    union_vec4_f16x2<Element> A_reg[((WARP_M*BLOCK_K)/(32*32))*2];
    //c mini tile is 32*32
    // vec4_fp32 o[(WARP_M/32)*(WARP_N/32)][4]={0};

    // __shared__ Element A_lds[STAGES*BLOCK_N * BLOCK_K];

    //wave size should be defined in launch file. Here use 64 threads

    int lane_id = threadIdx.x & 63; //lane id, 0-63

    int row = lane_id % 4;
    int col = lane_id / 4;

    int stage_id = 0;

    if(STAGES > 1 && (!Is_preload_A)) {
        int m_loop = 0;
        int A_block_buffer_load_global_offset = m_loop * BLOCK_M;
        int A_lds_stage_offset = stage_id * BLOCK_M * BLOCK_K;
        prefetch_to_lds_gfx938<false, BLOCK_M, BLOCK_K, Element, ElementAccum, Is_even_MN>(A_ptr, A_block_buffer_load_global_offset, A_lds + A_lds_stage_offset, seqlen_A_stride, warp_id);
    }

    #if 1   
    // int lds_offset = row * 8 + col * 32;
    for(int m_loop = 1; m_loop<(M/BLOCK_M) + 1; m_loop++) {
        if(STAGES > 1) {
            if constexpr(Is_preload_A || Is_store_A){
                stage_id ++;
            } else {
                stage_id = stage_id ^ 1;
            }
        }

        if(STAGES == 1) {
            m_loop--;
        }

        if((!Is_preload_A)&& m_loop < (M/BLOCK_M)) {
            int A_block_buffer_load_global_offset = m_loop*BLOCK_M;
            int A_lds_stage_offset = (stage_id)*BLOCK_K*BLOCK_M;
            prefetch_to_lds_gfx938<false, BLOCK_M, BLOCK_K, Element, ElementAccum, Is_even_MN>(A_ptr, A_block_buffer_load_global_offset, A_lds + A_lds_stage_offset, seqlen_A_stride, warp_id);
        }

        //BM = 32, BK = 32
        if(warp_id == 0) {
            if(!Is_preload_A){
                if(STAGES > 1 && m_loop < (M/BLOCK_M)) {
                    vmcnt_wait(1);
                } else {
                    vmcnt_wait(0);
                }
            }
        }

        if constexpr (STAGES > 1) {
            if constexpr(Is_preload_A || Is_store_A){
                stage_id --;
            } else {
                stage_id = stage_id ^ 1;
            }
        }
        //lds -> vgpr use ds_read_m; left matrix        

        //由于ds_read方式发生了改变，mmac结果矩阵layout变化，存储的时候，offset要进行修改
        {
            int A_lds_stage_offset = stage_id * BLOCK_K * BLOCK_M;
            // DS_READ_MATRIX_32X32_B16(ds_offset_cast(A_lds + A_lds_stage_offset), A_reg[0].f16, A_reg[1].f16, false);
            if constexpr (std::is_same_v<Element, half_t>) {
                A_reg[0].f16x8 =  __builtin_hcu_ds_read_matrix_format_f16(A_lds + A_lds_stage_offset, 0, 2, 1, 0);
                A_reg[1].f16x8 =  __builtin_hcu_ds_read_matrix_format_f16(A_lds + A_lds_stage_offset, 1024, 2, 1, 0);
            } else {
                A_reg[0].f16x8 =  __builtin_hcu_ds_read_matrix_format_bf16(A_lds + A_lds_stage_offset, 0, 2, 1, 0);
                A_reg[1].f16x8 =  __builtin_hcu_ds_read_matrix_format_bf16(A_lds + A_lds_stage_offset, 1024, 2, 1, 0);
            }
        }

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0); 

        if constexpr (STAGES == 1){
            m_loop++;
        }
        asm volatile("s_setprio 1");
        #pragma unroll
        for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
            #pragma unroll
            for(int m_idx=0; m_idx<(WARP_M/32); m_idx++) {
                #pragma unroll
                for(int k_idx=0; k_idx<(BLOCK_K/32); k_idx++) {  //BLOCK_K mini size is 32
                    //min tile is 32*32, mmac size is 16x16x16,so min_tile_n=32/16, min_tile_m=32/16
                    #pragma unroll
                    for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                        #pragma unroll
                        for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                            #pragma unroll
                            for(int min_tile_k=0; min_tile_k<2; min_tile_k++) {
                                if constexpr (std::is_same<Element,Float8_e4m3_t>::value){
                                } else {
                                    //A采用ds_read后对应的mmac
                                    C_reg[m_loop-1][min_tile_n*2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                                        //BN = 32, BK = 32
                                        // vec4_Element<Element>{A_reg[min_tile_k].f16[0*2 + min_tile_m], A_reg[min_tile_k].f16[1*2 + min_tile_m], A_reg[min_tile_k].f16[2*2 + min_tile_m], A_reg[min_tile_k].f16[3*2 + min_tile_m]},
                                        B_reg[min_tile_k].f16x4[min_tile_n],
                                        A_reg[min_tile_k].f16x4[min_tile_m],
                                        C_reg[m_loop-1][min_tile_n*2 + min_tile_m].f32);
                                }
                            }
                        }
                    }
                }
            }
        }

        // //test only
        // for(int min_tile_n = 0; min_tile_n < 2; ++ min_tile_n) {
        //     for(int min_tile_m = 0; min_tile_m < 2; ++ min_tile_m) {
        //         C_reg[m_loop-1][min_tile_n*2 + min_tile_m].f32[0] = UpCast<Element,float, true>(B_reg[min_tile_m].f16x4[min_tile_n][0]);
        //         C_reg[m_loop-1][min_tile_n*2 + min_tile_m].f32[1] = UpCast<Element,float, true>(B_reg[min_tile_m].f16x4[min_tile_n][1]);
        //         C_reg[m_loop-1][min_tile_n*2 + min_tile_m].f32[2] = UpCast<Element,float, true>(B_reg[min_tile_m].f16x4[min_tile_n][2]);
        //         C_reg[m_loop-1][min_tile_n*2 + min_tile_m].f32[3] = UpCast<Element,float, true>(B_reg[min_tile_m].f16x4[min_tile_n][3]);
        //     }
        // }
        


        asm volatile("s_setprio 0");
        if(STAGES > 1) {
            if constexpr(Is_preload_A || Is_store_A){
                stage_id ++;
            } else {
                stage_id ^=1;
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __syncthreads();
                __builtin_amdgcn_sched_barrier(0);
            }
        } else {
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __syncthreads();
            __builtin_amdgcn_sched_barrier(0);
        }
    }
    #endif
#endif
}