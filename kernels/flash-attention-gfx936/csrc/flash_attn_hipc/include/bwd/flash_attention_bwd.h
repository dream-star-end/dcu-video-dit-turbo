#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <fstream>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "assert.h" 
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"

#include "flash.h"
#include "utils.h"
#include "wait.h"
#include "../numeric_types.h"
#include "philox.cuh"
#include "softmax_tiling.h"
#include "gpu_gemm_nn.h"
#include "gpu_gemm_tt.h"
#include "intrinsic.h"
#include "intrinsic_mls_ds.h"
#include "static_switch.h"
#include "dot_do_o.h"
#include "dot_do_o_gfx938.h"
#include "dot_do_o_gfx946.h"
#include "prefetch.h"
#include "flash_singleton.h"
#include "flash_attention_dv_dk_bwd.h"
#include "flash_attention_dv_dk_bwd_gfx938.h"
#include "flash_attention_dv_dk_bwd_gfx946.h"
#include "flash_attention_dq_bwd.h"
#include "flash_attention_dq_bwd_gfx938.h"
#include "flash_attention_dq_bwd_gfx946.h"

using std::make_shared;
using std::shared_ptr;


template <int kBlockM_, int kBlockN_, int WARP_M_, int WARP_N_, typename Element>
inline __device__ void reshape(Element* smem, vec4_Element<Element> ds_reg_fp16[(WARP_N_/32)*(WARP_M_/32)][4], int warp_id) {
    int lane_id = threadIdx.x & 63; //lane id, 0-63
    
    #pragma unroll
    for(int m_idx=0; m_idx<(WARP_M_/32); m_idx++) {
        #pragma unroll
        for(int n_idx=0; n_idx<(WARP_N_/32); n_idx++) {
            #pragma unroll
            for(int min_tile_m = 0; min_tile_m<2; min_tile_m++) {
                #pragma unroll
                for(int min_tile_n = 0; min_tile_n<2; min_tile_n++) {
                    #pragma unroll
                    for(int vec_idx=0; vec_idx<4; vec_idx++) {
                        int lds_offset = warp_id*(WARP_N_/32)*33*kBlockM_ + n_idx*33*kBlockM_ + m_idx*32*33 +  min_tile_m*16*33 + vec_idx*4*33 + (lane_id>>4)*33 + min_tile_n*16 + (lane_id&15);
                        Element ds_reg_tmp = ds_reg_fp16[(WARP_N_/32)*m_idx + n_idx][min_tile_m*2 + min_tile_n][vec_idx];
                        {                            
                             smem[lds_offset] = ds_reg_fp16[(WARP_N_/32)*m_idx + n_idx][min_tile_m*2 + min_tile_n][vec_idx];
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for(int m_idx=0; m_idx<(kBlockM_/32); m_idx++) {
        #pragma unroll    
        for(int n_idx=0; n_idx<(WARP_N_/32); n_idx++) {
            #pragma unroll        
            for(int min_tile_m = 0; min_tile_m<2; min_tile_m++) {
                #pragma unroll            
                for(int min_tile_n = 0; min_tile_n<2; min_tile_n++) {
                    #pragma unroll                
                    for(int vec_idx=0; vec_idx<4; vec_idx++) {
                        int lds_offset = warp_id*33*kBlockM_ + m_idx*32*33 +  min_tile_m*16 + vec_idx*4 + (lane_id>>4) + min_tile_n*16*33 + (lane_id&15)*33;                        
                        ds_reg_fp16[(WARP_N_/32)*m_idx + n_idx][min_tile_m*2 + min_tile_n][vec_idx] = smem[lds_offset];
                    }
                }
            }
        }
    }
}


/*
 * q_ptr: Transposed 32x16 matrix
 * k_ptr: Non-transposed 32x16 matrix
 * qk_ptr: Non-transposed 32x32 matrixseqlen_q
 */




template<class DataType>
int check_param(int seqlen_q, int seqlen_k, int K, int kBlockM_, int kBlockN_, int kBlockK_, int WARP_M_, int WARP_N_, dim3 blockDim, dim3 gridDim, int maxBlockThreads, int STAGES) {
    // min warp size is 32x32
    if(WARP_M_<32 || WARP_N_<32) {
        std::cout<<"Error, WARP_M_<32 or WARP_N_<32!"<<std::endl;
        assert(((WARP_M_>=32) && (WARP_N_>=32)));
    }
    // check block threads number
    const int blockThreads = ((kBlockM_*kBlockN_)/(WARP_M_*WARP_N_)*64);
    if(blockThreads > maxBlockThreads) {
        std::cout<<"Error,Block threads is greater than maxBlockThreads! "<<std::endl;
        assert(blockThreads <= maxBlockThreads);
    }

    //check lds data numbers
    int DataTypeSize = sizeof(DataType);
    const int q_lds_size = STAGES * kBlockM_ * kBlockK_ * DataTypeSize;
    const int k_lds_size = STAGES * kBlockN_ * kBlockK_ * DataTypeSize;
    if(((q_lds_size + k_lds_size)/1024) > 64) {
        std::cout<<"Error, shared memory size is greater than 64KB"<<std::endl;
        assert(((q_lds_size + k_lds_size)/1024) <= 64);  //BW lds 64KB
    }
}
