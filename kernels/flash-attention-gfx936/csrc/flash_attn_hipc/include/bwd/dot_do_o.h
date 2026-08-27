#pragma once
#include <block_info.h>
#include "utils.h"

// Just compute dot(do, o) and write the result (softmax_d) to global memory as a separate kernel.
// This is used in the case where we want to parallelize the backward across seqlen_k.
template<bool Clear_dQaccum=true, bool Is_even_MN, class Element, class ElementAccumType,  int kBlockM_, int kBlockN_, int WARP_M_, int WARP_N_, int kHeadDim_, int STAGES, bool USE_BSHD_LAYOUT, typename Params>
inline __device__ void compute_dot_do_o(const Params &params) {
    Element *do_ptr = static_cast<Element*>(params.do_ptr);
    Element *o_ptr = static_cast<Element*>(params.o_ptr);
    ElementAccumType* dsoftmax_sum = static_cast<ElementAccumType*>(params.dsoftmax_sum);
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.z;
    // The block index for the head.
    const int bidh = blockIdx.y;
    // The thread index.
    const int tidx = threadIdx.x;
    //wave size should be defined in launch file. Here use 64 threads
    int lane_id = threadIdx.x & 63; //lane id, 0-63

    int warp_id_vec = threadIdx.x / 64; //warp id in a block
    int warp_id =0;

    __shared__ Element do_lds[STAGES*(kBlockM_/32) * (kBlockN_/32)*(32*34)];
    __shared__ Element o_lds[STAGES*(kBlockM_/32) * (kBlockN_/32)*(32*34)];

    float dP_sum_cur[(kBlockM_/16)] = {0.0f};

    int stage_id = 0;

    constexpr int kBlockM = kBlockM_;
    constexpr int kBlockN = kBlockN_;
    constexpr int kHeadDim = kHeadDim_;
    const int WARP_NUM = (kBlockM_)/(WARP_M_);

    const flash::BlockInfo</*Varlen=*/!Is_even_MN, false, USE_BSHD_LAYOUT> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    int seqlen_do_stride = params.do_row_stride;
    int seqlen_o_stride = params.o_row_stride;

    const int row_offset_do = binfo.q_offset1(params.do_batch_stride, params.do_row_stride, bidb) + binfo.q_offset2(params.do_head_stride,bidh) + m_block * kBlockM * seqlen_do_stride;
    const int row_offset_o =  binfo.q_offset1(params.o_batch_stride, params.o_row_stride, bidb)  + binfo.q_offset2(params.o_head_stride,bidh) + m_block * kBlockM * seqlen_o_stride;

    const int row_offset_dpsum = (bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM;

    // Element *gdO = reinterpret_cast<Element *>(do_ptr) + row_offset_do;
    auto gdO = tcp_cache_swizzle_func<kHeadDim_, Element>(reinterpret_cast<Element *>(do_ptr) + row_offset_do);
    // Element *gO = reinterpret_cast<Element *>(o_ptr) + row_offset_o;
    auto gO = tcp_cache_swizzle_func<kHeadDim_, Element>(reinterpret_cast<Element *>(o_ptr) + row_offset_o);
    ElementAccumType *dP_sum = reinterpret_cast<ElementAccumType *>(dsoftmax_sum) + row_offset_dpsum;

    asm volatile("v_readfirstlane_b32 %0,%1"
                : "=s"(warp_id)
                : "v"(warp_id_vec)
                :);
    vec2_Element<Element> do_reg[(kHeadDim_/kBlockN_)*((WARP_M_*kBlockN_)/(32*32))*2][4];  //ds_read mini size is 32*32,2 is seq, 4 is head dim
    vec2_Element<Element> o_reg[(kHeadDim_/kBlockN_)*((WARP_M_*kBlockN_)/(32*32))*2][4];  //ds_read mini size is 32*32,2 is seq, 4 is head dim


    // int A_lane_m_idx =  (lane_id >> 4);
    int do_lane_m_idx = ((lane_id >> 4) & 1)*2 + ((lane_id >> 4) >> 1); //(0, 1, 2, 3) --> (0, 2, 1, 3)
    int do_lane_head_dim_idx = (lane_id & 15);
    //global->lds, left matrix

    // printf("kBlockN_==%d, kHeadDim_=%d, WARP_M_=%d\n",kBlockN_, kHeadDim_, WARP_M_);

    for(int k_loop=0; k_loop<kHeadDim_/kBlockN_; k_loop++) {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);
        int do_block_buffer_load_global_offset = k_loop * kBlockN_;
        // do_ptr buffer load mini size is 4*32, (kBlockM_ * kBlockN_) mini size is (32*32)
        const int do_lds_load_num = (kBlockM_ * kBlockN_) / (4*32);
        int do_lds_stage_offset = stage_id * (kBlockM_/32) * (kBlockN_/32)*(32*34);

        for(int warp_loop=warp_id; warp_loop < do_lds_load_num; warp_loop+=WARP_NUM) {
            int padding = (warp_loop & 7)*2; // padding size in shared memory per buffer load, to avoid bank conflict
            int do_warp_buffer_load_m_id = (warp_loop & (kBlockM_/4 - 1)); //这样子对L1和utlc1有啥影响呢？
            int do_warp_buffer_load_k_id = (warp_loop / (kBlockM_/4));
            int do_warp_buffer_load_lds_offset     =  do_lds_stage_offset + (do_warp_buffer_load_k_id * kBlockM_ * 34) + ((do_warp_buffer_load_m_id >> 3)*(32*34) + (do_warp_buffer_load_m_id & 7)*(4*32)) ;
            int do_warp_buffer_load_global_offset  =  (do_warp_buffer_load_k_id * 32);

            int gsOffset   = (do_block_buffer_load_global_offset + do_warp_buffer_load_global_offset)/2 ;
            // int gvOffset   = (do_lane_m_idx * kHeadDim_)/2  +  do_lane_head_dim_idx;
            int lds_offset = (do_warp_buffer_load_lds_offset + padding)/2;
            {
                int gvOffset;
                if constexpr (!Is_even_MN) {
                    gvOffset = (min((do_lane_m_idx + (do_warp_buffer_load_m_id * 4)),binfo.actual_seqlen_q - m_block * kBlockM - 1) * seqlen_do_stride)/2  +  do_lane_head_dim_idx;
                } else {
                    gvOffset = ((do_lane_m_idx + (do_warp_buffer_load_m_id * 4)) * seqlen_do_stride)/2  +  do_lane_head_dim_idx; 
                }
                builtin_buffer_load_dword_lds(do_lds, gdO, lds_offset, gsOffset, gvOffset);
            }
            {
                int gvOffset;
                if constexpr (!Is_even_MN) {
                    gvOffset = (min((do_lane_m_idx + (do_warp_buffer_load_m_id * 4)),binfo.actual_seqlen_q - m_block * kBlockM - 1) * seqlen_o_stride)/2  +  do_lane_head_dim_idx;
                } else {
                    gvOffset = ((do_lane_m_idx + (do_warp_buffer_load_m_id * 4)) * seqlen_o_stride)/2  +  do_lane_head_dim_idx; 
                }
                builtin_buffer_load_dword_lds(o_lds, gO, lds_offset, gsOffset, gvOffset);
            }
        }

        vmcnt_wait(0);

        // By right we need to scale dP up by 1/params.p_dropout, but instead we don't and only scale the final
        // results (dQ and dK) by 1/params.p_dropout. So we need to keep dP_sum scaled down by params.p_dropout here,
        // so that (dP - dP_sum) is on the same scale.


        {
            //lds -> vgpr use ds_read_m; left matrix
            int do_warp_m_id = (warp_id & ((kBlockM_/WARP_M_) - 1));
            
            int do_lds_stage_offset = stage_id * (kBlockM_/32) * (kBlockN_/32)*(32*17);

            vec2_Element<Element> *do_lds_v2fp16 = (vec2_Element<Element> *)(do_lds);
            vec2_Element<Element> *o_lds_v2fp16 = (vec2_Element<Element> *)(o_lds);

            #pragma unroll
            for(int head_dim_idx=0; head_dim_idx<(kBlockN_/32); head_dim_idx++) { //32 half in col direction
                #pragma unroll
                for(int m_idx=0; m_idx<(WARP_M_/32); m_idx++) {
                    //a warp load min size is (row, col) = (32,16) float
                    #pragma unroll
                    for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {       //sequence direction
                        #pragma unroll
                        for(int vec_id=0; vec_id<4; vec_id++) {   //head dim direction
                            int lds_offset = do_lds_stage_offset + head_dim_idx*kBlockM_*17 + (warp_id*(WARP_M_/32) + m_idx)*(32*17) + vec_id*2 + min_tile_m*32 + (lane_id & 1)*16 + ((lane_id & 15)>>1)*64 + /*padding*/ ((lane_id & 15)>>1) + ((lane_id/16) &1)*8 + (lane_id/32);
                            inline_ds_read_b32_wait(do_lds_v2fp16, lds_offset, do_reg[/*(k_loop)*((WARP_M_*kBlockN_)/(32*32))*2 +*/ (head_dim_idx*(WARP_M_/32) + m_idx)*2 + min_tile_m][vec_id]);
                        }
                    }
                }
            }

            #pragma unroll
            for(int head_dim_idx=0; head_dim_idx<(kBlockN_/32); head_dim_idx++) { //32 half in col direction
                #pragma unroll
                for(int m_idx=0; m_idx<(WARP_M_/32); m_idx++) {
                    //a warp load min size is (row, col) = (32,16) float
                    #pragma unroll
                    for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {       //sequence direction
                        #pragma unroll
                        for(int vec_id=0; vec_id<4; vec_id++) {   //head dim direction
                            int lds_offset = do_lds_stage_offset + head_dim_idx*kBlockM_*17 + (warp_id*(WARP_M_/32) + m_idx)*(32*17) + vec_id*2 + min_tile_m*32 + (lane_id & 1)*16 + ((lane_id & 15)>>1)*64 + /*padding*/ ((lane_id & 15)>>1) + ((lane_id/16) &1)*8 + (lane_id/32);
                            inline_ds_read_b32_wait(o_lds_v2fp16, lds_offset, o_reg[/*(k_loop)*((WARP_M_*kBlockN_)/(32*32))*2 +*/ (head_dim_idx*(WARP_M_/32) + m_idx)*2 + min_tile_m][vec_id]);
                        }
                    }
                }
            }
        }

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        #pragma unroll
        for (int mi = 0; mi < (WARP_M_/32); ++mi) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                #pragma unroll
                for (int head_dim_idx = 0; head_dim_idx < (kBlockN/32); ++head_dim_idx) {
                    #pragma unroll
                    for(int vec_id = 0; vec_id<4; vec_id++) {
                        #pragma unroll
                        for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                            if (Is_even_MN || (m_block * kBlockM + mi*32 + min_tile_m + (threadIdx.x & 15)*2) < binfo.actual_seqlen_q) {
                                dP_sum_cur[mi*2 + min_tile_m] += UpCast<Element,float,true>(do_reg[(head_dim_idx*(WARP_M_/32) + mi)*2 + min_tile_m][vec_id][min_tile_n]) * UpCast<Element,float,true>(o_reg[(head_dim_idx*(WARP_M_/32) + mi)*2 + min_tile_m][vec_id][min_tile_n]);
                            }
                        }
                    }
                }
            }
        }
    }


    #pragma unroll
    for (int mi = 0; mi < (WARP_M_/32); ++mi) {
        #pragma unroll
        for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
            flash::SumOp<float> sum_op;
            dP_sum_cur[mi*2 + min_tile_m] = flash::Allreduce<64>::run(dP_sum_cur[mi*2 + min_tile_m], sum_op) * params.p_dropout;
            if ((threadIdx.x >> 4) == 0) {
                dP_sum[mi*32 + min_tile_m + (threadIdx.x & 15)*2] = dP_sum_cur[mi*2 + min_tile_m];
            }
        }
    }
}