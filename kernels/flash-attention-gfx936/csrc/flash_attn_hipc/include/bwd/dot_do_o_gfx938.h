#pragma once
#include <block_info.h>
#include "utils.h"
#include "prefetch.h"
// Just compute dot(do, o) and write the result (softmax_d) to global memory as a separate kernel.
// This is used in the case where we want to parallelize the backward across seqlen_k.
template<bool Clear_dQaccum=true, bool Is_even_MN, class Element, class ElementAccum,  int kBlockM, int kBlockN, int WARP_M, int WARP_N, int K, int STAGES, bool USE_BSHD_LAYOUT, typename Params>
inline __device__ void compute_dot_do_o_gfx938(const Params &params) {
    Element *do_ptr = static_cast<Element*>(params.do_ptr);
    Element *o_ptr = static_cast<Element*>(params.o_ptr);
    ElementAccum* dsoftmax_sum = static_cast<ElementAccum*>(params.dsoftmax_sum);
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
    int warp_id = 0;

    __shared__ Element dO_lds[kBlockM * kBlockN];
    __shared__ Element O_lds[kBlockM * kBlockN];

    float dP_sum_cur[(kBlockM/16)] = {0.0f};

    const int WARP_NUM = (kBlockM)/(WARP_M);

    const flash::BlockInfo</*Varlen=*/!Is_even_MN, false, USE_BSHD_LAYOUT> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    int seqlen_do_stride = params.do_row_stride;
    int seqlen_o_stride = params.o_row_stride;

    const int row_offset_do = binfo.q_offset1(params.do_batch_stride, params.do_row_stride, bidb) + binfo.q_offset2(params.do_head_stride,bidh) + m_block * kBlockM * seqlen_do_stride;
    const int row_offset_o =  binfo.q_offset1(params.o_batch_stride, params.o_row_stride, bidb)  + binfo.q_offset2(params.o_head_stride,bidh) + m_block * kBlockM * seqlen_o_stride;

    const int row_offset_dpsum = (bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM;

    auto gdO = prepare_for_matrix_load_gfx938<Element>(reinterpret_cast<Element *>(do_ptr) + row_offset_do, seqlen_do_stride);
    auto gO  = prepare_for_matrix_load_gfx938<Element>(reinterpret_cast<Element *>(o_ptr) + row_offset_o, seqlen_o_stride);

    ElementAccum *dP_sum = reinterpret_cast<ElementAccum *>(dsoftmax_sum) + row_offset_dpsum;

    warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
    union_vec4_f16x2<Element> dO_reg[((WARP_M*kBlockN)/(32*32))*2];
    union_vec4_f16x2<Element> O_reg[((WARP_M*kBlockN)/(32*32))*2];


    for(int k_loop=0; k_loop<K/kBlockN; k_loop++) {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);
        int do_block_buffer_load_global_offset = k_loop * kBlockN;
        //read 32 * 128
        prefetch_to_lds_gfx938<true, kBlockM, kBlockN, Element, ElementAccum, Is_even_MN, 1>(gdO, do_block_buffer_load_global_offset, dO_lds, binfo.actual_seqlen_q - m_block * kBlockM, warp_id);
        prefetch_to_lds_gfx938<true, kBlockM, kBlockN, Element, ElementAccum, Is_even_MN, 1>(gO, do_block_buffer_load_global_offset, O_lds, binfo.actual_seqlen_q - m_block * kBlockM, warp_id);
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();

        for(int i = 0; i < kBlockN / 32; ++i) {
            DS_READ_MATRIX_32X32_B16(ds_offset_cast(dO_lds + i * 32 * 32), dO_reg[i * 2 + 0].f16, dO_reg[i * 2 + 1].f16, true);
            DS_READ_MATRIX_32X32_B16(ds_offset_cast(O_lds + i * 32 * 32), O_reg[i * 2 + 0].f16, O_reg[i * 2 + 1].f16, true);
            // if constexpr (std::is_same_v<Element, half_t>) {
            //     dO_reg[i*2 + 0].f16x8 =  __builtin_hcu_ds_read_matrix_trans_format_f16(dO_lds + i * 32 * 32, 0, 2, 1, 0);
            //     dO_reg[i*2 + 1].f16x8 =  __builtin_hcu_ds_read_matrix_trans_format_f16(dO_lds + i * 32 * 32, 1024, 2, 1, 0);
            //     O_reg[i*2 + 0].f16x8  =  __builtin_hcu_ds_read_matrix_trans_format_f16(O_lds + i * 32 * 32,  0, 2, 1, 0);
            //     O_reg[i*2 + 1].f16x8  =  __builtin_hcu_ds_read_matrix_trans_format_f16(O_lds + i * 32 * 32,  1024, 2, 1, 0);
            // } else {
            //     dO_reg[i*2 + 0].f16x8 =  __builtin_hcu_ds_read_matrix_trans_format_bf16(dO_lds + i * 32 * 32, 0, 2, 1, 0);
            //     dO_reg[i*2 + 1].f16x8 =  __builtin_hcu_ds_read_matrix_trans_format_bf16(dO_lds + i * 32 * 32, 1024, 2, 1, 0);
            //     O_reg[i*2 + 0].f16x8  =  __builtin_hcu_ds_read_matrix_trans_format_bf16(O_lds + i * 32 * 32,  0, 2, 1, 0);
            //     O_reg[i*2 + 1].f16x8  =  __builtin_hcu_ds_read_matrix_trans_format_bf16(O_lds + i * 32 * 32,  1024, 2, 1, 0);
            // }
        }
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        #pragma unroll
        for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
            #pragma unroll
            for (int head_dim_idx = 0; head_dim_idx < (kBlockN/32); ++head_dim_idx) {
                #pragma unroll
                for(int vec_id = 0; vec_id<4; vec_id++) {
                    #pragma unroll
                    for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                        if (Is_even_MN || (m_block * kBlockM + min_tile_m*16 + (threadIdx.x & 15)) < binfo.actual_seqlen_q) {
                            dP_sum_cur[min_tile_m] += UpCast<Element,float,false>(dO_reg[head_dim_idx*2 + min_tile_m].f16[vec_id * 2 + min_tile_n]) * UpCast<Element,float,false>(O_reg[head_dim_idx*2 + min_tile_m].f16[vec_id * 2 + min_tile_n]);
                        }
                    }
                }
            }
        }
    }


    #pragma unroll
    for (int mi = 0; mi < (WARP_M/32); ++mi) {
        #pragma unroll
        for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
            flash::SumOp<float> sum_op;
            dP_sum_cur[mi*2 + min_tile_m] = flash::Allreduce<64>::run(dP_sum_cur[mi*2 + min_tile_m], sum_op) * params.p_dropout;
            if ((threadIdx.x >> 4) == 0) {
                dP_sum[mi*32 + min_tile_m * 16 + (threadIdx.x & 15)] = dP_sum_cur[mi*2 + min_tile_m];
            }
        }
    }
}
