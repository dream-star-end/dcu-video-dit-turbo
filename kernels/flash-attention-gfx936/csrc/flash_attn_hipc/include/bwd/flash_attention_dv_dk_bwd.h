#define print_kq(block_id_m, bidb, bidh) {                                                                                                \
    __builtin_amdgcn_sched_barrier(0);\
    __builtin_amdgcn_s_waitcnt(0);\
    __syncthreads();\
    __builtin_amdgcn_sched_barrier(0);\
    int qk_warp_m_id = (warp_id / (kBlockN_/WARP_N_));                                                            \
    int qk_warp_n_id = (warp_id & (kBlockN_/WARP_N_ - 1));                                                                \
    int qk_global_offset = bidb*(params.h* params.seqlen_q * params.seqlen_k) + bidh * (params.seqlen_q * params.seqlen_k) + n_block*kBlockN_ \
                         + block_id_m*kBlockM_*params.seqlen_k + qk_warp_n_id*WARP_N_ + qk_warp_m_id*WARP_M_*params.seqlen_k; \
    for(int n_idx=0; n_idx<(WARP_N_/32); n_idx++) {                                                               \
        for(int m_block_idx=0; m_block_idx<kBlockM_/WARP_M_; m_block_idx++) {                                        \
        for(int m_idx=0; m_idx<(WARP_M_/32); m_idx++) {                                                              \
            for(int reg_id=0; reg_id<4; reg_id++) {                                                                 \
            for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {                                                     \
                for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {                                                 \
                        if(((n_block*kBlockN_ + qk_warp_n_id*WARP_N_ + n_idx * 32 + ((lane_id & 15) << 1) + min_tile_n) < params.seqlen_k) &&                        \
                           ((block_id_m*kBlockM_ +  qk_warp_m_id*WARP_M_ + m_idx*32 + reg_id * 8 + min_tile_m + ((lane_id / 16) * 2)) < params.seqlen_q))    {           \
                            int offset = qk_global_offset + n_idx * 32 + m_block_idx * WARP_M_ * params.seqlen_k + m_idx*32*params.seqlen_k+ ((lane_id & 15) << 1)  + min_tile_m*params.seqlen_k + ((lane_id / 16) * 2) *params.seqlen_k + min_tile_n ; \
                            kq_ptr[offset + reg_id * 8 *params.seqlen_k] = 0;(s_reg[m_block_idx * (WARP_M_/32) *(WARP_N_/32)+ m_idx*(WARP_N_/32) + n_idx][min_tile_m*2 + min_tile_n].f32[reg_id]); \
                        }                                                                                           \
                    }                                                                                               \
                }                                                                                                   \
            }                                                                                                       \
        }                                                                                                           \
        }                                                                                                           \
    }                                                                                                               \
    __builtin_amdgcn_sched_barrier(0);\
    __builtin_amdgcn_s_waitcnt(0);\
    __syncthreads();\
    __builtin_amdgcn_sched_barrier(0);\
}

// #define print_kq(block_id_m, bidb, bidh) {                                                                                                \
//     __builtin_amdgcn_sched_barrier(0);\
//     __builtin_amdgcn_s_waitcnt(0);\
//     __syncthreads();\
//     __builtin_amdgcn_sched_barrier(0);\
//     int qk_warp_m_id = (warp_id / (kBlockN_/WARP_N_));                                                            \
//     int qk_warp_n_id = (warp_id & (kBlockN_/WARP_N_ - 1));                                                                \
//     int qk_global_offset = bidb*(params.h* params.seqlen_q * params.seqlen_k) + bidh * (params.seqlen_q * params.seqlen_k) + n_block*kBlockN_ \
//                          + block_id_m*kBlockM_*params.seqlen_k + qk_warp_n_id*WARP_N_ + qk_warp_m_id*WARP_M_*params.seqlen_k; \
//     for(int n_idx=0; n_idx<(WARP_N_/32); n_idx++) {                                                               \
//         for(int m_block_idx=0; m_block_idx<kBlockM_/WARP_M_; m_block_idx++) {                                        \
//         for(int m_idx=0; m_idx<(WARP_M_/32); m_idx++) {                                                              \
//             for(int reg_id=0; reg_id<4; reg_id++) {                                                                 \
//             for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {                                                     \
//                 for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {                                                 \
//                         if(((n_block*kBlockN_ + qk_warp_n_id*WARP_N_ + n_idx * 32 + (lane_id & 15) + min_tile_n*16) < params.seqlen_k) &&                        \
//                             ((block_id_m*kBlockM_ +  qk_warp_m_id*WARP_M_ + m_idx*32 + reg_id + min_tile_m*16 + ((lane_id / 16) * 4)) < params.seqlen_q))    {           \
//                             int offset = qk_global_offset + n_idx * 32 + m_block_idx * WARP_M_ * params.seqlen_k + m_idx*32*params.seqlen_k+ (lane_id & 15)  + min_tile_m*params.seqlen_k*16 + ((lane_id / 16) * 4) *params.seqlen_k + min_tile_n*16 ; \
//                             kq_ptr[offset + reg_id *params.seqlen_k] = 0;(s_reg[m_block_idx * (WARP_M_/32) *(WARP_N_/32)+ m_idx*(WARP_N_/32) + n_idx][min_tile_m*2 + min_tile_n].f32[reg_id]); \
//                         }                                                                                           \
//                     }                                                                                               \
//                 }                                                                                                   \
//             }                                                                                                       \
//         }                                                                                                           \
//         }                                                                                                           \
//     }                                                                                                               \
//     __builtin_amdgcn_sched_barrier(0);\
//     __builtin_amdgcn_s_waitcnt(0);\
//     __syncthreads();\
//     __builtin_amdgcn_sched_barrier(0);\
// }

#define print_softmax_rescale_o(block_id_m, bidb, bidh)    {                                                        \
    __builtin_amdgcn_sched_barrier(0);\
    __builtin_amdgcn_s_waitcnt(0);\
    __syncthreads();\
    __builtin_amdgcn_sched_barrier(0);\
    int s_warp_m_id = (warp_id / (kBlockN_/WARP_N_));                                                             \
    int s_warp_n_id = (warp_id & (kBlockN_/WARP_N_ - 1));                                                                 \
    int s_global_offset = bidb*(params.h * params.seqlen_q * params.seqlen_k) + bidh * (params.seqlen_q * params.seqlen_k) + n_block*kBlockN_ + block_id_m*kBlockM_*params.seqlen_k + s_warp_n_id*WARP_N_ + s_warp_m_id*WARP_M_*params.seqlen_k;    \
    for(int n_idx=0; n_idx<(WARP_N_/32); n_idx++) {                                                                  \
        for(int m_block_idx=0; m_block_idx<kBlockM_/WARP_M_; m_block_idx++) {                                        \
        for(int m_idx=0; m_idx<(WARP_M_/32); m_idx++) {                                                              \
            for(int reg_id=0; reg_id<4; reg_id++) {                                                                 \
                for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {                                                     \
                    for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {                                                 \
                        if(((n_block*kBlockN_ + s_warp_n_id*WARP_N_ + n_idx * 32 + ((lane_id & 15) << 1) + min_tile_n) < params.seqlen_k) &&                        \
                           ((block_id_m*kBlockM_ +  s_warp_m_id*WARP_M_ + m_idx*32 + reg_id * 8 + min_tile_m + ((lane_id / 16) * 2)) < params.seqlen_q))    {           \
                            int offset = s_global_offset + n_idx * 32  + m_block_idx * WARP_M_ * params.seqlen_k + m_idx*32*params.seqlen_k+ ((lane_id & 15) << 1) + min_tile_m*params.seqlen_k + ((lane_id / 16) * 2)*params.seqlen_k + min_tile_n ;\
                            s_ptr[offset + reg_id * 8 * params.seqlen_k] = (s_reg[m_block_idx * (WARP_M_/32) *(WARP_N_/32)+ m_idx*(WARP_N_/32) + n_idx][min_tile_n + min_tile_m*2].f32[reg_id]);\
                        }                                                                                           \
                    }                                                                                               \
                }                                                                                                   \
            }                                                                                                       \
        }                                                                                                           \
        }                                                                                                           \
    }                                                                                                               \
    __builtin_amdgcn_sched_barrier(0);\
    __builtin_amdgcn_s_waitcnt(0);\
    __syncthreads();\
    __builtin_amdgcn_sched_barrier(0);\
}

#define print_ds(block_id_m, bidb, bidh)    {                                                        \
    __builtin_amdgcn_sched_barrier(0);\
    __builtin_amdgcn_s_waitcnt(0);\
    __syncthreads();\
    __builtin_amdgcn_sched_barrier(0);\
    int ds_warp_m_id = (warp_id / (kBlockN_/WARP_N_));                                                             \
    int ds_warp_n_id = (warp_id & (kBlockN_/WARP_N_ - 1));                                                                 \
    int ds_global_offset = bidb*(params.h * params.seqlen_q * params.seqlen_k) + bidh * (params.seqlen_q * params.seqlen_k) + n_block*kBlockN_ + block_id_m*kBlockM_*params.seqlen_k + ds_warp_n_id*WARP_N_ + ds_warp_m_id*WARP_M_*params.seqlen_k;    \
    for(int n_idx=0; n_idx<(WARP_N_/32); n_idx++) {                                                                  \
        for(int m_block_idx=0; m_block_idx<kBlockM_/WARP_M_; m_block_idx++) {                                        \
        for(int m_idx=0; m_idx<(WARP_M_/32); m_idx++) {                                                              \
            for(int reg_id=0; reg_id<4; reg_id++) {                                                                 \
                for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {                                                     \
                    for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {                                                 \
                        if(((n_block*kBlockN_ + ds_warp_n_id*WARP_N_ + n_idx * 32 + ((lane_id & 15) << 1) + min_tile_n) < params.seqlen_k) &&                        \
                           ((block_id_m*kBlockM_ +  ds_warp_m_id*WARP_M_ + m_idx*32 + reg_id * 8 + min_tile_m + ((lane_id / 16) * 2)) < params.seqlen_q))    {           \
                            int offset = ds_global_offset + n_idx * 32  + m_block_idx * WARP_M_ * params.seqlen_k + m_idx*32*params.seqlen_k+ ((lane_id & 15) << 1) + min_tile_m*params.seqlen_k + ((lane_id / 16) * 2)*params.seqlen_k + min_tile_n ;\
                            ds_ptr[offset + reg_id * 8 * params.seqlen_k] = (dS_reg[m_block_idx * (WARP_M_/32) *(WARP_N_/32)+ m_idx*(WARP_N_/32) + n_idx][min_tile_n + min_tile_m*2].f32[reg_id]);\
                        }                                                                                           \
                    }                                                                                               \
                }                                                                                                   \
            }                                                                                                       \
        }                                                                                                           \
        }                                                                                                           \
    }                                                                                                               \
    __builtin_amdgcn_sched_barrier(0);\
    __builtin_amdgcn_s_waitcnt(0);\
    __syncthreads();\
    __builtin_amdgcn_sched_barrier(0);\
}

#define print_ds_fp16(block_id_m, bidb, bidh)    {                                                        \
    __builtin_amdgcn_sched_barrier(0);\
    __builtin_amdgcn_s_waitcnt(0);\
    __syncthreads();\
    __builtin_amdgcn_sched_barrier(0);\
    int ds_warp_m_id = (warp_id / (kBlockN_/WARP_N_));                                                             \
    int ds_warp_n_id = (warp_id & (kBlockN_/WARP_N_ - 1));                                                                 \
    int ds_global_offset = bidb*(params.h * params.seqlen_q * params.seqlen_k) + bidh * (params.seqlen_q * params.seqlen_k) + n_block*kBlockN_ + block_id_m*kBlockM_*params.seqlen_k + ds_warp_n_id*WARP_N_ + ds_warp_m_id*WARP_M_*params.seqlen_k;    \
    for(int n_idx=0; n_idx<(WARP_N_/32); n_idx++) {                                                                  \
        for(int m_block_idx=0; m_block_idx<kBlockM_/WARP_M_; m_block_idx++) {                                        \
        for(int m_idx=0; m_idx<(WARP_M_/32); m_idx++) {                                                              \
            for(int reg_id=0; reg_id<4; reg_id++) {                                                                 \
                for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {                                                     \
                    for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {                                                 \
                        if(((n_block*kBlockN_ + ds_warp_n_id*WARP_N_ + n_idx * 32 + ((lane_id & 15) << 1) + min_tile_n) < params.seqlen_k) &&                        \
                           ((block_id_m*kBlockM_ +  ds_warp_m_id*WARP_M_ + m_idx*32 + reg_id * 8 + min_tile_m + ((lane_id / 16) * 2)) < params.seqlen_q))    {           \
                            int offset = ds_global_offset + n_idx * 32  + m_block_idx * WARP_M_ * params.seqlen_k + m_idx*32*params.seqlen_k+ ((lane_id & 15) << 1) + min_tile_m*params.seqlen_k + ((lane_id / 16) * 2)*params.seqlen_k + min_tile_n ;\
                            ds_ptr[offset + reg_id * 8 * params.seqlen_k] = UpCast<Element,float>(dS_reg_fp16[m_block_idx * (WARP_M_/32) *(WARP_N_/32)+ m_idx*(WARP_N_/32) + n_idx][min_tile_n + min_tile_m*2].f16[reg_id]);\
                        }                                                                                           \
                    }                                                                                               \
                }                                                                                                   \
            }                                                                                                       \
        }                                                                                                           \
        }                                                                                                           \
    }                                                                                                               \
    __builtin_amdgcn_sched_barrier(0);\
    __builtin_amdgcn_s_waitcnt(0);\
    __syncthreads();\
    __builtin_amdgcn_sched_barrier(0);\
}

#define print_dp(block_id_m, bidb, bidh)    {                                                                         \
    __builtin_amdgcn_sched_barrier(0);\
    __builtin_amdgcn_s_waitcnt(0);\
    __syncthreads();\
    __builtin_amdgcn_sched_barrier(0);\
    int dp_warp_m_id = (warp_id / (kBlockN_/WARP_N_));                                                             \
    int dp_warp_n_id = (warp_id & (kBlockN_/WARP_N_ - 1));                                                                 \
    int dp_global_offset = bidb*(params.h * params.seqlen_q * params.seqlen_k) + bidh * (params.seqlen_q * params.seqlen_k) + n_block*kBlockN_ + block_id_m*kBlockM_*params.seqlen_k + dp_warp_n_id*WARP_N_ + dp_warp_m_id*WARP_M_*params.seqlen_k;    \
    for(int n_idx=0; n_idx<(WARP_N_/32); n_idx++) {\
        for(int m_block_idx=0; m_block_idx<kBlockM_/WARP_M_; m_block_idx++) {                                                                    \
        for(int m_idx=0; m_idx<(WARP_M_/32); m_idx++) {                                                              \
            for(int reg_id=0; reg_id<4; reg_id++) {                                                                 \
            for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {                                                     \
                for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {                                                 \
                        if(((n_block*kBlockN_ + dp_warp_n_id*WARP_N_ + n_idx * 32 + ((lane_id & 15) << 1) + min_tile_n) < params.seqlen_k) &&                        \
                           ((block_id_m*kBlockM_ +  dp_warp_m_id*WARP_M_ + m_idx*32 + reg_id * 8 + min_tile_m + ((lane_id / 16) * 2)) < params.seqlen_q))    {           \
                            int offset = dp_global_offset + n_idx * 32 + m_block_idx * WARP_M_ * params.seqlen_k + m_idx*32*params.seqlen_k+ ((lane_id & 15) << 1) + min_tile_m*params.seqlen_k + ((lane_id / 16) * 2)*params.seqlen_k + min_tile_n ;\
                            dp_ptr[offset + reg_id * 8 * params.seqlen_k] = (dp_reg[m_block_idx * (WARP_M_/32) *(WARP_N_/32) + m_idx*(WARP_N_/32) + n_idx][min_tile_n + min_tile_m*2].f32[reg_id]);\
                        }                                                                                           \
                    }                                                                                               \
                }                                                                                                   \
            }                                                                                                       \
        }                                                                                                           \
        }\
    }                                                                                                               \
    __builtin_amdgcn_sched_barrier(0);\
    __builtin_amdgcn_s_waitcnt(0);\
    __syncthreads();\
    __builtin_amdgcn_sched_barrier(0);\
}

template<class Element, class ElementAccumType, bool Is_dropout, bool Is_causal , bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_first, bool Is_last, bool Seq_parallel=false,  int kBlockM_, int kBlockN_, int K, int K_v, int kBlockK_, int WARP_M_, int WARP_N_, bool USE_BSHD_LAYOUT, typename Params>
__forceinline__ __device__ void compute_dk_dv_1colblock(Params &params, int bidb, int bidh, int n_block
    ) {
#ifdef DEBUGING
    ElementAccumType * kq_ptr = static_cast<ElementAccumType*>(params.kq_ptr);
    ElementAccumType * s_ptr = static_cast<ElementAccumType*>(params.s_ptr);
    ElementAccumType * dp_ptr = static_cast<ElementAccumType*>(params.dp_ptr);
    ElementAccumType * ds_ptr = static_cast<ElementAccumType*>(params.ds_ptr);
#endif
    Element* q_ptr = static_cast<Element*>(params.q_ptr);
    Element* k_ptr = static_cast<Element*>(params.k_ptr);
    Element* v_ptr = static_cast<Element*>(params.v_ptr);
    Element* o_ptr = static_cast<Element*>(params.o_ptr);
    Element* p_ptr = static_cast<Element*>(params.p_ptr);
    // Element* dq_ptr = static_cast<Element*>(params.dq_ptr);
    Element* dk_ptr = static_cast<Element*>(params.dk_ptr);
    Element* dv_ptr = static_cast<Element*>(params.dv_ptr);
    Element* do_ptr = static_cast<Element*>(params.do_ptr);
    ElementAccumType* softmax_lse_ptr = static_cast<ElementAccumType*>(params.softmax_lse_ptr);
    ElementAccumType* dsoftmax_sum = static_cast<ElementAccumType*>(params.dsoftmax_sum);
    //flash-attention QK, kBlockN_==WARP_N_;
    // static_assert(kBlockM_=WARP_M_,"Error: kBlockM_ not equal WARP_M_!");
    const int WARP_NUM = (kBlockM_*kBlockN_)/(WARP_M_*WARP_N_);
    const int M_BLOCK_NUM = params.seqlen_q/kBlockM_;
    const int N_BLOCK_NUM = params.seqlen_k/kBlockN_;

    extern __shared__ Element smem[];

    int K_lds_ratio;

    const int K_prefetch_level = 3;

    const int STAGES = 2;

    const bool Is_store_Q = true;
    const bool Is_store_dO = true;
    const bool Is_preload_Q = true;
    const bool Is_preload_dO = true;

    const int dP_dO_prefetch_level = Is_store_dO ? 1 : 0;
    const int Q_prefetech_level = Is_preload_Q ? 1 : 0;
    if constexpr (K_prefetch_level == 2){
        K_lds_ratio = (K / kBlockK_) / 2;
    } else {
        K_lds_ratio = (K_prefetch_level == 3) ? 0 : STAGES;
    }

    // Element* K_lds = (Element*)&(smem);
    // Element* Q_lds = K_lds + (kBlockN_/32) * (kBlockK_/32)*(32*34) * K_lds_ratio;
    // Element* V_lds = K_prefetch_level == 2 ? Q_lds : K_lds;
    // Element* dO_lds = Q_lds;

    Element* K_lds = (Element*)&(smem);
    Element* dO_lds = K_lds + (kBlockN_/32) * (kBlockK_/32)*(32*34) * K_lds_ratio;
    Element* V_lds = K_prefetch_level == 2 ? dO_lds : K_lds;
    Element* Q_lds = Is_store_Q ? dO_lds + (kBlockM_/32) * (K_v/32)*(32*34) : dO_lds;

#if 0//defined(__gfx936__)
        auto pointwise_mult = [](vec2_fp32 p, vec2_fp32 dp, vec2_fp32 d) {
            auto d0 = (!Is_dropout || p[0] >= 0 ? dp[0] - d[0] : d[0]);
            auto d1 = (!Is_dropout || p[1] >= 0 ? dp[1] - d[1] : d[1]);
            // return vec2_fp32{p[0]*d0,p[1]*d1};
            // return __builtin_hcu_pk_mul_f32(p, vec2_fp32{d0, d1});
            return __builtin_hcu_v_pk_mul_f32(p, vec2_fp32{d0, d1});
        };
#else
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
#endif

    int tidx    = threadIdx.x;
    int lane_id = threadIdx.x & 63; //lane id, 0-63

    const flash::BlockInfo</*Varlen=*/!Is_even_MN, false, USE_BSHD_LAYOUT> binfo(params, bidb);
    if (n_block * kBlockN_ >= binfo.actual_seqlen_k || binfo.actual_seqlen_q == 0) return;


    const int seqlen_k_minus_q = binfo.actual_seqlen_k - binfo.actual_seqlen_q;
    const int m_block_count = ceil_div(binfo.actual_seqlen_q, kBlockM_);
    const int m_block_min = (!Is_causal && !Is_local) ? 0
        : std::max(0, (n_block * kBlockN_ - seqlen_k_minus_q - params.window_size_right) / kBlockM_);
    const int m_block_max_unclamped = !Is_local ? m_block_count
        : std::min(m_block_count, ceil_div((n_block + 1) * kBlockN_ - seqlen_k_minus_q + params.window_size_left, kBlockM_));
    const int m_block_max = Is_local ? std::max(1, m_block_max_unclamped) : m_block_max_unclamped;

    int seqlen_q_stride = params.q_row_stride;
    int seqlen_k_stride = params.k_row_stride;
    int seqlen_v_stride = params.v_row_stride;
    int seqlen_do_stride = params.do_row_stride;
    int seqlen_o_stride = params.o_row_stride;
    int seqlen_dk_stride = params.dk_row_stride;
    int seqlen_dv_stride = params.dv_row_stride;

    // We move K and V to the last block.
    const int row_offset_q = binfo.q_offset1(params.q_batch_stride, params.q_row_stride, bidb) + binfo.q_offset2(params.q_head_stride,bidh) + (m_block_max - 1) * kBlockM_* seqlen_q_stride;
    const int row_offset_k = binfo.k_offset1(params.k_batch_stride, params.k_row_stride, bidb) + binfo.k_offset2(params.k_head_stride,bidh/params.h_h_k_ratio) + n_block * kBlockN_ * seqlen_k_stride;
    const int row_offset_v = binfo.k_offset1(params.v_batch_stride, params.v_row_stride, bidb) + binfo.k_offset2(params.v_head_stride,bidh/params.h_h_k_ratio) + n_block * kBlockN_ * seqlen_v_stride;

    const int row_offset_dO = binfo.q_offset1(params.do_batch_stride, params.do_row_stride, bidb) + binfo.q_offset2(params.do_head_stride,bidh) + (m_block_max - 1) * kBlockM_ * seqlen_do_stride;
    const int row_offset_o  = binfo.q_offset1(params.o_batch_stride, params.o_row_stride, bidb)  + binfo.q_offset2(params.o_head_stride,bidh) + (m_block_max - 1) * kBlockM_ * seqlen_o_stride;

    // const int row_offset_lse   = (bidb * params.h + bidh) * params.seqlen_q + (m_block_max - 1) * kBlockM_;
    const int row_offset_lse   = params.cu_seqlens_q == nullptr ? (bidb * params.h + bidh) * params.seqlen_q + (m_block_max - 1) * kBlockM_ : bidh * params.total_q + binfo.sum_s_q + (m_block_max - 1) * kBlockM_;
    const int row_offset_dpsum = (bidb * params.h + bidh) * params.seqlen_q_rounded + (m_block_max - 1) * kBlockM_;

    // Element * gQ = reinterpret_cast<Element *>(q_ptr) + row_offset_q;
    auto gQ = tcp_cache_swizzle_func<K, Element>(reinterpret_cast<Element *>(q_ptr) + row_offset_q);
    // Element * gK = reinterpret_cast<Element *>(k_ptr) + row_offset_k;
    auto gK = tcp_cache_swizzle_func<K, Element>(reinterpret_cast<Element *>(k_ptr) + row_offset_k);
    // Element * gV = reinterpret_cast<Element *>(v_ptr) + row_offset_v;
    auto gV = tcp_cache_swizzle_func<K_v, Element>(reinterpret_cast<Element *>(v_ptr) + row_offset_v);

    // Element * gdO = reinterpret_cast<Element *>(do_ptr) + row_offset_dO;
    auto gdO = tcp_cache_swizzle_func<K_v, Element>(reinterpret_cast<Element *>(do_ptr) + row_offset_dO);
    Element * gO  = reinterpret_cast<Element *>(o_ptr) + row_offset_o;
    // Element * gdQ = reinterpret_cast<Element *>(dq_ptr) + row_offset_dq;

    ElementAccumType *gLSE = reinterpret_cast<ElementAccumType *>(softmax_lse_ptr) + row_offset_lse;
    ElementAccumType *gdPsum   = reinterpret_cast<ElementAccumType *>(dsoftmax_sum) + row_offset_dpsum;

    constexpr int m_masking_steps = (!Is_causal && !Is_local)
    ? 0
    : flash::ceil_div(kBlockN_, kBlockM_);
    /***************************************************************************************************************************/
        // int warp_id =0;
        int warp_id_vec = threadIdx.x / 64; //warp id in a block
        int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);

    union_vec2_f16x2<Element> k_reg[(K/kBlockK_)*((WARP_N_*kBlockK_)/(32*32))*2/((K_prefetch_level == 3)? 1 : 2)][2];  //ds_read mini size is 32*32,2 is seq, 4 is head dim
    union_vec2_f16x2<Element> v_reg[(K_v/kBlockK_)*((WARP_N_*kBlockK_)/(32*32))*2][2];
    __builtin_amdgcn_sched_barrier(0);
    prefetch_to_vgpr<K_v, kBlockN_, kBlockK_, WARP_N_, Element, ElementAccumType, Is_even_MN>(gV, V_lds, v_reg, (binfo.actual_seqlen_k - n_block * kBlockN_), seqlen_v_stride);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    prefetch_to_vgpr<K, kBlockN_, kBlockK_, WARP_N_, Element, ElementAccumType, Is_even_MN>(gK, K_lds, k_reg, (binfo.actual_seqlen_k - n_block * kBlockN_), seqlen_k_stride);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    // __builtin_amdgcn_sched_barrier(0);
    // __builtin_amdgcn_s_waitcnt(0);
    // __syncthreads();
    // __builtin_amdgcn_sched_barrier(0);
    if constexpr (Is_preload_Q){
        prefetch_to_tmp_lds_wait<Is_even_MN, K, kBlockN_, kBlockM_, kBlockK_, WARP_N_, WARP_M_, Element>(gQ, Q_lds, (binfo.actual_seqlen_q - (m_block_max - 1) * kBlockM_), warp_id, seqlen_q_stride);
    }
    if constexpr (Is_preload_dO){
        prefetch_to_tmp_lds_wait<Is_even_MN, K_v, kBlockN_, kBlockM_, kBlockK_, WARP_N_, WARP_M_, Element>(gdO, dO_lds, (binfo.actual_seqlen_q - (m_block_max - 1) * kBlockM_), warp_id, seqlen_do_stride);
    }
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    __builtin_amdgcn_sched_barrier(0);


    union_vec4_fp32 acc_dv[(K_v/kBlockK_) * ((WARP_N_/32)*(kBlockK_/32))][4]={0};
    union_vec4_fp32 acc_dk[(K/kBlockK_) * ((WARP_N_/32)*(kBlockK_/32))][4]={0};

    for (int m_block = m_block_max - 1; m_block >= m_block_min; --m_block) {
        union_vec2_f16x2<Element> q_reg[((WARP_M_*kBlockK_)/(32*32))*2][2];

        // int warp_id =0;
        int warp_id_vec = threadIdx.x / 64; //warp id in a block
        warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
        //c mini tile is 32*32
        union_vec4_fp32 s_reg[(WARP_N_/32)*(kBlockM_/32)][4]= {0};

        //qk gemm
        gemm_tt_kq<Is_store_Q, Is_preload_dO, Is_even_MN, K_prefetch_level,  Q_prefetech_level, K, kBlockN_, kBlockM_, kBlockK_, WARP_N_, WARP_M_, STAGES, Element>(gK, gQ, K_lds, Q_lds, (binfo.actual_seqlen_k - n_block * kBlockN_), (binfo.actual_seqlen_q - m_block * kBlockM_), k_reg, q_reg, s_reg, warp_id, seqlen_k_stride, seqlen_q_stride);

        float lse[kBlockM_/4];

        #pragma unroll
        for(int vec_idx=0; vec_idx<4; vec_idx++) {
            #pragma unroll
            for (int mi = 0; mi < (kBlockM_/32); ++mi) {
                #pragma unroll
                for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                    const int lse_idx = mi*32 + vec_idx*8 + (lane_id >> 4)*2 + min_tile_m;
                    lse[(mi*2 + min_tile_m)*4 + vec_idx] = Is_even_MN || lse_idx < binfo.actual_seqlen_q - m_block * kBlockM_ ? gLSE[lse_idx] : INFINITY;
                }
            }
        }
        const int mask_m_minus_n = (n_block * kBlockN_ + warp_id * 32) - m_block * kBlockM_ - seqlen_k_minus_q;
        if constexpr (Is_local) {
            apply_mask_bwd<Is_even_MN, 3>(s_reg, binfo.actual_seqlen_k - n_block * kBlockN_ - warp_id * 32, binfo.actual_seqlen_q - m_block * kBlockM_, mask_m_minus_n, params.window_size_right, params.window_size_left);
        } else if constexpr (Is_causal) {
            apply_mask_bwd<Is_even_MN, 2>(s_reg, binfo.actual_seqlen_k - n_block * kBlockN_ - warp_id * 32, binfo.actual_seqlen_q - m_block * kBlockM_, mask_m_minus_n, params.window_size_right, params.window_size_left);
        } else {
            apply_mask_bwd<Is_even_MN, 0>(s_reg, binfo.actual_seqlen_k - n_block * kBlockN_ - warp_id * 32, binfo.actual_seqlen_q - m_block * kBlockM_, mask_m_minus_n, params.window_size_right, params.window_size_left);
        }

#ifdef DEBUGING
        print_kq(m_block, bidb, bidh);
#endif

        float dP_sum_reg[kBlockM_/4];
        #pragma unroll
        for (int vec_idx = 0; vec_idx < (kBlockM_/8); ++vec_idx) {
            for(int min_tile_m = 0; min_tile_m<2; min_tile_m++) {
                dP_sum_reg[vec_idx*2 + min_tile_m] =  gdPsum[vec_idx*8 + ((lane_id >> 4)*2) + min_tile_m];
            }
        }
        {
            scale_apply_exp2_bwd</*scale_max=*/false, kBlockM_, WARP_N_>(s_reg, lse, params.scale_softmax_log2);
        }
        if constexpr (Is_dropout) {
            const unsigned long long seed = params.rand_seed;
            const unsigned long long offset = params.rand_offset + ((bidb * params.h + bidh) << 6) + lane_id;
            const int warp_m_id = warp_id / (kBlockN_ / WARP_N_);
            const int warp_n_id = warp_id & (kBlockN_ / WARP_N_ - 1);
            union_vec2_uint rowcol;
            rowcol.u32.x = m_block * (kBlockM_ / 32) + warp_m_id;
            rowcol.u32.y = n_block * (kBlockN_ / WARP_N_) + warp_n_id;
            apply_dropout_transposed_kq<true, union_vec4_fp32, kBlockM_, WARP_N_, kBlockM_ / WARP_M_, Is_even_MN>(
                s_reg,
                binfo.actual_seqlen_q,
                binfo.actual_seqlen_k,
                m_block * kBlockM_ + warp_m_id * WARP_M_,
                n_block * kBlockN_ + warp_n_id * WARP_N_,
                bidb * params.h + bidh,
                seed,
                offset,
                params.p_dropout_in_uint8_t,
                rowcol,
                params.dropout_debug_count);
        }

#ifdef DEBUGING
        print_softmax_rescale_o(m_block, bidb, bidh);
#endif

        // //TODO:drop
        union_vec2_f16x2<Element> p_reg[(kBlockM_/32)*(WARP_N_/32)][4];

        if constexpr (Is_dropout) {
            const int warp_m_id = warp_id / (kBlockN_ / WARP_N_);
            const int warp_n_id = warp_id & (kBlockN_ / WARP_N_ - 1);
            convert_pk_type_dropout_dv<kBlockM_, WARP_N_, Element, Is_even_MN>(
                p_reg,
                s_reg,
                binfo.actual_seqlen_q,
                binfo.actual_seqlen_k,
                m_block * kBlockM_ + warp_m_id * WARP_M_,
                n_block * kBlockN_ + warp_n_id * WARP_N_,
                bidb * params.h + bidh,
                params.rand_seed,
                params.rand_offset + ((bidb * params.h + bidh) << 6) + lane_id,
                params.p_dropout_in_uint8_t,
                params.rp_dropout);
        } else {
            convert_pk_type<kBlockM_, WARP_N_, Element>(p_reg, s_reg);
        }

        //QK(seq_q, seq_kv), seq_q is continuous, seq_kv is not continuous

        if constexpr (Is_dropout || !Is_even_K || Is_causal || K_v <= 96) {
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_waitcnt(0);
            __syncthreads();
            __builtin_amdgcn_sched_barrier(0);
        }

        {
            //dv gemm,dO*P
            gpu_gemm_B_in_reg<Is_preload_dO, Is_store_dO, false, Is_even_MN, K_v, kBlockK_, kBlockN_,  kBlockM_, kBlockK_, WARP_N_, 2, Element>(gdO, gQ, dO_lds, p_reg, acc_dv, (binfo.actual_seqlen_k - n_block * kBlockN_), (binfo.actual_seqlen_q - m_block * kBlockM_),  warp_id, seqlen_do_stride);
        }

        if constexpr (!Is_even_K || K_v <= 96) {
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_waitcnt(0);
            __syncthreads();
            __builtin_amdgcn_sched_barrier(0);
        }

        union_vec2_f16x2<Element> dO_reg[((WARP_M_*kBlockK_)/(32*32))*2][2];
        union_vec4_fp32 dp_reg[(WARP_N_/32)*(kBlockM_/32)][4]= {0};

        {
            // dP gemm
            gemm_tt_kq<Is_store_dO, false, Is_even_MN, 3, dP_dO_prefetch_level, K_v, kBlockN_, kBlockM_, kBlockK_, WARP_N_, WARP_M_, STAGES, Element>(
                gV, gdO, V_lds, dO_lds,  (binfo.actual_seqlen_k - n_block * kBlockN_), (binfo.actual_seqlen_q - m_block * kBlockM_), v_reg, dO_reg, dp_reg, warp_id, seqlen_v_stride, seqlen_do_stride);
        }

#ifdef DEBUGING
        print_dp(m_block, bidb, bidh);
#endif

        union_vec4_fp32 dS_reg[(WARP_N_/32)*(kBlockM_/32)][4];
        #pragma unroll
        for (int mi = 0; mi < (kBlockM_/32); ++mi) {
            #pragma unroll
            for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                #pragma unroll
                for (int ni = 0; ni < (WARP_N_/32); ++ni)  {
                    #pragma unroll
                    for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                        #if 0//defined(__gfx936__)
                            #pragma unroll
                            for(int vec_idx=0; vec_idx<2; vec_idx++) {
                                // result register ds_reg reuse dp_reg
                                dS_reg[ni + mi*(WARP_N_/32)][min_tile_m*2 + min_tile_n].u64[vec_idx] = pointwise_mult(
                                    s_reg[ni + mi*(WARP_N_/32)][min_tile_m*2 + min_tile_n].u64[vec_idx],
                                    dp_reg[ni + mi*(WARP_N_/32)][min_tile_m*2 + min_tile_n].u64[vec_idx],
                                    vec2_fp32{gdPsum[vec_idx*16  + mi*8*4 + ((lane_id >> 4)*2) + min_tile_m], gdPsum[vec_idx*16  + mi*8*4 + ((lane_id >> 4)*2) + min_tile_m + 8]});
                            }
                        #else
                            #pragma unroll
                            for(int vec_idx=0; vec_idx<4; vec_idx++) {
                                // result register ds_reg reuse dp_reg
                                // if((m_block*kBlockM_ + vec_idx * 8 + min_tile_m + ((lane_id / 16) * 2)) < params.seqlen_q){
                                    dS_reg[ni + mi*(WARP_N_/32)][min_tile_m*2 + min_tile_n].f32[vec_idx] = pointwise_mult(s_reg[ni + mi*(WARP_N_/32)][min_tile_m*2 + min_tile_n].f32[vec_idx], dp_reg[ni + mi*(WARP_N_/32)][min_tile_m*2 + min_tile_n].f32[vec_idx], dP_sum_reg[vec_idx*2 + min_tile_m]);
                                // }
                                // else{
                                //     dS_reg[ni + mi*(WARP_N_/32)][min_tile_m*2 + min_tile_n].f32[vec_idx] = 0;
                                // }
                            }
                        #endif
                    }
                }
            }
        }

#ifdef DEBUGING
        print_ds(m_block, bidb, bidh);
#endif

        union_vec2_f16x2<Element> dS_reg_fp16[(WARP_N_/32)*(kBlockM_/32)][4];
        convert_pk_type<kBlockM_, WARP_N_, Element>(dS_reg_fp16, dS_reg);

// #ifdef DEBUGING
//         print_ds_fp16(m_block, bidb, bidh);
// #endif

        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);

        {
            //dk gemm, Q*dS
            gpu_gemm_B_in_reg<Is_store_Q , false , false, Is_even_MN, K, kBlockK_, kBlockN_,  kBlockM_, kBlockK_, WARP_N_, 2, Element>(gQ, gdO, Q_lds, dS_reg_fp16, acc_dk, (binfo.actual_seqlen_k - n_block * kBlockN_), (binfo.actual_seqlen_q - m_block * kBlockM_),  warp_id, seqlen_q_stride);
        }

        gLSE = gLSE + (-int(kBlockM_));
        gdPsum = gdPsum - kBlockM_;
        *(uint64_t*)&gQ -= ((kBlockM_ * seqlen_q_stride) * sizeof(Element));
        *(uint64_t*)&gdO -= ((kBlockM_ * seqlen_do_stride) * sizeof(Element));
        {
            __syncthreads();
            if (Is_preload_Q && m_block > m_block_min){
                prefetch_to_tmp_lds_wait<Is_even_MN, K, kBlockN_, kBlockM_, kBlockK_, WARP_N_, WARP_M_, Element>(gQ, Q_lds, (binfo.actual_seqlen_q - (m_block - 1) * kBlockM_), warp_id, seqlen_q_stride);
            }
            // __syncthreads();
            if (Is_preload_dO && m_block > m_block_min){
                prefetch_to_tmp_lds_wait<Is_even_MN, K_v, kBlockN_, kBlockM_, kBlockK_, WARP_N_, WARP_M_, Element>(gdO, dO_lds, (binfo.actual_seqlen_q - (m_block - 1) * kBlockM_), warp_id, seqlen_do_stride);
            }
        }
    }

    {
        // dv_ptr = dv_ptr + binfo.k_offset1(params.dv_batch_stride, params.dv_row_stride, bidb)*params.h_h_k_ratio + binfo.k_offset2(params.dv_head_stride,bidh);
        dv_ptr = dv_ptr + binfo.k_offset1_write(params.dv_batch_stride, params.dv_row_stride, bidb) + binfo.k_offset2(params.dv_head_stride,bidh);
        auto gdV = tcp_cache_swizzle_func<K_v, Element>(dv_ptr);
        int dv_lane_seq_idx      = (lane_id >> 4);
        int dv_lane_head_dim_idx = (lane_id & 15);

        int dv_global_addr_offset=0;
        #pragma unroll
        for(int k_loop = 0; k_loop<(K_v/kBlockK_); k_loop++) {
            #pragma unroll
            for(int warp_n_idx=0; warp_n_idx<(WARP_N_/32); warp_n_idx++) {
                #pragma unroll
                for(int k_tile_idx=0; k_tile_idx<(kBlockK_/32); k_tile_idx++) {
                    #pragma unroll
                    for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                        #pragma unroll
                        for(int vec_index=0;  vec_index<4; vec_index++) {
                            int v_offset = (dv_lane_head_dim_idx*2) + (dv_lane_seq_idx*2 * seqlen_dv_stride);
                            int s_offset = (min_tile_n*seqlen_dv_stride + vec_index * 8 * seqlen_dv_stride) + (k_tile_idx*32) + ((warp_id*WARP_N_ + warp_n_idx*32) * seqlen_dv_stride) + (k_loop * kBlockK_ + n_block * kBlockN_ * seqlen_dv_stride);
                            int known_offset = 0;
                            vec2_Element<Element> v_data;
                            v_data[0] =  DownCast<float,Element,true>(acc_dv[k_loop * ((WARP_N_/32)*(kBlockK_/32)) +  (warp_n_idx*(kBlockK_/32) + k_tile_idx)][min_tile_n*2].f32[vec_index]);
                            v_data[1] =  DownCast<float,Element,true>(acc_dv[k_loop * ((WARP_N_/32)*(kBlockK_/32)) +  (warp_n_idx*(kBlockK_/32) + k_tile_idx)][min_tile_n*2 + 1].f32[vec_index]);
                            const int seq_idx = n_block * kBlockN_ + warp_id * WARP_N_ + warp_n_idx * 32 + dv_lane_seq_idx * 2 + min_tile_n + vec_index * 8;
                            if constexpr (Is_even_K) {
                                if (Is_even_MN || seq_idx < binfo.actual_seqlen_k) {
                                    inline_buffer_store_dword_glc_slc<vec2_Element<Element>, 1>(v_data, v_offset, gdV, s_offset, /* immediate integer */known_offset);
                                }
                            } else {
                                const int head_idx = k_loop * kBlockK_ + k_tile_idx * 32 + dv_lane_head_dim_idx * 2;
                                const int global_addr = seq_idx * seqlen_dv_stride + head_idx;
                                if (seq_idx < binfo.actual_seqlen_k) {
                                    if (head_idx + 1 < params.d_value) {
                                        inline_buffer_store_dword_glc_slc<vec2_Element<Element>, 1>(v_data, v_offset, gdV, s_offset, /* immediate integer */known_offset);
                                    } else if (head_idx < params.d_value) {
                                        dv_ptr[global_addr] = v_data[0];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    {
        // dk_ptr = dk_ptr + binfo.k_offset1(params.dk_batch_stride, params.dk_row_stride, bidb)*params.h_h_k_ratio + binfo.k_offset2(params.dk_head_stride,bidh);
        dk_ptr = dk_ptr + binfo.k_offset1_write(params.dk_batch_stride, params.dk_row_stride, bidb) + binfo.k_offset2(params.dk_head_stride,bidh);
        auto gdK = tcp_cache_swizzle_func<K, Element>(dk_ptr);
        int dk_lane_seq_idx      = (lane_id >> 4);
        int dk_lane_head_dim_idx = (lane_id & 15);

        int dk_global_addr_offset=0;
        #pragma unroll
        for(int k_loop = 0; k_loop<(K/kBlockK_); k_loop++) {
            #pragma unroll
            for(int warp_n_idx=0; warp_n_idx<(WARP_N_/32); warp_n_idx++) {
                #pragma unroll
                for(int k_tile_idx=0; k_tile_idx<(kBlockK_/32); k_tile_idx++) {
                    #pragma unroll
                    for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                        #pragma unroll
                        for(int vec_index=0;  vec_index<4; vec_index++) {
                            vec2_Element<Element> v_data;
                            int v_offset = dk_lane_head_dim_idx*2 + (dk_lane_seq_idx*2) * seqlen_dk_stride;
                            int s_offset = n_block * kBlockN_ * seqlen_dk_stride + (warp_id*WARP_N_) * seqlen_dk_stride + (min_tile_n*seqlen_dk_stride + vec_index * 8 * seqlen_dk_stride + k_tile_idx*32 + k_loop * kBlockK_ + warp_n_idx*32);
                            int known_offset = 0;
                            v_data[0] = DownCast<float,Element,true>(acc_dk[k_loop * ((WARP_N_/32)*(kBlockK_/32)) +  (warp_n_idx*(kBlockK_/32) + k_tile_idx)][min_tile_n*2].f32[vec_index] * params.scale_softmax_rp_dropout);
                            v_data[1] = DownCast<float,Element,true>(acc_dk[k_loop * ((WARP_N_/32)*(kBlockK_/32)) +  (warp_n_idx*(kBlockK_/32) + k_tile_idx)][min_tile_n*2 + 1].f32[vec_index] * params.scale_softmax_rp_dropout);
                            const int seq_idx = n_block * kBlockN_ + warp_id * WARP_N_ + warp_n_idx * 32 + dk_lane_seq_idx * 2 + min_tile_n + vec_index * 8;
                            if constexpr (Is_even_K) {
                                if (Is_even_MN || seq_idx < binfo.actual_seqlen_k) {
                                    inline_buffer_store_dword_glc_slc<vec2_Element<Element>, 1>(v_data, v_offset, gdK, s_offset, /* immediate integer */known_offset);
                                }
                            } else {
                                const int head_idx = k_loop * kBlockK_ + k_tile_idx * 32 + dk_lane_head_dim_idx * 2;
                                const int global_addr = seq_idx * seqlen_dk_stride + head_idx;
                                if (seq_idx < binfo.actual_seqlen_k) {
                                    if (head_idx + 1 < params.d) {
                                        inline_buffer_store_dword_glc_slc<vec2_Element<Element>, 1>(v_data, v_offset, gdK, s_offset, /* immediate integer */known_offset);
                                    } else if (head_idx < params.d) {
                                        dk_ptr[global_addr] = v_data[0];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#undef print_kq
#undef print_dq
#undef print_softmax_rescale_o
#undef print_ds
#undef print_ds_fp16
#undef print_dp
