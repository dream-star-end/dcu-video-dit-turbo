#ifdef DEBUGING
#define print_qk(block_id_m, bidb, bidh) {\
    int qk_warp_n_offset = warp_id * WARP_M_ * params.seqlen_k; \
    int qk_global_offset = bidb*(params.h* params.seqlen_q * params.seqlen_k) + bidh * (params.seqlen_q * params.seqlen_k) + n_block*kBlockN_ \
        + block_id_m*kBlockM_*params.seqlen_k + qk_warp_n_offset; \
    for(int block_n_idx = 0; block_n_idx < kBlockN_/WARP_N_; ++block_n_idx){ \
        for(int warp_n_idx = 0; warp_n_idx < WARP_N_/16; ++warp_n_idx){ \
            for(int warp_m_idx = 0; warp_m_idx < WARP_M_/16; ++warp_m_idx){ \
                for(int vec_idx = 0; vec_idx < 4 ; ++vec_idx) {\
                    int offset = qk_global_offset + block_n_idx * WARP_N_ + lane_id%16 * params.seqlen_k + warp_m_idx * params.seqlen_k * 16 + warp_n_idx*16 + lane_id/16*4 + vec_idx; \
                    kq_ptr[offset]  = s_reg[block_n_idx][warp_n_idx*(WARP_M_/16) + warp_m_idx].f32[vec_idx]; \
                } \
            }  \
        } \
    }\
}

#define print_softmax_rescale_o(block_id_m, bidb, bidh)    {                                                        \
    int qk_warp_n_offset = warp_id * WARP_M_ * params.seqlen_k; \
    int qk_global_offset = bidb*(params.h* params.seqlen_q * params.seqlen_k) + bidh * (params.seqlen_q * params.seqlen_k) + n_block*kBlockN_ \
        + block_id_m*kBlockM_*params.seqlen_k + qk_warp_n_offset; \
    for(int block_n_idx = 0; block_n_idx < kBlockN_/WARP_N_; ++block_n_idx){ \
        for(int warp_n_idx = 0; warp_n_idx < WARP_N_/16; ++warp_n_idx){ \
            for(int warp_m_idx = 0; warp_m_idx < WARP_M_/16; ++warp_m_idx){ \
                for(int vec_idx = 0; vec_idx < 4 ; ++vec_idx) {\
                    int offset = qk_global_offset + block_n_idx * WARP_N_ + lane_id%16 * params.seqlen_k + warp_m_idx * params.seqlen_k * 16 + warp_n_idx*16 + lane_id/16*4 + vec_idx; \
                    s_ptr[offset]  = s_reg[block_n_idx][warp_n_idx*(WARP_M_/16) + warp_m_idx].f32[vec_idx]; \
                } \
            }  \
        } \
    }\
}

#define print_dp(block_id_m, bidb, bidh) {\
    int dp_warp_n_offset = warp_id * WARP_M_ * params.seqlen_k; \
    int dp_global_offset = bidb*(params.h* params.seqlen_q * params.seqlen_k) + bidh * (params.seqlen_q * params.seqlen_k) + n_block*kBlockN_ \
        + block_id_m*kBlockM_*params.seqlen_k + dp_warp_n_offset; \
    for(int block_n_idx = 0; block_n_idx < kBlockN_/WARP_N_; ++block_n_idx){ \
        for(int warp_n_idx = 0; warp_n_idx < WARP_N_/16; ++warp_n_idx){ \
            for(int warp_m_idx = 0; warp_m_idx < WARP_M_/16; ++warp_m_idx){ \
                for(int vec_idx = 0; vec_idx < 4 ; ++vec_idx) {\
                    int offset = dp_global_offset + block_n_idx * WARP_N_ + lane_id%16 * params.seqlen_k + warp_m_idx * params.seqlen_k * 16 + warp_n_idx*16 + lane_id/16*4 + vec_idx; \
                    dp_ptr[offset]  = dp_reg[block_n_idx][warp_n_idx*(WARP_M_/16) + warp_m_idx].f32[vec_idx]; \
                } \
            }  \
        } \
    }\
}

#define print_ds(block_id_m, bidb, bidh) {\
    int ds_warp_n_offset = warp_id * WARP_M_ * params.seqlen_k; \
    int ds_global_offset = bidb*(params.h* params.seqlen_q * params.seqlen_k) + bidh * (params.seqlen_q * params.seqlen_k) + n_block*kBlockN_ \
        + block_id_m*kBlockM_*params.seqlen_k + ds_warp_n_offset; \
    for(int block_n_idx = 0; block_n_idx < kBlockN_/WARP_N_; ++block_n_idx){ \
        for(int warp_n_idx = 0; warp_n_idx < WARP_N_/16; ++warp_n_idx){ \
            for(int warp_m_idx = 0; warp_m_idx < WARP_M_/16; ++warp_m_idx){ \
                for(int vec_idx = 0; vec_idx < 4 ; ++vec_idx) {\
                    int offset = ds_global_offset + block_n_idx * WARP_N_ + lane_id%16 * params.seqlen_k + warp_m_idx * params.seqlen_k * 16 + warp_n_idx*16 + lane_id/16*4 + vec_idx; \
                    ds_ptr[offset]  = dS_reg[block_n_idx][warp_n_idx*(WARP_M_/16) + warp_m_idx].f32[vec_idx]; \
                } \
            }  \
        } \
    }\
}
#endif

template<class Element, class ElementAccum, bool Is_dropout, bool Is_causal , bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_first, bool Is_last, bool Seq_parallel,  int kBlockM_, int kBlockN_, int K, int K_v, int kBlockK_, int WARP_M_, int WARP_N_, int STAGES, int USE_BSHD_LAYOUT, typename Params>
__forceinline__ __device__ void compute_dq_1colblock_gfx946(Params &params, int bidb, int bidh, int m_block 
    ) {
#ifdef DEBUGING
    ElementAccum * kq_ptr = static_cast<ElementAccum*>(params.kq_ptr);
    ElementAccum * s_ptr = static_cast<ElementAccum*>(params.s_ptr);
    ElementAccum * dp_ptr = static_cast<ElementAccum*>(params.dp_ptr);
    ElementAccum * ds_ptr = static_cast<ElementAccum*>(params.ds_ptr);
#endif                                                                                      
    Element* q_ptr = static_cast<Element*>(params.q_ptr);
    Element* k_ptr = static_cast<Element*>(params.k_ptr);
    Element* v_ptr = static_cast<Element*>(params.v_ptr);
    Element* o_ptr = static_cast<Element*>(params.o_ptr);
    Element* dq_ptr = static_cast<Element*>(params.dq_ptr);
    Element* dk_ptr = static_cast<Element*>(params.dk_ptr);
    Element* dv_ptr = static_cast<Element*>(params.dv_ptr);
    Element* do_ptr = static_cast<Element*>(params.do_ptr);
    ElementAccum* softmax_lse_ptr = static_cast<ElementAccum*>(params.softmax_lse_ptr);
    ElementAccum* dsoftmax_sum = static_cast<ElementAccum*>(params.dsoftmax_sum);
    //flash-attention QK, kBlockN_==WARP_N_;
    const int M_BLOCK_NUM = params.seqlen_q/kBlockM_;
    const int N_BLOCK_NUM = params.seqlen_k/kBlockN_;

    extern __shared__ Element smem[];

#if 1//defined(__gfx936__)
    const bool Is_store_K = true;
    const bool Is_preload_K = true;
    const bool Is_preload_V = true;
#else
    const bool Is_store_K = false;
    const bool Is_preload_K = false;
    const bool Is_preload_V = false;
#endif
    const int K_prefetch_level = Is_preload_K ? 1 : 0;
    const int V_prefetch_level = Is_preload_V ? 1 : 0;
    const int Q_prefetch_level = 3;
    
    Element* K_lds  = (Element*)&(smem);
    Element* Q_lds  = (Element*)&(smem);
    Element* dO_lds = (Element*)&(smem);
    Element* V_lds  = (Element*)&(smem) + kBlockN_* K;

    int tidx    = threadIdx.x;
    int lane_id = threadIdx.x & 63; //lane id, 0-63

    const flash::BlockInfo</*Varlen=*/!Is_even_MN, false, USE_BSHD_LAYOUT> binfo(params, bidb);
    if (m_block < 0 || m_block * kBlockM_ >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM_ - params.window_size_left) / kBlockN_);
    const int n_block_max = (!Is_causal && !Is_local) ? ceil_div(binfo.actual_seqlen_k, kBlockN_) : std::min(ceil_div(binfo.actual_seqlen_k, kBlockN_), flash::ceil_div((m_block + 1) * kBlockM_ + params.window_size_right, kBlockN_));
    
    int seqlen_q_stride = params.q_row_stride;
    int seqlen_k_stride = params.k_row_stride;
    int seqlen_v_stride = params.v_row_stride;
    int seqlen_do_stride = params.do_row_stride;
    int seqlen_o_stride = params.o_row_stride;
    int seqlen_dq_stride = params.dq_row_stride;

    // We move K and V to the last block.
    const int row_offset_q = binfo.q_offset1(params.q_batch_stride, params.q_row_stride, bidb) + binfo.q_offset2(params.q_head_stride,bidh) + m_block * kBlockM_ * seqlen_q_stride;
    const int row_offset_k = binfo.k_offset1(params.k_batch_stride, params.k_row_stride, bidb) + binfo.k_offset2(params.k_head_stride,bidh/params.h_h_k_ratio) + (n_block_max - 1) * kBlockN_ * seqlen_k_stride;
    const int row_offset_v = binfo.k_offset1(params.v_batch_stride, params.v_row_stride, bidb) + binfo.k_offset2(params.v_head_stride,bidh/params.h_h_k_ratio) + (n_block_max - 1) * kBlockN_ * seqlen_v_stride;

    const int row_offset_dO = binfo.q_offset1(params.do_batch_stride, params.do_row_stride, bidb) + binfo.q_offset2(params.do_head_stride,bidh) + m_block * kBlockM_ * seqlen_do_stride;
    const int row_offset_o  = binfo.q_offset1(params.o_batch_stride,  params.o_row_stride,  bidb) + binfo.q_offset2(params.o_head_stride,bidh)  + m_block * kBlockM_ * seqlen_o_stride;
    const int row_offset_dq = binfo.q_offset1(params.dq_batch_stride, params.dq_row_stride, bidb) + binfo.q_offset2(params.dq_head_stride,bidh) + m_block * kBlockM_ * seqlen_dq_stride;

    const int row_offset_lse   = params.cu_seqlens_q == nullptr ? (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM_ : bidh * params.total_q + binfo.sum_s_q + m_block * kBlockM_;
    const int row_offset_dpsum = (bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM_;

    auto gQ = prepare_for_matrix_load_gfx938<Element>(reinterpret_cast<Element *>(q_ptr) + row_offset_q, seqlen_q_stride);
    auto gK = prepare_for_matrix_load_gfx938<Element>(reinterpret_cast<Element *>(k_ptr) + row_offset_k, seqlen_k_stride);
    auto gV = prepare_for_matrix_load_gfx938<Element>(reinterpret_cast<Element *>(v_ptr) + row_offset_v, seqlen_v_stride);
    auto gdO = prepare_for_matrix_load_gfx938<Element>(reinterpret_cast<Element *>(do_ptr) + row_offset_dO, seqlen_do_stride);

    Element * gO  = reinterpret_cast<Element *>(o_ptr) + row_offset_o;
    dq_ptr = reinterpret_cast<Element *>(dq_ptr) + row_offset_dq;


    ElementAccum *gLSE = reinterpret_cast<ElementAccum *>(softmax_lse_ptr) + row_offset_lse;
    ElementAccum *gdPsum   = reinterpret_cast<ElementAccum *>(dsoftmax_sum) + row_offset_dpsum;

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
    ? 1
    :  ((Is_even_MN && Is_causal) ? flash::ceil_div(kBlockM_, kBlockN_) : flash::ceil_div(kBlockM_, kBlockN_) + 1);

    // int warp_id =0;
    int warp_id_vec = threadIdx.x / 64; //warp id in a block
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);

    union_vec4_f16x2<Element> q_reg[(K/kBlockK_)*((WARP_M_*kBlockK_)/(32*32))*2];
    union_vec4_f16x2<Element> dO_reg[(K_v/kBlockK_)*((WARP_M_*kBlockK_)/(32*32))*2];

    union_vec4_fp32 acc_dq[(K/kBlockK_) * ((WARP_M_/32)*(kBlockK_/32))][4]={0};
    
    float lse[WARP_M_/16];
    
    #pragma unroll
    for (int mi = 0; mi < (WARP_M_/32); ++mi) {
        for(int min_tile_m = 0; min_tile_m<2; min_tile_m++) {
            int lse_idx = warp_id*WARP_M_ + mi*32 + (lane_id & 15) + min_tile_m * 16;
            lse[mi*2 + min_tile_m] =   (Is_even_MN || lse_idx < binfo.actual_seqlen_q - m_block * kBlockM_) ? gLSE[lse_idx] : INFINITY; 
        }
    } 

    float dP_sum_reg[WARP_M_/16];
    #pragma unroll
    for (int mi = 0; mi < (WARP_M_/32); ++mi) { 
        for(int min_tile_m = 0; min_tile_m<2; min_tile_m++) {
            int dP_sum_idx = warp_id*WARP_M_ + mi*32 + (lane_id & 15) + min_tile_m * 16;
            dP_sum_reg[mi*2 + min_tile_m] = gdPsum[dP_sum_idx]; 
        }
    }
    prefetch_to_vgpr_gfx938<true, kBlockM_, K, Element, ElementAccum, Is_even_MN>(gQ, Q_lds, q_reg, (binfo.actual_seqlen_q - m_block * kBlockM_), warp_id);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    prefetch_to_vgpr_gfx938<true, kBlockM_, K, Element, ElementAccum, Is_even_MN>(gdO, dO_lds, dO_reg, (binfo.actual_seqlen_q - m_block * kBlockM_), warp_id);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    if constexpr (Is_preload_V){
        prefetch_to_lds_gfx938<true, kBlockN_, K_v, Element, ElementAccum, Is_even_MN>(gV, 0, V_lds, (binfo.actual_seqlen_k - (n_block_max - 1) * kBlockN_), warp_id);
    }

    if constexpr (Is_preload_K){
        prefetch_to_lds_gfx938<true, kBlockN_, K, Element, ElementAccum, Is_even_MN>(gK, 0, K_lds, (binfo.actual_seqlen_k - (n_block_max - 1) * kBlockN_), warp_id);
    }
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    for (int n_block = n_block_max - 1; n_block >= n_block_min ;  --n_block) {

        union_vec4_f16x2<Element> v_reg[((WARP_N_*kBlockK_)/(32*32))*2];
        union_vec4_fp32 dp_reg[(WARP_M_/32)*(kBlockN_/32)][4]= {0};

        //dP gemm
        gemm_tt_kq_gfx938<false, Is_preload_K, Is_even_MN, 3, V_prefetch_level, K_v, kBlockM_, kBlockN_, kBlockK_, WARP_N_, WARP_N_, STAGES, Element>(
            gdO, gV, dO_lds, V_lds,  (binfo.actual_seqlen_q - m_block * kBlockM_),  (binfo.actual_seqlen_k - n_block * kBlockN_), dO_reg, v_reg, dp_reg, warp_id, seqlen_do_stride, seqlen_v_stride
        );

#ifdef DEBUGING
        print_dp(m_block, bidb, bidh);
#endif

        union_vec4_f16x2<Element> k_reg[((WARP_M_*kBlockK_)/(32*32))*2];
        //c mini tile is 32*32
        union_vec4_fp32 s_reg[(WARP_N_/32)*(kBlockM_/32)][4]= {0};  

        //qk gemm
        gemm_tt_kq_gfx938<Is_store_K, false, Is_even_MN, Q_prefetch_level, K_prefetch_level, K, kBlockM_, kBlockN_, kBlockK_, WARP_N_, WARP_N_, STAGES, Element>(
            gQ, gK, Q_lds, K_lds, (binfo.actual_seqlen_q - m_block * kBlockM_), (binfo.actual_seqlen_k - n_block * kBlockN_),  q_reg, k_reg, s_reg, warp_id, seqlen_q_stride, seqlen_k_stride
        );
        
        *(uint64_t*)&gV -= ((kBlockN_ * seqlen_v_stride) * sizeof(Element));

        if (Is_preload_V && n_block > n_block_min){
            prefetch_to_lds_gfx938<true, kBlockN_, K_v, Element, ElementAccum, Is_even_MN>(gV, 0, V_lds, (binfo.actual_seqlen_k - (n_block - 1) * kBlockN_), warp_id);
        }

        apply_mask_bwd_gfx938<Is_even_MN, Is_local ? 3 : (Is_causal ? 1 : 0)>(s_reg, binfo.actual_seqlen_q - m_block * kBlockM_ - warp_id * 32, binfo.actual_seqlen_k - n_block * kBlockN_, (m_block * kBlockM_ + warp_id * 32) - (n_block * kBlockN_), params.window_size_left, params.window_size_right);

#ifdef DEBUGING
        print_qk(m_block, bidb, bidh);
#endif
        scale_apply_exp2_bwd_seq_q_major</*scale_max=*/false, WARP_M_, kBlockN_, union_vec4_fp32, ElementAccum>(s_reg, lse, params.scale_softmax_log2);

#ifdef DEBUGING
        print_softmax_rescale_o(m_block, bidb, bidh)
#endif

        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
            // return p * (dp - d);
        };

        union_vec4_fp32 dS_reg[(WARP_M_/32)*(kBlockN_/32)][4];

        #pragma unroll
        for (int ni = 0; ni < (kBlockN_/32); ++ni)  {
            #pragma unroll
            for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                #pragma unroll
                for (int mi = 0; mi < (WARP_M_/32); ++mi) {
                    #pragma unroll
                    for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                        #pragma unroll
                        for(int vec_idx=0; vec_idx<4; vec_idx++) {
                            // result register ds_reg reuse dp_reg
                            dS_reg[ni*(WARP_M_/32) + mi][min_tile_n*2 + min_tile_m].f32[vec_idx] = pointwise_mult(
                                s_reg[ni*(WARP_M_/32) + mi][min_tile_n*2 + min_tile_m].f32[vec_idx], 
                                dp_reg[ni*(WARP_M_/32) + mi][min_tile_n*2 + min_tile_m].f32[vec_idx], 
                                dP_sum_reg[min_tile_m + mi*2]);

                            // dS_reg[ni*(WARP_M_/32) + mi][min_tile_n*2 + min_tile_m].f32[vec_idx] = s_reg[ni*(WARP_M_/32) + mi][min_tile_n*2 + min_tile_m].f32[vec_idx];
                        }
                    }
                }
            }
        }        

#ifdef DEBUGING
        print_ds(m_block, bidb, bidh);
#endif
        union_vec4_f16x2<Element> dS_reg_fp16[(WARP_M_/32)*(kBlockN_/32)*2];
        convert_pk_type_gfx938<WARP_M_, kBlockN_, Element>(dS_reg_fp16, dS_reg);
        {
            //dq gemm, K*dS
            gpu_gemm_B_in_reg_gfx946<Is_store_K , false , Is_even_MN, K, kBlockK_, kBlockM_,  kBlockN_, kBlockK_, WARP_M_, 2, Element>(gK, gK, K_lds, dS_reg_fp16, acc_dq, (binfo.actual_seqlen_k - n_block * kBlockN_), (binfo.actual_seqlen_q - m_block * kBlockM_),  warp_id, seqlen_k_stride);
        }

        *(uint64_t*)&gK -= ((kBlockN_ * seqlen_k_stride) * sizeof(Element));
        // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
        //     printf("(binfo.actual_seqlen_k - n_block * kBlockN_) = %d\n", (binfo.actual_seqlen_k - n_block * kBlockN_));
        // }
#if 1//defined(__gfx936__)
        {
            __syncthreads();
            if (Is_preload_K && n_block > n_block_min){
                prefetch_to_lds_gfx938<true, kBlockN_, K, Element, ElementAccum, Is_even_MN>(gK, 0, K_lds, (binfo.actual_seqlen_k - (n_block - 1) * kBlockN_), warp_id);
            }
        }
#else
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);
#endif
    }

    #if 1
    //这是正常的MLS+ds_read_matrix的layout
    {
        dq_ptr = dq_ptr + binfo.q_offset1(params.dq_batch_stride, params.dq_row_stride, bidb) + binfo.q_offset2(params.dq_head_stride,bidh);
        auto gdQ = tcp_cache_swizzle_func<K_v, Element>(dq_ptr);
        int dq_lane_seq_idx      = (lane_id >> 4);
        int dq_lane_head_dim_idx = (lane_id & 15);

        int dq_global_addr_offset=0;
        #pragma unroll
        for(int k_loop = 0; k_loop<(K_v/kBlockK_); k_loop++) {
            #pragma unroll
            for(int warp_m_idx=0; warp_m_idx<(WARP_M_/32); warp_m_idx++) {
                #pragma unroll
                for(int k_tile_idx=0; k_tile_idx<(kBlockK_/32); k_tile_idx++) {
                    #pragma unroll
                    for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                        #pragma unroll
                        for(int vec_index=0;  vec_index<4; vec_index++) {
                            int v_offset = dq_lane_head_dim_idx * seqlen_dq_stride + dq_lane_seq_idx * 4;
                            int s_offset = (min_tile_m * seqlen_dq_stride * 16 + vec_index % 2 * 2 + vec_index / 2 * 16) + (k_tile_idx*32) + ((warp_id*WARP_M_ + warp_m_idx*32) * seqlen_dq_stride) + (k_loop * kBlockK_ + m_block * kBlockM_ * seqlen_dq_stride);
                            int known_offset = 0;
                            vec2_Element<Element> v_data;
                            v_data[0] =  DownCast<float,Element,true>(acc_dq[k_loop * ((WARP_M_/32)*(kBlockK_/32)) +  (warp_m_idx*(kBlockK_/32) + k_tile_idx)][min_tile_m*2 + vec_index / 2].f32[vec_index % 2 * 2] * params.scale_softmax_rp_dropout);
                            v_data[1] =  DownCast<float,Element,true>(acc_dq[k_loop * ((WARP_M_/32)*(kBlockK_/32)) +  (warp_m_idx*(kBlockK_/32) + k_tile_idx)][min_tile_m*2 + vec_index / 2].f32[vec_index % 2 * 2 + 1] * params.scale_softmax_rp_dropout);
                            if (Is_even_MN || min_tile_m*16 + (warp_id*WARP_M_ + warp_m_idx*32) + m_block * kBlockM_ + dq_lane_head_dim_idx < binfo.actual_seqlen_q){
                                inline_buffer_store_dword_glc_slc<vec2_Element<Element>, 1>(v_data, v_offset, gdQ, s_offset, /* immediate integer */known_offset);
                            }
                        }
                    }
                }
            }
        }
    }
    #endif
}

#undef print_qk
#undef print_softmax_rescale_o
#undef print_dp
#undef print_ds

