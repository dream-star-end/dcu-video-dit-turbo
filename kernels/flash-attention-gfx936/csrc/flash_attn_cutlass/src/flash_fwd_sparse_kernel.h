/******************************************************************************
 * Copyright (c) 2024, PAI, Alibaba Cloud.
 ******************************************************************************/

#pragma once

#include "flash_fwd_kernel.h"

namespace flash {

using namespace cute;
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void sparse_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {
    #define S_WAITCNT asm volatile("s_waitcnt vmcnt(3) \n s_barrier")
    #define S_BARRIER asm volatile("s_barrier")
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);
    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)  
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(0, 0));  // (kBlockN, kHeadDim, nblocksN)


    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(0, 0));  // (kBlockN, kHeadDim, nblocksN)
    const auto gK_data = gK.data();
    const auto gV_data = gV.data();
    const index_t row_offset_k_token =
        binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v_token =
        binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gKToken = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k_token),
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_stride(_0{}, _1{}));
    Tensor gVToken = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v_token),
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_stride(_0{}, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});

    typename Kernel_traits::TiledMma16x64  tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(gK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVtSplit);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x128, kHeadDimV/32>(tOsVt8x64.layout()));
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    int n_block = n_block_max - 1;

    int num_blks = params.block_count[(bidb * params.h + bidh) * params.NUM_ROWS + m_block];
    auto* blks_ptr = params.block_offset + ((bidb * params.h + bidh) * params.NUM_ROWS + m_block) * params.NNZ_S;
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);
    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    int num_cols = params.column_count[(bidb * params.h + bidh) * params.NUM_ROWS + m_block];
    int num_cols_block = (num_cols + kBlockN - 1)/ kBlockN;
    constexpr int kStages = Kernel_traits::kStages;
    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    if (num_blks <= 0 && num_cols_block <= 0) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    if (num_blks > 0) {
        int block_index = num_blks - 1;
        int actual_block = blks_ptr[block_index];
        gK.data() = gK_data + actual_block * int64_t(params.k_row_stride);
        gV.data() = gV_data + actual_block * int64_t(params.v_row_stride);
        
        #pragma unroll
        for (int i = 0; i < kStages; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - actual_block);
        }

        for (int masking_step = 0; masking_step < n_masking_steps && block_index >= 0; ++masking_step, --block_index) {
            Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
            clear(acc_s_ori);
            #pragma unroll
            for (int i = 0; i < k0_loops - kStages; ++i) {
                lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - actual_block);
                S_WAITCNT;
                flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i);
                S_BARRIER;
            }
            #pragma unroll
            for (int i = 0; i < kStages; ++i) { // tail kStages
                lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, i, params.v_row_stride, params.d, binfo.actual_seqlen_k - actual_block);
                S_WAITCNT;
                flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, k0_loops - kStages + i);
                S_BARRIER;
            }
            Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
            if constexpr (Is_softcap){
                apply_softcap(acc_s, params.softcap);
            }
            {
                const int wave_id = (tidx / 64);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx % 16) + (wave_id_to_row_block_id * 16);
                const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
                mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                    acc_s, actual_block, row_idx_offset_, (kNWarps << 4)
                );
            }
            
            // Sparse Warp Online Softmax: compute max_diff for dynamic PV skip
            float max_diff = -INFINITY;
            bool skip_pv = false;
            bool is_first_block = (masking_step == 0);

            if (is_first_block) {
                max_diff = softmax.template softmax_rescale_o_with_diff</*Is_first=*/true, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);
            } else {
                max_diff = softmax.template softmax_rescale_o_with_diff</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

                // Check if we can skip P@V computation based on dynamic threshold
                // Skip when max_diff + pv_threshold <= 0 (current block's contribution is negligible)
                if (params.enable_dynamic_skip) {
                    // Reduce max_diff across the warp to get the maximum value
                    MaxOp<float> max_op;
                    float warp_max_diff = Allreduce<64>::run(max_diff, max_op);

                    // All threads in the warp must agree on skip decision
                    skip_pv = (warp_max_diff + params.pv_threshold <= 0.0f);
                }
            }

            Tensor rP = flash::convert_type<Element>(acc_s);
            {   // dropout
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

                const int block_row_idx = row_idx_offset_;
                const int block_col_idx = n_block * (kBlockN);
                if constexpr (Return_softmax) {
                    Tensor rP_drop = make_fragment_like(rP);
                    cute::copy(rP, rP_drop);
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                    Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                    cute::copy(rP_drop_back, tSgS);
                    tSgS.data() = tSgS.data() + (-kBlockN);
                }
                if constexpr (Is_dropout) {
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }

            // Sparse Warp Online Softmax: Skip P@V if contribution is negligible
            if (!skip_pv) {
                // Accumulate softmax sum (must be done AFTER confirming we're not skipping)
                // This aligns with SpargeAttn's accumulate_d() pattern
                if (is_first_block) {
                    softmax.template accumulate_softmax_sum</*Is_first=*/true>(acc_s);
                } else {
                    softmax.template accumulate_softmax_sum</*Is_first=*/false>(acc_s);
                }
                lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, kStages + 0, params.v_row_stride, params.d, binfo.actual_seqlen_k - actual_block);
                S_WAITCNT;
                flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
                

                #if 1

                asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
                flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
                // S_BARRIER;
                // k = 2
                asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
                flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
                // S_BARRIER;
                // k = 3
                asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
                flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
                S_BARRIER;
                #endif
                
                // if (thread0())
                // {
                //     // asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
                //     printf("block_index =  %d actual_block = %d \n \n", block_index, actual_block);
                //     for (int i = 0; i < 64; i++)
                //     {
                //         for (int j = 0; j < 128; j++) {
                //             printf(" %.2f ", float(sV(i, j)));
                //         }
                //         printf("\n");
                //     }
                    
                // }
            }
            
            // BUGFIX: Prefetch next block data regardless of skip decision
            // This ensures pipeline is always filled for next iteration
            if (block_index > 0) {
                actual_block = blks_ptr[block_index - 1];
                gK.data() = gK_data + actual_block * int64_t(params.k_row_stride);
                gV.data() = gV_data + actual_block * int64_t(params.v_row_stride);

                lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
            }
        }
        #if 1
        for (; block_index >= 0; --block_index) {
            Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
            clear(acc_s_ori);
            #pragma unroll
            for (int i = 0; i < k0_loops - kStages; ++i) {
                lds_direct_copy<Is_even_K, true>(gK, sK, kStages + i, params.k_row_stride, params.d);
                S_WAITCNT;
                flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i);
                S_BARRIER;
            }
            #pragma unroll
            for (int i = 0; i < kStages; ++i) { // tail kStages
                lds_direct_copy<Is_even_K, true, _16x128>(gV, sV, i, params.v_row_stride, params.d);
                S_WAITCNT;
                flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, k0_loops - kStages + i);
                S_BARRIER;
            }
            Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
            if constexpr (Is_softcap){
                apply_softcap(acc_s, params.softcap);
            }
            // {
            //     const int wave_id = (tidx / 64);
            //     const int wave_id_to_row_block_id = wave_id;
            //     const int warp_row_stride = 16;
            //     const int row_idx_offset_in_block = (tidx % 16) + (wave_id_to_row_block_id * 16);
            //     const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            //     mask.template apply_mask_continuous<false>(
            //         acc_s, actual_block, row_idx_offset_, (kNWarps << 4)
            //     );
            // }
            
            // Sparse Warp Online Softmax: compute max_diff for dynamic PV skip
            float max_diff = softmax.template softmax_rescale_o_with_diff</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

            bool skip_pv = false;
            if (params.enable_dynamic_skip) {
                // Reduce max_diff across the warp to get the maximum value
                MaxOp<float> max_op;
                float warp_max_diff = Allreduce<64>::run(max_diff, max_op);

                // All threads in the warp must agree on skip decision
                skip_pv = (warp_max_diff + params.pv_threshold <= 0.0f);
            }

            Tensor rP = flash::convert_type<Element>(acc_s);
            {   // dropout
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

                const int block_row_idx = row_idx_offset_;
                const int block_col_idx = n_block * (kBlockN);
                if constexpr (Return_softmax) {
                    Tensor rP_drop = make_fragment_like(rP);
                    cute::copy(rP, rP_drop);
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                    Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                    cute::copy(rP_drop_back, tSgS);
                    tSgS.data() = tSgS.data() + (-kBlockN);
                }
                if constexpr (Is_dropout) {
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }

            // Sparse Warp Online Softmax: Skip P@V if contribution is negligible
            if (!skip_pv) {
                // Accumulate softmax sum (must be done AFTER confirming we're not skipping)
                softmax.template accumulate_softmax_sum</*Is_first=*/false>(acc_s);
                lds_direct_copy<Is_even_K, true, _16x128>(gV, sV, kStages + 0, params.v_row_stride, params.d);
                S_WAITCNT;
                flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
                
                #if 1
                asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
                flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
                // S_BARRIER;
                // k = 2
                asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
                flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
                // S_BARRIER;
                // k = 3
                asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
                flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
                S_BARRIER;
                #endif
            }
            
            // BUGFIX: Prefetch next block data regardless of skip decision
            // This ensures pipeline is always filled for next iteration
            #if 1
            if (block_index > 0) {
                const int actual_block = blks_ptr[block_index - 1];
                gK.data() = gK_data + actual_block * int64_t(params.k_row_stride);
                gV.data() = gV_data + actual_block * int64_t(params.v_row_stride);

                lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
            }
            #endif
        
        }

        #endif
    }
    

    #if 1
    if (num_cols > 0) {
        const int* cols_ptr = params.column_index + ((bidb * params.h + bidh) * params.NUM_ROWS + m_block) * params.NNZ_V;
        int row = tidx % 16;
        int lane = tidx % 64;
        // int col = lane / 16;
        int k_row_offset =  row * 4 + tidx / 64 < num_cols ? cols_ptr[row * 4 + tidx / 64] : binfo.actual_seqlen_k;
        int v_row_offset_0 = (tidx % 64) / 4 < num_cols ? cols_ptr[(tidx % 64) / 4] : binfo.actual_seqlen_k;
        int v_row_offset_1 = (tidx % 64) / 4 + 16 < num_cols ? cols_ptr[(tidx % 64) / 4 + 16] :  binfo.actual_seqlen_k;
        int v_row_offset_2 = (tidx % 64) / 4 + 32 < num_cols ? cols_ptr[(tidx % 64) / 4 + 2 * 16] :  binfo.actual_seqlen_k;
        int v_row_offset_3 = (tidx % 64) / 4 + 48 < num_cols ? cols_ptr[(tidx % 64) / 4 + 3 * 16] : binfo.actual_seqlen_k;
        #pragma unroll
        for (int i = 0; i < kStages; ++i) {
            lds_direct_copy_for_vertical_sparse<Is_even_K, Is_even_MN>(gKToken, sK, i, params.k_row_stride, k_row_offset, params.d, binfo.actual_seqlen_k);
        }

        // if (thread0())
        // {
        //     printf(" num_cols_block = %d n_masking_steps = %d \n ", num_cols_block, n_masking_steps);
        // }
        // asm volatile("s_waitcnt vmcnt(0) \n s_barrier");

        #if 1
        for (int n = 0; n < num_cols_block; ++n) {
            Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
            clear(acc_s_ori);


            #pragma unroll
            for (int i = 0; i < k0_loops - kStages; ++i) {
                lds_direct_copy_for_vertical_sparse<Is_even_K, Is_even_MN>(gKToken, sK,  kStages + i, params.k_row_stride, k_row_offset, params.d, binfo.actual_seqlen_k);
                S_WAITCNT;
                flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i);
                S_BARRIER;
            }
            {
                lds_direct_copy_for_vertical_sparse<Is_even_K, Is_even_MN, _16x128>(gVToken, sV, 0, params.v_row_stride, v_row_offset_0, params.d, binfo.actual_seqlen_k);
                S_WAITCNT;
                flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, k0_loops - kStages + 0);
                S_BARRIER;
                lds_direct_copy_for_vertical_sparse<Is_even_K, Is_even_MN, _16x128>(gVToken, sV, 1, params.v_row_stride, v_row_offset_1, params.d, binfo.actual_seqlen_k);
                S_WAITCNT;
                flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, k0_loops - kStages + 1);
                S_BARRIER;
                lds_direct_copy_for_vertical_sparse<Is_even_K, Is_even_MN, _16x128>(gVToken, sV, 2, params.v_row_stride, v_row_offset_2, params.d, binfo.actual_seqlen_k);
                S_WAITCNT;
                flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, k0_loops - kStages + 2);
                S_BARRIER;
            }

            Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
            if constexpr (Is_softcap){
                apply_softcap(acc_s, params.softcap);
            }
            if (n >= num_cols_block - n_masking_steps) {
                Tensor tensor = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int lane_id = threadIdx.x & 63;
                const int col_idx_offset = n * kBlockN + ((lane_id >> 4) << 2);
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int row_idx_offset_in_block = (tidx & (16 - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset = m_block * kBlockM + row_idx_offset_in_block;
                const int warp_row_stride = kNWarps * 16;
                const int max_seqlen_k = binfo.actual_seqlen_k;
                const int max_seqlen_q = binfo.actual_seqlen_q;
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    const int row_idx = row_idx_offset + mi * warp_row_stride;
                    const int col_idx_limit = Is_causal ? std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : max_seqlen_k;
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                        const int col_idx_base = col_idx_offset + (nj << 4);
                        for (int j = 0; j < size<1, 0>(tensor); ++j) {
                            const int col_idx = col_idx_base + j;
                            
                            // if (block0() && threadIdx.x < 64) {
                            //     printf(" threadIdx.x = %d num_cols = %d col_idx = %d cols_ptr[col_idx] = %d col_idx_limit = %d\n", threadIdx.x, num_cols, col_idx, cols_ptr[col_idx], col_idx_limit);
                            // }

                            if (col_idx >= num_cols || cols_ptr[col_idx] >= col_idx_limit) {
                                tensor(mi, make_coord(j, nj)) = -INFINITY;
                            }
                        }
                    }
                }
            }

            // Sparse Warp Online Softmax: compute max_diff for dynamic PV skip
            float max_diff = -INFINITY;
            bool skip_pv = false;
            bool is_first_block = (num_blks <= 0 && n == 0);

            if (is_first_block) {
                max_diff = softmax.template softmax_rescale_o_with_diff</*Is_first=*/true, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);
            } else {
                max_diff = softmax.template softmax_rescale_o_with_diff</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

                // Check if we can skip P@V computation based on dynamic threshold
                if (params.enable_dynamic_skip) {
                    // Reduce max_diff across the warp to get the maximum value
                    MaxOp<float> max_op;
                    float warp_max_diff = Allreduce<64>::run(max_diff, max_op);

                    // All threads in the warp must agree on skip decision
                    skip_pv = (warp_max_diff + params.pv_threshold <= 0.0f);
                }
            }

            Tensor rP = flash::convert_type<Element>(acc_s);
            // if (block(1))
            // {
            //     printf("block1 tidx %d rP %.2f %.2f %.2f %.2f \n", tidx, float(rP(0)), float(rP(1)), float(rP(2)), float(rP(3)));
            // }
            {   // dropout
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

                const int block_row_idx = row_idx_offset_;
                const int block_col_idx = n_block * (kBlockN);
                if constexpr (Return_softmax) {
                    Tensor rP_drop = make_fragment_like(rP);
                    cute::copy(rP, rP_drop);
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                    Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                    cute::copy(rP_drop_back, tSgS);
                    tSgS.data() = tSgS.data() + (-kBlockN);
                }
                if constexpr (Is_dropout) {
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }

            // Sparse Warp Online Softmax: Skip P@V if contribution is negligible
            if (!skip_pv) {
                // Accumulate softmax sum (must be done AFTER confirming we're not skipping)
                // This aligns with SpargeAttn's accumulate_d() pattern
                if (is_first_block) {
                    softmax.template accumulate_softmax_sum</*Is_first=*/true>(acc_s);
                } else {
                    softmax.template accumulate_softmax_sum</*Is_first=*/false>(acc_s);
                }
                lds_direct_copy_for_vertical_sparse<Is_even_K, Is_even_MN, _16x128>(gVToken, sV, 3, params.v_row_stride, v_row_offset_3, params.d, binfo.actual_seqlen_k);
                S_WAITCNT;
                flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
                
                // if (block0() && threadIdx.x < 64)
                // {
                //     printf("tidx = %d acc_s rp = %.2f %.2f %.2f binfo.actual_seqlen_k = %d v_row_offset_0 = %d v_row_offset_1 = %d v_row_offset_2 = %d v_row_offset_3 = %d\n", threadIdx.x, float(rP(0)), float(tOrVt(0)), acc_o(0), binfo.actual_seqlen_k, v_row_offset_0, v_row_offset_1, v_row_offset_2, v_row_offset_3);
                // }

                #if 1
                asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
                flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
                // S_BARRIER;
                // k = 2
                asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
                flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
                // S_BARRIER;
                // k = 3
                asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
                flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
                S_BARRIER;
                #endif
            }
            
            // BUGFIX: Prefetch next block data regardless of skip decision
            // This ensures pipeline is always filled for next iteration
            #if 1
            if (n + 1 < num_cols_block) {
                // cols_ptr += kBlockN;
                int row = tidx % 16;
                // int col = lane / 16;
                k_row_offset = (n + 1) * kBlockN + row * 4 + tidx / 64 < num_cols ? cols_ptr[ (n + 1) * kBlockN + row * 4 + tidx / 64] : binfo.actual_seqlen_k;
                v_row_offset_0 =(n + 1) * kBlockN + (tidx % 64) / 4 < num_cols ? cols_ptr[(n + 1) * kBlockN + (tidx % 64) / 4] : binfo.actual_seqlen_k;
                v_row_offset_1 = (n + 1) * kBlockN + (tidx % 64) / 4 + 16 < num_cols ? cols_ptr[(n + 1) * kBlockN + (tidx % 64) / 4 + 16] : binfo.actual_seqlen_k;
                v_row_offset_2 = (n + 1) * kBlockN + (tidx % 64) / 4 + 32 < num_cols ? cols_ptr[(n + 1) * kBlockN + (tidx % 64) / 4 + 2 * 16] : binfo.actual_seqlen_k;
                v_row_offset_3 = (n + 1) * kBlockN + (tidx % 64) / 4 + 48 < num_cols ? cols_ptr[(n + 1) * kBlockN + (tidx % 64) / 4 + 3 * 16] : binfo.actual_seqlen_k;
                lds_direct_copy_for_vertical_sparse<Is_even_K, Is_even_MN>(gKToken, sK, 0, params.k_row_stride, k_row_offset, params.d, binfo.actual_seqlen_k);
                lds_direct_copy_for_vertical_sparse<Is_even_K, Is_even_MN>(gKToken, sK, 1, params.k_row_stride, k_row_offset, params.d, binfo.actual_seqlen_k);
                lds_direct_copy_for_vertical_sparse<Is_even_K, Is_even_MN>(gKToken, sK, 2, params.k_row_stride, k_row_offset, params.d, binfo.actual_seqlen_k);
                // lds_direct_copy_for_vertical_sparse<Is_even_K, Is_even_MN>(gKToken, sK, 3, params.k_row_stride, cols_ptr, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            }
            #endif
        }
    
        #endif
    }

    // if (thread0())
    // {
    //     printf(" acc_o %.2f %.2f \n", float(acc_o(0))), float(acc_o(1));
    // }

    #endif
    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
    
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) + ni * 32;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ++ei) {
                    // wangaq debug
                    // if(thread(0, 0)) {
                    //     printf("mi:%d ni:%d ei:%d row:%d col:%d acc_o:%.4f\n", 
                    //     mi, ni, ei, row, col, acc_o(ei, mi, ni));
                    // }
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                    }
                    //  else 
                    //     gO(row, col) = Element(0.0);
                    col += 4;
                }
            }
        }
    } 


}

#if 1
template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void sparse_attn_1rowblock_sla(const Params &params, const int bidb, const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
    const int n_block_min = 0;
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    
    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)  
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(0, 0));  // (kBlockN, kHeadDim, nblocksN)


    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(0, 0));  // (kBlockN, kHeadDim, nblocksN)
    const auto gK_data = gK.data();
    const auto gV_data = gV.data();
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
        Shape<Int<kBlockM>, Int<kBlockN>>{}, make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});

    typename Kernel_traits::TiledMma16x64  tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(gK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVtSplit);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x128, kHeadDimV/32>(tOsVt8x64.layout()));
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    int n_block = n_block_max - 1;

    int num_blks = params.block_count[(bidb * params.h + bidh) * params.NUM_ROWS + m_block];
    auto* blks_ptr = params.block_offset + ((bidb * params.h + bidh) * params.NUM_ROWS + m_block) * params.NNZ_S;
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);
    flash::Softmax<size<1>(acc_o)> softmax;

    flash::Mask<false, false, false> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, 0.0);
    constexpr int n_masking_steps = 1;
    constexpr int kStages = Kernel_traits::kStages;
    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    if (num_blks <= 0) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    int block_index = num_blks - 1;
    int actual_block = blks_ptr[block_index];
    gK.data() = gK_data + actual_block * int64_t(params.k_row_stride);
    gV.data() = gV_data + actual_block * int64_t(params.v_row_stride);
    
    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - actual_block);
    }

    for (int masking_step = 0; masking_step < n_masking_steps && block_index >= 0; ++masking_step, --block_index) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - actual_block);
            s_waitcnt<3>();
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i);
            s_barrier();
        }
        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, i, params.v_row_stride, params.d, binfo.actual_seqlen_k - actual_block);
            s_waitcnt<3>();
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, k0_loops - kStages + i);
            s_barrier();
        }
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        {
            const int wave_id = (tidx / 64);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx % 16) + (wave_id_to_row_block_id * 16);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false, Is_even_MN>(
                acc_s, actual_block, row_idx_offset_, (kNWarps << 4)
            );
        }
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/false>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false>(acc_s, acc_o, params.scale_softmax_log2);
        Tensor rP = flash::convert_type<Element>(acc_s);
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, kStages + 0, params.v_row_stride, params.d, binfo.actual_seqlen_k - actual_block);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 2
        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 3
        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        // if (thread0())
        // {
        //     // asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        //     printf("block_index =  %d actual_block = %d \n \n", block_index, actual_block);
        //     for (int i = 0; i < 64; i++)
        //     {
        //         for (int j = 0; j < 128; j++) {
        //             printf(" %.2f ", float(sV(i, j)));
        //         }
        //         printf("\n");
        //     }
            
        // }

        if (block_index > 0) {
            actual_block = blks_ptr[block_index - 1];
            gK.data() = gK_data + actual_block * int64_t(params.k_row_stride);
            gV.data() = gV_data + actual_block * int64_t(params.v_row_stride);

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
        }
    }
    #if 1
    for (; block_index >= 0; --block_index) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            lds_direct_copy<Is_even_K, true>(gK, sK, kStages + i, params.k_row_stride, params.d);
            s_waitcnt<3>();
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i);
            s_barrier();
        }
        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            lds_direct_copy<Is_even_K, true, _16x128>(gV, sV, i, params.v_row_stride, params.d);
            s_waitcnt<3>();
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, k0_loops - kStages + i);
            s_barrier();
        }
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false>(acc_s, acc_o, params.scale_softmax_log2);
        Tensor rP = flash::convert_type<Element>(acc_s);

        lds_direct_copy<Is_even_K, true, _16x128>(gV, sV, kStages + 0, params.v_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        
        #if 1
        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 2
        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 3
        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        if (block_index > 0) {
            const int actual_block = blks_ptr[block_index - 1];
            gK.data() = gK_data + actual_block * int64_t(params.k_row_stride);
            gV.data() = gV_data + actual_block * int64_t(params.v_row_stride);

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
        }
        #endif
    
    }

    #endif
    

    Tensor lse = softmax.template normalize_softmax_lse<false>(acc_o, params.scale_softmax, params.rp_dropout);
    
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) + ni * 32;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ++ei) {
                    // wangaq debug
                    // if(thread(0, 0)) {
                    //     printf("mi:%d ni:%d ei:%d row:%d col:%d acc_o:%.4f\n", 
                    //     mi, ni, ei, row, col, acc_o(ei, mi, ni));
                    // }
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                    }
                    //  else 
                    //     gO(row, col) = Element(0.0);
                    col += 4;
                }
            }
        }
    } 


}
#endif


template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void sparse_attn_1rowblock_sla_dim64(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    // constexpr int kStages = Kernel_traits::kStages;
    
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = 0;
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(0, 0));  // (kBlockN, kHeadDim, nblocksN)


    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(0, 0));  // (kBlockN, kHeadDim, nblocksN)
    const auto gK_data = gK.data();
    const auto gV_data = gV.data();
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});

    typename Kernel_traits::TiledMma16x64  tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(gK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    //
    // Copy Atom retiling
    //

    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVtSplit);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x64_64, kHeadDimV/32>(tOsVt8x64.layout()));


    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    
    int num_blks = params.block_count[(bidb * params.h + bidh) * params.NUM_ROWS + m_block];
    auto* blks_ptr = params.block_offset + ((bidb * params.h + bidh) * params.NUM_ROWS + m_block) * params.NNZ_S;
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    flash::Mask<false, false, false> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, 0.0);
    constexpr int n_masking_steps = 1;
    
    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    static_assert(k0_loops == 2 && k1_loops == 4);
    
    if (num_blks <= 0) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    int block_index = num_blks - 1;
    int actual_block = blks_ptr[block_index];
    gK.data() = gK_data + actual_block * int64_t(params.k_row_stride);
    gV.data() = gV_data + actual_block * int64_t(params.v_row_stride);
    
    #pragma unroll
    for (int i = 0; i < k0_loops; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - actual_block);
    }
    #pragma unroll
    for (int i = 0; i < k1_loops; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gV, sV, i, params.v_row_stride, params.d, binfo.actual_seqlen_k - actual_block);
    }
    
#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps && block_index >= 0; ++masking_step, --block_index) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        s_waitcnt<5>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);

        s_waitcnt<4>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        s_barrier();

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("acc_s tid:%3d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, n_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15],
        //     tmp[16], tmp[17], tmp[18], tmp[19], 
        //     tmp[20], tmp[21], tmp[22], tmp[23],
        //     tmp[24], tmp[25], tmp[26], tmp[27], 
        //     tmp[28], tmp[29], tmp[30], tmp[31]
        //     );
        // }

        #if 1
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false, Is_even_MN>(
                acc_s, actual_block, row_idx_offset_, (kNWarps << 4)
            );
        }

        // wangaq debug
        // __syncthreads();
        // if (thread0() && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     Element * tmp = reinterpret_cast<Element*>(sV.data().get());
        //     int col = 32;
        //     for (int i = 0; i < size(sV)/col; ++i) {
        //         printf("V:%d nblock:%d ", i, n_block);
        //         for (int j = 0; j < col; ++j) {
        //             printf("%.4f ", float(tmp[i*col+j]));
        //         }
        //         printf("\n");
        //     }
        // }
        
        softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/false>(acc_s, acc_o, params.scale_softmax_log2);

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("exp_s tid:%d n_block:%d row_max:%10.4f %10.4f row_sum:%10.4f %10.4f | %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, n_block, softmax.row_max(0), softmax.row_max(1), softmax.row_sum(0), softmax.row_sum(1), 
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15],
        //     tmp[16], tmp[17], tmp[18], tmp[19], 
        //     tmp[20], tmp[21], tmp[22], tmp[23],
        //     tmp[24], tmp[25], tmp[26], tmp[27], 
        //     tmp[28], tmp[29], tmp[30], tmp[31]
        //     );
        // }

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        
        if (block_index > 0) {
            actual_block = blks_ptr[block_index - 1];
            gK.data() = gK_data + actual_block * int64_t(params.k_row_stride);
            gV.data() = gV_data + actual_block * int64_t(params.v_row_stride);

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            

            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 0, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 1, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 2, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 3, params.v_row_stride, params.d);

        }
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_o.data());
        //     printf("acc_o tid:%d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, n_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15],
        //     tmp[16], tmp[17], tmp[18], tmp[19], 
        //     tmp[20], tmp[21], tmp[22], tmp[23],
        //     tmp[24], tmp[25], tmp[26], tmp[27], 
        //     tmp[28], tmp[29], tmp[30], tmp[31]
        //     );
        // }

        #endif
    }

    #pragma unroll
    for (; block_index >= 0; --block_index) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        s_waitcnt<5>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        
        s_waitcnt<4>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        s_barrier();
        
        // __builtin_amdgcn_sched_barrier(1);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("acc_s tid:%d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, n_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15],
        //     tmp[16], tmp[17], tmp[18], tmp[19], 
        //     tmp[20], tmp[21], tmp[22], tmp[23],
        //     tmp[24], tmp[25], tmp[26], tmp[27], 
        //     tmp[28], tmp[29], tmp[30], tmp[31]
        //     );
        // }
        
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false>(acc_s, acc_o, params.scale_softmax_log2);
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("exp_s tid:%d n_block:%d row_max:%10.4f %10.4f row_sum:%10.4f %10.4f | %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, n_block, softmax.row_max(0), softmax.row_max(1), softmax.row_sum(0), softmax.row_sum(1), 
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15],
        //     tmp[16], tmp[17], tmp[18], tmp[19], 
        //     tmp[20], tmp[21], tmp[22], tmp[23],
        //     tmp[24], tmp[25], tmp[26], tmp[27], 
        //     tmp[28], tmp[29], tmp[30], tmp[31]
        //     );
        // }

        Tensor rP = flash::convert_type<Element>(acc_s);

        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        
        if (block_index > 0) {
            actual_block = blks_ptr[block_index - 1];
            gK.data() = gK_data + actual_block * int64_t(params.k_row_stride);
            gV.data() = gV_data + actual_block * int64_t(params.v_row_stride);

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            

            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 0, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 1, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 2, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 3, params.v_row_stride, params.d);

        }
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_o.data());
        //     printf("acc_o tid:%d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, n_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15],
        //     tmp[16], tmp[17], tmp[18], tmp[19], 
        //     tmp[20], tmp[21], tmp[22], tmp[23],
        //     tmp[24], tmp[25], tmp[26], tmp[27], 
        //     tmp[28], tmp[29], tmp[30], tmp[31]
        //     );
        // }
    }

    // Epilogue
    Tensor lse = softmax.template normalize_softmax_lse<false>(acc_o, params.scale_softmax, params.rp_dropout);

    // S_BARRIER;

    // wangaq debug
    // __syncthreads();
    // if (blockIdx.x == 0) {
    //     float * tmp = reinterpret_cast<float*>(acc_o.data());
    //     printf("acc_o tid:%d n_block:%d 0:%10.4f 1:%10.4f 2:%10.4f 3:%10.4f 4:%10.4f 5:%10.4f 6:%10.4f 7:%10.4f "
    //     "8:%10.4f 9:%10.4f 10:%10.4f 11:%10.4f 12:%10.4f 13:%10.4f 14:%10.4f 15:%10.4f "
    //     "16:%10.4f 17:%10.4f 18:%10.4f 19:%10.4f 20:%10.4f 21:%10.4f 22:%10.4f 23:%10.4f "
    //     "24:%10.4f 25:%10.4f 26:%10.4f 27:%10.4f 28:%10.4f 29:%10.4f 30:%10.4f 31:%10.4f\n", tidx, n_block,
    //     // tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7],
    //     // tmp[8], tmp[9], tmp[10], tmp[11], tmp[12], tmp[13], tmp[14], tmp[15],
    //     // tmp[16], tmp[17], tmp[18], tmp[19], tmp[20], tmp[21], tmp[22], tmp[23],
    //     // tmp[24], tmp[25], tmp[26], tmp[27], tmp[28], tmp[29], tmp[30], tmp[31],
    //     tmp[32], tmp[33], tmp[34], tmp[35], tmp[36], tmp[37], tmp[38], tmp[30],
    //     tmp[40], tmp[41], tmp[42], tmp[43], tmp[44], tmp[45], tmp[46], tmp[47],
    //     tmp[48], tmp[49], tmp[50], tmp[51], tmp[52], tmp[53], tmp[54], tmp[55],
    //     tmp[56], tmp[57], tmp[58], tmp[59], tmp[60], tmp[61], tmp[62], tmp[63]
    //     );
    // }

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) 
                                + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) + ni * 32;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ++ei) {
                    // wangaq debug
                    // if(thread(0, 0)) {
                    //     printf("mi:%d ni:%d ei:%d row:%d col:%d acc_o:%.4f\n", 
                    //     mi, ni, ei, row, col, acc_o(ei, mi, ni));
                    // }
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                    }
                    //  else 
                    //     gO(row, col) = Element(0.0);
                    col += 4;
                }
            }
        }
    } 
#endif
}


template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void sparse_attn_1rowblock_sla_fp8(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementOUT = typename Kernel_traits::ElementO;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    const index_t binfo_o_offset =  binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = 0;
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));  // (kBlockM, kHeadDim)128,128

    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(0, 0));  // (kBlockN, kHeadDim, nblocksN)


    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(0, 0));  // (kBlockN, kHeadDim, nblocksN)
    const auto gK_data = gK.data();
    const auto gV_data = gV.data();
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
        Shape<Int<kBlockM>, Int<kBlockN>>{}, make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + 8192, typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    // Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});

    typename Kernel_traits::TiledMma16x64  tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(gK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);
   
    //
    // Copy Atom retiling
    //

    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    // auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol_fp8<_64x64, 128/64>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x32_B8, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVt);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_fp8<_32x128>(tOsVt8x64.layout()));



    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    
    int n_block = n_block_max - 1;

    int num_blks = params.block_count[(bidb * params.h + bidh) * params.NUM_ROWS + m_block];
    auto* blks_ptr = params.block_offset + ((bidb * params.h + bidh) * params.NUM_ROWS + m_block) * params.NNZ_S;
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    flash::Mask<false, false, false> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, 0.0);

    if (num_blks <= 0) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementOUT*>(params.o_ptr) + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<ElementOUT>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    float q_descale = params.q_descale_ptr == nullptr ? 1.0f : params.q_descale_ptr[bidb * params.q_descale_batch_stride + bidh * params.q_descale_head_stride];
    float k_descale = params.k_descale_ptr == nullptr ? 1.0f : params.k_descale_ptr[bidb * params.k_descale_batch_stride + bidh * params.k_descale_head_stride];
    float v_descale = params.v_descale_ptr == nullptr ? 1.0f : params.v_descale_ptr[bidb * params.v_descale_batch_stride + bidh * params.v_descale_head_stride];

    float scale_softmax_log2 = params.scale_softmax_log2*q_descale*k_descale;
    float scale_softmax = params.scale_softmax*q_descale*k_descale;

    constexpr int n_masking_steps = 1;
    int block_index = num_blks - 1;
    int actual_block = blks_ptr[block_index];
    gK.data() = gK_data + actual_block * int64_t(params.k_row_stride);
    gV.data() = gV_data + actual_block * int64_t(params.v_row_stride);
    
    Tensor tCrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
    Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);

   
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _32x128>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _32x128>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);



    #if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps && block_index >= 0; ++masking_step, --block_index) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        s_waitcnt<3>();

        cute::copy(smem_tiled_copy_K, tSsK(_, _, 0), tCrK_copy_view(_, _, 0));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 0), tSrK(_, _, 0), acc_s_ori);

        s_waitcnt<2>();
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 1), tCrK_copy_view(_, _, 1));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 1), tSrK(_, _, 1), acc_s_ori);

        s_barrier();


        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
      
        #if 1
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<false, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

    
        softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/false>(acc_s, acc_o, scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type_fp8<Element>(acc_s);

        s_waitcnt<1>();
       
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 0), tOrVt_copy_view(_, _, 0));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 0), tOrVt(_, _, 0), acc_o);

        s_waitcnt<0>();

        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 1), tOrVt_copy_view(_, _, 1));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 1), tOrVt(_, _, 1), acc_o);

        s_barrier();
     
    
        if (block_index > 0) {
            actual_block = blks_ptr[block_index - 1];
            gK.data() = gK_data + actual_block * int64_t(params.k_row_stride);
            gV.data() = gV_data + actual_block * int64_t(params.v_row_stride);

            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 0, params.v_row_stride, params.d_value);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 1, params.v_row_stride, params.d_value);
        }

        #endif

    }
    #endif

    #if 1
    #pragma unroll
    for (; block_index >= 0; --block_index) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        s_waitcnt<3>();

        cute::copy(smem_tiled_copy_K, tSsK(_, _, 0), tCrK_copy_view(_, _, 0));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 0), tSrK(_, _, 0), acc_s_ori);

        s_waitcnt<2>();
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 1), tCrK_copy_view(_, _, 1));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 1), tSrK(_, _, 1), acc_s_ori);

        s_barrier();

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));


        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false>(acc_s, acc_o, scale_softmax_log2);
      
        Tensor rP = flash::convert_type_fp8<Element>(acc_s);


        s_waitcnt<1>();
      
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 0), tOrVt_copy_view(_, _, 0));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 0), tOrVt(_, _, 0), acc_o);

        s_waitcnt<0>();
  
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 1), tOrVt_copy_view(_, _, 1));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 1), tOrVt(_, _, 1), acc_o);

        s_barrier();

        if (block_index > 0) {
            const int actual_block = blks_ptr[block_index - 1];
            gK.data() = gK_data + actual_block * int64_t(params.k_row_stride);
            gV.data() = gV_data + actual_block * int64_t(params.v_row_stride);

            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 0, params.v_row_stride, params.d_value);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 1, params.v_row_stride, params.d_value);
        }
       
       
    }
    #endif

    #if 1
    // Epilogue
    Tensor lse = softmax.template normalize_softmax_lse_fp8<false>(acc_o, scale_softmax,v_descale, params.rp_dropout);
    
 

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementOUT*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    using result_type = cutlass::Array<bfloat16_t, 2>;
    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16)*4 + ni * 32;
                {
                    auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(0, mi, ni),  acc_o(1, mi, ni), false);
                    auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(2, mi, ni),   acc_o(3, mi, ni), false);
                    auto res0 = reinterpret_cast<result_type const &>(d0);
                    auto res1 = reinterpret_cast<result_type const &>(d1);

                    gO(row, col)     = res0[0];
                    gO(row, col + 1) = res0[1];
                    gO(row, col + 2) = res1[0];
                    gO(row, col + 3) = res1[1];

                    col += 16;
                }
              
                {
                    auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(4, mi, ni), acc_o(5, mi, ni), false);
                    auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(6, mi, ni), acc_o(7, mi, ni), false);
                    auto res0 = reinterpret_cast<result_type const &>(d0);
                    auto res1 = reinterpret_cast<result_type const &>(d1);
                    gO(row, col)     = res0[0];
                    gO(row, col + 1) = res0[1];
                    gO(row, col + 2) = res1[0];
                    gO(row, col + 3) = res1[1];
                }
                
               

            }
        }
    } 
  
    #endif
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_sparse_attn(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.z;
    // The block index for the head.
    const int bidh = blockIdx.y;

    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.
#if (defined(__gfx936__) || defined(__gfx938__) )
    flash::sparse_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block);
#endif
}

#if 1
template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void compute_sparse_attn_sla(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.z;
    // The block index for the head.
    const int bidh = blockIdx.y;
    #if (defined(__gfx936__) || defined(__gfx938__) )
    if constexpr (Kernel_traits::kHeadDim == 128) {
        flash::sparse_attn_1rowblock_sla<Kernel_traits, Is_even_MN, Is_even_K, Return_softmax>(params, bidb, bidh, m_block);
    } else if constexpr (Kernel_traits::kHeadDim == 64) {
        flash::sparse_attn_1rowblock_sla_dim64<Kernel_traits, Is_even_MN, Is_even_K, Return_softmax>(params, bidb, bidh, m_block);
    }
    #endif
}
#endif

#if 1
template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void compute_sparse_attn_sla_fp8(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.z;
    // The block index for the head.
    const int bidh = blockIdx.y;
    #if defined(__gfx938__) ||defined(__gfx92a__)
    flash::sparse_attn_1rowblock_sla_fp8<Kernel_traits, Is_even_MN, Is_even_K, Return_softmax>(params, bidb, bidh, m_block);
    #endif
}
#endif

} // namespace FLASH_NAMESPACE