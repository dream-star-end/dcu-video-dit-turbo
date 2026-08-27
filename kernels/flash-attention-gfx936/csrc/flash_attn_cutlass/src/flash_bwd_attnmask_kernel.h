/******************************************************************************
 * Copyright (c) 2026, Attnmask extension.
 * Backward kernel for attention with explicit mask support.
 * 
 * This file contains the backward kernels modified to support explicit attention masks.
 * The key modification is applying the attention mask when recomputing S = QK^T,
 * before the softmax (scale_apply_exp2), to ensure P = 0 at masked positions.
 ******************************************************************************/

#pragma once

#include "flash_bwd_kernel.h"
#include "flash_attnmask.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// dQ computation with attention mask support (dim128 prefetch version)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, 
         bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Use_mask, typename Params>
inline __device__ void compute_dq_1rowblock_16x64_prefetch_attnmask(const Params &params, const int bidb, const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int warpId = __builtin_amdgcn_readfirstlane(tidx / 64);
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kStages = Kernel_traits::kStages;

    using SdP_TiledShape_MNK = typename Kernel_traits::TiledMmaSdP::TiledShape_MNK;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(SdP_TiledShape_MNK{}))::value;;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    // ============ Attention Mask initialization ============
    bool* mask_ptr = Use_mask ? reinterpret_cast<bool*>(params.mask_ptr) 
        + bidb * params.mask_batch_stride + bidh * params.mask_head_stride : nullptr;
    // ======================================================

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + m_block * kBlockM;
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + m_block * kBlockM;
    
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    // ============ Attention Mask tensor setup ============
    Tensor mM = make_tensor(mask_ptr,
                            make_shape(binfo.actual_seqlen_q, binfo.actual_seqlen_k),
                            make_stride(params.mask_seq_q_stride, _1{}));
    Tensor gM = local_tile(mM, Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block, _));

    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKGemm0{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKGemm1transposed{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutVGemm0{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(gQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(gdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);

    // dQ
    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKt);

    auto gmem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_QdO = gmem_tiled_copy_QdO.get_thread_slice(tidx);

    Tensor tSgQ = gmem_thr_copy_QdO.partition_S(gQ);
    Tensor tdPgdO = gmem_thr_copy_QdO.partition_S(gdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sVtemp = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor tdPsVBLayout = smem_thr_copy_BLayout.partition_S(sVtemp);
    Tensor tdPsV = make_tensor(tdPsVBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDim/32>(tdPsVBLayout.layout()));

    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDim/32>(tSsKBLayout.layout()));

    auto smem_tiled_copy_Kt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16_WITH_8x64, Element>{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt8x64 = smem_thr_copy_Kt.partition_S(sKt);
    Tensor tdQsKt = make_tensor(tdQsKt8x64.data(), convert_layout_B_rowcol<_16x128>(tdQsKt8x64.layout()));

    // PREDICATES
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor cdO = make_identity_tensor(make_shape(size<0>(gdO), size<1>(gdO)));
    Tensor tQcQ = gmem_thr_copy_QdO.partition_D(cQ);
    Tensor tdOcdO = gmem_thr_copy_QdO.partition_D(cdO);

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdPgdO)));
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdOpdO); ++k) { tdOpdO(k) = get<1>(tdOcdO(0, 0, k)) < params.d; }
    }

    // ============ Attention Mask partition ============
    Tensor tSgM = thr_mma_sdp.partition_C(gM(_, _, n_block_max > 0 ? n_block_max - 1 : 0));
    Tensor tSrM = make_fragment_like<uint8_t>(tSgM);
    clear(tSrM);
    
    // Identity tensor for mask predicates
    // gM shape: [kBlockM, kBlockN], get<0> is Q direction, get<1> is K direction

    // Prologue
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
        
        Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.dq_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
        Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
        clear(tdQrdQ);
        Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
        Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
        if constexpr(!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
        }
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        return;
    }

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tSgQ, tSrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tdPgdO, tdPrdO, tdOcdO, tdOpdO, binfo.actual_seqlen_q - m_block * kBlockM);

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);
    Tensor taccScS_row = taccScS(0, _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    Tensor dP_sum = make_fragment_like(lse);
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

    int n_block = n_block_max - 1;

    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tdPsV);
    constexpr int k2_loops = size<2>(tdQsKt);
    static_assert(kStages <= k0_loops && kStages <= k1_loops && kStages <= k2_loops , "kStages is error");

    #pragma unroll
    for (int i = 0; i < kStages; i++) {
        lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }

    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_dq);

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    // ============ Pre-read mask for first iteration ============
    if constexpr (Use_mask) {
        if (n_block >= n_block_min) {
            tSgM = thr_mma_sdp.partition_C(gM(_, _, n_block));
            cute::copy(tSgM, tSrM);
        }
    }

    #pragma unroll
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; i++) {
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, i);
            asm volatile("s_barrier");
        }
        #pragma unroll
        for (int i = 0; i < kStages; i++) {
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gV, sV, i, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k0_loops - kStages + i);
            asm volatile("s_barrier");
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), flash::convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }

        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

        [[maybe_unused]] Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            // Compute dtanh before masking to avoid -inf -> NaN in backward
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }

        // ============ Apply attention mask ============
        // Apply mask BEFORE alibi and causal masking, after softcap
        flash::apply_atten_mask<Use_mask>(tSrM, acc_s_ori, params.masked_value);

        #if 1
        if constexpr (Has_alibi) {
            const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int col_idx_offset = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = AtomLayoutMS * 16;
            alibi.apply_alibi_continuous(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }
        #endif

        #if 1
        if constexpr (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                const int col_idx_offset_ = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
                flash::apply_mask_continuous(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if constexpr (Is_causal) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_causal_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        AtomLayoutMS * 16);
            }
        } else if constexpr (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_local_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }
        #endif

        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);

        #if 1
        if constexpr (Is_dropout) {
            const int wave_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            if constexpr (kHeadDim==128){
                dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, m_block * kBlockM, block_col_idx, AtomLayoutMS * 16
                );
            }else{
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, block_row_idx, block_col_idx, AtomLayoutMS * 16
                );
            }
        }
        #endif

        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_dp_ori);

        #pragma unroll
        for (int i = 0; i < k1_loops - kStages; i++) {
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gV, sV, kStages + i, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, i);
            asm volatile("s_barrier");
        }

        #pragma unroll
        for (int i = 0; i < kStages; i++) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gK, sKt, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k1_loops - kStages +  i);
            asm volatile("s_barrier");
        }

        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
#if 1
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            } 
        }
#endif

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        Tensor tdQrdS = flash::convert_type<Element>(dS_reshaped);
        #pragma unroll
        for (int i = 0; i < k2_loops - kStages; i++) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gK, sKt, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, i);
            asm volatile("s_barrier");
        }

        if (n_block > n_block_min) {
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            
            // ============ Pre-read mask for next iteration ============
            if constexpr (Use_mask) {
                tSgM = thr_mma_sdp.partition_C(gM(_, _, n_block - 1));
                cute::copy(tSgM, tSrM);
            }
            
            #pragma unroll 
            for (int i = 0; i < kStages; i++) {
                lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
                asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
                flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, k2_loops - kStages + i);
                asm volatile("s_barrier");
            }
        }
        else if (kStages == 3){
            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, 1);
            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, 2);
            asm volatile("s_barrier");
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, 3);
            asm volatile("s_barrier");
        } else {
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, k2_loops - kStages + i);
                asm volatile("s_barrier");
            }
        }
    }

    // Epilogue: write dQ
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));

    using GmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    auto gmem_tiled_copy_dQ = make_tiled_copy_C(GmemCopyAtom{}, tiled_mma_dq);
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    Tensor taccdQrdQ = gmem_thr_copy_dQ.retile_S(acc_dq);
    Tensor taccdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);

    #pragma unroll
    for (int m = 0; m < size<1>(taccdQrdQ); m++) {
        if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int k = 0; k < size<2>(taccdQrdQ); k++) {                
                const int col_id = get<1>(tdQcdQ(0, 0, k));
                for (int i = 0; i < size<0>(taccdQrdQ); i++) {          
                    if (Is_even_K || col_id + i * 4 < params.d) {
                        taccdQgdQ(i, m, k) = flash::convert_type<Element>(taccdQrdQ(i, m, k) * params.scale_softmax_rp_dropout);
                    }   
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// dQ computation with attention mask support (dim64 version)
// Based on compute_dq_1rowblock_16x64_dim64_prefetch with mask logic inserted.
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi,
         bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Use_mask, typename Params>
inline __device__ void compute_dq_1rowblock_16x64_dim64_prefetch_attnmask(
        const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    const int warpId = __builtin_amdgcn_readfirstlane(tidx / 64);
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    using SdP_TiledShape_MNK = typename Kernel_traits::TiledMmaSdP::TiledShape_MNK;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(SdP_TiledShape_MNK{}))::value;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    // ============ Attention Mask initialization ============
    bool* mask_ptr = Use_mask ? reinterpret_cast<bool*>(params.mask_ptr)
        + bidb * params.mask_batch_stride + bidh * params.mask_head_stride : nullptr;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse
        ? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        : (bidb * params.h + bidh) * params.seqlen_q) + m_block * kBlockM;
    const index_t row_offset_dpsum = (params.unpadded_lse
        ? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb
        : (bidb * params.h + bidh) * params.seqlen_q_rounded) + m_block * kBlockM;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    // ============ Attention Mask tensor setup ============
    Tensor mM = make_tensor(mask_ptr,
                            make_shape(binfo.actual_seqlen_q, binfo.actual_seqlen_k),
                            make_stride(params.mask_seq_q_stride, _1{}));
    Tensor gM = local_tile(mM, Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block, _));

    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKGemm0{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKGemm1transposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKGemm1transposedNoSwizzle{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutVGemm0{});

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(gQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(gdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);

    auto gmem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_QdO = gmem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_QdO.partition_S(gQ);
    Tensor tdPgdO = gmem_thr_copy_QdO.partition_S(gdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sVtemp = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor tdPsVBLayout = smem_thr_copy_BLayout.partition_S(sVtemp);
    Tensor tdPsV = make_tensor(tdPsVBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDim/32>(tdPsVBLayout.layout()));
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDim/32>(tSsKBLayout.layout()));

    auto smem_tiled_copy_Kt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16_WITH_8x64, Element>{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt8x64 = smem_thr_copy_Kt.partition_S(sKt);
    Tensor tdQsKt = make_tensor(tdQsKt8x64.data(), convert_layout_B_rowcol<_16x64_64>(tdQsKt8x64.layout()));

    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor cdO = make_identity_tensor(make_shape(size<0>(gdO), size<1>(gdO)));
    Tensor tQcQ = gmem_thr_copy_QdO.partition_D(cQ);
    Tensor tdOcdO = gmem_thr_copy_QdO.partition_D(cdO);

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdPgdO)));
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdOpdO); ++k) { tdOpdO(k) = get<1>(tdOcdO(0, 0, k)) < params.d; }
    }

    // ============ Attention Mask partition ============
    Tensor tSgM = thr_mma_sdp.partition_C(gM(_, _, n_block_max > 0 ? n_block_max - 1 : 0));
    Tensor tSrM = make_fragment_like<uint8_t>(tSgM);
    clear(tSrM);


    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
        Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.dq_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
        Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
        clear(tdQrdQ);
        Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
        Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
        if constexpr(!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
        }
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        return;
    }

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tSgQ, tSrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tdPgdO, tdPrdO, tdOcdO, tdOpdO, binfo.actual_seqlen_q - m_block * kBlockM);

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);
    Tensor taccScS_row = taccScS(0, _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    Tensor dP_sum = make_fragment_like(lse);
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

    int n_block = n_block_max - 1;

    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tdPsV);
    constexpr int k2_loops = size<2>(tdQsKt);
    static_assert(k0_loops == 2 && k1_loops == 2 && k2_loops == 4 && kBlockN == 64, "kblockn should be 64");

    lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

    lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, 0, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, 1, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_dq);

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    // ============ Pre-read mask for first iteration ============
    if constexpr (Use_mask) {
        if (n_block >= n_block_min) {
            tSgM = thr_mma_sdp.partition_C(gM(_, _, n_block));
            cute::copy(tSgM, tSrM);
        }
    }

    #pragma unroll
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 0);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 1);
        asm volatile("s_barrier");

        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gK, sKt, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gK, sKt, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gK, sKt, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gK, sKt, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), flash::convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }

        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        [[maybe_unused]] Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }

        // ============ Apply attention mask ============
        flash::apply_atten_mask<Use_mask>(tSrM, acc_s_ori, params.masked_value);

        #if 1
        if constexpr (Has_alibi) {
            const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int col_idx_offset = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = AtomLayoutMS * 16;
            alibi.apply_alibi_continuous(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }
        #endif

        #if 1
        if constexpr (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                const int col_idx_offset_ = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
                flash::apply_mask_continuous(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if constexpr (Is_causal) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_causal_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        AtomLayoutMS * 16);
            }
        } else if constexpr (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_local_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }
        #endif

        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);

        #if 1
        if constexpr (Is_dropout) {
            const int wave_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS * 16
            );
        }
        #endif

        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_dp_ori);

        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 0);
        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 1);
        asm volatile("s_barrier");

        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));

        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };

        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        Tensor tdQrdS = flash::convert_type<Element>(dS_reshaped);

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        asm volatile("s_barrier");

        if (n_block > n_block_min) {
            gV.data() = gV.data() + (-int(kBlockN * params.k_row_stride));
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));

            // ============ Pre-read mask for next iteration ============
            if constexpr (Use_mask) {
                tSgM = thr_mma_sdp.partition_C(gM(_, _, n_block - 1));
                cute::copy(tSgM, tSrM);
            }

            lds_direct_copy<Is_even_K, true>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

            lds_direct_copy<Is_even_K, true>(gV, sV, 0, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true>(gV, sV, 1, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }
    }

    // Epilogue: write dQ
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;

    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));

    using GmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    auto gmem_tiled_copy_dQ = make_tiled_copy_C(GmemCopyAtom{}, tiled_mma_dq);
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    Tensor taccdQrdQ = gmem_thr_copy_dQ.retile_S(acc_dq);
    Tensor taccdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);

    #pragma unroll
    for (int m = 0; m < size<1>(taccdQrdQ); m++) {
        if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int k = 0; k < size<2>(taccdQrdQ); k++) {
                const int col_id = get<1>(tdQcdQ(0, 0, k));
                for (int i = 0; i < size<0>(taccdQrdQ); i++) {
                    if (Is_even_K || col_id + i * 4 < params.d) {
                        taccdQgdQ(i, m, k) = flash::convert_type<Element>(taccdQrdQ(i, m, k) * params.scale_softmax_rp_dropout);
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// dQ wrapper with attention mask support (dispatches by kHeadDim)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, 
         bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Use_mask, typename Params>
inline __device__ void compute_dq_seqq_parallel_16x64_prefetch_attnmask(const Params &params) {
    const int bidb = blockIdx.z;
    const int bidh = blockIdx.y;
    int m_block = blockIdx.x;

    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    if constexpr (kHeadDim == 128) {
        compute_dq_1rowblock_16x64_prefetch_attnmask<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
            Is_even_MN, Is_even_K, Is_softcap, Use_mask>(params, bidb, bidh, m_block);
        #ifndef NO_CAUSAL_OPT
        if constexpr (Is_causal) {
            const int num_blocks = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
            if (num_blocks - m_block - 1 != m_block) {
                compute_dq_1rowblock_16x64_prefetch_attnmask<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                    Is_even_MN, Is_even_K, Is_softcap, Use_mask>(params, bidb, bidh, num_blocks - m_block - 1);
            }
        }
        #endif
    } else if constexpr (kHeadDim == 64) {
        compute_dq_1rowblock_16x64_dim64_prefetch_attnmask<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
            Is_even_MN, Is_even_K, Is_softcap, Use_mask>(params, bidb, bidh, m_block);
        #ifndef NO_CAUSAL_OPT
        if constexpr (Is_causal) {
            const int num_blocks = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
            if (num_blocks - m_block - 1 != m_block) {
                compute_dq_1rowblock_16x64_dim64_prefetch_attnmask<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                    Is_even_MN, Is_even_K, Is_softcap, Use_mask>(params, bidb, bidh, num_blocks - m_block - 1);
            }
        }
        #endif
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// dK/dV computation with attention mask support (dim128 prefetch version)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, 
         bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Use_mask, typename Params>
inline __device__ void compute_dk_dv_trans_1colblock_16x64_prefetch_attnmask(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;

    constexpr int kSmemOffset = Kernel_traits::kSmemOffset;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    // ============ Attention Mask initialization ============
    bool* mask_ptr = Use_mask ? reinterpret_cast<bool*>(params.mask_ptr) 
        + bidb * params.mask_batch_stride + bidh * params.mask_head_stride : nullptr;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    if constexpr (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;
    
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(params.do_row_stride, _1{}));

    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    // ============ Attention Mask tensor setup (transposed for dK/dV) ============
    // Original mask layout: [seqlen_q, seqlen_k] with mask[q,k] indicating if q attends to k
    // In dK/dV: S = K @ Q^T has shape [kBlockN, kBlockM] i.e. [key, query]
    // So we need transposed view: [seqlen_k, seqlen_q] to match S layout
    // This way mask_transposed[k,q] = mask[q,k] aligns with S[k,q]
    Tensor mM = make_tensor(mask_ptr,
                            make_shape(binfo.actual_seqlen_k, binfo.actual_seqlen_q),
                            make_stride(_1{}, params.mask_seq_q_stride));
    // For dK/dV: fixed n_block (key block), varying m_block (query block)
    // gM shape is [kBlockN, kBlockM] to match S = K @ Q^T layout
    Tensor gM = local_tile(mM, Shape<Int<kBlockN>, Int<kBlockM>>{}, make_coord(n_block, _));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQGemm0{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQGemm1transposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQGemm1transposedNoSwizzle{});

    Tensor sdO = make_tensor(sQ.data() + kSmemOffset, typename Kernel_traits::SmemLayoutdOGemm0{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutdOGemm1transposed{});
    Tensor sdOtNoSwizzle = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutdOGemm1transposedNoSwizzle{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrK = thr_mma_sdp.partition_fragment_A(gK);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_B(sQ);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_A(gV);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_B(sdO);

    // dV/dK
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtNoSwizzle);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);

    // Copy Atom retiling
    auto gmem_tiled_copy_KV = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSgK = gmem_thr_copy_KV.partition_S(gK);
    Tensor tdPgV = gmem_thr_copy_KV.partition_S(gV);
    
    auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sQtemp = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQ{});
    Tensor tSsQBLayout = smem_thr_copy_BLayout.partition_S(sQtemp);
    Tensor tSsQ = make_tensor(tSsQBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDim/32>(tSsQBLayout.layout()));
    Tensor sdOtemp = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutdO{});
    Tensor tdPsdOBLayout = smem_thr_copy_BLayout.partition_S(sdOtemp);
    Tensor tdPsdO = make_tensor(tdPsdOBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDimV/32>(tdPsdOBLayout.layout()));

    auto smem_tiled_copy_QdOt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16_WITH_8x64, Element>{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt8x64 = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdVsdOt = make_tensor(tdVsdOt8x64.data(), convert_layout_B_rowcol<_16x128>(tdVsdOt8x64.layout()));
    Tensor tdKsQt8x64 = smem_thr_copy_QdOt.partition_S(sQt);
    Tensor tdKsQt = make_tensor(tdKsQt8x64.data(), convert_layout_B_rowcol<_16x128>(tdKsQt8x64.layout()));

    // PREDICATES
    Tensor cK = make_identity_tensor(make_shape(size<0>(gK), size<1>(gK)));
    Tensor cV = make_identity_tensor(make_shape(size<0>(gV), size<1>(gV)));
    Tensor tKcK = gmem_thr_copy_KV.partition_D(cK);
    Tensor tVcV = gmem_thr_copy_KV.partition_D(cV);

    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tSgK)));
    Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tdPgV)));

    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) < params.d_value; }
    }

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);

    // ============ Attention Mask partition for dK/dV ============
    // gM is now transposed: shape [kBlockN, kBlockM, num_m_tiles] matching S = K @ Q^T
    // gM(_, _, m_block) selects the m_block-th query tile, giving [kBlockN, kBlockM]
    // get<0> is K direction, get<1> is Q direction
    Tensor tSgM = thr_mma_sdp.partition_C(gM(_, _, m_block_max > 0 ? m_block_max - 1 : 0));
    Tensor tSrM = make_fragment_like<uint8_t>(tSgM);
    clear(tSrM);
    
    // Identity tensor for mask predicates (transposed layout)


    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                                make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdK = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));
        Tensor cdV = make_identity_tensor(make_shape(size<0>(gdV), size<1>(gdV)));
        Tensor tdKcdK = gmem_thr_copy_dKV.partition_D(cdK);
        Tensor tdVcdV = gmem_thr_copy_dKV.partition_D(cdV);
        Tensor tdKpdK = make_tensor<bool>(make_shape(size<2>(tdKcdK)));
        Tensor tdVpdV = make_tensor<bool>(make_shape(size<2>(tdVcdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKpdK); ++k) { tdKpdK(k) = get<1>(tdKcdK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdVpdV); ++k) { tdVpdV(k) = get<1>(tdVcdV(0, 0, k)) < params.d_value; }
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKcdK, tdKpdK, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdVcdV, tdVpdV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }
    
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tSgK, tSrK, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN
    );

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tdPgV, tdPrV, tVcV, tVpV, binfo.actual_seqlen_k - n_block * kBlockN
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);  
    
    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
        bidb, bidh, tidx, params.h);

    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDimV>>{});
    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    clear(acc_dv);
    clear(acc_dk);
    
    Tensor taccScS_row = taccScS(_, 0, _);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    constexpr int kS_loops = size<2>(tSsQ);
    constexpr int kdV_loops = size<2>(tdVsdOt);
    constexpr int kdP_loops = size<2>(tdPsdO);
    constexpr int kdK_loops = size<2>(tdKsQt);
    static_assert(kStages <= kS_loops && kStages <= kdV_loops && kStages <= kdP_loops && kStages <= kdK_loops, "kStages is error");
    
    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    }

    // ============ Pre-read mask for first iteration ============
    if constexpr (Use_mask) {
        if (m_block >= m_block_min) {
            tSgM = thr_mma_sdp.partition_C(gM(_, _, m_block));
            cute::copy(tSgM, tSrM);
        }
    }

    #pragma unroll
    for (; m_block >= m_block_min; m_block--) {

        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < kS_loops - kStages; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, kStages + i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, i);
            S_BARRIER;
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gdO, sdOt, i, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, kS_loops - kStages + i);
            S_BARRIER;
        }
        
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }

        Tensor scores_trans = make_tensor(acc_s.data(), flash::convert_trans_layout_acc_rowcol(acc_s.layout()));
        [[maybe_unused]] Tensor dtanh_trans = make_tensor_like(scores_trans);
        if constexpr (Is_softcap) {
            // Compute dtanh before masking to avoid -inf -> NaN in backward
            flash::calculate_dtanh(scores_trans, dtanh_trans, params.softcap);
        }

        // ============ Apply attention mask (transposed for dK/dV) ============
        // For dK/dV, S has shape [kBlockN, kBlockM] (transposed)
        // Apply mask AFTER softcap to ensure masked positions stay at -inf
        flash::apply_atten_mask<Use_mask>(tSrM, acc_s_ori, params.masked_value);

        #if 1
        if constexpr (Has_alibi) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            const int wave_id = tidx / 64;
            const int col_idx_offset =  m_block * kBlockM;
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            alibi.apply_alibi_trans(scores, col_idx_offset, row_idx_offset_, kNWarps * 16);
        }
        #endif

        #if 1
        if constexpr(!Is_causal && !Is_local) {
            if (!Is_even_MN && (m_block + 1) * kBlockM >= binfo.actual_seqlen_q) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int warp_id = tidx / 64;
                const int col_idx_offset_ = m_block * kBlockM;
                flash::apply_mask_trans(scores, binfo.actual_seqlen_q, col_idx_offset_);
            }
        } else if constexpr(Is_causal) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_causal_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16
                );
            }
        } else if constexpr(Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_local_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    params.window_size_left, params.window_size_right
                );
            }
        }
        #endif
        
        flash::scale_apply_exp2</*scale_max=*/false>(scores_trans, lse, params.scale_softmax_log2);

        Tensor dP_sum = make_fragment_like(lse);

        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
            dP_sum(mi) = gdPsum(row);
        }
        if (m_block > m_block_min) {
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
                lse(mi) = gLSE(row);
            }
            
            // ============ Pre-read mask for next iteration ============
            if constexpr (Use_mask) {
                tSgM = thr_mma_sdp.partition_C(gM(_, _, m_block - 1));
                cute::copy(tSgM, tSrM);
            }
        }

        if constexpr (Is_dropout) {
            const int warp_id = tidx / 64;
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = (kHeadDim == 128) ? (n_block * kBlockN) : (n_block * kBlockN + row_idx_offset_in_block);
            int block_row_idx = row_idx_offset_;
            int block_col_idx = m_block * kBlockM;
            if constexpr (kHeadDim==128){
                dropout.template apply_dropout_trans_opt</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, n_block * kBlockN, m_block * kBlockM, kNWarps * 16
                );
            }else{
                dropout.template apply_dropout_trans</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, block_row_idx, block_col_idx, kNWarps * 16
                );
            }
        }

        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);

        #pragma unroll
        for (int i = 0; i < kdV_loops - kStages; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gdO, sdOt, kStages + i, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
            S_WAITCNT;
            flash::gemm_k_rs(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, i);
            S_BARRIER;
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, i, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
            S_WAITCNT;
            flash::gemm_k_rs(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, kdV_loops - kStages + i);
            S_BARRIER;
        }

        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{});
        clear(acc_dp_ori);

        #pragma unroll
        for (int i = 0; i < kdP_loops - kStages; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, kStages + i, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
            S_WAITCNT;
            flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, i);
            S_BARRIER;
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gQ, sQt, i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            S_WAITCNT;
            flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, kdP_loops - kStages + i);
            S_BARRIER;
        }
        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        Tensor dS = make_tensor(acc_dp.data(), scores_trans.layout());

        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores_trans(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh_trans(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        Tensor tdKrdSt = flash::convert_type<Element>(acc_dp);

        #pragma unroll
        for (int i = 0; i < kdK_loops - kStages; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gQ, sQt, kStages + i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            S_WAITCNT;
            flash::gemm_k_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, i);
            S_BARRIER;
        }
        S_WAITCNT2;
        flash::gemm_k_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, 1);
        S_BARRIER;
        S_WAITCNT1;
        flash::gemm_k_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, 2);
        S_BARRIER;
        S_WAITCNT0;
        flash::gemm_k_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, 3);
        S_BARRIER;
        if (m_block > m_block_min) {
            gQ.data() = gQ.data() + (-int(kBlockM * params.q_row_stride));
            gdO.data() = gdO.data() + (-int(kBlockM * params.do_row_stride));
            lds_direct_copy<Is_even_K>(gQ, sQ, 0, params.q_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gQ, sQ, 1, params.q_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gQ, sQ, 2, params.q_row_stride, params.d);
        }
    }

    // Epilogue: write dK and dV
    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.dv_row_stride, _1{}));
    
    int row, col;
    if constexpr (size<1>(acc_dk) == size<1>(acc_dv) && size<2>(acc_dk) == size<2>(acc_dv)) {
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dk); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dk); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdK(row, col) = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                            gdV(row, col) = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout );
                        }
                        col += 4;
                    }
                }
            }
        } 
    } else {
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dk); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dk); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdK(row, col) = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                        }
                        col += 4;
                    }
                }
            }
        } 
        
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dv); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dv); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dv); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdV(row, col) = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout);
                        }
                        col += 4;
                    }
                }
            }
        } 
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// dK/dV computation with attention mask support (dim64 version)
// Based on compute_dk_dv_trans_1colblock_16x64_dim64_prefetch with mask logic inserted.
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi,
         bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Use_mask, typename Params>
inline __device__ void compute_dk_dv_trans_1colblock_16x64_dim64_prefetch_attnmask(
        const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    // ============ Attention Mask initialization ============
    bool* mask_ptr = Use_mask ? reinterpret_cast<bool*>(params.mask_ptr)
        + bidb * params.mask_batch_stride + bidh * params.mask_head_stride : nullptr;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    if constexpr (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse
        ? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        : (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    const index_t row_offset_dpsum = (params.unpadded_lse
        ? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb
        : (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(params.do_row_stride, _1{}));

    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    // ============ Attention Mask tensor setup (transposed for dK/dV) ============
    Tensor mM = make_tensor(mask_ptr,
                            make_shape(binfo.actual_seqlen_k, binfo.actual_seqlen_q),
                            make_stride(_1{}, params.mask_seq_q_stride));
    Tensor gM = local_tile(mM, Shape<Int<kBlockN>, Int<kBlockM>>{}, make_coord(n_block, _));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQGemm0{});
    Tensor sQt = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutQGemm1transposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQGemm1transposedNoSwizzle{});

    Tensor sdO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutdOGemm0{});
    Tensor sdOt = make_tensor(sdO.data() + size(sQ), typename Kernel_traits::SmemLayoutdOGemm1transposed{});
    Tensor sdOtNoSwizzle = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutdOGemm1transposedNoSwizzle{});

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrK = thr_mma_sdp.partition_fragment_A(gK);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_B(sQ);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_A(gV);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_B(sdO);

    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtNoSwizzle);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);

    auto gmem_tiled_copy_KV = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSgK = gmem_thr_copy_KV.partition_S(gK);
    Tensor tdPgV = gmem_thr_copy_KV.partition_S(gV);

    auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sQtemp = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQ{});
    Tensor tSsQBLayout = smem_thr_copy_BLayout.partition_S(sQtemp);
    Tensor tSsQ = make_tensor(tSsQBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDim/32>(tSsQBLayout.layout()));
    Tensor sdOtemp = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutdO{});
    Tensor tdPsdOBLayout = smem_thr_copy_BLayout.partition_S(sdOtemp);
    Tensor tdPsdO = make_tensor(tdPsdOBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDimV/32>(tdPsdOBLayout.layout()));

    auto smem_tiled_copy_QdOt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16_WITH_8x64, Element>{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt8x64 = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdVsdOt = make_tensor(tdVsdOt8x64.data(), convert_layout_B_rowcol<_16x64_64>(tdVsdOt8x64.layout()));
    Tensor tdKsQt8x64 = smem_thr_copy_QdOt.partition_S(sQt);
    Tensor tdKsQt = make_tensor(tdKsQt8x64.data(), convert_layout_B_rowcol<_16x64_64>(tdKsQt8x64.layout()));

    Tensor cK = make_identity_tensor(make_shape(size<0>(gK), size<1>(gK)));
    Tensor cV = make_identity_tensor(make_shape(size<0>(gV), size<1>(gV)));
    Tensor tKcK = gmem_thr_copy_KV.partition_D(cK);
    Tensor tVcV = gmem_thr_copy_KV.partition_D(cV);

    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tSgK)));
    Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tdPgV)));

    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) < params.d; }
    }

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);

    // ============ Attention Mask partition for dK/dV ============
    Tensor tSgM = thr_mma_sdp.partition_C(gM(_, _, m_block_max > 0 ? m_block_max - 1 : 0));
    Tensor tSrM = make_fragment_like<uint8_t>(tSgM);
    clear(tSrM);



    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                                make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdK = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));
        Tensor cdV = make_identity_tensor(make_shape(size<0>(gdV), size<1>(gdV)));
        Tensor tdKcdK = gmem_thr_copy_dKV.partition_D(cdK);
        Tensor tdVcdV = gmem_thr_copy_dKV.partition_D(cdV);
        Tensor tdKpdK = make_tensor<bool>(make_shape(size<2>(tdKcdK)));
        Tensor tdVpdV = make_tensor<bool>(make_shape(size<2>(tdVcdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKpdK); ++k) { tdKpdK(k) = get<1>(tdKcdK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdVpdV); ++k) { tdVpdV(k) = get<1>(tdVcdV(0, 0, k)) < params.d; }
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKcdK, tdKpdK, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdVcdV, tdVpdV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tSgK, tSrK, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN
    );
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tdPgV, tdPrV, tVcV, tVpV, binfo.actual_seqlen_k - n_block * kBlockN
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);

    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
        bidb, bidh, tidx, params.h);

    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDimV>>{});
    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    clear(acc_dv);
    clear(acc_dk);

    Tensor taccScS_row = taccScS(_, 0, _);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, 0, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, 1, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);

    lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gdO, sdOt, 0, params.do_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gdO, sdOt, 1, params.do_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gdO, sdOt, 2, params.do_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gdO, sdOt, 3, params.do_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);

    // ============ Pre-read mask for first iteration ============
    if constexpr (Use_mask) {
        if (m_block >= m_block_min) {
            tSgM = thr_mma_sdp.partition_C(gM(_, _, m_block));
            cute::copy(tSgM, tSrM);
        }
    }

    #pragma unroll
    for (; m_block >= m_block_min; m_block--) {

        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{});
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 0);
        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 1);
        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        Tensor scores_trans = make_tensor(acc_s.data(), flash::convert_trans_layout_acc_rowcol(acc_s.layout()));
        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        [[maybe_unused]] Tensor dtanh_trans = make_tensor_like(scores_trans);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores_trans, dtanh_trans, params.softcap);
        }

        // ============ Apply attention mask (transposed for dK/dV) ============
        flash::apply_atten_mask<Use_mask>(tSrM, acc_s_ori, params.masked_value);

        #if 1
        if constexpr (Has_alibi) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            const int wave_id = tidx / 64;
            const int col_idx_offset =  m_block * kBlockM;
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            alibi.apply_alibi_trans(scores, col_idx_offset, row_idx_offset_, kNWarps * 16);
        }
        #endif

        #if 1
        if constexpr(!Is_causal && !Is_local) {
            if (!Is_even_MN && (m_block + 1) * kBlockM >= binfo.actual_seqlen_q) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int col_idx_offset_ = m_block * kBlockM;
                flash::apply_mask_trans(scores, binfo.actual_seqlen_q, col_idx_offset_);
            }
        } else if constexpr(Is_causal) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_causal_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16
                );
            }
        } else if constexpr(Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_local_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    params.window_size_left, params.window_size_right
                );
            }
        }
        #endif

        flash::scale_apply_exp2</*scale_max=*/false>(scores_trans, lse, params.scale_softmax_log2);

        Tensor dP_sum = make_fragment_like(lse);

        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
            dP_sum(mi) = gdPsum(row);
        }
        if (m_block > m_block_min) {
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
                lse(mi) = gLSE(row);
            }

            // ============ Pre-read mask for next iteration ============
            if constexpr (Use_mask) {
                tSgM = thr_mma_sdp.partition_C(gM(_, _, m_block - 1));
                cute::copy(tSgM, tSrM);
            }
        }

        if constexpr (Is_dropout) {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            int block_row_idx = row_idx_offset_;
            int block_col_idx = m_block * kBlockM;
            dropout.template apply_dropout_trans</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, kNWarps * 16
            );
        }

        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);

        lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);

        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_barrier");

        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gQ, sQt, 0, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gQ, sQt, 1, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gQ, sQt, 2, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gQ, sQt, 3, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);

        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{});
        clear(acc_dp_ori);

        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 0);
        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 1);
        asm volatile("s_barrier");

        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        Tensor dS = make_tensor(acc_dp.data(), scores_trans.layout());

        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };

        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores_trans(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh_trans(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        Tensor tdKrdSt = flash::convert_type<Element>(acc_dp);
        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_barrier");

        if (m_block > m_block_min) {
            gQ.data() = gQ.data() + (-int(kBlockM * params.q_row_stride));
            gdO.data() = gdO.data() + (-int(kBlockM * params.do_row_stride));

            lds_direct_copy<Is_even_K, true>(gQ, sQ, 0, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            lds_direct_copy<Is_even_K, true>(gQ, sQ, 1, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);

            lds_direct_copy<Is_even_K, true, _16x64_64>(gdO, sdOt, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
            lds_direct_copy<Is_even_K, true, _16x64_64>(gdO, sdOt, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
            lds_direct_copy<Is_even_K, true, _16x64_64>(gdO, sdOt, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
            lds_direct_copy<Is_even_K, true, _16x64_64>(gdO, sdOt, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        }
    }

    // Epilogue: write dK and dV
    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.dv_row_stride, _1{}));

    int row, col;
    if constexpr (size<1>(acc_dk) == size<1>(acc_dv) && size<2>(acc_dk) == size<2>(acc_dv)) {
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dk); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dk); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdK(row, col) = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                            gdV(row, col) = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout );
                        }
                        col += 4;
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dk); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dk); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdK(row, col) = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                        }
                        col += 4;
                    }
                }
            }
        }

        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dv); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dv); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dv); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdV(row, col) = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout);
                        }
                        col += 4;
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// dK/dV wrapper with attention mask support (dispatches by kHeadDim)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, 
         bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Use_mask, typename Params>
inline __device__ void compute_dk_dv_trans_16x64_prefetch_attnmask(const Params &params) {
    const int bidb = blockIdx.z;
    const int bidh = blockIdx.y;
    const int n_block = blockIdx.x;

    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    if constexpr (kHeadDim == 128) {
        compute_dk_dv_trans_1colblock_16x64_prefetch_attnmask<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
            Is_even_MN, Is_even_K, Is_softcap, Use_mask>(params, bidb, bidh, n_block);
        #ifndef NO_CAUSAL_OPT
        if constexpr (Is_causal) {
            const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
            if (num_n_block - n_block - 1 != num_n_block) {
                compute_dk_dv_trans_1colblock_16x64_prefetch_attnmask<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                    Is_even_MN, Is_even_K, Is_softcap, Use_mask>(params, bidb, bidh, num_n_block - n_block - 1);
            }
        }
        #endif
    } else if constexpr (kHeadDim == 64) {
        compute_dk_dv_trans_1colblock_16x64_dim64_prefetch_attnmask<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
            Is_even_MN, Is_even_K, Is_softcap, Use_mask>(params, bidb, bidh, n_block);
    }
}

} // namespace flash
