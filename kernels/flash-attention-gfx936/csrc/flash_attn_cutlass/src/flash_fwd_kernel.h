/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"
#include "rotary.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ElementAccum, typename Params, int kBlockM, bool Is_even_MN>
__forceinline__ __device__ auto get_lse_tile(const Params &params, const int bidb, const int bidh, const int m_block, const BlockInfo</*Varlen=*/!Is_even_MN> &binfo) {
        // When params.unpadded_lse is false, LSE is written as (b, h, seqlen_q) - this is non-variable seqlen path.
        // Otherwise, when params.seqlenq_ngroups_swapped is true, it is written as (h, seqlen_q, b) to account for seqlen_q <-> h swapping trick.
        // Otherwise, it's written as (h, b, seqlen_q).
        const bool varlen_q = params.unpadded_lse && !params.seqlenq_ngroups_swapped;
        auto lse_offset = varlen_q ? binfo.q_offset(params.seqlen_q, 1, bidb) : 0;
        auto gmem_ptr_lse = make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset);

        auto lse_shape = varlen_q ? make_shape(1, params.h, params.total_q) : make_shape(params.b, params.h, params.seqlen_q);
        auto lse_stride = params.seqlenq_ngroups_swapped ? make_stride(1, params.seqlen_q * params.b, params.b) : (
            params.unpadded_lse ? make_stride(params.h * params.total_q, params.total_q, 1) :  make_stride(params.h * params.seqlen_q, params.seqlen_q, 1)
            );

        auto lse_layout = make_layout(lse_shape, lse_stride);
        Tensor mLSE = make_tensor(gmem_ptr_lse, lse_layout);
        auto mLSE_slice = varlen_q ? mLSE(0, bidh, _) : mLSE(bidb, bidh, _);
        return local_tile(mLSE_slice, Shape<Int<kBlockM>>{}, make_coord(m_block));
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
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
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo_k_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo_v_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    #ifdef GEMM1_AMATRIX_WITH_SMEM
    Tensor sAccs = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutAccs{});
    #endif
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    #ifdef GEMM1_AMATRIX_WITH_SMEM
    Tensor sV = make_tensor(size(sK) > size(sAccs) ? sK.data() + size(sK):
        sAccs.data() + size(sAccs),
        typename Kernel_traits::SmemLayoutKV{});
    #else
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutV{});
    #endif
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    #ifndef GEMM1_AMATRIX_WITH_SMEM
    typename Kernel_traits::TiledMma_FOR_GEMM1 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    #endif
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    #ifdef GEMM1_AMATRIX_WITH_SMEM
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)
    #else
    Tensor tOrVt  = thr_mma_for_gemm1.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)
    #endif

    Tensor tSgS  = thr_mma.partition_C(gP);

    #ifdef GEMM1_AMATRIX_WITH_SMEM
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K
    #else
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K
    #endif

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    #ifdef GEMM1_AMATRIX_WITH_SMEM
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    #else
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    #endif

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);
    if constexpr (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if constexpr (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    auto tKrK = make_fragment_like(tKsK);
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKrK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);

    if constexpr (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        // flash::cp_async_wait<1>();
        // __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    // if (cute::thread0()) {
    //     printf("n_masking_steps = %d\n", n_masking_steps);
    // }
    // return;
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        __syncthreads();
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s);
        // return;
        cute::copy(tKrK, tKsK);
        __syncthreads();
        auto tVrV = make_fragment_like(tVsV);

        if (masking_step > 0) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV);
        }
        else {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        // return;
        flash::gemm_rs(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K);
        
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        cute::copy(tVrV, tVsV);
        __syncthreads();
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int warp_row_stride = 16;

            const int block_row_idx = m_block * (kBlockM >> 4) + wave_id;
            const int block_col_idx = n_block * (kBlockN >> 4);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps
                );
                cute::copy(rP_drop, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
            }
        }

        #ifdef GEMM1_AMATRIX_WITH_SMEM
        Tensor tOrP = flash::convert_layout_acc_Aregs(tiled_mma, rP, sAccs);
        Tensor TOrVtCoal = make_tensor(tOrVt.data(), make_shape(size<0>(tOrVt),size<1>(tOrVt),size<2>(tOrVt)));
        flash::gemm_rs(acc_o, tOrP, TOrVtCoal, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // wangaq debug
        // if (thread0()) {
        //     printf("rP layout:"); print(rP.layout());printf("\n");
        //     printf("tOrP layout:"); print(tOrP.layout());printf("\n");
        //     printf("tOrVt layout:"); print(tOrVt.layout());printf("\n");
        // }
        // return;
        #else
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        #endif

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }
    #if 1
    // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        // if (n_block == n_block_max - 1) {
        __syncthreads();
        // }
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 

        clear(acc_s);
        cute::copy(tKrK, tKsK);
        __syncthreads();

        auto tVrV = make_fragment_like(tVsV);
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV);


        // flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
        //     acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        //     smem_thr_copy_Q, smem_thr_copy_K
        // );
        flash::gemm_rs(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K);
        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        cute::copy(tVrV, tVsV);
        __syncthreads();
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }

        softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int warp_row_stride = 16;
            const int block_row_idx = m_block * (kBlockM >> 4) + wave_id;
            const int block_col_idx = n_block * (kBlockN >> 4);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps
                );
                cute::copy(rP_drop, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
            }
        }

        #ifdef GEMM1_AMATRIX_WITH_SMEM
        Tensor tOrP = flash::convert_layout_acc_Aregs(tiled_mma, rP, sAccs);
        // Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        Tensor TOrVtCoal = make_tensor(tOrVt.data(), make_shape(size<0>(tOrVt),size<1>(tOrVt),size<2>(tOrVt)));

        flash::gemm_rs(acc_o, tOrP, TOrVtCoal, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        #else
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        #endif
    }

    // Epilogue

    // ★ Attention Sinks: conditional normalize (direct global memory load) ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) { tSrS_aux(mi) = s_aux_val; }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }
    // if (cute::thread0())
    // {
    //     // for (int i = 0; i < size(acc_o); i++) {
    //     //     printf("i = %d acc_o = %f\n", i, float(acc_o[i]));
    //     // }
    //         // print_tensor(tOsVt);
    // }
    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    // Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    // CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
    #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
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
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo_k_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo_v_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + (Kernel_traits::Share_K_V_smem ? 0 : size(sK)), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    using Tensor_tSrQ = decltype(thr_mma.partition_fragment_A(sQ));
    using Tensor_tGrQ = decltype(thr_mma.partition_fragment_A(gQ));
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    Tensor_tSrQ tSrQ;
    Tensor_tGrQ tGrQ;
    if constexpr(Kernel_traits::Is_Q_use_smem) {
        tSrQ = thr_mma.partition_fragment_A(sQ);
        Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

        auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
        
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
        Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

        // Set predicates for k bounds
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
        }

        
        // printf("tid:%d m:%d %d %d %d n:%d max_MN:%d Is_even_MN:%d\n", tidx, get<0>(tQcQ(0, 0, 0)), get<0>(tQcQ(0, 1, 0)), 
        //     get<0>(tQcQ(0, 2, 0)), get<0>(tQcQ(0, 3, 0)), get<0>(tKVcKV(0, 0, 0)), binfo.actual_seqlen_q - m_block * kBlockM, Is_even_MN);
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);
        
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view)); 
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        if constexpr(Kernel_traits::MMA_Atom_Use_K32) { asm volatile("s_waitcnt lgkmcnt(0)\n\t" : :); } // for 16x64x32
    } else {
        tGrQ = thr_mma.partition_fragment_A(gQ);
        auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
        Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);
        
        Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
        Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

        // Set predicates for k bounds
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
        }

        // printf("tid:%d m:%d %d %d %d n:%d max_MN:%d Is_even_MN:%d\n", tidx, get<0>(tQcQ(0, 0, 0)), get<0>(tQcQ(0, 1, 0)), 
        //     get<0>(tQcQ(0, 2, 0)), get<0>(tQcQ(0, 3, 0)), get<0>(tKVcKV(0, 0, 0)), binfo.actual_seqlen_q - m_block * kBlockM, Is_even_MN);
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);
    }
    __syncthreads();

    // Prologue

    int n_block = n_block_max - 1;
    auto tKrK = make_fragment_like(tKsK);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKrK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        __syncthreads();
        cute::copy(tKrK, tKsK);

        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        auto tVrV = make_fragment_like(tVsV);
        if (masking_step > 0) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV);
        }
        else {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        __syncthreads();
        if constexpr(Kernel_traits::Is_Q_use_smem) {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs_swait(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        } else {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs_swait(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
                // flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        __syncthreads();
        cute::copy(tVrV, tVsV);
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

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

        __syncthreads();
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        __syncthreads();
        cute::copy(tKrK, tKsK);
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 

        clear(acc_s_ori);

        auto tVrV = make_fragment_like(tVsV);
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV);

        __syncthreads();
        if constexpr(Kernel_traits::Is_Q_use_smem) {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs_swait(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        } else {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs_swait(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
                // flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }
        
        __syncthreads();
        cute::copy(tVrV, tVsV);
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }

        softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);
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

        __syncthreads();
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    // ★ Attention Sinks: conditional normalize (direct global memory load) ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) { tSrS_aux(mi) = s_aux_val; }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    if constexpr(Kernel_traits::Is_Q_use_smem || !Kernel_traits::Share_K_V_smem) {
        Tensor sO = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
        // Partition sO to match the accumulator partitioning
        auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_for_gemm1);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        __syncthreads();

        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                            + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_coord(m_block, 0));  // (kBlockM, kHeadDim)
        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

        __syncthreads();

        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        // static_assert(decltype(size<0>(taccOcO))::value == 4);
        // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
        // Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
        // CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }

        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
    } else {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        auto gmem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_for_gemm1);
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor taccOrO = gmem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOgO = gmem_thr_copy_O.partition_D(gO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }

        // Construct identity layout for gO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(taccOgO)));
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, taccOrO, taccOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_dim64(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
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
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo_k_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo_v_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + (Kernel_traits::Share_K_V_smem ? 0 : size(sK)), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    using Tensor_tSrQ = decltype(thr_mma.partition_fragment_A(sQ));
    using Tensor_tGrQ = decltype(thr_mma.partition_fragment_A(gQ));
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    Tensor_tSrQ tSrQ;
    Tensor_tGrQ tGrQ;
    if constexpr(Kernel_traits::Is_Q_use_smem) {
        tSrQ = thr_mma.partition_fragment_A(sQ);
        Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

        auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
        
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
        Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

        // Set predicates for k bounds
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
        }

        
        // printf("tid:%d m:%d %d %d %d n:%d max_MN:%d Is_even_MN:%d\n", tidx, get<0>(tQcQ(0, 0, 0)), get<0>(tQcQ(0, 1, 0)), 
        //     get<0>(tQcQ(0, 2, 0)), get<0>(tQcQ(0, 3, 0)), get<0>(tKVcKV(0, 0, 0)), binfo.actual_seqlen_q - m_block * kBlockM, Is_even_MN);
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);
        
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view)); 
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        if constexpr(Kernel_traits::MMA_Atom_Use_K32) { asm volatile("s_waitcnt lgkmcnt(0)\n\t" : :); } // for 16x64x32
    } else {
        tGrQ = thr_mma.partition_fragment_A(gQ);
        auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
        Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);
        
        Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
        Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

        // Set predicates for k bounds
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
        }

        // printf("tid:%d m:%d %d %d %d n:%d max_MN:%d Is_even_MN:%d\n", tidx, get<0>(tQcQ(0, 0, 0)), get<0>(tQcQ(0, 1, 0)), 
        //     get<0>(tQcQ(0, 2, 0)), get<0>(tQcQ(0, 3, 0)), get<0>(tKVcKV(0, 0, 0)), binfo.actual_seqlen_q - m_block * kBlockM, Is_even_MN);
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);
    }
    __syncthreads();

    // Prologue

    int n_block = n_block_max - 1;
    auto tKrK = make_fragment_like(tKsK);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKrK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        __syncthreads();
        cute::copy(tKrK, tKsK);
        __syncthreads();
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        auto tVrV = make_fragment_like(tVsV);
        if (masking_step > 0) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV);
        }
        else {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        // __syncthreads();
        if constexpr(Kernel_traits::Is_Q_use_smem) {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        } else {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
                // flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        __syncthreads();
        cute::copy(tVrV, tVsV);
        __syncthreads();

        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
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

        // __syncthreads();
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        // __syncthreads();
        cute::copy(tKrK, tKsK);
        __syncthreads();

        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 

        clear(acc_s_ori);

        auto tVrV = make_fragment_like(tVsV);
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV);

        // __syncthreads();
        if constexpr(Kernel_traits::Is_Q_use_smem) {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        } else {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
                // flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }
        
        // __syncthreads();
        cute::copy(tVrV, tVsV);
        __syncthreads();

        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }

        softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

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

        // __syncthreads();
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

    }

    // Epilogue

    // ★ Attention Sinks: conditional normalize (direct global memory load) ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) { tSrS_aux(mi) = s_aux_val; }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    if constexpr(Kernel_traits::Is_Q_use_smem || !Kernel_traits::Share_K_V_smem || kHeadDim == 64) {
        Tensor sO = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
        // Partition sO to match the accumulator partitioning
        auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_for_gemm1);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        __syncthreads();

        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                            + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_coord(m_block, 0));  // (kBlockM, kHeadDim)
        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

        __syncthreads();

        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        // static_assert(decltype(size<0>(taccOcO))::value == 4);
        // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
        // Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
        // CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }

        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
    } else {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        auto gmem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_for_gemm1);
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor taccOrO = gmem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOgO = gmem_thr_copy_O.partition_D(gO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }

        // Construct identity layout for gO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(taccOgO)));
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, taccOrO, taccOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_mla(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {
// #if 1
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
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
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo_k_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo_v_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d_value),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + (Kernel_traits::Share_K_V_smem ? 0 : size(sK)), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // if (thread0())
    // {
    //     print("tVgV "); print(tVgV); print("\n");
    //     print("tVsV "); print(tVsV); print("\n");

    // }
#if 1
    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    using Tensor_tSrQ = decltype(thr_mma.partition_fragment_A(sQ));
    using Tensor_tGrQ = decltype(thr_mma.partition_fragment_A(gQ));
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(sV), size<1>(sV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tVcV = gmem_thr_copy_QKV.partition_S(cV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tVsV)));

    Tensor_tSrQ tSrQ;
    Tensor_tGrQ tGrQ;
    if constexpr(Kernel_traits::Is_Q_use_smem) {
        tSrQ = thr_mma.partition_fragment_A(sQ);
        Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

        auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
        
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
        Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

        // Set predicates for k bounds
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) < params.d_value; }
        }

        
        // printf("tid:%d m:%d %d %d %d n:%d max_MN:%d Is_even_MN:%d\n", tidx, get<0>(tQcQ(0, 0, 0)), get<0>(tQcQ(0, 1, 0)), 
        //     get<0>(tQcQ(0, 2, 0)), get<0>(tQcQ(0, 3, 0)), get<0>(tKVcKV(0, 0, 0)), binfo.actual_seqlen_q - m_block * kBlockM, Is_even_MN);
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);
        
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view)); 
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        if constexpr(Kernel_traits::MMA_Atom_Use_K32) { asm volatile("s_waitcnt lgkmcnt(0)\n\t" : :); } // for 16x64x32
    } else {
        tGrQ = thr_mma.partition_fragment_A(gQ);
        auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
        Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);
        
        Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
        Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

        // Set predicates for k bounds
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
            #pragma unroll
            for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) < params.d_value; }
        }

        // printf("tid:%d m:%d %d %d %d n:%d max_MN:%d Is_even_MN:%d\n", tidx, get<0>(tQcQ(0, 0, 0)), get<0>(tQcQ(0, 1, 0)), 
        //     get<0>(tQcQ(0, 2, 0)), get<0>(tQcQ(0, 3, 0)), get<0>(tKVcKV(0, 0, 0)), binfo.actual_seqlen_q - m_block * kBlockM, Is_even_MN);
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);
    }
    __syncthreads();

    // Prologue

    int n_block = n_block_max - 1;
    auto tKrK = make_fragment_like(tKsK);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKrK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        // __syncthreads();
        asm volatile("s_barrier\n\t");
        cute::copy(tKrK, tKsK);
        __syncthreads();
        // wangaq debug
        // __syncthreads();
        // if (cute::thread(0, 0)) {
        //     __half * tmp = reinterpret_cast<__half*>(sK.data().get());
        //     int col = 8;
        //     for (int i = 0; i < size(sK)/col; ++i) {
        //         printf("K %3d: ", i);
        //         for (int j = 0; j < col; ++j) {
        //             printf("%10.4f ", __half2float(tmp[i*col+j]));
        //         }
        //         printf("\n");
        //     }
        // }
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        auto tVrV = make_fragment_like(tVsV);
        if (masking_step > 0) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tVcV, tVpV);
        }
        else {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tVcV, tVpV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        // __syncthreads();
        if constexpr(Kernel_traits::Is_Q_use_smem) {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        } else {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
                // flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        // __syncthreads();
        asm volatile("s_barrier\n\t");
        cute::copy(tVrV, tVsV);
        __syncthreads();

        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
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

        // __syncthreads();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        // 
        asm volatile("s_barrier\n\t");
        cute::copy(tKrK, tKsK);
        __syncthreads();
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 

        clear(acc_s_ori);

        auto tVrV = make_fragment_like(tVsV);
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tVcV, tVpV);

        // __syncthreads();
        if constexpr(Kernel_traits::Is_Q_use_smem) {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        } else {
            if constexpr(Kernel_traits::MMA_Atom_Use_K16) {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x16
            } else {
                flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
                // flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K); // for 16x64x32
            }
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }
        
        // __syncthreads();
        asm volatile("s_barrier\n\t");
        cute::copy(tVrV, tVsV);
        __syncthreads();
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }

        softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

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

        // __syncthreads();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    if constexpr(Kernel_traits::Is_Q_use_smem || !Kernel_traits::Share_K_V_smem || true) {
        Tensor sO = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
        // Partition sO to match the accumulator partitioning
        auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_for_gemm1);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // __syncthreads();
        asm volatile("s_barrier\n\t");

        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                            + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_coord(m_block, 0));  // (kBlockM, kHeadDim)
        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

        __syncthreads();

        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }

        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
    } else {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        auto gmem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_for_gemm1);
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor taccOrO = gmem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOgO = gmem_thr_copy_O.partition_D(gO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }

        // Construct identity layout for gO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(taccOgO)));
        if constexpr (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, taccOrO, taccOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }
#endif
}

#define S_WAITCNT asm volatile("s_waitcnt vmcnt(3) \n s_barrier")
#define S_BARRIER asm volatile("s_barrier")
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_prefetch(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

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

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
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

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
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
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x128, kHeadDimV/32>(tOsVt8x64.layout()));

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
    // __syncthreads();
    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    static_assert(kStages <= k0_loops && kStages <= k1_loops, "kStages is error");
    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }
#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            s_waitcnt<3>();
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i);
            s_barrier();
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, i, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            s_waitcnt<3>();
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, k0_loops - kStages + i);
            s_barrier();
        }
        

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
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
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, kStages + 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        
        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 2
        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 3
        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            lds_direct_copy<Is_even_K>(gK, sK, kStages + i, params.k_row_stride, params.d);
            s_waitcnt<3>();
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i);
            s_barrier();
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128>(gV, sV, i, params.v_row_stride, params.d_value);
            s_waitcnt<3>();
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, k0_loops - kStages + i);
            s_barrier();
        }
        
        // __builtin_amdgcn_sched_barrier(1);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);
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
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128>(gV, sV, kStages + 0, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 2
        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        // k = 3
        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
        }
    }

    // Epilogue

    // ★ Attention Sinks: conditional normalize (direct global memory load) ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) {
            tSrS_aux(mi) = s_aux_val;
        }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
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
                    if constexpr (Is_even_K) {
                        col = (laneId / 16) * 2 + ni * 32;
                        using result_type = cutlass::Array<Element, 2>;
                        for (int ei = 0; ei < 4; ++ei)
                        {
                            result_type res;
                            res[0] = flash::convert_type<Element>(acc_o(ei, mi, ni));
                            res[1] = flash::convert_type<Element>(acc_o(ei + 4, mi, ni));
                            *(result_type*)(&gO(row, col)) = res;
                            col += 8;
                        }
                    } else {
                        col = (laneId / 16) * 2 + ni * 32;
                        if (col < params.d_value) {
                            using result_type = cutlass::Array<Element, 2>;
                            for (int ei = 0; ei < 4; ++ei)
                            {
                                result_type res;
                                res[0] = flash::convert_type<Element>(acc_o(ei, mi, ni));
                                res[1] = flash::convert_type<Element>(acc_o(ei + 4, mi, ni));
                                *(result_type*)(&gO(row, col)) = res;
                                col += 8;
                            }
                        }
                    }
                    // TODO dim!=128 with risk
                    //  else 
                    //     gO(row, col) = Element(0.0);
                    // col += 4;
                }
            }
        }
    } 
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_prefetch_fp8(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementOUT = typename Kernel_traits::ElementO;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

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

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementOUT*>(params.o_ptr)
                                              + binfo_o_offset),
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

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)128,128

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));//64,128

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutKV{});
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
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    float q_descale = params.q_descale_ptr == nullptr ? 1.0f : params.q_descale_ptr[bidb * params.q_descale_batch_stride + bidh * params.q_descale_head_stride];
    float k_descale = params.k_descale_ptr == nullptr ? 1.0f : params.k_descale_ptr[bidb * params.k_descale_batch_stride + bidh * params.k_descale_head_stride];
    float v_descale = params.v_descale_ptr == nullptr ? 1.0f : params.v_descale_ptr[bidb * params.v_descale_batch_stride + bidh * params.v_descale_head_stride];

    float scale_softmax_log2 = params.scale_softmax_log2*q_descale*k_descale;
    float scale_softmax = params.scale_softmax*q_descale*k_descale;

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    Tensor tCrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
    Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);

   
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _32x128>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _32x128>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);



#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(3) \n\t s_barrier\n\t");

        cute::copy(smem_tiled_copy_K, tSsK(_, _, 0), tCrK_copy_view(_, _, 0));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 0), tSrK(_, _, 0), acc_s_ori);

        asm volatile("s_waitcnt vmcnt(2) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 1), tCrK_copy_view(_, _, 1));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 1), tSrK(_, _, 1), acc_s_ori);

        asm volatile("s_barrier");


        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
      
        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

    
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type_fp8<Element>(acc_s);

      
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
                dropout.template apply_dropout_continuous_fp8</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back_fp8(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous_fp8(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
       
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 0), tOrVt_copy_view(_, _, 0));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 0), tOrVt(_, _, 0), acc_o);

        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");

        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 1), tOrVt_copy_view(_, _, 1));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 1), tOrVt(_, _, 1), acc_o);

        asm volatile("s_barrier");
     
    
        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 0, params.v_row_stride, params.d_value);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 1, params.v_row_stride, params.d_value);
        }

       
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif

    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

         asm volatile("s_waitcnt vmcnt(3) \n\t s_barrier\n\t");

        cute::copy(smem_tiled_copy_K, tSsK(_, _, 0), tCrK_copy_view(_, _, 0));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 0), tSrK(_, _, 0), acc_s_ori);

        asm volatile("s_waitcnt vmcnt(2) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 1), tCrK_copy_view(_, _, 1));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 1), tSrK(_, _, 1), acc_s_ori);

        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

       
        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, scale_softmax_log2);
      
        Tensor rP = flash::convert_type_fp8<Element>(acc_s);

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
                dropout.template apply_dropout_continuous_fp8</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back_fp8(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous_fp8(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
      
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 0), tOrVt_copy_view(_, _, 0));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 0), tOrVt(_, _, 0), acc_o);

        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
  
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 1), tOrVt_copy_view(_, _, 1));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 1), tOrVt(_, _, 1), acc_o);

        asm volatile("s_barrier");

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 0, params.v_row_stride, params.d_value);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 1, params.v_row_stride, params.d_value);
        }
       
       
    }

    // Epilogue
    Tensor lse = softmax.template normalize_softmax_lse_fp8<Is_dropout>(acc_o, scale_softmax,v_descale, params.rp_dropout);
    
 

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
                    auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(0, mi, ni), acc_o(1, mi, ni), false);
                    auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(2, mi, ni), acc_o(3, mi, ni), false);
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



template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_dim96_prefetch(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages - 1;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
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

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

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
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x96, kHeadDimV/32>(tOsVt8x64.layout()));

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
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    static_assert(kStages <= k0_loops && kStages <= k1_loops, "kStages is error");
    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }

#if 1
    #define BIDX 0
    #define BIDY 0
    #define BIDZ 0

    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<2>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<4>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);
        S_BARRIER;
        

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));


        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);


        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int wave_id = (tidx / 64);
            const int row_idx_offset_in_block = (tidx % 16) + (wave_id * 16);
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

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        
        
        s_waitcnt<6>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;
        
        s_waitcnt<4>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;
        // k = 2
        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;
        // k = 3
        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d);
            }
        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
        s_waitcnt<2>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        S_BARRIER;

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x96_multi_ins>(gV, sV, 0, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        S_BARRIER;

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x96_multi_ins>(gV, sV, 1, params.v_row_stride, params.d_value);
        s_waitcnt<4>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);
        S_BARRIER;
        
        // __builtin_amdgcn_sched_barrier(1);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

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

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x96_multi_ins>(gV, sV, 2, params.v_row_stride, params.d_value);
        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x96_multi_ins>(gV, sV, 3, params.v_row_stride, params.d_value);
        
        s_waitcnt<6>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        s_waitcnt<4>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;
        // k = 2
        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;
        // k = 3
        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d);
            }
        }
    }

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
    
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
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

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_dim32_prefetch(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

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

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
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

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutK{});
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
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    // Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x64_64, kHeadDimV/32>(tOsVt8x64.layout()));


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
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    // static_assert(k0_loops == 1 && k1_loops == 8);
    // if constexpr(Is_even_K)
    {
        lds_direct_copy_even_k<0, /*Is_even_MN=*/Is_even_MN, _64x32, true, 0>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<0, /*Is_even_MN=*/Is_even_MN, _64x32, true, 1>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        // lds_direct_copy_even_k<1, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        
        lds_direct_copy_even_k<0, Is_even_MN, _64x32_load_v>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<1, Is_even_MN, _64x32_load_v>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        // lds_direct_copy_even_k<1, Is_even_MN, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        // lds_direct_copy_even_k<2, Is_even_MN, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        // lds_direct_copy_even_k<3, Is_even_MN, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
    }
    

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        // flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        cute::copy(smem_tiled_copy_K, tSsK(_, 0, 0), tSrK(_, 0, 0));
        for (int m = 0; m < size<1>(tGrQ); m++)
        {
            cute::gemm(tiled_mma, tGrQ(_, m, 0), tSrK(_, 0, 0), acc_s_ori(_, m, 0));
        }
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        cute::copy(smem_tiled_copy_K, tSsK(_, 1, 0), tSrK(_, 1, 0));
        for (int m = 0; m < size<1>(tGrQ); m++)
        {
            cute::gemm(tiled_mma, tGrQ(_, m, 0), tSrK(_, 1, 0), acc_s_ori(_, m, 1));
        }
        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
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
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }

                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/false>(
                        rP, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/false>(
                        rP, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
            }
        }

        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<4>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<5>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<6>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<7>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        
        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            // if constexpr(Is_even_K)
            {
                lds_direct_copy_even_k<0, true, _64x32, true, 0>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<0, true, _64x32, true, 1>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                
                lds_direct_copy_even_k<0, true, _64x32_load_v>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<1, true, _64x32_load_v>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                // lds_direct_copy_even_k<1, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                // lds_direct_copy_even_k<2, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                // lds_direct_copy_even_k<3, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            }

        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        // flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        cute::copy(smem_tiled_copy_K, tSsK(_, 0, 0), tSrK(_, 0, 0));
        for (int m = 0; m < size<1>(tGrQ); m++)
        {
            cute::gemm(tiled_mma, tGrQ(_, m, 0), tSrK(_, 0, 0), acc_s_ori(_, m, 0));
        }
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        cute::copy(smem_tiled_copy_K, tSsK(_, 1, 0), tSrK(_, 1, 0));
        for (int m = 0; m < size<1>(tGrQ); m++)
        {
            cute::gemm(tiled_mma, tGrQ(_, m, 0), tSrK(_, 1, 0), acc_s_ori(_, m, 1));
        }
        asm volatile("s_barrier");
        
        // __builtin_amdgcn_sched_barrier(1);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

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
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/false>(
                        rP, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/false>(
                        rP, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
            }
        }

        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<4>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<5>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<6>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        flash::gemm_k_rs_ds_read_m32x16_alt<7>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        
        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            // if constexpr(Is_even_K)
            {
                lds_direct_copy_even_k<0, true, _64x32, true, 0>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<0, true, _64x32, true, 1>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                
                lds_direct_copy_even_k<0, true, _64x32_load_v>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<1, true, _64x32_load_v>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                // lds_direct_copy_even_k<1, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                // lds_direct_copy_even_k<2, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                // lds_direct_copy_even_k<3, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            }


        }
    }

    // ★ Attention Sinks: conditional normalize (direct global memory load) ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) { tSrS_aux(mi) = s_aux_val; }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Tensor taccOcO = thr_mma_for_gemm1.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    // if (get<1>(taccOcO(0)) == 0) {
    //     #pragma unroll
    //     for (int mi = 0; mi < size(lse); ++mi) {
    //         const int row = get<0>(taccOcO(0, mi, 0));
    //         if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
    //     }
    // }

    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) * 2 + ni * 32;
                if constexpr (Is_even_K)
                {
                    using result_type = cutlass::Array<Element, 2>;
                    for (int ei = 0; ei < 4; ++ei)
                    {
                        result_type res;
                        res[0] = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        res[1] = flash::convert_type<Element>(acc_o(ei + 4, mi, ni));
                        *(result_type*)(&gO(row, col)) = res;
                        col += 8;
                    }
                }
                else
                {
                    using result_type = cutlass::Array<Element, 2>;
                    for (int ei = 0; ei < 4; ++ei)
                    {
                        result_type res;
                        res[0] = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        res[1] = flash::convert_type<Element>(acc_o(ei + 4, mi, ni));
                        if (col < params.d_value)
                        {
                            gO(row, col) = res[0];
                        }
                        if (col + 1 < params.d_value)
                        {
                            gO(row, col + 1) = res[1];
                        }
                        col += 8;
                    }
                }

            }

            if (laneId / 16 == 0) {
                gLSE(row) = lse(mi);
            }
                
        }
    } 
   
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_dim64_prefetch(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

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

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
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

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutK{});
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
    
    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    static_assert(k0_loops == 2 && k1_loops == 4);
    if constexpr(Is_even_K)
    {
        lds_direct_copy_even_k<0, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<1, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        
        lds_direct_copy_even_k<0, Is_even_MN, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<1, Is_even_MN, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<2, Is_even_MN, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<3, Is_even_MN, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < k0_loops; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        #pragma unroll
        for (int i = 0; i < k1_loops; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gV, sV, i, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }
    }

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);

        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
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
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }

                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/false>(
                        rP, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/false>(
                        rP, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
            }
        }

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        
        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            if constexpr(Is_even_K)
            {
                lds_direct_copy_even_k<0, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<1, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                
                lds_direct_copy_even_k<0, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<1, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<2, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<3, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            }
            else
            {
                lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
                

                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 0, params.v_row_stride, params.d);
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 1, params.v_row_stride, params.d);
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 2, params.v_row_stride, params.d);
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 3, params.v_row_stride, params.d);
            }
        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);

        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        asm volatile("s_barrier");
        
        // __builtin_amdgcn_sched_barrier(1);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

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
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==64){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/false>(
                        rP, m_block * kBlockM, block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/false>(
                        rP, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
            }
        }

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        
        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            if constexpr(Is_even_K)
            {
                lds_direct_copy_even_k<0, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<1, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                
                lds_direct_copy_even_k<0, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<1, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<2, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<3, true, _16x64_64>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            }
            else
            {
                lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
                

                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 0, params.v_row_stride, params.d);
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 1, params.v_row_stride, params.d);
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 2, params.v_row_stride, params.d);
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 3, params.v_row_stride, params.d);
            }

        }
    }

    // ★ Attention Sinks: conditional normalize (direct global memory load) ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) { tSrS_aux(mi) = s_aux_val; }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
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
                col = (laneId / 16) * 2 + ni * 32;
                if constexpr (Is_even_K)
                {
                    using result_type = cutlass::Array<Element, 2>;
                    for (int ei = 0; ei < 4; ++ei)
                    {
                        result_type res;
                        res[0] = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        res[1] = flash::convert_type<Element>(acc_o(ei + 4, mi, ni));
                        *(result_type*)(&gO(row, col)) = res;
                        col += 8;
                    }
                }
                else
                {
                    using result_type = cutlass::Array<Element, 2>;
                    for (int ei = 0; ei < 4; ++ei)
                    {
                        result_type res;
                        res[0] = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        res[1] = flash::convert_type<Element>(acc_o(ei + 4, mi, ni));
                        if (col < params.d_value)
                        {
                            gO(row, col) = res[0];
                        }
                        if (col + 1 < params.d_value)
                        {
                            gO(row, col + 1) = res[1];
                        }
                        col += 8;
                    }
                }

            }
        }
    } 
   
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_dim256_prefetch(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

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

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    // if (tidx == 0) printf("bidb:%d bidh:%d m_block:%d\n", bidb, bidh, m_block);
    // printf();

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
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

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),typename Kernel_traits::SmemLayoutK{});
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
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x128, 4>(tOsVt8x64.layout()));

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
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_split = local_tile(acc_o, Shape<Int<8>, Int<kBlockM/64>, Int<kHeadDimV / 32 / 2>>{}, make_coord(0, 0, _)); 
    auto acc_o_temp0 = acc_o_split(_, _, _, 0);
    auto acc_o_temp1 = acc_o_split(_, _, _, 1);
    clear(acc_o);
    constexpr static int K_BUFF_SIZE = 4;
    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    s_waitcnt<0>();
    __syncthreads();
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }
    } else {
        lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
    }

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 4, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<4, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 5, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<5, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 6, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<6, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 7, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<7, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 0);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 1);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 6, 2);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 7, 3);
        s_barrier();

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

    #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int block_row_idx =  m_block * kBlockM + (tidx % 16) + (tidx / 64 * 16);
            const int block_col_idx = n_block * kBlockN;
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                } else {
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                } else {
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            if constexpr(!Is_even_K) {
                #pragma unroll 
                for (int i = 0; i < 3; i ++) {
                    lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d);
                }
            } else {
                lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            }
        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    #endif
    }

    #if 1
    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true>(gK, sK, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true>(gK, sK, 4, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<4, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true>(gK, sK, 5, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<5, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true>(gK, sK, 6, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<6, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true>(gK, sK, 7, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<7, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 0);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(0, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, true, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 1);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(0, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, true, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 6, 2);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(0, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, true, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 7, 3);
        s_barrier();

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        s_barrier();
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        s_barrier();

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
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(0, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, true, _16x256, 0>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(1, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, true, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(1, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, true, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(1, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, true, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, true, _16x256>(1, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, true, _16x256, 1>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();


        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            if constexpr(!Is_even_K) {
                #pragma unroll 
                for (int i = 0; i < 3; i ++) {
                    lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d);
                }
            } else {
                lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            }
        }
    }
    #endif

    __builtin_amdgcn_s_barrier();
    // Epilogue
    
    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
    s_waitcnt<0>();

    Tensor rO = flash::convert_type<Element>(acc_o);
    
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{}, make_coord(m_block, 0));  // (kBlockM, kHeadDim)
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

    // __builtin_amdgcn_sched_barrier(0);
    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    const int qo_len = binfo.actual_seqlen_q - m_block * kBlockM;

    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
            // asm volatile("v_cmpx_lt_i32 exec, %0, %1":: "v"(row), "v"(qo_len) :);
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                if constexpr (Is_even_K) {
                    col = (laneId / 16) * 2 + ni * 32;
                    using result_type = cutlass::Array<Element, 2>;
                    for (int ei = 0; ei < 4; ++ei)
                    {
                        result_type res;
                        res[0] = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        res[1] = flash::convert_type<Element>(acc_o(ei + 4, mi, ni));
                        *(result_type*)(&gO(row, col)) = res;
                        col += 8;
                    }
                } else {
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_o); ++ei) {
                        col = (laneId / 16) + ni * 32 + ei * 4;
                        // wangaq debug
                        // printf("bidx:%d bidy:%d bidz:%d tid:%d mi:%d ni:%d ei:%d row:%d col:%d acc_o:%10.4f\n", 
                        // blockIdx.x, blockIdx.y, blockIdx.z, tidx, mi, ni, ei, row, col, acc_o(ei, mi, ni));
                        if (col < params.d_value) {
                            gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        } 
                        // else 
                        //     gO(row, col) = Element(0.0);
                    }
                }
            }
            // asm volatile("s_mov_b64 exec, 0xFFFFFFFFFFFFFFFF");
        }
    } 
    // __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_dim512_prefetch(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

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

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    // if (tidx == 0) printf("bidb:%d bidh:%d m_block:%d\n", bidb, bidh, m_block);
    // printf();

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
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

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),typename Kernel_traits::SmemLayoutK{});
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
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x128, 4>(tOsVt8x64.layout()));

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
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_split = local_tile(acc_o, Shape<Int<8>, Int<1>, Int<kHeadDimV / 32 / 4>>{}, make_coord(0, 0, _)); 
    auto acc_o_temp0 = acc_o_split(_, _, _, 0);
    auto acc_o_temp1 = acc_o_split(_, _, _, 1);
    auto acc_o_temp2 = acc_o_split(_, _, _, 2);
    auto acc_o_temp3 = acc_o_split(_, _, _, 3);

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    s_waitcnt<0>();
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }
#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gK, sK, 4, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gK, sK, 5, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gK, sK, 6, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gK, sK, 7, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gK, sK, 8, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gK, sK, 9, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 6, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gK, sK, 10, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 7, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gK, sK, 11, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 8, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gK, sK, 12, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 9, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gK, sK, 13, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 10, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gK, sK, 14, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 11, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gK, sK, 15, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 12, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 13, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 14, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 15, 3);


        s_barrier();

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

    #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        {   // dropout
            const int block_row_idx =  m_block * kBlockM + (tidx % 16) + (tidx / 64 * 16);
            const int block_col_idx = n_block * kBlockN;
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                } else {
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                } else {
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();


        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d);
            }
        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    #endif
    }

    #if 1
    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
        s_barrier();
        
         lds_direct_copy<Is_even_K>(gK, sK, 3, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);
        s_barrier();

        lds_direct_copy<Is_even_K>(0, gK, sK, 4, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        s_barrier();

        lds_direct_copy<Is_even_K>(1, gK, sK, 5, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);
        s_barrier();

        lds_direct_copy<Is_even_K>(2, gK, sK, 6, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3);
        s_barrier();

        lds_direct_copy<Is_even_K>(3, gK, sK, 7, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 0);
        s_barrier();

        lds_direct_copy<Is_even_K>(0, gK, sK, 8, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 1);
        s_barrier();

        lds_direct_copy<Is_even_K>(1, gK, sK, 9, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 6, 2);
        s_barrier();

        lds_direct_copy<Is_even_K>(2, gK, sK, 10, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 7, 3);
        s_barrier();

        lds_direct_copy<Is_even_K>(3, gK, sK, 11, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 8, 0);
        s_barrier();

        lds_direct_copy<Is_even_K>(0, gK, sK, 12, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 9, 1);
        s_barrier();

        lds_direct_copy<Is_even_K>(1, gK, sK, 13, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 10, 2);
        s_barrier();

        lds_direct_copy<Is_even_K>(2, gK, sK, 14, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 11, 3);
        s_barrier();

        lds_direct_copy<Is_even_K>(3, gK, sK, 15, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 12, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(0, 0, gV, sV, 0, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 13, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(0, 1, gV, sV, 1, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 14, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(0, 2, gV, sV, 2, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 15, 3);


        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        s_barrier();
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        s_barrier();

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
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }
        lds_direct_copy<Is_even_K, true, _16x256>(0, 3, gV, sV, 3, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, true, _16x256>(1, 0, gV, sV, 0, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, true, _16x256>(1, 1, gV, sV, 1, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, true, _16x256>(1, 2, gV, sV, 2, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, true, _16x256>(1, 3, gV, sV, 3, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(2, 0, gV, sV, 0, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(2, 1, gV, sV, 1, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(2, 2, gV, sV, 2, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(2, 3, gV, sV, 3, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(3, 0, gV, sV, 0, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(3, 1, gV, sV, 1, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(3, 2, gV, sV, 2, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp2, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        lds_direct_copy<Is_even_K, true, _16x256>(3, 3, gV, sV, 3, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp3, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d);
            }
        }

    }
    #endif

    __builtin_amdgcn_s_barrier();
    // Epilogue
    #if 1

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
    s_waitcnt<0>();

    Tensor rO = flash::convert_type<Element>(acc_o);
    

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d_value),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDimV>>{}, make_coord(m_block, 0));  // (kBlockM, kHeadDim)
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

    // __builtin_amdgcn_sched_barrier(0);
    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    const int qo_len = binfo.actual_seqlen_q - m_block * kBlockM;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
            // asm volatile("v_cmpx_lt_i32 exec, %0, %1":: "v"(row), "v"(qo_len) :);
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ++ei) {
                    col = (laneId / 16) + ni * 32 + ei * 4;
                    // wangaq debug
                    // printf("bidx:%d bidy:%d bidz:%d tid:%d mi:%d ni:%d ei:%d row:%d col:%d acc_o:%10.4f\n", 
                    // blockIdx.x, blockIdx.y, blockIdx.z, tidx, mi, ni, ei, row, col, acc_o(ei, mi, ni));
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                    } 
                    // else 
                    //     gO(row, col) = Element(0.0);
                }
            }
            // asm volatile("s_mov_b64 exec, 0xFFFFFFFFFFFFFFFF");
        }
    } 
    #endif
    // __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
#endif
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_dim64_prefetch_padding_mask(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

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

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
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
    
    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutK{});
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
    
    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    static_assert(k0_loops == 2 && k1_loops == 4);
    #pragma unroll
    for (int i = 0; i < k0_loops; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }
    #pragma unroll
    for (int i = 0; i < k1_loops; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gV, sV, i, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);

        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
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
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP,  m_block * kBlockM, n_block * (kBlockN), kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        
        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            

            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 0, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 1, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 2, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 3, params.v_row_stride, params.d);

        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);

        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        asm volatile("s_barrier");
        
        // __builtin_amdgcn_sched_barrier(1);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

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
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM, n_block * (kBlockN), kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }               

                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP,  m_block * kBlockM, block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;

        
        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            

            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 0, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 1, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 2, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x64_64>(gV, sV, 3, params.v_row_stride, params.d);

        }
    }

    // Epilogue
    #if 1  // use all lds
    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    // static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    // Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    // CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
    #endif
#endif
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_mla_prefetch(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

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
    constexpr int K_BUFF_SIZE = 4;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
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

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    // Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});


    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

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
    // auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_BLayout);
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, 128/32>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVt);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol<_16x128>(tOsVt8x64.layout()));

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
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    constexpr int k0_loops = kHeadDim / 32;
    constexpr int k1_loops = kBlockN / 16;
    static_assert(kStages <= k0_loops && kStages <= k1_loops, "kStages is error");
    #pragma unroll
    for (int i = 0; i < kStages; ++i) { // 0 1 2
        lds_direct_copy<Is_even_K, Is_even_MN, _64x32, K_BUFF_SIZE>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }
#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            // load 3 4 5 -> 3 0 1 
            // k0/k1 0 1 2 
            lds_direct_copy<Is_even_K, Is_even_MN, _64x32, K_BUFF_SIZE>(gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i, i);
            S_BARRIER;
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            // load 2 3 0
            // k0 3 4 5
            // k1 3 0 1
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128, K_BUFF_SIZE>(gV, sV, (i+2)%4, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i+3, (i+3)%4);
            S_BARRIER;
        }
        

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
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

        // load  1
        // k0/k1 2
        lds_direct_copy<Is_even_K, Is_even_MN, _16x128, K_BUFF_SIZE>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        S_WAITCNT;
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        
        // tail kStages == 3
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;
        if (n_block > n_block_min) { 
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy<Is_even_K, true, _64x32, K_BUFF_SIZE>(gK, sK, i, params.k_row_stride, params.d);
            }
        }  

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            // load 3 4 5 -> 3 0 1 
            // k0 0 1 2
            // k1 0 1 2 
            lds_direct_copy<Is_even_K, true, _64x32, K_BUFF_SIZE>(gK, sK,  kStages + i, params.k_row_stride, params.d);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i, i);
            S_BARRIER;
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            // load 2 3 0
            // k0 3 4 5
            // k1 3 0 1
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128, K_BUFF_SIZE>(gV, sV, (i+2)%4, params.v_row_stride, params.d_value);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i+3, (i+3)%4);
            S_BARRIER;
        }
        
        // __builtin_amdgcn_sched_barrier(1);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }
        

        softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);
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

        
        // load  1
        // k0/k1 2
        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128, K_BUFF_SIZE>(gV, sV, 1, params.v_row_stride, params.d_value);
        S_WAITCNT;
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        
        // tail kStages == 3
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        // S_BARRIER;
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        S_BARRIER;
        if (n_block > n_block_min) { 
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy<Is_even_K, true, _64x32, K_BUFF_SIZE>(gK, sK, i, params.k_row_stride, params.d);
            }
        }  
    }

    // Epilogue
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) { tSrS_aux(mi) = s_aux_val; }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }



    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
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
                    } else 
                        gO(row, col) = Element(0.0);
                    col += 4;
                }
            }
        }
    } 
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_mla_prefetch_fp8(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementOUT = typename Kernel_traits::ElementO;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

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
    constexpr int K_BUFF_SIZE = 4;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementOUT*>(params.o_ptr)
                                              + binfo_o_offset),
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

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0)); 

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));
    
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});//64,192
    Tensor sV = make_tensor(sK.data() + 8192, typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    // Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});


    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)

    //Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVtNoSwizzle);      // (MMA, MMA_K,MMA_N)
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)
    Tensor tSgS  = thr_mma.partition_C(gP);


    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K 128,128

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
    // auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_BLayout);
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});//192,64
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol_fp8<_64x64, 192/64>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x32_B8, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    // Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
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
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    float q_descale = params.q_descale_ptr == nullptr ? 1.0f : params.q_descale_ptr[bidb * params.q_descale_batch_stride + bidh * params.q_descale_head_stride];
    float k_descale = params.k_descale_ptr == nullptr ? 1.0f : params.k_descale_ptr[bidb * params.k_descale_batch_stride + bidh * params.k_descale_head_stride];
    float v_descale = params.v_descale_ptr == nullptr ? 1.0f : params.v_descale_ptr[bidb * params.v_descale_batch_stride + bidh * params.v_descale_head_stride];

    float scale_softmax_log2 = params.scale_softmax_log2*q_descale*k_descale;
    float scale_softmax = params.scale_softmax*q_descale*k_descale;

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    // constexpr int k0_loops = kHeadDim / 64;
    // constexpr int k1_loops = kBlockN / 32;
    Tensor tCrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
    Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);

    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64>(gK, sK, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _32x128>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);


#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
  
    
        asm volatile("s_waitcnt vmcnt(3) \n\t s_barrier\n\t");

        cute::copy(smem_tiled_copy_K, tSsK(_, _, 0), tCrK_copy_view(_, _, 0));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 0), tSrK(_, _, 0), acc_s_ori);

        asm volatile("s_waitcnt vmcnt(2) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 1), tCrK_copy_view(_, _, 1));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 1), tSrK(_, _, 1), acc_s_ori);

        asm volatile("s_waitcnt vmcnt(1) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 2), tCrK_copy_view(_, _, 2));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 2), tSrK(_, _, 2), acc_s_ori);

        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        
        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);//0,1,2,3
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);//tid%warp_row_stride
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        masking_step == 0
        ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, scale_softmax_log2)
        : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, scale_softmax_log2);

     
        // Convert acc_s from fp32 to fp8
        Tensor rP = flash::convert_type_fp8<Element>(acc_s);
  
        {   
            //dropout
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
                dropout.template apply_dropout_continuous_fp8</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back_fp8(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous_fp8(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }

     
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _32x128>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);


        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 1), tOrVt_copy_view(_, _, 1));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 1), tOrVt(_, _, 1), acc_o);

        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");

        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 0), tOrVt_copy_view(_, _, 0));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 0), tOrVt(_, _, 0), acc_o);

        asm volatile("s_barrier");

        if (n_block > n_block_min) { 

            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 2, params.k_row_stride, params.d);
            
            // lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 0, params.v_row_stride, params.d_value);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 1, params.v_row_stride, params.d_value);


        }  


        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
  
      
        asm volatile("s_waitcnt vmcnt(3) \n\t s_barrier\n\t");

        cute::copy(smem_tiled_copy_K, tSsK(_, _, 0), tCrK_copy_view(_, _, 0));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 0), tSrK(_, _, 0), acc_s_ori);

        asm volatile("s_waitcnt vmcnt(2) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 1), tCrK_copy_view(_, _, 1));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 1), tSrK(_, _, 1), acc_s_ori);

        asm volatile("s_waitcnt vmcnt(1) \n\t s_barrier\n\t");
        cute::copy(smem_tiled_copy_K, tSsK(_, _, 2), tCrK_copy_view(_, _, 2));
        
        cute::gemm(tiled_mma, tGrQ(_, _, 2), tSrK(_, _, 2), acc_s_ori);

        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));


        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);//0,1,2,3
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);//tid%warp_row_stride
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/ Is_local>(acc_s, acc_o, scale_softmax_log2);
      
        Tensor rP = flash::convert_type_fp8<Element>(acc_s);
   
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
                dropout.template apply_dropout_continuous_fp8</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                );
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back_fp8(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous_fp8(rP, block_row_idx, block_col_idx, kNWarps * 16);
            }
        }
        
        lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 0, params.v_row_stride, params.d_value);


        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
      
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 1), tOrVt_copy_view(_, _, 1));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 1), tOrVt(_, _, 1), acc_o);

        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
  
        cute::copy(smem_tiled_copy_V, tOsVt(_, _, 0), tOrVt_copy_view(_, _, 0));
        cute::gemm(tiled_mma_for_gemm1, rP(_, _, 0), tOrVt(_, _, 0), acc_o);

        asm volatile("s_barrier");

        if (n_block > n_block_min) { 
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64>(gK, sK, 2, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _32x128>(gV, sV, 1, params.v_row_stride, params.d_value);

          
        }  
       
   
    }

    //Epilogue
    Tensor lse = softmax.template normalize_softmax_lse_fp8<Is_dropout>(acc_o, scale_softmax,v_descale, params.rp_dropout);
    


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
                    auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(2, mi, ni),  acc_o(3, mi, ni), false);
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


////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {

    #if 1
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    const int n_block_min = !Is_local
        ? n_split_idx * n_blocks_per_split
        : std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    if (n_block_min >= n_block_max) {  // This also covers the case where n_block_max <= 0
        // We exit early and write 0 to gOaccum and -inf to gLSEaccum.
        // Otherwise we might read OOB elements from gK and gV,
        // or get wrong results when we combine gOaccum from different blocks.
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
            + m_block * kBlockM) * params.d_rounded;
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                      Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                     make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                      Shape<Int<kBlockM>>{}, Stride<_1>{});

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        clear(tOrOaccum);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgOaccum); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSEaccum(row) = Split ? -INFINITY : INFINITY; }
        }
        return;
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = block_table == nullptr
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = block_table == nullptr
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    #ifdef GEMM1_AMATRIX_WITH_SMEM
    Tensor sAccs = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutAccs{});
    #endif
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutKV{});
    #ifdef GEMM1_AMATRIX_WITH_SMEM
    Tensor sV = make_tensor(size(sK) > size(sAccs) ? sK.data() + size(sK): sAccs.data() + size(sAccs),
        typename Kernel_traits::SmemLayoutKV{});
    #else
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutV{});
    #endif
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyQKVPaged gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    Tensor tKgK_ = gmem_thr_copy_KV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK_ = gmem_thr_copy_KV.partition_D(sK);
    Tensor tVgV_ = gmem_thr_copy_KV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV_ = gmem_thr_copy_KV.partition_D(sV);

    Tensor tKgK = make_tensor(tKgK_.data(), reshape_thread_tile(tKgK_.layout()));
    Tensor tKsK = make_tensor(tKsK_.data(), reshape_thread_tile(tKsK_.layout()));
    Tensor tVgV = make_tensor(tVgV_.data(), reshape_thread_tile(tVgV_.layout()));
    Tensor tVsV = make_tensor(tVsV_.data(), reshape_thread_tile(tVsV_.layout()));

    if (block_table != nullptr) {
        tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max, params.page_block_size,
            block_table, params.k_batch_stride, params.k_row_stride);
        tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max, params.page_block_size,
            block_table, params.v_batch_stride, params.v_row_stride);
    }

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    #ifndef GEMM1_AMATRIX_WITH_SMEM
    typename Kernel_traits::TiledMma_FOR_GEMM1 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    #endif
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    #ifdef GEMM1_AMATRIX_WITH_SMEM
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K
    #else
    Tensor tOrVt  = thr_mma_for_gemm1.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)
    Tensor acc_o = partition_fragment_C(thr_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K
    #endif


    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    #ifdef GEMM1_AMATRIX_WITH_SMEM
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    #else
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_for_gemm1);
    #endif
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV_ = gmem_thr_copy_KV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tKVcKV = make_tensor(tKVcKV_.data(), reshape_thread_tile(tKVcKV_.layout()));

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // Copy from Knew to K, optionally apply rotary embedding.
    // typename Kernel_traits::GmemTiledCopyRotcossin gmem_tiled_copy_rotary;
    // auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
    // typename Kernel_traits::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont;
    // auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);
    // if (cute::thread0())
    // {
    //     printf("Append_KV = %d params.rotary_dim = %d\n", Append_KV, params.rotary_dim);
    // }
    if constexpr (Append_KV) {
        typename Kernel_traits::GmemTiledCopyRotcossinPaged gmem_tiled_copy_rotary;
        auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
        typename Kernel_traits::GmemTiledCopyRotcossinContPaged gmem_tiled_copy_rotary_cont;
        auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);
        // Even if we have MQA / GQA, all threadblocks responsible for the same KV head are writing to
        // gmem. Technically it's a race condition, but they all write the same content anyway, and it's safe.
        // We want to do this so that all threadblocks can proceed right after they finish writing the KV cache.
        const index_t row_offset_cossin = ((n_block_max - 1) * kBlockN + (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])) * (params.rotary_dim / 2);
        Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockN>, Int<kHeadDim / 2>>{},
                                  make_stride(params.rotary_dim / 2, _1{}));
        Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockN>, Int<kHeadDim / 2>>{},
                                  make_stride(params.rotary_dim / 2, _1{}));
        Tensor gCosCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                      Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                      make_stride(params.rotary_dim / 2, _1{}));
        Tensor gSinCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                      Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                      make_stride(params.rotary_dim / 2, _1{}));
        Tensor tRgCos_ = gmem_thr_copy_rotary.partition_S(gCos);
        Tensor tRgSin_ = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor tRgCosCont_ = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        Tensor tRgSinCont_ = gmem_thr_copy_rotary_cont.partition_S(gSinCont);

        Tensor tRgCos = make_tensor(tRgCos_.data(), reshape_thread_tile(tRgCos_.layout()));
        Tensor tRgSin = make_tensor(tRgSin_.data(), reshape_thread_tile(tRgSin_.layout()));
        Tensor tRgCosCont = make_tensor(tRgCosCont_.data(), reshape_thread_tile(tRgCosCont_.layout()));
        Tensor tRgSinCont = make_tensor(tRgSinCont_.data(), reshape_thread_tile(tRgSinCont_.layout()));
        // if (cute::thread(0, 0)) { printf("rotary_cos_ptr = %p, gCos.data() = %p, tRgCos.data() = %p, rotary_dim = %d\n", params.rotary_cos_ptr, gCos.data(), tRgCos.data(), params.rotary_dim); }
        // if (cute::thread(8, 0)) { print_tensor(gCos); }
        // if (cute::thread(0, 0)) { print_tensor(tRgCos); }

        // const index_t row_offset_knew = binfo.k_offset(params.knew_batch_stride, params.knew_row_stride, bidb)
        const index_t row_offset_knew = bidb * params.knew_batch_stride
            + ((n_block_max - 1) * kBlockN) * params.knew_row_stride + (bidh / params.h_h_k_ratio) * params.knew_head_stride;
        // const index_t row_offset_vnew = binfo.k_offset(params.vnew_batch_stride, params.vnew_row_stride, bidb)
        const index_t row_offset_vnew = bidb * params.vnew_batch_stride
            + ((n_block_max - 1) * kBlockN) * params.vnew_row_stride + (bidh / params.h_h_k_ratio) * params.vnew_head_stride;
        // Subtract seqlen_k_cache * row stride so that conceptually gK and gKnew "line up". When we access them,
        // e.g. if gK has 128 rows and gKnew has 64 rows, we access gK[:128] and gKNew[128:128 + 64].
        // This maps to accessing the first 64 rows of knew_ptr.
        Tensor gKnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.knew_ptr)
                                                + row_offset_knew - binfo.seqlen_k_cache * params.knew_row_stride),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                  make_stride(params.knew_row_stride, _1{}));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("knew_ptr = %p, row_offset_knew = %d, gKnew_ptr = %p\n", params.knew_ptr, row_offset_knew, gKnew.data()); }
        Tensor gVnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.vnew_ptr)
                                                + row_offset_vnew - binfo.seqlen_k_cache * params.vnew_row_stride),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                  make_stride(params.vnew_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopyQKVPaged gmem_tiled_copy_KV_new;
        auto gmem_thr_copy_KV_new = gmem_tiled_copy_KV_new.get_thread_slice(tidx);
        Tensor tKgKnew_ = gmem_thr_copy_KV_new.partition_S(gKnew);  // (KCPY, KCPY_N, KCPY_K)
        Tensor tVgVnew_ = gmem_thr_copy_KV_new.partition_S(gVnew);  // (VCPY, VCPY_N, VCPY_K)
        auto tKgKnew = make_tensor(tKgKnew_.data(), reshape_thread_tile(tKgKnew_.layout()));
        auto tVgVnew = make_tensor(tVgVnew_.data(), reshape_thread_tile(tVgVnew_.layout()));

        const int n_block_copy_min = std::max(n_block_min, binfo.seqlen_k_cache / kBlockN);
        auto tKgK_data = tKgK.data();
        auto tVgV_data = tVgV.data();
        for (int n_block = n_block_max - 1; n_block >= n_block_copy_min; n_block--) {
            flash::copy_w_min_idx<Is_even_K>(
                tVgVnew, tVgV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN
            );
            // __syncthreads();
            // if (thread0())
            // {
            //     print(tVgVnew.layout());
            //     print(tVgV.layout());
            //     print(reshape_thread_tile(tVgVnew.layout()));
            //     print(reshape_thread_tile(tVgV.layout()));
            // }
            // return;
            tVgVnew.data() = tVgVnew.data() + (-int(kBlockN * params.vnew_row_stride));
            if (params.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K>(
                    tKgKnew, tKgK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN
                );
            } else {
                if (params.is_rotary_interleaved) {
                    // Don't clear OOB_K because we're writing to global memory
                    flash::copy_rotary_interleaved<Is_even_K, /*Clear_OOB_K=*/false>(
                        tKgKnew, tKgK, tRgCos, tRgSin, tKVcKV, binfo.actual_seqlen_k - n_block * kBlockN,
                        binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim
                    );
                    tRgCos.data() = tRgCos.data() + (-int(kBlockN * params.rotary_dim / 2));
                    tRgSin.data() = tRgSin.data() + (-int(kBlockN * params.rotary_dim / 2));
                } else {
                    // Don't clear OOB_K because we're writing to global memory
                    flash::copy_rotary_contiguous<Is_even_K, /*Clear_OOB_K=*/false>(
                        tKgKnew, tKgK, tRgCosCont, tRgSinCont, tKVcKV, binfo.actual_seqlen_k - n_block * kBlockN,
                        binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim
                    );
                    tRgCosCont.data() = tRgCosCont.data() + (-int(kBlockN * params.rotary_dim / 2));
                    tRgSinCont.data() = tRgSinCont.data() + (-int(kBlockN * params.rotary_dim / 2));

                }
            }
            tKgKnew.data() = tKgKnew.data() + (-int(kBlockN * params.knew_row_stride));
            if (block_table == nullptr) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                if (n_block > n_block_copy_min) {
                    tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size, 
                        block_table, params.v_batch_stride, params.v_row_stride);
                    tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size, 
                        block_table, params.k_batch_stride, params.k_row_stride);
                }
            }
        }
        // Need this before we can read in K again, so that we'll see the updated K values.
        __syncthreads();
        tKgK.data() = tKgK_data;
        tVgV.data() = tVgV_data;
    }
    // Read Q from gmem to smem, optionally apply rotary embedding.
    if (!Append_KV || params.rotary_dim == 0) {
        // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ,
                                           binfo.actual_seqlen_q - m_block * kBlockM);
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    } else {
        typename Kernel_traits::GmemTiledCopyRotcossin gmem_tiled_copy_rotary;
        auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
        typename Kernel_traits::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont;
        auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);
        const index_t row_offset_cossin = (binfo.seqlen_k_cache + (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb]) + (Is_causal || Is_local ? m_block * kBlockM : 0)) * (params.rotary_dim / 2);
        // If not causal, all the queries get the same the cos/sin, taken at location seqlen_k_cache.
        // We do this by setting the row stride of gCos / gSin to 0.
        Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        Tensor tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        Tensor tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);
        if (params.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K>(
                tQgQ, tQsQ, tRgCos, tRgSin, tQcQ, binfo.actual_seqlen_q - m_block * kBlockM,
                0, params.d, params.rotary_dim
            );
        } else {
            flash::copy_rotary_contiguous<Is_even_K>(
                tQgQ, tQsQ, tRgCosCont, tRgSinCont, tQcQ, binfo.actual_seqlen_q - m_block * kBlockM,
                0, params.d, params.rotary_dim
            );
        }
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    // if (blockIdx.x == 0 && blockIdx.z == 0 && tidx == 0)
    // {
    //     printf("n_block_min = %d n_block_max = %d\n", n_block_min, n_block_max);    
    // }

    int n_block = n_block_max - 1;
    auto tKrK = make_fragment_like(tKsK);
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KV, tKgK, tKrK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
    // if (cute::thread0()) {
    //     __syncthreads();
    //     printf("tKrK = %f\n", float(tKrK(0)));
    // }

    // flash::cp_async_wait<0>();
    // __syncthreads();
    // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(tKsK); }
    // __syncthreads();

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        __syncthreads();
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        cute::copy(tKrK, tKsK);
        __syncthreads();

        auto tVrV = make_fragment_like(tVsV);
        // Advance gV
        if (masking_step > 0) {
            if (block_table == nullptr) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            } else {
                tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block + 1, params.page_block_size,
                    block_table, params.v_batch_stride, params.v_row_stride);
            }
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_KV, tVgV, tVrV, tKVcKV, tKVpKV);
        } else {
            // Clear the smem tiles to account for predicated off loads
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_KV, tVgV, tVrV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        // __syncthreads();
        // cute::cp_async_fence();
        flash::gemm_rs(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K);
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        { 
            const int wave_id = tidx / 64;
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx % 16) +
                wave_id_to_row_block_id * warp_row_stride;
            const int row_idx_offset_ = m_block * kBlockM + 
                row_idx_offset_in_block;
            mask.template apply_mask<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, warp_row_stride * kNWarps
            );
        }


        // __syncthreads();
        cute::copy(tVrV, tVsV);
        __syncthreads();
        // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(tVsV); }
        // __syncthreads();

        if (n_block > n_block_min) {
            // Advance gK
            if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size, 
                    block_table, params.k_batch_stride, params.k_row_stride);
            }
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_KV, tKgK, tKrK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        // We have key_padding_mask so we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || !Is_even_MN>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || !Is_even_MN>(acc_s, acc_o, params.scale_softmax_log2);
        // if (cute::thread0()) { print(scores_max); print(scores_sum); print(scores); }

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);
        #ifdef GEMM1_AMATRIX_WITH_SMEM
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        // Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        Tensor tOrP = flash::convert_layout_acc_Aregs(tiled_mma, rP, sAccs);
        Tensor TOrVtCoal = make_tensor(tOrVt.data(), make_shape(size<0>(tOrVt),size<1>(tOrVt),size<2>(tOrVt)));

        flash::gemm_rs(acc_o, tOrP, TOrVtCoal, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        #else
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        #endif
        //  __syncthreads();
        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
         __syncthreads();
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        cute::copy(tKrK, tKsK);
        __syncthreads();
        auto tVrV = make_fragment_like(tVsV);
        // Advance gV
        if (block_table == nullptr) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        } else {
            tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block + 1, params.page_block_size, 
                block_table, params.v_batch_stride, params.v_row_stride);
        }
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_KV, tVgV, tVrV, tKVcKV, tKVpKV);
        
        flash::gemm_rs(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K);

        // flash::gemm(
        //     acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        //     smem_thr_copy_Q, smem_thr_copy_K
        // );
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        { 
            const int wave_id = tidx / 64;
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx % 16) +
                wave_id_to_row_block_id * warp_row_stride;
            const int row_idx_offset_ = m_block * kBlockM + 
                row_idx_offset_in_block;
            mask.template apply_mask<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, warp_row_stride * kNWarps
            );
        }

        cute::copy(tVrV, tVsV);
        __syncthreads();
        if (n_block > n_block_min) {
            // Advance gK
            if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size, 
                    block_table, params.k_batch_stride, params.k_row_stride); 
            }
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_KV, tKgK, tKrK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        // mask.template apply_mask</*Causal_mask=*/false>(
        //     acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        // );
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        #ifdef GEMM1_AMATRIX_WITH_SMEM
        Tensor tOrP = flash::convert_layout_acc_Aregs(tiled_mma, rP, sAccs);
        Tensor TOrVtCoal = make_tensor(tOrVt.data(), make_shape(size<0>(tOrVt),size<1>(tOrVt),size<2>(tOrVt)));
        flash::gemm_rs(acc_o, tOrP, TOrVtCoal, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        #else
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        #endif
    }

    // // Epilogue

    // ★ Attention Sinks: Conditional normalize based on split index ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr && n_split_idx == 0) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) {
            tSrS_aux(mi) = s_aux_val;
        }
        lse = softmax.template normalize_softmax_lse_with_sinks</*Is_dropout=*/false, Split>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2);
    } else {
        lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(
            acc_o, params.scale_softmax);
    }

    // Tensor sOaccum = make_tensor(sAccs.data() + size(sAccs), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    using SmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::SmemCopyAtomO,
        typename Kernel_traits::SmemCopyAtomOaccum
    >;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO = flash::convert_type<ElementO>(acc_o);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sOaccum is larger than sQ, so we need to syncthreads here
    // TODO: allocate enough smem for sOaccum
    // if constexpr (Split) { __syncthreads(); }
    __syncthreads();

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                         + m_block * kBlockM) * params.d_rounded;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q : bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});


    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);

    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );

    // __syncthreads();
    // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { 
    //     printf("row_offset_oaccum = %d, bidx = %d, gOaccum = %p gOaccum = %f\n", row_offset_oaccum, blockIdx.x, (uint64_t)(&gOaccum(0)), float(gOaccum(0))); 
    // }
#endif
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_16x64_mla(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    const int n_block_min = !Is_local
        ? n_split_idx * n_blocks_per_split
        : std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if (n_block_min >= n_block_max) {  // This also covers the case where n_block_max <= 0
        // We exit early and write 0 to gOaccum and -inf to gLSEaccum.
        // Otherwise we might read OOB elements from gK and gV,
        // or get wrong results when we combine gOaccum from different blocks.
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
            + m_block * kBlockM) * params.d_value_rounded;
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                      Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                     make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                      Shape<Int<kBlockM>>{}, Stride<_1>{});

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        clear(tOrOaccum);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgOaccum); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSEaccum(row) = Split ? -INFINITY : INFINITY; }
        }
        return;
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = block_table == nullptr
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = block_table == nullptr
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + (0), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKVPaged gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);

    Tensor tKgK_ = gmem_thr_copy_KV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK_ = gmem_thr_copy_KV.partition_D(sK);
    Tensor tVgV_ = gmem_thr_copy_KV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV_ = gmem_thr_copy_KV.partition_D(sV);

    Tensor tKgK = make_tensor(tKgK_.data(), reshape_thread_tile(tKgK_.layout()));
    Tensor tKsK = make_tensor(tKsK_.data(), reshape_thread_tile(tKsK_.layout()));
    Tensor tVgV = make_tensor(tVgV_.data(), reshape_thread_tile(tVgV_.layout()));
    Tensor tVsV = make_tensor(tVsV_.data(), reshape_thread_tile(tVsV_.layout()));

    if (block_table != nullptr) {
        tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max, params.page_block_size,
            block_table, params.k_batch_stride, params.k_row_stride);
        tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block_max, params.page_block_size,
            block_table, params.v_batch_stride, params.v_row_stride);
    }

    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    
    #if 1
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)


    Tensor tOrVt  = thr_mma_for_gemm1.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)
    Tensor acc_o = partition_fragment_C(thr_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K


    //
    // Copy Atom retiling
    //
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(sV), size<1>(sV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tKcK_ = gmem_thr_copy_KV.partition_S(cK);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tVcV_ = gmem_thr_copy_KV.partition_S(cV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tVcV = make_tensor(tVcV_.data(), reshape_thread_tile(tVcV_.layout()));
    Tensor tKcK = make_tensor(tKcK_.data(), reshape_thread_tile(tKcK_.layout()));

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));
    Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tVsV)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) < params.d_value; }
    }

    // Prologue

    // Copy from Knew to K, optionally apply rotary embedding.
    // typename Kernel_traits::GmemTiledCopyRotcossin gmem_tiled_copy_rotary;
    // auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
    // typename Kernel_traits::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont;
    // auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);
    // if (cute::thread0())
    // {
    //     printf("Append_KV = %d params.rotary_dim = %d\n", Append_KV, params.rotary_dim);
    // }
    #if 1
    if constexpr (Append_KV) {
        typename Kernel_traits::GmemTiledCopyRotcossinPaged gmem_tiled_copy_rotary;
        auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
        typename Kernel_traits::GmemTiledCopyRotcossinContPaged gmem_tiled_copy_rotary_cont;
        auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);
        // Even if we have MQA / GQA, all threadblocks responsible for the same KV head are writing to
        // gmem. Technically it's a race condition, but they all write the same content anyway, and it's safe.
        // We want to do this so that all threadblocks can proceed right after they finish writing the KV cache.
        const index_t row_offset_cossin = ((n_block_max - 1) * kBlockN + (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])) * (params.rotary_dim / 2);
        Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockN>, Int<kHeadDim / 2>>{},
                                  make_stride(params.rotary_dim / 2, _1{}));
        Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockN>, Int<kHeadDim / 2>>{},
                                  make_stride(params.rotary_dim / 2, _1{}));
        Tensor gCosCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                      Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                      make_stride(params.rotary_dim / 2, _1{}));
        Tensor gSinCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                      Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                      make_stride(params.rotary_dim / 2, _1{}));
        Tensor tRgCos_ = gmem_thr_copy_rotary.partition_S(gCos);
        Tensor tRgSin_ = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor tRgCosCont_ = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        Tensor tRgSinCont_ = gmem_thr_copy_rotary_cont.partition_S(gSinCont);

        Tensor tRgCos = make_tensor(tRgCos_.data(), reshape_thread_tile(tRgCos_.layout()));
        Tensor tRgSin = make_tensor(tRgSin_.data(), reshape_thread_tile(tRgSin_.layout()));
        Tensor tRgCosCont = make_tensor(tRgCosCont_.data(), reshape_thread_tile(tRgCosCont_.layout()));
        Tensor tRgSinCont = make_tensor(tRgSinCont_.data(), reshape_thread_tile(tRgSinCont_.layout()));
        // if (cute::thread(0, 0)) { printf("rotary_cos_ptr = %p, gCos.data() = %p, tRgCos.data() = %p, rotary_dim = %d\n", params.rotary_cos_ptr, gCos.data(), tRgCos.data(), params.rotary_dim); }
        // if (cute::thread(8, 0)) { print_tensor(gCos); }
        // if (cute::thread(0, 0)) { print_tensor(tRgCos); }

        // const index_t row_offset_knew = binfo.k_offset(params.knew_batch_stride, params.knew_row_stride, bidb)
        const index_t row_offset_knew = bidb * params.knew_batch_stride
            + ((n_block_max - 1) * kBlockN) * params.knew_row_stride + (bidh / params.h_h_k_ratio) * params.knew_head_stride;
        // const index_t row_offset_vnew = binfo.k_offset(params.vnew_batch_stride, params.vnew_row_stride, bidb)
        const index_t row_offset_vnew = bidb * params.vnew_batch_stride
            + ((n_block_max - 1) * kBlockN) * params.vnew_row_stride + (bidh / params.h_h_k_ratio) * params.vnew_head_stride;
        // Subtract seqlen_k_cache * row stride so that conceptually gK and gKnew "line up". When we access them,
        // e.g. if gK has 128 rows and gKnew has 64 rows, we access gK[:128] and gKNew[128:128 + 64].
        // This maps to accessing the first 64 rows of knew_ptr.
        Tensor gKnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.knew_ptr)
                                                + row_offset_knew - binfo.seqlen_k_cache * params.knew_row_stride),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                  make_stride(params.knew_row_stride, _1{}));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("knew_ptr = %p, row_offset_knew = %d, gKnew_ptr = %p\n", params.knew_ptr, row_offset_knew, gKnew.data()); }
        Tensor gVnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.vnew_ptr)
                                                + row_offset_vnew - binfo.seqlen_k_cache * params.vnew_row_stride),
                                  Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                                  make_stride(params.vnew_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopyQKVPaged gmem_tiled_copy_KV_new;
        auto gmem_thr_copy_KV_new = gmem_tiled_copy_KV_new.get_thread_slice(tidx);
        Tensor tKgKnew_ = gmem_thr_copy_KV_new.partition_S(gKnew);  // (KCPY, KCPY_N, KCPY_K)
        Tensor tVgVnew_ = gmem_thr_copy_KV_new.partition_S(gVnew);  // (VCPY, VCPY_N, VCPY_K)
        auto tKgKnew = make_tensor(tKgKnew_.data(), reshape_thread_tile(tKgKnew_.layout()));
        auto tVgVnew = make_tensor(tVgVnew_.data(), reshape_thread_tile(tVgVnew_.layout()));

        const int n_block_copy_min = std::max(n_block_min, binfo.seqlen_k_cache / kBlockN);
        auto tKgK_data = tKgK.data();
        auto tVgV_data = tVgV.data();
        for (int n_block = n_block_max - 1; n_block >= n_block_copy_min; n_block--) {
            flash::copy_w_min_idx<Is_even_K>(
                tVgVnew, tVgV, tVcV, tVpV, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN
            );
            // __syncthreads();
            // if (thread0())
            // {
            //     print(tVgVnew.layout());
            //     print(tVgV.layout());
            //     print(reshape_thread_tile(tVgVnew.layout()));
            //     print(reshape_thread_tile(tVgV.layout()));
            // }
            // return;
            tVgVnew.data() = tVgVnew.data() + (-int(kBlockN * params.vnew_row_stride));
            if (params.rotary_dim == 0) {
                flash::copy_w_min_idx<Is_even_K>(
                    tKgKnew, tKgK, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN
                );
            } else {
                if (params.is_rotary_interleaved) {
                    // Don't clear OOB_K because we're writing to global memory
                    flash::copy_rotary_interleaved<Is_even_K, /*Clear_OOB_K=*/false>(
                        tKgKnew, tKgK, tRgCos, tRgSin, tKcK, binfo.actual_seqlen_k - n_block * kBlockN,
                        binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim
                    );
                    tRgCos.data() = tRgCos.data() + (-int(kBlockN * params.rotary_dim / 2));
                    tRgSin.data() = tRgSin.data() + (-int(kBlockN * params.rotary_dim / 2));
                } else {
                    // Don't clear OOB_K because we're writing to global memory
                    flash::copy_rotary_contiguous<Is_even_K, /*Clear_OOB_K=*/false>(
                        tKgKnew, tKgK, tRgCosCont, tRgSinCont, tKcK, binfo.actual_seqlen_k - n_block * kBlockN,
                        binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim
                    );
                    tRgCosCont.data() = tRgCosCont.data() + (-int(kBlockN * params.rotary_dim / 2));
                    tRgSinCont.data() = tRgSinCont.data() + (-int(kBlockN * params.rotary_dim / 2));

                }
            }
            tKgKnew.data() = tKgKnew.data() + (-int(kBlockN * params.knew_row_stride));
            if (block_table == nullptr) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                if (n_block > n_block_copy_min) {
                    tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size, 
                        block_table, params.v_batch_stride, params.v_row_stride);
                    tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size, 
                        block_table, params.k_batch_stride, params.k_row_stride);
                }
            }
        }
        // Need this before we can read in K again, so that we'll see the updated K values.
        __syncthreads();
        tKgK.data() = tKgK_data;
        tVgV.data() = tVgV_data;
    }
    #endif
    // Read Q from gmem to smem, optionally apply rotary embedding.
    if (!Append_KV || params.rotary_dim == 0) {
        // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tSrQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);
        __syncthreads();
    } else {
        #if 1
        typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_Q;
        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
        Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);
        Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);

        auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

        typename Kernel_traits::GmemTiledCopyRotcossin gmem_tiled_copy_rotary;
        auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
        typename Kernel_traits::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont;
        auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);
        const index_t row_offset_cossin = (binfo.seqlen_k_cache + (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb]) + (Is_causal || Is_local ? m_block * kBlockM : 0)) * (params.rotary_dim / 2);
        // If not causal, all the queries get the same the cos/sin, taken at location seqlen_k_cache.
        // We do this by setting the row stride of gCos / gSin to 0.
        Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        Tensor tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        Tensor tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);
        if (params.is_rotary_interleaved) {
            flash::copy_rotary_interleaved<Is_even_K>(
                tQgQ, tQsQ, tRgCos, tRgSin, tQcQ, binfo.actual_seqlen_q - m_block * kBlockM,
                0, params.d, params.rotary_dim
            );
        } else {
            flash::copy_rotary_contiguous<Is_even_K>(
                tQgQ, tQsQ, tRgCosCont, tRgSinCont, tQcQ, binfo.actual_seqlen_q - m_block * kBlockM,
                0, params.d, params.rotary_dim
            );
        }
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
        #endif
    }

    int n_block = n_block_max - 1;
    auto tKrK = make_fragment_like(tKsK);
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KV, tKgK, tKrK, tKcK, tKpK,
                                       binfo.actual_seqlen_k - n_block * kBlockN);

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        // __syncthreads();
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s_ori);
        asm volatile("s_barrier\n\t");
        cute::copy(tKrK, tKsK);
        __syncthreads();

        auto tVrV = make_fragment_like(tVsV);
        // Advance gV
        if (masking_step > 0) {
            if (block_table == nullptr) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            } else {
                tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block + 1, params.page_block_size,
                    block_table, params.v_batch_stride, params.v_row_stride);
            }
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_KV, tVgV, tVrV, tVcV, tVpV);
        } else {
            // Clear the smem tiles to account for predicated off loads
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_KV, tVgV, tVrV, tVcV, tVpV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }

        flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K);
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        { 

            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        asm volatile("s_barrier\n\t");
        cute::copy(tVrV, tVsV);
        __syncthreads();

        if (n_block > n_block_min) {
            // Advance gK
            if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size, 
                    block_table, params.k_batch_stride, params.k_row_stride);
            }
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_KV, tKgK, tKrK, tKcK, tKpK);
        }

        // We have key_padding_mask so we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || !Is_even_MN>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || !Is_even_MN>(acc_s, acc_o, params.scale_softmax_log2);
        // if (cute::thread0()) { print(scores_max); print(scores_sum); print(scores); }

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);
        
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }
#endif
#if 1
    for (; n_block >= n_block_min; --n_block) {
        //  __syncthreads();
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s_ori);
        asm volatile("s_barrier\n\t");
        cute::copy(tKrK, tKsK);
        __syncthreads();
        auto tVrV = make_fragment_like(tVsV);
        // Advance gV
        if (block_table == nullptr) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        } else {
            tVgV.data() = gV.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block + 1, params.page_block_size, 
                block_table, params.v_batch_stride, params.v_row_stride);
        }
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_KV, tVgV, tVrV, tVcV, tVpV);
        
        flash::gemm_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K);
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        asm volatile("s_barrier\n\t");
        cute::copy(tVrV, tVsV);
        __syncthreads();
        if (n_block > n_block_min) {
            // Advance gK
            if (block_table == nullptr) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            } else {
                tKgK.data() = gK.data() + flash::resolve_thread_kv_page_slice_offset<Kernel_traits>(tidx, n_block, params.page_block_size, 
                    block_table, params.k_batch_stride, params.k_row_stride); 
            }
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_KV, tKgK, tKrK, tKcK, tKpK);
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
    }
#endif
    // // Epilogue
#if 1
    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, params.scale_softmax);
    
    // Tensor sOaccum = make_tensor(sAccs.data() + size(sAccs), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    using SmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::SmemCopyAtomO,
        typename Kernel_traits::SmemCopyAtomOaccum
    >;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO = flash::convert_type<ElementO>(acc_o);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sOaccum is larger than sQ, so we need to syncthreads here
    // TODO: allocate enough smem for sOaccum
    // if constexpr (Split) { __syncthreads(); }
    __syncthreads();

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                         + m_block * kBlockM) * params.d_value_rounded;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                 make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});


    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma_for_gemm1.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 8);

    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
#endif
    // __syncthreads();
    // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { 
    //     printf("row_offset_oaccum = %d, bidx = %d, gOaccum = %p gOaccum = %f\n", row_offset_oaccum, blockIdx.x, (uint64_t)(&gOaccum(0)), float(gOaccum(0))); 
    // }
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch(const Params &params, const int bidb, const int bidh, const int m_block) {
#if 1
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = (params.seqlen_k + kBlockN - 1) / kBlockN ;
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = !Has_block_table? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = !Has_block_table
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = !Has_block_table
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gV_tail = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v + 64 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + (0), typename Kernel_traits::SmemLayoutV{});
    Tensor sV_tail = make_tensor(sK.data() + size(sV), typename Kernel_traits::SmemLayoutV{});
    // Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    // Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});


    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x64x16 tiled_mma_gemm1;
    auto thr_mma_gemm1 = tiled_mma_gemm1.get_thread_slice(tidx);

    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    // Tensor tSgS  = thr_mma.partition_C(gP);
    #if 1
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)


    Tensor tOrV  = thr_mma_gemm1.partition_fragment_B(sV);                // (MMA, MMA_K,MMA_N)



    //
    // Copy Atom retiling
    //
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    // auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_BLayout);
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, 128/32>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tSsV = smem_thr_copy_V.partition_S(sV);
    auto tSsV_tail = smem_thr_copy_V.partition_S(sV_tail);

    // if (thread0())
    // {
    //     printf(" 01 %p \n", &(tVsV(0, 0, 1)));
    //     printf(" 10 %p \n", &(tVsV(0, 1, 0)));
    //     printf(" 11 %p \n", &(tVsV(0, 1, 1)));
    // }
#endif
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                binfo.actual_seqlen_q - m_block * kBlockM);
    __syncthreads();

    int n_block = n_block_max - 1;
    Tensor acc_o_ori = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_tail_ori = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o_ori);
    clear(acc_o_tail_ori);
    Tensor acc_o = make_tensor(acc_o_ori.data(), convert_layout_acc(acc_o_ori.layout()));
    Tensor acc_o_tail = make_tensor(acc_o_tail_ori.data(), convert_layout_acc(acc_o_tail_ori.layout()));

    flash::Softmax<size<1>(acc_o_ori)> softmax;
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    constexpr int k0_loops = 4;
    constexpr int k1_loops = 2;
    constexpr int kStages = 3;
    auto gK_data = gK.data();
    auto gV_data = gV.data();
    auto gV_tail_data = gV_tail.data();
    {
        int cur_block_table;
        const int *cur_block_table_ptr = block_table + (n_block);
        // cur_block_table = block_table[n_block];
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
             "=s"(cur_block_table));
        index_t offset_k = cur_block_table * params.k_batch_stride;
        // index_t offset_v = cur_block_table * params.v_batch_stride;
        gK.data() = gK_data + (offset_k);
        gV.data() = gV_data + (offset_k);
        gV_tail.data() = gV_tail_data + (offset_k);
    }
    {
        int max_mn = binfo.actual_seqlen_k - n_block * kBlockN;
        int row = tidx % 64;
        int end_loop_idx = ((max_mn + 3) / 4 ) * 4;
        const int warp_id = tidx / 64;
        // if (thread0())
        // {
        //     printf(" %.2f max_mn = %d \n ", float(gV0(row, max_mn)), max_mn);
        // }
        for (; max_mn < end_loop_idx; max_mn++)
        {
            if (warp_id == 0) {
                gV(row, max_mn) = Element(0);
            }
            else if (warp_id == 1) {
                gV_tail(row, max_mn) = Element(0);
            }
            
        }
    }
    __syncthreads();
    __builtin_amdgcn_sched_barrier(0);
    lds_direct_copy_even_k<0, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_even_k<1, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_even_k<2, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);


    // lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    // lds_direct_copy<Is_even_K, Is_even_MN,  _64x32, 0, false>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    // lds_direct_copy<Is_even_K, Is_even_MN,  _64x32, 0, false>(gK, sK, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    __builtin_amdgcn_sched_barrier(0);
    const bool Is_need_pad = binfo.actual_seqlen_k % 4 != 0;
#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
        {
            lds_direct_copy_even_k<3, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(4) \n s_barrier");;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
            S_BARRIER;

            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2, 2);
            S_BARRIER;

            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV_tail, sV_tail, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV_tail, sV_tail, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(6) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3, 3);
            S_BARRIER;
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );

        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, acc_o_tail, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, acc_o_tail, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);


        {
            __builtin_amdgcn_sched_barrier(0);
            // int token_id = n_block * kBlockN + ((tidx % 64) / 16) * 4;
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV_tail, sV_tail, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV_tail, sV_tail, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(7) \n s_barrier");
            __builtin_amdgcn_sched_barrier(0);
            // if (!Is_even_MN && Is_need_pad && masking_step == 0) {
                // flash::gemm_k_rs_pad_ws<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
                // asm volatile("s_waitcnt vmcnt(6) \n s_barrier");
                // flash::gemm_k_rs_pad_ws<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
                // asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
                // flash::gemm_k_rs_pad_ws<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
                // asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
                // flash::gemm_k_rs_pad_ws<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);
                // asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
                // flash::gemm_k_rs_pad_ws<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
                // asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
                // flash::gemm_k_rs_pad_ws<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
                // asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
                // flash::gemm_k_rs_pad_ws<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
                // asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
                // flash::gemm_k_rs_pad_ws<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);
                // S_BARRIER;
            // } else {
                flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
                asm volatile("s_waitcnt vmcnt(6) \n s_barrier");
                flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
                asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
                flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
                asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
                flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);
                asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
                flash::gemm_k_rs(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
                asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
                flash::gemm_k_rs(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
                asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
                flash::gemm_k_rs(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
                asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
                flash::gemm_k_rs(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);
                S_BARRIER;
            // }
        }

        if (n_block > 0) { 
            int cur_block_table;
            const int *cur_block_table_ptr = block_table + (n_block - 1);
            // cur_block_table = block_table[n_block - 1];
            asm volatile("s_load_dword %1, %0, 0x0\n\t"
                        "s_waitcnt lgkmcnt(0)\n\t":
                        "+s"(cur_block_table_ptr),
                "=s"(cur_block_table));
            index_t offset_k = cur_block_table * params.k_batch_stride;
            // index_t offset_v = cur_block_table * params.v_batch_stride;
            gK.data() = gK_data + (offset_k);
            gV.data() = gV_data + (offset_k);
            gV_tail.data() = gV_tail_data + (offset_k);

            lds_direct_copy_even_k<0, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_even_k<1, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_even_k<2, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);

        }
        
        if (n_masking_steps > 1 && n_block <= 0) {
            --n_block;
            break;
        }
    }

    #pragma unroll
    for (; n_block >= 0; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
        {
            lds_direct_copy_even_k<3, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
            S_BARRIER;
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(4) \n s_barrier");;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
            S_BARRIER;

            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2, 2);
            S_BARRIER;

            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV_tail, sV_tail, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV_tail, sV_tail, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(6) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3, 3);
            S_BARRIER;
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
    
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, acc_o_tail, params.scale_softmax_log2);
        
        Tensor rP = flash::convert_type<Element>(acc_s);
        {
            S_BARRIER;
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV_tail, sV_tail, 2, params.v_row_stride, params.d);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV_tail, sV_tail, 3, params.v_row_stride, params.d);
            asm volatile("s_waitcnt vmcnt(7) \n s_barrier");
            
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(6) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);
            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            flash::gemm_k_rs(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            flash::gemm_k_rs(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            flash::gemm_k_rs(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);
            S_BARRIER;
        }

        if (n_block > 0) { 
            int cur_block_table;
            const int *cur_block_table_ptr = block_table + (n_block - 1);
            // cur_block_table = block_table[n_block-1];
            asm volatile("s_load_dword %1, %0, 0x0\n\t"
                        "s_waitcnt lgkmcnt(0)\n\t":
                        "+s"(cur_block_table_ptr),
                "=s"(cur_block_table));
            index_t offset_k = cur_block_table * params.k_batch_stride;
            // index_t offset_v = cur_block_table * params.v_batch_stride;
            gK.data() = gK_data + (offset_k);
            gV.data() = gV_data + (offset_k);
            gV_tail.data() = gV_tail_data + (offset_k);
            
            lds_direct_copy_even_k<0, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_even_k<1, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_even_k<2, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);

            // #pragma unroll
            // for (int i = 0; i < kStages; ++i) {
            //     lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, i, params.k_row_stride, params.d);
            // }
        }

    }
#endif
    // // Epilogue
#if 1
    // Tensor acc_s = make_tensor(acc_o_ori.data(), convert_layout_acc(acc_o_ori.layout()));
    // Tensor acc_o_tail = make_tensor(acc_o_tail_ori.data(), convert_layout_acc(acc_o_tail_ori.layout()));

    // ★ Attention Sinks: Conditional normalize based on split index ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr ) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) {
            tSrS_aux(mi) = s_aux_val;
        }
        lse = softmax.template normalize_softmax_lse_with_sinks_tail</*Is_dropout=*/false, Split>(
            acc_o, acc_o_tail, tSrS_aux, params.scale_softmax, params.scale_softmax_log2);
    } else {
        lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(
            acc_o, acc_o_tail, params.scale_softmax);
    }
    
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ( bidb * params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
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
                col = (laneId / 16) * 4 + ni * 16;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ei += 4) {
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o(ei + 0, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o(ei + 3, mi, ni));
                    } 
                    col += 4;
                }
            }
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 64;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ei += 4) {
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o_tail(ei + 0, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o_tail(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o_tail(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o_tail(ei + 3, mi, ni));
                    } 
                    // else 
                    //     gO(row, col) = Element(0.0);
                    col += 4;
                }
            }
        }
    } 
#endif
#endif
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_fp8(const Params &params, const int bidb, const int bidh, const int m_block) {
#if 1
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;
    using ElementOUT = typename Kernel_traits::ElementO;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, ElementOUT, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
    const int n_blocks_per_split = (params.seqlen_k + kBlockN - 1) / kBlockN ;
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = !Has_block_table? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = !Has_block_table
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = !Has_block_table
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gV_tail = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v + 64 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + (0), typename Kernel_traits::SmemLayoutV{});
    Tensor sV_tail = make_tensor(sK.data() + size(sV), typename Kernel_traits::SmemLayoutV{});



    typename Kernel_traits::TiledMma16x64_LIT tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x64x32_NN tiled_mma_gemm1;
    auto thr_mma_gemm1 = tiled_mma_gemm1.get_thread_slice(tidx);

    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)

    // Tensor tSgS  = thr_mma.partition_C(gP);
    #if 1
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrV  = thr_mma_gemm1.partition_fragment_B(sV);                // (MMA, MMA_K,MMA_N)
    //
    // Copy Atom retiling
    //
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64_Blayout_LIT tiled_mma_BLayout;
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x64, 128/64>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tSsV = smem_thr_copy_V.partition_S(sV);
    auto tSsV_tail = smem_thr_copy_V.partition_S(sV_tail);

#endif
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                binfo.actual_seqlen_q - m_block * kBlockM);
    __syncthreads();

    int n_block = n_block_max - 1;
    Tensor acc_o_ori = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_tail_ori = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o_ori);
    clear(acc_o_tail_ori);
    Tensor acc_o = make_tensor(acc_o_ori.data(), convert_layout_acc(acc_o_ori.layout()));
    Tensor acc_o_tail = make_tensor(acc_o_tail_ori.data(), convert_layout_acc(acc_o_tail_ori.layout()));

    flash::Softmax<size<1>(acc_o_ori)> softmax;
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    constexpr int k0_loops = 2;
    constexpr int k1_loops = 2;
    constexpr int kStages = 1;
    auto gK_data = gK.data();
    auto gV_data = gV.data();
    auto gV_tail_data = gV_tail.data();
    {
        int cur_block_table;
        const int *cur_block_table_ptr = block_table + (n_block);
        // cur_block_table = block_table[n_block - 1];
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
             "=s"(cur_block_table));
        index_t offset_k = cur_block_table * params.k_batch_stride;
        // index_t offset_v = cur_block_table * params.v_batch_stride;
        gK.data() = gK_data + (offset_k);
        gV.data() = gV_data + (offset_k);
        gV_tail.data() = gV_tail_data + (offset_k);
    }

    float q_descale = params.q_descale_ptr == nullptr ? 1.0f : params.q_descale_ptr[0];
    float k_descale = params.k_descale_ptr == nullptr ? 1.0f : params.k_descale_ptr[0];
    float v_descale = params.v_descale_ptr == nullptr ? 1.0f : params.v_descale_ptr[0];

    const float scale_softmax_log2 = params.scale_softmax_log2*q_descale*k_descale;
    const float scale_softmax = params.scale_softmax*q_descale*k_descale;

    Tensor tCrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
    Tensor tCrV_copy_view = smem_thr_copy_V.retile_D(tOrV);

    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64_LIT, 0, false>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64_LIT, 0, false>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");;
        // flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
        Tensor tGrQ_  = recast<uint_byte_t<16>>(tGrQ); 
        __builtin_amdgcn_sched_barrier(0);
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 0)), (tCrK_copy_view(_, _, 0)));
        __builtin_amdgcn_sched_barrier(0);
        cute::gemm(tiled_mma, tGrQ_(_, _, 0), tSrK(_, _, 0), acc_s_ori);
        S_BARRIER;     
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");;
        // flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 1)), (tCrK_copy_view(_, _, 1)));
        cute::gemm(tiled_mma, tGrQ_(_, _, 1), tSrK(_, _, 1), acc_s_ori);
        S_BARRIER;     

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );

        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, acc_o_tail, scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, acc_o_tail, scale_softmax_log2);

        Tensor rP = flash::convert_type_fp8<Element>(acc_s);


        {
            lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV_tail, sV_tail, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV_tail, sV_tail, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
           
            const int Max_Mn = binfo.actual_seqlen_k - n_block * kBlockN;
            const int need_pad_k_idx = Max_Mn / 32;
            const int round_4 = Max_Mn % 8;

            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);

            if (need_pad_k_idx == 0 && round_4 != 0) {
                {
                    const int col = 0 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 0) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 0);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori);

            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            if (need_pad_k_idx == 1 && round_4 != 0) {
                {
                    const int col = 1 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 1) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 1);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori);

            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV_tail(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            if (need_pad_k_idx == 0 && round_4 != 0) {
                {
                    const int col = 0 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 0) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 0);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_tail_ori);

            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV_tail(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            if (need_pad_k_idx == 1 && round_4 != 0) {
                {
                    const int col = 1 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 1) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 1);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_tail_ori);
            S_BARRIER;

        }

        if (n_block > 0) { 

            if constexpr (Has_block_table) {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                // cur_block_table = block_table[n_block - 1];
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                // index_t offset_v = cur_block_table * params.v_batch_stride;
                gK.data() = gK_data + (offset_k);
                gV.data() = gV_data + (offset_k);
                gV_tail.data() = gV_tail_data + (offset_k);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            }

            lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 0, params.k_row_stride, params.d);

        }
        
        if (n_masking_steps > 1 && n_block <= 0) {
            --n_block;
            break;
        }
    }

    #pragma unroll
    for (; n_block >= 0; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");;
        // flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 0)), (tCrK_copy_view(_, _, 0)));
        Tensor tGrQ_  = recast<uint_byte_t<16>>(tGrQ); 
        cute::gemm(tiled_mma, tGrQ_(_, _, 0), tSrK(_, _, 0), acc_s_ori);
        S_BARRIER;
        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);

        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");;
        // flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 1)), (tCrK_copy_view(_, _, 1)));
        cute::gemm(tiled_mma, tGrQ_(_, _, 1), tSrK(_, _, 1), acc_s_ori);
        S_BARRIER;

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, acc_o_tail, scale_softmax_log2);
        
        Tensor rP = flash::convert_type_fp8<Element>(acc_s);

        {   
            lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV_tail, sV_tail, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV_tail, sV_tail, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);

            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            // flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            cute::copy(smem_tiled_copy_V, (tSsV(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori);

            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            // flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            cute::copy(smem_tiled_copy_V, (tSsV(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori);

            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            // flash::gemm_k_rs(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            cute::copy(smem_tiled_copy_V, (tSsV_tail(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_tail_ori);
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            // flash::gemm_k_rs(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            cute::copy(smem_tiled_copy_V, (tSsV_tail(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_tail_ori);
            S_BARRIER;

            
        }

        if (n_block > 0) { 
            if constexpr (Has_block_table) {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                // cur_block_table = block_table[n_block - 1];
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                // index_t offset_v = cur_block_table * params.v_batch_stride;
                gK.data() = gK_data + (offset_k);
                gV.data() = gV_data + (offset_k);
                gV_tail.data() = gV_tail_data + (offset_k);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            }
            lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 0, params.k_row_stride, params.d);
        }

    }
#endif
    // // Epilogue
#if 1

    Tensor lse = softmax.template normalize_softmax_lse_fp8</*Is_dropout=*/false, Split>(acc_o, acc_o_tail, scale_softmax, v_descale);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ( bidb * params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }
    using result_type = cutlass::Array<ElementO, 2>;
    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) * 4 + ni * 16;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ei += 4) {

                    if (Is_even_K || col < params.d_value) {
                        if constexpr (std::is_same_v<ElementO, cutlass::bfloat16_t>) {
                            auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(ei, mi, ni),   acc_o(ei+1, mi, ni), false);
                            auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(ei+2, mi, ni), acc_o(ei+3, mi, ni), false);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        } else {
                            auto d0 = __builtin_hcu_cvt_pk_f16_f32(acc_o(ei, mi, ni),   acc_o(ei+1, mi, ni), false, 0);
                            auto d1 = __builtin_hcu_cvt_pk_f16_f32(acc_o(ei+2, mi, ni), acc_o(ei+3, mi, ni), false, 0);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        }
                    } 
                    col += 4;
                }
            }
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 64;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ei += 4) {

                    if (Is_even_K || col < params.d_value) {
                        if constexpr (std::is_same_v<ElementO, cutlass::bfloat16_t>) {
                            auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o_tail(ei, mi, ni),   acc_o_tail(ei+1, mi, ni), false);
                            auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o_tail(ei+2, mi, ni), acc_o_tail(ei+3, mi, ni), false);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        } else {
                            auto d0 = __builtin_hcu_cvt_pk_f16_f32(acc_o_tail(ei, mi, ni),    acc_o_tail(ei+1, mi, ni), false, 0);
                            auto d1 = __builtin_hcu_cvt_pk_f16_f32(acc_o_tail(ei+2, mi, ni),  acc_o_tail(ei+3, mi, ni), false, 0);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        }
                    } 

                    col += 4;
                }
            }
        }
    } 
#endif
#endif
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_fp8_dim192(const Params &params, const int bidb, const int bidh, const int m_block) {
#if 1
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;
    using ElementOUT = typename Kernel_traits::ElementO;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, ElementOUT, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = (params.seqlen_k + kBlockN - 1) / kBlockN ;
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = !Has_block_table? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = !Has_block_table
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = !Has_block_table
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV0 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gV1 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v + 64 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));    
    Tensor gV2 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v + 2 * 64 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV0 = make_tensor(sK.data() + (0), typename Kernel_traits::SmemLayoutV{});
    Tensor sV1 = make_tensor(sK.data() + size(sV0), typename Kernel_traits::SmemLayoutV{});    
    Tensor sV2 = make_tensor(sK.data() + 2*size(sV0), typename Kernel_traits::SmemLayoutV{});

    typename Kernel_traits::TiledMma16x64_LIT tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x64x32_NN tiled_mma_gemm1;
    auto thr_mma_gemm1 = tiled_mma_gemm1.get_thread_slice(tidx);

    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)

    // Tensor tSgS  = thr_mma.partition_C(gP);
    #if 1
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrV  = thr_mma_gemm1.partition_fragment_B(sV0);                // (MMA, MMA_K,MMA_N)
    //
    // Copy Atom retiling
    //
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64_Blayout_LIT tiled_mma_BLayout;
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x64, 192/64>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tSsV0 = smem_thr_copy_V.partition_S(sV0);
    auto tSsV1 = smem_thr_copy_V.partition_S(sV1);
    auto tSsV2 = smem_thr_copy_V.partition_S(sV2);

#endif
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                binfo.actual_seqlen_q - m_block * kBlockM);
    __syncthreads();

    int n_block = n_block_max - 1;
    Tensor acc_o_ori0 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_ori1 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_ori2 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o_ori0);
    clear(acc_o_ori1);
    clear(acc_o_ori2);
    Tensor acc_o0 = make_tensor(acc_o_ori0.data(), convert_layout_acc(acc_o_ori0.layout()));
    Tensor acc_o1 = make_tensor(acc_o_ori1.data(), convert_layout_acc(acc_o_ori1.layout()));
    Tensor acc_o2 = make_tensor(acc_o_ori2.data(), convert_layout_acc(acc_o_ori2.layout()));


    flash::Softmax<size<1>(acc_o_ori0)> softmax;
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    // constexpr int k0_loops = 2;
    // constexpr int k1_loops = 2;
    // constexpr int kStages = 1;
    auto gK_data = gK.data();
    auto gV0_data = gV0.data();
    auto gV1_data = gV1.data();
    auto gV2_data = gV2.data();

    {
        const int blocks_per_page = params.page_block_size / kBlockN;
        const int page_idx = (n_block) / blocks_per_page;
        const int tile_in_page = (n_block) % blocks_per_page;

        const int *cur_block_table_ptr = block_table + page_idx;
        int cur_block_table;
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));

        index_t offset_k = cur_block_table * params.k_batch_stride
                        + tile_in_page * kBlockN * kHeadDim;
        index_t offset_v = cur_block_table * params.v_batch_stride
                        + tile_in_page * kBlockN ;
        gK.data() = gK_data + (offset_k);
        gV0.data() = gV0_data + (offset_v);
        gV1.data() = gV1_data + (offset_v);
        gV2.data() = gV2_data + (offset_v);

    }

    float q_descale = params.q_descale_ptr == nullptr ? 1.0f : params.q_descale_ptr[0];
    float k_descale = params.k_descale_ptr == nullptr ? 1.0f : params.k_descale_ptr[0];
    float v_descale = params.v_descale_ptr == nullptr ? 1.0f : params.v_descale_ptr[0];


    const float scale_softmax_log2 = params.scale_softmax_log2*q_descale*k_descale;
    const float scale_softmax = params.scale_softmax*q_descale*k_descale;
    
    Tensor tCrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
    Tensor tCrV_copy_view = smem_thr_copy_V.retile_D(tOrV);

    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64_LIT, 0, false>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64_LIT, 0, false>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64_LIT, 0, false>(gK, sK, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        // lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64_LIT, 0, false>(gK, sK, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");;
        Tensor tGrQ_  = recast<uint_byte_t<16>>(tGrQ); 
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 0)), (tCrK_copy_view(_, _, 0)));
        cute::gemm(tiled_mma, tGrQ_(_, _, 0), tSrK(_, _, 0), acc_s_ori);
        S_BARRIER;     
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV0, sV0, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV0, sV0, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");;
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 1)), (tCrK_copy_view(_, _, 1)));
        cute::gemm(tiled_mma, tGrQ_(_, _, 1), tSrK(_, _, 1), acc_s_ori);
        S_BARRIER;     
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV1, sV1, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV1, sV1, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");;
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 2)), (tCrK_copy_view(_, _, 2)));
        cute::gemm(tiled_mma, tGrQ_(_, _, 2), tSrK(_, _, 2), acc_s_ori);
        S_BARRIER;  
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV2, sV2, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV2, sV2, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);

        S_BARRIER;  

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );

        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o0, acc_o1, acc_o2, scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o0, acc_o1, acc_o2, scale_softmax_log2);

        Tensor rP = flash::convert_type_fp8<Element>(acc_s);


        {
           
            const int Max_Mn = binfo.actual_seqlen_k - n_block * kBlockN;
            const int need_pad_k_idx = Max_Mn / 32;
            const int round_4 = Max_Mn % 8;

            asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV0(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);

            if (need_pad_k_idx == 0 && round_4 != 0) {
                {
                    const int col = 0 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 0) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 0);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori0);

            asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV0(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            if (need_pad_k_idx == 1 && round_4 != 0) {
                {
                    const int col = 1 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 1) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 1);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori0);

            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV1(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            if (need_pad_k_idx == 0 && round_4 != 0) {
                {
                    const int col = 0 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 0) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 0);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori1);

            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV1(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            if (need_pad_k_idx == 1 && round_4 != 0) {
                {
                    const int col = 1 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 1) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 1);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori1);

            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV2(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            if (need_pad_k_idx == 0 && round_4 != 0) {
                {
                    const int col = 0 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 0) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 0);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori2);

            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV2(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            if (need_pad_k_idx == 1 && round_4 != 0) {
                {
                    const int col = 1 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 1) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 1);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori2);

            S_BARRIER;

        }

        if (n_block > 0) { 

            if constexpr (Has_block_table) {
                const int blocks_per_page = params.page_block_size / kBlockN;
                const int page_idx = (n_block-1) / blocks_per_page;
                const int tile_in_page = (n_block-1) % blocks_per_page;

                const int *cur_block_table_ptr = block_table + page_idx;
                int cur_block_table;
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                            "=s"(cur_block_table));

                index_t offset_k = cur_block_table * params.k_batch_stride
                                + tile_in_page * kBlockN * kHeadDim;
                index_t offset_v = cur_block_table * params.v_batch_stride
                                + tile_in_page * kBlockN ;
                gK.data() = gK_data + (offset_k);
                gV0.data() = gV0_data + (offset_v);
                gV1.data() = gV1_data + (offset_v);
                gV2.data() = gV2_data + (offset_v);

            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV0.data() = gV0.data() + (-int(kBlockN * params.v_row_stride));
            }

            lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 2, params.k_row_stride, params.d);

        }
        
        if (n_masking_steps > 1 && n_block <= 0) {
            --n_block;
            break;
        }
    }

    #pragma unroll
    for (; n_block >= 0; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");;
        // flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 0)), (tCrK_copy_view(_, _, 0)));
        Tensor tGrQ_  = recast<uint_byte_t<16>>(tGrQ); 
        cute::gemm(tiled_mma, tGrQ_(_, _, 0), tSrK(_, _, 0), acc_s_ori);
        S_BARRIER;

        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV0, sV0, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV0, sV0, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");;
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 1)), (tCrK_copy_view(_, _, 1)));
        cute::gemm(tiled_mma, tGrQ_(_, _, 1), tSrK(_, _, 1), acc_s_ori);
        S_BARRIER;

        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV1, sV1, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV1, sV1, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");;
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 2)), (tCrK_copy_view(_, _, 2)));
        cute::gemm(tiled_mma, tGrQ_(_, _, 2), tSrK(_, _, 2), acc_s_ori);
        S_BARRIER;

        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV2, sV2, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV2, sV2, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        S_BARRIER;

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o0, acc_o1, acc_o2, scale_softmax_log2);
        
        Tensor rP = flash::convert_type_fp8<Element>(acc_s);

        {   

            asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
            cute::copy(smem_tiled_copy_V, (tSsV0(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori0);

            asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
            cute::copy(smem_tiled_copy_V, (tSsV0(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori0);

            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            cute::copy(smem_tiled_copy_V, (tSsV1(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori1);
            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            cute::copy(smem_tiled_copy_V, (tSsV1(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori1);

            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            cute::copy(smem_tiled_copy_V, (tSsV2(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori2);

            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            cute::copy(smem_tiled_copy_V, (tSsV2(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori2);

            S_BARRIER;

            
        }

        if (n_block > 0) { 
            if constexpr (Has_block_table) {
                const int blocks_per_page = params.page_block_size / kBlockN;
                const int page_idx = (n_block-1) / blocks_per_page;
                const int tile_in_page = (n_block-1) % blocks_per_page;

                const int *cur_block_table_ptr = block_table + page_idx;
                int cur_block_table;
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                            "=s"(cur_block_table));

                index_t offset_k = cur_block_table * params.k_batch_stride
                                + tile_in_page * kBlockN * kHeadDim;
                index_t offset_v = cur_block_table * params.v_batch_stride
                                + tile_in_page * kBlockN ;
                gK.data() = gK_data + (offset_k);
                gV0.data() = gV0_data + (offset_v);
                gV1.data() = gV1_data + (offset_v);
                gV2.data() = gV2_data + (offset_v);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV0.data() = gV0.data() + (-int(kBlockN * params.v_row_stride));
            }
            lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 2, params.k_row_stride, params.d);
        }

    }
#endif
    // // Epilogue
#if 1

    Tensor lse = softmax.template normalize_softmax_lse_fp8</*Is_dropout=*/false, Split>(acc_o0, acc_o1, acc_o2, scale_softmax, v_descale);
   
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ( bidb * params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }
    using result_type = cutlass::Array<ElementO, 2>;
    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o0); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o0); ++ni) {
                col = (laneId / 16) * 4 + ni * 16;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o0); ei += 4) {

                    if (Is_even_K || col < params.d_value) {
                        if constexpr (std::is_same_v<ElementO, cutlass::bfloat16_t>) {
                            auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o0(ei, mi, ni), acc_o0(ei+1, mi, ni), false);
                            auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o0(ei+2, mi, ni), acc_o0(ei+3, mi, ni), false);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        } else {
                            auto d0 = __builtin_hcu_cvt_pk_f16_f32(acc_o0(ei, mi, ni),    acc_o0(ei+1, mi, ni), false, 0);
                            auto d1 = __builtin_hcu_cvt_pk_f16_f32(acc_o0(ei+2, mi, ni),  acc_o0(ei+3, mi, ni), false, 0);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        }
                    } 
                    col += 4;
                }
            }
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o1); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 64;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o1); ei += 4) {

                    if (Is_even_K || col < params.d_value) {
                        if constexpr (std::is_same_v<ElementO, cutlass::bfloat16_t>) {
                            auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o1(ei, mi, ni),    acc_o1(ei+1, mi, ni), false);
                            auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o1(ei+2, mi, ni),  acc_o1(ei+3, mi, ni), false);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        } else {
                            auto d0 = __builtin_hcu_cvt_pk_f16_f32(acc_o1(ei, mi, ni),    acc_o1(ei+1, mi, ni), false, 0);
                            auto d1 = __builtin_hcu_cvt_pk_f16_f32(acc_o1(ei+2, mi, ni),  acc_o1(ei+3, mi, ni), false, 0);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        }
                    } 

                    col += 4;
                }
            }
            for (int ni = 0; ni < size<2>(acc_o2); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 128;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o2); ei += 4) {

                    if (Is_even_K || col < params.d_value) {
                        if constexpr (std::is_same_v<ElementO, cutlass::bfloat16_t>) {
                            auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o2(ei, mi, ni),    acc_o2(ei+1, mi, ni), false);
                            auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o2(ei+2, mi, ni),  acc_o2(ei+3, mi, ni), false);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        } else {
                            auto d0 = __builtin_hcu_cvt_pk_f16_f32(acc_o2(ei, mi, ni),    acc_o2(ei+1, mi, ni), false, 0);
                            auto d1 = __builtin_hcu_cvt_pk_f16_f32(acc_o2(ei+2, mi, ni),  acc_o2(ei+3, mi, ni), false, 0);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        }
                    } 

                    col += 4;
                }
            }
        }
    } 
#endif
#endif
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_fp8_dim256(const Params &params, const int bidb, const int bidh, const int m_block) {
#if 1
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;
    using ElementOUT = typename Kernel_traits::ElementO;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, ElementOUT, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = (params.seqlen_k + kBlockN - 1) / kBlockN ;
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = !Has_block_table? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = !Has_block_table
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = !Has_block_table
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV0 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gV1 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v + 64 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));    
    Tensor gV2 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v + 2 * 64 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gV3 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v + 3 * 64 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV0 = make_tensor(sK.data() + (0), typename Kernel_traits::SmemLayoutV{});
    Tensor sV1 = make_tensor(sK.data() + size(sV0), typename Kernel_traits::SmemLayoutV{});    
    Tensor sV2 = make_tensor(sK.data() + 2*size(sV0), typename Kernel_traits::SmemLayoutV{});
    Tensor sV3 = make_tensor(sK.data() + 3*size(sV0), typename Kernel_traits::SmemLayoutV{});



    typename Kernel_traits::TiledMma16x64_LIT tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x64x32_NN tiled_mma_gemm1;
    auto thr_mma_gemm1 = tiled_mma_gemm1.get_thread_slice(tidx);

    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)

    // Tensor tSgS  = thr_mma.partition_C(gP);
    #if 1
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrV  = thr_mma_gemm1.partition_fragment_B(sV0);                // (MMA, MMA_K,MMA_N)
    //
    // Copy Atom retiling
    //
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64_Blayout_LIT tiled_mma_BLayout;
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x64, 256/64>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tSsV0 = smem_thr_copy_V.partition_S(sV0);
    auto tSsV1 = smem_thr_copy_V.partition_S(sV1);
    auto tSsV2 = smem_thr_copy_V.partition_S(sV2);
    auto tSsV3 = smem_thr_copy_V.partition_S(sV3);

#endif
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                binfo.actual_seqlen_q - m_block * kBlockM);
    __syncthreads();

    int n_block = n_block_max - 1;
    Tensor acc_o_ori0 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_ori1 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_ori2 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_ori3 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K

    clear(acc_o_ori0);
    clear(acc_o_ori1);
    clear(acc_o_ori2);
    clear(acc_o_ori3);
    Tensor acc_o0 = make_tensor(acc_o_ori0.data(), convert_layout_acc(acc_o_ori0.layout()));
    Tensor acc_o1 = make_tensor(acc_o_ori1.data(), convert_layout_acc(acc_o_ori1.layout()));
    Tensor acc_o2 = make_tensor(acc_o_ori2.data(), convert_layout_acc(acc_o_ori2.layout()));
    Tensor acc_o3 = make_tensor(acc_o_ori3.data(), convert_layout_acc(acc_o_ori3.layout()));


    flash::Softmax<size<1>(acc_o_ori0)> softmax;
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    // constexpr int k0_loops = 2;
    // constexpr int k1_loops = 2;
    // constexpr int kStages = 1;
    auto gK_data = gK.data();
    auto gV0_data = gV0.data();
    auto gV1_data = gV1.data();
    auto gV2_data = gV2.data();
    auto gV3_data = gV3.data();

    {
        int cur_block_table;
        const int *cur_block_table_ptr = block_table + (n_block);
        // cur_block_table = block_table[n_block - 1];
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
             "=s"(cur_block_table));
        index_t offset_k = cur_block_table * params.k_batch_stride;
        gK.data() = gK_data + (offset_k);
        gV0.data() = gV0_data + (offset_k);
        gV1.data() = gV1_data + (offset_k);
        gV2.data() = gV2_data + (offset_k);
        gV3.data() = gV3_data + (offset_k);

    }

    // float q_descale = params.q_descale_ptr == nullptr ? 1.0f : params.q_descale_ptr[0];
    // float k_descale = params.k_descale_ptr == nullptr ? 1.0f : params.k_descale_ptr[0];
    // float v_descale = params.v_descale_ptr == nullptr ? 1.0f : params.v_descale_ptr[0];


    // const float scale_softmax_log2 = params.scale_softmax_log2;
    // const float scale_softmax = params.scale_softmax;
    constexpr float scale_softmax =0.0625;
    constexpr float scale_softmax_log2=scale_softmax*1.4426950408889634;
    
    Tensor tCrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
    Tensor tCrV_copy_view = smem_thr_copy_V.retile_D(tOrV);

    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64_LIT, 0, false>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64_LIT, 0, false>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64_LIT, 0, false>(gK, sK, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64_LIT, 0, false>(gK, sK, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");;
        Tensor tGrQ_  = recast<uint_byte_t<16>>(tGrQ); 
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 0)), (tCrK_copy_view(_, _, 0)));
        cute::gemm(tiled_mma, tGrQ_(_, _, 0), tSrK(_, _, 0), acc_s_ori);
        S_BARRIER;     
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV0, sV0, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV0, sV0, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");;
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 1)), (tCrK_copy_view(_, _, 1)));
        cute::gemm(tiled_mma, tGrQ_(_, _, 1), tSrK(_, _, 1), acc_s_ori);
        S_BARRIER;     
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV1, sV1, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV1, sV1, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");;
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 2)), (tCrK_copy_view(_, _, 2)));
        cute::gemm(tiled_mma, tGrQ_(_, _, 2), tSrK(_, _, 2), acc_s_ori);
        S_BARRIER;  
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV2, sV2, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV2, sV2, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(6) \n s_barrier");;
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 3)), (tCrK_copy_view(_, _, 3)));
        cute::gemm(tiled_mma, tGrQ_(_, _, 3), tSrK(_, _, 3), acc_s_ori);
        S_BARRIER;  

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );

        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o0, acc_o1, acc_o2, acc_o3, scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o0, acc_o1, acc_o2, acc_o3, scale_softmax_log2);

        Tensor rP = flash::convert_type_fp8<Element>(acc_s);


        {

            lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV3, sV3, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV3, sV3, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
           
            const int Max_Mn = binfo.actual_seqlen_k - n_block * kBlockN;
            const int need_pad_k_idx = Max_Mn / 32;
            const int round_4 = Max_Mn % 8;

            asm volatile("s_waitcnt vmcnt(7) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV0(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);

            if (need_pad_k_idx == 0 && round_4 != 0) {
                {
                    const int col = 0 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 0) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 0);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori0);

            asm volatile("s_waitcnt vmcnt(6) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV0(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            if (need_pad_k_idx == 1 && round_4 != 0) {
                {
                    const int col = 1 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 1) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 1);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori0);

            // __syncthreads();
            // if(thread0()){
            //     printf("gV0 is\n");
            //     for(int i = 0;i < size(gV0);++i){
            //         printf("%f ,", float(gV0(i)));
            //         if((i + 1) % 64 == 0) printf("\n");
            //     }
            //     printf("\n");
            // }

            asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV1(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            if (need_pad_k_idx == 0 && round_4 != 0) {
                {
                    const int col = 0 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 0) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 0);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori1);

            asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV1(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            if (need_pad_k_idx == 1 && round_4 != 0) {
                {
                    const int col = 1 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 1) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 1);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori1);

            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV2(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            if (need_pad_k_idx == 0 && round_4 != 0) {
                {
                    const int col = 0 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 0) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 0);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori2);

            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV2(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            if (need_pad_k_idx == 1 && round_4 != 0) {
                {
                    const int col = 1 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 1) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 1);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori2);

            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV3(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            if (need_pad_k_idx == 0 && round_4 != 0) {
                {
                    const int col = 0 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 0) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 0);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori3);

            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV3(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            {auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            if (need_pad_k_idx == 1 && round_4 != 0) {
                {
                    const int col = 1 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 1) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 1);
                        }
                    }
                }
            }}
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori3);
            S_BARRIER;

        }

        if (n_block > 0) { 

            if constexpr (Has_block_table) {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                // cur_block_table = block_table[n_block - 1];
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                gK.data() = gK_data + (offset_k);
                gV0.data() = gV0_data + (offset_k);
                gV1.data() = gV1_data + (offset_k);
                gV2.data() = gV2_data + (offset_k);
                gV3.data() = gV3_data + (offset_k);

            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV0.data() = gV0.data() + (-int(kBlockN * params.v_row_stride));
            }

            lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 2, params.k_row_stride, params.d);

        }
        
        if (n_masking_steps > 1 && n_block <= 0) {
            --n_block;
            break;
        }
    }

    #pragma unroll
    for (; n_block >= 0; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");;
        // flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 0)), (tCrK_copy_view(_, _, 0)));
        Tensor tGrQ_  = recast<uint_byte_t<16>>(tGrQ); 
        cute::gemm(tiled_mma, tGrQ_(_, _, 0), tSrK(_, _, 0), acc_s_ori);
        S_BARRIER;

        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV0, sV0, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV0, sV0, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");;
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 1)), (tCrK_copy_view(_, _, 1)));
        cute::gemm(tiled_mma, tGrQ_(_, _, 1), tSrK(_, _, 1), acc_s_ori);
        S_BARRIER;

        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV1, sV1, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV1, sV1, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");;
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 2)), (tCrK_copy_view(_, _, 2)));
        cute::gemm(tiled_mma, tGrQ_(_, _, 2), tSrK(_, _, 2), acc_s_ori);
        S_BARRIER;

        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV2, sV2, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV2, sV2, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(6) \n s_barrier");;
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 3)), (tCrK_copy_view(_, _, 3)));
        cute::gemm(tiled_mma, tGrQ_(_, _, 3), tSrK(_, _, 3), acc_s_ori);
        S_BARRIER;

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o0, acc_o1, acc_o2, acc_o3, scale_softmax_log2);
        
        Tensor rP = flash::convert_type_fp8<Element>(acc_s);

        {   

            lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV3, sV3, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV3, sV3, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);

            asm volatile("s_waitcnt vmcnt(7) \n s_barrier");
            cute::copy(smem_tiled_copy_V, (tSsV0(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori0);

            asm volatile("s_waitcnt vmcnt(6) \n s_barrier");
            cute::copy(smem_tiled_copy_V, (tSsV0(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori0);

            asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
            cute::copy(smem_tiled_copy_V, (tSsV1(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori1);
            asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
            cute::copy(smem_tiled_copy_V, (tSsV1(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori1);

            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            cute::copy(smem_tiled_copy_V, (tSsV2(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori2);

            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            cute::copy(smem_tiled_copy_V, (tSsV2(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori2);

            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            cute::copy(smem_tiled_copy_V, (tSsV3(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori3);
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            cute::copy(smem_tiled_copy_V, (tSsV3(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori3);
            S_BARRIER;

            
        }

        if (n_block > 0) { 
            if constexpr (Has_block_table) {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                // cur_block_table = block_table[n_block - 1];
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                gK.data() = gK_data + (offset_k);
                gV0.data() = gV0_data + (offset_k);
                gV1.data() = gV1_data + (offset_k);
                gV2.data() = gV2_data + (offset_k);
                gV3.data() = gV3_data + (offset_k);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV0.data() = gV0.data() + (-int(kBlockN * params.v_row_stride));
            }
            lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 2, params.k_row_stride, params.d);
        }

    }
#endif
    // // Epilogue
#if 1

    Tensor lse = softmax.template normalize_softmax_lse_fp8</*Is_dropout=*/false, Split>(acc_o0, acc_o1, acc_o2, acc_o3, scale_softmax, 1.0f);
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ( bidb * params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }
    using result_type = cutlass::Array<ElementO, 2>;
    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o0); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o0); ++ni) {
                col = (laneId / 16) * 4 + ni * 16;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o0); ei += 4) {

                    if (Is_even_K || col < params.d_value) {
                        if constexpr (std::is_same_v<ElementO, cutlass::bfloat16_t>) {
                            auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o0(ei, mi, ni),   acc_o0(ei+1, mi, ni), false);
                            auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o0(ei+2, mi, ni),  acc_o0(ei+3, mi, ni), false);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        } else {
                            auto d0 = __builtin_hcu_cvt_pk_f16_f32(acc_o0(ei, mi, ni),   acc_o0(ei+1, mi, ni), false, 0);
                            auto d1 = __builtin_hcu_cvt_pk_f16_f32(acc_o0(ei+2, mi, ni), acc_o0(ei+3, mi, ni), false, 0);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        }
                    } 
                    col += 4;
                }
            }
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o1); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 64;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o1); ei += 4) {

                    if (Is_even_K || col < params.d_value) {
                        if constexpr (std::is_same_v<ElementO, cutlass::bfloat16_t>) {
                            auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o1(ei, mi, ni),   acc_o1(ei+1, mi, ni), false);
                            auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o1(ei+2, mi, ni), acc_o1(ei+3, mi, ni), false);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        } else {
                            auto d0 = __builtin_hcu_cvt_pk_f16_f32(acc_o1(ei, mi, ni),   acc_o1(ei+1, mi, ni), false, 0);
                            auto d1 = __builtin_hcu_cvt_pk_f16_f32(acc_o1(ei+2, mi, ni), acc_o1(ei+3, mi, ni), false, 0);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        }
                    } 

                    col += 4;
                }
            }
            for (int ni = 0; ni < size<2>(acc_o2); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 128;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o2); ei += 4) {

                    if (Is_even_K || col < params.d_value) {
                        if constexpr (std::is_same_v<ElementO, cutlass::bfloat16_t>) {
                            auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o2(ei, mi, ni),   acc_o2(ei+1, mi, ni), false);
                            auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o2(ei+2, mi, ni), acc_o2(ei+3, mi, ni), false);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        } else {
                            auto d0 = __builtin_hcu_cvt_pk_f16_f32(acc_o2(ei, mi, ni),   acc_o2(ei+1, mi, ni), false, 0);
                            auto d1 = __builtin_hcu_cvt_pk_f16_f32(acc_o2(ei+2, mi, ni),  acc_o2(ei+3, mi, ni), false, 0);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        }
                    } 

                    col += 4;
                }
            }
            for (int ni = 0; ni < size<2>(acc_o3); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 192;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o3); ei += 4) {

                    if (Is_even_K || col < params.d_value) {
                        if constexpr (std::is_same_v<ElementO, cutlass::bfloat16_t>) {
                            auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o3(ei, mi, ni),   acc_o3(ei+1, mi, ni), false);
                            auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o3(ei+2, mi, ni), acc_o3(ei+3, mi, ni), false);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        } else {
                            auto d0 = __builtin_hcu_cvt_pk_f16_f32(acc_o3(ei, mi, ni),   acc_o3(ei+1, mi, ni), false, 0);
                            auto d1 = __builtin_hcu_cvt_pk_f16_f32(acc_o3(ei+2, mi, ni),  acc_o3(ei+3, mi, ni), false, 0);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        }
                    } 

                    col += 4;
                }
            }
        }
    } 
#endif
#endif
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_kv_fp8(const Params &params, const int bidb, const int bidh, const int m_block) {
#if 1
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;
    using ElementKV = typename Kernel_traits::ElementKV;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = (params.seqlen_k + kBlockN - 1) / kBlockN ;
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = !Has_block_table? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = !Has_block_table
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = !Has_block_table
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKV *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKV *>(params.v_ptr) + row_offset_v),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gV_tail = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKV *>(params.v_ptr) + row_offset_v + 64 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor sV_tail = make_tensor(sK.data() + size(sV), typename Kernel_traits::SmemLayoutV{});

    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x64x16 tiled_mma_gemm1;
    auto thr_mma_gemm1 = tiled_mma_gemm1.get_thread_slice(tidx);

    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    // Tensor tSgS  = thr_mma.partition_C(gP);
    #if 1
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)


    Tensor tOrV  = thr_mma_gemm1.partition_fragment_B(sV);                // (MMA, MMA_K,MMA_N)



    //
    // Copy Atom retiling
    //
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    // auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_BLayout);
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, 128/32>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tSsV = smem_thr_copy_V.partition_S(sV);
    auto tSsV_tail = smem_thr_copy_V.partition_S(sV_tail);

#endif
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                binfo.actual_seqlen_q - m_block * kBlockM);
    __syncthreads();

    int n_block = n_block_max - 1;
    Tensor acc_o_ori = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_tail_ori = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o_ori);
    clear(acc_o_tail_ori);
    Tensor acc_o = make_tensor(acc_o_ori.data(), convert_layout_acc(acc_o_ori.layout()));
    Tensor acc_o_tail = make_tensor(acc_o_tail_ori.data(), convert_layout_acc(acc_o_tail_ori.layout()));

    flash::Softmax<size<1>(acc_o_ori)> softmax;
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    constexpr int k0_loops = 4;
    constexpr int k1_loops = 2;
    constexpr int kStages = 3;
    auto gK_data = gK.data();
    auto gV_data = gV.data();
    auto gV_tail_data = gV_tail.data();
    {
        int cur_block_table;
        const int *cur_block_table_ptr = block_table + (n_block);
        // cur_block_table = block_table[n_block - 1];
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
             "=s"(cur_block_table));
        index_t offset_k = cur_block_table * params.k_batch_stride;
        index_t offset_v = cur_block_table * params.v_batch_stride;
        gK.data() = gK_data + (offset_k);
        gV.data() = gV_data + (offset_v);
        gV_tail.data() = gV_tail_data + (offset_v);
    }

    float k_descale = params.k_descale_ptr == nullptr ? 1.0f : params.k_descale_ptr[0];
    float v_descale = params.v_descale_ptr == nullptr ? 1.0f : params.v_descale_ptr[0];

    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
        lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x32, 0, false>(k_descale, gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x32, 0, false>(k_descale, gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(3) lgkmcnt(3) \n s_barrier");;

            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i, i);
            S_BARRIER;
        }
        
        {
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(4) lgkmcnt(4) \n s_barrier");;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
            S_BARRIER;

            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(5) lgkmcnt(5) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2, 2);
            S_BARRIER;

            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV_tail, sV_tail, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV_tail, sV_tail, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(6) lgkmcnt(6) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3, 3);
            S_BARRIER;
        }


        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );

        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, acc_o_tail, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, acc_o_tail, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);


        {
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV_tail, sV_tail, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV_tail, sV_tail, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(7) lgkmcnt(7) \n s_barrier");
            

            flash::gemm_k_rs_pad<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(6) lgkmcnt(6) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(5) lgkmcnt(5) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(4) lgkmcnt(4) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(3) lgkmcnt(3) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(2) lgkmcnt(2) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(1) lgkmcnt(1) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);
            S_BARRIER;

            
        }

        if (n_block > 0) { 
            if constexpr (Has_block_table) {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                // cur_block_table = block_table[n_block - 1];
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                index_t offset_v = cur_block_table * params.v_batch_stride;
                gK.data() = gK_data + (offset_k);
                gV.data() = gV_data + (offset_v);
                gV_tail.data() = gV_tail_data + (offset_v);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            }
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x32, 0, false>(k_descale, gK, sK, i, params.k_row_stride, params.d);
            }
        }
        
        if (n_masking_steps > 1 && n_block <= 0) {
            --n_block;
            break;
        }
    }

    #pragma unroll
    for (; n_block >= 0; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x32, 0, false>(k_descale, gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(3) lgkmcnt(3) \n s_barrier");;

            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i, i);
            S_BARRIER;
        }
        
        {
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(4) lgkmcnt(4) \n s_barrier");;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
            S_BARRIER;

            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(5) lgkmcnt(5) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2, 2);
            S_BARRIER;

            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV_tail, sV_tail, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV_tail, sV_tail, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(6) lgkmcnt(6) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3, 3);
            S_BARRIER;
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, acc_o_tail, params.scale_softmax_log2);
        
        Tensor rP = flash::convert_type<Element>(acc_s);
        {
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV_tail, sV_tail, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV_tail, sV_tail, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(7) lgkmcnt(7) \n s_barrier");
            
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(6) lgkmcnt(6) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(5) lgkmcnt(5) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(4) lgkmcnt(4) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);
            asm volatile("s_waitcnt vmcnt(3) lgkmcnt(3) \n s_barrier");
            flash::gemm_k_rs(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(2) lgkmcnt(2) \n s_barrier");
            flash::gemm_k_rs(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(1) lgkmcnt(1) \n s_barrier");
            flash::gemm_k_rs(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0) \n s_barrier");
            flash::gemm_k_rs(acc_o_tail_ori, rP, tOrV, tSsV_tail, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);
            S_BARRIER;
            
        }

        if (n_block > 0) { 
            if constexpr (Has_block_table) {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                // cur_block_table = block_table[n_block - 1];
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                index_t offset_v = cur_block_table * params.v_batch_stride;
                gK.data() = gK_data + (offset_k);
                gV.data() = gV_data + (offset_v);
                gV_tail.data() = gV_tail_data + (offset_v);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            }
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x32, 0, false>(k_descale, gK, sK, i, params.k_row_stride, params.d);
            }
        }

    }
#endif
#if 1

    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, acc_o_tail, params.scale_softmax);
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            (bidb* params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
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
                col = (laneId / 16) * 4 + ni * 16;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ei += 4) {
                    // wangaq debug
                    // if(thread(0, 0)) {
                    //     printf("mi:%d ni:%d ei:%d row:%d col:%d acc_o:%.4f\n", 
                    //     mi, ni, ei, row, col, acc_o(ei, mi, ni));
                    // }
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o(ei + 3, mi, ni));
                    } 
                    // else 
                    //     gO(row, col) = Element(0.0);
                    col += 4;
                }
            }
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 64;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ei += 4) {
                    // wangaq debug
                    // if(thread(0, 0)) {
                    //     printf("mi:%d ni:%d ei:%d row:%d col:%d acc_o:%.4f\n", 
                    //     mi, ni, ei, row, col, acc_o(ei, mi, ni));
                    // }
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o_tail(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o_tail(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o_tail(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o_tail(ei + 3, mi, ni));
                    } 
                    // else 
                    //     gO(row, col) = Element(0.0);
                    col += 4;
                }
            }
        }
    } 
#endif
#endif

}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_kv_fp8_dim64(const Params &params, const int bidb, const int bidh, const int m_block) {
#if 1
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;
    using ElementKV = typename Kernel_traits::ElementKV;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = (params.seqlen_k + kBlockN - 1) / kBlockN ;
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = !Has_block_table? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = !Has_block_table
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = !Has_block_table
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKV *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKV *>(params.v_ptr) + row_offset_v),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutV{});

    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x64x16 tiled_mma_gemm1;
    auto thr_mma_gemm1 = tiled_mma_gemm1.get_thread_slice(tidx);

    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    // Tensor tSgS  = thr_mma.partition_C(gP);
    #if 1
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)

    Tensor tOrV  = thr_mma_gemm1.partition_fragment_B(sV);                // (MMA, MMA_K,MMA_N)



    //
    // Copy Atom retiling
    //
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, 64/32>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tSsV = smem_thr_copy_V.partition_S(sV);

#endif
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                binfo.actual_seqlen_q - m_block * kBlockM);
    __syncthreads();

    int n_block = n_block_max - 1;
    Tensor acc_o_ori = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o_ori);
    Tensor acc_o = make_tensor(acc_o_ori.data(), convert_layout_acc(acc_o_ori.layout()));

    flash::Softmax<size<1>(acc_o_ori)> softmax;
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    auto gK_data = gK.data();
    auto gV_data = gV.data();
    {
        int cur_block_table;
        const int *cur_block_table_ptr = block_table + (n_block);
        // cur_block_table = block_table[n_block - 1];
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
             "=s"(cur_block_table));
        index_t offset_k = cur_block_table * params.k_batch_stride;
        index_t offset_v = cur_block_table * params.v_batch_stride;
        gK.data() = gK_data + (offset_k);
        gV.data() = gV_data + (offset_v);
    }

    float k_descale = params.k_descale_ptr == nullptr ? 1.0f : params.k_descale_ptr[0];
    float v_descale = params.v_descale_ptr == nullptr ? 1.0f : params.v_descale_ptr[0];

    lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x32, 0, false>(k_descale, gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);


#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x32, 0, false>(k_descale, gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(1) lgkmcnt(1) \n s_barrier");;

        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
        S_BARRIER;

        {
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(2) lgkmcnt(2) \n s_barrier");;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
            S_BARRIER;

            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            S_BARRIER;
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );

        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);


        {
            asm volatile("s_waitcnt vmcnt(3) lgkmcnt(3) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(2) lgkmcnt(2) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(1) lgkmcnt(1) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);
            S_BARRIER;

            
        }

        if (n_block > 0) { 
            if constexpr (Has_block_table) {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                // cur_block_table = block_table[n_block - 1];
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                index_t offset_v = cur_block_table * params.v_batch_stride;
                gK.data() = gK_data + (offset_k);
                gV.data() = gV_data + (offset_v);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            }
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x32, 0, false>(k_descale, gK, sK, 0, params.k_row_stride, params.d);

        }
        
        if (n_masking_steps > 1 && n_block <= 0) {
            --n_block;
            break;
        }
    }

    #pragma unroll
    for (; n_block >= 0; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x32, 0, false>(k_descale, gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(1) lgkmcnt(1) \n s_barrier");;

        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
        S_BARRIER;

        
        {
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(2) lgkmcnt(2) \n s_barrier");;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
            S_BARRIER;

            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            S_BARRIER;
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        
        Tensor rP = flash::convert_type<Element>(acc_s);
        {
            asm volatile("s_waitcnt vmcnt(3) lgkmcnt(3) \n s_barrier");
            
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(2) lgkmcnt(2) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(1) lgkmcnt(1) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);
            S_BARRIER;
            
        }

        if (n_block > 0) { 
            if constexpr (Has_block_table) {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                // cur_block_table = block_table[n_block - 1];
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                index_t offset_v = cur_block_table * params.v_batch_stride;
                gK.data() = gK_data + (offset_k);
                gV.data() = gV_data + (offset_v);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            }
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x32, 0, false>(k_descale, gK, sK, 0, params.k_row_stride, params.d);

        }

    }
#endif
#if 1

    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, params.scale_softmax);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ( bidb* params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
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
                col = (laneId / 16) * 4 + ni * 16;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ei += 4) {
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o(ei + 3, mi, ni));
                    } 
                    col += 4;
                }
            }
        }
    } 
#endif
#endif

}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_kv_fp8_dim256(const Params &params, const int bidb, const int bidh, const int m_block) {
#if 1
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;
    using ElementKV = typename Kernel_traits::ElementKV;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = (params.seqlen_k + kBlockN - 1) / kBlockN ;
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = !Has_block_table? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = !Has_block_table
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = !Has_block_table
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKV *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV0 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKV *>(params.v_ptr) + row_offset_v),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gV1 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKV *>(params.v_ptr) + row_offset_v + 64 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gV2 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKV *>(params.v_ptr) + row_offset_v + 128 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gV3 = make_tensor(make_gmem_ptr(reinterpret_cast<ElementKV *>(params.v_ptr) + row_offset_v + 192 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV0 = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor sV1 = make_tensor(sK.data() + 1*size(sV0), typename Kernel_traits::SmemLayoutV{});
    Tensor sV2 = make_tensor(sK.data() + 2*size(sV0), typename Kernel_traits::SmemLayoutV{});
    Tensor sV3 = make_tensor(sK.data() + 3*size(sV0), typename Kernel_traits::SmemLayoutV{});

    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x64x16 tiled_mma_gemm1;
    auto thr_mma_gemm1 = tiled_mma_gemm1.get_thread_slice(tidx);

    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    // Tensor tSgS  = thr_mma.partition_C(gP);
    #if 1
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)


    Tensor tOrV  = thr_mma_gemm1.partition_fragment_B(sV0);                // (MMA, MMA_K,MMA_N)



    //
    // Copy Atom retiling
    //
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    // auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_BLayout);
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, 256/32>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tSsV0 = smem_thr_copy_V.partition_S(sV0);
    auto tSsV1 = smem_thr_copy_V.partition_S(sV1);
    auto tSsV2 = smem_thr_copy_V.partition_S(sV2);
    auto tSsV3 = smem_thr_copy_V.partition_S(sV3);


#endif
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                binfo.actual_seqlen_q - m_block * kBlockM);
    __syncthreads();

    int n_block = n_block_max - 1;
    Tensor acc_o_ori0 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_ori1 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_ori2 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_ori3 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K

    clear(acc_o_ori0);
    clear(acc_o_ori1);
    clear(acc_o_ori2);
    clear(acc_o_ori3);

    Tensor acc_o0 = make_tensor(acc_o_ori0.data(), convert_layout_acc(acc_o_ori0.layout()));
    Tensor acc_o1 = make_tensor(acc_o_ori1.data(), convert_layout_acc(acc_o_ori1.layout()));
    Tensor acc_o2 = make_tensor(acc_o_ori2.data(), convert_layout_acc(acc_o_ori2.layout()));
    Tensor acc_o3 = make_tensor(acc_o_ori3.data(), convert_layout_acc(acc_o_ori3.layout()));

    flash::Softmax<size<1>(acc_o_ori0)> softmax;
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    constexpr int k0_loops = 8;
    constexpr int k1_loops = 2;
    constexpr int kStages = 7;
    auto gK_data = gK.data();
    auto gV0_data = gV0.data();
    auto gV1_data = gV1.data();
    auto gV2_data = gV2.data();
    auto gV3_data = gV3.data();

    {
        int cur_block_table;
        const int *cur_block_table_ptr = block_table + (n_block);
        // cur_block_table = block_table[n_block - 1];
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
             "=s"(cur_block_table));
        index_t offset_k = cur_block_table * params.k_batch_stride;
        index_t offset_v = cur_block_table * params.v_batch_stride;
        gK.data() = gK_data + (offset_k);
        gV0.data() = gV0_data + (offset_v);
        gV1.data() = gV1_data + (offset_v);
        gV2.data() = gV2_data + (offset_v);
        gV3.data() = gV3_data + (offset_v);

    }

    float k_descale = params.k_descale_ptr == nullptr ? 1.0f : params.k_descale_ptr[0];
    float v_descale = params.v_descale_ptr == nullptr ? 1.0f : params.v_descale_ptr[0];

    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
        lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x32, 0, false>(k_descale, gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x32, 0, false>(k_descale, gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(7) lgkmcnt(7) \n s_barrier");;

            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i, i);
            S_BARRIER;
        }
        
        {
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV0, sV0, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV0, sV0, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(8) lgkmcnt(8) \n s_barrier");;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
            S_BARRIER;

            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV0, sV0, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV0, sV0, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(9) lgkmcnt(9) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2, 2);
            S_BARRIER;

            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV1, sV1, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV1, sV1, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(10) lgkmcnt(10) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3, 3);
            S_BARRIER;
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV1, sV1, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV1, sV1, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(11) lgkmcnt(11) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 4);
            S_BARRIER;
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV2, sV2, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV2, sV2, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(12) lgkmcnt(12) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 5);
            S_BARRIER;
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV2, sV2, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV2, sV2, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(13) lgkmcnt(13) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 6, 6);
            S_BARRIER;
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV3, sV3, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV3, sV3, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(14) lgkmcnt(14) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 7, 7);
            S_BARRIER;
        }


        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );

        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o0, acc_o1, acc_o2, acc_o3, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o0, acc_o1, acc_o2, acc_o3, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);


        {
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV3, sV3, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, Is_even_MN, _64x16, 0, false>(v_descale, gV3, sV3, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);

            asm volatile("s_waitcnt vmcnt(15) lgkmcnt(15) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(14) lgkmcnt(14) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(13) lgkmcnt(13) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(12) lgkmcnt(12) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);


            asm volatile("s_waitcnt vmcnt(11) lgkmcnt(11) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(10) lgkmcnt(10) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(9) lgkmcnt(9) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(8) lgkmcnt(8) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);

            asm volatile("s_waitcnt vmcnt(7) lgkmcnt(7) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(6) lgkmcnt(6) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(5) lgkmcnt(5) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(4) lgkmcnt(4) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);

            asm volatile("s_waitcnt vmcnt(3) lgkmcnt(3) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(2) lgkmcnt(2) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(1) lgkmcnt(1) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);
            S_BARRIER;

            
        }

        if (n_block > 0) { 
            if constexpr (Has_block_table) {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                // cur_block_table = block_table[n_block - 1];
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                index_t offset_v = cur_block_table * params.v_batch_stride;
                gK.data() = gK_data + (offset_k);
                gV0.data() = gV0_data + (offset_v);
                gV1.data() = gV1_data + (offset_v);
                gV2.data() = gV2_data + (offset_v);
                gV3.data() = gV3_data + (offset_v);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV0.data() = gV0.data() + (-int(kBlockN * params.v_row_stride));
            }
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x32, 0, false>(k_descale, gK, sK, i, params.k_row_stride, params.d);
            }
        }
        
        if (n_masking_steps > 1 && n_block <= 0) {
            --n_block;
            break;
        }
    }

    #pragma unroll
    for (; n_block >= 0; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x32, 0, false>(k_descale, gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(7) lgkmcnt(7) \n s_barrier");;

            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i, i);
            S_BARRIER;
        }
        
        {
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV0, sV0, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV0, sV0, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(8) lgkmcnt(8) \n s_barrier");;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
            S_BARRIER;

            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV0, sV0, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV0, sV0, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(9) lgkmcnt(9) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2, 2);
            S_BARRIER;

            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV1, sV1, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV1, sV1, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(10) lgkmcnt(10) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3, 3);
            S_BARRIER;

            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV1, sV1, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV1, sV1, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(11) lgkmcnt(11) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 4);
            S_BARRIER;

            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV2, sV2, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV2, sV2, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(12) lgkmcnt(12) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 5);
            S_BARRIER;

            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV2, sV2, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV2, sV2, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(13) lgkmcnt(13) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 6, 6);
            S_BARRIER;

            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV3, sV3, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV3, sV3, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(14) lgkmcnt(14) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 7, 7);
            S_BARRIER;
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o0, acc_o1, acc_o2, acc_o3, params.scale_softmax_log2);
        
        Tensor rP = flash::convert_type<Element>(acc_s);
        {
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV3, sV3, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x16, 0, false>(v_descale, gV3, sV3, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(15) lgkmcnt(15) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(14) lgkmcnt(14) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(13) lgkmcnt(13) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(12) lgkmcnt(12) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);

            asm volatile("s_waitcnt vmcnt(11) lgkmcnt(11) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(10) lgkmcnt(10) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(9) lgkmcnt(9) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(8) lgkmcnt(8) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);

            asm volatile("s_waitcnt vmcnt(7) lgkmcnt(7) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(6) lgkmcnt(6) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(5) lgkmcnt(5) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(4) lgkmcnt(4) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);

            asm volatile("s_waitcnt vmcnt(3) lgkmcnt(3) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(2) lgkmcnt(2) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(1) lgkmcnt(1) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);
            S_BARRIER;
            
        }

        if (n_block > 0) { 
            if constexpr (Has_block_table) {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                // cur_block_table = block_table[n_block - 1];
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                index_t offset_v = cur_block_table * params.v_batch_stride;
                gK.data() = gK_data + (offset_k);
                gV0.data() = gV0_data + (offset_v);
                gV1.data() = gV1_data + (offset_v);
                gV2.data() = gV2_data + (offset_v);
                gV3.data() = gV3_data + (offset_v);

            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV0.data() = gV0.data() + (-int(kBlockN * params.v_row_stride));
            }
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy_kv_fp8<Element, Is_even_K, true, _64x32, 0, false>(k_descale, gK, sK, i, params.k_row_stride, params.d);
            }
        }

    }
#endif
#if 1
    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o0, acc_o1, acc_o2, acc_o3, params.scale_softmax);
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ( bidb * params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o0); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o0); ++ni) {
                col = (laneId / 16) * 4 + ni * 16;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o0); ei += 4) {
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o0(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o0(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o0(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o0(ei + 3, mi, ni));
                    } 
                    col += 4;
                }
            }
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o1); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 64;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o1); ei += 4) {
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o1(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o1(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o1(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o1(ei + 3, mi, ni));
                    } 
                    col += 4;
                }
            }
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o2); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 128;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o2); ei += 4) {
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o2(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o2(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o2(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o2(ei + 3, mi, ni));
                    } 
                    col += 4;
                }
            }
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o3); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 192;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o3); ei += 4) {
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o3(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o3(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o3(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o3(ei + 3, mi, ni));
                    } 
                    col += 4;
                }
            }
        }
    } 
#endif
#endif

}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_dim192(const Params &params, const int bidb, const int bidh, const int m_block) {
#if 1
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = (params.seqlen_k + kBlockN - 1) / kBlockN ;
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = !Has_block_table? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = !Has_block_table
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = !Has_block_table
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV0 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gV1 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v + 64 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gV2 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v + 128 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));    
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV0 = make_tensor(sK.data() + (0), typename Kernel_traits::SmemLayoutV{});
    Tensor sV1 = make_tensor(sK.data() + size(sV0), typename Kernel_traits::SmemLayoutV{});
    Tensor sV2 = make_tensor(sK.data() + size(sV0)*2, typename Kernel_traits::SmemLayoutV{});
    // Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    // Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});


    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x64x16 tiled_mma_gemm1;
    auto thr_mma_gemm1 = tiled_mma_gemm1.get_thread_slice(tidx);

    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    // Tensor tSgS  = thr_mma.partition_C(gP);
    #if 1
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrV  = thr_mma_gemm1.partition_fragment_B(sV0);                // (MMA, MMA_K,MMA_N)
    //
    // Copy Atom retiling
    //
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    // auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_BLayout);
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, 192/32>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tSsV0 = smem_thr_copy_V.partition_S(sV0);
    auto tSsV1 = smem_thr_copy_V.partition_S(sV1);
    auto tSsV2 = smem_thr_copy_V.partition_S(sV2);
#endif
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                binfo.actual_seqlen_q - m_block * kBlockM);
    __syncthreads();

    int n_block = n_block_max - 1;
    Tensor acc_o_ori0 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_ori1 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_ori2 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o_ori0);
    clear(acc_o_ori1);
    clear(acc_o_ori2);
    Tensor acc_o0 = make_tensor(acc_o_ori0.data(), convert_layout_acc(acc_o_ori0.layout()));
    Tensor acc_o1 = make_tensor(acc_o_ori1.data(), convert_layout_acc(acc_o_ori1.layout()));
    Tensor acc_o2 = make_tensor(acc_o_ori2.data(), convert_layout_acc(acc_o_ori2.layout()));

    flash::Softmax<size<1>(acc_o_ori0)> softmax;
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    constexpr int k0_loops = 6;
    constexpr int kStages = 5;
    auto gK_data = gK.data();
    auto gV0_data = gV0.data();
    auto gV1_data = gV1.data();
    auto gV2_data = gV2.data();
    {
        const int blocks_per_page = params.page_block_size / kBlockN;
        const int page_idx = (n_block) / blocks_per_page;
        const int tile_in_page = (n_block) % blocks_per_page;

        const int *cur_block_table_ptr = block_table + page_idx;
        int cur_block_table;
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));

        index_t offset_k = cur_block_table * params.k_batch_stride
                        + tile_in_page * kBlockN * kHeadDim;
        index_t offset_v = cur_block_table * params.v_batch_stride
                        + tile_in_page * kBlockN ;
        gK.data() = gK_data + (offset_k);
        gV0.data() = gV0_data + (offset_v);
        gV1.data() = gV1_data + (offset_v);
        gV2.data() = gV2_data + (offset_v);
    }

    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            // load 3 4 5 -> 3 0 1 
            // k0/k1 0 1 2 
            lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i, i);
            S_BARRIER;
        }
        
        {
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV0, sV0, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV0, sV0, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(6) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV0, sV0, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV0, sV0, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(7) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2, 2);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV1, sV1, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV1, sV1, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(8) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3, 3);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV1, sV1, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV1, sV1, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(9) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 4);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV2, sV2, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV2, sV2, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(10) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 5);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV2, sV2, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV2, sV2, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            S_BARRIER;
        }
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );

        }
        //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o0, acc_o1, acc_o2,  params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o0, acc_o1, acc_o2,  params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);
        {
            asm volatile("s_waitcnt vmcnt(11) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(10) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(9) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(8) \n s_barrier");

            flash::gemm_k_rs_pad<Element>(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(7) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(6) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);

            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);
            S_BARRIER;
            
        }

        if (n_block > 0) { 
            if constexpr (Has_block_table) {
                const int blocks_per_page = params.page_block_size / kBlockN;
                const int page_idx = (n_block-1) / blocks_per_page;
                const int tile_in_page = (n_block-1) % blocks_per_page;

                const int *cur_block_table_ptr = block_table + page_idx;
                int cur_block_table;
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                            "=s"(cur_block_table));

                index_t offset_k = cur_block_table * params.k_batch_stride
                                + tile_in_page * kBlockN * kHeadDim;
                index_t offset_v = cur_block_table * params.v_batch_stride
                                + tile_in_page * kBlockN ;
                gK.data() = gK_data + (offset_k);
                gV0.data() = gV0_data + (offset_v);
                gV1.data() = gV1_data + (offset_v);
                gV2.data() = gV2_data + (offset_v);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV0.data() = gV0.data() + (-int(kBlockN * params.v_row_stride));
            }
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, i, params.k_row_stride, params.d);
            }
        }
        
        if (n_masking_steps > 1 && n_block <= 0) {
            --n_block;
            break;
        }
    }

    #pragma unroll
    for (; n_block >= 0; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            // load 3 4 5 -> 3 0 1 
            // k0/k1 0 1 2 
            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i, i);
        }
        
        {
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV0, sV0, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV0, sV0, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(6) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
            S_BARRIER;
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV0, sV0, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV0, sV0, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(7) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2, 2);
            S_BARRIER;
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV1, sV1, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV1, sV1, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(8) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3, 3);
            S_BARRIER;
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV1, sV1, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV1, sV1, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(9) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 4);
            S_BARRIER;
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV2, sV2, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV2, sV2, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(10) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 5);
            S_BARRIER;
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV2, sV2, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV2, sV2, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            S_BARRIER;
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        //TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o0, acc_o1, acc_o2, params.scale_softmax_log2);
        
        Tensor rP = flash::convert_type<Element>(acc_s);
        {
            asm volatile("s_waitcnt vmcnt(11) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(10) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(9) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(8) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);

            asm volatile("s_waitcnt vmcnt(7) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(6) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);

            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);

            S_BARRIER;            
        }

        if (n_block > 0) { 
            if constexpr (Has_block_table) {
                const int blocks_per_page = params.page_block_size / kBlockN;
                const int page_idx = (n_block-1) / blocks_per_page;
                const int tile_in_page = (n_block-1) % blocks_per_page;

                const int *cur_block_table_ptr = block_table + page_idx;
                int cur_block_table;
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                            "=s"(cur_block_table));

                index_t offset_k = cur_block_table * params.k_batch_stride
                                + tile_in_page * kBlockN * kHeadDim;
                index_t offset_v = cur_block_table * params.v_batch_stride
                                + tile_in_page * kBlockN ;
                gK.data() = gK_data + (offset_k);
                gV0.data() = gV0_data + (offset_v);
                gV1.data() = gV1_data + (offset_v);
                gV2.data() = gV2_data + (offset_v);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV0.data() = gV0.data() + (-int(kBlockN * params.v_row_stride));
            }
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, i, params.k_row_stride, params.d);
            }
        }

    }
#endif
    // // Epilogue
#if 1
    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o0, acc_o1, acc_o2, params.scale_softmax);
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ( bidb * params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o0); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o0); ++ni) {
                col = (laneId / 16) * 4 + ni * 16;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o0); ei += 4) {
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o0(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o0(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o0(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o0(ei + 3, mi, ni));
                    } 
                    col += 4;
                }
            }
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o1); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 64;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o1); ei += 4) {
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o1(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o1(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o1(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o1(ei + 3, mi, ni));
                    } 
                    col += 4;
                }
            }
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o2); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 128;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o2); ei += 4) {
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o2(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o2(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o2(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o2(ei + 3, mi, ni));
                    } 
                    col += 4;
                }
            }
        }
    } 
#endif
#endif
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_dim256(const Params &params, const int bidb, const int bidh, const int m_block) {
#if 1
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = (params.seqlen_k + kBlockN - 1) / kBlockN ;
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = !Has_block_table? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = !Has_block_table
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = !Has_block_table
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV0 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gV1 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v + 64 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gV2 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v + 128 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));    
    Tensor gV3 = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v + 192 * params.v_row_stride),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV0 = make_tensor(sK.data() + (0), typename Kernel_traits::SmemLayoutV{});
    Tensor sV1 = make_tensor(sK.data() + size(sV0), typename Kernel_traits::SmemLayoutV{});
    Tensor sV2 = make_tensor(sK.data() + size(sV0)*2, typename Kernel_traits::SmemLayoutV{});
    Tensor sV3 = make_tensor(sK.data() + size(sV0)*3, typename Kernel_traits::SmemLayoutV{});
    // Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    // Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});


    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x64x16 tiled_mma_gemm1;
    auto thr_mma_gemm1 = tiled_mma_gemm1.get_thread_slice(tidx);

    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    // Tensor tSgS  = thr_mma.partition_C(gP);
    #if 1
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)


    Tensor tOrV  = thr_mma_gemm1.partition_fragment_B(sV0);                // (MMA, MMA_K,MMA_N)



    //
    // Copy Atom retiling
    //
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    // auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_BLayout);
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, 256/32>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tSsV0 = smem_thr_copy_V.partition_S(sV0);
    auto tSsV1 = smem_thr_copy_V.partition_S(sV1);
    auto tSsV2 = smem_thr_copy_V.partition_S(sV2);
    auto tSsV3 = smem_thr_copy_V.partition_S(sV3);

#endif
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                binfo.actual_seqlen_q - m_block * kBlockM);
    __syncthreads();

    int n_block = n_block_max - 1;
    Tensor acc_o_ori0 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_ori1 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_ori2 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_ori3 = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o_ori0);
    clear(acc_o_ori1);
    clear(acc_o_ori2);
    clear(acc_o_ori3);
    Tensor acc_o0 = make_tensor(acc_o_ori0.data(), convert_layout_acc(acc_o_ori0.layout()));
    Tensor acc_o1 = make_tensor(acc_o_ori1.data(), convert_layout_acc(acc_o_ori1.layout()));
    Tensor acc_o2 = make_tensor(acc_o_ori2.data(), convert_layout_acc(acc_o_ori2.layout()));
    Tensor acc_o3 = make_tensor(acc_o_ori3.data(), convert_layout_acc(acc_o_ori3.layout()));

    flash::Softmax<size<1>(acc_o_ori0)> softmax;
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    constexpr int k0_loops = 8;
    constexpr int k1_loops = 2;
    constexpr int kStages = 7;
    auto gK_data = gK.data();
    auto gV0_data = gV0.data();
    auto gV1_data = gV1.data();
    auto gV2_data = gV2.data();
    auto gV3_data = gV3.data();
    {
        int cur_block_table;
        const int *cur_block_table_ptr = block_table + (n_block);
        // cur_block_table = block_table[n_block - 1];
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
             "=s"(cur_block_table));
        index_t offset_k = cur_block_table * params.k_batch_stride;
        // index_t offset_v = cur_block_table * params.v_batch_stride;
        gK.data() = gK_data + (offset_k);
        gV0.data() = gV0_data + (offset_k);
        gV1.data() = gV1_data + (offset_k);
        gV2.data() = gV2_data + (offset_k);
        gV3.data() = gV3_data + (offset_k);
    }

    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            // load 3 4 5 -> 3 0 1 
            // k0/k1 0 1 2 
            lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(7) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i, i);
            S_BARRIER;
        }
        
        {
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV0, sV0, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV0, sV0, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(8) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV0, sV0, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV0, sV0, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(9) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2, 2);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV1, sV1, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV1, sV1, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(10) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3, 3);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV1, sV1, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV1, sV1, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(11) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 4);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV2, sV2, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV2, sV2, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(12) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 5);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV2, sV2, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV2, sV2, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(13) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 6, 6);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV3, sV3, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV3, sV3, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(14) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 7, 7);
            S_BARRIER;
        }
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );

        }
        //TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o0, acc_o1, acc_o2, acc_o3,  params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o0, acc_o1, acc_o2, acc_o3,  params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);


        {
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV3, sV3, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV3, sV3, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            
            asm volatile("s_waitcnt vmcnt(15) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(14) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(13) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(12) \n s_barrier");

            flash::gemm_k_rs_pad<Element>(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(11) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(10) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(9) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(8) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);

            asm volatile("s_waitcnt vmcnt(7) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(6) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);

            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);
            S_BARRIER;
            
        }

        if (n_block > 0) { 
            // gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            // gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            if constexpr (Has_block_table) {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                // cur_block_table = block_table[n_block - 1];
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                // index_t offset_v = cur_block_table * params.v_batch_stride;
                gK.data() = gK_data + (offset_k);
                gV0.data() = gV0_data + (offset_k);
                gV1.data() = gV1_data + (offset_k);
                gV2.data() = gV2_data + (offset_k);
                gV3.data() = gV3_data + (offset_k);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV0.data() = gV0.data() + (-int(kBlockN * params.v_row_stride));
            }
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, i, params.k_row_stride, params.d);
            }
        }
        
        if (n_masking_steps > 1 && n_block <= 0) {
            --n_block;
            break;
        }
    }

    #pragma unroll
    for (; n_block >= 0; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; ++i) {
            // load 3 4 5 -> 3 0 1 
            // k0/k1 0 1 2 
            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(7) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, i, i);

        }
        
        {
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV0, sV0, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV0, sV0, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(8) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
            S_BARRIER;
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV0, sV0, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV0, sV0, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(9) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2, 2);
            S_BARRIER;
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV1, sV1, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV1, sV1, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(10) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3, 3);
            S_BARRIER;
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV1, sV1, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV1, sV1, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(11) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 4);
            S_BARRIER;
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV2, sV2, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV2, sV2, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(12) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 5);
            S_BARRIER;
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV2, sV2, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV2, sV2, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(13) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 6, 6);
            S_BARRIER;
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV3, sV3, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV3, sV3, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(14) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 7, 7);
            S_BARRIER;
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }
        //TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o0, acc_o1, acc_o2, acc_o3, params.scale_softmax_log2);
        
        Tensor rP = flash::convert_type<Element>(acc_s);
        {
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV3, sV3, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV3, sV3, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);


            asm volatile("s_waitcnt vmcnt(15) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(14) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(13) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(12) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori0, rP, tOrV, tSsV0, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);

            asm volatile("s_waitcnt vmcnt(11) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(10) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(9) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(8) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori1, rP, tOrV, tSsV1, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);

            asm volatile("s_waitcnt vmcnt(7) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(6) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori2, rP, tOrV, tSsV2, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);

            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori3, rP, tOrV, tSsV3, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);
            S_BARRIER;            
        }

        if (n_block > 0) { 
            if constexpr (Has_block_table) {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                // cur_block_table = block_table[n_block - 1];
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                // index_t offset_v = cur_block_table * params.v_batch_stride;
                gK.data() = gK_data + (offset_k);
                gV0.data() = gV0_data + (offset_k);
                gV1.data() = gV1_data + (offset_k);
                gV2.data() = gV2_data + (offset_k);
                gV3.data() = gV3_data + (offset_k);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV0.data() = gV0.data() + (-int(kBlockN * params.v_row_stride));
            }
            #pragma unroll
            for (int i = 0; i < kStages; ++i) {
                lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, i, params.k_row_stride, params.d);
            }
        }

    }
#endif
    // // Epilogue
#if 1
    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o0, acc_o1, acc_o2, acc_o3, params.scale_softmax);
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ( bidb * params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o0); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o0); ++ni) {
                col = (laneId / 16) * 4 + ni * 16;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o0); ei += 4) {
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o0(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o0(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o0(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o0(ei + 3, mi, ni));
                    } 
                    col += 4;
                }
            }
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o1); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 64;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o1); ei += 4) {
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o1(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o1(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o1(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o1(ei + 3, mi, ni));
                    } 
                    col += 4;
                }
            }
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o2); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 128;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o2); ei += 4) {
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o2(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o2(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o2(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o2(ei + 3, mi, ni));
                    } 
                    col += 4;
                }
            }
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o3); ++ni) {
                col = (laneId / 16) * 4 + ni * 16 + 192;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o3); ei += 4) {
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o3(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o3(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o3(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o3(ei + 3, mi, ni));
                    } 
                    col += 4;
                }
            }
        }
    } 
#endif
#endif
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, bool Use_alibi_sqrt, bool Use_qq_bias, bool Use_mm_prefix, typename Params>
inline __device__ void compute_attn_1rowblock_unified_16x64_prefetch(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {
#if 1
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    const int n_block_min = !Is_local
        ? n_split_idx * n_blocks_per_split
        : std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
    static constexpr bool Effective_causal = Is_causal || (Use_alibi_sqrt && !Is_local);
    if ((Effective_causal || Is_local)) {
    // if ((Is_causal || Is_local) && !Use_mm_prefix) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    if (n_block_min >= n_block_max) {  // This also covers the case where n_block_max <= 0
        // We exit early and write 0 to gOaccum and -inf to gLSEaccum.
        // Otherwise we might read OOB elements from gK and gV,
        // or get wrong results when we combine gOaccum from different blocks.
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
            + m_block * kBlockM) * params.d_value_rounded;
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                      Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                     make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                      Shape<Int<kBlockM>>{}, Stride<_1>{});

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        clear(tOrOaccum);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgOaccum); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSEaccum(row) = Split ? -INFINITY : INFINITY; }
        }
        return;
    }
    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = !Has_block_table? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = !Has_block_table
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = !Has_block_table
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + (0), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});
    #if 1
    typename Kernel_traits::GmemTiledCopyQKVPaged gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);

    Tensor tKgK_ = gmem_thr_copy_KV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK_ = gmem_thr_copy_KV.partition_D(sK);
    Tensor tVgV_ = gmem_thr_copy_KV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV_ = gmem_thr_copy_KV.partition_D(sV);

    Tensor tKgK = make_tensor(tKgK_.data(), reshape_thread_tile(tKgK_.layout()));
    Tensor tKsK = make_tensor(tKsK_.data(), reshape_thread_tile(tKsK_.layout()));
    Tensor tVgV = make_tensor(tVgV_.data(), reshape_thread_tile(tVgV_.layout()));
    Tensor tVsV = make_tensor(tVsV_.data(), reshape_thread_tile(tVsV_.layout()));

    #endif

    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    // Tensor tSgS  = thr_mma.partition_C(gP);
    #if 1
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)


    Tensor tOrVt  = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    //
    // Copy Atom retiling
    //
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    // auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_BLayout);
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, 128/32>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVtSplit);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x128, 4>(tOsVt8x64.layout()));

    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));


    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);


    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));


    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }

    }

    // Read Q from gmem to smem, optionally apply rotary embedding.

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                    binfo.actual_seqlen_q - m_block * kBlockM);
    __syncthreads();


    int n_block = n_block_max - 1;
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Effective_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Effective_causal) ? cute::ceil_div(kBlockM, kBlockN)
                                            : cute::ceil_div(kBlockM, kBlockN) + 1);


    auto gK_data = gK.data();
    auto gV_data = gV.data();

    if constexpr (Has_block_table) {
        const int blocks_per_page = params.page_block_size / kBlockN;
        const int page_idx = (n_block) / blocks_per_page;
        const int tile_in_page = (n_block) % blocks_per_page;

        const int *cur_block_table_ptr = block_table + page_idx;
        int cur_block_table;
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));

        index_t offset_k = cur_block_table * params.k_batch_stride
                        + tile_in_page * kBlockN * params.k_row_stride;
        index_t offset_v = cur_block_table * params.v_batch_stride
                        + tile_in_page * kBlockN * params.v_row_stride;

        gK.data() = gK_data + offset_k;
        gV.data() = gV_data + offset_v;
    }
    __builtin_amdgcn_sched_barrier(0);
    lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    __builtin_amdgcn_sched_barrier(0);

    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s_ori);
        {
            lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128, 0, false>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128, 0, false>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2, 2);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128, 0, false>(gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3, 3);
            S_BARRIER;
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_unified<Effective_causal, Is_even_MN, Use_alibi_sqrt, Use_qq_bias, Use_mm_prefix>(
                acc_s, n_block * kBlockN, row_idx_offset_, warp_row_stride, binfo.actual_seqlen_k - binfo.actual_seqlen_q,
                params.qq_bias_ptr, params.qq_bias_stride_0, params.mm_prefix_range_ptr, params.max_mm_ranges,bidb,params.scale_softmax);
        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Effective_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Effective_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        {
            __builtin_amdgcn_sched_barrier(0);
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128, 0, false>(gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            s_waitcnt<3>();
            __builtin_amdgcn_sched_barrier(0);
            flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
            s_waitcnt<2>();
            flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
            s_waitcnt<1>();
            flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
            s_waitcnt<0>();
            flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
            S_BARRIER;
            __builtin_amdgcn_sched_barrier(0);
        }

        if (n_block > n_block_min) {
            if (Has_block_table) {
                const int blocks_per_page = params.page_block_size / kBlockN;
                const int page_idx = (n_block -1) / blocks_per_page;
                const int tile_in_page = (n_block -1) % blocks_per_page;

                const int *cur_block_table_ptr = block_table + page_idx;
                int cur_block_table;
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                            "=s"(cur_block_table));

                index_t offset_k = cur_block_table * params.k_batch_stride
                                + tile_in_page * kBlockN * params.k_row_stride;
                index_t offset_v = cur_block_table * params.v_batch_stride
                                + tile_in_page * kBlockN * params.v_row_stride;

                gK.data() = gK_data + offset_k;
                gV.data() = gV_data + offset_v;
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            }

            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, 2, params.k_row_stride, params.d);
        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s_ori);
        {
            __builtin_amdgcn_sched_barrier(0);
            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK,  3, params.k_row_stride, params.d);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128, 0, false>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128, 0, false>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2, 2);
            S_BARRIER;
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128, 0, false>(gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3, 3);
            S_BARRIER;
            __builtin_amdgcn_sched_barrier(0);
        }

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            mask.template apply_mask_continuous_unified</*Causal_mask=*/false,/*Is_even_MN=*/true, Use_alibi_sqrt, Use_qq_bias, Use_mm_prefix>(
                acc_s, n_block * kBlockN, row_idx_offset_, warp_row_stride, binfo.actual_seqlen_k - binfo.actual_seqlen_q,
                params.qq_bias_ptr, params.qq_bias_stride_0, params.mm_prefix_range_ptr, params.max_mm_ranges, bidb, params.scale_softmax);
        }


        softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        {
            __builtin_amdgcn_sched_barrier(0);
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128, 0, false>(gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            s_waitcnt<3>();
            __builtin_amdgcn_sched_barrier(0);
            flash::gemm_k_rs_ds_read_m32x16<0>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
            s_waitcnt<2>();
            flash::gemm_k_rs_ds_read_m32x16<1>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
            s_waitcnt<1>();
            flash::gemm_k_rs_ds_read_m32x16<2>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
            s_waitcnt<0>();
            flash::gemm_k_rs_ds_read_m32x16<3>(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
            S_BARRIER;
            __builtin_amdgcn_sched_barrier(0);
        }

        if (n_block > n_block_min) {
            if (Has_block_table) {
                const int blocks_per_page = params.page_block_size / kBlockN;
                const int page_idx = (n_block -1) / blocks_per_page;
                const int tile_in_page = (n_block -1) % blocks_per_page;

                const int *cur_block_table_ptr = block_table + page_idx;
                int cur_block_table;
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                            "=s"(cur_block_table));

                index_t offset_k = cur_block_table * params.k_batch_stride
                                + tile_in_page * kBlockN * params.k_row_stride;
                index_t offset_v = cur_block_table * params.v_batch_stride
                                + tile_in_page * kBlockN * params.v_row_stride;

                gK.data() = gK_data + offset_k;
                gV.data() = gV_data + offset_v;
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            }

            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, 2, params.k_row_stride, params.d);
        }
    }

#endif
    // // Epilogue
#if 1
    // ★ Attention Sinks: Conditional normalize based on split index ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr && n_split_idx == 0) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) {
            tSrS_aux(mi) = s_aux_val;
        }
        lse = softmax.template normalize_softmax_lse_with_sinks</*Is_dropout=*/false, Split>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2);
    } else {
        lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(
            acc_o, params.scale_softmax);
    }
    if constexpr (Split) {
        // Tensor sOaccum = make_tensor(sAccs.data() + size(sAccs), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
        Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
        // Partition sO to match the accumulator partitioning
        using SmemTiledCopyO = std::conditional_t<
            !Split,
            typename Kernel_traits::SmemCopyAtomO,
            typename Kernel_traits::SmemCopyAtomOaccum
        >;
        auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma_for_gemm1);
        auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor rO = flash::convert_type<ElementO>(acc_o);
        Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        __syncthreads();

        cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                            + m_block * kBlockM) * params.d_value_rounded;
        const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
                ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
            ) + m_block * kBlockM;

        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                    Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                    make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                    Shape<Int<kBlockM>>{}, Stride<_1>{});


        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

        __syncthreads();

        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma_for_gemm1.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        static_assert(decltype(size<0>(taccOcO))::value == 8);

        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
            }
        }

        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
    } else {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
                ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
            ) + m_block * kBlockM;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                    Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                    make_stride(params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                    Shape<Int<kBlockM>>{}, Stride<_1>{});

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)

        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
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
                        if (Is_even_K || col < params.d_value) {
                            gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        } else
                            gO(row, col) = Element(0.0);
                        col += 4;
                    }
                }
            }
        }
    }
#endif
#endif
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, bool Use_alibi_sqrt, bool Use_qq_bias, bool Use_mm_prefix, typename Params>
inline __device__ void compute_attn_1rowblock_unified_16x64_prefetch_dim256(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {
#if 1
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    const int n_block_min = !Is_local
        ? n_split_idx * n_blocks_per_split
        : std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
    constexpr bool Effective_causal = Is_causal || (Use_alibi_sqrt && !Is_local);
    if ((Effective_causal || Is_local)) {
    // if ((Is_causal || Is_local) && !Use_mm_prefix) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    if (n_block_min >= n_block_max) {  // This also covers the case where n_block_max <= 0
        // We exit early and write 0 to gOaccum and -inf to gLSEaccum.
        // Otherwise we might read OOB elements from gK and gV,
        // or get wrong results when we combine gOaccum from different blocks.
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
            + m_block * kBlockM) * params.d_value_rounded;
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                      Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                     make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                      Shape<Int<kBlockM>>{}, Stride<_1>{});

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        clear(tOrOaccum);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgOaccum); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSEaccum(row) = Split ? -INFINITY : INFINITY; }
        }
        return;
    }
    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = !Has_block_table? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = !Has_block_table
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = block_table == nullptr
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + (0), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtSplit = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransSplit{});
    #if 1
    typename Kernel_traits::GmemTiledCopyQKVPaged gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);

    Tensor tKgK_ = gmem_thr_copy_KV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK_ = gmem_thr_copy_KV.partition_D(sK);
    Tensor tVgV_ = gmem_thr_copy_KV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV_ = gmem_thr_copy_KV.partition_D(sV);

    Tensor tKgK = make_tensor(tKgK_.data(), reshape_thread_tile(tKgK_.layout()));
    Tensor tKsK = make_tensor(tKsK_.data(), reshape_thread_tile(tKsK_.layout()));
    Tensor tVgV = make_tensor(tVgV_.data(), reshape_thread_tile(tVgV_.layout()));
    Tensor tVsV = make_tensor(tVsV_.data(), reshape_thread_tile(tVsV_.layout()));

    #endif

    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x32 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);
    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    // Tensor tSgS  = thr_mma.partition_C(gP);
    #if 1
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)


    Tensor tOrVt  = thr_mma_for_gemm1.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)

    //
    // Copy Atom retiling
    //
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    // auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_BLayout);
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, 256/32>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt8x64 = smem_thr_copy_V.partition_S(sVtSplit);
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x128, 4>(tOsVt8x64.layout()));

    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));


    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);


    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));


    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }

    }

    // Read Q from gmem to smem, optionally apply rotary embedding.

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                    binfo.actual_seqlen_q - m_block * kBlockM);
    __syncthreads();


    int n_block = n_block_max - 1;
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    // acc_o+1;
    Tensor acc_o_split = local_tile(acc_o, Shape<Int<8>, Int<1>, Int<kHeadDimV / 32 / 2>>{}, make_coord(0, 0, _)); 
    auto acc_o_temp0 = acc_o_split(_, _, _, 0);
    // acc_o_temp0+1;

    auto acc_o_temp1 = acc_o_split(_, _, _, 1);
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Effective_causal && !Is_local)
    ? 1
    : ((Is_even_MN && Effective_causal) ? cute::ceil_div(kBlockM, kBlockN)
                                        : cute::ceil_div(kBlockM, kBlockN) + 1);
    

    auto gK_data = gK.data();
    auto gV_data = gV.data();

    if constexpr (Has_block_table) {
        const int blocks_per_page = params.page_block_size / kBlockN;
        const int page_idx = (n_block) / blocks_per_page;
        const int tile_in_page = (n_block) % blocks_per_page;

        const int *cur_block_table_ptr = block_table + page_idx;
        int cur_block_table;
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));

        index_t offset_k = cur_block_table * params.k_batch_stride
                        + tile_in_page * kBlockN * params.k_row_stride;
        index_t offset_v = cur_block_table * params.v_batch_stride
                        + tile_in_page * kBlockN * params.v_row_stride;

        gK.data() = gK_data + offset_k;
        gV.data() = gV_data + offset_v;
    }
    
    lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);


    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
        S_BARRIER;
        lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, 4, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
        S_BARRIER;
        lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, 5, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2, 2);
        S_BARRIER;
        lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, 6, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3, 3);
        S_BARRIER;
        lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, 7, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 4);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 5);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 6, 6);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 7, 7);
        S_BARRIER;

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_unified<Effective_causal, Is_even_MN, Use_alibi_sqrt, Use_qq_bias, Use_mm_prefix>(
                acc_s, n_block * kBlockN, row_idx_offset_, warp_row_stride, binfo.actual_seqlen_k - binfo.actual_seqlen_q,
                params.qq_bias_ptr, params.qq_bias_stride_0, params.mm_prefix_range_ptr, params.max_mm_ranges,bidb,params.scale_softmax);
        }
        
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Effective_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Effective_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 0, gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 1, gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 2, gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 3, gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        if (n_block > n_block_min) { 
            if (Has_block_table) {
                const int blocks_per_page = params.page_block_size / kBlockN;
                const int page_idx = (n_block -1) / blocks_per_page;
                const int tile_in_page = (n_block -1) % blocks_per_page;

                const int *cur_block_table_ptr = block_table + page_idx;
                int cur_block_table;
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                            "=s"(cur_block_table));

                index_t offset_k = cur_block_table * params.k_batch_stride
                                + tile_in_page * kBlockN * params.k_row_stride;
                index_t offset_v = cur_block_table * params.v_batch_stride
                                + tile_in_page * kBlockN * params.v_row_stride;

                gK.data() = gK_data + offset_k;
                gV.data() = gV_data + offset_v;
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            }

            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, 2, params.k_row_stride, params.d);
        }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK,  3, params.k_row_stride, params.d);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
        S_BARRIER;        
        lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK,  4, params.k_row_stride, params.d);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
        S_BARRIER;        
        lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK,  5, params.k_row_stride, params.d);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2, 2);
        S_BARRIER;
        lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK,  6, params.k_row_stride, params.d);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3, 3);
        S_BARRIER;
        lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK,  7, params.k_row_stride, params.d);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 4, 4);
        S_BARRIER;

        lds_direct_copy<Is_even_K, true, _16x256>(0, 0, gV, sV, 0, params.v_row_stride, params.d_value);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 5, 5);
        S_BARRIER;

        lds_direct_copy<Is_even_K, true, _16x256>(0, 1, gV, sV, 1, params.v_row_stride, params.d_value);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 6, 6);
        S_BARRIER;

        lds_direct_copy<Is_even_K, true, _16x256>(0, 2, gV, sV, 2, params.v_row_stride, params.d_value);
        S_WAITCNT;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 7, 7);
        S_BARRIER;

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            
            mask.template apply_mask_continuous_unified</*Causal_mask=*/false,/*Is_even_MN=*/true, Use_alibi_sqrt, Use_qq_bias, Use_mm_prefix>(
                acc_s, n_block * kBlockN, row_idx_offset_, warp_row_stride, binfo.actual_seqlen_k - binfo.actual_seqlen_q,
                params.qq_bias_ptr, params.qq_bias_stride_0, params.mm_prefix_range_ptr, params.max_mm_ranges, bidb, params.scale_softmax);
        }
        

        softmax.template softmax_rescale_o</*Is_first=*/false,  /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);
        
        lds_direct_copy<Is_even_K, true, _16x256>(0, 3, gV, sV, 3, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, true, _16x256>(1, 0, gV, sV, 0, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, true, _16x256>(1, 1, gV, sV, 1, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, true, _16x256>(1, 2, gV, sV, 2, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp0, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
        
        lds_direct_copy<Is_even_K, true, _16x256>(1, 3, gV, sV, 3, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();

        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_o_temp1, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
        s_barrier();
                
        if (n_block > n_block_min) { 
            if (Has_block_table) {
                const int blocks_per_page = params.page_block_size / kBlockN;
                const int page_idx = (n_block -1) / blocks_per_page;
                const int tile_in_page = (n_block -1) % blocks_per_page;

                const int *cur_block_table_ptr = block_table + page_idx;
                int cur_block_table;
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                            "=s"(cur_block_table));

                index_t offset_k = cur_block_table * params.k_batch_stride
                                + tile_in_page * kBlockN * params.k_row_stride;
                index_t offset_v = cur_block_table * params.v_batch_stride
                                + tile_in_page * kBlockN * params.v_row_stride;

                gK.data() = gK_data + offset_k;
                gV.data() = gV_data + offset_v;
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            }

            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, 2, params.k_row_stride, params.d);
        }
    }

#endif
    // // Epilogue
#if 1
    // ★ Attention Sinks: Conditional normalize based on split index ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr && n_split_idx == 0) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) {
            tSrS_aux(mi) = s_aux_val;
        }
        lse = softmax.template normalize_softmax_lse_with_sinks</*Is_dropout=*/false, Split>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2);
    } else {
        lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(
            acc_o, params.scale_softmax);
    }
    if constexpr (Split) {
        // Tensor sOaccum = make_tensor(sAccs.data() + size(sAccs), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
        Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
        // Partition sO to match the accumulator partitioning
        using SmemTiledCopyO = std::conditional_t<
            !Split,
            typename Kernel_traits::SmemCopyAtomO,
            typename Kernel_traits::SmemCopyAtomOaccum
        >;
        auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma_for_gemm1);
        auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor rO = flash::convert_type<ElementO>(acc_o);
        Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        __syncthreads();

        cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                            + m_block * kBlockM) * params.d_value_rounded;
        const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
                ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
            ) + m_block * kBlockM;

        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                    Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                    make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                    Shape<Int<kBlockM>>{}, Stride<_1>{});


        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

        __syncthreads();

        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma_for_gemm1.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        static_assert(decltype(size<0>(taccOcO))::value == 8);

        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
            }
        }

        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_value; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
    } else {
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
                ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
            ) + m_block * kBlockM;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                    Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                    make_stride(params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                    Shape<Int<kBlockM>>{}, Stride<_1>{});

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        
        if (get<1>(taccOcO(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO(0, mi, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
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
                        if (Is_even_K || col < params.d_value) {
                            gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        } else 
                            gO(row, col) = Element(0.0);
                        col += 4;
                    }
                }
            }
        } 
    }
#endif
#endif
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_fp8_dim64(const Params &params, const int bidb, const int bidh, const int m_block) {
#if 1
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;
    using ElementOUT = typename Kernel_traits::ElementO;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, ElementOUT, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = (params.seqlen_k + kBlockN - 1) / kBlockN ;
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = !Has_block_table? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = !Has_block_table
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = !Has_block_table
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + (0), typename Kernel_traits::SmemLayoutV{});



    typename Kernel_traits::TiledMma16x64_LIT tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x64x32_NN tiled_mma_gemm1;
    auto thr_mma_gemm1 = tiled_mma_gemm1.get_thread_slice(tidx);

    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)

    // Tensor tSgS  = thr_mma.partition_C(gP);
    #if 1
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrV  = thr_mma_gemm1.partition_fragment_B(sV);                // (MMA, MMA_K,MMA_N)
    //
    // Copy Atom retiling
    //
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64_Blayout_LIT tiled_mma_BLayout;
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x64, 128/64>(tSsKBLayout.layout()));

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tSsV = smem_thr_copy_V.partition_S(sV);

#endif
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                binfo.actual_seqlen_q - m_block * kBlockM);
    __syncthreads();

    int n_block = n_block_max - 1;
    Tensor acc_o_ori = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o_ori);
    Tensor acc_o = make_tensor(acc_o_ori.data(), convert_layout_acc(acc_o_ori.layout()));

    flash::Softmax<size<1>(acc_o_ori)> softmax;
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    constexpr int k0_loops = 2;
    constexpr int k1_loops = 2;
    constexpr int kStages = 1;
    auto gK_data = gK.data();
    auto gV_data = gV.data();
    {
        int cur_block_table;
        const int *cur_block_table_ptr = block_table + (n_block);
        // cur_block_table = block_table[n_block - 1];
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
             "=s"(cur_block_table));
        index_t offset_k = cur_block_table * params.k_batch_stride;
        index_t offset_v = cur_block_table * params.v_batch_stride;
        gK.data() = gK_data + (offset_k);
        gV.data() = gV_data + (offset_v);
    }

    float q_descale = params.q_descale_ptr == nullptr ? 1.0f : params.q_descale_ptr[0];
    float k_descale = params.k_descale_ptr == nullptr ? 1.0f : params.k_descale_ptr[0];
    float v_descale = params.v_descale_ptr == nullptr ? 1.0f : params.v_descale_ptr[0];


    const float scale_softmax_log2 = params.scale_softmax_log2*q_descale*k_descale;
    const float scale_softmax = params.scale_softmax*q_descale*k_descale;

    Tensor tCrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
    Tensor tCrV_copy_view = smem_thr_copy_V.retile_D(tOrV);


#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x64_LIT, 0, false>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");;
        // flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
        Tensor tGrQ_  = recast<uint_byte_t<16>>(tGrQ); 
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 0)), (tCrK_copy_view(_, _, 0)));
        cute::gemm(tiled_mma, tGrQ_(_, _, 0), tSrK(_, _, 0), acc_s_ori);
        S_BARRIER;     
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, Is_even_MN, _64x32, 0, false>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        S_BARRIER;     

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );

        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, scale_softmax_log2);

        Tensor rP = flash::convert_type_fp8<Element>(acc_s);


        {
           
            const int Max_Mn = binfo.actual_seqlen_k - n_block * kBlockN;
            const int need_pad_k_idx = Max_Mn / 32;
            const int round_4 = Max_Mn % 8;
            auto acc_layout = tOrV.layout();
            auto k32_layout = make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)),get<1>(acc_layout)),get<2>(acc_layout));
            auto tOrV_k32 = make_tensor(tOrV.data(), k32_layout);
            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV(_, _, 0)), (tCrV_copy_view(_, _, 0)));

            if (need_pad_k_idx == 0 && round_4 != 0) {
                {
                    const int col = 0 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 0) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 0);
                        }
                    }
                }
            }
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori);

            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            // flash::gemm_k_rs_pad_V_K32<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            cute::copy(smem_tiled_copy_V, (tSsV(_, _, 1)), (tCrV_copy_view(_, _, 1)));

            if (need_pad_k_idx == 1 && round_4 != 0) {
                {
                    const int col = 1 * 32 + ((tidx >> 4) & 3) * 8;

                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tOrV_k32); ni++) {
                        #pragma unroll
                        for (int ei = 0; ei < size<0>(tOrV_k32); ei++) {
                            tOrV_k32(ei, ni, 1) = col + ei >= Max_Mn ? Element(0) : tOrV_k32(ei, ni, 1);
                        }
                    }
                }
            }
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori);
            S_BARRIER;

        }

        if (n_block > 0) { 

            if constexpr (Has_block_table) {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                // cur_block_table = block_table[n_block - 1];
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                index_t offset_v = cur_block_table * params.v_batch_stride;
                gK.data() = gK_data + (offset_k);
                gV.data() = gV_data + (offset_v);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            }


        }
        
        if (n_masking_steps > 1 && n_block <= 0) {
            --n_block;
            break;
        }
    }

    #pragma unroll
    for (; n_block >= 0; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy_fp8<Is_even_K, true, _64x64_LIT, 0, false>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");;
        // flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
        cute::copy(smem_tiled_copy_K, (tSsK(_, _, 0)), (tCrK_copy_view(_, _, 0)));
        Tensor tGrQ_  = recast<uint_byte_t<16>>(tGrQ); 
        cute::gemm(tiled_mma, tGrQ_(_, _, 0), tSrK(_, _, 0), acc_s_ori);
        S_BARRIER;
        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_fp8<Is_even_K, true, _64x32, 0, false>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);


        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous_fp8<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, scale_softmax_log2);
        
        Tensor rP = flash::convert_type_fp8<Element>(acc_s);

        {   
            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            // flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            cute::copy(smem_tiled_copy_V, (tSsV(_, _, 0)), (tCrV_copy_view(_, _, 0)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 0), tOrV(_, _, 0), acc_o_ori);

            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            // flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            cute::copy(smem_tiled_copy_V, (tSsV(_, _, 1)), (tCrV_copy_view(_, _, 1)));
            cute::gemm(tiled_mma_gemm1, rP(_, _, 1), tOrV(_, _, 1), acc_o_ori);
            S_BARRIER;

            
        }

        if (n_block > 0) { 
            if constexpr (Has_block_table) {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                // cur_block_table = block_table[n_block - 1];
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                index_t offset_v = cur_block_table * params.v_batch_stride;
                gK.data() = gK_data + (offset_k);
                gV.data() = gV_data + (offset_v);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            }
        }

    }
#endif
    // // Epilogue
#if 1

    Tensor lse = softmax.template normalize_softmax_lse_fp8</*Is_dropout=*/false, Split>(acc_o, scale_softmax,v_descale);
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            (bidb * params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }
    using result_type = cutlass::Array<ElementO, 2>;
    int row, col;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_o); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                col = (laneId / 16) * 4 + ni * 16;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ei += 4) {

                    if (Is_even_K || col < params.d_value) {
                        if constexpr (std::is_same_v<ElementO, cutlass::bfloat16_t>) {
                            auto d0 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(ei, mi, ni),   acc_o(ei+1, mi, ni), false);
                            auto d1 = __builtin_hcu_cvt_pk_bf16_f32(acc_o(ei+2, mi, ni), acc_o(ei+3, mi, ni), false);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        } else {
                            auto d0 = __builtin_hcu_cvt_pk_f16_f32(acc_o(ei, mi, ni),   acc_o(ei+1, mi, ni), false, 0);
                            auto d1 = __builtin_hcu_cvt_pk_f16_f32(acc_o(ei+2, mi, ni),  acc_o(ei+3, mi, ni), false, 0);
                            auto res0 = reinterpret_cast<result_type const &>(d0);
                            auto res1 = reinterpret_cast<result_type const &>(d1);
                            gO(row, col)     = res0[0];
                            gO(row, col + 1) = res0[1];
                            gO(row, col + 2) = res1[0];
                            gO(row, col + 3) = res1[1];
                        }
                    } 
                    col += 4;
                }
            }
        }
    } 
#endif
#endif
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_dim64(const Params &params, const int bidb, const int bidh, const int m_block) {
#if 1
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;

    // Shared memory.
    extern __shared__ char smem_[];
    // (Attention Sinks) s_aux is read from global memory; no shared-memory staging needed.

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    // (Attention Sinks) s_aux is read from global memory; no shared-memory staging needed.
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = (params.seqlen_k + kBlockN - 1) / kBlockN ;
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = !Has_block_table? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = !Has_block_table
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = !Has_block_table
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<64>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    // Tensor gV_tail = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v + 64 * params.v_row_stride),
    //                         Shape<Int<64>, Int<kBlockN>>{},
    //                         make_stride(params.v_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + (0), typename Kernel_traits::SmemLayoutV{});
    // Tensor sV_tail = make_tensor(sK.data() + size(sV), typename Kernel_traits::SmemLayoutV{});
    // Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    // Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});


    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x64x16 tiled_mma_gemm1;
    auto thr_mma_gemm1 = tiled_mma_gemm1.get_thread_slice(tidx);

    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    // Tensor tSgS  = thr_mma.partition_C(gP);
    #if 1
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)


    Tensor tOrV  = thr_mma_gemm1.partition_fragment_B(sV);                // (MMA, MMA_K,MMA_N)

    //
    // Copy Atom retiling
    //
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    // auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_BLayout);
                                                                                                            //Blayout中将nk更换了位置，后续的shape和stride也转换了，相当于逻辑上转置
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout); //正常拷贝进lds 在skv 按正常layout写进去
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);                            //skv是kn*128的 sk是(kn*2)*64的 
    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});                          //然后按照sk的blayout方式将数据读出来计算
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, 64/32>(tSsKBLayout.layout()));


    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tSsV = smem_thr_copy_V.partition_S(sV);
    // auto tSsV_tail = smem_thr_copy_V.partition_S(sV_tail);

#endif
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                binfo.actual_seqlen_q - m_block * kBlockM);
    __syncthreads();

    int n_block = n_block_max - 1;
    Tensor acc_o_ori = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    // Tensor acc_o_tail_ori = partition_fragment_C(tiled_mma_gemm1, Shape<Int<kBlockM>, Int<64>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o_ori);
    // clear(acc_o_tail_ori);
    Tensor acc_o = make_tensor(acc_o_ori.data(), convert_layout_acc(acc_o_ori.layout()));
    // Tensor acc_o_tail = make_tensor(acc_o_tail_ori.data(), convert_layout_acc(acc_o_tail_ori.layout()));

    flash::Softmax<size<1>(acc_o_ori)> softmax;
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    constexpr int k0_loops = 2;
    constexpr int kStages = 1;
    auto gK_data = gK.data();
    auto gV_data = gV.data();
    // auto gV_tail_data = gV_tail.data();
    {
        const int blocks_per_page = params.page_block_size / kBlockN;
        const int page_idx = (n_block) / blocks_per_page;
        const int tile_in_page = (n_block) % blocks_per_page;

        const int *cur_block_table_ptr = block_table + page_idx;
        int cur_block_table;
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));

        index_t offset_k = cur_block_table * params.k_batch_stride
                        + tile_in_page * kBlockN * kHeadDim;
        index_t offset_v = cur_block_table * params.v_batch_stride
                        + tile_in_page * kBlockN ;
        gK.data() = gK_data + (offset_k);
        gV.data() = gV_data + (offset_v);
        // gV_tail.data() = gV_tail_data + (offset_v);
    }


    lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);


#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy<Is_even_K, Is_even_MN, _64x32, 0, false>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        S_WAITCNT;
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
        // S_BARRIER;
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
        S_BARRIER;

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );

        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);


        {
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);

            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _64x16, 0, false>(gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);

            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            flash::gemm_k_rs_pad<Element>(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, binfo.actual_seqlen_k - n_block * kBlockN);
            S_BARRIER;
            
        }

        if (n_block > 0) { 
            if constexpr (Has_block_table) {
                const int blocks_per_page = params.page_block_size / kBlockN;
                const int page_idx = (n_block-1) / blocks_per_page;
                const int tile_in_page = (n_block-1) % blocks_per_page;

                const int *cur_block_table_ptr = block_table + page_idx;
                int cur_block_table;
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                            "=s"(cur_block_table));

                index_t offset_k = cur_block_table * params.k_batch_stride
                                + tile_in_page * kBlockN * kHeadDim;
                index_t offset_v = cur_block_table * params.v_batch_stride
                                + tile_in_page * kBlockN ;
                gK.data() = gK_data + (offset_k);
                gV.data() = gV_data + (offset_v);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            }
            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, 0, params.k_row_stride, params.d);
        }
        
        if (n_masking_steps > 1 && n_block <= 0) {
            --n_block;
            break;
        }
    }

    #pragma unroll
    for (; n_block >= 0; n_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);


        lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        S_WAITCNT;
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0, 0);
        // S_BARRIER;
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");;
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1, 1);
        S_BARRIER;


        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        
        Tensor rP = flash::convert_type<Element>(acc_s);
        {
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);

            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, true, _64x16, 0, false>(gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);

            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 0, 0);
            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 1, 1);
            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 2, 2);
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            flash::gemm_k_rs(acc_o_ori, rP, tOrV, tSsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, 3, 3);
            S_BARRIER;
            
        }

        if (n_block > 0) { 
            if constexpr (Has_block_table) {
                const int blocks_per_page = params.page_block_size / kBlockN;
                const int page_idx = (n_block-1) / blocks_per_page;
                const int tile_in_page = (n_block-1) % blocks_per_page;

                const int *cur_block_table_ptr = block_table + page_idx;
                int cur_block_table;
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                            "=s"(cur_block_table));

                index_t offset_k = cur_block_table * params.k_batch_stride
                                + tile_in_page * kBlockN * kHeadDim;
                index_t offset_v = cur_block_table * params.v_batch_stride
                                + tile_in_page * kBlockN ;
                gK.data() = gK_data + (offset_k);
                gV.data() = gV_data + (offset_v);
            } else {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            }
            lds_direct_copy<Is_even_K, true, _64x32, 0, false>(gK, sK, 0, params.k_row_stride, params.d);

        }

    }
#endif
    // // Epilogue
#if 1

    // ★ Attention Sinks: Conditional normalize based on split index ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) {
            tSrS_aux(mi) = s_aux_val;
        }
        lse = softmax.template normalize_softmax_lse_with_sinks</*Is_dropout=*/false, Split>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2);
    } else {
        lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(
            acc_o, params.scale_softmax);
    }
    // Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, acc_o_tail, params.scale_softmax);
    
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ( bidb * params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                make_stride(params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
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
                col = (laneId / 16) * 4 + ni * 16;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ei += 4) {
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o(ei + 3, mi, ni));
                    } 
                    // else 
                    //     gO(row, col) = Element(0.0);
                    col += 4;
                }
            }
        }
    } 
#endif
#endif
}
template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_gfx928(const Params &params, const int bidb, const int bidh, const int m_block) {
#if 1
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Element = typename Kernel_traits::Element;

    // Shared memory.
    extern __shared__ char smem_[];
    // (Attention Sinks) s_aux is read from global memory; no shared-memory staging needed.

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = (params.seqlen_k + kBlockN - 1) / kBlockN ;
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = !Has_block_table? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const index_t row_offset_k = !Has_block_table
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = !Has_block_table
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kHeadDimV>, Int<kBlockN>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutV{});
    // Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    // Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});


    typename Kernel_traits::TiledMma16x64 tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma16x64x16 tiled_mma_gemm1;
    auto thr_mma_gemm1 = tiled_mma_gemm1.get_thread_slice(tidx);

    Tensor tGrQ  = thr_mma.partition_fragment_A(gQ);                           // (MMA,MMA_M,MMA_K)
    // Tensor tSgS  = thr_mma.partition_C(gP);
    #if 1
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)


    Tensor tOrV  = thr_mma_gemm1.partition_fragment_B(sV);                // (MMA, MMA_K,MMA_N)



    //
    // Copy Atom retiling
    //
    auto gmem_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_Q.partition_S(gQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    auto tOsV = smem_thr_copy_V.partition_S(sV);

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);

    Tensor tKgK = gmem_thr_copy_KV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_KV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_KV.partition_D(sV);


#endif
    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));

    Tensor cK = make_identity_tensor(make_shape(size<0>(gK), size<1>(gK)));
    Tensor tKcK = gmem_thr_copy_KV.partition_S(cK);
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));

    Tensor cV = make_identity_tensor(make_shape(size<0>(gV), size<1>(gV)));
    Tensor tVcV = gmem_thr_copy_KV.partition_S(cV);
    Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tVsV)));


    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_Q, tSgQ, tGrQ, tQcQ, tQpQ,
                                binfo.actual_seqlen_q - m_block * kBlockM);
    __syncthreads();

    int n_block = n_block_max - 1;

    #pragma unroll
    for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) <  binfo.actual_seqlen_k - n_block * kBlockN; }


    Tensor acc_o_ori = partition_fragment_C(thr_mma_gemm1, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o_ori);
    Tensor acc_o = make_tensor(acc_o_ori.data(), convert_layout_acc(acc_o_ori.layout()));
    flash::Softmax<size<1>(acc_o_ori)> softmax;
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);

    // auto gK_data = gK.data();
    // auto gV_data = gV.data();
    auto tKgK_data = tKgK.data();
    auto tVgV_data = tVgV.data();
    {
        int cur_block_table;
        const int *cur_block_table_ptr = block_table + (n_block);
        // cur_block_table = block_table[n_block - 1];
        asm volatile("s_load_dword %1, %0, 0x0\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t":
                    "+s"(cur_block_table_ptr),
             "=s"(cur_block_table));
        index_t offset_k = cur_block_table * params.k_batch_stride;
        // index_t offset_v = cur_block_table * params.v_batch_stride;
        tKgK.data() = tKgK_data + (offset_k);
        tVgV.data() = tVgV_data + (offset_k);
    }
#if 1
    auto tKrK = make_fragment_like(tKsK);
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KV, tKgK, tKrK, tKcK, tKpK,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
#endif

#if 1
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        __syncthreads();
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
        cute::copy(tKrK, tKsK);
        __syncthreads();

        auto tVrV = make_fragment_like(tVsV);
        // V矩阵转置了
        flash::copy_v<true, true, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_KV, tVgV, tVrV, tVcV, tVpV
        );
        
        flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );

        }

        cute::copy(tVrV, tVsV);
        __syncthreads();

        if (n_block > 0) {
            // Advance gK
            {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                cur_block_table = block_table[n_block - 1];
                // asm volatile("s_load_dword %1, %0, 0x0\n\t"
                //             "s_waitcnt lgkmcnt(0)\n\t":
                //             "+s"(cur_block_table_ptr),
                //     "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                index_t offset_v = cur_block_table * params.v_batch_stride;
                tKgK.data() = tKgK_data + (offset_k);
                tVgV.data() = tVgV_data + (offset_v);
            }
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_KV, tKgK, tKrK, tKcK, tKpK);
        }

        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);
        flash::gemm_rs_pad<Element>(acc_o_ori, rP, tOrV, tOsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V, binfo.actual_seqlen_k - n_block * kBlockN);

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= 0) {
            --n_block;
            break;
        }
    }

    #pragma unroll
    for (; n_block >= 0; n_block--) {
        __syncthreads();
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);
        cute::copy(tKrK, tKsK);
        __syncthreads();
        auto tVrV = make_fragment_like(tVsV);

        flash::copy_v<true, true, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_KV, tVgV, tVrV, tVcV, tVpV, binfo.actual_seqlen_k - n_block * kBlockN
        );

        flash::gemm_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K);


        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        cute::copy(tVrV, tVsV);
        __syncthreads();

        if (n_block > 0) {
            // Advance gK
            {
                int cur_block_table;
                const int *cur_block_table_ptr = block_table + (n_block - 1);
                // cur_block_table = block_table[n_block - 1];
                asm volatile("s_load_dword %1, %0, 0x0\n\t"
                            "s_waitcnt lgkmcnt(0)\n\t":
                            "+s"(cur_block_table_ptr),
                    "=s"(cur_block_table));
                index_t offset_k = cur_block_table * params.k_batch_stride;
                // index_t offset_v = cur_block_table * params.v_batch_stride;
                tKgK.data() = tKgK_data + (offset_k);
                tVgV.data() = tVgV_data + (offset_k);
            }
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_KV, tKgK, tKrK, tKcK, tKpK);
        }


        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local>(acc_s, acc_o, params.scale_softmax_log2);
        
        Tensor rP = flash::convert_type<Element>(acc_s);
        flash::gemm_rs(acc_o_ori, rP, tOrV, tOsV, tiled_mma_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

    }
#endif

#if 1

    // ★ Attention Sinks: Conditional normalize based on split index ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Load s_aux directly from global memory (no shared-memory staging).
        float s_aux_val = static_cast<float>(reinterpret_cast<Element const*>(params.s_aux_ptr)[bidh]);
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) {
            tSrS_aux(mi) = s_aux_val;
        }
        lse = softmax.template normalize_softmax_lse_with_sinks</*Is_dropout=*/false, Split>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2);
    } else {
        lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(
            acc_o, params.scale_softmax);
    }
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            (bidb* params.h + bidh) * params.seqlen_q : bidh * params.seqlen_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                Shape<Int<kBlockM>, Int<128>>{},
                                make_stride(params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma_gemm1.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
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
                col = (laneId / 16) * 4 + ni * 16;
                #pragma unroll
                for (int ei = 0; ei < size<0>(acc_o); ei += 4) {
                    // wangaq debug
                    // if(thread(0, 0)) {
                    //     printf("mi:%d ni:%d ei:%d row:%d col:%d acc_o:%.4f params.d_value = %d \n", 
                    //     mi, ni, ei, row, col, acc_o(ei, mi, ni), params.d_value);
                    // }
                    if (Is_even_K || col < params.d_value) {
                        gO(row, col) = flash::convert_type<Element>(acc_o(ei, mi, ni));
                        gO(row, col + 1) = flash::convert_type<Element>(acc_o(ei + 1, mi, ni));
                        gO(row, col + 2) = flash::convert_type<Element>(acc_o(ei + 2, mi, ni));
                        gO(row, col + 3) = flash::convert_type<Element>(acc_o(ei + 3, mi, ni));
                    } 
                    // else 
                    //     gO(row, col) = Element(0.0);
                    col += 4;
                }
            }

        }
    } 
#endif
#endif
}

#if 0
#else
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params,  typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_16x64_prefetch_skip_softmax(const Params &params, const int bidb, const int bidh, const int m_block, index_t binfo_q_offset, index_t binfo_k_offset, index_t binfo_v_offset, index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];
    // ★ Attention Sinks: Shared memory for s_aux (max 64 heads) ★
    __shared__ ElementAccum smem_s_aux[64];
    __shared__ uint32_t skip_softmax_vote[1];

    // The thread index.
    const int tidx = threadIdx.x;

    
    // wangaq debug
    // if (tidx == 0) {
    //     uint32_t * gSkipInfo_ptr = reinterpret_cast<uint32_t*>(params.skip_blocks_info_ptr) + (bidb * params.h + bidh) * 2;
    //     printf("bidx:%d gSkipInfo_ptr:%d %d \n", blockIdx.x, *gSkipInfo_ptr, *(gSkipInfo_ptr + 1));
    // }

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    // ★ Attention Sinks: Cooperatively load s_aux into shared memory (CuTe style) ★
    if (params.s_aux_ptr != nullptr) {
        const int num_heads = params.h;
        Tensor gS_aux = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element const*>(params.s_aux_ptr)),
            make_shape(num_heads)
        );
        Tensor sS_aux = make_tensor(
            make_smem_ptr(smem_s_aux),
            Layout<Shape<Int<64>>>{}
        );
        if (tidx < num_heads) {
            sS_aux(tidx) = gS_aux(tidx);
        }
        __syncthreads();
    }

    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
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

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    const index_t row_offset_k = binfo_k_offset + (n_block_max - 1) * kBlockN * params.k_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));

    const index_t row_offset_v = binfo_v_offset + (n_block_max - 1) * kBlockN * params.v_row_stride 
                                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

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
    Tensor tOsVt = make_tensor(tOsVt8x64.data(), convert_layout_B_rowcol_<_16x128, kHeadDimV/32>(tOsVt8x64.layout()));

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
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K
    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;
    softmax.skip_softmax_threshold = params.skip_softmax_threshold_scale_factor / binfo.actual_seqlen_k;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tOsVt);
    static_assert(kStages <= k0_loops && kStages <= k1_loops, "kStages is error");
    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }
    // __builtin_amdgcn_sched_barrier(0);
    // wangaq debug
    // __syncthreads();
    // if (thread0() && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     printf("params.d:%d\n", params.d);
    //     // __half * tmp = reinterpret_cast<__half*>(sK.data().get());
    //     // int col = 8;
    //     // for (int i = 0; i < size(sK)/col; ++i) {
    //     //     printf("K:%d nblock:%d ", i, n_block);
    //     //     for (int j = 0; j < col; ++j) {
    //     //         printf("%.4f ", __half2float(tmp[i*col+j]));
    //     //     }
    //     //     printf("\n");
    //     // }
    //     printf("tSrK rank:%d %d %d %d\n", rank(tSrK).value, size<0>(tSrK).value, size<1>(tSrK).value, size<2>(tSrK).value);
    //     printf("tSsK rank:%d %d %d %d\n", rank(tSsK).value, size<0>(tSsK).value, size<1>(tSsK).value, size<2>(tSsK).value);
    // }
    // return;

    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        if (tidx == 0) *skip_softmax_vote = 1;
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);

        lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, 0, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, 1, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, 2, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3);
        s_barrier();
        

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("acc_s tid:%d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, n_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], tmp[12], tmp[13], tmp[14], tmp[15],
        //     tmp[16], tmp[17], tmp[18], tmp[19], tmp[20], tmp[21], tmp[22], tmp[23],
        //     tmp[24], tmp[25], tmp[26], tmp[27], tmp[28], tmp[29], tmp[30], tmp[31]
        //     );
        // }

        #if 1
        if constexpr (Is_softcap){
            apply_softcap(acc_s, params.softcap);
        }
        
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
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
        bool skip = false;
        if (masking_step == 0) {
            skip = softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2, skip_softmax_vote);
        } else {
            skip = softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2, skip_softmax_vote);
        }

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("exp_s tid:%d n_block:%d row_max:%10.4f %10.4f row_sum:%10.4f %10.4f | %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, n_block, softmax.row_max(0), softmax.row_max(1), softmax.row_sum(0), softmax.row_sum(1), 
        //     tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], tmp[12], tmp[13], tmp[14], tmp[15],
        //     tmp[16], tmp[17], tmp[18], tmp[19], tmp[20], tmp[21], tmp[22], tmp[23],
        //     tmp[24], tmp[25], tmp[26], tmp[27], tmp[28], tmp[29], tmp[30], tmp[31]
        //     );
        // }

        if (skip) {
            s_waitcnt<0>();
            if (n_block > n_block_min) {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

                lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
            }

            if (n_masking_steps > 1 && n_block <= n_block_min) {
                --n_block;
                break;
            }
            continue;
        }

        // Convert acc_s from fp32 to fp16/bf16
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
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gV, sV, 3, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
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

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
        }

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_o.data());
        //     printf("out tid:%d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, n_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], tmp[12], tmp[13], tmp[14], tmp[15],
        //     tmp[16], tmp[17], tmp[18], tmp[19], tmp[20], tmp[21], tmp[22], tmp[23],
        //     tmp[24], tmp[25], tmp[26], tmp[27], tmp[28], tmp[29], tmp[30], tmp[31]
        //     );
        // }

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
        #endif
    }

    #if 1
    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        if (tidx == 0) *skip_softmax_vote = 1;
        Tensor acc_s_ori = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
        clear(acc_s_ori);

        lds_direct_copy<Is_even_K>(gK, sK, 3, params.k_row_stride, params.d);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 0);

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128>(gV, sV, 0, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 1);

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128>(gV, sV, 1, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 2);

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128>(gV, sV, 2, params.v_row_stride, params.d_value);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tGrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_thr_copy_K, 3);
        s_barrier();
        
        // __builtin_amdgcn_sched_barrier(1);

        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        if constexpr (Is_softcap) {
            apply_softcap(acc_s, params.softcap);
        }
        
        { 
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                acc_s, n_block * kBlockN, row_idx_offset_,  (kNWarps << 4)
            );
        }
        
        bool skip = softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2, skip_softmax_vote);
        

        if (skip) {
            s_waitcnt<0>();
            if (n_block > n_block_min) {
                gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
                gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

                lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
                lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
            }
            continue;
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
                if constexpr (kHeadDim==128){
                    dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, m_block * kBlockM , block_col_idx, kNWarps * 16
                    );
                }else{
                    dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                        rP_drop, block_row_idx, block_col_idx, kNWarps * 16
                    );
                }
                Tensor rP_drop_back = make_tensor(rP_drop.data(), convert_layout_acc_back(rP_drop.layout()));
                cute::copy(rP_drop_back, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                if constexpr (kHeadDim==128){
                    dropout.apply_dropout_continuous_opt(rP, m_block * kBlockM , block_col_idx, kNWarps * 16);
                }else{
                    dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps * 16);
                }
            }
        }

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/true, _16x128>(gV, sV, 3, params.v_row_stride, params.d_value);
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

        if (n_block > n_block_min) {
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
        }
        #endif
    }
    #endif

    // Epilogue
    #if 1
    // ★ Attention Sinks: conditional normalize ★
    typename decltype(softmax)::TensorT lse;
    if (params.s_aux_ptr != nullptr) {
        using TensorT = typename decltype(softmax)::TensorT;
        TensorT tSrS_aux;
        // Read from shared memory (already loaded at kernel start) using CuTe Tensor
        Tensor sS_aux = make_tensor(
            make_smem_ptr(smem_s_aux),
            Layout<Shape<Int<64>>>{}
        );
        float s_aux_val = static_cast<float>(sS_aux(bidh));
        #pragma unroll
        for (int mi = 0; mi < size(tSrS_aux); ++mi) {
            tSrS_aux(mi) = s_aux_val;
        }
        lse = softmax.template normalize_softmax_lse_with_sinks<Is_dropout>(
            acc_o, tSrS_aux, params.scale_softmax, params.scale_softmax_log2, params.rp_dropout);
    } else {
        lse = softmax.template normalize_softmax_lse<Is_dropout>(
            acc_o, params.scale_softmax, params.rp_dropout);
    }

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
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

    if (tidx == 0) {
        uint32_t * gSkipInfo_ptr = reinterpret_cast<uint32_t*>(params.skip_blocks_info_ptr) + (bidb * params.h + bidh) * 2;
        // wangaq debug
        // printf("bidx:%d gSkipInfo_ptr:%d %d total_blocks:%d skipped_blocks:%d\n", blockIdx.x,
        //     *gSkipInfo_ptr, *(gSkipInfo_ptr + 1), softmax.total_blocks, softmax.skipped_blocks);
        atomicAdd(gSkipInfo_ptr, softmax.total_blocks);
        atomicAdd(gSkipInfo_ptr + 1, softmax.skipped_blocks);
    }
    
    #endif
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn(const Params &params) {
    const int m_block = Is_causal?gridDim.z - 1 - blockIdx.z:blockIdx.x;
    const int bidb = Is_causal?blockIdx.y:blockIdx.z;
    const int bidh = Is_causal?blockIdx.x:blockIdx.y;
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    using index_t = typename Kernel_traits::index_t;
    index_t binfo_q_offset = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb);
    index_t binfo_k_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb);
    index_t binfo_v_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb);
    index_t binfo_o_offset =  binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);

    flash::compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_16x64(const Params &params) {
    bool need_switch = Is_causal || (params.seqlen_q>8192&&params.h==40);
    const int m_block = need_switch?blockIdx.z:blockIdx.x;
    const int bidb = need_switch?blockIdx.y:blockIdx.z;
    const int bidh = need_switch?blockIdx.x:blockIdx.y;
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    using index_t = typename Kernel_traits::index_t;
    index_t binfo_q_offset = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb);
    index_t binfo_k_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb);
    index_t binfo_v_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb);
    index_t binfo_o_offset =  binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);

    if constexpr (Kernel_traits::kHeadDim == 192 && Kernel_traits::kHeadDimV == 128) { 
        flash::compute_attn_1rowblock_16x64_mla<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
    }
    else if constexpr (Kernel_traits::kHeadDim == 128) {
        flash::compute_attn_1rowblock_16x64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
    }
    else if constexpr (Kernel_traits::kHeadDim == 64) {
        flash::compute_attn_1rowblock_16x64_dim64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
    }

}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_16x64_prefetch(const Params &params) {
    const int m_block = Is_causal?gridDim.z - 1 - blockIdx.z:blockIdx.x;
    const int bidb = Is_causal?blockIdx.y:blockIdx.z;
    const int bidh = Is_causal?blockIdx.x:blockIdx.y;
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    using index_t = typename Kernel_traits::index_t;
    index_t binfo_q_offset = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb);
    index_t binfo_k_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb);
    index_t binfo_v_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb);
    index_t binfo_o_offset =  binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);
    __syncthreads();
    if constexpr (Kernel_traits::kHeadDim == 192 && Kernel_traits::kHeadDimV == 128){
        flash::compute_attn_1rowblock_16x64_mla_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
    } else if constexpr (Kernel_traits::kHeadDim == 128){
        if constexpr(Kernel_traits::ENABLE_SKIP_SOFTMAX) {
            flash::compute_attn_1rowblock_16x64_prefetch_skip_softmax<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
        } else {
            flash::compute_attn_1rowblock_16x64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
        }
    } else if constexpr (Kernel_traits::kHeadDim == 96){
        flash::compute_attn_1rowblock_16x64_dim96_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
    } else if constexpr(Kernel_traits::kHeadDim == 64) {
        flash::compute_attn_1rowblock_16x64_dim64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
    } else if  constexpr(Kernel_traits::kHeadDim == 32) { 
        flash::compute_attn_1rowblock_16x64_dim32_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
    } else if constexpr(Kernel_traits::kHeadDim == 256) {
        flash::compute_attn_1rowblock_16x64_dim256_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
        __syncthreads();
    }  else if constexpr(Kernel_traits::kHeadDim == 512) {
        flash::compute_attn_1rowblock_16x64_dim512_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
        __syncthreads();
    }

}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_16x64_prefetch_fp8(const Params &params) {
    const int m_block = Is_causal?gridDim.z - 1 - blockIdx.z:blockIdx.x;
    const int bidb = Is_causal?blockIdx.y:blockIdx.z;
    const int bidh = Is_causal?blockIdx.x:blockIdx.y;
    
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    using index_t = typename Kernel_traits::index_t;
    index_t binfo_q_offset = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb);
    index_t binfo_k_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb);
    index_t binfo_v_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb);
    index_t binfo_o_offset =  binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);
    __syncthreads();
    #if (defined(__gfx938__)||defined(__gfx92a__))
        if constexpr (Kernel_traits::kHeadDim == 192 && Kernel_traits::kHeadDimV == 128){
            flash::compute_attn_1rowblock_16x64_mla_prefetch_fp8<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
        } else if constexpr (Kernel_traits::kHeadDim == 128){
            flash::compute_attn_1rowblock_16x64_prefetch_fp8<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
        } 
    #endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_16x64_prefetch_padding_mask(const Params &params) {
    const int m_block = Is_causal?gridDim.z - 1 - blockIdx.z:blockIdx.x;
    const int bidb = Is_causal?blockIdx.y:blockIdx.z;
    const int bidh = Is_causal?blockIdx.x:blockIdx.y;
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb, true);
    using index_t = typename Kernel_traits::index_t;
    index_t binfo_q_offset = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb);
    index_t binfo_k_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb);
    index_t binfo_v_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb);
    index_t binfo_o_offset =  binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);
    flash::compute_attn_1rowblock_16x64_dim64_prefetch_padding_mask<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block, binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_splitkv(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
    // The block index for the head.
    const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
    const int n_split_idx = Split ? blockIdx.y : 0;
    const int num_n_splits = Split ? gridDim.y : 1;
    flash::compute_attn_1rowblock_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_splitkv_16x64_vllm_kvcache_prefetch(const Params &params) {
    const int m_block = Is_causal?gridDim.z - 1 - blockIdx.z:blockIdx.x;
    const int bidb = Is_causal?blockIdx.y:blockIdx.z;
    const int bidh = Is_causal?blockIdx.x:blockIdx.y;
    #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
    if constexpr (Kernel_traits::kHeadDim == 64){
        flash::compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_dim64<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params, bidb, bidh, m_block);
    }else if constexpr (Kernel_traits::kHeadDim == 128) {
        flash::compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params, bidb, bidh, m_block);
    }else if constexpr (Kernel_traits::kHeadDim == 192) {
        flash::compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_dim192<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params, bidb, bidh, m_block);
    }else if constexpr (Kernel_traits::kHeadDim == 256) {
        flash::compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_dim256<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params, bidb, bidh, m_block);
    }
    #endif

}
template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_splitkv_16x64_vllm_kvcache_prefetch_kv_fp8(const Params &params) {
    const int m_block = Is_causal?gridDim.z - 1 - blockIdx.z:blockIdx.x;
    const int bidb = Is_causal?blockIdx.y:blockIdx.z;
    const int bidh = Is_causal?blockIdx.x:blockIdx.y;

    #if (defined(__gfx936__))
    if constexpr (Kernel_traits::kHeadDim == 64){
        flash::compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_kv_fp8_dim64<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params, bidb, bidh, m_block);
    }else if constexpr (Kernel_traits::kHeadDim == 128) {
        flash::compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_kv_fp8<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params, bidb, bidh, m_block);
    }else if constexpr (Kernel_traits::kHeadDim == 256) {
        // printf("进到kernel里 dim128");
        flash::compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_kv_fp8_dim256<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params, bidb, bidh, m_block);
    }
    #endif

}
template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_splitkv_16x64_vllm_kvcache_prefetch_fp8(const Params &params) {
    const int m_block = Is_causal?gridDim.z - 1 - blockIdx.z:blockIdx.x;
    const int bidb = Is_causal?blockIdx.y:blockIdx.z;
    const int bidh = Is_causal?blockIdx.x:blockIdx.y;
    #if (defined(__gfx938__)||defined(__gfx92a__))
    if constexpr (Kernel_traits::kHeadDim == 64){
        flash::compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_fp8_dim64<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params, bidb, bidh, m_block);
    }else if constexpr (Kernel_traits::kHeadDim == 128) {
        flash::compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_fp8<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params, bidb, bidh, m_block);
    }else if constexpr (Kernel_traits::kHeadDim == 192) {
        flash::compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_fp8_dim192<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params, bidb, bidh, m_block);
    }else if constexpr (Kernel_traits::kHeadDim == 256) {
        flash::compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_prefetch_fp8_dim256<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params, bidb, bidh, m_block);
    }
    #endif
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, bool Is_need_balance, bool Use_alibi_sqrt, bool Use_qq_bias, bool Use_mm_prefix, typename Params>
inline __device__ void compute_attn_unified_16x64_prefetch(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
    // The block index for the head.
    const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
    const int n_split_idx = Split ? blockIdx.y : 0;
    const int num_n_splits = Split ? gridDim.y : 1;


    #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
    if constexpr (Kernel_traits::kHeadDim == 256) {
        flash::compute_attn_1rowblock_unified_16x64_prefetch_dim256<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table,  Use_alibi_sqrt, Use_qq_bias, Use_mm_prefix>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
        if constexpr(Is_causal)
        {
            const int num_blocks = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
            #ifndef NO_CAUSAL_OPT
            // 处理奇数个block的情况
            if (num_blocks - m_block - 1 != m_block)
            {
                flash::compute_attn_1rowblock_unified_16x64_prefetch_dim256<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table,  Use_alibi_sqrt, Use_qq_bias, Use_mm_prefix>(params, bidb, bidh, num_blocks - m_block - 1, n_split_idx, num_n_splits);
            }
            #endif
        }
        
        return;
    } else if (Kernel_traits::kHeadDim == 128) {
        flash::compute_attn_1rowblock_unified_16x64_prefetch<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table,  Use_alibi_sqrt, Use_qq_bias, Use_mm_prefix>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
        if constexpr (Is_causal)
        {
            const int num_blocks = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
            #ifndef NO_CAUSAL_OPT
            // 处理奇数个block的情况
            if (num_blocks - m_block - 1 != m_block)
            {
                flash::compute_attn_1rowblock_unified_16x64_prefetch<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table,  Use_alibi_sqrt, Use_qq_bias, Use_mm_prefix>(params, bidb, bidh, num_blocks - m_block - 1, n_split_idx, num_n_splits);
            }
            #endif
        }

        return;
    }
    #endif

}
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, typename Params>
inline __device__ void compute_attn_splitkv_16x64_vllm_kvcache_gfx928(const Params &params) {
    const int m_block = Is_causal?gridDim.z - 1 - blockIdx.z:blockIdx.x;
    const int bidb = Is_causal?blockIdx.y:blockIdx.z;
    const int bidh = Is_causal?blockIdx.x:blockIdx.y;
    #if (defined(__gfx928__))
        flash::compute_attn_1rowblock_splitkv_16x64_vllm_kvcache_gfx928<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params, bidb, bidh, m_block);
    #endif

}
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K, typename Params>
inline __device__ void combine_attn_seqk_parallel(const Params &params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kMaxSplits = 1 << Log_max_splits;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNThreads = Kernel_traits::kNThreads;

    static_assert(kMaxSplits <= 128, "kMaxSplits must be <= 128");
    static_assert(kBlockM == 4 || kBlockM == 8 || kBlockM == 16 || kBlockM == 32, "kBlockM must be 4, 8, 16 or 32");
    static_assert(kNThreads == 256, "We assume that each block has 128 threads");

    // Shared memory.
    // kBlockM + 1 instead of kBlockM to reduce bank conflicts.
    __shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];

    // The thread and block index.
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    const index_t lse_size = params.b * params.h * params.seqlen_q;

    const index_t row_offset_lse = bidx * kBlockM;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lse),
                                   Shape<Int<kMaxSplits>, Int<kBlockM>>{},
                                   make_stride(lse_size, _1{}));
    // LSE format is different depending on params.unpadded_lse and params.seqlenq_ngroups_swapped, see comment in get_lse_tile.
    // This tensor's layout maps row_offset_lse to {bidb, bidh, q_offset}.
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    // This layout maps row_offset_lse to {bidh, q_offset, bidb} or {bidh, bidb, q_offset}.
    Layout flat_layout = make_layout(lse_size);
    Layout orig_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b));
    auto transposed_stride = params.seqlenq_ngroups_swapped ? make_stride(params.b, params.seqlen_q * params.b, 1) : make_stride(1, params.seqlen_q * params.b, params.seqlen_q);
    Layout remapped_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b), transposed_stride);
    Layout final_layout = cute::composition(remapped_layout, cute::composition(orig_layout, flat_layout));

    Tensor gLSE_unpadded = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr)), final_layout);
    // 1
    constexpr int kNLsePerThread = (kMaxSplits * kBlockM + kNThreads - 1) / kNThreads;

    // if (cute::thread0())
    // {
    //     printf("kNLsePerThread = %d \n", kNLsePerThread);
    // }
    // 8
    // Read the LSE values from gmem and store them in shared memory, then transpose them.
    constexpr int kRowsPerLoadLSE = kNThreads / kBlockM; 
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadLSE + tidx / kBlockM;
        const int col = tidx % kBlockM;
        ElementAccum lse = (row < params.num_splits && col < lse_size - bidx * kBlockM) ? gLSEaccum(row, col) : -INFINITY;
        __syncthreads();
        if (row < kMaxSplits) { sLSE[row][col] = lse; }
        // if (bidx == 0 && tidx < 64) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse); }
    }
    // if (bidx == 1 && tidx < 32) { printf("tidx = %d, row_offset_lse = %d, lse = %f\n", tidx, row_offset_lse, lse_accum(0)); }
    __syncthreads();

    Tensor lse_accum = make_tensor<ElementAccum>(Shape<Int<kNLsePerThread>>{});
    constexpr int kRowsPerLoadTranspose = std::min(kRowsPerLoadLSE, kMaxSplits);
    static_assert(kNLsePerThread * kRowsPerLoadTranspose <= kMaxSplits);
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        lse_accum(l) = (row < kMaxSplits && col < kBlockM) ? sLSE[row][col] : -INFINITY;
        // if (bidx == 0 && tidx < 64) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse_accum(l)); }
    }
    __syncthreads();
    // Compute the logsumexp of the LSE along the split dimension.
    ElementAccum lse_max = lse_accum(0);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_max = max(lse_max, lse_accum(l)); }
    MaxOp<float> max_op;
    lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
    lse_max = lse_max == -INFINITY ? 0.0f : lse_max;  // In case all local LSEs are -inf
    float lse_sum = expf(lse_accum(0) - lse_max);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_sum += expf(lse_accum(l) - lse_max); }
    SumOp<float> sum_op;
    lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);
    // For the case where all local lse == -INFINITY, we want to set lse_logsum to INFINITY. Otherwise
    // lse_logsum is log(0.0) = -INFINITY and we get NaN when we do lse_accum(l) - lse_logsum.
    ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;
    // if (bidx == 0 && tidx < 32) { printf("tidx = %d, lse = %f, lse_max = %f, lse_logsum = %f\n", tidx, lse_accum(0), lse_max, lse_logsum); }
    if (tidx % kRowsPerLoadTranspose == 0 && tidx / kRowsPerLoadTranspose < kBlockM) {
        if (params.unpadded_lse) {
            const index_t lse_offset = row_offset_lse + tidx / kRowsPerLoadTranspose;
            if (lse_offset < lse_size) {
                gLSE_unpadded(lse_offset) = lse_logsum;
            }
        } else {
            gLSE(tidx / kRowsPerLoadTranspose) = lse_logsum;
        }
    }
    // Store the scales exp(lse - lse_logsum) in shared memory.
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        if (row < params.num_splits && col < kBlockM) { sLSE[row][col] = expf(lse_accum(l) - lse_logsum); }
    }
    __syncthreads();

    const index_t row_offset_oaccum = bidx * kBlockM * params.d_rounded;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 Stride<Int<kHeadDim>, _1>{});
    constexpr int kBlockN = kNThreads / kBlockM;
    using GmemLayoutAtomOaccum = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);
    // Predicates
    Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    // Repeat the partitioning with identity layouts
    Tensor tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cOaccum);
    Tensor tOpOaccum = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpOaccum); ++k) { tOpOaccum(k) = get<1>(tOcOaccum(0, 0, k)) < params.d; }
    }
    __syncthreads();
    // Load Oaccum in then scale and accumulate to O
    for (int split = 0; split < params.num_splits; ++split) {
        flash::copy</*Is_even_MN=*/false, Is_even_K>(
            gmem_tiled_copy_Oaccum, tOgOaccum, tOrOaccum, tOcOaccum, tOpOaccum, params.b * params.h * params.seqlen_q - bidx * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOrOaccum); ++m) {
            int row = get<0>(tOcOaccum(0, m, 0));
            ElementAccum lse_scale = sLSE[split][row];
            
            #pragma unroll
            for (int k = 0; k < size<2>(tOrOaccum); ++k) {
                #pragma unroll
                for (int i = 0; i < size<0>(tOrOaccum); ++i) {
                    // auto temp = tOrO(i, m, k);
                    tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
                }

            }
            // if (cute::thread0()) { printf("lse_scale = %f, %f\n", sLSE[split][0], sLSE[split][1]);}
        }
        tOgOaccum.data() = tOgOaccum.data() + params.b * params.h * params.seqlen_q * params.d_rounded;
    }
    
    Tensor rO = flash::convert_type<Element>(tOrO);
   
    #pragma unroll
    for (int m = 0; m < size<1>(rO); ++m) {
        const int idx = bidx * kBlockM + get<0>(tOcOaccum(0, m, 0));
        if (idx < params.b * params.h * params.seqlen_q) {
            const int batch_idx = idx / (params.h * params.seqlen_q);
            const int head_idx = (idx - batch_idx * (params.h * params.seqlen_q)) / params.seqlen_q;
            // The index to the rows of Q
            const int row = idx - batch_idx * (params.h * params.seqlen_q) - head_idx * params.seqlen_q;
            auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride
                + head_idx * params.o_head_stride + row * params.o_row_stride;
            #pragma unroll
            for (int k = 0; k < size<2>(rO); ++k) {
                if (Is_even_K || tOpOaccum(k)) {
                    const int col = get<1>(tOcOaccum(0, m, k));
                    Tensor gO = make_tensor(make_gmem_ptr(o_ptr + col),
                                            Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
                    // TODO: Should check if this is using vectorized store, but it seems pretty fast
                    copy(rO(_, m, k), gO);
                }
            }
        }
    }
}

} // namespace flash
