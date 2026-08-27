/******************************************************************************
 * Copyright (c) 2024, PAI, Alibaba Cloud.
 ******************************************************************************/

#pragma once

#include "flash_fwd_kernel.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Attnmask kernel using 16x64x32 MMA for Q*K and 16x16x16 for P*V.
/// The 16x64x32 MMA produces accumulator layout ((4,4), MMA_M, MMA_N) which is incompatible
/// with the standard kernel's assumption of (4, MMA_M, MMA_N) from 16x16x16 MMA.
/// Based on predmaskbeta's compute_attn_1rowblock_bias.
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi,
         bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, bool Use_mask,
         typename Params, typename index_t, typename Binfo>
inline __device__ void compute_attn_1rowblock_attnmask(const Params &params, const int bidb, const int bidh,
                                                              const int m_block, index_t binfo_q_offset,
                                                              index_t binfo_k_offset, index_t binfo_v_offset,
                                                              index_t binfo_o_offset, const Binfo& binfo) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;

    extern __shared__ char smem_[];

    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    // Attention Mask pointer initialization
    bool* mask_ptr = Use_mask ? reinterpret_cast<bool*>(params.mask_ptr) + bidb * params.mask_batch_stride + bidh * params.mask_head_stride : nullptr;

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
    }
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo_o_offset),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
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

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo_q_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + binfo_k_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + binfo_v_offset),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    // matrix mask
    Tensor mM = make_tensor(mask_ptr,
                            make_shape(binfo.actual_seqlen_q, binfo.actual_seqlen_k),
                            make_stride(params.mask_seq_q_stride, _1{}));
    Tensor gM = local_tile(mM, Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block, _));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    auto tKrK = make_fragment_like(tKsK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // 16x64x32 MMA for Q*K
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    // 16x16x16 MMA with ValLayout<1,2,1> for P*V
    typename Kernel_traits::TiledMma_FOR_GEMM1 tiled_mma_for_gemm1;
    auto thr_mma_for_gemm1 = tiled_mma_for_gemm1.get_thread_slice(tidx);

    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma_for_gemm1.partition_fragment_B(sVtNoSwizzle);

    // Mask tensors partitioned with tiled_mma (16x64x32) - compatible since gM is (kBlockM, kBlockN) = (64, 64) and MMA_N=64
    Tensor tSgM = thr_mma.partition_C(gM(_, _, 0));
    Tensor tSrM = make_fragment_like<uint8_t>(tSgM);
    clear(tSrM);

    Tensor tSgS  = thr_mma.partition_C(gP);

    // acc_o uses tiled_mma_for_gemm1 since kHeadDim may be < MMA_N=64 of 16x64x32; 16x16x16 handles all supported kHeadDim values
    Tensor acc_o = partition_fragment_C(tiled_mma_for_gemm1, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    //
    // Copy Atom retiling
    //
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    // V copy uses SmemCopyAtomV (GFX928_DS_READ_DS_M32x16_B16_RAW) with tiled_mma_for_gemm1
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomV{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));

    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Mask identity tensor for predicate checks
    Tensor cM = make_identity_tensor(make_shape(size<0>(gM(_, _, 0)), size<1>(gM(_, _, 0))));
    Tensor tScM = thr_mma.partition_C(cM);

    // m direction predicate - each thread is responsible for one row
    bool pm_m = false;

    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // m direction predicate check for mask (using make_coord for ((4,4),...) layout)
    if constexpr (!Is_even_MN) {
        pm_m = get<0>(tScM(make_coord(0, 0), 0, 0)) < (binfo.actual_seqlen_q - m_block * kBlockM);
    }

    // Prologue: load Q
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);
    if constexpr (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    if constexpr (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // Load first K block
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKrK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);

    if constexpr (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    // Mask pre-read before main loop
    if constexpr (Use_mask) {
        tSgM = thr_mma.partition_C(gM(_, _, n_block));
        #pragma unroll
        for (int i = 0; i < size<0, 1>(tSgM); ++i) {
            if constexpr (Is_even_MN) {
                cute::copy(tSgM, tSrM);
            } else {
                if ((get<1>(tScM(make_coord(0, i), 0, 0)) < (binfo.actual_seqlen_k - n_block * kBlockN)) && pm_m) {
                    cute::copy(tSgM(make_coord(_, i), 0, 0),
                            tSrM(make_coord(_, i), 0, 0));
                }
            }
        }
    }

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
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        cute::copy(tKrK, tKsK);
        __syncthreads();
        auto tVrV = make_fragment_like(tVsV);

        if (masking_step > 0) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV);
        } else {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }

        // Q*K GEMM using 16x64x32 MMA (A in regs from prologue)
        flash::gemm</* A_in_regs */ true>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
                              smem_tiled_copy_Q, smem_tiled_copy_K,
                              smem_thr_copy_Q, smem_thr_copy_K);

        // Reshape acc_s from ((4,4), MMA_M, MMA_N) to (4, MMA_M, 4*MMA_N) for mask/softmax compatibility
        auto naive_acc_s = make_tensor(acc_s.data(),
            make_shape(shape<0, 0>(acc_s), shape<1>(acc_s), shape<0, 1>(acc_s) * shape<2>(acc_s)),
            make_stride(_1{}, _0{}, _4{}));

        if constexpr (Is_softcap) {
            apply_softcap(naive_acc_s, params.softcap);
        }

        // Apply attention mask on native acc_s layout (element-wise, works with any shape)
        flash::apply_atten_mask<Use_mask>(tSrM, acc_s, params.masked_value);

        // Apply causal/local mask using naive_acc_s (needs (4, MMA_M, MMA_N) layout)
        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<Is_causal, Is_even_MN>(
                naive_acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }

        // Use_mask requires Check_inf=true because explicit mask can create all-masked rows
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || Use_mask>(naive_acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || Use_mask>(naive_acc_s, acc_o, params.scale_softmax_log2);

        // Read mask for next iteration
        if (n_block > n_block_min) {
            if constexpr (Use_mask) {
                tSgM = thr_mma.partition_C(gM(_, _, n_block - 1));
                if constexpr (Is_even_MN) {
                    cute::copy(tSgM, tSrM);
                } else {
                    if (pm_m) {
                        cute::copy(tSgM, tSrM);
                    }
                }
            }
        }

        // Write V to smem and sync
        cute::copy(tVrV, tVsV);

        // Convert naive_acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(naive_acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int warp_row_stride = 16;
            const int block_row_idx = m_block * (kBlockM >> 4) + wave_id;
            const int block_col_idx = n_block * (kBlockN >> 4);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps
                );
                cute::copy(rP_drop, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps);
            }
        }

        // Wait for V in smem before P*V GEMM
        asm volatile("s_waitcnt lgkmcnt(0)\n\t"
                      "s_barrier \n\t"
                      : : : "memory");
        // P*V GEMM using tiled_mma_for_gemm1 (16x16x16)
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);

        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // Non-masking loop
    #pragma unroll
    for (; n_block >= n_block_min; n_block--) {
        __syncthreads();
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        cute::copy(tKrK, tKsK);
        __syncthreads();

        auto tVrV = make_fragment_like(tVsV);
        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVrV, tKVcKV, tKVpKV);

        flash::gemm</* A_in_regs */ true>(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
                              smem_tiled_copy_Q, smem_tiled_copy_K,
                              smem_thr_copy_Q, smem_thr_copy_K);

        auto naive_acc_s = make_tensor(acc_s.data(),
            make_shape(shape<0, 0>(acc_s), shape<1>(acc_s), shape<0, 1>(acc_s) * shape<2>(acc_s)),
            make_stride(_1{}, _0{}, _4{}));

        if constexpr (Is_softcap) {
            apply_softcap(naive_acc_s, params.softcap);
        }

        flash::apply_atten_mask<Use_mask>(tSrM, acc_s, params.masked_value);

        {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;
            mask.template apply_mask_continuous<false>(
                naive_acc_s, n_block * kBlockN, row_idx_offset_, (kNWarps << 4)
            );
        }

        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKrK, tKVcKV, tKVpKV);
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/ Is_local || Use_mask>(naive_acc_s, acc_o, params.scale_softmax_log2);

        if (n_block > n_block_min) {
            if constexpr (Use_mask) {
                tSgM = thr_mma.partition_C(gM(_, _, n_block - 1));
                if constexpr (Is_even_MN) {
                    cute::copy(tSgM, tSrM);
                } else {
                    if (pm_m) {
                        cute::copy(tSgM, tSrM);
                    }
                }
            }
        }

        cute::copy(tVrV, tVsV);

        Tensor rP = flash::convert_type<Element>(naive_acc_s);

        {   // dropout
            const int wave_id = (tidx >> 6);
            const int warp_row_stride = 16;
            const int block_row_idx = m_block * (kBlockM >> 4) + wave_id;
            const int block_col_idx = n_block * (kBlockN >> 4);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps
                );
                cute::copy(rP_drop, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout_continuous(rP, block_row_idx, block_col_idx, kNWarps);
            }
        }

        asm volatile("s_waitcnt lgkmcnt(0)\n\t"
                      "s_barrier \n\t"
                      : : : "memory");
        flash::gemm_rs(acc_o, rP, tOrVt, tOsVt, tiled_mma_for_gemm1, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});

    // Use tiled_mma_for_gemm1 for output smem copy (output shape (kBlockM, kHeadDim))
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_for_gemm1);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo_o_offset),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO2 = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO2);

    // Use thr_mma_for_gemm1 for LSE write (kHeadDim may be < MMA_N=64 of 16x64x32)
    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma_for_gemm1.partition_C(caccO);
    if (get<1>(taccOcO(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO(0, mi, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO2, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Wrapper for attnmask kernel dispatch.
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local,
         bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap,
         bool Return_softmax, bool Use_mask, typename Params>
inline __device__ void compute_attn_attnmask(const Params &params) {
    const int m_block = blockIdx.x;
    const int bidb = blockIdx.z;
    const int bidh = blockIdx.y;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    using index_t = typename Kernel_traits::index_t;
    index_t binfo_q_offset = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb);
    index_t binfo_k_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb);
    index_t binfo_v_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb);
    index_t binfo_o_offset = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);

    compute_attn_1rowblock_attnmask<Kernel_traits, Is_dropout, Is_causal, Is_local,
        Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Use_mask>(
        params, bidb, bidh, m_block,
        binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);

#ifndef NO_CAUSAL_OPT
    if constexpr (Is_causal)
    {
        const int num_blocks = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
        if (num_blocks - m_block - 1 != m_block)
        {
            compute_attn_1rowblock_attnmask<Kernel_traits, Is_dropout, Is_causal, Is_local,
                Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax, Use_mask>(
                params, bidb, bidh, num_blocks - m_block - 1,
                binfo_q_offset, binfo_k_offset, binfo_v_offset, binfo_o_offset, binfo);
        }
    }
#endif
}

} // namespace flash
