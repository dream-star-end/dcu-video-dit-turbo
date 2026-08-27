/******************************************************************************
 * Copyright (c) 2026, Attnmask extension.
 * Launch template for backward pass with explicit mask support.
 ******************************************************************************/

#pragma once

#include <ATen/cuda/CUDAContext.h>

#include "flash_bwd_launch_template.h"
#include "flash_bwd_attnmask_kernel.h"
#include "flash_attnmask.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// Backward pass with explicit mask support.
//
// The mask is applied when recomputing S = QK^T in the backward pass, 
// before the softmax (scale_apply_exp2), to ensure P = 0 at masked positions.
////////////////////////////////////////////////////////////////////////////////////////////////////

// Define kernel wrapper macros for attnmask backward
#define DEFINE_FLASH_BACKWARD_ATTNMASK_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_bwd_params_attnmask params)

// dK/dV kernel with mask support
DEFINE_FLASH_BACKWARD_ATTNMASK_KERNEL(flash_bwd_dk_dv_attnmask_kernel, 
    bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));
        flash::compute_dk_dv_trans_16x64_prefetch_attnmask<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, 
            Is_even_MN, Is_even_K, Is_softcap, /*Use_mask=*/true>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

// dQ kernel with mask support
DEFINE_FLASH_BACKWARD_ATTNMASK_KERNEL(flash_bwd_dq_attnmask_kernel, 
    bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));
        flash::compute_dq_seqq_parallel_16x64_prefetch_attnmask<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, 
            Is_even_MN, Is_even_K, Is_softcap, /*Use_mask=*/true>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Launch function for backward pass with mask support (prefetch version for gfx936/gfx938)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename Kernel_trans_traits, bool Is_dropout, bool Is_causal>
void run_flash_bwd_attnmask_prefetch(Flash_bwd_params_attnmask &params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD
#ifdef NO_CAUSAL_OPT
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
#else
    const int non_causal_num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    const int num_n_block = (Is_causal && Kernel_trans_traits::kHeadDim != 64) ? (non_causal_num_n_block + 1) >> 1 : non_causal_num_n_block;
    const int non_causal_num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    const int num_m_block = Is_causal ? (non_causal_num_m_block + 1) >> 1 : non_causal_num_m_block;
#endif
    dim3 grid_m(num_m_block, params.h, params.b);
    dim3 grid_m_do((params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM, params.b, params.h);
    dim3 grid_n(num_n_block, params.h, params.b);

    // Preprocess: compute dO * O sum
    flash_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m_do, Kernel_traits::kNThreads, 0, stream>>>(
        static_cast<Flash_bwd_params&>(params));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr 
        && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_MN_trans = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr 
        && params.seqlen_q % Kernel_trans_traits::kBlockM == 0 && params.seqlen_k % Kernel_trans_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;

    constexpr int smem_size_dropout = Kernel_trans_traits::kBlockM * Kernel_trans_traits::kBlockN;
    constexpr int smem_size_dk_dv = Kernel_trans_traits::kSmemPrefetchSize;
    constexpr int smem_size_dk_dv_total = (Kernel_trans_traits::kHeadDim == 128) ? (smem_size_dk_dv + smem_size_dropout) : smem_size_dk_dv;
    constexpr int smem_size_dq = Kernel_traits::kSmemPrefetchSize;

    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal, Is_local, [&] {
                ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                    SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                        BOOL_SWITCH(is_even_MN_trans, IsEvenMNTransConst, [&] {
                            // Launch dK/dV kernel
                            auto kernel_dkdv = &flash_bwd_dk_dv_attnmask_kernel<
                                Kernel_trans_traits, 
                                Is_dropout && !Is_softcap, Is_causal, 
                                Is_local && !Is_causal, 
                                Has_alibi, 
                                IsEvenMNTransConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, 
                                IsEvenKConst, 
                                Is_softcap>;
                            kernel_dkdv<<<grid_n, Kernel_traits::kNThreads, smem_size_dk_dv_total, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                        
                        // Launch dQ kernel
                        auto kernel_dq = &flash_bwd_dq_attnmask_kernel<
                            Kernel_traits, 
                            Is_dropout && !Is_softcap, Is_causal, 
                            Is_local && !Is_causal, 
                            Has_alibi, 
                            IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, 
                            IsEvenKConst, 
                            Is_softcap>;
                        kernel_dq<<<grid_m, Kernel_traits::kNThreads, smem_size_dq, stream>>>(params);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Launch function for backward pass with mask support (seqk parallel version)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename Kernel_trans_traits, bool Is_dropout, bool Is_causal>
void run_flash_bwd_attnmask_seqk_parallel_trans(Flash_bwd_params_attnmask &params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.h, params.b);
    dim3 grid_m_do(num_m_block, params.b, params.h);
    const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    dim3 grid_n(num_n_block, params.h, params.b);

    // Preprocess: compute dO * O sum
    flash_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m_do, Kernel_traits::kNThreads, 0, stream>>>(
        static_cast<Flash_bwd_params&>(params));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr 
        && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_MN_trans = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr 
        && params.seqlen_q % Kernel_trans_traits::kBlockM == 0 && params.seqlen_k % Kernel_trans_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;

    constexpr int smem_size_dq_dk_dv = Kernel_trans_traits::kSmemSizeTrans1colblock;
    constexpr int smem_size_dq = Kernel_traits::kSmemSize1rowblock;

    BOOL_SWITCH(is_even_MN_trans, IsEvenMNTransConst, [&] {
        BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
            EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
                LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !params.is_causal, Is_local, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            // Launch dK/dV kernel
                            auto kernel_dkdv = &flash_bwd_dk_dv_attnmask_kernel<
                                Kernel_trans_traits, 
                                Is_dropout && !Is_softcap, Is_causal, 
                                Is_local && !Is_causal, 
                                Has_alibi, 
                                IsEvenMNTransConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, 
                                IsEvenKConst, 
                                Is_softcap>;
                            kernel_dkdv<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                            
                            // Launch dQ kernel
                            auto kernel_dq = &flash_bwd_dq_attnmask_kernel<
                                Kernel_traits, 
                                Is_dropout && !Is_softcap, Is_causal, 
                                Is_local && !Is_causal, 
                                Has_alibi, 
                                IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, 
                                IsEvenKConst, 
                                Is_softcap>;
                            kernel_dq<<<grid_m, Kernel_traits::kNThreads, smem_size_dq, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Backward dispatch for attnmask - uses the new kernels with mask support
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, bool Is_causal>
void run_mha_bwd_attnmask_hdim128(Flash_bwd_params_attnmask &params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD
    constexpr static int Headdim = 128;
    
    // Attnmask backward kernel requires prefetch traits (uses kStages, SmemLayoutKGemm0, etc.)
    // Unified to use prefetch traits for all devices
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        using kernel_trans_traits = Flash_bwd_kernel_trans_16x64_prefetch_traits<Headdim, /*kBlockM_*/64, /*kBlockN_*/64, /*kNWarps_*/4, T, 3>;
        using kernel_traits = Flash_bwd_kernel_dq_16x64_prefetch_traits<Headdim, /*kBlockM_*/Is_dropout ? (Is_causal ? 64 : 128) : 128, /*kBlockN_*/64, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/1, /*AtomLayoutMdQ*/4, /*Is_V_in_regs_*/false, 
            /*No_double_buffer_*/true, /*Is_Q_in_regs_*/false, /*Share_Q_K_smem_*/true, T, 3>;
        run_flash_bwd_attnmask_prefetch<kernel_traits, kernel_trans_traits, Is_dropout, Is_causal>(params, stream);
    });
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Entry point for backward pass with mask, dispatched by element type and causal flag.
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// Backward dispatch for attnmask - hdim64
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, bool Is_causal>
void run_mha_bwd_attnmask_hdim64(Flash_bwd_params_attnmask &params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD
    constexpr static int Headdim = 64;

    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        using kernel_trans_traits = Flash_bwd_kernel_trans_16x64_prefetch_traits<Headdim, /*kBlockM_*/64, /*kBlockN_*/128, /*kNWarps_*/4, T, 3>;
        using kernel_traits = Flash_bwd_kernel_dq_16x64_prefetch_traits<Headdim, /*kBlockM_*/128, /*kBlockN_*/64, /*kNWarps_*/4,
            /*AtomLayoutMSdP_*/4, /*AtomLayoutNdKV*/1, /*AtomLayoutMdQ*/4, /*Is_V_in_regs_*/false,
            /*No_double_buffer_*/true, /*Is_Q_in_regs_*/false, /*Share_Q_K_smem_*/true, T, 3>;
        run_flash_bwd_attnmask_prefetch<kernel_traits, kernel_trans_traits, Is_dropout, Is_causal>(params, stream);
    });
#endif
}

template<typename T, int Headdim, bool Is_causal>
void run_mha_bwd_attnmask_(Flash_bwd_params_attnmask &params, cudaStream_t stream) {
    static_assert(Headdim == 64 || Headdim == 128);
    if constexpr (Headdim == 128) {
        run_mha_bwd_attnmask_hdim128<T, Is_causal>(params, stream);
    } else if constexpr (Headdim == 64) {
        run_mha_bwd_attnmask_hdim64<T, Is_causal>(params, stream);
    }
}
