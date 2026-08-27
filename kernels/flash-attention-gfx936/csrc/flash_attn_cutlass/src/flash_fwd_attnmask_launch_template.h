/******************************************************************************
 * Copyright (c) 2026, Attnmask extension.
 ******************************************************************************/

#pragma once

#include <ATen/cuda/CUDAContext.h>

#include "flash_fwd_launch_template.h"
#include "flash_fwd_attnmask_kernel.h"
#include "flash_attnmask.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// Attnmask kernel entry point (uses 16x64x32 MMA for Q*K, 16x16x16 for P*V).
////////////////////////////////////////////////////////////////////////////////////////////////////

#define DEFINE_FLASH_FORWARD_ATTNMASK_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__launch_bounds__(Kernel_traits::kNThreads) \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params_attnmask params)

DEFINE_FLASH_FORWARD_ATTNMASK_KERNEL(flash_fwd_attnmask_kernel, bool Is_dropout, bool Is_causal, bool Is_local,
                                     bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap,
                                     bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));
        flash::compute_attn_attnmask<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi,
                                     Is_even_MN, Is_even_K, Is_softcap, Return_softmax,
                                     /*Use_mask=*/true>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Attnmask launch function.
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_attnmask(Flash_fwd_params_attnmask &params, cudaStream_t stream) {
    static_assert(Kernel_traits::Share_Q_K_smem, "Share_Q_K_smem must be true");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    #ifdef NO_CAUSAL_OPT
        const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    #else
        const int non_causal_num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
        const int num_m_block = Is_causal ? (non_causal_num_m_block + 1) >> 1 :
            non_causal_num_m_block;
    #endif

    dim3 grid(num_m_block, params.h, params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr
        && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            auto kernel = &flash_fwd_attnmask_kernel<Kernel_traits,
                                Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi,
                                IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128,
                                IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Default attnmask dispatch: use a conservative kernel configuration to keep the
// implementation compact. Specialize per Headdim if needed for performance tuning.
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_attnmask_(Flash_fwd_params_attnmask &params, cudaStream_t stream) {
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd_attnmask<Flash_fwd_kernel_traits_attnmask<Headdim, 64, 64, 4, false, true, T>,
                               Is_dropout, Is_causal>(params, stream);
    });
}
