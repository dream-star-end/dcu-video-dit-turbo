/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <ATen/cuda/CUDAContext.h>

#include "static_switch.h"
#include "flash.h"
#include "flash_fwd_kernel.h"

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

#if defined(DCU_ASM)
    #define ARCH_SUPPORTS_FLASH
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

static const bool prefetch = get_env_("FLASH_ATTENTION_FWD_PREFETCH");
static const bool force_blockm128 = get_env_("FLASH_ATTENTION_force_blockm128");
static const bool force_blockm64 = get_env_("FLASH_ATTENTION_force_blockm64");
static const bool prefix_cache_force_use_128= get_env_("FLASH_ATTENTION_prefix_cache_force_use_128");
static const bool debug_env = get_env_("FLASH_ATTENTION_debug_env");

static const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__launch_bounds__(Kernel_traits::kNThreads) \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        flash::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel_16x64, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        flash::compute_attn_16x64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel_16x64_prefetch, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        flash::compute_attn_16x64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel_16x64_prefetch_fp8, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        flash::compute_attn_16x64_prefetch_fp8<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel_16x64_prefetch_padding_mask, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        flash::compute_attn_16x64_prefetch_padding_mask<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV) {
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_attn_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}


DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel_16x64_vllm_kvcache_prefetch, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table) {
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_attn_splitkv_16x64_vllm_kvcache_prefetch<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel_16x64_vllm_kvcache_prefetch_fp8, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table) {
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_attn_splitkv_16x64_vllm_kvcache_prefetch_fp8<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel_16x64_vllm_kvcache_prefetch_kv_fp8, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table) {
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_attn_splitkv_16x64_vllm_kvcache_prefetch_kv_fp8<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel_16x64_vllm_kvcache_gfx928, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table) {
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_attn_splitkv_16x64_vllm_kvcache_gfx928<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}
DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_unified_kernel_16x64_prefetch, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Has_block_table, bool Is_need_balance, bool Use_alibi_sqrt, bool Use_qq_bias, bool Use_mm_prefix) {
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_attn_unified_16x64_prefetch<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, Has_block_table, Is_need_balance, Use_alibi_sqrt, Use_qq_bias, Use_mm_prefix>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_combine_kernel, int kBlockM, int Log_max_splits, bool Is_even_K) {
    static_assert(Log_max_splits >= 1);
    flash::combine_attn_seqk_parallel<Kernel_traits, kBlockM, Log_max_splits, Is_even_K>(params);
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(Kernel_traits::Share_Q_K_smem, "Share_Q_K_smem must be true");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;//2
    dim3 grid;
    if (Is_causal||(params.seqlen_q>8192&&params.h==40))grid=dim3(params.h, params.b,num_m_block);
    else grid=dim3(num_m_block,params.h, params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_16x64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;//2
    dim3 grid;
    if (Is_causal||(params.seqlen_q>8192&&params.h==40))grid=dim3(params.h, params.b,num_m_block);
    else grid=dim3(num_m_block,params.h, params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            auto kernel = &flash_fwd_kernel_16x64<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                            // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_16x64_prefetch(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(Kernel_traits::Share_Q_K_smem, "Share_Q_K_smem must be true");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;//2
    dim3 grid;
    if constexpr(Is_causal)grid=dim3(params.h, params.b,num_m_block);
    else grid=dim3(num_m_block,params.h, params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            // constexpr static bool IsEvenKConst = true;
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                // constexpr static bool Is_local = false;
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    // constexpr static bool ReturnSoftmaxConst = true;
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        // constexpr static bool Has_alibi = false;
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            // constexpr static bool Is_softcap = false;
                            auto kernel = &flash_fwd_kernel_16x64_prefetch<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                            // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_16x64_prefetch_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(Kernel_traits::Share_Q_K_smem, "Share_Q_K_smem must be true");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;//2
    dim3 grid;
    if constexpr(Is_causal)grid=dim3(params.h, params.b,num_m_block);
    else grid=dim3(num_m_block,params.h, params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
  
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    // printf("%d,%d\n",Is_causal, is_even_MN);
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
							    auto kernel = &flash_fwd_kernel_16x64_prefetch_fp8<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                                // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                                C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
}





template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd_16x64_prefetch_padding_mask(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(Kernel_traits::Share_Q_K_smem, "Share_Q_K_smem must be true");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;//2
    dim3 grid;
    if constexpr(Is_causal)grid=dim3(params.h, params.b,num_m_block);
    else grid=dim3(num_m_block,params.h, params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    // BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        constexpr static bool IsEvenMNConst = false;
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            auto kernel = &flash_fwd_kernel_16x64_prefetch_padding_mask<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                            // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    // });
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(Kernel_traits::Is_Q_in_regs, "SplitKV implementation must support Is_Q_in_regs");
    static_assert(Kernel_traits::Share_Q_K_smem, "SplitKV implementation must support Share_Q_K_smem");
    // params.num_splits大于1的时候,输出值是float类型，是大于Q的。这里改动的本质原因是q与kv共享lds导致的
    const size_t smem_size = params.num_splits > 1 ? std::max(Kernel_traits::kSmemQSize * 2, Kernel_traits::kSmemSize) : Kernel_traits::kSmemSize;
    // printf("smem_size = %d\n", smem_size);
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    BOOL_SWITCH(params.knew_ptr != nullptr, Append_KV, [&] {
                        ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                            SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                                constexpr static bool is_even_MN=false;
                                constexpr static bool Append_KV=false;
                                auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && !Append_KV && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, Split, Append_KV>;

                                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                                C10_CUDA_KERNEL_LAUNCH_CHECK();
                            });
                        });
                    });
                });
            });
        });
    });
        // printf(" run_flash_splitkv_fwd params.num_splits = %d\n", params.num_splits);
    if (params.num_splits > 1) {
        // We want kBlockM to be as small as possible for more parallelism.
        // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
        // If headdim is divisible by 64, then we set kBlockM = 8, etc.
        constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 32 : (Kernel_traits::kHeadDim % 64 == 0 ? 32 : 32);
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            if (params.num_splits <= 2) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 1, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } 
            else if (params.num_splits <= 4) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 8) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 3, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 16) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 4, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 32) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 5, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 64) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 6, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 128) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 7, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }

}

template<typename Kernel_traits, typename Combine_Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch(Flash_fwd_params &params, cudaStream_t stream) {
    // params.num_splits大于1的时候,输出值是float类型，是大于Q的。这里改动的本质原因是q与kv共享lds导致的
    params.num_splits=1;
    const size_t smem_size = params.num_splits > 1 ? std::max(Kernel_traits::kSmemOSize * 2, Kernel_traits::kSmemSize) : Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;//2
    dim3 grid;
    if constexpr(Is_causal)grid=dim3(params.h, params.b,num_m_block);
    else grid=dim3(num_m_block,params.h, params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = true;
    // const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                                constexpr static bool IsEvenMNConst = false;
                                constexpr static bool IsEvenKConst = true;
                                // constexpr static bool Is_local = false;
                                constexpr static bool Is_softcap = false;
                                constexpr static bool Has_block_table = true;
                                constexpr static bool Append_KV = false;
                                constexpr static bool Split = false;
                                auto kernel = &flash_fwd_splitkv_kernel_16x64_vllm_kvcache_prefetch<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && !Append_KV && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, Split, Append_KV, Has_block_table>;
                                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                        });
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits, typename Combine_Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch_kv_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    // params.num_splits大于1的时候,输出值是float类型，是大于Q的。这里改动的本质原因是q与kv共享lds导致的
    params.num_splits=1;
    const size_t smem_size = params.num_splits > 1 ? std::max(Kernel_traits::kSmemOSize * 2, Kernel_traits::kSmemSize) : Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;//2
    dim3 grid;
    if constexpr(Is_causal)grid=dim3(params.h, params.b,num_m_block);
    else grid=dim3(num_m_block,params.h, params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = true;
    
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            constexpr static bool IsEvenMNConst = false;
                            constexpr static bool IsEvenKConst = true;
                            // constexpr static bool Is_local = false;
                            constexpr static bool Is_softcap = false;
                            constexpr static bool Has_block_table = true;
                            constexpr static bool Append_KV = false;
                            constexpr static bool Split = false;
                            auto kernel = &flash_fwd_splitkv_kernel_16x64_vllm_kvcache_prefetch_kv_fp8<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && !Append_KV && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, Split, Append_KV, Has_block_table>;
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits, typename Combine_Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    // params.num_splits大于1的时候,输出值是float类型，是大于Q的。这里改动的本质原因是q与kv共享lds导致的
    params.num_splits=1;
    const size_t smem_size = params.num_splits > 1 ? std::max(Kernel_traits::kSmemOSize * 2, Kernel_traits::kSmemSize) : Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;//2
    dim3 grid;
    if constexpr(Is_causal)grid=dim3(params.h, params.b,num_m_block);
    else grid=dim3(num_m_block,params.h, params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = true;
    // const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            constexpr static bool  IsEvenMNConst = false;
                            constexpr static bool IsEvenKConst = true;
                            // constexpr static bool Is_local = false;
                            constexpr static bool Is_softcap = false;
                            constexpr static bool Has_block_table = true;
                            constexpr static bool Append_KV = false;
                            constexpr static bool Split = false;
                            // constexpr static bool Has_alibi = false;
                            auto kernel = &flash_fwd_splitkv_kernel_16x64_vllm_kvcache_prefetch_fp8<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && !Append_KV && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, Split, Append_KV, Has_block_table>;
                            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                            C10_CUDA_KERNEL_LAUNCH_CHECK();
                        });
                    });
                });
            });
        });
    });

}

template<typename Kernel_traits, typename Combine_Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd_16x64_unified_prefetch(Flash_fwd_params &params, cudaStream_t stream) {
    // params.num_splits大于1的时候,输出值是float类型，是大于Q的。这里改动的本质原因是q与kv共享lds导致的
    const size_t smem_size = params.num_splits > 1 ? std::max(Kernel_traits::kSmemOSize * 2, Kernel_traits::kSmemSize) : Kernel_traits::kSmemSize;
    // printf("smem_size = %d\n", smem_size);
#ifdef NO_CAUSAL_OPT
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
#else
    const int non_causal_num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    const int grid_y = params.num_splits > 1 ? params.num_splits : params.b;
    const int grid_z = params.num_splits > 1 ? params.b * params.h : params.h;
    const bool need_balance = Is_causal && (non_causal_num_m_block * grid_y * grid_z > 80);
    const int num_m_block = need_balance ? (non_causal_num_m_block + 1 ) >> 1 : 
        non_causal_num_m_block;
#endif

    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    // BOOL_SWITCH(params.knew_ptr != nullptr, Append_KV, [&] {
                        ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                            SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                                    BOOL_SWITCH(need_balance, Is_need_balance, [&] {
                                        BOOL_SWITCH(params.use_alibi_sqrt, Use_alibi_sqrt, [&] {
                                            BOOL_SWITCH(params.qq_bias_ptr != nullptr, Use_qq_bias, [&] {
                                                BOOL_SWITCH(params.mm_prefix_range_ptr != nullptr, Use_mm_prefix, [&] {
                                                constexpr static bool IsEvenMNConst = false;
                                                constexpr static bool IsEvenKConst = true;
                                                constexpr static bool Has_block_table = true;
                                                constexpr static bool Append_KV = false;
                                                auto kernel = &flash_fwd_unified_kernel_16x64_prefetch<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && !Append_KV && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, Split, Append_KV, Has_block_table, Is_need_balance, Use_alibi_sqrt, Use_qq_bias, Use_mm_prefix>;                                    
                                                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                                                C10_CUDA_KERNEL_LAUNCH_CHECK();
                                            });
                                        });
                                    });
                                });
                            });
                        });
                    // });
                });
            });
        });
    });
    // printf(" run_flash_splitkv_fwd params.num_splits = %d\n", params.num_splits);
    if (params.num_splits > 1) {
        // We want kBlockM to be as small as possible for more parallelism.
        // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
        // If headdim is divisible by 64, then we set kBlockM = 8, etc.
        constexpr static int kBlockM = Combine_Kernel_traits::kHeadDim % 128 == 0 ? 32 : (Kernel_traits::kHeadDim % 64 == 0 ? 32 : 32);
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        params.d = params.d_value;
        params.d_rounded = params.d_value_rounded;

        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            if (params.num_splits <= 2) {
                flash_fwd_splitkv_combine_kernel<Combine_Kernel_traits, kBlockM, 1, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } 
            else if (params.num_splits <= 4) {
                flash_fwd_splitkv_combine_kernel<Combine_Kernel_traits, kBlockM, 2, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 8) {
                flash_fwd_splitkv_combine_kernel<Combine_Kernel_traits, kBlockM, 3, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 16) {
                flash_fwd_splitkv_combine_kernel<Combine_Kernel_traits, kBlockM, 4, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 32) {
                flash_fwd_splitkv_combine_kernel<Combine_Kernel_traits, kBlockM, 5, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 64) {
                flash_fwd_splitkv_combine_kernel<Combine_Kernel_traits, kBlockM, 6, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 128) {
                flash_fwd_splitkv_combine_kernel<Combine_Kernel_traits, kBlockM, 7, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
}


template<typename Kernel_traits, typename Combine_Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd_16x64_vllm_kvcache_gfx928(Flash_fwd_params &params, cudaStream_t stream) {
    params.num_splits=1;
    const size_t smem_size = params.num_splits > 1 ? std::max(Kernel_traits::kSmemOSize * 2, Kernel_traits::kSmemSize) : Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;//2
    dim3 grid;
    if constexpr(Is_causal)grid=dim3(params.h, params.b,num_m_block);
    else grid=dim3(num_m_block,params.h, params.b);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = true;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    BOOL_SWITCH(params.knew_ptr != nullptr, Append_KV, [&] {
                        ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                            SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                                BOOL_SWITCH(params.block_table != nullptr, Has_block_table, [&] {
                                        constexpr static bool IsEvenMNConst = false;
                                        constexpr static bool IsEvenKConst = true;
                                        // constexpr static bool Is_local = false;
                                        constexpr static bool Is_softcap = false;
                                        constexpr static bool Has_block_table = true;
                                        constexpr static bool Append_KV = false;
                                        constexpr static bool Split = false;
                                        auto kernel = &flash_fwd_splitkv_kernel_16x64_vllm_kvcache_gfx928<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && !Append_KV && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, Split, Append_KV, Has_block_table>;
                                        kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                                });
                            });
                        });
                    });
                });
            });
        });
    });

}

template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int kBlockM = 64;  // Fixed for all head dimensions
    // TD [2023-08-28]: nvcc segfaults for headdim 96 with block size 64 x 256,
    // and for headdim 192 with block size 64 x 128.
    // Also for headdim 160 with block size 64 x 128 after the rotary addition.
    constexpr static int kBlockN = Headdim <= 128 ? 64 : (Headdim % 64 == 0 ? 32 : 64);
    int mblocks = (params.seqlen_q + 64 - 1) / 64;
    bool is_small = (params.seqlen_q <= 64||params.h * params.b * mblocks< 4*sm_count);
    if (params.is_vllm_kvcache) {
        if constexpr(Headdim == 64) {
            if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
                using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<64, kBlockM, kBlockN, 4, false, false, T, 64>;
                if (is_small) {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_dim64<64, 64, 64, 4, T, 1, 64>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                } else {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_dim64<64, 128, 64, 4, T, 1, 64>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                }
            }
            else {
                using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<64, kBlockM, kBlockN, 4, false, false, T, 64>;
                if (is_small) {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_vllm_kvcache_traits<64, 64, 64, 4, T, 3, 64>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_gfx928<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                } else {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_vllm_kvcache_traits<64, 128, 64, 4, T, 3, 64>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_gfx928<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                }
            }
        }else if constexpr(Headdim == 128) {
            if (get_device_name() == "gfx936"||get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
                assert(params.knew_ptr == nullptr && params.block_table != nullptr);
                using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<128, kBlockM, kBlockN, 4, false, false, T, 128>;
                if (is_small) {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits<128, 64, 64, 4, T, 3, 128>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                } else {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits<128, 128, 64, 4, T, 3, 128>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                } 
            }
            else {
                using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<128, kBlockM, kBlockN, 4, false, false, T, 128>;
                if (is_small) {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_vllm_kvcache_traits<128, 64, 64, 4, T, 3, 128>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_gfx928<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                } else {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_vllm_kvcache_traits<128, 128, 64, 4, T, 3, 128>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_gfx928<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                }
            }
        }else if constexpr(Headdim == 192) {
            if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
                using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<192, kBlockM, kBlockN, 4, false, false, T, 192>;
                using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_dim192<192, 64, 64, 4, T, 3, 192>;
                run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
            }
        }else if constexpr(Headdim == 256) {
            if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
                using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<256, kBlockM, kBlockN, 4, false, false, T, 256>;
                using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_dim256<256, 64, 64, 4, T, 3, 256>;
                run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
            }else {
                using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<256, kBlockM, kBlockN, 4, false, false, T, 256>;
                using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_vllm_kvcache_traits<256, 64, 64, 4, T, 3, 256>;
                run_flash_splitkv_fwd_16x64_vllm_kvcache_gfx928<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
            }
        }
    }else{
        run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, true, T>, Is_causal>(params, stream);
    } 
}

template<typename Q,typename KV, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch_kv_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int kBlockM = 64;  // Fixed for all head dimensions
    // TD [2023-08-28]: nvcc segfaults for headdim 96 with block size 64 x 256,
    // and for headdim 192 with block size 64 x 128.
    // Also for headdim 160 with block size 64 x 128 after the rotary addition.
    constexpr static int kBlockN = Headdim <= 128 ? 64 : (Headdim % 64 == 0 ? 32 : 64);
    int mblocks = (params.seqlen_q + 64 - 1) / 64;
    bool is_small = (params.seqlen_q <= 64||params.h * params.b * mblocks< 4*sm_count);
    // printf("kBlockM = %d, kBlockN = %d", kBlockM, kBlockN);
#ifndef FLASHATTENTION_DISABLE_SPLITKV
    if constexpr(Headdim == 64) {
        if (get_device_name() == "gfx936") {
            if (params.is_vllm_kvcache) {
                using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<64, kBlockM, kBlockN, 4, false, false, Q, 64>;
                if (is_small) {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_kv_fp8_dim64<64, 64, 64, 4, Q, KV, 1, 64>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch_kv_fp8<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                }else {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_kv_fp8_dim64<64, 128, 64, 4, Q, KV, 1, 64>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch_kv_fp8<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                }
            }
        }
    }else if constexpr(Headdim == 128) {
        if (get_device_name() == "gfx936") {
            if (params.is_vllm_kvcache) {
			     using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<128, kBlockM, kBlockN, 4, false, false, Q, 128>;
                if (is_small) {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_kv_fp8<128, 64, 64, 4, Q, KV, 1, 128>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch_kv_fp8<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                } else {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_kv_fp8<128, 128, 64, 4, Q, KV, 1, 128>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch_kv_fp8<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                }
            }
        }
    }else if constexpr(Headdim == 256) {
        if (get_device_name() == "gfx936") {
            if (params.is_vllm_kvcache) {
                using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<256, kBlockM, kBlockN, 4, false, false, Q, 256>;
                using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_kv_fp8_dim256<256, 64, 64, 4, Q, KV, 1, 256>;
                run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch_kv_fp8<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
            }
        }
    }
#endif
}

template<typename T,typename TO, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int kBlockM = 64;  // Fixed for all head dimensions
    // TD [2023-08-28]: nvcc segfaults for headdim 96 with block size 64 x 256,
    // and for headdim 192 with block size 64 x 128.
    // Also for headdim 160 with block size 64 x 128 after the rotary addition.
    constexpr static int kBlockN = Headdim <= 128 ? 64 : (Headdim % 64 == 0 ? 32 : 64);
    // printf("kBlockM = %d, kBlockN = %d", kBlockM, kBlockN);
#ifndef FLASHATTENTION_DISABLE_SPLITKV
    if constexpr(Headdim == 64) {
        if (get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
            if (params.is_vllm_kvcache) {
                using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<64, kBlockM, kBlockN, 4, false, false, TO, 64>;
                if (params.seqlen_q < 64) {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_fp8_dim64<64, 64, 64, 4, T, TO, 1, 64>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch_fp8<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                } else {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_fp8_dim64<64, 128, 64, 4, T, TO, 1, 64>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch_fp8<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                } 
            }
        }
    }else if constexpr(Headdim == 128) {
        if (get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
            if (params.is_vllm_kvcache) {
                using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<128, kBlockM, kBlockN, 4, false, false, TO, 128>;
                if (params.seqlen_q < 64) {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_fp8<128, 64, 64, 4, T, TO, 1, 128>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch_fp8<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                } else {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_fp8<128, 128, 64, 4, T, TO, 1, 128>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch_fp8<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                }
            }
        }
    }else if constexpr(Headdim == 192) {
        if (get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
            if (params.is_vllm_kvcache) {
                using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<192, kBlockM, kBlockN, 4, false, false, TO, 192>;
                if (params.seqlen_q < 64) {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_fp8_dim192<192, 64, 64, 4, T, TO, 1, 192>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch_fp8<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                } else {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_fp8_dim192<192, 128, 64, 4, T, TO, 1, 192>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch_fp8<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                } 
            }
        }
    }else if constexpr(Headdim == 256) {
        if (get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
            if (params.is_vllm_kvcache) {
                using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<256, kBlockM, kBlockN, 4, false, false, TO, 256>;
                if (params.seqlen_q < 64) {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_fp8_dim256<256, 64, 64, 4, T, TO, 1, 256>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch_fp8<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                } else {
                    using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_fp8_dim256<256, 128, 64, 4, T, TO, 1, 256>;
                    run_flash_splitkv_fwd_16x64_vllm_kvcache_prefetch_fp8<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
                } 
            }
        }
    }
#endif
}

template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_unified_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int kBlockM = 64;
    constexpr static int kBlockN = Headdim <= 128 ? 64 : (Headdim % 64 == 0 ? 32 : 64);
    if constexpr(Headdim == 256) {
        if (get_device_name() == "gfx938" || get_device_name() == "gfx936"|| get_device_name() == "gfx92a") {

            using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_unified_traits_dim256<256, 64, 64, 4, T, 3, 256>;
            using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<256, kBlockM, kBlockN, 4, false, false, T, 256>;
            run_flash_splitkv_fwd_16x64_unified_prefetch<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
        }
    } else if constexpr (Headdim == 128) {
        if (get_device_name() == "gfx936"||get_device_name() == "gfx938") {
            assert(params.knew_ptr == nullptr && params.block_table != nullptr);
            using prefetch_kernel_traits = Flash_fwd_kernel_16x64_splitkv_prefetch_unified_traits<128, 64, 64, 4, T, 3, 128>;
            using combine_kernel_traits = Flash_fwd_kernel_16x64_traits_splitkv<128, kBlockM, kBlockN, 4, false, false, T, 128>;
            run_flash_splitkv_fwd_16x64_unified_prefetch<prefetch_kernel_traits, combine_kernel_traits, Is_causal>(params, stream);
        }
    } else {
        assert(false && "unified attn only supported headdim=128/256");
    }
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
#if 1
    constexpr static int Headdim = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx936" || get_device_name() == "gfx938" && !Is_dropout && !params.s_aux_ptr && params.d == 32 && params.seqlen_q > 64) {
            run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim64<Headdim, 256, 128, 4, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
#endif
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
    #if 0
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 256, 64, 4, false, /*Share_Q_K_smem_=*/true, T>, Is_dropout, Is_causal>(params, stream);
    #else
    if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
        int mblocks = (params.seqlen_q + 64 - 1) / 64;
        if (params.seqlen_q <= 64||params.h * params.b * mblocks< 4*sm_count) {
            run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim64<Headdim, 64, 64, 4, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim64<Headdim, Is_dropout ? 128 : 256, 64, 4, T>, Is_dropout, Is_causal>(params, stream);
        }
    } else {
        run_flash_fwd_16x64<Flash_fwd_kernel_16x64_traits<Headdim, 256, 64, 4, /*Is_Q_use_smem_=*/false, /*Share_K_V_smem_=*/false, T>, Is_dropout, Is_causal>(params, stream);
    }
    #endif

    });
}
template<typename T, bool Is_causal>
void run_mha_fwd_padding_mask_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a")  {
            run_flash_fwd_16x64_prefetch_padding_mask<Flash_fwd_kernel_16x64_prefetch_traits_dim64<Headdim, 128, 64, 4, T>, Is_dropout, Is_causal>(params, stream);
        }
        else {
            run_flash_fwd_16x64<Flash_fwd_kernel_16x64_traits<Headdim, 128, 64, 4, /*Is_Q_use_smem_=*/false, /*Share_K_V_smem_=*/false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
            int mblocks = (params.seqlen_q + 64 - 1) / 64;
            if(params.seqlen_q <= 64||params.h * params.b * mblocks< 4*sm_count){
                run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim96<Headdim, 64, 64, 4, T, 3>, Is_dropout, Is_causal>(params, stream);
            }
            else{
                run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim96<Headdim, 128, 64, 4, T, 3>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal, bool Is_skip_softmax = false>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    // printf("run_mha_fwd_hdim128\n");
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") { 
            int mblocks = (params.seqlen_q + 64 - 1) / 64;
            if (params.seqlen_q <= 64||params.h * params.b * mblocks< 4*sm_count) {
                run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits<Headdim, 64, 64, 4, T, 3, Is_skip_softmax>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits<Headdim, 128, 64, 4, T, 3, Is_skip_softmax>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd_16x64<Flash_fwd_kernel_16x64_traits<Headdim, 128, 64, 4, /*Is_Q_use_smem_=*/true, /*Share_K_V_smem_=*/false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim128_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    static constexpr bool Is_FP8 = cute::is_same_v<T, cutlass::float_e4m3_t> || cute::is_same_v<T, cutlass::float_e5m2_t>;
    using T_out = std::conditional_t<!Is_FP8, T, cutlass::bfloat16_t>;
    // printf("run_mha_fwd_hdim128\n");
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx938"|| get_device_name() == "gfx92a") { 
            int mblocks = (params.seqlen_q + 64 - 1) / 64;
            if (params.seqlen_q <= 64||params.h * params.b * mblocks< 4*sm_count) {
                run_flash_fwd_16x64_prefetch_fp8<Flash_fwd_kernel_16x64_prefetch_traits_fp8<Headdim, 64, 64, 4, T,T_out, 3>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd_16x64_prefetch_fp8<Flash_fwd_kernel_16x64_prefetch_traits_fp8<Headdim, 128, 64, 4, T,T_out, 3>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            printf("this device is not supoort fp8");
        }
    });
}


template<typename T, bool Is_causal>
void run_mha_fwd_hdim160(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 160;
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major == 8 && dprops->minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim192_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
            run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_mla_traits</*Headdim*/192, 128, 64, 4, T, 3, /*HeaddimV*/128>, Is_dropout, Is_causal>(params, stream);
        }
        else {
            run_flash_fwd_16x64<Flash_fwd_kernel_16x64_traits_MLA</*Headdim*/192, 128, 64, 4, /*Is_Q_use_smem_=*/false, /*Share_K_V_smem_=*/true, T, 128>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim192_hdim128_fp8(Flash_fwd_params &params, cudaStream_t stream) {

    static constexpr bool Is_FP8 = cute::is_same_v<T, cutlass::float_e4m3_t> || cute::is_same_v<T, cutlass::float_e5m2_t>;
    using T_out = std::conditional_t<!Is_FP8, T, cutlass::bfloat16_t>;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
            run_flash_fwd_16x64_prefetch_fp8<Flash_fwd_kernel_16x64_prefetch_mla_traits_fp8</*Headdim*/192, 128, 64, 4, T,T_out, 3, /*HeaddimV*/128>, Is_dropout, Is_causal>(params, stream);
        }
        else {
            printf("this device is not supoort fp8");
        }
    });

}
template<typename T, bool Is_causal>
void run_mha_fwd_hdim224(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 224;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // constexpr static int Is_dropout = false;
        if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
            run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim256<Headdim, 128, 64, 4, T, 3>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim512(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 512;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // constexpr static int Is_dropout = false;
        if (get_device_name() == "gfx936" || get_device_name() == "gfx938"|| get_device_name() == "gfx92a") {
            run_flash_fwd_16x64_prefetch<Flash_fwd_kernel_16x64_prefetch_traits_dim512<Headdim, 64, 64, 4, T, 3>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}
