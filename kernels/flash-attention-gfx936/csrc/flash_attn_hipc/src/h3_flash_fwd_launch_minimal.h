#pragma once

#include "numeric_types.h"

// OpenDAS main currently targets a newer aicc frontend.  DTK 26.04's dcc can
// emit the same packed vector arithmetic, but does not expose this newer alias.
#if !__has_builtin(__builtin_hcu_pk_add_f32)
__device__ __forceinline__ __float2 h3_dcc_pk_add_f32(__float2 lhs, __float2 rhs) {
#ifdef H3_PACKED_F32_SHIMS
    __float2 result;
    asm volatile("v_pk_add_f32 %0, %1, %2 ; h3_dcc_pk_add_f32"
                 : "=v"(result)
                 : "v"(lhs), "v"(rhs));
    return result;
#else
    return lhs + rhs;
#endif
}
#define __builtin_hcu_pk_add_f32(lhs, rhs) h3_dcc_pk_add_f32((lhs), (rhs))
#endif

#if !__has_builtin(__builtin_hcu_pk_mul_f32)
__device__ __forceinline__ __float2 h3_dcc_pk_mul_f32(__float2 lhs, __float2 rhs) {
#ifdef H3_PACKED_F32_SHIMS
    __float2 result;
    asm volatile("v_pk_mul_f32 %0, %1, %2 ; h3_dcc_pk_mul_f32"
                 : "=v"(result)
                 : "v"(lhs), "v"(rhs));
    return result;
#else
    return lhs * rhs;
#endif
}
#define __builtin_hcu_pk_mul_f32(lhs, rhs) h3_dcc_pk_mul_f32((lhs), (rhs))
#endif

#if !__has_builtin(__builtin_hcu_pk_fma_f32)
__device__ __forceinline__ __float2 h3_dcc_pk_fma_f32(
    __float2 lhs,
    __float2 rhs,
    __float2 accumulator) {
#ifdef H3_PACKED_F32_SHIMS
    __float2 result;
    asm volatile("v_pk_fma_f32 %0, %1, %2, %3 ; h3_dcc_pk_fma_f32"
                 : "=v"(result)
                 : "v"(lhs), "v"(rhs), "v"(accumulator));
    return result;
#else
    return __float2{
        __builtin_fmaf(lhs[0], rhs[0], accumulator[0]),
        __builtin_fmaf(lhs[1], rhs[1], accumulator[1])};
#endif
}
#define __builtin_hcu_pk_fma_f32(lhs, rhs, accumulator) \
    h3_dcc_pk_fma_f32((lhs), (rhs), (accumulator))
#endif

#include "config.h"
#include "static_switch.h"
#include "flash.h"
#include "flash_fwd_b16_fa.h"
#include "flash_singleton.h"

#ifndef H3_BLOCK_N
#define H3_BLOCK_N 128
#endif


template <typename KernelTraits>
__global__ void __launch_bounds__(256, 1) h3_flash_fwd_bf16_kernel(Flash_fwd_params params) {
    flash::compute_attn<
        KernelTraits,
        /*Is_training=*/true,
        /*Is_dropout=*/false,
        /*Is_causal=*/false,
        /*Is_local=*/false,
        /*Is_even_MN=*/false,
        /*Return_softmax=*/false,
        /*Has_alibi=*/false,
        /*Is_GQA=*/false,
        /*Layout=*/0,
        Flash_fwd_params,
        int32_t,
        /*IsVarlen=*/false>(params);
}


inline void run_h3_flash_fwd_bf16_hdim128(Flash_fwd_params& params, hipStream_t stream) {
    using Traits = Flash_fwd_kernel_traits<
        128, 128, 128, H3_BLOCK_N, 32, 32, 32, 2,
        false, false, BFloat16>;

    auto& properties = DeviceProperties<Traits, FAFUNC::FORWARD>::GetInstance();
    params.cu_count = properties.cu_count;
    const size_t smem_size = properties.lds_size;
    const int query_blocks = (params.seqlen_q + Traits::kBlockM - 1) / Traits::kBlockM;
    const dim3 grid(query_blocks, params.h, params.b);
    auto kernel = &h3_flash_fwd_bf16_kernel<Traits>;
    if (smem_size >= 64 * 1024) {
        HIP_CHECK(hipFuncSetAttribute(
            reinterpret_cast<const void*>(kernel),
            hipFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }
    kernel<<<grid, Traits::kNThreads, smem_size, stream>>>(params);
}
