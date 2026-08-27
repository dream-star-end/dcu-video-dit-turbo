/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
// #include <cutlass/arch/memory_buffer.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////
template<const int COUNT>
__forceinline__ __device__ void s_nop() {
    asm volatile("s_nop %0":: "B"(COUNT) :);
}

__forceinline__ __device__ void s_barrier() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_barrier");
    __builtin_amdgcn_sched_barrier(0);
}

template<const int COUNT>
__forceinline__ __device__ void s_waitcnt() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile(
      "s_waitcnt vmcnt(%0)\n\t"
      "s_barrier\n"
      :: "B"(COUNT)
      :);
    __builtin_amdgcn_sched_barrier(0);
}

template<const int COUNT>
__forceinline__ __device__ void s_waitcnt_nosync() {
    asm volatile(
      "s_waitcnt vmcnt(%0)\n\t"
      :: "B"(COUNT)
      :);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__forceinline__ __device__ uint32_t relu2(const uint32_t x);

template<>
__forceinline__ __device__ uint32_t relu2<cutlass::half_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;

#ifdef __HIP_DEVICE_COMPILE__ 
    // 暂时不使用ptx指令，后续优化点
    const auto x_p = reinterpret_cast<const cutlass::half_t*>(&x);
    auto res_p = reinterpret_cast<cutlass::half_t*>(&res);

    res_p[0] = (x_p[0] >= cutlass::half_t(0)) ? x_p[0] : cutlass::half_t(0);
    res_p[1] = (x_p[1] >= cutlass::half_t(0)) ? x_p[1] : cutlass::half_t(0);
#else
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
#else
    // asm volatile( \
    //     "{\n" \
    //     "\t .reg .f16x2 sela;\n" \
    //     "\t set.gtu.u32.f16x2 sela, %1, %2;\n" \
    //     "\t and.b32 %0, sela, %1;\n" 
    //     "}\n" : "=r"(res) : "r"(x), "r"(zero));
#endif
#endif

    return res;
}

template<>
__forceinline__ __device__ uint32_t relu2<cutlass::bfloat16_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;

#ifdef __HIP_DEVICE_COMPILE__ 
    // 暂时不使用ptx指令，后续优化点
    const auto x_p = reinterpret_cast<const cutlass::bfloat16_t*>(&x);
    auto res_p = reinterpret_cast<cutlass::bfloat16_t*>(&res);

    res_p[0] = (x_p[0] >= cutlass::bfloat16_t(0)) ? x_p[0] : cutlass::bfloat16_t(0);
    res_p[1] = (x_p[1] >= cutlass::bfloat16_t(0)) ? x_p[1] : cutlass::bfloat16_t(0);
#else
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // asm volatile("max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
#endif
#endif

    return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

template<typename T>
__forceinline__ __device__ uint32_t convert_relu2(const float2 x);

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::half_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    // asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::bfloat16_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    // asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 64 || THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4 || THREADS == 2);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor(x, OFFSET, 64));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<1> {
    // static_assert(THREADS == 64 || THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4 || THREADS == 2);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        return x;
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<32> {
template<typename T, typename Operator> 
static __device__ __forceinline__ T run(T x, Operator &op) {
     x = op(x, __shfl_xor(x, 16, 64));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    if(!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if(!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
#pragma unroll
    for(int i = 0; i < size<2>(tCrA); ++i) {
        if(i < size<2>(tCrA) - 1) {
            if(!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if(!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }

        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        if (i == 0) {
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(1);
            __builtin_amdgcn_sched_barrier(0);
        }
    }
    
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }

        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        if (i == 0) {
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(1);
            __builtin_amdgcn_sched_barrier(0);
        }
    }
    
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
}

template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_B(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 16);
    // static_assert(decltype(size<2>(acc_layout))::value == 1);
    static_assert(decltype(rank(acc_layout))::value == 3);

    // return make_layout(get<0>(get<0>(acc_layout)), get<1>(acc_layout), get<1>(get<0>(acc_layout)));
    return make_layout(get<0>(get<0>(acc_layout)), make_layout(get<1>(get<0>(acc_layout)), get<1>(acc_layout)), get<2>(acc_layout));
};

template<typename Element, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs_pad(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B, int max_mn) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        auto tCrB_ = make_tensor(tCrB.data(), convert_layout_acc_B(tCrB.layout()));
        int col = i * 16 + ((threadIdx.x % 64) / 16) * 4;
        for (int j = 0; j < size<0>(tCrB_); j++) {
            for (int k = 0; k < size<1>(tCrB_); k++) {
                tCrB_(j, k, i) = col + j >= max_mn ? Element(0.0f) : tCrB_(j, k, i);
            }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        if (i == 0) {
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(1);
            __builtin_amdgcn_sched_barrier(0);
        }
    }
    
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
}

// template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
//          typename TiledMma, typename TiledCopy, typename ThrCopy>
// __forceinline__ __device__ void gemm_rs_debug__(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
//                                TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
//                                ThrCopy smem_thr_copy_B) {
//     CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
//     CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
//     CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
//     Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
//     CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
//     cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));

//     if (block0())
//     {
//         printf("tidx = %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f  \n", threadIdx.x, 
//             float(tCrB(0, 0, 0)),
//             float(tCrB(1, 0, 0)),
//             float(tCrB(2, 0, 0)),
//             float(tCrB(3, 0, 0)),
//             float(tCrB(4, 0, 0)),
//             float(tCrB(5, 0, 0)),
//             float(tCrB(6, 0, 0)),
//             float(tCrB(7, 0, 0))
//         );
//     }

//     // #pragma unroll
//     // for (int i = 0; i < size<2>(tCrA); ++i) {
//     //     if (i < size<2>(tCrA) - 1) {
//     //         cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
//     //     }

//     //     cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
//     //     if (i == 0) {
//     //         __builtin_amdgcn_sched_barrier(0);
//     //         __builtin_amdgcn_s_setprio(1);
//     //         __builtin_amdgcn_sched_barrier(0);
//     //     }
//     // }
    
//     __builtin_amdgcn_sched_barrier(0);
//     __builtin_amdgcn_s_setprio(0);
//     __builtin_amdgcn_sched_barrier(0);
// }
template<typename Element,typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_k_rs_pad_ws(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B, int k_idx, int Max_Mn) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    // __builtin_amdgcn_sched_barrier(0);
    // __builtin_amdgcn_s_setprio(0);
    // __builtin_amdgcn_sched_barrier(0);
    // using From_type = typename Tensor0::Engine::value_type;
    int tidx = threadIdx.x;
    // __builtin_amdgcn_sched_barrier(0);
    cute::copy(smem_tiled_copy_B, tCsB(_, _, k_idx), tCrB_copy_view(_, _, k_idx));
    // __builtin_amdgcn_sched_barrier(0);
    int need_pad_k_idx = Max_Mn / 16;
    if (need_pad_k_idx == k_idx) {
        auto tCrB_ = make_tensor(tCrB.data(), convert_layout_acc_B(tCrB.layout()));
        for (int ni = 0; ni < size<1>(tCrB_); ni++) {
            int col = k_idx * 16 + ((tidx % 64) / 16) * 4;
            for (int ei = 0; ei < size<0>(tCrB_); ei++) {
                tCrB_(ei, ni, k_idx) = col + ei >= Max_Mn ? Element(0) : tCrB_(ei, ni, k_idx);
            }
        }
    }
    
    cute::gemm(tiled_mma, tCrA(_, _, k_idx), tCrB(_, _, k_idx), acc);
}

template<typename Element,typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_k_rs_pad(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B, int k_idx, int Max_Mn) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    // __builtin_amdgcn_sched_barrier(0);
    // __builtin_amdgcn_s_setprio(0);
    // __builtin_amdgcn_sched_barrier(0);
    // using From_type = typename Tensor0::Engine::value_type;
    int tidx = threadIdx.x;
    cute::copy(smem_tiled_copy_B, tCsB(_, _, k_idx), tCrB_copy_view(_, _, k_idx));
    int need_pad_k_idx = Max_Mn / 16;
    int round_4 = Max_Mn % 4;
    if (need_pad_k_idx == k_idx && round_4 != 0) {
        auto tCrB_ = make_tensor(tCrB.data(), convert_layout_acc_B(tCrB.layout()));
        
        for (int ni = 0; ni < size<1>(tCrB_); ni++)
        {
            int col = k_idx * 16 + ((tidx % 64) / 16) * 4;
            for (int ei = 0; ei < size<0>(tCrB_); ei++)
            {
                tCrB_(ei, ni, k_idx) = col + ei >= Max_Mn ? Element(0) : tCrB_(ei, ni, k_idx);
            }
        }
    }
    // __builtin_amdgcn_sched_barrier(0);
    // __builtin_amdgcn_s_setprio(1);
    // __builtin_amdgcn_sched_barrier(0);
    cute::gemm(tiled_mma, tCrA(_, _, k_idx), tCrB(_, _, k_idx), acc);
}

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_k_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B, int k_idx) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    // __builtin_amdgcn_sched_barrier(0);
    // __builtin_amdgcn_s_setprio(0);
    // __builtin_amdgcn_sched_barrier(0);
    cute::copy(smem_tiled_copy_B, tCsB(_, _, k_idx), tCrB_copy_view(_, _, k_idx));
    // __builtin_amdgcn_sched_barrier(0);
    // __builtin_amdgcn_s_setprio(1);
    // __builtin_amdgcn_sched_barrier(0);
    cute::gemm(tiled_mma, tCrA(_, _, k_idx), tCrB(_, _, k_idx), acc);
}

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_k_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B, int kA_idx, int kB_idx) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    // CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, kB_idx), tCrB_copy_view(_, _, kB_idx));
    cute::gemm(tiled_mma, tCrA(_, _, kA_idx), tCrB(_, _, kB_idx), acc);
}

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_k_rs_debug(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B, int k_idx) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, k_idx), tCrB_copy_view(_, _, k_idx));
    cute::gemm(tiled_mma, tCrA(_, _, k_idx), tCrB(_, _, k_idx), acc);
    
    int tidx = threadIdx.x;
    printf("tid:%d k_idx:%d tCrA:%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        "tCrB:%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        "acc:%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f \n", tidx, k_idx, 
        float(tCrA(0, 0, k_idx)), float(tCrA(1, 0, k_idx)), float(tCrA(2, 0, k_idx)), float(tCrA(3, 0, k_idx)), 
        float(tCrA(4, 0, k_idx)), float(tCrA(5, 0, k_idx)), float(tCrA(6, 0, k_idx)), float(tCrA(7, 0, k_idx)), 
        float(tCrB(0, 0, k_idx)), float(tCrB(1, 0, k_idx)), float(tCrB(2, 0, k_idx)), float(tCrB(3, 0, k_idx)), 
        float(tCrB(4, 0, k_idx)), float(tCrB(5, 0, k_idx)), float(tCrB(6, 0, k_idx)), float(tCrB(7, 0, k_idx)),
        acc(0), acc(1), acc(2), acc(3), acc(4), acc(5), acc(6), acc(7), 
        acc(8), acc(9), acc(10), acc(11), acc(12), acc(13), acc(14), acc(15), 
        acc(16), acc(17), acc(18), acc(19), acc(20), acc(21), acc(22), acc(23), 
        acc(24), acc(25), acc(26), acc(27), acc(28), acc(29), acc(30), acc(31) 
    );
}

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_k_rs_debug(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B, int kA_idx, int kB_idx) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    // CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, kB_idx), tCrB_copy_view(_, _, kB_idx));
    
    cute::gemm(tiled_mma, tCrA(_, _, kA_idx), tCrB(_, _, kB_idx), acc);
    
    int tidx = threadIdx.x;
    printf("tid:%d kA_idx:%d kB_idx:%d tCrA:%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        "tCrB:%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        "acc:%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f \n", tidx, kA_idx, kB_idx, 
        float(tCrA(0, 0, kA_idx)), float(tCrA(1, 0, kA_idx)), float(tCrA(2, 0, kA_idx)), float(tCrA(3, 0, kA_idx)), 
        float(tCrA(4, 0, kA_idx)), float(tCrA(5, 0, kA_idx)), float(tCrA(6, 0, kA_idx)), float(tCrA(7, 0, kA_idx)), 
        float(tCrB(0, 0, kB_idx)), float(tCrB(1, 0, kB_idx)), float(tCrB(2, 0, kB_idx)), float(tCrB(3, 0, kB_idx)), 
        float(tCrB(4, 0, kB_idx)), float(tCrB(5, 0, kB_idx)), float(tCrB(6, 0, kB_idx)), float(tCrB(7, 0, kB_idx)),
        float(tCrB(8, 0, kB_idx)), float(tCrB(9, 0, kB_idx)), float(tCrB(10, 0, kB_idx)), float(tCrB(11, 0, kB_idx)), 
        float(tCrB(12, 0, kB_idx)), float(tCrB(13, 0, kB_idx)), float(tCrB(14, 0, kB_idx)), float(tCrB(15, 0, kB_idx)),
        float(tCrB(16, 0, kB_idx)), float(tCrB(17, 0, kB_idx)), float(tCrB(18, 0, kB_idx)), float(tCrB(19, 0, kB_idx)), 
        float(tCrB(20, 0, kB_idx)), float(tCrB(21, 0, kB_idx)), float(tCrB(22, 0, kB_idx)), float(tCrB(23, 0, kB_idx)),
        float(tCrB(24, 0, kB_idx)), float(tCrB(25, 0, kB_idx)), float(tCrB(26, 0, kB_idx)), float(tCrB(27, 0, kB_idx)), 
        float(tCrB(28, 0, kB_idx)), float(tCrB(29, 0, kB_idx)), float(tCrB(30, 0, kB_idx)), float(tCrB(31, 0, kB_idx)),
        acc(0), acc(1), acc(2), acc(3), acc(4), acc(5), acc(6), acc(7), 
        acc(8), acc(9), acc(10), acc(11), acc(12), acc(13), acc(14), acc(15), 
        acc(16), acc(17), acc(18), acc(19), acc(20), acc(21), acc(22), acc(23), 
        acc(24), acc(25), acc(26), acc(27), acc(28), acc(29), acc(30), acc(31) 
    );
}

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs_swait(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    asm volatile("s_waitcnt lgkmcnt(0)\n\t" : :);
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            asm volatile("s_waitcnt lgkmcnt(0)\n\t" : :);
        }

        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        if (i == 0) {
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(1);
            __builtin_amdgcn_sched_barrier(0);
        }
    }
    
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
}

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_k_rs_swait(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B, int k_idx) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, k_idx), tCrB_copy_view(_, _, k_idx));
    asm volatile("s_waitcnt lgkmcnt(0)");
    // int tidx = threadIdx.x;
    // printf("tid:%d k_idx:%d tCrA:%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "tCrB:%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, k_idx, 
    //     float(tCrA(0, 0, k_idx)), float(tCrA(1, 0, k_idx)), float(tCrA(2, 0, k_idx)), float(tCrA(3, 0, k_idx)), 
    //     float(tCrA(4, 0, k_idx)), float(tCrA(5, 0, k_idx)), float(tCrA(6, 0, k_idx)), float(tCrA(7, 0, k_idx)), 
    //     float(tCrA(0, 1, k_idx)), float(tCrA(1, 1, k_idx)), float(tCrA(2, 1, k_idx)), float(tCrA(3, 1, k_idx)), 
    //     float(tCrA(4, 1, k_idx)), float(tCrA(5, 1, k_idx)), float(tCrA(6, 1, k_idx)), float(tCrA(7, 1, k_idx)), 
    //     float(tCsB(0, 0, k_idx)), float(tCsB(1, 0, k_idx)), float(tCsB(2, 0, k_idx)), float(tCsB(3, 0, k_idx)), 
    //     float(tCsB(4, 0, k_idx)), float(tCsB(5, 0, k_idx)), float(tCsB(6, 0, k_idx)), float(tCsB(7, 0, k_idx))
    // );
    cute::gemm(tiled_mma, tCrA(_, _, k_idx), tCrB(_, _, k_idx), acc);
}

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs_debug(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    // wangaq debug
    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     int offset = reinterpret_cast<const char *>(&tCsB(0, 0, _0{})) - (char *)(0x1000000000000);
    //     printf("tid:%d i:0 tCsB:%p %p %p %p "
    //         "%p %p %p %p "
    //         "offset:%d row:%d col:%d\n", threadIdx.x, 
    //         &tCsB(0, 0, _0{}), &tCsB(1, 0, _0{}), &tCsB(2, 0, _0{}), &tCsB(3, 0, _0{}), 
    //         &tCsB(4, 0, _0{}), &tCsB(5, 0, _0{}), &tCsB(6, 0, _0{}), &tCsB(7, 0, _0{}), 
    //         offset, offset/128, (offset % 128)/16);
    // }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            // wangaq debug
            // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            //     int offset = reinterpret_cast<const char *>(&tCsB(0, 0, _0{})) - (char *)(0x1000000000000);
            //     printf("tid:%d i:%d tCsB:%p %p %p %p "
            //         "%p %p %p %p "
            //         "offset:%d row:%d col:%d\n", threadIdx.x, i + 1,
            //         &tCsB(0, 0, i + 1), &tCsB(1, 0, i + 1), &tCsB(2, 0, i + 1), &tCsB(3, 0, i + 1), 
            //         &tCsB(4, 0, i + 1), &tCsB(5, 0, i + 1), &tCsB(6, 0, i + 1), &tCsB(7, 0, i + 1), 
            //         offset, offset/128, (offset % 128)/16);
            // }
        }

        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);

        // wangaq debug
        // if(thread0()) {
        //     printf("i:%d tCrA:%7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f | "
        //         "%7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f \n"
        //         "tCrB:%7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f | "
        //         "%7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f \n"
        //         "acc:%7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f | "
        //         "%7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f | %7.4f %7.4f %7.4f %7.4f \n", i, 
        //         float(tCrA(0,  0, i)),  float(tCrA(1,  0, i)),  float(tCrA(2,  0, i)),  float(tCrA(3,  0, i)), 
        //         float(tCrA(4,  0, i)),  float(tCrA(5,  0, i)),  float(tCrA(6,  0, i)),  float(tCrA(7,  0, i)), 
        //         float(tCrA(8,  0, i)),  float(tCrA(9,  0, i)),  float(tCrA(10, 0, i)),  float(tCrA(11, 0, i)), 
        //         float(tCrA(12, 0, i)),  float(tCrA(13, 0, i)),  float(tCrA(14, 0, i)),  float(tCrA(15, 0, i)), 
        //         float(tCrA(0,  1, i)),  float(tCrA(1,  1, i)),  float(tCrA(2,  1, i)),  float(tCrA(3,  1, i)), 
        //         float(tCrA(4,  1, i)),  float(tCrA(5,  1, i)),  float(tCrA(6,  1, i)),  float(tCrA(7,  1, i)), 
        //         float(tCrA(8,  1, i)),  float(tCrA(9,  1, i)),  float(tCrA(10, 1, i)),  float(tCrA(11, 1, i)), 
        //         float(tCrA(12, 1, i)),  float(tCrA(13, 1, i)),  float(tCrA(14, 1, i)),  float(tCrA(15, 1, i)), 
        //         float(tCrB(0,  0, i)),  float(tCrB(1,  0, i)),  float(tCrB(2,  0, i)),  float(tCrB(3,  0, i)), 
        //         float(tCrB(4,  0, i)),  float(tCrB(5,  0, i)),  float(tCrB(6,  0, i)),  float(tCrB(7,  0, i)), 
        //         float(tCrB(0,  1, i)),  float(tCrB(1,  1, i)),  float(tCrB(2,  1, i)),  float(tCrB(3,  1, i)), 
        //         float(tCrB(4,  1, i)),  float(tCrB(5,  1, i)),  float(tCrB(6,  1, i)),  float(tCrB(7,  1, i)), 
        //         float(tCrB(0,  2, i)),  float(tCrB(1,  2, i)),  float(tCrB(2,  2, i)),  float(tCrB(3,  2, i)), 
        //         float(tCrB(4,  2, i)),  float(tCrB(5,  2, i)),  float(tCrB(6,  2, i)),  float(tCrB(7,  2, i)), 
        //         float(tCrB(0,  3, i)),  float(tCrB(1,  3, i)),  float(tCrB(2,  3, i)),  float(tCrB(3,  3, i)), 
        //         float(tCrB(4,  3, i)),  float(tCrB(5,  3, i)),  float(tCrB(6,  3, i)),  float(tCrB(7,  3, i)), 
        //         acc(0), acc(1), acc(2), acc(3), acc(4), acc(5), acc(6), acc(7), 
        //         acc(8), acc(9), acc(10), acc(11), acc(12), acc(13), acc(14), acc(15), 
        //         acc(16), acc(17), acc(18), acc(19), acc(20), acc(21), acc(22), acc(23), 
        //         acc(24), acc(25), acc(26), acc(27), acc(28), acc(29), acc(30), acc(31) 
        //     );
        // }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm_rr(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, TiledMma tiled_mma) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        if (i == 0) {
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(1);
            __builtin_amdgcn_sched_barrier(0);
        }
    }
    
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template<int row, int col, typename Tensor0, typename Tensor1>
__forceinline__ __device__  static void __ds_read_m32x16_row_col_alt(Tensor0& src, Tensor1& dst)
{

    auto lds = reinterpret_cast<__fp16 *>(src.data().get());
    auto layout  = src.layout();
    constexpr short offset = layout(0, row, col) * 2;

    auto d = __builtin_amdgcn_ds_read_m32x16f16_alt((__attribute__((address_space(3))) __fp16*)(lds), offset);

    uint16_t * d_ptr = reinterpret_cast<uint16_t*>(&d);
    uint16_t * dst_ptr = reinterpret_cast<uint16_t*>(&(dst(0, row, col)));

    dst_ptr[0] = d_ptr[0];
    dst_ptr[1] = d_ptr[1];
    dst_ptr[2] = d_ptr[2];
    dst_ptr[3] = d_ptr[3];
    dst_ptr[4] = d_ptr[4];
    dst_ptr[5] = d_ptr[5];
    dst_ptr[6] = d_ptr[6];
    dst_ptr[7] = d_ptr[7];
}

template<int k_idx, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_k_rs_ds_read_m32x16_alt(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    auto shape = tCsB.shape();
    constexpr int rows = get<1>(shape);
    static_assert(rows == 6 || rows == 4 || rows == 3 || rows == 2 || rows == 1);
    if constexpr (rows == 6) {
        __ds_read_m32x16_row_col_alt<0, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col_alt<1, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col_alt<2, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col_alt<3, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col_alt<4, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col_alt<5, k_idx>(tCsB, tCrB_copy_view);
    } else if constexpr (rows == 4) {
        __ds_read_m32x16_row_col_alt<0, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col_alt<1, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col_alt<2, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col_alt<3, k_idx>(tCsB, tCrB_copy_view);
    } else if constexpr (rows == 3) {
        __ds_read_m32x16_row_col_alt<0, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col_alt<1, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col_alt<2, k_idx>(tCsB, tCrB_copy_view);
    } else if constexpr (rows == 2) {
        __ds_read_m32x16_row_col_alt<0, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col_alt<1, k_idx>(tCsB, tCrB_copy_view);
    } else if constexpr (rows == 1) {
        __ds_read_m32x16_row_col_alt<0, k_idx>(tCsB, tCrB_copy_view);
    }         
                        
    // cute::copy(smem_tiled_copy_B, tCsB(_, _, k_idx), tCrB_copy_view(_, _, k_idx));
    cute::gemm(tiled_mma, tCrA(_, _, k_idx), tCrB(_, _, k_idx), acc);
}

template<int row, int col, typename Tensor0, typename Tensor1>
__forceinline__ __device__  static void __ds_read_m32x16_row_col(Tensor0& src, Tensor1& dst)
{

    auto lds = reinterpret_cast<__fp16 *>(src.data().get());
    auto layout  = src.layout();
    constexpr short offset = layout(0, row, col) * 2;

    auto d = __builtin_amdgcn_ds_read_m32x16f16((__attribute__((address_space(3))) __fp16*)(lds), offset);

    uint16_t * d_ptr = reinterpret_cast<uint16_t*>(&d);
    uint16_t * dst_ptr = reinterpret_cast<uint16_t*>(&(dst(0, row, col)));

    dst_ptr[0] = d_ptr[0];
    dst_ptr[1] = d_ptr[1];
    dst_ptr[2] = d_ptr[2];
    dst_ptr[3] = d_ptr[3];
    dst_ptr[4] = d_ptr[4];
    dst_ptr[5] = d_ptr[5];
    dst_ptr[6] = d_ptr[6];
    dst_ptr[7] = d_ptr[7];
}

template<int k_idx, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_k_rs_ds_read_m32x16(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    auto shape = tCsB.shape();
    constexpr int rows = get<1>(shape);
    static_assert(rows == 6 || rows == 4 || rows == 3 || rows == 2);
    if constexpr (rows == 6) {
        __ds_read_m32x16_row_col<0, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<1, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<2, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<3, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<4, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<5, k_idx>(tCsB, tCrB_copy_view);
    } else if constexpr (rows == 4) {
        __ds_read_m32x16_row_col<0, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<1, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<2, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<3, k_idx>(tCsB, tCrB_copy_view);
    } else if constexpr (rows == 3) {
        __ds_read_m32x16_row_col<0, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<1, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<2, k_idx>(tCsB, tCrB_copy_view);
    } 
    else if constexpr (rows == 2) {
        __ds_read_m32x16_row_col<0, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<1, k_idx>(tCsB, tCrB_copy_view);
    }                        
    // cute::copy(smem_tiled_copy_B, tCsB(_, _, k_idx), tCrB_copy_view(_, _, k_idx));
    cute::gemm(tiled_mma, tCrA(_, _, k_idx), tCrB(_, _, k_idx), acc);
}

template<int k_idxA, int k_idxB, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_k_rs_ds_read_m32x16(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    auto shape = tCsB.shape();
    constexpr int rows = get<1>(shape);
    static_assert(rows == 6 || rows == 4 || rows == 3 || rows == 2);
    if constexpr (rows == 6) {
        __ds_read_m32x16_row_col<0, k_idxB>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<1, k_idxB>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<2, k_idxB>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<3, k_idxB>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<4, k_idxB>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<5, k_idxB>(tCsB, tCrB_copy_view);
    } else if constexpr (rows == 4) {
        __ds_read_m32x16_row_col<0, k_idxB>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<1, k_idxB>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<2, k_idxB>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<3, k_idxB>(tCsB, tCrB_copy_view);
    } else if constexpr (rows == 3) {
        __ds_read_m32x16_row_col<0, k_idxB>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<1, k_idxB>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<2, k_idxB>(tCsB, tCrB_copy_view);
    } 
    else if constexpr (rows == 2) {
        __ds_read_m32x16_row_col<0, k_idxB>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<1, k_idxB>(tCsB, tCrB_copy_view);
    }                        
    cute::gemm(tiled_mma, tCrA(_, _, k_idxA), tCrB(_, _, k_idxB), acc);
}

template<int k_idx, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_k_rs_ds_read_m32x16_debug(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    auto shape = tCsB.shape();
    constexpr int rows = get<1>(shape);
    static_assert(rows == 6 || rows == 4 || rows == 3 || rows == 2);
    if constexpr (rows == 6) {
        __ds_read_m32x16_row_col<0, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<1, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<2, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<3, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<4, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<5, k_idx>(tCsB, tCrB_copy_view);
    } else if constexpr (rows == 4) {
        __ds_read_m32x16_row_col<0, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<1, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<2, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<3, k_idx>(tCsB, tCrB_copy_view);
    } else if constexpr (rows == 3) {
        __ds_read_m32x16_row_col<0, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<1, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<2, k_idx>(tCsB, tCrB_copy_view);
    } 
    else if constexpr (rows == 2) {
        __ds_read_m32x16_row_col<0, k_idx>(tCsB, tCrB_copy_view);
        __ds_read_m32x16_row_col<1, k_idx>(tCsB, tCrB_copy_view);
    }                        
    // cute::copy(smem_tiled_copy_B, tCsB(_, _, k_idx), tCrB_copy_view(_, _, k_idx));
    cute::gemm(tiled_mma, tCrA(_, _, k_idx), tCrB(_, _, k_idx), acc);

    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        int tidx = threadIdx.x;
        printf("tid:%d k_idx:%d tCrA:%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
            "tCrB:%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
            "acc:%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
            "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
            "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
            "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f \n", tidx, k_idx, 
            float(tCrA(0, 0, k_idx)), float(tCrA(1, 0, k_idx)), float(tCrA(2, 0, k_idx)), float(tCrA(3, 0, k_idx)), 
            float(tCrA(4, 0, k_idx)), float(tCrA(5, 0, k_idx)), float(tCrA(6, 0, k_idx)), float(tCrA(7, 0, k_idx)), 
            float(tCrB(0, 0, k_idx)), float(tCrB(1, 0, k_idx)), float(tCrB(2, 0, k_idx)), float(tCrB(3, 0, k_idx)), 
            float(tCrB(4, 0, k_idx)), float(tCrB(5, 0, k_idx)), float(tCrB(6, 0, k_idx)), float(tCrB(7, 0, k_idx)),
            acc(0), acc(1), acc(2), acc(3), acc(4), acc(5), acc(6), acc(7), 
            acc(8), acc(9), acc(10), acc(11), acc(12), acc(13), acc(14), acc(15), 
            acc(16), acc(17), acc(18), acc(19), acc(20), acc(21), acc(22), acc(23), 
            acc(24), acc(25), acc(26), acc(27), acc(28), acc(29), acc(30), acc(31) 
        );
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    // static_assert(decltype(size<0>(acc_layout))::value == 4 || decltype(size<0>(acc_layout))::value == 8);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_1>{});   // (_4,_1,_2):(_1,_0,_4) -> ((_1,_4),_1,_2):((_0,_1),_0,_4)

    return make_layout(make_layout(get<1>(l)), make_layout(get<1>(get<0>(l)), get<2>(l)));  // (1, (4, 2)):((_0),(_1,_4))
};

template<typename Layout>
__forceinline__ __device__ auto convert_trans_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    
    return make_layout(
        make_layout(get<0>(acc_layout), get<2>(acc_layout)), 
        make_layout(get<1>(acc_layout)));
};

template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 16);
    // static_assert(decltype(size<2>(acc_layout))::value == 1);
    static_assert(decltype(rank(acc_layout))::value == 3);

    // return make_layout(get<0>(get<0>(acc_layout)), get<1>(acc_layout), get<1>(get<0>(acc_layout)));
    return make_layout(get<0>(get<0>(acc_layout)), get<1>(acc_layout), make_layout(get<1>(get<0>(acc_layout)), get<2>(acc_layout)));
};

template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_fp8(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 16);
    // static_assert(decltype(size<2>(acc_layout))::value == 1);
    static_assert(decltype(rank(acc_layout))::value == 3);

    // return make_layout(get<0>(get<0>(acc_layout)), get<1>(acc_layout), get<1>(get<0>(acc_layout)));
    return make_layout(get<0>(get<0>(acc_layout)), get<1>(acc_layout), make_layout(get<1>(get<0>(acc_layout)), get<2>(acc_layout)));
};


// template<typename Layout>
// __forceinline__ __device__ auto convert_layout_acc_back(Layout acc_layout) {
//     using X = Underscore;
//     static_assert(decltype(size<0>(acc_layout))::value == 4);
//     static_assert(decltype(rank(acc_layout))::value == 3);
//     auto l = logical_divide(acc_layout, Shape<X, X, _1>{});

//     return make_layout(make_layout(get<0>(l), get<1>(get<2>(l))), get<1>(l), get<0>(get<2>(l)));
// };

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
// template<typename MMA_traits, typename Layout>
// __forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
//     using X = Underscore;
//     static_assert(decltype(size<0>(acc_layout))::value == 4);
//     static_assert(decltype(rank(acc_layout))::value == 3);
//     constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
//     static_assert(mma_shape_K == 8 || mma_shape_K == 16);
//     // if constexpr (mma_shape_K == 8) {
//     //     return acc_layout;
//     // } else {
//     //     auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
//     //     return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
//     // }
// };

template <class TiledMma,
        typename Engine0, typename Layout0,
        typename Engine1, typename Layout1
        >
__forceinline__ __device__ auto convert_layout_acc_Aregs(const TiledMma& tiled_mma, Tensor<Engine0, Layout0> const& tOrP,
    Tensor<Engine1, Layout1> const& sAcc)
{
    int tidx = threadIdx.x;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);

    auto smem_tiled_copy_ACC = make_tiled_copy_C(Copy_Atom<DefaultCopy, cute::half_t>{}, tiled_mma);
    auto smem_thr_copy_ACC = smem_tiled_copy_ACC.get_thread_slice(tidx);
    Tensor taccOr = smem_thr_copy_ACC.retile_S(tOrP);
    Tensor taccOs = smem_thr_copy_ACC.partition_D(sAcc); 
    // if (cute::thread0())
    // { taccOr
        // raw_ptr_16b(0x2000000000010) o ((_1,_4),_1,_4):((_0,_1),_0,_4)
    //     print("taccOr\n"); print(taccOr); print("\n");
    // }
    cute::copy(smem_tiled_copy_ACC, taccOr, taccOs); 
    
    // asm volatile("s_waitcnt lgkmcnt(0)\n\t");
    __syncthreads();



    auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<DefaultCopy, cute::half_t>{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(tidx);

    Tensor tSsACC = smem_thr_copy_A.partition_S(sAcc);
    Tensor tSrACC  = thr_mma.partition_fragment_A(sAcc);  
    Tensor tSrACC_copy_view = smem_thr_copy_A.retile_D(tSrACC);

    cute::copy(smem_tiled_copy_ACC, tSsACC, tSrACC_copy_view);

    // asm volatile("s_waitcnt lgkmcnt(0)\n\t");
    // __syncthreads(); // 取消这个sync,2024.06.13

    return tSrACC;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_dropout(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    // auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
    auto l = logical_divide(acc_layout, Shape<X, X, _1>{});  // (4, MMA_M, (1, MMA_N)))

    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));  // ((4, 1), 1, 2):((1, 0), 0, 4)
};

template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_back(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<X, X, _1>{});

    return make_layout(make_layout(get<0>(l), get<1>(get<2>(l))), get<1>(l), get<0>(get<2>(l)));
};

template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_back_fp8(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 8);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<X, X, _1>{});

    return make_layout(make_layout(get<0>(l), get<1>(get<2>(l))), get<1>(l), get<0>(get<2>(l)));
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    Tensor tensor_To_type = make_tensor<To_type>(layout(tensor));
    cutlass::Array<To_type, numel> *result_ptr = reinterpret_cast<cutlass::Array<To_type, numel> *>(tensor_To_type.data());
    if constexpr (std::is_same_v<To_type, cutlass::bfloat16_t>) {
#ifndef FLASH_ATTENTION_BF16_TYPE
#define FLASH_ATTENTION_BF16_TYPE 0
#endif
#if FLASH_ATTENTION_BF16_TYPE == 1
        cutlass::NumericArrayConverter<To_type, From_type, numel, cutlass::FloatRoundStyle::round_toward_zero> convert_op;
#elif FLASH_ATTENTION_BF16_TYPE == 2
        cutlass::NumericArrayConverter<To_type, From_type, numel, cutlass::FloatRoundStyle::round_to_nearest> convert_op;
#else
        cutlass::NumericArrayConverter<To_type, From_type, numel, cutlass::FloatRoundStyle::round_half_ulp_truncate> convert_op;
#endif
        *result_ptr = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    } else {
        cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
        *result_ptr = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    }
    return tensor_To_type;
    // cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // // HACK: this requires tensor to be "contiguous"
    // auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    // return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type_fp8(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    if constexpr (std::is_same_v<To_type, From_type>)
    {
        return tensor;
    }
    
    constexpr int numel = decltype(size(tensor))::value;
    Tensor tensor_To_type = make_tensor<To_type>(layout(tensor));
    cutlass::Array<To_type, numel> *result_ptr = reinterpret_cast<cutlass::Array<To_type, numel> *>(tensor_To_type.data());
    
    if constexpr (std::is_same_v<To_type, cutlass::bfloat16_t>) {
        cutlass::NumericArrayConverter<To_type, From_type, numel, cutlass::FloatRoundStyle::round_to_nearest> convert_op;
        *result_ptr = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
        } 
        else if constexpr (std::is_same_v<To_type, cutlass::float_e4m3_t>) {
        
            cutlass::NumericArrayConverter<To_type, From_type, numel,cutlass::FloatRoundStyle::round_to_nearest> convert_op;
            *result_ptr = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
        }
        else if constexpr (std::is_same_v<To_type, cutlass::float_e5m2_t>) {
        
            cutlass::NumericArrayConverter<To_type, From_type, numel,cutlass::FloatRoundStyle::round_to_nearest> convert_op;
            *result_ptr = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
        }
        else {
            cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
            *result_ptr = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
        }
        return tensor_To_type;
   
}


template <typename To_type, typename From_type>
__forceinline__ __device__ auto convert_type(From_type const &from) {
    if constexpr (std::is_same_v<To_type, cutlass::bfloat16_t>) {
        #ifndef FLASH_ATTENTION_BF16_TYPE
        #define FLASH_ATTENTION_BF16_TYPE 0
        #endif
        #if FLASH_ATTENTION_BF16_TYPE == 1
        cutlass::NumericConverter<To_type, From_type, cutlass::FloatRoundStyle::round_toward_zero> convert_;
        #elif FLASH_ATTENTION_BF16_TYPE == 2
        cutlass::NumericConverter<To_type, From_type, cutlass::FloatRoundStyle::round_to_nearest> convert_;
        #else
        cutlass::NumericConverter<To_type, From_type, cutlass::FloatRoundStyle::round_half_ulp_truncate> convert_;
        #endif
        return convert_(from);
    } else {
        cutlass::NumericConverter<To_type, From_type> convert_;
        return convert_(from);
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void relu_(Tensor<Engine, Layout> &tensor) {
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
    using value_t = typename Engine::value_type;
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_uint32 = recast<uint32_t>(tensor);
    #pragma unroll
    for (int i = 0; i < size(tensor_uint32); ++i) {
        tensor_uint32(i) = relu2<value_t>(tensor_uint32(i));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// On SM80 and above, we can fuse fp32 -> fp16/bf16 conversion and relu into 1 instruction
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type_relu(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    static_assert(std::is_same_v<To_type, cutlass::half_t> || std::is_same_v<To_type, cutlass::bfloat16_t>);
    static_assert(std::is_same_v<float, From_type>);
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_float2 = recast<float2>(tensor);
    Tensor out_uint32 = make_tensor<uint32_t>(tensor_float2.layout());
    #pragma unroll
    for (int i = 0; i < size(out_uint32); ++i) {
        out_uint32(i) = convert_relu2<To_type>(tensor_float2(i));
    }
    Tensor out = make_tensor(make_rmem_ptr<To_type>(out_uint32.data()), tensor.layout());
#else
    Tensor out = flash::convert_type<To_type>(tensor);
    flash::relu_(out);
#endif
    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    // asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_v(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        // if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
        //     #pragma unroll
        //     for (int k = 0; k < size<2>(S); ++k) {
        //         if (Is_even_K || predicate_K(k)) {
        //             cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
        //         } else if (Clear_OOB_K) {
        //             cute::clear(D(_, m, k));
        //         }
        //     }
        // } 
        //     else if (Clear_OOB_MN) {
        //     cute::clear(D(_, m, _));
        // }
        if (Is_even_K || predicate_K(m)) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_MN || get<0>(identity_MN(0, 0, k)) < max_MN) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } 
        
            else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }

        
    }
}

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_k_idx(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, int k_idx, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            if (Is_even_K || predicate_K(k_idx)) {
                cute::copy(tiled_copy, S(_, m, k_idx), D(_, m, k_idx));
            } else if (Clear_OOB_K) {
                cute::clear(D(_, m, k_idx));
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, k_idx));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S,
                                      Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                      Tensor<Engine3, Layout3> const &predicate_K,
                                      const int max_MN=0, const int min_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, max_MN = %d, min_MN = %d\n", blockIdx.y, max_MN, min_MN); }
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("Inner loop, blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}

#if 1

/*
for _64x32, use thread layout is 64x4, per thread get 8 elements, get 64x32 data, put data in lds with 32x64 
for _16x128, use thread layout is 16x16, per thread get 8 elements, get 16x128 data, put data in lds with 32x64 
for _16x192, use thread layout is 16x16, per thread get 12 elements, get 16x192 data, put data in lds with 48x64 
for _16x64_128, use thread layout is 16x16, per thread get 4 elements with offset 128, get 16x64 data, put data in lds with 16x64 
*/
enum MMA_LAYOUT{ _64x32 /* for gemm0 load K */,_64x64_LIT, _64x16 /* for gemm1 load V */, _16x128 /* for gemm1 load V */, _16x192 /* for dim 192 */, _16x64_128 /* for dim 64 */, _16x64_64 /*for load dim 64 V*/ ,
    _16x96 /*for load dim 96 V*/,
    _16x96_multi_ins /*for load dim 96 V*/,
    _16x256 /* for dim 256 read V */,
	_64x64, _32x128 /* for dim 192,128 fp8 read KV */,
    _64x32_load_v
};
template <bool Is_even_K=true, 
          bool Is_even_MN=true, 
          MMA_LAYOUT mma_layout = _64x32,
          int K_BUFF_SIZE = 0,
          bool Use_cache_swizzle = true,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
lds_direct_copy(
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst,
     int k_idx_, const int row_stride, 
     const int max_K = 0, const int max_MN=0)
{
    constexpr int warp_size = 64;
    int tidx = threadIdx.x;
    int warp_id = __builtin_amdgcn_readfirstlane(tidx / warp_size) % 4;
    int lane = tidx % warp_size;
    constexpr int element_size = 2;
    int k_idx = __builtin_amdgcn_readfirstlane(k_idx_);
    int k_slide = k_idx;
    if constexpr(K_BUFF_SIZE) {
        k_slide = (k_idx % K_BUFF_SIZE);
    }

    const int offset_s = 0;

    // global addr
    struct PtrWrapper {
        uint32_t former;
        uint32_t latter;
    };
    PtrWrapper glob_ptr;
    *(uint64_t*)&glob_ptr = reinterpret_cast<uint64_t>(src.data().get());
    if constexpr (Use_cache_swizzle) {
        glob_ptr.latter += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
    }

    uint32x4_t global_addr = {0};
    global_addr[0] = __builtin_amdgcn_readfirstlane(glob_ptr.former);
    global_addr[1] = __builtin_amdgcn_readfirstlane(glob_ptr.latter);
    global_addr[2] = 0x80000000;
    global_addr[3] = 0x00020000;

    if constexpr(mma_layout == _64x32) {
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 32*64;
        int row = tidx % 16;
        int col = lane / 16;
        int row_offset = row * 4 + warp_id;
        int col_offset = col * elements_per_thread + k_idx * 32;
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        __builtin_amdgcn_sched_barrier(0);
        #endif
    } else if constexpr(mma_layout == _64x16) {
        constexpr int elements_per_thread = 4;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*64;
        int row = tidx % 16;
        int col = lane / 16;
        int row_offset = row * 4 + warp_id;
        int col_offset = col * elements_per_thread + k_idx * 16;
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_MN && col_offset >= max_MN) offset_v = -1;
        // if (!Is_even_K && row_offset >= max_K) offset_v = -1;
        // const int offset_s = warp_id * row_stride * 2;
        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx2 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        __builtin_amdgcn_sched_barrier(0);
        #endif
    } else if constexpr(mma_layout == _16x128) {
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*128;
        int row = lane / 4;
        int col = tidx % 4;
        int row_offset = row + k_idx * 16;
        int col_offset = col * elements_per_thread + warp_id * 32;
        
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        __builtin_amdgcn_sched_barrier(0);
        #endif
    } else if constexpr(mma_layout == _16x192) {
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 48*64;
        int row = lane / 4;
        int col = tidx % 4;
        int row_offset = row + k_idx * 16;
        int col_offset = col * elements_per_thread + warp_id * 32;
        
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        __builtin_amdgcn_sched_barrier(0);
        #endif

        constexpr int elements_per_thread_tail = 4;
        constexpr int bytes_per_warp_tail = warp_size * elements_per_thread_tail * element_size;

        row = (tidx / 8) % 16;
        col = tidx % 8;
        row_offset = row + k_idx * 16;
        col_offset = col * elements_per_thread_tail + warp_id / 2 * 32 + /* pre offset */128 ;

        offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;
        
        ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + /* pre offset */64*32 * element_size + warp_id * bytes_per_warp_tail + k_slide * mma_k * element_size;
        
        // if (thread0()) printf("tid:%d offset_v:%d ldsAddrPerWave:%d\n", tidx, offset_v, ldsAddrPerWave);
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx2 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        __builtin_amdgcn_sched_barrier(0);
        #endif
    } else if constexpr(mma_layout == _16x64_128) {
        constexpr int elements_per_thread = 4;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*64;
        int row = (tidx / 8) % 16;
        int col = tidx % 8;
        int row_offset = row + k_idx * 16;
        int col_offset = col * elements_per_thread + warp_id / 2 * 32 + /* pre offset */128 ;

        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;
        
        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        // if (thread0()) printf("tid:%d offset_v:%d ldsAddrPerWave:%d\n", tidx, offset_v, ldsAddrPerWave);
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx2 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        __builtin_amdgcn_sched_barrier(0);
        #endif
    } else if constexpr(mma_layout == _16x64_64) {
        constexpr int elements_per_thread = 4;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*64;
        int row = (tidx / 8) % 16;
        int col = tidx % 8;
        int row_offset = row + k_idx * 16;
        int col_offset = col * elements_per_thread + warp_id / 2 * 32;

        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;
        
        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        // if (tidx < 64) printf("tid:%d offset_v:%d ldsAddrPerWave:%d\n", tidx, offset_v, ldsAddrPerWave);
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx2 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        __builtin_amdgcn_sched_barrier(0);
        #endif
    } else if constexpr(mma_layout == _16x96) {
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*96;
        int row = lane / 4;
        int col = tidx % 4;
        int row_offset = row + k_idx * 16;
        int col_offset = col * elements_per_thread + warp_id * 32;
        
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        if (warp_id < 3) {
            #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
            __builtin_amdgcn_sched_barrier(0);
            asm volatile(
                "s_mov_b32 m0, %1 \n\t"
                "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
                "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
            :);
            __builtin_amdgcn_sched_barrier(0);
            #endif
        }
    } else if constexpr(mma_layout == _16x96_multi_ins) {
        constexpr int elements_per_thread = 4;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*96;
        int row = lane / 8;
        int col = tidx % 8;
        int row_offset = row + (warp_id % 2) * 8 + k_idx * 16;
        int col_offset = col * elements_per_thread + warp_id / 2 * 32;
        
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx2 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        __builtin_amdgcn_sched_barrier(0);
        #endif
        
        constexpr int elements_per_thread_tail = 2;
        constexpr int bytes_per_warp_tail = warp_size * elements_per_thread_tail * element_size;
        
        row = lane / 16;
        col = tidx % 16;
        row_offset = row + warp_id * 4 + k_idx * 16;
        col_offset = col * elements_per_thread_tail + /* pre offset */64 ;

        offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;
        
        ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + /* pre offset */16*64 * element_size + warp_id * bytes_per_warp_tail + k_slide * mma_k * element_size;
        
        // if (thread0()) printf("tid:%d offset_v:%d ldsAddrPerWave:%d\n", tidx, offset_v, ldsAddrPerWave);
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dword %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        __builtin_amdgcn_sched_barrier(0);
        #endif
    }

}

template <
        int k_idx,
        int K_BUFF_SIZE = 0,
        bool Is_even_MN=true, 
        MMA_LAYOUT mma_layout = _64x32, 
        int n_idx = 0,
        bool Use_cache_swizzle = true,
        class SrcEngine, class SrcLayout,
        class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
lds_direct_copy_even_k_dim256(
         Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst,
     const int row_stride, 
     const int max_MN=0)
{
    constexpr int warp_size = 64;
    int tidx = threadIdx.x;
    int warp_id = __builtin_amdgcn_readfirstlane(tidx / warp_size);
    int lane = tidx % warp_size;
    constexpr int element_size = 2;
    struct PtrWrapper {
        uint32_t former;
        uint32_t latter;
    };
    PtrWrapper glob_ptr;
    *(uint64_t*)&glob_ptr = reinterpret_cast<uint64_t>(src.data().get());
    if constexpr (Use_cache_swizzle) {
        glob_ptr.latter += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
    }
    uint32x4_t global_addr = {0};
    global_addr[0] = __builtin_amdgcn_readfirstlane(glob_ptr.former);
    global_addr[1] = __builtin_amdgcn_readfirstlane(glob_ptr.latter);
    global_addr[2] = 0x80000000;
    global_addr[3] = 0x00020000;

    if constexpr(mma_layout == _64x32) {
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 32*64;
        int row = tidx % 16;
        int col = lane / 16;
        int row_offset = row * 4 + warp_id;
        int col_offset = col * elements_per_thread;
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + (k_idx % K_BUFF_SIZE) * mma_k * element_size;
        const int offset_s = k_idx * 32 * element_size;
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        __builtin_amdgcn_sched_barrier(0);
        #endif
    }  else if constexpr(mma_layout == _16x256) {
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*128;
        int row = lane / 4;
        int col = tidx % 4;
        int row_offset = row + k_idx * 16;
        int col_offset = col * elements_per_thread;
        
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_idx * mma_k * element_size;
        const int offset_s = (warp_id * 32 + n_idx * 128) * element_size;
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        __builtin_amdgcn_sched_barrier(0);
        #endif
    } 
}

template <
         int k_idx,
          bool Is_even_MN=true, 
          MMA_LAYOUT mma_layout = _64x32, 
          bool Use_cache_swizzle = true,
          int n_idx = 0,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
lds_direct_copy_even_k(
         Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst,
     const int row_stride, 
     const int max_MN=0)
{
    constexpr int warp_size = 64;
    int tidx = threadIdx.x;
    int warp_id = __builtin_amdgcn_readfirstlane(tidx / warp_size);
    int lane = tidx % warp_size;
    constexpr int element_size = 2;
    struct PtrWrapper {
        uint32_t former;
        uint32_t latter;
    };
    PtrWrapper glob_ptr;
    *(uint64_t*)&glob_ptr = reinterpret_cast<uint64_t>(src.data().get());
    if constexpr (Use_cache_swizzle) {
        glob_ptr.latter += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
    }
    uint32x4_t global_addr = {0};
    global_addr[0] = __builtin_amdgcn_readfirstlane(glob_ptr.former);
    global_addr[1] = __builtin_amdgcn_readfirstlane(glob_ptr.latter);
    global_addr[2] = 0x80000000;
    global_addr[3] = 0x00020000;

    if constexpr(mma_layout == _64x32) {
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 32*64;
        int row = tidx % 16;
        int col = lane / 16;
        int row_offset = row * 4 + warp_id + n_idx * 64;
        int col_offset = col * elements_per_thread;
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + (k_idx + n_idx) * mma_k * element_size;
        const int offset_s = k_idx * 32 * element_size;
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        __builtin_amdgcn_sched_barrier(0);
        #endif
    } else if constexpr(mma_layout == _16x64_64) {
        constexpr int elements_per_thread = 4;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*64;
        int row = (tidx / 8) % 16;
        int col = tidx % 8;
        int row_offset = row + k_idx * 16;
        int col_offset = col * elements_per_thread;

        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;
        
        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_idx * mma_k * element_size;
        
        // if (tidx < 64) printf("tid:%d offset_v:%d ldsAddrPerWave:%d\n", tidx, offset_v, ldsAddrPerWave);
        const int offset_s = warp_id / 2 * 32 * element_size;
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx2 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        __builtin_amdgcn_sched_barrier(0);
        #endif
    } else if constexpr(mma_layout == _16x128) {
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*128;
        int row = lane / 4;
        int col = tidx % 4;
        int row_offset = row + k_idx * 16;
        int col_offset = col * elements_per_thread;
        
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        // if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_idx * mma_k * element_size;
        const int offset_s = warp_id * 32 * 2;
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        __builtin_amdgcn_sched_barrier(0);
        #endif
    } else if constexpr (mma_layout == _64x32_load_v) {
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 32*64;
        int row = (lane / 4);
        int col = lane % 4;
        int row_offset = row + warp_id * 16 + k_idx * 64;
        int col_offset = col * elements_per_thread;
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_idx * mma_k * element_size;
        const int offset_s = 0;
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        __builtin_amdgcn_sched_barrier(0);
        #endif
    }
}
#define fp8 unsigned char
__forceinline__ __device__ float fp8e5m2_to_fp32(const fp8& input) {
  union uf16{
    uint16_t as_bits;
    _Float16 as_value;
  } ;
  union uf32 {
    uint32_t as_bits;
    float as_value;
  };
  uf16 u16;
  uf32 u32;
  u16.as_bits = (uint16_t)input << 8;
  u32.as_value = (float)u16.as_value;
  return u32.as_value;
}

template <typename Element, bool Is_even_K=true, 
          bool Is_even_MN=true, 
          MMA_LAYOUT mma_layout = _64x32,
          int K_BUFF_SIZE = 0,
          bool Use_cache_swizzle = true,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
lds_direct_copy_kv_fp8(float scale,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst,
     int k_idx_, const int row_stride, 
     const int max_K = 0, const int max_MN=0)
{
    constexpr int warp_size = 64;
    int tidx = threadIdx.x;
    int warp_id = __builtin_amdgcn_readfirstlane(tidx / warp_size) % 4;
    int lane = tidx % warp_size;
    constexpr int element_size = 2;
    int k_idx = __builtin_amdgcn_readfirstlane(k_idx_);
    int k_slide = k_idx;
    if constexpr(K_BUFF_SIZE) {
        k_slide = (k_idx % K_BUFF_SIZE);
    }

    if constexpr(mma_layout == _64x32) {
        constexpr int elements_per_thread = 8;
        int mma_k = 32*64;
        int row = tidx % 16;
        int col = lane / 16;
        int row_offset = row * 4 + warp_id;
        int col_offset = col * elements_per_thread + k_idx * 32;
        
        Element rst[8];
        cutlass::NumericConverter<Element, float, cutlass::FloatRoundStyle::round_toward_zero> convert_;
        
        const fp8* src_ptr = reinterpret_cast<const fp8*>(src.data().get());
        
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((Is_even_K || col_offset < max_K) && 
                (Is_even_MN || row_offset < max_MN)) {
                int offset = row_offset * row_stride + col_offset + i;
                float f = fp8e5m2_to_fp32(src_ptr[offset]) * scale;
                rst[i] = convert_(f);
            } else {
                rst[i] = Element(0);
            }
        }
        
        int element_offset = warp_id * warp_size * elements_per_thread + k_slide * mma_k + lane * elements_per_thread;
        Element* lds_ptr = dst.data().get() + element_offset;
        *reinterpret_cast<uint4*>(lds_ptr) = *reinterpret_cast<uint4*>(rst);

    } else if constexpr(mma_layout == _64x16) {
        constexpr int elements_per_thread = 4;
        int mma_k = 16*64;
        int row = tidx % 16;
        int col = lane / 16;
        int row_offset = row * 4 + warp_id;
        int col_offset = col * elements_per_thread + k_idx * 16;
        
        Element rst[4];
        cutlass::NumericConverter<Element, float, cutlass::FloatRoundStyle::round_toward_zero> convert_;
        
        const fp8* src_ptr = reinterpret_cast<const fp8*>(src.data().get());
        
        bool valid = (Is_even_K || row_offset < max_K) &&
                    (Is_even_MN || col_offset < max_MN);  // 不检查 col_offset+i
                    
        for (int i = 0; i < 4; ++i) {
            if (valid) {
                int offset = row_offset * row_stride + col_offset + i;
                float f = fp8e5m2_to_fp32(src_ptr[offset]) * scale;
                rst[i] = convert_(f);
            } else {
                rst[i] = Element(0);
            }
        }
        
        int element_offset = warp_id * warp_size * elements_per_thread + k_slide * mma_k + lane * elements_per_thread;
        Element* lds_ptr = dst.data().get() + element_offset;
        *reinterpret_cast<uint2*>(lds_ptr) = *reinterpret_cast<uint2*>(rst);
    } 
}


template <bool Is_even_K=true, 
          bool Is_even_MN=true, 
          MMA_LAYOUT mma_layout = _64x32,
          bool Use_cache_swizzle = true,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
lds_direct_copy(int k_slide, 
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst,
     int k_idx_, const int row_stride, 
     const int max_K = 0, const int max_MN=0)
{
    constexpr int warp_size = 64;
    int tidx = threadIdx.x;
    int warp_id = __builtin_amdgcn_readfirstlane(tidx / warp_size) % 4;
    int lane = tidx % warp_size;
    constexpr int element_size = 2;
    int k_idx = __builtin_amdgcn_readfirstlane(k_idx_);

    const int offset_s = 0;

    // global addr
    struct PtrWrapper {
        uint32_t former;
        uint32_t latter;
    };
    PtrWrapper glob_ptr;
    *(uint64_t*)&glob_ptr = reinterpret_cast<uint64_t>(src.data().get());
    if constexpr (Use_cache_swizzle) {
        glob_ptr.latter += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
    }

    uint32x4_t global_addr = {0};
    global_addr[0] = __builtin_amdgcn_readfirstlane(glob_ptr.former);
    global_addr[1] = __builtin_amdgcn_readfirstlane(glob_ptr.latter);
    global_addr[2] = 0x80000000;
    global_addr[3] = 0x00020000;

    if constexpr(mma_layout == _64x32) {
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 32*64;
        int row = tidx % 16;
        int col = lane / 16;
        int row_offset = row * 4 + warp_id;
        int col_offset = col * elements_per_thread + k_idx * 32;
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        #endif
    } else if constexpr(mma_layout == _16x128) {
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*128;
        int row = lane / 4;
        int col = tidx % 4;
        int row_offset = row + k_idx * 16;
        int col_offset = col * elements_per_thread + warp_id * 32;
        
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        #endif
    } else if constexpr(mma_layout == _16x192) {
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 48*64;
        int row = lane / 4;
        int col = tidx % 4;
        int row_offset = row + k_idx * 16;
        int col_offset = col * elements_per_thread + warp_id * 32;
        
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        #endif

        constexpr int elements_per_thread_tail = 4;
        constexpr int bytes_per_warp_tail = warp_size * elements_per_thread_tail * element_size;

        row = (tidx / 8) % 16;
        col = tidx % 8;
        row_offset = row + k_idx * 16;
        col_offset = col * elements_per_thread_tail + warp_id / 2 * 32 + /* pre offset */128 ;

        offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;
        
        ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + /* pre offset */64*32 * element_size + warp_id * bytes_per_warp_tail + k_slide * mma_k * element_size;
        
        // if (thread0()) printf("tid:%d offset_v:%d ldsAddrPerWave:%d\n", tidx, offset_v, ldsAddrPerWave);
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx2 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        #endif
    } else if constexpr(mma_layout == _16x64_128) {
        constexpr int elements_per_thread = 4;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*64;
        int row = (tidx / 8) % 16;
        int col = tidx % 8;
        int row_offset = row + k_idx * 16;
        int col_offset = col * elements_per_thread + warp_id / 2 * 32 + /* pre offset */128 ;

        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;
        
        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        // if (thread0()) printf("tid:%d offset_v:%d ldsAddrPerWave:%d\n", tidx, offset_v, ldsAddrPerWave);
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx2 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        #endif
    } else if constexpr(mma_layout == _16x64_64) {
        constexpr int elements_per_thread = 4;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*64;
        int row = (tidx / 8) % 16;
        int col = tidx % 8;
        int row_offset = row + k_idx * 16;
        int col_offset = col * elements_per_thread + warp_id / 2 * 32;

        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;
        
        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        // if (tidx < 64) printf("tid:%d offset_v:%d ldsAddrPerWave:%d\n", tidx, offset_v, ldsAddrPerWave);
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx2 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        #endif
    } else if constexpr(mma_layout == _16x96) {
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*96;
        int row = lane / 4;
        int col = tidx % 4;
        int row_offset = row + k_idx * 16;
        int col_offset = col * elements_per_thread + warp_id * 32;
        
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        if (warp_id < 3) {
            #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
            asm volatile(
                "s_mov_b32 m0, %1 \n\t"
                "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
                "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
            :);
            #endif
        }
    } else if constexpr(mma_layout == _16x96_multi_ins) {
        constexpr int elements_per_thread = 4;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*96;
        int row = lane / 8;
        int col = tidx % 8;
        int row_offset = row + (warp_id % 2) * 8 + k_idx * 16;
        int col_offset = col * elements_per_thread + warp_id / 2 * 32;
        
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx2 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        #endif
        
        constexpr int elements_per_thread_tail = 2;
        constexpr int bytes_per_warp_tail = warp_size * elements_per_thread_tail * element_size;
        
        row = lane / 16;
        col = tidx % 16;
        row_offset = row + warp_id * 4 + k_idx * 16;
        col_offset = col * elements_per_thread_tail + /* pre offset */64 ;

        offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;
        
        ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + /* pre offset */16*64 * element_size + warp_id * bytes_per_warp_tail + k_slide * mma_k * element_size;
        
        // if (thread0()) printf("tid:%d offset_v:%d ldsAddrPerWave:%d\n", tidx, offset_v, ldsAddrPerWave);
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dword %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        #endif
    }

}

template <bool Is_even_K=true, 
          bool Is_even_MN=true, 
          MMA_LAYOUT mma_layout = _16x256,
          bool Use_cache_swizzle = true,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
lds_direct_copy(int n_idx, int k_slide, 
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst,
     int k_idx_, const int row_stride, 
     const int max_K = 0, const int max_MN=0)
{
    constexpr int warp_size = 64;
    const int tidx = threadIdx.x;
    const int warp_id = __builtin_amdgcn_readfirstlane(tidx / warp_size) % 4;
    const int lane = tidx % warp_size;
    constexpr int element_size = 2;
    int k_idx = __builtin_amdgcn_readfirstlane(k_idx_);

    const int offset_s = 0;

    // global addr
    struct PtrWrapper {
        uint32_t former;
        uint32_t latter;
    };
    PtrWrapper glob_ptr;
    *(uint64_t*)&glob_ptr = reinterpret_cast<uint64_t>(src.data().get());
    if constexpr (Use_cache_swizzle) {
        glob_ptr.latter += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
    }

    uint32x4_t global_addr = {0};
    global_addr[0] = __builtin_amdgcn_readfirstlane(glob_ptr.former);
    global_addr[1] = __builtin_amdgcn_readfirstlane(glob_ptr.latter);
    global_addr[2] = 0x80000000;
    global_addr[3] = 0x00020000;

    if constexpr(mma_layout == _16x256) {
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*128;
        int row = lane / 4;
        int col = tidx % 4;
        int row_offset = row + k_idx * 16;
        int col_offset = col * elements_per_thread + warp_id * 32 + n_idx * 128;
        
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        #endif
    } 
}

template <bool Is_even_K=true, 
          bool Is_even_MN=true, 
          MMA_LAYOUT mma_layout = _64x64,
          int K_BUFF_SIZE = 0,
          bool Use_cache_swizzle = true,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
lds_direct_copy_fp8(
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst,
     int k_idx_, const int row_stride, 
     const int max_K = 0, const int max_MN=0)
{
    constexpr int warp_size = 64;
    int tidx = threadIdx.x;
    int warp_id = __builtin_amdgcn_readfirstlane(tidx / warp_size);
    int lane = tidx % warp_size;
    constexpr int element_size = 1;
    int k_idx = __builtin_amdgcn_readfirstlane(k_idx_);
    int k_slide = k_idx;
    if constexpr(K_BUFF_SIZE) {
        k_slide = (k_idx % K_BUFF_SIZE);
    }

    const int offset_s = 0;

    // global addr
    struct PtrWrapper {
        uint32_t former;
        uint32_t latter;
    };
    PtrWrapper glob_ptr;
    *(uint64_t*)&glob_ptr = reinterpret_cast<uint64_t>(src.data().get());
    if constexpr (Use_cache_swizzle) {
        glob_ptr.latter += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
    }

    uint32x4_t global_addr = {0};
    global_addr[0] = __builtin_amdgcn_readfirstlane(glob_ptr.former);
    global_addr[1] = __builtin_amdgcn_readfirstlane(glob_ptr.latter);
    global_addr[2] = 0x80000000;
    global_addr[3] = 0x00020000;

    if constexpr(mma_layout == _64x64) {
        // constexpr int elements_per_thread = 16;
        // constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        // int mma_k = 64*64;
        // int row = tidx % 16;
        // int col = lane / 16;
        // int row_offset = row * 4 + warp_id;
        // int col_offset = col * elements_per_thread + k_idx * 64;
        // int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        // if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        // if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        constexpr int elements_per_thread = 16;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 64*64;
        int row = tidx % 16;
        int col = lane / 16;
        int row_offset = row * 2 + warp_id +(warp_id/2)*30;
        int col_offset = col * elements_per_thread + k_idx * 64;
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if defined(__gfx938__)||defined(__gfx92a__)
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        #endif
    }else if constexpr(mma_layout == _64x64_LIT) {

        constexpr int elements_per_thread = 16;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 64*64;
        int row = tidx % 16;
        int col = lane / 16;
        int row_offset = row * 2 + warp_id +(warp_id/2)*30;
        int col_offset = col * elements_per_thread + k_idx * 64;
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if defined(__gfx938__)||defined(__gfx92a__)
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        #endif
    }else if constexpr(mma_layout == _64x32) {
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 32*64;
        int row = tidx % 16;
        int col = lane / 16;
        int row_offset = row * 4 + warp_id;
        int col_offset = col * elements_per_thread + k_idx * 32;
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_MN && col_offset >= max_MN) offset_v = -1;
        if (!Is_even_K && row_offset >= max_K) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if defined(__gfx938__)||defined(__gfx92a__)
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx2 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);

        #endif
    } else if constexpr(mma_layout == _32x128) {
        constexpr int elements_per_thread = 16;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 32*128;
        int row = lane / 2;
        int col = tidx % 2;
        int row_offset = row + k_idx * 32;
        int col_offset = col * elements_per_thread + warp_id * 32;
        
        int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if defined(__gfx938__)||defined(__gfx92a__)
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        #endif
    } 

}

template <bool Is_even_K=true, 
          bool Is_even_MN=true, 
          MMA_LAYOUT mma_layout = _64x32,
          int K_BUFF_SIZE = 0,
          bool Use_cache_swizzle = true,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
lds_direct_copy_for_vertical_sparse(
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst,
     int k_idx_, const int row_stride, int row_offset,
     const int max_K = 0, const int max_MN=0)
{
    constexpr int warp_size = 64;
    int tidx = threadIdx.x;
    int warp_id = __builtin_amdgcn_readfirstlane(tidx / warp_size);
    int lane = tidx % warp_size;
    constexpr int element_size = 2;
    int k_idx = __builtin_amdgcn_readfirstlane(k_idx_);
    int k_slide = k_idx;
    if constexpr(K_BUFF_SIZE) {
        k_slide = (k_idx % K_BUFF_SIZE);
    }

    const int offset_s = 0;



    if constexpr(mma_layout == _64x32) {
        // global addr
        struct PtrWrapper {
            uint32_t former;
            uint32_t latter;
        };
        PtrWrapper glob_ptr;
        *(uint64_t*)&glob_ptr = reinterpret_cast<uint64_t>(src.data().get());
        {
            // 设置stride值为16, 因为一个线程读取8个元素, 16字节
            glob_ptr.latter += ((row_stride * 2 ) << 16); // 62 bit: cache swizzle;  48~61: Stride
        }

        uint32x4_t global_addr = {0};
        global_addr[0] = __builtin_amdgcn_readfirstlane(glob_ptr.former);
        global_addr[1] = __builtin_amdgcn_readfirstlane(glob_ptr.latter);
        global_addr[2] = __builtin_amdgcn_readfirstlane(max_MN); // number records  95:64
        global_addr[3] = 0x00020000;
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 32*64;
        int row = tidx % 16;
        int col = lane / 16;
        // int row_offset = cols_ptr[row * 4 + warp_id];
        int col_offset = col * elements_per_thread + k_idx * 32;
        typedef uint32_t uint32x2_t __attribute__((ext_vector_type(2)));

        uint32x2_t offset_v = {0};
        offset_v[0] = row_offset;
        offset_v[1] = col_offset * 2;
        // int offset_v = (row_offset * row_stride + col_offset) * element_size; // bytes
        // if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        // if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;

        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,idxen offen  offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        #endif
    } else if constexpr(mma_layout == _16x128) {  
        // global addr
        struct PtrWrapper {
            uint32_t former;
            uint32_t latter;
        };
        PtrWrapper glob_ptr;
        *(uint64_t*)&glob_ptr = reinterpret_cast<uint64_t>(src.data().get());
        // if constexpr (Use_cache_swizzle) 
        {
            // 设置stride值为16, 因为一个线程读取8个元素, 16字节
            glob_ptr.latter += ((row_stride * 2 ) << 16); // 62 bit: cache swizzle;  48~61: Stride
        }

        uint32x4_t global_addr = {0};
        global_addr[0] = __builtin_amdgcn_readfirstlane(glob_ptr.former);
        global_addr[1] = __builtin_amdgcn_readfirstlane(glob_ptr.latter);
        global_addr[2] = __builtin_amdgcn_readfirstlane(max_MN); // number records  95:64
        global_addr[3] = 0x00020000;
        constexpr int elements_per_thread = 8;
        constexpr int bytes_per_warp = warp_size * elements_per_thread * element_size;
        int mma_k = 16*128;
        int row = lane / 4;
        int col = tidx % 4;
        // int row_offset = cols_ptr[row + k_idx * 16];
        int col_offset = col * elements_per_thread + warp_id * 32;
        
        // int64_t offset_v = (row_offset + col_offset) / 8; // bytes
        typedef uint32_t uint32x2_t __attribute__((ext_vector_type(2)));

        uint32x2_t offset_v = {0};
        offset_v[0] = row_offset ;
        offset_v[1] = col_offset * 2;
        // if (!Is_even_K && col_offset >= max_K) offset_v = -1;
        // if (!Is_even_MN && row_offset >= max_MN) offset_v = -1;
        // int index_v = offset_v;
        int ldsAddrPerWave = reinterpret_cast<size_t>(dst.data().get()) + warp_id * bytes_per_warp + k_slide * mma_k * element_size;
        
        #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
        asm volatile(
            "s_mov_b32 m0, %1 \n\t"
            "buffer_load_dwordx4 %0, %2, %3 ,idxen  offen offset:0, lds \n" ::"v"(offset_v),
            "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
        :);
        #endif
    }
}

template<MMA_LAYOUT mma_layout = _64x32, int divide = 1, typename Layout>
__forceinline__ __device__ auto convert_layout_B_rowcol(Layout B_layout) {
  static_assert(decltype(rank(B_layout))::value == 3);
  if constexpr(mma_layout == _64x32||mma_layout == _64x64) {
    auto layout = make_layout(get<1>(B_layout), get<0>(B_layout), get<2>(B_layout));  
    auto l = logical_divide(layout, Shape<Int<divide>>{});

    // if (thread0()) {
    //   printf("l:"); print(l); printf("\n");
    // }
    return make_layout(get<1>(l), get<1>(get<0>(l)), get<0>(get<0>(l)));
  } else if constexpr(mma_layout == _16x128) {
    return make_layout(get<0>(B_layout), get<2>(B_layout), get<1>(B_layout));
  } else if constexpr(mma_layout == _16x192) {
    // disgusting!!! hard code 
    auto layout = make_layout(Shape<Shape<_8, _1>, _6, _4>{}, Stride<Stride<_1, _0>, _512, Int<3072>>{}); // ((_8,_1),_6,_4):((_1,_0),_3072,_512)
    // if (thread0()) {
    //   printf("layout:"); print(layout); printf("\n");
    // }
    return layout;
  } else if constexpr(mma_layout == _16x64_128) {
    // disgusting!!! hard code 
    return make_layout(Shape<Shape<_8, _1>, _2, _4>{}, Stride<Stride<_1, _0>, _512, Int<1024>>{});
  } else if constexpr(mma_layout == _16x64_64) {
    // disgusting!!! hard code 
    return make_layout(Shape<Shape<_8, _1>, _2, _4>{}, Stride<Stride<_1, _0>, _512, Int<1024>>{});
  }
};

template<MMA_LAYOUT mma_layout = _64x64, int divide = 1, typename Layout>
__forceinline__ __device__ auto convert_layout_B_rowcol_fp8(Layout B_layout) {
  static_assert(decltype(rank(B_layout))::value == 3);
  if constexpr(mma_layout == _64x64) {
    auto layout = make_layout(get<1>(B_layout), get<0>(B_layout), get<2>(B_layout));  
    auto l = logical_divide(layout, Shape<Int<divide>>{});

    return make_layout(get<1>(l), get<1>(get<0>(l)), get<0>(get<0>(l)));
  } else if constexpr(mma_layout == _32x128) {
   
    auto layout = make_layout(Shape<Shape<_16, _1>, _4, _2>{}, Stride<Stride<_1, _0>, _1024, Int<4096>>{});
    return layout;
  } 
};


template<MMA_LAYOUT mma_layout = _64x32, int divide = 1, typename Layout>
__forceinline__ __device__ auto convert_layout_B_rowcol_(Layout B_layout) {
  static_assert(decltype(rank(B_layout))::value == 3);
  if constexpr(mma_layout == _64x32) {
    auto layout = make_layout(get<2>(B_layout), get<0>(B_layout), get<1>(B_layout));  
    auto l = logical_divide(layout, Shape<Int<divide>>{});
    return make_layout(get<1>(l), get<0>(get<0>(l)), get<1>(get<0>(l)));
  } else if constexpr(mma_layout == _16x128 || mma_layout == _16x192 || mma_layout == _16x64_64 || mma_layout == _16x96) {
    auto layout = make_layout(get<1>(B_layout), get<0>(B_layout), get<2>(B_layout));  
    auto l = logical_divide(layout, Shape<Int<divide>>{});
    return make_layout(get<1>(l), get<0>(get<0>(l)), get<1>(get<0>(l)));
  }
};

#endif
////////////////////////////////////////////////////////////////////////////////////////////////////

// resolves offset of a slice of a paged kv copy from gmem.
// assumes that the tensor has already been positioned at the correct head.
template <typename Kernel_traits>
__forceinline__ __device__
int64_t resolve_thread_kv_page_slice_offset(const int tidx, const int n_block_max, const int page_block_size, 
                            const int* block_table, const int page_stride, const int row_stride) {
    constexpr int kGmemThreadsPerRow = Kernel_traits::kGmemThreadsPerRow;
    constexpr int kGmemRowsPerThread = Kernel_traits::kGmemRowsPerThread;
    constexpr int kGmemElemsPerLoad = Kernel_traits::kGmemElemsPerLoad;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    
    const int64_t col_offset = tidx % kGmemThreadsPerRow * kGmemElemsPerLoad;
    const int64_t block_row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread;
    const int64_t global_row_offset = block_row_offset + (n_block_max - 1) * kBlockN;
    const int64_t page_offset = global_row_offset % page_block_size;
    const int64_t virtual_page_idx = global_row_offset / page_block_size;

    return ((int64_t) block_table[virtual_page_idx]) * ((int64_t) page_stride)
        + page_offset * ((int64_t) row_stride)
        + col_offset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Layout reshape function. Given a layout with modes ((v1, v2), m, k), returns (v1, v2, k),         
// where v2 may be a tuple itself, in the case of swizzled smem-backed thread tiles. This ensures
// that paged and non-paged copies result in equivalently shaped, if not necessarily strided, tensors.
template <class Shape, class Stride>
__forceinline__ __device__
auto reshape_thread_tile(Layout<Shape, Stride> l) {
    return make_layout(append(get<0>(l.shape()), get<2>(l.shape())),
                        append(get<0>(l.stride()), get<2>(l.stride())));
}

// reshapes and flattens the thread tile layout. A separate function is needed for the case where
// one of the modes of l is a layout itself and must be flattened, as opposed to keeping it intact
// for the case of swizzled layouts
template <class Shape, class Stride>
__forceinline__ __device__
auto reshape_flatten_thread_tile(Layout<Shape, Stride> l) {
    auto mode_0 = filter(flatten(get<0>(l)));
    return make_layout(append(mode_0.shape(), get<2>(l.shape())),
                        append(mode_0.stride(), get<2>(l.stride())));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_softcap(Tensor<Engine, Layout> &tensor, const float softcap){
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = cutlass::fast_tanh(tensor(i) * softcap);
    }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void calculate_dtanh(Tensor<Engine0, Layout0> &src_tensor, Tensor<Engine1, Layout1> &dst_tensor, const float softcap){
    #pragma unroll
    for (int i = 0; i < size(src_tensor); ++i) {
        dst_tensor(i) = (1.f - (src_tensor(i) * src_tensor(i))) * softcap;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Use_mask, typename TensorSrc, typename TensorAcc>
__forceinline__ __device__ void apply_atten_mask(TensorSrc const& src, TensorAcc& accum, float value = -INFINITY) {
    if constexpr(Use_mask) {
        CUTE_STATIC_ASSERT_V(size(src) == size(accum));
        #pragma unroll
        for (int i = 0; i < size(src); i++) {
            accum(i) = src(i) ? accum(i) : value;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    原来的 exp2f 对于极小数有特殊处理, 对于小于 -126 的输入 x , exp2f 计算方式是 2^(x + 64) * 2^{-64}
    但是对于深度学习来说, 2^-126 的数字其实没那么重要了, 因此只需要保留 v_exp_f32 直接暴力计算即可
*/
extern __device__ __attribute__((const)) float __llvm_exp2_f32(float) __asm("llvm.exp2.f32");

__forceinline__ __device__ float custom_exp2f(float x) {
    #if 0
    return __exp2f(x);
    #elif 0
    return __llvm_exp2_f32(x);
    #else
    return __builtin_amdgcn_exp2f(x);
    #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
