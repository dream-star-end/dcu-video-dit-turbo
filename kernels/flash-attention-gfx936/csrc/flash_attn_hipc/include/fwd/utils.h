#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include "numeric_types.h"
#include "wait.h"
#include "intrinsic.h"

namespace flash {

__forceinline__ __device__ void raise_priority(const int priority_level=2) {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_setprio %0":: "n"(priority_level));
    __builtin_amdgcn_sched_barrier(0);
}


__forceinline__ __device__ void lower_priority() {
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_setprio 0");
    __builtin_amdgcn_sched_barrier(0);
}

inline __device__ constexpr int ceil_div(int const& a, int const& b) {
    return (a + b - 1) / b;
}

inline __device__ constexpr int floor_div(int const& a, int const& b) {
    return a / b;
}

template<class T, class AccumType>
inline __device__ vec4_fp32 mmac(const vec4_Element<T> &v1, const vec4_Element<T> &v2, const vec4_fp32 &v3)
{
#if defined(__gfx916__) || defined(__gfx926__)
    return {0, 0, 0, 0};
#else
    return __builtin_hcu_mmac_f32_16x16x16_f16(v1, v2, v3);
#endif
}

template<>
inline __device__ vec4_fp32 mmac<half_t, float>(const vec4_fp16 &v1, const vec4_fp16 &v2, const vec4_fp32 &v3)
{
#if defined(__gfx916__) || defined(__gfx926__)
    return {0, 0, 0, 0};
#else
    return __builtin_hcu_mmac_f32_16x16x16_f16(v1, v2, v3);
#endif
}

template<>
inline __device__ vec4_fp32 mmac<bhalf_t, float>(const vec4_bf16 &v1, const vec4_bf16 &v2, const vec4_fp32 &v3)
{
#if defined(__gfx916__) || defined(__gfx926__)
    return {0, 0, 0, 0};
#else
    return __builtin_hcu_mmac_f32_16x16x16_bf16(v1, v2, v3);
#endif
}
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ inline T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ inline float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ inline T operator()(T const & x, T const & y) {
      T res = (x + y); 
      return res;
     }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
__forceinline__ __device__ T __shfl_xor_tmp(T x, const int lane_mask) {
    int lane_id = threadIdx.x & 63;
    int index   = (lane_id ^ lane_mask) << 2;
    int res = __builtin_amdgcn_ds_bpermute(index, *(int*)&x); // attention, __builtin only support int
    return *(T*)&res;
}


template<typename T>
__forceinline__ __device__ T __shfl_swap16(T x) {
    int result = __builtin_amdgcn_ds_swizzle(*(int*)&x, 0x401F);
    return *(T*)&result;
}


template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 64);
    template<typename Operator>
    static __device__ inline union_vec2_fp32 run(union_vec2_fp32 x, Operator &op) {
        union_vec2_fp32 res;
        if constexpr (std::is_same<Operator, SumOp<float> >::value) {
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
            res.f32[0] = __shfl_xor_tmp(x.f32[0], 32);
            res.f32[1] = __shfl_xor_tmp(x.f32[1], 32);
            x.u64 = __builtin_hcu_pk_add_f32(x.u64, res.u64);
            res.f32[0] = __shfl_swap16(x.f32[0]); // __shfl_xor_tmp(x.f32[0], 16);
            res.f32[1] = __shfl_swap16(x.f32[1]); // __shfl_xor_tmp(x.f32[1], 16);
            res.u64 = __builtin_hcu_pk_add_f32(res.u64, x.u64);
        #else
            x.f32[0] = x.f32[0] + __shfl_xor_tmp(x.f32[0], 32);
            x.f32[1] = x.f32[1] + __shfl_xor_tmp(x.f32[1], 32);
            res.f32[0] = x.f32[0] + __shfl_xor_tmp(x.f32[0], 16);
            res.f32[1] = x.f32[1] + __shfl_xor_tmp(x.f32[1], 16);
        #endif
        }
        else if constexpr (std::is_same<Operator, MaxOp<float> >::value) {
            x.f32[0] = op(x.f32[0], __shfl_xor_tmp(x.f32[0], 32));
            x.f32[1] = op(x.f32[1], __shfl_xor_tmp(x.f32[1], 32));
            res.f32[0] = op(x.f32[0], __shfl_swap16(x.f32[0]));
            res.f32[1] = op(x.f32[1], __shfl_swap16(x.f32[1]));
        }
        return res;
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////



template<const int kHeadDim, typename T, bool Do_CacheSwizzle=true>
__device__ __forceinline__ vec4_uint prepare_for_buffer_load(T* ptr) {
    vec4_uint res;
    *(uint64_t*)&res = reinterpret_cast<uint64_t>(ptr);
    if constexpr (Do_CacheSwizzle) {
        if constexpr (kHeadDim == 128) {
            res[1] += 0x41000000; // 62 bit: cache swizzle;  48~61: Stride
        } else if constexpr (kHeadDim == 192) {
            res[1] += 0x41800000; // stride 192Bytes and change tagram
        } else if constexpr (kHeadDim == 64) {
            res[1] += 0x40800000; // stride 128Bytes and change tagram
        }
    }
    res[2] = 0x80000000;
    res[3] = 0x00020000;
    return res;
}


// for matrix_load
template<const int kHeadDim, typename T>
__device__ __forceinline__ vec4_uint prepare_for_matrix_load(T* ptr) {
    vec4_uint res;
    *(uint64_t*)&res = reinterpret_cast<uint64_t>(ptr);
    res[2] = 0x0;
    res[3] = 0x0;
    return res;
}


template<int K_LOOP_COUNT, int M_WARP_COUNT, int K_WARP_COUNT, int M_MMAC_COUNT, typename ElementAccum>
__forceinline__ __device__ void attention_initialize(
    vec2_Accum<ElementAccum> scores_max[M_WARP_COUNT],
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT],
    vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4]
) {
    #pragma unroll
    for (int i = 0; i < M_WARP_COUNT; ++i) {
        scores_max[i].f32[0] = -INFINITY;
        scores_max[i].f32[1] = -INFINITY;
        scores_sum[i].f32[0] = 0;
        scores_sum[i].f32[1] = 0;
    }
    uint64_t pk_zero = 0;
    #pragma unroll
    for (int i = 0; i < K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT; ++i) {
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            #pragma unroll
            for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                #if defined(H3_STANDARD_FWD_ONLY)
                acc_o[i][min_tile_n * 2 + min_tile_m].f32[0] = 0;
                acc_o[i][min_tile_n * 2 + min_tile_m].f32[1] = 0;
                acc_o[i][min_tile_n * 2 + min_tile_m].f32[2] = 0;
                acc_o[i][min_tile_n * 2 + min_tile_m].f32[3] = 0;
                #elif defined(__gfx936__)
                acc_o[i][min_tile_n * 2 + min_tile_m].u64[0] = __builtin_hcu_mov_b64(pk_zero);
                acc_o[i][min_tile_n * 2 + min_tile_m].u64[1] = __builtin_hcu_mov_b64(pk_zero);
                #elif defined(__gfx938__) || defined(__gfx946__)
                asm volatile("v_mov_b64 %0, 0x0"
                    : "=v"(acc_o[i][min_tile_n * 2 + min_tile_m].u64[0])
                    :);
                asm volatile("v_mov_b64 %0, 0x0"
                    : "=v"(acc_o[i][min_tile_n * 2 + min_tile_m].u64[1])
                    :);
                #else
                acc_o[i][min_tile_n * 2 + min_tile_m].f32[0] = 0;
                acc_o[i][min_tile_n * 2 + min_tile_m].f32[1] = 0;
                acc_o[i][min_tile_n * 2 + min_tile_m].f32[2] = 0;
                acc_o[i][min_tile_n * 2 + min_tile_m].f32[3] = 0;
                #endif
            }
        }
    }
}


template<int kHeadDim, int WARP_M, int WARP_N, typename ElementAccum>
__forceinline__ __device__ void fp8_attention_initialize(
    ElementAccum scores_max[WARP_M / 16],
    ElementAccum scores_sum[WARP_M / 16],
    vec4_Accum<ElementAccum> acc_o[kHeadDim / 32][WARP_M / 16][WARP_N / 16]
) {
    #pragma unroll
    for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
        scores_max[m_idx] = -INFINITY;
        scores_sum[m_idx] = 0;
    }
    #pragma unroll
    for (int pv_loop = 0; pv_loop < kHeadDim / 32; ++pv_loop) {
        #pragma unroll
        for (int m_idx = 0; m_idx < WARP_M / 16; ++m_idx) {
            #pragma unroll
            for (int n_idx = 0; n_idx < WARP_N / 16; ++n_idx) {
                inline_vgpr4_init_zero(acc_o[pv_loop][m_idx][n_idx]);
            }
        }
    }
}



template<int kHeadDimV, typename ElementAccum>
__forceinline__ __device__ void prefix_prefill_hdim512_16x64_clear_acc_o(
        vec4_Accum<ElementAccum> acc_o[kHeadDimV / 16]) {
    #pragma unroll
    for (int i = 0; i < kHeadDimV / 16; ++i) {
        inline_vgpr4_init_zero(acc_o[i]);
    }
}

} // namespace flash
