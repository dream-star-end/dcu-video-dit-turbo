#pragma once
#include "hip/hip_fp16.h"
#include "hip/hip_bf16.h"
#include "hip/hip_runtime.h"
#include "hip/hip_fp8.h"
using bhalf_t = __hip_bfloat16;
using half_t  = __half;
using fp8_e4m3  = __hip_fp8_e4m3;
using fp8_e5m2  = __hip_fp8_e5m2;
using BFloat16 = bhalf_t;
using Float16 = half_t;
using Int32 = int;
using Int16 = unsigned short;
using Float32 = float;
using f8_t = uint8_t;

//fp8_e4m3 definitions
struct alignas(1) Float8_e4m3_t{
    /// Data container
    uint8_t data;
    __host__ __device__ Float8_e4m3_t() = default;
    __host__ __device__ Float8_e4m3_t(uint8_t value): data(value) {}
};

typedef short __attribute__((ext_vector_type(8))) vec8_bf16;
typedef _Float16  __attribute__((ext_vector_type(8))) vec8_fp16;

using vec4_fp16 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
// using vec8_fp16 = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using vec2_fp16 = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
using vec16_fp16 = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;

using vec4_bf16 = __attribute__((__vector_size__(4 * sizeof(unsigned short)))) unsigned short;
// using vec8_bf16 = __attribute__((__vector_size__(8 * sizeof(unsigned short)))) unsigned short;
using vec2_bf16 = __attribute__((__vector_size__(2 * sizeof(unsigned short)))) unsigned short;
using vec16_bf16 = __attribute__((__vector_size__(16 * sizeof(unsigned short)))) unsigned short;

using vec4_uint = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using vec2_uint = __attribute__((__vector_size__(2 * sizeof(uint32_t)))) uint32_t;
using vec4_int = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) int32_t;

using vec4_fp8 = __attribute__((__vector_size__(4 * sizeof(uint8_t)))) uint8_t;
using vec8_fp8 = __attribute__((__vector_size__(8 * sizeof(uint8_t)))) uint8_t;
using vec2_fp8 = __attribute__((__vector_size__(2 * sizeof(uint8_t)))) uint8_t;
using vec16_fp8 = __attribute__((__vector_size__(16 * sizeof(uint8_t)))) uint8_t;

using vec4_int8 = __attribute__((__vector_size__(4 * sizeof(int8_t)))) int8_t;
using vec8_int8 = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;
using vec2_int8 = __attribute__((__vector_size__(2 * sizeof(int8_t)))) int8_t;
using vec16_int8 = __attribute__((__vector_size__(16 * sizeof(int8_t)))) int8_t;
using vec4_int32 = __attribute__((__vector_size__(4 * sizeof(int)))) int;

using __builtin_half2 = __attribute__((ext_vector_type(2))) __fp16;
using __float2  = __attribute__((ext_vector_type(2))) float;

using vec4_fp32 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using vec2_fp32 = __attribute__((__vector_size__(2 * sizeof(float)))) float;


union union_vec4_fp32 {
    vec4_fp32 f32;
    double data[2];
    int64_t b64[2];
    __float2 u64[2];
    vec2_fp32 f32x2[2];
    half_t f16[8];
};

union union_vec2_fp32 {
    vec2_fp32 f32;
    double data;
    __float2 u64;
};

union union_vec_fp32 {
    float f32[1];
};

union union_vec4_uint {
    unsigned long long u64[2]; // 128 bits
    uint4 u32;
    uint8_t u8[16];
};

union union_vec2_uint {
    uint2 u32;
    unsigned long long u64;
};

union union_vec8_fp32 {
    float f32[8];
    vec4_fp32 f32x4[2];
    double data[4];
    __float2 u64[4];
};

union union_vec4_int32 {
    vec4_int32 int32;
    int64_t b64[2];
    __float2 u64[2];
};

template<typename T>
using vec2_Element = 
    std::conditional_t
    <
        std::is_same_v<T, half_t>, 
        vec2_fp16, 
        std::conditional_t
        <
            std::is_same_v<T, bhalf_t>, 
            vec2_bf16,
            std::conditional_t
            <
                std::is_same_v<T, Float8_e4m3_t> || std::is_same_v<T, int8_t>, 
                vec2_fp8,
                std::conditional_t
                <
                    std::is_same_v<T, int8_t>, 
                    vec2_int8,
                    vec2_fp32
                >
            >    
        >
    >;

template<typename T>
using vec4_Element = 
    std::conditional_t
    <
        std::is_same_v<T, half_t>, 
        vec4_fp16, 
        std::conditional_t
        <
            std::is_same_v<T, bhalf_t>, 
            vec4_bf16,
            std::conditional_t
            <
                std::is_same_v<T, Float8_e4m3_t>, 
                vec4_fp8,
                std::conditional_t
                <
                    std::is_same_v<T, int8_t>, 
                    vec4_int8,
                    void
                > 
            >    
        >
    >;

template<typename T>
using vec8_Element = 
    std::conditional_t
    <
        std::is_same_v<T, half_t>, 
        vec8_fp16, 
        std::conditional_t
        <
            std::is_same_v<T, bhalf_t>, 
            vec8_bf16,
            std::conditional_t
            <
                std::is_same_v<T, Float8_e4m3_t>, 
                vec8_fp8,
                std::conditional_t
                <
                    std::is_same_v<T, int8_t>, 
                    vec8_int8,
                    void
                > 
            >    
        >
    >;

template<typename Element>
union union_vec2_f16x2 {
    vec2_fp32 f32;
    double data;
    __float2 u64;
    vec2_Element<Element> f16x2[2];
    vec4_Element<Element> f8x4[2];
    vec4_Element<Element> f16x4;
    Element f16[4];
    Element f8[8];
    int32_t i32[2];
};

template<typename Element>
union union_vec2_f8x2 {
    // vec2_u16 u16;
    float data;
    unsigned short u16[2];
    vec2_Element<Element> i8x2[2];
    vec4_Element<Element> i8x4;
    Element i8[4];
};

template<typename Element>
union union_vec4_f16x2 {
    vec4_fp32 f32;
    double data[2];
    __float2 u64[2];
    vec2_Element<Element> f16x2[4];
    vec4_Element<Element> f16x4[2];
    vec8_Element<Element> f16x8;
    vec4_int8 f8x4[4];
    Element f16[8];
    __builtin_half2 b16x2[4];
};

template<typename Element>
union union_vec2_int8x2 {
    // vec2_u16 u16;
    float data;
    unsigned short u16[2];
    vec2_Element<Element> i8x2[2];
    vec4_Element<Element> i8x4;
    Element i8[4];
};

union union_vec16_fp8 {
    int8_t i8[16];
    vec8_int8 i8x8[2];
    vec4_int8 i8x4[4];
    vec4_fp32 f32x4;
    vec4_int i32x4;
    __float2 u64[2];
    int32_t i32[4];
};

union union_vec32_fp8 {
    int8_t i8[32];
    vec8_int8 i8x8[4];
    vec4_int8 i8x4[8];
    vec4_int i32x4[2];
    int32_t i32[8];
};

template<typename T>
using vec4_Accum = std::conditional_t<std::is_same_v<T, float>, union_vec4_fp32, vec4_bf16>;

template<typename T>
using vec2_Accum = std::conditional_t<std::is_same_v<T, float>, union_vec2_fp32, vec4_bf16>;

template<typename T>
using vec_Accum = std::conditional_t<std::is_same_v<T, float>, union_vec_fp32, vec2_bf16>;

template<typename T>
__forceinline__ __device__ vec4_Element<T> make_vec4_f16(T a, T b, T c, T d) {
    return {a, b, c, d};
}

template<>
__forceinline__ __device__ vec4_Element<bhalf_t> make_vec4_f16(bhalf_t a, bhalf_t b, bhalf_t c, bhalf_t d) {
#ifdef ROCM_5_7
    return {a.data, b.data, c.data, d.data};
#else
    return {__hip_bfloat16_raw(a).x, __hip_bfloat16_raw(b).x, __hip_bfloat16_raw(c).x, __hip_bfloat16_raw(d).x};
    // return {*(unsigned short*)(&a), *(unsigned short*)(&b), *(unsigned short*)(&c), *(unsigned short*)(&d)};
#endif
}

