#pragma once
#include <vector>
#include "numeric_types.h"
#include "intrinsic.h"

// ======================================================= MLS ===========================================================

#define VA_LIMIT_BITS(x) (0xffffffffffff & x)


#define MATRIX_LOAD_32X32_B16_LDS_TRANS(LDSADDR, SRSRC, R, T) \
    int soffset = LDSADDR + 0x80000000; \
    asm volatile("s_nop 0\n\t" \
                 "matrix_load_32x32_b16 %0, %1 moffset:%2 "#R #T" lds\n" \
                 :: "s"(SRSRC), "s"(soffset), "n"(0) \
                 :);

#define MATRIX_LOAD_32X32_B16_LDS_TRANS_GFX946(LDSADDR, SRSRC, R, T) \
    asm volatile("s_nop 0\n\t" \
                 "matrix_load_32x32_b16 %0, %1 moffset:%2 "#R #T" lds\n" \
                 :: "s"(SRSRC), "s"(LDSADDR), "n"(0) \
                 :);

template<int r, int t, class DataType>
__forceinline__ __device__ void inline_matrix_load_32x32_b16_lds_trans(DataType *shared_addr, vec4_uint srsrc, int &lds_offset, const int offset) {
#if defined(__gfx938__)
    int lds_addr_per_wave = reinterpret_cast<size_t>(shared_addr) + (lds_offset);
    /*
    matrix_load_32x32_b16 VDATA, SRSRC, m0 moffset:8 r:1 t:1 lds:1 glc:1 slc:1
    VDATA:DST
    SRSRC: {sgpr[SRSRC+1], sgpr[SRSRC]}: global base address
            sgpr[SRSRC+2]: stride
            sgpr[SRSRC+3]: m/nm_filter, cache swizzle, interleave
    */
    if constexpr (r && t) {
        MATRIX_LOAD_32X32_B16_LDS_TRANS(lds_addr_per_wave, srsrc, r, t);
    } else if constexpr (r && !t) {
        MATRIX_LOAD_32X32_B16_LDS_TRANS(lds_addr_per_wave, srsrc, r,);
    } else if constexpr (!r && t) {
        MATRIX_LOAD_32X32_B16_LDS_TRANS(lds_addr_per_wave, srsrc,, t);
    } else {
        MATRIX_LOAD_32X32_B16_LDS_TRANS(lds_addr_per_wave, srsrc,,);
    }
#elif defined(__gfx946__) || defined(__gfx92a__)
    int lds_addr_per_wave = reinterpret_cast<size_t>(shared_addr) + (lds_offset);
    if constexpr (r && t) {
        MATRIX_LOAD_32X32_B16_LDS_TRANS_GFX946(lds_addr_per_wave, srsrc, r, t);
    } else if constexpr (r && !t) {
        MATRIX_LOAD_32X32_B16_LDS_TRANS_GFX946(lds_addr_per_wave, srsrc, r,);
    } else if constexpr (!r && t) {
        MATRIX_LOAD_32X32_B16_LDS_TRANS_GFX946(lds_addr_per_wave, srsrc,, t);
    } else {
        MATRIX_LOAD_32X32_B16_LDS_TRANS_GFX946(lds_addr_per_wave, srsrc,,);
    }
#endif
}

#define MATRIX_LOAD_32X32_B16_LDS(LDSADDR, SRSRC, R, T) \
    int soffset = LDSADDR + 0x00000000; \
    asm volatile("s_nop 0\n\t" \
                 "matrix_load_32x32_b16 %0, %1 moffset:%2 "#R #T" lds\n" \
                 :: "s"(SRSRC), "s"(soffset), "n"(0) \
                 :);

#define MATRIX_LOAD_32X32_B16_LDS_GFX946(LDSADDR, SRSRC, R, T) \
    asm volatile("s_nop 0\n\t" \
                 "matrix_load_32x32_b16 %0, %1 moffset:%2 "#R #T" lds\n" \
                 :: "s"(SRSRC), "s"(LDSADDR), "n"(0) \
                 :);


template<int r, int t, class DataType>
__forceinline__ __device__ void inline_matrix_load_32x32_b16_lds(DataType *shared_addr, vec4_uint srsrc, int &lds_offset, const int offset) {
#if defined(__gfx938__)
    int lds_addr_per_wave = reinterpret_cast<size_t>(shared_addr) + (lds_offset);
    /*
    matrix_load_32x32_b16 VDATA, SRSRC, m0 moffset:8 r:1 t:1 lds:1 glc:1 slc:1
    VDATA:DST
    SRSRC: {sgpr[SRSRC+1], sgpr[SRSRC]}: global base address
            sgpr[SRSRC+2]: stride
            sgpr[SRSRC+3]: m/nm_filter, cache swizzle, interleave
    */
    if constexpr (r && t) {
        MATRIX_LOAD_32X32_B16_LDS(lds_addr_per_wave, srsrc, r, t);
    } else if constexpr (r && !t) {
        MATRIX_LOAD_32X32_B16_LDS(lds_addr_per_wave, srsrc, r,);
    } else if constexpr (!r && t) {
        MATRIX_LOAD_32X32_B16_LDS(lds_addr_per_wave, srsrc,, t);
    } else {
        MATRIX_LOAD_32X32_B16_LDS(lds_addr_per_wave, srsrc,,);
    }
#elif defined(__gfx946__) || defined(__gfx92a__)
    int lds_addr_per_wave = reinterpret_cast<size_t>(shared_addr) + (lds_offset);
    if constexpr (r && t) {
        MATRIX_LOAD_32X32_B16_LDS_GFX946(lds_addr_per_wave, srsrc, r, t);
    } else if constexpr (r && !t) {
        MATRIX_LOAD_32X32_B16_LDS_GFX946(lds_addr_per_wave, srsrc, r,);
    } else if constexpr (!r && t) {
        MATRIX_LOAD_32X32_B16_LDS_GFX946(lds_addr_per_wave, srsrc,, t);
    } else {
        MATRIX_LOAD_32X32_B16_LDS_GFX946(lds_addr_per_wave, srsrc,,);
    }
#endif
}

// ======================================================= MLS32x16 ===========================================================
#define MATRIX_LOAD_32X16_B16_LDS_TRANS(LDSADDR, SRSRC, R, T) \
    int soffset = LDSADDR + 0x80000000; \
    asm volatile("s_nop 0\n\t" \
                 "matrix_load_32x16_b16 %0, %1 moffset:%2 "#R #T" lds\n" \
                 :: "s"(SRSRC), "s"(soffset), "n"(0) \
                 :);

#define MATRIX_LOAD_32X16_B16_LDS_TRANS_GFX946(LDSADDR, SRSRC, R, T) \
    asm volatile("s_nop 0\n\t" \
                 "matrix_load_32x16_b16 %0, %1 moffset:%2 "#R #T" lds\n" \
                 :: "s"(SRSRC), "s"(LDSADDR), "n"(0) \
                 :);

template<int r, int t, class DataType>
__forceinline__ __device__ void inline_matrix_load_32x16_b16_lds_trans(DataType *shared_addr, vec4_uint srsrc, int &lds_offset, const int offset) {
#if defined(__gfx938__)
    int lds_addr_per_wave = reinterpret_cast<size_t>(shared_addr) + (lds_offset);
    /*
    matrix_load_32x32_b16 VDATA, SRSRC, m0 moffset:8 r:1 t:1 lds:1 glc:1 slc:1
    VDATA:DST
    SRSRC: {sgpr[SRSRC+1], sgpr[SRSRC]}: global base address
            sgpr[SRSRC+2]: stride
            sgpr[SRSRC+3]: m/nm_filter, cache swizzle, interleave
    */
    if constexpr (r && t) {
        MATRIX_LOAD_32X16_B16_LDS_TRANS(lds_addr_per_wave, srsrc, r, t);
    } else if constexpr (r && !t) {
        MATRIX_LOAD_32X16_B16_LDS_TRANS(lds_addr_per_wave, srsrc, r,);
    } else if constexpr (!r && t) {
        MATRIX_LOAD_32X16_B16_LDS_TRANS(lds_addr_per_wave, srsrc,, t);
    } else {
        MATRIX_LOAD_32X16_B16_LDS_TRANS(lds_addr_per_wave, srsrc,,);
    }
#elif defined(__gfx946__) || defined(__gfx92a__)
    int lds_addr_per_wave = reinterpret_cast<size_t>(shared_addr) + (lds_offset);
    if constexpr (r && t) {
        MATRIX_LOAD_32X16_B16_LDS_TRANS_GFX946(lds_addr_per_wave, srsrc, r, t);
    } else if constexpr (r && !t) {
        MATRIX_LOAD_32X16_B16_LDS_TRANS_GFX946(lds_addr_per_wave, srsrc, r,);
    } else if constexpr (!r && t) {
        MATRIX_LOAD_32X16_B16_LDS_TRANS_GFX946(lds_addr_per_wave, srsrc,, t);
    } else {
        MATRIX_LOAD_32X16_B16_LDS_TRANS_GFX946(lds_addr_per_wave, srsrc,,);
    }
#endif
}


#define MATRIX_LOAD_32X16_B16_LDS(LDSADDR, SRSRC, R, T) \
    int soffset = LDSADDR + 0x00000000; \
    asm volatile("s_nop 0\n\t" \
                 "matrix_load_32x16_b16 %0, %1 moffset:%2 "#R #T" lds\n" \
                 :: "s"(SRSRC), "s"(soffset), "n"(0) \
                 :);

#define MATRIX_LOAD_32X16_B16_LDS_GFX946(LDSADDR, SRSRC, R, T) \
    asm volatile("s_nop 0\n\t" \
                 "matrix_load_32x16_b16 %0, %1 moffset:%2 "#R #T" lds\n" \
                 :: "s"(SRSRC), "s"(LDSADDR), "n"(0) \
                 :);


template<int r, int t, class DataType>
__forceinline__ __device__ void inline_matrix_load_32x16_b16_lds(DataType *shared_addr, vec4_uint srsrc, int &lds_offset, const int offset) {
#if defined(__gfx938__)
    int lds_addr_per_wave = reinterpret_cast<size_t>(shared_addr) + (lds_offset);
    /*
    matrix_load_32x32_b16 VDATA, SRSRC, m0 moffset:8 r:1 t:1 lds:1 glc:1 slc:1
    VDATA:DST
    SRSRC: {sgpr[SRSRC+1], sgpr[SRSRC]}: global base address
            sgpr[SRSRC+2]: stride
            sgpr[SRSRC+3]: m/nm_filter, cache swizzle, interleave
    */
    if constexpr (r && t) {
        MATRIX_LOAD_32X16_B16_LDS(lds_addr_per_wave, srsrc, r, t);
    } else if constexpr (r && !t) {
        MATRIX_LOAD_32X16_B16_LDS(lds_addr_per_wave, srsrc, r,);
    } else if constexpr (!r && t) {
        MATRIX_LOAD_32X16_B16_LDS(lds_addr_per_wave, srsrc,, t);
    } else {
        MATRIX_LOAD_32X16_B16_LDS(lds_addr_per_wave, srsrc,,);
    }
#elif defined(__gfx946__) || defined(__gfx92a__)
    int lds_addr_per_wave = reinterpret_cast<size_t>(shared_addr) + (lds_offset);
    if constexpr (r && t) {
        MATRIX_LOAD_32X16_B16_LDS_GFX946(lds_addr_per_wave, srsrc, r, t);
    } else if constexpr (r && !t) {
        MATRIX_LOAD_32X16_B16_LDS_GFX946(lds_addr_per_wave, srsrc, r,);
    } else if constexpr (!r && t) {
        MATRIX_LOAD_32X16_B16_LDS_GFX946(lds_addr_per_wave, srsrc,, t);
    } else {
        MATRIX_LOAD_32X16_B16_LDS_GFX946(lds_addr_per_wave, srsrc,,);
    }
#endif
}

// ======================================================= DS ===========================================================

#define DS_READ_MATRIX_32X32_B16(OFFSET, REG, REG1, TRANS) \
    if constexpr (TRANS) { \
        asm volatile( \
            "s_mov_b32 m0, %2\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_trans_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\t" \
            "ds_read_matrix_trans_format %1, m0 offset:1024 element:0x2 row:0x2 col:0x1 alt:0x0\n" \
            : "=v"(REG), "=v"(REG1) \
            : "s"(OFFSET) \
            :); \
    } else { \
        asm volatile( \
            "s_mov_b32 m0, %2\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\t" \
            "ds_read_matrix_format %1, m0 offset:1024 element:0x2 row:0x2 col:0x1 alt:0x0\n" \
            : "=v"(REG), "=v"(REG1) \
            : "s"(OFFSET) \
            :); \
    }

#define DS_READ_MATRIX_32X32_B16_GFX946(OFFSET, REG, REG1, TRANS) \
    if constexpr (TRANS) { \
        asm volatile( \
            "s_nop 0\n\t" \
            "ds_read_matrix_trans_format %0, %2 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\t" \
            "ds_read_matrix_trans_format %1, %2 offset:1024 element:0x2 row:0x2 col:0x1 alt:0x0\n" \
            : "=v"(REG), "=v"(REG1) \
            : "s"(OFFSET) \
            :); \
    } else { \
        asm volatile( \
            "s_nop 0\n\t" \
            "ds_read_matrix_format %0, %2 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\t" \
            "ds_read_matrix_format %1, %2 offset:1024 element:0x2 row:0x2 col:0x1 alt:0x0\n" \
            : "=v"(REG), "=v"(REG1) \
            : "s"(OFFSET) \
            :); \
    }

#define DS_READ_MATRIX_32X16_B16(OFFSET, REG, TRANS) \
    if constexpr (TRANS) { \
        asm volatile( \
            "s_mov_b32 m0, %1\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_trans_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\t" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    } else { \
        asm volatile( \
            "s_mov_b32 m0, %1\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x0\n\t" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    }


#define DS_READ_MATRIX_32X16_B16_ALT2(OFFSET, REG, TRANS) \
    if constexpr (TRANS) { \
        asm volatile( \
            "s_mov_b32 m0, %1\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_trans_format %0, m0 offset:0 element:0x2 row:0x1 col:0x2 alt:0x1\n\t" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    } else { \
        asm volatile( \
            "s_mov_b32 m0, %1\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x1\n\t" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    }

#define DS_READ_MATRIX_32X32_B16_ALT2(OFFSET, REG, REG1, TRANS) \
    if constexpr (TRANS) { \
        asm volatile( \
            "s_mov_b32 m0, %2\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_trans_format %0, m0 offset:0 element:0x2 row:0x1 col:0x2 alt:0x1\n\t" \
            "ds_read_matrix_trans_format %1, m0 offset:1024 element:0x2 row:0x1 col:0x2 alt:0x1\n\t" \
            : "=v"(REG), "=v"(REG1) \
            : "s"(OFFSET) \
            :); \
    } else { \
        asm volatile( \
            "s_mov_b32 m0, %2\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_format %0, m0 offset:0 element:0x2 row:0x2 col:0x1 alt:0x1\n\t" \
            "ds_read_matrix_format %1, m0 offset:1024 element:0x2 row:0x2 col:0x1 alt:0x1\n\t" \
            : "=v"(REG), "=v"(REG1) \
            : "s"(OFFSET) \
            :); \
    }


template<int min_value, int max_value>
__forceinline__ __device__ int inline_min_max(int source) {
    /*
        To avoid usage of v_med3_i32
            ----> to avoid usage of __builtin_amdgcn_readfirstlane
                ----> to avoid usage of 5 nops for mls data hazard
    */
    return max(min_value, min(max_value, source));
    // int result;
    // asm volatile("s_max_i32 %0, %1, %2\n\t"
    //              "s_min_i32 %0, %0, %3\n"
    //            : "=s"(result)
    //            : "s"(source), "n"(min_value), "n"(max_value)
    //            :);
    // return result;
}

// ======================================================= def ===========================================================
#define YY_USE_MPERMUTE

template<typename VEC>
__forceinline__ __device__ void ds_mpermute_kdim_for_mmac(VEC& data) {
    asm volatile(
        "ds_mpermute_dwordx2 %0, %0 offset:6\n"
        : "+v"(data)
        : );
}

template<typename VEC>
__forceinline__ __device__ void ds_mpermute_kdim_for_mmac_wait(VEC& data) {
    asm volatile(
        "ds_mpermute_dwordx2 %0, %0 offset:6\n\ts_waitcnt lgkmcnt(0)\n"
        : "+v"(data)
        : );
}


// ======================================================= mmac ===========================================================
template<class T, class AccumType>
inline __device__ vec4_fp32 mmac_4interleave(const vec4_Element<T> &v1, const vec4_Element<T> &v2, const vec4_fp32 &v3)
{
    return __builtin_hcu_mmac_f32_16x16x16_f16(v1, v2, v3);
}

template<>
inline __device__ vec4_fp32 mmac_4interleave<half_t, float>(const vec4_fp16 &v1, const vec4_fp16 &v2, const vec4_fp32 &v3)
{
#if defined(__gfx938__) || defined(__gfx946__)
    return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(v1, v2, v3, 1, 0);
#else
    return __builtin_hcu_mmac_f32_16x16x16_f16(v1, v2, v3);
#endif
}

template<>
inline __device__ vec4_fp32 mmac_4interleave<bhalf_t, float>(const vec4_bf16 &v1, const vec4_bf16 &v2, const vec4_fp32 &v3)
{
#if defined(__gfx938__) || defined(__gfx946__)
    return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(v1, v2, v3, 1, 0);
#else
    return __builtin_hcu_mmac_f32_16x16x16_bf16(v1, v2, v3);
#endif
}
