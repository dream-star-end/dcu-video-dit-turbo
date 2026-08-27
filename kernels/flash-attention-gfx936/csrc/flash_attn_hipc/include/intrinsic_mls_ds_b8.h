#pragma once


#define MATRIX_LOAD_128X16_B8_LDS_TRANS(LDSADDR, SRSRC, MATRIX_OFFSET, R, T) \
    int soffset = LDSADDR + 0x80000000; \
    asm volatile("s_nop 4\n\t" \
                 "matrix_load_128x16_b8 %0, %1, moffset:%2 "#R #T" lds\n" \
                 :: "s"(SRSRC), "s"(soffset), "n"(MATRIX_OFFSET) \
                 :);

template<int r, int t, class DataType>
__forceinline__ __device__ void inline_matrix_load_128x16_b8_lds_trans(DataType *shared_addr, vec4_uint srsrc, int lds_offset, const int matrix_offset) {
#if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
    int lds_addr_per_wave = reinterpret_cast<size_t>(shared_addr) + (lds_offset);
    if constexpr (r && t) {
        MATRIX_LOAD_128X16_B8_LDS_TRANS(lds_addr_per_wave, srsrc, matrix_offset, r, t);
    } else if constexpr (r && !t) {
        MATRIX_LOAD_128X16_B8_LDS_TRANS(lds_addr_per_wave, srsrc, matrix_offset, r,);
    } else if constexpr (!r && t) {
        MATRIX_LOAD_128X16_B8_LDS_TRANS(lds_addr_per_wave, srsrc, matrix_offset,, t);
    } else {
        MATRIX_LOAD_128X16_B8_LDS_TRANS(lds_addr_per_wave, srsrc, matrix_offset,,);
    }
#endif
}

#define DS_READ_MATRIX_64x16_B8(OFFSET, REG, TRANS) \
    if constexpr (TRANS) { \
        asm volatile( \
            "s_add_u32 m0, %1, 0x80000000\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_trans_format %0, m0 offset:0 element:0x1 row:0x3 col:0x1 alt:0x0\n" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    } else { \
        asm volatile( \
            "s_nop 0\n\t" \
            "ds_read_matrix_format %0, m0 offset:0 element:0x1 row:0x3 col:0x1 alt:0x0\n" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    }


#define MATRIX_LOAD_64x32_B8_LDS_REARRANGE(LDSADDR, SRSRC, MATRIX_OFFSET, R, T) \
    asm volatile("s_nop 4\n\t" \
                 "matrix_load_64x32_b8 %0, %1, moffset:%2 "#R #T" lds\n" \
                 :: "s"(SRSRC), "s"(LDSADDR), "n"(MATRIX_OFFSET) \
                 :);

template<int r, int t, class DataType>
__forceinline__ __device__ void inline_matrix_load_64x32_b8_lds_rearrange(DataType *shared_addr, vec4_uint srsrc, int lds_offset, const int matrix_offset) {
#if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
    int lds_addr_per_wave = reinterpret_cast<size_t>(shared_addr) + (lds_offset);
    if constexpr (r && t) {
        MATRIX_LOAD_64x32_B8_LDS_REARRANGE(lds_addr_per_wave, srsrc, matrix_offset, r, t);
    } else if constexpr (r && !t) {
        MATRIX_LOAD_64x32_B8_LDS_REARRANGE(lds_addr_per_wave, srsrc, matrix_offset, r,);
    } else if constexpr (!r && t) {
        MATRIX_LOAD_64x32_B8_LDS_REARRANGE(lds_addr_per_wave, srsrc, matrix_offset,, t);
    } else {
        MATRIX_LOAD_64x32_B8_LDS_REARRANGE(lds_addr_per_wave, srsrc, matrix_offset,,);
    }
#endif
}


#define DS_READ_MATRIX_32x32_B8(OFFSET, REG, TRANS) \
    if constexpr (TRANS) { \
        asm volatile( \
            "s_add_u32 m0, %1, 0x80000000\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_trans_format %0, m0 offset:0 element:0x1 row:0x2 col:0x2 alt:0x0\n" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    } else { \
        asm volatile( \
            "s_mov_b32 m0, %1\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_format %0, m0 offset:0 element:0x1 row:0x2 col:0x2 alt:0x0\n" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    }


#define DS_READ_MATRIX_32x32_B8_ALT2(OFFSET, REG, TRANS) \
    if constexpr (TRANS) { \
        asm volatile( \
            "s_add_u32 m0, %1, 0x80000000\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_trans_format %0, m0 offset:0 element:0x1 row:0x2 col:0x2 alt:0x1\n" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    } else { \
        asm volatile( \
            "s_mov_b32 m0, %1\n\t" \
            "s_nop 0\n\t" \
            "ds_read_matrix_format %0, m0 offset:0 element:0x1 row:0x2 col:0x2 alt:0x1\n" \
            : "=v"(REG) \
            : "s"(OFFSET) \
            :); \
    }

template<class T, class AccumType>
inline __device__ vec4_fp32 mmac_4interleave_b8(const vec8_Element<T> &v1, const vec8_Element<T> &v2, const vec4_fp32 &v3)
{
    return __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(v1, v2, v3, 1, 0);
}
