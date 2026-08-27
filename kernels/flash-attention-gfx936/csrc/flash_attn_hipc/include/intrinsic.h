#pragma once
#include <vector>
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "numeric_types.h"

#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
#define USE_BUFFER_LOAD_DWORDX4
// #define USE_BUFFER_LOAD_DWORDX2
#endif



template<typename VEC, typename pointerType>
__forceinline__ __device__ void inline_global_load_dwordx1(VEC& v_data, const int v_offset, const pointerType* s_addr) {
    const int v_offset_bytes = v_offset * sizeof(pointerType);
    asm volatile(
        "global_load_dword %0, %1, %2\n"
        : "=v"(v_data)
        : "v"(v_offset_bytes), "s"(s_addr)
        :);
}

template<const int shfl_count, bool bypass, class DataType>
__forceinline__ __device__ void inline_buffer_load_dwordx1(DataType& v_data, const int v_offset, const vec4_uint global_addr) {

    int v_offset_bytes = v_offset << shfl_count;
    if constexpr (bypass) {
        asm volatile(
            "buffer_load_dword %0, %1, %2, 0, offen offset:0 glc slc\n"
            : "=v"(v_data)
            : "v"(v_offset_bytes), "s"(global_addr)
            :);
    } else {
        asm volatile(
            "buffer_load_dword %0, %1, %2, 0, offen offset:0\n"
            : "=v"(v_data)
            : "v"(v_offset_bytes), "s"(global_addr)
            :);
    }
}


template<class DataType>
__forceinline__ __device__ void inline_utcl2_warmup_dword(DataType buffer_resource) {
    int container;
    int offset = 0;
    __builtin_amdgcn_sched_barrier(0);
    asm volatile(
        "s_nop 4\n\t"
        "buffer_load_dword %0, %1, %2, 0, offen offset:0 glc slc\n\t"
        "s_waitcnt vmcnt(0)\n"
        : "=v"(container)
        : "v"(offset), "s"(buffer_resource)
    );
    __builtin_amdgcn_sched_barrier(0);
}


template<class DataType, const int shfl_count=2>
__forceinline__ __device__ void inline_buffer_load_dword_lds(DataType *const shared_addr, const vec4_uint global_addr, const int &lds_offset, const int &gvOffset_s, const int &gvOffset_v) {

  int ldsAddrPerWave = reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count);
  int offset_s = gvOffset_s << shfl_count;
  int offset_v = gvOffset_v << shfl_count;

  asm volatile("s_mov_b32 m0, %1 \n\t"
               "buffer_load_dword %0, %2, %3 ,offen  offset:0, lds \n"
               :: "v"(offset_v), "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
               :);
}

template<class DataType, const int shfl_count=2>
__forceinline__ __device__ void inline_buffer_load_dwordx2_lds(DataType *const shared_addr, const vec4_uint global_addr, const int &lds_offset, const int &gvOffset_s, const int &gvOffset_v) {

  int ldsAddrPerWave = reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count);
  int offset_s = gvOffset_s << shfl_count;
  int offset_v = gvOffset_v << shfl_count;

  asm volatile("s_mov_b32 m0, %1 \n\t"
               "buffer_load_dwordx2 %0, %2, %3 ,offen  offset:0, lds \n"
               :: "v"(offset_v), "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
               :);
}

template<class DataType, const int shfl_count=2>
__forceinline__ __device__ void inline_buffer_load_dwordx4_lds(DataType *const shared_addr, const vec4_uint global_addr, const int &lds_offset, const int &gvOffset_s, const int &gvOffset_v) {

  int ldsAddrPerWave = reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count);
  int offset_s = gvOffset_s << shfl_count;
  int offset_v = gvOffset_v << shfl_count;

  asm volatile("s_mov_b32 m0, %1 \n\t"
               "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n"
               :: "v"(offset_v), "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
               :);
}

template<class DataType, const int shfl_count=2>
__forceinline__ __device__ void safe_inline_buffer_load_dwordx4_lds(DataType *const shared_addr, const vec4_uint global_addr, const int &lds_offset, const int &offset_s, const int &offset_v) {

  int lds_addr_per_wave = reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count);
  int __offset_s = offset_s << shfl_count;
  int __offset_v = offset_v << shfl_count;

  asm volatile("s_nop 3\n\t"
               "s_mov_b32 m0, %1\n\t"
               "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds\n"
               :: "v"(__offset_v), "s"(lds_addr_per_wave), "s"(global_addr), "s"(__offset_s)
               :);
}

template<class DataType, const int shfl_count=2>
__forceinline__ __device__ void inline_buffer_load_dword_lds_bypass_glc_slc(DataType *const shared_addr, const vec4_uint global_addr, const int &lds_offset, const int &gvOffset_s, const int &gvOffset_v) {

  int ldsAddrPerWave = reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count);
  int offset_s = gvOffset_s << shfl_count;
  int offset_v = gvOffset_v << shfl_count;

  asm volatile("s_mov_b32 m0, %1 \n\t"
               "buffer_load_dword %0, %2, %3 ,offen  offset:0 glc slc lds\n"
               :: "v"(offset_v), "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
               :);
}

template<class DataType, const int shfl_count=2>
__forceinline__ __device__ void inline_buffer_load_dword_lds_bypass_l1_glc(DataType *const shared_addr, const vec4_uint global_addr, const int &lds_offset, const int &gvOffset_s, const int &gvOffset_v) {

  int ldsAddrPerWave = reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count);
  int offset_s = gvOffset_s << shfl_count;
  int offset_v = gvOffset_v << shfl_count;

  asm volatile("s_mov_b32 m0, %1 \n\t"
               "buffer_load_dword %0, %2, %3 ,offen  offset:0 glc lds\n"
               :: "v"(offset_v), "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
               :);
}

template<class DataType, const int shfl_count=2>
__forceinline__ __device__ void inline_buffer_load_dword_lds_bypass_l2_slc(DataType *const shared_addr, const vec4_uint global_addr, const int &lds_offset, const int &gvOffset_s, const int &gvOffset_v) {

  int ldsAddrPerWave = reinterpret_cast<size_t>(shared_addr) + (lds_offset << shfl_count);
  int offset_s = gvOffset_s << shfl_count;
  int offset_v = gvOffset_v << shfl_count;

  asm volatile("s_mov_b32 m0, %1 \n\t"
               "buffer_load_dword %0, %2, %3 ,offen  offset:0 slc lds\n"
               :: "v"(offset_v), "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
               :);
}

template<typename src_type=half_t, typename dst_type=float, const int dword_count=1, const int auxilariy=0>
__forceinline__ __device__ void builtin_buffer_load_dword_lds(
    src_type *const shared_addr,
    const vec4_uint rsrc,
    const int &lds_offset,
    const int gvOffset_s,
    const int &gvOffset_v) {
#if defined(H3_STANDARD_FWD_ONLY)
    static_assert(dword_count == 1);
    static_assert(sizeof(dst_type) == 4);
    inline_buffer_load_dword_lds(
        shared_addr, rsrc, lds_offset, gvOffset_s, gvOffset_v);
#else
    constexpr int bytes_per_element = sizeof(dst_type);
    using lds_int_ptr = __attribute__((address_space(3))) int*;
    lds_int_ptr ptr = reinterpret_cast<lds_int_ptr>(shared_addr) + lds_offset;
    __builtin_hcu_raw_buffer_load_lds(
        rsrc,
        ptr,
        dword_count * 4,
        gvOffset_v * bytes_per_element,
        gvOffset_s * bytes_per_element,
        0, /* immediate offset, instruction offset */
        auxilariy /* auxilariy data| bit 0: glc, bit 1: slc, bit 2: dlc, bit 3: cache swizzle */
    );
#endif
}

template<typename src_type=half_t, typename dst_type=float>
__forceinline__ __device__ void builtin_buffer_load_dword_lds_bypass_glc_slc(
    src_type *const shared_addr,
    const vec4_uint rsrc,
    const int &lds_offset,
    const int gvOffset_s,
    const int &gvOffset_v) {
#if defined(H3_STANDARD_FWD_ONLY)
    static_assert(sizeof(dst_type) == 4);
    inline_buffer_load_dword_lds_bypass_glc_slc(
        shared_addr, rsrc, lds_offset, gvOffset_s, gvOffset_v);
#else
    constexpr int bytes_per_element = sizeof(dst_type);
    using lds_int_ptr = __attribute__((address_space(3))) int*;
    lds_int_ptr ptr = reinterpret_cast<lds_int_ptr>(shared_addr) + lds_offset;
    __builtin_hcu_raw_buffer_load_lds(
        rsrc,
        ptr,
        4,
        gvOffset_v * bytes_per_element,
        gvOffset_s * bytes_per_element,
        0, /* immediate offset, instruction offset */
        11 /* auxilariy data| bit 0: glc, bit 1: slc, bit 2: dlc, bit 3: cache swizzle */
    );
#endif
}

template<class DataType, const int shfl_count>
__forceinline__ __device__ void inline_buffer_store_dword(const DataType v_data, const int &v_offset, const vec4_uint global_addr, const int &s_offset, const int s_constant=0) {

  int v_offset_bytes = v_offset << shfl_count;
  int s_offset_bybes = s_offset << shfl_count;
  const int s_constant_bytes = s_constant << shfl_count;

  asm volatile(
      "buffer_store_dword %0, %1, %2, %3, offen  offset:%4 \n"
      :: "v"(v_data), "v"(v_offset_bytes), "s"(global_addr), "s"(s_offset_bybes), "B"(s_constant_bytes)
      :);
}

template<class DataType, const int shfl_count>
__forceinline__ __device__ void inline_buffer_store_dwordx4(const DataType v_data, const int &v_offset, const vec4_uint global_addr, const int &s_offset, const int s_constant=0) {

  int v_offset_bytes = v_offset << shfl_count;
  int s_offset_bybes = s_offset << shfl_count;
  const int s_constant_bytes = s_constant << shfl_count;

  asm volatile(
      "buffer_store_dwordx4 %0, %1, %2, %3, offen offset:%4 \n"
      :: "v"(v_data), "v"(v_offset_bytes), "s"(global_addr), "s"(s_offset_bybes), "B"(s_constant_bytes)
      :);
}

template<class DataType, const int shfl_count>
__forceinline__ __device__ void inline_buffer_store_dword_glc_slc(DataType v_data, int &v_offset, vec4_uint global_addr, int &s_offset, const int s_constant=0) {

  int v_offset_bytes = v_offset << shfl_count;
  int s_offset_bybes = s_offset << shfl_count;
  const int s_constant_bytes = s_constant << shfl_count;

#if !defined(__gfx916__) && !defined(__gfx926__)
  asm volatile(
      "buffer_store_dword %0, %1, %2, %3, offen  offset:%4  glc slc\n"
      :: "v"(v_data), "v"(v_offset_bytes), "s"(global_addr), "s"(s_offset_bybes), "B"(s_constant_bytes)
      :);
#endif
}

template<typename VEC>
__forceinline__ __device__ void  inline_ds_read_b16_no_wait_bytes(const int &lds_offset, VEC &reg_val) {
  asm volatile(
      "ds_read_u16 %0 ,%1\n"
      : "=v"(reg_val)
      : "v"(lds_offset)
      :);
}

template<typename VEC>
__forceinline__  __device__ void  inline_ds_read_b32_no_wait(VEC *const shared_addr, const int &lds_offset, VEC &reg_val) {
  int ldsAddr = reinterpret_cast<size_t>(shared_addr) + lds_offset * 4;
  asm volatile(
      "ds_read_b32 %0, %1\n"
      : "=v"(reg_val)
      : "v"(ldsAddr)
      :);
}

template<typename VEC>
__forceinline__  __device__ void  inline_ds_read_b32_no_wait_bytes(const int &lds_offset, VEC &reg_val) {
  asm volatile(
      "ds_read_b32 %0, %1\n"
      : "=v"(reg_val)
      : "v"(lds_offset)
      :);
}

template<typename VEC, typename dwordx2>
__forceinline__  __device__ void  inline_ds_read2_b32_no_wait(VEC *shared_addr, const int &lds_offset, dwordx2& reg_val, const int offset1) {
  int ldsAddr = reinterpret_cast<size_t>(shared_addr) + lds_offset * 4;
  asm volatile(
      "ds_read2_b32 %0 ,%1 offset0:0 offset1:%2\n"
      : "=v"(reg_val)
      : "v"(ldsAddr), "B"(offset1)
      :);
}

template<typename dwordx2>
__forceinline__  __device__ void  inline_ds_read2_b32_no_wait_bytes(const int &lds_offset, dwordx2& reg_val, const int offset1) {
  asm volatile(
      "ds_read2_b32 %0, %1 offset0:0 offset1:%2\n"
      : "=v"(reg_val)
      : "v"(lds_offset), "B"(offset1)
      :);
}


template<typename DataType>
__forceinline__  __device__ void inline_ds_read2_b64(const int lds_offset, DataType& reg_val, const int offset0, const int offset1) {
    asm volatile(
        "ds_read2_b64 %0, %1, offset0:%2, offset1:%3\n"
        : "=v"(reg_val)
        : "s"(lds_offset), "B"(offset0), "B"(offset1)
        :);
}


template<typename dwordx2>
__forceinline__  __device__ void  inlineasm_fa_ds_read2_b32(float* shared_addr, const int &lds_offset, dwordx2& reg_val, const int offset0, const int offset1) {
  int lds_addr = reinterpret_cast<size_t>(shared_addr) + lds_offset * 4;
  asm volatile(
      "ds_read2_b32 %0, %1 offset0:%2 offset1:%3\n"
      : "=v"(reg_val)
      : "v"(lds_addr), "B"(offset0), "B"(offset1)
      :);
}

__forceinline__  __device__ void  inline_ds_write2_b32_no_wait_bytes(float* shared_addr, int lds_offset, const __float2& reg_val, const int offset0, const int offset1) {
  int lds_addr = reinterpret_cast<size_t>(shared_addr) + lds_offset * 4;
  asm volatile(
      "ds_write2_b32 %0, %1, %2 offset0:%3 offset1:%4\n"
      : "=v"(lds_addr)
      : "v"(reg_val[0]), "v"(reg_val[1]), "B"(offset0), "B"(offset1)
      :);
}

template<typename VEC>
__forceinline__  __device__ void  inline_ds_read_b32_wait(VEC *const shared_addr, const int &lds_offset, VEC &reg_val) {
  reg_val = shared_addr[lds_offset];
}


template<typename VEC>
__forceinline__ __device__ void inlineasm_ds_read_b128(int lds_offset, VEC& data) {
    asm volatile("ds_read_b128 %0, %1\n"
        : "=v"(data)
        : "v"(lds_offset)
        :);
}

template<typename VEC>
__forceinline__ __device__ void inlineasm_ds_write_b128(int lds_offset, VEC& data) {
    asm volatile("ds_write_b128 %0, %1\n"
        :: "v"(lds_offset), "v"(data)
        :);
}


template<typename VEC>
__forceinline__  __device__ void inline_vgpr_init_zero(VEC &dst, const int idx) {
  asm ("v_mov_b32 %0, 0x0"
      : "=v"(dst[idx])
      :);
}

template<typename VEC>
__forceinline__  __device__ void inline_vgpr2_init_zero(VEC &dst) {
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
  asm ("v_mov_b64 %0, 0x0"
      : "=v"(dst)
      :);
#else
  dst = 0x0;
#endif
}

template<typename VEC>
__forceinline__  __device__ void inline_vgpr4_init_zero(VEC &dst) {
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
  asm ("v_mov_b64 %0, 0x0\n\t"
       "v_mov_b64 %1, 0x0\n\t"
      : "=v"(dst.u64[0]), "=v"(dst.u64[1])
      :);
#else
  asm ("v_mov_b32 %0, 0x0\n\t"
       "v_mov_b32 %1, 0x0\n\t"
       "v_mov_b32 %2, 0x0\n\t"
       "v_mov_b32 %3, 0x0\n\t"
      : "=v"(dst.f32[0]), "=v"(dst.f32[1]), "=v"(dst.f32[2]), "=v"(dst.f32[3])
      :);
#endif
}

template<typename VEC>
__forceinline__  __device__ void inline_vgpr4_init_zero_4x4x4(VEC s_reg[4][4]) {
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
  asm ("v_mov_b64 %0, 0x0\n\t"
       "v_mov_b64 %1, 0x0\n\t"
       "v_mov_b64 %2, 0x0\n\t"
       "v_mov_b64 %3, 0x0\n\t"
       "v_mov_b64 %4, 0x0\n\t"
       "v_mov_b64 %5, 0x0\n\t"
       "v_mov_b64 %6, 0x0\n\t"
       "v_mov_b64 %7, 0x0\n\t"
       "v_mov_b64 %8, 0x0\n\t"
       "v_mov_b64 %9, 0x0\n\t"
       "v_mov_b64 %10, 0x0\n\t"
       "v_mov_b64 %11, 0x0\n\t"
       "v_mov_b64 %12, 0x0\n\t"
       "v_mov_b64 %13, 0x0\n\t"
       "v_mov_b64 %14, 0x0\n\t"
       "v_mov_b64 %15, 0x0\n\t"
       "v_mov_b64 %16, 0x0\n\t"
       "v_mov_b64 %17, 0x0\n\t"
       "v_mov_b64 %18, 0x0\n\t"
       "v_mov_b64 %19, 0x0\n\t"
       "v_mov_b64 %20, 0x0\n\t"
       "v_mov_b64 %21, 0x0\n\t"
       "v_mov_b64 %22, 0x0\n\t"
       "v_mov_b64 %23, 0x0\n\t"
       "v_mov_b64 %24, 0x0\n\t"
       "v_mov_b64 %25, 0x0\n\t"
       "v_mov_b64 %26, 0x0\n\t"
       "v_mov_b64 %27, 0x0\n\t"
       "v_mov_b64 %28, 0x0\n\t"
       "v_mov_b64 %29, 0x0\n\t"
       "v_mov_b64 %30, 0x0\n\t"
       "v_mov_b64 %31, 0x0\n"
      : "=v"(s_reg[0][0].u64[0]), "=v"(s_reg[0][0].u64[1]), "=v"(s_reg[0][1].u64[0]), "=v"(s_reg[0][1].u64[1]), "=v"(s_reg[0][2].u64[0]), "=v"(s_reg[0][2].u64[1]), "=v"(s_reg[0][3].u64[0]), "=v"(s_reg[0][3].u64[1]), "=v"(s_reg[1][0].u64[0]), "=v"(s_reg[1][0].u64[1]), "=v"(s_reg[1][1].u64[0]), "=v"(s_reg[1][1].u64[1]), "=v"(s_reg[1][2].u64[0]), "=v"(s_reg[1][2].u64[1]), "=v"(s_reg[1][3].u64[0]), "=v"(s_reg[1][3].u64[1]), "=v"(s_reg[2][0].u64[0]), "=v"(s_reg[2][0].u64[1]), "=v"(s_reg[2][1].u64[0]), "=v"(s_reg[2][1].u64[1]), "=v"(s_reg[2][2].u64[0]), "=v"(s_reg[2][2].u64[1]), "=v"(s_reg[2][3].u64[0]), "=v"(s_reg[2][3].u64[1]), "=v"(s_reg[3][0].u64[0]), "=v"(s_reg[3][0].u64[1]), "=v"(s_reg[3][1].u64[0]), "=v"(s_reg[3][1].u64[1]), "=v"(s_reg[3][2].u64[0]), "=v"(s_reg[3][2].u64[1]), "=v"(s_reg[3][3].u64[0]), "=v"(s_reg[3][3].u64[1])
      :);
#else
  uint64_t pk_zero = 0x0;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
          s_reg[i][j].u64[0] = pk_zero;
          s_reg[i][j].u64[1] = pk_zero;
      }
  }
#endif
}

template<typename VEC>
__forceinline__  __device__ void inline_vgpr4_init_zero_4x2x4(VEC s_reg[4][4]) {
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
  asm ("v_mov_b64 %0, 0x0\n\t"
       "v_mov_b64 %1, 0x0\n\t"
       "v_mov_b64 %2, 0x0\n\t"
       "v_mov_b64 %3, 0x0\n\t"
       "v_mov_b64 %4, 0x0\n\t"
       "v_mov_b64 %5, 0x0\n\t"
       "v_mov_b64 %6, 0x0\n\t"
       "v_mov_b64 %7, 0x0\n\t"
       "v_mov_b64 %8, 0x0\n\t"
       "v_mov_b64 %9, 0x0\n\t"
       "v_mov_b64 %10, 0x0\n\t"
       "v_mov_b64 %11, 0x0\n\t"
       "v_mov_b64 %12, 0x0\n\t"
       "v_mov_b64 %13, 0x0\n\t"
       "v_mov_b64 %14, 0x0\n\t"
       "v_mov_b64 %15, 0x0\n\t"
      : "=v"(s_reg[0][0].u64[0]), "=v"(s_reg[0][0].u64[1]), "=v"(s_reg[0][2].u64[0]), "=v"(s_reg[0][2].u64[1]), "=v"(s_reg[1][0].u64[0]), "=v"(s_reg[1][0].u64[1]), "=v"(s_reg[1][2].u64[0]), "=v"(s_reg[1][2].u64[1]), "=v"(s_reg[2][0].u64[0]), "=v"(s_reg[2][0].u64[1]), "=v"(s_reg[2][2].u64[0]), "=v"(s_reg[2][2].u64[1]), "=v"(s_reg[3][0].u64[0]), "=v"(s_reg[3][0].u64[1]), "=v"(s_reg[3][2].u64[0]), "=v"(s_reg[3][2].u64[1])
      :);
#else
  uint64_t pk_zero = 0x0;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
      #pragma unroll
      for (int j = 0; j < 4; j += 2) {
          s_reg[i][j].u64[0] = pk_zero;
          s_reg[i][j].u64[1] = pk_zero;
      }
  }
#endif
}


template<typename VEC>
__forceinline__  __device__ void inline_vgpr4_init_zero_1x4x4(VEC s_reg[1][4]) {
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
  asm ("v_mov_b64 %0, 0x0\n\t"
       "v_mov_b64 %1, 0x0\n\t"
       "v_mov_b64 %2, 0x0\n\t"
       "v_mov_b64 %3, 0x0\n\t"
       "v_mov_b64 %4, 0x0\n\t"
       "v_mov_b64 %5, 0x0\n\t"
       "v_mov_b64 %6, 0x0\n\t"
       "v_mov_b64 %7, 0x0\n\t"
      : "=v"(s_reg[0][0].u64[0]), "=v"(s_reg[0][0].u64[1]), "=v"(s_reg[0][1].u64[0]), "=v"(s_reg[0][1].u64[1]), "=v"(s_reg[0][2].u64[0]), "=v"(s_reg[0][2].u64[1]), "=v"(s_reg[0][3].u64[0]), "=v"(s_reg[0][3].u64[1])
      :);
#endif
}

template<typename VEC>
__forceinline__  __device__ void inline_vgpr4_init_zero_1x2x4(VEC s_reg[1][4]) {
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
  asm ("v_mov_b64 %0, 0x0\n\t"
       "v_mov_b64 %1, 0x0\n\t"
       "v_mov_b64 %2, 0x0\n\t"
       "v_mov_b64 %3, 0x0\n\t"
      : "=v"(s_reg[0][0].u64[0]), "=v"(s_reg[0][0].u64[1]), "=v"(s_reg[0][2].u64[0]), "=v"(s_reg[0][2].u64[1])
      :);
#endif
}


// to simplify f32 -> bf16 conversion, filter some branch
inline __HOST_DEVICE__ bhalf_t inlineasm_float2bfloat16_nonan(const float f) {
	bhalf_t ret;
#if defined(__gfx938__)
    // ret.data = __builtin_hcu_cvt_bf16_f32(f, /*clamp*/false, /*dst_sel*/false);
    *(unsigned short*)&ret = __builtin_hcu_cvt_bf16_f32(f, /*clamp*/false, /*dst_sel*/false);
// #elif __HIP_DEVICE_COMPILE__
// inline asm may lead to spill in scratch memory
#elif 0
    unsigned int help = 0x7fff; // this line can be optimized, so as to use v_add3_u32
    unsigned int tmp;
    asm volatile(
        "v_lshrrev_b32 %0, 16, %1\n\t"
        "v_and_b32 %0, 0x1, %0\n\t"
        : "=v"(tmp)
        : "v"(f)
        :);
    asm volatile(
        "v_add3_u32 %0, %2, %3, %4\n"
        "v_lshrrev_b32 %1, 16, %0\n"
        : "=v"(tmp), "=v"(ret.data)
        : "v"(tmp), "s"(help), "v"(f)
        :);
#else
    // for inf, 0x7f80-0000 + 0x0000-7fff + (0x7f80 & 1) = 0x7f80-7ffff, and >> 16 -> 0x7f80, still inf
    // for nan, no process, for input is from softmax, > 0 and no nan
    // for others, actually, not totally rounding to nearest even, no case of mantissa 1000
    union {
		float fp32;
		unsigned int u32;
	} u = {f};
	u.u32 += 0x7fff + ((u.u32 >> 16) & 1);
	*(unsigned short*)&ret = (u.u32 >> 16);
#endif
	return ret;
}

// this seems to have no provement while writing global memory
inline __HOST_DEVICE__ unsigned short inlineasm_float2bfloat16_ushort_nonan(const float f) {
    bhalf_t ret = inlineasm_float2bfloat16_nonan(f);
    return *(unsigned short*)&ret;
}



// d = a * b + c
inline __device__ __float2 inlineasm_fa_v_pk_fma_f32(__float2 &a, const __float2& b, const __float2& c) {
    __float2 d;
    asm volatile("v_pk_fma_f32 %0, %1, %2, %3  ; inlineasm_fa_v_pk_fma_f32"
               : "=v"(d)
               : "v"(a), "v"(b), "v"(c)
               :);
    return d;
}


inline __device__ void inlineasm_fa_v_mov_b64(__float2 &c, const __float2 &a) {
    asm volatile("v_mov_b64 %0, %1 ; inlineasm_fa_v_mov_b64"
               : "=v"(c)
               : "v"(a)
               :);
}

extern __device__ __attribute__((const)) __float2 __llvm_v_pk_fma_f32(__float2, __float2, __float2) __asm("llvm.fma.v2f32");


inline __device__ void inlineasm_fa_v_pk_mul_f32(__float2 &dst, const __float2 &src, const __float2 &scale) {
    asm volatile("v_pk_mul_f32 %0, %1, %2 ; inlineasm_fa_v_pk_mul_f32"
               : "=v"(dst)
               : "v"(src), "v"(scale)
               :);
}

// c = a + b
inline __device__ void inline_v_pk_add_f32(__float2 &c, const __float2 &a, const __float2& b) {
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
    asm volatile("v_pk_add_f32 %0, %1, %2 ; inline_v_pk_add_f32"
               : "=v"(c)
               : "v"(a), "v"(b)
               :);
#else
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
#endif
}

/*
    原来的 exp2f 对于极小数有特殊处理, 对于小于 -126 的输入 x , exp2f 计算方式是 2^(x + 64) * 2^{-64}
    但是对于深度学习来说, 2^-126 的数字其实没那么重要了, 因此只需要保留 v_exp_f32 直接暴力计算即可
*/
extern __device__ __attribute__((const)) float __llvm_exp2_f32(float) __asm("llvm.exp2.f32");
extern __device__ __attribute__((const)) float __llvm_log2_f32(float) __asm("llvm.log2.f32");
extern __device__ __attribute__((const)) float __llvm_fma_f32(float, float, float) __asm("llvm.fma.f32");
extern __device__ __attribute__((const)) int64_t __builtin_hcu_mov_b64(int64_t) __asm("llvm.hcu.mov64");

/* 直接内联汇编单独测试没问题, 但放到 flash attention 里面结果不对, 很奇怪 */
inline __device__ float inlineasm_fa_v_exp2_f32(const float x) {
    // return exp2f(x);
    float y;
    asm volatile(
              // "s_waitcnt lgkmcnt(0)\n\t"
              "v_exp_f32 %0, %1"
               : "=v"(y)
               : "v"(x)
               :);
    return y;
}



#if !defined(__NVCC__)
// fp8_e5m2
constexpr int32_t e5m2_exp_bits  = 5;
constexpr int32_t e5m2_mant_bits = 2;
constexpr int32_t e5m2_bits      = 8;
constexpr int32_t e5m2_bias      = (1 << (e5m2_exp_bits - 1)) - 1;
// fp8_e4m3
constexpr int32_t e4m3_exp_bits  = 4;
constexpr int32_t e4m3_mant_bits = 3;
constexpr int32_t e4m3_bits      = 8;
constexpr int32_t e4m3_bias      = (1 << (e4m3_exp_bits - 1)) - 1;
// fp16
constexpr int32_t fp16_exp_bits  = 5;
constexpr int32_t fp16_mant_bits = 10;
constexpr int32_t fp16_bits      = 16;
constexpr int32_t fp16_bias      = (1 << (fp16_exp_bits - 1)) - 1;
// fp32
constexpr int32_t fp32_exp_bits  = 8;
constexpr int32_t fp32_mant_bits = 23;
constexpr int32_t fp32_bits      = 32;
constexpr int32_t fp32_bias      = (1 << (fp32_exp_bits - 1)) - 1;

__host__ __device__
static uint8_t __float2e4m3(const float src) {
    // conversion from float to unsigned int(32 bits) for convience to fetching each bit
    uint32_t __src = *(unsigned int*)&src;
    // fetch sign bits
    uint8_t sign_bits = (__src & 0x80000000) >> (fp32_bits - e5m2_bits);
    // fetch exponent bitss
    uint32_t exp_bits  = __src & 0x7f800000;
    // fetch mantissa bits
    uint32_t mant_bits = __src & 0x007fffff;
    // fetch absolute value
    uint32_t data_scale = __src & 0x7fffffff;
    // categorical discussions
    /* NAN */
    uint8_t result = 0x0;
    if (exp_bits == 0x7f800000 and mant_bits > 0x0) {
        // result = sign_bits | 0x7f; // output qNAN
        result = 0x7f; // for NV's __nv_cvt_float_to_fp8:cvt.rn.satfinite.e4m3x2.f32, output are all 0x7f
    }
    /* inf or greater than MAX value of E5M2 */
    else if ((exp_bits == 0x7f800000 and mant_bits == 0x0) or (data_scale > 0x43e00000)) {
        result = sign_bits | 0x7e; // output MAX
    }
    /* less than MIN of denorm */
    else if (exp_bits <= 0x3a800000) {
        result = (exp_bits == 0x3a800000 and mant_bits > 0x0) ? sign_bits | 0x1: sign_bits;
    }
    /* others */
    else {
        /* norm fp32 can be represented as denorm fp8 / norm */
        mant_bits = exp_bits < 0x3c800000 ? (0x800000 | mant_bits) >> ((0x3c800000 - exp_bits) >> fp32_mant_bits): mant_bits;
        exp_bits  = exp_bits < 0x3c800000 ? 0x0: ((exp_bits >> fp32_mant_bits) - (fp32_bias - e4m3_bias)) << e4m3_mant_bits;
        // get discard bits
        uint32_t discard_bits = mant_bits & 0xfffff;
        // rounding
        bool carry_a_bit = discard_bits > 0x80000 or (discard_bits == 0x80000 and (mant_bits & 0x100000));
        mant_bits = (mant_bits & 0x700000) >> (fp32_mant_bits - e4m3_mant_bits);
        mant_bits = carry_a_bit ? mant_bits + 1: mant_bits;
        result = sign_bits + exp_bits + mant_bits; // + rather than |, since mant may carry a bit to exp
    }
    return result;
}

__host__ __device__
static float __e4m32float(const uint8_t src) {
    // initialize ret value
    float result;
    // conversion from float to unsigned int(32 bits) for convience to fetching each bit
    uint8_t __src = *(uint8_t*)&src;
    // fetch sign bits
    uint32_t sign_bits = __src & 0x80;
    // fetch exponent bits
    uint32_t exp_bits  = (__src & 0x78) >> e4m3_mant_bits;
    // fetch mantissa bits
    uint32_t mant_bits = __src & 0x7;
    // denorm or 0
    if (exp_bits == 0x0 and mant_bits >= 0x0) {
        result = 0.0078125f * ((mant_bits & 0x4) >> 2) + 0.00390625f * ((mant_bits & 0x2) >> 1) + 0.001953125f * (mant_bits & 0x1);
        result = sign_bits > 0 ? -result: result;
    } else {
        uint32_t tmp = (exp_bits == 0xf and mant_bits == 0x7)
            ? /*input NaN*/0x7fffffff
            : /*input norm*/(sign_bits << (fp32_bits - e4m3_bits)) + ((exp_bits - e4m3_bias + fp32_bias) << fp32_mant_bits) + (mant_bits << (fp32_mant_bits - e4m3_mant_bits));
        result = *(float*)&tmp;
    }
    return result;
}
#endif // end of #if !defined(__NVCC__)


////////////////////////////////////////////////////////////////////////////////////////////////////
//数据类型转化封装
//DownCast
//fp32转fp16
template<class FromType, class ToType, bool Is_short = false, typename std::enable_if<std::is_same<FromType, float>::value && std::is_same<ToType,half_t>::value, int>::type = 0>
__host__ __device__ ToType DownCast(const FromType &from_var) {
    return __float2half(from_var);
}
//fp32转bf16，并返回其内置数据类型
template<class FromType, class ToType, bool Is_short = false, typename std::enable_if<std::is_same<FromType, float>::value && Is_short && std::is_same<ToType, BFloat16>::value, int>::type = 0>
__host__ __device__ unsigned short DownCast(const FromType &from_var) {
#if defined(__gfx928__) || defined(__gfx936__)
    return inlineasm_float2bfloat16_ushort_nonan(from_var);
#else
    bhalf_t ret = __float2bfloat16(from_var);
    return *(unsigned short*)&ret;
#endif
}
//fp32转bf16，返回其结构体本身
template<class FromType, class ToType, bool Is_short = false, typename std::enable_if<std::is_same<FromType, float>::value && !Is_short && std::is_same<ToType, BFloat16>::value, int>::type = 0>
__host__ __device__ BFloat16 DownCast(const float &from_var) {
#if 1
    return inlineasm_float2bfloat16_nonan(from_var);
#else
    return __float2bfloat16(from_var);
#endif
}
//fp32转fp8，返回其内置数据类型
template<class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<std::is_same<FromType, float>::value && Is_uint8 && std::is_same<ToType, Float8_e4m3_t>::value, int>::type = 0>
__host__ __device__ uint8_t DownCast(const float &from_var) {
    return __float2e4m3(from_var);
}
//fp32转fp8，返回其结构体本身
template<class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<std::is_same<FromType, float>::value && !Is_uint8 && std::is_same<ToType, Float8_e4m3_t>::value, int>::type = 0>
__host__ __device__ Float8_e4m3_t DownCast(const float &from_var) {
    return Float8_e4m3_t(__float2e4m3(from_var));
}
//fp16转fp8，返回其内置数据类型
template<class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<std::is_same<FromType,half_t>::value && Is_uint8 && std::is_same<ToType, Float8_e4m3_t>::value, int>::type = 0>
__host__ __device__ uint8_t DownCast(const half_t &from_var) {
    float src_f32 = __half2float(from_var);
    return __float2e4m3(src_f32);
}
//fp16转fp8，返回其结构体本身
template<class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<std::is_same<FromType,half_t>::value && !Is_uint8 && std::is_same<ToType, Float8_e4m3_t>::value, int>::type = 0>
__host__ __device__ Float8_e4m3_t DownCast(const half_t &from_var) {
    float src_f32 = __half2float(from_var);
    return Float8_e4m3_t(__float2e4m3(src_f32));
}
//fp32转fp16
template<class FromType, class ToType, bool Is_short = false, typename std::enable_if<std::is_same<FromType, float>::value && std::is_same<ToType, float>::value, int>::type = 0>
__host__ __device__ ToType DownCast(const FromType &from_var) {
    return from_var;
}


//UpCast
//fp16转fp32
template<class FromType=half_t, class ToType=float, bool Is_short = false, typename std::enable_if<std::is_same<FromType,half_t>::value && std::is_same<ToType, float>::value, int>::type = 0>
__host__ __device__ float UpCast(const half_t &from_var) {
    return __half2float(from_var);
}
//bf16的内置数据类型转fp32
template<class FromType, class ToType, bool Is_short = false, typename std::enable_if<Is_short && std::is_same<FromType, BFloat16>::value && std::is_same<ToType, float>::value, int>::type = 0>
__host__ __device__ float UpCast(const unsigned short &from_var) {
    bhalf_t x;
    *(unsigned short*)&x = from_var;
    return  __bfloat162float(x);
}
//bf16转fp32
template<class FromType=bhalf_t, class ToType=float, bool Is_short = false,typename std::enable_if<!Is_short && std::is_same<FromType, BFloat16>::value && std::is_same<ToType,float>::value, int>::type = 0>
__host__ __device__ float UpCast(const BFloat16 &from_var) {
    return __bfloat162float(from_var);
}
//fp8的内置数据类型转fp32
template<class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<Is_uint8 && std::is_same<FromType, Float8_e4m3_t>::value && std::is_same<ToType, float>::value, int>::type = 0>
__host__ __device__ float UpCast(const uint8_t &from_var) {
    return __e4m32float(from_var);
}
//fp8转fp32
template<class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<!Is_uint8 && std::is_same<FromType, Float8_e4m3_t>::value && std::is_same<ToType, float>::value, int>::type = 0>
__host__ __device__ float UpCast(const Float8_e4m3_t &from_var) {
    return __e4m32float(from_var.data);
}
//fp8的内置数据类型转fp16
template<class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<Is_uint8 && std::is_same<FromType, Float8_e4m3_t>::value && std::is_same<ToType,half_t>::value, int>::type = 0>
__host__ __device__ half_t UpCast(const uint8_t &from_var) {
    float src_f32 = __e4m32float(from_var);
    return __float2half(src_f32);
}
//fp8转fp16
template<class FromType, class ToType, bool Is_uint8 = false, typename std::enable_if<!Is_uint8 && std::is_same<FromType, Float8_e4m3_t>::value && std::is_same<ToType,half_t>::value, int>::type = 0>
__host__ __device__ half_t UpCast(const Float8_e4m3_t &from_var) {
    float src_f32 = __e4m32float(from_var.data);
    return __float2half(src_f32);
}



////////////////////////////////////////////////////////////////////////////////////////////////////
template<class FromType, class ToType>
inline __host__ __device__ auto DownCastPair(const vec2_Element<FromType>& source) {
    static_assert(false and "No Cvt method for DownCastPair!");
    return vec2_Element<ToType>(0);
}


template<>
inline __host__ __device__ auto DownCastPair<float, half_t>(const vec2_Element<float>& source) {
#if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
    auto result = __builtin_hcu_cvt_pk_f16_f32(source[0], source[1], false/*clamp*/, 0/*o_modifier*/);
    return *(vec2_Element<half_t>*)(&result);
#else
    return __builtin_amdgcn_cvt_pkrtz(source[0], source[1]);
#endif
}

template<>
inline __host__ __device__ auto DownCastPair<float, bhalf_t>(const vec2_Element<float>& source) {
#if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
    auto result = __builtin_hcu_cvt_pk_bf16_f32(source[0], source[1], false/*clamp*/);
    return *(vec2_Element<bhalf_t>*)(&result);
#else
    vec2_Element<bhalf_t> result;
    result[0] = inlineasm_float2bfloat16_ushort_nonan(source[0]);
    result[1] = inlineasm_float2bfloat16_ushort_nonan(source[1]);
    return result;
#endif
}

// Support when src0 and src1 are not contiguously rearranged
template<class FromType, class ToType>
inline __host__ __device__ auto DownCastPairNoPack(const FromType src0, const FromType src1) {
    static_assert(false and "No Cvt method for DownCastPairNoPack!");
    return vec2_Element<ToType>(0);
}

template<>
inline __host__ __device__ auto DownCastPairNoPack<float, half_t>(const float src0, const float src1) {
#if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
    auto result = __builtin_hcu_cvt_pk_f16_f32(src0, src1, false/*clamp*/, 0/*o_modifier*/);
    return *(vec2_Element<half_t>*)(&result);
#else
    return __builtin_amdgcn_cvt_pkrtz(src0, src1);
#endif
}

template<>
inline __host__ __device__ auto DownCastPairNoPack<float, bhalf_t>(const float src0, const float src1) {
#if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
    auto result = __builtin_hcu_cvt_pk_bf16_f32(src0, src1, false/*clamp*/);
    return *(vec2_Element<bhalf_t>*)(&result);
#else
    vec2_Element<bhalf_t> result;
    result[0] = inlineasm_float2bfloat16_ushort_nonan(src0);
    result[1] = inlineasm_float2bfloat16_ushort_nonan(src1);
    return result;
#endif
}

template<>
inline __host__ __device__ auto DownCastPairNoPack<float, float>(const float src0, const float src1) {
    __float2 result;
    result[0] = src0;
    result[1] = src1;
    return *(vec2_Element<float>*)(&result);
}

////////////////////////////////////////////////////////////////////////////////////////////////////


// distinct upcast function to avoid regression, used in splitkv
template<typename accumType, class FromType>
__host__ __device__ float splitkv_upcast_to_f32(const FromType &from_var) {
    if constexpr (std::is_same<FromType, half_t>::value or std::is_same<FromType, __half>::value) {
        return __half2float(from_var);
    } else if constexpr (std::is_same<FromType, __hip_bfloat16>::value) {
        return __bfloat162float(from_var);
    } else if constexpr (std::is_same<FromType, unsigned short>::value) {
        bhalf_t container;
        *(unsigned short*)&container = from_var;
        return __bfloat162float(container);
    } else {
        return from_var;
    }
}


template<typename output_dtype>
__forceinline__ __device__ void __builtin_hcu_cvt_pk4_fp8_f32(const vec4_fp32& source, int32_t &container) {
#if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
    if constexpr (std::is_same<output_dtype, fp8_e4m3>::value) {
        container = __builtin_hcu_cvt_pk_fp8_f32(source[0], source[1], container, false/*op_sel:[0,0,0,0]*/);
        container = __builtin_hcu_cvt_pk_fp8_f32(source[2], source[3], container, true/*op_sel:[0,0,0,1]*/);
    } else if constexpr (std::is_same<output_dtype, fp8_e5m2>::value) {
        container = __builtin_hcu_cvt_pk_bf8_f32(source[0], source[1], container, false/*op_sel:[0,0,0,0]*/);
        container = __builtin_hcu_cvt_pk_bf8_f32(source[2], source[3], container, true/*op_sel:[0,0,0,1]*/);
    } else {
        static_assert (false and "Inputs of invalid dtype fed to __builtin_hcu_cvt_pk4_fp8_f32");
    }
#endif
}
