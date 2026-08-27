#pragma once

#include "hip/hip_fp16.h"
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include "numeric_types.h"
#include "intrinsic.h"



#if defined(__gfx936__) || defined(__gfx938__)
    #define parallel_degree 3
#else
    #define parallel_degree 2
#endif

template<typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result) {
        throw std::runtime_error(std::string("[GPU][ERROR] HIP runtime error: ") + hipGetErrorString(result) + " " + file + ":" + std::to_string(line) + " \n");
    }
}
#define check_hip_error(val) check((val), #val, __FILE__, __LINE__)

namespace flash {

inline __device__ constexpr int ceil_div(int const& a, int const& b) {
    return (a + b - 1) / b;
}

template<class T>
__device__ vec4_fp32 mmac(const vec4_Element<T> &v1, const vec4_Element<T> &v2, const vec4_fp32 &v3)
{
#if defined(__gfx916__) || defined(__gfx926__)
    return {0, 0, 0, 0};
#else
    return __builtin_hcu_mmac_f32_16x16x16_f16(v1, v2, v3);
#endif
}

template<>
__device__ vec4_fp32 mmac<half_t>(const vec4_fp16 &v1, const vec4_fp16 &v2, const vec4_fp32 &v3)
{
#if defined(__gfx916__) || defined(__gfx926__)
    return {0, 0, 0, 0};
#else
    return __builtin_hcu_mmac_f32_16x16x16_f16(v1, v2, v3);
#endif
}

template<>
__device__ vec4_fp32 mmac<bhalf_t>(const vec4_bf16 &v1, const vec4_bf16 &v2, const vec4_fp32 &v3)
{
#if defined(__gfx916__) || defined(__gfx926__)
    return {0, 0, 0, 0};
#else
    return __builtin_hcu_mmac_f32_16x16x16_bf16(v1, v2, v3);
#endif
}


template<typename T>
__forceinline__ __device__ T __shfl_xor_tmp(T x, const int lane_mask) {
    int lane_id = threadIdx.x & 63;
    int index   = (lane_id ^ lane_mask) << 2;
    int res = __builtin_amdgcn_ds_bpermute(index, *(int*)&x); // attention, __builtin only support int
    return *(T*)&res;
}


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

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 64);
    template<typename T, typename Operator>
    static __device__ inline T run(T x, Operator &op) {

        x = op(x, __shfl_xor_tmp(x, 32));
        return op(x, __shfl_xor_tmp(x, 16));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<32> {
template<typename T, typename Operator>
static __device__ inline T run(T x, Operator &op) {
    //x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    x = op(x, __shfl_xor_tmp(x, 16));

    return x;
}
};

template<typename T, int WARP_M>
void copy(T *src, T *dst) {
    for(int i=0; i<(WARP_M/16); i++) {
        dst[i] = src[i];
    }
}

//TODO:后续优化得用上V_CVT_PKRTZ_FP16_FP32
//QK(seq_q, seq_k), two float in seq_k dim convert to two half, and packed into a U32
template <int BLOCK_M, int WARP_N, typename ElementType>
inline __device__ void convert_pk_type(union_vec2_f16x2<ElementType> p_reg[(BLOCK_M/32)*(WARP_N/32)][4], union_vec4_fp32 s_reg[(BLOCK_M/32)*(WARP_N/32)][4]) {
    #pragma unroll
    for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
        #pragma unroll
        for(int m_idx=0; m_idx<(BLOCK_M/32); m_idx++) {
            #pragma unroll
            for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                #pragma unroll
                for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                    // for(int vec_idx=0; vec_idx<4; vec_idx++) {
                    //     p_reg[n_idx + m_idx*(WARP_N/32)][min_tile_n*2 + min_tile_m].f16[vec_idx] =  DownCast<float,ElementType,true>(s_reg[n_idx + m_idx*(WARP_N/32)][min_tile_n*2 + min_tile_m].f32[vec_idx]);
                    // }
                    for(int vec_idx=0; vec_idx<2; vec_idx++) {
                        p_reg[n_idx + m_idx*(WARP_N/32)][min_tile_n*2 + min_tile_m].f16x2[vec_idx][0] =  DownCast<float,ElementType,true>(s_reg[n_idx + m_idx*(WARP_N/32)][min_tile_n*2 + min_tile_m].f32[vec_idx*2]);
                        p_reg[n_idx + m_idx*(WARP_N/32)][min_tile_n*2 + min_tile_m].f16x2[vec_idx][1] =  DownCast<float,ElementType,true>(s_reg[n_idx + m_idx*(WARP_N/32)][min_tile_n*2 + min_tile_m].f32[vec_idx*2+1]);
                    }
                }
            }
        }
    }
}

template <int BLOCK_M, int WARP_N, typename ElementType>
inline __device__ void convert_pk_type_zero_negative(union_vec2_f16x2<ElementType> p_reg[(BLOCK_M/32)*(WARP_N/32)][4], union_vec4_fp32 s_reg[(BLOCK_M/32)*(WARP_N/32)][4]) {
    #pragma unroll
    for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
        #pragma unroll
        for(int m_idx=0; m_idx<(BLOCK_M/32); m_idx++) {
            #pragma unroll
            for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                #pragma unroll
                for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                    #pragma unroll
                    for(int vec_idx=0; vec_idx<2; vec_idx++) {
                        const float p0 = s_reg[n_idx + m_idx*(WARP_N/32)][min_tile_n*2 + min_tile_m].f32[vec_idx*2];
                        const float p1 = s_reg[n_idx + m_idx*(WARP_N/32)][min_tile_n*2 + min_tile_m].f32[vec_idx*2+1];
                        p_reg[n_idx + m_idx*(WARP_N/32)][min_tile_n*2 + min_tile_m].f16x2[vec_idx][0] =
                            DownCast<float,ElementType,true>(p0 < 0.0f ? 0.0f : p0);
                        p_reg[n_idx + m_idx*(WARP_N/32)][min_tile_n*2 + min_tile_m].f16x2[vec_idx][1] =
                            DownCast<float,ElementType,true>(p1 < 0.0f ? 0.0f : p1);
                    }
                }
            }
        }
    }
}

inline __device__ bool is_dropout_encoded(float value) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    return (bits.u & 0x80000000u) != 0;
}

inline __device__ float clear_dropout_encoded_sign(float value) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    bits.u &= 0x7fffffffu;
    return bits.f;
}

template <int BLOCK_M, int WARP_N, typename ElementType>
inline __device__ void convert_pk_type_zero_negative_transposed_sign(union_vec2_f16x2<ElementType> p_reg[(BLOCK_M/32)*(WARP_N/32)][4], union_vec4_fp32 s_reg[(BLOCK_M/32)*(WARP_N/32)][4]) {
    #pragma unroll
    for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
        #pragma unroll
        for(int m_idx=0; m_idx<(BLOCK_M/32); m_idx++) {
            #pragma unroll
            for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                #pragma unroll
                for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                    #pragma unroll
                    for(int vec_idx=0; vec_idx<2; vec_idx++) {
                        const int reg_idx = n_idx + m_idx*(WARP_N/32);
                        const int value_idx = min_tile_n*2 + min_tile_m;
                        const int sign_idx = min_tile_m*2 + min_tile_n;
                        const float p0 = clear_dropout_encoded_sign(s_reg[reg_idx][value_idx].f32[vec_idx*2]);
                        const float p1 = clear_dropout_encoded_sign(s_reg[reg_idx][value_idx].f32[vec_idx*2+1]);
                        const bool drop0 = is_dropout_encoded(s_reg[reg_idx][sign_idx].f32[vec_idx*2]);
                        const bool drop1 = is_dropout_encoded(s_reg[reg_idx][sign_idx].f32[vec_idx*2+1]);
                        p_reg[reg_idx][value_idx].f16x2[vec_idx][0] =
                            DownCast<float,ElementType,true>(drop0 ? 0.0f : p0);
                        p_reg[reg_idx][value_idx].f16x2[vec_idx][1] =
                            DownCast<float,ElementType,true>(drop1 ? 0.0f : p1);
                    }
                }
            }
        }
    }
}

//TODO:后续优化得用上V_CVT_PKRTZ_FP16_FP32
//QK(seq_q, seq_k), two float in seq_k dim convert to two half, and packed into a U32
template <int BLOCK_M, int WARP_N, typename ElementType>
inline __device__ void convert_pk_type_gfx938(union_vec4_f16x2<ElementType> p_reg[(BLOCK_M/32)*(WARP_N/32)*2], union_vec4_fp32 s_reg[(BLOCK_M/32)*(WARP_N/32)][4]) {
    #pragma unroll
    for(int n_idx=0; n_idx<(WARP_N/32); n_idx++) {
        #pragma unroll
        for(int m_idx=0; m_idx<(BLOCK_M/32); m_idx++) {
            #pragma unroll
            for(int min_tile_n=0; min_tile_n<2; min_tile_n++) {
                #pragma unroll
                for(int min_tile_m=0; min_tile_m<2; min_tile_m++) {
                    for(int vec_idx=0; vec_idx<4; vec_idx++) {
                        p_reg[(n_idx + m_idx*(WARP_N/32)) * 2 + min_tile_n].f16[min_tile_m * 4 + vec_idx] =  DownCast<float,ElementType,false>(s_reg[n_idx + m_idx*(WARP_N/32)][min_tile_n*2 + min_tile_m].f32[vec_idx]);
                        // p_reg[(n_idx + m_idx*(WARP_N/32)) * 2 + min_tile_n].f16[min_tile_m * 4 + vec_idx] =  s_reg[n_idx + m_idx*(WARP_N/32)][min_tile_n*2 + min_tile_m].f32[vec_idx];
                    }
                }
            }
        }
    }
}

template<const int kHeadDim, typename T>
__device__ __forceinline__ vec4_uint tcp_cache_swizzle_func(T* ptr) {
    vec4_uint res;
    *(uint64_t*)&res = reinterpret_cast<uint64_t>(ptr);
    if constexpr (kHeadDim == 196) {
        res[1] += 0x41800000; // 62 bit: cache swizzle;  48~61: Stride
    } else if constexpr (kHeadDim == 128) {
        res[1] += 0x41000000; // stride 256 Bytes and change tagram
    } else if constexpr (kHeadDim == 64) {
        res[1] += 0x40800000; // stride 128 Bytes and change tagram
    }
    res[2] = 0x80000000;
    res[3] = 0x00020000;
    return res;
}

template<typename T>
__device__ __forceinline__ vec4_uint prepare_for_matrix_load_gfx938(T* ptr, int row_stride) {
    vec4_uint srsrc;
    *(uint64_t*)&srsrc = reinterpret_cast<uint64_t>(ptr);
    srsrc[2] = row_stride;
    srsrc[3] = 0;
    return srsrc;
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

//封装buffer_load
template<int Is_M_equal,int WARP_NUM,int N_row_len,int M,int N,typename Element>
__forceinline__ __device__ void buffer_load_lds_tile(vec4_uint global_ptr, Element* lds_ptr, int global_offset, int lds_stage_offset, int max_M_len, int warp_id, int lane_id) {
    int bytes_per_Element = 2;
    if constexpr (std::is_same<Element, int8_t>::value || std::is_same<Element, Float8_e4m3_t>::value) {
        bytes_per_Element = 1;
    }
    int Elment_per_dword = 4/bytes_per_Element;
    //M维度index变换，(0, 1, 2, 3) --> (0, 2, 1, 3)
    int lane_M_idx = ((lane_id >> 4) & 1)*2 + ((lane_id >> 4) >> 1);
    int lane_N_idx = lane_id & 15;
    const int lds_load_num = (M*N*bytes_per_Element) / (4*64);
    // for(int warp_loop=warp_id; warp_loop<lds_load_num; warp_loop+=WARP_NUM) {
    for(int load = 0,warp_loop = warp_id; load < lds_load_num/WARP_NUM; warp_loop += WARP_NUM, ++load) {
        int warp_buffer_load_lds_offset     =  lds_stage_offset + warp_loop * (4*32);
        int gsOffset   = global_offset/Elment_per_dword;
        int gvOffset;
        if constexpr (Is_M_equal){
            gvOffset   = (warp_loop * 4 + lane_M_idx) * N_row_len/Elment_per_dword  +  lane_N_idx;
        } else {
            gvOffset   = (min(warp_loop * 4 + lane_M_idx, max_M_len - 1) * N_row_len)/Elment_per_dword  +  lane_N_idx;
        }
        int lds_offset = warp_buffer_load_lds_offset/Elment_per_dword;
        builtin_buffer_load_dword_lds(lds_ptr, global_ptr, lds_offset, gsOffset, gvOffset);
    }
}
//封装buffer_load
template<int Is_M_equal,int WARP_NUM,int N_row_len,int M,int N,typename Element>
__forceinline__ __device__ void buffer_load_lds_tile_pad(vec4_uint global_ptr, Element* lds_ptr, int global_offset, int lds_stage_offset, int max_M_len, int warp_id, int lane_id) {
    int bytes_per_Element = 2;
    if constexpr (std::is_same<Element, int8_t>::value || std::is_same<Element, Float8_e4m3_t>::value) {
        bytes_per_Element = 1;
    }
    int Elment_per_dword = 4/bytes_per_Element;
    //M维度index变换，(0, 1, 2, 3) --> (0, 2, 1, 3)
    int lane_M_idx = ((lane_id >> 4) & 1)*2 + ((lane_id >> 4) >> 1);
    int lane_N_idx = lane_id & 15;
    const int lds_load_num = (M*N*bytes_per_Element) / (4*64);
    for(int load = 0,warp_loop = warp_id; load < lds_load_num/WARP_NUM; warp_loop += WARP_NUM, ++load) {
        int padding = (warp_loop & 7)*2; // padding size in shared memory per buffer load, to avoid bank conflict
        int warp_buffer_load_lds_offset     =  lds_stage_offset  + ((warp_loop >> 3)*(32*34) + ( warp_loop & 7)*(4*32));
        int gsOffset   = global_offset/Elment_per_dword;
        int gvOffset;
        if constexpr (Is_M_equal){
            gvOffset   = (warp_loop * 4 + lane_M_idx) * N_row_len/Elment_per_dword  +  lane_N_idx;
        } else {
            gvOffset   = (min(warp_loop * 4 + lane_M_idx, max_M_len - 1) * N_row_len)/Elment_per_dword  +  lane_N_idx;
        }
        int lds_offset = (warp_buffer_load_lds_offset + padding)/Elment_per_dword;
        builtin_buffer_load_dword_lds(lds_ptr, global_ptr, lds_offset, gsOffset, gvOffset);
    }
}

//封装ds_read
template<int M, int N, int WARP_NUM, typename Element>
__forceinline__ __device__ void ds_read_tile_pad(vec2_Element<Element>* lds_v2fp16, int lds_stage_offset, union_vec2_f16x2<Element> (*reg)[2], int loop, int warp_id, int lane_id){
    #pragma unroll
    for(int m_idx = 0; m_idx < M / 32; m_idx ++){
        #pragma unroll
        for(int n_idx = 0; n_idx < N / 32; n_idx ++){
            #pragma unroll
            for(int i=0; i<2; i++) {
                #pragma unroll
                for(int j=0; j<4; j++) {
                    int lds_offset = lds_stage_offset + n_idx*M*17 + (warp_id*(M/32) + m_idx)*(N*17) + j*2 + i*32 + (lane_id & 1)*16 + ((lane_id & 15)>>1)*64 + /*padding*/ ((lane_id & 15)>>1) + ((lane_id/16) &1)*8 + (lane_id/32);
                    inline_ds_read_b32_wait(lds_v2fp16, lds_offset, reg[(loop)*((M*N)/(32*32))*2 + (n_idx*(M/32) + m_idx)*2 + i][j/2].f16x2[j%2]);
                }
            }
        }
    }
}

//封装ds_read2
template<int M, int N, int WARP_NUM, typename Element>
__forceinline__ __device__ void ds_read2_tile_pad_no_wait(vec2_Element<Element>* lds_v2fp16, int lds_stage_offset, union_vec2_f16x2<Element> (*reg)[2], int loop, int warp_id, int lane_id){
    #pragma unroll
    for(int m_idx = 0; m_idx < M / 32; m_idx ++){
        #pragma unroll
        for(int n_idx = 0; n_idx < N / 32; n_idx ++){
            #pragma unroll
            for(int i=0; i<2; i++) {
                #pragma unroll
                for(int j=0; j<2; j++) {
                    int lds_offset = lds_stage_offset + n_idx*M*17 + (warp_id*(M/32) + m_idx)*(N*17) + j*4 + i*32 + (lane_id & 1)*16 + ((lane_id & 15)>>1)*64 + /*padding*/ ((lane_id & 15)>>1) + ((lane_id/16) &1)*8 + (lane_id/32);
                    inline_ds_read2_b32_no_wait(lds_v2fp16, lds_offset, reg[(loop)*((M*N)/(32*32))*2 + (n_idx*(M/32) + m_idx)*2 + i][j].f32, 2);
                }
            }
        }
    }
}




//封装buffer_load
#define buffer_load_lds_tile_pad(Is_M_equal, WARP_NUM, N_row_len, M, N, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id)\
{\
    int bytes_per_Element = 2;\
    if constexpr (std::is_same<Element, int8_t>::value || std::is_same<Element, Float8_e4m3_t>::value) {\
        bytes_per_Element = 1;\
    }\
    int Elment_per_dword = 4/bytes_per_Element;\
    int lane_M_idx = ((lane_id >> 4) & 1)*2 + ((lane_id >> 4) >> 1);\
    int lane_N_idx = lane_id & 15;\
    const int lds_load_num = (M*N*bytes_per_Element) / (4*64);\
    for(int load = 0,warp_loop = warp_id; load < lds_load_num/WARP_NUM; warp_loop += WARP_NUM, ++load) {\
        int padding = (warp_loop & 7);\
        int gsOffset   = global_offset/Elment_per_dword;\
        int gvOffset;\
        if constexpr (Is_M_equal){\
            gvOffset   = (warp_loop * 4 + lane_M_idx) * N_row_len/Elment_per_dword  +  lane_N_idx;\
        } else {\
            gvOffset   = (min(warp_loop * 4 + lane_M_idx, max_M_len - 1) * N_row_len)/Elment_per_dword  +  lane_N_idx;\
        }\
        int lds_offset = lds_stage_offset/Elment_per_dword + padding + warp_loop * 64;\
        builtin_buffer_load_dword_lds(lds_ptr, global_ptr, lds_offset, gsOffset, gvOffset);\
    }\
}

//封装buffer_load
#define buffer_load_lds_tile_pad_1(Is_M_equal, WARP_NUM, N_row_len, M, N, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id)\
{\
    int bytes_per_Element = 2;\
    if constexpr (std::is_same<Element, int8_t>::value || std::is_same<Element, Float8_e4m3_t>::value) {\
        bytes_per_Element = 1;\
    }\
    int Elment_per_dword = 4/bytes_per_Element;\
    int lane_M_idx = ((lane_id >> 4) & 1)*2 + ((lane_id >> 4) >> 1);\
    int lane_N_idx = lane_id & 15;\
    const int lds_load_num = (M*N*bytes_per_Element) / (4*64);\
    for(int load = 0,warp_loop = warp_id; load < lds_load_num/WARP_NUM; warp_loop += WARP_NUM, ++load) {\
        int padding = (warp_loop & 7);\
        int gsOffset   = global_offset/Elment_per_dword;\
        int gvOffset;\
            gvOffset   = (warp_loop * 4 + lane_M_idx) * N_row_len/Elment_per_dword  +  lane_N_idx;\
        int lds_offset = lds_stage_offset/Elment_per_dword + padding + warp_loop * 64;\
        builtin_buffer_load_dword_lds(lds_ptr, global_ptr, lds_offset, gsOffset, gvOffset);\
    }\
}

//封装buffer_load
#define buffer_load_lds_tile(Is_M_equal, WARP_NUM, N_row_len, M, N, Element, global_ptr, lds_ptr, global_offset, lds_stage_offset, max_M_len, warp_id, lane_id)\
{\
    int bytes_per_Element = 2;\
    if constexpr (std::is_same<Element, int8_t>::value || std::is_same<Element, Float8_e4m3_t>::value) {\
        bytes_per_Element = 1;\
    }\
    int Elment_per_dword = 4/bytes_per_Element;\
    int lane_M_idx = ((lane_id >> 4) & 1)*2 + ((lane_id >> 4) >> 1);\
    int lane_N_idx = lane_id & 15;\
    const int lds_load_num = (M*N*bytes_per_Element) / (4*64);\
    for(int load = 0,warp_loop = warp_id; load < lds_load_num/WARP_NUM; warp_loop += WARP_NUM, ++load) {\
        int gsOffset   = global_offset/Elment_per_dword;\
        int gvOffset;\
        if constexpr (Is_M_equal){\
            gvOffset   = (warp_loop * 4 + lane_M_idx) * N_row_len/Elment_per_dword  +  lane_N_idx;\
        } else {\
            gvOffset   = (min(warp_loop * 4 + lane_M_idx, max_M_len - 1) * N_row_len)/Elment_per_dword  +  lane_N_idx;\
        }\
        int lds_offset = lds_stage_offset/Elment_per_dword + warp_loop * 64;\
        builtin_buffer_load_dword_lds(lds_ptr, global_ptr, lds_offset, gsOffset, gvOffset);\
    }\
}

#define ds_read_tile_pad(M, N, WARP_NUM, Element, lds_v2fp16, lds_stage_offset, reg, loop, warp_id, lane_id)\
{\
    for(int m_idx = 0; m_idx < M / 32; m_idx ++){\
        for(int n_idx = 0; n_idx < N / 32; n_idx ++){\
            for(int i=0; i<2; i++) {\
                for(int j=0; j<4; j++) {\
                    int lds_offset = lds_stage_offset + n_idx*M*17 + (warp_id*(M/32) + m_idx)*(N*17) + j*2 + i*32 + (lane_id & 1)*16 + ((lane_id & 15)>>1)*64 + /*padding*/ ((lane_id & 15)>>1) + ((lane_id/16) &1)*8 + (lane_id/32);\
                    inline_ds_read_b32_wait(lds_v2fp16, lds_offset, reg[(loop)*((M*N)/(32*32))*2 + (n_idx*(M/32) + m_idx)*2 + i][j/2].f16x2[j%2]);\
                }\
            }\
        }\
    }\
}

#define ds_read2_tile_pad_no_wait(M,N,WARP_NUM,Element,lds_v2fp16,precompute_offset,reg,loop,warp_id,lane_id)\
{\
for(int m_idx = 0; m_idx < M / 32; m_idx ++){\
    for(int n_idx = 0; n_idx < N / 32; n_idx ++){\
        for(int i=0; i<2; i++) {\
            for(int j=0; j<2; j++) {\
                inline_ds_read2_b32_no_wait(lds_v2fp16, precompute_B_lds_offset[i*2+j], reg[(loop)*((M*N)/(32*32))*2 + (n_idx*(M/32) + m_idx)*2 + i][j].f32, 2);            \
            }\
        }\
    }\
}\
}

#define ds_offset_cast(OFFSET) \
static_cast<int>(reinterpret_cast<uintptr_t>(OFFSET) & 0xFFFFFFFF)
}
