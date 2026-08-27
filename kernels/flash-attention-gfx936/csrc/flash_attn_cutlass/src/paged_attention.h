#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define WARP_SIZE 64
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define max_tmp_offset 5000000
#define out_tmp_offset 10000000
// Input validation macros (consistent with flash_api.cpp and flash_api_sparse.cpp)
#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
using uint8x4_t = __attribute__( (__vector_size__(4 * sizeof(uint8_t)) )) uint8_t;
using half4_t = __attribute__( (__vector_size__(4 * sizeof(_Float16)) )) _Float16;
using half8_t = __attribute__( (__vector_size__(8 * sizeof(_Float16)) )) _Float16;
using v4bh = __attribute__( (__vector_size__(4 * sizeof(short)) )) short;
using float4_t = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
using float2_t = __attribute__( (__vector_size__(2 * sizeof(float)) )) float;
using intx2 = __attribute__( (__vector_size__(2 * sizeof(int)) )) int;
using intx4 = __attribute__( (__vector_size__(4 * sizeof(int)) )) int;


// static constexpr int PARTITION_SIZE=512;
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
template<typename scalar_t> 
static __device__ inline void from_float(scalar_t &out ,float f){
  if constexpr(std::is_same<scalar_t, _Float16>::value||std::is_same<scalar_t, float>::value){
    out=f;
  }
  else{
    uint32_t u = *(uint32_t*)(&f);
    // u += 0x7fff + ((u >> 16) & 1); 
    u += 0x8000; 
    out = u>>16;
  }
}

template<typename scalar_t> 
static __device__ inline float to_float(scalar_t in){
  if constexpr(std::is_same<scalar_t, _Float16>::value||std::is_same<scalar_t, float>::value){
    return in;
  }
  else{
    union{
        uint32_t int32;
        float    fp32;
    } u = {uint32_t(in) << 16};
    return u.fp32;
  }
}


inline __device__ float uint82float(const uint8_t& input) {
#if (defined(__gfx938__) ||defined(__gfx92a__))
  return __builtin_hcu_cvt_f32_fp8(input,false,0,0);
#else
  const uint32_t w = (uint32_t)input << 24;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  uint32_t renorm_shift = __clz(nonsign);
  renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;
  uint32_t result = sign | ((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23));
  return c10::detail::fp32_from_bits(result);
#endif
}
template<typename scalar_t,bool is_e4m3>
__forceinline__ __device__ scalar_t uint82half(const uint8_t& input) {
  union uf16{
    uint16_t as_bits;
    _Float16 as_value;
  } ;
  union uf32 {
    uint32_t as_bits;
    float as_value;
  };
  if constexpr(!is_e4m3){
    uf16 u16;
    u16.as_bits = (uint16_t)input << 8;
    if constexpr(std::is_same<scalar_t, _Float16>::value){
      return u16.as_value;
    }
    else{
      uf32 u32;
      u32.as_value = (float)u16.as_value;
      return u32.as_bits>>16;
    }
  }
  else{
    uf32 u32;
    u32.as_value = uint82float(input);
    if constexpr(std::is_same<scalar_t, _Float16>::value){
      return (_Float16)(u32.as_value);
    }
    else{
      return (uint16_t)(u32.as_bits >> 16);
    }
  }
}
template <bool is_e4m3>
static __device__ int to_f8_from_f32(float v1,float v2,float v3,float v4) {
  int val=0;
  #if (defined(__gfx938__) || defined(__gfx92a__))
  if constexpr(is_e4m3){
    val = __builtin_hcu_cvt_pk_fp8_f32(v1,v2,val,false);
    val = __builtin_hcu_cvt_pk_fp8_f32(v3,v4,val,true);
  }
  else{
    val = __builtin_hcu_cvt_pk_bf8_f32(v1,v2,val,false);
    val = __builtin_hcu_cvt_pk_bf8_f32(v3,v4,val,true);
  }
  #endif
  return val;
}

template <bool is_e4m3>
static __device__ float4_t to_fp32_from_fp8(int val) {
  float4_t ret;
  #if (defined(__gfx938__) || defined(__gfx92a__))
  if constexpr(is_e4m3){
    ret[0] = __builtin_hcu_cvt_f32_fp8(val,false,0,0);
    ret[1] = __builtin_hcu_cvt_f32_fp8(val,false,0,1);
    ret[2] = __builtin_hcu_cvt_f32_fp8(val,false,0,2);
    ret[3] = __builtin_hcu_cvt_f32_fp8(val,false,0,3);
  }
  else{
    ret[0] = __builtin_hcu_cvt_f32_bf8(val,false,0,0);
    ret[1] = __builtin_hcu_cvt_f32_bf8(val,false,0,1);
    ret[2] = __builtin_hcu_cvt_f32_bf8(val,false,0,2);
    ret[3] = __builtin_hcu_cvt_f32_bf8(val,false,0,3);
  }
  #endif
  return ret;
}

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define Output_Type_SWITCH(SRC_DTYPE, ...)     \
  [&] {                                       \
    if (SRC_DTYPE == at::ScalarType::Half) {  \
      using scalar_t=_Float16;                \
      return __VA_ARGS__();                   \
    }else {                                   \
      using scalar_t=uint16_t;                \
      return __VA_ARGS__();                   \
    }                                         \
  }()

#define Cache_Type_SWITCH(scalar_t,dtype, ...)      \
  [&] {                                             \
    if(dtype==torch::kFloat8_e5m2){                 \
      using cache_t=uint8_t;                        \
      constexpr bool is_e4m3=false;                 \
      return __VA_ARGS__();                         \
    }else if(dtype==torch::kFloat8_e4m3fn){         \
      using cache_t=uint8_t;                        \
      constexpr bool is_e4m3=true;                  \
      return __VA_ARGS__();                         \
    }else {                                         \
      using cache_t=scalar_t;                       \
      constexpr bool is_e4m3=false;                 \
      return __VA_ARGS__();                         \
    }                                               \
  }()

#define Input_Type_SWITCH(scalar_t,qdtype,kdtype,...)   \
  [&] {                                                 \
    if(qdtype==torch::kFloat8_e5m2){                    \
      constexpr bool is_e4m3=false;                     \
      using q_type = uint8_t;                           \
      return __VA_ARGS__();                             \
    }else if(qdtype==torch::kFloat8_e4m3fn){            \
      constexpr bool is_e4m3=true;                      \
      using q_type = uint8_t;                           \
      return __VA_ARGS__();                             \
    }else if(kdtype==torch::kFloat8_e5m2){              \
      constexpr bool is_e4m3=false;                     \
      using q_type = scalar_t;                          \
      return __VA_ARGS__();                             \
    }else{                                              \
      constexpr bool is_e4m3=true;                      \
      using q_type = scalar_t;                          \
      return __VA_ARGS__();                             \
    }                                                   \
  }()

#define REUSEKV_SWITCH(reusekv,...)                     \
[&] {                                                   \
    if (reusekv==64){                                   \
        constexpr static int REUSE_KV_TIMES = 64;       \
        return __VA_ARGS__();                           \
    }else if (reusekv==48){                             \
        constexpr static int REUSE_KV_TIMES = 48;       \
        return __VA_ARGS__();                           \
        }else if (reusekv==32){                         \
        constexpr static int REUSE_KV_TIMES = 32;       \
        return __VA_ARGS__();                           \
    }else if (reusekv==24){                             \
        constexpr static int REUSE_KV_TIMES = 24;       \
        return __VA_ARGS__();                           \
    }else if (reusekv==16){                             \
        constexpr static int REUSE_KV_TIMES = 16;       \
        return __VA_ARGS__();                           \
    }else if (reusekv==8){                              \
        constexpr static int REUSE_KV_TIMES = 8;        \
        return __VA_ARGS__();                           \
    }else                {                              \
        constexpr static int REUSE_KV_TIMES = 4;        \
        return __VA_ARGS__();                           \
    }                                                   \
}()

#define HEADSIZE_SWITCH(headsize,...)                   \
[&] {                                                   \
    if (headsize==64){                                  \
        constexpr static int HEAD_SIZE = 64;            \
        return __VA_ARGS__();                           \
    }else if(headsize==128){                            \
        constexpr static int HEAD_SIZE = 128;           \
        return __VA_ARGS__();                           \
    }else if(headsize==192){                            \
        constexpr static int HEAD_SIZE = 192;           \
        return __VA_ARGS__();                           \
    }else {                                             \
        constexpr static int HEAD_SIZE = 256;           \
        return __VA_ARGS__();                           \
    }                                                   \
}()

extern const std::string device_name;
extern const int PA_PRINT_PARAM ;
extern const int PA_PARTITION_SIZE ;


template<int vec> 
struct half4vec{
  half4_t data[vec];
};
using half4x2 = half4vec<2>;
using half4x4 = half4vec<4>;

template<int vec> 
struct int2vec{
  intx2 data[vec];
};

template<int vec> 
struct uint8x4vec{
  uint8x4_t data[vec];
};

using uint8x4x2 = uint8x4vec<2>;
using uint8x4x4 = uint8x4vec<4>;

template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  int warp = __builtin_amdgcn_readfirstlane(threadIdx.x / WARP_SIZE);
  int lane = threadIdx.x % WARP_SIZE;
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor(sum, mask);
  }
  if (lane == 0) {
    red_smem[warp] = sum;
  }
  __syncthreads();
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor(sum, mask);
  }
  return __shfl(sum, 0);
}

template<bool is_half>
inline __device__ void builtin_amdgcn_mmac(const half4_t& reg_a, const half4_t& reg_b, float4_t& reg_c)
{
  if constexpr (is_half){
    reg_c=__builtin_hcu_mmac_f32_16x16x16_f16(reg_a,reg_b,reg_c);
  }else{
    reg_c=__builtin_hcu_mmac_f32_16x16x16_bf16(*(v4bh*)&reg_a,*(v4bh*)&reg_b,reg_c);
  }
}

template<bool is_e4m3>
inline __device__ void builtin_amdgcn_mmac_fp8(const intx2& reg_a, const intx2& reg_b, float4_t& reg_c)
{
  if constexpr(is_e4m3){
    reg_c=__builtin_hcu_mmac_f32_16x16x32_fp8_fp8(reg_a,reg_b,reg_c);
  }else{
    reg_c=__builtin_hcu_mmac_f32_16x16x32_bf8_bf8(reg_a,reg_b,reg_c);
  }
}
  


template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS, 1) void paged_attention_combine(
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
    scalar_t* out_tmp,  // [num_seqs, num_heads,
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_partitions,
    int num_heads,
    int PARTITION_SIZE) {
  extern __shared__ char shared_mem[];
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int seq_len = __builtin_amdgcn_readfirstlane(seq_lens[seq_idx]);
  const int lane = threadIdx.x;
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);
  if(num_partitions==1)return;
  float* shared_exp_sums=reinterpret_cast<float*>(shared_mem);
  float* shared_max_logits=shared_exp_sums+num_partitions;
  float max_logit = -FLT_MAX;
  float global_exp_sum = 0.0f;
  int offset = seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions;
  const float * exp_sums=reinterpret_cast<float*>(out_tmp);
  const float * max_logits=reinterpret_cast<float*>(out_tmp+max_tmp_offset);
  const float* max_logits_ptr = max_logits + offset;
  const float* exp_sums_ptr = exp_sums + offset;
  const scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
  const scalar_t* tmp_out_ptr = out_tmp + out_tmp_offset + offset* HEAD_SIZE;
  for(int i=lane;i<num_partitions;i+=WARP_SIZE){
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l; 
    max_logit = fmaxf(max_logit,l);
  }
  #pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, __shfl_xor(max_logit, mask));
  }
  for(int i=lane;i<num_partitions;i+=WARP_SIZE){
    float rescaled_exp_sum = exp_sums_ptr[i] * __builtin_amdgcn_exp2f(shared_max_logits[i] - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  #pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    global_exp_sum += __shfl_xor(global_exp_sum, mask);
  }
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);
  constexpr int vec_size_o=HEAD_SIZE/64;
  constexpr int vec_size = vec_size_o==3?4:vec_size_o;
  using half_vec= __attribute__( (__vector_size__(vec_size * sizeof(scalar_t)) )) scalar_t;
  using float_vec= __attribute__( (__vector_size__(vec_size * sizeof(float)) )) float;
  float_vec acc = {0.0f};
  half_vec acc_half;
  if(lane<HEAD_SIZE/vec_size){
    for (int j = 0; j < num_partitions; ++j) {
      half_vec tout= *(half_vec*)(tmp_out_ptr + j * HEAD_SIZE + lane * vec_size);
      float temp_sum=shared_exp_sums[j]*inv_global_exp_sum;
      #pragma unroll
      for(int i=0;i<vec_size;i++){
        acc[i] += to_float(tout[i])*temp_sum;
      }
    }
    #pragma unroll
    for(int i=0;i<vec_size;i++){
      scalar_t temp;
      from_float(temp,acc[i]);
      acc_half[i]=temp;
    }
    *(half_vec*)(out_ptr+lane*vec_size)=acc_half;
  }
}

int get_reusekv(int qhead,int kv_head);

void paged_attention_fp8_bhsd(
    torch::Tensor& out,    // [num_seqs,seqlen, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& value_cache,// [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    const c10::optional<torch::Tensor>& alibi_slopes,
    const c10::optional<torch::Tensor>& q_scale,
    const c10::optional<torch::Tensor>& k_scale,
    const c10::optional<torch::Tensor>& v_scale,
    int max_seq_len,
    const c10::optional<at::Tensor> &s_aux_,
    float *tmp_out_ptr,
    int PARTITION_SIZE);  // ★ Attention Sinks ★

void paged_attention_fp8(
    torch::Tensor& out,    // [num_seqs,seqlen, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& value_cache,// [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    const c10::optional<torch::Tensor>& alibi_slopes,
    const c10::optional<torch::Tensor>& q_scale,
    const c10::optional<torch::Tensor>& k_scale,
    const c10::optional<torch::Tensor>& v_scale,
    int max_seq_len,
    const c10::optional<at::Tensor> &s_aux_,
    float *tmp_out_ptr,
    int PARTITION_SIZE);  // ★ Attention Sinks ★

void paged_attention_bhsd(
    torch::Tensor& out,    // [num_seqs,seqlen, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& value_cache,// [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    const c10::optional<torch::Tensor>& alibi_slopes,
    int max_seq_len,
    const c10::optional<at::Tensor> &s_aux_,
    float *tmp_out_ptr,
    int PARTITION_SIZE);  // ★ Attention Sinks ★

extern "C"
void paged_attention(
    torch::Tensor& out,    // [num_seqs,seqlen, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, num_heads, block_size, head_size]
    torch::Tensor& value_cache,// [num_blocks, num_heads, head_size, block_size]
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, //auto,int8,fp8/fp8_e4m3
    const c10::optional<torch::Tensor>& q_scale,
    const c10::optional<torch::Tensor>& k_scale,
    const c10::optional<torch::Tensor>& v_scale,
    int max_seq_len,
    const c10::optional<at::Tensor> &s_aux_,
    bool is_bhsd);
