#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

template<typename scalar_t> 
static __device__ inline void from_float(scalar_t &out ,float f){
  if constexpr(std::is_same<scalar_t, _Float16>::value||std::is_same<scalar_t, float>::value){
    out=f;
  }
  else{
    uint32_t u = *(uint32_t*)(&f);
    u += 0x7fff + ((u >> 16) & 1); 
    // u += 0x8000; 
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


#define Input_Type_SWITCH(SRC_DTYPE, ...)     \
  [&] {                                       \
    if (SRC_DTYPE == at::ScalarType::Half) {  \
      using scalar_t=_Float16;                \
      return __VA_ARGS__();                   \
    }else {                                   \
      using scalar_t=uint16_t;                \
      return __VA_ARGS__();                   \
    }                                         \
  }()

#define BLK_SWITCH(blk,...)                   \
[&] {                                         \
    if (blk==64){                             \
        constexpr static int BLK = 64;        \
        return __VA_ARGS__();                 \
    }else {                                   \
        constexpr static int BLK = 128;       \
        return __VA_ARGS__();                 \
    }                                         \
}()

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

template<typename scalar_t,int blocksize,int DIM,int BLK,bool has_mean>
__global__ void mean_pool_fast_kernel(scalar_t *out, const scalar_t *input,int L_BLOCKS,int b,int s,int h ,const scalar_t* mean){
  int tid = threadIdx.x;
  if(blockIdx.x<L_BLOCKS-1||s==L_BLOCKS*BLK){
    const scalar_t* input_cur = input + blockIdx.z*s*h*DIM + blockIdx.y*DIM + (blockIdx.x*BLK+tid/16)*h*DIM + tid%16*8;
    scalar_t* out_cur = out+blockIdx.z*h*L_BLOCKS*DIM + blockIdx.y*L_BLOCKS*DIM + blockIdx.x * DIM;
    const scalar_t* mean_cur = has_mean? mean+blockIdx.z*h*DIM + blockIdx.y*DIM + tid%16*8:nullptr;
    constexpr int n = DIM*BLK;
    using half_vec= __attribute__( (__vector_size__(8 * sizeof(scalar_t)) )) scalar_t;
    using float_vec= __attribute__( (__vector_size__(8 * sizeof(float)) )) float;
    __shared__ float lds_ptr[blocksize*8];
    {
      float_vec sum={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
      half_vec mean_temp;
      if constexpr(has_mean){
          mean_temp = *reinterpret_cast<const half_vec*>(mean_cur);
          // if(tid==0)printf("mean_temp =%.5f,%.5f,%.5f,%.5f,  %.5f,%.5f,%.5f,%.5f,\n", to_float(mean_temp[0]), to_float(mean_temp[1]), to_float(mean_temp[2]), to_float(mean_temp[3])
          //                                                                             , to_float(mean_temp[4]), to_float(mean_temp[5]), to_float(mean_temp[6]), to_float(mean_temp[7]));
      }
      for(int i=0;i<n;i+=blocksize*8){
        half_vec temp = *reinterpret_cast<const half_vec*>(input_cur+i*h);
        for(int ii=0;ii<8;ii++){
          if constexpr(has_mean){
            sum[ii] += to_float(temp[ii]) - to_float(mean_temp[ii]);
          }
          else{
            sum[ii] += to_float(temp[ii]);
          }
        }
      }
      *reinterpret_cast<float_vec*>(lds_ptr+tid*8)=sum;
      __syncthreads();
    }
    float sum=0.0f;
    for(int i=0;i<8;i++){
       sum+=lds_ptr[tid+DIM*i];
    }
    sum/=BLK;
    from_float(out_cur[tid],sum);
  }
  else{
    int s_lenth = s % BLK;
    const scalar_t* input_cur = input + blockIdx.z*s*h*DIM + blockIdx.y*DIM + (blockIdx.x*BLK)*h*DIM + tid;
    scalar_t* out_cur = out+blockIdx.z*h*L_BLOCKS*DIM + blockIdx.y*L_BLOCKS*DIM + blockIdx.x * DIM;
    const scalar_t* mean_cur = has_mean? mean+blockIdx.z*h*DIM + blockIdx.y*DIM + tid:nullptr;
    float sum=0.0f;
    float mean_temp=0.0f;
    if constexpr(has_mean){
        mean_temp = to_float(*(mean_cur));
    }
    for(int i=0;i<s_lenth;i++){
      scalar_t temp = *(input_cur+i*h*DIM);
      if constexpr(has_mean){
        sum+=(to_float(temp)-mean_temp);
      }
      else{
        sum+=to_float(temp);
      }
    }
    sum /= s_lenth;
    from_float(out_cur[tid],sum);
  }
}

at::Tensor mean_pool_fast(const at::Tensor &input,int blk,const c10::optional<at::Tensor> &mean){
  //assume dim=128
  int b=input.size(0);
  int s=input.size(1);
  int h=input.size(2);
  int d=input.size(3);
  int L_BLOCKS = (s + blk - 1) / blk;
  auto out = torch::empty({b, h, L_BLOCKS,d}, input.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(L_BLOCKS,h,b);
  Input_Type_SWITCH(input.scalar_type(),[&]{
    BLK_SWITCH(blk,[&]{
      const scalar_t *mean_ptr = mean?reinterpret_cast<const scalar_t*>(mean.value().data_ptr()):nullptr;
      BOOL_SWITCH(mean_ptr!=nullptr,has_mean,[&]{
        const scalar_t *input_ptr = reinterpret_cast<const scalar_t*>(input.data_ptr());
        scalar_t *out_ptr = reinterpret_cast<scalar_t*>(out.data_ptr());
        mean_pool_fast_kernel<scalar_t,128,128,BLK,has_mean><<<grid,128,0,stream>>>(out_ptr,input_ptr,L_BLOCKS,b,s,h,mean_ptr);
      });
    });
  });
  return out;
}