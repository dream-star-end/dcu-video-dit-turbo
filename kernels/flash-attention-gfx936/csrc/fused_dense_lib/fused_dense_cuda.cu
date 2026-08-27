// Adapted from https://github.com/NVIDIA/apex/blob/master/csrc/fused_dense_cuda.cu
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <torch/torch.h>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "reduce.h"

// FP16 Tensor core wrapper around cublas GEMMEx
cublasStatus_t gemm_bias(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int64_t m,
    int64_t n,
    int64_t k,
    const float* alpha,
    const at::Half* A,
    int64_t lda,
    const at::Half* B,
    int64_t ldb,
    const float* beta,
    at::Half* C,
    int64_t ldc) {
  return cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      HIPBLAS_R_16F,
      lda,
      B,
      HIPBLAS_R_16F,
      ldb,
      beta,
      C,
      HIPBLAS_R_16F,
      ldc,
      HIPBLAS_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// BF16 Tensor core wrapper around cublas GEMMEx
cublasStatus_t gemm_bias(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int64_t m,
    int64_t n,
    int64_t k,
    const float* alpha,
    const at::BFloat16* A,
    int64_t lda,
    const at::BFloat16* B,
    int64_t ldb,
    const float* beta,
    at::BFloat16* C,
    int64_t ldc) {
  return cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      HIPBLAS_R_16B,
      lda,
      B,
      HIPBLAS_R_16B,
      ldb,
      beta,
      C,
      HIPBLAS_R_16B,
      ldc,
      HIPBLAS_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}


template <typename T>
int linear_bias_wgrad_cuda(const T *input, const T *d_output, int64_t in_features, int64_t batch_size, int64_t out_features, T *d_weight, T *d_bias, void *lt_workspace, size_t workspaceSize) {
    const float alpha          = 1.0;
    const float beta_zero      = 0.0;
    int status = 1;
    if (status != 0){
        cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
        status = gemm_bias(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          in_features,
          out_features,
          batch_size,
          &alpha,
          input,
          in_features,
          d_output,
          out_features,
          &beta_zero,
          d_weight,
          in_features);
    }
    if(d_bias!=nullptr)at::native::bias_reduce_sum((T* )d_bias,(T* )d_output, batch_size*out_features, out_features);
    return status;
}

template <typename T>
int linear_act_forward_cuda(const T *input, const T *weight, const T *bias, int64_t in_features, int64_t batch_size, int64_t out_features, bool is_gelu, int heuristic, T *output, void *pre_act, void *lt_workspace, size_t workspaceSize) {
    return 1;
}

template <typename T>
int bias_act_linear_dgrad_bgrad_cuda(const T *weight, const T *d_output, const void *pre_act, int64_t in_features, int64_t batch_size, int64_t out_features, bool is_gelu, int heuristic, T *d_input, T *d_bias, void *lt_workspace, size_t workspaceSize) {

    return 1;

}

template int linear_bias_wgrad_cuda<at::Half>(const at::Half *input, const at::Half *d_output, int64_t in_features, int64_t batch_size, int64_t out_features, at::Half *d_weight, at::Half *d_bias, void *lt_workspace, size_t workspaceSize);
template int linear_bias_wgrad_cuda<at::BFloat16>(const at::BFloat16 *input, const at::BFloat16 *d_output, int64_t in_features, int64_t batch_size, int64_t out_features, at::BFloat16 *d_weight, at::BFloat16 *d_bias, void *lt_workspace, size_t workspaceSize);

template int linear_act_forward_cuda<at::Half>(const at::Half *input, const at::Half *weight, const at::Half *bias, int64_t in_features, int64_t batch_size, int64_t out_features, bool is_gelu, int heuristic, at::Half *output, void *pre_act, void *lt_workspace, size_t workspaceSize);
template int linear_act_forward_cuda<at::BFloat16>(const at::BFloat16 *input, const at::BFloat16 *weight, const at::BFloat16 *bias, int64_t in_features, int64_t batch_size, int64_t out_features, bool is_gelu, int heuristic, at::BFloat16 *output, void *pre_act, void *lt_workspace, size_t workspaceSize);

template int bias_act_linear_dgrad_bgrad_cuda<at::Half>(const at::Half *weight, const at::Half *d_output, const void *pre_act, int64_t in_features, int64_t batch_size, int64_t out_features, bool is_gelu, int heuristic, at::Half *d_input, at::Half *d_bias, void *lt_workspace, size_t workspaceSize);
template int bias_act_linear_dgrad_bgrad_cuda<at::BFloat16>(const at::BFloat16 *weight, const at::BFloat16 *d_output, const void *pre_act, int64_t in_features, int64_t batch_size, int64_t out_features, bool is_gelu, int heuristic, at::BFloat16 *d_input, at::BFloat16 *d_bias, void *lt_workspace, size_t workspaceSize);
