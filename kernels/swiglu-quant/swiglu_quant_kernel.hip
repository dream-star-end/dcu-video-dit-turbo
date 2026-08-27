#include <torch/extension.h>

#include <ATen/ATen.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <c10/util/BFloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr float kInv127 = 0.0078740157186985015869140625f;

__device__ __forceinline__ c10::BFloat16 eager_swiglu_bf16(
    c10::BFloat16 gate_bf16,
    c10::BFloat16 up_bf16) {
  const float gate = static_cast<float>(gate_bf16);
  // ATen SiLU for reduced floating types computes in FP32, returns BF16, then
  // comfy-kitchen's mul_ performs a second FP32 operation and BF16 rounding.
  const c10::BFloat16 silu(gate / (1.0f + expf(-gate)));
  return c10::BFloat16(
      static_cast<float>(silu) * static_cast<float>(up_bf16));
}

__global__ void swiglu_quantize_int8_rowwise_bf16_kernel(
    const c10::BFloat16* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t columns) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t input_columns = columns * 2;
  const int64_t input_row = row * input_columns;
  const int64_t output_row = row * columns;
  extern __shared__ c10::BFloat16 activated[];

  float local_max = 0.0f;
  int local_nan = 0;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    const c10::BFloat16 value_bf16 = eager_swiglu_bf16(
        input[input_row + column], input[input_row + columns + column]);
    activated[column] = value_bf16;
    const float value = static_cast<float>(value_bf16);
    if (isnan(value)) {
      local_nan = 1;
    } else {
      local_max = fmaxf(local_max, fabsf(value));
    }
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    local_nan = max(local_nan, __shfl_down(local_nan, offset));
  }

  __shared__ float warp_max[32];
  __shared__ int warp_nan[32];
  __shared__ float scale_math_shared;
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
  if (lane == 0) {
    warp_max[warp] = local_max;
    warp_nan[warp] = local_nan;
  }
  __syncthreads();

  if (warp == 0) {
    float block_max = lane < warp_count ? warp_max[lane] : 0.0f;
    int block_nan = lane < warp_count ? warp_nan[lane] : 0;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      block_max = fmaxf(block_max, __shfl_down(block_max, offset));
      block_nan = max(block_nan, __shfl_down(block_nan, offset));
    }
    if (lane == 0) {
      float scale = block_nan ? NAN : block_max * kInv127;
      if (!isnan(scale) && scale < 1.0e-30f) {
        scale = 1.0e-30f;
      }
      scales[row] = scale;
      c10::BFloat16 scale_bf16(scale);
      float scale_math = static_cast<float>(scale_bf16);
      if (scale_math == 0.0f) {
        scale_math = static_cast<float>(c10::BFloat16(1.1754943508222875e-38f));
      }
      scale_math_shared = scale_math;
    }
  }
  __syncthreads();

  const float scale_math = scale_math_shared;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    const c10::BFloat16 divided_bf16(
        static_cast<float>(activated[column]) / scale_math);
    float rounded = nearbyintf(static_cast<float>(divided_bf16));
    if (rounded > 127.0f) {
      rounded = 127.0f;
    } else if (rounded < -128.0f) {
      rounded = -128.0f;
    }
    output[output_row + column] =
        isfinite(rounded) ? static_cast<int8_t>(rounded) : static_cast<int8_t>(0);
  }
}

}  // namespace

std::vector<torch::Tensor> h3_swiglu_quantize_int8_rowwise_hip(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be on HIP");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be BF16");
  TORCH_CHECK(input.dim() == 2, "input must be [M, 2K]");
  TORCH_CHECK(input.size(1) % 2 == 0, "input width must be even");
  const int64_t rows = input.size(0);
  const int64_t columns = input.size(1) / 2;
  auto output = torch::empty({rows, columns}, input.options().dtype(torch::kInt8));
  auto scales = torch::empty({rows, 1}, input.options().dtype(torch::kFloat32));
  if (rows == 0 || columns == 0) {
    return {output, scales};
  }
  const auto stream = c10::hip::getCurrentHIPStream(input.get_device());
  hipLaunchKernelGGL(
      swiglu_quantize_int8_rowwise_bf16_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kThreads),
      static_cast<size_t>(columns * sizeof(c10::BFloat16)),
      stream.stream(),
      input.data_ptr<c10::BFloat16>(),
      output.data_ptr<int8_t>(),
      scales.data_ptr<float>(),
      columns);
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {output, scales};
}
