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

// Match comfy-kitchen eager, in this exact order:
//   abs_max = x.abs().amax(-1, keepdim=True)                [BF16]
//   scale = (abs_max.float() / 127.0).clamp(min=1e-30)     [FP32]
//   scale_math = scale.to(BF16), replacing zero by BF16 tiny
//   q = (x / scale_math).round().clamp(-128, 127).to(INT8) [BF16 math]
// The input is deliberately read twice. That preserves a simple deterministic
// reduction while still eliminating eager's abs temporary and four launches.
__global__ void quantize_int8_rowwise_bf16_kernel(
    const c10::BFloat16* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int64_t columns) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t row_offset = row * columns;

  float local_max = 0.0f;
  int local_nan = 0;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    const float value = static_cast<float>(input[row_offset + column]);
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
      // ATen's elementwise HIP kernel strength-reduces `/ 127.0f` to a
      // multiplication by the correctly rounded FP32 reciprocal. Spell that
      // operation explicitly; a hardware FP32 division differs by one ULP for
      // some BF16 maxima (for example 4.5).
      constexpr float kInv127 = 0.0078740157186985015869140625f;
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
    const float value = static_cast<float>(input[row_offset + column]);
    // ATen's BF16 division rounds the FP32 quotient back to BF16 before round().
    const c10::BFloat16 divided_bf16(value / scale_math);
    const float divided = static_cast<float>(divided_bf16);
    float rounded = nearbyintf(divided);  // IEEE round-to-nearest, ties-to-even.
    if (rounded > 127.0f) {
      rounded = 127.0f;
    } else if (rounded < -128.0f) {
      rounded = -128.0f;
    }
    // Actual H3 activations are finite. Match ATen's integral conversion for
    // non-finite test values as well: NaN/Inf become zero after clamp/cast.
    output[row_offset + column] =
        isfinite(rounded) ? static_cast<int8_t>(rounded) : static_cast<int8_t>(0);
  }
}

void check_input(const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "input must be on the HIP device");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bfloat16");
  TORCH_CHECK(input.dim() == 2, "input must be a two-dimensional [M, K] tensor");
  TORCH_CHECK(input.size(0) <= static_cast<int64_t>(UINT32_MAX), "M exceeds launch grid");
}

}  // namespace

std::vector<torch::Tensor> h3_quantize_int8_rowwise_hip(torch::Tensor input) {
  check_input(input);
  const int64_t rows = input.size(0);
  const int64_t columns = input.size(1);
  auto output = torch::empty(input.sizes(), input.options().dtype(torch::kInt8));
  auto scales = torch::empty({rows, 1}, input.options().dtype(torch::kFloat32));
  if (input.numel() == 0) {
    if (rows != 0 && columns == 0) {
      scales.fill_(1.0e-30f);
    }
    return {output, scales};
  }

  const auto stream = c10::hip::getCurrentHIPStream(input.get_device());
  hipLaunchKernelGGL(
      quantize_int8_rowwise_bf16_kernel,
      dim3(static_cast<unsigned int>(rows)),
      dim3(kThreads),
      0,
      stream.stream(),
      input.data_ptr<c10::BFloat16>(),
      output.data_ptr<int8_t>(),
      scales.data_ptr<float>(),
      columns);
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {output, scales};
}
