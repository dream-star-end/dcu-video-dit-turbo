#include <torch/extension.h>

#include <ATen/ATen.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <c10/util/BFloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

enum class BiasKind : int32_t {
  None = 0,
  BFloat16 = 1,
  Float32 = 2,
};

// The two explicit BF16 roundings match the eager implementation:
//   1. (acc.float() * (row_scale * col_scale)).to(torch.bfloat16)
//   2. result + bias.to(torch.bfloat16)
// Compilation disables fast math, unsafe reassociation, and FP contraction.
__global__ void h3_epilogue_kernel(
    const int32_t* __restrict__ accumulator,
    const float* __restrict__ row_scale,
    const float* __restrict__ col_scale,
    const void* __restrict__ bias,
    c10::BFloat16* __restrict__ output,
    int64_t columns,
    BiasKind bias_kind) {
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  const int64_t column =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= columns) {
    return;
  }
  const int64_t index = row * columns + column;

  // Preserve eager parenthesization: accumulator * (row_scale * col_scale).
  const float combined_scale = row_scale[row] * col_scale[column];
  const float scaled = static_cast<float>(accumulator[index]) * combined_scale;
  c10::BFloat16 rounded(scaled);

  if (bias_kind != BiasKind::None) {
    c10::BFloat16 rounded_bias;
    if (bias_kind == BiasKind::BFloat16) {
      rounded_bias = static_cast<const c10::BFloat16*>(bias)[column];
    } else {
      rounded_bias = c10::BFloat16(static_cast<const float*>(bias)[column]);
    }
    rounded = c10::BFloat16(
        static_cast<float>(rounded) + static_cast<float>(rounded_bias));
  }

  output[index] = rounded;
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on the HIP device");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor h3_epilogue_hip(
    torch::Tensor accumulator,
    torch::Tensor row_scale,
    torch::Tensor col_scale,
    torch::Tensor bias) {
  check_cuda_contiguous(accumulator, "accumulator");
  check_cuda_contiguous(row_scale, "row_scale");
  check_cuda_contiguous(col_scale, "col_scale");
  check_cuda_contiguous(bias, "bias");

  TORCH_CHECK(accumulator.scalar_type() == torch::kInt32,
              "accumulator must be int32");
  TORCH_CHECK(row_scale.scalar_type() == torch::kFloat32,
              "row_scale must be float32");
  TORCH_CHECK(col_scale.scalar_type() == torch::kFloat32,
              "col_scale must be float32");
  TORCH_CHECK(accumulator.dim() == 2, "accumulator must be [M, N]");

  const int64_t rows = accumulator.size(0);
  const int64_t columns = accumulator.size(1);
  TORCH_CHECK(rows <= 65535,
              "M exceeds the 2D launch grid limit: ", rows);
  TORCH_CHECK(row_scale.numel() == rows,
              "row_scale must contain exactly M values");
  TORCH_CHECK(col_scale.numel() == columns,
              "col_scale must contain exactly N values");
  TORCH_CHECK(row_scale.device() == accumulator.device() &&
                  col_scale.device() == accumulator.device() &&
                  bias.device() == accumulator.device(),
              "all inputs must be on the same device");

  BiasKind bias_kind = BiasKind::None;
  if (bias.numel() != 0) {
    TORCH_CHECK(bias.numel() == columns,
                "bias must be empty or contain exactly N values");
    if (bias.scalar_type() == torch::kBFloat16) {
      bias_kind = BiasKind::BFloat16;
    } else if (bias.scalar_type() == torch::kFloat32) {
      bias_kind = BiasKind::Float32;
    } else {
      TORCH_CHECK(false, "bias must be empty, bfloat16, or float32");
    }
  }

  auto output = torch::empty(
      accumulator.sizes(), accumulator.options().dtype(torch::kBFloat16));
  if (accumulator.numel() == 0) {
    return output;
  }

  constexpr int threads = 256;
  const int blocks_x = static_cast<int>((columns + threads - 1) / threads);
  const auto stream = c10::hip::getCurrentHIPStream(accumulator.get_device());
  hipLaunchKernelGGL(
      h3_epilogue_kernel,
      dim3(blocks_x, static_cast<unsigned int>(rows)),
      dim3(threads),
      0,
      stream.stream(),
      accumulator.data_ptr<int32_t>(),
      row_scale.data_ptr<float>(),
      col_scale.data_ptr<float>(),
      bias.numel() == 0 ? nullptr : bias.const_data_ptr(),
      output.data_ptr<c10::BFloat16>(),
      columns,
      bias_kind);
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return output;
}
