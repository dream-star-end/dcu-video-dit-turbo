#include <torch/extension.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <c10/util/BFloat16.h>
#include <hip/hip_runtime.h>

__global__ void epilogue_vec4_kernel(
    const int32_t* __restrict__ accumulator,
    const float* __restrict__ row_scale,
    const float* __restrict__ col_scale,
    c10::BFloat16* __restrict__ output,
    int64_t columns) {
  int64_t row = static_cast<int64_t>(blockIdx.y);
  int64_t column = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (column >= columns) return;
  int64_t index = row * columns + column;
  const float rs = row_scale[row];
  if (column + 3 < columns) {
    int4 a = *reinterpret_cast<const int4*>(accumulator + index);
    float4 cs = *reinterpret_cast<const float4*>(col_scale + column);
    output[index + 0] = c10::BFloat16(static_cast<float>(a.x) * (rs * cs.x));
    output[index + 1] = c10::BFloat16(static_cast<float>(a.y) * (rs * cs.y));
    output[index + 2] = c10::BFloat16(static_cast<float>(a.z) * (rs * cs.z));
    output[index + 3] = c10::BFloat16(static_cast<float>(a.w) * (rs * cs.w));
  } else {
    for (int i = 0; i < 4 && column + i < columns; ++i) {
      output[index + i] = c10::BFloat16(
          static_cast<float>(accumulator[index + i]) * (rs * col_scale[column + i]));
    }
  }
}

torch::Tensor h3_epilogue_vec4_hip(
    torch::Tensor acc, torch::Tensor row, torch::Tensor col) {
  TORCH_CHECK(acc.is_cuda() && row.is_cuda() && col.is_cuda(), "HIP tensors required");
  TORCH_CHECK(acc.is_contiguous() && row.is_contiguous() && col.is_contiguous(), "contiguous required");
  TORCH_CHECK(acc.scalar_type() == torch::kInt32 && row.scalar_type() == torch::kFloat32 && col.scalar_type() == torch::kFloat32, "dtype mismatch");
  TORCH_CHECK(acc.dim() == 2 && row.numel() == acc.size(0) && col.numel() == acc.size(1), "shape mismatch");
  auto out = torch::empty(acc.sizes(), acc.options().dtype(torch::kBFloat16));
  constexpr int threads = 256;
  int64_t vec_columns = (acc.size(1) + 3) / 4;
  int blocks_x = static_cast<int>((vec_columns + threads - 1) / threads);
  auto stream = c10::hip::getCurrentHIPStream(acc.get_device());
  hipLaunchKernelGGL(epilogue_vec4_kernel, dim3(blocks_x, static_cast<unsigned>(acc.size(0))), dim3(threads), 0, stream.stream(),
      acc.data_ptr<int32_t>(), row.data_ptr<float>(), col.data_ptr<float>(), out.data_ptr<c10::BFloat16>(), acc.size(1));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return out;
}
