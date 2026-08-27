#include <torch/extension.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <c10/util/BFloat16.h>
#include <hip/hip_runtime.h>

__global__ void mod_scale_shift_kernel(
    c10::BFloat16* x,
    const c10::BFloat16* scale,
    const c10::BFloat16* shift,
    int64_t elements,
    int64_t columns) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  int64_t column = index % columns;
  // Preserve all three eager BF16 roundings: 1+scale, h*factor, +shift.
  c10::BFloat16 factor(1.0f + static_cast<float>(scale[column]));
  c10::BFloat16 product(
      static_cast<float>(x[index]) * static_cast<float>(factor));
  x[index] = c10::BFloat16(
      static_cast<float>(product) + static_cast<float>(shift[column]));
}

torch::Tensor h3_mod_scale_shift_hip(
    torch::Tensor x, torch::Tensor scale, torch::Tensor shift) {
  TORCH_CHECK(x.is_cuda() && scale.is_cuda() && shift.is_cuda(), "HIP tensors required");
  TORCH_CHECK(x.is_contiguous() && scale.is_contiguous() && shift.is_contiguous(), "contiguous tensors required");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && scale.scalar_type() == torch::kBFloat16 && shift.scalar_type() == torch::kBFloat16, "BF16 required");
  TORCH_CHECK(x.dim() == 2 && scale.numel() == x.size(1) && shift.numel() == x.size(1), "shape mismatch");
  constexpr int threads = 256;
  int64_t elements = x.numel();
  int blocks = static_cast<int>((elements + threads - 1) / threads);
  auto stream = c10::hip::getCurrentHIPStream(x.get_device());
  hipLaunchKernelGGL(mod_scale_shift_kernel, dim3(blocks), dim3(threads), 0, stream.stream(),
      x.data_ptr<c10::BFloat16>(), scale.data_ptr<c10::BFloat16>(), shift.data_ptr<c10::BFloat16>(), elements, x.size(1));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return x;
}
