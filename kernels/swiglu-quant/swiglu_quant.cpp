#include <torch/extension.h>

std::vector<torch::Tensor> h3_swiglu_quantize_int8_rowwise_hip(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "swiglu_quantize_int8_rowwise",
      &h3_swiglu_quantize_int8_rowwise_hip,
      "Exact fused BF16 SwiGLU and rowwise INT8 quantization (HIP)");
}
