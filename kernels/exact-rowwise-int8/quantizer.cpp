#include <torch/extension.h>

std::vector<torch::Tensor> h3_quantize_int8_rowwise_hip(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "quantize_int8_rowwise",
      &h3_quantize_int8_rowwise_hip,
      "Exact H3 BF16 rowwise INT8 quantization (HIP)");
}
