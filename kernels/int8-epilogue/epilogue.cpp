#include <torch/extension.h>

torch::Tensor h3_epilogue_hip(
    torch::Tensor accumulator,
    torch::Tensor row_scale,
    torch::Tensor col_scale,
    torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "epilogue",
      &h3_epilogue_hip,
      "H3 INT8 GEMM epilogue (int32 + row/column scales + optional bias -> BF16)");
}
