#include <torch/extension.h>
torch::Tensor h3_epilogue_vec4_hip(torch::Tensor acc, torch::Tensor row, torch::Tensor col);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("epilogue", &h3_epilogue_vec4_hip, "Exact vectorized H3 INT8 epilogue");
}
