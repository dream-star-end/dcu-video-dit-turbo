#include <torch/extension.h>
torch::Tensor h3_mod_scale_shift_hip(torch::Tensor x, torch::Tensor scale, torch::Tensor shift);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mod_scale_shift_", &h3_mod_scale_shift_hip, "Exact BF16 mod scale/shift in place");
}
