#include "flash.h"
#include "numeric_types.h"

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>


namespace py = pybind11;


void check_int32_max_offset(const at::Tensor& tensor, const char* name) {
    int64_t remaining = std::numeric_limits<int>::max();
    for (int dimension = 0; dimension < tensor.dim(); ++dimension) {
        const int64_t stride = tensor.stride(dimension);
        const int64_t extent = tensor.size(dimension) - 1;
        TORCH_CHECK(stride >= 0 && stride <= std::numeric_limits<int>::max(),
                    name, " requires non-negative int32 tensor strides");
        TORCH_CHECK(extent == 0 || stride <= remaining / extent,
                    name, " maximum tensor offset exceeds INT_MAX");
        remaining -= extent * stride;
    }
}


at::Tensor h3_flash_fwd_bhsd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double softmax_scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, and v must be on a HIP device");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "H3 kernel requires BF16 q");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
                "q, k, and v must have the same dtype");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, and v must use BHSD layout");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "H3 kernel currently requires self-attention with identical QKV shapes");
    TORCH_CHECK(q.get_device() == k.get_device() && q.get_device() == v.get_device(),
                "q, k, and v must be on the same HIP device");
    TORCH_CHECK(q.size(3) == 128, "H3 kernel requires head dimension 128");
    TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
                "q, k, and v must have contiguous last dimensions");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0,
                "batch, heads, and sequence length must be positive");
    TORCH_CHECK(std::isfinite(softmax_scale) && softmax_scale > 0.0,
                "softmax_scale must be finite and positive");
    constexpr int64_t int_max = std::numeric_limits<int>::max();
    for (const at::Tensor* tensor : {&q, &k, &v}) {
        check_int32_max_offset(*tensor, "H3 input");
    }
    TORCH_CHECK(q.size(0) <= int_max && q.size(1) <= int_max && q.size(2) <= int_max,
                "H3 kernel requires int32 batch, head, and sequence sizes");

    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());

    const int batch = static_cast<int>(q.size(0));
    const int heads = static_cast<int>(q.size(1));
    const int sequence = static_cast<int>(q.size(2));
    const int rounded_sequence = (sequence + 31) / 32 * 32;
    // Back the logical BHSD result with contiguous BSHD storage. H3 consumes
    // BSHD-flattened output, so its subsequent transpose+flatten becomes a
    // view instead of copying roughly 170 MiB after every transformer block.
    auto out_storage = at::empty({batch, sequence, heads, 128}, q.options());
    auto out = out_storage.transpose(1, 2);
    check_int32_max_offset(out, "H3 output");
    TORCH_CHECK(sequence == 0 || batch <= int_max / sequence,
                "H3 total_q exceeds INT_MAX");

    Flash_fwd_params params{};
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();

    params.q_batch_stride = static_cast<int>(q.stride(0));
    params.k_batch_stride = static_cast<int>(k.stride(0));
    params.v_batch_stride = static_cast<int>(v.stride(0));
    params.o_batch_stride = static_cast<int>(out.stride(0));
    params.q_head_stride = static_cast<int>(q.stride(1));
    params.k_head_stride = static_cast<int>(k.stride(1));
    params.v_head_stride = static_cast<int>(v.stride(1));
    params.o_head_stride = static_cast<int>(out.stride(1));
    params.q_row_stride = static_cast<int>(q.stride(2));
    params.k_row_stride = static_cast<int>(k.stride(2));
    params.v_row_stride = static_cast<int>(v.stride(2));
    params.o_row_stride = static_cast<int>(out.stride(2));
    params.v_dim_stride = static_cast<int>(v.stride(3));

    params.b = batch;
    params.b_k = batch;
    params.h = heads;
    params.h_k = heads;
    params.h_h_k_ratio = 1;
    params.seqlen_q = sequence;
    params.seqlen_k = sequence;
    params.seqlen_q_rounded = rounded_sequence;
    params.seqlen_k_rounded = rounded_sequence;
    params.total_q = batch * sequence;
    params.total_k = batch * sequence;
    params.d = 128;
    params.d_rounded = 128;
    params.d_value = 128;
    params.d_value_rounded = 128;

    params.scale_softmax = static_cast<float>(softmax_scale);
    params.scale_softmax_log2 = static_cast<float>(softmax_scale * M_LOG2E);
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.is_bf16 = true;
    params.is_causal = false;
    params.is_local = false;
    params.num_splits = 1;
    params.partition_size = 0;
    params.layout = 0;
    params.qkvheaddim_compute = 128;
    params.qkvheaddim_tail_tile16 = 2;

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    run_mha_fwd_<BFloat16, 128, 128>(params, stream);
    return out;
}


PYBIND11_MODULE(h3_flash_attn, module) {
    module.doc() = "Minimal OpenDAS BF16 FlashAttention forward kernel for H3 on gfx936";
    module.def(
        "fwd_bhsd",
        &h3_flash_fwd_bhsd,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("softmax_scale"));
}
