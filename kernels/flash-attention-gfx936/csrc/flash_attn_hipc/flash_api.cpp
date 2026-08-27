/******************************************************************************
* Copyright (c) 2023, Tri Dao.
******************************************************************************/

#include "flash_c_api.h"

#ifndef BUILD_C_INTERFACE
// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
// #include <torch/python.h>
#include <torch/nn/functional.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/TensorIndexing.h>
#include <ATen/core/Tensor.h>

#if defined(USE_ROCM)
#include <ATen/hip/HIPGeneratorImpl.h>
#else
#ifndef TORCH_CUDA_CPP_API
#define TORCH_CUDA_CPP_API TORCH_API
#endif
#include <ATen/cuda/CUDAGeneratorImpL.h>
#endif

#define CHECK_DEVICE(x)     TORCH_CHECK(x.is_cuda(), #x " must be on CUDA (", __FILE__, ":", __LINE__, ")")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == at::IntArrayRef({__VA_ARGS__}), #x " must have shape (", at::IntArrayRef({__VA_ARGS__}), "), but got ", x.sizes(), " (", __FILE__, ":", __LINE__, ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous (", __FILE__, ":", __LINE__, ")")

static inline bool is_runtime_gfx92a(const std::string &gcn_arch_name) {
    return gcn_arch_name.rfind("gfx92a", 0) == 0;
}

static inline int runtime_gfx_arch_id(const std::string &gcn_arch_name) {
    return is_runtime_gfx92a(gcn_arch_name) ? 930 : std::stoi(gcn_arch_name.substr(3, 3));
}

static inline bool is_supported_hg_mla_arch(const std::string &gcn_arch_name, const int gcn_arch) {
    return is_runtime_gfx92a(gcn_arch_name) || gcn_arch >= 936;
}


void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      const int d_v,
                      const int d_v_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      float softcap=0.0,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false,
                      const bool is_kvcache=false,
                      const bool is_seqlens_k_cumulative=false,
                      const int layout=0,
                      const bool is_flashmla=false,
                      const bool is_prefix=false
                    ) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_int8 = q.dtype() == at::ScalarType::Char;
    if (!params.is_int8) {
        params.is_bf16 = q.dtype() == at::ScalarType::BFloat16;
    }
    params.is_e4m3 = q.dtype() == at::ScalarType::Float8_e4m3fn;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.o_ptr = out.data_ptr();
    params.layout = layout;
    if (cu_seqlens_k_d == nullptr and !is_kvcache) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);

        params.q_row_stride = params.layout ? q.stride(1): q.stride(2);
        params.k_row_stride = params.layout ? k.stride(1): k.stride(2);
        params.v_row_stride = params.layout ? v.stride(1): v.stride(2);
        params.o_row_stride = params.layout ? out.stride(1): out.stride(2);

        params.q_head_stride = params.layout ? q.stride(2): q.stride(1);
        params.k_head_stride = params.layout ? k.stride(2): k.stride(1);
        params.v_head_stride = params.layout ? v.stride(2): v.stride(1);
        params.o_head_stride = params.layout ? out.stride(2): out.stride(1);
        params.is_seqlens_k_cumulative = false;
        // params.varlen_proj_qkv_head = h; // uniform computation to reduce vgpr/sgpr
    }
    else {
        params.is_seqlens_k_cumulative = is_seqlens_k_cumulative;
        if (is_kvcache) {
            // when kvcache, q/o shape is different from training/prefill
            params.q_batch_stride = q.stride(0);
            params.o_batch_stride = out.stride(0);
            params.q_head_stride  = (layout == 1) ? q.stride(2): q.stride(1);
            params.k_head_stride  = (layout == 1) ? k.stride(2): k.stride(1);
            params.v_head_stride  = (layout == 1) ? v.stride(2): v.stride(1);
            params.o_head_stride  = (layout == 1) ? out.stride(2): out.stride(1);
            params.q_row_stride   = (layout == 1) ? q.stride(1): q.stride(2);
            params.k_row_stride   = (layout == 1) ? k.stride(1): k.stride(2);
            params.v_row_stride   = (layout == 1) ? v.stride(1): v.stride(2);
            params.o_row_stride   = (layout == 1) ? out.stride(1): out.stride(2);
        } else if (is_flashmla) {
            params.q_batch_stride = q.stride(0);
            params.o_batch_stride = out.stride(0);
            params.q_head_stride  = (layout == 1) ? q.stride(2): q.stride(1);
            params.k_head_stride  = (layout == 1) ? k.stride(2): k.stride(1);
            params.v_head_stride  = params.k_head_stride;
            params.o_head_stride  = (layout == 1) ? out.stride(2): out.stride(1);
            if (seqlenq_ngroups_swapped) params.o_head_stride *= seqlen_q;
            params.q_row_stride   = (layout == 1) ? q.stride(1): q.stride(2);
            params.k_row_stride   = (layout == 1) ? k.stride(1): k.stride(2);
            params.v_row_stride   = params.k_row_stride;
            params.o_row_stride   = (layout == 1) ? out.stride(1): out.stride(2);
        } else if (is_prefix) {
            if (params.layout == 1) {
                params.q_head_stride = q.stride(-2);
                params.k_head_stride = k.stride(-2);
                params.v_head_stride = v.stride(-2);
                params.o_head_stride = out.stride(1);
                params.q_row_stride  = q.stride(0);
                params.k_row_stride  = k.stride(1);
                params.v_row_stride  = v.stride(1);
                params.o_row_stride  = out.stride(0);
            } else {
                params.q_head_stride = q.stride(1);
                params.k_head_stride = k.stride(1);
                params.v_head_stride = v.stride(1);
                params.o_head_stride = out.stride(1);
                params.q_row_stride  = q.stride(0);
                params.k_row_stride  = k.stride(2);
                params.v_row_stride  = v.stride(2);
                params.o_row_stride  = out.stride(0);
            }
        } else {
            params.q_head_stride = params.layout ? q.stride(-2): q.stride(0);
            params.k_head_stride = params.layout ? k.stride(-2): k.stride(0);
            params.v_head_stride = params.layout ? v.stride(-2): v.stride(0);
            params.o_head_stride = params.layout ? out.stride(1): out.stride(0);
            params.q_row_stride = params.layout ? q.stride(0): params.q_head_stride/*also .stride(0)*/;
            params.k_row_stride = params.layout ? k.stride(0): params.k_head_stride;
            params.v_row_stride = params.layout ? v.stride(0): params.v_head_stride;
            params.o_row_stride = params.layout ? out.stride(0): params.o_head_stride;
            // params.varlen_proj_qkv_head = params.layout ? k.stride(-3) / k.stride(-2): 0;
            // in vllm, K and V is not contiguous due to rope, but Q is contiguous. However, in some sceniros, K is contiguous but V is not contiguous()
        }
    }
    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;
    params.d_value = d_v;
    params.d_value_rounded = d_v_rounded;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
        TORCH_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
    #endif
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        // Set the different scale values.
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }
    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
}

void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      const int d_v,
                      const int d_v_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *p_d,
#ifdef DEBUGING
                      void *kq_ptr,
                      void *s_ptr,
                      void *dp_ptr,
                      void *ds_ptr,
#endif
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap=0.0,
                      bool deterministic=false,
                      const bool unpadded_lse=false,
                      const int layout=0) {

    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     d_v, d_v_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     nullptr,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     false, // seqlenq_ngroups_swapped
                     unpadded_lse,
                     false,
                     true,
                     layout);
    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);

        params.dq_row_stride = params.layout ? dq.stride(-3):dq.stride(-2);
        params.dk_row_stride = params.layout ? dk.stride(-3):dk.stride(-2);
        params.dv_row_stride = params.layout ? dv.stride(-3):dv.stride(-2);
        params.do_row_stride = params.layout ? dout.stride(-3):dout.stride(-2);
        params.dq_head_stride = params.layout ? dq.stride(-2) : dq.stride(-3);
        params.dk_head_stride = params.layout ? dk.stride(-2) : dk.stride(-3);
        params.dv_head_stride = params.layout ? dv.stride(-2) : dv.stride(-3);
        params.do_head_stride = params.layout ? dout.stride(-2) : dout.stride(-3);
    }
    else {
        params.q_batch_stride = q.stride(0);
        params.o_batch_stride = out.stride(0);

        params.dq_head_stride = dq.stride(-2);
        params.dk_head_stride = dk.stride(-2);
        params.dv_head_stride = dv.stride(-2);
        params.do_head_stride = dout.stride(-2);

        params.dq_row_stride = params.layout ? dq.stride(-3) : dq.stride(-2);
        params.dk_row_stride = params.layout ? dk.stride(-3) : dk.stride(-2);
        params.dv_row_stride = params.layout ? dv.stride(-3) : dv.stride(-2);
        params.do_row_stride = params.layout ? dout.stride(-3) : dout.stride(-2);
    }
    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    // deterministic
    params.deterministic = deterministic;
    // PRINT_BWD_PARAMS
#ifdef DEBUGING
    params.kq_ptr = kq_ptr;
    params.s_ptr = s_ptr;
    params.dp_ptr = dp_ptr;
    params.ds_ptr = ds_ptr;
#endif
}


// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}


void set_params_dropout(Flash_fwd_params& params, float p_dropout, int counter_offset, at::Tensor& rng_state, c10::optional<at::Generator> gen_, at::TensorOptions opts, at::Tensor& dropout_debug_count) {
    if (p_dropout > 0) {
        rng_state = at::empty({2}, opts.dtype(at::ScalarType::Long));
        // Forward kernel will populate memory with the seed and offset.
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        at::PhiloxCudaState philox_args = gen->philox_cuda_state(counter_offset);
        // at::cuda::philox::unpack(philox_args) not supported on ROCm
        params.rand_seed = philox_args.seed_.val;
        params.rand_offset = philox_args.offset_.val;
        // For dropout debugging tensor
        #ifdef FA_DEBUG
        dropout_debug_count = at::zeros({2}, opts.dtype(at::ScalarType::UInt32));
        params.dropout_debug_count = reinterpret_cast<uint32_t*>(dropout_debug_count.data_ptr());
        #endif
    } else {
        params.rng_state = nullptr;
    }
}


void set_params_alibi(Flash_fwd_params &params, c10::optional<at::Tensor> &alibi_slopes_, int batch_size, int num_heads){
#ifdef FLASHATTENTION_DISABLE_ALIBI
    TORCH_CHECK(!alibi_slopes_.has_value(), "This flash attention build does not support alibi.");
    params.alibi_slopes_ptr = nullptr;
#else
    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        TORCH_CHECK(alibi_slopes.dtype() == at::ScalarType::Float, "ALiBi slopes must have dtype fp32");
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == at::IntArrayRef({num_heads}) || alibi_slopes.sizes() == at::IntArrayRef({batch_size, num_heads}));
        params.alibi_slopes_ptr = alibi_slopes.data_ptr();
        params.alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    } else {
        params.alibi_slopes_ptr = nullptr;
    }
#endif
}



std::vector<at::Tensor>
fwd_base(at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        c10::optional<at::Tensor> &out_,
        c10::optional<at::Tensor> &alibi_slopes_,
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_,
        const int layout,
        c10::optional<at::Tensor> q_descale_,
        c10::optional<at::Tensor> k_descale_,
        c10::optional<at::Tensor> v_descale_,
        const bool is_bf16_output) {
#if defined(BUILD_FA_FWD)
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());

    auto q_dtype = q.dtype();
    const bool fp8_used = q_dtype == at::ScalarType::Float8_e4m3fn;
    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16 || fp8_used,
                "FlashAttention only supports fp16, bf16, and fp8_e4m3 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    if (fp8_used) {
        TORCH_CHECK(q_descale_.has_value() && k_descale_.has_value() && v_descale_.has_value(),
                    "FP8 forward requires q_descale, k_descale, and v_descale");
    }

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const bool use_bshd_layout = bool(layout == 1);
    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    int num_heads = use_bshd_layout ? sizes[2]: sizes[1];
    int seqlen_q = use_bshd_layout ? sizes[1]: sizes[2];
    const int head_size_og = sizes[3];
    const int head_size_og_value = v.size(3);
    const int num_heads_k = use_bshd_layout ? k.size(2): k.size(1);
    const int seqlen_k = use_bshd_layout ? k.size(1): k.size(2);
    TORCH_CHECK(seqlen_q == seqlen_k || is_causal == false, "FlashAttention forward do not support 'seqlen_k != seqlen_q && is_causal == true' for now")
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size_og <= 512, "FlashAttention forward only supports head dimension at most 512");
    TORCH_CHECK(head_size_og_value <= 512, "FlashAttention forward only supports head dimension at most 512");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(head_size_og >= head_size_og_value, "Head dimension of query/key must greater or equal to head dimension in query");

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    TORCH_CHECK(int64_t(batch_size * num_heads * seqlen_q * head_size_og) < /*2^31*/int64_t(2147483648), "The data amount of q must be smaller than the representation range of int");
    TORCH_CHECK(int64_t(batch_size * num_heads_k * seqlen_k * head_size_og) < /*2^31*/int64_t(2147483648), "The data amount of k/v must be smaller than the representation range of int");

    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }  // causal=true is the same as causal=false in this case
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && layout == 0 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size_og % 8 == 0;
    if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        if (layout == 0) q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og});
        else if (layout == 1) q = q.transpose(1, 2).reshape({batch_size, ngroups, num_heads_k, head_size_og});
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    if (layout == 0) {
        CHECK_SHAPE(q, batch_size, num_heads, seqlen_q, head_size_og);
        CHECK_SHAPE(k, batch_size, num_heads_k, seqlen_k, head_size_og);
        CHECK_SHAPE(v, batch_size, num_heads_k, seqlen_k, head_size_og_value);
    } else if (layout == 1) {
        CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
        CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
        CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og_value);
    }

    // For better performance for cases where headdim is not even multiple times of 32, assign head_size granularity
    const char* headdim_granularity_env = std::getenv("FA_HEADDIM_GRANULARITY");
    int headdim_granularity = headdim_granularity_env == nullptr ? 64: std::atoi(headdim_granularity_env);
    if (head_size_og % 32 == 0 or head_size_og_value % 32 == 0) { headdim_granularity = 32; }

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % headdim_granularity != 0) {
        q_padded = at::pad(q, {0, headdim_granularity - head_size_og % headdim_granularity});
        k_padded = at::pad(k, {0, headdim_granularity - head_size_og % headdim_granularity});
    } else {
        q_padded = q;
        k_padded = k;
    }

    if (head_size_og_value % headdim_granularity != 0) {
        v_padded = at::pad(v, {0, headdim_granularity - head_size_og_value % headdim_granularity});
    } else {
        v_padded = v;
    }

    at::Tensor out;
    auto opts = q.options();
    auto out_opts = fp8_used
        ? (is_bf16_output ? opts.dtype(at::ScalarType::BFloat16) : opts.dtype(at::ScalarType::Half))
        : opts;
    if (out_.has_value()) {
        out = out_.value();
        if (fp8_used) {
            TORCH_CHECK(out.dtype() == at::ScalarType::Half || out.dtype() == at::ScalarType::BFloat16,
                        "FP8 forward output must have fp16 or bf16 dtype");
        } else {
            TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        }
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        if (layout == 0) {
            CHECK_SHAPE(out, batch_size, num_heads, seqlen_q, head_size_og_value);
        } else if (layout == 1) {
            CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og_value);
        }
    } else {
        if (layout == 0) {
            out = at::empty({batch_size, num_heads, seqlen_q, head_size_og_value}, out_opts);
        } else if (layout == 1) {
            out = at::empty({batch_size, seqlen_q, num_heads, head_size_og_value}, out_opts);
        } else if (layout == 2) {
            out = at::empty({seqlen_q, batch_size, num_heads, head_size_og_value}, out_opts);
        }
    }

    if (head_size_og_value % headdim_granularity != 0) {
        out = at::pad(out, {0, headdim_granularity - head_size_og_value % headdim_granularity});
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, headdim_granularity);
    const int head_size_v = round_multiple(head_size_og_value, 8);
    const int head_size_v_rounded = round_multiple(head_size_v, headdim_granularity);
    const int seqlen_q_rounded = round_multiple(seqlen_q, headdim_granularity);
    const int seqlen_k_rounded = round_multiple(seqlen_k, headdim_granularity);

    auto softmax_lse = at::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = at::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     head_size_v, head_size_v_rounded,
                     q_padded, k_padded, v_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     /*seqlenq_ngroups_swapped*/seqlenq_ngroups_swapped,
                     /*unpadded_lse*/false,
                     /*is_kvcache*/false,
                     /*is_seqlens_k_cumulative*/false,
                     /*layout*/layout
                     );

    if (fp8_used) {
        at::Tensor q_descale = q_descale_.value();
        at::Tensor k_descale = k_descale_.value();
        at::Tensor v_descale = v_descale_.value();
        TORCH_CHECK(q_descale.dtype() == at::ScalarType::Float, "q_descale must have dtype float32");
        TORCH_CHECK(k_descale.dtype() == at::ScalarType::Float, "k_descale must have dtype float32");
        TORCH_CHECK(v_descale.dtype() == at::ScalarType::Float, "v_descale must have dtype float32");
        CHECK_DEVICE(q_descale); CHECK_DEVICE(k_descale); CHECK_DEVICE(v_descale);
        TORCH_CHECK(q_descale.dim() >= 2 && k_descale.dim() >= 2 && v_descale.dim() >= 2,
                    "FP8 descale tensors must have at least [batch, head] dimensions");
        params.is_bf16 = is_bf16_output;
        params.q_descale_ptr = reinterpret_cast<float*>(q_descale.data_ptr());
        params.k_descale_ptr = reinterpret_cast<float*>(k_descale.data_ptr());
        params.v_descale_ptr = reinterpret_cast<float*>(v_descale.data_ptr());
        params.q_descale_batch_stride = q_descale.stride(0);
        params.q_descale_head_stride  = q_descale.stride(1);
        params.k_descale_batch_stride = k_descale.stride(0);
        params.k_descale_head_stride  = k_descale.stride(1);
        params.v_descale_batch_stride = v_descale.stride(0);
        params.v_descale_head_stride  = v_descale.stride(1);
    }

    if (head_size_og % headdim_granularity != 0 or head_size_og_value % headdim_granularity != 0) {
        params.d       = head_size_rounded;
        params.d_value = head_size_v_rounded;
        params.qkvheaddim_compute = (int(std::max(head_size_og, head_size_og_value) / 32) + 1) * 32;
        params.qkvheaddim_tail_tile16 = std::max((head_size_og % 32 + 16 - 1) / 16, (head_size_og_value % 32 + 16 - 1) / 16);
    }

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (seqlen_q + 64 - 1) / 64;
    params.num_splits = 1;
    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks,/*num_SMs*/ 1 /*dprops->multiProcessorCount*/, num_n_blocks, 128);
        if (params.num_splits > 1) {
            at::Tensor softmax_lse_accum = at::empty({params.num_splits, batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
            at::Tensor out_accum = at::empty({params.num_splits, batch_size, num_heads, seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
            params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
            params.oaccum_ptr = out_accum.data_ptr();
        }
        TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    at::Tensor rng_state;
    at::Tensor dropout_debug_count;
    int counter_offset = batch_size * num_heads * 64;
    set_params_dropout(params, p_dropout, counter_offset, rng_state, gen_, opts, dropout_debug_count);

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    const char* fa_debug = std::getenv("FA_DEBUG");
    if (fa_debug != nullptr) {
        if (std::strcmp(fa_debug, "1") == 0) { PRINT_PARAMS }
        else if (std::strcmp(fa_debug, "2") == 0) { PRINT_PARAMS_ONELINE }
        PRINT_QKV_INFO(q, k, v)
    }

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    run_mha_fwd(params, stream);

    if (fa_debug != nullptr && std::strcmp(fa_debug, "dropout") == 0 && p_dropout > 0) {
        HIP_CHECK(hipDeviceSynchronize());
        std::cout << "rng_state: " << rng_state[0].item() << ", " << rng_state[1].item() << std::endl;
        std::cout << "dropout_debug_count: " << dropout_debug_count[0].item() << std::endl;
    }

    at::Tensor out_padded = out;
    if (head_size_og_value % headdim_granularity != 0) {
        out = out.index({"...", at::indexing::Slice(at::indexing::None, head_size_og_value)}).contiguous();
        // if (out_.has_value()) { out_.value().copy_(out); }
    }

    if (seqlenq_ngroups_swapped) {
        if (layout == 0) {
            out = out.reshape({batch_size, num_heads_k * seqlen_q, 1, head_size_og_value});
            out_padded  = out_padded.reshape({batch_size, num_heads_k * seqlen_q, 1, head_size_og_value});
            q_padded    = q_padded.reshape({batch_size, num_heads_k * seqlen_q, 1, head_size_og_value});
            softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
        } else if (layout == 1) {
            out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og_value});
            out_padded  = out_padded.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og_value});
            q_padded    = q_padded.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og_value});
            softmax_lse = softmax_lse.transpose(1, 2).reshape({batch_size, num_heads_k * seqlen_q, 1});
        }
    }
    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
#else
    return {};
#endif
}



std::vector<at::Tensor>
hg_fwd_bhsd(at::Tensor &q,                           // batch_size x num_heads x seqlen_q x head_size
        const at::Tensor &k,                      // batch_size x num_heads x seqlen_q x head_size
        const at::Tensor &v,                      // batch_size x num_heads x seqlen_q x head_size
        c10::optional<at::Tensor> &out_,          // batch_size x num_heads x seqlen_q x head_size
        c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_,
        c10::optional<at::Tensor> q_descale_,
        c10::optional<at::Tensor> k_descale_,
        c10::optional<at::Tensor> v_descale_,
        const bool is_bf16_output) {
    return fwd_base(q, k, v, out_, alibi_slopes_, p_dropout, softmax_scale, is_causal, window_size_left, window_size_right, softcap, return_softmax, gen_, 0/*bhsd*/, q_descale_, k_descale_, v_descale_, is_bf16_output);
}

std::vector<at::Tensor>
hg_fwd_bshd(at::Tensor &q,                           // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,                      // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &v,                      // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &out_,          // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_,
        c10::optional<at::Tensor> q_descale_,
        c10::optional<at::Tensor> k_descale_,
        c10::optional<at::Tensor> v_descale_,
        const bool is_bf16_output) {
    return fwd_base(q, k, v, out_, alibi_slopes_, p_dropout, softmax_scale, is_causal, window_size_left, window_size_right, softcap, return_softmax, gen_, 1/*bshd*/, q_descale_, k_descale_, v_descale_, is_bf16_output);
}

std::vector<at::Tensor>
fwd_padding_mask(at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        const at::Tensor &padding_mask,
        c10::optional<at::Tensor> &out_,
        c10::optional<at::Tensor> &alibi_slopes_,
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_,
        int layout) {
#if defined(BUILD_FA_FWD)
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const bool use_bshd_layout = bool(layout == 1);
    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    int num_heads = use_bshd_layout ? sizes[2]: sizes[1];
    int seqlen_q = use_bshd_layout ? sizes[1]: sizes[2];
    const int head_size_og = sizes[3];
    const int head_size_og_value = v.size(3);
    const int num_heads_k = use_bshd_layout ? k.size(2): k.size(1);
    const int seqlen_k = use_bshd_layout ? k.size(1): k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size_og_value <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(head_size_og >= head_size_og_value, "Head dimension of query/key must greater or equal to head dimension in query");
    if ((head_size_og != 64 and head_size_og != 128) or (head_size_og_value != 64 and head_size_og_value != 128)) {
        printf("\x1b[31mOnly headdim 64/128 is supported for padding mask yet!\033[0m\n");
        return {};
    }

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    TORCH_CHECK(int64_t(batch_size * num_heads * seqlen_q * head_size_og) < /*2^31*/int64_t(2147483648), "The data amount of q must be smaller than the representation range of int");
    TORCH_CHECK(int64_t(batch_size * num_heads_k * seqlen_k * head_size_og) < /*2^31*/int64_t(2147483648), "The data amount of k/v must be smaller than the representation range of int");

    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }  // causal=true is the same as causal=false in this case
    if (is_causal) {
        window_size_right = 0;
        printf("\x1b[31mCausal mask is not supported for padding mask yet!\033[0m\n");
        return {};
    }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size_og % 8 == 0;
    if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og});
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }
    // CHECK_SHAPE(q, batch_size, num_heads, seqlen_q, head_size_og);
    // CHECK_SHAPE(k, batch_size, num_heads_k, seqlen_k, head_size_og);
    // CHECK_SHAPE(v, batch_size, num_heads_k, seqlen_k, head_size_og_value);

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 32 != 0) {
        q_padded = at::pad(q, {0, 32 - head_size_og % 32});
        k_padded = at::pad(k, {0, 32 - head_size_og % 32});
    } else {
        q_padded = q;
        k_padded = k;
    }

    if (head_size_og_value % 32 != 0) {
        v_padded = at::pad(v, {0, 32 - head_size_og_value % 32});
    } else {
        v_padded = v;
    }

    at::Tensor out;
    auto opts = q.options();
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        // CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og_value);
    } else {
        if (layout == 0) {
            out = at::zeros({batch_size, num_heads, seqlen_q, head_size_og_value}, opts);
        } else if (layout == 1) {
            out = at::zeros({batch_size, seqlen_q, num_heads, head_size_og_value}, opts);
        } else if (layout == 2) {
            out = at::zeros({seqlen_q, batch_size, num_heads, head_size_og_value}, opts);
        }
    }

    if (head_size_og_value % 32 != 0) {
        out = at::pad(out, {0, 32 - head_size_og_value % 32});
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int head_size_v = round_multiple(head_size_og_value, 8);
    const int head_size_v_rounded = round_multiple(head_size_v, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 32);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 32);

    auto softmax_lse = at::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p, rng_state;

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     head_size_v, head_size_v_rounded,
                     q_padded, k_padded, v_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     /*seqlenq_ngroups_swapped*/false,
                     /*unpadded_lse*/false,
                     /*is_kvcache*/false,
                     /*is_seqlens_k_cumulative*/false,
                     /*layout*/layout
                     );
    params.padding_mask = padding_mask.data_ptr<int32_t>();

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    const char* fa_debug = std::getenv("FA_DEBUG");
    if (fa_debug != nullptr) {
        if (std::strcmp(fa_debug, "1") == 0) { PRINT_PARAMS }
        else if (std::strcmp(fa_debug, "2") == 0) { PRINT_PARAMS_ONELINE }
        PRINT_QKV_INFO(q, k, v)
    }

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    run_mha_fwd(params, stream);

    at::Tensor out_padded = out;
    if (head_size_og_value % 32 != 0) {
        out = out.index({"...", at::indexing::Slice(at::indexing::None, head_size_og_value)}).contiguous();
    }

    if (seqlenq_ngroups_swapped) {
        out = out.reshape({batch_size, num_heads_k * seqlen_q, 1, head_size_og_value});
        out_padded = out_padded.reshape({batch_size, num_heads_k * seqlen_q, 1, head_size_og_value});
        q_padded = q_padded.reshape({batch_size, num_heads_k * seqlen_q, 1, head_size_og_value});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
#else
    return {};
#endif
}


std::vector<at::Tensor>
fwd_attn_mask(at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        const at::Tensor &attn_mask,
        c10::optional<at::Tensor> &out_,
        c10::optional<at::Tensor> &alibi_slopes_,
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_,
        int layout) {
#if defined(BUILD_FA_FWD)
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const bool use_bshd_layout = bool(layout == 1);
    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    int num_heads = use_bshd_layout ? sizes[2]: sizes[1];
    int seqlen_q = use_bshd_layout ? sizes[1]: sizes[2];
    const int head_size_og = sizes[3];
    const int head_size_og_value = v.size(3);
    const int num_heads_k = use_bshd_layout ? k.size(2): k.size(1);
    const int seqlen_k = use_bshd_layout ? k.size(1): k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size_og_value <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(head_size_og >= head_size_og_value, "Head dimension of query/key must greater or equal to head dimension in query");
    if (head_size_og != 128 or head_size_og_value != 128) {
        printf("\x1b[31mOnly headdim 128 is supported for attn mask yet!\033[0m\n");
        return {};
    }

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    TORCH_CHECK(int64_t(batch_size * num_heads * seqlen_q * head_size_og) < /*2^31*/int64_t(2147483648), "The data amount of q must be smaller than the representation range of int");
    TORCH_CHECK(int64_t(batch_size * num_heads_k * seqlen_k * head_size_og) < /*2^31*/int64_t(2147483648), "The data amount of k/v must be smaller than the representation range of int");

    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }  // causal=true is the same as causal=false in this case
    if (is_causal) {
        window_size_right = 0;
        printf("\x1b[31mCausal mask is not supported for attn mask yet!\033[0m\n");
        return {};
    }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size_og % 8 == 0;
    if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og});
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    if (layout == 0) {
        CHECK_SHAPE(q, batch_size, num_heads, seqlen_q, head_size_og);
        CHECK_SHAPE(k, batch_size, num_heads_k, seqlen_k, head_size_og);
        CHECK_SHAPE(v, batch_size, num_heads_k, seqlen_k, head_size_og_value);
    } else {
        CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
        CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
        CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og_value);
    }

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 32 != 0) {
        q_padded = at::pad(q, {0, 32 - head_size_og % 32});
        k_padded = at::pad(k, {0, 32 - head_size_og % 32});
    } else {
        q_padded = q;
        k_padded = k;
    }

    if (head_size_og_value % 32 != 0) {
        v_padded = at::pad(v, {0, 32 - head_size_og_value % 32});
    } else {
        v_padded = v;
    }

    at::Tensor out;
    auto opts = q.options();
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        if (layout == 0) {
            CHECK_SHAPE(out, batch_size, num_heads, seqlen_q, head_size_og_value);
        } else if (layout == 1) {
            CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og_value);
        }
    } else {
        if (layout == 0) {
            out = at::zeros({batch_size, num_heads, seqlen_q, head_size_og_value}, opts);
        } else if (layout == 1) {
            out = at::zeros({batch_size, seqlen_q, num_heads, head_size_og_value}, opts);
        } else if (layout == 2) {
            out = at::zeros({seqlen_q, batch_size, num_heads, head_size_og_value}, opts);
        }
    }

    if (head_size_og_value % 32 != 0) {
        out = at::pad(out, {0, 32 - head_size_og_value % 32});
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int head_size_v = round_multiple(head_size_og_value, 8);
    const int head_size_v_rounded = round_multiple(head_size_v, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 32);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 32);

    auto softmax_lse = at::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p, rng_state;

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     head_size_v, head_size_v_rounded,
                     q_padded, k_padded, v_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     /*seqlenq_ngroups_swapped*/false,
                     /*unpadded_lse*/false,
                     /*is_kvcache*/false,
                     /*is_seqlens_k_cumulative*/false,
                     /*layout*/layout
                     );
    params.attn_mask = attn_mask.data_ptr<int32_t>();

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    const char* fa_debug = std::getenv("FA_DEBUG");
    if (fa_debug != nullptr) {
        if (std::strcmp(fa_debug, "1") == 0) { PRINT_PARAMS }
        else if (std::strcmp(fa_debug, "2") == 0) { PRINT_PARAMS_ONELINE }
        PRINT_QKV_INFO(q, k, v)
    }

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    run_mha_fwd(params, stream);

    at::Tensor out_padded = out;
    if (head_size_og_value % 32 != 0) {
        out = out.index({"...", at::indexing::Slice(at::indexing::None, head_size_og_value)}).contiguous();
    }

    if (seqlenq_ngroups_swapped) {
        out = out.reshape({batch_size, num_heads_k * seqlen_q, 1, head_size_og_value});
        out_padded = out_padded.reshape({batch_size, num_heads_k * seqlen_q, 1, head_size_og_value});
        q_padded = q_padded.reshape({batch_size, num_heads_k * seqlen_q, 1, head_size_og_value});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
#else
    return {};
#endif
}


std::vector<at::Tensor> varlen_fwd(
        const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        const int num_heads,
        const int num_heads_k,
        c10::optional<at::Tensor> &out_,
        const at::Tensor &cu_seqlens_q,
        const at::Tensor &cu_seqlens_k,
        c10::optional<at::Tensor> &seqused_k,
        c10::optional<at::Tensor> &alibi_slopes_,
        const int max_seqlen_q,
        const int max_seqlen_k,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_,
        const int layout,
        c10::optional<at::Tensor> q_descale_,
        c10::optional<at::Tensor> k_descale_,
        c10::optional<at::Tensor> v_descale_,
        const bool is_bf16_output
) {
#if defined(BUILD_FA_FWD)
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
    if (is_causal) { window_size_right = 0; }

    auto q_dtype = q.dtype();
    const bool fp8_used = q_dtype == at::ScalarType::Float8_e4m3fn;
    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16 || fp8_used,
                "FlashAttention only supports fp16, bf16, and fp8_e4m3 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    if (fp8_used) {
        TORCH_CHECK(q_descale_.has_value() && k_descale_.has_value() && v_descale_.has_value(),
                    "FP8 varlen forward requires q_descale, k_descale, and v_descale");
    }
    TORCH_CHECK(cu_seqlens_q.dtype() == at::ScalarType::Int, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == at::ScalarType::Int, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const bool use_bshd_layout = bool(layout == 1);
    const auto query_size   = q.sizes();
    const auto k_size       = k.sizes();
    const auto v_size       = v.sizes();
    const int head_size_og  = use_bshd_layout ? query_size[2]: query_size[1];
    const int head_size_value = use_bshd_layout ? v_size[2]: v_size[1]; //TODO:FBH
    const int total_q       = use_bshd_layout ? query_size[0] * query_size[1] / num_heads: query_size[0] / num_heads; // cu_seqlens_q[-1].item<int>();
    const int total_k       = use_bshd_layout ? k_size[0]:  k_size[0] / num_heads_k; // cu_seqlens_k[-1].item<int>();
    const int batch_size    = cu_seqlens_q.numel() - 1;
    if (use_bshd_layout) {
        TORCH_CHECK(q.dim() == 3, "In varlen forward attention, when layout is 'bshd', q must be 3-dimension tensor");
        TORCH_CHECK(k.dim() == 3, "In varlen forward attention, when layout is 'bshd', q must be 3-dimension tensor");
        TORCH_CHECK(v.dim() == 3, "In varlen forward attention, when layout is 'bshd', q must be 3-dimension tensor");
    } else {
        TORCH_CHECK(q.dim() == 2, "In varlen forward attention, when layout is 'bhsd', q must be 2-dimension tensor");
        TORCH_CHECK(k.dim() == 2, "In varlen forward attention, when layout is 'bhsd', q must be 2-dimension tensor");
        TORCH_CHECK(v.dim() == 2, "In varlen forward attention, when layout is 'bhsd', q must be 2-dimension tensor");
    }
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og == 32 || head_size_og == 40 || head_size_og == 64 || head_size_og == 72 || head_size_og == 80 ||
                head_size_og == 96 || head_size_og == 128 || head_size_og == 160 || head_size_og == 192 || head_size_og == 224 ||
                head_size_og == 256, "FlashAttention forward only supports head dimension 32,40,64,72,80,96,128,160,192,224,256");
    TORCH_CHECK(head_size_value == 32 || head_size_value == 40 || head_size_value == 64 || head_size_value == 72 || head_size_value == 80 ||
                head_size_value == 96 || head_size_value == 128 || head_size_value == 160 || head_size_value == 192 || head_size_value == 224 ||
                head_size_value == 256, "FlashAttention forward only supports head dimension value 32,40,64,72,80,96,128,160,192,224,256");
    TORCH_CHECK(head_size_og == head_size_value || (head_size_og == 192 && head_size_value==128), "FlashAttention forward only supports head dimension be same, or d192 dv128");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(int64_t(query_size[0] * head_size_og) < /*2^31*/int64_t(2147483648), "The data amount of q must be smaller than the representation range of int");
    TORCH_CHECK(int64_t(k_size[0] * head_size_value) < /*2^31*/int64_t(2147483648), "The data amount of k/v must be smaller than the representation range of int");
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (seqused_k.has_value()) {
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == at::ScalarType::Int, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    // For better performance for cases where headdim is not even multiple times of 32, assign head_size granularity
    const char* headdim_granularity_env = std::getenv("FA_HEADDIM_GRANULARITY");
    int headdim_granularity = headdim_granularity_env == nullptr ? 64: std::atoi(headdim_granularity_env);
    if (head_size_og % 32 == 0 or head_size_value % 32 == 0) { headdim_granularity = 32; }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, headdim_granularity);
    const int head_size_v = round_multiple(head_size_value, 8);
    const int head_size_v_rounded = round_multiple(head_size_v, headdim_granularity);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, headdim_granularity);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, headdim_granularity);

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % headdim_granularity != 0) {
        q_padded = at::pad(q, {0, headdim_granularity - head_size_og % headdim_granularity});
        k_padded = at::pad(k, {0, headdim_granularity - head_size_og % headdim_granularity});
    } else {
        q_padded = q;
        k_padded = k;
    }

    if (head_size_value % headdim_granularity != 0) {
        v_padded = at::pad(v, {0, headdim_granularity - head_size_value % headdim_granularity});
    } else {
        v_padded = v;
    }

    auto opts = q.options();
    auto out_opts = fp8_used
        ? (is_bf16_output ? opts.dtype(at::ScalarType::BFloat16) : opts.dtype(at::ScalarType::Half))
        : opts;
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        if (fp8_used) {
            TORCH_CHECK(out.dtype() == at::ScalarType::Half || out.dtype() == at::ScalarType::BFloat16,
                        "FP8 varlen forward output must have fp16 or bf16 dtype");
        } else {
            TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        }
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        if (head_size_value % headdim_granularity != 0) {
            out = at::pad(out, {0, headdim_granularity - head_size_value % headdim_granularity});
        }
    } else {
        if (layout == 0) {out = at::empty({query_size[0], head_size_v_rounded}, out_opts);}
        else if (layout == 1) {out = at::empty({query_size[0], query_size[1], head_size_v_rounded}, out_opts);}
    }

    auto softmax_lse = at::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = at::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }

    if (zero_tensors) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) {p.zero_();}
    }
    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     head_size_v, head_size_v_rounded,
                     q_padded, k_padded, v_padded, out,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     false,
                     /*unpadded_lse*/false,
                     /*is_kvcache*/false,
                     /*is_seqlens_k_cumulative*/cu_seqlens_k.size(0) == (batch_size + 1),
                     layout
                    );
    params.total_q = total_q;
    params.total_k = total_k;
    if (fp8_used) {
        at::Tensor q_descale = q_descale_.value();
        at::Tensor k_descale = k_descale_.value();
        at::Tensor v_descale = v_descale_.value();
        TORCH_CHECK(q_descale.dtype() == at::ScalarType::Float, "q_descale must have dtype float32");
        TORCH_CHECK(k_descale.dtype() == at::ScalarType::Float, "k_descale must have dtype float32");
        TORCH_CHECK(v_descale.dtype() == at::ScalarType::Float, "v_descale must have dtype float32");
        CHECK_DEVICE(q_descale); CHECK_DEVICE(k_descale); CHECK_DEVICE(v_descale);
        TORCH_CHECK(q_descale.dim() >= 2 && k_descale.dim() >= 2 && v_descale.dim() >= 2,
                    "FP8 descale tensors must have at least [batch, head] dimensions");
        params.is_bf16 = is_bf16_output;
        params.q_descale_ptr = reinterpret_cast<float*>(q_descale.data_ptr());
        params.k_descale_ptr = reinterpret_cast<float*>(k_descale.data_ptr());
        params.v_descale_ptr = reinterpret_cast<float*>(v_descale.data_ptr());
        params.q_descale_batch_stride = q_descale.stride(0);
        params.q_descale_head_stride  = q_descale.stride(1);
        params.k_descale_batch_stride = k_descale.stride(0);
        params.k_descale_head_stride  = k_descale.stride(1);
        params.v_descale_batch_stride = v_descale.stride(0);
        params.v_descale_head_stride  = v_descale.stride(1);
    }
    if (head_size_og % headdim_granularity != 0 or head_size_value % headdim_granularity != 0) {
        params.d       = head_size_rounded;
        params.d_value = head_size_v_rounded;
        params.qkvheaddim_compute = (int(std::max(head_size_og, head_size_value) / 32) + 1) * 32/*mls32x32粒度是32*/;
        params.qkvheaddim_tail_tile16 = std::max((head_size_og % 32 + 16 - 1) / 16, (head_size_value % 32 + 16 - 1) / 16);
    }

    at::Tensor rng_state;
    at::Tensor dropout_debug_count;
    int counter_offset = batch_size * num_heads * 64;
    set_params_dropout(params, p_dropout, counter_offset, rng_state, gen_, opts, dropout_debug_count);

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    const char* fa_debug = std::getenv("FA_DEBUG");
    if (fa_debug != nullptr) {
        if (std::strcmp(fa_debug, "1") == 0) { PRINT_PARAMS }
        else if (std::strcmp(fa_debug, "2") == 0) {
            PRINT_PARAMS_ONELINE
            auto temp_tensor = cu_seqlens_k.to(at::DeviceType::CPU).contiguous();
            std::vector<int32_t> temp_vector(temp_tensor.data_ptr<int32_t>(), temp_tensor.data_ptr<int32_t>() + temp_tensor.numel());
            printf("cu_seqlens_k: ["); for (const auto val: temp_vector) { printf("%d ", val); } printf("]\n");
        }
        PRINT_QKV_INFO(q, k, v)
    }

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    run_mha_fwd(params, stream);

    if (fa_debug != nullptr && std::strcmp(fa_debug, "dropout") == 0 && p_dropout > 0) {
        HIP_CHECK(hipDeviceSynchronize());
        std::cout << "rng_state: " << rng_state[0].item() << ", " << rng_state[1].item() << std::endl;
        std::cout << "dropout_debug_count: " << dropout_debug_count[0].item() << std::endl;
    }

    at::Tensor out_padded = out;
    if (head_size_value % headdim_granularity != 0) {
        out = out.index({"...", at::indexing::Slice(at::indexing::None, head_size_value)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
#else
    return {};
#endif
}



std::vector<at::Tensor> hg_varlen_fwd_bshd(
        at::Tensor &q,
        at::Tensor &k,
        at::Tensor &v,
        c10::optional<at::Tensor> &out_,
        const at::Tensor &cu_seqlens_q,
        const at::Tensor &cu_seqlens_k,
        c10::optional<at::Tensor> &seqused_k,
        c10::optional<at::Tensor> &alibi_slopes_,
        const int max_seqlen_q,
        const int max_seqlen_k,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_,
        c10::optional<at::Tensor> q_descale_,
        c10::optional<at::Tensor> k_descale_,
        c10::optional<at::Tensor> v_descale_,
        const bool is_bf16_output) {
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
    // [batch x seqlen, num_head, headdim] ----> [batch x num_head x seqlen, headdim]
    const auto query_size = q.sizes();
    const bool tensor_is_4dim = query_size.size() == 4;
    const int num_heads = tensor_is_4dim ? query_size[2]: query_size[1];
    const int num_heads_kv = tensor_is_4dim ? k.size(2): k.size(1);
    // FA kernel
    return varlen_fwd(
        q,
        k,
        v,
        num_heads,
        num_heads_kv,
        out_,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        alibi_slopes_,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen_,
        1/*bshd*/,
        q_descale_,
        k_descale_,
        v_descale_,
        is_bf16_output);
}

// Preserved for original inference interface
at::Tensor varlen_fwd_bshd_infer(
        at::Tensor &q,
        at::Tensor &k,
        at::Tensor &v,
        c10::optional<at::Tensor> &out_,
        const at::Tensor &cu_seqlens_q,
        const at::Tensor &cu_seqlens_k,
        c10::optional<at::Tensor> &seqused_k,
        c10::optional<at::Tensor> &alibi_slopes_,
        const int max_seqlen_q,
        const int max_seqlen_k,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_,
        c10::optional<at::Tensor> q_descale_,
        c10::optional<at::Tensor> k_descale_,
        c10::optional<at::Tensor> v_descale_,
        const bool is_bf16_output) {
    return hg_varlen_fwd_bshd(q, k, v, out_, cu_seqlens_q, cu_seqlens_k, seqused_k, alibi_slopes_, max_seqlen_q, max_seqlen_k, p_dropout, softmax_scale, zero_tensors, is_causal, window_size_left, window_size_right, softcap, return_softmax, gen_, q_descale_, k_descale_, v_descale_, is_bf16_output)[0];
}




std::vector<at::Tensor> varlen_fwd_bhsd(
        const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        c10::optional<at::Tensor> &out_,
        const at::Tensor &cu_seqlens_q,
        const at::Tensor &cu_seqlens_k,
        c10::optional<at::Tensor> &seqused_k,
        c10::optional<at::Tensor> &alibi_slopes_,
        const int max_seqlen_q,
        const int max_seqlen_k,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_
){
#if defined(BUILD_FA_FWD)
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
    if (is_causal) { window_size_right = 0; }

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == at::ScalarType::Int, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == at::ScalarType::Int, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();
    const int total_q_heads = q.numel() / sizes[1];
    const int total_q       = cu_seqlens_q[-1].item<int>();
    const int batch_size    = cu_seqlens_q.numel() - 1;
    const int num_heads     = total_q_heads / total_q;
    const int head_size_og  = sizes[1];
    const int head_size_value = v.size(1);
    const int total_k_heads = k.numel() / k.size(1);
    const int total_k       = cu_seqlens_k[-1].item<int>();
    const int num_heads_k   = total_k_heads / total_k;
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size_value <= 256, "FlashAttention forward only supports head dimension at most 256 for V");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(int64_t(total_q_heads * head_size_og) < /*2^31*/int64_t(2147483648), "The data amount of q must be smaller than the representation range of int");
    TORCH_CHECK(int64_t(total_k_heads * head_size_og) < /*2^31*/int64_t(2147483648), "The data amount of k/v must be smaller than the representation range of int");

    CHECK_SHAPE(q, total_q_heads, head_size_og);
    CHECK_SHAPE(k, total_k_heads, head_size_og);
    CHECK_SHAPE(v, total_k_heads, head_size_og);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (seqused_k.has_value()) {
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == at::ScalarType::Int, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int head_size_v = round_multiple(head_size_value, 8);
    const int head_size_v_rounded = round_multiple(head_size_v, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 32);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 32);

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 32 != 0) {
        q_padded = at::pad(q, {0, 32 - head_size_og % 32});
        k_padded = at::pad(k, {0, 32 - head_size_og % 32});
    } else {
        q_padded = q;
        k_padded = k;
    }
    if (head_size_value % 32 != 0) {
        v_padded = at::pad(v, {0, 32 - head_size_value % 32});
    } else {
        v_padded = v;
    }

    auto opts = q.options();
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        // CHECK_SHAPE(out, total_q, num_heads, head_size_value);
        if (head_size_value % 32 != 0) {
            out = at::pad(out, {0, 32 - head_size_value % 32});
        }
    } else {
        out = at::empty({total_q_heads, head_size_v_rounded}, opts);
    }

    auto softmax_lse = at::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = at::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }

    if (zero_tensors) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) {p.zero_();}
    }
    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     head_size_v, head_size_v_rounded,
                     q_padded, k_padded, v_padded, out,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     return_softmax ? p.data_ptr() : nullptr,
                     seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     false,
                     /*unpadded_lse*/false,
                     /*is_kvcache*/false,
                     /*is_seqlens_k_cumulative*/cu_seqlens_k.size(0) == (batch_size + 1),
                     /*layout*/0
                    );
    params.total_q = total_q;
    params.total_k = total_k;

    at::Tensor rng_state;
    if (p_dropout > 0) {
        auto options = at::TensorOptions().dtype(at::ScalarType::Float).device(at::DeviceType::CUDA);
        rng_state = at::empty({2}, options.dtype(at::ScalarType::Long));
        // Forward kernel will populate memory with the seed and offset.
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());
    } else {
        params.rng_state = nullptr;
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    const char* fa_debug = std::getenv("FA_DEBUG");
    if (fa_debug != nullptr) {
        if (std::strcmp(fa_debug, "1") == 0) { PRINT_PARAMS }
        else if (std::strcmp(fa_debug, "2") == 0) {
            PRINT_PARAMS_ONELINE
            auto temp_tensor = cu_seqlens_k.to(at::DeviceType::CPU).contiguous();
            std::vector<int32_t> temp_vector(temp_tensor.data_ptr<int32_t>(), temp_tensor.data_ptr<int32_t>() + temp_tensor.numel());
            printf("cu_seqlens_k: ["); for (const auto val: temp_vector) { printf("%d ", val); } printf("]\n");
        }
        PRINT_QKV_INFO(q, k, v)
    }

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    run_mha_fwd(params, stream);

    at::Tensor out_padded = out;
    if (head_size_value % 32 != 0) {
        out = out.index({"...", at::indexing::Slice(at::indexing::None, head_size_value)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
#else
    return {};
#endif
}

int get_attention_sink_type(at::ScalarType dtype) {
  if (dtype == at::ScalarType::Float) {
    return 1;
  }
  if (dtype == at::ScalarType::Half) {
    return 2;
  }
  if (dtype == at::ScalarType::BFloat16) {
    return 3;
  }
  TORCH_CHECK(false, "Attention sink only supports fp32/fp16/bf16 dtype");
  return 0;
}

std::vector<at::Tensor> hg_prefix_prefill_varlen_fwd(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    c10::optional<at::Tensor> &out_, const at::Tensor &cu_seqlens_q,
    c10::optional<at::Tensor> &cu_seqlens_k, at::Tensor &seqused_k,
    c10::optional<at::Tensor> &alibi_slopes_, at::Tensor &block_table,
    const int max_seqlen_q, const int max_seqlen_k, const float p_dropout,
    const float softmax_scale, const bool zero_tensors, const bool is_causal,
    int window_size_left, int window_size_right, const float softcap,
    const bool return_softmax, const int layout,
    c10::optional<at::Tensor> scales_q_ = c10::nullopt,
    c10::optional<at::Tensor> scales_k_ = c10::nullopt,
    c10::optional<at::Tensor> scales_v_ = c10::nullopt,
    c10::optional<at::Tensor> s_aux_ = c10::nullopt,
    const bool is_bf16_output = false) {
#if defined(BUILD_FA_FWD)
  const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
  // TORCH_CHECK(is_causal == true, "for prefix prefill, only causal mask = True
  // is supported!");
  if (is_causal) {
    window_size_right = 0;
  }

  auto q_dtype = q.dtype();
  const bool int8_used = q_dtype == at::ScalarType::Char;
  const bool fp8_used = q_dtype == at::ScalarType::Float8_e4m3fn;
  TORCH_CHECK(q_dtype == at::ScalarType::Half ||
                  q_dtype == at::ScalarType::BFloat16 ||
                  q_dtype == at::ScalarType::Char ||
                  q_dtype == at::ScalarType::Float8_e4m3fn,
              "FlashAttention only support fp16 and bf16 and int8 and fp8 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == at::ScalarType::Int,
              "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(seqused_k.dtype() == at::ScalarType::Int,
              "seqused_k must have dtype int32");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(cu_seqlens_q);
  CHECK_DEVICE(seqused_k);

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(seqused_k);

  TORCH_CHECK(layout == 0 || layout == 1,
              "Prefix prefill only supports layout 0 (bhsd) or 1 (bshd)");
  const bool use_bshd_layout = layout == 1;
  const auto query_size = q.sizes();
  const auto k_size = k.sizes();
  const auto v_size = v.sizes();
  const int batch_size = cu_seqlens_q.numel() - 1;
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  if (use_bshd_layout) {
    TORCH_CHECK(q.dim() == 3,
                "Prefix prefill bshd expects q shape [total_q, num_heads, head_dim]");
    TORCH_CHECK(k.dim() == 4,
                "Prefix prefill bshd expects k shape [num_blocks, block_size, num_heads_k, head_dim]");
    TORCH_CHECK(v.dim() == 4,
                "Prefix prefill bshd expects v shape [num_blocks, block_size, num_heads_k, head_dim_v]");
  } else {
    TORCH_CHECK(q.dim() == 3,
                "Prefix prefill bhsd expects q shape [total_q, num_heads, head_dim]");
    TORCH_CHECK(k.dim() == 4,
                "Prefix prefill bhsd expects k shape [num_blocks, num_heads_k, block_size, head_dim]");
    TORCH_CHECK(v.dim() == 4,
                "Prefix prefill bhsd expects v shape [num_blocks, num_heads_k, block_size, head_dim_v]");
  }
  const int total_q = query_size[0];
  TORCH_CHECK(total_q > 0, "total_q must be positive");
  const int num_heads = query_size[1];
  TORCH_CHECK(num_heads > 0, "num_heads must be positive");
  const int num_heads_k = use_bshd_layout ? k_size[2] : k_size[1];
  const int head_size_og = query_size[2];
  const int head_size_value = v_size[3];
  const int page_block_size = use_bshd_layout ? k_size[1] : k_size[2];
  TORCH_CHECK(page_block_size == 128 || (!int8_used && page_block_size == 64),
              "Prefix prefill only supports page block_size 128, plus b16/fp8 page block_size 64");
  if (!use_bshd_layout) {
    TORCH_CHECK(q.is_contiguous(),
                "Prefix prefill bhsd requires q to be contiguous");
    TORCH_CHECK(k.is_contiguous(),
                "Prefix prefill bhsd requires k cache to be contiguous");
    TORCH_CHECK(v.is_contiguous(),
                "Prefix prefill bhsd requires v cache to be contiguous");
  }
  TORCH_CHECK((head_size_og == 128 and head_size_value == 128) or
                  (head_size_og == 192 and head_size_value == 128) or
                  (head_size_og == 192 and head_size_value == 192) or
                  (head_size_og == 256 and head_size_value == 256) or
                  (!int8_used && !fp8_used &&
                   head_size_og == 512 && head_size_value == 512),
              "Prefix prefill only supports head dimension "
              "128+128/192+128/192+192/256+256, plus b16 512+512");
  if (fp8_used) {
    TORCH_CHECK((head_size_og == 128 and head_size_value == 128) or
                    (head_size_og == 192 and head_size_value == 128) or
                    (head_size_og == 256 and head_size_value == 256),
                "FP8 prefix prefill only supports head dimension 128+128/192+128/256+256 on gfx938");
    TORCH_CHECK(scales_q_.has_value() && scales_k_.has_value() &&
                    scales_v_.has_value(),
                "FP8 prefix prefill requires q/k/v descale tensors");
  }
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");
  TORCH_CHECK(int64_t(q.numel()) <
                  /*2^31*/ int64_t(2147483648),
              "The data amount of q must be smaller than the representation "
              "range of int");
  TORCH_CHECK(int64_t(k.numel()) <
                  /*2^31*/ int64_t(2147483648),
              "The data amount of k/v must be smaller than the representation "
              "range of int");
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(seqused_k, batch_size);

  if (softcap > 0.f) {
    TORCH_CHECK(p_dropout == 0.f,
                "Softcapping does not support dropout for now");
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);
  const int head_size_v = round_multiple(head_size_value, 8);
  const int head_size_v_rounded = round_multiple(head_size_v, 32);
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, 32);
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, 32);

  at::Tensor q_padded, k_padded, v_padded;
  if (head_size_og % 32 != 0) {
    q_padded = at::pad(q, {0, 32 - head_size_og % 32});
    k_padded = at::pad(k, {0, 32 - head_size_og % 32});
  } else {
    q_padded = q;
    k_padded = k;
  }

  if (head_size_value % 32 != 0) {
    v_padded = at::pad(v, {0, 32 - head_size_value % 32});
  } else {
    v_padded = v;
  }

  auto opts = q.options();
  at::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    if (!int8_used && !fp8_used) {
      TORCH_CHECK(out.dtype() == q_dtype,
                  "Output must have the same dtype as inputs");
    } else if (fp8_used) {
      TORCH_CHECK(out.dtype() == at::ScalarType::Half ||
                      out.dtype() == at::ScalarType::BFloat16,
                  "FP8 prefix prefill output must be fp16 or bf16");
    }
    CHECK_DEVICE(out);
    TORCH_CHECK(out.stride(-1) == 1,
                "Output tensor must have contiguous last dimension");
    if (!use_bshd_layout) {
      TORCH_CHECK(out.is_contiguous(),
                  "Prefix prefill bhsd requires output to be contiguous");
    }
    if (head_size_value % 32 != 0) {
      out = at::pad(out, {0, 32 - head_size_value % 32});
    }
  } else {
    // for (bs)hd layout
    if (int8_used || fp8_used) {
      auto int8_opts = is_bf16_output ? opts.dtype(at::ScalarType::BFloat16)
                                      : opts.dtype(at::ScalarType::Half);
      out = at::empty({query_size[0], query_size[1], head_size_v_rounded},
                      int8_opts);
    } else {
      out = at::empty({query_size[0], query_size[1], head_size_v_rounded},
                      opts);
    }
  }

  auto softmax_lse = at::empty({num_heads, total_q}, opts.dtype(at::kFloat));
  at::Tensor p;
  // Only return softmax if there's dropout to reduce compilation time
  if (false /*return_softmax*/) {
    TORCH_CHECK(p_dropout > 0.0f,
                "return_softmax is only supported when p_dropout > 0.0");
    p = at::empty({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded},
                  opts.dtype(at::kFloat));
  }

  if (zero_tensors) {
    out.zero_();
    softmax_lse.fill_(-std::numeric_limits<float>::infinity());
    if (return_softmax) {
      p.zero_();
    }
  }
  Flash_fwd_params params;
  set_params_fprop(
      params, batch_size, max_seqlen_q, max_seqlen_k, seqlen_q_rounded,
      seqlen_k_rounded, num_heads, num_heads_k, head_size, head_size_rounded,
      head_size_v, head_size_v_rounded, q_padded, k_padded, v_padded, out,
      cu_seqlens_q.data_ptr(), seqused_k.data_ptr(),
      return_softmax ? nullptr /*p.data_ptr()*/ : nullptr, seqused_k.data_ptr(),
      softmax_lse.data_ptr(), p_dropout, softmax_scale, window_size_left,
      window_size_right, softcap, false,
      /*unpadded_lse*/ false,
      /*is_kvcache*/ false,
      /*is_seqlens_k_cumulative*/ seqused_k.size(0) == (batch_size + 1),
      layout /*layout*/, false /*is_flashmla*/, true /*is_prefix*/
  );
  params.s_aux_ptr = nullptr;
  params.s_aux_type = 0;
  if (s_aux_.has_value()) {
    auto s_aux = s_aux_.value();
    const auto expected_sink_dtype = out.scalar_type();
    TORCH_CHECK(s_aux.scalar_type() == expected_sink_dtype,
                "Attention sink dtype must match prefix output dtype. Got ",
                s_aux.dtype(), ", expected ", expected_sink_dtype);
    CHECK_DEVICE(s_aux);
    CHECK_CONTIGUOUS(s_aux);
    CHECK_SHAPE(s_aux, num_heads);
    params.s_aux_ptr = s_aux.data_ptr();
    params.s_aux_type = get_attention_sink_type(s_aux.scalar_type());
  }
  params.total_q = total_q;
  params.block_table = block_table.data_ptr<int>();
  params.block_table_batch_stride = block_table.stride(0);
  params.k_batch_stride = k_padded.stride(0);
  params.v_batch_stride = v_padded.stride(0);
  params.page_block_size = page_block_size;
  params.seqused_k = reinterpret_cast<int *>(seqused_k.data_ptr());
  params.layout = layout;

  params.is_int8 = int8_used;
  if (int8_used) {
    params.is_bf16 = is_bf16_output;
    at::Tensor scales_k;
    scales_k = scales_k_.value();
    params.scales_k_ptr = scales_k.data_ptr();
    at::Tensor scales_v;
    scales_v = scales_v_.value();
    params.scales_v_ptr = scales_v.data_ptr();
    at::Tensor scales_q;
    scales_q = scales_q_.value();
    params.scales_q_ptr = scales_q.data_ptr();
    params.total_scale_q = scales_q.numel();
  }
  if (fp8_used) {
    params.is_bf16 = out.dtype() == at::ScalarType::BFloat16;
    params.is_e4m3 = true;
    auto set_fp8_descale = [](const at::Tensor &descale, const char *name) {
      CHECK_DEVICE(descale);
      TORCH_CHECK(descale.dtype() == at::ScalarType::Float,
                  name, " must have dtype float32");
      TORCH_CHECK(descale.numel() >= 1,
                  name, " must contain at least one element");
      return reinterpret_cast<float*>(descale.data_ptr());
    };
    at::Tensor scales_q = scales_q_.value();
    params.q_descale_ptr = set_fp8_descale(scales_q, "q_descale");
    params.q_descale_batch_stride = 0;
    params.q_descale_head_stride = 0;
    at::Tensor scales_k = scales_k_.value();
    params.k_descale_ptr = set_fp8_descale(scales_k, "k_descale");
    params.k_descale_batch_stride = 0;
    params.k_descale_head_stride = 0;
    at::Tensor scales_v = scales_v_.value();
    params.v_descale_ptr = set_fp8_descale(scales_v, "v_descale");
    params.v_descale_batch_stride = 0;
    params.v_descale_head_stride = 0;
  }

  at::Tensor rng_state;
  if (p_dropout > 0) {
    auto options = at::TensorOptions()
                       .dtype(at::ScalarType::Float)
                       .device(at::DeviceType::CUDA);
    rng_state = at::empty({2}, options.dtype(at::ScalarType::Long));
    // Forward kernel will populate memory with the seed and offset.
    params.rng_state = reinterpret_cast<uint64_t *>(rng_state.data_ptr());
  } else {
    params.rng_state = nullptr;
  }

  set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

  const char *fa_debug = std::getenv("FA_DEBUG");
  if (fa_debug != nullptr) {
    if (std::strcmp(fa_debug, "1") == 0) {
      PRINT_PARAMS
    } else if (std::strcmp(fa_debug, "2") == 0) {
      PRINT_PARAMS_ONELINE
      auto temp_tensor = seqused_k.to(at::DeviceType::CPU).contiguous();
      std::vector<int32_t> temp_vector(temp_tensor.data_ptr<int32_t>(),
                                       temp_tensor.data_ptr<int32_t>() +
                                           temp_tensor.numel());
      printf("seqused_k: [");
      for (const auto val : temp_vector) {
        printf("%d ", val);
      }
      printf("]\n");
    }
    PRINT_QKV_INFO(q, k, v)
  }

  const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
  run_mha_fwd(params, stream);

  at::Tensor out_padded = out;
  if (head_size_value % 32 != 0) {
    out = out.index(
        {"...", at::indexing::Slice(at::indexing::None, head_size_value)});
    if (out_.has_value()) {
      out_.value().copy_(out);
    }
  }

  // return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
  if (return_softmax) return {out, softmax_lse};
  else return {out};
#else
  return {};
#endif
}


std::vector<at::Tensor> prefix_prefill_varlen_fwd_mla(
    at::Tensor &q,
    at::Tensor &kcache,
    at::Tensor &vcache,
    at::Tensor &qv,
    at::Tensor &page_table,
    at::Tensor &cache_seqlens,
    at::Tensor &cu_seqlens_q,
    at::Tensor &cu_seqlens_k_new,
    const int max_seqlen_q,
    const float softmax_scale,
    const bool causal,
    const float softcap,
    c10::optional<const at::Tensor> &k_descale,
    c10::optional<const at::Tensor> &v_descale,
    const bool return_softmax_lse,
    const bool is_mtp
) {
#if defined(BUILD_FA_FWD)
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
    // 类型检查
    TORCH_CHECK(q.dtype() == at::ScalarType::Half || q.dtype() == at::ScalarType::BFloat16, "Prefix prefill forward mla only support fp16 and bf16 data type for q");
    TORCH_CHECK(kcache.dtype() == at::ScalarType::Half || kcache.dtype() == at::ScalarType::BFloat16, "Prefix prefill forward mla only support fp16 and bf16 data type for kcache");
    TORCH_CHECK(vcache.dtype() == at::ScalarType::Half || vcache.dtype() == at::ScalarType::BFloat16, "Prefix prefill forward mla only support fp16 and bf16 data type for vcache");
    TORCH_CHECK(qv.dtype() == at::ScalarType::Half || qv.dtype() == at::ScalarType::BFloat16, "Prefix prefill forward mla only support fp16 and bf16 data type for qv");
    TORCH_CHECK(page_table.dtype() == at::ScalarType::Int, "Prefix prefill forward mla only support int32_t data type for page_table");
    TORCH_CHECK(cache_seqlens.dtype() == at::ScalarType::Int, "Prefix prefill forward mla only support int32_t data type for cache_seqlens");
    TORCH_CHECK(cu_seqlens_q.dtype() == at::ScalarType::Int, "Prefix prefill forward mla only support int32_t data type for cu_seqlens_q");
    TORCH_CHECK(cu_seqlens_k_new.dtype() == at::ScalarType::Int, "Prefix prefill forward mla only support int32_t data type for cu_seqlens_k_new");
    // device 检查
    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache); CHECK_DEVICE(qv); CHECK_DEVICE(page_table); CHECK_DEVICE(cache_seqlens); CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k_new);
    // 连续性检查
    CHECK_CONTIGUOUS(page_table); CHECK_CONTIGUOUS(cache_seqlens); CHECK_CONTIGUOUS(cu_seqlens_q); CHECK_CONTIGUOUS(cu_seqlens_k_new);
    // 张量 shape 检查, 是否是 3/4 维这种
    TORCH_CHECK(q.dim() == 3, "In prefix prefill forward mla, q must be 3-dimension tensor");
    TORCH_CHECK(kcache.dim() == 4, "In prefix prefill forward mla, kcache must be 4-dimension tensor");
    TORCH_CHECK(vcache.dim() == 4, "In prefix prefill forward mla, vcache must be 4-dimension tensor");
    TORCH_CHECK(qv.dim() == 3, "In prefix prefill forward mla, qv must be 3-dimension tensor");
    TORCH_CHECK(page_table.dim() == 2, "In prefix prefill forward mla, page_table must be 2-dimension tensor");
    // 获取基本信息
    const auto q_size = q.sizes();
    const auto qv_size = qv.sizes();
    const auto kcache_size = kcache.sizes();
    const auto vcache_size = vcache.sizes();
    const int batch_size = page_table.size(0);
    const int qheads = q_size[1];
    const int kvheads = kcache_size[2];
    const int headdim_v = vcache_size[3];
    const int headdim_rope = q_size[2];
    const int headdim_qk = headdim_v + headdim_rope;
    const int page_block_size = kcache_size[1];
    // 检查 size 是否符合要求
    TORCH_CHECK(qheads % kvheads == 0, "In prefix prefill forward mla, qheads must be multiple of kvheads");
    TORCH_CHECK(headdim_v == 512, "In prefix prefill forward mla, headdim_v must be 512");
    TORCH_CHECK(headdim_rope == 64, "In prefix prefill forward mla, headdim_rope must be 64");
    TORCH_CHECK(headdim_qk == 576, "In prefix prefill forward mla, headdim_qk must be 576");
    TORCH_CHECK(page_block_size == 128, "In prefix prefill forward mla, page_block_size must be 128")
    // 检查 size 是否匹配
    TORCH_CHECK(q_size[2] == kcache_size[3], "In prefix prefill forward mla, headdim must match between q and kcache");
    TORCH_CHECK(qv_size[2] == vcache_size[3], "In prefix prefill forward mla, headdim must match between qv and vcache");
    // 检查平台
    hipDeviceProp_t props;
    auto hipResult = hipGetDeviceProperties(&props, 0);
    std::string gcn_arch_name(props.gcnArchName);
    const int gcn_arch = runtime_gfx_arch_id(gcn_arch_name);
    TORCH_CHECK(is_supported_hg_mla_arch(gcn_arch_name, gcn_arch), "In prefix prefill forward mla, only gfx92a or arch id >= gfx936 is supported!");
    // 准备输出变量
    auto opts = q.options();
    at::Tensor out, softmax_lse, scores_max, scores_sum;
    out = at::empty({q_size[0], q_size[1], headdim_v}, opts);
    if (true/*return_softmax_lse*/) {
        auto scores_memory = at::empty({3, qheads, q_size[0]}, opts.dtype(at::kFloat));
        scores_max = scores_memory.index({0});
        scores_sum = scores_memory.index({1});
        softmax_lse = scores_memory.index({2});
    }
    // 准备 kernel 需要的参数列表
    Flash_fwd_mla_params params;
    memset(&params, 0, sizeof(params));
    params.layout           = 1;
    params.b                = batch_size;
    params.h                = qheads;
    params.h_k              = kvheads;
    params.h_h_k_ratio      = int(qheads / kvheads);
    params.total_q          = q_size[0];
    params.scale_softmax    = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.cu_seqlens_q     = reinterpret_cast<int32_t*>(cu_seqlens_q.data_ptr());
    params.cu_seqlens_k_new = reinterpret_cast<int32_t*>(cu_seqlens_k_new.data_ptr());
    params.q_ptr            = q.data_ptr();
    params.qv_ptr           = qv.data_ptr();
    params.k_ptr            = kcache.data_ptr();
    params.v_ptr            = vcache.data_ptr();
    params.o_ptr            = out.data_ptr();
    params.softmax_lse_ptr  = softmax_lse.data_ptr<float>();
    params.scores_max_ptr   = scores_max.data_ptr<float>();
    params.scores_sum_ptr   = scores_sum.data_ptr<float>();
    params.block_table      = reinterpret_cast<int32_t*>(page_table.data_ptr());
    params.block_table_batch_stride = page_table.stride(0);
    params.page_block_size  = page_block_size;
    params.is_causal        = causal;
    params.q_row_stride     = q.stride(0);
    params.q_head_stride    = q.stride(1);
    params.qv_row_stride    = qv.stride(0);
    params.qv_head_stride   = qv.stride(1);
    params.k_batch_stride   = kcache.stride(0);
    params.k_row_stride     = kcache.stride(1);
    params.k_head_stride    = kcache.stride(2);
    params.v_batch_stride   = vcache.stride(0);
    params.v_row_stride     = vcache.stride(1);
    params.v_head_stride    = vcache.stride(2);
    params.o_row_stride     = out.stride(0);
    params.o_head_stride    = out.stride(1);
    params.seqlen_q         = max_seqlen_q;
    params.is_bf16          = q.dtype() == at::ScalarType::BFloat16;
    params.cu_count         = props.multiProcessorCount;
    params.mtp              = is_mtp;                   // A flag to ensure whether prefill or decode

    // 准备启动 kernel
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    const char* fa_debug = std::getenv("FA_DEBUG");
    if (fa_debug != nullptr) {
        PRINT_MLA_PARAMS
        if (strcmp(fa_debug, "2") == 0) { // print operations listed below may interrupt cudagraph, and thus only print tensors util FA_DEBUG=2
            PRINT_TENSOR(cache_seqlens, "cache_seqlens")
            PRINT_TENSOR(cu_seqlens_q, "cu_seqlens_q")
            PRINT_TENSOR(cu_seqlens_k_new, "cu_seqlens_k_new")
        }
        PRINT_TENSOR_INFO(q, "q")
        PRINT_TENSOR_INFO(kcache, "kcache")
        PRINT_TENSOR_INFO(vcache, "vcache")
        PRINT_TENSOR_INFO(qv, "qv")
    }

    if (max_seqlen_q > 0 and std::getenv("MLA_PREFILL_EMPTY") == nullptr) {
        run_fwd_prefix_prefill_mla(params, stream);
    } else {
        out.zero_();
    }

    return {out, softmax_lse, scores_max, scores_sum};
#else
    return {};
#endif
}

#if defined(BUILD_FA_BWD)
#include "flash_sumout_api.h"
namespace inner {
    void sum_out(at::Tensor &output, at::Tensor input, int dim) {
     auto dtype = input.dtype();
     const int stride0       = input.stride(dim);
     const int stride1       = input.stride(dim-1);
     const int num_elem      = output.numel();
     const int num_thread    = 256;
     const int num_grid      = num_elem / num_thread;
     if(dtype == at::ScalarType::Half)
         flash_sum_out<Float16><<<num_grid, num_thread>>>(reinterpret_cast<Float16 *>(output.data_ptr()), reinterpret_cast<Float16 *>(input.data_ptr()), stride0, stride1);
     else if (dtype == at::ScalarType::BFloat16)
         flash_sum_out<BFloat16><<<num_grid, num_thread>>>(reinterpret_cast<BFloat16 *>(output.data_ptr()), reinterpret_cast<BFloat16 *>(input.data_ptr()), stride0, stride1);
 }
 }
#endif

std::vector<at::Tensor>
bwd_base(const at::Tensor &dout,  // batch_size x num_heads x seqlen_q x head_size_og
        const at::Tensor &q,   // batch_size x num_heads x seqlen_q x head_size
        const at::Tensor &k,   // batch_size x num_heads x seqlen_q x head_size
        const at::Tensor &v,   // batch_size x num_heads x seqlen_q x head_size
        const at::Tensor &out,   // batch_size x num_heads x seqlen_q x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        c10::optional<at::Tensor> &dq_,   // batch_size x num_heads x seqlen_q x head_size
        c10::optional<at::Tensor> &dk_,   // batch_size x num_heads x seqlen_q x head_size
        c10::optional<at::Tensor> &dv_,   // batch_size x num_heads x seqlen_q x head_size
        c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        c10::optional<at::Generator> gen_,
        c10::optional<at::Tensor> &rng_state,
        const int layout
    ) {
#if defined(BUILD_FA_BWD)
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
    if (is_causal) { window_size_right = 0; }

    bool is_dropout = p_dropout > 0.0;
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16 || q_dtype == at::ScalarType::Float8_e4m3fn,
            "FlashAttention only support fp16,bf16,e4m3 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    TORCH_CHECK(layout == 0 || layout == 1, "layout only supports 0 or 1");
    const bool use_bshd_layout = bool(layout == 1);
    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int num_heads = use_bshd_layout ? sizes[2]: sizes[1];
    const int seqlen_q = use_bshd_layout ? sizes[1]: sizes[2];
    const int head_size_value = v.size(3);
    const int head_size = sizes[3];
    const int num_heads_k = use_bshd_layout ? k.size(2): k.size(1);
    const int seqlen_k = use_bshd_layout ? k.size(1): k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(head_size_value <= 256, "FlashAttention backward only supports head dimension at most 256");


    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(int64_t(batch_size * num_heads * seqlen_q * head_size) < /*2^31*/int64_t(2147483648), "The data amount of q must be smaller than the representation range of int");
    TORCH_CHECK(int64_t(batch_size * num_heads_k * seqlen_k * head_size) < /*2^31*/int64_t(2147483648), "The data amount of k/v must be smaller than the representation range of int");
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int head_size_value_rounded = round_multiple(head_size_value, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    if (layout == 0) {
        CHECK_SHAPE(q, batch_size, num_heads, seqlen_q, head_size);
        CHECK_SHAPE(k, batch_size, num_heads_k, seqlen_k, head_size);
        CHECK_SHAPE(v, batch_size, num_heads_k, seqlen_k, head_size_value);
        CHECK_SHAPE(out, batch_size, num_heads, seqlen_q, head_size_value);
        CHECK_SHAPE(dout, batch_size, num_heads, seqlen_q, dout.size(-1));
    } else {
        CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
        CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
        CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_value);
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_value);
        CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, dout.size(-1));
    }

    auto opts = q.options();
    at::Tensor q_padded, k_padded, v_padded, out_padded, dq_padded, dk_padded, dv_padded, dout_padded;
    if (head_size % 32 != 0) {
        q_padded = at::pad(q, {0, 32 - head_size % 32});
        k_padded = at::pad(k, {0, 32 - head_size % 32});
    } else {
        q_padded = q;
        k_padded = k;
    }

    if (head_size_value % 32 != 0) {
        v_padded = at::pad(v, {0, 32 - head_size_value % 32});
        out_padded = at::pad(out, {0, 32 - head_size_value % 32});
    } else {
        v_padded = v;
        out_padded = out;
    }

    if (dout.size(-1) % 32 != 0) {
        dout_padded = at::pad(dout, {0, 32 - dout.size(-1) % 32});
    } else {
        dout_padded = dout;
    }

    if(dq_.has_value()){
        if(layout == 0) {
            CHECK_SHAPE(dq_.value(), batch_size, num_heads, seqlen_q, head_size);
        } else {
            CHECK_SHAPE(dq_.value(), batch_size, seqlen_q, num_heads, head_size);
        }
        if (head_size % 32 != 0) {
            dq_padded = at::pad(dq_.value(), {0, 32 - head_size % 32});
        } else {
            dq_padded = dq_.value();
        }
    } else {
        if (layout == 0) {
            dq_padded = at::empty({batch_size, num_heads, seqlen_q, head_size_rounded}, opts);
        } else {
            dq_padded = at::empty({batch_size, seqlen_q, num_heads, head_size_rounded}, opts);
        }
    }

    if(dk_.has_value()){
        if(layout == 0) {
            CHECK_SHAPE(dk_.value(), batch_size, num_heads_k, seqlen_k, head_size);
        } else {
            CHECK_SHAPE(dk_.value(), batch_size, seqlen_k, num_heads_k, head_size);
        }
        if (head_size % 32 != 0) {
            dk_padded = at::pad(dk_.value(), {0, 32 - head_size % 32});
        } else {
            dk_padded = dk_.value();
        }
    } else {
        if (layout == 0) {
            dk_padded = at::empty({batch_size, num_heads_k, seqlen_k, head_size_rounded}, opts);
        } else {
            dk_padded = at::empty({batch_size, seqlen_k, num_heads_k, head_size_rounded}, opts);
        }
    }

    if(dv_.has_value()){
        if(layout == 0) {
            CHECK_SHAPE(dv_.value(), batch_size, num_heads_k, seqlen_k, head_size_value);
        } else {
            CHECK_SHAPE(dv_.value(), batch_size, seqlen_k, num_heads_k, head_size_value);
        }
        if (head_size_value % 32 != 0) {
            dv_padded = at::pad(dv_.value(), {0, 32 - head_size_value % 32});
        } else {
            dv_padded = dv_.value();
        }
    } else {
        if (layout == 0) {
            dv_padded = at::empty({batch_size, num_heads_k, seqlen_k, head_size_value_rounded}, opts);
        } else {
            dv_padded = at::empty({batch_size, seqlen_k, num_heads_k, head_size_value_rounded}, opts);
        }
    }

    // // Otherwise the kernel will be launched from cuda:0 device
    // // Cast to char to avoid compiler warning about narrowing
    // at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto softmax_d = at::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    at::Tensor dk_accum, dv_accum;
    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        if(layout == 0){
            dk_expanded = at::empty({batch_size, num_heads, seqlen_k, head_size_rounded}, opts);
            dv_expanded = at::empty({batch_size, num_heads, seqlen_k, head_size_value_rounded}, opts);
        } else{
            dk_expanded = at::empty({batch_size, seqlen_k, num_heads, head_size_rounded}, opts);
            dv_expanded = at::empty({batch_size, seqlen_k, num_heads, head_size_value_rounded}, opts);
        }

    } else {
        dk_expanded = dk_padded;
        dv_expanded = dv_padded;
    }

#ifdef DEBUGING
        at::Tensor dev_kq, dev_s, dev_dp, dev_ds;
        if(layout == 0){
            dev_kq = at::empty({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(at::kFloat));
            dev_kq.fill_(float('-inf'));
            dev_s  = at::zeros({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(at::kFloat));
            dev_dp = at::zeros({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(at::kFloat));
            dev_ds = at::zeros({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(at::kFloat));
        } else {
            dev_kq = at::zeros({batch_size, seqlen_q, num_heads, seqlen_k}, opts.dtype(at::kFloat));
            dev_kq.fill_(float('-inf'));
            dev_s  = at::zeros({batch_size, seqlen_q, num_heads, seqlen_k}, opts.dtype(at::kFloat));
            dev_dp = at::zeros({batch_size, seqlen_q, num_heads, seqlen_k}, opts.dtype(at::kFloat));
            dev_ds = at::zeros({batch_size, seqlen_q, num_heads, seqlen_k}, opts.dtype(at::kFloat));
        }
#endif

    // std::cout << "q_padded:\n" << q_padded << std::endl;
    // std::cout << "k_padded:\n" << k_padded << std::endl;
    // std::cout << "v_padded:\n" << v_padded << std::endl;
    // std::cout << "out_padded:\n" << out_padded << std::endl;
    // std::cout << "dout_padded:\n" << dout_padded << std::endl;

    Flash_bwd_params params;
    set_params_dgrad(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     head_size_value, head_size_value_rounded,
                     q_padded, k_padded, v_padded, out_padded,
                     dout_padded, dq_padded, dk_expanded, dv_expanded,
                     nullptr,
                     nullptr,
                     nullptr/*p_d.data_ptr()*/,
#ifdef DEBUGING
                     dev_kq.data_ptr(),
                     dev_s.data_ptr(),
                     dev_dp.data_ptr(),
                     dev_ds.data_ptr(),
#endif
                     nullptr,
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     deterministic,
                     /*unpadded_lse*/false,
                     layout
                    );
    // std::cout<<"params.q_row_stride = "<< params.q_row_stride<<std::endl;
    // std::cout<<"params.k_row_stride = "<<params.k_row_stride<<std::endl;
    // std::cout<<"params.v_row_stride = "<<params.v_row_stride<<std::endl;
    // std::cout<<"params.o_row_stride = "<<params.o_row_stride<<std::endl;
    // std::cout<<"params.q_head_stride = "<<params.q_head_stride<<std::endl;
    // std::cout<<"params.k_head_stride = "<<params.k_head_stride<<std::endl;
    // std::cout<<"params.v_head_stride = "<<params.v_head_stride<<std::endl;
    // std::cout<<"params.o_head_stride = "<<params.o_head_stride<<std::endl;
    // std::cout<<"params.dq_row_stride = "<< params.dq_row_stride<<std::endl;
    // std::cout<<"params.dk_row_stride = "<<params.dk_row_stride<<std::endl;
    // std::cout<<"params.dv_row_stride = "<<params.dv_row_stride<<std::endl;
    // std::cout<<"params.do_row_stride = "<<params.do_row_stride<<std::endl;
    // std::cout<<"params.dq_head_stride = "<<params.dq_head_stride<<std::endl;
    // std::cout<<"params.dk_head_stride = "<<params.dk_head_stride<<std::endl;
    // std::cout<<\"params.dv_head_stride = \"<<params.dv_head_stride<<std::endl;
    // std::cout<<\"params.do_head_stride = \"<<params.do_head_stride<<std::endl;

    auto launch = &run_mha_bwd;
    // launch(params, stream, /*configure=*/true);

    // auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
    //     gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    at::Tensor rng_state_tensor;
    if ( rng_state.has_value() ) {
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
        const char *fa_debug = std::getenv("FA_DEBUG");
        if (fa_debug != nullptr && std::strcmp(fa_debug, "dropout") == 0) {
            std::cout << "params.rng_state[0] = " << params.rng_state[0] << std::endl;
            std::cout << "params.rng_state[1] = " << params.rng_state[1] << std::endl;
        }
    }
    else if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        rng_state_tensor = at::empty({2}, opts.dtype(at::ScalarType::Long));
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state_tensor.data_ptr());
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        std::lock_guard<std::mutex> lock(gen->mutex_);
        at::PhiloxCudaState philox_args = gen->philox_cuda_state(counter_offset);
        // at::cuda::philox::unpack(philox_args) not supported on ROCm
        params.rng_state[0] = philox_args.seed_.val;
        params.rng_state[1] = philox_args.offset_.val;
    }
    if (is_dropout) {
        params.rand_seed = params.rng_state[0];
        params.rand_offset = params.rng_state[1];
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    const hipStream_t stream = nullptr;//at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    launch(params, stream, /*configure=*/false);
    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        if(layout == 0){
            sum_out(dk_padded, at::reshape(dk_expanded, {batch_size, num_heads_k, num_heads / num_heads_k, seqlen_k, head_size_rounded}), 2);
            sum_out(dv_padded, at::reshape(dv_expanded, {batch_size, num_heads_k, num_heads / num_heads_k, seqlen_k, head_size_value_rounded}), 2);
        } else {
            sum_out(dk_padded, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size_rounded}), 3);
            sum_out(dv_padded, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size_value_rounded}), 3);
        }

    }
    at::Tensor dq, dk, dv;
    if (head_size % 32 != 0) {
        dq = dq_padded.index({"...", at::indexing::Slice(at::indexing::None, head_size)});
        dk = dk_padded.index({"...", at::indexing::Slice(at::indexing::None, head_size)});
    } else {
        dq = dq_padded;
        dk = dk_padded;
    }
    if (head_size_value % 32 != 0) {
        dv = dv_padded.index({"...", at::indexing::Slice(at::indexing::None, head_size_value)});
    } else {
        dv = dv_padded;
    }

    // std::cout<<"q.sizes() = "<<q.sizes()<<std::endl;
    // std::cout<<"k.sizes() = "<<k.sizes()<<std::endl;
    // std::cout<<"out.sizes() = "<<out.sizes()<<std::endl;
    // std::cout<<"num_heads = "<<num_heads<<std::endl;
    // std::cout<<"dk.sizes() = "<<dk.sizes()<<std::endl;
    // std::cout<<"layout="<<layout<<std::endl;
    // std::cout<<"dq.sizes() = "<<dq.sizes()<<std::endl;
    // std::cout<<"dq.stride() = "<<dq.stride(0)<<" "<<dq.stride(1)<<" "<<dq.stride(2)<<" "<<dq.stride(3)<<std::endl;
    // std::cout<<"q.stride() = "<<q.stride(0)<<" "<<q.stride(1)<<" "<<q.stride(2)<<" "<<q.stride(3)<<std::endl;
    // std::cout<<"dk.sizes() = "<<dk.sizes()<<std::endl;
    // std::cout<<"dv.sizes() = "<<dv.sizes()<<std::endl;
    // std::cout<<"num_heads_k = "<<num_heads_k<<std::endl;
    // std::cout<<"num_heads = "<<num_heads<<std::endl;
    // std::cout<<"dq.sizes() = "<<dq.sizes()<<std::endl;
    // std::cout<<"dk.sizes() = "<<dk.sizes()<<std::endl;
    // std::cout<<"dv.sizes() = "<<dv.sizes()<<std::endl;

    #ifdef DEBUGING
        return { dq, dk, dv, softmax_d, dev_kq.clone(), dev_s.clone(), dev_dp.clone(), dev_ds.clone()};
    #else
        return { dq, dk, dv, softmax_d };
    #endif
#else
    return {};
#endif
}

std::vector<at::Tensor>
hg_bwd_bhsd(const at::Tensor &dout,  // batch_size x num_heads x seqlen_q x head_size_og
    const at::Tensor &q,   // batch_size x num_heads x seqlen_q x head_size
    const at::Tensor &k,   // batch_size x num_heads x seqlen_q x head_size
    const at::Tensor &v,   // batch_size x num_heads x seqlen_q x head_size
    const at::Tensor &out,   // batch_size x num_heads x seqlen_q x head_size
    const at::Tensor &softmax_lse,     // b x h x seqlen_q
    c10::optional<at::Tensor> &dq_,   // batch_size x num_heads x seqlen_q x head_size
    c10::optional<at::Tensor> &dk_,   // batch_size x num_heads x seqlen_q x head_size
    c10::optional<at::Tensor> &dv_,   // batch_size x num_heads x seqlen_q x head_size
    c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
    const float p_dropout,         // probability to drop
    const float softmax_scale,
    const bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool deterministic,
    c10::optional<at::Generator> gen_,
    c10::optional<at::Tensor> &rng_state
) {
    return bwd_base(dout, q, k, v, out, softmax_lse, dq_, dk_, dv_, alibi_slopes_, p_dropout, softmax_scale, is_causal, window_size_left, window_size_right, softcap, deterministic, gen_, rng_state, 0);
}

std::vector<at::Tensor>
hg_bwd_bshd(const at::Tensor &dout,  // batch_size x num_heads x seqlen_q x head_size_og
    const at::Tensor &q,   // batch_size x num_heads x seqlen_q x head_size
    const at::Tensor &k,   // batch_size x num_heads x seqlen_q x head_size
    const at::Tensor &v,   // batch_size x num_heads x seqlen_q x head_size
    const at::Tensor &out,   // batch_size x num_heads x seqlen_q x head_size
    const at::Tensor &softmax_lse,     // b x h x seqlen_q
    c10::optional<at::Tensor> &dq_,   // batch_size x num_heads x seqlen_q x head_size
    c10::optional<at::Tensor> &dk_,   // batch_size x num_heads x seqlen_q x head_size
    c10::optional<at::Tensor> &dv_,   // batch_size x num_heads x seqlen_q x head_size
    c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
    const float p_dropout,         // probability to drop
    const float softmax_scale,
    const bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool deterministic,
    c10::optional<at::Generator> gen_,
    c10::optional<at::Tensor> &rng_state
) {
    return bwd_base(dout, q, k, v, out, softmax_lse, dq_, dk_, dv_, alibi_slopes_, p_dropout, softmax_scale, is_causal, window_size_left, window_size_right, softcap, deterministic, gen_, rng_state, 1);
}

std::vector<at::Tensor>
hg_varlen_bwd_bshd(const at::Tensor &dout,  // total_q_heads x head_size, total_q_heads := \sum_{i=0}^{b} s_i x num_heads
                    const at::Tensor &q,  // total_q_heads x head_size, total_q_heads := \sum_{i=0}^{b} s_i x num_heads
                    const at::Tensor &k,  // total_k_heads x head_size, total_k_heads := \sum_{i=0}^{b} s_i x num_heads_k
                    const at::Tensor &v,  // total_k_heads x head_size, total_k_heads := \sum_{i=0}^{b} s_i x num_heads_k
                    const at::Tensor &out, // total_q_heads x head_size, total_q_heads := \sum_{i=0}^{b} s_i x num_heads
                    const at::Tensor &softmax_lse,     // b x h x s   softmax logsumexp
                    c10::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                    c10::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                    c10::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                    const at::Tensor &cu_seqlens_q,  // b+1
                    const at::Tensor &cu_seqlens_k,  // b+1
                    c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
                    const int max_seqlen_q,
                    const int max_seqlen_k,          // max sequence length to choose the kernel
                    const float p_dropout,         // probability to drop
                    const float softmax_scale,
                    const bool zero_tensors,
                    const bool is_causal,
                    int window_size_left,
                    int window_size_right,
                    const float softcap,
                    const bool deterministic,
                    c10::optional<at::Generator> gen_,
                    c10::optional<at::Tensor> &rng_state
        #ifdef DEBUGING
                        ,
                        const at::Tensor &dev_kq,
                        const at::Tensor &dev_s,
                        const at::Tensor &dev_dp,
                        const at::Tensor &dev_ds
        #endif
                    ) {
#if defined(BUILD_FA_BWD)
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
    const int layout = 1;
    if (is_causal) { window_size_right = 0; }

    bool is_dropout = p_dropout > 0.0;
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16 || q_dtype == at::ScalarType::Float8_e4m3fn,
            "FlashAttention only support fp16,bf16,e4m3 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    //support MLA
    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size_value = v.size(2);
    const int head_size = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(head_size_value <= 256, "FlashAttention backward only supports head dimension at most 256");


    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(int64_t(total_q * num_heads * head_size) < /*2^31*/int64_t(2147483648), "The data amount of q must be smaller than the representation range of int");
    TORCH_CHECK(int64_t(total_k * num_heads_k * head_size) < /*2^31*/int64_t(2147483648), "The data amount of k/v must be smaller than the representation range of int");
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int head_size_value_rounded = round_multiple(head_size_value, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);
    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(dout, total_q, num_heads, dout.size(-1));
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    auto opts = q.options();
    at::Tensor q_padded, k_padded, v_padded, out_padded, dq_padded, dk_padded, dv_padded, dout_padded;
    if (head_size % 32 != 0) {
        q_padded = at::pad(q, {0, 32 - head_size % 32});
        k_padded = at::pad(k, {0, 32 - head_size % 32});
    } else {
        q_padded = q;
        k_padded = k;
    }

    if (head_size_value % 32 != 0) {
        v_padded = at::pad(v, {0, 32 - head_size_value % 32});
        out_padded = at::pad(out, {0, 32 - head_size_value % 32});
    } else {
        v_padded = v;
        out_padded = out;
    }

    if (dout.size(-1) % 32 != 0) {
        dout_padded = at::pad(dout, {0, 32 - dout.size(-1) % 32});
    } else {
        dout_padded = dout;
    }

    if(dq_.has_value()){
        CHECK_SHAPE(dq_.value(), total_q, num_heads, head_size);
        if (head_size % 32 != 0) {
            dq_padded = at::pad(dq_.value(), {0, 32 - head_size % 32});
        } else {
            dq_padded = dq_.value();
        }
    } else {
        dq_padded = at::empty({total_q, num_heads, head_size_rounded}, opts);
    }

    if(dk_.has_value()){
        CHECK_SHAPE(dk_.value(), total_k, num_heads_k, head_size);
        if (head_size % 32 != 0) {
            dk_padded = at::pad(dk_.value(), {0, 32 - head_size % 32});
        } else {
            dk_padded = dk_.value();
        }
    } else {
        dk_padded = at::empty({total_k, num_heads_k, head_size_rounded}, opts);
    }

    if(dv_.has_value()){
        CHECK_SHAPE(dv_.value(), total_k, num_heads_k, head_size_value);
        if (head_size_value % 32 != 0) {
            dv_padded = at::pad(dv_.value(), {0, 32 - head_size_value % 32});
        } else {
            dv_padded = dv_.value();
        }
    } else {
        dv_padded = at::empty({total_k, num_heads_k, head_size_value_rounded}, opts);
    }

    auto softmax_d = at::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    at::Tensor dk_accum, dv_accum;
    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = at::empty({total_k, num_heads, head_size_rounded}, opts);
        dv_expanded = at::empty({total_k, num_heads, head_size_value_rounded}, opts);
    } else {
        dk_expanded = dk_padded;
        dv_expanded = dv_padded;
    }

    #ifdef DEBUGING
        at::Tensor dev_kq, dev_s, dev_dp, dev_ds;
        if(layout == 0){
            dev_kq = at::empty({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(at::kFloat));
            dev_kq.fill_(float('-inf'));
            dev_s  = at::zeros({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(at::kFloat));
            dev_dp = at::zeros({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(at::kFloat));
            dev_ds = at::zeros({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(at::kFloat));
        } else {
            dev_kq = at::zeros({batch_size, seqlen_q, num_heads, seqlen_k}, opts.dtype(at::kFloat));
            dev_kq.fill_(float('-inf'));
            dev_s  = at::zeros({batch_size, seqlen_q, num_heads, seqlen_k}, opts.dtype(at::kFloat));
            dev_dp = at::zeros({batch_size, seqlen_q, num_heads, seqlen_k}, opts.dtype(at::kFloat));
            dev_ds = at::zeros({batch_size, seqlen_q, num_heads, seqlen_k}, opts.dtype(at::kFloat));
        }
    #endif

    // std::cout << "q_padded:\n" << q_padded << std::endl;
    // std::cout << "k_padded:\n" << k_padded << std::endl;
    // std::cout << "v_padded:\n" << v_padded << std::endl;
    // std::cout << "out_padded:\n" << out_padded << std::endl;
    // std::cout << "dout_padded:\n" << dout_padded << std::endl;

    Flash_bwd_params params;
    set_params_dgrad(params,
                    batch_size,
                    max_seqlen_q, max_seqlen_k,
                    seqlen_q_rounded, seqlen_k_rounded,
                    num_heads, num_heads_k,
                    head_size, head_size_rounded,
                    head_size_value, head_size_value_rounded,
                    q_padded, k_padded, v_padded, out_padded,
                    dout_padded, dq_padded, dk_expanded, dv_expanded,
                    cu_seqlens_q.data_ptr(),
                    cu_seqlens_k.data_ptr(),
                    nullptr/*p_d.data_ptr()*/,
    #ifdef DEBUGING
                    dev_kq.data_ptr(),
                    dev_s.data_ptr(),
                    dev_dp.data_ptr(),
                    dev_ds.data_ptr(),
    #endif
                    nullptr,
                    nullptr,
                    nullptr,
                    softmax_lse.data_ptr(),
                    softmax_d.data_ptr(),
                    p_dropout,
                    softmax_scale,
                    window_size_left,
                    window_size_right,
                    softcap,
                    deterministic,
                    /*unpadded_lse*/false,
                    layout
                    );
    params.total_q = total_q;
    params.total_k = total_k;
    auto launch = &run_mha_bwd;
    // launch(params, stream, /*configure=*/true);

    // auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
    //     gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;

    at::Tensor rng_state_tensor;
    if ( rng_state.has_value() ) {
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
    }
    else if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        rng_state_tensor = at::empty({2}, opts.dtype(at::ScalarType::Long));
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state_tensor.data_ptr());
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        std::lock_guard<std::mutex> lock(gen->mutex_);
        at::PhiloxCudaState philox_args = gen->philox_cuda_state(counter_offset);
        // at::cuda::philox::unpack(philox_args) not supported on ROCm
        params.rng_state[0] = philox_args.seed_.val;
        params.rng_state[1] = philox_args.offset_.val;
    }
    if (is_dropout) {
        params.rand_seed = params.rng_state[0];
        params.rand_offset = params.rng_state[1];
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    const hipStream_t stream = nullptr;//at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    launch(params, stream, /*configure=*/false);
    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        // inner::sum_out(dk_padded, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size_rounded}), 2);
        // inner::sum_out(dv_padded, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size_value_rounded}), 2);
        at::sum_out(dk_padded, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size_rounded}), {2});
        at::sum_out(dv_padded, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size_value_rounded}), {2});
    }
    at::Tensor dq, dk, dv;
    if (head_size % 32 != 0) {
        dq = dq_padded.index({"...", at::indexing::Slice(at::indexing::None, head_size)});
        dk = dk_padded.index({"...", at::indexing::Slice(at::indexing::None, head_size)});
    } else {
        dq = dq_padded;
        dk = dk_padded;
    }
    if (head_size_value % 32 != 0) {
        dv = dv_padded.index({"...", at::indexing::Slice(at::indexing::None, head_size_value)});
    } else {
        dv = dv_padded;
    }

    // std::cout<<"q.sizes() = "<<q.sizes()<<std::endl;
    // std::cout<<"k.sizes() = "<<k.sizes()<<std::endl;
    // std::cout<<"out.sizes() = "<<out.sizes()<<std::endl;
    // std::cout<<"num_heads = "<<num_heads<<std::endl;
    // std::cout<<"dk.sizes() = "<<dk.sizes()<<std::endl;
    // std::cout<<"layout="<<layout<<std::endl;
    // std::cout<<"dq.sizes() = "<<dq.sizes()<<std::endl;
    // std::cout<<"dq.stride() = "<<dq.stride(0)<<" "<<dq.stride(1)<<" "<<dq.stride(2)<<" "<<dq.stride(3)<<std::endl;
    // std::cout<<"q.stride() = "<<q.stride(0)<<" "<<q.stride(1)<<" "<<q.stride(2)<<" "<<q.stride(3)<<std::endl;
    // std::cout<<"dk.sizes() = "<<dk.sizes()<<std::endl;
    // std::cout<<"dv.sizes() = "<<dv.sizes()<<std::endl;
    // std::cout<<"num_heads_k = "<<num_heads_k<<std::endl;
    // std::cout<<"num_heads = "<<num_heads<<std::endl;
    // std::cout<<"dq.sizes() = "<<dq.sizes()<<std::endl;
    // std::cout<<"dk.sizes() = "<<dk.sizes()<<std::endl;
    // std::cout<<"dv.sizes() = "<<dv.sizes()<<std::endl;

    #ifdef DEBUGING
        return { dq, dk, dv, softmax_d, dev_kq.clone(), dev_s.clone(), dev_dp.clone(), dev_ds.clone()};
    #else
        return { dq, dk, dv, softmax_d };
    #endif
#else
    return {};
#endif
}

std::vector<at::Tensor>
mha_varlen_bwd_bhsd(const at::Tensor &dout,  // total_q_heads x head_size, total_q_heads := \sum_{i=0}^{b} s_i x num_heads
                    const at::Tensor &q,  // total_q_heads x head_size, total_q_heads := \sum_{i=0}^{b} s_i x num_heads
                    const at::Tensor &k,  // total_k_heads x head_size, total_k_heads := \sum_{i=0}^{b} s_i x num_heads_k
                    const at::Tensor &v,  // total_k_heads x head_size, total_k_heads := \sum_{i=0}^{b} s_i x num_heads_k
                    const at::Tensor &out, // total_q_heads x head_size, total_q_heads := \sum_{i=0}^{b} s_i x num_heads
                    const at::Tensor &softmax_lse,     // b x h x s   softmax logsumexp
                    c10::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                    c10::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                    c10::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                    const at::Tensor &cu_seqlens_q,  // b+1
                    const at::Tensor &cu_seqlens_k,  // b+1
                    c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
                    const int max_seqlen_q,
                    const int max_seqlen_k,          // max sequence length to choose the kernel
                    const float p_dropout,         // probability to drop
                    const float softmax_scale,
                    const bool zero_tensors,
                    const bool is_causal,
                    int window_size_left,
                    int window_size_right,
                    const float softcap,
                    const bool deterministic,
                    c10::optional<at::Generator> gen_,
                    c10::optional<at::Tensor> &rng_state
        #ifdef DEBUGING
                        ,
                        const at::Tensor &dev_kq,
                        const at::Tensor &dev_s,
                        const at::Tensor &dev_dp,
                        const at::Tensor &dev_ds
        #endif
                    ) {
#if defined(BUILD_FA_BWD)
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
    const int layout = 0;
    if (is_causal) { window_size_right = 0; }

    bool is_dropout = p_dropout > 0.0;
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16 || q_dtype == at::ScalarType::Float8_e4m3fn,
            "FlashAttention only support fp16,bf16,e4m3 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int total_q_heads = sizes[0];
    const int total_q = cu_seqlens_q[-1].item<int>();
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = total_q_heads / total_q;
    const int head_size_value = v.size(-1);
    const int head_size = sizes[1];
    const int total_k_heads = k.size(0);
    const int total_k = cu_seqlens_k[-1].item<int>();
    const int num_heads_k = total_k_heads / total_k;

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(head_size_value <= 256, "FlashAttention backward only supports head dimension at most 256");


    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(int64_t(total_q_heads * head_size) < /*2^31*/int64_t(2147483648), "The data amount of q must be smaller than the representation range of int");
    TORCH_CHECK(int64_t(total_k_heads * head_size) < /*2^31*/int64_t(2147483648), "The data amount of k/v must be smaller than the representation range of int");
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int head_size_value_rounded = round_multiple(head_size_value, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);
    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(dout, total_q_heads, dout.size(-1));
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    auto opts = q.options();
    at::Tensor q_padded, k_padded, v_padded, out_padded, dq_padded, dk_padded, dv_padded, dout_padded;
    if (head_size % 32 != 0) {
        q_padded = at::pad(q, {0, 32 - head_size % 32});
        k_padded = at::pad(k, {0, 32 - head_size % 32});
    } else {
        q_padded = q;
        k_padded = k;
    }

    if (head_size_value % 32 != 0) {
        v_padded = at::pad(v, {0, 32 - head_size_value % 32});
        out_padded = at::pad(out, {0, 32 - head_size_value % 32});
    } else {
        v_padded = v;
        out_padded = out;
    }

    if (dout.size(-1) % 32 != 0) {
        dout_padded = at::pad(dout, {0, 32 - dout.size(-1) % 32});
    } else {
        dout_padded = dout;
    }

    if(dq_.has_value()){
        CHECK_SHAPE(dq_.value(), total_q_heads, head_size);
        if (head_size % 32 != 0) {
            dq_padded = at::pad(dq_.value(), {0, 32 - head_size % 32});
        } else {
            dq_padded = dq_.value();
        }
    } else {
        dq_padded = at::empty({total_q_heads, head_size_rounded}, opts);
    }

    if(dk_.has_value()){
        CHECK_SHAPE(dk_.value(), total_k_heads, head_size);
        if (head_size % 32 != 0) {
            dk_padded = at::pad(dk_.value(), {0, 32 - head_size % 32});
        } else {
            dk_padded = dk_.value();
        }
    } else {
        dk_padded = at::empty({total_k_heads, head_size_rounded}, opts);
    }

    if(dv_.has_value()){
        CHECK_SHAPE(dv_.value(), total_k_heads, head_size_value);
        if (head_size_value % 32 != 0) {
            dv_padded = at::pad(dv_.value(), {0, 32 - head_size_value % 32});
        } else {
            dv_padded = dv_.value();
        }
    } else {
        dv_padded = at::empty({total_k_heads, head_size_value_rounded}, opts);
    }

    auto softmax_d = at::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));

    at::Tensor dk_accum, dv_accum;
    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = at::empty({total_k_heads * (num_heads / num_heads_k), head_size_rounded}, opts);
        dv_expanded = at::empty({total_k_heads * (num_heads / num_heads_k), head_size_value_rounded}, opts);
    } else {
        dk_expanded = dk_padded;
        dv_expanded = dv_padded;
    }

    #ifdef DEBUGING
        at::Tensor dev_kq, dev_s, dev_dp, dev_ds;
        if(layout == 0){
            dev_kq = at::empty({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(at::kFloat));
            dev_kq.fill_(float('-inf'));
            dev_s  = at::zeros({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(at::kFloat));
            dev_dp = at::zeros({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(at::kFloat));
            dev_ds = at::zeros({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(at::kFloat));
        } else {
            dev_kq = at::zeros({batch_size, seqlen_q, num_heads, seqlen_k}, opts.dtype(at::kFloat));
            dev_kq.fill_(float('-inf'));
            dev_s  = at::zeros({batch_size, seqlen_q, num_heads, seqlen_k}, opts.dtype(at::kFloat));
            dev_dp = at::zeros({batch_size, seqlen_q, num_heads, seqlen_k}, opts.dtype(at::kFloat));
            dev_ds = at::zeros({batch_size, seqlen_q, num_heads, seqlen_k}, opts.dtype(at::kFloat));
        }
    #endif

    // std::cout << "q_padded:\n" << q_padded << std::endl;
    // std::cout << "k_padded:\n" << k_padded << std::endl;
    // std::cout << "v_padded:\n" << v_padded << std::endl;
    // std::cout << "out_padded:\n" << out_padded << std::endl;
    // std::cout << "dout_padded:\n" << dout_padded << std::endl;

    Flash_bwd_params params;
    set_params_dgrad(params,
                    batch_size,
                    max_seqlen_q, max_seqlen_k,
                    seqlen_q_rounded, seqlen_k_rounded,
                    num_heads, num_heads_k,
                    head_size, head_size_rounded,
                    head_size_value, head_size_value_rounded,
                    q_padded, k_padded, v_padded, out_padded,
                    dout_padded, dq_padded, dk_expanded, dv_expanded,
                    cu_seqlens_q.data_ptr(),
                    cu_seqlens_k.data_ptr(),
                    nullptr/*p_d.data_ptr()*/,
    #ifdef DEBUGING
                    dev_kq.data_ptr(),
                    dev_s.data_ptr(),
                    dev_dp.data_ptr(),
                    dev_ds.data_ptr(),
    #endif
                    nullptr,
                    nullptr,
                    nullptr,
                    softmax_lse.data_ptr(),
                    softmax_d.data_ptr(),
                    p_dropout,
                    softmax_scale,
                    window_size_left,
                    window_size_right,
                    softcap,
                    deterministic,
                    /*unpadded_lse*/false,
                    layout
                    );
    params.total_q = total_q;
    params.total_k = total_k;
    auto launch = &run_mha_bwd;
    // launch(params, stream, /*configure=*/true);

    // auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
    //     gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;

    at::Tensor rng_state_tensor;
    if ( rng_state.has_value() ) {
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
    }
    else if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        rng_state_tensor = at::empty({2}, opts.dtype(at::ScalarType::Long));
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state_tensor.data_ptr());
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        std::lock_guard<std::mutex> lock(gen->mutex_);
        at::PhiloxCudaState philox_args = gen->philox_cuda_state(counter_offset);
        // at::cuda::philox::unpack(philox_args) not supported on ROCm
        params.rng_state[0] = philox_args.seed_.val;
        params.rng_state[1] = philox_args.offset_.val;
    }
    if (is_dropout) {
        params.rand_seed = params.rng_state[0];
        params.rand_offset = params.rng_state[1];
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    const hipStream_t stream = nullptr;//at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    launch(params, stream, /*configure=*/false);
    // For MQA/GQA we need to sum dK and dV across the groups
    // b * h * s, d
    if (num_heads_k != num_heads) {
        for(int i = 0; i< batch_size; ++i) {
            at::Tensor tmp_dk = at::reshape(at::reshape(dk_expanded.index({at::indexing::Slice(cu_seqlens_k[i].item<int>() * num_heads, cu_seqlens_k[i+1].item<int>() * num_heads)}), {num_heads_k, num_heads / num_heads_k, -1, head_size_rounded}).sum(1), {-1, head_size_rounded});
            dk_padded.index({at::indexing::Slice(cu_seqlens_k[i].item<int>() * num_heads_k, cu_seqlens_k[i+1].item<int>() * num_heads_k)}) = tmp_dk;
            at::Tensor tmp_dv = at::reshape(at::reshape(dv_expanded.index({at::indexing::Slice(cu_seqlens_k[i].item<int>() * num_heads, cu_seqlens_k[i+1].item<int>() * num_heads)}), {num_heads_k, num_heads / num_heads_k, -1, head_size_value_rounded}).sum(1), {-1, head_size_value_rounded});
            dv_padded.index({at::indexing::Slice(cu_seqlens_k[i].item<int>() * num_heads_k, cu_seqlens_k[i+1].item<int>() * num_heads_k)}) = tmp_dv;
        }
    }
    at::Tensor dq, dk, dv;
    if (head_size % 32 != 0) {
        dq = dq_padded.index({"...", at::indexing::Slice(at::indexing::None, head_size)});
        dk = dk_padded.index({"...", at::indexing::Slice(at::indexing::None, head_size)});
    } else {
        dq = dq_padded;
        dk = dk_padded;
    }
    if (head_size_value % 32 != 0) {
        dv = dv_padded.index({"...", at::indexing::Slice(at::indexing::None, head_size_value)});
    } else {
        dv = dv_padded;
    }

    // std::cout<<"q.sizes() = "<<q.sizes()<<std::endl;
    // std::cout<<"k.sizes() = "<<k.sizes()<<std::endl;
    // std::cout<<"out.sizes() = "<<out.sizes()<<std::endl;
    // std::cout<<"num_heads = "<<num_heads<<std::endl;
    // std::cout<<"dk.sizes() = "<<dk.sizes()<<std::endl;
    // std::cout<<"layout="<<layout<<std::endl;
    // std::cout<<"dq.sizes() = "<<dq.sizes()<<std::endl;
    // std::cout<<"dq.stride() = "<<dq.stride(0)<<" "<<dq.stride(1)<<" "<<dq.stride(2)<<" "<<dq.stride(3)<<std::endl;
    // std::cout<<"q.stride() = "<<q.stride(0)<<" "<<q.stride(1)<<" "<<q.stride(2)<<" "<<q.stride(3)<<std::endl;
    // std::cout<<"dk.sizes() = "<<dk.sizes()<<std::endl;
    // std::cout<<"dv.sizes() = "<<dv.sizes()<<std::endl;
    // std::cout<<"num_heads_k = "<<num_heads_k<<std::endl;
    // std::cout<<"num_heads = "<<num_heads<<std::endl;
    // std::cout<<"dq.sizes() = "<<dq.sizes()<<std::endl;
    // std::cout<<"dk.sizes() = "<<dk.sizes()<<std::endl;
    // std::cout<<"dv.sizes() = "<<dv.sizes()<<std::endl;

    #ifdef DEBUGING
        return { dq, dk, dv, softmax_d, dev_kq.clone(), dev_s.clone(), dev_dp.clone(), dev_ds.clone()};
    #else
        return { dq, dk, dv, softmax_d };
    #endif
#else
    return {};
#endif
}


std::vector<at::Tensor> mha_fwd_kvcache_base(
    at::Tensor &q,
    const at::Tensor &kcache,
    const at::Tensor &vcache,
    c10::optional<const at::Tensor> &k_,
    c10::optional<const at::Tensor> &v_,
    c10::optional<const at::Tensor> &seqlens_q_,
    c10::optional<const at::Tensor> &seqlens_k_,
    int max_seqlen_k,
    c10::optional<const at::Tensor> &rotary_cos_,
    c10::optional<const at::Tensor> &rotary_sin_,
    c10::optional<const at::Tensor> &cache_batch_idx_,
    c10::optional<const at::Tensor> &leftpad_k_,
    c10::optional<at::Tensor> &block_table_,
    c10::optional<at::Tensor> &alibi_slopes_,
    c10::optional<at::Tensor> &out_,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    bool is_rotary_interleaved,
    int partition_size,
    c10::optional<at::Tensor> &scores_raw,
    c10::optional<at::Tensor> &tmp_output,
    const int layout,
    c10::optional<at::Tensor> scales_q_,
    c10::optional<at::Tensor> scales_k_,
    c10::optional<at::Tensor> scales_v_,
    const bool is_bf16_output
) {
#if defined(BUILD_FA_KVCACHE)
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
    const bool int8_used = scales_k_.has_value();
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16 || q_dtype == at::ScalarType::Char,
                "FlashAttention only support fp16 and bf16 and int8 data type");
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    TORCH_CHECK(paged_KV, "Only PagedAttention KVcache is suppprted yet!");
    if (paged_KV) {
        TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == at::ScalarType::Int, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    const auto sizes       = q.sizes();
    const int batch_size   = sizes[0];
    int       num_heads    = (layout == 1) ? sizes[2]: sizes[1];
    int       seqlen_q     = (layout == 1) ? sizes[1]: sizes[2];
    const int head_size_og = sizes[3];
    const int qk_head_size = q.size(3);
    const int v_head_size  = vcache.size(3);
    const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks   = kcache.size(0);
    const int page_block_size = (layout == 1) ? kcache.size(1): kcache.size(2);
    const int num_heads_k  = (layout == 1) ? kcache.size(2): kcache.size(1);
    const int batch_size_c = batch_size;
    // multi token prediction
    const int mtp          = (layout == 1) ? sizes[1]: sizes[2];

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    const bool b16_hdim512 = !int8_used && qk_head_size == 512 && v_head_size == 512;
    TORCH_CHECK(qk_head_size <= 256 or qk_head_size == 576 or b16_hdim512,
                "PagedAttention only supports head dimension at most 256, MLA-QK-576, or b16 QK-512");
    TORCH_CHECK(v_head_size <= 256 or v_head_size == 512, "PagedAttention only supports head dimension at most 256 or MLA-V-512");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // acquire varlen information of Q
    void *cu_seqlens_q = seqlens_q_.has_value() ? seqlens_q_.value().data_ptr(): nullptr;

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int ngroups = num_heads / num_heads_k;
    const int seqlenq_ngroups_swapped = (!int8_used or layout == 0) && (v_head_size == 128 or v_head_size == 64) && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
    if (seqlenq_ngroups_swapped) {
        // when batch size is small, cu occupancy is likely low, and thus reuse less KV to dispatch more threadgroups
        if (batch_size <= 2) {
            PA_GQA_REGROUP_SWITCH(ngroups, [&] {
                if (layout == 0) {
                    q = q.view({batch_size, num_heads_k * int(ngroups / GQA_REGROUP), GQA_REGROUP * mtp, qk_head_size});
                } else {
                    q = q.view({batch_size, mtp, -1, GQA_REGROUP, qk_head_size}).transpose(2, 3).contiguous().view({batch_size, mtp * GQA_REGROUP, -1, qk_head_size});
                }
                seqlen_q  = GQA_REGROUP * mtp;
                num_heads = num_heads_k * int(ngroups / GQA_REGROUP);
            });
        } else {
            // default reuse strategy
            if (layout == 0) {
                q = q.view({batch_size, num_heads_k * int(ngroups / ngroups), ngroups * mtp, qk_head_size});
            } else {
                q = q.view({batch_size, mtp, -1, ngroups, qk_head_size}).transpose(2, 3).contiguous().view({batch_size, mtp * ngroups, -1, qk_head_size});
            }
            seqlen_q  = ngroups * mtp;
            num_heads = num_heads_k;
        }
    }

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    TORCH_CHECK(int64_t(batch_size * num_heads * seqlen_q * qk_head_size) < /*2^31*/int64_t(2147483648), "The data amount of q must be smaller than the representation range of int");
    // TORCH_CHECK(int64_t(total_k_heads * head_size_og) < /*2^31*/int64_t(2147483648), "The data amount of k/v must be smaller than the representation range of int");
    if (!paged_KV) {
        CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, qk_head_size);
        CHECK_SHAPE(kcache, batch_size_c, seqlen_q, num_heads_k, qk_head_size);
        CHECK_SHAPE(vcache, batch_size_c, seqlen_q, num_heads_k, v_head_size);
    } else {
        // CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
        // CHECK_SHAPE(q, total_q_heads, qk_head_size);
        // CHECK_SHAPE(kcache, total_k_heads, head_size_og);
        // CHECK_SHAPE(vcache, total_k_heads, head_size_og);
    }

    at::Tensor q_padded, kcache_padded, vcache_padded, accum_output_padded;
    constexpr int HEADDIM_GRANULARITY = 32; // headdim 模板参数化的最小粒度是 32
    const bool QK_IS_NOT_COMMON_HEADDIM  = (qk_head_size % HEADDIM_GRANULARITY != 0);
    if (QK_IS_NOT_COMMON_HEADDIM) {
        q_padded = at::pad(q, {0, HEADDIM_GRANULARITY - qk_head_size % HEADDIM_GRANULARITY});
        kcache_padded = at::pad(kcache, {0, HEADDIM_GRANULARITY - qk_head_size % HEADDIM_GRANULARITY});
    } else {
        q_padded = q;
        kcache_padded = kcache;
    }

    const bool V_IS_NOT_COMMON_HEADDIM  = (v_head_size % HEADDIM_GRANULARITY != 0);

    if (V_IS_NOT_COMMON_HEADDIM) {
        vcache_padded = at::pad(vcache, {0, HEADDIM_GRANULARITY - v_head_size % HEADDIM_GRANULARITY});
        if (tmp_output.has_value()) accum_output_padded = at::pad(tmp_output.value(), {0, HEADDIM_GRANULARITY - v_head_size % HEADDIM_GRANULARITY});
    } else {
        vcache_padded = vcache;
        if (tmp_output.has_value()) accum_output_padded = tmp_output.value();
    }

    auto opts = q.options();
    at::Tensor out;
    bool output_allocated_outside = out_.has_value();
    if (output_allocated_outside) {
        out = out_.value();
        if (!int8_used){
            TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        }
        CHECK_DEVICE(out);
        // TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        // CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, v_head_size);
        if (V_IS_NOT_COMMON_HEADDIM) { out = at::empty_like(q_padded); }
        // out = out.view_as(q);
        out = out.view({q.size(0), q.size(1), q.size(2), -1});
    } else {
        if (!int8_used) {
            out = at::empty({{q.size(0), q.size(1), q.size(2), vcache_padded.size(-1)}}, opts);
        } else {
            auto int8_opts = is_bf16_output ? opts.dtype(at::ScalarType::BFloat16) : opts.dtype(at::ScalarType::Half);
            out = at::empty({{q.size(0), q.size(1), q.size(2), vcache_padded.size(-1)}}, int8_opts);
        }
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int qk_head_size_rounded = round_multiple(round_multiple(qk_head_size, 8), HEADDIM_GRANULARITY);
    const int v_head_size_rounded = round_multiple(round_multiple(v_head_size, 8), HEADDIM_GRANULARITY);

    const int seqlen_q_rounded = round_multiple(seqlen_q, 32);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 32);

    // auto softmax_lse = at::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    bool seqlens_k_has_value = seqlens_k_.has_value();
    if (seqlens_k_has_value) {
        auto seqlens_k = seqlens_k_.value();
        TORCH_CHECK(seqlens_k.dtype() == at::ScalarType::Int, "seqlens_k must have dtype int32");
        CHECK_DEVICE(seqlens_k);
        CHECK_CONTIGUOUS(seqlens_k);
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     qk_head_size, qk_head_size_rounded,
                     v_head_size, v_head_size_rounded,
                     q_padded, kcache_padded, vcache_padded, out,
                     /*cu_seqlens_q_d=*/cu_seqlens_q,
                     /*cu_seqlens_k_d=*/seqlens_k_has_value ? seqlens_k_.value().data_ptr(): nullptr,
                     /*seqused_k=*/nullptr,
                     /*p_ptr=*/nullptr,
                     /*softmax_lse.data_ptr()*/nullptr,
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     seqlenq_ngroups_swapped,
                     /*unpadded_lse*/true,
                     /*is_kvcache*/true,
                     /*is_seqlens_k_cumulative*/seqlens_k_has_value ? (seqlens_k_.value().size(0) == (batch_size + 1)): false,
                     layout
                    );

    if (int8_used){
        params.is_bf16 = is_bf16_output;
        at::Tensor scales_q;
        scales_q = scales_q_.value();
        params.scales_q_ptr = scales_q.data_ptr();
        params.total_scale_q = scales_q.numel();
        at::Tensor scales_k;
        scales_k = scales_k_.value();
        params.scales_k_ptr = scales_k.data_ptr();
        at::Tensor scales_v;
        scales_v = scales_v_.value();
        params.scales_v_ptr = scales_v.data_ptr();
    }
    if (k_.has_value()) {
        at::Tensor k, v, k_padded, v_padded;
        TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
        TORCH_CHECK(seqlens_k_.has_value(), "If key is supplied, seqlens_k must also be passed in");
        TORCH_CHECK(seqlen_q <= max_seqlen_k, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");
        k = k_.value();
        v = v_.value();
        if (!int8_used){
            TORCH_CHECK(k.dtype() == q_dtype, "Key must have the same dtype as query");
            TORCH_CHECK(v.dtype() == q_dtype, "Value must have the same dtype as query");
        }
        CHECK_DEVICE(k); CHECK_DEVICE(v);
        // TORCH_CHECK(k.stride(-1) == 1, "Key tensor must have contiguous last dimension");
        // TORCH_CHECK(v.stride(-1) == 1, "Value tensor must have contiguous last dimension");
        int seqlen_knew = k.size(1);
        // CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, qk_head_size);
        // CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, v_head_size);
        if (QK_IS_NOT_COMMON_HEADDIM) {
            k_padded = at::pad(k, {0, HEADDIM_GRANULARITY - qk_head_size % HEADDIM_GRANULARITY});
        } else {
            k_padded = k;
        }

        if (V_IS_NOT_COMMON_HEADDIM) {
            v_padded = at::pad(v, {0, HEADDIM_GRANULARITY - v_head_size % HEADDIM_GRANULARITY});
        } else {
            v_padded = v;
        }
        params.seqlen_knew = seqlen_knew;
        params.knew_ptr = k_padded.data_ptr();
        params.vnew_ptr = v_padded.data_ptr();
        // All stride are in elements, not bytes.
        params.knew_batch_stride = k_padded.stride(0);
        params.vnew_batch_stride = v_padded.stride(0);
        params.knew_row_stride = k_padded.stride(-3);
        params.vnew_row_stride = v_padded.stride(-3);
        params.knew_head_stride = k_padded.stride(-2);
        params.vnew_head_stride = v_padded.stride(-2);
    }

    // params.is_seqlens_k_cumulative = !(seqlens_k_.has_value());
    if (leftpad_k_.has_value()) {
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(leftpad_k.dtype() == at::ScalarType::Int, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    }

    if (rotary_cos_.has_value()) {
        TORCH_CHECK(k_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
        auto rotary_cos = rotary_cos_.value();
        CHECK_DEVICE(rotary_cos);
        params.rotary_dim = rotary_cos.size(1) * 2;
        TORCH_CHECK(params.rotary_dim <= qk_head_size, "rotary_dim must be <= headdim");
        TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
        const int seqlen_ro = rotary_cos.size(0);
        TORCH_CHECK(seqlen_ro >= max_seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
        CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
        CHECK_CONTIGUOUS(rotary_cos);
        TORCH_CHECK(rotary_cos.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");

        TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
        auto rotary_sin = rotary_sin_.value();
        CHECK_DEVICE(rotary_sin);
        CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
        CHECK_CONTIGUOUS(rotary_sin);
        TORCH_CHECK(rotary_sin.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");
        params.rotary_cos_ptr = rotary_cos.data_ptr();
        params.rotary_sin_ptr = rotary_sin.data_ptr();
        params.is_rotary_interleaved = is_rotary_interleaved;
    } else {
        params.rotary_dim = 0;
    }

    if (cache_batch_idx_.has_value()) {
        auto cache_batch_idx = cache_batch_idx_.value();
        CHECK_DEVICE(cache_batch_idx);
        CHECK_CONTIGUOUS(cache_batch_idx);
        TORCH_CHECK(cache_batch_idx.scalar_type() == at::ScalarType::Int, "cache_batch_idx must have dtype int32");
        params.cache_batch_idx = reinterpret_cast<int *>(cache_batch_idx.data_ptr());
    }

    // Acquire cu count
    hipDeviceProp_t props;
    auto hipResult = hipGetDeviceProperties(&props, 0);
    params.cu_count = props.multiProcessorCount;

    // check if splitkv is forbidden
    bool allow_splitkv = bool(std::getenv("PA_NO_SPLITKV") == nullptr) and (v_head_size_rounded == 128 or v_head_size_rounded == 512 or v_head_size_rounded == 64);

    // Keep references to these tensors to extend their lifetime
    at::Tensor scores_sum, scores_max, out_accum;
    if (allow_splitkv and partition_size > 0) {
        // compare with official methods, we don't consider the relationship between partition_size and cu_count
        // since we don't support arbitrary partition size yet
        bool partition_size_assigned = scores_raw.has_value() and tmp_output.has_value();
        at::Tensor raw_memory;
        if (partition_size_assigned) {
            params.partition_size = partition_size;
            params.num_splits = std::max<int32_t>(1, std::floor(max_seqlen_k * 1.f / params.partition_size));
            TORCH_CHECK(params.num_splits <= 1024, "num_splits > 128 not supported");
            TORCH_CHECK(params.partition_size >= 128, "partition_size >= 128 is required");
            TORCH_CHECK(params.partition_size % page_block_size == 0, "partition_size must be multiple of page_block_size");
            raw_memory = scores_raw.value().view({2, params.num_splits, batch_size, num_heads, seqlen_q});
        } else {
            // 指定的不是 partition_size 而是 num_splits, 这样 batch_size, num_splits, num_heads 都是固定的, 可以跑 cudagraph
            params.num_splits = partition_size;
            params.partition_size = std::max<int32_t>(128, std::ceil(max_seqlen_k * 1.f / (params.num_splits * page_block_size)) * page_block_size);
            raw_memory = at::empty({2, params.num_splits, batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
            if (layout == 0)      accum_output_padded = at::empty({params.num_splits, batch_size, num_heads, seqlen_q, v_head_size_rounded}, opts.dtype(q_dtype));
            else if (layout == 1) accum_output_padded = at::empty({params.num_splits, batch_size, seqlen_q, num_heads, v_head_size_rounded}, opts.dtype(q_dtype));
        }
        scores_sum = raw_memory.index({0});
        scores_max = raw_memory.index({1});
        out_accum  = /*original is tmp_output*/accum_output_padded.view({params.num_splits, batch_size, num_heads, seqlen_q, v_head_size_rounded}); // 看官方的写法, O_accum 用的更高精度去写的, 而不是半精度
        params.scores_sum_ptr = reinterpret_cast<float*>(scores_sum.data_ptr());
        params.scores_max_ptr = reinterpret_cast<float*>(scores_max.data_ptr());
        params.oaccum_ptr     = out_accum.data_ptr();
    }
    // 如果没有指定 partition size, 且 headdim 128, 自主决定切分策略
    if (allow_splitkv and !tmp_output.has_value() and partition_size == 0) {
        const char* partition_size_env = std::getenv("PA_PARTITION_SIZE");
        const int partition_size_assign = partition_size_env ? std::atoi(partition_size_env): 0;
        // 没有指定 splitkv 分块大小, 则启发式
        if (partition_size_assign == 0) {
            // 如果初步能划分的 block 数量对应的利用率不高
            constexpr int device_cu = 128;
            const int threshold     = device_cu;
            // 如果 gqa 组数不是常见的 16/8/4/2/9/7/5/3 的倍数, ngroup 会被全部 re-group 到 seqlen 维度上, 会导致发的 TG 比较少, 因此算最优 partition size 的时候还是要认为 ngroup = 1
            // 原始是 GQA, 但做了最大程度的 regroup
            const bool use_max_regroup  = (ngroups > 1 and ngroups != 29 and ngroups != 16 and ngroups != 8 and ngroups != 4 and ngroups != 2 and ngroups != 9 and ngroups != 7 and ngroups != 5 and ngroups != 3);
            int actual_ngroup = use_max_regroup ? 1: ngroups;
            // 如果目前能发的 TG 数量比较少而且最大的 seqkv 不是很短
            // 或者 seqkv 比较长, 可以做切分
            if ((batch_size * 1/*seq_q_len*/ * actual_ngroup < threshold and max_seqlen_k >= 1024) or (max_seqlen_k >= 8192)) {
                // 根据一个 batch 里最大的 seqKV 长度, 决定相应的划分 size
                if (max_seqlen_k <= 1024)       partition_size = 128;
                else if (max_seqlen_k <= 2048)  partition_size = 256;
                else if (max_seqlen_k <= 32768) partition_size = 512;
                else                            partition_size = 1024;
                // 如果是 MHA, 无法做 GQA ngroup-swapped 优化, 可以发更多的 TG, 不需要划分那么多小块, 可以划分大一点的块
                if (ngroups == 1) partition_size = 1024;
                // 如果按照上述划分之后, 利用率还不是很高, partition size 继续减半
                while (ngroups > 1 and (batch_size * 1/*seq_q_len*/ * actual_ngroup * (max_seqlen_k / partition_size)) < threshold) {
                    // 目前支持的最小 partition size 是 128
                    if (partition_size < 256) break;
                    partition_size = int(partition_size / 2);
                }
            }
        } else if (partition_size_assign >= 128 and partition_size_assign <= 1024) {
            // 指定的 partition_size 满足需求, 可以开始划分
            partition_size = partition_size_assign;
        }
        // 如果划分满足最小粒度 128 的倍数, 且不超过 1024 个划分, 则允许 splitkv 算法
        // 128 的倍数, 对应 kernel: int this_split_seqlen_start = Split ? split_id * params.partition_size: 0; 暂不支持任意长度的 splitkv
        if (partition_size >= 128 and partition_size % page_block_size == 0) {
            // 截断最后一个切分到前一个 block 上去计算
            const int num_splits = std::max<int32_t>(1, std::floor(max_seqlen_k * 1.f / partition_size));
            // 最大支持 1024 个划分
            if (num_splits <= 1024) {
                // 传递给 kernel args
                params.partition_size = partition_size;
                params.num_splits     = num_splits;
                auto raw_memory = at::empty({2, params.num_splits, batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
                scores_sum = raw_memory.index({0});
                scores_max = raw_memory.index({1});
                if (layout == 0)      out_accum  = at::empty({params.num_splits, batch_size, num_heads, seqlen_q, v_head_size_rounded}, opts.dtype(q_dtype));
                else if (layout == 1) out_accum  = at::empty({params.num_splits, batch_size, seqlen_q, num_heads, v_head_size_rounded}, opts.dtype(q_dtype));
                params.scores_sum_ptr = reinterpret_cast<float*>(scores_sum.data_ptr());
                params.scores_max_ptr = reinterpret_cast<float*>(scores_max.data_ptr());
                params.oaccum_ptr     = out_accum.data_ptr();
            }
        }
    }

    // decide accumulation dtype when splitkv
    if (params.partition_size > 0 and params.num_splits > 1) {
        params.splitkv_use_fp32_as_accum = out_accum.dtype() == at::ScalarType::Float;
    }

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
        params.k_batch_stride = kcache_padded.stride(0);
        params.v_batch_stride = vcache_padded.stride(0);
    }
    params.page_block_size = page_block_size;
    params.mtp = mtp;

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    // print main args
    bool fa_debug = (std::getenv("FA_DEBUG") != nullptr);
    if (fa_debug) {
        PRINT_PARAMS
        auto temp_tensor = seqlens_k_.value().to(at::DeviceType::CPU).contiguous();
        std::vector<int32_t> temp_vector(temp_tensor.data_ptr<int32_t>(), temp_tensor.data_ptr<int32_t>() + temp_tensor.numel());
        printf("seqlens_k: ["); for (const auto val: temp_vector) { printf("%d ", val); } printf("]\n");
        PRINT_QKV_INFO(q, kcache, vcache)
        std::cout << "block_table: " << block_table.sizes() << "\n";
    }

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // Only split kernel supports appending to KV cache, or indexing to the cache with cache_batch_idx,
    // or paged KV cache
    // run_mha_fwd(params, stream, /*force_split_kernel=*/k_.has_value() || cache_batch_idx_.has_value() || paged_KV);
    if (max_seqlen_k > 0 and std::getenv("PA_EMPTY") == nullptr) {
        if (!int8_used){
            run_mha_fwd_kvcache(params, stream, paged_KV);
        } else{
            run_int8_fwd_kvcache(params, stream, paged_KV);
        }
    } else {
        out.zero_();
        // softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if (seqlenq_ngroups_swapped) {
        if (layout == 0) {
            out = out.view({batch_size, -1, mtp, v_head_size_rounded});
        } else if (layout == 1) {
            out = out.view({batch_size, mtp, -1, num_heads, v_head_size_rounded}).transpose(2, 3).contiguous().view({batch_size, mtp, -1, v_head_size_rounded});
            if (output_allocated_outside and out_.has_value()) { out_.value().copy_(out.clone()); } // strange, without this line, result is wrong
        }
    }

    if (QK_IS_NOT_COMMON_HEADDIM) {
        if (k_.has_value()) {
            // It's expensive to copy the KV cache here for the case where head size not divisible by 8,
            // but we don't expect to get this case in practice. This is just so that the code works for that case.
            kcache.copy_(kcache_padded.index({"...", at::indexing::Slice(at::indexing::None, qk_head_size)}));
        }
    }

    if (V_IS_NOT_COMMON_HEADDIM) {
        out = out.index({"...", at::indexing::Slice(at::indexing::None, v_head_size)});
        if (out_.has_value()) { out_.value().copy_(out); }
        if (v_.has_value()) {
            // It's expensive to copy the KV cache here for the case where head size not divisible by 8,
            // but we don't expect to get this case in practice. This is just so that the code works for that case.
            vcache.copy_(vcache_padded.index({"...", at::indexing::Slice(at::indexing::None, v_head_size)}));
        }
    }

    if (output_allocated_outside) {
        return {out};
    } else {
        return {out, out_accum, scores_max, scores_sum, at::tensor(params.partition_size, at::dtype(at::ScalarType::Int))};
    }
#else
    return {};
#endif
}


std::vector<at::Tensor> mha_fwd_kvcache_bhsd(
        at::Tensor &q,
        const at::Tensor &kcache,
        const at::Tensor &vcache,
        c10::optional<const at::Tensor> &k_,
        c10::optional<const at::Tensor> &v_,
        c10::optional<const at::Tensor> &seqlens_q_,
        c10::optional<const at::Tensor> &seqlens_k_,
        int max_seqlen_k,
        c10::optional<const at::Tensor> &rotary_cos_,
        c10::optional<const at::Tensor> &rotary_sin_,
        c10::optional<const at::Tensor> &cache_batch_idx_,
        c10::optional<const at::Tensor> &leftpad_k_,
        c10::optional<at::Tensor> &block_table_,
        c10::optional<at::Tensor> &alibi_slopes_,
        c10::optional<at::Tensor> &out_,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        bool is_rotary_interleaved,
        int partition_size,
        c10::optional<at::Tensor> &scores_raw,
        c10::optional<at::Tensor> &tmp_output,
        c10::optional<at::Tensor> scales_q_,
        c10::optional<at::Tensor> scales_k_,
        c10::optional<at::Tensor> scales_v_,
        const bool is_bf16_output
    ) {
    return mha_fwd_kvcache_base(q, kcache, vcache,
        k_, v_, seqlens_q_, seqlens_k_, max_seqlen_k, rotary_cos_, rotary_sin_, cache_batch_idx_, leftpad_k_, block_table_, alibi_slopes_, out_, softmax_scale, is_causal, window_size_left, window_size_right, softcap, is_rotary_interleaved, partition_size, scores_raw, tmp_output,
        0/*bhsd*/, scales_q_, scales_k_, scales_v_, is_bf16_output
    );
}


std::vector<at::Tensor> hg_fwd_kvcache_bshd(
        at::Tensor &q,
        const at::Tensor &kcache,
        const at::Tensor &vcache,
        c10::optional<const at::Tensor> &k_,
        c10::optional<const at::Tensor> &v_,
        c10::optional<const at::Tensor> &seqlens_q_,
        c10::optional<const at::Tensor> &seqlens_k_,
        int max_seqlen_k,
        c10::optional<const at::Tensor> &rotary_cos_,
        c10::optional<const at::Tensor> &rotary_sin_,
        c10::optional<const at::Tensor> &cache_batch_idx_,
        c10::optional<const at::Tensor> &leftpad_k_,
        c10::optional<at::Tensor> &block_table_,
        c10::optional<at::Tensor> &alibi_slopes_,
        c10::optional<at::Tensor> &out_,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        bool is_rotary_interleaved,
        int partition_size,
        c10::optional<at::Tensor> &scores_raw,
        c10::optional<at::Tensor> &tmp_output,
        c10::optional<at::Tensor> scales_q_,
        c10::optional<at::Tensor> scales_k_,
        c10::optional<at::Tensor> scales_v_,
        const bool is_bf16_output
    ) {
    return mha_fwd_kvcache_base(q, kcache, vcache,
        k_, v_, seqlens_q_, seqlens_k_, max_seqlen_k, rotary_cos_, rotary_sin_, cache_batch_idx_, leftpad_k_, block_table_, alibi_slopes_, out_, softmax_scale, is_causal, window_size_left, window_size_right, softcap, is_rotary_interleaved, partition_size, scores_raw, tmp_output,
        1/*bshd*/, scales_q_, scales_k_, scales_v_, is_bf16_output
    );
}




std::vector<at::Tensor> hg_prefix_decode_varlen_fwd(
    at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    c10::optional<at::Tensor> &out_, const at::Tensor &cu_seqlens_q,
    c10::optional<at::Tensor> &cu_seqlens_k, at::Tensor &seqused_k,
    c10::optional<at::Tensor> &alibi_slopes_, at::Tensor &block_table,
    const int max_seqlen_q, const int max_seqlen_k, const float p_dropout,
    const float softmax_scale, const bool zero_tensors, const bool is_causal,
    int window_size_left, int window_size_right, const float softcap,
    const bool return_softmax, const int layout,
    c10::optional<at::Tensor> scales_q_ = c10::nullopt,
    c10::optional<at::Tensor> scales_k_ = c10::nullopt,
    c10::optional<at::Tensor> scales_v_ = c10::nullopt,
    c10::optional<at::Tensor> s_aux_ = c10::nullopt,
    const bool is_bf16_output = false ) {
#if defined(BUILD_FA_KVCACHE)
  const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
  // TORCH_CHECK(is_causal == true, "For prefix decode, only causal mask = True
  // is supported!");
  if (is_causal) {
    window_size_right = 0;
  }

  auto q_dtype = q.dtype();
  const bool fp8_used = q_dtype == at::ScalarType::Float8_e4m3fn;
  TORCH_CHECK(q_dtype == at::ScalarType::Half ||
              q_dtype == at::ScalarType::BFloat16 ||
              q_dtype == at::ScalarType::Float8_e4m3fn,
              "For prefix decode, only support fp16/bf16/fp8_e4m3 data type");
  TORCH_CHECK(k.dtype() == q_dtype,
              "For prefix decode, query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype,
              "For prefix decode, query and value must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == at::ScalarType::Int,
              "For prefix decode, cu_seqlens_q must have dtype int32");
  TORCH_CHECK(seqused_k.dtype() == at::ScalarType::Int,
              "For prefix decode, seqused_k must have dtype int32");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(cu_seqlens_q);
  CHECK_DEVICE(seqused_k);

  TORCH_CHECK(
      q.stride(-1) == 1,
      "For prefix decode, Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      k.stride(-1) == 1,
      "For prefix decode, Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      v.stride(-1) == 1,
      "For prefix decode, Input tensor must have contiguous last dimension");
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(seqused_k);

  const bool use_bshd_layout = layout == 1;
  const auto query_size = q.sizes();
  const auto k_size = k.sizes();
  const auto v_size = v.sizes();
  int num_heads = query_size[1];
  const int original_num_heads = num_heads;
  const int num_heads_k = k_size[2];
  const int head_size_og = use_bshd_layout ? query_size[2] : query_size[1];
  const int head_size_value = use_bshd_layout ? v_size[3] : v_size[2];
  const int total_q =
      use_bshd_layout ? query_size[0] : query_size[0] / num_heads;
  const int batch_size = cu_seqlens_q.numel() - 1;
  const int page_block_size = use_bshd_layout ? k_size[1] : k_size[2];
  TORCH_CHECK(batch_size > 0, "For prefix decode, batch size must be positive");
  TORCH_CHECK(page_block_size == 128 || page_block_size == 64,
              "For prefix decode, only supports page block_size 128 or 64");
  TORCH_CHECK((head_size_og == 128 and head_size_value == 128) or
              (head_size_og == 192 and head_size_value == 128) or
              (head_size_og == 192 and head_size_value == 192) or
              (head_size_og == 256 and head_size_value == 256) or
              (!fp8_used && head_size_og == 512 && head_size_value == 512),
              "For prefix decode, only supports head dimension "
              "128+128/192+128/192+192/256+256, plus b16 512+512");
  if (fp8_used) {
    TORCH_CHECK((head_size_og == 128 and head_size_value == 128) or
                (head_size_og == 192 and head_size_value == 128) or
                (head_size_og == 256 and head_size_value == 256),
                "For fp8 prefix decode, only supports head dimension "
                "128+128/192+128/256+256 on gfx938 MLS kernel");
    TORCH_CHECK(scales_q_.has_value() && scales_k_.has_value() && scales_v_.has_value(),
                "For fp8 prefix decode, q/k/v descale tensors must be provided");
  }
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");
  TORCH_CHECK(int64_t(query_size[0] * head_size_og) <
                  /*2^31*/ int64_t(2147483648),
              "The data amount of q must be smaller than the representation "
              "range of int");
  TORCH_CHECK(int64_t(k_size[0] * head_size_value) <
                  /*2^31*/ int64_t(2147483648),
              "The data amount of k/v must be smaller than the representation "
              "range of int");
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(seqused_k, batch_size);

  if (softcap > 0.f) {
    TORCH_CHECK(
        p_dropout == 0.f,
        "For prefix decode, Softcapping does not support dropout for now");
  }

  int ngroups = num_heads / num_heads_k;
  const int ngroups_limit = std::getenv("PA_USE_TILE32X32") == nullptr
                                ? 32
                                : 16 /*32 is not supported for 32x32tile yet*/;
  while (ngroups > 1) {
    if (ngroups * max_seqlen_q <= ngroups_limit and
        (num_heads % ngroups == 0 and num_heads / ngroups % num_heads_k == 0))
      break;
    --ngroups;
  }
  if (ngroups > 1) {
    num_heads = num_heads / ngroups;
    q = q.view({total_q, num_heads, ngroups, -1})
            .transpose(1, 2)
            .contiguous()
            .view({total_q * ngroups, num_heads, -1});
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);
  const int head_size_v = round_multiple(head_size_value, 8);
  const int head_size_v_rounded = round_multiple(head_size_v, 32);
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, 32);
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, 32);

  at::Tensor q_padded, k_padded, v_padded;
  if (head_size_og % 32 != 0) {
    q_padded = at::pad(q, {0, 32 - head_size_og % 32});
    k_padded = at::pad(k, {0, 32 - head_size_og % 32});
  } else {
    q_padded = q;
    k_padded = k;
  }

  if (head_size_value % 32 != 0) {
    v_padded = at::pad(v, {0, 32 - head_size_value % 32});
  } else {
    v_padded = v;
  }

  auto opts = q.options();
  at::Tensor out;
  bool output_allocated_outside = out_.has_value();
  if (output_allocated_outside) {
    out = out_.value();
    if (!fp8_used){
      TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
    } else {
      TORCH_CHECK(out.dtype() == at::ScalarType::Half ||
                  out.dtype() == at::ScalarType::BFloat16,
                  "For fp8 prefix decode, output must be fp16 or bf16");
    }
    if (out.is_contiguous()) {
      out = out.view({q.size(0), q.size(1), -1});
      CHECK_DEVICE(out);
      TORCH_CHECK(out.stride(-1) == 1, "For prefix decode, output tensor must "
                                       "have contiguous last dimension");
    } else {
      out = at::empty({q.size(0), q.size(1), v_padded.size(-1)}, opts);
    }
  } else {
    // for (bs)hd layout
    if (fp8_used) {
      auto fp8_opts = is_bf16_output ? opts.dtype(at::ScalarType::BFloat16) : opts.dtype(at::ScalarType::Half);
      out = at::empty({q.size(0), q.size(1), head_size_v_rounded}, fp8_opts);
    } else {
      out = at::empty({q.size(0), q.size(1), v_padded.size(-1)}, opts);
    }
  }

  auto softmax_lse =
      at::empty({num_heads * ngroups, total_q}, opts.dtype(at::kFloat));
  if (zero_tensors) {
    out.zero_();
    softmax_lse.fill_(-std::numeric_limits<float>::infinity());
  }

  Flash_fwd_params params;
  set_params_fprop(
      params, batch_size, max_seqlen_q, max_seqlen_k, seqlen_q_rounded,
      seqlen_k_rounded, num_heads, num_heads_k, head_size, head_size_rounded,
      head_size_v, head_size_v_rounded, q_padded, k_padded, v_padded, out,
      cu_seqlens_q.data_ptr(), seqused_k.data_ptr(),
      return_softmax ? nullptr /*p.data_ptr()*/ : nullptr, seqused_k.data_ptr(),
      softmax_lse.data_ptr(), p_dropout, softmax_scale, window_size_left,
      window_size_right, softcap, false,
      /*unpadded_lse*/ false,
      /*is_kvcache*/ false,
      /*is_seqlens_k_cumulative*/ seqused_k.size(0) == (batch_size + 1),
      layout /*layout*/, false /*is_flashmla*/, true /*is_prefix*/
  );
  params.s_aux_ptr = nullptr;
  params.s_aux_type = 0;
  if (s_aux_.has_value()) {
    auto s_aux = s_aux_.value();
    const auto expected_sink_dtype = out.scalar_type();
    TORCH_CHECK(s_aux.scalar_type() == expected_sink_dtype,
                "Attention sink dtype must match prefix output dtype. Got ",
                s_aux.dtype(), ", expected ", expected_sink_dtype);
    CHECK_DEVICE(s_aux);
    CHECK_CONTIGUOUS(s_aux);
    CHECK_SHAPE(s_aux, original_num_heads);
    params.s_aux_ptr = s_aux.data_ptr();
    params.s_aux_type = get_attention_sink_type(s_aux.scalar_type());
  }
  params.total_q = total_q;
  params.block_table = block_table.data_ptr<int>();
  params.block_table_batch_stride = block_table.stride(0);
  params.k_batch_stride = k_padded.stride(0);
  params.v_batch_stride = v_padded.stride(0);
  params.page_block_size = page_block_size;
  params.seqused_k = reinterpret_cast<int *>(seqused_k.data_ptr());
  params.layout = 1; // only bshd (layout = 1) is supported yet
  params.mtp = max_seqlen_q;
  params.seqlen_q *= ngroups;
  params.ngroups = ngroups;
  params.seqlenq_ngroups_swapped = ngroups > 1;

  if (fp8_used) {
    params.is_bf16 = out.dtype() == at::ScalarType::BFloat16;
    params.is_e4m3 = true;
    auto set_fp8_descale = [](const at::Tensor &descale, const char *name) {
      CHECK_DEVICE(descale);
      TORCH_CHECK(descale.dtype() == at::ScalarType::Float,
                  name, " must have dtype float32");
      TORCH_CHECK(descale.numel() >= 1,
                  name, " must contain at least one element");
      return reinterpret_cast<float*>(descale.data_ptr());
    };
    at::Tensor scales_q = scales_q_.value();
    params.q_descale_ptr = set_fp8_descale(scales_q, "q_descale");
    params.q_descale_batch_stride = 0;
    params.q_descale_head_stride = 0;
    at::Tensor scales_k = scales_k_.value();
    params.k_descale_ptr = set_fp8_descale(scales_k, "k_descale");
    params.k_descale_batch_stride = 0;
    params.k_descale_head_stride = 0;
    at::Tensor scales_v = scales_v_.value();
    params.v_descale_ptr = set_fp8_descale(scales_v, "v_descale");
    params.v_descale_batch_stride = 0;
    params.v_descale_head_stride = 0;
  }
  set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

  at::Tensor softmax_lseaccum;
  at::Tensor out_accum;
  hipDeviceProp_t props;
  auto hipResult = hipGetDeviceProperties(&props, 0);
  params.cu_count = props.multiProcessorCount;
  params.num_splits = 1;
  const int arch_id = getArch();
  const bool enable_split =
      (arch_id >= 938 &&
       (head_size_value == 128 || head_size_value == 64 ||
        (!fp8_used && head_size_value == 512))) ||
      (arch_id == 936 && !fp8_used &&
       (head_size_value == 128 || head_size_value == 512));
  if (enable_split) {
    if (batch_size * params.h < params.cu_count / 2) {
      params.partition_size = PA_FIX_PARTITION;
      params.num_splits = 8;
      while (batch_size * params.h * params.num_splits < params.cu_count) {
        params.num_splits *= 2;
      }
      params.num_splits = std::min(64, params.num_splits);
      const bool has_local_window =
          window_size_left >= 0 || window_size_right >= 0;
      if (has_local_window) {
        const int local_left =
            window_size_left < 0 ? max_seqlen_k : window_size_left;
        const int local_right =
            window_size_right < 0 ? max_seqlen_k : window_size_right;
        const int local_seqlen_k =
            std::min(max_seqlen_k, local_left + max_seqlen_q + local_right);
        if (local_seqlen_k <= 2048) {
          params.num_splits = 1;
        }
      }
      if (params.num_splits > 1) {
        // 申请空间
        softmax_lseaccum =
            at::empty({params.num_splits, num_heads * ngroups, total_q},
                      opts.dtype(at::kFloat));
        out_accum = at::empty(
            {params.num_splits, out.size(0), out.size(1), out.size(2)},
            fp8_used ? out.options() : opts);
        params.softmax_lseaccum_ptr =
            reinterpret_cast<float *>(softmax_lseaccum.data_ptr());
        params.oaccum_ptr = out_accum.data_ptr();
      }
    }
  }

  const char *fa_debug = std::getenv("FA_DEBUG");
  if (fa_debug != nullptr) {
    if (std::strcmp(fa_debug, "1") == 0) {
      PRINT_PARAMS
    } else if (std::strcmp(fa_debug, "2") == 0) {
      PRINT_PARAMS_ONELINE
      auto temp_tensor = seqused_k.to(at::DeviceType::CPU).contiguous();
      std::vector<int32_t> temp_vector(temp_tensor.data_ptr<int32_t>(),
                                       temp_tensor.data_ptr<int32_t>() +
                                           temp_tensor.numel());
      printf("seqused_k: [");
      for (const auto val : temp_vector) {
        printf("%d ", val);
      }
      printf("]\n");
    }
    PRINT_QKV_INFO(q, k, v)
  }

  const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
  if (std::getenv("PA_EMPTY") == nullptr) {
    run_mha_fwd_kvcache(params, stream);
  }

  at::Tensor out_padded = out;
  if (head_size_value % 32 != 0) {
    out = out.index(
        {"...", at::indexing::Slice(at::indexing::None, head_size_value)});
    if (out_.has_value()) {
      out_.value().copy_(out);
    }
  }

  if (ngroups > 1) {
    out = out.view({total_q, num_heads * ngroups, -1});
    if (output_allocated_outside) { out_.value().copy_(out); }
  }

  if (return_softmax) return {out, softmax_lse};
  else return {out};
#else
    return {};
#endif
}



std::vector<at::Tensor> fwd_kvcache_mla_decoding(
    at::Tensor &q,
    const at::Tensor &kcache,
    c10::optional<const at::Tensor> &vcache,
    const int head_dim_v,
    const at::Tensor &cache_seqlens,
    const at::Tensor &block_table,
    const float softmax_scale,
    bool is_causal,
    const c10::optional<const at::Tensor> &tile_scheduler_metadata,
    const c10::optional<const at::Tensor> &num_splits,
    c10::optional<at::Tensor> &out_,
    int max_seqlen_k
) {
#if defined(BUILD_FLASHMLA)
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
    // OptionalHIPStreamGuardMasqueradingAsCUDA ?

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16, "FlashMLA only support fp16 and bf16 data type");
    TORCH_CHECK(kcache.dtype() == q_dtype, "Query and key must have the same dtype");
    CHECK_DEVICE(q); CHECK_DEVICE(kcache);
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_DEVICE(block_table);
    TORCH_CHECK(block_table.dtype() == at::ScalarType::Int, "block_table must have dtype torch.int32");
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");

    // decide layout ----> 0: bhsd, 1: bshd
    const int layout = (kcache.size(1) % 32 == 0/*page block size*/) and (kcache.size(2) == 1/*kvhead = 1, MQA*/);

    const auto sizes       = q.sizes();
    const int o_batch_size = sizes[0]; // fake batch size, may be padded in sglang, and thus o_batch_size >= batch_size
    int       num_heads    = layout == 1 ? sizes[2]: sizes[1];
    int       seqlen_q     = layout == 1 ? sizes[1]: sizes[2];
    const int head_size_og = sizes[3];
    const int head_dim_qk  = q.size(3);
    const int batch_size   = block_table.size(0); // true batch size
    const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks   = kcache.size(0);
    const int page_block_size = layout == 1 ? kcache.size(1): kcache.size(2);
    const int num_heads_k  = layout == 1 ? kcache.size(2): kcache.size(1);
    const int mtp = seqlen_q;
    TORCH_CHECK(batch_size > 0 and o_batch_size > 0, "batch size must be positive");
    TORCH_CHECK(o_batch_size >= batch_size, "batch size of query must be larger than batch_size of query");
    // TORCH_CHECK(block_table.size(0) == batch_size, "For FlashMLA, batch size of block table is not compatible with query! Please check shape!");
    TORCH_CHECK(head_dim_qk == 576, "FlashMLA only supports QK headdim 576");
    TORCH_CHECK(head_dim_v == 512, "FlashMLA only supports V headdim 512");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(mtp <= 128, "FlashMLA only support mtp <= 128 yet");
    TORCH_CHECK(not (num_heads == 128 and mtp > 1), "FlashMLA decoding doesn't support mtp when qheads = 128, not supported yet");

    // causal=true is the same as causal=false in this case
    if (mtp == 1) { is_causal = false; } else { is_causal = true; }

    // for ours flashmla, mtp and regroup are limited
    const bool use_tile_16x32 = std::getenv("MLA_USE_TILE32X32") == nullptr;
    const int MTP_REGROUP_COUNT = use_tile_16x32 ? 4: 8;
    const int MAX_MTP_ALLOWED = use_tile_16x32 ? 16 / MTP_REGROUP_COUNT: 32 / MTP_REGROUP_COUNT;

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    const int ngroups = num_heads / num_heads_k;
    const char* mla_regroup_control = std::getenv("MLA_REGROUP");
    const int mla_regroup = mla_regroup_control ? std::atoi(mla_regroup_control): 0;
    const int seqlenq_ngroups_swapped = (mtp == 1 or (mtp <= MAX_MTP_ALLOWED and num_heads <= 16)) and num_heads > num_heads_k and (mla_regroup == 0/*默认不指定 regroup*/ or (mla_regroup > 1 and mla_regroup <= num_heads/*指定的 regroup 在合理范围内*/ and (num_heads % mla_regroup == 0/*可以做 regroup*/)));
    if (seqlenq_ngroups_swapped) {
        // default reuse strategy
        if (mla_regroup == 0) {
            // limited seqlen_q_regroup due to 16x576 lds load limit
            int regroup_discount = std::ceil(ngroups * 1.f / 16);
            if (mtp > 1) {
                seqlen_q  = mtp * MTP_REGROUP_COUNT;
                num_heads = int(num_heads / MTP_REGROUP_COUNT);
            } else {
                seqlen_q  = int(ngroups / regroup_discount);
                num_heads = int(num_heads_k * regroup_discount);
            }
            if (layout == 0)      q = q.view({o_batch_size, num_heads, seqlen_q, head_dim_qk});
            else if (layout == 1) q = q.view({o_batch_size, seqlen_q, num_heads, head_dim_qk});
        } else { // use self-assigned regroup strategy
            seqlen_q  = mla_regroup;
            num_heads = num_heads_k * int(ngroups / mla_regroup);
            if (layout == 0)      q = q.view({o_batch_size, num_heads, mla_regroup, head_dim_qk});
            else if (layout == 1) q = q.view({o_batch_size, mla_regroup, num_heads, head_dim_qk});
        }
    }
    TORCH_CHECK(seqlen_q <= 128, "FlashMLA only support seqlen_q * hq / hk <= 128 yet");
    TORCH_CHECK(int64_t(o_batch_size * num_heads * seqlen_q * head_dim_qk) < /*2^31*/int64_t(2147483648), "The data amount of q must be smaller than the representation range of int");

    // Allocate and check output
    auto opts = q.options();
    at::Tensor out;
    bool output_allocated_outside = out_.has_value();
    if (output_allocated_outside) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        // CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_dim_v);
        out = out.view({q.size(0), q.size(1), q.size(2), head_dim_v});
    } else {
        out = at::empty({q.size(0), q.size(1), q.size(2), head_dim_v}, opts);
    }

    // Acquire and check cache_seqlens length information
    TORCH_CHECK(cache_seqlens.dtype() == at::ScalarType::Int, "seqlens_k must have dtype int32");
    CHECK_DEVICE(cache_seqlens);
    CHECK_CONTIGUOUS(cache_seqlens);
    auto cache_seqlens_ptr = cache_seqlens.data_ptr();

    Flash_fwd_mla_params params;
    // Reset the parameters
    memset(&params, 0, sizeof(params));
    // Set the status.
    params.layout = layout;
    params.mtp = mtp;
    params.is_bf16 = q.dtype() == at::ScalarType::BFloat16;
    params.is_e4m3 = q.dtype() == at::ScalarType::Float8_e4m3fn;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
    params.is_seqlens_k_cumulative = cache_seqlens.size(0) == (batch_size + 1);
    // Set the pointers.
    params.q_ptr = q.data_ptr();
    params.k_ptr = kcache.data_ptr();
    params.v_ptr = kcache.data_ptr();
    params.o_ptr = out.data_ptr();
    params.cu_seqlens_q = static_cast<int *>(cache_seqlens_ptr);
    params.cu_seqlens_k = static_cast<int *>(cache_seqlens_ptr);
    // Set the strides.
    params.q_batch_stride = q.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_head_stride  = (layout == 1) ? q.stride(2): q.stride(1);
    params.k_head_stride  = (layout == 1) ? kcache.stride(2): kcache.stride(1);
    params.v_head_stride  = params.k_head_stride;
    params.o_head_stride  = (layout == 1) ? out.stride(2): out.stride(1);
    params.q_row_stride   = (layout == 1) ? q.stride(1): q.stride(2);
    params.k_row_stride   = (layout == 1) ? kcache.stride(1): kcache.stride(2);
    params.v_row_stride   = params.k_row_stride;
    params.o_row_stride   = (layout == 1) ? out.stride(1): out.stride(2);
    // Set the dimensions etc.
    params.b   = batch_size;
    params.h   = num_heads;
    params.h_k = num_heads_k;
    params.d   = head_dim_qk;
    params.d_v = head_dim_v;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    // Set the block table.
    params.block_table     = block_table.data_ptr<int>();
    params.page_block_size = page_block_size;
    params.block_table_batch_stride = block_table.stride(0);
    params.k_batch_stride  = kcache.stride(0);
    params.v_batch_stride  = kcache.stride(0);

    // get cu_count
    hipDeviceProp_t props;
    auto hipResult  = hipGetDeviceProperties(&props, 0);
    params.cu_count = props.multiProcessorCount;

    at::Tensor out_accum, softmax_lse_accum;
    // MTP == 1, 而且没有禁止 splitkv 的情况下, 对 seqkv 进行划分
    bool env_allow_splitkv = bool(std::getenv("MLA_NO_SPLITKV") == nullptr);
    bool allow_splitkv = max_seqlen_k >= 128 and mtp <= 128 and env_allow_splitkv;
    if (allow_splitkv) {
        int partition_size = 0;
        const char* partition_size_env = std::getenv("MLA_PARTITION_SIZE");
        const int partition_size_assign = partition_size_env ? std::atoi(partition_size_env): 0;
        // 如果没有指定 partition size, 启发式决定切分策略
        if (partition_size_assign == 0) {
            // 如果初步能划分的 block 数量对应的利用率不高
            constexpr int device_cu = 100;
            const int threshold     = device_cu * 0.8;
            constexpr int large_seq = 4096;
            // 如果目前能发的 TG 数量比较少而且最大的 seqkv 不是很短, 根据 batch 来决定切多大
            if (batch_size * num_heads * mtp < threshold and max_seqlen_k >= 512 and max_seqlen_k < large_seq) {
                if (batch_size < 8)       partition_size = 128;
                else if (batch_size < 16) partition_size = 256;
                else if (batch_size < 32) partition_size = 512;
                else if (batch_size < 64) partition_size = 1024;
            } else if (max_seqlen_k >= large_seq) { // 或者 seqkv 足够长, 直接根据 seqkv 来决定切多大
                partition_size = 1024;
                // 如果按照上述划分之后, 利用率还不是很高, partition size 继续减半
                int splits = std::ceil(max_seqlen_k / partition_size);
                while (batch_size * num_heads * mtp * splits < threshold) {
                    // 目前支持的最小 partition size 是 128
                    if (partition_size < 256) break;
                    partition_size = int(partition_size / 2);
                    splits *= 2;
                }
            }
        } else if (partition_size_assign >= 128 and partition_size_assign % 128 == 0 and partition_size_assign <= max_seqlen_k) {
            // 指定的 partition_size 满足划分的需求, 目前只支持 128 的倍数, 则可以开始划分
            partition_size = partition_size_assign;
        }
        int num_splits = std::ceil(max_seqlen_k * 1.f / partition_size);
        // 如果划分成功
        if (partition_size > 0 and partition_size >= 128/*partition_size 本身合理*/ and num_splits <= 1024/*最多只能支持 1024 个划分*/) {
            // 传递给 kernel args
            params.partition_size = partition_size;
            params.num_splits     = num_splits;
            // 申请 scores_max/sum 和 out_accum 的空间
            auto raw_memory = at::empty({1, params.num_splits, o_batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
            softmax_lse_accum = raw_memory.index({0});
            if (layout == 0) out_accum  = at::empty({params.num_splits, o_batch_size, num_heads, seqlen_q, head_dim_v}, opts.dtype(q_dtype));
            else if (layout == 1) out_accum  = at::empty({params.num_splits, o_batch_size, seqlen_q, num_heads, head_dim_v}, opts.dtype(q_dtype));
            params.softmax_lse_ptr = reinterpret_cast<float*>(softmax_lse_accum.data_ptr());
            params.oaccum_ptr      = out_accum.data_ptr();
        }
    } else if (env_allow_splitkv) { // 开启 cuda graph 可走这里
        const int num_splits_assigned = 8;
        if (num_splits_assigned > 1 and batch_size <= 32) {
            // 传递给 kernel args
            params.partition_size = MLA_FIX_PARTITION;
            params.num_splits     = num_splits_assigned;
            while (o_batch_size * params.num_splits < 64) {
                params.num_splits *= 2;
            }
            params.num_splits = o_batch_size == 1 ? 32: params.num_splits; // for tiny batch size 1, splitkv reduce 64 may be the bottleneck
            params.num_splits = std::min(64, params.num_splits);
            // 申请 scores_max/sum 和 out_accum 的空间
            auto raw_memory = at::empty({1, params.num_splits, o_batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
            softmax_lse_accum = raw_memory.index({0});
            if (layout == 0) out_accum  = at::empty({params.num_splits, o_batch_size, num_heads, seqlen_q, head_dim_v}, opts.dtype(q_dtype));
            else if (layout == 1) out_accum  = at::empty({params.num_splits, o_batch_size, seqlen_q, num_heads, head_dim_v}, opts.dtype(q_dtype));
            params.softmax_lse_ptr = reinterpret_cast<float*>(softmax_lse_accum.data_ptr());
            params.oaccum_ptr      = out_accum.data_ptr();
        }
    }

    // decide accumulation dtype when splitkv
    if (params.partition_size > 0 and params.num_splits > 1) {
        params.splitkv_use_fp32_as_accum = out_accum.dtype() == at::ScalarType::Float;
    }

    const char* env_info = std::getenv("FA_DEBUG");
    if (env_info != nullptr) {
        PRINT_MLA_PARAMS
        PRINT_QKV_INFO(q, kcache, kcache);
        PRINT_TENSOR_INFO(out, "out");
        std::cout << "block_table: " << block_table.sizes() << std::endl;
        if (std::strcmp(env_info, "2") == 0) {
            auto temp_tensor = cache_seqlens.to(at::DeviceType::CPU).contiguous(); // to cpu op may interrupt cudagraph
            std::vector<int32_t> temp_vector(temp_tensor.data_ptr<int32_t>(), temp_tensor.data_ptr<int32_t>() + temp_tensor.numel());
            printf("cache_seqlens: ["); for (const auto val: temp_vector) { printf("%d ", val); } printf("]\n");
        }
    }

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if (max_seqlen_k > 0 and std::getenv("MLA_DECODE_EMPTY") == nullptr) {
        FP16_SWITCH(!params.is_bf16, [&] {
            run_mla_fwd_splitkv_dispatch<elem_type, 576, 512>(params, stream);
        });
    } else {
        out.zero_();
    }

    if (seqlenq_ngroups_swapped) {
        if (layout == 0) {
            if (mtp > 1) {
                out = out.view({o_batch_size, num_heads * MTP_REGROUP_COUNT, mtp, head_dim_v});
                if (params.partition_size > 0) out_accum = out_accum.view({params.num_splits, o_batch_size, num_heads * MTP_REGROUP_COUNT, mtp, head_dim_v});
            } else {
                out = out.view({o_batch_size, num_heads_k * ngroups, mtp, head_dim_v});
                if (params.partition_size > 0) out_accum = out_accum.view({params.num_splits, o_batch_size, num_heads_k * ngroups, mtp, head_dim_v});
            }
        } else if (layout == 1) {
            if (mtp > 1) {
                out = out.view({o_batch_size, mtp, num_heads * MTP_REGROUP_COUNT, head_dim_v});
                if (params.partition_size > 0) out_accum = out_accum.view({params.num_splits, o_batch_size, mtp, num_heads * MTP_REGROUP_COUNT, head_dim_v});
            } else {
                out = out.view({o_batch_size, mtp, num_heads_k * ngroups, head_dim_v}); // kheads 为 1, 所以不用加一步 contiguous()
                if (params.partition_size > 0) out_accum = out_accum.view({params.num_splits, o_batch_size, mtp, num_heads_k * ngroups, head_dim_v});
            }
        }
    }

    if (output_allocated_outside) {
        return {};
    } else {
        return {out, out_accum, softmax_lse_accum};
    }
#else
    return {};
#endif
}



std::vector<at::Tensor> fwd_kvcache_mla_dataparallel(
    at::Tensor &q_all,
    at::Tensor &kvcache,
    c10::optional<const at::Tensor> &vcache_,
    const int headdim_v,
    const at::Tensor &cache_seqlens,
    const at::Tensor &page_table,
    const float softmax_scale,
    const bool is_causal,
    const c10::optional<const at::Tensor> &tile_scheduler_metadata,
    const c10::optional<const at::Tensor> &num_splits,
    c10::optional<at::Tensor> &out_,
    int max_seqlen_k
) {
#if defined(BUILD_FLASHMLA)
    // 类型检查
    TORCH_CHECK(q_all.dtype() == at::ScalarType::Half || q_all.dtype() == at::ScalarType::BFloat16, "Fwd_kvcache_mla only support fp16 and bf16 data type for q");
    TORCH_CHECK(kvcache.dtype() == at::ScalarType::Half || kvcache.dtype() == at::ScalarType::BFloat16, "Fwd_kvcache_mla mla only support fp16 and bf16 data type for kcache");
    TORCH_CHECK(cache_seqlens.dtype() == at::ScalarType::Int, "Fwd_kvcache_mla only support int32_t data type for cache_seqlens");
    TORCH_CHECK(page_table.dtype() == at::ScalarType::Int, "Fwd_kvcache_mla only support int32_t data type for page_table");
    // device 检查
    CHECK_DEVICE(q_all); CHECK_DEVICE(kvcache); CHECK_DEVICE(page_table); CHECK_DEVICE(cache_seqlens);
    // 连续性检查
    CHECK_CONTIGUOUS(page_table); CHECK_CONTIGUOUS(cache_seqlens);
    // 张量 shape 检查, 是否是 3/4 维这种
    TORCH_CHECK(q_all.dim() == 4, "In fwd_kvcache_mla, q must be 4-dimension tensor");
    TORCH_CHECK(kvcache.dim() == 4, "In fwd_kvcache_mla, kvcache must be 4-dimension tensor");
    TORCH_CHECK(page_table.dim() == 2, "In fwd_kvcache_mla, page_table must be 2-dimension tensor");
    // 获取基本信息
    const auto q_size       = q_all.sizes();
    const int o_batch_size  = q_size[0];
    const int headdim_qk    = q_size[3];
    const int headdim_rope  = headdim_qk - headdim_v;
    const int batch_size    = page_table.size(0);
    const int num_heads_ori = q_size[2];
    const int num_heads_k   = kvcache.size(2);
    const int page_block_size = kvcache.size(1);
    TORCH_CHECK(batch_size > 0 and o_batch_size > 0, "batch size must be positive");
    TORCH_CHECK(o_batch_size >= batch_size, "batch size of query must be larger than batch_size of query");
    // TORCH_CHECK(page_table.size(0) == q_size[0], "In fwd_kvcache_mla, batch_size of page_table must be compatible with query! Please check shape!");
    // GQA regroup 优化
    TORCH_CHECK(num_heads_ori % num_heads_k == 0, "In fwd_kvcache_mla, qheads must be multiple of kvheads! Please check layout and shape!");
    const int seqlen_q_ori = q_size[1];
    const int ngroups = num_heads_ori / num_heads_k;
    const int seqlen_q = seqlen_q_ori * ngroups;
    const int num_heads = num_heads_k;
    q_all = q_all.view({o_batch_size, seqlen_q, num_heads_k, headdim_qk});
    // 沿着 headdim 切分 q
    const auto q = q_all.narrow(-1, headdim_v, headdim_rope);
    const auto qv = q_all.narrow(-1, 0, headdim_v);
    // 沿着 headdim 切分 k, v
    const auto kcache = kvcache.narrow(-1, headdim_v, headdim_rope);
    const auto vcache = kvcache.narrow(-1, 0, headdim_v);
    const auto kcache_size = kcache.sizes();
    const auto vcache_size = vcache.sizes();
    // 检查 size 是否符合要求
    TORCH_CHECK(headdim_v == 512, "In fwd_kvcache_mla, headdim_v must be 512");
    TORCH_CHECK(headdim_rope == 64, "In fwd_kvcache_mla, headdim_rope must be 64");
    TORCH_CHECK(headdim_qk == 576, "In fwd_kvcache_mla, headdim_qk must be 576");
    TORCH_CHECK(page_block_size == 128, "In fwd_kvcache_mla, page_block_size must be 128")
    // 检查平台
    hipDeviceProp_t props;
    auto hipResult = hipGetDeviceProperties(&props, 0);
    std::string gcn_arch_name(props.gcnArchName);
    const int gcn_arch = runtime_gfx_arch_id(gcn_arch_name);
    TORCH_CHECK(is_supported_hg_mla_arch(gcn_arch_name, gcn_arch), "In fwd_kvcache_mla, only gfx92a or arch id >= gfx936 is supported!");
    // 准备输出变量
    auto opts = q.options();
    at::Tensor out, softmax_lse, scores_max, scores_sum;
    out = at::empty({q.size(0), q.size(1), q.size(2), headdim_v}, opts);
    if (true/*return_softmax_lse*/) { // extra op for return_softmax_lse may lead to 2.3% performance drop, slightly
        auto scores_memory = at::empty({3, o_batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
        scores_max = scores_memory.index({0});
        scores_sum = scores_memory.index({1});
        softmax_lse = scores_memory.index({2});
    }

    // NMZ走MLS FlashMLA
    bool IS_DP_MLA_MLS = false;
    if (gcn_arch >= 938 and std::getenv("MLA_DP_DECODE_NO_MLS") == nullptr and o_batch_size>= 16) IS_DP_MLA_MLS = true;

    // 准备 kernel 需要的参数列表
    Flash_fwd_mla_params params;
    memset(&params, 0, sizeof(params));
    params.layout           = 1;
    params.b                = batch_size;
    params.h                = num_heads;
    params.h_k              = num_heads_k;
    params.h_h_k_ratio      = params.h / params.h_k;
    params.mtp              = seqlen_q_ori;
    params.d                = headdim_qk;
    params.d_v              = headdim_v;
    params.scale_softmax    = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.cu_seqlens_q     = nullptr; // <int32_t*>(cu_seqlens_q.data_ptr());
    params.cu_seqlens_k     = reinterpret_cast<int32_t*>(cache_seqlens.data_ptr());
    params.q_ptr            = IS_DP_MLA_MLS ? q_all.data_ptr() : q.data_ptr();
    params.qv_ptr           = IS_DP_MLA_MLS ? nullptr : qv.data_ptr();
    params.k_ptr            = IS_DP_MLA_MLS ? kvcache.data_ptr() : kcache.data_ptr();
    params.v_ptr            = IS_DP_MLA_MLS ? kvcache.data_ptr() : vcache.data_ptr();
    params.o_ptr            = out.data_ptr();
    params.softmax_lse_ptr  = softmax_lse.data_ptr<float>();
    params.scores_max_ptr   = scores_max.data_ptr<float>();
    params.scores_sum_ptr   = scores_sum.data_ptr<float>();
    params.block_table      = reinterpret_cast<int32_t*>(page_table.data_ptr());
    params.block_table_batch_stride = page_table.stride(0);
    params.page_block_size  = page_block_size;
    params.is_causal        = is_causal;
    params.q_batch_stride   = IS_DP_MLA_MLS ? q_all.stride(0) : q.stride(0);
    params.q_row_stride     = IS_DP_MLA_MLS ? q_all.stride(1) : q.stride(1);
    params.q_head_stride    = IS_DP_MLA_MLS ? q_all.stride(2) : q.stride(2);
    params.qv_batch_stride  = qv.stride(0);
    params.qv_row_stride    = qv.stride(1);
    params.qv_head_stride   = qv.stride(2);
    params.k_batch_stride   = IS_DP_MLA_MLS ? kvcache.stride(0) : kcache.stride(0);
    params.k_row_stride     = IS_DP_MLA_MLS ? kvcache.stride(1) : kcache.stride(1);
    params.k_head_stride    = IS_DP_MLA_MLS ? kvcache.stride(2) : kcache.stride(2);
    params.v_batch_stride   = IS_DP_MLA_MLS ? kvcache.stride(0) : vcache.stride(0);
    params.v_row_stride     = IS_DP_MLA_MLS ? kvcache.stride(1) : vcache.stride(1);
    params.v_head_stride    = IS_DP_MLA_MLS ? kvcache.stride(2) : vcache.stride(2);
    params.o_batch_stride   = out.stride(0);
    params.o_row_stride     = out.stride(1);
    params.o_head_stride    = out.stride(2);
    params.seqlen_q         = seqlen_q;
    params.ngroups          = ngroups;
    params.is_bf16          = q.dtype() == at::ScalarType::BFloat16;
    params.cu_count         = props.multiProcessorCount;
    params.seqlenq_ngroups_swapped = true;

    // DEBUG
    const char* fa_debug = std::getenv("FA_DEBUG");
    if (fa_debug != nullptr) {
        PRINT_MLA_PARAMS
        if (strcmp(fa_debug, "2") == 0) { // print operations listed below may interrupt cudagraph, and thus only print tensors util FA_DEBUG=2
            PRINT_TENSOR(cache_seqlens, "cache_seqlens")
        }
        PRINT_TENSOR_INFO(q, "q")
        PRINT_TENSOR_INFO(kcache, "kcache")
        PRINT_TENSOR_INFO(vcache, "vcache")
        PRINT_TENSOR_INFO(qv, "qv")
    }

    // 准备启动 kernel
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if (std::getenv("MLA_DECODE_EMPTY") == nullptr) {
        FP16_SWITCH(!params.is_bf16, [&] {
            run_mla_fwd_dispatch<elem_type, 576, 512>(params, stream);
        });
    } else {
        out.zero_();
    }

    // GQA 优化重排
    out = out.view({o_batch_size, seqlen_q_ori, ngroups * num_heads_k, headdim_v});
    if (params.mtp == 1) {
        softmax_lse = softmax_lse.view({o_batch_size, ngroups * num_heads_k, seqlen_q_ori});
    } else {
        softmax_lse = softmax_lse.view({o_batch_size, num_heads_k, seqlen_q_ori, ngroups}).transpose(-1, -2).contiguous().view({o_batch_size, ngroups * num_heads_k, seqlen_q_ori});
    }

    return {out, softmax_lse, scores_max, scores_sum};
#else
    return {};
#endif
}


std::vector<at::Tensor> hg_fwd_kvcache_mla(
    at::Tensor &q_all,
    at::Tensor &kvcache,
    c10::optional<const at::Tensor> &vcache_,
    const int headdim_v,
    const at::Tensor &seqlens_k,
    const at::Tensor &block_table,
    const float softmax_scale,
    const bool is_causal,
    const c10::optional<const at::Tensor> &tile_scheduler_metadata,
    const c10::optional<const at::Tensor> &num_splits,
    c10::optional<at::Tensor> &out_,
    int max_seqlen_k
) {
    int qheads = max(q_all.size(1), q_all.size(2));
    if (qheads == 128)
        return fwd_kvcache_mla_dataparallel(q_all, kvcache, vcache_, headdim_v, seqlens_k, block_table, softmax_scale, is_causal, tile_scheduler_metadata, num_splits, out_, max_seqlen_k);
    return fwd_kvcache_mla_decoding(q_all, kvcache, vcache_, headdim_v, seqlens_k, block_table, softmax_scale, is_causal, tile_scheduler_metadata, num_splits, out_, max_seqlen_k);
}




std::vector<at::Tensor> fwd_kvcache_mla_decoding_fp8(
    at::Tensor &q,
    const at::Tensor &kcache,
    c10::optional<const at::Tensor> &vcache,
    const int head_dim_v,
    const at::Tensor &cache_seqlens,
    const at::Tensor &block_table,
    const float softmax_scale,
    bool is_causal,
    const c10::optional<const at::Tensor> &tile_scheduler_metadata,
    const c10::optional<const at::Tensor> &num_splits,
    c10::optional<at::Tensor> &out_,
    int max_seqlen_k,
    const at::Tensor& descale_q,
    const at::Tensor& descale_k
) {
#if defined(BUILD_FLASHMLA)
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == at::ScalarType::Float8_e4m3fn, "FlashMLA_FP8 only support fp8_e4m3 data type");
    TORCH_CHECK(kcache.dtype() == q_dtype, "Query and key must have the same dtype");
    CHECK_DEVICE(q); CHECK_DEVICE(kcache);
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_DEVICE(block_table);
    TORCH_CHECK(block_table.dtype() == at::ScalarType::Int, "block_table must have dtype torch.int32");
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    CHECK_DEVICE(descale_q);
    TORCH_CHECK(descale_q.dtype() == at::ScalarType::Float, "descale_q must have dtype torch.float32");
    TORCH_CHECK(descale_q.is_contiguous(), "descale_q must be contiguous");
    CHECK_SHAPE(descale_q, 1);
    CHECK_DEVICE(descale_k);
    TORCH_CHECK(descale_k.dtype() == at::ScalarType::Float, "descale_k must have dtype torch.float32");
    TORCH_CHECK(descale_k.is_contiguous(), "descale_k must be contiguous");
    CHECK_SHAPE(descale_k, 1);

    // decide layout ----> 0: bhsd, 1: bshd
    const int layout = (kcache.size(1) % 32 == 0/*page block size*/) and (kcache.size(2) == 1/*kvhead = 1, MQA*/);

    const auto sizes       = q.sizes();
    const int o_batch_size = sizes[0]; // fake batch size, may be padded in sglang, and thus o_batch_size >= batch_size
    int       num_heads    = layout == 1 ? sizes[2]: sizes[1];
    int       seqlen_q     = layout == 1 ? sizes[1]: sizes[2];
    const int head_size_og = sizes[3];
    const int head_dim_qk  = q.size(3);
    const int batch_size   = block_table.size(0); // true batch size
    const int max_num_blocks_per_seq = block_table.size(1);
    const int num_blocks   = kcache.size(0);
    const int page_block_size = layout == 1 ? kcache.size(1): kcache.size(2);
    const int num_heads_k  = layout == 1 ? kcache.size(2): kcache.size(1);
    const int mtp = seqlen_q;
    const bool is_prefill = bool(mtp > 16); // seqlen_q > 16, usage for prefill
    TORCH_CHECK(batch_size > 0 and o_batch_size > 0, "batch size must be positive");
    TORCH_CHECK(o_batch_size >= batch_size, "batch size of query must be larger than batch_size of query");
    // TORCH_CHECK(block_table.size(0) == batch_size, "For FlashMLA, batch size of block table is not compatible with query! Please check shape!");
    TORCH_CHECK(head_dim_qk == 576, "FlashMLA only supports QK headdim 576");
    TORCH_CHECK(head_dim_v == 512, "FlashMLA only supports V headdim 512");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    TORCH_CHECK(not (num_heads == 128 and mtp > 1), "FlashMLA decoding doesn't support mtp when qheads = 128, not supported yet");

    // causal=true is the same as causal=false in this case
    if (mtp == 1) { is_causal = false; } else { is_causal = true; }

    // for ours flashmla, mtp and regroup are limited
    const bool use_tile_16x32 = std::getenv("MLA_USE_TILE32X32") == nullptr;
    const int MTP_REGROUP_COUNT = use_tile_16x32 ? 4: 8;
    const int MAX_MTP_ALLOWED = use_tile_16x32 ? 16 / MTP_REGROUP_COUNT: 32 / MTP_REGROUP_COUNT;

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    const int ngroups = num_heads / num_heads_k;
    const char* mla_regroup_control = std::getenv("MLA_REGROUP");
    const int mla_regroup = mla_regroup_control ? std::atoi(mla_regroup_control): 0;
    const int seqlenq_ngroups_swapped = (mtp == 1 or (mtp <= MAX_MTP_ALLOWED and num_heads <= 16)) and num_heads > num_heads_k and (mla_regroup == 0/*默认不指定 regroup*/ or (mla_regroup > 1 and mla_regroup <= num_heads/*指定的 regroup 在合理范围内*/ and (num_heads % mla_regroup == 0/*可以做 regroup*/)));
    if (seqlenq_ngroups_swapped) {
        // default reuse strategy
        if (mla_regroup == 0) {
            // limited seqlen_q_regroup due to 16x576 lds load limit
            int regroup_discount = std::ceil(ngroups * 1.f / 16);
            if (mtp > 1) {
                seqlen_q  = mtp * MTP_REGROUP_COUNT;
                num_heads = int(num_heads / MTP_REGROUP_COUNT);
            } else {
                seqlen_q  = int(ngroups / regroup_discount);
                num_heads = int(num_heads_k * regroup_discount);
            }
            if (layout == 0)      q = q.view({o_batch_size, num_heads, seqlen_q, head_dim_qk});
            else if (layout == 1) q = q.view({o_batch_size, seqlen_q, num_heads, head_dim_qk});
        } else { // use self-assigned regroup strategy
            seqlen_q  = mla_regroup;
            num_heads = num_heads_k * int(ngroups / mla_regroup);
            if (layout == 0)      q = q.view({o_batch_size, num_heads, mla_regroup, head_dim_qk});
            else if (layout == 1) q = q.view({o_batch_size, mla_regroup, num_heads, head_dim_qk});
        }
    }
    TORCH_CHECK(int64_t(o_batch_size * num_heads * seqlen_q * head_dim_qk) < /*2^31*/int64_t(2147483648), "The data amount of q must be smaller than the representation range of int");

    // Allocate and check output
    auto opts = q.options();
    at::Tensor out;
    bool output_allocated_outside = out_.has_value();
    constexpr auto MLAFP8OutputDtype = at::ScalarType::BFloat16;
    if (output_allocated_outside) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == MLAFP8OutputDtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        // CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_dim_v);
        out = out.view({q.size(0), q.size(1), q.size(2), head_dim_v});
    } else {
        out = at::empty({q.size(0), q.size(1), q.size(2), head_dim_v}, opts.dtype(MLAFP8OutputDtype));
    }

    // Acquire and check cache_seqlens length information
    TORCH_CHECK(cache_seqlens.dtype() == at::ScalarType::Int, "seqlens_k must have dtype int32");
    CHECK_DEVICE(cache_seqlens);
    CHECK_CONTIGUOUS(cache_seqlens);
    auto cache_seqlens_ptr = cache_seqlens.data_ptr();

    Flash_fwd_mla_params params;
    // Reset the parameters
    memset(&params, 0, sizeof(params));
    // Set the status.
    params.layout = layout;
    params.mtp = mtp;
    params.is_e4m3 = true;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
    params.is_seqlens_k_cumulative = cache_seqlens.size(0) == (batch_size + 1);
    // Set the pointers.
    params.q_ptr = q.data_ptr();
    params.k_ptr = kcache.data_ptr();
    params.v_ptr = kcache.data_ptr();
    params.o_ptr = out.data_ptr();
    params.cu_seqlens_q = static_cast<int *>(cache_seqlens_ptr);
    params.cu_seqlens_k = static_cast<int *>(cache_seqlens_ptr);
    // Set the descale
    params.scales_q_ptr = reinterpret_cast<float *>(descale_q.data_ptr<float>());
    params.scales_k_ptr = reinterpret_cast<float *>(descale_k.data_ptr<float>());
    // Set the strides.
    params.q_batch_stride = q.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_head_stride  = (layout == 1) ? q.stride(2): q.stride(1);
    params.k_head_stride  = (layout == 1) ? kcache.stride(2): kcache.stride(1);
    params.v_head_stride  = params.k_head_stride;
    params.o_head_stride  = (layout == 1) ? out.stride(2): out.stride(1);
    params.q_row_stride   = (layout == 1) ? q.stride(1): q.stride(2);
    params.k_row_stride   = (layout == 1) ? kcache.stride(1): kcache.stride(2);
    params.v_row_stride   = params.k_row_stride;
    params.o_row_stride   = (layout == 1) ? out.stride(1): out.stride(2);
    // Set the dimensions etc.
    params.b   = batch_size;
    params.h   = num_heads;
    params.h_k = num_heads_k;
    params.d   = head_dim_qk;
    params.d_v = head_dim_v;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    // Set the block table.
    params.block_table     = block_table.data_ptr<int>();
    params.page_block_size = page_block_size;
    params.block_table_batch_stride = block_table.stride(0);
    params.k_batch_stride  = kcache.stride(0);
    params.v_batch_stride  = kcache.stride(0);

    at::Tensor out_accum, softmax_lse_accum;
    // 对 seqkv 进行划分
    bool allow_splitkv = bool(std::getenv("MLA_NO_SPLITKV") == nullptr) and !is_prefill;
    if (allow_splitkv) {
        const int num_splits_assigned = 8;
        if (num_splits_assigned > 1 and batch_size <= 32) {
            // 传递给 kernel args
            params.partition_size = MLA_FIX_PARTITION;
            params.num_splits     = num_splits_assigned;
            while (o_batch_size * params.num_splits < 64) {
                params.num_splits *= 2;
            }
            params.num_splits = o_batch_size == 1 ? 32: params.num_splits; // for tiny batch size 1, splitkv reduce 64 may be the bottleneck
            params.num_splits = std::min(64, params.num_splits);
            // 申请 scores_max/sum 和 out_accum 的空间
            auto raw_memory = at::empty({1, params.num_splits, o_batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
            softmax_lse_accum = raw_memory.index({0});
            if (layout == 0) out_accum  = at::empty({params.num_splits, o_batch_size, num_heads, seqlen_q, head_dim_v}, opts.dtype(MLAFP8OutputDtype));
            else if (layout == 1) out_accum  = at::empty({params.num_splits, o_batch_size, seqlen_q, num_heads, head_dim_v}, opts.dtype(MLAFP8OutputDtype));
            params.softmax_lse_ptr = reinterpret_cast<float*>(softmax_lse_accum.data_ptr());
            params.oaccum_ptr      = out_accum.data_ptr();
        }
    }

    // decide accumulation dtype when splitkv
    if (params.partition_size > 0 and params.num_splits > 1) {
        params.splitkv_use_fp32_as_accum = out_accum.dtype() == at::ScalarType::Float;
    }

    const char* env_info = std::getenv("FA_DEBUG");
    if (env_info != nullptr) {
        PRINT_MLA_PARAMS
        PRINT_QKV_INFO(q, kcache, kcache);
        PRINT_TENSOR_INFO(out, "out");
        std::cout << "block_table: " << block_table.sizes() << std::endl;
        if (std::strcmp(env_info, "2") == 0) {
            auto temp_tensor = cache_seqlens.to(at::DeviceType::CPU).contiguous(); // to cpu op may interrupt cudagraph
            std::vector<int32_t> temp_vector(temp_tensor.data_ptr<int32_t>(), temp_tensor.data_ptr<int32_t>() + temp_tensor.numel());
            printf("cache_seqlens: ["); for (const auto val: temp_vector) { printf("%d ", val); } printf("]\n");
        }
    }

    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if (max_seqlen_k > 0 and std::getenv("MLA_DECODE_EMPTY") == nullptr) {
        run_fp8_mla_fwd_splitkv_dispatch<BFloat16, 576, 512>(params, stream);
    } else {
        out.zero_();
    }

    if (seqlenq_ngroups_swapped) {
        if (layout == 0) {
            if (mtp > 1) {
                out = out.view({o_batch_size, num_heads * MTP_REGROUP_COUNT, mtp, head_dim_v});
                if (params.partition_size > 0) out_accum = out_accum.view({params.num_splits, o_batch_size, num_heads * MTP_REGROUP_COUNT, mtp, head_dim_v});
            } else {
                out = out.view({o_batch_size, num_heads_k * ngroups, mtp, head_dim_v});
                if (params.partition_size > 0) out_accum = out_accum.view({params.num_splits, o_batch_size, num_heads_k * ngroups, mtp, head_dim_v});
            }
        } else if (layout == 1) {
            if (mtp > 1) {
                out = out.view({o_batch_size, mtp, num_heads * MTP_REGROUP_COUNT, head_dim_v});
                if (params.partition_size > 0) out_accum = out_accum.view({params.num_splits, o_batch_size, mtp, num_heads * MTP_REGROUP_COUNT, head_dim_v});
            } else {
                out = out.view({o_batch_size, mtp, num_heads_k * ngroups, head_dim_v}); // kheads 为 1, 所以不用加一步 contiguous()
                if (params.partition_size > 0) out_accum = out_accum.view({params.num_splits, o_batch_size, mtp, num_heads_k * ngroups, head_dim_v});
            }
        }
    }

    if (output_allocated_outside) {
        return {};
    } else {
        return {out, out_accum, softmax_lse_accum};
    }
#else
    return {};
#endif
}


at::Tensor flash_mla_convert_query_to_fp8(
    at::Tensor& q_nope,
    at::Tensor& q_rope,
    const bool is_fp8
) {
#if defined(BUILD_FLASHMLA)
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q_nope.device().index());
    TORCH_CHECK(is_fp8, "flash_mla_convert_query only support return tensor of fp8 yet! Bf8 is not supported yet!");
    TORCH_CHECK(q_nope.dtype() == at::ScalarType::Half || q_nope.dtype() == at::ScalarType::BFloat16, "flash_mla_convert_query only support fp16 and bf16 data type for q");
    TORCH_CHECK(q_rope.dtype() == q_rope.dtype(), "flash_mla_convert_query only support same dtype for q_nope, q_rope");
    CHECK_DEVICE(q_nope);
    CHECK_DEVICE(q_rope);
    // Acquire basic information
    const int batch_size = q_nope.size(0);
    const int qheads = q_nope.size(-2);
    const int headdim_nope = q_nope.size(-1);
    const int headdim_rope = q_rope.size(-1);
    const int headdim_qk = headdim_nope + headdim_rope;
    const int seqlen_q = q_nope.dim() == 3 ? 1: q_nope.size(1);
    const bool is_bf16 = q_nope.dtype() == at::ScalarType::BFloat16;
    // Prepare output tensor
    at::Tensor query_fp8;
    query_fp8 = at::empty({batch_size, seqlen_q, qheads, headdim_qk}, q_nope.options().dtype(at::ScalarType::Float8_e4m3fn));
    if (q_nope.dim() == 3) query_fp8 = query_fp8.view({batch_size * seqlen_q, qheads, headdim_qk});
    // Params
    Flash_fwd_mla_params params;
    params.o_ptr          = query_fp8.data_ptr();
    params.qv_ptr         = q_nope.data_ptr();
    params.q_ptr          = q_rope.data_ptr();
    params.o_head_stride  = query_fp8.stride(-2);
    params.qv_head_stride = q_nope.stride(-2);
    params.q_head_stride  = q_rope.stride(-2);
    params.total_blocks   = batch_size * seqlen_q * qheads;
    params.qv_row_stride  = q_nope.stride(-3);
    params.q_row_stride   = q_rope.stride(-3);
    params.h              = qheads;
    // Debug
    const char* env_info = std::getenv("FA_DEBUG");
    if (env_info != nullptr) {
        std::cout << "flash_mla_convert_query_to_fp8 kernel: " << std::endl;
        std::cout << "batch_size: " << batch_size / seqlen_q << std::endl;
        std::cout << "q_nope: " << q_nope.sizes() << ", " << q_nope.strides() << ", " << q_nope.dtype() << std::endl;
        std::cout << "q_rope: " << q_rope.sizes() << ", " << q_rope.strides() << ", " << q_rope.dtype() << std::endl;
    }
    // Launch Kernel
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    FP16_SWITCH(!is_bf16, [&]{
        run_fp8_mla_convert_q_to_fp8_dispatch<elem_type, 576, 512>(params, stream);
    });
    return query_fp8;
#else
    return at::Tensor();
#endif
}


#ifdef BUILD_FA_PERMUTE
#include "flash_permute_api.h"

// Preserved for emergency
at::Tensor varlen_fwd_bshd_with_permute(
        at::Tensor &q,
        at::Tensor &k,
        at::Tensor &v,
        c10::optional<at::Tensor> &out_,
        const at::Tensor &cu_seqlens_q,
        const at::Tensor &cu_seqlens_k,
        c10::optional<at::Tensor> &seqused_k,
        c10::optional<at::Tensor> &alibi_slopes_,
        const int max_seqlen_q,
        const int max_seqlen_k,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_) {
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
    // [batch x seqlen, num_head, headdim] ----> [batch x num_head x seqlen, headdim]
    const auto query_size = q.sizes();
    const bool tensor_is_4dim = query_size.size() == 4;
    const int num_heads = tensor_is_4dim ? query_size[2]: query_size[1];
    const int num_heads_kv = tensor_is_4dim ? k.size(2): k.size(1);
    auto pre_permuted = varlen_fwd_permute_bshd2bhsd(q, k, v, cu_seqlens_q, max_seqlen_q); // 默认 cu_seqlens_q = cu_seqlens_k
    c10::optional<at::Tensor> q_descale_ = c10::nullopt;
    c10::optional<at::Tensor> k_descale_ = c10::nullopt;
    c10::optional<at::Tensor> v_descale_ = c10::nullopt;
    // FA kernel
    auto fa_out = varlen_fwd(
        pre_permuted[0],
        pre_permuted[1],
        pre_permuted[2],
        num_heads,
        num_heads_kv,
        out_,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        alibi_slopes_,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen_,
        0/*bhsd*/,
        q_descale_,
        k_descale_,
        v_descale_,
        false)[0];
    // [batch x num_head x seqlen, headdim] ----> [batch x seqlen, num_head, headdim]
    return varlen_fwd_permute_bhsd2bshd(fa_out, cu_seqlens_q, num_heads, max_seqlen_q);
}



/**
 * @brief FA Kernel, for sbhd layouts
 * @param main are listed below
           q [seqlen, batch_size, num_head, head_dim]
           k [seqlen, batch_size, num_head, head_dim]
           v [seqlen, batch_size, num_head, head_dim]
 * @return
           fa_output: a list of tensor, element [0] is of [seqlen, batch_size, num_head, head_dim] layouts
           Attention! Other returned results are of bhsd layouts, only output is changed by fwd_permute_bhsd2bshd
 */
std::vector<at::Tensor> fwd_sbhd(
        at::Tensor &q,                            // seqlen_q x batch_size x num_heads x head_size
        at::Tensor &k,                            // seqlen_q x batch_size x num_heads x head_size
        at::Tensor &v,                            // seqlen_q x batch_size x num_heads x head_size
        c10::optional<at::Tensor> &out_,          // seqlen_q x batch_size x num_heads x head_size
        c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_) {
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
    // [s, b, h, d] ---> [b, h, s, d]
    auto qkv_bhsd = fwd_permute_sbhd2bhsd(q, k, v);
    c10::optional<at::Tensor> q_descale_ = c10::nullopt;
    c10::optional<at::Tensor> k_descale_ = c10::nullopt;
    c10::optional<at::Tensor> v_descale_ = c10::nullopt;
    // bhsd FA kernel
    auto fa_output = hg_fwd_bhsd(
        qkv_bhsd[0],
        qkv_bhsd[1],
        qkv_bhsd[2],
        out_,
        alibi_slopes_,
        p_dropout,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen_,
        q_descale_,
        k_descale_,
        v_descale_,
        false);
    // [b, h, s, d] ---> [s, b, h x d]
    if (not fa_output.empty()) fa_output[0] = fwd_permute_bhsd2sbhd(fa_output[0]);
    // in this api call, some memory share operations can be applied to reduce hipMalloc
    return fa_output;
}


/**
 * @brief FA Kernel, for bshd layouts
 * @param main are listed below
           q [batch_size, seqlen, num_head, head_dim]
           k [batch_size, seqlen, num_head, head_dim]
           v [batch_size, seqlen, num_head, head_dim]
 * @return
           fa_output: a list of tensor, element [0] is of [batch_size, seqlen, num_head, head_dim] layouts
           Attention! Other returned results are of bhsd layouts, only output is changed by fwd_permute_bhsd2bshd
 */
std::vector<at::Tensor> fwd_bshd_with_permute(
        at::Tensor &q,                            // seqlen_q x batch_size x num_heads x head_size
        at::Tensor &k,                            // seqlen_q x batch_size x num_heads x head_size
        at::Tensor &v,                            // seqlen_q x batch_size x num_heads x head_size
        c10::optional<at::Tensor> &out_,          // seqlen_q x batch_size x num_heads x head_size
        c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_) {
    const at::cuda::HIPGuardMasqueradingAsCUDA device_guard(q.device().index());
    // [b, s, h, d] ---> [b, h, s, d]
    auto qkv_bhsd = fwd_permute_bshd2bhsd(q, k, v);
    c10::optional<at::Tensor> q_descale_ = c10::nullopt;
    c10::optional<at::Tensor> k_descale_ = c10::nullopt;
    c10::optional<at::Tensor> v_descale_ = c10::nullopt;
    // bhsd FA kernel
    auto fa_output = hg_fwd_bhsd(
        qkv_bhsd[0],
        qkv_bhsd[1],
        qkv_bhsd[2],
        out_,
        alibi_slopes_,
        p_dropout,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        gen_,
        q_descale_,
        k_descale_,
        v_descale_,
        false);
    // [b, h, s, d] ---> [b, s, h, d]
    if (not fa_output.empty()) fa_output[0] = fwd_permute_bhsd2bshd(fa_output[0]);
    return fa_output;
}

#endif // end of BUILD_FA_PERMUTE

#define PREFIX_PREFILL_PY_ARGS                                                 \
  py::arg("q") = py::none(), py::arg("k") = py::none(),                        \
  py::arg("v") = py::none(), py::arg("out_") = py::none(),                     \
  py::arg("cu_seqlens_q") = py::none(), py::arg("cu_seqlens_k") = py::none(),  \
  py::arg("seqused_k") = py::none(), py::arg("alibi_slopes_") = py::none(),    \
  py::arg("block_table") = py::none(), py::arg("max_seqlen_q") = py::none(),   \
  py::arg("max_seqlen_k") = py::none(), py::arg("p_dropout") = py::none(),     \
  py::arg("softmax_scale") = py::none(), py::arg("zero_tensors") = py::none(), \
  py::arg("is_causal") = py::none(), py::arg("window_size_left") = py::none(), \
  py::arg("window_size_right") = py::none(), py::arg("softcap") = py::none(),  \
  py::arg("return_softmax") = py::none(), py::arg("layout") = py::none(),      \
  py::arg("scales_q_") = py::none(), py::arg("scales_k_") = py::none(),        \
  py::arg("scales_v_") = py::none(), py::arg("s_aux_") = py::none(),           \
  py::arg("is_bf16_output") = py::none()

#define FWD_PY_ARGS                                                            \
  py::arg("q") = py::none(), py::arg("k") = py::none(),                        \
  py::arg("v") = py::none(), py::arg("out_") = py::none(),                     \
  py::arg("alibi_slopes_") = py::none(), py::arg("p_dropout") = py::none(),    \
  py::arg("softmax_scale") = py::none(), py::arg("is_causal") = py::none(),    \
  py::arg("window_size_left") = py::none(),                                    \
  py::arg("window_size_right") = py::none(), py::arg("softcap") = py::none(),  \
  py::arg("return_softmax") = py::none(), py::arg("gen_") = py::none(),        \
  py::arg("q_descale_") = py::none(), py::arg("k_descale_") = py::none(),      \
  py::arg("v_descale_") = py::none(), py::arg("is_bf16_output") = true

#define VARLEN_FWD_PY_ARGS                                                     \
  py::arg("q") = py::none(), py::arg("k") = py::none(),                        \
  py::arg("v") = py::none(), py::arg("out_") = py::none(),                     \
  py::arg("cu_seqlens_q") = py::none(), py::arg("cu_seqlens_k") = py::none(),  \
  py::arg("seqused_k") = py::none(), py::arg("alibi_slopes_") = py::none(),    \
  py::arg("max_seqlen_q") = py::none(), py::arg("max_seqlen_k") = py::none(),  \
  py::arg("p_dropout") = py::none(), py::arg("softmax_scale") = py::none(),    \
  py::arg("zero_tensors") = py::none(), py::arg("is_causal") = py::none(),     \
  py::arg("window_size_left") = py::none(),                                    \
  py::arg("window_size_right") = py::none(), py::arg("softcap") = py::none(),  \
  py::arg("return_softmax") = py::none(), py::arg("gen_") = py::none(),        \
  py::arg("q_descale_") = py::none(), py::arg("k_descale_") = py::none(),      \
  py::arg("v_descale_") = py::none(), py::arg("is_bf16_output") = true

PYBIND11_MODULE(flash_attn_hg_cuda, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &hg_fwd_bshd, FWD_PY_ARGS, "Forward pass");
    m.def("bwd", &hg_bwd_bshd, "Backward pass");
    m.def("hg_fwd_bshd", &hg_fwd_bshd, FWD_PY_ARGS, "Forward pass, for inputs of bshd layout and return bshd layout");
    m.def("hg_fwd_bhsd", &hg_fwd_bhsd, FWD_PY_ARGS, "Forward pass, for inputs of bhsd layout and return bhsd layout");
    m.def("fwd_bshd", &hg_fwd_bshd, FWD_PY_ARGS, "Forward pass, for inputs of bshd layout and return bshd layout");
    m.def("fwd_bhsd", &hg_fwd_bhsd, FWD_PY_ARGS, "Forward pass, for inputs of bhsd layout and return bhsd layout");
    m.def("fwd_padding_mask", &fwd_padding_mask, "Forward pass, for inputs with padding mask in bert-liked models");
    m.def("fwd_attn_mask", &fwd_attn_mask, "Forward pass, for inputs of self-defined attn mask");
    m.def("hg_bwd_bshd", &hg_bwd_bshd, "Backward pass, for inputs of bshd layout and return bshd layout");
    m.def("hg_bwd_bhsd", &hg_bwd_bhsd, "Backward pass, for inputs of bhsd layout and return bhsd layout");
    m.def("bwd_bshd", &hg_bwd_bshd, "Backward pass, for inputs of bshd layout and return bshd layout");
    m.def("bwd_bhsd", &hg_bwd_bhsd, "Backward pass, for inputs of bhsd layout and return bhsd layout");
    m.def("varlen_fwd", &hg_varlen_fwd_bshd, VARLEN_FWD_PY_ARGS, "Forward pass (variable length), for inputs of bshd layout");
    m.def("hg_varlen_fwd_bshd", &varlen_fwd_bshd_infer, VARLEN_FWD_PY_ARGS, "Forward pass (variable length), for inputs of bshd layout, only return output, preserved for vllm/sglang interface");
    m.def("varlen_fwd_bshd", &varlen_fwd_bshd_infer, VARLEN_FWD_PY_ARGS, "Forward pass (variable length), for inputs of bshd layout, only return output, preserved for vllm/sglang interface");
    m.def("varlen_fwd_bhsd", &varlen_fwd_bhsd,  "Forward pass (variable length), for inputs of bhsd layout");
    m.def("varlen_fwd_inner", &varlen_fwd, "Forward pass (variable length) base function");
    m.def("varlen_bwd", &hg_varlen_bwd_bshd, "backward pass (variable length)");
    m.def("varlen_bwd_bshd", &hg_varlen_bwd_bshd, "backward pass (variable length), for inputs of bshd layout");
    m.def("varlen_bwd_bhsd", &mha_varlen_bwd_bhsd, "backward pass (variable length), for inputs of bhsd layout");
    m.def("fwd_kvcache", &hg_fwd_kvcache_bshd, "Forward pass, with KV-cache");
    m.def("fwd_kvcache_bhsd", &mha_fwd_kvcache_bhsd, "Forward pass, with KV-cache");
    m.def("fwd_kvcache_bshd", &hg_fwd_kvcache_bshd, "Forward pass, with KV-cache");
    m.def("hg_fwd_kvcache_mla", &hg_fwd_kvcache_mla, "HG forward pass, with FlashMLA decoding");
    m.def("fwd_kvcache_mla_fp8", &fwd_kvcache_mla_decoding_fp8, "Forward pass, with FlashMLA fp8 decoding");
    m.def("flash_mla_convert_query_to_fp8", &flash_mla_convert_query_to_fp8, "Forward pass, for convert q into fp8 dtype in FlashMLA fp8 decoding");
    m.def("hg_prefix_prefill_varlen_fwd", &hg_prefix_prefill_varlen_fwd, PREFIX_PREFILL_PY_ARGS, "Forward pass, for prefix prefill attention(bshd).");
    m.def("prefix_prefill_varlen_fwd_mla", &prefix_prefill_varlen_fwd_mla, "Forward pass, for prefix prefill attention(bshd).");
    m.def("hg_prefix_decode_varlen_fwd", &hg_prefix_decode_varlen_fwd, PREFIX_PREFILL_PY_ARGS, "Forward pass, for prefix decode attention(bshd).");
#ifdef BUILD_FA_PERMUTE
    m.def("varlen_fwd_permute_bshd2bhsd", &varlen_fwd_permute_bshd2bhsd, "Forward pass (variable length), for permute layout");
    m.def("varlen_fwd_permute_bhsd2bshd", &varlen_fwd_permute_bhsd2bshd, "Forward pass (variable length), for permute layout");
    m.def("varlen_fwd_bshd_with_permute", &varlen_fwd_bshd_with_permute, "Forward pass (variable length), for inputs of bshd layout");
    m.def("fwd_permute_sbhd2bhsd", &fwd_permute_sbhd2bhsd, "Forward pass layout transformation, for inputs of sbhd -> bhsd layout");
    m.def("fwd_permute_bhsd2sbhd", &fwd_permute_bhsd2sbhd, "Forward pass layout transformation, for inputs of bhsd -> sbhd layout");
    m.def("fwd_permute_bshd2bhsd", &fwd_permute_bshd2bhsd, "Forward pass layout transformation, for inputs of bshd -> bhsd layout");
    m.def("fwd_permute_bhsd2bshd", &fwd_permute_bhsd2bshd, "Forward pass layout transformation, for inputs of bhsd -> bshd layout");
    m.def("fwd_sbhd", &fwd_sbhd, "Forward pass, for inputs of sbhd layout and return sb(hd) layout");
    m.def("fwd_bshd_with_permute", &fwd_bshd_with_permute, "Forward pass, for inputs of bshd layout and return bshd layout");
    m.def("bwd_permute_bhsd2sbhd", &bwd_permute_bhsd2sbhd, "Backward pass layout transformation, for inputs of bhsd -> sbhd layout");
    m.def("bwd_permute_bhsd2bshd", &bwd_permute_bhsd2bshd, "Backward pass layout transformation, for inputs of bhsd -> bshd layout");
    m.def("bwd_permute_sbhd2bhsd", &bwd_permute_sbhd2bhsd, "Backward pass layout transformation, for inputs of sbhd -> bhsd layout");
    m.def("bwd_permute_bshd2bhsd", &bwd_permute_bshd2bhsd, "Backward pass layout transformation, for inputs of bshd -> bhsd layout");
    m.def("permute_sbhd2bhsd", &permute_sbhd2bhsd, "Uniform layout transformation, for inputs of sbhd -> bhsd layout");
    m.def("permute_bhsd2sbhd", &permute_bhsd2sbhd, "Uniform layout transformation, for inputs of bhsd -> sbhd layout");
    m.def("permute_bhsd2bshd", &permute_bhsd2bshd, "Uniform layout transformation, for inputs of bhsd -> bshd layout");
    m.def("permute_bshd2bhsd", &permute_bshd2bhsd, "Uniform layout transformation, for inputs of bshd -> bhsd layout");
#endif // end of BUILD_FA_PERMUTE
}

#endif // else of no-def BUILD_C_INTERFACE
