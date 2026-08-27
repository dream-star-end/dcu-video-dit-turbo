// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>  // For at::Generator and at::PhiloxCudaState
// #include "philox_unpack.cuh"  // For at::cuda::philox::unpack

#include <cutlass/numeric_types.h>

// #include "namespace_config.h"
// #include "hardware_info.h"
#include "flash_attnmask.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

static const bool print_param = get_env_("FLASH_ATTENTION_PRINT_PARAM");

////////////////////////////////////////////////////////////////////////////////////////////////////
// External declarations - reuse from flash_api.cpp

void set_params_fprop(Flash_fwd_params &params,
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
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
                      const float softcap,
                      bool is_bhsd = false,
                      bool seqlenq_ngroups_swapped = false,
                      const bool unpadded_lse = false,
                      int d_v = 0,
                      int d_v_rounded = 0,
                      bool is_vllm_kvcache = false);

std::tuple<at::Tensor, at::Tensor> set_params_splitkv(
    Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits, cudaDeviceProp *dprops, struct c10::TensorOptions opts);

void set_params_alibi(Flash_fwd_params &params, c10::optional<at::Tensor> &alibi_slopes_, int batch_size, int num_heads);

void set_params_dgrad(Flash_bwd_params &params,
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
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
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool deterministic,
                      const bool unpadded_lse,
                      bool is_bhsd,
                      int d_v,
                      int d_v_rounded);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Attnmask-specific parameter setup

void set_params_fprop_attnmask(Flash_fwd_params_attnmask &params,
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
                               // device pointers
                               const at::Tensor q,
                               const at::Tensor k,
                               const at::Tensor v,
                               const at::Tensor attn_mask,
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
                               const float softcap,
                               const float masked_value,
                               bool is_bhsd = false,
                               bool unpadded_lse = false) {
    // Call base class setup
    set_params_fprop(params,
                     b, seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d, cu_seqlens_k_d, seqused_k,
                     p_d, softmax_lse_d,
                     p_dropout, softmax_scale,
                     window_size_left, window_size_right,
                     softcap, is_bhsd,
                     /*seqlenq_ngroups_swapped=*/false,
                     unpadded_lse);

    // Set attnmask-specific parameters
    // Expected mask layout: [b, h, seqlen_q, seqlen_k] with K dimension contiguous
    params.mask_ptr = attn_mask.data_ptr();
    params.mask_batch_stride = attn_mask.stride(0);
    params.mask_head_stride = attn_mask.stride(1);
    params.mask_seq_q_stride = attn_mask.stride(2);
    // K dimension must be contiguous (stride = 1), enforced by caller
    params.masked_value = masked_value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Attnmask forward dispatch

void run_mha_fwd_attnmask(Flash_fwd_params_attnmask &params, cudaStream_t stream) {
    TORCH_CHECK(params.num_splits <= 1, "run_mha_fwd_attnmask does not support splitkv.");
    TORCH_CHECK(params.d == 32 || params.d == 64 || params.d == 128,
                "run_mha_fwd_attnmask only supports headdim=32, 64, or 128 for now.");

    FP16_SWITCH(!params.is_bf16, [&] {
        if (params.d <= 32) {
            constexpr static int kHeadDim = 32;
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_attnmask_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        } else if (params.d <= 64) {
            constexpr static int kHeadDim = 64;
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_attnmask_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        } else {
            constexpr static int kHeadDim = 128;
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_attnmask_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        }
    });
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Python-facing API

extern "C"
std::vector<at::Tensor>
mha_fwd_attnmask(at::Tensor &q,              // batch_size x seqlen_q x num_heads x head_size
                 const at::Tensor &k,        // batch_size x seqlen_k x num_heads_k x head_size
                 const at::Tensor &v,        // batch_size x seqlen_k x num_heads_k x head_size
                 const at::Tensor &attn_mask,// batch_size x num_heads x seqlen_q x seqlen_k (bool)
                 c10::optional<at::Tensor> &out_,
                 c10::optional<at::Tensor> &alibi_slopes_,
                 const float p_dropout,
                 const float softmax_scale,
                 bool is_causal,
                 int window_size_left,
                 int window_size_right,
                 const float softcap,
                 const float masked_value,   // Value to use when mask is false (e.g., -inf)
                 const bool return_softmax,
                 c10::optional<at::Generator> gen_,
                 bool is_bhsd = false,
                 const c10::optional<at::Tensor> &s_aux_ = c10::nullopt) {  // Attention Sinks: precomputed LSE for sink tokens

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention attnmask only supports fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v); CHECK_DEVICE(attn_mask);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    // Mask validation
    TORCH_CHECK(attn_mask.dtype() == torch::kBool || attn_mask.dtype() == torch::kUInt8,
                "attn_mask must have dtype bool or uint8");
    TORCH_CHECK(attn_mask.stride(-1) == 1,
                "attn_mask must have contiguous last dimension (seqlen_k)");

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    int seqlen_q = !is_bhsd ? sizes[1] : sizes[2];
    int num_heads = !is_bhsd ? sizes[2] : sizes[1];
    const int head_size_og = sizes[3];
    const int seqlen_k = !is_bhsd ? k.size(1) : k.size(2);
    const int num_heads_k = !is_bhsd ? k.size(2) : k.size(1);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size_og == 32 || head_size_og == 64 || head_size_og == 128,
                "FlashAttention attnmask currently only supports head dimension 32, 64, or 128");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // Validate mask shape: [b, h, seqlen_q, seqlen_k]
    CHECK_SHAPE(attn_mask, batch_size, num_heads, seqlen_q, seqlen_k);

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    // Backward correctness requires masked_value = -inf so that P=0 at masked positions
    TORCH_CHECK(std::isinf(masked_value) && masked_value < 0,
                "attnmask requires masked_value = -INFINITY for backward pass correctness");

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    if (!is_bhsd) {
        CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
        CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
        CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);
    } else {
        CHECK_SHAPE(q, batch_size, num_heads, seqlen_q, head_size_og);
        CHECK_SHAPE(k, batch_size, num_heads_k, seqlen_k, head_size_og);
        CHECK_SHAPE(v, batch_size, num_heads_k, seqlen_k, head_size_og);
    }

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        k_padded = k;
        v_padded = v;
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        if (!is_bhsd) {
            CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
        } else {
            CHECK_SHAPE(out, batch_size, num_heads, seqlen_q, head_size_og);
        }
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Internally pad mask to kernel block-aligned sequence lengths:
    // Q aligned to blockM=128, K aligned to blockN=64.
    at::Tensor attn_mask_padded = attn_mask;
    const int mask_seqlen_q_padded = round_multiple(seqlen_q, 128);
    const int mask_seqlen_k_padded = round_multiple(seqlen_k, 64);
    if (mask_seqlen_q_padded != seqlen_q || mask_seqlen_k_padded != seqlen_k) {
        attn_mask_padded = torch::zeros(
            {batch_size, num_heads, mask_seqlen_q_padded, mask_seqlen_k_padded},
            attn_mask.options());
        attn_mask_padded.narrow(/*dim=*/2, /*start=*/0, /*length=*/seqlen_q)
                        .narrow(/*dim=*/3, /*start=*/0, /*length=*/seqlen_k)
                        .copy_(attn_mask);
    }

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    at::Tensor p;
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, opts);
    }

    Flash_fwd_params_attnmask params;
    set_params_fprop_attnmask(params,
                              batch_size,
                              seqlen_q, seqlen_k,
                              seqlen_q_rounded, seqlen_k_rounded,
                              num_heads, num_heads_k,
                              head_size, head_size_rounded,
                              q_padded, k_padded, v_padded,
                              attn_mask_padded,
                              out,
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
                              masked_value,
                              is_bhsd);

    auto dprops = at::cuda::getCurrentDeviceProperties();
    at::Tensor softmax_lse_accum, out_accum;
    std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
        params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
        head_size_rounded, p_dropout, /*num_splits*/ 1, dprops, opts);

    int64_t counter_offset = params.b * params.h * 32;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if (p_dropout > 0.0) {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    // Attention Sinks: set s_aux_ptr
    if (s_aux_.has_value()) {
        auto s_aux = s_aux_.value();
        // s_aux must match Q/K/V dtype (Element type) for mixed precision
        TORCH_CHECK(s_aux.dtype() == q.dtype(),
                    "s_aux must have the same dtype as Q/K/V. Got s_aux dtype: ", s_aux.dtype(),
                    ", Q dtype: ", q.dtype());
        TORCH_CHECK(s_aux.dtype() == torch::kFloat16 || s_aux.dtype() == torch::kBFloat16,
                    "s_aux must have dtype float16 or bfloat16 (to match Q/K/V). Got: ", s_aux.dtype());
        TORCH_CHECK(num_heads <= 64, "Attention Sinks only supports up to 64 heads (shared memory limit), got ", num_heads);
        CHECK_DEVICE(s_aux);
        CHECK_SHAPE(s_aux, num_heads);
        CHECK_CONTIGUOUS(s_aux);
        params.s_aux_ptr = s_aux.data_ptr();
    } else {
        params.s_aux_ptr = nullptr;
    }

    if (seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd_attnmask(params, stream);
    } else {
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    at::Tensor out_padded = out;
    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Variable-length version

extern "C"
std::vector<at::Tensor>
mha_varlen_fwd_attnmask(at::Tensor &q,              // total_q x num_heads x head_size
                        const at::Tensor &k,        // total_k x num_heads_k x head_size
                        const at::Tensor &v,        // total_k x num_heads_k x head_size
                        const at::Tensor &attn_mask,// batch_size x num_heads x max_seqlen_q x max_seqlen_k (bool)
                        c10::optional<at::Tensor> &out_,
                        const at::Tensor &cu_seqlens_q,  // b+1
                        const at::Tensor &cu_seqlens_k,  // b+1
                        c10::optional<at::Tensor> &seqused_k,
                        c10::optional<at::Tensor> &alibi_slopes_,
                        int max_seqlen_q,
                        const int max_seqlen_k,
                        const float p_dropout,
                        const float softmax_scale,
                        const bool zero_tensors,
                        bool is_causal,
                        int window_size_left,
                        int window_size_right,
                        const float softcap,
                        const float masked_value,
                        const bool return_softmax,
                        c10::optional<at::Generator> gen_,
                        const c10::optional<at::Tensor> &s_aux_ = c10::nullopt) {  // Attention Sinks: precomputed LSE for sink tokens

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention attnmask only supports fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v); CHECK_DEVICE(attn_mask);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    // Mask validation
    TORCH_CHECK(attn_mask.dtype() == torch::kBool || attn_mask.dtype() == torch::kUInt8,
                "attn_mask must have dtype bool or uint8");
    TORCH_CHECK(attn_mask.stride(-1) == 1,
                "attn_mask must have contiguous last dimension (seqlen_k)");

    const auto sizes = q.sizes();
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int total_q = sizes[0];
    int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size_og == 32 || head_size_og == 64 || head_size_og == 128,
                "FlashAttention attnmask currently only supports head dimension 32, 64, or 128");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // Validate mask shape: [b, h, max_seqlen_q, max_seqlen_k]
    CHECK_SHAPE(attn_mask, batch_size, num_heads, max_seqlen_q, max_seqlen_k);

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    // Backward correctness requires masked_value = -inf so that P=0 at masked positions
    TORCH_CHECK(std::isinf(masked_value) && masked_value < 0,
                "attnmask requires masked_value = -INFINITY for backward pass correctness");

    if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, total_q, num_heads, head_size_og);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    if (seqused_k.has_value()) {
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        k_padded = k;
        v_padded = v;
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, total_q, num_heads, head_size_og);
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    at::Tensor attn_mask_padded = attn_mask;
    const int mask_seqlen_q_padded = round_multiple(max_seqlen_q, 128);
    const int mask_seqlen_k_padded = round_multiple(max_seqlen_k, 64);
    if (mask_seqlen_q_padded != max_seqlen_q || mask_seqlen_k_padded != max_seqlen_k) {
        attn_mask_padded = torch::zeros(
            {batch_size, num_heads, mask_seqlen_q_padded, mask_seqlen_k_padded},
            attn_mask.options());
        attn_mask_padded.narrow(/*dim=*/2, /*start=*/0, /*length=*/max_seqlen_q)
                        .narrow(/*dim=*/3, /*start=*/0, /*length=*/max_seqlen_k)
                        .copy_(attn_mask);
    }

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

    at::Tensor p;
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, opts);
    }

    if (zero_tensors) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) { p.zero_(); }
    }

    Flash_fwd_params_attnmask params;
    set_params_fprop_attnmask(params,
                              batch_size,
                              max_seqlen_q, max_seqlen_k,
                              seqlen_q_rounded, seqlen_k_rounded,
                              num_heads, num_heads_k,
                              head_size, head_size_rounded,
                              q_padded, k_padded, v_padded,
                              attn_mask_padded,
                              out,
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
                              masked_value,
                              /*is_bhsd=*/false,
                              /*unpadded_lse=*/true);
    params.total_q = total_q;

    auto dprops = at::cuda::getCurrentDeviceProperties();
    at::Tensor softmax_lse_accum, out_accum;
    std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
        params, batch_size, num_heads, head_size, max_seqlen_k, max_seqlen_q,
        head_size_rounded, p_dropout, /*num_splits*/ 1, dprops, opts);

    int64_t counter_offset = params.b * params.h * 32;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if (p_dropout > 0.0) {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    // Attention Sinks: set s_aux_ptr
    if (s_aux_.has_value()) {
        auto s_aux = s_aux_.value();
        // s_aux must match Q/K/V dtype (Element type) for mixed precision
        TORCH_CHECK(s_aux.dtype() == q.dtype(),
                    "s_aux must have the same dtype as Q/K/V. Got s_aux dtype: ", s_aux.dtype(),
                    ", Q dtype: ", q.dtype());
        TORCH_CHECK(s_aux.dtype() == torch::kFloat16 || s_aux.dtype() == torch::kBFloat16,
                    "s_aux must have dtype float16 or bfloat16 (to match Q/K/V). Got: ", s_aux.dtype());
        TORCH_CHECK(num_heads <= 64,
                    "Attention Sinks only supports up to 64 heads (shared memory limit), got ", num_heads);
        CHECK_DEVICE(s_aux);
        CHECK_SHAPE(s_aux, num_heads);
        CHECK_CONTIGUOUS(s_aux);
        params.s_aux_ptr = s_aux.data_ptr();
    } else {
        params.s_aux_ptr = nullptr;
    }

    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd_attnmask(params, stream);
    } else {
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    at::Tensor out_padded = out;
    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Backward pass API
////////////////////////////////////////////////////////////////////////////////////////////////////

// Attnmask-specific backward parameter setup
void set_params_dgrad_attnmask(Flash_bwd_params_attnmask &params,
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
                               // device pointers
                               const at::Tensor q,
                               const at::Tensor k,
                               const at::Tensor v,
                               const at::Tensor attn_mask,
                               const at::Tensor out,
                               const at::Tensor dout,
                               at::Tensor dq,
                               at::Tensor dk,
                               at::Tensor dv,
                               void *cu_seqlens_q_d,
                               void *cu_seqlens_k_d,
                               void *dq_accum_d,
                               void *dk_accum_d,
                               void *dv_accum_d,
                               void *softmax_lse_d,
                               void *dsoftmax_sum_d,
                               float p_dropout,
                               float softmax_scale,
                               int window_size_left,
                               int window_size_right,
                               const float softcap,
                               bool deterministic,
                               bool unpadded_lse = false,
                               bool is_bhsd = false,
                               int d_v = 0,
                               int d_v_rounded = 0) {
    // Call base class setup
    set_params_dgrad(params,
                     b, seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     h, h_k, d, d_rounded,
                     q, k, v, out, dout,
                     dq, dk, dv,
                     cu_seqlens_q_d, cu_seqlens_k_d,
                     dq_accum_d, dk_accum_d, dv_accum_d,
                     softmax_lse_d, dsoftmax_sum_d,
                     p_dropout, softmax_scale,
                     window_size_left, window_size_right,
                     softcap, deterministic,
                     unpadded_lse, is_bhsd,
                     d_v, d_v_rounded);

    // Set attnmask-specific parameters
    // Expected mask layout: [b, h, seqlen_q, seqlen_k] with K dimension contiguous
    params.mask_ptr = attn_mask.data_ptr();
    params.mask_batch_stride = attn_mask.stride(0);
    params.mask_head_stride = attn_mask.stride(1);
    params.mask_seq_q_stride = attn_mask.stride(2);
    // K dimension must be contiguous (stride = 1), enforced by caller

    // Forward pass enforces masked_value = -INFINITY, so we can hardcode it here
    params.masked_value = -INFINITY;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Attnmask backward dispatch

void run_mha_bwd_attnmask(Flash_bwd_params_attnmask &params, cudaStream_t stream) {
    TORCH_CHECK(params.d == 64 || params.d == 128, "run_mha_bwd_attnmask only supports headdim=64 or 128.");

    FP16_SWITCH(!params.is_bf16, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            if (params.d <= 64) {
                constexpr static int kHeadDim = 64;
                run_mha_bwd_attnmask_<elem_type, kHeadDim, Is_causal>(params, stream);
            } else {
                constexpr static int kHeadDim = 128;
                run_mha_bwd_attnmask_<elem_type, kHeadDim, Is_causal>(params, stream);
            }
        });
    });
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Python-facing backward API

extern "C"
std::vector<at::Tensor>
mha_bwd_attnmask(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads x head_size_og
                 const at::Tensor &q,     // batch_size x seqlen_q x num_heads x head_size
                 const at::Tensor &k,     // batch_size x seqlen_k x num_heads_k x head_size
                 const at::Tensor &v,     // batch_size x seqlen_k x num_heads_k x head_size
                 const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
                 const at::Tensor &softmax_lse,  // b x h x seqlen_q
                 const at::Tensor &attn_mask,    // batch_size x num_heads x seqlen_q x seqlen_k (bool)
                 c10::optional<at::Tensor> &dq_, // batch_size x seqlen_q x num_heads x head_size
                 c10::optional<at::Tensor> &dk_, // batch_size x seqlen_k x num_heads_k x head_size
                 c10::optional<at::Tensor> &dv_, // batch_size x seqlen_k x num_heads_k x head_size
                 c10::optional<at::Tensor> &alibi_slopes_,
                 const float p_dropout,
                 const float softmax_scale,
                 const bool is_causal,
                 int window_size_left,
                 int window_size_right,
                 const float softcap,
                 const bool deterministic,
                 c10::optional<at::Generator> gen_,
                 c10::optional<at::Tensor> &rng_state,
                 bool is_bhsd = false) {

    if (is_causal) { window_size_right = 0; }
    auto dprops = at::cuda::getCurrentDeviceProperties();

    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention attnmask backward only supports fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(attn_mask);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    // Mask validation
    TORCH_CHECK(attn_mask.dtype() == torch::kBool || attn_mask.dtype() == torch::kUInt8,
                "attn_mask must have dtype bool or uint8");
    TORCH_CHECK(attn_mask.stride(-1) == 1,
                "attn_mask must have contiguous last dimension (seqlen_k)");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen_q = !is_bhsd ? sizes[1] : sizes[2];
    const int num_heads = !is_bhsd ? sizes[2] : sizes[1];
    const int head_size_og = dout.size(3);
    const int head_size = sizes[3];
    const int head_size_value = v.sizes()[3];
    const int seqlen_k = !is_bhsd ? k.size(1) : k.size(2);
    const int num_heads_k = !is_bhsd ? k.size(2) : k.size(1);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size == 64 || head_size == 128, "FlashAttention attnmask backward only supports head dimension 64 or 128");
    TORCH_CHECK(head_size_value <= 256, "FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // Validate mask shape: [b, h, seqlen_q, seqlen_k]
    CHECK_SHAPE(attn_mask, batch_size, num_heads, seqlen_q, seqlen_k);

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int head_size_value_rounded = round_multiple(head_size_value, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    at::Tensor attn_mask_padded = attn_mask;
    const int mask_seqlen_q_padded = round_multiple(seqlen_q, 128);
    const int mask_seqlen_k_padded = round_multiple(seqlen_k, 64);
    if (mask_seqlen_q_padded != seqlen_q || mask_seqlen_k_padded != seqlen_k) {
        attn_mask_padded = torch::zeros(
            {batch_size, num_heads, mask_seqlen_q_padded, mask_seqlen_k_padded},
            attn_mask.options());
        attn_mask_padded.narrow(/*dim=*/2, /*start=*/0, /*length=*/seqlen_q)
                        .narrow(/*dim=*/3, /*start=*/0, /*length=*/seqlen_k)
                        .copy_(attn_mask);
    }

    TORCH_CHECK(head_size_value == round_multiple(head_size_og, 8), "head_size_value must be head_size_og rounded to a multiple of 8");
    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    if (!is_bhsd) {
        CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
        CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
        CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_value);
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_value);
        CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_og);
    } else {
        CHECK_SHAPE(q, batch_size, num_heads, seqlen_q, head_size);
        CHECK_SHAPE(k, batch_size, num_heads_k, seqlen_k, head_size);
        CHECK_SHAPE(v, batch_size, num_heads_k, seqlen_k, head_size_value);
        CHECK_SHAPE(out, batch_size, num_heads, seqlen_q, head_size_value);
        CHECK_SHAPE(dout, batch_size, num_heads, seqlen_q, head_size_og);
    }

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        if (!is_bhsd) {
            CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
        } else {
            CHECK_SHAPE(dq, batch_size, num_heads, seqlen_q, head_size);
        }
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        if (!is_bhsd) {
            CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
        } else {
            CHECK_SHAPE(dk, batch_size, num_heads_k, seqlen_k, head_size);
        }
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        if (!is_bhsd) {
            CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size_value);
        } else {
            CHECK_SHAPE(dv, batch_size, num_heads_k, seqlen_k, head_size_value);
        }
    } else {
        dv = torch::empty_like(v);
    }

    at::Tensor dout_padded;
    if (head_size_og % 8 != 0) {
        dout_padded = torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        dout_padded = dout;
    }

    // For simplicity, set loop = false (no dq_accum)
    bool loop = false;

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = !is_bhsd ? torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts)
            : torch::empty({batch_size, num_heads, seqlen_k, head_size}, opts);
        dv_expanded = !is_bhsd ? torch::empty({batch_size, seqlen_k, num_heads, head_size_value}, opts)
            : torch::empty({batch_size, num_heads, seqlen_k, head_size_value}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    Flash_bwd_params_attnmask params;

    set_params_dgrad_attnmask(params,
                              batch_size,
                              seqlen_q, seqlen_k,
                              seqlen_q_rounded, seqlen_k_rounded,
                              num_heads, num_heads_k,
                              head_size, head_size_rounded,
                              q, k, v, attn_mask_padded, out,
                              dout_padded, dq, dk_expanded, dv_expanded,
                              nullptr,  // cu_seqlens_q
                              nullptr,  // cu_seqlens_k
                              loop ? dq_accum.data_ptr() : nullptr,
                              nullptr,  // dk_accum
                              nullptr,  // dv_accum
                              softmax_lse.data_ptr(),
                              softmax_d.data_ptr(),
                              p_dropout,
                              softmax_scale,
                              window_size_left,
                              window_size_right,
                              softcap,
                              deterministic,
                              /*unpadded_lse=*/false,
                              is_bhsd,
                              head_size_value,
                              head_size_value_rounded);
    params.dq_accum_split_stride = 0;

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    int64_t counter_offset = params.b * params.h * 32;

    if (rng_state.has_value()) {
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
    } else if (is_dropout) {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
        auto seeds = at::cuda::philox::unpack(params.philox_args);
        params.rng_state[0] = std::get<0>(seeds);
        params.rng_state[1] = std::get<1>(seeds);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (seqlen_q > 0) {
        run_mha_bwd_attnmask(params, stream);
    } else {
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        if (!is_bhsd) {
            at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
            at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size_value}), {3});
        } else {
            at::sum_out(dk, at::reshape(dk_expanded, {batch_size, num_heads_k, num_heads / num_heads_k, seqlen_k, head_size}), {2});
            at::sum_out(dv, at::reshape(dv_expanded, {batch_size, num_heads_k, num_heads / num_heads_k, seqlen_k, head_size_value}), {2});
        }
    }
    if (head_size_og % 8 != 0) {
        dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dk = dk.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dv = dv.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_value)});
    }

    return { dq, dk, dv, softmax_d };
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Variable-length backward version

extern "C"
std::vector<at::Tensor>
mha_varlen_bwd_attnmask(const at::Tensor &dout,   // total_q x num_heads x head_size_og
                        const at::Tensor &q,      // total_q x num_heads x head_size
                        const at::Tensor &k,      // total_k x num_heads_k x head_size
                        const at::Tensor &v,      // total_k x num_heads_k x head_size
                        const at::Tensor &out,    // total_q x num_heads x head_size
                        const at::Tensor &softmax_lse,  // h x total_q
                        const at::Tensor &attn_mask,    // batch_size x num_heads x max_seqlen_q x max_seqlen_k (bool)
                        c10::optional<at::Tensor> &dq_, // total_q x num_heads x head_size
                        c10::optional<at::Tensor> &dk_, // total_k x num_heads_k x head_size
                        c10::optional<at::Tensor> &dv_, // total_k x num_heads_k x head_size
                        const at::Tensor &cu_seqlens_q, // b+1
                        const at::Tensor &cu_seqlens_k, // b+1
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
                        const bool deterministic,
                        c10::optional<at::Generator> gen_,
                        c10::optional<at::Tensor> &rng_state) {

    if (is_causal) { window_size_right = 0; }
    auto dprops = at::cuda::getCurrentDeviceProperties();

    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention attnmask backward only supports fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);
    CHECK_DEVICE(attn_mask);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    // Mask validation
    TORCH_CHECK(attn_mask.dtype() == torch::kBool || attn_mask.dtype() == torch::kUInt8,
                "attn_mask must have dtype bool or uint8");
    TORCH_CHECK(attn_mask.stride(-1) == 1,
                "attn_mask must have contiguous last dimension (seqlen_k)");

    const auto sizes = q.sizes();

    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size_og = dout.size(2);
    const int head_size = sizes[2];
    const int head_size_value = v.size(2);
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_value % 8 == 0, "head_size_value should be a multiple of 8");
    TORCH_CHECK(head_size == 64 || head_size == 128, "FlashAttention attnmask backward only supports head dimension 64 or 128");
    TORCH_CHECK(head_size_value <= 256, "FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    // Validate mask shape: [b, h, max_seqlen_q, max_seqlen_k]
    CHECK_SHAPE(attn_mask, batch_size, num_heads, max_seqlen_q, max_seqlen_k);

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int head_size_value_rounded = round_multiple(head_size_value, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    at::Tensor attn_mask_padded = attn_mask;
    const int mask_seqlen_q_padded = round_multiple(max_seqlen_q, 128);
    const int mask_seqlen_k_padded = round_multiple(max_seqlen_k, 64);
    if (mask_seqlen_q_padded != max_seqlen_q || mask_seqlen_k_padded != max_seqlen_k) {
        attn_mask_padded = torch::zeros(
            {batch_size, num_heads, mask_seqlen_q_padded, mask_seqlen_k_padded},
            attn_mask.options());
        attn_mask_padded.narrow(/*dim=*/2, /*start=*/0, /*length=*/max_seqlen_q)
                        .narrow(/*dim=*/3, /*start=*/0, /*length=*/max_seqlen_k)
                        .copy_(attn_mask);
    }

    TORCH_CHECK(head_size_value == round_multiple(head_size_og, 8), "head_size_value must be head_size_og rounded to a multiple of 8");

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_value);
    CHECK_SHAPE(out, total_q, num_heads, head_size_value);
    CHECK_SHAPE(dout, total_q, num_heads, head_size_og);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, total_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, total_k, num_heads_k, head_size_value);
    } else {
        dv = torch::empty_like(v);
    }

    at::Tensor dout_padded;
    if (head_size_og % 8 != 0) {
        dout_padded = torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        dout_padded = dout;
    }

    // For simplicity, set loop = false (no dq_accum)
    bool loop = false;

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    auto softmax_d = torch::empty({num_heads, total_q + 128 * batch_size}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = torch::empty({total_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({total_k, num_heads, head_size_value}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    if (zero_tensors) {
        dq.zero_();
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    Flash_bwd_params_attnmask params;

    set_params_dgrad_attnmask(params,
                              batch_size,
                              max_seqlen_q, max_seqlen_k,
                              seqlen_q_rounded, seqlen_k_rounded,
                              num_heads, num_heads_k,
                              head_size, head_size_rounded,
                              q, k, v, attn_mask_padded, out,
                              dout_padded, dq, dk_expanded, dv_expanded,
                              cu_seqlens_q.data_ptr(),
                              cu_seqlens_k.data_ptr(),
                              loop ? dq_accum.data_ptr() : nullptr,
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
                              /*unpadded_lse=*/true,
                              /*is_bhsd=*/false,
                              head_size_value,
                              head_size_value_rounded);
    params.dq_accum_split_stride = 0;
    params.total_q = total_q;

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    int64_t counter_offset = params.b * params.h * 32;

    if (rng_state.has_value()) {
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
    } else if (is_dropout) {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
        auto seeds = at::cuda::philox::unpack(params.philox_args);
        params.rng_state[0] = std::get<0>(seeds);
        params.rng_state[1] = std::get<1>(seeds);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (max_seqlen_q > 0) {
        run_mha_bwd_attnmask(params, stream);
    } else {
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
        at::sum_out(dv, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size_value}), {2});
    }
    if (head_size_og % 8 != 0) {
        dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dk = dk.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dv = dv.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_value)});
    }

    return { dq, dk, dv, softmax_d };
}
