/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

//#include "nvfuser_resources/PhiloxCudaStateRaw.h"
// #include "ATen/hip/detail/PhiloxCudaStateRaw.cuh"

#include "hip/hip_runtime.h"
#include <vector>
// #ifdef OLD_GENERATOR_PATH
// #include <ATen/CUDAGeneratorImpl.h>
// #else
// #include <ATen/cuda/CUDAGeneratorImpl.h>
// #endif

// #include <ATen/cuda/CUDAGraphsUtils.cuh> // For at::cuda::philox::unpack

constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    int q_batch_stride;
    int k_batch_stride;
    int v_batch_stride;
    int q_row_stride;
    int k_row_stride;
    int v_row_stride;
    int q_head_stride;
    int k_head_stride;
    int v_head_stride;
    int v_dim_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,

    int cu_count; // added by liuch, may be useful to balance workload
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

#ifdef DEBUGING
    void *__restrict__ qk_ptr;
    void *__restrict__ qk_softmax_ptr;
#endif

    // The stride between rows of O.
    int o_batch_stride;
    int o_row_stride;
    int o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // Attention sink values, one scalar per original query head.
    // s_aux_type: 0 none, 1 fp32, 2 fp16, 3 bf16.
    void * __restrict__ s_aux_ptr;
    int s_aux_type;

    // For FP8 scaling
    float * __restrict__ q_descale_ptr;
    float * __restrict__ k_descale_ptr;
    float * __restrict__ v_descale_ptr;
    int q_descale_batch_stride;
    int q_descale_head_stride;
    int k_descale_batch_stride;
    int k_descale_head_stride;
    int v_descale_batch_stride;
    int v_descale_head_stride;

    // The pointer of scales_q and scales_k for int8
    void *__restrict__ scales_q_ptr;
    void *__restrict__ scales_k_ptr;
    void *__restrict__ scales_v_ptr;
    int total_scale_q;

    // The pointers for scores_sum/scores_max
    float * __restrict__ scores_sum_ptr;
    float * __restrict__ scores_max_ptr;

    // The pointer of quantilized k
    void *__restrict__ q_quant_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, varlen_proj_qkv_head;
    int total_q, total_k, total_knew;
    int b_k;  // When having KV cache and with cache_batch_idx, K & V might have larger batch size than Q
    int d_value, d_value_rounded;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ attn_mask;
    int * __restrict__ leftpad_k;
    int * __restrict__ padding_mask;

    // If provided, the actual length of each q/k sequence.
    int *__restrict__ seqused_q;
    int *__restrict__ seqused_k;

    // The stride between rows of Oaccum.
    int oaccum_split_stride;
    int oaccum_batch_stride;
    int oaccum_row_stride;
    int oaccum_head_stride;

    // The stride between rows of LSEaccum.
    int lseaccum_split_stride;
    int lseaccum_batch_stride;
    int lseaccum_head_stride;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    int knew_batch_stride;
    int vnew_batch_stride;
    int knew_row_stride;
    int vnew_row_stride;
    int knew_head_stride;
    int vnew_head_stride;

    void *__restrict__ qv_ptr;
    int qv_batch_stride;
    int qv_row_stride;
    int qv_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // The indices to index into the KV cache.
    int *__restrict__ cache_batch_idx;

    // Paged KV cache
    int * __restrict__ block_table;
    int block_table_batch_stride;
    int page_block_size;
    int num_pages;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Local window size
    int window_size_left, window_size_right;
    int sink_token_length;
    float softcap;

    // Random state.
    // at::PhiloxCudaState philox_args;
    unsigned long long rand_seed;
    unsigned long long rand_offset;
    uint32_t *dropout_debug_count;

    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t * rng_state;

    bool is_bf16;
    bool is_fp32;
    bool is_e4m3;
    bool is_int8;
    bool is_causal;

    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative;

    bool is_local;

    bool is_rotary_interleaved;

    int num_splits;  // For split-KV version
    int partition_size;
    bool pack_gqa;

    int * __restrict__ tile_count_semaphore;

    int arch;

    // Alibi
    void *alibi_slopes_ptr;
    int alibi_slopes_batch_stride;

    bool unpadded_lse; // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b, nheads, seqlen_q].
    bool seqlenq_ngroups_swapped;  // q has been transposed from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d).

    int layout; // 0: bhsd, 1: bshd, 2: sbhd
    int qkvheaddim_compute, qkvheaddim_tail_tile16; // for special headdim, like 72, 80 in gfx938
    int mtp;
    int ngroups; // for pa/flashmla gqa regroups optimization
    bool splitkv_use_fp32_as_accum;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {

    // The dO and dQKV matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // To accumulate dQ
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

#ifdef DEBUGING
    void *__restrict__ kq_ptr;
    void *__restrict__ s_ptr;
    void *__restrict__ dp_ptr;
    void *__restrict__ ds_ptr;
#endif
    // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
    // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
    // dv_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    int do_batch_stride;
    int do_row_stride;
    int do_head_stride;
    int dq_batch_stride;
    int dk_batch_stride;
    int dv_batch_stride;
    int dq_row_stride;
    int dk_row_stride;
    int dv_row_stride;
    int dq_head_stride;
    int dk_head_stride;
    int dv_head_stride;

    // The pointer to the softmax d sum.
    void *__restrict__ dsoftmax_sum;
    void *__restrict__ softmax_lse_log2_ptr;

    int *__restrict__ dq_semaphore;
    int *__restrict__ dk_semaphore;
    int *__restrict__ dv_semaphore;

    bool deterministic;
    int dq_accum_split_stride;
    int se_balance_cnt;
};




// almostly aligned to deepseek official params
struct Flash_fwd_mla_params {
    using index_t = int; //  int64_t;

    int *__restrict__ cu_seqlens_q;
    int *__restrict__ cu_seqlens_k;
    int *__restrict__ cu_seqlens_k_new;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ o_ptr;

    void *__restrict__ qv_ptr;

    int *__restrict__ block_table;

    void *__restrict__ oaccum_ptr;
    float *__restrict__ scores_max_ptr;
    float *__restrict__ scores_sum_ptr;
    float *__restrict__ softmax_lse_ptr;

    int b, seqlen_q, seqlen_k, d, d_v;
    int h, h_k, h_h_k_ratio, ngroups, total_q;
    float scale_softmax, scale_softmax_log2;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t o_batch_stride;
    index_t qv_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t o_row_stride;
    index_t qv_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t o_head_stride;
    index_t qv_head_stride;

    index_t block_table_batch_stride;
    int page_block_size;

    int num_splits, partition_size;

    int layout;
    int mtp;
    int q_blocks, total_blocks, cu_count;
    bool is_bf16;
    bool is_e4m3;
    bool is_int8;
    bool is_causal;
    bool seqlenq_ngroups_swapped;
    bool is_seqlens_k_cumulative;
    bool splitkv_use_fp32_as_accum;

    // not used params
    float *__restrict__ scales_q_ptr;
    float *__restrict__ scales_k_ptr;
    int *__restrict__ leftpad_k;
    int *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int num_sm_parts;
};


struct Flash_fwd_mla_reduce_params {
    // pointers, 16 dword aligned
    float* softmax_lse_ptr;
    void* oaccum_ptr;
    void* o_ptr;
    int32_t* cu_seqlens_k;
    // intermiate variables, within 16 dword
    int num_splits;
    int partition_size;
    int h;
    int seqlen_q;
    int layout;
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim, int HeaddimV> void run_mha_fwd_(Flash_fwd_params &params, hipStream_t stream);

template<typename T, int Headdim, int HeaddimV> void run_fp8_mha_fwd_(Flash_fwd_params &params, hipStream_t stream);

template<typename T, int Headdim, int HeaddimV> void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, hipStream_t stream);

template<typename T, int Headdim, int HeaddimV> void run_int8_fwd_splitkv_dispatch(Flash_fwd_params &params, hipStream_t stream);

template<typename T, int Headdim, int HeaddimV> void run_mla_fwd_splitkv_dispatch(Flash_fwd_mla_params &params, hipStream_t stream);

template<typename T, int Headdim, int HeaddimV> void run_fp8_mla_fwd_splitkv_dispatch(Flash_fwd_mla_params &params, hipStream_t stream);

template<typename T, int Headdim, int HeaddimV> void run_fp8_mla_convert_q_to_fp8_dispatch(Flash_fwd_mla_params &params, hipStream_t stream);

template<typename T, int Headdim, int HeaddimV> void run_mha_bwd_(Flash_bwd_params &params, hipStream_t stream, const bool configure);

template<typename T, int Headdim, int HeaddimV> void run_mha_fwd_padding_mask_(Flash_fwd_params &params, hipStream_t stream);

template<typename T, int Headdim, int HeaddimV> void run_mha_fwd_attn_mask_(Flash_fwd_params &params, hipStream_t stream);

template<typename T, int Headdim, int HeaddimV> void run_mha_fwd_prefix_prefill_(Flash_fwd_params &params, hipStream_t stream);

template<typename T, int Headdim, int HeaddimV> void run_int8_mha_fwd_prefix_prefill_(Flash_fwd_params &params, hipStream_t stream);

template<typename T, int Headdim, int HeaddimV> void run_fp8_mha_fwd_prefix_prefill_(Flash_fwd_params &params, hipStream_t stream);

template<typename T, int Headdim, int HeaddimV> void run_mla_fwd_prefix_prefill_dispatch_(Flash_fwd_mla_params &params, hipStream_t stream);

template<typename T, int Headdim, int HeaddimV> void run_mla_fwd_dispatch(Flash_fwd_mla_params &params, hipStream_t stream);

// C interface
void run_mha_fwd(Flash_fwd_params &params, hipStream_t stream, bool force_split_kernel);
void run_mha_bwd(Flash_bwd_params &params, hipStream_t stream, const bool configure);
void run_mha_fwd_kvcache(Flash_fwd_params &params, hipStream_t stream, bool force_split_kernel);
void run_int8_fwd_kvcache(Flash_fwd_params &params, hipStream_t stream, bool force_split_kernel);
void run_fwd_flashmla(Flash_fwd_mla_params &params, hipStream_t stream, bool force_split_kernel);
void run_fwd_prefix_prefill_mla(Flash_fwd_mla_params &params, hipStream_t stream);
