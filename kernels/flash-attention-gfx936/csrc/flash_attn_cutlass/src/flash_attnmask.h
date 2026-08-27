#pragma once

#include "flash.h"

// Parameter struct for attention mask forward path, isolated from the main
// Flash_fwd_params to avoid polluting the common kernel interface.
struct Flash_fwd_params_attnmask : public Flash_fwd_params {
    // Attention mask pointer and strides.
    // Expected layout: [b, h, seqlen_q, seqlen_k], with K dim contiguous
    // (K stride == 1). Only the Q stride is configurable here.
    void * __restrict__ mask_ptr;
    index_t mask_batch_stride;
    index_t mask_head_stride;
    index_t mask_seq_q_stride;

    // Value to write when mask is false.
    float masked_value;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Forward entry point for attention with explicit mask.
template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_attnmask_(Flash_fwd_params_attnmask &params, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////

// Parameter struct for attention mask backward path.
struct Flash_bwd_params_attnmask : public Flash_bwd_params {
    // Attention mask pointer and strides.
    // Expected layout: [b, h, seqlen_q, seqlen_k], with K dim contiguous
    // (K stride == 1). Only the Q stride is configurable here.
    void * __restrict__ mask_ptr;
    index_t mask_batch_stride;
    index_t mask_head_stride;
    index_t mask_seq_q_stride;

    // Value used when mask is false (typically -INFINITY in forward).
    // In backward, positions where mask is false have P=0, so dS should also be 0.
    float masked_value;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Backward entry point for attention with explicit mask.
template<typename T, int Headdim, bool Is_causal>
void run_mha_bwd_attnmask_(Flash_bwd_params_attnmask &params, cudaStream_t stream);
