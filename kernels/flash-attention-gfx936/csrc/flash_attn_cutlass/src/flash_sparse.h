#pragma once

#include "flash.h"

// namespace FLASH_NAMESPACE {

struct Flash_fwd_params_sparse : public Flash_fwd_params {
    // For sparse attention
    const int* block_count;
    const int* block_offset;
    const int* column_count;
    const int* column_index;
    int NUM_ROWS;
    int NNZ_S;
    int NNZ_V;

    // Dynamic PV skip optimization parameters
    float pv_threshold;        // Threshold for skipping P@V computation (default: 50.0, matching SpargeAttn)
    bool enable_dynamic_skip;  // Whether to enable dynamic skip (default: true)
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_sparse_(Flash_fwd_params_sparse &params, cudaStream_t stream);
template<typename T, int Headdim> void run_mha_fwd_sparse_sla_(Flash_fwd_params_sparse &params, cudaStream_t stream);
template<typename T, int Headdim> void run_mha_fwd_sparse_sla_fp8_(Flash_fwd_params_sparse &params, cudaStream_t stream);

// }  // namespace FLASH_NAMESPACE