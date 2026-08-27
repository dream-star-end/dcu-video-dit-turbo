// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
// #include "namespace_config.h"
#include "flash_fwd_sparse_launch_template.h"

// namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_sparse_<cutlass::half_t, 128, false>(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    run_mha_fwd_sparse_hdim128<cutlass::half_t, false>(params, stream);
}

template<>
void run_mha_fwd_sparse_sla_<cutlass::half_t, 128>(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    run_mha_fwd_sparse_sla_hdim128<cutlass::half_t>(params, stream);
}

template<>
void run_mha_fwd_sparse_sla_fp8_<cutlass::float_e4m3_t, 128>(Flash_fwd_params_sparse &params, cudaStream_t stream) {
    run_mha_fwd_sparse_sla_hdim128_fp8<cutlass::float_e4m3_t>(params, stream);
}

// } // namespace FLASH_NAMESPACE