// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_mla_<cutlass::float_e4m3_t, 192, 128, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim192_hdim128_fp8<cutlass::float_e4m3_t, true>(params, stream);
}
