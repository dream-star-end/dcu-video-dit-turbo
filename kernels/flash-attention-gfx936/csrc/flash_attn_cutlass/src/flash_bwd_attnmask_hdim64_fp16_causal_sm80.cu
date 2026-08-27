// Copyright (c) 2026, Attnmask extension.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_bwd_attnmask_launch_template.h"

template void run_mha_bwd_attnmask_<cutlass::half_t, 64, true>(
    Flash_bwd_params_attnmask &params, cudaStream_t stream);
