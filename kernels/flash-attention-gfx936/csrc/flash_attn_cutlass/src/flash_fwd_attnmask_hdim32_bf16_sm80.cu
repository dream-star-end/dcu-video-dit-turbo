// Copyright (c) 2026, Attnmask extension.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_attnmask_launch_template.h"

template void run_mha_fwd_attnmask_<cutlass::bfloat16_t, 32, false>(
    Flash_fwd_params_attnmask &params, cudaStream_t stream);
