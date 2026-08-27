// Copyright (c) 2025, Xin Zhou.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "../flash_fwd_launch_template.h"

template<>
void run_fp8_mha_fwd_prefix_prefill_<Float16, 256, 256>(Flash_fwd_params &params, hipStream_t stream) {
#ifdef BUILD_FA_FWD
    run_fp8_flash_fwd_prefix_prefill<Float16, 256, 256>(params, stream);
#endif
}
