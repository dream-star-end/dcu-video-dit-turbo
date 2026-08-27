// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "../flash_fwd_launch_template.h"

template<>
void run_mha_fwd_prefix_prefill_<BFloat16, 128, 128>(Flash_fwd_params &params, hipStream_t stream) {
#ifdef BUILD_FA_FWD
    run_flash_fwd_prefix_prefill<BFloat16, 128, 128>(params, stream);
#endif
}

template<>
void run_mha_fwd_prefix_prefill_<BFloat16, 192, 128>(Flash_fwd_params &params, hipStream_t stream) {
#ifdef BUILD_FA_FWD
    run_flash_fwd_prefix_prefill<BFloat16, 192, 128>(params, stream);
#endif
}

template<>
void run_mha_fwd_prefix_prefill_<BFloat16, 192, 192>(Flash_fwd_params &params, hipStream_t stream) {
#ifdef BUILD_FA_FWD
    run_flash_fwd_prefix_prefill<BFloat16, 192, 192>(params, stream);
#endif
}

template<>
void run_mha_fwd_prefix_prefill_<BFloat16, 256, 256>(Flash_fwd_params &params, hipStream_t stream) {
#ifdef BUILD_FA_FWD
    run_flash_fwd_prefix_prefill<BFloat16, 256, 256>(params, stream);
#endif
}