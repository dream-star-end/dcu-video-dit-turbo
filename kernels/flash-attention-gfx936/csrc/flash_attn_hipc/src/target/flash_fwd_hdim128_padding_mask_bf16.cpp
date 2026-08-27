// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "../flash_fwd_launch_template.h"

template<>
void run_mha_fwd_padding_mask_<BFloat16, 128, 128>(Flash_fwd_params &params, hipStream_t stream) {
#ifdef BUILD_FA_FWD
    run_flash_fwd_padding_mask<BFloat16, 128>(params, stream);
#endif
}

template<>
void run_mha_fwd_padding_mask_<BFloat16, 64, 64>(Flash_fwd_params &params, hipStream_t stream) {
#ifdef BUILD_FA_FWD
    run_flash_fwd_padding_mask<BFloat16, 64>(params, stream);
#endif
}