// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "../flash_fwd_launch_template.h"

#if defined(BUILD_FA_FWD) && !defined(HEADDIM_128_ONLY)
template<>
void run_mha_fwd_<BFloat16, 192, 192>(Flash_fwd_params &params, hipStream_t stream) {
    run_mha_fwd_hdim192<BFloat16>(params, stream);
}
#endif