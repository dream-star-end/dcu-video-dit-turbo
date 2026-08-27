// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#ifdef H3_STANDARD_FWD_ONLY
#include "../h3_flash_fwd_launch_minimal.h"
#else
#include "../flash_fwd_launch_template.h"
#endif

template<>
void run_mha_fwd_<BFloat16, 128, 128>(Flash_fwd_params &params, hipStream_t stream) {
#ifdef BUILD_FA_FWD
#ifdef H3_STANDARD_FWD_ONLY
    run_h3_flash_fwd_bf16_hdim128(params, stream);
#else
    run_mha_fwd_hdim128<BFloat16>(params, stream);
#endif
#endif
}
