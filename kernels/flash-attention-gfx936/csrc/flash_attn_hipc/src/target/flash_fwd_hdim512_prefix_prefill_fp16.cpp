// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "../flash_fwd_launch_template.h"

template<>
void run_mha_fwd_prefix_prefill_<Float16, 512, 512>(Flash_fwd_params &params, hipStream_t stream) {
#ifdef BUILD_FA_FWD
    run_flash_fwd_prefix_prefill<Float16, 512, 512>(params, stream);
#endif
}
