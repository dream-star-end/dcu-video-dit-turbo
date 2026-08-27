// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "../flash_fwd_launch_template.h"

template<>
void run_mla_fwd_prefix_prefill_dispatch_<Float16, 576, 512>(Flash_fwd_mla_params &params, hipStream_t stream) {
#ifdef BUILD_FA_FWD
    run_mla_fwd_prefix_prefill_dispatch<Float16, 576, 512>(params, stream);
#endif
}
