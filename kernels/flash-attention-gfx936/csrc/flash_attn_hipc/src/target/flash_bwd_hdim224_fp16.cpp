// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "../flash_bwd_launch_template.h"

#if defined(BUILD_FA_BWD) && !defined(HEADDIM_128_ONLY)
template<>
void run_mha_bwd_<Float16, 224, 224>(Flash_bwd_params &params, hipStream_t stream, const bool configure) {
    run_mha_bwd_hdim224<Float16>(params, stream, configure);
}
#endif
