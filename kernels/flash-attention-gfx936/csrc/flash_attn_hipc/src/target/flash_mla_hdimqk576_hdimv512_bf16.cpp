// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "../flash_fwd_launch_template.h"

#if defined(BUILD_FLASHMLA)
template void run_mla_fwd_splitkv_dispatch<BFloat16, 576, 512>(Flash_fwd_mla_params &params, hipStream_t stream);

template void run_mla_fwd_dispatch<BFloat16, 576, 512>(Flash_fwd_mla_params &params, hipStream_t stream);
#endif