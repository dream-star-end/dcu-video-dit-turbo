// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "../flash_fwd_launch_template.h"

#if defined(BUILD_FA_KVCACHE) && !defined(HEADDIM_128_ONLY)
template void run_mha_fwd_splitkv_dispatch<Float16, 576, 512>(Flash_fwd_params &params, hipStream_t stream);
#endif