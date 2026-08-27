// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "../flash_fwd_launch_template.h"

#ifdef BUILD_FA_KVCACHE
template void run_int8_fwd_splitkv_dispatch<Float16, 128, 128>(Flash_fwd_params &params, hipStream_t stream);
#endif