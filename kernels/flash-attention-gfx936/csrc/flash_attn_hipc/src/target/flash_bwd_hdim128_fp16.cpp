// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "../flash_bwd_launch_template.h"

template<>
void run_mha_bwd_<Float16, 128, 128>(Flash_bwd_params &params, hipStream_t stream, const bool configure) {
#ifdef BUILD_FA_BWD
    run_mha_bwd_hdim128<Float16>(params, stream, configure);
#endif
}
