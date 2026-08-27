#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_mla_<cutlass::half_t, 192, 128, false>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim192_hdim128<cutlass::half_t, false>(params, stream);
}