#include <hip/hip_runtime.h>
#include "flash_sumout_api.h"

template __global__ void flash_sum_out<BFloat16>(BFloat16*, BFloat16*, int, int);
template __global__ void flash_sum_out<Float16>(Float16*, Float16*, int, int);