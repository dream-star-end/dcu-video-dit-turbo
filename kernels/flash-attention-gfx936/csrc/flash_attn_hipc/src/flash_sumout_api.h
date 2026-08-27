#include "numeric_types.h"
template<typename T>
__global__ void flash_sum_out(T* output, T* input, int stride0, int stride1) {
    const int num_threads = blockDim.x;
    const int reduce_num = stride1 / stride0;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int output_offset = bid * num_threads + tid;
    const int input_offset = output_offset / stride0 * stride1 + output_offset % stride0;
    T result = 0;
    for(int i = 0; i < reduce_num; ++i) {
        result += input[input_offset + i * stride0];
    }
    output[output_offset] = result;
}