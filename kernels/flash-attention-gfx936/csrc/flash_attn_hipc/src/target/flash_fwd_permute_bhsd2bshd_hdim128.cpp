#ifdef BUILD_FA_PERMUTE
#include <hip/hip_runtime.h>
#include "../flash_permute_hdim128.h"

template<>
__global__ void flash_permute_bhsd2bshd<128, 1, 32>(void* output, void* input, int32_t seqlen, int32_t num_heads) {
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    constexpr int32_t kHeadDim = 128;
    // 获取当前 block 的任务 id
    const int32_t bidb = blockIdx.z;
    const int32_t split_id = blockIdx.x;
    const int32_t bidh = blockIdx.y;
    // 计算当前任务读取数据的起始偏移
    int64_t batch_offset = bidb * num_heads * seqlen * kHeadDim;
    int64_t head_offset  = bidh * seqlen * kHeadDim;
    // 把输入当作一个一个 dword 处理
    float* read_ptr  = reinterpret_cast<float*>(input) + ((batch_offset + head_offset) >> 1);
    // 使用寄存器缓存数据
    float register_buffer[SEQLEN_PER_BLOCK];
    // 每个 block 处理 32 行 128 个 Half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 计算 seqlen 方向边界
        int32_t seqlen_limit = min(seqlen - 1, split_id * SEQLEN_PER_BLOCK + fetch);
        // block 偏移
        int64_t block_offset = seqlen_limit * kHeadDim;
        // thread 偏移
        int32_t thread_offset = threadIdx.x << 1;
        // 读取数据
        register_buffer[fetch] = read_ptr[(block_offset + thread_offset) >> 1];
    }
    __builtin_amdgcn_sched_barrier(0);
    // 计算当前任务写数据的起始偏移
    int64_t write_offset = (bidb * seqlen * num_heads + bidh) * kHeadDim;
    float*  write_ptr    = reinterpret_cast<float*>(output) + (write_offset >> 1);
    // 一次性写出
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 计算 seqlen 方向边界
        int32_t seqlen_limit = min(seqlen - 1, split_id * SEQLEN_PER_BLOCK + fetch);
        // thread 偏移
        int32_t thread_offset = threadIdx.x << 1;
        // 写出数据
        write_ptr[(seqlen_limit * num_heads * kHeadDim + thread_offset) >> 1] = register_buffer[fetch];
    }
}

#endif // end of BUILD_FA_PERMUTE