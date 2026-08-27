#ifdef BUILD_FA_PERMUTE
#include <hip/hip_runtime.h>
#include "../flash_permute_hdim128.h"

template<>
__global__ void flash_permute_bhsd2sbhd<128, 1, 32>(void* output, void* input, int32_t seqlen, int real_headdim) {
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    constexpr int32_t kHeadDim = 128;
    // 获取任务 id
    const int32_t split_id = blockIdx.x;
    const int32_t bid_bh   = blockIdx.y;
    // 把输入输出看成一个一个 dword
    float* read_ptr = reinterpret_cast<float*>(input) + (bid_bh * seqlen * kHeadDim >> 1);
    // 使用寄存器缓存数据
    float register_buffer[SEQLEN_PER_BLOCK];
    // 每个 block 处理 32 行的 128 个 Half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 计算 seqlen 方向边界
        const int32_t seqlen_limit = min(seqlen - 1, fetch + split_id * SEQLEN_PER_BLOCK);
        // block 偏移
        const int64_t block_read_offset = seqlen_limit * kHeadDim;
        // thread 偏移
        const int32_t thread_offset = threadIdx.x << 1;
        // 读取数据
        register_buffer[fetch] = read_ptr[(block_read_offset + thread_offset) >> 1];
    }
    __builtin_amdgcn_sched_barrier(0);
    // 一次性写出
    const int32_t batch_heads = gridDim.y;
    float* write_ptr = reinterpret_cast<float*>(output) + (bid_bh * kHeadDim >> 1);
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // thread 偏移
        const int32_t thread_offset = threadIdx.x << 1;
        // 计算 seqlen 方向边界
        const int32_t seqlen_limit = min(seqlen - 1, fetch + split_id * SEQLEN_PER_BLOCK);
        // 写出数据
        write_ptr[(seqlen_limit * batch_heads * kHeadDim + thread_offset) >> 1] = register_buffer[fetch];
    }
}




template<>
__global__ void __launch_bounds__(128, 1) flash_permute_bhsd2sbhd<64, 1, 32>(void* output, void* input, int32_t seqlen, int real_headdim) {
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    constexpr int32_t kHeadDim = 64;
    // 获取任务 id
    const int32_t split_id = blockIdx.x;
    const int32_t bid_bh   = blockIdx.y;
    // 把输入输出看成一个一个 dword
    uint16_t* read_ptr = reinterpret_cast<uint16_t*>(input) + bid_bh * seqlen * kHeadDim;
    // 使用寄存器缓存数据
    uint16_t register_buffer[SEQLEN_PER_BLOCK * 1];
    // 每个 block 处理 32 行的 64 个 Half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 计算 seqlen 方向边界
        const int32_t seqlen_limit = min(seqlen - 1, fetch + split_id * SEQLEN_PER_BLOCK);
        // block 偏移
        const int64_t block_read_offset = seqlen_limit * kHeadDim;
        // thread 偏移
        const int32_t thread_offset = threadIdx.x;
        // 读取数据
        #pragma unroll
        for (int32_t s = 0; s < 1; ++s) {
            register_buffer[fetch * 1 + s] = read_ptr[block_read_offset + thread_offset + 64 * s];
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    // 一次性写出
    const int32_t batch_heads = gridDim.y;
    uint16_t* write_ptr = reinterpret_cast<uint16_t*>(output) + bid_bh * kHeadDim;
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // thread 偏移
        const int32_t thread_offset = threadIdx.x;
        // 计算 seqlen 方向边界
        const int32_t seqlen_limit = min(seqlen - 1, fetch + split_id * SEQLEN_PER_BLOCK);
        // 写出数据
        #pragma unroll
        for (int32_t s = 0; s < 1; ++s) {
            write_ptr[seqlen_limit * batch_heads * kHeadDim + thread_offset + 64 * s] = register_buffer[fetch * 1 + s];
        }
    }
}



template<>
__global__ void __launch_bounds__(128, 1) flash_permute_bhsd2sbhd<192, 1, 32>(void* output, void* input, int32_t seqlen, int real_headdim) {
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    constexpr int32_t kHeadDim = 192;
    // 获取任务 id
    const int32_t split_id = blockIdx.x;
    const int32_t bid_bh   = blockIdx.y;
    // 把输入输出看成一个一个 dword
    uint16_t* read_ptr = reinterpret_cast<uint16_t*>(input) + bid_bh * seqlen * kHeadDim;
    // 使用寄存器缓存数据
    uint16_t register_buffer[SEQLEN_PER_BLOCK * 3];
    // 每个 block 处理 32 行的 192 个 Half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 计算 seqlen 方向边界
        const int32_t seqlen_limit = min(seqlen - 1, fetch + split_id * SEQLEN_PER_BLOCK);
        // block 偏移
        const int64_t block_read_offset = seqlen_limit * kHeadDim;
        // thread 偏移
        const int32_t thread_offset = threadIdx.x;
        // 读取数据
        #pragma unroll
        for (int32_t s = 0; s < 3; ++s) {
            register_buffer[fetch * 3 + s] = read_ptr[block_read_offset + thread_offset + 64 * s];
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    // 一次性写出
    const int32_t batch_heads = gridDim.y;
    uint16_t* write_ptr = reinterpret_cast<uint16_t*>(output) + bid_bh * kHeadDim;
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // thread 偏移
        const int32_t thread_offset = threadIdx.x;
        // 计算 seqlen 方向边界
        const int32_t seqlen_limit = min(seqlen - 1, fetch + split_id * SEQLEN_PER_BLOCK);
        // 写出数据
        #pragma unroll
        for (int32_t s = 0; s < 3; ++s) {
            write_ptr[seqlen_limit * batch_heads * kHeadDim + thread_offset + 64 * s] = register_buffer[fetch * 3 + s];
        }
    }
}



template<>
__global__ void __launch_bounds__(128, 1) flash_permute_bhsd2sbhd<256, 1, 32>(void* output, void* input, int32_t seqlen, int real_headdim) {
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    constexpr int32_t kHeadDim = 256;
    using vec4_b16 = __attribute__((__vector_size__(2 * sizeof(uint16_t)))) uint16_t;
    // 获取任务 id
    const int32_t split_id = blockIdx.x;
    const int32_t bid_bh   = blockIdx.y;
    // 把输入输出看成一个一个 dword
    uint16_t* read_ptr = reinterpret_cast<uint16_t*>(input) + bid_bh * seqlen * kHeadDim;
    // 使用寄存器缓存数据
    vec4_b16 register_buffer[SEQLEN_PER_BLOCK];
    // 每个 block 处理 32 行的 192 个 Half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 计算 seqlen 方向边界
        const int32_t seqlen_limit = min(seqlen - 1, fetch + split_id * SEQLEN_PER_BLOCK);
        // block 偏移
        const int64_t block_read_offset = seqlen_limit * kHeadDim;
        // thread 偏移
        const int32_t thread_offset = threadIdx.x << 2;
        // 读取数据
        register_buffer[fetch] = *(vec4_b16*)(read_ptr + block_read_offset + thread_offset);
    }
    __builtin_amdgcn_sched_barrier(0);
    // 一次性写出
    const int32_t batch_heads = gridDim.y;
    uint16_t* write_ptr = reinterpret_cast<uint16_t*>(output) + bid_bh * kHeadDim;
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // thread 偏移
        const int32_t thread_offset = threadIdx.x << 2;
        // 计算 seqlen 方向边界
        const int32_t seqlen_limit = min(seqlen - 1, fetch + split_id * SEQLEN_PER_BLOCK);
        // 写出数据
        *(vec4_b16*)(write_ptr + seqlen_limit * batch_heads * kHeadDim + thread_offset) = register_buffer[fetch];
    }
}



template<>
__global__ void __launch_bounds__(128, 1) flash_permute_bhsd2sbhd<0, 1, 32>(void* output, void* input, int32_t seqlen, int real_headdim) {
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    constexpr int32_t kHeadDimLimit = 256;
    // 获取任务 id
    const int32_t split_id = blockIdx.x;
    const int32_t bid_bh   = blockIdx.y;
    // 把输入输出看成一个一个 dword
    uint16_t* read_ptr = reinterpret_cast<uint16_t*>(input) + bid_bh * seqlen * real_headdim;
    // 使用寄存器缓存数据
    constexpr int32_t ELEM_COUNT = (kHeadDimLimit >> 6);
    uint16_t register_buffer[SEQLEN_PER_BLOCK * ELEM_COUNT];
    // 每个 block 处理 32 行的最多 256 个 Half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 计算 seqlen 方向边界
        const int32_t seqlen_limit = min(seqlen - 1, fetch + split_id * SEQLEN_PER_BLOCK);
        // block 偏移
        const int64_t block_read_offset = seqlen_limit * real_headdim;
        // thread 偏移
        const int32_t thread_offset = threadIdx.x;
        // 读取数据
        #pragma unroll
        for (int32_t s = 0; s < ELEM_COUNT; ++s) {
            int32_t offset = thread_offset + 64 * s;
            if (offset < real_headdim)
                register_buffer[fetch * ELEM_COUNT + s] = read_ptr[block_read_offset + offset];
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    // 一次性写出
    const int32_t batch_heads = gridDim.y;
    uint16_t* write_ptr = reinterpret_cast<uint16_t*>(output) + bid_bh * real_headdim;
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // thread 偏移
        const int32_t thread_offset = threadIdx.x;
        // 计算 seqlen 方向边界
        const int32_t seqlen_limit = min(seqlen - 1, fetch + split_id * SEQLEN_PER_BLOCK);
        // 写出数据
        #pragma unroll
        for (int32_t s = 0; s < ELEM_COUNT; ++s) {
            int32_t offset = thread_offset + 64 * s;
            if (offset < real_headdim)
                write_ptr[seqlen_limit * batch_heads * real_headdim + offset] = register_buffer[fetch * ELEM_COUNT + s];
        }
    }
}

#endif // end of BUILD_FA_PERMUTE