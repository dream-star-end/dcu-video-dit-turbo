#ifdef BUILD_FA_PERMUTE
#include <hip/hip_runtime.h>
#include "../flash_permute_hdim128.h"

template<>
__global__ void flash_permute_sbhd2bhsd<128, 1, 32>(void* output, void* input, int32_t seqlen, int real_headdim) {
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    constexpr int32_t kHeadDim = 128;
    // 获取当前任务信息
    int32_t split_id = blockIdx.x;
    int32_t bid_bh   = blockIdx.y;
    // 获取该任务读取数据的起始位移
    int32_t batch_heads = gridDim.y;
    // 把输入都看作一个一个 dword
    int64_t bid_bh_read_offset = bid_bh * kHeadDim;
    float* read_ptr  = reinterpret_cast<float*>(input) + (bid_bh_read_offset >> 1);
    // 使用寄存器缓存所有数据
    float register_buffer[SEQLEN_PER_BLOCK];
    // 每个 block 负责处理 32 条 128 个 Half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制 seqlen 方向位移
        int32_t seqlen_limit = min(seqlen - 1, fetch + (split_id * SEQLEN_PER_BLOCK));
        // block 位移
        int64_t block_read_offset = seqlen_limit * batch_heads * kHeadDim;
        // thread 位移
        int32_t thread_offset = threadIdx.x << 1;
        // 读取数据
        register_buffer[fetch] = read_ptr[(block_read_offset + thread_offset) >> 1];
    }
    __builtin_amdgcn_sched_barrier(0);
    // 获取该任务写出数据的起始位移
    int64_t bid_bh_write_offset = bid_bh * seqlen * kHeadDim;
    float* write_ptr = reinterpret_cast<float*>(output) + (bid_bh_write_offset >> 1);
    // 一次性把数据写出到 global memory, 跟 global_load 分离开
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // thread 位移
        int32_t thread_offset = threadIdx.x << 1;
        // 限制 seqlen 方向位移
        int32_t seqlen_limit = min(seqlen - 1, fetch + (split_id * SEQLEN_PER_BLOCK));
        // 数据写出
        write_ptr[(seqlen_limit * kHeadDim + thread_offset) >> 1] = register_buffer[fetch];
    }
}



template<>
__global__ void __launch_bounds__(64, 1) flash_permute_sbhd2bhsd<64, 1, 32>(void* output, void* input, int32_t seqlen, int real_headdim) {
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    constexpr int32_t kHeadDim = 64;
    // 获取当前任务信息
    int32_t split_id = blockIdx.x;
    int32_t bid_bh   = blockIdx.y;
    // 获取该任务读取数据的起始位移
    int32_t batch_heads = gridDim.y;
    // 把输入都看作一个一个 dword
    int64_t bid_bh_read_offset = bid_bh * kHeadDim;
    uint16_t* read_ptr  = reinterpret_cast<uint16_t*>(input) + bid_bh_read_offset;
    // 使用寄存器缓存所有数据
    constexpr int32_t ELEM_COUNT = (kHeadDim >> 6);
    uint16_t register_buffer[SEQLEN_PER_BLOCK * ELEM_COUNT];
    // 每个 block 负责处理 32 条 192 个 Half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制 seqlen 方向位移
        int32_t seqlen_limit = min(seqlen - 1, fetch + (split_id * SEQLEN_PER_BLOCK));
        // block 位移
        int64_t block_read_offset = seqlen_limit * batch_heads * kHeadDim;
        // thread 位移
        int32_t thread_offset = threadIdx.x;
        // 读取数据
        #pragma unroll
        for (int32_t s = 0; s < ELEM_COUNT; ++s) {
            register_buffer[fetch * ELEM_COUNT + s] = read_ptr[block_read_offset + thread_offset + 64 * s];
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    // 获取该任务写出数据的起始位移
    int64_t bid_bh_write_offset = bid_bh * seqlen * kHeadDim;
    uint16_t* write_ptr = reinterpret_cast<uint16_t*>(output) + bid_bh_write_offset;
    // 一次性把数据写出到 global memory, 跟 global_load 分离开
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // thread 位移
        int32_t thread_offset = threadIdx.x;
        // 限制 seqlen 方向位移
        int32_t seqlen_limit = min(seqlen - 1, fetch + (split_id * SEQLEN_PER_BLOCK));
        // 数据写出
        #pragma unroll
        for (int32_t s = 0; s < ELEM_COUNT; ++s) {
            write_ptr[seqlen_limit * kHeadDim + thread_offset + 64 * s] = register_buffer[fetch * ELEM_COUNT + s];
        }
    }
}



template<>
__global__ void __launch_bounds__(128, 1) flash_permute_sbhd2bhsd<192, 1, 32>(void* output, void* input, int32_t seqlen, int real_headdim) {
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    constexpr int32_t kHeadDim = 192;
    // 获取当前任务信息
    int32_t split_id = blockIdx.x;
    int32_t bid_bh   = blockIdx.y;
    // 获取该任务读取数据的起始位移
    int32_t batch_heads = gridDim.y;
    // 把输入都看作一个一个 dword
    int64_t bid_bh_read_offset = bid_bh * kHeadDim;
    uint16_t* read_ptr  = reinterpret_cast<uint16_t*>(input) + bid_bh_read_offset;
    // 使用寄存器缓存所有数据
    constexpr int32_t ELEM_COUNT = (kHeadDim >> 6);
    uint16_t register_buffer[SEQLEN_PER_BLOCK * ELEM_COUNT];
    // 每个 block 负责处理 32 条 192 个 Half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制 seqlen 方向位移
        int32_t seqlen_limit = min(seqlen - 1, fetch + (split_id * SEQLEN_PER_BLOCK));
        // block 位移
        int64_t block_read_offset = seqlen_limit * batch_heads * kHeadDim;
        // thread 位移
        int32_t thread_offset = threadIdx.x;
        // 读取数据
        #pragma unroll
        for (int32_t s = 0; s < ELEM_COUNT; ++s) {
            register_buffer[fetch * ELEM_COUNT + s] = read_ptr[block_read_offset + thread_offset + 64 * s];
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    // 获取该任务写出数据的起始位移
    int64_t bid_bh_write_offset = bid_bh * seqlen * kHeadDim;
    uint16_t* write_ptr = reinterpret_cast<uint16_t*>(output) + bid_bh_write_offset;
    // 一次性把数据写出到 global memory, 跟 global_load 分离开
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // thread 位移
        int32_t thread_offset = threadIdx.x;
        // 限制 seqlen 方向位移
        int32_t seqlen_limit = min(seqlen - 1, fetch + (split_id * SEQLEN_PER_BLOCK));
        // 数据写出
        #pragma unroll
        for (int32_t s = 0; s < ELEM_COUNT; ++s) {
            write_ptr[seqlen_limit * kHeadDim + thread_offset + 64 * s] = register_buffer[fetch * ELEM_COUNT + s];
        }
    }
}



template<>
__global__ void __launch_bounds__(128, 1) flash_permute_sbhd2bhsd<256, 1, 32>(void* output, void* input, int32_t seqlen, int real_headdim) {
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    constexpr int32_t kHeadDim = 256;
    using vec2_fp32 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
    // 获取当前任务信息
    int32_t split_id = blockIdx.x;
    int32_t bid_bh   = blockIdx.y;
    // 获取该任务读取数据的起始位移
    int32_t batch_heads = gridDim.y;
    // 把输入都看作一个一个 dword
    int64_t bid_bh_read_offset = bid_bh * kHeadDim;
    vec2_fp32* read_ptr  = reinterpret_cast<vec2_fp32*>(input) + (bid_bh_read_offset >> 2);
    // 使用寄存器缓存所有数据
    constexpr int32_t ELEM_COUNT = (kHeadDim >> 6) >> 2;
    vec2_fp32 register_buffer[SEQLEN_PER_BLOCK * ELEM_COUNT];
    // 每个 block 负责处理 32 条 192 个 Half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制 seqlen 方向位移
        int32_t seqlen_limit = min(seqlen - 1, fetch + (split_id * SEQLEN_PER_BLOCK));
        // block 位移
        int64_t block_read_offset = (seqlen_limit * batch_heads * kHeadDim) >> 2;
        // thread 位移
        int32_t thread_offset = threadIdx.x;
        // 读取数据
        #pragma unroll
        for (int32_t s = 0; s < ELEM_COUNT; ++s) {
            register_buffer[fetch * ELEM_COUNT + s] = read_ptr[block_read_offset + thread_offset + 256 * s];
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    // 获取该任务写出数据的起始位移
    int64_t bid_bh_write_offset = bid_bh * seqlen * kHeadDim;
    vec2_fp32* write_ptr = reinterpret_cast<vec2_fp32*>(output) + (bid_bh_write_offset >> 2);
    // 一次性把数据写出到 global memory, 跟 global_load 分离开
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // thread 位移
        int32_t thread_offset = threadIdx.x;
        // 限制 seqlen 方向位移
        int32_t seqlen_limit = min(seqlen - 1, fetch + (split_id * SEQLEN_PER_BLOCK));
        // 数据写出
        #pragma unroll
        for (int32_t s = 0; s < ELEM_COUNT; ++s) {
            write_ptr[(seqlen_limit * kHeadDim >> 2) + thread_offset + 256 * s] = register_buffer[fetch * ELEM_COUNT + s];
        }
    }
}

template<>
__global__ void __launch_bounds__(128, 1) flash_permute_sbhd2bhsd<0, 1, 32>(void* output, void* input, int32_t seqlen, int real_headdim) {
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    constexpr int32_t kHeadDimLimit = 256;
    // 获取当前任务信息
    int32_t split_id = blockIdx.x;
    int32_t bid_bh   = blockIdx.y;
    // 获取该任务读取数据的起始位移
    int32_t batch_heads = gridDim.y;
    // 把输入都看作一个一个 dword
    int64_t bid_bh_read_offset = bid_bh * real_headdim;
    uint16_t* read_ptr  = reinterpret_cast<uint16_t*>(input) + bid_bh_read_offset;
    // 使用寄存器缓存所有数据
    constexpr int32_t ELEM_COUNT = (kHeadDimLimit >> 6);
    uint16_t register_buffer[SEQLEN_PER_BLOCK * ELEM_COUNT];
    // 每个 block 负责处理 32 条 192 个 Half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制 seqlen 方向位移
        int32_t seqlen_limit = min(seqlen - 1, fetch + (split_id * SEQLEN_PER_BLOCK));
        // block 位移
        int64_t block_read_offset = seqlen_limit * batch_heads * real_headdim;
        // thread 位移
        int32_t thread_offset = threadIdx.x;
        // 读取数据
        #pragma unroll
        for (int32_t s = 0; s < ELEM_COUNT; ++s) {
            int offset = thread_offset + 64 * s;
            if (offset < real_headdim) {
                register_buffer[fetch * ELEM_COUNT + s] = read_ptr[block_read_offset + offset];
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    // 获取该任务写出数据的起始位移
    int64_t bid_bh_write_offset = bid_bh * seqlen * real_headdim;
    uint16_t* write_ptr = reinterpret_cast<uint16_t*>(output) + bid_bh_write_offset;
    // 一次性把数据写出到 global memory, 跟 global_load 分离开
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // thread 位移
        int32_t thread_offset = threadIdx.x;
        // 限制 seqlen 方向位移
        int32_t seqlen_limit = min(seqlen - 1, fetch + (split_id * SEQLEN_PER_BLOCK));
        // 数据写出
        #pragma unroll
        for (int32_t s = 0; s < ELEM_COUNT; ++s) {
            int offset = thread_offset + 64 * s;
            if (offset < real_headdim) {
                write_ptr[seqlen_limit * real_headdim + offset] = register_buffer[fetch * ELEM_COUNT + s];
            }
        }
    }
}

#endif // end of BUILD_FA_PERMUTE