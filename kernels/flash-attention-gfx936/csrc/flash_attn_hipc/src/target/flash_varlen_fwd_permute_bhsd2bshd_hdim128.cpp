#ifdef BUILD_FA_PERMUTE
#include <hip/hip_runtime.h>
#include "../flash_fwd_permute_hdim128.h"

template<>
__global__ void flash_fwd_varlen_permute_bhsd2bshd<128, 1, 0>(
        void* output, void* input, void* split_sizes, int num_heads, int real_headdim) {
    using index_t = int64_t;
    constexpr int32_t kHeadDim = 128;
    // 获取 batch id
    int32_t bidb = blockIdx.y;
    // 获取 head id, 第几个头
    int32_t bidh = blockIdx.x;
    // 获取当前 batch 的 seqlen 长度, 决定要取多少次数据
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 根据 batch id, 获取这个 Batch 的起始地址, 注意的转置都是在 batch 内部的, 不同 batch 的内容不会相互影响
    index_t batch_offset = split[bidb] * num_heads * kHeadDim;
    // 获取当前 head 在这个 batch 内的读取地址
    index_t head_offset = bidh * cur_seqlen_q * kHeadDim;
    // 得到读取的指针
    float* read_ptr = reinterpret_cast<float*>(input) + ((batch_offset + head_offset) >> 1);
    // 得到写出的指针
    float* write_ptr = reinterpret_cast<float*>(output) + ((batch_offset + bidh * kHeadDim) >> 1);
    // 循环取 seqlen 次数据
    for (int32_t fetch = 0; fetch < cur_seqlen_q; ++fetch) {
        // 当前 head 的数据内 seqlen x head_dim, 读取 seqlen 次, 每个 64 个线程读取 128 个 Half
        int32_t block_offset  = fetch * kHeadDim;
        int32_t thread_offset = threadIdx.x << 1;
        // 读取数据, 这里可以用 global_load_dwordx4 优化, 逻辑会稍微复杂一点
        float content = read_ptr[(block_offset + thread_offset) >> 1];
        // 写出去
        index_t write_block_offset = fetch * num_heads * kHeadDim;
        write_ptr[(write_block_offset + thread_offset) >> 1] = content;
    }
}



template<>
__global__ void flash_fwd_varlen_permute_bhsd2bshd<128, 1, 32>(
        void* output, void* input, void* split_sizes, int num_heads, int real_headdim) {
    using index_t = int64_t;
    constexpr int32_t kHeadDim = 128;
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    // 获取 batch id
    int32_t bidb = blockIdx.z;
    // 获取当前 seqlen 的第几个块
    int32_t seq_id = blockIdx.y;
    // 获取 head id, 第几个头
    int32_t bidh = blockIdx.x;
    // 获取当前 batch 的 seqlen 长度, 决定要取多少次数据
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 得到当前处理的小块从哪里开始取数据
    index_t seq_offset = seq_id * SEQLEN_PER_BLOCK * kHeadDim;
    // 获取当前处理的小块实际的 seqlen 跨度
    int32_t seqlen_block_count = (cur_seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    int32_t actual_seqlen = (seq_id >= seqlen_block_count - 1) ? cur_seqlen_q - seq_id * SEQLEN_PER_BLOCK: SEQLEN_PER_BLOCK;
    // if (threadIdx.x == 0) { printf("bidb: %d | seq_id: %d | bidh: %d | cur_seqlen_q: %d | actual_seqlen: %d\n", bidb, seq_id, bidh, cur_seqlen_q, actual_seqlen); }
    if (actual_seqlen <= 0 or seq_id * SEQLEN_PER_BLOCK > cur_seqlen_q) return;
    // 根据 batch id, 获取这个 Batch 的起始地址, 注意的转置都是在 batch 内部的, 不同 batch 的内容不会相互影响
    index_t batch_offset = split[bidb] * num_heads * kHeadDim;
    // 获取当前 head 在这个 batch 内的读取地址
    index_t head_offset = bidh * cur_seqlen_q * kHeadDim;
    // 得到读取的指针
    float* read_ptr = reinterpret_cast<float*>(input) + ((batch_offset + head_offset + seq_offset) >> 1);
    // 得到写出的指针
    float* write_ptr = reinterpret_cast<float*>(output) + ((batch_offset + seq_id * SEQLEN_PER_BLOCK * num_heads * kHeadDim + bidh * kHeadDim) >> 1);
    // 循环取 seqlen 次数据
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit = min(actual_seqlen - 1, fetch);
        // 当前 head 的数据内 seqlen x head_dim, 读取 seqlen 次, 每个 64 个线程读取 128 个 Half
        int32_t block_offset  = seqlen_limit * kHeadDim;
        int32_t thread_offset = threadIdx.x << 1;
        // 读取数据, 这里可以用 global_load_dwordx4 优化, 逻辑会稍微复杂一点
        float content = read_ptr[(block_offset + thread_offset) >> 1];
        // 写出去
        index_t write_block_offset = seqlen_limit * num_heads * kHeadDim;
        write_ptr[(write_block_offset + thread_offset) >> 1] = content;
    }
}


/*
    目标用上 dwordx4 指令, 可以利用 seq 维度用上 load_dwordx4, 也可以利用 num_heads 维度用上 store_dwordx4
*/
template<>
__global__ void flash_fwd_varlen_permute_bhsd2bshd<128, 4, 32>(
        void* output, void* input, void* split_sizes, int num_heads, int real_headdim) {
    using index_t   = int64_t;
    using vec4_fp32 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    using vec2_fp32 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
    using vec4_uint = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
    constexpr int32_t kHeadDim = 128;
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    // 获取 batch id
    int32_t bidb = blockIdx.z;
    // 获取当前 seqlen 的第几个块
    int32_t seq_id = blockIdx.y;
    // 获取 head id, 第几个头
    int32_t bidh = blockIdx.x;
    // 获取当前 batch 的 seqlen 长度, 决定要取多少次数据
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 得到当前处理的小块从哪里开始取数据
    index_t seq_offset = seq_id * SEQLEN_PER_BLOCK * kHeadDim;
    // 获取当前处理的小块实际的 seqlen 跨度
    int32_t seqlen_block_count = (cur_seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    int32_t actual_seqlen = (seq_id >= seqlen_block_count - 1) ? cur_seqlen_q - seq_id * SEQLEN_PER_BLOCK: SEQLEN_PER_BLOCK;
    // if (threadIdx.x == 0) { printf("bidb: %d | seq_id: %d | bidh: %d | cur_seqlen_q: %d | actual_seqlen: %d\n", bidb, seq_id, bidh, cur_seqlen_q, actual_seqlen); }
    if (actual_seqlen <= 0 or seq_id * SEQLEN_PER_BLOCK > cur_seqlen_q) return;
    // 根据 batch id, 获取这个 Batch 的起始地址, 注意的转置都是在 batch 内部的, 不同 batch 的内容不会相互影响
    index_t batch_offset = split[bidb] * num_heads * kHeadDim;
    // 获取当前 head 在这个 batch 内的读取地址
    index_t head_offset = bidh * cur_seqlen_q * kHeadDim;
    // 得到读取的指针
    float* read_ptr = reinterpret_cast<float*>(input) + ((batch_offset + head_offset + seq_offset) >> 1);
    // 配置读指针的 buffer resource
    vec4_uint read_buffer;
    *(uint64_t*)&read_buffer = reinterpret_cast<uint64_t>(read_ptr);
    read_buffer[2] = 0x80000000;
    read_buffer[3] = 0x00020000;
    // 得到写出的指针
    float* write_ptr = reinterpret_cast<float*>(output) + ((batch_offset + seq_id * SEQLEN_PER_BLOCK * num_heads * kHeadDim + bidh * kHeadDim) >> 1);
    // 计算一些需要的 lane 下标
    int32_t lane_id = threadIdx.x;
    int32_t lane_id_row = lane_id >> 4;
    int32_t lane_id_col = lane_id % 16;
    // 需要用 lds, 至少需要 4x128 个 Half
    __shared__ float lds[1024];
    // 循环取 seqlen 次数据
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK / 4; ++fetch) {
        // 限制边界
        int32_t seqlen_limit = min(actual_seqlen - 1, fetch * 4 + lane_id_row);
        // 当前 head 的数据内 seqlen x head_dim, 读取 seqlen 次, 每个 64 个线程读取 128 个 Half
        int32_t block_offset  = seqlen_limit * kHeadDim;
        int32_t thread_offset = lane_id_col * 8;
        // 一次读取 4x128 的 Half 到 LDS
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
            __builtin_hcu_raw_buffer_load_lds(
                read_buffer,
                lds + lane_id * 4,
                16,
                (block_offset + thread_offset) << 1, /* v_offset */
                0, /* s_offset */
                0, /* immediate offset, instruction offset */
                0  /* auxilariy data| bit 0: glc, bit 1: slc, bit 2: dlc, bit 3: cache swizzle */
            );
        #else
            *(vec4_fp32*)(lds + lane_id * 4) = *(vec4_fp32*)(read_ptr + ((block_offset + thread_offset) >> 1));
        #endif
        // 从 LDS 转置后, 64 个线程写 4 行, 每次写 128 个 Half, 对应 fetch * 4 + [0,3] 的 seqlen
        vec2_fp32 data0 = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float*)lds + lane_id, 0, 64, false);
        vec2_fp32 data1 = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float*)lds + lane_id + 128, 0, 64, false);
        write_ptr[(min(actual_seqlen - 1, fetch * 4 + 0) * num_heads * kHeadDim + (lane_id << 1)) >> 1] = data0[0];
        write_ptr[(min(actual_seqlen - 1, fetch * 4 + 1) * num_heads * kHeadDim + (lane_id << 1)) >> 1] = data0[1];
        write_ptr[(min(actual_seqlen - 1, fetch * 4 + 2) * num_heads * kHeadDim + (lane_id << 1)) >> 1] = data1[0];
        write_ptr[(min(actual_seqlen - 1, fetch * 4 + 3) * num_heads * kHeadDim + (lane_id << 1)) >> 1] = data1[1];
    }
}



template<>
__global__ void flash_fwd_varlen_permute_bhsd2bshd<192, 1, 32>(
        void* output, void* input, void* split_sizes, int num_heads, int real_headdim) {
    using index_t = int64_t;
    constexpr int32_t kHeadDim = 192;
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    // 获取 batch id
    int32_t bidb = blockIdx.z;
    // 获取当前 seqlen 的第几个块
    int32_t seq_id = blockIdx.y;
    // 获取 head id, 第几个头
    int32_t bidh = blockIdx.x;
    // 获取当前 batch 的 seqlen 长度, 决定要取多少次数据
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 得到当前处理的小块从哪里开始取数据
    index_t seq_offset = seq_id * SEQLEN_PER_BLOCK * kHeadDim;
    // 获取当前处理的小块实际的 seqlen 跨度
    int32_t seqlen_block_count = (cur_seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    int32_t actual_seqlen = (seq_id >= seqlen_block_count - 1) ? cur_seqlen_q - seq_id * SEQLEN_PER_BLOCK: SEQLEN_PER_BLOCK;
    // if (threadIdx.x == 0) { printf("bidb: %d | seq_id: %d | bidh: %d | cur_seqlen_q: %d | actual_seqlen: %d\n", bidb, seq_id, bidh, cur_seqlen_q, actual_seqlen); }
    if (actual_seqlen <= 0 or seq_id * SEQLEN_PER_BLOCK > cur_seqlen_q) return;
    // 根据 batch id, 获取这个 Batch 的起始地址, 注意的转置都是在 batch 内部的, 不同 batch 的内容不会相互影响
    index_t batch_offset = split[bidb] * num_heads * kHeadDim;
    // 获取当前 head 在这个 batch 内的读取地址
    index_t head_offset = bidh * cur_seqlen_q * kHeadDim;
    // 得到读取的指针
    uint16_t* read_ptr = reinterpret_cast<uint16_t*>(input) + (batch_offset + head_offset + seq_offset);
    // 得到写出的指针
    uint16_t* write_ptr = reinterpret_cast<uint16_t*>(output) + (batch_offset + seq_id * SEQLEN_PER_BLOCK * num_heads * kHeadDim + bidh * kHeadDim);
    // 循环取 seqlen 次数据
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit = min(actual_seqlen - 1, fetch);
        // 当前 head 的数据内 seqlen x kHeadDim, 读取 seqlen 次, 每个 64 个线程读取 192 个 Half
        int32_t block_offset  = seqlen_limit * kHeadDim;
        int32_t thread_offset = threadIdx.x;
        // 读取数据, 3 个 half, 可以当成读取一次 int32_t 加一次 int16_t 的数据
        // uint16_t a = read_ptr[block_offset + thread_offset + 0 * 64];
        // uint16_t b = read_ptr[block_offset + thread_offset + 1 * 64];
        // uint16_t c = read_ptr[block_offset + thread_offset + 2 * 64];
        uint32_t ab = *(uint32_t*)(read_ptr + block_offset + thread_offset * 2);
        uint16_t c = read_ptr[block_offset + thread_offset + 2 * 64];
        // 写出去
        // index_t write_block_offset = seqlen_limit * num_heads * kHeadDim;
        // write_ptr[write_block_offset + thread_offset + 0 * 64] = a;
        // write_ptr[write_block_offset + thread_offset + 1 * 64] = b;
        // write_ptr[write_block_offset + thread_offset + 2 * 64] = c;
        index_t write_block_offset = seqlen_limit * num_heads * kHeadDim;
        *(uint32_t*)(write_ptr + write_block_offset + thread_offset * 2) = ab;
        write_ptr[write_block_offset + thread_offset + 2 * 64] = c;
    }
}



template<>
__global__ void flash_fwd_varlen_permute_bhsd2bshd<64, 1, 32>(
        void* output, void* input, void* split_sizes, int num_heads, int real_headdim) {
    using index_t = int64_t;
    constexpr int32_t kHeadDim = 64;
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    // 获取 batch id
    int32_t bidb = blockIdx.z;
    // 获取当前 seqlen 的第几个块
    int32_t seq_id = blockIdx.y;
    // 获取 head id, 第几个头
    int32_t bidh = blockIdx.x;
    // 获取当前 batch 的 seqlen 长度, 决定要取多少次数据
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 得到当前处理的小块从哪里开始取数据
    index_t seq_offset = seq_id * SEQLEN_PER_BLOCK * kHeadDim;
    // 获取当前处理的小块实际的 seqlen 跨度
    int32_t seqlen_block_count = (cur_seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    int32_t actual_seqlen = (seq_id >= seqlen_block_count - 1) ? cur_seqlen_q - seq_id * SEQLEN_PER_BLOCK: SEQLEN_PER_BLOCK;
    // if (threadIdx.x == 0) { printf("bidb: %d | seq_id: %d | bidh: %d | cur_seqlen_q: %d | actual_seqlen: %d\n", bidb, seq_id, bidh, cur_seqlen_q, actual_seqlen); }
    if (actual_seqlen <= 0 or seq_id * SEQLEN_PER_BLOCK > cur_seqlen_q) return;
    // 根据 batch id, 获取这个 Batch 的起始地址, 注意的转置都是在 batch 内部的, 不同 batch 的内容不会相互影响
    index_t batch_offset = split[bidb] * num_heads * kHeadDim;
    // 获取当前 head 在这个 batch 内的读取地址
    index_t head_offset = bidh * cur_seqlen_q * kHeadDim;
    // 得到读取的指针
    uint16_t* read_ptr = reinterpret_cast<uint16_t*>(input) + (batch_offset + head_offset + seq_offset);
    // 得到写出的指针
    uint16_t* write_ptr = reinterpret_cast<uint16_t*>(output) + (batch_offset + seq_id * SEQLEN_PER_BLOCK * num_heads * kHeadDim + bidh * kHeadDim);
    // 循环取 seqlen 次数据
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit = min(actual_seqlen - 1, fetch);
        // 当前 head 的数据内 seqlen x kHeadDim, 读取 seqlen 次, 每个 64 个线程读取 192 个 Half
        int32_t block_offset  = seqlen_limit * kHeadDim;
        index_t write_block_offset = seqlen_limit * num_heads * kHeadDim;
        int32_t thread_offset = threadIdx.x;
        uint16_t data = read_ptr[block_offset + thread_offset];
        write_ptr[write_block_offset + thread_offset] = data;
    }
}


template<>
__global__ void flash_fwd_varlen_permute_bhsd2bshd<0, 1, 32>(
        void* output, void* input, void* split_sizes, int num_heads, int real_headdim) {
    using index_t = int64_t;
    constexpr int32_t HEADDIM_LIMIT    = 256;
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    // 获取 batch id
    int32_t bidb = blockIdx.z;
    // 获取当前 seqlen 的第几个块
    int32_t seq_id = blockIdx.y;
    // 获取 head id, 第几个头
    int32_t bidh = blockIdx.x;
    // 获取当前 batch 的 seqlen 长度, 决定要取多少次数据
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 得到当前处理的小块从哪里开始取数据
    index_t seq_offset = seq_id * SEQLEN_PER_BLOCK * real_headdim;
    // 获取当前处理的小块实际的 seqlen 跨度
    int32_t seqlen_block_count = (cur_seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    int32_t actual_seqlen = (seq_id >= seqlen_block_count - 1) ? cur_seqlen_q - seq_id * SEQLEN_PER_BLOCK: SEQLEN_PER_BLOCK;
    // if (threadIdx.x == 0) { printf("bidb: %d | seq_id: %d | bidh: %d | cur_seqlen_q: %d | actual_seqlen: %d\n", bidb, seq_id, bidh, cur_seqlen_q, actual_seqlen); }
    if (actual_seqlen <= 0 or seq_id * SEQLEN_PER_BLOCK > cur_seqlen_q) return;
    // 根据 batch id, 获取这个 Batch 的起始地址, 注意的转置都是在 batch 内部的, 不同 batch 的内容不会相互影响
    index_t batch_offset = split[bidb] * num_heads * real_headdim;
    // 获取当前 head 在这个 batch 内的读取地址
    index_t head_offset = bidh * cur_seqlen_q * real_headdim;
    // 得到读取的指针
    uint16_t* read_ptr = reinterpret_cast<uint16_t*>(input) + (batch_offset + head_offset + seq_offset);
    // 得到写出的指针
    uint16_t* write_ptr = reinterpret_cast<uint16_t*>(output) + (batch_offset + seq_id * SEQLEN_PER_BLOCK * num_heads * real_headdim + bidh * real_headdim);
    // 循环取 seqlen 次数据
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit = min(actual_seqlen - 1, fetch);
        // 当前 head 的数据内 seqlen x real_headdim, 读取 seqlen 次, 每个 64 个线程读取 192 个 Half
        int32_t block_offset  = seqlen_limit * real_headdim;
        index_t write_block_offset = seqlen_limit * num_heads * real_headdim;
        #pragma unroll
        for (int i = 0; i < (HEADDIM_LIMIT / 64); ++i) {
            int32_t thread_offset = threadIdx.x + i * 64;
            if (thread_offset < real_headdim) {
                uint16_t data = read_ptr[block_offset + thread_offset];
                write_ptr[write_block_offset + thread_offset] = data;
            }
        }
    }
}


template<>
__global__ void flash_fwd_varlen_permute_bhsd2bshd<256, 1, 32>(
        void* output, void* input, void* split_sizes, int num_heads, int real_headdim) {
    using index_t = int64_t;
    constexpr int32_t kHeadDim = 256;
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    using vec2_int = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    // 获取 batch id
    int32_t bidb = blockIdx.z;
    // 获取当前 seqlen 的第几个块
    int32_t seq_id = blockIdx.y;
    // 获取 head id, 第几个头
    int32_t bidh = blockIdx.x;
    // 获取当前 batch 的 seqlen 长度, 决定要取多少次数据
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 得到当前处理的小块从哪里开始取数据
    index_t seq_offset = seq_id * SEQLEN_PER_BLOCK * kHeadDim;
    // 获取当前处理的小块实际的 seqlen 跨度
    int32_t seqlen_block_count = (cur_seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    int32_t actual_seqlen = (seq_id >= seqlen_block_count - 1) ? cur_seqlen_q - seq_id * SEQLEN_PER_BLOCK: SEQLEN_PER_BLOCK;
    // if (threadIdx.x == 0) { printf("bidb: %d | seq_id: %d | bidh: %d | cur_seqlen_q: %d | actual_seqlen: %d\n", bidb, seq_id, bidh, cur_seqlen_q, actual_seqlen); }
    if (actual_seqlen <= 0 or seq_id * SEQLEN_PER_BLOCK > cur_seqlen_q) return;
    // 根据 batch id, 获取这个 Batch 的起始地址, 注意的转置都是在 batch 内部的, 不同 batch 的内容不会相互影响
    index_t batch_offset = split[bidb] * num_heads * kHeadDim;
    // 获取当前 head 在这个 batch 内的读取地址
    index_t head_offset = bidh * cur_seqlen_q * kHeadDim;
    // 得到读取的指针
    uint16_t* read_ptr = reinterpret_cast<uint16_t*>(input) + (batch_offset + head_offset + seq_offset);
    // 得到写出的指针
    uint16_t* write_ptr = reinterpret_cast<uint16_t*>(output) + (batch_offset + seq_id * SEQLEN_PER_BLOCK * num_heads * kHeadDim + bidh * kHeadDim);
    // 循环取 seqlen 次数据
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit = min(actual_seqlen - 1, fetch);
        // 当前 head 的数据内 seqlen x kHeadDim, 读取 seqlen 次, 每个 64 个线程读取 192 个 Half
        int32_t block_offset  = seqlen_limit * kHeadDim;
        index_t write_block_offset = seqlen_limit * num_heads * kHeadDim;
        int32_t thread_offset = threadIdx.x * 4;
        vec2_int data = *(vec2_int*)(read_ptr + block_offset + thread_offset);
        *(vec2_int*)(write_ptr + write_block_offset + thread_offset) = data;
    }
}



#endif