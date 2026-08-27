#ifdef BUILD_FA_PERMUTE
#include <hip/hip_runtime.h>
#include "../flash_fwd_permute_hdim128.h"


/* 最早的版本使用如下两个 kernel, 问题主要出在读写冲突, 但是使用 split 版本之后, 读写次数是固定的、可展开的, 给了编译器很大的优化空间

// 决定任务划分
dim3 block(64);
// 如果 numhead 是 4 的倍数, 而且 numhead 切分后保证一定的 CU 利用率, 则每个 block 处理 4x128 的内容
PERMUTE_DWORD_SWITCH((num_heads % 4 == 0) and (batch_size * (num_heads / 4) > 80 * 4 * 2), DWORD_PER_TX, [&] {
    dim3 grid(num_heads / DWORD_PER_TX, batch_size); // num_heads 放 grid.x 位置, 下发的 TG 访问地址更接近, 对 utcl1 更友好
    // 启动任务, 转置 q
    flash_fwd_varlen_permute_bshd2bhsd<128, DWORD_PER_TX><<<grid, block, 0, stream>>>(
        q_output.data_ptr(),
        query.data_ptr(),
        split_sizes.data_ptr(),
        query.stride(1),
        num_heads,
        head_dim
    );
    // 再转置 kv, 更新任务划分
    grid.x = num_heads_kv / DWORD_PER_TX;
    flash_fwd_varlen_permute_bshd2bhsd<128, DWORD_PER_TX><<<grid, block, 0, stream>>>(
        k_output.data_ptr(),
        key.data_ptr(),
        split_sizes.data_ptr(),
        key.stride(1),
        num_heads_kv,
        head_dim
    );
    flash_fwd_varlen_permute_bshd2bhsd<128, DWORD_PER_TX><<<grid, block, 0, stream>>>(
        v_output.data_ptr(),
        value.data_ptr(),
        split_sizes.data_ptr(),
        value.stride(1),
        num_heads_kv,
        head_dim
    );
});

*/

template<>
__global__ void flash_fwd_varlen_permute_bshd2bhsd<128, 1, 0>(
        void* output, void* query, void* split_sizes, int64_t head_stride, int32_t num_heads, int real_headdim) {
    using index_t = int64_t;
    constexpr int32_t head_dim = 128;
    // 获取 batch id, 以一个 1 x seqlen_q x num_heads x head_dim 为一个单位, 做变换
    int32_t bidb = blockIdx.y;
    // 获取第几个 head
    int32_t bidh = blockIdx.x;
    // 根据 batch id 计算整个 batch 所有头的起始地址; 需要注意的是, 输入连续时, head_stride = num_heads * head_dim; 输入不连续时, 就不是
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    index_t batch_offset = split[bidb] * head_stride;
    // 得到当前的 seqlen_q, 决定要取多少次数据
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 一行 128 个 Half, 对应 64 个 dword, 每个线程只需要读取 1 个 dword, 所以采用 float 这种 32 比特的数据类型
    float *read_ptr  = reinterpret_cast<float*>(query);
    // 同样地, 写出去的时候, 每个线程只需要写出 1 个 dword, 也用 float 数据类型
    // 写的时候, num_head 在前面, seqlen_q x head_dim 在后面, 因此按照 head id 来写出
    index_t write_head_offset = num_heads * split[bidb] * head_dim/*batch_offset, 输出必然连续*/ + bidh * cur_seqlen_q * head_dim;
    float   *write_ptr = reinterpret_cast<float*>(output) + (write_head_offset >> 1);
    // 开始循环读取和写出 seqlen_q 次, 每个 block(warp, 64threads) 写一行 128 个 half
    for (int32_t fetch = 0; fetch < cur_seqlen_q; ++fetch) {
        // 每个 block 64 个线程读取的起始地址
        index_t block_offset = batch_offset + bidh * head_dim + fetch * head_stride;
        // 每个线程读取的起始地址, 每个线程读取 2 个 half
        int32_t thread_offset = threadIdx.x << 1;
        // block 地址 + thread 地址, >> 1 是获取 dword 偏移量
        float content = read_ptr[(block_offset + thread_offset) >> 1];
        // 循环 seqlen_q 次, 每次间隔一行 128 个 half
        // 这里写出的时候可以用 global_store_dwordx4 优化
        write_ptr[(fetch * head_dim + thread_offset) >> 1] = content;
    }
}



template<>
__global__ void flash_fwd_varlen_permute_bshd2bhsd<128, 4, 0>(
        void* output, void* query, void* split_sizes, int64_t head_stride, int32_t num_heads, int real_headdim) {
    using index_t   = int64_t;
    using vec4_fp32 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    using vec2_fp32 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
    using vec4_uint = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
    constexpr int32_t head_dim = 128;
    // 获取 batch id, 以一个 1 x seqlen_q x num_heads x head_dim 为一个单位, 做变换
    int32_t bidb = blockIdx.y;
    // 获取第几个 head
    int32_t bidh = blockIdx.x;
    // 根据 batch id 计算整个 batch 所有头的起始地址; 需要注意的是, 输入连续时, head_stride = num_heads * head_dim; 输入不连续时, 就不是
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    index_t batch_offset = split[bidb] * head_stride;
    // 得到当前的 seqlen_q, 决定要取多少次数据
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 一行 128 个 Half, 对应 64 个 dword, 每个线程只需要读取 1 个 dword, 所以采用 float 这种 32 比特的数据类型
    float* read_ptr = reinterpret_cast<float*>(query) + (batch_offset >> 1);
    // 配置读指针的 buffer resource
    vec4_uint read_buffer;
    *(uint64_t*)&read_buffer = reinterpret_cast<uint64_t>(read_ptr);
    read_buffer[2] = 0x80000000;
    read_buffer[3] = 0x00020000;
    // 写的时候, num_head 在前面, seqlen_q x head_dim 在后面, 因此按照 head id 来写出
    index_t write_head_offset = num_heads * split[bidb] * head_dim/*batch_offset, 输出必然连续*/ + bidh * 4 * cur_seqlen_q * head_dim;
    float   *write_ptr = reinterpret_cast<float*>(output) + (write_head_offset >> 1);
    // 计算一些需要的 lane 下标
    int32_t lane_id = threadIdx.x;
    int32_t lane_id_row = lane_id >> 4;
    int32_t lane_id_col = lane_id % 16;
    // 需要用 lds, 至少需要 4x128 个 Half
    __shared__ float lds[1024];
    // 开始循环读取和写出 seqlen_q 次, 每个 block(warp, 64threads) 写一行 128 个 half
    for (int32_t fetch = 0; fetch < cur_seqlen_q; ++fetch) {
        // 每个 block 64 个线程读取的起始地址
        int32_t block_offset = bidh * 4 * head_dim + fetch * head_stride;
        // 接下来, 这个 block 要读取 4x128 的内容, 15 个线程读取一行 128 个 half(这里写死了 head_dim = 128), 每个线程读取 8 个 half
        int32_t thread_offset = lane_id_row * 128 + lane_id_col * 8;
        // block 地址 + thread 地址, << 1 是获取偏移的字节数, 写到 lds 是为了转置一下
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
        // 写到 lds 不需要同步, 因为只有一个 wave
        // 循环 seqlen_q 次, 每次间隔 4 x 128 个 half, 需要写 4 次
        vec2_fp32 data0 = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float*)lds + lane_id, 0, 64, false);
        vec2_fp32 data1 = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float*)lds + lane_id + 128, 0, 64, false);
        write_ptr[(fetch * head_dim + (lane_id << 1) + 0 * cur_seqlen_q * head_dim) >> 1] = data0[0];
        write_ptr[(fetch * head_dim + (lane_id << 1) + 1 * cur_seqlen_q * head_dim) >> 1] = data0[1];
        write_ptr[(fetch * head_dim + (lane_id << 1) + 2 * cur_seqlen_q * head_dim) >> 1] = data1[0];
        write_ptr[(fetch * head_dim + (lane_id << 1) + 3 * cur_seqlen_q * head_dim) >> 1] = data1[1];
    }
}


/* 默认的 varlen permute 方法, 优点:
    1. 实现简单
    2. 编译期循环可展开, 能优化读写冲突
   后续的优化点:
    1. 拖尾效应, 可以效仿 PA, 放到最后一个 block 上, 预估成效不高, 因为默认 block 32 的 seqlen 长度, overhead 不是很高
    2. KV 的两次转置可以融合成一个新的 kernel
*/
template<>
__global__ void __launch_bounds__(64, 1) flash_fwd_varlen_permute_bshd2bhsd<128, 1, 32>(
        void* output, void* query, void* split_sizes, int64_t head_stride, int32_t num_heads, int real_headdim) {
    using index_t = int64_t;
    constexpr int32_t head_dim = 128;
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    // 获取 batch id, 以一个 1 x seqlen_q x num_heads x head_dim 为一个单位, 做变换
    int32_t bidb = blockIdx.z;
    // 获取 seqlen 方向第几个块
    int32_t seq_id = blockIdx.y;
    // 获取第几个 head
    int32_t bidh = blockIdx.x;
    // 根据 batch id 计算整个 batch 所有头的起始地址; 需要注意的是, 输入连续时, head_stride = num_heads * head_dim; 输入不连续时, 就不是
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    index_t batch_offset = split[bidb] * head_stride;
    // 得到当前的 seqlen_q, 决定要取多少次数据
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 得到当前处理的小块从哪里开始取数据
    index_t seq_offset = seq_id * SEQLEN_PER_BLOCK * head_stride;
    // 获取当前处理的小块实际的 seqlen 跨度
    int32_t seqlen_block_count = (cur_seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    int32_t actual_seqlen = (seq_id >= seqlen_block_count - 1) ? cur_seqlen_q - seq_id * SEQLEN_PER_BLOCK: SEQLEN_PER_BLOCK;
    // if (threadIdx.x == 0) { printf("bidb: %d | seq_id: %d | bidh: %d | cur_seqlen_q: %d | actual_seqlen: %d\n", bidb, seq_id, bidh, cur_seqlen_q, actual_seqlen); }
    if (actual_seqlen <= 0 or seq_id * SEQLEN_PER_BLOCK > cur_seqlen_q) return;
    // 一行 128 个 Half, 对应 64 个 dword, 每个线程只需要读取 1 个 dword, 所以采用 float 这种 32 比特的数据类型
    float *read_ptr  = reinterpret_cast<float*>(query);
    // 写的时候, num_head 在前面, seqlen_q x head_dim 在后面, 因此按照 head id 来写出
    index_t write_head_offset = num_heads * split[bidb] * head_dim/*batch_offset, 输出必然连续*/ + bidh * cur_seqlen_q * head_dim + seq_id * SEQLEN_PER_BLOCK * head_dim;
    float   *write_ptr = reinterpret_cast<float*>(output) + (write_head_offset >> 1);
    // 使用寄存器缓冲一部分数据
    float registers_buffer[SEQLEN_PER_BLOCK];
    // 开始循环读取和写出 seqlen_q 次, 每个 block(warp, 64threads) 写一行 128 个 half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit = min(actual_seqlen - 1, fetch);
        // 每个 block 64 个线程读取的起始地址
        index_t block_offset = batch_offset + seq_offset + bidh * head_dim + seqlen_limit * head_stride;
        // 每个线程读取的起始地址, 每个线程读取 2 个 half
        int32_t thread_offset = threadIdx.x << 1;
        // block 地址 + thread 地址, >> 1 是获取 dword 偏移量
        registers_buffer[fetch] = read_ptr[(block_offset + thread_offset) >> 1];
    }
    __builtin_amdgcn_sched_barrier(0);

    // 把所有 buffer_load 指令下发之后, 才开始写
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit = min(actual_seqlen - 1, fetch);
        int32_t thread_offset = threadIdx.x << 1;
        // 循环 seqlen_q 次, 每次间隔一行 128 个 half
        write_ptr[(seqlen_limit * head_dim + thread_offset) >> 1] = registers_buffer[fetch];
    }
}


#if 1
template<>
__global__ void flash_fwd_varlen_permute_bshd2bhsd<128, 4, 32>(
        void* output, void* query, void* split_sizes, int64_t head_stride, int32_t num_heads, int real_headdim) {
    using index_t   = int64_t;
    using vec4_fp32 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    using vec2_fp32 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
    using vec4_uint = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
    constexpr int32_t head_dim = 128;
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    // 获取 batch id, 以一个 1 x seqlen_q x num_heads x head_dim 为一个单位, 做变换
    int32_t bidb = blockIdx.z;
    // 获取 seqlen 方向第几个块
    int32_t seq_id = blockIdx.y;
    // 获取第几个 head
    int32_t bidh = blockIdx.x;
    // 根据 batch id 计算整个 batch 所有头的起始地址; 需要注意的是, 输入连续时, head_stride = num_heads * head_dim; 输入不连续时, 就不是
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    index_t batch_offset = split[bidb] * head_stride;
    // 得到当前的 seqlen_q, 决定要取多少次数据
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 得到当前处理的小块从哪里开始取数据
    index_t seq_offset = seq_id * SEQLEN_PER_BLOCK * head_stride;
    // 获取当前处理的小块实际的 seqlen 跨度
    int32_t seqlen_block_count = (cur_seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    int32_t actual_seqlen = (seq_id >= seqlen_block_count - 1) ? cur_seqlen_q - seq_id * SEQLEN_PER_BLOCK: SEQLEN_PER_BLOCK;
    // if (threadIdx.x == 0) { printf("bidb: %d | seq_id: %d | bidh: %d | cur_seqlen_q: %d | actual_seqlen: %d\n", bidb, seq_id, bidh, cur_seqlen_q, actual_seqlen); }
    if (actual_seqlen <= 0 or seq_id * SEQLEN_PER_BLOCK > cur_seqlen_q) return;
    // 一行 128 个 Half, 对应 64 个 dword, 每个线程只需要读取 1 个 dword, 所以采用 float 这种 32 比特的数据类型
    float *read_ptr  = reinterpret_cast<float*>(query) + ((batch_offset + seq_offset) >> 1);
    // 配置读指针的 buffer resource
    vec4_uint read_buffer;
    *(uint64_t*)&read_buffer = reinterpret_cast<uint64_t>(read_ptr);
    read_buffer[2] = 0x80000000;
    read_buffer[3] = 0x00020000;
    // 写的时候, num_head 在前面, seqlen_q x head_dim 在后面, 因此按照 head id 来写出
    index_t write_head_offset = num_heads * split[bidb] * head_dim/*batch_offset, 输出必然连续*/ + bidh * 4 * cur_seqlen_q * head_dim + seq_id * SEQLEN_PER_BLOCK * head_dim;
    float   *write_ptr = reinterpret_cast<float*>(output) + (write_head_offset >> 1);
    // 计算一些需要的 lane 下标
    int32_t lane_id = threadIdx.x;
    int32_t lane_id_row = lane_id >> 4;
    int32_t lane_id_col = lane_id % 16;
    // 需要用 lds, 至少需要 4x128 个 Half
    __shared__ float lds[1024];
    // 开始循环读取和写出 seqlen_q 次, 每个 block(warp, 64threads) 写一行 128 个 half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit = min(actual_seqlen - 1, fetch);
        // 每个 block 64 个线程读取的起始地址
        int32_t block_offset = bidh * 4 * head_dim + seqlen_limit * head_stride;
        // 接下来, 这个 block 要读取 4x128 的内容, 15 个线程读取一行 128 个 half(这里写死了 head_dim = 128), 每个线程读取 8 个 half
        int32_t thread_offset = lane_id_row * 128 + lane_id_col * 8;
        // block 地址 + thread 地址, << 1 是获取偏移的字节数, 写到 lds 是为了转置一下
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
        // 写到 lds 不需要同步, 因为只有一个 wave
        // 循环 seqlen_q 次, 每次间隔 4 x 128 个 half, 需要写 4 次
        vec2_fp32 data0 = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float*)lds + lane_id, 0, 64, false);
        vec2_fp32 data1 = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float*)lds + lane_id + 128, 0, 64, false);
        write_ptr[(seqlen_limit * head_dim + (lane_id << 1) + 0 * cur_seqlen_q * head_dim) >> 1] = data0[0];
        write_ptr[(seqlen_limit * head_dim + (lane_id << 1) + 1 * cur_seqlen_q * head_dim) >> 1] = data0[1];
        write_ptr[(seqlen_limit * head_dim + (lane_id << 1) + 2 * cur_seqlen_q * head_dim) >> 1] = data1[0];
        write_ptr[(seqlen_limit * head_dim + (lane_id << 1) + 3 * cur_seqlen_q * head_dim) >> 1] = data1[1];
    }
}

#else
template<>
__global__ void __launch_bounds__(64, 1) flash_fwd_varlen_permute_bshd2bhsd<128, 4, 32>(
        void* output, void* query, void* split_sizes, int64_t head_stride, int32_t num_heads, int real_headdim) {
    using index_t   = int64_t;
    using vec4_fp32 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    using vec2_fp32 = __attribute__((__vector_size__(2 * sizeof(float)))) float;
    using vec4_uint = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
    constexpr int32_t head_dim = 128;
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    // 获取 batch id, 以一个 1 x seqlen_q x num_heads x head_dim 为一个单位, 做变换
    int32_t bidb = blockIdx.z;
    // 获取 seqlen 方向第几个块
    int32_t seq_id = blockIdx.y;
    // 获取第几个 head
    int32_t bidh = blockIdx.x;
    // 根据 batch id 计算整个 batch 所有头的起始地址; 需要注意的是, 输入连续时, head_stride = num_heads * head_dim; 输入不连续时, 就不是
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    index_t batch_offset = split[bidb] * head_stride;
    // 得到当前的 seqlen_q, 决定要取多少次数据
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 得到当前处理的小块从哪里开始取数据
    index_t seq_offset = seq_id * SEQLEN_PER_BLOCK * head_stride;
    // 获取当前处理的小块实际的 seqlen 跨度
    int32_t seqlen_block_count = (cur_seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    int32_t actual_seqlen = (seq_id >= seqlen_block_count - 1) ? cur_seqlen_q - seq_id * SEQLEN_PER_BLOCK: SEQLEN_PER_BLOCK;
    // if (threadIdx.x == 0) { printf("bidb: %d | seq_id: %d | bidh: %d | cur_seqlen_q: %d | actual_seqlen: %d\n", bidb, seq_id, bidh, cur_seqlen_q, actual_seqlen); }
    if (actual_seqlen <= 0 or seq_id * SEQLEN_PER_BLOCK > cur_seqlen_q) return;
    // 一行 128 个 Half, 对应 64 个 dword, 每个线程只需要读取 1 个 dword, 所以采用 float 这种 32 比特的数据类型
    float *read_ptr  = reinterpret_cast<float*>(query) + ((batch_offset + seq_offset) >> 1);
    // 配置读指针的 buffer resource
    vec4_uint read_buffer;
    *(uint64_t*)&read_buffer = reinterpret_cast<uint64_t>(read_ptr);
    read_buffer[2] = 0x80000000;
    read_buffer[3] = 0x00020000;
    // 写的时候, num_head 在前面, seqlen_q x head_dim 在后面, 因此按照 head id 来写出
    index_t write_head_offset = num_heads * split[bidb] * head_dim/*batch_offset, 输出必然连续*/ + bidh * 4 * cur_seqlen_q * head_dim + seq_id * SEQLEN_PER_BLOCK * head_dim;
    float   *write_ptr = reinterpret_cast<float*>(output) + (write_head_offset >> 1);
    // 计算一些需要的 lane 下标
    int32_t lane_id = threadIdx.x;
    int32_t lane_id_row = lane_id >> 4;
    int32_t lane_id_col = lane_id % 16;
    // 需要用 lds, 至少需要 4x128 个 Half, 为了让 buffer_load 与 ds_read 相互重叠, 需要申请多倍的 lds 空间
    __shared__ float lds[(4 * 128 >> 1) * SEQLEN_PER_BLOCK];
    vec2_fp32 registers_buffer[2 * SEQLEN_PER_BLOCK];
    // 开始循环读取和写出 seqlen_q 次, 每个 block(warp, 64threads) 写一行 128 个 half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit = min(actual_seqlen - 1, fetch);
        // 每个 block 64 个线程读取的起始地址
        int32_t block_offset = bidh * 4 * head_dim + seqlen_limit * head_stride;
        // 接下来, 这个 block 要读取 4x128 的内容, 15 个线程读取一行 128 个 half(这里写死了 head_dim = 128), 每个线程读取 8 个 half
        int32_t thread_offset = lane_id_row * 128 + lane_id_col * 8;
        // block 地址 + thread 地址, << 1 是获取偏移的字节数, 写到 lds 是为了转置一下
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
            int m0_offset = reinterpret_cast<size_t>(lds) + (fetch * 256 << 2);
            int offset_v = (block_offset + thread_offset) << 1;
            asm volatile(
                "s_mov_b32 m0, %1 \n\t"
                "buffer_load_dwordx4 %0, %2, %3, offen  offset:0, lds \n"
                :: "v"(offset_v), "s"(m0_offset), "s"(read_buffer), "s"(0)
                :);
            // __builtin_hcu_raw_buffer_load_lds(
            //     read_buffer,
            //     lds + fetch * 256,
            //     16,
            //     (block_offset + thread_offset) << 1, /* v_offset */
            //     0, /* s_offset */
            //     0, /* immediate offset, instruction offset */
            //     0  /* auxilariy data| bit 0: glc, bit 1: slc, bit 2: dlc, bit 3: cache swizzle */
            // );
        #else
            *(vec4_fp32*)(lds + fetch * 256 + lane_id * 4) = *(vec4_fp32*)(read_ptr + ((block_offset + thread_offset) >> 1));
        #endif
    }

    // 把所有的 buffer_load 指令下发之后, 再从 lds 开始读取
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_waitcnt vmcnt(%0)\n" :: "B"(SEQLEN_PER_BLOCK - fetch - 1));
            __builtin_amdgcn_sched_barrier(0);
        #endif
        // 循环 seqlen_q 次, 每次间隔 4 x 128 个 half, 需要写 4 次
        registers_buffer[fetch * 2]     = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float*)lds + fetch * 256 + lane_id, 0, 64, false);
        registers_buffer[fetch * 2 + 1] = __builtin_hcu_ds_read2_f32((__attribute__((address_space(3))) float*)lds + fetch * 256 + lane_id + 128, 0, 64, false);
    }
    __builtin_amdgcn_sched_barrier(0);

    // 把所有的 ds_read 指令下发且读取到寄存器后, 写出 global memory
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit = min(actual_seqlen - 1, fetch);
        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
            // 计算固定的偏移, 字节数目
            int32_t v_addr = (seqlen_limit * head_dim << 1) + (lane_id << 2);
            // 循环 seqlen_q 次, 每次间隔 4 x 128 个 half, 需要写 4 次
            asm volatile("global_store_dword %0, %1, %2\n" :: "v"(v_addr), "v"(registers_buffer[fetch * 2][0]), "s"(write_ptr));
            v_addr += cur_seqlen_q * head_dim * 2;
            asm volatile("global_store_dword %0, %1, %2\n" :: "v"(v_addr), "v"(registers_buffer[fetch * 2][1]), "s"(write_ptr));
            v_addr += cur_seqlen_q * head_dim * 2;
            asm volatile("global_store_dword %0, %1, %2\n" :: "v"(v_addr), "v"(registers_buffer[fetch * 2 + 1][0]), "s"(write_ptr));
            v_addr += cur_seqlen_q * head_dim * 2;
            asm volatile("global_store_dword %0, %1, %2\n" :: "v"(v_addr), "v"(registers_buffer[fetch * 2 + 1][1]), "s"(write_ptr));
        #else
            int32_t write_offset = (seqlen_limit * head_dim >> 1) + lane_id;
            write_ptr[(0 * cur_seqlen_q * head_dim >> 1) + write_offset] = registers_buffer[fetch * 2][0];
            write_ptr[(1 * cur_seqlen_q * head_dim >> 1) + write_offset] = registers_buffer[fetch * 2][1];
            write_ptr[(2 * cur_seqlen_q * head_dim >> 1) + write_offset] = registers_buffer[fetch * 2 + 1][0];
            write_ptr[(3 * cur_seqlen_q * head_dim >> 1) + write_offset] = registers_buffer[fetch * 2 + 1][1];
        #endif
    }
}

#endif



template<>
__global__ void __launch_bounds__(64, 1) flash_fwd_varlen_permute_bshd2bhsd<192, 1, 32>(
        void* output, void* query, void* split_sizes, int64_t head_stride, int32_t num_heads, int real_headdim) {
    using index_t = int64_t;
    constexpr int32_t head_dim = 192;
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    // 获取 batch id, 以一个 1 x seqlen_q x num_heads x head_dim 为一个单位, 做变换
    int32_t bidb = blockIdx.z;
    // 获取 seqlen 方向第几个块
    int32_t seq_id = blockIdx.y;
    // 获取第几个 head
    int32_t bidh = blockIdx.x;
    // 根据 batch id 计算整个 batch 所有头的起始地址; 需要注意的是, 输入连续时, head_stride = num_heads * head_dim; 输入不连续时, 就不是
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    index_t batch_offset = split[bidb] * head_stride;
    // 得到当前的 seqlen_q, 决定要取多少次数据
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 得到当前处理的小块从哪里开始取数据
    index_t seq_offset = seq_id * SEQLEN_PER_BLOCK * head_stride;
    // 获取当前处理的小块实际的 seqlen 跨度
    int32_t seqlen_block_count = (cur_seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    int32_t actual_seqlen = (seq_id >= seqlen_block_count - 1) ? cur_seqlen_q - seq_id * SEQLEN_PER_BLOCK: SEQLEN_PER_BLOCK;
    // if (threadIdx.x == 0) { printf("bidb: %d | seq_id: %d | bidh: %d | cur_seqlen_q: %d | actual_seqlen: %d\n", bidb, seq_id, bidh, cur_seqlen_q, actual_seqlen); }
    if (actual_seqlen <= 0 or seq_id * SEQLEN_PER_BLOCK > cur_seqlen_q) return;
    // 一行 192 个 Half, 每个线程只需要读取 3 个 half, 所以采用 uint16_t 这种 16 比特的数据类型
    uint16_t *read_ptr  = reinterpret_cast<uint16_t*>(query);
    // 写的时候, num_head 在前面, seqlen_q x head_dim 在后面, 因此按照 head id 来写出
    index_t write_head_offset = num_heads * split[bidb] * head_dim/*batch_offset, 输出必然连续*/ + bidh * cur_seqlen_q * head_dim + seq_id * SEQLEN_PER_BLOCK * head_dim;
    uint16_t   *write_ptr = reinterpret_cast<uint16_t*>(output) + write_head_offset;
    // 使用寄存器缓冲一部分数据
    uint16_t registers_buffer[SEQLEN_PER_BLOCK * 3];
    // 开始循环读取和写出 seqlen_q 次, 每个 block(warp, 64threads) 写一行 192 个 half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit  = min(actual_seqlen - 1, fetch);
        // 每个 block 64 个线程读取的起始地址
        index_t block_offset  = batch_offset + seq_offset + bidh * head_dim + seqlen_limit * head_stride;
        // 每个线程读取的起始地址, 每个线程读取 3 个 half
        int32_t thread_offset = threadIdx.x;
        // block 地址 + thread 地址
        #pragma unroll
        for (int elem = 0; elem < 3; ++elem) {
            registers_buffer[fetch + elem * SEQLEN_PER_BLOCK] = read_ptr[block_offset + thread_offset + elem * 64];
        }
    }
    __builtin_amdgcn_sched_barrier(0);

    // 把所有 buffer_load 指令下发之后, 才开始写
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit  = min(actual_seqlen - 1, fetch);
        int32_t thread_offset = threadIdx.x;
        // 循环 seqlen_q 次, 每次间隔一行 192 个 half
        #pragma unroll
        for (int elem = 0; elem < 3; ++elem) {
            write_ptr[seqlen_limit * head_dim + thread_offset + elem * 64] = registers_buffer[fetch + elem * SEQLEN_PER_BLOCK];
        }
    }
}

template<>
__global__ void __launch_bounds__(64, 1) flash_fwd_varlen_permute_bshd2bhsd<192, 4, 32>(
        void* output, void* query, void* split_sizes, int64_t head_stride, int32_t num_heads, int real_headdim) {}



template<>
__global__ void __launch_bounds__(64, 1) flash_fwd_varlen_permute_bshd2bhsd<0, 1, 32>(
        void* output, void* query, void* split_sizes, int64_t head_stride, int32_t num_heads, int real_headdim) {
    using index_t = int64_t;
    // constexpr int32_t kHeadDim         = 232;
    constexpr int32_t HEADDIM_LIMIT    = 256;
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    // 获取 batch id, 以一个 1 x seqlen_q x num_heads x real_headdim 为一个单位, 做变换
    int32_t bidb = blockIdx.z;
    // 获取 seqlen 方向第几个块
    int32_t seq_id = blockIdx.y;
    // 获取第几个 head
    int32_t bidh = blockIdx.x;
    // 根据 batch id 计算整个 batch 所有头的起始地址; 需要注意的是, 输入连续时, head_stride = num_heads * real_headdim; 输入不连续时, 就不是
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    index_t batch_offset = split[bidb] * head_stride;
    // 得到当前的 seqlen_q, 决定要取多少次数据
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 得到当前处理的小块从哪里开始取数据
    index_t seq_offset = seq_id * SEQLEN_PER_BLOCK * head_stride;
    // 获取当前处理的小块实际的 seqlen 跨度
    int32_t seqlen_block_count = (cur_seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    int32_t actual_seqlen = (seq_id >= seqlen_block_count - 1) ? cur_seqlen_q - seq_id * SEQLEN_PER_BLOCK: SEQLEN_PER_BLOCK;
    // if (threadIdx.x == 0) { printf("bidb: %d | seq_id: %d | bidh: %d | cur_seqlen_q: %d | actual_seqlen: %d\n", bidb, seq_id, bidh, cur_seqlen_q, actual_seqlen); }
    if (actual_seqlen <= 0 or seq_id * SEQLEN_PER_BLOCK > cur_seqlen_q) return;
    // 一行 real_headdim 个 Half, 每个线程只需要读取 real_headdim / 64 个 half, 所以采用 uint16_t 这种 16 比特的数据类型
    uint16_t *read_ptr  = reinterpret_cast<uint16_t*>(query);
    // 写的时候, num_head 在前面, seqlen_q x real_headdim 在后面, 因此按照 head id 来写出
    index_t write_head_offset = num_heads * split[bidb] * real_headdim/*batch_offset, 输出必然连续*/ + bidh * cur_seqlen_q * real_headdim + seq_id * SEQLEN_PER_BLOCK * real_headdim;
    uint16_t   *write_ptr = reinterpret_cast<uint16_t*>(output) + write_head_offset;
    // 使用寄存器缓冲一部分数据
    uint16_t registers_buffer[SEQLEN_PER_BLOCK * (HEADDIM_LIMIT / 64)];
    // 开始循环读取和写出 seqlen_q 次, 每个 block(warp, 64threads) 写一行 real_headdim 个 half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit  = min(actual_seqlen - 1, fetch);
        // 每个 block 64 个线程读取的起始地址
        index_t block_offset  = batch_offset + seq_offset + bidh * real_headdim + seqlen_limit * head_stride;
        // block 地址 + thread 地址
        #pragma unroll
        for (int elem = 0; elem < (HEADDIM_LIMIT / 64); ++elem) {
            // 每个线程读取的起始地址, 每个线程读取 real_headdim / 64 个 half
            int32_t thread_offset = min(real_headdim, threadIdx.x + elem * 64);
            registers_buffer[fetch + elem * SEQLEN_PER_BLOCK] = read_ptr[block_offset + thread_offset];
        }
    }
    __builtin_amdgcn_sched_barrier(0);

    // 把所有 buffer_load 指令下发之后, 才开始写
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit  = min(actual_seqlen - 1, fetch);
        // 循环 seqlen_q 次, 每次间隔一行 real_headdim 个 half
        #pragma unroll
        for (int elem = 0; elem < (HEADDIM_LIMIT / 64); ++elem) {
            int32_t thread_offset = threadIdx.x + elem * 64;
            if (thread_offset < real_headdim) {
                // 这里有 bug, 同一个 wave 多个线程写同一个数据, 会导致结果错误, 需要定位下是不是硬件的问题
                write_ptr[seqlen_limit * real_headdim + thread_offset] = registers_buffer[fetch + elem * SEQLEN_PER_BLOCK];
            }
        }
    }
}


template<>
__global__ void __launch_bounds__(64, 1) flash_fwd_varlen_permute_bshd2bhsd<0, 4, 32>(
        void* output, void* query, void* split_sizes, int64_t head_stride, int32_t num_heads, int real_headdim) {}


template<>
__global__ void __launch_bounds__(64, 1) flash_fwd_varlen_permute_bshd2bhsd<64, 1, 32>(
        void* output, void* query, void* split_sizes, int64_t head_stride, int32_t num_heads, int real_headdim) {
    using index_t = int64_t;
    constexpr int32_t kHeadDim         = 64;
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    // 获取 batch id, 以一个 1 x seqlen_q x num_heads x kHeadDim 为一个单位, 做变换
    int32_t bidb = blockIdx.z;
    // 获取 seqlen 方向第几个块
    int32_t seq_id = blockIdx.y;
    // 获取第几个 head
    int32_t bidh = blockIdx.x;
    // 根据 batch id 计算整个 batch 所有头的起始地址; 需要注意的是, 输入连续时, head_stride = num_heads * kHeadDim; 输入不连续时, 就不是
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    index_t batch_offset = split[bidb] * head_stride;
    // 得到当前的 seqlen_q, 决定要取多少次数据
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 得到当前处理的小块从哪里开始取数据
    index_t seq_offset = seq_id * SEQLEN_PER_BLOCK * head_stride;
    // 获取当前处理的小块实际的 seqlen 跨度
    int32_t seqlen_block_count = (cur_seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    int32_t actual_seqlen = (seq_id >= seqlen_block_count - 1) ? cur_seqlen_q - seq_id * SEQLEN_PER_BLOCK: SEQLEN_PER_BLOCK;
    // if (threadIdx.x == 0) { printf("bidb: %d | seq_id: %d | bidh: %d | cur_seqlen_q: %d | actual_seqlen: %d\n", bidb, seq_id, bidh, cur_seqlen_q, actual_seqlen); }
    if (actual_seqlen <= 0 or seq_id * SEQLEN_PER_BLOCK > cur_seqlen_q) return;
    // 一行 64 个 Half, 每个线程只需要读取 1 个 half, 所以采用 uint16_t 这种 16 比特的数据类型
    uint16_t *read_ptr  = reinterpret_cast<uint16_t*>(query);
    // 写的时候, num_head 在前面, seqlen_q x kHeadDim 在后面, 因此按照 head id 来写出
    index_t write_head_offset = num_heads * split[bidb] * kHeadDim/*batch_offset, 输出必然连续*/ + bidh * cur_seqlen_q * kHeadDim + seq_id * SEQLEN_PER_BLOCK * kHeadDim;
    uint16_t   *write_ptr = reinterpret_cast<uint16_t*>(output) + write_head_offset;
    // 使用寄存器缓冲一部分数据
    uint16_t registers_buffer[SEQLEN_PER_BLOCK];
    // 开始循环读取和写出 seqlen_q 次, 每个 block(warp, 64threads) 写一行 64 个 half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit  = min(actual_seqlen - 1, fetch);
        // 每个 block 64 个线程读取的起始地址
        index_t block_offset  = batch_offset + seq_offset + bidh * kHeadDim + seqlen_limit * head_stride;
        // 每个线程读取的起始地址, 每个线程读取 1 个 half
        int32_t thread_offset = min(kHeadDim, threadIdx.x);
        registers_buffer[fetch] = read_ptr[block_offset + thread_offset];
    }
    __builtin_amdgcn_sched_barrier(0);

    // 把所有 buffer_load 指令下发之后, 才开始写
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit  = min(actual_seqlen - 1, fetch);
        // 循环 seqlen_q 次, 每次间隔一行 192 个 half
        int32_t thread_offset = threadIdx.x;
        // 这里有 bug, 同一个 wave 多个线程写同一个数据, 会导致结果错误, 需要定位下是不是硬件的问题
        write_ptr[seqlen_limit * kHeadDim + thread_offset] = registers_buffer[fetch];
    }
}


template<>
__global__ void __launch_bounds__(64, 1) flash_fwd_varlen_permute_bshd2bhsd<64, 4, 32>(
        void* output, void* query, void* split_sizes, int64_t head_stride, int32_t num_heads, int real_headdim) {}


template<>
__global__ void __launch_bounds__(64, 1) flash_fwd_varlen_permute_bshd2bhsd<256, 1, 32>(
        void* output, void* query, void* split_sizes, int64_t head_stride, int32_t num_heads, int real_headdim) {
    using index_t = int64_t;
    constexpr int32_t kHeadDim         = 256;
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    // 获取 batch id, 以一个 1 x seqlen_q x num_heads x kHeadDim 为一个单位, 做变换
    int32_t bidb = blockIdx.z;
    // 获取 seqlen 方向第几个块
    int32_t seq_id = blockIdx.y;
    // 获取第几个 head
    int32_t bidh = blockIdx.x;
    // 根据 batch id 计算整个 batch 所有头的起始地址; 需要注意的是, 输入连续时, head_stride = num_heads * real_headdim; 输入不连续时, 就不是
    int32_t *split = reinterpret_cast<int32_t*>(split_sizes);
    index_t batch_offset = split[bidb] * head_stride;
    // 得到当前的 seqlen_q, 决定要取多少次数据
    int32_t cur_seqlen_q = split[bidb + 1] - split[bidb];
    // 得到当前处理的小块从哪里开始取数据
    index_t seq_offset = seq_id * SEQLEN_PER_BLOCK * head_stride;
    // 获取当前处理的小块实际的 seqlen 跨度
    int32_t seqlen_block_count = (cur_seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    int32_t actual_seqlen = (seq_id >= seqlen_block_count - 1) ? cur_seqlen_q - seq_id * SEQLEN_PER_BLOCK: SEQLEN_PER_BLOCK;
    // if (threadIdx.x == 0) { printf("bidb: %d | seq_id: %d | bidh: %d | cur_seqlen_q: %d | actual_seqlen: %d\n", bidb, seq_id, bidh, cur_seqlen_q, actual_seqlen); }
    if (actual_seqlen <= 0 or seq_id * SEQLEN_PER_BLOCK > cur_seqlen_q) return;
    // 一行 real_headdim 个 Half, 每个线程只需要读取 real_headdim / 64 个 half, 所以采用 uint16_t 这种 16 比特的数据类型
    uint16_t *read_ptr  = reinterpret_cast<uint16_t*>(query);
    // 写的时候, num_head 在前面, seqlen_q x real_headdim 在后面, 因此按照 head id 来写出
    index_t write_head_offset = num_heads * split[bidb] * kHeadDim/*batch_offset, 输出必然连续*/ + bidh * cur_seqlen_q * kHeadDim + seq_id * SEQLEN_PER_BLOCK * kHeadDim;
    uint16_t   *write_ptr = reinterpret_cast<uint16_t*>(output) + write_head_offset;
    // 使用寄存器缓冲一部分数据
    uint16_t registers_buffer[SEQLEN_PER_BLOCK * (kHeadDim / 64)];
    // 开始循环读取和写出 seqlen_q 次, 每个 block(warp, 64threads) 写一行 real_headdim 个 half
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit  = min(actual_seqlen - 1, fetch);
        // 每个 block 64 个线程读取的起始地址
        index_t block_offset  = batch_offset + seq_offset + bidh * kHeadDim + seqlen_limit * head_stride;
        // block 地址 + thread 地址
        #pragma unroll
        for (int elem = 0; elem < (kHeadDim / 64); ++elem) {
            // 每个线程读取的起始地址, 每个线程读取 kHeadDim / 64 个 half
            int32_t thread_offset = threadIdx.x + elem * 64;
            registers_buffer[fetch + elem * SEQLEN_PER_BLOCK] = read_ptr[block_offset + thread_offset];
        }
    }
    __builtin_amdgcn_sched_barrier(0);

    // 把所有 buffer_load 指令下发之后, 才开始写
    #pragma unroll
    for (int32_t fetch = 0; fetch < SEQLEN_PER_BLOCK; ++fetch) {
        // 限制边界
        int32_t seqlen_limit  = min(actual_seqlen - 1, fetch);
        // 循环 seqlen_q 次, 每次间隔一行 real_headdim 个 half
        #pragma unroll
        for (int elem = 0; elem < (kHeadDim / 64); ++elem) {
            int32_t thread_offset = threadIdx.x + elem * 64;
            write_ptr[seqlen_limit * real_headdim + thread_offset] = registers_buffer[fetch + elem * SEQLEN_PER_BLOCK];
        }
    }
}


template<>
__global__ void __launch_bounds__(64, 1) flash_fwd_varlen_permute_bshd2bhsd<256, 4, 32>(
        void* output, void* query, void* split_sizes, int64_t head_stride, int32_t num_heads, int real_headdim) {}

#endif // end of BUILD_FA_PERMUTE