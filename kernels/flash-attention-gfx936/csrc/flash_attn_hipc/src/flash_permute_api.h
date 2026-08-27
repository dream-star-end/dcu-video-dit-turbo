#include <torch/nn/functional.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <vector>
#include <iostream>
// self
#include "static_switch.h"
#include "flash_permute_hdim128.h"
#include "flash_fwd_permute_hdim128.h"

#ifdef BUILD_FA_PERMUTE

std::vector<at::Tensor> varlen_fwd_permute_bshd2bhsd(
        at::Tensor &query,
        at::Tensor &key,
        at::Tensor &value,
        const at::Tensor split_sizes,
        const int32_t max_seqlen_qkv,
        const bool use_bshd_layout=false) {
    // 获取 size
    auto query_size = query.sizes();
    int beg_dim = 0;
    if (query_size.size() == 4) { beg_dim = 1; } // 默认只需要 [batch_seqlen, num_heads, head_dim] 这 3 个维度, 如果传进来 4 维, 取维度的时候改成 3 维需要的 size
    const int32_t batch_seqlen = query_size[beg_dim + 0];
    const int32_t num_heads    = query_size[beg_dim + 1];
    const int32_t head_dim_q   = query_size[beg_dim + 2];
    const int32_t batch_size   = split_sizes.sizes()[0] - 1;
    const int32_t num_heads_kv = key.sizes()[beg_dim + 1];
    const int32_t head_dim_k   = key.sizes()[beg_dim + 2];
    const int32_t head_dim_v   = value.sizes()[beg_dim + 2];
    // printf("varlen_fwd_permute_bshd2bhsd | batch_seqlen: %d\nnum_heads: %d\nnum_heads_kv: %d\nhead_dim: %d\nbatch_size: %d\n", batch_seqlen, num_heads, num_heads_kv, head_dim, batch_size);
    // 检查变量的合法性, 是不是 headdim 128 等
    TORCH_CHECK (head_dim_q == 128 or head_dim_q == 192 or (head_dim_q > 0 and head_dim_q <= 256), "For fa forward varlen permute bshd2bhsd, only head_dim_q = 128/192/<=256 is supported!");
    TORCH_CHECK (head_dim_k == 128 or head_dim_k == 192 or (head_dim_k > 0 and head_dim_k <= 256), "For fa forward varlen permute bshd2bhsd, only head_dim_k = 128/192/<=256 is supported!");
    TORCH_CHECK (head_dim_v == 128 or head_dim_v == 192 or (head_dim_v > 0 and head_dim_v <= 256), "For fa forward varlen permute bshd2bhsd, only head_dim_v = 128/192/<=256 is supported!");
    TORCH_CHECK (batch_size > 0, "For fa forward varlen permute bshd2bhsd, split_sizes have 2 elements at least!");
    TORCH_CHECK (sizeof(query.dtype()) == 2 and sizeof(key.dtype()) == 2 and sizeof(value.dtype()) == 2, "For fa forward varlen permute bshd2bhsd, only dtypes of 2 bytes are supported!");
    // 准备张量
    at::Tensor q_output, k_output, v_output, adjacent_memory;
    // 如果输入都是连续的, 使用最普通的计算流程
    // 获取数据类型和设备, 准备需要返回的张量
    auto opts = query.options();
    // 由于 MLA, KV 需要的显存也不一定一样大了
    q_output = use_bshd_layout ? query: torch::empty({batch_seqlen * num_heads, head_dim_q}, opts);
    k_output = torch::empty({batch_seqlen * num_heads_kv, head_dim_k}, opts);
    v_output = torch::empty({batch_seqlen * num_heads_kv, head_dim_v}, opts);
    // 获取当前执行流, 方便被 hipgraph 捕捉到
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // 根据 batch_size 来决定任务划分, 避免 batch_size 太小导致 CU 空闲, 至少每个 cu 4 个 SIMD 都利用上, 每个 SIMD 上跑 2 个 wave: 比如 batch_size * num_heads < 80 * 4 * 2
    // 但由于 spllit seqlen 可以做循环展开, 有较大优势, 因此默认对 seq 维度做切分
    // 后续可以根据 batch size 对 seqlen_qkv 做切分, 大 batch 用更大的 block, 减少 TG 的 dispatch
    // 如果 num_heads 是 4 的倍数, 每个 block 可以一次性处理 4x128 的数据, 使用 buffer_load_dwordx4 与连续读 256 bytes 的硬件特性
    PERMUTE_DWORD_SWITCH(num_heads_kv % 4 == 0 and head_dim_q == 128 and head_dim_k == 128, DWORD_PER_TX, [&]{
        const int32_t SEQLEN_PER_BLOCK = 32;
        int32_t seqlen_split = (max_seqlen_qkv + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
        dim3 block(64);
        dim3 grid(num_heads / DWORD_PER_TX, seqlen_split, batch_size);
        // 启动任务
        if (not use_bshd_layout) {
            PERMUTE_HEADDIM_SWITCH(head_dim_q, [&]{
                // 如果不强制 kernel 用 bshd layout, 则需要自己转换一下 layout
                flash_fwd_varlen_permute_bshd2bhsd<kHeadDim, DWORD_PER_TX, SEQLEN_PER_BLOCK><<<grid, block, 0, stream>>>(
                    q_output.data_ptr(),
                    query.data_ptr(),
                    split_sizes.data_ptr(),
                    query.stride(-3),
                    num_heads,
                    head_dim_q
                );
            });
        }
        grid.x = num_heads_kv / DWORD_PER_TX;
        PERMUTE_HEADDIM_SWITCH(head_dim_k, [&]{
            flash_fwd_varlen_permute_bshd2bhsd<kHeadDim, DWORD_PER_TX, SEQLEN_PER_BLOCK><<<grid, block, 0, stream>>>(
                k_output.data_ptr(),
                key.data_ptr(),
                split_sizes.data_ptr(),
                key.stride(-3),
                num_heads_kv,
                head_dim_k
            );
        });
        PERMUTE_HEADDIM_SWITCH(head_dim_v, [&]{
            flash_fwd_varlen_permute_bshd2bhsd<kHeadDim, DWORD_PER_TX, SEQLEN_PER_BLOCK><<<grid, block, 0, stream>>>(
                v_output.data_ptr(),
                value.data_ptr(),
                split_sizes.data_ptr(),
                value.stride(-3),
                num_heads_kv,
                head_dim_v
            );
        });
    });
    return {q_output, k_output, v_output};
}


at::Tensor varlen_fwd_permute_bhsd2bshd(
        at::Tensor &fa_output,
        const at::Tensor split_sizes,
        const int num_heads,
        const int max_seqlen_qkv) {
    // 获取 size
    const auto output_size   = fa_output.sizes();
    const int32_t batch_seqlen_heads = output_size[0];
    const int32_t head_dim   = output_size[1];
    const int32_t batch_size = split_sizes.sizes()[0] - 1;
    const int batch_seqlen   = int32_t(batch_seqlen_heads / num_heads);
    // printf("varlen_fwd_permute_bhsd2bshd | batch_size: %d | num_heads: %d | head_dim: %d | batch_seqlen: %d\n", batch_size, num_heads, head_dim, batch_seqlen);
    // 检查变量的合法性, 是不是 headdim 128 等
    TORCH_CHECK (head_dim == 128 or head_dim == 192 or (head_dim > 0 and head_dim <= 256), "For fa forward varlen permute bhsd2bshd, only headdim = 128/192/<=256 is supported!");
    TORCH_CHECK (batch_size > 0, "For fa forward varlen permute bhsd2bshd, split_sizes have 2 elements at least!");
    TORCH_CHECK (sizeof(fa_output.dtype()) == 2, "For fa forward varlen permute bhsd2bshd, only dtypes of 2 bytes are supported!");
    // 如果数据不连续, 先改成连续的, 注意会修改引用
    if (not fa_output.is_contiguous()) fa_output = fa_output.contiguous();
    // 获取数据类型和设备, 准备输出的张量
    auto opts = fa_output.options();
    auto output = torch::empty({batch_seqlen, num_heads, head_dim}, opts);
    // 获取当前执行流, 方便被 hipgraph 捕捉到
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // 决定任务划分
    const int32_t SEQLEN_PER_BLOCK = 32;
    int32_t seqlen_split = (max_seqlen_qkv + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    dim3 block(64);
    dim3 grid(num_heads, seqlen_split, batch_size);
    void (*kernel_function)(void*, void*, void*, int, int);
    if (head_dim == 64) {
        kernel_function = flash_fwd_varlen_permute_bhsd2bshd<64, 1, SEQLEN_PER_BLOCK>;
    } else if (head_dim == 128) {
        kernel_function = flash_fwd_varlen_permute_bhsd2bshd<128, 4, SEQLEN_PER_BLOCK>;
    } else if (head_dim == 192) {
        kernel_function = flash_fwd_varlen_permute_bhsd2bshd<192, 1, SEQLEN_PER_BLOCK>;
    } else if (head_dim < 256) {
        kernel_function = flash_fwd_varlen_permute_bhsd2bshd<0, 1, SEQLEN_PER_BLOCK>;
    } else if (head_dim == 256) {
        kernel_function = flash_fwd_varlen_permute_bhsd2bshd<256, 1, SEQLEN_PER_BLOCK>;
    }
    kernel_function<<<grid, block, 0, stream>>>(
        output.data_ptr(),
        fa_output.data_ptr(),
        split_sizes.data_ptr(),
        num_heads,
        head_dim
    );
    return output;
}




/**
 * @brief sbhd --> bhsd, tranpose Q/K/V togather
 * @param query [seqlen, batch_size, num_head, head_dim]
          key   [seqlen, batch_size, num_head, head_dim]
          value [seqlen, batch_size, num_head, head_dim]
 * @return q_output [batch_size, num_head, seqlen, head_dim]
           k_output [batch_size, num_head, seqlen, head_dim]
           v_output [batch_size, num_head, seqlen, head_dim]
 */
std::vector<at::Tensor> fwd_permute_sbhd2bhsd(
        at::Tensor &query,
        at::Tensor &key,
        at::Tensor &value) {
    // acquire task scale
    const auto query_size      = query.sizes();
    const int32_t seqlen_q     = query_size[0];
    const int32_t batch_size   = query_size[1];
    const int32_t num_heads    = query_size[2];
    const int32_t head_dim_qk  = query_size[3];
    const auto    v_size       = value.sizes();
    const int32_t num_heads_kv = v_size[2];
    const int32_t head_dim_v   = v_size[3];
    // check
    // printf("seqlen_q: %d | batch_size: %d | num_heads: %d | head_dim_qk: %d | head_dim_v: %d\n", seqlen_q, batch_size, num_heads, head_dim_qk, head_dim_v);
    TORCH_CHECK (head_dim_qk <= 256 and head_dim_v <= 256, "For fwd_permute_sbhd2bhsd, only head_dim <= 256 is supported!");
    TORCH_CHECK (sizeof(query.dtype()) == 2, "For fwd_permute_sbhd2bhsd, only dtypes of 2 bytes are supported!");
    TORCH_CHECK (seqlen_q == v_size[0], "For fwd_permute_sbhd2bhsd, seqlen of Q/K/V must be same!");
    TORCH_CHECK (query.is_contiguous() and key.is_contiguous() and value.is_contiguous(), "For fwd_permute_sbhd2bhsd, Q/K/V must be contiguous!");
    // prepare space for outputs
    at::Tensor q_output, k_output, v_output;
    auto opts = query.options();
    q_output = torch::empty({batch_size, num_heads, seqlen_q, head_dim_qk}, opts);
    k_output = torch::empty({batch_size, num_heads_kv, seqlen_q, head_dim_qk}, opts);
    v_output = torch::empty({batch_size, num_heads_kv, seqlen_q, head_dim_v}, opts);
    // acquire current execution stream
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // dispatch tasks
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    int32_t seqlen_split = (seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    dim3 block(64);
    dim3 grid(seqlen_split, batch_size * num_heads);
    // process Q
    PERMUTE_HEADDIM_SWITCH(head_dim_qk, [&]{
        flash_permute_sbhd2bhsd<kHeadDim, 1, SEQLEN_PER_BLOCK><<<grid, block, 0, stream>>>(
            q_output.data_ptr(),
            query.data_ptr(),
            seqlen_q,
            head_dim_qk
        );
    });
    // process K/V
    grid.y = batch_size * num_heads_kv;
    PERMUTE_HEADDIM_SWITCH(head_dim_qk, [&]{
        flash_permute_sbhd2bhsd<kHeadDim, 1, SEQLEN_PER_BLOCK><<<grid, block, 0, stream>>>(
            k_output.data_ptr(),
            key.data_ptr(),
            seqlen_q,
            head_dim_qk
        );
    });
    PERMUTE_HEADDIM_SWITCH(head_dim_v, [&]{
        flash_permute_sbhd2bhsd<kHeadDim, 1, SEQLEN_PER_BLOCK><<<grid, block, 0, stream>>>(
            v_output.data_ptr(),
            value.data_ptr(),
            seqlen_q,
            head_dim_v
        );
    });
    return {q_output, k_output, v_output};
}


/**
 * @brief bhsd --> sbhd
 * @param context_layer [batch_size, num_head, seqlen, head_dim]
 * @return output: [seqlen, batch_size, num_head, head_dim]
 */
at::Tensor fwd_permute_bhsd2sbhd(at::Tensor &context_layer) {
    // acquire problem scale
    const auto sizes = context_layer.sizes();
    const int32_t batch_size = sizes[0];
    const int32_t num_heads  = sizes[1];
    const int32_t seqlen     = sizes[2];
    const int32_t head_dim   = sizes[3];
    // check
    // printf("batch_size: %d | num_heads: %d | seqlen_q: %d | head_dim: %d\n", batch_size, num_heads, seqlen, head_dim);
    TORCH_CHECK (head_dim <= 256, "For fwd_permute_bhsd2sbhd, only head_dim <= 256 is supported!");
    TORCH_CHECK (sizeof(context_layer.dtype()) == 2, "For fwd_permute_bhsd2sbhd, only dtypes of 2 bytes are supported!");
    TORCH_CHECK (context_layer.is_contiguous(), "For fwd_permute_bhsd2sbhd, input must be contiguous!");
    // prepare outputs
    auto opts = context_layer.options();
    at::Tensor output = torch::empty({seqlen, batch_size, num_heads * head_dim}, opts);
    // acquire current execution stream
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // dispatch tasks
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    int32_t seqlen_split = (seqlen + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    dim3 block(64);
    dim3 grid(seqlen_split, batch_size * num_heads);
    // process Q
    PERMUTE_HEADDIM_SWITCH(head_dim, [&]{
        flash_permute_bhsd2sbhd<kHeadDim, 1, SEQLEN_PER_BLOCK><<<grid, block, 0, stream>>>(
            output.data_ptr(),
            context_layer.data_ptr(),
            seqlen,
            head_dim
        );
    });
    return output;
};


/**
 * @brief bshd --> bhsd, tranpose Q/K/V togather
 * @param query [batch_size, seqlen, num_head, head_dim]
          key   [batch_size, seqlen, num_head, head_dim]
          value [batch_size, seqlen, num_head, head_dim]
 * @return q_output [batch_size, num_head, seqlen, head_dim]
           k_output [batch_size, num_head, seqlen, head_dim]
           v_output [batch_size, num_head, seqlen, head_dim]
 */
std::vector<at::Tensor> fwd_permute_bshd2bhsd(
        at::Tensor &query,
        at::Tensor &key,
        at::Tensor &value) {
    // acquire problem scale
    const auto query_size    = query.sizes();
    const int32_t batch_size = query_size[0];
    const int32_t seqlen_q   = query_size[1];
    const int32_t num_heads  = query_size[2];
    const int32_t head_dim   = query_size[3];
    const auto kv_size = key.sizes();
    const int32_t num_heads_kv = kv_size[2];
    // check
    // printf("batch_size: %d | seqlen_q: %d | num_heads: %d | head_dim: %d\n", batch_size, seqlen_q, num_heads, head_dim);
    TORCH_CHECK (head_dim == 128, "For fwd_permute_bshd2bhsd, only head_dim 128 is supported!");
    TORCH_CHECK (sizeof(query.dtype()) == 2, "For fwd_permute_bshd2bhsd, only dtypes of 2 bytes are supported!");
    TORCH_CHECK (seqlen_q == kv_size[1], "For fwd_permute_bshd2bhsd, seqlen of Q/K/V must be same!");
    TORCH_CHECK (query.is_contiguous() and key.is_contiguous() and value.is_contiguous(), "For fwd_permute_bshd2bhsd, Q/K/V must be contiguous!");
    // prepare space for outputs
    at::Tensor q_output, k_output, v_output;
    auto opts = query.options();
    q_output = torch::empty({batch_size, num_heads, seqlen_q, head_dim}, opts);
    auto adjacent_memory = torch::empty({2, batch_size, num_heads_kv, seqlen_q, head_dim}, opts);
    k_output = adjacent_memory.index({0});
    v_output = adjacent_memory.index({1});
    // acquire current execution stream
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // dispatch tasks
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    int32_t seqlen_split = (seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    dim3 block(64);
    dim3 grid(seqlen_split, num_heads, batch_size);
    flash_permute_bshd2bhsd<128, 1, 32><<<grid, block, 0, stream>>>(
        q_output.data_ptr(),
        query.data_ptr(),
        seqlen_q,
        num_heads
    );
    grid.y = num_heads_kv;
    flash_permute_bshd2bhsd<128, 1, 32><<<grid, block, 0, stream>>>(
        k_output.data_ptr(),
        key.data_ptr(),
        seqlen_q,
        num_heads_kv
    );
    flash_permute_bshd2bhsd<128, 1, 32><<<grid, block, 0, stream>>>(
        v_output.data_ptr(),
        value.data_ptr(),
        seqlen_q,
        num_heads_kv
    );
    return {q_output, k_output, v_output};
}


/**
 * @brief bhsd --> bshd
 * @param context_layer [batch_size, num_head, seqlen, head_dim]
 * @return output: [batch_size, seqlen, num_head, head_dim]
 */
at::Tensor fwd_permute_bhsd2bshd(at::Tensor& context_layer) {
    // acquire problem scale
    const auto sizes = context_layer.sizes();
    const int32_t batch_size = sizes[0];
    const int32_t num_heads  = sizes[1];
    const int32_t seqlen     = sizes[2];
    const int32_t head_dim   = sizes[3];
    // check
    // printf("batch_size: %d | num_heads: %d | seqlen: %d | head_dim: %d\n", batch_size, num_heads, seqlen, head_dim);
    TORCH_CHECK (head_dim == 128, "For fwd_permute_bhsd2bshd, only head_dim 128 is supported!");
    TORCH_CHECK (sizeof(context_layer.dtype()) == 2, "For fwd_permute_bhsd2bshd, only dtypes of 2 bytes are supported!");
    TORCH_CHECK (context_layer.is_contiguous(), "For fwd_permute_bhsd2bshd, input must be contiguous!");
    // prepare space for outputs
    auto opts = context_layer.options();
    at::Tensor output = torch::empty({batch_size, seqlen, num_heads, head_dim}, opts);
    // acquire current execution stream
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // dispatch tasks
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    int32_t seqlen_split = (seqlen + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    dim3 block(64);
    dim3 grid(seqlen_split, num_heads, batch_size);
    flash_permute_bhsd2bshd<128, 1, 32><<<grid, block, 0, stream>>>(
        output.data_ptr(),
        context_layer.data_ptr(),
        seqlen,
        num_heads
    );
    return output;
}


/**
 * @brief bhsd --> sbhd
 * @param dq [batch_size, num_head, seqlen, head_dim]
          dk [batch_size, num_head, seqlen, head_dim]
          dv [batch_size, num_head, seqlen, head_dim]
 * @return output_dq [seqlen, batch_size, num_head, head_dim]
           output_dk [seqlen, batch_size, num_head, head_dim]
           output_dv [seqlen, batch_size, num_head, head_dim]
 */
std::vector<at::Tensor> bwd_permute_bhsd2sbhd(
        at::Tensor &dq, at::Tensor &dk, at::Tensor &dv) {
    // acquire problem scale
    const auto sizes = dq.sizes();
    const int32_t batch_size   = sizes[0];
    const int32_t num_heads    = sizes[1];
    const int32_t seqlen       = sizes[2];
    const int32_t head_dim_qk  = sizes[3];
    const int32_t num_heads_kv = dv.size(1);
    const int32_t head_dim_v   = dv.size(3);
    // check
    // printf("batch_size: %d | num_heads: %d | seqlen_q: %d | head_dim_qk: %d | head_dim_v: %d\n", batch_size, num_heads, seqlen, head_dim_qk, head_dim_v);
    TORCH_CHECK (head_dim_qk <= 256 and head_dim_v <= 256, "For bwd_permute_bhsd2sbhd, only head_dim <= 256 is supported!");
    TORCH_CHECK (sizeof(dq.dtype()) == 2 and sizeof(dk.dtype()) == 2 and sizeof(dv.dtype()) == 2, "For bwd_permute_bhsd2sbhd, only dtypes of 2 bytes are supported!");
    TORCH_CHECK (dq.is_contiguous() and dk.is_contiguous() and dv.is_contiguous(), "For bwd_permute_bhsd2sbhd, input must be contiguous!");
    // prepare outputs
    auto opts = dq.options();
    at::Tensor output_dq = torch::empty({seqlen, batch_size, num_heads, head_dim_qk}, opts);
    at::Tensor output_dk = torch::empty({seqlen, batch_size, num_heads_kv, head_dim_qk}, opts);
    at::Tensor output_dv = torch::empty({seqlen, batch_size, num_heads_kv, head_dim_v}, opts);
    // acquire current execution stream
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // dispatch tasks
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    int32_t seqlen_split = (seqlen + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    dim3 block(64);
    dim3 grid(seqlen_split, batch_size * num_heads);
    // process dq
    PERMUTE_HEADDIM_SWITCH(head_dim_qk, [&]{
        flash_permute_bhsd2sbhd<kHeadDim, 1, SEQLEN_PER_BLOCK><<<grid, block, 0, stream>>>(
            output_dq.data_ptr(),
            dq.data_ptr(),
            seqlen,
            head_dim_qk
        );
    });
    // process dk/dv
    grid.y = batch_size * num_heads_kv;
    PERMUTE_HEADDIM_SWITCH(head_dim_qk, [&]{
        flash_permute_bhsd2sbhd<kHeadDim, 1, SEQLEN_PER_BLOCK><<<grid, block, 0, stream>>>(
            output_dk.data_ptr(),
            dk.data_ptr(),
            seqlen,
            head_dim_qk
        );
    });
    PERMUTE_HEADDIM_SWITCH(head_dim_v, [&]{
        flash_permute_bhsd2sbhd<kHeadDim, 1, SEQLEN_PER_BLOCK><<<grid, block, 0, stream>>>(
            output_dv.data_ptr(),
            dv.data_ptr(),
            seqlen,
            head_dim_v
        );
    });
    return {output_dq, output_dk, output_dv};
};


/**
* @brief bhsd --> bshd, process Q/K/V togather
* @param dq [batch_size, num_head, seqlen, head_dim]
*        dk [batch_size, num_head, seqlen, head_dim]
*        dv [batch_size, num_head, seqlen, head_dim]
* @return output_dq [batch_size, seqlen, num_head, head_dim]
*         output_dk [batch_size, seqlen, num_head, head_dim]
*         output_dv [batch_size, seqlen, num_head, head_dim]
*/
std::vector<at::Tensor> bwd_permute_bhsd2bshd(
        at::Tensor& dq, at::Tensor& dk, at::Tensor& dv) {
    // acquire problem scale
    const auto sizes = dq.sizes();
    const int32_t batch_size = sizes[0];
    const int32_t num_heads  = sizes[1];
    const int32_t seqlen     = sizes[2];
    const int32_t head_dim   = sizes[3];
    const int32_t num_heads_kv = dk.sizes()[1];
    // check
    // printf("batch_size: %d | num_heads: %d | seqlen: %d | head_dim: %d\n", batch_size, num_heads, seqlen, head_dim);
    TORCH_CHECK (head_dim == 128, "For bwd_permute_bhsd2bshd, only head_dim 128 is supported!");
    TORCH_CHECK (sizeof(dq.dtype()) == 2 and sizeof(dk.dtype()) == 2 and sizeof(dv.dtype()) == 2, "For bwd_permute_bhsd2bshd, only dtypes of 2 bytes are supported!");
    TORCH_CHECK (dq.is_contiguous() and dk.is_contiguous() and dv.is_contiguous(), "For bwd_permute_bhsd2bshd, input must be contiguous!");
    // prepare space for outputs
    auto opts = dq.options();
    at::Tensor output_dq = torch::empty({batch_size, seqlen, num_heads, head_dim}, opts);
    at::Tensor output_dk = torch::empty({batch_size, seqlen, num_heads_kv, head_dim}, opts);
    at::Tensor output_dv = torch::empty({batch_size, seqlen, num_heads_kv, head_dim}, opts);
    // acquire current execution stream
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // dispatch tasks
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    int32_t seqlen_split = (seqlen + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    dim3 block(64);
    dim3 grid(seqlen_split, num_heads, batch_size);
    flash_permute_bhsd2bshd<128, 1, 32><<<grid, block, 0, stream>>>(
        output_dq.data_ptr(),
        dq.data_ptr(),
        seqlen,
        num_heads
    );
    grid.y = num_heads_kv;
    flash_permute_bhsd2bshd<128, 1, 32><<<grid, block, 0, stream>>>(
        output_dk.data_ptr(),
        dk.data_ptr(),
        seqlen,
        num_heads_kv
    );
    flash_permute_bhsd2bshd<128, 1, 32><<<grid, block, 0, stream>>>(
        output_dv.data_ptr(),
        dv.data_ptr(),
        seqlen,
        num_heads_kv
    );
    return {output_dq, output_dk, output_dv};
}


/**
 * @brief sbhd --> bhsd
 * @param context_layer [seqlen, batch_size, num_head, head_dim]
 * @return output: [batch_size, num_head, seqlen, head_dim]
 */
at::Tensor bwd_permute_sbhd2bhsd(at::Tensor &context_layer) {
    // acquire task scale
    const auto sizes         = context_layer.sizes();
    const int32_t seqlen_q   = sizes[0];
    const int32_t batch_size = sizes[1];
    const int32_t num_heads  = sizes[2];
    const int32_t head_dim   = sizes[3];
    // check
    // printf("seqlen_q: %d | batch_size: %d | num_heads: %d | head_dim: %d\n", seqlen_q, batch_size, num_heads, head_dim);
    TORCH_CHECK (head_dim <= 256, "For bwd_permute_sbhd2bhsd, only head_dim <= 256 is supported!");
    TORCH_CHECK (sizeof(context_layer.dtype()) == 2, "For bwd_permute_sbhd2bhsd, only dtypes of 2 bytes are supported!");
    TORCH_CHECK (context_layer.is_contiguous(), "For bwd_permute_sbhd2bhsd, input must be contiguous!");
    // prepare space for outputs
    auto opts = context_layer.options();
    at::Tensor output = torch::empty({batch_size, num_heads, seqlen_q, head_dim}, opts);
    // acquire current execution stream
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // dispatch tasks
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    int32_t seqlen_split = (seqlen_q +  SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    dim3 block(64);
    dim3 grid(seqlen_split, batch_size * num_heads);
    // process Q
    PERMUTE_HEADDIM_SWITCH(head_dim, [&]{
        flash_permute_sbhd2bhsd<kHeadDim, 1, SEQLEN_PER_BLOCK><<<grid, block, 0, stream>>>(
            output.data_ptr(),
            context_layer.data_ptr(),
            seqlen_q,
            head_dim
        );
    });
    return output;
}


/**
 * @brief bshd --> bhsd
 * @param context_layer [batch_size, seqlen, num_head, head_dim]
 * @return output: [batch_size, num_head, seqlen, head_dim]
 */
at::Tensor bwd_permute_bshd2bhsd(at::Tensor &context_layer) {
    // acquire problem scale
    const auto sizes         = context_layer.sizes();
    const int32_t batch_size = sizes[0];
    const int32_t seqlen_q   = sizes[1];
    const int32_t num_heads  = sizes[2];
    const int32_t head_dim   = sizes[3];
    // check
    // printf("batch_size: %d | seqlen_q: %d | num_heads: %d | head_dim: %d\n", batch_size, seqlen_q, num_heads, head_dim);
    TORCH_CHECK (head_dim == 128, "For bwd_permute_bshd2bhsd, only head_dim 128 is supported!");
    TORCH_CHECK (sizeof(context_layer.dtype()) == 2, "For bwd_permute_bshd2bhsd, only dtypes of 2 bytes are supported!");
    TORCH_CHECK (context_layer.is_contiguous(), "For bwd_permute_bshd2bhsd, Q/K/V must be contiguous!");
    // prepare space for outputs
    auto opts = context_layer.options();
    at::Tensor output = torch::empty({batch_size, num_heads, seqlen_q, head_dim}, opts);
    // acquire current execution stream
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // dispatch tasks
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    int32_t seqlen_split = (seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    dim3 block(64);
    dim3 grid(seqlen_split, num_heads, batch_size);
    flash_permute_bshd2bhsd<128, 1, 32><<<grid, block, 0, stream>>>(
        output.data_ptr(),
        context_layer.data_ptr(),
        seqlen_q,
        num_heads
    );
    return output;
}



/**
 * @brief sbhd --> bhsd
 * @param input [seqlen, batch_size, num_head, head_dim]
 * @return output: [batch_size, num_head, seqlen, head_dim]
 */
at::Tensor permute_sbhd2bhsd(at::Tensor &input) {
    // acquire task scale
    const auto sizes         = input.sizes();
    const int32_t seqlen_q   = sizes[0];
    const int32_t batch_size = sizes[1];
    const int32_t num_heads  = sizes[2];
    const int32_t head_dim   = sizes[3];
    // check
    // printf("seqlen_q: %d | batch_size: %d | num_heads: %d | head_dim: %d\n", seqlen_q, batch_size, num_heads, head_dim);
    TORCH_CHECK (head_dim <= 256, "For permute_sbhd2bhsd, only head_dim <= 256 is supported!");
    TORCH_CHECK (sizeof(input.dtype()) == 2, "For permute_sbhd2bhsd, only dtypes of 2 bytes are supported!");
    TORCH_CHECK (input.is_contiguous(), "For permute_sbhd2bhsd, input must be contiguous!");
    // prepare space for outputs
    auto opts = input.options();
    at::Tensor output = torch::empty({batch_size, num_heads, seqlen_q, head_dim}, opts);
    // acquire current execution stream
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // dispatch tasks
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    int32_t seqlen_split = (seqlen_q +  SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    dim3 block(64);
    dim3 grid(seqlen_split, batch_size * num_heads);
    // process Q
    PERMUTE_HEADDIM_SWITCH(head_dim, [&]{
        flash_permute_sbhd2bhsd<kHeadDim, 1, SEQLEN_PER_BLOCK><<<grid, block, 0, stream>>>(
            output.data_ptr(),
            input.data_ptr(),
            seqlen_q,
            head_dim
        );
    });
    return output;
}


/**
 * @brief bhsd --> sbhd
 * @param input [batch_size, num_head, seqlen, head_dim]
 * @return output: [seqlen, batch_size, num_head, head_dim]
 */
at::Tensor permute_bhsd2sbhd(at::Tensor &input) {
    // acquire problem scale
    const auto sizes = input.sizes();
    const int32_t batch_size = sizes[0];
    const int32_t num_heads  = sizes[1];
    const int32_t seqlen     = sizes[2];
    const int32_t head_dim   = sizes[3];
    // check
    // printf("batch_size: %d | num_heads: %d | seqlen_q: %d | head_dim: %d\n", batch_size, num_heads, seqlen, head_dim);
    TORCH_CHECK (head_dim <= 256, "For permute_bhsd2sbhd, only head_dim <= 256 is supported!");
    TORCH_CHECK (sizeof(input.dtype()) == 2, "For permute_bhsd2sbhd, only dtypes of 2 bytes are supported!");
    TORCH_CHECK (input.is_contiguous(), "For permute_bhsd2sbhd, input must be contiguous!");
    // prepare outputs
    auto opts = input.options();
    at::Tensor output = torch::empty({seqlen, batch_size, num_heads * head_dim}, opts);
    // acquire current execution stream
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // dispatch tasks
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    int32_t seqlen_split = (seqlen + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    dim3 block(64);
    dim3 grid(seqlen_split, batch_size * num_heads);
    // process Q
    PERMUTE_HEADDIM_SWITCH(head_dim, [&]{
        flash_permute_bhsd2sbhd<kHeadDim, 1, SEQLEN_PER_BLOCK><<<grid, block, 0, stream>>>(
            output.data_ptr(),
            input.data_ptr(),
            seqlen,
            head_dim
        );
    });
    return output;
};


/**
 * @brief bhsd --> bshd
 * @param input [batch_size, num_head, seqlen, head_dim]
 * @return output: [batch_size, seqlen, num_head, head_dim]
 */
at::Tensor permute_bhsd2bshd(at::Tensor& input) {
    // acquire problem scale
    const auto sizes = input.sizes();
    const int32_t batch_size = sizes[0];
    const int32_t num_heads  = sizes[1];
    const int32_t seqlen     = sizes[2];
    const int32_t head_dim   = sizes[3];
    // check
    // printf("batch_size: %d | num_heads: %d | seqlen: %d | head_dim: %d\n", batch_size, num_heads, seqlen, head_dim);
    TORCH_CHECK (head_dim == 128, "For permute_bhsd2bshd, only head_dim 128 is supported!");
    TORCH_CHECK (sizeof(input.dtype()) == 2, "For permute_bhsd2bshd, only dtypes of 2 bytes are supported!");
    TORCH_CHECK (input.is_contiguous(), "For permute_bhsd2bshd, input must be contiguous!");
    // prepare space for outputs
    auto opts = input.options();
    at::Tensor output = torch::empty({batch_size, seqlen, num_heads, head_dim}, opts);
    // acquire current execution stream
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // dispatch tasks
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    int32_t seqlen_split = (seqlen + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    dim3 block(64);
    dim3 grid(seqlen_split, num_heads, batch_size);
    flash_permute_bhsd2bshd<128, 1, 32><<<grid, block, 0, stream>>>(
        output.data_ptr(),
        input.data_ptr(),
        seqlen,
        num_heads
    );
    return output;
}



/**
 * @brief bshd --> bhsd
 * @param input [batch_size, seqlen, num_head, head_dim]
 * @return output: [batch_size, num_head, seqlen, head_dim]
 */
at::Tensor permute_bshd2bhsd(at::Tensor &input) {
    // acquire problem scale
    const auto sizes         = input.sizes();
    const int32_t batch_size = sizes[0];
    const int32_t seqlen_q   = sizes[1];
    const int32_t num_heads  = sizes[2];
    const int32_t head_dim   = sizes[3];
    // check
    // printf("batch_size: %d | seqlen_q: %d | num_heads: %d | head_dim: %d\n", batch_size, seqlen_q, num_heads, head_dim);
    TORCH_CHECK (head_dim == 128, "For permute_bshd2bhsd, only head_dim 128 is supported!");
    TORCH_CHECK (sizeof(input.dtype()) == 2, "For permute_bshd2bhsd, only dtypes of 2 bytes are supported!");
    TORCH_CHECK (input.is_contiguous(), "For permute_bshd2bhsd, Q/K/V must be contiguous!");
    // prepare space for outputs
    auto opts = input.options();
    at::Tensor output = torch::empty({batch_size, num_heads, seqlen_q, head_dim}, opts);
    // acquire current execution stream
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    // dispatch tasks
    constexpr int32_t SEQLEN_PER_BLOCK = 32;
    int32_t seqlen_split = (seqlen_q + SEQLEN_PER_BLOCK - 1) / SEQLEN_PER_BLOCK;
    dim3 block(64);
    dim3 grid(seqlen_split, num_heads, batch_size);
    flash_permute_bshd2bhsd<128, 1, 32><<<grid, block, 0, stream>>>(
        output.data_ptr(),
        input.data_ptr(),
        seqlen_q,
        num_heads
    );
    return output;
}


#endif // end of BUILD_FA_PERMUTE