#pragma once

#include <iostream>
#include "flash.h"

#  define FLASH_HOST_DEVICE __forceinline__ __host__ __device__
#  define FLASH_DEVICE      __forceinline__          __device__
#  define FLASH_HOST        __forceinline__ __host__

#define HIP_KERNEL_LAUNCH_CHECK() {  \
    hipError_t error = hipGetLastError(); \
    if (error != hipSuccess) { \
        std::cout << "HIP Kernel Launch error: " << hipGetErrorString(error) << std::endl;\
    }\
} 

#define HIP_CHECK(func) { \
    hipError_t error = func;    \
    if (error != hipSuccess) { \
        std::cout << "HIP API call error: " << hipGetErrorString(error) << std::endl;\
    }\
}


#define PRINT_TENSOR_INFO(tensor, name) \
    std::cout << name << ": shape " << tensor.sizes() << ", stride " << tensor.strides() << ", contiguous " << std::boolalpha << tensor.is_contiguous() << "\n";


#define PRINT_QKV_INFO(q, k, v) \
    std::cout << "qkv shape: " << q.sizes() << ", " << k.sizes() << ", " << v.sizes() << "\n"; \
    std::cout << "qkv stride: " << q.strides() << ", " << k.strides() << ", " << v.strides() << "\n"; \
    std::cout << "qkv contiguous: " << std::boolalpha << q.is_contiguous() << ", " << k.is_contiguous() << ", " << v.is_contiguous() << "\n";


#define PRINT_TENSOR(tensor, name) \
    { \
        auto temp_tensor = tensor.to(at::DeviceType::CPU).contiguous(); \
        std::vector<int32_t> temp_vector(temp_tensor.data_ptr<int32_t>(), temp_tensor.data_ptr<int32_t>() + temp_tensor.numel()); \
        printf("%s: [", name); \
        for (const auto val: temp_vector) { printf("%d ", val); } \
        printf("]\n"); \
    }


#define PRINT_PARAMS \
    printf("layout: %d\n", params.layout);\
    printf("mtp: %d\n", params.mtp);\
    printf("is_causal: %d\n", params.is_causal); \
    printf("is_bf16: %d\n", params.is_bf16); \
    printf("is_e4m3: %d\n", params.is_e4m3); \
    printf("b: %d\n", params.b);\
    printf("h: %d\n", params.h);\
    printf("h_k: %d\n", params.h_k);\
    printf("h_h_k_ratio: %d\n", params.h_h_k_ratio);\
    printf("seqlen_q: %d\n", params.seqlen_q);\
    printf("seqlen_k: %d\n", params.seqlen_k);\
    printf("total_q: %d\n", params.total_q);\
    printf("seqlen_knew: %d\n", params.seqlen_knew);\
    printf("seqlen_q_rounded: %d\n", params.seqlen_q_rounded);\
    printf("seqlen_k_rounded: %d\n", params.seqlen_k_rounded);\
    printf("ngroups: %d\n", params.ngroups);\
    printf("seqlenq_ngroups_swapped: %d\n", params.seqlenq_ngroups_swapped);\
    printf("d: %d\n", params.d);\
    printf("dv: %d\n", params.d_value);\
    printf("q_batch_stride: %d\n", params.q_batch_stride);\
    printf("q_row_stride: %d\n", params.q_row_stride);\
    printf("q_head_stride: %d\n", params.q_head_stride);\
    printf("k_batch_stride: %d\n", params.k_batch_stride);\
    printf("k_row_stride: %d\n", params.k_row_stride);\
    printf("k_head_stride: %d\n", params.k_head_stride);\
    printf("v_batch_stride: %d\n", params.v_batch_stride);\
    printf("v_row_stride: %d\n", params.v_row_stride);\
    printf("v_head_stride: %d\n", params.v_head_stride);\
    printf("o_batch_stride: %d\n", params.o_batch_stride);\
    printf("o_row_stride: %d\n", params.o_row_stride);\
    printf("o_head_stride: %d\n", params.o_head_stride);\
    printf("varlen_proj_qkv_head: %d\n", params.varlen_proj_qkv_head);\
    printf("knew_batch_stride: %d\n", params.knew_batch_stride);\
    printf("vnew_batch_stride: %d\n", params.vnew_batch_stride);\
    printf("knew_row_stride: %d\n", params.knew_row_stride);\
    printf("vnew_row_stride: %d\n", params.vnew_row_stride);\
    printf("knew_head_stride: %d\n", params.knew_head_stride);\
    printf("vnew_head_stride: %d\n", params.vnew_head_stride);\
    printf("window_size_left: %d\n", params.window_size_left);\
    printf("window_size_right: %d\n", params.window_size_right);\
    printf("qkvheaddim_compute: %d\n", params.qkvheaddim_compute);\
    printf("qkvheaddim_tail_tile16: %d\n", params.qkvheaddim_tail_tile16);\
    printf("softcap: %.12f\n", params.softcap);\
    printf("p_dropout: %.12f\n", params.p_dropout);\
    printf("rp_dropout: %.12f\n", params.rp_dropout);\
    printf("p_dropout_in_uint8_t: %u\n", params.p_dropout_in_uint8_t);\
    printf("random_seed: %llu\n", params.rand_seed);\
    printf("random_offset: %llu\n", params.rand_offset);\
    printf("scale_softmax: %.12f\n", params.scale_softmax);\
    printf("scale_softmax_log2: %.12f\n", params.scale_softmax_log2);\
    printf("is_seqlens_k_cumulative: %d\n", params.is_seqlens_k_cumulative); \
    printf("q_ptr: %p\n", params.q_ptr); \
    printf("k_ptr: %p\n", params.k_ptr); \
    printf("v_ptr: %p\n", params.v_ptr); \
    printf("o_ptr: %p\n", params.o_ptr); \
    printf("qv_ptr: %p\n", params.qv_ptr); \
    printf("cu_seqlens_q: %p\n", params.cu_seqlens_q); \
    printf("cu_seqlens_k: %p\n", params.cu_seqlens_k); \
    printf("seqused_k: %p\n", params.seqused_k); \
    printf("attn_mask: %p\n", params.attn_mask); \
    printf("padding_mask: %p\n", params.padding_mask); \
    printf("softmax_lse_ptr: %p\n", params.softmax_lse_ptr); \
    printf("softmax_lseaccum_ptr: %p\n", params.softmax_lseaccum_ptr); \
    printf("scores_sum_ptr: %p\n", params.scores_sum_ptr); \
    printf("scores_max_ptr: %p\n", params.scores_max_ptr); \
    printf("oaccum_ptr: %p\n", params.oaccum_ptr); \
    printf("block_table : %p\n", params.block_table); \
    printf("page_block_size: %d\n", params.page_block_size); \
    printf("block_table_batch_stride: %d\n", params.block_table_batch_stride); \
    printf("num_pages : %d\n", params.num_pages); \
    printf("num_splits : %d\n", params.num_splits); \
    printf("partition_size : %d\n", params.partition_size); \
    printf("splitkv_use_fp32_as_accum: %d\n", params.splitkv_use_fp32_as_accum);

// output details in oneline
#define PRINT_PARAMS_ONELINE \
    printf("{layout: %d, is_causal: %d, is_bf16: %d, is_e4m3: %d, b: %d, h: %d, h_k: %d, h_h_k_ratio: %d, seqlen_q: %d, seqlen_k: %d, total_q: %d, ngroups: %d, seqlenq_ngroups_swapped: %d, d: %d, dv: %d,\
q_batch_stride: %d, q_row_stride: %d, q_head_stride: %d, k_batch_stride: %d, k_row_stride: %d, k_head_stride: %d, v_batch_stride: %d, v_row_stride: %d, v_head_stride: %d, o_batch_stride: %d, o_row_stride: %d, o_head_stride: %d,\
varlen_proj_qkv_head: %d, window_size_left: %d, window_size_right: %d, softcap: %.12f, scale_softmax: %.12f, scale_softmax_log2: %.12f, is_seqlens_k_cumulative: %d,\
q_ptr: %p, k_ptr: %p, v_ptr: %p, o_ptr: %p, qv: %p, cu_seqlens_q: %p, cu_seqlens_k: %p, seqused_k: %p, attn_mask: %p, padding_mask: %p, softmax_lse_ptr: %p, scores_sum_ptr: %p, scores_max_ptr: %p, oaccum_ptr: %p, block_table : %p,\
page_block_size: %d, block_table_batch_stride: %d, num_pages : %d, num_splits : %d, partition_size : %d, splitkv_use_fp32_as_accum: %d}\n",\
        params.layout, \
        params.is_causal, \
        params.is_bf16, \
        params.is_e4m3, \
        params.b, \
        params.h, \
        params.h_k, \
        params.h_h_k_ratio, \
        params.seqlen_q, \
        params.seqlen_k, \
        params.total_q, \
        params.ngroups, \
        params.seqlenq_ngroups_swapped, \
        params.d, \
        params.d_value, \
        params.q_batch_stride, \
        params.q_row_stride, \
        params.q_head_stride, \
        params.k_batch_stride, \
        params.k_row_stride, \
        params.k_head_stride, \
        params.v_batch_stride, \
        params.v_row_stride, \
        params.v_head_stride, \
        params.o_batch_stride, \
        params.o_row_stride, \
        params.o_head_stride, \
        params.varlen_proj_qkv_head, \
        params.window_size_left, \
        params.window_size_right, \
        params.softcap, \
        params.scale_softmax, \
        params.scale_softmax_log2, \
        params.is_seqlens_k_cumulative, \
        params.q_ptr, \
        params.k_ptr, \
        params.v_ptr, \
        params.o_ptr, \
        params.qv_ptr, \
        params.cu_seqlens_q, \
        params.cu_seqlens_k, \
        params.seqused_k, \
        params.attn_mask, \
        params.padding_mask, \
        params.softmax_lse_ptr, \
        params.scores_sum_ptr, \
        params.scores_max_ptr, \
        params.oaccum_ptr, \
        params.block_table , \
        params.page_block_size, \
        params.block_table_batch_stride, \
        params.num_pages , \
        params.num_splits , \
        params.partition_size , \
        params.splitkv_use_fp32_as_accum \
    );


__attribute__((weak)) void printFlashBwdParams(const Flash_bwd_params& params) {
    std::cout << "Flash_bwd_params:\n";

    // 打印 Flash_fwd_params 成员
    std::cout << "o_ptr: " << params.o_ptr << "\n";
    std::cout << "oaccum_ptr: " << params.oaccum_ptr << "\n";
#ifdef DEBUGING
    std::cout << "qk_ptr: " << params.qk_ptr << "\n";
    std::cout << "qk_softmax_ptr: " << params.qk_softmax_ptr << "\n";
#endif
    std::cout << "o_batch_stride: " << params.o_batch_stride << "\n";
    std::cout << "o_row_stride: " << params.o_row_stride << "\n";
    std::cout << "o_head_stride: " << params.o_head_stride << "\n";
    std::cout << "p_ptr: " << params.p_ptr << "\n";
    std::cout << "softmax_lse_ptr: " << params.softmax_lse_ptr << "\n";
    std::cout << "softmax_lseaccum_ptr: " << params.softmax_lseaccum_ptr << "\n";
    std::cout << "scores_sum_ptr: " << params.scores_sum_ptr << "\n";
    std::cout << "scores_max_ptr: " << params.scores_max_ptr << "\n";
    std::cout << "b: " << params.b << "\n";
    std::cout << "seqlen_q: " << params.seqlen_q << "\n";
    std::cout << "seqlen_k: " << params.seqlen_k << "\n";
    std::cout << "seqlen_knew: " << params.seqlen_knew << "\n";
    std::cout << "d: " << params.d << "\n";
    std::cout << "seqlen_q_rounded: " << params.seqlen_q_rounded << "\n";
    std::cout << "seqlen_k_rounded: " << params.seqlen_k_rounded << "\n";
    std::cout << "d_rounded: " << params.d_rounded << "\n";
    std::cout << "rotary_dim: " << params.rotary_dim << "\n";
    std::cout << "total_q: " << params.total_q << "\n";
    std::cout << "scale_softmax: " << params.scale_softmax << "\n";
    std::cout << "scale_softmax_log2: " << params.scale_softmax_log2 << "\n";
    std::cout << "cu_seqlens_q: " << params.cu_seqlens_q << "\n";
    std::cout << "cu_seqlens_k: " << params.cu_seqlens_k << "\n";
    std::cout << "leftpad_k: " << params.leftpad_k << "\n";
    std::cout << "seqused_k: " << params.seqused_k << "\n";
//  std::cout << "blockmask: " << params.blockmask << "\n";
    std::cout << "knew_ptr: " << params.knew_ptr << "\n";
    std::cout << "vnew_ptr: " << params.vnew_ptr << "\n";
    std::cout << "knew_batch_stride: " << params.knew_batch_stride << "\n";
    std::cout << "vnew_batch_stride: " << params.vnew_batch_stride << "\n";
    std::cout << "knew_row_stride: " << params.knew_row_stride << "\n";
    std::cout << "vnew_row_stride: " << params.vnew_row_stride << "\n";
    std::cout << "knew_head_stride: " << params.knew_head_stride << "\n";
    std::cout << "vnew_head_stride: " << params.vnew_head_stride << "\n";
    std::cout << "rotary_cos_ptr: " << params.rotary_cos_ptr << "\n";
    std::cout << "rotary_sin_ptr: " << params.rotary_sin_ptr << "\n";
    std::cout << "cache_batch_idx: " << params.cache_batch_idx << "\n";
    std::cout << "block_table: " << params.block_table << "\n";
    std::cout << "block_table_batch_stride: " << params.block_table_batch_stride << "\n";
    std::cout << "page_block_size: " << params.page_block_size << "\n";
    std::cout << "p_dropout: " << params.p_dropout << "\n";
    std::cout << "p_dropout_in_uint8_t: " << (int)params.p_dropout_in_uint8_t << "\n";
    std::cout << "rp_dropout: " << params.rp_dropout << "\n";
    std::cout << "scale_softmax_rp_dropout: " << params.scale_softmax_rp_dropout << "\n";
    std::cout << "window_size_left: " << params.window_size_left << "\n";
    std::cout << "window_size_right: " << params.window_size_right << "\n";
    std::cout << "softcap: " << params.softcap << "\n";
    std::cout << "rand_seed: " << params.rand_seed << "\n";
    std::cout << "rand_offset: " << params.rand_offset << "\n";
    std::cout << "dropout_debug_count: " << params.dropout_debug_count << "\n";
    std::cout << "rng_state: " << params.rng_state << "\n";
    std::cout << "is_bf16: " << params.is_bf16 << "\n";
    std::cout << "is_causal: " << params.is_causal << "\n";
    std::cout << "is_seqlens_k_cumulative: " << params.is_seqlens_k_cumulative << "\n";
    std::cout << "is_rotary_interleaved: " << params.is_rotary_interleaved << "\n";
    std::cout << "num_splits: " << params.num_splits << "\n";
    std::cout << "partition_size: " << params.partition_size << "\n";
    std::cout << "alibi_slopes_ptr: " << params.alibi_slopes_ptr << "\n";
    std::cout << "alibi_slopes_batch_stride: " << params.alibi_slopes_batch_stride << "\n";
    std::cout << "unpadded_lse: " << params.unpadded_lse << "\n";
    std::cout << "seqlenq_ngroups_swapped: " << params.seqlenq_ngroups_swapped << "\n";

    // 打印 Flash_bwd_params 独有成员
    std::cout << "q_ptr: " << params.q_ptr << "\n";
    std::cout << "k_ptr: " << params.k_ptr << "\n";
    std::cout << "v_ptr: " << params.v_ptr << "\n";
    std::cout << "o_ptr: " << params.o_ptr << "\n";
    std::cout << "softmax_lse_ptr: " << params.softmax_lse_ptr << "\n";
    std::cout << "do_ptr: " << params.do_ptr << "\n";
    std::cout << "dq_ptr: " << params.dq_ptr << "\n";
    std::cout << "dk_ptr: " << params.dk_ptr << "\n";
    std::cout << "dv_ptr: " << params.dv_ptr << "\n";
    std::cout << "dq_accum_ptr: " << params.dq_accum_ptr << "\n";
    std::cout << "dk_accum_ptr: " << params.dk_accum_ptr << "\n";
    std::cout << "dv_accum_ptr: " << params.dv_accum_ptr << "\n";
#ifdef DEBUGING
    std::cout << "kq_ptr: " << params.kq_ptr << "\n";
    std::cout << "s_ptr: " << params.s_ptr << "\n";
    std::cout << "dp_ptr: " << params.dp_ptr << "\n";
    std::cout << "ds_ptr: " << params.ds_ptr << "\n";
#endif
    std::cout << "do_batch_stride: " << params.do_batch_stride << "\n";
    std::cout << "do_row_stride: " << params.do_row_stride << "\n";
    std::cout << "do_head_stride: " << params.do_head_stride << "\n";
    std::cout << "dq_batch_stride: " << params.dq_batch_stride << "\n";
    std::cout << "dk_batch_stride: " << params.dk_batch_stride << "\n";
    std::cout << "dv_batch_stride: " << params.dv_batch_stride << "\n";
    std::cout << "dq_row_stride: " << params.dq_row_stride << "\n";
    std::cout << "dk_row_stride: " << params.dk_row_stride << "\n";
    std::cout << "dv_row_stride: " << params.dv_row_stride << "\n";
    std::cout << "dq_head_stride: " << params.dq_head_stride << "\n";
    std::cout << "dk_head_stride: " << params.dk_head_stride << "\n";
    std::cout << "dv_head_stride: " << params.dv_head_stride << "\n";
    std::cout << "dsoftmax_sum: " << params.dsoftmax_sum << "\n";
    std::cout << "deterministic: " << params.deterministic << "\n";
    std::cout << "dq_accum_split_stride: " << params.dq_accum_split_stride << "\n";
}



#define PRINT_BWD_PARAMS printf("b is %d\n", params.b); \
    printf("params.h is %d \n",params.h); \
    printf("params.h_k is %d \n",params.h_k); \
    printf("params.h_h_k_ratio is %d \n",params.h_h_k_ratio); \
    printf("params.d is %d \n",params.d); \
    printf("params.seqlen_q is %d \n",params.seqlen_q); \
    printf("params.seqlen_k is %d \n",params.seqlen_k); \
    printf("params.q_row_stride is %d \n",params.q_row_stride); \
    printf("params.k_row_stride is %d \n",params.k_row_stride); \
    printf("params.v_row_stride is %d \n",params.v_row_stride); \
    printf("params.o_row_stride is %d \n",params.o_row_stride); \
    printf("params.do_row_stride is %d \n",params.do_row_stride); \
    printf("params.dq_row_stride is %d \n",params.dq_row_stride); \
    printf("params.dk_row_stride is %d \n",params.dk_row_stride); \
    printf("params.dv_row_stride is %d \n",params.dv_row_stride); \
    printf("params.q_head_stride is %d \n",params.q_head_stride); \
    printf("params.k_head_stride is %d \n",params.k_head_stride); \
    printf("params.v_head_stride is %d \n",params.v_head_stride); \
    printf("params.o_head_stride is %d \n",params.o_head_stride); \
    printf("params.dq_head_stride is %d \n",params.dq_head_stride); \
    printf("params.do_head_stride is %d \n",params.do_head_stride); \
    printf("params.dk_head_stride is %d \n",params.dk_head_stride); \
    printf("params.dv_head_stride is %d \n",params.dv_head_stride); \
    printf("params.q_batch_stride is %d \n",params.q_batch_stride); \
    printf("params.k_batch_stride is %d \n",params.k_batch_stride); \
    printf("params.o_batch_stride is %d \n",params.o_batch_stride); \
    printf("params.do_batch_stride is %d \n",params.do_batch_stride); \
    printf("params.dq_batch_stride is %d \n",params.dq_batch_stride); \
    printf("params.dk_batch_stride is %d \n",params.dk_batch_stride); \
    printf("params.dv_batch_stride is %d \n",params.dv_batch_stride); \
    printf("params.scale_softmax is %d \n",params.scale_softmax); \
    printf("params.deterministic is %d \n",params.deterministic);



#define PRINT_MLA_PARAMS \
    printf("layout: %d\n", params.layout);\
    printf("mtp: %d\n", params.mtp);\
    printf("is_causal: %d\n", params.is_causal); \
    printf("is_bf16: %d\n", params.is_bf16); \
    printf("is_e4m3: %d\n", params.is_e4m3); \
    printf("b: %d\n", params.b);\
    printf("h: %d\n", params.h);\
    printf("h_k: %d\n", params.h_k);\
    printf("h_h_k_ratio: %d\n", params.h_h_k_ratio);\
    printf("total_q: %d\n", params.total_q);\
    printf("seqlen_q: %d\n", params.seqlen_q);\
    printf("seqlen_k: %d\n", params.seqlen_k);\
    printf("ngroups: %d\n", params.ngroups);\
    printf("seqlenq_ngroups_swapped: %d\n", params.seqlenq_ngroups_swapped);\
    printf("d: %d\n", params.d);\
    printf("q_batch_stride: %d\n", params.q_batch_stride);\
    printf("q_row_stride: %d\n", params.q_row_stride);\
    printf("q_head_stride: %d\n", params.q_head_stride);\
    printf("k_batch_stride: %d\n", params.k_batch_stride);\
    printf("k_row_stride: %d\n", params.k_row_stride);\
    printf("k_head_stride: %d\n", params.k_head_stride);\
    printf("v_batch_stride: %d\n", params.v_batch_stride);\
    printf("v_row_stride: %d\n", params.v_row_stride);\
    printf("v_head_stride: %d\n", params.v_head_stride);\
    printf("qv_batch_stride: %d\n", params.qv_batch_stride);\
    printf("qv_row_stride: %d\n", params.qv_row_stride);\
    printf("qv_head_stride: %d\n", params.qv_head_stride);\
    printf("o_batch_stride: %d\n", params.o_batch_stride);\
    printf("o_row_stride: %d\n", params.o_row_stride);\
    printf("o_head_stride: %d\n", params.o_head_stride);\
    printf("scale_softmax: %.12f\n", params.scale_softmax);\
    printf("scale_softmax_log2: %.12f\n", params.scale_softmax_log2);\
    printf("is_seqlens_k_cumulative: %d\n", params.is_seqlens_k_cumulative); \
    printf("q_ptr: %p\n", params.q_ptr); \
    printf("k_ptr: %p\n", params.k_ptr); \
    printf("v_ptr: %p\n", params.v_ptr); \
    printf("qv_ptr: %p\n", params.qv_ptr); \
    printf("o_ptr: %p\n", params.o_ptr); \
    printf("cu_seqlens_q: %p\n", params.cu_seqlens_q); \
    printf("cu_seqlens_k: %p\n", params.cu_seqlens_k); \
    printf("cu_seqlens_k_new: %p\n", params.cu_seqlens_k_new); \
    printf("leftpad_k: %p\n", params.leftpad_k); \
    printf("num_splits_ptr: %p\n", params.num_splits_ptr); \
    printf("tile_scheduler_metadata_ptr: %p\n", params.tile_scheduler_metadata_ptr); \
    printf("oaccum_ptr: %p\n", params.oaccum_ptr); \
    printf("scores_max_ptr: %p\n", params.scores_max_ptr); \
    printf("scores_sum_ptr: %p\n", params.scores_sum_ptr); \
    printf("softmax_lse_ptr: %p\n", params.softmax_lse_ptr); \
    printf("block_table : %p\n", params.block_table); \
    printf("page_block_size: %d\n", params.page_block_size); \
    printf("block_table_batch_stride: %d\n", params.block_table_batch_stride); \
    printf("num_splits : %d\n", params.num_splits); \
    printf("partition_size : %d\n", params.partition_size); \
    printf("splitkv_use_fp32_as_accum: %d\n", params.splitkv_use_fp32_as_accum); \
    printf("cu_count : %d\n", params.cu_count);