#pragma once
#include "flash_fwd_b16_mla.h"

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////
#include "kvcache/kvcache_qk_gemm_prefetch_v.h"
#include "kvcache/kvcache_pv_gemm_prefetch_k.h"
#include "kvcache/kvcache_softmax.h"
#include "kvcache/kvcache_acco_reduce.h"
#include "kvcache/kvcache_epilogue.h"

template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_Varlen, bool Is_local, bool Is_even_K, bool Return_softmax, bool Has_alibi, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, typename Params>
inline __device__ void compute_attn_mha_1rowblock_splitkv(const Params &params, const int bidb, const int bidh, const int WARP_ID) {
    using Element           = typename Kernel_traits::Element;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    using index_t           = typename Kernel_traits::index_t;
    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kBlockK   = Kernel_traits::kBlockK;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps   = Kernel_traits::kNWarps;
    constexpr int WARP_M    = Kernel_traits::kWaveM;
    constexpr int WARP_N    = Kernel_traits::kWaveN;
    constexpr int STAGES    = Kernel_traits::STAGES;
    constexpr int WARP_NUM = kBlockN / WARP_N;

    // flash::BlockInfo</*Varlen=*/true, !Is_Varlen> binfo(params, bidb);
    flash::SafeDecodeBlockInfo binfo;
    binfo.set_params<Params, /*Is_Q_Varlen=*/Is_Varlen, /*Is_K_Cumulative*/false>(params, bidb);

    // recompute the true actual_seqlen_k and num_split according to split_id, especially the last block
    int split_id;
    int partition_size;
    if constexpr (Split) {
        split_id = blockIdx.y;
        if constexpr (Is_Varlen) {
            partition_size = splitkv_get_partitionsize_of_fix_numsplits(binfo.actual_seqlen_k, params.num_splits);
            binfo.actual_seqlen_k = min(binfo.actual_seqlen_k - split_id * partition_size, partition_size);
        } else {
            partition_size = params.partition_size;
            int num_splits = max(1, floor_div(binfo.actual_seqlen_k, partition_size));
            binfo.actual_seqlen_k = (split_id == num_splits - 1)
                ? binfo.actual_seqlen_k - split_id * partition_size: partition_size;
            binfo.actual_seqlen_k = (split_id >= num_splits) ? 0: binfo.actual_seqlen_k;
            if (split_id >= num_splits) return;
        }
    }

    // kvcache doesn't has mask, no need to balance workload
    const int m_block = blockIdx.x;

    // 适配 varlen 场景
    int ngroups, actual_seqlen_q;
    if constexpr (Is_Varlen) {
        ngroups = params.ngroups;
        actual_seqlen_q = binfo.actual_seqlen_q * ngroups;
    } else {
        actual_seqlen_q = binfo.actual_seqlen_q;
    }

    // when groups is more than 32, this may lead to incorrect results
    if (m_block * kBlockM >= actual_seqlen_q || binfo.actual_seqlen_k == 0) return;

    // decide lds partition
    extern __shared__ Element smem[];
    // load Q --> QK gemm load K --> PV gemm load V, no conflicts
    Element* q_lds = reinterpret_cast<Element*>(smem) + 512/*1KB, 512 halfs, configured for max_lds*/;
    Element* k_lds = q_lds;
    Element* v_lds = q_lds;
    ElementAccum* acc_o_lds = reinterpret_cast<ElementAccum*>(smem); // prepare lds for max and acc whiling reducing results across 4 waves
    ElementAccum* max_lds   = acc_o_lds; // max and acc_o_lds has no conflicts while using lds

    // acquire stride over seqlen dimension
    const int query_seqlen_stride  = params.q_row_stride;
    const int kcache_seqlen_stride = params.k_row_stride;
    const int vcache_seqlen_stride = params.v_row_stride;

    // compute block table
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    block_table = block_table + (Split ? ceil_div(split_id * partition_size, page_block_size) : 0); // // if split, block_table begin from the new split!
    const int block_table_idx = 0;
    const int block_table_offset = 0;
    const int64_t row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const int64_t row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    // const int64_t row_offset_q = bidb * int64_t(params.q_batch_stride) + bidh * params.q_head_stride + m_block * kBlockM * query_seqlen_stride;
    const int64_t row_offset_q = Is_Varlen
        ? binfo.sum_s_q * ngroups * query_seqlen_stride + bidh * params.q_head_stride + m_block * kBlockM * query_seqlen_stride
        : bidb * int64_t(params.q_batch_stride) + bidh * params.q_head_stride + m_block * kBlockM * query_seqlen_stride;
    #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
        constexpr bool USE_CACHE_SWIZZLE = false;
    #else
        constexpr bool USE_CACHE_SWIZZLE = true; // for gfx928, cache swizzle have significant influence
    #endif
    auto gQ = prepare_for_buffer_load<kHeadDim, Element, USE_CACHE_SWIZZLE>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
    auto gK = prepare_for_buffer_load<kHeadDim, Element, USE_CACHE_SWIZZLE>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto gV = prepare_for_buffer_load<kHeadDimV, Element, USE_CACHE_SWIZZLE>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v);
    // compute lse offset of necessary for splitkv
    int row_offset_lse;
    ElementAccum * scores_sum_ptr;
    ElementAccum * scores_max_ptr;
    ElementAccum * softmax_lse_ptr;
    if constexpr (Split) {
        int row_offset_scores_split;
        if constexpr (Is_Varlen) {
            row_offset_lse = bidh * ngroups * params.total_q + binfo.sum_s_q + m_block * kBlockM;
            row_offset_scores_split = split_id * (params.h * ngroups * params.total_q);
            softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + row_offset_lse + row_offset_scores_split;
        } else {
            row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
            row_offset_scores_split = split_id * (params.b * params.h * params.seqlen_q);
            scores_sum_ptr = reinterpret_cast<ElementAccum*>(params.scores_sum_ptr) + row_offset_lse + row_offset_scores_split;
            scores_max_ptr = reinterpret_cast<ElementAccum*>(params.scores_max_ptr) + row_offset_lse + row_offset_scores_split;
        }
    } else {
        if constexpr (Is_Varlen) {
            row_offset_lse = bidh * ngroups * params.total_q + binfo.sum_s_q + m_block * kBlockM;
            softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + row_offset_lse;
        } else {
            row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
            softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + row_offset_lse;
        }
    }

    // prepare for alibi
    float gAlibi;
    if constexpr (Has_alibi) {
        gAlibi = reinterpret_cast<ElementAccum*>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    }

    // prefetch Q to vgprs
    union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2];
    kvcache_prefetch_q_to_vgpr<kHeadDim, kBlockM, kBlockK, WARP_M, WARP_NUM, Element, STAGES, REUSE_KV_TIMES, M_MMAC_COUNT>(
        gQ, q_lds, q_reg, WARP_ID, query_seqlen_stride, actual_seqlen_q - m_block * kBlockM);

    // initialze
    vec2_Accum<ElementAccum> scores_max[WARP_M / 32];
    vec2_Accum<ElementAccum> scores_sum[WARP_M / 32];
    vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4];
    attention_initialize<kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, M_MMAC_COUNT, ElementAccum>(scores_max, scores_sum, acc_o);

    // Mainloop
    int n_block_loop = 0;
    int n_block_max  = ceil_div(binfo.actual_seqlen_k, kBlockN);
    for (; n_block_loop < n_block_max - 1; ++n_block_loop) {

        int warp_offset_in_seqkv = n_block_loop * kBlockN + WARP_ID * WARP_N;
        int warp_seqkv_limit     = binfo.actual_seqlen_k - warp_offset_in_seqkv;

        if constexpr (STAGES > 1) {
            kvcache_prefetch_k_to_lds<kHeadDim, kBlockM, kBlockK, WARP_M, WARP_N, Element, STAGES, WARP_NUM>(gK, k_lds, WARP_ID, kcache_seqlen_stride, warp_seqkv_limit);
        }

        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (WARP_N / 32)][4];

        STAGES <= 2
        ? kvcache_qk_gemm_prefetch_v<kHeadDim, kHeadDimV, kBlockM, WARP_N, kBlockK, WARP_M, WARP_N, WARP_NUM, STAGES, M_MMAC_COUNT, Element, ElementAccum>(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, kcache_seqlen_stride, vcache_seqlen_stride, warp_seqkv_limit)
        : kvcache_qk_gemm_prefetch_v_3stage<kHeadDim, kHeadDimV, kBlockM, WARP_N, kBlockK, WARP_M, WARP_N, WARP_NUM, STAGES, M_MMAC_COUNT, Element, ElementAccum>(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, kcache_seqlen_stride, vcache_seqlen_stride, warp_seqkv_limit);

        if constexpr (Has_alibi) {
            kvcache_apply_alibi<vec4_Accum<ElementAccum>, WARP_M, WARP_N, M_MMAC_COUNT>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, m_block * kBlockM, actual_seqlen_q, gAlibi);
        }

        if constexpr (Is_Varlen) {
            kvcache_apply_mask_causal<vec4_Accum<ElementAccum>, WARP_M, WARP_N, M_MMAC_COUNT>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, m_block * kBlockM, actual_seqlen_q, ngroups);
        }

        if constexpr (Is_local) {
            kvcache_apply_mask_local</*HasWSLeft=*/Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN, M_MMAC_COUNT>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, m_block * kBlockM + WARP_ID * WARP_M, actual_seqlen_q, params.window_size_left, params.window_size_right);
        }

        kvcache_softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, ElementAccum, kHeadDimV, kBlockK, WARP_M, WARP_N, WARP_NUM, M_MMAC_COUNT>(s_reg, scores_max, scores_sum, acc_o, max_lds, WARP_ID, params.scale_softmax_log2);

        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (WARP_N / 32)][4];
        kvcache_convert_pk_type<WARP_M, WARP_N, M_MMAC_COUNT, Element, ElementAccum>(p_reg, s_reg);

        const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;

        *(int64_t*)&gK += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * params.k_row_stride) * sizeof(Element);

        STAGES <= 2
        ? kvcache_pv_gemm_prefetch_k<false/*prefetch_k*/, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, WARP_N, STAGES, WARP_NUM, M_MMAC_COUNT, Element, ElementAccum>(gV, gK, v_lds, k_lds, p_reg, acc_o, WARP_ID, vcache_seqlen_stride, warp_seqkv_limit)
        : kvcache_pv_gemm_prefetch_k_3stage<false/*prefetch_k*/, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, WARP_N, STAGES, WARP_NUM, M_MMAC_COUNT, Element, ElementAccum>(gV, gK, v_lds, k_lds, p_reg, acc_o, WARP_ID, vcache_seqlen_stride, warp_seqkv_limit);

        *(int64_t*)&gV += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * params.v_row_stride) * sizeof(Element);

    }

    // rest loop, mask need to be applied
    {
        int warp_offset_in_seqkv = n_block_loop * kBlockN + WARP_ID * WARP_N;
        int warp_seqkv_limit     = binfo.actual_seqlen_k - warp_offset_in_seqkv;

        if constexpr (STAGES > 1) {
            kvcache_prefetch_k_to_lds<kHeadDim, kBlockM, kBlockK, WARP_M, WARP_N, Element, STAGES, WARP_NUM>(gK, k_lds, WARP_ID, kcache_seqlen_stride, warp_seqkv_limit);
        }

        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (WARP_N / 32)][4];

        STAGES <= 2
        ? kvcache_qk_gemm_prefetch_v<kHeadDim, kHeadDimV, kBlockM, WARP_N, kBlockK, WARP_M, WARP_N, WARP_NUM, STAGES, M_MMAC_COUNT, Element, ElementAccum>(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, kcache_seqlen_stride, vcache_seqlen_stride, warp_seqkv_limit)
        : kvcache_qk_gemm_prefetch_v_3stage<kHeadDim, kHeadDimV, kBlockM, WARP_N, kBlockK, WARP_M, WARP_N, WARP_NUM, STAGES, M_MMAC_COUNT, Element, ElementAccum>(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, kcache_seqlen_stride, vcache_seqlen_stride, warp_seqkv_limit);

        if constexpr (Has_alibi) {
            kvcache_apply_alibi<vec4_Accum<ElementAccum>, WARP_M, WARP_N, M_MMAC_COUNT>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, m_block * kBlockM, actual_seqlen_q, gAlibi);
        }

        if constexpr (not Is_causal and not Is_Varlen) {
            kvcache_apply_mask<vec4_Accum<ElementAccum>, WARP_M, WARP_N, M_MMAC_COUNT>(s_reg, warp_seqkv_limit);
        } else {
            kvcache_apply_mask_causal<vec4_Accum<ElementAccum>, WARP_M, WARP_N, M_MMAC_COUNT>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, m_block * kBlockM, actual_seqlen_q, ngroups);
        }

        if constexpr (Is_local) {
            kvcache_apply_mask_local</*HasWSLeft=*/Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN, M_MMAC_COUNT>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, m_block * kBlockM + WARP_ID * WARP_M, actual_seqlen_q, params.window_size_left, params.window_size_right);
        }

        kvcache_softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, ElementAccum, kHeadDimV, kBlockK, WARP_M, WARP_N, WARP_NUM, M_MMAC_COUNT>(s_reg, scores_max, scores_sum, acc_o, max_lds, WARP_ID, params.scale_softmax_log2);

        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (WARP_N / 32)][4];
        kvcache_convert_pk_type<WARP_M, WARP_N, M_MMAC_COUNT, Element, ElementAccum>(p_reg, s_reg);

        STAGES <= 2
        ? kvcache_pv_gemm_prefetch_k<false/*prefetch_k*/, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, WARP_N, STAGES, WARP_NUM, M_MMAC_COUNT, Element, ElementAccum>(gV, gK, v_lds, k_lds, p_reg, acc_o, WARP_ID, vcache_seqlen_stride, warp_seqkv_limit)
        : kvcache_pv_gemm_prefetch_k_3stage<false/*prefetch_k*/, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, WARP_N, STAGES, WARP_NUM, M_MMAC_COUNT, Element, ElementAccum>(gV, gK, v_lds, k_lds, p_reg, acc_o, WARP_ID, vcache_seqlen_stride, warp_seqkv_limit);
    }
    __syncthreads();

    int lane_id = threadIdx.x & 63;
    if constexpr (WARP_NUM > 1) {
        // reduce acc_o across 4 waves
        kvcache_acco_reduce<REUSE_KV_TIMES, kHeadDimV, kBlockK, WARP_M, M_MMAC_COUNT, WARP_NUM, ElementAccum>(acc_o, acc_o_lds, params.seqlen_q, WARP_ID, lane_id);
    }

    /**********************************************************************************************************************************/
    // Epilogue, 收尾工作
    // 收尾 1: 根据最后的归一化求和, 做 rescale
    kvcache_epilugue_rescale_acco<kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, M_MMAC_COUNT, ElementAccum>(acc_o, scores_sum);

    if constexpr (Is_Varlen) {
        kvcache_epilogue_store_softmax_lse<Is_Varlen, false, WARP_M / 32, M_MMAC_COUNT, ElementAccum>(
            scores_max, scores_sum, softmax_lse_ptr, params.scale_softmax, WARP_ID, threadIdx.x, lane_id, 0/*headdim_split_id*/, actual_seqlen_q - m_block * kBlockM, params.total_q, params.ngroups);

        // const int64_t row_offset_o = int64_t(binfo.sum_s_q * ngroups) * int64_t(params.o_row_stride) + bidh * params.o_head_stride/* + m_block * kBlockM * query_seqlen_stride*/;
        // {total_q, ngroups, num_heads, -1} --> {total_q, num_heads, ngroups, -1}
        const int64_t row_offset_o = binfo.sum_s_q * ngroups * int64_t(params.o_row_stride) + bidh * ngroups * params.o_head_stride + m_block * kBlockM * int64_t(params.o_row_stride);
        kvcache_varlen_epilogue_store_output<Params, kHeadDimV, kHeadDimV, Split, false/*Is_16x32*/, typename Kernel_traits::SplitkvAccumType, ElementAccum, kBlockM, kBlockK, WARP_NUM, kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, M_MMAC_COUNT>(
            acc_o, params, row_offset_o, actual_seqlen_q - m_block * kBlockM, bidb, bidh, m_block, split_id, 0/*headdim_split_id*/, WARP_ID, lane_id);
    } else {
        // 收尾 2: splitkv, 或者开启 debug 的情况下, 写出 scores_max, scores_sum
        kvcache_epilogue_store_max_sum<Split, false, WARP_M / 32, M_MMAC_COUNT, ElementAccum>(
            scores_max, scores_sum, scores_max_ptr, scores_sum_ptr, params.scale_softmax, WARP_ID, threadIdx.x, lane_id, 0, actual_seqlen_q - m_block * kBlockM);
        // 收尾 3: 写出 output
        kvcache_epilogue_store_output<Params, kHeadDimV, kHeadDimV, Split, false/*Is_16x32*/, typename Kernel_traits::SplitkvAccumType, ElementAccum, kBlockM, kBlockK, WARP_NUM, kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, M_MMAC_COUNT>(
            acc_o, params, bidb, bidh, m_block, split_id, 0, WARP_ID, lane_id);
    }

}



template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_Varlen, bool Is_local, bool Is_even_K, bool Return_softmax, bool Has_alibi, bool Is_GQA, bool Is_softcap, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, bool Append_KV, typename Params>
inline __device__ void compute_attn_splitkv(const Params &params) {
    // block id in sequence dimension
    const int m_block = blockIdx.x;

    // The block index for the head.
    const int bidh = Split ? blockIdx.z % params.h : blockIdx.y; // batch x num_head, num_head first

    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.z;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id     = __builtin_amdgcn_readfirstlane(warp_id_vec);
    flash::compute_attn_mha_1rowblock_splitkv<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_Varlen, Is_local, Is_even_K, Return_softmax, Has_alibi, Split, M_MMAC_COUNT, REUSE_KV_TIMES, Flash_fwd_params>(params, bidb, bidh, warp_id);
}





////////////////////////////////////////////////////////////////////////////////////////////////////
#include "kvcache/kvcache_qk_gemm_prefetch_v_tile16x32.h"
#include "kvcache/kvcache_pv_gemm_prefetch_k_tile16x32.h"
#include "kvcache/kvcache_softmax_tile16x32.h"
#include "kvcache/kvcache_acco_reduce_tile16x32.h"

template<typename Kernel_traits, bool Is_causal, bool Is_Varlen, bool Split, bool Is_local, int M_MMAC_COUNT, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_tile16x32(const Params &params, const int bidb, const int bidh, const int warp_id) {
    using Element          = typename Kernel_traits::Element;
    using ElementAccum     = typename Kernel_traits::ElementAccum;
    using index_t          = typename Kernel_traits::index_t;
    using SplitkvAccumType = typename Kernel_traits::SplitkvAccumType;
    constexpr int kBlockM  = Kernel_traits::kBlockM;
    constexpr int kBlockN  = Kernel_traits::kBlockN;
    constexpr int kBlockK  = Kernel_traits::kBlockK;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps  = Kernel_traits::kNWarps;
    constexpr int WARP_M   = Kernel_traits::kWaveM;
    constexpr int WARP_N   = Kernel_traits::kWaveN;
    constexpr int STAGES   = Kernel_traits::STAGES;
    constexpr int WARP_NUM = kBlockN / WARP_N;
    // split-D, 为了节省寄存器, 对输出 headdim 做切分, 获取当前 block 在 headdim 方向上的处理的长度
    constexpr int kHeadDimVSplit = kHeadDimV / HEADDIM_V_SPLIT;

    // flash::BlockInfo</*Varlen=*/true, !Is_Varlen> binfo(params, bidb);
    flash::SafeDecodeBlockInfo binfo;
    binfo.set_params<Params, /*Is_Q_Varlen=*/Is_Varlen, /*Is_K_Cumulative*/false>(params, bidb);

    // splitKV, 根据 split id 确定当前 split 在 seqlen_kv 上处理的长度
    int split_id;
    int original_actual_seqlen_k = binfo.actual_seqlen_k;
    int partition_size;
    if constexpr (Split) {
        split_id = blockIdx.y;
        if constexpr (Is_Varlen) {
            partition_size = splitkv_get_partitionsize_of_fix_numsplits(binfo.actual_seqlen_k, params.num_splits);
            binfo.actual_seqlen_k = min(binfo.actual_seqlen_k - split_id * partition_size, partition_size);
        } else {
            partition_size = params.partition_size;
            int num_splits = max(1, floor_div(binfo.actual_seqlen_k, partition_size));
            binfo.actual_seqlen_k = (split_id == num_splits - 1)
                ? binfo.actual_seqlen_k - split_id * partition_size: partition_size;
            binfo.actual_seqlen_k = (split_id >= num_splits) ? 0: binfo.actual_seqlen_k;
            if (split_id >= num_splits) return;
        }
    }

    // "seqlen_q 方向切块" 和 "head_size_v 切块" 共用一个 grid.x 维度, 以 head_size_v 方向优先排列
    int block_x = blockIdx.x;
    const int m_block          = block_x / HEADDIM_V_SPLIT;
    const int headdim_split_id = block_x & (HEADDIM_V_SPLIT - 1);

    // 适配 varlen 场景
    int ngroups, actual_seqlen_q;
    if constexpr (Is_Varlen) {
        ngroups = params.ngroups;
        actual_seqlen_q = binfo.actual_seqlen_q * ngroups;
    } else {
        ngroups = 1;
        actual_seqlen_q = binfo.actual_seqlen_q;
    }

    // 确定运行边界
    if (m_block * kBlockM >= actual_seqlen_q || binfo.actual_seqlen_k == 0) return;

    // 决定 lds 用量划分
    // 2 stage: 加载 Q 32x576 到 vgpr, 有足够的 lds 给 K/V 做 2-stage 预取, 此时 Q/K/V 的 lds 用量可以复用
    // 1 stage: 加载 Q 16x576 到 vgpr, 没足够的 lds 给 K/V 做 2-stage 预取, 此时 Q 的 lds 完全独占, K/V 可以复用
    // 4 wave 更新 scores_max/scores_sum 的值, 留充足的 1 KB
    // 4 wave 更新求和 acc_o, 此时 Q/K/V 使用完毕, 最大值也不需要再更新, acc_o 可以使用所有的 lds
    extern __shared__ Element smem[];
    Element* q_lds = reinterpret_cast<Element*>(smem) + 512/*1KB, 512 halfs, configured for max_lds*/;
    Element* k_lds = reinterpret_cast<Element*>(smem);
    Element* v_lds = q_lds;
    ElementAccum* acc_o_lds = reinterpret_cast<ElementAccum*>(smem);
    ElementAccum* max_lds   = acc_o_lds;

    // 计算 Q/Kvcache 在 seqlen 方向上的跨度
    int query_seqlen_stride  = params.q_row_stride;
    int kcache_seqlen_stride = params.k_row_stride;
    int vcache_seqlen_stride = params.v_row_stride;

    // 计算 seqlen_k 方向上的起点, 针对 local mask 设计
    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);

    // 计算当前 workgroup 的 Q/KVcache 起始地址
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    int this_split_seqlen_start = Split ? split_id * partition_size: 0;
    block_table = block_table + (Split ? ceil_div(this_split_seqlen_start, page_block_size) : 0);
    const int block_table_idx = n_block_min;
    const int block_table_offset = 0;
    const int64_t row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const int64_t row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const int64_t row_offset_q = Is_Varlen
        ? binfo.sum_s_q * ngroups * query_seqlen_stride + bidh * params.q_head_stride + m_block * kBlockM * query_seqlen_stride
        : bidb * int64_t(params.q_batch_stride) + bidh * params.q_head_stride + m_block * kBlockM * query_seqlen_stride;

    // 准备读取数据的 buffer resource 寄存器
    #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
        constexpr bool USE_CACHE_SWIZZLE = false;
    #else
        constexpr bool USE_CACHE_SWIZZLE = true; // for gfx928, cache swizzle have significant influence
    #endif
    auto q_addr = prepare_for_buffer_load<kHeadDim, Element, USE_CACHE_SWIZZLE>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
    auto k_addr = prepare_for_buffer_load<kHeadDim, Element, USE_CACHE_SWIZZLE>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto v_addr = prepare_for_buffer_load<kHeadDimV, Element, USE_CACHE_SWIZZLE>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v + headdim_split_id * kHeadDimVSplit);

    // splitkv, debug 场景下需要写出一些值, 例如 scores_max/scores_sum
    int row_offset_lse;
    ElementAccum * scores_sum_ptr;
    ElementAccum * scores_max_ptr;
    ElementAccum * softmax_lse_ptr;
    if constexpr (Split) {
        int row_offset_scores_split;
        if constexpr (Is_Varlen) {
            row_offset_lse = bidh * ngroups * params.total_q + binfo.sum_s_q + m_block * kBlockM;
            row_offset_scores_split = split_id * (params.h * ngroups * params.total_q);
            softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + row_offset_lse + row_offset_scores_split;
        } else {
            row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
            row_offset_scores_split = split_id * (params.b * params.h * params.seqlen_q);
            scores_sum_ptr = reinterpret_cast<ElementAccum*>(params.scores_sum_ptr) + row_offset_lse + row_offset_scores_split;
            scores_max_ptr = reinterpret_cast<ElementAccum*>(params.scores_max_ptr) + row_offset_lse + row_offset_scores_split;
        }
    } else {
        if constexpr (Is_Varlen) {
            row_offset_lse = bidh * ngroups * params.total_q + binfo.sum_s_q + m_block * kBlockM;
            softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + row_offset_lse;
        } else {
            row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
            softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + row_offset_lse;
        }
    }

    // 目前最小的计算块是 32x32, 平摊到 64 个线程是 16 个 Half, 4 组 16x16, 每一组一次性从 lds 读取 2 个 Half, 读取 2 次
    constexpr int M_WARP_COUNT = WARP_M / 32;
    constexpr int K_WARP_COUNT = kBlockK / 32;
    constexpr int N_WARP_COUNT = WARP_N / 32;
    constexpr int K_LOOP_COUNT = kHeadDimVSplit / kBlockK;
    constexpr int Q_LOAD_BLOCKS = STAGES == 2 ? (kHeadDim / kBlockK): 1;
    union_vec4_f16x2<Element> q_reg[Q_LOAD_BLOCKS * M_WARP_COUNT * K_WARP_COUNT * 2];

    // 预取 Q 到寄存器或者 lds
    kvcache_prefetch_q_to_vgpr_tile16x32<kHeadDim, kBlockM, kBlockK, WARP_M, WARP_NUM, Element, STAGES, M_MMAC_COUNT>(
        q_addr, q_lds, q_reg, warp_id, query_seqlen_stride, actual_seqlen_q - m_block * kBlockM);

    // 准备初始状态, scores_max/scores_max/acc_o
    vec2_Accum<ElementAccum> scores_max[M_WARP_COUNT];
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT];
    vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4];
    attention_initialize<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(scores_max, scores_sum, acc_o);

    /**********************************************************************************************************************************/
    // 主循环, 沿着 seqlenKV 维度, 每次 4 个 wave 共同计算一个 kBLOCKN
    const int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN); // Split ? ceil_div(Partition_Size, kBlockN): ceil_div(binfo.actual_seqlen_k, kBlockN);
    int n_block_loop      = n_block_min;
    for (; n_block_loop < n_block_max; ++n_block_loop) {

        int warp_offset_in_seqkv = n_block_loop * kBlockN + warp_id * WARP_N;
        int warp_seqkv_limit     = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        kvcache_prefetch_k_to_lds_tile16x32<kBlockK, WARP_N, Element, STAGES, WARP_NUM>(k_addr, k_lds, warp_id, kcache_seqlen_stride, warp_seqkv_limit);

        vec4_Accum<ElementAccum> s_reg[M_WARP_COUNT * N_WARP_COUNT][4];

        kvcache_qk_gemm_prefetch_v_tile16x32<kHeadDim, kHeadDimVSplit, kBlockM, WARP_N, kBlockK, WARP_M, WARP_N, WARP_NUM, STAGES, M_MMAC_COUNT, Element, ElementAccum>(
            q_addr, k_addr, v_addr, q_lds, k_lds, v_lds, q_reg, s_reg, warp_id, kcache_seqlen_stride, vcache_seqlen_stride, warp_seqkv_limit);

        if constexpr (Is_causal) {
            if constexpr (Is_Varlen) { /*varlen Q and mtp cannot work togather yet*/
                if constexpr (Is_local) {
                    kvcache_apply_local_mask_causal_tile16x32<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, warp_offset_in_seqkv + this_split_seqlen_start, original_actual_seqlen_k, m_block * kBlockM, actual_seqlen_q, ngroups, params.window_size_left, params.window_size_right);
                } else {
                    kvcache_apply_mask_causal_tile16x32<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, warp_offset_in_seqkv + this_split_seqlen_start, original_actual_seqlen_k, m_block * kBlockM, actual_seqlen_q, ngroups);
                }
            } else {
                kvcache_apply_mask_causal_tile16x32_mtp<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, warp_offset_in_seqkv + this_split_seqlen_start, original_actual_seqlen_k, m_block * kBlockM, actual_seqlen_q, params.mtp, params.layout);
            }
        } else {
            kvcache_apply_mask_tile16x32<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, warp_seqkv_limit, warp_id * WARP_N);
        }

        mla_softmax_rescale_o<Is_causal, ElementAccum, K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, N_WARP_COUNT, WARP_NUM, M_MMAC_COUNT>(
            s_reg, scores_max, scores_sum, acc_o, max_lds, warp_id, params.scale_softmax_log2);

        union_vec2_f16x2<Element> p_reg[M_WARP_COUNT * N_WARP_COUNT][4];
        mla_convert_pk_type<M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT, Element, ElementAccum>(p_reg, s_reg);

        const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;

        kvcache_pv_gemm_prefetch_k_tile16x32<K_LOOP_COUNT, kBlockM, kBlockK, kBlockN, M_WARP_COUNT, K_WARP_COUNT/*kBlockK*/, N_WARP_COUNT/*WARP_N*/, STAGES, WARP_NUM, M_MMAC_COUNT, Element, ElementAccum>(
            v_addr, k_addr, v_lds, k_lds, p_reg, acc_o, warp_id, vcache_seqlen_stride, warp_seqkv_limit);

        *(int64_t*)&k_addr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);
        *(int64_t*)&v_addr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);

    }

    __syncthreads();

    // 多个 wave 同步 acc_o
    int thread_id = threadIdx.x;
    int lane_id   = thread_id & 63;
    if constexpr (WARP_NUM > 1) {
        int reduced_q_len = Is_Varlen ? params.seqlen_q: actual_seqlen_q;
        kvcache_acco_reduce_tile16x32<K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, M_MMAC_COUNT, WARP_NUM, 4/*Padding*/, ElementAccum>(acc_o, acc_o_lds, reduced_q_len, warp_id, lane_id);
    }

    /**********************************************************************************************************************************/
    // Epilogue, 收尾工作
    // 收尾 1: 根据最后的归一化求和, 做 rescale
    if (params.s_aux_ptr != nullptr) {
        if constexpr (Split) {
            if (split_id == 0) {
                kvcache_apply_attention_sink<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
                    acc_o, scores_max, scores_sum, params.s_aux_ptr, params.s_aux_type,
                    bidh, ngroups, m_block, kBlockM, lane_id, params.scale_softmax);
            }
        } else {
            kvcache_apply_attention_sink<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
                acc_o, scores_max, scores_sum, params.s_aux_ptr, params.s_aux_type,
                bidh, ngroups, m_block, kBlockM, lane_id, params.scale_softmax);
        }
    }
    kvcache_epilugue_rescale_acco<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(acc_o, scores_sum);

    if constexpr (Is_Varlen) {
        kvcache_epilogue_store_softmax_lse<Is_Varlen, true, M_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
            scores_max, scores_sum, softmax_lse_ptr, params.scale_softmax, warp_id, thread_id, lane_id, headdim_split_id, actual_seqlen_q - m_block * kBlockM, params.total_q, ngroups);

        // {total_q, ngroups, num_heads, -1} --> {total_q, num_heads, ngroups, -1}
        const int64_t row_offset_o = binfo.sum_s_q * ngroups * int64_t(params.o_row_stride) + bidh * ngroups * params.o_head_stride + headdim_split_id * kHeadDimVSplit;
        kvcache_varlen_epilogue_store_output<Params, kHeadDimV, kHeadDimVSplit, Split, true/*Is_16x32*/, SplitkvAccumType, ElementAccum, kBlockM, kBlockK, WARP_NUM, K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT>(
            acc_o, params, row_offset_o, actual_seqlen_q - m_block * kBlockM, bidb, bidh, m_block, split_id, headdim_split_id, warp_id, lane_id);
    } else {
        // 收尾 2: splitkv, 或者开启 debug 的情况下, 写出 scores_max, scores_sum
        kvcache_epilogue_store_max_sum<Split, true/*Is_16x32*/, M_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
            scores_max, scores_sum, scores_max_ptr, scores_sum_ptr, params.scale_softmax, warp_id, thread_id, lane_id, headdim_split_id, actual_seqlen_q - m_block * kBlockM);
        // 收尾 3: 写出 output
        kvcache_epilogue_store_output<Params, kHeadDimV, kHeadDimVSplit, Split, true/*Is_16x32*/, SplitkvAccumType, ElementAccum, kBlockM, kBlockK, WARP_NUM, K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT>(
            acc_o, params, bidb, bidh, m_block, split_id, headdim_split_id, warp_id, lane_id);
    }

}


template<typename Kernel_traits, bool Is_causal, bool Is_Varlen, bool Split, bool Is_local, int M_MMAC_COUNT, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
inline __device__ void compute_attn_splitkv_tile16x32(const Params &params) {

    // The block index for the head.
    const int bidh = Split ? blockIdx.z % params.h : blockIdx.y; // batch x num_head, num_head first

    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.z;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id     = __builtin_amdgcn_readfirstlane(warp_id_vec);

    flash::compute_attn_1rowblock_splitkv_tile16x32<Kernel_traits, Is_causal, Is_Varlen, Split, Is_local, M_MMAC_COUNT, HEADDIM_V_SPLIT, Partition_Size * 128, Params>(params, bidb, bidh, warp_id);
}



////////////////////////////////////////////////////////////////////////////////////////////////////
//                                MLS-based Paged Attention, >= gfx938
////////////////////////////////////////////////////////////////////////////////////////////////////
#include "kvcache/gfx938/kvcache_qk_gemm_prefetch_v_gfx938.h"
#include "kvcache/gfx938/kvcache_pv_gemm_prefetch_k_gfx938.h"
#include "kvcache/gfx938/kvcache_softmax_gfx938.h"
#include "kvcache/gfx938/kvcache_epilogue_gfx938.h"

template<typename Kernel_traits, bool Is_causal, bool Is_Varlen, bool Split, int M_MMAC_COUNT, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_gfx938(const Params &params, const int bidb, const int bidh, const int warp_id) {
    using Element          = typename Kernel_traits::Element;
    using ElementAccum     = typename Kernel_traits::ElementAccum;
    using index_t          = typename Kernel_traits::index_t;
    using SplitkvAccumType = typename Kernel_traits::SplitkvAccumType;
    constexpr int kBlockM  = Kernel_traits::kBlockM;
    constexpr int kBlockN  = Kernel_traits::kBlockN;
    constexpr int kBlockK  = Kernel_traits::kBlockK;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps  = Kernel_traits::kNWarps;
    constexpr int WARP_M   = Kernel_traits::kWaveM;
    constexpr int WARP_N   = Kernel_traits::kWaveN;
    constexpr int STAGES   = Kernel_traits::STAGES;
    constexpr int WARP_NUM = kBlockN / WARP_N;
    // split-D, 为了节省寄存器, 对输出 headdim 做切分, 获取当前 block 在 headdim 方向上的处理的长度
    constexpr int kHeadDimVSplit = kHeadDimV / HEADDIM_V_SPLIT;

    // flash::BlockInfo</*Varlen=*/true, /*Is_Kvcache*/true> binfo(params, bidb);
    flash::SafeDecodeBlockInfo binfo;
    binfo.set_params<Params, /*Is_Q_varlen=*/Is_Varlen, /*Is_K_Cumulative=*/false>(params, bidb);

    // splitKV, 根据 split id 确定当前 split 在 seqlen_kv 上处理的长度
    int split_id = 0;
    int original_actual_seqlen_k = binfo.actual_seqlen_k;
    int partition_size = 0;
    if constexpr (Split) {
        split_id = blockIdx.y;
        if constexpr (Is_Varlen) {
            partition_size = splitkv_get_partitionsize_of_fix_numsplits(binfo.actual_seqlen_k, params.num_splits);
            binfo.actual_seqlen_k = min(binfo.actual_seqlen_k - split_id * partition_size, partition_size);
        } else {
            partition_size = params.partition_size;
            int num_splits = max(1, floor_div(binfo.actual_seqlen_k, partition_size));
            binfo.actual_seqlen_k = (split_id == num_splits - 1)
                ? binfo.actual_seqlen_k - split_id * partition_size: partition_size;
            binfo.actual_seqlen_k = (split_id >= num_splits) ? 0: binfo.actual_seqlen_k;
            if (split_id >= num_splits) return;
        }
    }

    // "seqlen_q 方向切块" 和 "head_size_v 切块" 共用一个 grid.x 维度, 以 head_size_v 方向优先排列
    int block_x = blockIdx.x;
    const int m_block          = block_x / HEADDIM_V_SPLIT;
    const int headdim_split_id = block_x & (HEADDIM_V_SPLIT - 1);

    // 适配 varlen 场景
    int ngroups, actual_seqlen_q;
    if constexpr (Is_Varlen) {
        ngroups = params.ngroups;
        actual_seqlen_q = binfo.actual_seqlen_q * ngroups;
    } else {
        ngroups = 1;
        actual_seqlen_q = binfo.actual_seqlen_q;
    }

    // 确定运行边界
    if (m_block * kBlockM >= actual_seqlen_q || binfo.actual_seqlen_k <= 0) return;

    // 决定 lds 用量划分
    extern __shared__ Element smem[];
    Element* q_lds = reinterpret_cast<Element*>(smem);
    Element* k_lds = reinterpret_cast<Element*>(smem);
    Element* v_lds = k_lds;
    ElementAccum* acc_o_lds = reinterpret_cast<ElementAccum*>(smem);
    ElementAccum* max_lds   = acc_o_lds + 1024/*from 4096 bytes*/;

    // 计算 Q/Kvcache 在 seqlen 方向上的跨度
    int query_seqlen_stride  = params.q_row_stride;
    int kcache_seqlen_stride = params.k_row_stride;
    int vcache_seqlen_stride = params.v_row_stride;

    // 计算当前 workgroup 的 Q/KVcache 起始地址
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    int this_split_seqlen_start = Split ? split_id * partition_size: 0;
    block_table = block_table + (Split ? ceil_div(this_split_seqlen_start, page_block_size) : 0);
    const int block_table_idx = 0;
    const int block_table_offset = 0;
    const int64_t row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const int64_t row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const int64_t row_offset_q = Is_Varlen
        ? binfo.sum_s_q * ngroups * query_seqlen_stride + bidh * params.q_head_stride + m_block * kBlockM * query_seqlen_stride
        : bidb * int64_t(params.q_batch_stride) + bidh * params.q_head_stride + m_block * kBlockM * query_seqlen_stride;

    // 准备读取数据的 buffer resource 寄存器
    auto q_addr = prepare_for_buffer_load<kHeadDim, Element, false>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
    auto k_addr = prepare_for_buffer_load<kHeadDim, Element, false>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto v_addr = prepare_for_buffer_load<kHeadDimV, Element, false>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v + headdim_split_id * kHeadDimVSplit);

    bool is_thread0 = threadIdx.x == 0;
    if (is_thread0) {
        // inline_utcl2_warmup_dword(q_addr);
        // inline_utcl2_warmup_dword(k_addr);
        // inline_utcl2_warmup_dword(v_addr);
    }
    // Keep warmup buffer loads out of the MLS vmcnt schedule below.
    flash::wait_all_buffer_data_arrived<true>();

    // splitkv, debug 场景下需要写出一些值, 例如 scores_max/scores_sum
    int row_offset_lse;
    ElementAccum * scores_sum_ptr;
    ElementAccum * scores_max_ptr;
    ElementAccum * softmax_lse_ptr;
    if constexpr (Split) {
        int row_offset_scores_split;
        if constexpr (Is_Varlen) {
            row_offset_lse = bidh * ngroups * params.total_q + binfo.sum_s_q + m_block * kBlockM;
            row_offset_scores_split = split_id * (params.h * ngroups * params.total_q);
            softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + row_offset_lse + row_offset_scores_split;
        } else {
            row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
            row_offset_scores_split = split_id * (params.b * params.h * params.seqlen_q);
            scores_sum_ptr = reinterpret_cast<ElementAccum*>(params.scores_sum_ptr) + row_offset_lse + row_offset_scores_split;
            scores_max_ptr = reinterpret_cast<ElementAccum*>(params.scores_max_ptr) + row_offset_lse + row_offset_scores_split;
        }
    } else {
        if constexpr (Is_Varlen) {
            row_offset_lse = bidh * ngroups * params.total_q + binfo.sum_s_q + m_block * kBlockM;
            softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + row_offset_lse;
        } else {
            row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
            softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + row_offset_lse;
        }
    }

    // 目前最小的计算块是 32x32, 平摊到 64 个线程是 16 个 Half, 4 组 16x16, 每一组一次性从 lds 读取 2 个 Half, 读取 2 次
    constexpr int M_WARP_COUNT = WARP_M / 32;
    constexpr int K_WARP_COUNT = kBlockK / 32;
    constexpr int N_WARP_COUNT = WARP_N / 32;
    constexpr int K_LOOP_COUNT = kHeadDimVSplit / kBlockK;
    constexpr int Q_LOAD_BLOCKS = STAGES == 2 ? (kHeadDim / kBlockK): 1;
    union_vec4_f16x2<Element> q_reg[Q_LOAD_BLOCKS * M_WARP_COUNT * K_WARP_COUNT * 2];

    // 预取 Q 到寄存器或者 lds
    kvcache_prefetch_q_to_vgpr_gfx938<kHeadDim, kBlockM, kBlockK, WARP_M, WARP_NUM, Element, STAGES, M_MMAC_COUNT>(
        q_addr, q_lds, q_reg, warp_id, query_seqlen_stride, actual_seqlen_q - m_block * kBlockM);

    // 准备初始状态, scores_max/scores_max/acc_o
    vec2_Accum<ElementAccum> scores_max[M_WARP_COUNT];
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT];
    vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4];
    attention_initialize<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(scores_max, scores_sum, acc_o);

    /**********************************************************************************************************************************/
    // 主循环, 沿着 seqlenKV 维度, 每次 4 个 wave 共同计算一个 kBLOCKN
    const int n_block_min = 0;
    const int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN); // Split ? ceil_div(Partition_Size, kBlockN): ceil_div(binfo.actual_seqlen_k, kBlockN);
    int n_block_loop      = n_block_min;
    for (; n_block_loop < n_block_max; ++n_block_loop) {

        int warp_offset_in_seqkv = n_block_loop * kBlockN + warp_id * WARP_N;
        int warp_seqkv_limit     = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        kvcache_prefetch_k_to_lds_gfx938<kBlockK, WARP_N, Element, STAGES, WARP_NUM>(k_addr, k_lds, warp_id, kcache_seqlen_stride, warp_seqkv_limit);

        vec4_Accum<ElementAccum> s_reg[M_WARP_COUNT * N_WARP_COUNT][4];

        kvcache_qk_gemm_prefetch_v_gfx938<kHeadDim, kHeadDimVSplit, kBlockM, WARP_N, kBlockK, WARP_M, WARP_N, WARP_NUM, STAGES, M_MMAC_COUNT, Element, ElementAccum>(
            q_addr, k_addr, v_addr, q_lds, k_lds, v_lds, q_reg, s_reg, warp_id, kcache_seqlen_stride, vcache_seqlen_stride, warp_seqkv_limit);

        if constexpr (Is_causal) {
            if constexpr (Is_Varlen) {
                kvcache_apply_mask_causal_gfx938<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, warp_offset_in_seqkv + this_split_seqlen_start, original_actual_seqlen_k, m_block * kBlockM, actual_seqlen_q, ngroups);
            } else {
                kvcache_apply_mask_causal_gfx938_mtp<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, warp_offset_in_seqkv + this_split_seqlen_start, original_actual_seqlen_k, m_block * kBlockM, actual_seqlen_q, params.mtp, params.layout);
            }
        } else {
            kvcache_apply_mask_gfx938<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, warp_seqkv_limit, warp_id * WARP_N);
        }

        mla_softmax_rescale_o<Is_causal, ElementAccum, K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, N_WARP_COUNT, WARP_NUM, M_MMAC_COUNT>(
            s_reg, scores_max, scores_sum, acc_o, max_lds, warp_id, params.scale_softmax_log2);

        union_vec2_f16x2<Element> p_reg[M_WARP_COUNT * N_WARP_COUNT][4];
        mla_convert_pk_type<M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT, Element, ElementAccum>(p_reg, s_reg);

        const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;

        kvcache_pv_gemm_prefetch_k_gfx938<K_LOOP_COUNT, kBlockM, kBlockK, kBlockN, M_WARP_COUNT, K_WARP_COUNT/*kBlockK*/, N_WARP_COUNT/*WARP_N*/, STAGES, WARP_NUM, M_MMAC_COUNT, Element, ElementAccum>(
            v_addr, k_addr, v_lds, k_lds, p_reg, acc_o, warp_id, vcache_seqlen_stride, warp_seqkv_limit);

        *(int64_t*)&k_addr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * params.k_row_stride) * sizeof(Element);
        *(int64_t*)&v_addr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * params.v_row_stride) * sizeof(Element);

    }

    __syncthreads();

    // 多个 wave 同步 acc_o
    int thread_id = threadIdx.x;
    int lane_id   = thread_id & 63;
    if constexpr (WARP_NUM > 1) {
        kvcache_acco_reduce_tile16x32<K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, M_MMAC_COUNT, WARP_NUM, 0/*Padding*/, ElementAccum>(acc_o, acc_o_lds, params.seqlen_q, warp_id, lane_id);
    }

    /**********************************************************************************************************************************/
    // Epilogue, 收尾工作
    // 收尾 1: 根据最后的归一化求和, 做 rescale
    if (params.s_aux_ptr != nullptr) {
        if constexpr (Split) {
            if (split_id == 0) {
                kvcache_apply_attention_sink<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
                    acc_o, scores_max, scores_sum, params.s_aux_ptr, params.s_aux_type,
                    bidh, ngroups, m_block, kBlockM, lane_id, params.scale_softmax);
            }
        } else {
            kvcache_apply_attention_sink<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
                acc_o, scores_max, scores_sum, params.s_aux_ptr, params.s_aux_type,
                bidh, ngroups, m_block, kBlockM, lane_id, params.scale_softmax);
        }
    }
    kvcache_epilugue_rescale_acco<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(acc_o, scores_sum);

    // 收尾 2: splitkv, 或者开启 debug 的情况下, 写出 scores_max, scores_sum
    if constexpr (Is_Varlen) {
        kvcache_epilogue_store_softmax_lse<Is_Varlen, true, M_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
            scores_max, scores_sum, softmax_lse_ptr, params.scale_softmax, warp_id, thread_id, lane_id, headdim_split_id, actual_seqlen_q - m_block * kBlockM, params.total_q, params.ngroups);
    } else {
        kvcache_epilogue_store_max_sum<Split, true/*Is_16x32*/, M_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
            scores_max, scores_sum, scores_max_ptr, scores_sum_ptr, params.scale_softmax, warp_id, thread_id, lane_id, headdim_split_id, actual_seqlen_q - m_block * kBlockM);
    }

    // 收尾 3: 写出 output
    if constexpr (Is_Varlen) {
        // {total_q, ngroups, num_heads, -1} --> {total_q, num_heads, ngroups, -1}
        const int64_t row_offset_o = binfo.sum_s_q * ngroups * int64_t(params.o_row_stride) + bidh * ngroups * params.o_head_stride + headdim_split_id * kHeadDimVSplit + m_block * kBlockM * int64_t(params.o_row_stride);
        kvcache_varlen_epilogue_store_output_gfx938<Params, kHeadDimV, kHeadDimVSplit, Split, SplitkvAccumType, ElementAccum, kBlockM, kBlockK, WARP_NUM, K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT>(
            acc_o, params, row_offset_o, actual_seqlen_q - m_block * kBlockM, bidb, bidh, m_block, split_id, headdim_split_id, warp_id, lane_id);
    } else {
        kvcache_epilogue_store_output_gfx938<Params, kHeadDimV, kHeadDimVSplit, true/*alt*/, Split, SplitkvAccumType, ElementAccum, kBlockM, kBlockK, WARP_NUM, K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT>(
            acc_o, params, bidb, bidh, m_block, split_id, headdim_split_id, warp_id, lane_id);
    }
}


template<typename Kernel_traits, bool Is_causal, bool Is_Varlen, bool Split, int M_MMAC_COUNT, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
inline __device__ void compute_attn_splitkv_gfx938(const Params &params) {

#if defined(__gfx938__) || defined(__gfx946__)
    // The block index for the head.
    const int bidh = Split ? blockIdx.z % params.h : blockIdx.y; // batch x num_head, num_head first

    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.z;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id     = __builtin_amdgcn_readfirstlane(warp_id_vec);

    flash::compute_attn_1rowblock_splitkv_gfx938<Kernel_traits, Is_causal, Is_Varlen, Split, M_MMAC_COUNT, HEADDIM_V_SPLIT, Partition_Size * 128, Params>(params, bidb, bidh, warp_id);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//                                MLS-based Paged Attention, gfx92a
////////////////////////////////////////////////////////////////////////////////////////////////////
#include "kvcache/gfx92a/f16_kvcache_gfx92a.h"

template<typename Kernel_traits, bool Is_causal, bool Is_Varlen, bool Is_monopolize, bool Split, int M_MMAC_COUNT, int HEADDIM_V_SPLIT, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_gfx92a(const Params &params, const int bidb, const int bidh, const int warp_id) {
    using Element          = typename Kernel_traits::Element;
    using ElementAccum     = typename Kernel_traits::ElementAccum;
    using index_t          = typename Kernel_traits::index_t;
    using SplitkvAccumType = typename Kernel_traits::SplitkvAccumType;
    constexpr int kBlockM  = Kernel_traits::kBlockM;
    constexpr int kBlockN  = Kernel_traits::kBlockN;
    constexpr int kBlockK  = Kernel_traits::kBlockK;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps  = Kernel_traits::kNWarps;
    constexpr int WARP_M   = Kernel_traits::kWaveM;
    constexpr int WARP_N   = Kernel_traits::kWaveN;
    constexpr int STAGES   = Kernel_traits::STAGES;
    constexpr int WARP_NUM = kBlockN / WARP_N;
    constexpr int kHeadDimVSplit = kHeadDimV / HEADDIM_V_SPLIT;

    // flash::BlockInfo</*Varlen=*/true, /*Is_Kvcache*/true> binfo(params, bidb);
    flash::SafeDecodeBlockInfo binfo;
    binfo.set_params<Params, /*Is_Q_varlen=*/Is_Varlen, /*Is_K_Cumulative=*/false>(params, bidb);

    // SplitKV processing
    int split_id;
    int original_actual_seqlen_k = binfo.actual_seqlen_k;
    int partition_size;
    if constexpr (Split) {
        split_id = blockIdx.y;
        if constexpr (Is_Varlen) {
            partition_size = splitkv_get_partitionsize_of_fix_numsplits(binfo.actual_seqlen_k, params.num_splits);
            binfo.actual_seqlen_k = min(binfo.actual_seqlen_k - split_id * partition_size, partition_size);
        } else {
            partition_size = params.partition_size;
            int num_splits = max(1, floor_div(binfo.actual_seqlen_k, partition_size));
            binfo.actual_seqlen_k = (split_id == num_splits - 1)
                ? binfo.actual_seqlen_k - split_id * partition_size: partition_size;
            binfo.actual_seqlen_k = (split_id >= num_splits) ? 0: binfo.actual_seqlen_k;
            if (split_id >= num_splits) return;
        }
    }

    // acquire TG id
    int block_x                = blockIdx.x;
    const int m_block          = block_x / HEADDIM_V_SPLIT;
    const int headdim_split_id = block_x & (HEADDIM_V_SPLIT - 1);

    // Compute seqQ
    int ngroups, actual_seqlen_q;
    if constexpr (Is_Varlen) {
        ngroups = params.ngroups;
        actual_seqlen_q = binfo.actual_seqlen_q * ngroups;
    } else {
        actual_seqlen_q = binfo.actual_seqlen_q;
    }

    // Running boundaries
    if (m_block * kBlockM >= actual_seqlen_q || binfo.actual_seqlen_k <= 0) return;

    // Decide lsa usage
    extern __shared__ Element smem[];
    Element* q_lds          = reinterpret_cast<Element*>(smem);
    Element* k_lds          = reinterpret_cast<Element*>(smem);
    Element* v_lds          = Is_monopolize ? k_lds + 16384: k_lds;
    ElementAccum* acc_o_lds = reinterpret_cast<ElementAccum*>(smem);
    ElementAccum* max_lds   = acc_o_lds + 1024/*from 4096 bytes*/;

    // Acquire stride along seq dimension of q/k/v
    int query_seqlen_stride  = params.q_row_stride;
    int kcache_seqlen_stride = params.k_row_stride;
    int vcache_seqlen_stride = params.v_row_stride;

    // Compute q and k/v block table address
    int page_block_size = params.page_block_size;
    int this_split_seqlen_start = Split ? split_id * partition_size: 0;
    int *block_table    = params.block_table + bidb * params.block_table_batch_stride;
    block_table         = block_table + (Split ? ceil_div(this_split_seqlen_start, page_block_size) : 0);
    const int block_table_idx    = 0;
    const int block_table_offset = 0;
    const int64_t row_offset_k   = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const int64_t row_offset_v   = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const int64_t row_offset_q   = Is_Varlen
        ? binfo.sum_s_q * ngroups * query_seqlen_stride + bidh * ngroups * params.q_head_stride + m_block * kBlockM * query_seqlen_stride
        : bidb * int64_t(params.q_batch_stride) + bidh * params.q_head_stride + m_block * kBlockM * query_seqlen_stride;

    // Prepare buffer resource for q/k/v
    Element* q_ptr = reinterpret_cast<Element*>(params.q_ptr) + row_offset_q;
    auto q_addr = prepare_for_buffer_load<kHeadDim, Element, false>(q_ptr);
    auto k_addr = prepare_for_buffer_load<kHeadDim, Element, false>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto v_addr = prepare_for_buffer_load<kHeadDimV, Element, false>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v + headdim_split_id * kHeadDimVSplit);

    // utcl2 warmup
    if constexpr (false) {
        bool is_thread0 = threadIdx.x == 0;
        if (is_thread0) {
            inline_utcl2_warmup_dword(q_addr);
            inline_utcl2_warmup_dword(k_addr);
            inline_utcl2_warmup_dword(v_addr);
        }
    }

    // Compute lse/max/sum pointers of this TG
    int row_offset_lse;
    ElementAccum * scores_sum_ptr;
    ElementAccum * scores_max_ptr;
    ElementAccum * softmax_lse_ptr;
    if constexpr (Split) {
        int row_offset_scores_split;
        if constexpr (Is_Varlen) {
            row_offset_lse = bidh * ngroups * params.total_q + binfo.sum_s_q + m_block * kBlockM;
            row_offset_scores_split = split_id * (params.h * ngroups * params.total_q);
            softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + row_offset_lse + row_offset_scores_split;
        } else {
            row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
            row_offset_scores_split = split_id * (params.b * params.h * params.seqlen_q);
            scores_sum_ptr = reinterpret_cast<ElementAccum*>(params.scores_sum_ptr) + row_offset_lse + row_offset_scores_split;
            scores_max_ptr = reinterpret_cast<ElementAccum*>(params.scores_max_ptr) + row_offset_lse + row_offset_scores_split;
        }
    } else {
        if constexpr (Is_Varlen) {
            row_offset_lse = bidh * ngroups * params.total_q + binfo.sum_s_q + m_block * kBlockM;
            softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + row_offset_lse;
        } else {
            row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
            softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + row_offset_lse;
        }
    }

    // hold q regs
    constexpr int M_WARP_COUNT  = WARP_M / 32;
    constexpr int K_WARP_COUNT  = kBlockK / 32;
    constexpr int N_WARP_COUNT  = WARP_N / 32;
    constexpr int K_LOOP_COUNT  = kHeadDimVSplit / kBlockK;
    constexpr int Q_LOAD_BLOCKS = STAGES == 2 ? (kHeadDim / kBlockK): 1;
    union_vec4_f16x2<Element> q_reg[Q_LOAD_BLOCKS * M_WARP_COUNT * K_WARP_COUNT * 2];

    // prefetch Q into vgprs, can be hide
    gfx92a::kvcache_prefetch_q_to_vgpr<Is_Varlen, kHeadDim, kBlockK, WARP_M, WARP_NUM, M_MMAC_COUNT, Element>(
        q_ptr, q_lds, q_reg, warp_id, query_seqlen_stride, params.q_head_stride, ngroups, actual_seqlen_q - m_block * kBlockM);

    // Initialize, scores_max/scores_max/acc_o
    vec2_Accum<ElementAccum> scores_max[M_WARP_COUNT];
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT];
    vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4];
    attention_initialize<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(scores_max, scores_sum, acc_o);

    // Mainloop, along seqlenkv dimension, 4 warps computes attention of a kBlockN
    const int n_block_min = 0;
    const int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    int n_block_loop      = n_block_min;
    for (; n_block_loop < n_block_max; ++n_block_loop) {

        int warp_offset_in_seqkv = n_block_loop * kBlockN + warp_id * WARP_N;
        int warp_seqkv_limit     = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        constexpr int prefetchKLevel = 4;
        constexpr int prefetchVLevel = Is_monopolize ? 4: 2;
        constexpr bool prefetchK     = Is_monopolize;
        if constexpr (prefetchK) {
            if (n_block_loop == n_block_min) gfx92a::kvcache_prefetch_k_to_lds<kBlockK, WARP_N, prefetchKLevel, Element>(k_addr, k_lds, warp_id, kcache_seqlen_stride, warp_seqkv_limit);
        } else {
            gfx92a::kvcache_prefetch_k_to_lds<kBlockK, WARP_N, prefetchKLevel, Element>(k_addr, k_lds, warp_id, kcache_seqlen_stride, warp_seqkv_limit);
        }

        vec4_Accum<ElementAccum> s_reg[M_WARP_COUNT * N_WARP_COUNT][4];

        gfx92a::kvcache_qk_gemm_prefetch_v<kHeadDim, kHeadDimVSplit, WARP_N, kBlockK, WARP_M, WARP_N, prefetchKLevel, prefetchVLevel, M_MMAC_COUNT, Element, ElementAccum>(
            k_addr, v_addr, k_lds, v_lds, q_reg, s_reg, warp_id, kcache_seqlen_stride, vcache_seqlen_stride, warp_seqkv_limit);

        const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;
        int table_diff;
        int table_cur, table_next;
        if constexpr (prefetchK) {
            inline_global_load_dwordx1(table_cur, block_table_idx_cur, block_table);
            inline_global_load_dwordx1(table_next, block_table_idx_next, block_table);
        }

        gfx92a::kvcache_prefetch_v_to_lds<kHeadDimV, kBlockK, kBlockK, STAGES, prefetchVLevel, Element>(v_addr, v_lds, warp_id, vcache_seqlen_stride, warp_seqkv_limit);

        if constexpr (Is_causal) {
            gfx92a::kvcache_apply_mask_causal<M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT, Is_Varlen>(s_reg, warp_offset_in_seqkv + this_split_seqlen_start, original_actual_seqlen_k, m_block * kBlockM, actual_seqlen_q, ngroups, params.mtp, params.layout);
        } else {
            gfx92a::kvcache_apply_mask<M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, warp_seqkv_limit, warp_id * WARP_N);
        }

        mla_softmax_rescale_o<Is_causal, ElementAccum, K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, N_WARP_COUNT, WARP_NUM, M_MMAC_COUNT>(
            s_reg, scores_max, scores_sum, acc_o, max_lds, warp_id, params.scale_softmax_log2);

        union_vec2_f16x2<Element> p_reg[M_WARP_COUNT * N_WARP_COUNT][4];
        gfx92a::convert_attn_f32_to_f16<M_MMAC_COUNT, Element, ElementAccum>(s_reg, p_reg);

        if constexpr (prefetchK) {
            flash::wait_buffer_data_arrived<true/*can be false*/>(prefetchVLevel/*4 for hdim 128*/);
            table_diff = __builtin_amdgcn_readfirstlane(table_next - table_cur);
        } else {
            table_diff = __builtin_amdgcn_readfirstlane(block_table[block_table_idx_next] - block_table[block_table_idx_cur]);
        }
        *(int64_t*)&k_addr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * params.k_row_stride) * sizeof(Element);

        gfx92a::kvcache_pv_gemm_prefetch_k<prefetchK, K_LOOP_COUNT, kBlockK, kBlockN, M_WARP_COUNT, K_WARP_COUNT, N_WARP_COUNT, STAGES, prefetchKLevel, prefetchVLevel, M_MMAC_COUNT, Element, ElementAccum>(
            v_addr, k_addr, v_lds, k_lds, p_reg, acc_o, warp_id, vcache_seqlen_stride, kcache_seqlen_stride, warp_seqkv_limit);

        *(int64_t*)&v_addr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * params.v_row_stride) * sizeof(Element);

    }

    // reduce pv results among 4 warps
    int thread_id = threadIdx.x;
    int lane_id   = thread_id & 63;
    if constexpr (WARP_NUM > 1) {
        gfx92a::kvcache_acco_reduce_tile16x32<K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, M_MMAC_COUNT, WARP_NUM, ElementAccum>(acc_o, acc_o_lds, params.seqlen_q, warp_id, lane_id);
    }

    /**********************************************************************************************************************************/
    // Epilogue 1: rescaling acc_o
    kvcache_epilugue_rescale_acco<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(acc_o, scores_sum);

    // Epilogue 2: store lse / max / sum for splitkv reduction
    if constexpr (Is_Varlen) {
        kvcache_epilogue_store_softmax_lse<Is_Varlen, true, M_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
            scores_max, scores_sum, softmax_lse_ptr, params.scale_softmax, warp_id, thread_id, lane_id, headdim_split_id, actual_seqlen_q - m_block * kBlockM, params.total_q, params.ngroups);
    } else {
        kvcache_epilogue_store_max_sum<Split, true/*Is_16x32*/, M_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
            scores_max, scores_sum, scores_max_ptr, scores_sum_ptr, params.scale_softmax, warp_id, thread_id, lane_id, headdim_split_id, actual_seqlen_q - m_block * kBlockM);
    }

    // Epilogue 3: store acc_o into global memory
    int64_t row_offset_o = Is_Varlen ? binfo.sum_s_q * ngroups * int64_t(params.o_row_stride) + bidh * ngroups * params.o_head_stride + headdim_split_id * kHeadDimVSplit + (Split ? split_id * params.ngroups * int64_t(params.total_q) * params.o_row_stride: 0)
                         : bidb * int64_t(params.o_batch_stride) + bidh * params.o_head_stride + headdim_split_id * kHeadDimVSplit + (Split ? split_id * params.b * params.o_batch_stride: 0);
    gfx92a::kvcache_varlen_epilogue_store_output<Is_Varlen, Split, kBlockK, WARP_NUM, K_LOOP_COUNT, M_MMAC_COUNT, SplitkvAccumType, ElementAccum>(
        acc_o, params, row_offset_o, actual_seqlen_q - m_block * kBlockM, warp_id, lane_id);
}


template<typename Kernel_traits, bool Is_causal, bool Is_Varlen, bool Is_monopolize, bool Split, int M_MMAC_COUNT, int HEADDIM_V_SPLIT, typename Params>
inline __device__ void compute_attn_splitkv_gfx92a(const Params &params) {

#if defined(__gfx92a__)
    // The block index for the head.
    const int bidh = Split ? blockIdx.z % params.h : blockIdx.y; // batch x num_head, num_head first

    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.z;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id     = __builtin_amdgcn_readfirstlane(warp_id_vec);

    flash::compute_attn_1rowblock_splitkv_gfx92a<Kernel_traits, Is_causal, Is_Varlen, Is_monopolize, Split, M_MMAC_COUNT, HEADDIM_V_SPLIT, Params>(params, bidb, bidh, warp_id);
#endif
}



////////////////////////////////////////////////////////////////////////////////////////////////////
//                                     FMA-based Paged Attention
////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename Kernel_traits, bool Is_causal, bool Split, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_splitkv_mha_fma_kernel(Params params) {

    using Element           = typename Kernel_traits::Element;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    using index_t           = typename Kernel_traits::index_t;
    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kBlockK   = Kernel_traits::kBlockK;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int WARP_NUM  = 4;
    constexpr int WARP_SIZE = 64;
    constexpr int LOAD_REQUESTS = kBlockN / WARP_NUM;

    // 获取任务划分
    const int bidb = blockIdx.z;
    const int bidh = blockIdx.y;
    const int m_block = blockIdx.x;
    int split_id = 0;

    // 获取当前任务需要处理的 seqlen 长度
    const int actual_seqlen_q = 1;
    const int actual_seqlen_k = params.cu_seqlens_k[bidb];

    // 确定运行边界
    if (m_block * kBlockM >= actual_seqlen_q || actual_seqlen_k <= 0) return;

    // 计算当前 block 读取数据 Q 的起始偏移量
    const int query_seqlen_stride = kHeadDim;/*bhsd*/
    int64_t row_offset_q = (bidb * params.h + bidh) * int64_t(query_seqlen_stride);
    int64_t row_offset_o = (bidb * params.h + bidh) * int64_t(query_seqlen_stride);

    // 计算当前 block 读取 KVcache 的起始偏移量
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    int this_split_seqlen_start = Split ? split_id * params.partition_size: 0;
    block_table = block_table + (Split ? ceil_div(this_split_seqlen_start, page_block_size) : 0);
    const int block_table_idx    = 0;
    const int block_table_offset = 0;
    const int64_t row_offset_k   = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + bidh * params.k_head_stride;
    const int64_t row_offset_v   = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + bidh * params.v_head_stride;

    // 得到 Q, kvcache 的起始地址
    Element* q_addr = reinterpret_cast<Element*>(params.q_ptr) + row_offset_q;
    Element* k_addr = reinterpret_cast<Element*>(params.k_ptr) + row_offset_k;
    Element* v_addr = reinterpret_cast<Element*>(params.v_ptr) + row_offset_v;
    Element* o_addr = reinterpret_cast<Element*>(params.o_ptr) + row_offset_o;

    // 4 个 wave, 都把数据读取到寄存器, 1 * 128 个 Half, 每个 wave 64 个线程分别读取 2 个 Half, 即 1 个 dword
    int thread_id = int(threadIdx.x);
    int lane_id   = thread_id & 63;
    int wave_id   = thread_id >> 6;
    vec2_Element<Element> q_reg = *(vec2_Element<Element>*)(q_addr + lane_id * 2);
    ElementAccum q_reg_f32s[2];
    q_reg_f32s[0] = splitkv_upcast_to_f32<Element>(q_reg[0]);
    q_reg_f32s[1] = splitkv_upcast_to_f32<Element>(q_reg[1]);

    // 初始化 acc_o, scores_sum, scores_max
    vec2_Accum<ElementAccum> acc_o; // 1 * 128, 每个线程负责 2 个 Half, 一个 dword, 但是 acc_o 需要 rescale 以及多 wave reduce, 所以采用 float 高精度
    for (int i = 0; i < 2; ++i) acc_o.f32[i] = 0;
    ElementAccum scores_max = -INFINITY; // seqlen_q = 1, 只需要一个最大值
    ElementAccum scores_sum = 0;

    // 准备必要的 lds
    #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
        __shared__ ElementAccum lds[4096]; // 16384 bytes, allow 4 waves per simd
    #else
        __shared__ ElementAccum lds[16384]; // 65536 bytes, allow 1 waves per simd for zd
    #endif

    // 累加循环需要偏移几个 kBlockN
    int kv_loop_count = ceil_div(actual_seqlen_k, kBlockN);

    for (int kv_loop = 0; kv_loop < kv_loop_count; ++kv_loop) {
        // ===================== load K =======================
        vec2_Element<Element> k_reg[LOAD_REQUESTS];
        // kBlockN, 每次 4 个 wave 读 4 行, 需要读取 32 次
        #pragma unroll
        for (int load = 0; load < LOAD_REQUESTS; ++load) {
            int k_row_offset  = min(load * WARP_NUM + wave_id, actual_seqlen_k - 1 - kv_loop * kBlockN);
            int k_load_offset = k_row_offset * kHeadDim + lane_id * 2;
            k_reg[load] = *(vec2_Element<Element>*)(k_addr + k_load_offset);
        }
        // ===================== QK gemm =======================
        ElementAccum s_reg[LOAD_REQUESTS];
        #pragma unroll
        for (int load = 0; load < LOAD_REQUESTS; ++load) {
            ElementAccum k_reg_f32s[2];
            k_reg_f32s[0] = splitkv_upcast_to_f32<Element>(k_reg[load][0]);
            k_reg_f32s[1] = splitkv_upcast_to_f32<Element>(k_reg[load][1]);
            s_reg[load]   = q_reg_f32s[0] * k_reg_f32s[0];
            s_reg[load]  += q_reg_f32s[1] * k_reg_f32s[1];
        }
        // 每个 wave, 64 个线程之间求和, 需要 reduce
        #pragma unroll
        for (int load = 0; load < LOAD_REQUESTS; ++load) {
            #pragma unroll
            for (int step = WARP_SIZE >> 1/*64 threads*/; step > 0; step = (step >> 1)) {
                s_reg[load] = s_reg[load] + __shfl_xor_tmp(s_reg[load], step);
            }
        }

        // ===================== Apply mask =======================
        #pragma unroll
        for (int load = 0; load < LOAD_REQUESTS; ++load) {
            bool within_mask = (load * WARP_NUM + wave_id + kv_loop * kBlockN) < actual_seqlen_k;
            s_reg[load] = within_mask ? s_reg[load]: -INFINITY;
        }

        // ===================== Max =======================
        // 现在每个 wave 得到了当前 wave 的 qk 矩阵乘结果了, 开始求这一行的最大值
        // 每个线程在这一行拥有 32 个数据(64 个线程拥有的值是一样的)
        ElementAccum max_in_wave = s_reg[0];
        #pragma unroll
        for (int load = 1; load < LOAD_REQUESTS; ++load) {
            max_in_wave = max(max_in_wave, s_reg[load]);
        }
        // 现在每个 wave 的线程都得到了这一行的最大值, 需要 4 个 wave 之间交换数据, 求 4 个 wave 的最大值
        if (lane_id == 0) lds[wave_id] = max_in_wave;
        __syncthreads();
        if (thread_id == 0) {
            float lds_max = lds[0]; for (int i = 1; i < WARP_NUM; ++i) lds_max = max(lds_max, lds[i]); lds[0] = lds_max;
        }
        __syncthreads();
        ElementAccum max_across_waves = lds[0];
        __syncthreads();
        // ===================== Update max =======================
        // 先拿前一笔的最大值 和 这一笔的最大值比较
        ElementAccum old_scores_max = scores_max;
        scores_max = max(scores_max, max_across_waves);
        // ===================== Rescale =======================
        // 获取 rescale 系数
        float scores_scale = kv_loop > 0 ? __llvm_exp2_f32((old_scores_max - scores_max) * params.scale_softmax_log2): 1.f;
        scores_sum   *= scores_scale;
        acc_o.f32[0] *= scores_scale;
        acc_o.f32[1] *= scores_scale;
        // ===================== Exp =======================
        #pragma unroll
        for (int load = 0; load < LOAD_REQUESTS; ++load) {
            s_reg[load] = __llvm_exp2_f32((s_reg[load] - scores_max) * params.scale_softmax_log2);
        }
        // 先 wave 内求和
        ElementAccum sum_in_wave = s_reg[0];
        #pragma unroll
        for (int load = 1; load < LOAD_REQUESTS; ++load) { // 这里一直累加有累加误差
            sum_in_wave = sum_in_wave + s_reg[load]; // 这里可以凑 fma
        }
        // 再 wave 间求和
        __syncthreads();
        if (lane_id == 0) lds[wave_id] = sum_in_wave;
        __syncthreads();
        if (thread_id == 0) {
            float lds_sum = lds[0]; for (int i = 1; i < WARP_NUM; ++i) lds_sum = lds_sum + lds[i]; lds[0] = lds_sum;
        }
        __syncthreads();
        ElementAccum sum_across_waves = lds[0];
        __syncthreads();
        // ===================== Update sum =======================
        scores_sum += sum_across_waves;

        // ===================== load V =======================
        vec2_Element<Element> v_reg[LOAD_REQUESTS];
        // kBlockN, 每次 4 个 wave 读 4 行, 需要读取 32 次
        #pragma unroll
        for (int load = 0; load < LOAD_REQUESTS; ++load) {
            int v_row_offset  = min(load * WARP_NUM + wave_id, actual_seqlen_k - 1 - kv_loop * kBlockN);
            int v_load_offset = v_row_offset * kHeadDimV + lane_id * 2;
            v_reg[load] = *(vec2_Element<Element>*)(v_addr + v_load_offset);
        }

        // ===================== PV gemm =======================
        // V 每个线程 hold 两个 Half, 实际可能是 2 个 float
        #pragma unroll
        for (int load = 0; load < LOAD_REQUESTS; ++load) {
            ElementAccum v_reg_f32s[2];
            v_reg_f32s[0] = splitkv_upcast_to_f32<Element>(v_reg[load][0]);
            v_reg_f32s[1] = splitkv_upcast_to_f32<Element>(v_reg[load][1]);
            acc_o.f32[0]  += s_reg[load] * v_reg_f32s[0];
            acc_o.f32[1]  += s_reg[load] * v_reg_f32s[1];
        }

        // 页表偏移
        const int block_table_idx_cur     = kv_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = kv_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(kv_loop_count - 1, kv_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(kv_loop_count - 1, kv_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;
        *(int64_t*)&k_addr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);
        *(int64_t*)&v_addr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);
    }

    // kv loop 结束, 4 个 wave 的 acc_o reduce 在一起
    // 每个 wave 都有 128 个 float, 4 个 wave 就有 512 个 float
    // 先各自把数据写到对应位置上
    __syncthreads();
    lds[thread_id] = acc_o.f32[0];
    lds[thread_id + WARP_NUM * WARP_SIZE/*256*/] = acc_o.f32[1];
    __syncthreads();
    if (wave_id == 0) {
        acc_o.f32[0] = lds[thread_id];
        acc_o.f32[1] = lds[thread_id + WARP_NUM * WARP_SIZE/*256*/];
        for (int i = 1; i < WARP_NUM; ++i) {
            acc_o.f32[0] += lds[thread_id + WARP_SIZE * i];
            acc_o.f32[1] += lds[thread_id + WARP_SIZE * i + WARP_NUM * WARP_SIZE/*256*/];
        }
        // 做 rescale, 归一化
        acc_o.f32[0] = acc_o.f32[0] / scores_sum;
        acc_o.f32[1] = acc_o.f32[1] / scores_sum;
        // 写出到 global memory, 1 x 128 个 float -> half, 每个线程负责 2 个 half
        vec2_Element<Element> output;
        output[0] = DownCast<ElementAccum, Element, true>(acc_o.f32[0]);
        output[1] = DownCast<ElementAccum, Element, true>(acc_o.f32[1]);
        *(vec2_Element<Element>*)(o_addr + thread_id * 2) = output;
    }
}




template<typename Kernel_traits, bool Is_causal, bool Split, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_splitkv_mha_kernel(Params params) {

    using Element           = typename Kernel_traits::Element;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    using index_t           = typename Kernel_traits::index_t;
    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kBlockK   = Kernel_traits::kBlockK;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int WARP_NUM  = 4;
    constexpr int WARP_SIZE = 64;
    constexpr int LOAD_DWORD_COUNT = 4; // 一次 load 4 个 dword
    constexpr int LOAD_REQUESTS = kBlockN / (WARP_NUM * LOAD_DWORD_COUNT);
    constexpr int LOAD_ELEMENTS = LOAD_DWORD_COUNT * 2; // 4 个 dword, 意味着每次 load 8 个 Half

    // 获取任务划分
    const int bidb = blockIdx.z;
    const int bidh = blockIdx.y;
    const int m_block = blockIdx.x;
    int split_id = 0;

    // 获取当前任务需要处理的 seqlen 长度
    const int actual_seqlen_q = 1;
    const int actual_seqlen_k = params.cu_seqlens_k[bidb];

    // 确定运行边界
    if (m_block * kBlockM >= actual_seqlen_q || actual_seqlen_k <= 0) return;

    // 计算当前 block 读取数据 Q 的起始偏移量
    const int query_seqlen_stride = kHeadDim;/*bhsd*/
    int64_t row_offset_q = (bidb * params.h + bidh) * int64_t(query_seqlen_stride);
    int64_t row_offset_o = (bidb * params.h + bidh) * int64_t(query_seqlen_stride);

    // 计算当前 block 读取 KVcache 的起始偏移量
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    int this_split_seqlen_start = Split ? split_id * params.partition_size: 0;
    block_table = block_table + (Split ? ceil_div(this_split_seqlen_start, page_block_size) : 0);
    const int block_table_idx    = 0;
    const int block_table_offset = 0;
    const int64_t row_offset_k   = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * int64_t(params.k_row_stride) + bidh * params.k_head_stride;
    const int64_t row_offset_v   = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * int64_t(params.v_row_stride) + bidh * params.v_head_stride;

    // 得到 Q, kvcache 的起始地址
    Element* q_addr = reinterpret_cast<Element*>(params.q_ptr) + row_offset_q;
    Element* k_addr = reinterpret_cast<Element*>(params.k_ptr) + row_offset_k;
    Element* v_addr = reinterpret_cast<Element*>(params.v_ptr) + row_offset_v;
    Element* o_addr = reinterpret_cast<Element*>(params.o_ptr) + row_offset_o;

    // 4 个 wave, 都把数据读取到寄存器, 1 * 128 个 Half, 每个 wave 64 个线程分别读取 2 个 Half, 即 1 个 dword
    int thread_id = int(threadIdx.x);
    int lane_id   = thread_id & 63;
    int wave_id   = thread_id >> 6;
    union_vec4_f16x2<Element> q_reg = *(union_vec4_f16x2<Element>*)(q_addr + (lane_id & 15) * LOAD_ELEMENTS); // 每个线程管理 8 个 half
    ElementAccum q_reg_f32s[LOAD_ELEMENTS];
    #pragma unroll
    for (int i = 0; i < LOAD_ELEMENTS; ++i) {
        q_reg_f32s[i] = splitkv_upcast_to_f32<Element>(q_reg.f16[i]);
    }

    // 判别数据类型是 fp16/bf16
    constexpr bool is_fp16 = std::is_same<Element, half_t>::value;

    // 初始化 acc_o, scores_sum, scores_max
    union_vec8_fp32 acc_o; // 1 * 128, 每个线程负责 8 个 Half, 一个 dword, 但是 acc_o 需要 rescale 以及多 wave reduce, 所以采用 float 高精度
    #pragma unroll
    for (int i = 0; i < LOAD_ELEMENTS; ++i) {
        acc_o.f32[i] = 0;
    }
    ElementAccum scores_max = -INFINITY; // seqlen_q = 1, 只需要一个最大值
    ElementAccum scores_sum = 0;

    // 准备必要的 lds
    #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
        __shared__ ElementAccum lds[4096]; // 16384 bytes, allow 4 waves per simd
    #else
        __shared__ ElementAccum lds[16384]; // 65536 bytes, allow 1 waves per simd for zd
    #endif

    // 累加循环需要偏移几个 kBlockN
    int kv_loop_count = ceil_div(actual_seqlen_k, kBlockN);

    for (int kv_loop = 0; kv_loop < kv_loop_count; ++kv_loop) { // potential promotion: QK fma, PV mmac
        // ===================== load K =======================
        union_vec4_f16x2<Element> k_reg[LOAD_REQUESTS];
        #pragma unroll
        for (int load = 0; load < LOAD_REQUESTS; ++load) {
            int k_row_offset = min(wave_id * 4 + (lane_id >> 4) + load * WARP_NUM * 4, actual_seqlen_k - 1 - kv_loop * kBlockN);
            int k_load_offset = k_row_offset * kHeadDim + (lane_id & 15) * LOAD_ELEMENTS;
            k_reg[load] = *(union_vec4_f16x2<Element>*)(k_addr + k_load_offset);
        }
        // ===================== QK gemm =======================
        ElementAccum s_reg[LOAD_REQUESTS];
        #pragma unroll
        for (int load = 0; load < LOAD_REQUESTS/*8次 Load*/; ++load) {
            // 每次 load 要做 8-8 个 Half 的乘法和加法
            s_reg[load] = 0;
            if constexpr (is_fp16) { // fp16 可以使用 v_dot2_f32_f16 指令加速
                #pragma unroll
                for (int i = 0; i < (LOAD_ELEMENTS / 2)/*8个 half*/; ++i) {
                    s_reg[load] = __builtin_amdgcn_fdot2(q_reg.b16x2[i], k_reg[load].b16x2[i], s_reg[load], true/*clamp*/);
                }
            } else { // gfx936/gfx928/gfx938 doesn't support v_dot2_f32_f16
                #pragma unroll
                for (int i = 0; i < LOAD_ELEMENTS/*8个 half*/; ++i) {
                    ElementAccum k_reg_f32 = splitkv_upcast_to_f32<Element>(k_reg[load].f16[i]);
                    s_reg[load] += q_reg_f32s[i] * k_reg_f32;
                }
            }
        }
        // 0-15, 16-31, 32-47, 48-63 各自持有的 8 个数据分别求和
        #pragma unroll
        for (int step = 16 >> 1; step > 0; step = (step >> 1)) {
            #pragma unroll
            for (int load = 0; load < LOAD_REQUESTS; ++load) {
                s_reg[load] = s_reg[load] + __shfl_xor_tmp(s_reg[load], step);
            }
        }
        // ===================== Apply mask =======================
        #pragma unroll
        for (int load = 0; load < LOAD_REQUESTS; ++load) {
            bool within_mask = wave_id * 4 + (lane_id >> 4) + load * WARP_NUM * 4 + kv_loop * kBlockN < actual_seqlen_k;
            s_reg[load] = within_mask ? s_reg[load]: -INFINITY;
        }
        // ===================== Max =======================
        __syncthreads();
        // 每个线程持有的 8 个数据求最大值
        ElementAccum max_in_reg = s_reg[0];
        #pragma unroll
        for (int load = 1; load < LOAD_REQUESTS; ++load) {
            max_in_reg = max(max_in_reg, s_reg[load]);
        }
        // 每个 wave 内部持有的 4 个线程求最大值
        ElementAccum max_in_wave = max_in_reg;
        max_in_wave = max(max_in_wave, __shfl_xor_tmp(max_in_wave, 32));
        max_in_wave = max(max_in_wave, __shfl_xor_tmp(max_in_wave, 16));
        // 4 个 wave 之间持有的最大值之间求最大值
        if (lane_id == 0) lds[wave_id] = max_in_wave;
        __syncthreads();
        if (thread_id == 0) {
            float lds_max = lds[0]; for (int i = 1; i < WARP_NUM; ++i) lds_max = max(lds_max, lds[i]); lds[0] = lds_max;
        }
        __syncthreads();
        ElementAccum max_across_waves = lds[0];
        // ===================== Update max =======================
        ElementAccum old_scores_max = scores_max;
        scores_max = max(scores_max, max_across_waves);
        // ===================== Rescale =======================
        // 获取 rescale 系数
        ElementAccum scale_softmax_log2 = params.scale_softmax_log2;
        float scores_scale = kv_loop > 0 ? __llvm_exp2_f32((old_scores_max - scores_max) * scale_softmax_log2): 1.f;
        scores_sum *= scores_scale;
        // ===================== Exp =======================
        ElementAccum exp_bias = -scores_max * scale_softmax_log2;
        #pragma unroll
        for (int load = 0; load < LOAD_REQUESTS; ++load) {
            ElementAccum scaled_qk = __llvm_fma_f32(s_reg[load], scale_softmax_log2, exp_bias);
            s_reg[load] = __llvm_exp2_f32(scaled_qk); // __llvm_exp2_f32((s_reg[load] - scores_max) * params.scale_softmax_log2);
        }
        // 先线程内求和
        ElementAccum sum_in_reg = s_reg[0];
        #pragma unroll
        for (int load = 1; load < LOAD_REQUESTS; ++load) {
            sum_in_reg = sum_in_reg + s_reg[load];
        }
        // 再 wave 内求和
        ElementAccum sum_in_wave = sum_in_reg;
        sum_in_wave = sum_in_wave + __shfl_xor_tmp(sum_in_wave, 32);
        sum_in_wave = sum_in_wave + __shfl_xor_tmp(sum_in_wave, 16);
        // 再 wave 间求和
        __syncthreads();
        if (lane_id == 0) lds[wave_id] = sum_in_wave;
        __syncthreads();
        if (thread_id == 0) {
            float lds_sum = lds[0]; for (int i = 1; i < WARP_NUM; ++i) lds_sum = lds_sum + lds[i]; lds[0] = lds_sum;
        }
        __syncthreads();
        ElementAccum sum_across_waves = lds[0];
        // ===================== Update sum =======================
        scores_sum += sum_across_waves;
        // ===================== load V =======================
        union_vec4_f16x2<Element> v_reg[LOAD_REQUESTS];
        #pragma unroll
        for (int load = 0; load < LOAD_REQUESTS; ++load) {
            int v_row_offset = min(wave_id * 4 + (lane_id >> 4) + load * WARP_NUM * 4, actual_seqlen_k - 1 - kv_loop * kBlockN);
            int v_load_offset = v_row_offset * kHeadDim + (lane_id & 15) * LOAD_ELEMENTS;
            v_reg[load] = *(union_vec4_f16x2<Element>*)(v_addr + v_load_offset);
        }
        // ===================== PV gemm =======================
        // s_reg 每个线程有 8 次 load 的计算结果, 要分别与 V 的 8 次 load, 每次 8 个线程分别计算
        union_vec8_fp32 pv_1;
        #pragma unroll
        for (int i = 0; i < LOAD_ELEMENTS; ++i) {
            pv_1.f32[i] = 0;
            #pragma unroll
            for (int load = 0; load < LOAD_REQUESTS; ++load) {
                ElementAccum v_reg_f32 = splitkv_upcast_to_f32<Element>(v_reg[load].f16[i]); // v_pk_fma_f32 can be applied by packing s_reg[load]
                pv_1.f32[i] += s_reg[load] * v_reg_f32;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < LOAD_ELEMENTS; ++i) { // wave 内求和
            pv_1.f32[i] = pv_1.f32[i] + __shfl_xor_tmp(pv_1.f32[i], 32);
            pv_1.f32[i] = pv_1.f32[i] + __shfl_xor_tmp(pv_1.f32[i], 16);
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < LOAD_ELEMENTS; ++i) { // wave 间求和
            lds[thread_id + WARP_NUM * WARP_SIZE * i] = pv_1.f32[i];
        }
        __syncthreads();
        if (wave_id == 0) { // 0 号 wave reduce 其他 wave 的结果
            #pragma unroll
            for (int i = 0; i < LOAD_ELEMENTS; ++i) {
                #pragma unroll
                for (int w = 1; w < WARP_NUM; ++w) {
                    pv_1.f32[i] += lds[thread_id + w * WARP_SIZE + WARP_NUM * WARP_SIZE * i];
                }
                acc_o.f32[i] = __llvm_fma_f32(acc_o.f32[i], scores_scale, pv_1.f32[i]);
            }
        }

        // 页表偏移
        const int block_table_idx_cur     = kv_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = kv_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(kv_loop_count - 1, kv_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(kv_loop_count - 1, kv_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;
        *(int64_t*)&k_addr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);
        *(int64_t*)&v_addr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);
    }

    // 写出到 global memory
    if (thread_id < 16) {
        union_vec4_f16x2<Element> acc_o_f16;
        #pragma unroll
        for (int i = 0; i < LOAD_ELEMENTS; ++i) {
            // 做 rescale, 归一化, 并转换为半精度
            acc_o_f16.f16[i] = DownCast<ElementAccum, Element, false>(acc_o.f32[i] / scores_sum);
        }
        *(union_vec4_f16x2<Element>*)(o_addr + thread_id * LOAD_ELEMENTS) = acc_o_f16;
    }
}

}
