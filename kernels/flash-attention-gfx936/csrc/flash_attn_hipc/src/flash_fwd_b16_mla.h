#pragma once
#include "block_info.h"
#include "kernel_traits.h"
#include "splitkv.h"

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////
#include "mla/mla_qk_gemm_prefetch_v.h"
#include "mla/mla_pv_gemm_prefetch_k.h"
#include "mla/mla_softmax.h"
#include "mla/mla_acco_reduce.h"
#include "mla/mla_epilogue.h"

template<typename Kernel_traits, bool Is_causal, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_mla(const Params &params, const int bidb, const int bidh, const int warp_id) {
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
    binfo.set_params<Params, /*Is_Q_varlen=*/false, /*Is_K_Cumulative=*/false>(params, bidb);

    // splitKV, 根据 split id 确定当前 split 在 seqlen_kv 上处理的长度
    int split_id;
    int original_actual_seqlen_k = binfo.actual_seqlen_k;
    int partition_size;
    constexpr bool MLA_FIX_NUM_SPLITS = Partition_Size > MLA_MAX_SPLITS;
    if constexpr (Split) {
        if constexpr (MLA_FIX_NUM_SPLITS) {
            split_id = blockIdx.y;
            partition_size = splitkv_get_partitionsize_of_fix_numsplits(binfo.actual_seqlen_k, params.num_splits);
            binfo.actual_seqlen_k = min(binfo.actual_seqlen_k - split_id * partition_size, partition_size);
        } else {
            split_id = blockIdx.y;
            partition_size = params.partition_size;
            int num_splits = ceil_div(binfo.actual_seqlen_k, partition_size);
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

    // 确定运行边界
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k <= 0) return;

    // 决定 lds 用量划分
    // 2 stage: 加载 Q 32x576 到 vgpr, 有足够的 lds 给 K/V 做 2-stage 预取, 此时 Q/K/V 的 lds 用量可以复用
    // 1 stage: 加载 Q 16x576 到 vgpr, 没足够的 lds 给 K/V 做 2-stage 预取, 此时 Q 的 lds 完全独占, K/V 可以复用
    // 4 wave 更新 scores_max/scores_sum 的值, 留充足的 1 KB
    // 4 wave 更新求和 acc_o, 此时 Q/K/V 使用完毕, 最大值也不需要再更新, acc_o 可以使用所有的 lds
    extern __shared__ Element smem[];
    Element* q_lds = reinterpret_cast<Element*>(smem) + 512/*1KB, 512 halfs, configured for max_lds*/;
    Element* k_lds = (STAGES == 2) ? q_lds: q_lds + (kHeadDim / kBlockK * WARP_M / 2/*16x34 is used only*/ * 34)/*halfs*/;
    Element* v_lds = k_lds;
    ElementAccum* acc_o_lds = reinterpret_cast<ElementAccum*>(smem);
    ElementAccum* max_lds   = acc_o_lds;

    // 计算 Q/Kvcache 在 seqlen 方向上的跨度
    int query_seqlen_stride   = params.q_row_stride;
    int kvcache_seqlen_stride = params.k_row_stride;

    // 计算当前 workgroup 的 Q/KVcache 起始地址
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    int this_split_seqlen_start = Split ? split_id * partition_size: 0;
    block_table = block_table + (Split ? ceil_div(this_split_seqlen_start, page_block_size) : 0);
    const int block_table_idx = 0;
    const int block_table_offset = 0;
    const int64_t row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const int64_t row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const int64_t row_offset_q = bidb * int64_t(params.q_batch_stride) + bidh * params.q_head_stride + m_block * kBlockM * query_seqlen_stride;

    // 准备读取数据的 buffer resource 寄存器
    constexpr bool USE_CACHE_SWIZZLE = false;
    auto q_addr = prepare_for_buffer_load<kHeadDim, Element, USE_CACHE_SWIZZLE>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
    auto k_addr = prepare_for_buffer_load<kHeadDim, Element, USE_CACHE_SWIZZLE>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto v_addr = prepare_for_buffer_load<kHeadDimV, Element, USE_CACHE_SWIZZLE>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v + headdim_split_id * kHeadDimVSplit);

    // splitkv, debug 场景下需要写出一些值, 例如 scores_max/scores_sum
    int row_offset_lse;
    ElementAccum * softmax_lse_ptr;
    if constexpr (Split) {
        row_offset_lse = split_id * (params.b * params.h * params.seqlen_q) + (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + row_offset_lse;
    }

    // 目前最小的计算块是 32x32, 平摊到 64 个线程是 16 个 Half, 4 组 16x16, 每一组一次性从 lds 读取 2 个 Half, 读取 2 次
    constexpr int M_WARP_COUNT = WARP_M / 32;
    constexpr int K_WARP_COUNT = kBlockK / 32;
    constexpr int N_WARP_COUNT = WARP_N / 32;
    constexpr int K_LOOP_COUNT = kHeadDimVSplit / kBlockK;
    constexpr int Q_LOAD_BLOCKS = STAGES == 2 ? (kHeadDim / kBlockK): 1;
    vec2_Element<Element> q_reg[Q_LOAD_BLOCKS * M_WARP_COUNT * K_WARP_COUNT * 2][4];

    // 预取 Q 到寄存器或者 lds
    if constexpr (STAGES == 2) {
        mla_prefetch_q_to_vgpr<kHeadDim, kBlockM, kBlockK, WARP_M, WARP_NUM, Element, STAGES, REUSE_KV_TIMES, M_MMAC_COUNT>(
            q_addr, q_lds, q_reg, warp_id, query_seqlen_stride, binfo.actual_seqlen_q - m_block * kBlockM);
    } else if constexpr (STAGES == 1) {
        mla_prefetch_q_to_lds_stage1<kHeadDim, kBlockM, kBlockK, WARP_NUM, Element, STAGES, REUSE_KV_TIMES, M_MMAC_COUNT>(
            q_addr, q_lds, warp_id, query_seqlen_stride, binfo.actual_seqlen_q - m_block * kBlockM);
    }

    // 准备初始状态, scores_max/scores_max/acc_o
    vec2_Accum<ElementAccum> scores_max[M_WARP_COUNT];
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT];
    vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4];
    attention_initialize<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(scores_max, scores_sum, acc_o);

    /**********************************************************************************************************************************/
    // 主循环, 沿着 seqlenKV 维度, 每次 4 个 wave 共同计算一个 kBLOCKN
    const int n_block_min = 0;
    const int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    int n_block_loop      = n_block_min;
    for (; n_block_loop < n_block_max - 1; ++n_block_loop) {

        int warp_offset_in_seqkv = n_block_loop * kBlockN + warp_id * WARP_N;
        int warp_seqkv_limit     = binfo.actual_seqlen_k - warp_offset_in_seqkv;

        if constexpr (STAGES == 2) {
            mla_prefetch_k_to_lds<kBlockK, WARP_N, Element, STAGES, WARP_NUM>(k_addr, k_lds, warp_id, kvcache_seqlen_stride, warp_seqkv_limit);
        }

        vec4_Accum<ElementAccum> s_reg[M_WARP_COUNT * N_WARP_COUNT][4];

        if constexpr (STAGES == 2) {
            mla_qk_gemm_prefetch_v<kHeadDim, kHeadDimVSplit, kBlockM, WARP_N, kBlockK, WARP_M, WARP_N, WARP_NUM, STAGES, M_MMAC_COUNT, Element, ElementAccum>(
                q_addr, k_addr, v_addr, q_lds, k_lds, v_lds, q_reg, s_reg, warp_id, kvcache_seqlen_stride, warp_seqkv_limit);
        } else if constexpr (STAGES == 1) {
            mla_qk_gemm_prefetch_v_qinlds<kHeadDim, kHeadDimVSplit, kBlockM, WARP_N/*kBlockN for each wave*/, kBlockK, M_WARP_COUNT, N_WARP_COUNT, WARP_NUM, STAGES, M_MMAC_COUNT, Element, ElementAccum>(
                q_addr, k_addr, v_addr, q_lds, k_lds, v_lds, q_reg, s_reg, warp_id, kvcache_seqlen_stride, warp_seqkv_limit);
        }

        if constexpr (Is_causal) {
            mla_apply_mask_causal<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT, 8/*MTP_REGROUP_COUNT*/, REUSE_KV_TIMES>(s_reg, warp_offset_in_seqkv + this_split_seqlen_start, original_actual_seqlen_k, m_block * kBlockM, binfo.actual_seqlen_q, params.mtp, params.layout);
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

        mla_pv_gemm_prefetch_k<K_LOOP_COUNT, kBlockM, kBlockK, kBlockN, M_WARP_COUNT, K_WARP_COUNT/*kBlockK*/, N_WARP_COUNT/*WARP_N*/, STAGES, WARP_NUM, M_MMAC_COUNT, Element, ElementAccum>(
            v_addr, k_addr, v_lds, k_lds, p_reg, acc_o, warp_id, kvcache_seqlen_stride, warp_seqkv_limit);

        *(int64_t*)&k_addr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * params.k_row_stride) * sizeof(Element);
        *(int64_t*)&v_addr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * params.v_row_stride) * sizeof(Element);

    }

    // 非 MTP 场景下, 只有最后一次循环需要做 mla_apply_mask, 判断边界
    {
        int warp_offset_in_seqkv = n_block_loop * kBlockN + warp_id * WARP_N;
        int warp_seqkv_limit     = binfo.actual_seqlen_k - warp_offset_in_seqkv;

        if constexpr (STAGES == 2) {
            mla_prefetch_k_to_lds<kBlockK, WARP_N, Element, STAGES, WARP_NUM>(k_addr, k_lds, warp_id, kvcache_seqlen_stride, warp_seqkv_limit);
        }

        vec4_Accum<ElementAccum> s_reg[M_WARP_COUNT * N_WARP_COUNT][4];

        if constexpr (STAGES == 2) {
            mla_qk_gemm_prefetch_v<kHeadDim, kHeadDimVSplit, kBlockM, WARP_N, kBlockK, WARP_M, WARP_N, WARP_NUM, STAGES, M_MMAC_COUNT, Element, ElementAccum>(
                q_addr, k_addr, v_addr, q_lds, k_lds, v_lds, q_reg, s_reg, warp_id, kvcache_seqlen_stride, warp_seqkv_limit);
        } else if constexpr (STAGES == 1) {
            mla_qk_gemm_prefetch_v_qinlds<kHeadDim, kHeadDimVSplit, kBlockM, WARP_N/*kBlockN for each wave*/, kBlockK, M_WARP_COUNT, N_WARP_COUNT, WARP_NUM, STAGES, M_MMAC_COUNT, Element, ElementAccum>(
                q_addr, k_addr, v_addr, q_lds, k_lds, v_lds, q_reg, s_reg, warp_id, kvcache_seqlen_stride, warp_seqkv_limit);
        }

        if constexpr (not Is_causal) {
            mla_apply_mask<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, warp_seqkv_limit);
        } else {
            mla_apply_mask_causal<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT, 8/*MTP_REGROUP_COUNT*/, REUSE_KV_TIMES>(s_reg, warp_offset_in_seqkv + this_split_seqlen_start, original_actual_seqlen_k, m_block * kBlockM, binfo.actual_seqlen_q, params.mtp, params.layout);
        }

        mla_softmax_rescale_o<Is_causal, ElementAccum, K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, N_WARP_COUNT, WARP_NUM, M_MMAC_COUNT>(
            s_reg, scores_max, scores_sum, acc_o, max_lds, warp_id, params.scale_softmax_log2);

        union_vec2_f16x2<Element> p_reg[M_WARP_COUNT * N_WARP_COUNT][4];
        mla_convert_pk_type<M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT, Element, ElementAccum>(p_reg, s_reg);

        mla_pv_gemm_prefetch_k<K_LOOP_COUNT, kBlockM, kBlockK, kBlockN, M_WARP_COUNT, K_WARP_COUNT/*kBlockK*/, N_WARP_COUNT/*WARP_N*/, STAGES, WARP_NUM, M_MMAC_COUNT, Element, ElementAccum>(
            v_addr, k_addr, v_lds, k_lds, p_reg, acc_o, warp_id, kvcache_seqlen_stride, warp_seqkv_limit);

    }
    __syncthreads();

    // 多个 wave 同步 acc_o
    int thread_id = threadIdx.x;
    int lane_id = thread_id & 63;
    if constexpr (WARP_NUM > 1) {
        mla_acco_reduce<REUSE_KV_TIMES, K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, M_MMAC_COUNT, WARP_NUM, ElementAccum>(acc_o, acc_o_lds, params.seqlen_q, warp_id, lane_id);
    }

    /**********************************************************************************************************************************/
    // Epilogue, 收尾工作
    // 收尾 1: 根据最后的归一化求和, 做 rescale
    mla_epilugue_rescale_acco<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(acc_o, scores_sum);

    // 收尾 2: splitkv, 或者开启 debug 的情况下, 写出 scores_max, scores_sum
    mla_tp8_epilogue_store_softmax_lse<Split, false/*Is_16x32*/, M_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
        scores_max, scores_sum, softmax_lse_ptr, params.scale_softmax, warp_id, thread_id, lane_id, headdim_split_id, binfo.actual_seqlen_q - m_block * kBlockM);

    // 收尾 3: 写出 output
    mla_epilogue_store_output<Params, kHeadDimV, kHeadDimVSplit, Split, SplitkvAccumType, ElementAccum, kBlockM, kBlockK, WARP_NUM, K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT>(
        acc_o, params, bidb, bidh, m_block, split_id, headdim_split_id, warp_id, lane_id);

}


////////////////////////////////////////////////////////////////////////////////////////////////////
#include "mla/mla_qk_gemm_prefetch_v_tile16x32.h"
#include "mla/mla_pv_gemm_prefetch_k_tile16x32.h"
#include "mla/mla_softmax_tile16x32.h"
#include "mla/mla_acco_reduce_tile16x32.h"
#include "mla/mla_epilogue_tile16x32.h"

template<typename Kernel_traits, bool Is_causal, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_mla_tile16x32(const Params &params, const int bidb, const int bidh, const int warp_id) {
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
    binfo.set_params<Params, /*Is_Q_varlen=*/false, /*Is_K_Cumulative=*/false>(params, bidb);

    // splitKV, 根据 split id 确定当前 split 在 seqlen_kv 上处理的长度
    int split_id;
    int original_actual_seqlen_k = binfo.actual_seqlen_k;
    int partition_size;
    constexpr bool MLA_FIX_NUM_SPLITS = Partition_Size > MLA_MAX_SPLITS;
    if constexpr (Split) {
        if constexpr (MLA_FIX_NUM_SPLITS) {
            split_id = blockIdx.y;
            partition_size = splitkv_get_partitionsize_of_fix_numsplits(binfo.actual_seqlen_k, params.num_splits);
            binfo.actual_seqlen_k = min(binfo.actual_seqlen_k - split_id * partition_size, partition_size);
        } else {
            split_id = blockIdx.y;
            partition_size = params.partition_size;
            int num_splits = ceil_div(binfo.actual_seqlen_k, partition_size);
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

    // 确定运行边界
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k <= 0) return;

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
    int query_seqlen_stride   = params.q_row_stride;
    int kvcache_seqlen_stride = params.k_row_stride;

    // 计算当前 workgroup 的 Q/KVcache 起始地址
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    int this_split_seqlen_start = Split ? split_id * partition_size: 0;
    block_table = block_table + (Split ? ceil_div(this_split_seqlen_start, page_block_size) : 0);
    const int block_table_idx = 0;
    const int block_table_offset = 0;
    const int64_t row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const int64_t row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const int64_t row_offset_q = bidb * int64_t(params.q_batch_stride) + bidh * params.q_head_stride + m_block * kBlockM * query_seqlen_stride;

    // 准备读取数据的 buffer resource 寄存器
    constexpr bool USE_CACHE_SWIZZLE = false; // avoid compiler buffer_loadx4 and buffer_storex2 error
    auto q_addr = prepare_for_buffer_load<kHeadDim, Element, USE_CACHE_SWIZZLE>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
    auto k_addr = prepare_for_buffer_load<kHeadDim, Element, USE_CACHE_SWIZZLE>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto v_addr = prepare_for_buffer_load<kHeadDimV, Element, USE_CACHE_SWIZZLE>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v + headdim_split_id * kHeadDimVSplit);

    // splitkv, debug 场景下需要写出一些值, 例如 scores_max/scores_sum
    int row_offset_lse;
    ElementAccum * softmax_lse_ptr;
    if constexpr (Split) {
        row_offset_lse = split_id * (params.b * params.h * params.seqlen_q) + (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + row_offset_lse;
    }

    // 目前最小的计算块是 32x32, 平摊到 64 个线程是 16 个 Half, 4 组 16x16, 每一组一次性从 lds 读取 2 个 Half, 读取 2 次
    constexpr int M_WARP_COUNT = WARP_M / 32;
    constexpr int K_WARP_COUNT = kBlockK / 32;
    constexpr int N_WARP_COUNT = WARP_N / 32;
    constexpr int K_LOOP_COUNT = kHeadDimVSplit / kBlockK;
    constexpr int Q_LOAD_BLOCKS = STAGES == 2 ? (kHeadDim / kBlockK): 1;
    union_vec4_f16x2<Element> q_reg[Q_LOAD_BLOCKS * M_WARP_COUNT * K_WARP_COUNT * 2];

    // 预取 Q 到寄存器或者 lds
    mla_prefetch_q_to_vgpr_tile16x32<kHeadDim, kBlockM, kBlockK, WARP_M, WARP_NUM, Element, STAGES, M_MMAC_COUNT>(
        q_addr, q_lds, q_reg, warp_id, query_seqlen_stride, binfo.actual_seqlen_q - m_block * kBlockM);

    // 准备初始状态, scores_max/scores_max/acc_o
    vec2_Accum<ElementAccum> scores_max[M_WARP_COUNT];
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT];
    vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4];
    attention_initialize<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(scores_max, scores_sum, acc_o);

    /**********************************************************************************************************************************/
    // 主循环, 沿着 seqlenKV 维度, 每次 4 个 wave 共同计算一个 kBLOCKN
    const int n_block_min = 0;
    #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
        const int n_block_max = (Split and !MLA_FIX_NUM_SPLITS) ? ceil_div(Partition_Size, kBlockN): ceil_div(binfo.actual_seqlen_k, kBlockN);
    #else
        const int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN); // temp workaround, unroll partition size for zd may lead to wrong results
    #endif
    int n_block_loop      = n_block_min;
    for (; n_block_loop < n_block_max; ++n_block_loop) {

        int warp_offset_in_seqkv = n_block_loop * kBlockN + warp_id * WARP_N;
        int warp_seqkv_limit     = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        mla_prefetch_k_to_lds_tile16x32<kBlockK, WARP_N, Element, STAGES, WARP_NUM>(k_addr, k_lds, warp_id, kvcache_seqlen_stride, warp_seqkv_limit);

        vec4_Accum<ElementAccum> s_reg[M_WARP_COUNT * N_WARP_COUNT][4];

        mla_qk_gemm_prefetch_v_tile16x32<kHeadDim, kHeadDimVSplit, kBlockM, WARP_N, kBlockK, WARP_M, WARP_N, WARP_NUM, STAGES, M_MMAC_COUNT, Element, ElementAccum>(
            q_addr, k_addr, v_addr, q_lds, k_lds, v_lds, q_reg, s_reg, warp_id, kvcache_seqlen_stride, warp_seqkv_limit);

        if constexpr (Is_causal) {
            mla_apply_mask_causal_tile16x32<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT, 4/*MTP_REGROUP_COUNT*/, REUSE_KV_TIMES>(s_reg, warp_offset_in_seqkv + this_split_seqlen_start, original_actual_seqlen_k, m_block * kBlockM, binfo.actual_seqlen_q, params.mtp, params.layout);
        } else {
            mla_apply_mask_tile16x32<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, warp_seqkv_limit, warp_id * WARP_N);
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

        mla_pv_gemm_prefetch_k_tile16x32<K_LOOP_COUNT, kBlockM, kBlockK, kBlockN, M_WARP_COUNT, K_WARP_COUNT/*kBlockK*/, N_WARP_COUNT/*WARP_N*/, STAGES, WARP_NUM, M_MMAC_COUNT, Element, ElementAccum>(
            v_addr, k_addr, v_lds, k_lds, p_reg, acc_o, warp_id, kvcache_seqlen_stride, warp_seqkv_limit);

        *(int64_t*)&k_addr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);
        *(int64_t*)&v_addr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);

    }

    __syncthreads();

    // 多个 wave 同步 acc_o
    int thread_id = threadIdx.x;
    int lane_id   = thread_id & 63;
    if constexpr (WARP_NUM > 1) {
        mla_acco_reduce_tile16x32<REUSE_KV_TIMES, K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, M_MMAC_COUNT, WARP_NUM, 4/*Padding*/, ElementAccum>(acc_o, acc_o_lds, params.seqlen_q, warp_id, lane_id);
    }

    /**********************************************************************************************************************************/
    // Epilogue, 收尾工作
    // 收尾 1: 根据最后的归一化求和, 做 rescale
    mla_epilugue_rescale_acco<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(acc_o, scores_sum);

    // 收尾 2: splitkv, 或者开启 debug 的情况下, 写出 scores_max, scores_sum
    mla_tp8_epilogue_store_softmax_lse<Split, true/*Is_16x32*/, M_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
        scores_max, scores_sum, softmax_lse_ptr, params.scale_softmax, warp_id, thread_id, lane_id, headdim_split_id, binfo.actual_seqlen_q - m_block * kBlockM);

    // 收尾 3: 写出 output
    mla_epilogue_store_output_tile16x32<Params, kHeadDimV, kHeadDimVSplit, Split, SplitkvAccumType, ElementAccum, kBlockM, kBlockK, WARP_NUM, K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT>(
        acc_o, params, bidb, bidh, m_block, split_id, headdim_split_id, warp_id, lane_id);

}


template<typename Kernel_traits, bool Is_causal, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
inline __device__ void compute_attn_splitkv_mla(const Params &params) {

    // The block index for the head.
    const int bidh = Split ? blockIdx.z % params.h : blockIdx.y; // batch x num_head, num_head first

    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.z;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id     = __builtin_amdgcn_readfirstlane(warp_id_vec);

    // decide 16x32 tile and 32x32 tile by split-D
    if constexpr (HEADDIM_V_SPLIT == 2) {
        flash::compute_attn_1rowblock_splitkv_mla<Kernel_traits, Is_causal, Split, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size * 128, Params>(params, bidb, bidh, warp_id);
    } else {
        flash::compute_attn_1rowblock_splitkv_mla_tile16x32<Kernel_traits, Is_causal, Split, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size * 128, Params>(params, bidb, bidh, warp_id);
    }
}



////////////////////////////////////////////////////////////////////////////////////////////////////
#include "kvcache/gfx938/kvcache_softmax_gfx938.h"
#include "kvcache/kvcache_acco_reduce_tile16x32.h"
#include "kvcache/kvcache_epilogue.h"
#include "mla/gfx938/fp8_mla_acco_reduce_gfx938.h"
#include "mla/gfx938/mla_tp8_qk_gemm_utils_gfx938.h"
#include "mla/gfx938/mla_tp8_epilogue_gfx938.h"
#include "mla/gfx938/f16_mla_tp8_qk_gemm_utils_gfx938.h"
#include "mla/gfx938/f16_mla_tp8_qk_gemm_gfx938.h"
#include "mla/gfx938/f16_mla_tp8_pv_gemm_gfx938.h"
// For FlashMLA, codes almostly copy codes from paged_attention with a few differences.
// Kernel codes listed below can be customized alone if neccessary.
// sgpr: 75, vgpr: 240 | base sgpr: 80, vgpr 254
template<typename Kernel_traits, bool Is_causal, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_mla_gfx938(const Params &params, const int bidb, const int bidh, const int warp_id) {
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
    binfo.set_params<Params, /*Is_Q_varlen=*/false, /*Is_K_Cumulative=*/false>(params, bidb);

    // splitKV, 根据 split id 确定当前 split 在 seqlen_kv 上处理的长度
    int split_id;
    int original_actual_seqlen_k = binfo.actual_seqlen_k;
    int partition_size;
    constexpr bool MLA_FIX_NUM_SPLITS = Partition_Size > MLA_MAX_SPLITS;
    if constexpr (Split) {
        if constexpr (MLA_FIX_NUM_SPLITS) {
            split_id = blockIdx.y;
            partition_size = splitkv_get_partitionsize_of_fix_numsplits(binfo.actual_seqlen_k, params.num_splits);
            binfo.actual_seqlen_k = min(binfo.actual_seqlen_k - split_id * partition_size, partition_size);
        } else {
            split_id = blockIdx.y;
            partition_size = params.partition_size;
            int num_splits = ceil_div(binfo.actual_seqlen_k, partition_size);
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

    // 确定运行边界
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k <= 0) return;

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
    const int64_t row_offset_q = bidb * int64_t(params.q_batch_stride) + bidh * params.q_head_stride + m_block * kBlockM * query_seqlen_stride;

    // 准备读取数据的 buffer resource 寄存器
    auto q_addr = prepare_for_buffer_load<kHeadDim, Element, false>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
    auto k_addr = prepare_for_buffer_load<kHeadDim, Element, false>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto v_addr = prepare_for_buffer_load<kHeadDimV, Element, false>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v + headdim_split_id * kHeadDimVSplit);

    bool is_thread0 = threadIdx.x == 0;
    if (is_thread0) {
        // inline_utcl2_warmup_dword(k_addr);
    }

    // splitkv, debug 场景下需要写出一些值, 例如 scores_max/scores_sum
    int row_offset_lse;
    ElementAccum * softmax_lse_ptr;
    if constexpr (Split) {
        row_offset_lse = split_id * (params.b * params.h * params.seqlen_q) + (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + row_offset_lse;
    }

    // 目前最小的计算块是 32x32, 平摊到 64 个线程是 16 个 Half, 4 组 16x16, 每一组一次性从 lds 读取 2 个 Half, 读取 2 次
    constexpr int M_WARP_COUNT = WARP_M / 32;
    constexpr int K_WARP_COUNT = kBlockK / 32;
    constexpr int N_WARP_COUNT = WARP_N / 32;
    constexpr int K_LOOP_COUNT = kHeadDimVSplit / kBlockK;
    constexpr int Q_LOAD_BLOCKS = STAGES == 2 ? (kHeadDim / kBlockK): 1;
    union_vec4_f16x2<Element> q_reg[Q_LOAD_BLOCKS * M_WARP_COUNT * K_WARP_COUNT * 2];

    // 准备初始状态, scores_max/scores_max/acc_o
    vec2_Accum<ElementAccum> scores_max[M_WARP_COUNT];
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT];
    vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4];

    // 预取 Q 到寄存器或者 lds
    mla_prefetch_q_to_vgpr_gfx938_with_initialization<kHeadDim, kHeadDimVSplit, kBlockM, kBlockK, WARP_M, WARP_NUM, M_MMAC_COUNT, Element, ElementAccum>(
        q_addr, q_lds, q_reg, warp_id, query_seqlen_stride, binfo.actual_seqlen_q - m_block * kBlockM, scores_max, scores_sum, acc_o);

    /**********************************************************************************************************************************/
    // 主循环, 沿着 seqlenKV 维度, 每次 4 个 wave 共同计算一个 kBLOCKN
    const int n_block_min = 0;
    const int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN); // Split ? ceil_div(Partition_Size, kBlockN): ceil_div(binfo.actual_seqlen_k, kBlockN);
    int n_block_loop      = n_block_min;
    for (; n_block_loop < n_block_max; ++n_block_loop) {

        int warp_offset_in_seqkv = n_block_loop * kBlockN + warp_id * WARP_N;
        int warp_seqkv_limit     = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        f16_mla_tp8_prefetch_k_to_lds_gfx938<kBlockK, WARP_N, Element, STAGES, WARP_NUM>(k_addr, k_lds, warp_id, kcache_seqlen_stride, warp_seqkv_limit);

        vec4_Accum<ElementAccum> s_reg[M_WARP_COUNT * N_WARP_COUNT][4];

        f16_mla_tp8_qk_gemm_gfx938<kHeadDim, kHeadDimVSplit, kBlockM, WARP_N, kBlockK, WARP_M, WARP_N, WARP_NUM, STAGES, M_MMAC_COUNT, Element, ElementAccum>(
            q_addr, k_addr, v_addr, q_lds, k_lds, v_lds, q_reg, s_reg, warp_id, kcache_seqlen_stride, vcache_seqlen_stride, warp_seqkv_limit);

        if constexpr (Is_causal) {
            kvcache_apply_mask_causal_gfx938_mtp<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, warp_offset_in_seqkv + this_split_seqlen_start, original_actual_seqlen_k, m_block * kBlockM, binfo.actual_seqlen_q, params.mtp, params.layout);
        } else {
            kvcache_apply_mask_gfx938<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, warp_seqkv_limit, warp_id * WARP_N);
        }

        mla_softmax_rescale_o<Is_causal, ElementAccum, K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, N_WARP_COUNT, WARP_NUM, M_MMAC_COUNT>(
            s_reg, scores_max, scores_sum, acc_o, max_lds, warp_id, params.scale_softmax_log2);

        union_vec2_f16x2<Element> p_reg[M_WARP_COUNT * N_WARP_COUNT][4];
        mla_convert_pk_type<M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT, Element, ElementAccum>(p_reg, s_reg);

        f16_mla_tp8_pv_gemm_gfx938<K_LOOP_COUNT, kBlockM, kBlockK, kBlockN, M_WARP_COUNT, K_WARP_COUNT/*kBlockK*/, N_WARP_COUNT/*WARP_N*/, STAGES, WARP_NUM, M_MMAC_COUNT, Element, ElementAccum>(
            v_addr, k_addr, v_lds, k_lds, p_reg, acc_o, warp_id, vcache_seqlen_stride, warp_seqkv_limit);

        const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;

        *(int64_t*)&k_addr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);
        *(int64_t*)&v_addr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);

    }

    __syncthreads();

    // 多个 wave 同步 acc_o
    int thread_id = threadIdx.x;
    int lane_id   = thread_id & 63;
    if constexpr (WARP_NUM > 1) {
        // kvcache_acco_reduce_tile16x32<REUSE_KV_TIMES, K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, M_MMAC_COUNT, WARP_NUM, 4/*Padding*/, ElementAccum>(acc_o, acc_o_lds, params.seqlen_q, warp_id, lane_id);
        fp8_mla_acco_reduce_tile16x32<REUSE_KV_TIMES, K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, M_MMAC_COUNT, WARP_NUM, 4/*Padding*/, ElementAccum>(acc_o, acc_o_lds, params.seqlen_q, warp_id, lane_id);
    }

    /**********************************************************************************************************************************/
    // Epilogue, 收尾工作
    // 收尾 1: 根据最后的归一化求和, 做 rescale
    kvcache_epilugue_rescale_acco<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(acc_o, scores_sum);

    // 收尾 2: splitkv, 或者开启 debug 的情况下, 写出 scores_max, scores_sum
    mla_tp8_epilogue_store_softmax_lse<Split, true/*Is_16x32*/, M_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
        scores_max, scores_sum, softmax_lse_ptr, params.scale_softmax, warp_id, thread_id, lane_id, headdim_split_id, binfo.actual_seqlen_q - m_block * kBlockM);

    // 收尾 3: 写出 output
    mla_tp8_epilogue_store_output_gfx938<Params, kHeadDimV, kHeadDimVSplit, true/*alt*/, Split, SplitkvAccumType, ElementAccum, kBlockM, kBlockK, WARP_NUM, K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT>(
        acc_o, params, bidb, bidh, m_block, split_id, headdim_split_id, warp_id, lane_id);
}


template<typename Kernel_traits, bool Is_causal, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
inline __device__ void compute_attn_splitkv_mla_gfx938(const Params &params) {

#if defined(__gfx938__) || defined(__gfx946__)
    // The block index for the head.
    const int bidh = Split ? blockIdx.z % params.h : blockIdx.y; // batch x num_head, num_head first

    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.z;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id     = __builtin_amdgcn_readfirstlane(warp_id_vec);

    flash::compute_attn_1rowblock_splitkv_mla_gfx938<Kernel_traits, Is_causal, Split, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size * 128, Params>(params, bidb, bidh, warp_id);
#endif
}



////////////////////////////////////////////////////////////////////////////////////////////////////
#include "mla/mla_prefix_prefill.h"

template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, bool Is_prefix, bool Is_causal, typename Element, typename ElementAccum, typename Params>
inline __device__ void flash_fwd_mla_prefix_prefill_kernel_base(const Params params, const int bidb, const int bidh, const int m_block, const int warp_id) {
    // 既定参数
    constexpr int kHeadDimVSplit = kHeadDimV >> 1;
    constexpr int WARP_M    = 16;
    constexpr int WARP_N    = 32;
    constexpr int PREFETCH_V_BLOCKS = 2;
    constexpr int WARP_NUM  = 8;
    constexpr bool Is_FlashMLA = !Is_prefix; // Is_prefix means prefix

    // 准备 lds, 64KB
    __shared__ Element lds[32768];
    Element* q_lds = lds + 0;
    Element* k_lds = lds + 0;
    Element* v_lds = lds + 0;
    Element* k_rope_lds = k_lds + 16384;
    ElementAccum* s_reg_lds = reinterpret_cast<ElementAccum*>(lds + 0);

    // 获取当前线程 id 和 warp id
    int tx = threadIdx.x;
    int lane_id = tx & 63;
    int warp_id_row = warp_id & 3;
    int warp_id_col = warp_id >> 2; // [0, 1, 2, 3]: 0, [4, 5, 6, 7]: 1

    // 获取当前任务的长度
    int sum_s_q = Is_FlashMLA ? 0: params.cu_seqlens_q[bidb];
    int actual_seqlen_q = Is_FlashMLA ? params.seqlen_q: params.cu_seqlens_q[bidb + 1] - sum_s_q;
    int actual_seqlen_k = Is_FlashMLA ? params.cu_seqlens_k[bidb]: params.cu_seqlens_k_new[bidb + 1] - params.cu_seqlens_k_new[bidb];

    // 判断边界是否越界
    if (m_block * kBlockM >= actual_seqlen_q || actual_seqlen_k <= 0) return;

    // 计算 q, k, v 的位移或指针
    int64_t row_offset_q, row_offset_k, row_offset_v, row_offset_o, row_offset_qv;
    int64_t row_offset_lse;
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = 0;
    const int block_table_offset = 0;
    // 注意, q, k, v, o 都只算一半
    if constexpr (Is_FlashMLA) {
        row_offset_q   = bidb * int64_t(params.q_batch_stride) + m_block * kBlockM * int64_t(params.q_row_stride) + bidh * params.q_head_stride;
        row_offset_qv  = bidb * int64_t(params.qv_batch_stride) + m_block * kBlockM * int64_t(params.qv_row_stride) + bidh * params.qv_head_stride;
        row_offset_o   = bidb * int64_t(params.o_batch_stride) + m_block * kBlockM * int64_t(params.o_row_stride) + bidh * params.o_head_stride;
        row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    } else {
        row_offset_q   = (sum_s_q + m_block * kBlockM) * int64_t(params.q_row_stride) + bidh * params.q_head_stride;
        row_offset_qv  = (sum_s_q + m_block * kBlockM) * int64_t(params.qv_row_stride) + bidh * params.qv_head_stride;
        row_offset_o   = (sum_s_q + m_block * kBlockM) * int64_t(params.o_row_stride) + bidh * params.o_head_stride;
        row_offset_lse = bidh * params.total_q + sum_s_q + m_block * kBlockM;
    }
    row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Element* qv_ptr = reinterpret_cast<Element*>(params.qv_ptr) + row_offset_qv;
    Element* q_ptr = reinterpret_cast<Element*>(params.q_ptr) + row_offset_q;
    auto k_buffer = prepare_for_buffer_load<kHeadDim>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto v_buffer = prepare_for_buffer_load<kHeadDimV>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v);

    ElementAccum scores_max[WARP_M / 16];
    ElementAccum scores_sum[WARP_M / 16];
    vec4_Accum<ElementAccum> acc_o[WARP_M / 16][kHeadDimVSplit / 16];
    mla_prefix_prefill_initialize<WARP_M, kHeadDimVSplit>(scores_max, scores_sum, acc_o);

    union_vec4_f16x2<Element> qv_regs[WARP_M / 16][8];
    union_vec4_f16x2<Element> q_regs[WARP_M / 16];
    mla_prefix_prefill_fetch_q_to_vgpr<kBlockM, WARP_M, WARP_NUM, Element>(qv_regs, q_regs, qv_ptr, q_ptr, m_block, warp_id_row, warp_id_col, lane_id, params.qv_row_stride, params.q_row_stride, actual_seqlen_q);

    // Mainloop
    int n_block_min = 0;
    int n_block_max = ceil_div(actual_seqlen_k, kBlockN);
    if constexpr (Is_prefix) {
        n_block_max = std::min(n_block_max, ceil_div((m_block + 1) * kBlockM + actual_seqlen_k - actual_seqlen_q, kBlockN));
    }
    for (int n_block_loop = n_block_min; n_block_loop < n_block_max; ++n_block_loop) {
        // 注意这里的 q, qv 分别对应 kcache, vcache
        const int seqlen_kv_limit = actual_seqlen_k - n_block_loop * kBlockN;
        // prefetch K
        mla_prefix_prefill_prefetch_k_nope_to_lds<kBlockN, WARP_NUM, Element>(v_lds, v_buffer, warp_id, lane_id, params.v_row_stride, seqlen_kv_limit);
        // qk gemm
        vec4_Accum<ElementAccum> s_reg[WARP_M / 16][(kBlockN / 16)]; // 1 * 8 * 4 = 32 vgprs
        mla_prefix_prefill_compute_fwd_qk_nope<kBlockN, WARP_M, WARP_N, WARP_NUM, Element, ElementAccum>(s_reg, qv_regs, v_buffer, v_lds, k_buffer, k_rope_lds, warp_id, lane_id, params.v_row_stride, params.k_row_stride, seqlen_kv_limit);
        mla_prefix_prefill_compute_fwd_qk_rope<kBlockN, WARP_M, WARP_N, WARP_NUM, Element, ElementAccum>(s_reg, q_regs, k_buffer, k_rope_lds, warp_id, lane_id, params.k_row_stride, seqlen_kv_limit);
        // 把 s_reg 两两 wave 之间相加
        mla_prefix_prefill_combine_s_reg_of_2waves<kBlockN, WARP_M, WARP_NUM, ElementAccum>(s_reg, s_reg_lds/*64KB needed*/, warp_id, lane_id);
        // 预取 v
        mla_prefix_prefill_prefetch_v_to_lds<PREFETCH_V_BLOCKS, WARP_NUM, Element>(v_buffer, v_lds, params.v_row_stride, warp_id, lane_id, seqlen_kv_limit);
        // causal mask / mtp mask / mask
        if constexpr (Is_prefix) {
            mla_prefix_prefill_apply_causal_mask<kBlockN, WARP_M, WARP_NUM, ElementAccum>(s_reg, lane_id, n_block_loop * kBlockN, actual_seqlen_k, m_block * kBlockM + warp_id_row * 16, actual_seqlen_q);
        } else if constexpr (Is_causal) {
            mla_prefix_prefill_apply_mtp_mask<kBlockN, WARP_M, WARP_NUM, ElementAccum>(s_reg, lane_id, n_block_loop * kBlockN, actual_seqlen_k, m_block * kBlockM + warp_id_row * 16, actual_seqlen_q, params.ngroups, params.mtp);
        } else {
            mla_prefix_prefill_apply_mask<kBlockN, WARP_M, WARP_NUM, ElementAccum>(s_reg, lane_id, n_block_loop * kBlockN, actual_seqlen_k, m_block * kBlockM + warp_id_row * 16, actual_seqlen_q);
        }
        // softmax
        mla_prefix_prefill_compute_fwd_softmax<kBlockN, WARP_M, kHeadDimVSplit, ElementAccum>(s_reg, scores_max, scores_sum, params.scale_softmax_log2, acc_o);
        // cvt
        union_vec2_f16x2<Element> p_reg[WARP_M / 16][kBlockN / 16];
        mla_prefix_prefill_cvt_dtype<kBlockN, WARP_M, ElementAccum, Element>(s_reg, p_reg);
        // 计算页表偏移等
        const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;
        // pv gemm
        mla_prefix_prefill_compute_fwd_pv<false, PREFETCH_V_BLOCKS, kBlockN, WARP_M, WARP_NUM, kHeadDimVSplit, Element, ElementAccum>(acc_o, p_reg, v_buffer, v_lds, warp_id, lane_id, params.v_row_stride, seqlen_kv_limit, 0);
        *(int64_t*)&k_buffer += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);
        *(int64_t*)&v_buffer += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);
    }

    mla_prefix_prefill_rescale_acc_o<kBlockM, WARP_M, WARP_NUM, kHeadDimVSplit, ElementAccum>(
        acc_o, params.scores_max_ptr, params.scores_sum_ptr, params.softmax_lse_ptr, scores_max, scores_sum, params.scale_softmax, row_offset_lse, m_block, warp_id, warp_id_row, lane_id, actual_seqlen_q);

    mla_prefix_prefill_store_output<kBlockM, WARP_M, WARP_NUM, kHeadDimVSplit, Element, ElementAccum>(acc_o, params.o_ptr, row_offset_o, m_block, warp_id_row, warp_id_col, lane_id, params.o_row_stride, actual_seqlen_q);

}

template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, bool Is_prefix, bool Is_causal, typename Element, typename ElementAccum, typename Params>
__global__ void __launch_bounds__(512, 1) flash_fwd_mla_prefix_prefill_fix_kernel(const Params params) {
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
    int q_blocks = params.q_blocks;
    for(int loop = blockIdx.x; loop < params.total_blocks; loop += params.cu_count) {
        int m_block = loop % q_blocks;
        int bidh = (loop / q_blocks) % params.h;
        int bidb = (loop / q_blocks) / params.h;

        int warp_id_vec = threadIdx.x / 64;
        int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);

        flash_fwd_mla_prefix_prefill_kernel_base<kHeadDim, kHeadDimV, kBlockM, kBlockN, Is_prefix, Is_causal, Element, ElementAccum, Params>(params, bidb, bidh, m_block, warp_id);
        __syncthreads();
        flash_fwd_mla_prefix_prefill_kernel_base<kHeadDim, kHeadDimV, kBlockM, kBlockN, Is_prefix, Is_causal, Element, ElementAccum, Params>(params, bidb, bidh, q_blocks * 2 - 1 - m_block, warp_id);
        __syncthreads();
    }
#elif 0
    // 获取当前任务
    int bidb = blockIdx.z;
    int bidh = blockIdx.y;
    int m_block = blockIdx.x;
    int warp_id_vec = threadIdx.x / 64;
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
    flash_fwd_mla_prefix_prefill_kernel_base<kHeadDim, kHeadDimV, kBlockM, kBlockN, Is_prefix, Is_causal, Element, ElementAccum, Params>(params, bidb, bidh, m_block, warp_id);
    __syncthreads();
    flash_fwd_mla_prefix_prefill_kernel_base<kHeadDim, kHeadDimV, kBlockM, kBlockN, Is_prefix, Is_causal, Element, ElementAccum, Params>(params, bidb, bidh, gridDim.x * 2 - 1 - m_block, warp_id);
#endif // zd doesn't support
}



template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, bool Is_prefix, bool Is_causal, typename Element, typename ElementAccum, typename Params>
__global__ void __launch_bounds__(512, 1) flash_fwd_mla_fix_kernel(const Params params) {
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
    int q_blocks = params.q_blocks;
    for(int loop = blockIdx.x; loop < params.total_blocks; loop += params.cu_count) {
        int m_block = loop % q_blocks;
        int bidh = (loop / q_blocks) % params.h;
        int bidb = (loop / q_blocks) / params.h;

        int warp_id_vec = threadIdx.x / 64;
        int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);

        flash_fwd_mla_prefix_prefill_kernel_base<kHeadDim, kHeadDimV, kBlockM, kBlockN, Is_prefix, Is_causal, Element, ElementAccum, Params>(params, bidb, bidh, m_block, warp_id);
        __syncthreads();
    }
#elif 0
    // 5% performance higher than listed above
    int bidb = blockIdx.z;
    int bidh = blockIdx.y;
    int m_block = blockIdx.x;
    int warp_id_vec = threadIdx.x / 64;
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
    flash_fwd_mla_prefix_prefill_kernel_base<kHeadDim, kHeadDimV, kBlockM, kBlockN, Is_prefix, Is_causal, Element, ElementAccum, Params>(params, bidb, bidh, m_block, warp_id);
#endif // zd doesn't support
}


////////////////////////////////////////////////////////////////////////////////////////////////////
template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, bool Is_prefix, bool Is_causal, typename Element, typename ElementAccum, typename Params>
inline __device__ void flash_fwd_mla_fast_prefix_prefill_kernel_base(const Params params, const int bidb, const int bidh, const int m_block, const int warp_id) {
    // 既定参数
    constexpr int kHeadDimVSplit = kHeadDimV >> 1;
    constexpr int WARP_M    = kBlockM >> 1;
    constexpr int WARP_N    = 32;
    constexpr int PREFETCH_V_BLOCKS = 2;
    constexpr int WARP_NUM  = 4;
    constexpr bool Is_FlashMLA = !Is_prefix; // Is_prefix means prefix

    // 准备 lds, 64KB
    __shared__ Element lds[16384];
    Element* q_lds = lds + 0;
    Element* k_lds = lds + 0;
    Element* v_lds = lds + 0;
    Element* k_rope_lds = k_lds;
    ElementAccum* s_reg_lds = reinterpret_cast<ElementAccum*>(lds + 0);

    // 获取当前线程 id 和 warp id
    int tx = threadIdx.x;
    int lane_id = tx & 63;
    int warp_id_row = warp_id >> 1;
    int warp_id_col = warp_id & 1; // [0, 1]: 0, [2, 3]: 1

    // 获取当前任务的长度
    int sum_s_q = Is_FlashMLA ? 0: params.cu_seqlens_q[bidb];
    int actual_seqlen_q = Is_FlashMLA ? params.seqlen_q: params.cu_seqlens_q[bidb + 1] - sum_s_q;
    int actual_seqlen_k = Is_FlashMLA ? params.cu_seqlens_k[bidb]: params.cu_seqlens_k_new[bidb + 1] - params.cu_seqlens_k_new[bidb];

    // 判断边界是否越界
    if (m_block * kBlockM >= actual_seqlen_q || actual_seqlen_k <= 0) return;

    // 计算 q, k, v 的位移或指针
    int64_t row_offset_q, row_offset_k, row_offset_v, row_offset_o, row_offset_qv;
    int64_t row_offset_lse;
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = 0;
    const int block_table_offset = 0;
    // 注意, q, k, v, o 都只算一半
    if constexpr (Is_FlashMLA) {
        row_offset_q   = bidb * int64_t(params.q_batch_stride) + m_block * kBlockM * int64_t(params.q_row_stride) + bidh * params.q_head_stride;
        row_offset_qv  = bidb * int64_t(params.qv_batch_stride) + m_block * kBlockM * int64_t(params.qv_row_stride) + bidh * params.qv_head_stride;
        row_offset_o   = bidb * int64_t(params.o_batch_stride) + m_block * kBlockM * int64_t(params.o_row_stride) + bidh * params.o_head_stride;
        row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    } else {
        row_offset_q   = (sum_s_q + m_block * kBlockM) * int64_t(params.q_row_stride) + bidh * params.q_head_stride;
        row_offset_qv  = (sum_s_q + m_block * kBlockM) * int64_t(params.qv_row_stride) + bidh * params.qv_head_stride;
        row_offset_o   = (sum_s_q + m_block * kBlockM) * int64_t(params.o_row_stride) + bidh * params.o_head_stride;
        row_offset_lse = bidh * params.total_q + sum_s_q + m_block * kBlockM;
    }
    row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    Element* qv_ptr = reinterpret_cast<Element*>(params.qv_ptr) + row_offset_qv;
    Element* q_ptr = reinterpret_cast<Element*>(params.q_ptr) + row_offset_q;
    auto k_buffer = prepare_for_buffer_load<kHeadDim>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto v_buffer = prepare_for_buffer_load<kHeadDimV>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v);

    ElementAccum scores_max[WARP_M / 16];
    ElementAccum scores_sum[WARP_M / 16];
    vec4_Accum<ElementAccum> acc_o[WARP_M / 16][kHeadDimVSplit / 16];
    mla_prefix_prefill_initialize<WARP_M, kHeadDimVSplit>(scores_max, scores_sum, acc_o);

    union_vec4_f16x2<Element> qv_regs[WARP_M / 16][8];
    union_vec4_f16x2<Element> q_regs[WARP_M / 16];
    mla_prefix_prefill_fetch_q_to_vgpr<kBlockM, WARP_M, WARP_NUM, Element>(qv_regs, q_regs, qv_ptr, q_ptr, m_block, warp_id_row, warp_id_col, lane_id, params.qv_row_stride, params.q_row_stride, actual_seqlen_q);

    constexpr bool PREFETCH_K = true;
    if constexpr (PREFETCH_K) {
        mla_prefix_prefill_prefetch_k_nope_to_lds<kBlockN, WARP_NUM, Element>(v_lds, v_buffer, warp_id, lane_id, params.v_row_stride, actual_seqlen_k);
    }

    // Mainloop
    int n_block_min = 0;
    int n_block_max = ceil_div(actual_seqlen_k, kBlockN);
    if constexpr (Is_prefix) {
        n_block_max = std::min(n_block_max, ceil_div((m_block + 1) * kBlockM + actual_seqlen_k - actual_seqlen_q, kBlockN));
    }
    for (int n_block_loop = n_block_min; n_block_loop < n_block_max; ++n_block_loop) {
        // 注意这里的 q, qv 分别对应 kcache, vcache
        const int seqlen_kv_limit = actual_seqlen_k - n_block_loop * kBlockN;
        // prefetch K
        if constexpr (not PREFETCH_K) {
            mla_prefix_prefill_prefetch_k_nope_to_lds<kBlockN, WARP_NUM, Element>(v_lds, v_buffer, warp_id, lane_id, params.v_row_stride, seqlen_kv_limit);
        }
        // qk gemm
        vec4_Accum<ElementAccum> s_reg[WARP_M / 16][(kBlockN / 16)]; // 1 * 8 * 4 = 32 vgprs
        mla_prefix_prefill_compute_fwd_qk_nope<kBlockN, WARP_M, WARP_N, WARP_NUM, Element, ElementAccum>(s_reg, qv_regs, v_buffer, v_lds, k_buffer, k_rope_lds, warp_id, lane_id, params.v_row_stride, params.k_row_stride, seqlen_kv_limit);
        mla_prefix_prefill_compute_fwd_qk_rope<kBlockN, WARP_M, WARP_N, WARP_NUM, Element, ElementAccum>(s_reg, q_regs, k_buffer, k_rope_lds, warp_id, lane_id, params.k_row_stride, seqlen_kv_limit);
        // 把 s_reg 两两 wave 之间相加
        mla_prefix_prefill_combine_s_reg_of_2waves<kBlockN, WARP_M, WARP_NUM, ElementAccum>(s_reg, s_reg_lds/*64KB needed*/, warp_id, lane_id);
        // 预取 v
        mla_prefix_prefill_prefetch_v_to_lds<4, WARP_NUM, Element>(v_buffer, v_lds, params.v_row_stride, warp_id, lane_id, seqlen_kv_limit);
        // causal mask / mtp mask / mask
        if constexpr (Is_prefix) {
            mla_prefix_prefill_apply_causal_mask<kBlockN, WARP_M, WARP_NUM, ElementAccum>(s_reg, lane_id, n_block_loop * kBlockN, actual_seqlen_k, m_block * kBlockM + warp_id_row * 16, actual_seqlen_q);
        } else if constexpr (Is_causal) {
            mla_prefix_prefill_apply_mtp_mask<kBlockN, WARP_M, WARP_NUM, ElementAccum>(s_reg, lane_id, n_block_loop * kBlockN, actual_seqlen_k, m_block * kBlockM + warp_id_row * 16, actual_seqlen_q, params.ngroups, params.mtp);
        } else {
            mla_prefix_prefill_apply_mask<kBlockN, WARP_M, WARP_NUM, ElementAccum>(s_reg, lane_id, n_block_loop * kBlockN, actual_seqlen_k, m_block * kBlockM + warp_id_row * 16, actual_seqlen_q);
        }
        // softmax
        mla_prefix_prefill_compute_fwd_softmax<kBlockN, WARP_M, kHeadDimVSplit, ElementAccum>(s_reg, scores_max, scores_sum, params.scale_softmax_log2, acc_o);
        // cvt
        union_vec2_f16x2<Element> p_reg[WARP_M / 16][kBlockN / 16];
        mla_prefix_prefill_cvt_dtype<kBlockN, WARP_M, ElementAccum, Element>(s_reg, p_reg);
        // 计算页表偏移等
        const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;
        const int64_t v_buffer_offset     = (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);
        // pv gemm
        mla_prefix_prefill_compute_fwd_pv<PREFETCH_K, PREFETCH_V_BLOCKS, kBlockN, WARP_M, WARP_NUM, kHeadDimVSplit, Element, ElementAccum>(acc_o, p_reg, v_buffer, v_lds, warp_id, lane_id, params.v_row_stride, seqlen_kv_limit, v_buffer_offset);
        *(int64_t*)&k_buffer += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);
        *(int64_t*)&v_buffer += v_buffer_offset;
    }

    mla_prefix_prefill_rescale_acc_o<kBlockM, WARP_M, WARP_NUM, kHeadDimVSplit, ElementAccum>(
        acc_o, params.scores_max_ptr, params.scores_sum_ptr, params.softmax_lse_ptr, scores_max, scores_sum, params.scale_softmax, row_offset_lse, m_block, warp_id, warp_id_row, lane_id, actual_seqlen_q);

    mla_prefix_prefill_store_output<kBlockM, WARP_M, WARP_NUM, kHeadDimVSplit, Element, ElementAccum>(acc_o, params.o_ptr, row_offset_o, m_block, warp_id_row, warp_id_col, lane_id, params.o_row_stride, actual_seqlen_q);

}




template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, bool Is_prefix, bool Is_causal, typename Element, typename ElementAccum, typename Params>
__global__ void __launch_bounds__(512, 1) flash_fwd_mla_prefix_prefill_kernel(const Params params) {
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
    const int q_blocks = params.q_blocks;
    for (int m_block = 0; m_block < q_blocks; ++m_block) {
        // 获取当前任务
        int bidb = blockIdx.z;
        int bidh = blockIdx.y;
        // int m_block = blockIdx.x;
        int warp_id_vec = threadIdx.x / 64;
        int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
        flash_fwd_mla_fast_prefix_prefill_kernel_base<kHeadDim, kHeadDimV, kBlockM, kBlockN, Is_prefix, Is_causal, Element, ElementAccum, Params>(params, bidb, bidh, (m_block << 1) + blockIdx.x, warp_id);
        __syncthreads();
        // flash_fwd_mla_fast_prefix_prefill_kernel_base<kHeadDim, kHeadDimV, kBlockM, kBlockN, Is_prefix, Is_causal, Element, ElementAccum, Params>(params, bidb, bidh, q_blocks * 2 - 1 - m_block, warp_id);
        // __syncthreads();
    }
#endif // zd doesn't support
}


template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, bool Is_prefix, bool Is_causal, typename Element, typename ElementAccum, typename Params>
__global__ void __launch_bounds__(512, 1) flash_fwd_mla_kernel(const Params params) {
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
    // 获取当前任务
    const int q_blocks = params.q_blocks;
    for (int m_block = 0; m_block < q_blocks; ++m_block) {
        int bidb = blockIdx.z;
        int bidh = blockIdx.y;
        int warp_id_vec = threadIdx.x / 64;
        int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
        flash_fwd_mla_fast_prefix_prefill_kernel_base<kHeadDim, kHeadDimV, kBlockM, kBlockN, Is_prefix, Is_causal, Element, ElementAccum, Params>(params, bidb, bidh, (m_block << 1) + blockIdx.x, warp_id);
    }
#endif // zd doesn't support
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#include "mla/gfx938/mla_qk_gemm_prefetch_v_mls_ds.h"
#include "mla/gfx938/mla_qk_gemm_utils_mls_ds.h"
#include "mla/gfx938/mla_pv_gemm_prefetch_k_mls_ds.h"
#include "mla/gfx938/mla_pv_gemm_utils_mls_ds.h"
#include "mla/gfx938/mla_softmax_gfx938.h"
#include "mla/gfx938/mla_epilogue_tile16x32_lit.h"

template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_prefix, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Is_MTP, int Layout, typename Params>
__global__ void __launch_bounds__(512, 1) flash_fwd_mla_prefill_kernel_gfx938(const Params params) {
#if defined(__gfx938__)
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
    constexpr int WARP_NUM  = kBlockM / WARP_M;
    
    {
        const int bidh = blockIdx.y;
        const int bidb = blockIdx.z;
        int warp_id_vec = threadIdx.x / 64;
        const int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
        const int m_block = blockIdx.x;

        constexpr bool Is_FlashMLA = !Is_prefix; // 仅支持prefix为true

        // 获取当前任务的长度
        int sum_s_q = params.cu_seqlens_q[bidb];
        int actual_seqlen_q = params.cu_seqlens_q[bidb + 1] - sum_s_q;
        int actual_seqlen_k = params.cu_seqlens_k_new[bidb + 1] - params.cu_seqlens_k_new[bidb];

        // 处理边界
        if (m_block * kBlockM >= actual_seqlen_q || actual_seqlen_k <= 0) return;
        const int warp_offset_in_seq_q = m_block * kBlockM + warp_id * WARP_M;
        const int warp_seqq_limit      = Is_even_MN ? 0: actual_seqlen_q - m_block * kBlockM;

        // 分配 lds Q/P same place, K/V same place;
        extern __shared__ Element smem[];
        Element* q_lds = (Element*)&(smem);                     // 16KB
        Element* k_lds = q_lds + ((16*1024) / sizeof(Element)); // 16KB
        Element* v_lds = q_lds;                                 // 32KB

        // 计算当前任务沿着 seqlenKV 方向的 block 起始位置和终止位置
        const int n_block_min = 0;
        int n_block_max = ceil_div(actual_seqlen_k, kBlockN);
        if constexpr (Is_causal) {
            n_block_max = std::min(n_block_max, flash::ceil_div((m_block + 1) * kBlockM + actual_seqlen_k - actual_seqlen_q + 0/* params.window_size_right */, kBlockN));
        }

        // 计算数据跨度
        int seqlen_q_stride = params.q_row_stride; 
        int seqlen_k_stride = params.k_row_stride; 
        int seqlen_v_stride = params.v_row_stride; 
        int seqlen_o_stride = params.o_row_stride;
        int seqlen_qv_stride = params.qv_row_stride;

        int row_offset_q, row_offset_k, row_offset_v, row_offset_o, row_offset_lse, row_offset_qv;
        int headdim_split_id = 0;

        const int page_block_size = params.page_block_size;
        int *block_table = params.block_table + bidb * params.block_table_batch_stride;
        const int block_table_idx = 0;
        const int block_table_offset = 0;

        if constexpr (Is_FlashMLA) {  
        row_offset_q   = bidb * int64_t(params.q_batch_stride) + m_block * kBlockM * int64_t(params.q_row_stride) + bidh * params.q_head_stride;
        row_offset_qv  = bidb * int64_t(params.qv_batch_stride) + m_block * kBlockM * int64_t(params.qv_row_stride) + bidh * params.qv_head_stride;
        row_offset_o   = bidb * int64_t(params.o_batch_stride) + m_block * kBlockM * int64_t(params.o_row_stride) + bidh * params.o_head_stride;
        row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        } else { // 目前仅支持此种情况
            row_offset_q   = (sum_s_q + m_block * kBlockM) * int64_t(params.q_row_stride) + bidh * params.q_head_stride;
            row_offset_qv  = (sum_s_q + m_block * kBlockM) * int64_t(params.qv_row_stride) + bidh * params.qv_head_stride;
            row_offset_o   = (sum_s_q + m_block * kBlockM) * int64_t(params.o_row_stride) + bidh * params.o_head_stride;
            row_offset_lse = bidh * params.total_q + sum_s_q + m_block * kBlockM;
        }
        row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;

        // 根据起始数据偏移量准备 Q/K/V 的 buffer resource 寄存器
        // q_ptr : 64 | qv_ptr : 512 | k_ptr : 64(k_rope) | v_ptr : 512(k_nope)  
        auto q_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
        auto qv_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.qv_ptr) + row_offset_qv);
        auto k_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
        auto v_ptr = prepare_for_matrix_load<kHeadDimV>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v);

        // apply causal mask 的步骤和 no causal mask 的步骤分开算
        constexpr int n_masking_steps = (!Is_causal) ? 1: flash::ceil_div(kBlockM, kBlockN);

        // 是否做 prefetch K, PV 结束后, prefetch K 有风险
        constexpr bool PREFETCH_K = (Is_even_MN) and( ( kHeadDim == 576 and kHeadDimV == 512 )); // 简单场景下开启
        constexpr bool ALLOW_PREFETCH = (STAGES > 1); // 客观上决定是否开启 prefetch
        if constexpr (PREFETCH_K and ALLOW_PREFETCH) {
            if (n_block_min < n_block_max - n_masking_steps) {
                prefetch_q_to_lds_mls_ds_576_512<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(q_ptr, q_lds, warp_id, seqlen_q_stride, Is_even_MN ? 0: actual_seqlen_q - m_block * kBlockM);
                prefetch_k_to_lds_mls_ds_576_512<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, Is_even_MN ? 0: actual_seqlen_k - n_block_min * kBlockN);
            }
        }

        vec_Accum<ElementAccum> scores_max[WARP_M / 16]; // 只处理16行,所以只有1个reg
        vec_Accum<ElementAccum> scores_sum[WARP_M / 16];
        vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 16) * (kBlockK / 32)][2];

        // 内联失败，手动展开
        {
            constexpr int K_LOOP_COUNT = kHeadDimV / kBlockK;
            constexpr int M_WARP_COUNT = WARP_M / 16;
            constexpr int K_WARP_COUNT = kBlockK / 32; 
            constexpr int M_MMAC_COUNT = 1;

            #pragma unroll
            for (int i = 0; i < M_WARP_COUNT; ++i) {
                scores_max[i].f32[0] = -INFINITY;
                scores_sum[i].f32[0] = 0;
            }
        
            uint64_t pk_zero = 0;
            #pragma unroll
            for (int i = 0; i < K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT; ++i) {
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                        acc_o[i][min_tile_n].u64[0] = __builtin_hcu_mov_b64(pk_zero);
                        acc_o[i][min_tile_n].u64[1] = __builtin_hcu_mov_b64(pk_zero);
                        #else
                        acc_o[i][min_tile_n].f32[0] = 0;
                        acc_o[i][min_tile_n].f32[1] = 0;
                        acc_o[i][min_tile_n].f32[2] = 0;
                        acc_o[i][min_tile_n].f32[3] = 0;
                        #endif
                    }
                }
            }
        }

        auto QK_GEMM_FUNC = &qk_gemm_prefetch_v_mls_ds_576_512<kHeadDim, kHeadDimV, kBlockM, kBlockN, kBlockK, WARP_M, WARP_N, STAGES, Element, ElementAccum, Is_even_MN, Is_FlashMLA>;
        auto PV_GEMM_FUNC = &pv_gemm_prefetch_k_mls_ds_576_512<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;
        auto PV_GEMM_FUNC_IN_MASK = &pv_gemm_prefetch_k_mls_ds_576_512<false, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;

        // Mainloop, 主循环, 不做 causal mask 的部分
        for (int n_block_loop = n_block_min; n_block_loop < n_block_max - n_masking_steps; ++n_block_loop) {

            // 计算当前 loop 下 seqlen_kv 的数据起始位置和边界
            int warp_offset_in_seqkv = n_block_loop * kBlockN;
            int warp_seqkv_limit     = Is_even_MN ? 0: actual_seqlen_k - warp_offset_in_seqkv;
        

            // 预取 K 的数据到 lds // 预取 k_ptr(k_rope), q_ptr 到 k_lds, q_lds
            if constexpr (not PREFETCH_K) {
                prefetch_q_to_lds_mls_ds_576_512<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(q_ptr, q_lds, warp_id, seqlen_q_stride, warp_seqq_limit);
                prefetch_k_to_lds_mls_ds_576_512<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, warp_seqkv_limit);
            }

            // 准备 QK gemm 输出的寄存器
            vec4_Accum<ElementAccum> s_reg[(WARP_M / 16) * (kBlockN / 32)][2];

            // QK gemm
            QK_GEMM_FUNC(qv_ptr /* qv512 */, q_ptr /* q64 */, v_ptr /* k512 */, v_ptr /* kv512 */, q_lds, k_lds, v_lds, s_reg, warp_id, seqlen_qv_stride, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, warp_seqq_limit, warp_seqkv_limit);

            if constexpr (Is_MTP) {
                prefill_mla_apply_mtp_mask_causal_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, warp_offset_in_seqkv, actual_seqlen_k, warp_offset_in_seq_q, actual_seqlen_q);
            }

            // 对 QK 输出做 softmax, 以及重放缩 acc_o/scores_sum
            prefill_mla_softmax_rescale_o<false, Is_causal, vec4_Accum<ElementAccum>, vec_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

            // softmax(QK) f32 -> f16
            union_vec2_f16x2<Element> p_reg[(WARP_M / 16) * (kBlockN / 32)][2];
            prefill_mla_convert_pk_type<WARP_M, kBlockN, Element, ElementAccum, 1/* M_MMAC_COUNT */>(p_reg, s_reg);

            // 计算页表偏移等
            const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
            const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
            const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
            const int offset_diff             = (block_table_idx_cur == block_table_idx_next) ? (64) : (-64);
            *(int64_t*)&k_ptr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);

            // PV gemm
            PV_GEMM_FUNC(q_ptr, k_ptr, v_ptr, q_lds, k_lds, v_lds, p_reg, acc_o, warp_id, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, warp_seqq_limit, warp_seqkv_limit);

            // 偏移 V 指针
            *(int64_t*)&v_ptr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);
        }

        // prefetch K 的话, 最后一次多取了一段 K, 为了不影响后续的操作, 需要同步等待
        if constexpr (PREFETCH_K) {
            buffer_load_lds_dwordx1_wait<0>();
        }

        // Rest loop, 做 causal mask 的部分
        int n_block_loop = max(n_block_max - n_masking_steps, n_block_min);
        #pragma unroll
        for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, ++n_block_loop) {

            // 计算当前 loop 下 seqlen_kv 的数据起始位置和边界
            int warp_offset_in_seqkv = n_block_loop * kBlockN;
            int warp_seqkv_limit     = Is_even_MN ? 0: actual_seqlen_k - warp_offset_in_seqkv;

            // 预取 Q K 的数据到 lds
            prefetch_q_to_lds_mls_ds_576_512<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(q_ptr, q_lds,/* q_reg,*/ warp_id, seqlen_q_stride, warp_seqq_limit);
            prefetch_k_to_lds_mls_ds_576_512<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, warp_seqkv_limit);

            // 准备 QK gemm 输出的寄存器
            vec4_Accum<ElementAccum> s_reg[(WARP_M / 16) * (kBlockN / 32)][2];

            // QK gemm
            QK_GEMM_FUNC(qv_ptr /* qv512 */, q_ptr /* q64 */, v_ptr /* k512 */, v_ptr /* kv512 */, q_lds, k_lds, v_lds, s_reg, warp_id, seqlen_qv_stride, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, warp_seqq_limit, warp_seqkv_limit);

            // 偏移 K 指针, 提前偏移准备预取 K
            const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
            const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
            const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
            const int offset_diff             = (block_table_idx_cur == block_table_idx_next) ? (64) : (-64);
            *(int64_t*)&k_ptr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);

            // Attention: mask, causal mask, local mask
            if constexpr (!Is_causal) {
                if constexpr (!Is_even_MN) { prefill_mla_apply_mask_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, warp_seqkv_limit); }
            } else {
                if constexpr (Is_causal) {
                    if constexpr (Is_MTP) {
                        prefill_mla_apply_mtp_mask_causal_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, warp_offset_in_seqkv, actual_seqlen_k, warp_offset_in_seq_q, actual_seqlen_q);
                    } else {
                        prefill_mla_apply_mask_causal_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, warp_offset_in_seqkv, actual_seqlen_k, warp_offset_in_seq_q, actual_seqlen_q);
                    }
                } 
            }

            // 对 QK 输出做 softmax, 以及重放缩 acc_o/scores_sum
            prefill_mla_softmax_rescale_o<false, Is_causal, vec4_Accum<ElementAccum>, vec_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);


            // softmax(QK) f32 -> f16
            union_vec2_f16x2<Element> p_reg[(WARP_M / 16) * (kBlockN / 32)][2];
            prefill_mla_convert_pk_type<WARP_M, kBlockN, Element, ElementAccum, 1/* M_MMAC_COUNT */>(p_reg, s_reg);

            // PV gemm
            PV_GEMM_FUNC_IN_MASK(q_ptr, k_ptr, v_ptr, q_lds, k_lds, v_lds, p_reg, acc_o, warp_id, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, warp_seqq_limit, warp_seqkv_limit);

            // 偏移 V 指针
            *(int64_t*)&v_ptr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);

        }
        /**********************************************************************************************************************************/
        // Epilogue: rescale acco
        vec_Accum<ElementAccum> lse[WARP_M / 16];
        prefill_mla_epilugue_rescale_acco<WARP_M, kBlockK, kHeadDimV, Is_dropout && Is_training, ElementAccum, vec_Accum<ElementAccum>, 1/* M_MMAC_COUNT */>(acc_o, lse, scores_max, scores_sum, params.scale_softmax, 0/* params.rp_dropout */);
        /**************************************************************************************************************************************/
        constexpr bool Is_Interleave = true;
        int lane_id = threadIdx.x & 63;
        if (params.softmax_lse_ptr != nullptr) {
            prefill_mla_epilogue_store_lse<WARP_M, Is_even_MN, false/*SplitD*/, Is_Interleave, ElementAccum, vec_Accum<ElementAccum>, 1/* M_MMAC_COUNT */>(lse, params.softmax_lse_ptr, row_offset_lse, warp_id, lane_id, 0, actual_seqlen_q - m_block * kBlockM);
        }
        /**************************************************************************************************************************************/
        Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;
        prefill_mla_epilogue_store_output<kHeadDimV, kBlockM, kBlockK, WARP_M, Is_even_MN, Is_Interleave, false/*TcpSwizzle*/, Element, ElementAccum, 1/* M_MMAC_COUNT */>(o_ptr, acc_o, m_block, warp_id, lane_id, seqlen_o_stride, actual_seqlen_q);
    }
    {
    if constexpr (Is_causal){
        flash::wait_all_warp_arrived();
        const int bidh = blockIdx.y;
        const int bidb = blockIdx.z;
        int warp_id_vec = threadIdx.x / 64;
        const int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
        const int m_block = gridDim.x * 2 - 1 - blockIdx.x;

        constexpr bool Is_FlashMLA = !Is_prefix;

        // 获取当前任务的长度
        int sum_s_q = params.cu_seqlens_q[bidb];
        int actual_seqlen_q = params.cu_seqlens_q[bidb + 1] - sum_s_q;
        int actual_seqlen_k = params.cu_seqlens_k_new[bidb + 1] - params.cu_seqlens_k_new[bidb];
        // ************************************************

        // 处理边界
        if (m_block * kBlockM >= actual_seqlen_q || actual_seqlen_k <= 0) return;
        const int warp_offset_in_seq_q = m_block * kBlockM + warp_id * WARP_M;
        const int warp_seqq_limit      = Is_even_MN ? 0: actual_seqlen_q - m_block * kBlockM;

        // 分配 lds Q/P same place, K/V same place;
        extern __shared__ Element smem[];
        Element* q_lds = (Element*)&(smem);                     // 16KB
        Element* k_lds = q_lds + ((16*1024) / sizeof(Element)); // 16KB
        Element* v_lds = q_lds;                                 // 32KB

        // 计算当前任务沿着 seqlenKV 方向的 block 起始位置和终止位置
        const int n_block_min = 0;
        int n_block_max = ceil_div(actual_seqlen_k, kBlockN);
        if constexpr (Is_causal) {
            n_block_max = std::min(n_block_max, flash::ceil_div((m_block + 1) * kBlockM + actual_seqlen_k - actual_seqlen_q + 0/* params.window_size_right */, kBlockN));
        }

        // 计算数据跨度
        int seqlen_q_stride = params.q_row_stride; 
        int seqlen_k_stride = params.k_row_stride; 
        int seqlen_v_stride = params.v_row_stride; 
        int seqlen_o_stride = params.o_row_stride;
        int seqlen_qv_stride = params.qv_row_stride;

        int row_offset_q, row_offset_k, row_offset_v, row_offset_o, row_offset_lse, row_offset_qv;
        int headdim_split_id = 0;

        const int page_block_size = params.page_block_size;
        int *block_table = params.block_table + bidb * params.block_table_batch_stride;
        const int block_table_idx = 0;
        const int block_table_offset = 0;
        // ************************************************

        if constexpr (Is_FlashMLA) {  
        row_offset_q   = bidb * int64_t(params.q_batch_stride) + m_block * kBlockM * int64_t(params.q_row_stride) + bidh * params.q_head_stride;
        row_offset_qv  = bidb * int64_t(params.qv_batch_stride) + m_block * kBlockM * int64_t(params.qv_row_stride) + bidh * params.qv_head_stride;
        row_offset_o   = bidb * int64_t(params.o_batch_stride) + m_block * kBlockM * int64_t(params.o_row_stride) + bidh * params.o_head_stride;
        row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        } else { // 目前仅支持此种情况
            row_offset_q   = (sum_s_q + m_block * kBlockM) * int64_t(params.q_row_stride) + bidh * params.q_head_stride;
            row_offset_qv  = (sum_s_q + m_block * kBlockM) * int64_t(params.qv_row_stride) + bidh * params.qv_head_stride;
            row_offset_o   = (sum_s_q + m_block * kBlockM) * int64_t(params.o_row_stride) + bidh * params.o_head_stride;
            row_offset_lse = bidh * params.total_q + sum_s_q + m_block * kBlockM;
        }
        row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
        // ********************************************

        #if 0
        if (int(threadIdx.x) == 0) {
            printf("bidb: %d | bidh: %d | actual_seqlen_q: %d | actual_seqlen_k: %d | n_block_max: %d | row_offset_q: %d | row_offset_k: %d | row_offset_v: %d | row_offset_o: %d | seqlen_q_stride: %d | seqlen_k_stride: %d | seqlen_v_stride: %d\n",
                bidb, bidh, actual_seqlen_q, actual_seqlen_k, n_block_max, row_offset_q, row_offset_k, row_offset_v, row_offset_o, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride);
        }
        #endif

        // 根据起始数据偏移量准备 Q/K/V 的 buffer resource 寄存器
        // q_ptr : 64 | qv_ptr : 512 | k_ptr : 64(k_rope) | v_ptr : 512(k_nope)  
        auto q_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
        auto qv_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.qv_ptr) + row_offset_qv);
        auto k_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
        auto v_ptr = prepare_for_matrix_load<kHeadDimV>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v);

        // apply causal mask 的步骤和 no causal mask 的步骤分开算
        constexpr int n_masking_steps = (!Is_causal) ? 1: flash::ceil_div(kBlockM, kBlockN);

        // 是否做 prefetch K, PV 结束后, prefetch K 有风险
        constexpr bool PREFETCH_K = (Is_even_MN) and( ( kHeadDim == 576 and kHeadDimV == 512 )); // 简单场景下开启
        constexpr bool ALLOW_PREFETCH = (STAGES > 1); // 客观上决定是否开启 prefetch
        if constexpr (PREFETCH_K and ALLOW_PREFETCH) {
            if (n_block_min < n_block_max - n_masking_steps) {
                prefetch_q_to_lds_mls_ds_576_512<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(q_ptr, q_lds, warp_id, seqlen_q_stride, Is_even_MN ? 0: actual_seqlen_q - m_block * kBlockM);
                prefetch_k_to_lds_mls_ds_576_512<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, Is_even_MN ? 0: actual_seqlen_k - n_block_min * kBlockN);
            }
        }

        /***************************************************************************************************************************/
        vec_Accum<ElementAccum> scores_max[WARP_M / 16];
        vec_Accum<ElementAccum> scores_sum[WARP_M / 16];
        vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 16) * (kBlockK / 32)][2];
        // 内联失败，手动展开
        {
            constexpr int K_LOOP_COUNT = kHeadDimV / kBlockK;
            constexpr int M_WARP_COUNT = WARP_M / 16;
            constexpr int K_WARP_COUNT = kBlockK / 32; 
            constexpr int M_MMAC_COUNT = 1;

            #pragma unroll
            for (int i = 0; i < M_WARP_COUNT; ++i) {
                scores_max[i].f32[0] = -INFINITY;
                scores_sum[i].f32[0] = 0;
            }
        
            uint64_t pk_zero = 0;
            #pragma unroll
            for (int i = 0; i < K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT; ++i) {
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                        acc_o[i][min_tile_n].u64[0] = __builtin_hcu_mov_b64(pk_zero);
                        acc_o[i][min_tile_n].u64[1] = __builtin_hcu_mov_b64(pk_zero);
                        #else
                        acc_o[i][min_tile_n].f32[0] = 0;
                        acc_o[i][min_tile_n].f32[1] = 0;
                        acc_o[i][min_tile_n].f32[2] = 0;
                        acc_o[i][min_tile_n].f32[3] = 0;
                        #endif
                    }
                }
            }
        }
        /***************************************************************************************************************************/

        auto QK_GEMM_FUNC = &qk_gemm_prefetch_v_mls_ds_576_512<kHeadDim, kHeadDimV, kBlockM, kBlockN, kBlockK, WARP_M, WARP_N, STAGES, Element, ElementAccum, Is_even_MN, Is_FlashMLA>;
        auto PV_GEMM_FUNC = &pv_gemm_prefetch_k_mls_ds_576_512<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;
        auto PV_GEMM_FUNC_IN_MASK = &pv_gemm_prefetch_k_mls_ds_576_512<false, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;

        // Mainloop, 主循环, 不做 causal mask 的部分
        for (int n_block_loop = n_block_min; n_block_loop < n_block_max - n_masking_steps; ++n_block_loop) {

            // 计算当前 loop 下 seqlen_kv 的数据起始位置和边界
            int warp_offset_in_seqkv = n_block_loop * kBlockN;
            int warp_seqkv_limit     = Is_even_MN ? 0: actual_seqlen_k - warp_offset_in_seqkv;
        

            // 预取 K 的数据到 lds // 预取 k_ptr(k_rope), q_ptr 到 k_lds, q_lds
            if constexpr (not PREFETCH_K) {
                prefetch_q_to_lds_mls_ds_576_512<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(q_ptr, q_lds, warp_id, seqlen_q_stride, warp_seqq_limit);
                prefetch_k_to_lds_mls_ds_576_512<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, warp_seqkv_limit);
            }

            // 准备 QK gemm 输出的寄存器
            vec4_Accum<ElementAccum> s_reg[(WARP_M / 16) * (kBlockN / 32)][2];

            // QK gemm
            QK_GEMM_FUNC(qv_ptr /* qv512 */, q_ptr /* q64 */, v_ptr /* k512 */, v_ptr /* kv512 */, q_lds, k_lds, v_lds, s_reg, warp_id, seqlen_qv_stride, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, warp_seqq_limit, warp_seqkv_limit);

            if constexpr (Is_MTP) {
                prefill_mla_apply_mtp_mask_causal_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, warp_offset_in_seqkv, actual_seqlen_k, warp_offset_in_seq_q, actual_seqlen_q);
            }

            // 对 QK 输出做 softmax, 以及重放缩 acc_o/scores_sum
            prefill_mla_softmax_rescale_o<false, Is_causal, vec4_Accum<ElementAccum>, vec_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

            // softmax(QK) f32 -> f16
            union_vec2_f16x2<Element> p_reg[(WARP_M / 16) * (kBlockN / 32)][2];
            prefill_mla_convert_pk_type<WARP_M, kBlockN, Element, ElementAccum, 1/* M_MMAC_COUNT */>(p_reg, s_reg);

            // 计算页表偏移等
            const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
            const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
            const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
            const int offset_diff             = (block_table_idx_cur == block_table_idx_next) ? (64) : (-64);
            *(int64_t*)&k_ptr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);

            // PV gemm
            PV_GEMM_FUNC(q_ptr, k_ptr, v_ptr, q_lds, k_lds, v_lds, p_reg, acc_o, warp_id, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, warp_seqq_limit, warp_seqkv_limit);

            // 偏移 V 指针
            *(int64_t*)&v_ptr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);
        }

        // prefetch K 的话, 最后一次多取了一段 K, 为了不影响后续的操作, 需要同步等待
        if constexpr (PREFETCH_K) {
            buffer_load_lds_dwordx1_wait<0>();
        }

        /***************************************************************************************************************************/
        // Rest loop, 做 causal mask 的部分
        int n_block_loop = max(n_block_max - n_masking_steps, n_block_min);
        #pragma unroll
        for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, ++n_block_loop) {

            // 计算当前 loop 下 seqlen_kv 的数据起始位置和边界
            int warp_offset_in_seqkv = n_block_loop * kBlockN;
            int warp_seqkv_limit     = Is_even_MN ? 0: actual_seqlen_k - warp_offset_in_seqkv;

            // 预取 Q K 的数据到 lds
            prefetch_q_to_lds_mls_ds_576_512<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(q_ptr, q_lds,/* q_reg,*/ warp_id, seqlen_q_stride, warp_seqq_limit);
            prefetch_k_to_lds_mls_ds_576_512<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, warp_seqkv_limit);

            // 准备 QK gemm 输出的寄存器
            vec4_Accum<ElementAccum> s_reg[(WARP_M / 16) * (kBlockN / 32)][2];

            // QK gemm
            QK_GEMM_FUNC(qv_ptr /* qv512 */, q_ptr /* q64 */, v_ptr /* k512 */, v_ptr /* kv512 */, q_lds, k_lds, v_lds, s_reg, warp_id, seqlen_qv_stride, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, warp_seqq_limit, warp_seqkv_limit);

            // 偏移 K 指针, 提前偏移准备预取 K
            const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
            const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
            const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
            const int offset_diff             = (block_table_idx_cur == block_table_idx_next) ? (64) : (-64);
            *(int64_t*)&k_ptr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);

            // Attention: mask, causal mask, local mask
            if constexpr (!Is_causal) {
                if constexpr (!Is_even_MN) { prefill_mla_apply_mask_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, warp_seqkv_limit); }
            } else {
                if constexpr (Is_causal) {
                    if constexpr (Is_MTP) {
                        prefill_mla_apply_mtp_mask_causal_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, warp_offset_in_seqkv, actual_seqlen_k, warp_offset_in_seq_q, actual_seqlen_q);
                    } else {
                        prefill_mla_apply_mask_causal_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, warp_offset_in_seqkv, actual_seqlen_k, warp_offset_in_seq_q, actual_seqlen_q);
                    }
                } 
            }

            // 对 QK 输出做 softmax, 以及重放缩 acc_o/scores_sum
            prefill_mla_softmax_rescale_o<false, Is_causal, vec4_Accum<ElementAccum>, vec_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);


            // softmax(QK) f32 -> f16
            union_vec2_f16x2<Element> p_reg[(WARP_M / 16) * (kBlockN / 32)][2];
            prefill_mla_convert_pk_type<WARP_M, kBlockN, Element, ElementAccum, 1/* M_MMAC_COUNT */>(p_reg, s_reg);

            // PV gemm
            PV_GEMM_FUNC_IN_MASK(q_ptr, k_ptr, v_ptr, q_lds, k_lds, v_lds, p_reg, acc_o, warp_id, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, warp_seqq_limit, warp_seqkv_limit);

            // 偏移 V 指针
            *(int64_t*)&v_ptr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);

        }
        /**********************************************************************************************************************************/
        // Epilogue: rescale acco
        vec_Accum<ElementAccum> lse[WARP_M / 16];
        prefill_mla_epilugue_rescale_acco<WARP_M, kBlockK, kHeadDimV, Is_dropout && Is_training, ElementAccum, vec_Accum<ElementAccum>, 1/* M_MMAC_COUNT */>(acc_o, lse, scores_max, scores_sum, params.scale_softmax, 0/* params.rp_dropout */);
        /**************************************************************************************************************************************/
        constexpr bool Is_Interleave = true;
        int lane_id = threadIdx.x & 63;
        if (params.softmax_lse_ptr != nullptr) {
            prefill_mla_epilogue_store_lse<WARP_M, Is_even_MN, false/*SplitD*/, Is_Interleave, ElementAccum, vec_Accum<ElementAccum>, 1/* M_MMAC_COUNT */>(lse, params.softmax_lse_ptr, row_offset_lse, warp_id, lane_id, 0, actual_seqlen_q - m_block * kBlockM);
        }
        /**************************************************************************************************************************************/
        Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;
        prefill_mla_epilogue_store_output<kHeadDimV, kBlockM, kBlockK, WARP_M, Is_even_MN, Is_Interleave, false/*TcpSwizzle*/, Element, ElementAccum, 1/* M_MMAC_COUNT */>(o_ptr, acc_o, m_block, warp_id, lane_id, seqlen_o_stride, actual_seqlen_q);
    }
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_prefix, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Is_MTP, int Layout, typename Params>
__global__ void __launch_bounds__(512, 1) flash_fwd_mla_decode_kernel_gfx938(const Params params) {
#if defined(__gfx938__)
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
    constexpr int WARP_NUM  = kBlockM / WARP_M;

    {
        const int bidh = blockIdx.y;
        const int bidb = blockIdx.z;
        int warp_id_vec = threadIdx.x / 64;
        const int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
        const int m_block = blockIdx.x;

        // Is_FlashMLA为true则无qv，为False则有qv，适配FA3和FlashMLA接口
        constexpr bool Is_FlashMLA = !Is_prefix;

        // 获取当前任务的长度
        int sum_s_q = Is_FlashMLA ? 0: params.cu_seqlens_q[bidb];
        int actual_seqlen_q = Is_FlashMLA ? params.seqlen_q: params.cu_seqlens_q[bidb + 1] - sum_s_q;
        int actual_seqlen_k = Is_FlashMLA ? params.cu_seqlens_k[bidb]: params.cu_seqlens_k_new[bidb + 1] - params.cu_seqlens_k_new[bidb];

        // 处理边界
        if (m_block * kBlockM >= actual_seqlen_q || actual_seqlen_k <= 0) return;
        const int warp_offset_in_seq_q = m_block * kBlockM + warp_id * WARP_M;
        const int warp_seqq_limit      = Is_even_MN ? 0: actual_seqlen_q - m_block * kBlockM;

        // 分配 lds Q/P same place, K/V same place;
        extern __shared__ Element smem[];
        Element* q_lds = (Element*)&(smem);                     // 16KB
        Element* k_lds = q_lds + ((16*1024) / sizeof(Element)); // 16KB
        Element* v_lds = q_lds;                                 // 32KB

        // 计算当前任务沿着 seqlenKV 方向的 block 起始位置和终止位置
        const int n_block_min = 0;
        int n_block_max = ceil_div(actual_seqlen_k, kBlockN);

        // 计算数据跨度
        int seqlen_q_stride = params.q_row_stride;
        int seqlen_k_stride = params.k_row_stride;
        int seqlen_v_stride = params.v_row_stride;
        int seqlen_o_stride = params.o_row_stride;
        int seqlen_qv_stride = params.qv_row_stride;    // 当走FlashMLA接口时，不会使用

        int row_offset_q, row_offset_k, row_offset_v, row_offset_o, row_offset_lse, row_offset_qv;
        int headdim_split_id = 0;

        const int page_block_size = params.page_block_size;
        int *block_table = params.block_table + bidb * params.block_table_batch_stride;
        const int block_table_idx = 0;
        const int block_table_offset = 0;

        if constexpr (Is_FlashMLA) {
            row_offset_q   = bidb * int64_t(params.q_batch_stride) + m_block * kBlockM * int64_t(params.q_row_stride) + bidh * params.q_head_stride;
            row_offset_o   = bidb * int64_t(params.o_batch_stride) + m_block * kBlockM * int64_t(params.o_row_stride) + bidh * params.o_head_stride;
            row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        } else {
            row_offset_q   = (sum_s_q + m_block * kBlockM) * int64_t(params.q_row_stride) + bidh * params.q_head_stride;
            row_offset_qv  = (sum_s_q + m_block * kBlockM) * int64_t(params.qv_row_stride) + bidh * params.qv_head_stride;
            row_offset_o   = (sum_s_q + m_block * kBlockM) * int64_t(params.o_row_stride) + bidh * params.o_head_stride;
            row_offset_lse = bidh * params.total_q + sum_s_q + m_block * kBlockM;
        }
        row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;

        // 根据起始数据偏移量准备 Q/K/V 的 buffer resource 寄存器
        // q_ptr : 64 | qv_ptr : 512 | k_ptr : 64(k_rope) | v_ptr : 512(k_nope)
        vec4_uint qv_ptr;
        auto q_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
        if constexpr (!Is_FlashMLA)
            qv_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.qv_ptr) + row_offset_qv);
        auto k_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
        auto v_ptr = prepare_for_matrix_load<kHeadDimV>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v);

        // apply causal mask 的步骤和 no causal mask 的步骤分开算
        constexpr int n_masking_steps = 1;

        // 是否做 prefetch K, PV 结束后, prefetch K 有风险
        constexpr bool PREFETCH_K = (Is_even_MN) and( ( kHeadDim == 576 and kHeadDimV == 512 )); // 简单场景下开启
        constexpr bool ALLOW_PREFETCH = (STAGES > 1); // 客观上决定是否开启 prefetch
        if constexpr (PREFETCH_K and ALLOW_PREFETCH) {
            if (n_block_min < n_block_max - n_masking_steps) {
                prefetch_q_to_lds_mls_ds_576_512<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(q_ptr, q_lds, warp_id, seqlen_q_stride, Is_even_MN ? 0: actual_seqlen_q - m_block * kBlockM);
                prefetch_k_to_lds_mls_ds_576_512<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, Is_even_MN ? 0: actual_seqlen_k - n_block_min * kBlockN);
            }
        }

        vec_Accum<ElementAccum> scores_max[WARP_M / 16]; // 只处理16行,所以只有1个reg
        vec_Accum<ElementAccum> scores_sum[WARP_M / 16];
        vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 16) * (kBlockK / 32)][2];

        // 内联失败，手动展开
        {
            constexpr int K_LOOP_COUNT = kHeadDimV / kBlockK;
            constexpr int M_WARP_COUNT = WARP_M / 16;
            constexpr int K_WARP_COUNT = kBlockK / 32;
            constexpr int M_MMAC_COUNT = 1;

            #pragma unroll
            for (int i = 0; i < M_WARP_COUNT; ++i) {
                scores_max[i].f32[0] = -INFINITY;
                scores_sum[i].f32[0] = 0;
            }

            uint64_t pk_zero = 0;
            #pragma unroll
            for (int i = 0; i < K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT; ++i) {
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__) || defined(__gfx92a__)
                        acc_o[i][min_tile_n].u64[0] = __builtin_hcu_mov_b64(pk_zero);
                        acc_o[i][min_tile_n].u64[1] = __builtin_hcu_mov_b64(pk_zero);
                        #else
                        acc_o[i][min_tile_n].f32[0] = 0;
                        acc_o[i][min_tile_n].f32[1] = 0;
                        acc_o[i][min_tile_n].f32[2] = 0;
                        acc_o[i][min_tile_n].f32[3] = 0;
                        #endif
                    }
                }
            }
        }

        auto QK_GEMM_FUNC = &qk_gemm_prefetch_v_mls_ds_576_512<kHeadDim, kHeadDimV, kBlockM, kBlockN, kBlockK, WARP_M, WARP_N, STAGES, Element, ElementAccum, Is_even_MN, Is_FlashMLA>;
        auto PV_GEMM_FUNC = &pv_gemm_prefetch_k_mls_ds_576_512<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;
        auto PV_GEMM_FUNC_IN_MASK = &pv_gemm_prefetch_k_mls_ds_576_512<false, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;

        // Mainloop, 主循环, 不做 causal mask 的部分
        for (int n_block_loop = n_block_min; n_block_loop < n_block_max; ++n_block_loop) {

            // 计算当前 loop 下 seqlen_kv 的数据起始位置和边界
            int warp_offset_in_seqkv = n_block_loop * kBlockN;
            int warp_seqkv_limit     = Is_even_MN ? 0: actual_seqlen_k - warp_offset_in_seqkv;

            // 预取 K 的数据到 lds // 预取 k_ptr(k_rope), q_ptr 到 k_lds, q_lds
            if constexpr (not PREFETCH_K) {
                prefetch_q_to_lds_mls_ds_576_512<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(q_ptr, q_lds, warp_id, seqlen_q_stride, warp_seqq_limit);
                prefetch_k_to_lds_mls_ds_576_512<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, warp_seqkv_limit);
            }

            // 准备 QK gemm 输出的寄存器
            vec4_Accum<ElementAccum> s_reg[(WARP_M / 16) * (kBlockN / 32)][2];

            // QK gemm
            if constexpr (Is_FlashMLA) {
                QK_GEMM_FUNC(q_ptr /* q576 */, q_ptr /* q576 */, k_ptr /* k576 */, v_ptr /* v512 */, q_lds, k_lds, v_lds, s_reg, warp_id, seqlen_qv_stride, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, warp_seqq_limit, warp_seqkv_limit);
            } else {
                QK_GEMM_FUNC(qv_ptr /* qv512 */, q_ptr /* q64 */, v_ptr /* k512 */, v_ptr /* kv512 */, q_lds, k_lds, v_lds, s_reg, warp_id, seqlen_qv_stride, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, warp_seqq_limit, warp_seqkv_limit);
            }

            if constexpr (!Is_causal) {
                if constexpr (!Is_even_MN) { prefill_mla_apply_mask_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, warp_seqkv_limit); }
            } else {
                    if constexpr (Is_MTP) {
                        if constexpr (Is_FlashMLA) {
                            flashmla_apply_mtp_mask_causal_gfx938<vec4_Accum<ElementAccum>, kBlockN, WARP_M, WARP_NUM>(s_reg, warp_offset_in_seqkv/* n_block_loop * kBlockN */, actual_seqlen_k, warp_offset_in_seq_q/* m_block * kBlockM + warp_id_row * 16 */, actual_seqlen_q, params.ngroups, params.mtp);
                        } else {
                            prefill_mla_apply_mtp_mask_causal_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, warp_offset_in_seqkv, actual_seqlen_k, warp_offset_in_seq_q, actual_seqlen_q);
                        }
                    } else {
                        prefill_mla_apply_mask_causal_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, warp_offset_in_seqkv, actual_seqlen_k, warp_offset_in_seq_q, actual_seqlen_q);
                    }
            }

            // 对 QK 输出做 softmax, 以及重放缩 acc_o/scores_sum
            prefill_mla_softmax_rescale_o<false, Is_causal, vec4_Accum<ElementAccum>, vec_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN, 1/* M_MMAC_COUNT */>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

            // softmax(QK) f32 -> f16
            union_vec2_f16x2<Element> p_reg[(WARP_M / 16) * (kBlockN / 32)][2];
            prefill_mla_convert_pk_type<WARP_M, kBlockN, Element, ElementAccum, 1/* M_MMAC_COUNT */>(p_reg, s_reg);

            // 计算页表偏移等
            const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
            const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
            const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
            const int offset_diff             = (block_table_idx_cur == block_table_idx_next) ? (64) : (-64);
            *(int64_t*)&k_ptr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);

            // PV gemm
            PV_GEMM_FUNC_IN_MASK(q_ptr, k_ptr, v_ptr, q_lds, k_lds, v_lds, p_reg, acc_o, warp_id, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, warp_seqq_limit, warp_seqkv_limit);

            // 偏移 V 指针
            *(int64_t*)&v_ptr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);
        }
        /**********************************************************************************************************************************/
        // Epilogue: rescale acco
        vec_Accum<ElementAccum> lse[WARP_M / 16];
        prefill_mla_epilugue_rescale_acco<WARP_M, kBlockK, kHeadDimV, Is_dropout && Is_training, ElementAccum, vec_Accum<ElementAccum>, 1/* M_MMAC_COUNT */>(acc_o, lse, scores_max, scores_sum, params.scale_softmax, 0/* params.rp_dropout */);
        /**************************************************************************************************************************************/
        constexpr bool Is_Interleave = true;
        int lane_id = threadIdx.x & 63;
        if (params.softmax_lse_ptr != nullptr) {
            prefill_mla_epilogue_store_lse<WARP_M, Is_even_MN, false/*SplitD*/, Is_Interleave, ElementAccum, vec_Accum<ElementAccum>, 1/* M_MMAC_COUNT */>(lse, params.softmax_lse_ptr, row_offset_lse, warp_id, lane_id, 0, actual_seqlen_q - m_block * kBlockM);
        }
        /**************************************************************************************************************************************/
        Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;
        prefill_mla_epilogue_store_output<kHeadDimV, kBlockM, kBlockK, WARP_M, Is_even_MN, Is_Interleave, false/*TcpSwizzle*/, Element, ElementAccum, 1/* M_MMAC_COUNT */>(o_ptr, acc_o, m_block, warp_id, lane_id, seqlen_o_stride, actual_seqlen_q);
    }
#endif
}

} // namespace flash