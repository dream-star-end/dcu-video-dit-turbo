#pragma once
#include "flash_fwd_b8_pa.h"
#include "mla/gfx938/fp8_mla_tp8_qk_gemm_utils_gfx938.h"
#include "mla/gfx938/fp8_mla_tp8_qk_gemm_gfx938.h"
#include "mla/gfx938/fp8_mla_tp8_pv_gemm_prefetch_k_gfx938.h"
#include "mla/gfx938/fp8_mla_softmax_gfx938.h"
#include "mla/gfx938/fp8_mla_epilogue_gfx938.h"
// #include "mla/gfx938/fp8_mla_acco_reduce_gfx938.h"

namespace flash {

/*
    sgpr: 86
    vgpr: 209
*/
template<typename Kernel_traits, bool Is_causal, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_fp8_mla_gfx938(const Params &params, const int bidb, const int bidh, const int warp_id) {
    using Element          = fp8_e4m3;
    using ElementAccum     = typename Kernel_traits::ElementAccum;
    using index_t          = typename Kernel_traits::index_t;
    using SplitkvAccumType = typename Kernel_traits::SplitkvAccumType;
    using AttnType         = half_t;
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

    // compute workload
    flash::SafeDecodeBlockInfo binfo;
    binfo.set_params<Params, /*Is_Q_varlen=*/false, /*Is_K_Cumulative=*/false>(params, bidb);

    // splitkv
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

    // acquire seq_q_id / headdim_split_id
    int block_x = blockIdx.x;
    const int m_block          = block_x / HEADDIM_V_SPLIT;
    const int headdim_split_id = block_x & (HEADDIM_V_SPLIT - 1);

    // 确定运行边界
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k <= 0) return;

    // 决定 lds 用量划分
    extern __shared__ Element fp8_smem[];
    Element* q_lds = reinterpret_cast<Element*>(fp8_smem);
    Element* k_lds = reinterpret_cast<Element*>(fp8_smem);
    Element* v_lds = k_lds;
    ElementAccum* acc_o_lds = reinterpret_cast<ElementAccum*>(fp8_smem);
    ElementAccum* max_lds   = acc_o_lds + 4096/*from 4096 bytes*/;

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

    // 读取 descale 因子
    ElementAccum* q_descale_ptr = reinterpret_cast<ElementAccum*>(params.scales_q_ptr);
    ElementAccum* k_descale_ptr = reinterpret_cast<ElementAccum*>(params.scales_k_ptr);
    ElementAccum q_descale = q_descale_ptr[0];
    ElementAccum k_descale = k_descale_ptr[0];
    __float2 qk_descale;
    qk_descale[0] = q_descale * k_descale;
    qk_descale[1] = qk_descale[0];

    // splitkv, debug 场景下需要写出一些值, 例如 scores_max/scores_sum
    int row_offset_lse;
    ElementAccum * softmax_lse_ptr;
    if constexpr (Split) {
        row_offset_lse = split_id * (params.b * params.h * params.seqlen_q) + (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        softmax_lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + row_offset_lse;
    }

    // 确认分块任务大小
    constexpr int M_WARP_COUNT = WARP_M / 32;
    constexpr int K_WARP_COUNT = kBlockK / 32;
    constexpr int N_WARP_COUNT = WARP_N / 32;
    constexpr int K_LOOP_COUNT = kHeadDimVSplit / kBlockK;

    // 准备初始状态, scores_max/scores_max/acc_o
    vec2_Accum<ElementAccum> scores_max[M_WARP_COUNT];
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT];
    vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4];

    // 预取 Q 到寄存器或者 lds
    union_vec16_fp8 q_reg[M_MMAC_COUNT][kHeadDim / 64];
    fp8_mla_tp8_prefetch_q_to_vgpr_gfx938_with_initialization<kHeadDim, kHeadDimVSplit, kBlockM, kBlockK, WARP_M, WARP_NUM, Element, ElementAccum, STAGES, M_MMAC_COUNT>(
        q_addr, q_lds, q_reg, warp_id, query_seqlen_stride, binfo.actual_seqlen_q - m_block * kBlockM, scores_max, scores_sum, acc_o);

    /**********************************************************************************************************************************/
    // 主循环, 沿着 seqlenKV 维度, 每次 4 个 wave 共同计算一个 kBLOCKN
    const int n_block_min = 0;
    const int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    int n_block_loop      = n_block_min;
    constexpr bool PrefetchK = true;
    if constexpr (PrefetchK) {
        int warp_seqkv_limit = binfo.actual_seqlen_k - n_block_min * kBlockN;
        fp8_mla_tp8_prefetch_k_gfx938<WARP_NUM, Element>(k_addr, k_lds, warp_id, kcache_seqlen_stride, warp_seqkv_limit);
    }
    for (; n_block_loop < n_block_max; ++n_block_loop) {

        int warp_offset_in_seqkv = n_block_loop * kBlockN + warp_id * WARP_N;
        int warp_seqkv_limit     = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        if constexpr (not PrefetchK) {
            fp8_mla_tp8_prefetch_k_gfx938<WARP_NUM, Element>(k_addr, k_lds, warp_id, kcache_seqlen_stride, warp_seqkv_limit);
        }

        vec4_Accum<ElementAccum> s_reg[M_WARP_COUNT * N_WARP_COUNT][4];
        fp8_mla_tp8_qk_gemm_gfx938<kHeadDim, kBlockK, WARP_M, WARP_N, WARP_NUM, M_MMAC_COUNT, Element, ElementAccum>(
            k_addr, k_lds, q_reg, s_reg, warp_id, kcache_seqlen_stride, warp_seqkv_limit);

        fp8_mla_tp8_prefetch_v_gfx938<K_LOOP_COUNT, kBlockK, WARP_NUM, Element>(v_addr, v_lds, warp_id, vcache_seqlen_stride, warp_seqkv_limit);

        fp8_mla_apply_descale_gfx938<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, qk_descale);

        if constexpr (Is_causal) {
            fp8_mla_apply_mask_causal_gfx938_mtp<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, warp_offset_in_seqkv + this_split_seqlen_start, original_actual_seqlen_k, m_block * kBlockM, binfo.actual_seqlen_q, params.mtp, params.layout);
        } else {
            fp8_mla_apply_mask_gfx938<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, warp_seqkv_limit, warp_id * WARP_N);
        }

        mla_softmax_rescale_o<Is_causal, ElementAccum, K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, N_WARP_COUNT, WARP_NUM, M_MMAC_COUNT>(
            s_reg, scores_max, scores_sum, acc_o, max_lds, warp_id, params.scale_softmax_log2);

        union_vec2_f16x2<AttnType> p_reg[M_WARP_COUNT * N_WARP_COUNT][4];
        mla_convert_pk_type<M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT, AttnType, ElementAccum>(p_reg, s_reg);

        const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;
        const int64_t k_addr_offset       = (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);

        fp8_mla_tp8_pv_gemm_prefetch_k_gfx938<PrefetchK, K_LOOP_COUNT, kBlockK, kBlockN, M_WARP_COUNT, K_WARP_COUNT, WARP_NUM, M_MMAC_COUNT, Element, AttnType, ElementAccum>(
            v_addr, k_addr, v_lds, k_lds, p_reg, acc_o, warp_id, kcache_seqlen_stride, vcache_seqlen_stride, warp_seqkv_limit, k_addr_offset);

        // *(int64_t*)&k_addr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);
        *(int64_t*)&v_addr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);
    }
    if constexpr (PrefetchK) {
        flash::wait_buffer_data_arrived<false/*sync*/>(0);
    }
    flash::wait_lds_data_arrived<true/*sync*/>(0);

    // 多个 wave 同步 acc_o
    int thread_id = threadIdx.x;
    int lane_id   = thread_id & 63;
    if constexpr (WARP_NUM > 1) {
        fp8_mla_acco_reduce_tile16x32<REUSE_KV_TIMES, K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, M_MMAC_COUNT, WARP_NUM, 4/*Padding*/, ElementAccum>(acc_o, acc_o_lds, params.seqlen_q, warp_id, lane_id);
    }

    /**********************************************************************************************************************************/
    // Epilogue, 收尾工作
    // 收尾 1: 根据最后的归一化求和, 做 rescale
    fp8_mla_epilugue_rescale_acco_gfx938<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(acc_o, scores_sum, k_descale/*v_descale*/);

    // 收尾 2: splitkv, 或者开启 debug 的情况下, 写出 scores_max, scores_sum
    mla_tp8_epilogue_store_softmax_lse<Split, true/*Is_16x32*/, M_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
        scores_max, scores_sum, softmax_lse_ptr, params.scale_softmax, warp_id, thread_id, lane_id, headdim_split_id, binfo.actual_seqlen_q - m_block * kBlockM);

    // 收尾 3: 写出 output
    mla_tp8_epilogue_store_output_gfx938<Params, kHeadDimV, kHeadDimVSplit, true/*alt*/, Split, SplitkvAccumType, ElementAccum, kBlockM, kBlockK, WARP_NUM, K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT>(
        acc_o, params, bidb, bidh, m_block, split_id, headdim_split_id, warp_id, lane_id);
}


template<typename Kernel_traits, bool Is_causal, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
inline __device__ void compute_attn_splitkv_fp8_mla_gfx938(const Params &params) {

#if defined(__gfx938__) || defined(__gfx946__)
    // The block index for the head.
    const int bidh = Split ? blockIdx.z % params.h : blockIdx.y; // batch x num_head, num_head first

    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.z;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id     = __builtin_amdgcn_readfirstlane(warp_id_vec);

    flash::compute_attn_1rowblock_splitkv_fp8_mla_gfx938<Kernel_traits, Is_causal, Split, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size * 128, Params>(params, bidb, bidh, warp_id);
#endif
}



// =====================================================================================================================================

template<typename input_dtype, typename output_dtype>
__device__ union_vec2_f16x2<input_dtype> flash_mla_convert_f16_to_fp8(const union_vec4_f16x2<input_dtype>& src_f16) {
    union_vec8_fp32 src_f32;
    for (int i = 0; i < 8; ++i) {
        src_f32.f32[i] = splitkv_upcast_to_f32<float, input_dtype>(src_f16.f16[i]);
    }
    union_vec2_f16x2<input_dtype> dst;
    __builtin_hcu_cvt_pk4_fp8_f32<output_dtype>(src_f32.f32x4[0], dst.i32[0]);
    __builtin_hcu_cvt_pk4_fp8_f32<output_dtype>(src_f32.f32x4[1], dst.i32[1]);
    return dst;
}

template<typename input_dtype, typename output_dtype, bool persistent>
__global__ void flash_mla_convert_query_to_fp8_kernel(
    output_dtype* output,
    input_dtype* q_nope,
    input_dtype* q_rope,
    const int total_blocks,
    const int o_stride,
    const int nope_head_stride,
    const int rope_head_stride,
    const int nope_row_stride,
    const int rope_row_stride,
    const int qheads) {
#if defined(__gfx938__) || defined(__gfx946__)
    if constexpr (persistent) {
        for (int bidb = blockIdx.x; bidb < total_blocks; bidb += gridDim.x) {
            // --------------------- nope -------------------------
            int bidh      = bidb % qheads;
            int bids      = bidb / qheads;
            int boffset   = bids * nope_row_stride + bidh * nope_head_stride;
            int tx        = threadIdx.x;
            int tx_offset = tx << 3;
            auto nope_f16 = *(union_vec4_f16x2<input_dtype>*)(q_nope + boffset + tx_offset);
            auto nope_fp8 = flash_mla_convert_f16_to_fp8<input_dtype, output_dtype>(nope_f16);
            *(double*)(output + bidb * o_stride + tx_offset) = nope_fp8.data;
            // --------------------- rope -------------------------
            if (tx < 8) {
                int boffset   = bids * rope_row_stride + bidh * rope_head_stride;
                auto rope_f16 = *(union_vec4_f16x2<input_dtype>*)(q_rope + boffset + tx_offset);
                auto rope_fp8 = flash_mla_convert_f16_to_fp8<input_dtype, output_dtype>(rope_f16);
                *(double*)(output + bidb * o_stride + 512 + tx_offset) = rope_fp8.data;
            }
        }
    } else {
        // 512 f16 + 64 f16 -> 576 fp8
        // --------------------- nope -------------------------
        int bidb      = blockIdx.x;
        int bidh      = bidb % qheads;
        int bids      = bidb / qheads;
        int boffset   = bids * nope_row_stride + bidh * nope_head_stride;
        int tx        = threadIdx.x;
        int tx_offset = tx << 3;
        auto nope_f16 = *(union_vec4_f16x2<input_dtype>*)(q_nope + boffset + tx_offset);
        auto nope_fp8 = flash_mla_convert_f16_to_fp8<input_dtype, output_dtype>(nope_f16);
        *(double*)(output + bidb * o_stride + tx_offset) = nope_fp8.data;
        // --------------------- rope -------------------------
        if (tx < 8) {
            int boffset   = bids * rope_row_stride + bidh * rope_head_stride;
            auto rope_f16 = *(union_vec4_f16x2<input_dtype>*)(q_rope + boffset + tx_offset);
            auto rope_fp8 = flash_mla_convert_f16_to_fp8<input_dtype, output_dtype>(rope_f16);
            *(double*)(output + bidb * o_stride + 512 + tx_offset) = rope_fp8.data;
        }
    }
#endif
}

} // namespace flash