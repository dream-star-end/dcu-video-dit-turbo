#pragma once
#include "flash_fwd_b16_pa.h"

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////


#include "kvcache/int8_kvcache_qk_gemm_prefetch_v.h"
#include "kvcache/int8_kvcache_pv_gemm_prefetch_k.h"
#include "kvcache/int8_kvcache_softmax.h"
#include "kvcache/int8_kvcache_acco_reduce.h"

template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_K, bool Return_softmax, bool Has_alibi, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, typename Params>
inline __device__ void compute_attn_mha_1rowblock_splitkv_int8(const Params &params, const int bidb, const int bidh, const int WARP_ID) {
    using Element          = typename Kernel_traits::Element;
    using Element_k        = typename Kernel_traits::Element_k;
    using ElementAccum     = typename Kernel_traits::ElementAccum;
    using index_t          = typename Kernel_traits::index_t;

    #ifndef NDEBUGING
    // ElementAccum * qk_ptr = static_cast<ElementAccum*>(params.qk_ptr);
    int * qk_ptr = static_cast<int*>(params.qk_ptr);
    #endif
    constexpr int kBlockM  = Kernel_traits::kBlockM;
    constexpr int kBlockN  = Kernel_traits::kBlockN;
    constexpr int kBlockK_int8  = Kernel_traits::kBlockK_int8;
    constexpr int kBlockK  = Kernel_traits::kBlockK;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps  = Kernel_traits::kNWarps;
    constexpr int WARP_M   = Kernel_traits::kWaveM;
    constexpr int WARP_N   = Kernel_traits::kWaveN;
    constexpr int STAGES   = Kernel_traits::STAGES;

    flash::BlockInfo</*Varlen=*/true, /*Is_Kvcache*/true> binfo(params, bidb);

    // recompute the true actual_seqlen_k and num_split according to split_id, especially the last block
    int split_id;
    if constexpr (Split) {
        split_id   = blockIdx.y;
        int num_splits = max(1, floor_div(binfo.actual_seqlen_k, params.partition_size));
        binfo.actual_seqlen_k = (split_id == num_splits - 1)
            ? binfo.actual_seqlen_k - split_id * params.partition_size: params.partition_size;
        binfo.actual_seqlen_k = (split_id >= num_splits) ? 0: binfo.actual_seqlen_k;
        if (split_id >= num_splits) return;
    }

    const int WARP_NUM = (kBlockN)/(WARP_N);

    // kvcache doesn't has mask, no need to balance workload
    const int m_block = blockIdx.x;

    // when groups is more than 32, this may lead to incorrect results
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0) return;
    extern __shared__ Element smem[];

    // decide lds partition
    // load Q --> QK gemm load K --> PV gemm load V, no conflicts
    #if defined(KVCACHE_USE_4STAGES_PINGPANG) && (defined(__gfx936__) || defined(__gfx938__))
        Element_k* q_lds = reinterpret_cast<Element_k*>(smem);
        Element_k* k_lds = q_lds;
        Element_k* v_lds = q_lds;
        ElementAccum* acc_o_lds = reinterpret_cast<ElementAccum*>(smem);
        ElementAccum* max_lds   = reinterpret_cast<ElementAccum*>(v_lds + (4096 - 256));
        /*默认占用 32KB, 最后一个 4x128 不预取, 留给 max_lds 做多 wave 之间的 reduce, 这里选的是 0 号 wave lds 最后一段空间(一共 8KB, 4096 个 half), 512 B 足够了, 即使 256 个 Half*/
    #else
        Element_k* q_lds = reinterpret_cast<Element_k*>(smem) + 512/*1KB, 512 halfs, configured for max_lds*/;
        Element_k* k_lds = q_lds;
        // Element_k* v_lds_int8 = reinterpret_cast<Element_k*>(q_lds);
        Element_k* v_lds = q_lds;
        // prepare lds for max and acc whiling reducing results across 4 waves
        // max and acc_o_lds has no conflicts while using lds
        ElementAccum* acc_o_lds = reinterpret_cast<ElementAccum*>(smem);
        ElementAccum* max_lds   = acc_o_lds;
        // float* scales_q = reinterpret_cast<float*>(smem);
        // __shared__ ElementAccum acc_o_lds[kHeadDim * 4];
        // __shared__ ElementAccum max_lds[WARP_NUM * WARP_M];
    #endif

    vec4_int8 q_reg[(kHeadDim/kBlockK_int8)*((WARP_M*kBlockK_int8)/(32*kBlockK_int8))*2][4];  //ds_read mini size is 32*32,2 is seq, 4 is head dim

    const int n_block_min = 0;
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);

    // acquire stride over seqlen dimension
    const int query_seqlen_stride  = params.q_row_stride;
    const int kcache_seqlen_stride = params.k_row_stride;
    const int vcache_seqlen_stride = params.v_row_stride;
    const int scale_kcache_seqlen_stride = kcache_seqlen_stride / kHeadDim;

    // compute block table
    const int page_block_size = params.page_block_size;
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    // if split, block_table begin from the new split!
    block_table = block_table + (Split ? ceil_div(split_id * params.partition_size, page_block_size) : 0);
    const int block_table_idx = 0;
    const int block_table_offset = 0;
    const int64_t row_offset_k = block_table == nullptr
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    // const index_t row_offset_v = block_table == nullptr
    //     ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
    //     + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
    //     : block_table[block_table_idx] * params.v_batch_stride + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;

    const int64_t row_offset_q = bidb * params.q_batch_stride + bidh * params.q_head_stride + m_block * kBlockM * query_seqlen_stride;

    #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
        constexpr bool USE_CACHE_SWIZZLE = false;
    #else
        constexpr bool USE_CACHE_SWIZZLE = true; // for gfx928, cache swizzle have significant influence
    #endif

    auto gQ = prepare_for_buffer_load<kHeadDim, Element_k, false>(reinterpret_cast<Element_k*>(params.q_ptr) + row_offset_q);
    auto gK = prepare_for_buffer_load<kHeadDim, Element_k, false>(reinterpret_cast<Element_k*>(params.k_ptr) + row_offset_k);
    auto gV = prepare_for_buffer_load<kHeadDimV, Element_k, false>(reinterpret_cast<Element_k*>(params.v_ptr) + row_offset_k);

    float *scales_q_ptr = reinterpret_cast<ElementAccum*>(params.scales_q_ptr);
    float *scales_k_ptr = reinterpret_cast<ElementAccum*>(params.scales_k_ptr);
    float *scales_v_ptr = reinterpret_cast<ElementAccum*>(params.scales_v_ptr);
    
    int lane_id = threadIdx.x & 63; // lane id, 0-63
    int q_seq_idx = (lane_id & 15) * 2;
    float scales_q[M_MMAC_COUNT];
    int scales_q_global_offset = row_offset_q / kHeadDim;
    for (int m_idx=0; m_idx<M_MMAC_COUNT; m_idx++){
        int scales_q_offset = min(scales_q_global_offset + q_seq_idx + m_idx, params.total_scale_q);
        scales_q[m_idx] = scales_q_ptr[scales_q_offset];
    }

    int row_offset_lse;
    ElementAccum * scores_sum_ptr;
    ElementAccum * scores_max_ptr;
    if constexpr (Split) {
        row_offset_lse = split_id * (params.b * params.h * params.seqlen_q) + (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        scores_sum_ptr = reinterpret_cast<ElementAccum*>(params.scores_sum_ptr) + row_offset_lse;
        scores_max_ptr = reinterpret_cast<ElementAccum*>(params.scores_max_ptr) + row_offset_lse;
    }

    float gAlibi;
    if constexpr (Has_alibi) {
        gAlibi = reinterpret_cast<ElementAccum*>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    }

    unsigned long long rand_seed, rand_offset;
    uint32_t p_dropout_in_8bits_value;
    union_vec2_uint warp_idx_for_dropout;
    if constexpr (Is_dropout and Is_training) {
        rand_seed                = params.rand_seed;
        rand_offset              = params.rand_offset + ((bidb * params.h + bidh) << 6) + threadIdx.x & 63; /* 参考官方写法 offset(offset + (bid * nheads + hid) * 32 + tid % 32) */
        p_dropout_in_8bits_value = params.p_dropout_in_uint8_t & 0xffffffff; /*DCU 不支持 16bit 和 8bit 的比较指令*/
        warp_idx_for_dropout.u32.x = 1 * m_block * (kBlockM / 32)/*前面几个 block 累积的 warp 数目, 这里不直接填 WARP_M, 参照 NV 的写法*/ + WARP_ID/*当前 block 内的 warp id*/;
        // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might exit early and no one saves the rng states.
        if (Is_training and m_block == 0 and bidb == 0 and bidh == 0 and threadIdx.x == 0) {
            params.rng_state[0] = rand_seed;
            params.rng_state[1] = rand_offset;
        }
    }


    int8_kvcache_prefetch_q_to_vgpr<kHeadDim, kBlockM, kBlockK_int8, WARP_M, WARP_NUM, Element, Element_k, STAGES, REUSE_KV_TIMES, M_MMAC_COUNT>(gQ, q_lds, q_reg, WARP_ID, (binfo.actual_seqlen_q - m_block * kBlockM));
    /***************************************************************************************************************************/
    vec2_Accum<ElementAccum> scores_max[WARP_M/32] = {-INFINITY};
    vec2_Accum<ElementAccum> scores_sum[WARP_M/32] = {0};
    // 由于当前编译器无法自动生成 v_mov_b64 指令, 主动用 builtin 还会被转译成 v_mov_b32, 因此用内联汇编控制
    #if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx946__)
        vec4_Accum<ElementAccum> acc_o[(kHeadDimV/kBlockK) * ((WARP_M/32)*(kBlockK/32))][4];
        if constexpr (kHeadDimV == 128) { // kHeadDim 128 是主要优化目标
            if constexpr (M_MMAC_COUNT == 1) {
                inline_vgpr4_init_zero_4x2x4(acc_o);
            } else {
                inline_vgpr4_init_zero_4x4x4(acc_o);
            }
            __builtin_amdgcn_sched_barrier(0);
        } else { // 非 kHeaddim 128, 交给编译器后续的优化了
            uint64_t pk_zero = 0;
            #pragma unroll
            for (int i = 0; i < (kHeadDimV/kBlockK) * ((WARP_M/32)*(kBlockK/32)); ++i) {
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        acc_o[i][min_tile_n * 2 + min_tile_m].u64[0] = __builtin_hcu_mov_b64(pk_zero);
                        acc_o[i][min_tile_n * 2 + min_tile_m].u64[1] = __builtin_hcu_mov_b64(pk_zero);
                    }
                }
            }
        }
    #else // gfx928
        vec4_Accum<ElementAccum> acc_o[(kHeadDimV/kBlockK) * ((WARP_M/32)*(kBlockK/32))][4] = {0};
    #endif
    
    constexpr bool PREFETCH_K = false; // KV lds 使用量较大, 暂不适合使用 prefetch K
    int block_table_idx_cur = block_table_idx;
    // int scales_k_global_offset = block_table[block_table_idx_cur] * params.scales_k_batch_stride + (bidh / params.h_h_k_ratio) * params.scales_k_head_stride;
    int64_t scales_k_global_offset = row_offset_k / kHeadDim;
    int table_diff = 0;
    int offset_diff = 0;
    // These are the iterations where we don't need masking on S
    // Separated processing of masked and unmasked would result in vgpr spill
    // Can be reviewed in the future
    int n_block_loop = n_block_min;
    float scales_k[2][4];
    float scales_v[2][4];
    int pre_scales_k_offset[2][4];
    #pragma unroll
    for (int min_tile_n=0; min_tile_n<2; min_tile_n++) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            pre_scales_k_offset[min_tile_n][k] = (lane_id/16*2 + k*8 + min_tile_n + WARP_N*WARP_ID)*scale_kcache_seqlen_stride;
        }
    }
    
    for (; n_block_loop < n_block_max - 1;) {
        int warp_id_vec = threadIdx.x / 64; // warp id in a block
        int warp_id     = __builtin_amdgcn_readfirstlane(warp_id_vec);
        int warp_offset_in_seqkv = n_block_loop * kBlockN + warp_id * WARP_N;
        int warp_seqkv_limit     = binfo.actual_seqlen_k - warp_offset_in_seqkv;
        scales_k_global_offset += (int64_t(table_diff) * int64_t(params.k_batch_stride) + int64_t(offset_diff) * int64_t(params.k_row_stride))/kHeadDim;

        if constexpr ((not PREFETCH_K) and (STAGES > 1)) {
            int8_kvcache_prefetch_k_to_lds<kHeadDim, kBlockM, kBlockK_int8, WARP_M, WARP_N, Element, Element_k, STAGES, WARP_NUM>(gK, k_lds, warp_id, kcache_seqlen_stride, warp_seqkv_limit);
        }

        #pragma unroll
        for (int min_tile_n=0; min_tile_n<2; min_tile_n++) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int scales_k_offset = scales_k_global_offset + pre_scales_k_offset[min_tile_n][k];
                scales_k[min_tile_n][k] = scales_k_ptr[scales_k_offset];
                // scales_v[min_tile_n][k] = scales_v_ptr[scales_k_offset];
            }
        }

        union_vec4_int32 s_reg[(WARP_M/32)*(WARP_N/32)][4];
        int8_kvcache_qk_gemm_prefetch_v_3stage<kHeadDim, kHeadDimV, kBlockM, WARP_N, kBlockK_int8, kBlockK, WARP_M, WARP_N, WARP_NUM, STAGES, M_MMAC_COUNT, Element, Element_k, ElementAccum>(gK, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, warp_id, kcache_seqlen_stride, kcache_seqlen_stride, warp_seqkv_limit);
        vec4_Accum<ElementAccum> s_reg_fp32[(WARP_M/32)*(WARP_N/32)][4];

        // 将 s_reg 的值转换为 fp32
        #pragma unroll
        for (int min_tile_n=0; min_tile_n<2; min_tile_n++) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int scales_k_offset = scales_k_global_offset + pre_scales_k_offset[min_tile_n][k];
                scales_v[min_tile_n][k] = scales_v_ptr[scales_k_offset];
                #pragma unroll
                for (int i = 0; i < (WARP_M/32)*(WARP_N/32); ++i) {
                    #pragma unroll
                    for (int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                        s_reg_fp32[i][min_tile_n*2 + min_tile_m].f32[k] = static_cast<float>(s_reg[i][min_tile_n*2 + min_tile_m].int32[k]) * scales_q[min_tile_m] * scales_k[min_tile_n][k];
                    }
                }
            }
        }

        if constexpr (Has_alibi) {
            int8_kvcache_apply_alibi<vec4_Accum<ElementAccum>, WARP_M, WARP_N, M_MMAC_COUNT>(s_reg_fp32, warp_offset_in_seqkv, binfo.actual_seqlen_k, (m_block * kBlockM), binfo.actual_seqlen_q, gAlibi);
        }

        // kvcache_apply_mask<vec4_Accum<ElementAccum>, WARP_M, WARP_N, M_MMAC_COUNT>(s_reg, warp_seqkv_limit);

        if constexpr (Is_local) {
            int8_kvcache_apply_mask_local</*HasWSLeft=*/Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN, M_MMAC_COUNT>(s_reg_fp32, warp_offset_in_seqkv, binfo.actual_seqlen_k,
                                                                                (m_block * kBlockM + warp_id * WARP_M),
                                                                                    binfo.actual_seqlen_q, params.window_size_left,
                                                                                    params.window_size_right);
        }

        int8_kvcache_softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, ElementAccum, kHeadDim, kBlockK, WARP_M, WARP_N, WARP_NUM, M_MMAC_COUNT>(s_reg_fp32, scores_max, scores_sum, acc_o, max_lds, warp_id, params.scale_softmax_log2);

        if constexpr (Is_dropout and Is_training) {
            warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
            int8_kvcache_apply_dropout<vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, M_MMAC_COUNT>(s_reg_fp32, binfo.actual_seqlen_k - warp_seqkv_limit, 0, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
        }

        union_vec2_f16x2<Element> p_reg[(WARP_M/32)*(WARP_N/32)][4];
        // convertType: float2half
        int8_kvcache_convert_pk_type<WARP_M, WARP_N, M_MMAC_COUNT, Element, ElementAccum>(p_reg, s_reg_fp32);

        // 如果要预取 K 的话, 需要提前偏移
        if constexpr (PREFETCH_K) {
            n_block_loop++;
        }

        block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        offset_diff             = block_table_offset_next - block_table_offset_cur;

        *(uint64_t*)&gK += ((int64_t(table_diff) * int64_t(params.k_batch_stride) + int64_t(offset_diff) * int64_t(params.k_row_stride)) * sizeof(Element_k));

        int8_kvcache_pv_gemm_prefetch_k_3stage<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, WARP_N, STAGES, WARP_NUM, M_MMAC_COUNT, Element, Element_k, ElementAccum>(gV, gK, v_lds, k_lds, scales_v, p_reg, acc_o, warp_id, kcache_seqlen_stride, warp_seqkv_limit);

        *(uint64_t*)&gV += ((int64_t(table_diff) * int64_t(params.v_batch_stride) + int64_t(offset_diff) * int64_t(params.v_row_stride)) * sizeof(Element_k));

        if constexpr (not PREFETCH_K) {
            n_block_loop++;
        }

    }

    {
        int warp_offset_in_seqkv = n_block_loop * kBlockN + WARP_ID * WARP_N;
        int warp_seqkv_limit     = binfo.actual_seqlen_k - warp_offset_in_seqkv;
        scales_k_global_offset += (int64_t(table_diff) * int64_t(params.k_batch_stride) + int64_t(offset_diff) * int64_t(params.k_row_stride))/kHeadDim;

        if constexpr ((not PREFETCH_K) and (STAGES > 1)) {
            int8_kvcache_prefetch_k_to_lds<kHeadDim, kBlockM, kBlockK_int8, WARP_M, WARP_N, Element, Element_k, STAGES, WARP_NUM>(gK, k_lds, WARP_ID, kcache_seqlen_stride, warp_seqkv_limit);
        }

        #pragma unroll
        for (int min_tile_n=0; min_tile_n<2; min_tile_n++) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int scales_k_offset = scales_k_global_offset + pre_scales_k_offset[min_tile_n][k];
                scales_k[min_tile_n][k] = scales_k_ptr[scales_k_offset];
            }
        }
        
        union_vec4_int32 s_reg[(WARP_M/32)*(WARP_N/32)][4];
        int8_kvcache_qk_gemm_prefetch_v_3stage<kHeadDim, kHeadDimV, kBlockM, WARP_N, kBlockK_int8, kBlockK, WARP_M, WARP_N, WARP_NUM, STAGES, M_MMAC_COUNT, Element, Element_k, ElementAccum>(gK, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, kcache_seqlen_stride, kcache_seqlen_stride, warp_seqkv_limit);
        vec4_Accum<ElementAccum> s_reg_fp32[(WARP_M/32)*(WARP_N/32)][4];
    
        // 将 s_reg 的值转换为 fp32
        #pragma unroll
        for (int min_tile_n=0; min_tile_n<2; min_tile_n++) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int scales_k_offset = scales_k_global_offset + pre_scales_k_offset[min_tile_n][k];
                scales_v[min_tile_n][k] = scales_v_ptr[scales_k_offset];
                #pragma unroll
                for (int i = 0; i < (WARP_M/32)*(WARP_N/32); ++i) {
                    #pragma unroll
                    for (int min_tile_m=0; min_tile_m<M_MMAC_COUNT; min_tile_m++) {
                        s_reg_fp32[i][min_tile_n*2 + min_tile_m].f32[k] = static_cast<float>(s_reg[i][min_tile_n*2 + min_tile_m].int32[k]) * scales_q[min_tile_m] * scales_k[min_tile_n][k];
                    }
                }
            }
        }
        // __syncthreads();
        // print_qk(m_block, bidb, bidh);

        if constexpr (Has_alibi) {
            int8_kvcache_apply_alibi<vec4_Accum<ElementAccum>, WARP_M, WARP_N, M_MMAC_COUNT>(s_reg_fp32, warp_offset_in_seqkv, binfo.actual_seqlen_k, (m_block * kBlockM), binfo.actual_seqlen_q, gAlibi);
        }

        if constexpr (true) {
            int8_kvcache_apply_mask<vec4_Accum<ElementAccum>, WARP_M, WARP_N, M_MMAC_COUNT>(s_reg_fp32, warp_seqkv_limit);
        }

        if constexpr (Is_local) {
            int8_kvcache_apply_mask_local</*HasWSLeft=*/Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN, M_MMAC_COUNT>(s_reg_fp32, warp_offset_in_seqkv, binfo.actual_seqlen_k,
                                                                                (m_block * kBlockM + WARP_ID * WARP_M),
                                                                                    binfo.actual_seqlen_q, params.window_size_left,
                                                                                    params.window_size_right);
        }

        int8_kvcache_softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, ElementAccum, kHeadDim, kBlockK, WARP_M, WARP_N, WARP_NUM, M_MMAC_COUNT>(s_reg_fp32, scores_max, scores_sum, acc_o, max_lds, WARP_ID, params.scale_softmax_log2);

        if constexpr (Is_dropout and Is_training) {
            warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
            int8_kvcache_apply_dropout<vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, M_MMAC_COUNT>(s_reg_fp32, binfo.actual_seqlen_k - warp_seqkv_limit, 0, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
        }

        union_vec2_f16x2<Element> p_reg[(WARP_M/32)*(WARP_N/32)][4];
        // convertType: float2half
        int8_kvcache_convert_pk_type<WARP_M, WARP_N, M_MMAC_COUNT, Element, ElementAccum>(p_reg, s_reg_fp32);

        int8_kvcache_pv_gemm_prefetch_k_3stage<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, WARP_N, STAGES, WARP_NUM, M_MMAC_COUNT, Element, Element_k, ElementAccum>(gV, gK, v_lds, k_lds, scales_v, p_reg, acc_o, WARP_ID, kcache_seqlen_stride, warp_seqkv_limit);

    }
    __syncthreads();

    if constexpr (WARP_NUM > 1) {
        // reduce acc_o across 4 waves
        kvcache_acco_reduce<REUSE_KV_TIMES, kHeadDimV, kBlockK, WARP_M, M_MMAC_COUNT, WARP_NUM, ElementAccum>(acc_o, acc_o_lds, params.seqlen_q, WARP_ID, lane_id);
    }

    /**********************************************************************************************************************************/

    kvcache_epilugue_rescale_acco<kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, M_MMAC_COUNT, ElementAccum>(acc_o, scores_sum);

    kvcache_epilogue_store_max_sum<Split, false, WARP_M / 32, M_MMAC_COUNT, ElementAccum>(
        scores_max, scores_sum, scores_max_ptr, scores_sum_ptr, params.scale_softmax, WARP_ID, threadIdx.x, lane_id, 0, binfo.actual_seqlen_q - m_block * kBlockM);

    /**************************************************************************************************************************************/
    kvcache_epilogue_store_output<Params, kHeadDimV, kHeadDimV, Split, false/*Is_16x32*/, typename Kernel_traits::SplitkvAccumType, ElementAccum, kBlockM, kBlockK, WARP_NUM, kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, M_MMAC_COUNT>(
        acc_o, params, bidb, bidh, m_block, split_id, 0, WARP_ID, lane_id);
}


template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_K, bool Return_softmax, bool Has_alibi, bool Is_GQA, bool Is_softcap, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, bool Append_KV, typename Params>
inline __device__ void compute_attn_splitkv_int8(const Params &params) {
    // block id in sequence dimension
    const int m_block = blockIdx.x;

    // The block index for the head.
    const int bidh = Split ? blockIdx.z % params.h : blockIdx.y; // batch x num_head, num_head first

    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.z;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id     = __builtin_amdgcn_readfirstlane(warp_id_vec);
    flash::compute_attn_mha_1rowblock_splitkv_int8<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_K, Return_softmax, Has_alibi, Split, M_MMAC_COUNT, REUSE_KV_TIMES, Flash_fwd_params>(params, bidb, bidh, warp_id);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// MLS-based FP8 Paged Attention, >= gfx938
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int REUSE_KV_TIMES, int K_LOOP_COUNT, int K_WARP_COUNT, int M_WARP_COUNT, int M_MMAC_COUNT, int WARP_NUM, typename ElementAccum>
__forceinline__ __device__ void fp8_kvcache_acco_reduce_compact_gfx938(
        vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4],
        ElementAccum* acc_o_lds,
        int seqlen_q,
        int warp_id,
        int lane_id) {
    constexpr int kReduceBlockK = 32;
    constexpr int kReduceRows = M_WARP_COUNT * M_MMAC_COUNT * 16;
    const int q_seq_idx = lane_id & 15;
    const int lane_dim_offset = (lane_id >> 4) * 4;
    const int even_reuse_kv_times = (REUSE_KV_TIMES > 0) ? ((REUSE_KV_TIMES + 1) / 2) * 2 : ((seqlen_q + 1) / 2) * 2;
    const bool is_valid_q_lane = q_seq_idx < even_reuse_kv_times;

    #pragma unroll
    for (int h_idx = 0; h_idx < K_LOOP_COUNT; ++h_idx) {
        #pragma unroll
        for (int k_idx = 0; k_idx < K_WARP_COUNT; ++k_idx) {
            if (is_valid_q_lane) {
                #pragma unroll
                for (int warp_m_idx = 0; warp_m_idx < M_WARP_COUNT; ++warp_m_idx) {
                    #pragma unroll
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        #pragma unroll
                        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            const int row_idx = warp_m_idx * M_MMAC_COUNT * 16 + min_tile_m * 16 + q_seq_idx;
                            const int lds_offset = (warp_id * kReduceRows + row_idx) * kReduceBlockK
                                                 + min_tile_n * 16 + lane_dim_offset;
                            const int tile_32x32_id = h_idx * M_WARP_COUNT * K_WARP_COUNT
                                                     + k_idx * M_WARP_COUNT + warp_m_idx;
                            *(vec4_fp32*)(acc_o_lds + lds_offset) = acc_o[tile_32x32_id][min_tile_n * 2 + min_tile_m].f32;
                        }
                    }
                }
            }
            __syncthreads();

            if constexpr (WARP_NUM > 1) {
                if (warp_id == 0) {
                    if (is_valid_q_lane) {
                        #pragma unroll
                        for (int warp_m_idx = 0; warp_m_idx < M_WARP_COUNT; ++warp_m_idx) {
                            #pragma unroll
                            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                                #pragma unroll
                                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                                    #pragma unroll
                                    for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                                        const int row_idx = warp_m_idx * M_MMAC_COUNT * 16 + min_tile_m * 16 + q_seq_idx;
                                        const int lds_offset = row_idx * kReduceBlockK
                                                             + min_tile_n * 16 + lane_dim_offset + vec_idx;
                                        ElementAccum acc_tmp = acc_o_lds[lds_offset];
                                        #pragma unroll
                                        for (int loop = 1; loop < WARP_NUM; ++loop) {
                                            acc_tmp += acc_o_lds[lds_offset + loop * kReduceRows * kReduceBlockK];
                                        }
                                        acc_o_lds[lds_offset] = acc_tmp;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();

            if (is_valid_q_lane) {
                #pragma unroll
                for (int warp_m_idx = 0; warp_m_idx < M_WARP_COUNT; ++warp_m_idx) {
                    #pragma unroll
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        #pragma unroll
                        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                            const int row_idx = warp_m_idx * M_MMAC_COUNT * 16 + min_tile_m * 16 + q_seq_idx;
                            const int lds_offset = row_idx * kReduceBlockK
                                                 + min_tile_n * 16 + lane_dim_offset;
                            const int tile_32x32_id = h_idx * M_WARP_COUNT * K_WARP_COUNT
                                                     + k_idx * M_WARP_COUNT + warp_m_idx;
                            acc_o[tile_32x32_id][min_tile_n * 2 + min_tile_m].f32 = *(vec4_fp32*)(acc_o_lds + lds_offset);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <typename DataType, int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT>
inline __device__ void fp8_kvcache_apply_mask_local_causal_gfx938(
        DataType tensor[M_WARP_COUNT * N_WARP_COUNT][4], const int col_idx_offset_,
        const int max_seqlen_k, const int row_idx_offset_, const int max_seqlen_q,
        const int ngroups, const int window_size_left, const int window_size_right) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15);
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4) * 8;
    #pragma unroll
    for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m * 16;
            const int logical_row = row_idx / ngroups;
            const int logical_q = max_seqlen_q / ngroups;
            const int col_idx_limit_left = max(0, logical_row + max_seqlen_k - logical_q - window_size_left);
            const int col_idx_limit_right = min(max_seqlen_k, logical_row + max_seqlen_k - logical_q + window_size_right);
            #pragma unroll
            for (int ni = 0; ni < N_WARP_COUNT; ++ni)  {
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n * 4;
                    #pragma unroll
                    for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx;
                        tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx] =
                            (col_idx < col_idx_limit_left || col_idx > col_idx_limit_right)
                                ? -INFINITY
                                : tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                    }
                }
            }
        }
    }
}

template<int kHeadDim, int kBlockM, int WARP_M, int M_MMAC_COUNT, typename Element>
__forceinline__ __device__ void fp8_mha_prefetch_q_to_vgpr_gfx938(
        vec4_uint q_addr,
        Element* q_lds,
        union_vec16_fp8 q_reg[M_MMAC_COUNT][kHeadDim / 64],
        int warp_id,
        int query_seqlen_stride,
        int max_seq_q_offset) {
    static_assert(kHeadDim == 128 || kHeadDim == 256);
    static_assert(WARP_M == 32);

    vec4_uint q_srsrc;
    q_srsrc[1] = q_addr[1];
    q_srsrc[2] = query_seqlen_stride;

    #pragma unroll
    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
        #pragma unroll
        for (int k_loop = 0; k_loop < kHeadDim / 128; ++k_loop) {
            if (warp_id == min_tile_m) {
                const int q_row_base = min_tile_m * 16;
                const int valid_rows = max_seq_q_offset - q_row_base;
                const int safe_q_row_base = valid_rows <= 0 ? 0 : q_row_base;
                const int nm_filter = inline_min_max<0, 16>(16 - valid_rows);
                q_srsrc[3] = valid_rows >= 16 ? 0 : (nm_filter << 8);

                const int64_t row_offset_bytes = int64_t(safe_q_row_base) * int64_t(query_seqlen_stride) * sizeof(Element);
                const int64_t dim_offset_bytes = int64_t(k_loop) * 128 * sizeof(Element);
                *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_addr + row_offset_bytes + dim_offset_bytes);

                const int lds_offset_bytes = (min_tile_m * (kHeadDim / 128) + k_loop) * 16 * 128 * sizeof(Element);
                inline_matrix_load_128x16_b8_lds_trans<0, 1>(q_lds, q_srsrc, lds_offset_bytes, 0);
            }
        }
    }
    flash::wait_buffer_data_arrived<true/*sync*/>(0);

    #pragma unroll
    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
        #pragma unroll
        for (int k_loop = 0; k_loop < kHeadDim / 128; ++k_loop) {
            const int lds_offset_bytes = (min_tile_m * (kHeadDim / 128) + k_loop) * 16 * 128 * sizeof(Element);
            const int q_lds_load_offset = reinterpret_cast<size_t>(q_lds) + lds_offset_bytes;
            DS_READ_MATRIX_64x16_B8(q_lds_load_offset,        q_reg[min_tile_m][k_loop * 2 + 0].i32x4, true/*transpose*/)
            DS_READ_MATRIX_64x16_B8(q_lds_load_offset + 1024, q_reg[min_tile_m][k_loop * 2 + 1].i32x4, true/*transpose*/)
            flash::wait_lds_data_arrived<true/*sync*/>(0);
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

template<int kBlockM, int WARP_M, int M_MMAC_COUNT, typename Element>
__forceinline__ __device__ void fp8_mha_prefetch_q_to_vgpr_hdim192_gfx938(
        vec4_uint q_addr,
        Element* q_lds,
        union_vec16_fp8 q_reg[M_MMAC_COUNT][3],
        int warp_id,
        int query_seqlen_stride,
        int max_seq_q_offset) {
    static_assert(WARP_M == 32);
    constexpr int kLoadBytes = 16 * 128 * sizeof(Element);

    vec4_uint q_srsrc;
    q_srsrc[1] = q_addr[1];
    q_srsrc[2] = query_seqlen_stride;

    #pragma unroll
    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
        if (warp_id == min_tile_m) {
            const int q_row_base = min_tile_m * 16;
            const int valid_rows = max_seq_q_offset - q_row_base;
            const int safe_q_row_base = valid_rows <= 0 ? 0 : q_row_base;
            const int nm_filter = inline_min_max<0, 16>(16 - valid_rows);
            q_srsrc[3] = valid_rows >= 16 ? 0 : (nm_filter << 8);

            const int64_t row_offset_bytes = int64_t(safe_q_row_base) * int64_t(query_seqlen_stride) * sizeof(Element);

            *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_addr + row_offset_bytes);
            const int q_lds_first_offset = (min_tile_m * 2 + 0) * kLoadBytes;
            inline_matrix_load_128x16_b8_lds_trans<0, 1>(q_lds, q_srsrc, q_lds_first_offset, 0);

            *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_addr + row_offset_bytes + 64 * sizeof(Element));
            const int q_lds_tail_offset = (min_tile_m * 2 + 1) * kLoadBytes;
            inline_matrix_load_128x16_b8_lds_trans<0, 1>(q_lds, q_srsrc, q_lds_tail_offset, 0);
        }
    }
    flash::wait_buffer_data_arrived<true/*sync*/>(0);

    #pragma unroll
    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
        const int q_lds_first_load = reinterpret_cast<size_t>(q_lds) + (min_tile_m * 2 + 0) * kLoadBytes;
        DS_READ_MATRIX_64x16_B8(q_lds_first_load,        q_reg[min_tile_m][0].i32x4, true/*transpose*/)
        DS_READ_MATRIX_64x16_B8(q_lds_first_load + 1024, q_reg[min_tile_m][1].i32x4, true/*transpose*/)
        flash::wait_lds_data_arrived<true/*sync*/>(0);

        const int q_lds_tail_load = reinterpret_cast<size_t>(q_lds) + (min_tile_m * 2 + 1) * kLoadBytes;
        DS_READ_MATRIX_64x16_B8(q_lds_tail_load + 1024, q_reg[min_tile_m][2].i32x4, true/*transpose*/)
        flash::wait_lds_data_arrived<true/*sync*/>(0);
    }
    __builtin_amdgcn_sched_barrier(0);
}


template<typename Kernel_traits, bool Is_causal, bool Is_Varlen, bool Split, bool Is_local, int M_MMAC_COUNT, int REUSE_KV_TIMES, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv_fp8_gfx938(const Params &params, const int bidb, const int bidh, const int warp_id) {
    using Element          = fp8_e4m3;
    using ElementAccum     = typename Kernel_traits::ElementAccum;
    using SplitkvAccumType = typename Kernel_traits::SplitkvAccumType;
    constexpr int kBlockM  = Kernel_traits::kBlockM;
    constexpr int kBlockN  = Kernel_traits::kBlockN;
    constexpr int kBlockK  = Kernel_traits::kBlockK;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int WARP_M   = Kernel_traits::kWaveM;
    constexpr int WARP_N   = Kernel_traits::kWaveN;
    constexpr int STAGES   = Kernel_traits::STAGES;
    constexpr int WARP_NUM = kBlockN / WARP_N;
    constexpr int kHeadDimVSplit = kHeadDimV / HEADDIM_V_SPLIT;

    static_assert(kBlockK == 64);
    static_assert(kHeadDim == 128 || kHeadDim == 256 || (kHeadDim == 192 && kHeadDimV == 128));
    static_assert(kHeadDimVSplit == 128);

    flash::SafeDecodeBlockInfo binfo;
    binfo.set_params<Params, /*Is_Q_varlen=*/Is_Varlen, /*Is_K_Cumulative=*/false>(params, bidb);

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
                ? binfo.actual_seqlen_k - split_id * partition_size : partition_size;
            binfo.actual_seqlen_k = (split_id >= num_splits) ? 0 : binfo.actual_seqlen_k;
            if (split_id >= num_splits) return;
        }
    }

    int block_x = blockIdx.x;
    const int m_block = block_x / HEADDIM_V_SPLIT;
    const int headdim_split_id = block_x & (HEADDIM_V_SPLIT - 1);

    int ngroups = 1;
    int actual_seqlen_q = binfo.actual_seqlen_q;
    if constexpr (Is_Varlen) {
        ngroups = params.ngroups;
        actual_seqlen_q = binfo.actual_seqlen_q * ngroups;
    }
    if (m_block * kBlockM >= actual_seqlen_q || binfo.actual_seqlen_k <= 0) return;

    extern __shared__ Element fp8_smem[];
    constexpr int q_smem_bytes = STAGES * kBlockM * kBlockK * sizeof(Element);
    constexpr int kv_smem_bytes = STAGES * kBlockK * WARP_N * sizeof(Element) * WARP_NUM;
    constexpr int gemm_smem_bytes = q_smem_bytes > kv_smem_bytes ? q_smem_bytes : kv_smem_bytes;
    Element* q_lds = reinterpret_cast<Element*>(fp8_smem);
    Element* k_lds = reinterpret_cast<Element*>(fp8_smem);
    Element* v_lds = k_lds;
    ElementAccum* acc_o_lds = reinterpret_cast<ElementAccum*>(fp8_smem);
    ElementAccum* max_lds = reinterpret_cast<ElementAccum*>(
        reinterpret_cast<char*>(fp8_smem) + gemm_smem_bytes);

    const int query_seqlen_stride = params.q_row_stride;
    const int kcache_seqlen_stride = params.k_row_stride;
    const int vcache_seqlen_stride = params.v_row_stride;

    int n_block_min = 0;
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_local) {
        const int q_row_start = m_block * kBlockM;
        const int q_row_end = min(actual_seqlen_q, (m_block + 1) * kBlockM) - 1;
        const int logical_q = Is_Varlen ? actual_seqlen_q / ngroups : actual_seqlen_q;
        const int logical_row_start = Is_Varlen ? q_row_start / ngroups : q_row_start;
        const int logical_row_end = Is_Varlen ? q_row_end / ngroups : q_row_end;
        const int split_seqlen_start = Split ? split_id * partition_size : 0;
        const int local_left = max(0, logical_row_start + original_actual_seqlen_k - logical_q - params.window_size_left);
        const int local_right = min(original_actual_seqlen_k, logical_row_end + original_actual_seqlen_k - logical_q + params.window_size_right + 1);
        const int split_local_left = local_left - split_seqlen_start;
        const int split_local_right = local_right - split_seqlen_start;
        const int n_block_count = n_block_max;
        const int raw_n_block_min = max(0, split_local_left / kBlockN);
        const int raw_n_block_max = ceil_div(max(0, split_local_right), kBlockN);
        n_block_min = min(max(raw_n_block_min, 0), max(0, n_block_count - 1));
        n_block_max = min(max(raw_n_block_max, n_block_min + 1), n_block_count);
    }

    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    const int this_split_seqlen_start = Split ? split_id * partition_size : 0;
    block_table = block_table + (Split ? ceil_div(this_split_seqlen_start, page_block_size) : 0);
    const int block_table_idx = n_block_min * kBlockN / page_block_size;
    const int block_table_offset = n_block_min * kBlockN - block_table_idx * page_block_size;
    const int64_t row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride)
        + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const int64_t row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride)
        + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const int64_t row_offset_q = Is_Varlen
        ? binfo.sum_s_q * ngroups * int64_t(query_seqlen_stride) + bidh * params.q_head_stride + m_block * kBlockM * int64_t(query_seqlen_stride)
        : bidb * int64_t(params.q_batch_stride) + bidh * params.q_head_stride + m_block * kBlockM * int64_t(query_seqlen_stride);

    auto q_addr = prepare_for_buffer_load<kHeadDim, Element, false>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
    auto k_addr = prepare_for_buffer_load<kHeadDim, Element, false>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto v_addr = prepare_for_buffer_load<kHeadDimV, Element, false>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v + headdim_split_id * kHeadDimVSplit);

    const ElementAccum q_descale = params.q_descale_ptr[0];
    const ElementAccum k_descale = params.k_descale_ptr[0];
    const ElementAccum v_descale = params.v_descale_ptr[0];
    __float2 qk_descale = {q_descale * k_descale, q_descale * k_descale};

    int row_offset_lse;
    ElementAccum *scores_sum_ptr = nullptr;
    ElementAccum *scores_max_ptr = nullptr;
    ElementAccum *softmax_lse_ptr = nullptr;
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

    constexpr int M_WARP_COUNT = WARP_M / 32;
    constexpr int K_WARP_COUNT = kBlockK / 32;
    constexpr int N_WARP_COUNT = WARP_N / 32;
    constexpr int K_LOOP_COUNT = kHeadDimVSplit / kBlockK;

    vec2_Accum<ElementAccum> scores_max[M_WARP_COUNT];
    vec2_Accum<ElementAccum> scores_sum[M_WARP_COUNT];
    vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4];

    union_vec16_fp8 q_reg[M_MMAC_COUNT][kHeadDim / 64];
    attention_initialize<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(scores_max, scores_sum, acc_o);
    if constexpr (kHeadDim == 192 && kHeadDimV == 128) {
        fp8_mha_prefetch_q_to_vgpr_hdim192_gfx938<kBlockM, WARP_M, M_MMAC_COUNT, Element>(
            q_addr, q_lds, q_reg, warp_id, query_seqlen_stride, actual_seqlen_q - m_block * kBlockM);
    } else {
        fp8_mha_prefetch_q_to_vgpr_gfx938<kHeadDim, kBlockM, WARP_M, M_MMAC_COUNT, Element>(
            q_addr, q_lds, q_reg, warp_id, query_seqlen_stride, actual_seqlen_q - m_block * kBlockM);
    }

    int n_block_loop = n_block_min;
    constexpr bool PrefetchK = true;
    if constexpr (PrefetchK) {
        int warp_seqkv_limit = binfo.actual_seqlen_k - n_block_min * kBlockN;
        fp8_kvcache_prefetch_k_gfx938<WARP_NUM, Element>(k_addr, k_lds, warp_id, kcache_seqlen_stride, warp_seqkv_limit);
    }
    for (; n_block_loop < n_block_max; ++n_block_loop) {
        const int warp_offset_in_seqkv = n_block_loop * kBlockN + warp_id * WARP_N;
        const int warp_seqkv_limit = binfo.actual_seqlen_k - n_block_loop * kBlockN;
        constexpr bool PrefetchVInQK = (kHeadDim == 128 && K_LOOP_COUNT == 2);

        if constexpr (!PrefetchK) {
            fp8_kvcache_prefetch_k_gfx938<WARP_NUM, Element>(k_addr, k_lds, warp_id, kcache_seqlen_stride, warp_seqkv_limit);
        }

        vec4_Accum<ElementAccum> s_reg[M_WARP_COUNT * N_WARP_COUNT][4];
        fp8_kvcache_qk_gemm_gfx938<PrefetchVInQK, K_LOOP_COUNT, kHeadDim, kBlockK, WARP_M, WARP_N, WARP_NUM, M_MMAC_COUNT, Element, ElementAccum>(
            k_addr, v_addr, k_lds, v_lds, q_reg, s_reg, warp_id, kcache_seqlen_stride, vcache_seqlen_stride, warp_seqkv_limit);
        if constexpr (!PrefetchVInQK) {
            fp8_kvcache_prefetch_v_gfx938<K_LOOP_COUNT, kBlockK, WARP_NUM, Element>(
                v_addr, v_lds, warp_id, vcache_seqlen_stride, warp_seqkv_limit);
        }

        fp8_kvcache_apply_descale_gfx938<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, qk_descale);

        if constexpr (Is_causal) {
            if constexpr (Is_Varlen) {
                if constexpr (Is_local) {
                    fp8_kvcache_apply_mask_local_causal_gfx938<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(
                        s_reg, warp_offset_in_seqkv + this_split_seqlen_start, original_actual_seqlen_k, m_block * kBlockM, actual_seqlen_q, ngroups, params.window_size_left, params.window_size_right);
                } else {
                    kvcache_apply_mask_causal_gfx938<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(
                        s_reg, warp_offset_in_seqkv + this_split_seqlen_start, original_actual_seqlen_k, m_block * kBlockM, actual_seqlen_q, ngroups);
                }
            } else {
                kvcache_apply_mask_causal_gfx938_mtp<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(
                    s_reg, warp_offset_in_seqkv + this_split_seqlen_start, original_actual_seqlen_k, m_block * kBlockM, actual_seqlen_q, params.mtp, params.layout);
            }
        } else {
            kvcache_apply_mask_gfx938<vec4_Accum<ElementAccum>, M_WARP_COUNT, N_WARP_COUNT, M_MMAC_COUNT>(s_reg, warp_seqkv_limit, warp_id * WARP_N);
        }

        mla_softmax_rescale_o<Is_causal || Is_local, ElementAccum, K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, N_WARP_COUNT, WARP_NUM, M_MMAC_COUNT>(
            s_reg, scores_max, scores_sum, acc_o, max_lds, warp_id, params.scale_softmax_log2);

        union_vec32_fp8 p_reg[M_MMAC_COUNT];
        fp8_kvcache_cvt_f32_to_fp8_gfx938<M_MMAC_COUNT, Element, ElementAccum>(p_reg, s_reg);

        const int block_table_idx_cur = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff = block_table_offset_next - block_table_offset_cur;
        const int64_t k_addr_offset = (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);

        fp8_kvcache_pv_gemm_fp8_prefetch_k_gfx938<PrefetchK, K_LOOP_COUNT, kBlockK, kBlockN, M_WARP_COUNT, K_WARP_COUNT, WARP_NUM, M_MMAC_COUNT, Element, ElementAccum>(
            v_addr, k_addr, v_lds, k_lds, p_reg, acc_o, warp_id, kcache_seqlen_stride, vcache_seqlen_stride, warp_seqkv_limit, k_addr_offset);

        *(int64_t*)&v_addr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);
    }
    if constexpr (PrefetchK) {
        flash::wait_buffer_data_arrived<false/*sync*/>(0);
    }
    flash::wait_lds_data_arrived<true/*sync*/>(0);

    const int thread_id = threadIdx.x;
    const int lane_id = thread_id & 63;
    if constexpr (WARP_NUM > 1) {
        fp8_kvcache_acco_reduce_compact_gfx938<REUSE_KV_TIMES, K_LOOP_COUNT, K_WARP_COUNT, M_WARP_COUNT, M_MMAC_COUNT, WARP_NUM, ElementAccum>(
            acc_o, acc_o_lds, params.seqlen_q, warp_id, lane_id);
    }

    if (params.s_aux_ptr != nullptr && split_id == 0) {
        kvcache_apply_attention_sink<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
            acc_o, scores_max, scores_sum, params.s_aux_ptr, params.s_aux_type,
            bidh, ngroups, m_block, kBlockM, lane_id, params.scale_softmax);
    }
    fp8_kvcache_epilogue_rescale_acco_gfx938<K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(acc_o, scores_sum, v_descale);

    if constexpr (Is_Varlen) {
        kvcache_epilogue_store_softmax_lse<Is_Varlen, true, M_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
            scores_max, scores_sum, softmax_lse_ptr, params.scale_softmax, warp_id, thread_id, lane_id, headdim_split_id, actual_seqlen_q - m_block * kBlockM, params.total_q, params.ngroups);
        const int64_t row_offset_o = binfo.sum_s_q * ngroups * int64_t(params.o_row_stride) + bidh * ngroups * params.o_head_stride + headdim_split_id * kHeadDimVSplit + m_block * kBlockM * int64_t(params.o_row_stride);
        kvcache_varlen_epilogue_store_output_gfx938<Params, kHeadDimV, kHeadDimVSplit, Split, SplitkvAccumType, ElementAccum, kBlockM, kBlockK, WARP_NUM, K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT>(
            acc_o, params, row_offset_o, actual_seqlen_q - m_block * kBlockM, bidb, bidh, m_block, split_id, headdim_split_id, warp_id, lane_id);
    } else {
        kvcache_epilogue_store_max_sum<Split, true/*Is_16x32*/, M_WARP_COUNT, M_MMAC_COUNT, ElementAccum>(
            scores_max, scores_sum, scores_max_ptr, scores_sum_ptr, params.scale_softmax, warp_id, thread_id, lane_id, headdim_split_id, actual_seqlen_q - m_block * kBlockM);
        kvcache_epilogue_store_output_gfx938<Params, kHeadDimV, kHeadDimVSplit, true/*alt*/, Split, SplitkvAccumType, ElementAccum, kBlockM, kBlockK, WARP_NUM, K_LOOP_COUNT, M_WARP_COUNT, K_WARP_COUNT, M_MMAC_COUNT>(
            acc_o, params, bidb, bidh, m_block, split_id, headdim_split_id, warp_id, lane_id);
    }
}


template<typename Kernel_traits, bool Is_causal, bool Is_Varlen, bool Split, bool Is_local, int M_MMAC_COUNT, int REUSE_KV_TIMES, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
inline __device__ void compute_attn_splitkv_fp8_gfx938(const Params &params) {

#if defined(__gfx938__)
    // The block index for the head.
    const int bidh = Split ? blockIdx.z % params.h : blockIdx.y; // batch x num_head, num_head first

    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.z;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id     = __builtin_amdgcn_readfirstlane(warp_id_vec);

    flash::compute_attn_1rowblock_splitkv_fp8_gfx938<Kernel_traits, Is_causal, Is_Varlen, Split, Is_local, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size * 128, Params>(params, bidb, bidh, warp_id);
#endif
}

template<typename Kernel_traits, bool Is_causal, bool Is_Varlen, bool Split, bool Is_local, int M_MMAC_COUNT, int REUSE_KV_TIMES, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
inline __device__ void compute_attn_splitkv_fp8_gfx938_hdim192_v128(const Params &params) {

#if defined(__gfx938__)
    static_assert(Kernel_traits::kHeadDim == 192 && Kernel_traits::kHeadDimV == 128);
    static_assert(HEADDIM_V_SPLIT == 1);

    const int bidh = Split ? blockIdx.z % params.h : blockIdx.y;
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.z;

    int warp_id_vec = threadIdx.x / 64;
    int warp_id     = __builtin_amdgcn_readfirstlane(warp_id_vec);

    flash::compute_attn_1rowblock_splitkv_fp8_gfx938<Kernel_traits, Is_causal, Is_Varlen, Split, Is_local, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size * 128, Params>(params, bidb, bidh, warp_id);
#endif
}
} // namespace flash
