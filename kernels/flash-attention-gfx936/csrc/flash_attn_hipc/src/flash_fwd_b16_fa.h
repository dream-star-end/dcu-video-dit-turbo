#pragma once
#include "block_info.h"
#include "kernel_traits.h"
#include "fwd/qk_gemm_prefetch_v.h"
#include "fwd/pv_gemm_prefetch_k.h"
#include "fwd/softmax.h"
#include "fwd/fwd_epilogue.h"
#include "fwd/fwd_prologue.h"

namespace flash {

template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Return_softmax, bool Has_alibi, int Layout, typename Params, typename OffsetT = std::conditional_t<Is_even_MN, int32_t, int64_t>, bool IsVarlen = !Is_even_MN>
inline __device__ void compute_attn_mha_1rowblock(const Params &params, const int bidb, const int __bidh, const int m_block, const int WARP_ID) {
    using Element           = typename Kernel_traits::Element;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    using index_t           = OffsetT;
    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kBlockK   = Kernel_traits::kBlockK;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimVSplit;
    constexpr int kNWarps   = Kernel_traits::kNWarps;
    constexpr int WARP_M    = Kernel_traits::kWaveM;
    constexpr int WARP_N    = Kernel_traits::kWaveN;
    constexpr int STAGES    = Kernel_traits::STAGES;
    constexpr int WARP_NUM  = kBlockM / WARP_M;
    constexpr int SplitD    = Kernel_traits::SplitD;
    constexpr int kHeadDimVOrigin = Kernel_traits::kHeadDimV;

    // 获取 splitD 结果
    const int bidh = __bidh / SplitD;

    // 获取当前 TG 处理的任务大小
    const flash::BlockInfo</*Varlen=*/IsVarlen> binfo(params, bidb);

    // 处理边界
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0) return;

    // 分配 lds
    extern __shared__ Element smem[];
    Element* q_lds = (Element*)&(smem);
    Element* v_lds = (Element*)&(smem);
    Element* k_lds = v_lds + (2 * STAGES * 32 * kBlockK);

    // 计算当前任务沿着 seqlenKV 方向的 block 起始位置和终止位置
    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, flash::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // 计算数据跨度
    int seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, seqlen_o_stride;
    index_t row_offset_q, row_offset_k, row_offset_v, row_offset_o;
    int row_offset_lse;
    int headdim_split_id = 0;
    fwd_prologue_compute_offset<Layout, kBlockM, kBlockN, kHeadDim, kHeadDimV, kHeadDimVOrigin, SplitD, Is_even_MN, false/*Is_PaddingMask*/, Params, decltype(binfo), decltype(row_offset_q), IsVarlen>(
        seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, seqlen_o_stride, row_offset_q, row_offset_k, row_offset_v, row_offset_o, row_offset_lse,
        headdim_split_id, bidb, bidh, __bidh, m_block, n_block_min, binfo, params
    );
    #if 0
    if (int(threadIdx.x) == 0) {
        printf("bidb: %d | bidh: %d | actual_seqlen_q: %d | actual_seqlen_k: %d | n_block_max: %d | row_offset_q: %d | row_offset_k: %d | row_offset_v: %d | row_offset_o: %d | seqlen_q_stride: %d | seqlen_k_stride: %d | seqlen_v_stride: %d\n",
            bidb, bidh, binfo.actual_seqlen_q, binfo.actual_seqlen_k, n_block_max, row_offset_q, row_offset_k, row_offset_v, row_offset_o, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride);
    }
    #endif
    // 根据起始数据偏移量准备 Q/K/V 的 buffer resource 寄存器
    auto gQ = prepare_for_buffer_load<kHeadDim>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
    auto gK = prepare_for_buffer_load<kHeadDim>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto gV = prepare_for_buffer_load<kHeadDimV>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v);

    // attention 变体: Alibi
    float gAlibi;
    if constexpr (Has_alibi) {
        gAlibi = reinterpret_cast<ElementAccum*>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    }

    // attention 插件: Dropout
    unsigned long long rand_seed, rand_offset;
    uint32_t p_dropout_in_8bits_value;
    union_vec2_uint warp_idx_for_dropout;
    if constexpr (Is_dropout and Is_training) {
        rand_seed                = params.rand_seed;
        rand_offset              = params.rand_offset + ((bidb * params.h + bidh) << 6) + (threadIdx.x & 63); /* 参考官方写法 offset(offset + (bid * nheads + hid) * 32 + tid % 32) */
        p_dropout_in_8bits_value = params.p_dropout_in_uint8_t & 0xffffffff; /*hcu 不支持 16bit 和 8bit 的比较指令*/
        warp_idx_for_dropout.u32.x = 1 * m_block * (kBlockM / 32)/*前面几个 block 累积的 warp 数目, 这里不直接填 WARP_M, 参照 NV 的写法*/ + WARP_ID/*当前 block 内的 warp id*/;
        // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might exit early and no one saves the rng states.
        if (Is_training and m_block == 0 and bidb == 0 and bidh == 0 and threadIdx.x == 0) {
            params.rng_state[0] = rand_seed;
            params.rng_state[1] = rand_offset;
        }
    }

    // 预取 Q 的数据到寄存器
    vec2_Element<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2][4]; // ds_read mini size is 32 * 32,2 is seq, 4 is head dim
    Is_even_MN
    ? prefetch_q_to_vgpr<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(gQ, q_lds, q_reg, WARP_ID, seqlen_q_stride)
    : prefetch_q_to_vgpr<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(gQ, q_lds, q_reg, WARP_ID, seqlen_q_stride, (binfo.actual_seqlen_q - m_block * kBlockM));

    // apply causal mask 的步骤和 no causal mask 的步骤分开算
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : flash::ceil_div(kBlockM, kBlockN) + (Is_even_MN ? 0 : 1);

    // 是否做 prefetch K, PV 结束后, prefetch K 有风险
    constexpr bool PREFETCH_K     = Is_even_MN and kHeadDim == 128 and kHeadDimV == 128;   // 简单场景下开启
    constexpr bool ALLOW_PREFETCH = (STAGES > 1); // 客观上决定是否开启 prefetch
    if constexpr (PREFETCH_K and ALLOW_PREFETCH) {
        if (n_block_min < (n_block_max - n_masking_steps)) {
            Is_even_MN
            ? prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride)
            : prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride, (binfo.actual_seqlen_k - n_block_min * kBlockN));
        }
    }

    /***************************************************************************************************************************/
    vec2_Accum<ElementAccum> scores_max[WARP_M / 32];
    vec2_Accum<ElementAccum> scores_sum[WARP_M / 32];
    vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4];
    attention_initialize<kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, 2/*M_MMAC_COUNT*/, ElementAccum>(scores_max, scores_sum, acc_o);
    /***************************************************************************************************************************/
    constexpr bool Aggressive = (kHeadDim == 128 or kHeadDim == 64);
    auto QK_GEMM_FUNC = Aggressive
        ? &qk_gemm_prefetch_v_headdim128<kHeadDim, kHeadDimV, kBlockM, kBlockN, kBlockK, WARP_M, WARP_N, STAGES, Element, ElementAccum, Is_even_MN>
        : &qk_gemm_prefetch_v<kHeadDim, kHeadDimV, kBlockM, kBlockN, kBlockK, WARP_M, WARP_N, STAGES, Element, ElementAccum, Is_even_MN>;

    auto PV_GEMM_FUNC = Aggressive
        ? &pv_gemm_prefetch_k_headdim128<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>
        : &pv_gemm_prefetch_k<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;
    // mask 循环中不需要做 prefetch K, 因此 prefetch K 固定为 false
    auto PV_GEMM_FUNC_IN_MASK = Aggressive
        ? &pv_gemm_prefetch_k_headdim128<false, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>
        : &pv_gemm_prefetch_k<false, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;

    // These are the iterations where we don't need masking on S
    for (int n_block_loop = n_block_min; n_block_loop < (n_block_max - n_masking_steps); ++n_block_loop) {

        const int seqlen_kv_limit = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        // c mini tile is 32 * 32
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (kBlockN / WARP_N)][4];

        if constexpr ((not PREFETCH_K) and ALLOW_PREFETCH) {
            Is_even_MN
            ? prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride)
            : prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride, seqlen_kv_limit);
        }

        Is_even_MN
        ? QK_GEMM_FUNC(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, seqlen_k_stride, seqlen_v_stride, 0)
        : QK_GEMM_FUNC(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, seqlen_k_stride, seqlen_v_stride, seqlen_kv_limit);

        if constexpr (Has_alibi) {
            apply_alibi<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, n_block_loop * kBlockN, binfo.actual_seqlen_k, m_block * kBlockM + WARP_ID * WARP_M, binfo.actual_seqlen_q, gAlibi);
        }

        if constexpr (Is_local) {
            apply_mask_local<Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, n_block_loop * kBlockN, binfo.actual_seqlen_k,
                                                                                    m_block * kBlockM + WARP_ID * WARP_M,
                                                                                    binfo.actual_seqlen_q, params.window_size_left,
                                                                                    params.window_size_right);
        }

        softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN, false/*IsInference*/>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        if constexpr (Return_softmax) {
            vec4_Accum<ElementAccum> s_reg_dmask[(WARP_M / 32) * (kBlockN / WARP_N)][4];
            #pragma unroll
            for (int idx = 0; idx < (WARP_M / 32) * (kBlockN / WARP_N); ++idx) {
                #pragma unroll
                for (int frag = 0; frag < 4; ++frag) {
                    s_reg_dmask[idx][frag] = s_reg[idx][frag];
                }
            }
            if constexpr (Is_dropout and Is_training) {
                warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
                apply_dropout<false, vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg, binfo.actual_seqlen_q, binfo.actual_seqlen_k, m_block * kBlockM + WARP_ID * WARP_M, n_block_loop * kBlockN, bidb * params.h + bidh, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
                apply_dropout<true, vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg_dmask, binfo.actual_seqlen_q, binfo.actual_seqlen_k, m_block * kBlockM + WARP_ID * WARP_M, n_block_loop * kBlockN, bidb * params.h + bidh, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
                __builtin_amdgcn_sched_barrier(0);
            }
            store_s_dmask_gfx936<vec4_Accum<ElementAccum>, Element, WARP_M, kBlockN, kBlockM, kBlockN, Is_even_MN>(
                params.p_ptr,
                s_reg_dmask,
                bidb,
                bidh,
                m_block,
                n_block_loop,
                WARP_ID,
                threadIdx.x & 63,
                params.h,
                params.seqlen_q_rounded,
                params.seqlen_k_rounded,
                binfo.actual_seqlen_q,
                binfo.actual_seqlen_k
            );
        } else if constexpr (Is_dropout and Is_training) {
            warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
            apply_dropout<false, vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg, binfo.actual_seqlen_q, binfo.actual_seqlen_k, m_block * kBlockM + WARP_ID * WARP_M, n_block_loop * kBlockN, bidb * params.h + bidh, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
            __builtin_amdgcn_sched_barrier(0);
        }

        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockN / 32)][4];
        // convertType: float2half
        convert_pk_type<WARP_M, kBlockN, Element, ElementAccum>(p_reg, s_reg);

        *(uint64_t*)&gK += kBlockN * seqlen_k_stride * sizeof(Element);

        Is_even_MN
        ? PV_GEMM_FUNC(gV, gK, v_lds, k_lds, p_reg, acc_o, WARP_ID, seqlen_k_stride, seqlen_v_stride, 0)
        : PV_GEMM_FUNC(gV, gK, v_lds, k_lds, p_reg, acc_o, WARP_ID, seqlen_k_stride, seqlen_v_stride, seqlen_kv_limit);

        // gV = gV + (-kBlockN * params.v_row_stride);
        *(uint64_t*)&gV += kBlockN * seqlen_v_stride * sizeof(Element);
    }

    // prefetch K 的话, 最后一次多取了一段 K, 为了不影响后续的操作, 需要同步等待
    if constexpr (PREFETCH_K) {
        buffer_load_lds_dwordx1_wait<0>();
    }

    /***************************************************************************************************************************/
    int n_block = max(n_block_max - n_masking_steps, n_block_min);

    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, n_block++) {

        const int seqlen_kv_limit = binfo.actual_seqlen_k - n_block * kBlockN;

        // c mini tile is 32 * 32
        vec4_Accum<ElementAccum> s_reg[(kBlockN / 32) * (WARP_M / 32)][4];

        if constexpr (ALLOW_PREFETCH) {
            Is_even_MN
            ? prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride)
            : prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride, seqlen_kv_limit);
        }

        Is_even_MN
        ? QK_GEMM_FUNC(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, seqlen_k_stride, seqlen_v_stride, 0)
        : QK_GEMM_FUNC(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, seqlen_k_stride, seqlen_v_stride, seqlen_kv_limit);

        *(uint64_t*)&gK += kBlockN * seqlen_k_stride * sizeof(Element);

        if constexpr (Has_alibi) {
            apply_alibi<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, n_block * kBlockN, binfo.actual_seqlen_k, m_block * kBlockM + WARP_ID * WARP_M, binfo.actual_seqlen_q, gAlibi);
        }

        if constexpr (!Is_causal && !Is_local) {
            if constexpr (!Is_even_MN) { apply_mask<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, seqlen_kv_limit); }
        } else {
            if constexpr (Is_causal) {
                apply_mask_causal<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, n_block * kBlockN, binfo.actual_seqlen_k,
                                                                                m_block * kBlockM + WARP_ID * WARP_M,
                                                                                binfo.actual_seqlen_q);
            }
            if constexpr (Is_local) {
                apply_mask_local<Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, n_block * kBlockN, binfo.actual_seqlen_k,
                                                                                m_block * kBlockM + WARP_ID * WARP_M,
                                                                                binfo.actual_seqlen_q, params.window_size_left,
                                                                                params.window_size_right);
            }
        }

#ifdef FA_FWD_ATTEN_DEBUG
        // Binarize s_reg: 0=masked(-inf), 1=unmasked, then save to p_ptr
        if constexpr (Return_softmax) {
            #pragma unroll
            for (int idx = 0; idx < (kBlockN/32)*(WARP_M/32); ++idx) {
                #pragma unroll
                for (int frag = 0; frag < 4; ++frag) {
                    #pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        s_reg[idx][frag].f32[v] = (s_reg[idx][frag].f32[v] <= -1e10) ? 0.0f : 1.0f;
                    }
                }
            }
            store_s_dmask_gfx936<vec4_Accum<ElementAccum>, Element, WARP_M, kBlockN, kBlockM, kBlockN, Is_even_MN>(
                params.p_ptr, s_reg, bidb, bidh, m_block, n_block, WARP_ID, threadIdx.x & 63,
                params.h, params.seqlen_q_rounded, params.seqlen_k_rounded,
                binfo.actual_seqlen_q, binfo.actual_seqlen_k);
        }
#endif

        // TODO: when we have key_padding_mask we'll need to Check_inf
        softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN, false/*IsInference*/>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        if constexpr (Return_softmax) {
            vec4_Accum<ElementAccum> s_reg_dmask[(kBlockN / 32) * (WARP_M / 32)][4];
            #pragma unroll
            for (int idx = 0; idx < (kBlockN / 32) * (WARP_M / 32); ++idx) {
                #pragma unroll
                for (int frag = 0; frag < 4; ++frag) {
                    s_reg_dmask[idx][frag] = s_reg[idx][frag];
                }
            }
            if constexpr (Is_dropout and Is_training) {
                warp_idx_for_dropout.u32.y = n_block * (kBlockN / WARP_N);
                apply_dropout<false, vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg, binfo.actual_seqlen_q, binfo.actual_seqlen_k, m_block * kBlockM + WARP_ID * WARP_M, n_block * kBlockN, bidb * params.h + bidh, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
                apply_dropout<true, vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg_dmask, binfo.actual_seqlen_q, binfo.actual_seqlen_k, m_block * kBlockM + WARP_ID * WARP_M, n_block * kBlockN, bidb * params.h + bidh, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
                __builtin_amdgcn_sched_barrier(0);
            }
#ifndef FA_FWD_ATTEN_DEBUG
            store_s_dmask_gfx936<vec4_Accum<ElementAccum>, Element, WARP_M, kBlockN, kBlockM, kBlockN, Is_even_MN>(
                params.p_ptr,
                s_reg_dmask,
                bidb,
                bidh,
                m_block,
                n_block,
                WARP_ID,
                threadIdx.x & 63,
                params.h,
                params.seqlen_q_rounded,
                params.seqlen_k_rounded,
                binfo.actual_seqlen_q,
                binfo.actual_seqlen_k
            );
#endif
        } else if constexpr (Is_dropout and Is_training) {
            warp_idx_for_dropout.u32.y = n_block * (kBlockN / WARP_N);
            apply_dropout<false, vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg, binfo.actual_seqlen_q, binfo.actual_seqlen_k, m_block * kBlockM + WARP_ID * WARP_M, n_block * kBlockN, bidb * params.h + bidh, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
            __builtin_amdgcn_sched_barrier(0);
        }

        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockN / 32)][4];
        // convertType: float2half
        convert_pk_type<WARP_M, kBlockN, Element, ElementAccum>(p_reg, s_reg);

        Is_even_MN
        ? PV_GEMM_FUNC_IN_MASK(gV, gK, v_lds, k_lds, p_reg, acc_o, WARP_ID, seqlen_k_stride, seqlen_v_stride, 0)
        : PV_GEMM_FUNC_IN_MASK(gV, gK, v_lds, k_lds, p_reg, acc_o, WARP_ID, seqlen_k_stride, seqlen_v_stride, seqlen_kv_limit);

        *(uint64_t*)&gV += kBlockN * seqlen_v_stride * sizeof(Element);
    }
    /**********************************************************************************************************************************/
    // Epilogue: rescale acco
    vec2_Accum<ElementAccum> lse[WARP_M / 32];
#ifndef H3_STANDARD_FWD_ONLY
    if (params.s_aux_ptr != nullptr) {
        const float sink_value = fwd_attention_sink_load(params.s_aux_ptr, params.s_aux_type, bidh);
        fwd_apply_attention_sink<WARP_M, kBlockK, kHeadDimV, ElementAccum>(
            acc_o, scores_max, scores_sum, params.scale_softmax, sink_value);
    }
#endif
#ifdef H3_STANDARD_FWD_ONLY
    fwd_epilugue_rescale_acco<WARP_M, kBlockK, kHeadDimV, Is_dropout && Is_training, ElementAccum, false>(acc_o, lse, scores_max, scores_sum, params.scale_softmax, params.rp_dropout);
#else
    fwd_epilugue_rescale_acco<WARP_M, kBlockK, kHeadDimV, Is_dropout && Is_training, ElementAccum>(acc_o, lse, scores_max, scores_sum, params.scale_softmax, params.rp_dropout);
#endif
    /**************************************************************************************************************************************/
    int lane_id = threadIdx.x & 63;
#ifndef H3_STANDARD_FWD_ONLY
    if (params.softmax_lse_ptr != nullptr) {
        fwd_epilogue_store_lse<WARP_M, Is_even_MN, SplitD, false/*Is_Interleaved*/, ElementAccum>(lse, params.softmax_lse_ptr, row_offset_lse, WARP_ID, lane_id, headdim_split_id, binfo.actual_seqlen_q - m_block * kBlockM);
    }
#endif
    /**************************************************************************************************************************************/
    Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;
    fwd_epilogue_store_output<kHeadDimV, kBlockM, kBlockK, WARP_M, Is_even_MN, true/*Is_Interleaved*/, false/*TcpSwizzle*/, Element, ElementAccum>(o_ptr, acc_o, m_block, WARP_ID, lane_id, seqlen_o_stride, binfo.actual_seqlen_q);
}


template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Return_softmax, bool Has_alibi, bool Is_GQA, int Layout, typename Params, typename OffsetT = std::conditional_t<Is_even_MN, int32_t, int64_t>, bool IsVarlen = !Is_even_MN>
inline __device__ void compute_attn(const Params &params) {

    constexpr bool Do_lpt = Is_causal and Is_GQA;

    const int bidh = Do_lpt ? blockIdx.x: blockIdx.y;

    const int bidb = Do_lpt ? blockIdx.y: blockIdx.z;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);

    int m_block = Do_lpt ? gridDim.z - 1 - blockIdx.z: blockIdx.x;

    flash::compute_attn_mha_1rowblock<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_MN, Return_softmax, Has_alibi, Layout, Params, OffsetT, IsVarlen>(params, bidb, bidh, m_block, warp_id);

    if constexpr (Is_causal and !Is_GQA/*MHA causal mask*/) {
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);
        flash::compute_attn_mha_1rowblock<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_MN, Return_softmax, Has_alibi, Layout, Params, OffsetT, IsVarlen>(params, bidb, bidh, gridDim.x * 2 - 1 - m_block, warp_id);
    }
}




////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Return_softmax, bool Has_alibi, int Layout, typename Params>
inline __device__ void compute_attn_mha_prefix_prefill_1rowblock(const Params &params, const int bidb, const int __bidh, const int m_block, const int warp_id) {
    using Element           = typename Kernel_traits::Element;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    using index_t           = typename Kernel_traits::index_t;
    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kBlockK   = Kernel_traits::kBlockK;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimVSplit;
    constexpr int kNWarps   = Kernel_traits::kNWarps;
    constexpr int WARP_M    = Kernel_traits::kWaveM;
    constexpr int WARP_N    = Kernel_traits::kWaveN;
    constexpr int STAGES    = Kernel_traits::STAGES;
    constexpr int WARP_NUM  = kBlockM / WARP_M;
    constexpr int SplitD    = Kernel_traits::SplitD;
    constexpr int kHeadDimVOrigin = Kernel_traits::kHeadDimV;

    // 获取 splitD 结果
    const int bidh = __bidh / SplitD;

    // 获取当前 TG 处理的任务大小
    // const flash::BlockInfo</*Varlen=*/!Is_even_MN, false/*Is_kvcache*/> binfo(params, bidb);
    flash::SafeDecodeBlockInfo binfo;
    binfo.set_params<Params, /*Is_Q_varlen=*/true, /*Is_K_Cumulative=*/false>(params, bidb);

    // 处理边界
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0) return;

    // 分配 lds
    extern __shared__ Element smem[];
    Element* q_lds = (Element*)&(smem);
    Element* v_lds = (Element*)&(smem);
    Element* k_lds = v_lds + (2 * STAGES * 32 * kBlockK);

    // 计算当前任务沿着 seqlenKV 方向的 block 起始位置和终止位置
    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, flash::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // 计算数据跨度
    int seqlen_q_stride = params.q_row_stride;
    int seqlen_k_stride = params.k_row_stride;
    int seqlen_v_stride = params.v_row_stride;
    int seqlen_o_stride = params.o_row_stride;
    int64_t row_offset_q, row_offset_k, row_offset_v, row_offset_o;
    int64_t row_offset_lse;
    // 获取页表信息
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = n_block_min * kBlockN / page_block_size;
    const int block_table_offset = n_block_min * kBlockN - block_table_idx * page_block_size;
    if constexpr (Layout == 1) { /*bshd layout, lse is num_heads, total_q*/
        row_offset_q = (binfo.sum_s_q + m_block * kBlockM) * int64_t(seqlen_q_stride) + params.q_head_stride * bidh;
        row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * seqlen_k_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * seqlen_v_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
        row_offset_lse = bidh * params.total_q + binfo.sum_s_q + m_block * kBlockM;
        row_offset_o = binfo.sum_s_q * int64_t(params.o_head_stride) * params.h + params.o_head_stride * bidh + m_block * kBlockM * seqlen_o_stride;
    } else { /*q/o: [total_q, h, d], k/v cache: [num_blocks, h_k, page_block_size, d]*/
        const int kv_head = bidh / params.h_h_k_ratio;
        row_offset_q = (binfo.sum_s_q + m_block * kBlockM) * int64_t(seqlen_q_stride) + int64_t(bidh) * params.q_head_stride;
        row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + kv_head * int64_t(params.k_head_stride) + block_table_offset * int64_t(seqlen_k_stride);
        row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + kv_head * int64_t(params.v_head_stride) + block_table_offset * int64_t(seqlen_v_stride);
        row_offset_lse = bidh * params.total_q + binfo.sum_s_q + m_block * kBlockM;
        row_offset_o = (binfo.sum_s_q + m_block * kBlockM) * int64_t(seqlen_o_stride) + int64_t(bidh) * params.o_head_stride;
    }
    #if 0
    if (int(threadIdx.x) == 0) {
        printf("bidb: %d | bidh: %d | actual_seqlen_q: %d | actual_seqlen_k: %d | n_block_max: %d | row_offset_q: %ld | row_offset_k: %ld | row_offset_v: %ld | row_offset_o: %ld | seqlen_q_stride: %d | seqlen_k_stride: %d | seqlen_v_stride: %d\n",
            bidb, bidh, binfo.actual_seqlen_q, binfo.actual_seqlen_k, n_block_max, row_offset_q, row_offset_k, row_offset_v, row_offset_o, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride);
    }
    #endif

    // 根据起始数据偏移量准备 Q/K/V 的 buffer resource 寄存器
    auto gQ = prepare_for_buffer_load<kHeadDim>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
    auto gK = prepare_for_buffer_load<kHeadDim>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto gV = prepare_for_buffer_load<kHeadDimV>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v);

    // attention 变体: Alibi
    float g_alibi;
    if constexpr (Has_alibi) {
        g_alibi = reinterpret_cast<ElementAccum*>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    }

    // attention 插件: Dropout
    unsigned long long rand_seed, rand_offset;
    uint32_t p_dropout_in_8bits_value;
    union_vec2_uint warp_idx_for_dropout;
    if constexpr (Is_dropout) {
        rand_seed                = params.rand_seed;
        rand_offset              = params.rand_offset + ((bidb * params.h + bidh) << 6) + (threadIdx.x & 63);
        p_dropout_in_8bits_value = params.p_dropout_in_uint8_t & 0xffffffff;
        warp_idx_for_dropout.u32.x = 1 * m_block * (kBlockM / 32) + warp_id;
        // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might exit early and no one saves the rng states.
        if (m_block == 0 and bidb == 0 and bidh == 0 and threadIdx.x == 0) {
            params.rng_state[0] = rand_seed;
            params.rng_state[1] = rand_offset;
        }
    }

    // 预取 Q 的数据到寄存器
    vec2_Element<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2][4];
    prefetch_q_to_vgpr<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(gQ, q_lds, q_reg, warp_id, seqlen_q_stride, binfo.actual_seqlen_q - m_block * kBlockM);

    /***************************************************************************************************************************/
    vec2_Accum<ElementAccum> scores_max[WARP_M / 32];
    vec2_Accum<ElementAccum> scores_sum[WARP_M / 32];
    vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4];
    attention_initialize<kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, 2/*M_MMAC_COUNT*/, ElementAccum>(scores_max, scores_sum, acc_o);
    /***************************************************************************************************************************/
    // 是否做 prefetch K, PV 结束后, prefetch K 有风险
    constexpr bool PREFETCH_K = false; // true;
    constexpr bool Aggressive = (kHeadDim == 128 or kHeadDim == 64);
    auto QK_GEMM_FUNC = Aggressive
        ? &qk_gemm_prefetch_v_headdim128<kHeadDim, kHeadDimV, kBlockM, kBlockN, kBlockK, WARP_M, WARP_N, STAGES, Element, ElementAccum, Is_even_MN>
        : &qk_gemm_prefetch_v<kHeadDim, kHeadDimV, kBlockM, kBlockN, kBlockK, WARP_M, WARP_N, STAGES, Element, ElementAccum, Is_even_MN>;

    auto PV_GEMM_FUNC = Aggressive
        ? &pv_gemm_prefetch_k_headdim128<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>
        : &pv_gemm_prefetch_k<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;
    auto PV_GEMM_FUNC_IN_MASK = Aggressive
        ? &pv_gemm_prefetch_k_headdim128<false, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>
        : &pv_gemm_prefetch_k<false, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;

    if constexpr (PREFETCH_K) {
        prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, warp_id, seqlen_k_stride, binfo.actual_seqlen_k - n_block_min * kBlockN);
    }

    // constexpr int n_masking_steps = (!Is_causal && !Is_local) ? 1 : flash::ceil_div(kBlockM, kBlockN);
    // These are the iterations where we don't need masking on S
    for (int n_block_loop = n_block_min; n_block_loop < n_block_max/*n_block_max - n_masking_steps*/; ++n_block_loop) {

        const int seqlen_kv_limit = binfo.actual_seqlen_k - n_block_loop * kBlockN;

        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (kBlockN / WARP_N)][4];

        if constexpr (not PREFETCH_K) {
            prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, warp_id, seqlen_k_stride, seqlen_kv_limit);
        }

        QK_GEMM_FUNC(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, warp_id, seqlen_k_stride, seqlen_v_stride, seqlen_kv_limit);

        if constexpr (Has_alibi) {
            apply_alibi<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, n_block_loop * kBlockN, binfo.actual_seqlen_k, m_block * kBlockM + warp_id * WARP_M, binfo.actual_seqlen_q, g_alibi);
        }

        if constexpr (Is_local) {
            apply_mask_local<Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, n_block_loop * kBlockN, binfo.actual_seqlen_k, m_block * kBlockM + warp_id * WARP_M, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right);
        } else if constexpr (Is_causal) {
            apply_mask_causal<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, n_block_loop * kBlockN, binfo.actual_seqlen_k, m_block * kBlockM + warp_id * WARP_M, binfo.actual_seqlen_q);
        } else if constexpr (!Is_even_MN) {
            apply_mask<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, seqlen_kv_limit);
        }

        softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        if constexpr (Is_dropout) {
            warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
            apply_dropout<false, vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg, binfo.actual_seqlen_q, binfo.actual_seqlen_k, m_block * kBlockM + warp_id * WARP_M, n_block_loop * kBlockN, bidb * params.h + bidh, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
        }

        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockN / 32)][4];
        convert_pk_type<WARP_M, kBlockN, Element, ElementAccum, true/*IsInference*/>(p_reg, s_reg);

        if constexpr (not PREFETCH_K) {
            PV_GEMM_FUNC(gV, gK, v_lds, k_lds, p_reg, acc_o, warp_id, seqlen_k_stride, seqlen_v_stride, seqlen_kv_limit);
        }

        const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;
        const int64_t k_delta             = int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(seqlen_k_stride);
        const int64_t v_delta             = int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(seqlen_v_stride);

        *(int64_t*)&gK += k_delta * sizeof(Element);

        if constexpr (PREFETCH_K) {
            PV_GEMM_FUNC(gV, gK, v_lds, k_lds, p_reg, acc_o, warp_id, seqlen_k_stride, seqlen_v_stride, seqlen_kv_limit);
        }

        *(int64_t*)&gV += v_delta * sizeof(Element);
    }

    /**********************************************************************************************************************************/
    // Epilogue: rescale acco
    vec2_Accum<ElementAccum> lse[WARP_M / 32];
    if (params.s_aux_ptr != nullptr) {
        const float sink_value = fwd_attention_sink_load(params.s_aux_ptr, params.s_aux_type, bidh);
        fwd_apply_attention_sink<WARP_M, kBlockK, kHeadDimV, ElementAccum>(
            acc_o, scores_max, scores_sum, params.scale_softmax, sink_value);
    }
    fwd_epilugue_rescale_acco<WARP_M, kBlockK, kHeadDimV, false/*Is_dropout*/, ElementAccum>(acc_o, lse, scores_max, scores_sum, params.scale_softmax, params.rp_dropout);
    /**************************************************************************************************************************************/
    int lane_id = threadIdx.x & 63;
    if (params.softmax_lse_ptr != nullptr) {
        fwd_epilogue_store_lse<WARP_M, Is_even_MN, SplitD, false/*Is_Interleaved*/, ElementAccum>(lse, params.softmax_lse_ptr, row_offset_lse, warp_id, lane_id, 0, binfo.actual_seqlen_q - m_block * kBlockM);
    }
    /**************************************************************************************************************************************/
    Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;
    fwd_epilogue_store_output<kHeadDimV, kBlockM, kBlockK, WARP_M, Is_even_MN, false/*Is_Interleaves*/, false/*TcpSwizzle*/, Element, ElementAccum>(o_ptr, acc_o, m_block, warp_id, lane_id, seqlen_o_stride, binfo.actual_seqlen_q);
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Return_softmax, bool Has_alibi, bool Is_GQA, int Layout, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_prefix_prefill_kernel(const Params params) {

    int bidh;
    int bidb;
    int m_block;
    if constexpr (Layout == 0) {
        const int group_size = params.h_h_k_ratio;
        const int m_index = blockIdx.x / group_size;
        const int head_in_group = blockIdx.x - m_index * group_size;
        bidh = blockIdx.y * group_size + head_in_group;
        bidb = blockIdx.z;
        m_block = gridDim.x / group_size - 1 - m_index;
    } else {
        bidh = blockIdx.x;
        bidb = blockIdx.y;
        m_block = gridDim.z - 1 - blockIdx.z;
    }

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);

    flash::compute_attn_mha_prefix_prefill_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Is_even_MN, Return_softmax, Has_alibi, Layout, Flash_fwd_params>(params, bidb, bidh, m_block, warp_id);
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Return_softmax, bool Has_alibi, int Layout, typename Params>
inline __device__ void compute_attn_mha_prefix_prefill_hdim512_16x64_1rowblock(
        const Params &params, const int bidb, const int bidh, const int m_block, const int warp_id) {
    using Element           = typename Kernel_traits::Element;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kBlockK   = Kernel_traits::kBlockK;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int WARP_M    = Kernel_traits::kWaveM;
    constexpr int WARP_N    = Kernel_traits::kWaveN;

    static_assert(kHeadDim == 512, "hdim512 16x64 prefill expects qk head dim 512");
    static_assert(kHeadDimV == 512, "hdim512 16x64 prefill expects v head dim 512");
    static_assert(kBlockM == 64 && kBlockN == 64 && kBlockK == 32);
    static_assert(WARP_M == 16 && WARP_N == 64);

    flash::SafeDecodeBlockInfo binfo;
    binfo.set_params<Params, /*Is_Q_varlen=*/true, /*Is_K_Cumulative=*/false>(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0) return;

    extern __shared__ Element smem[];
    Element* kv_lds = smem;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, flash::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const int seqlen_q_stride = params.q_row_stride;
    const int seqlen_k_stride = params.k_row_stride;
    const int seqlen_v_stride = params.v_row_stride;
    const int seqlen_o_stride = params.o_row_stride;

    int64_t row_offset_q = 0, row_offset_k = 0, row_offset_v = 0, row_offset_o = 0, row_offset_lse = 0;
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = n_block_min * kBlockN / page_block_size;
    const int block_table_offset = n_block_min * kBlockN - block_table_idx * page_block_size;

    if constexpr (Layout == 1) {
        row_offset_q = (binfo.sum_s_q + m_block * kBlockM) * int64_t(seqlen_q_stride) + params.q_head_stride * bidh;
        row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
        row_offset_lse = bidh * params.total_q + binfo.sum_s_q + m_block * kBlockM;
        row_offset_o = binfo.sum_s_q * int64_t(params.o_head_stride) * params.h + params.o_head_stride * bidh + m_block * kBlockM * int64_t(seqlen_o_stride);
    }

    const Element* q_ptr = reinterpret_cast<const Element*>(params.q_ptr) + row_offset_q;
    auto gK = prepare_for_buffer_load<kHeadDim>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto gV = prepare_for_buffer_load<kHeadDimV>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v);

    float g_alibi = 0.f;
    if constexpr (Has_alibi) {
        g_alibi = reinterpret_cast<ElementAccum*>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    }

    const int lane_id = threadIdx.x & 63;
    union_vec4_f16x2<Element> q_regs[kHeadDim / kBlockK];
    prefix_prefill_hdim512_16x64_fetch_q_to_vgpr<kHeadDim, kBlockK, WARP_M, Element, Is_even_MN>(
        q_ptr, q_regs, warp_id, lane_id, seqlen_q_stride, binfo.actual_seqlen_q - m_block * kBlockM);

    ElementAccum scores_max = -INFINITY;
    ElementAccum scores_sum = 0.f;
    vec4_Accum<ElementAccum> acc_o[kHeadDimV / 16];
    prefix_prefill_hdim512_16x64_clear_acc_o<kHeadDimV, ElementAccum>(acc_o);

    for (int n_block_loop = n_block_min; n_block_loop < n_block_max; ++n_block_loop) {
        const int seqlen_kv_limit = binfo.actual_seqlen_k - n_block_loop * kBlockN;
        vec4_Accum<ElementAccum> s_reg[kBlockN / 16];

        prefix_prefill_hdim512_16x64_qk<kHeadDim, kBlockN, kBlockK, WARP_M, Element, ElementAccum, Is_even_MN>(
            gK, kv_lds, q_regs, s_reg, warp_id, lane_id, seqlen_k_stride, seqlen_kv_limit);

        if constexpr (Has_alibi) {
            prefix_prefill_hdim512_16x64_apply_alibi<kBlockN, WARP_M, ElementAccum>(
                s_reg, lane_id, n_block_loop * kBlockN, m_block * kBlockM + warp_id * WARP_M, g_alibi);
        }
        if constexpr (Is_local) {
            prefix_prefill_hdim512_16x64_apply_local_mask<Is_local, kBlockN, WARP_M, ElementAccum>(
                s_reg, lane_id, n_block_loop * kBlockN, binfo.actual_seqlen_k,
                m_block * kBlockM + warp_id * WARP_M, binfo.actual_seqlen_q,
                params.window_size_left, params.window_size_right);
        } else if constexpr (Is_causal) {
            prefix_prefill_hdim512_16x64_apply_causal_mask<kBlockN, WARP_M, ElementAccum>(
                s_reg, lane_id, n_block_loop * kBlockN, binfo.actual_seqlen_k,
                m_block * kBlockM + warp_id * WARP_M, binfo.actual_seqlen_q);
        } else if constexpr (!Is_even_MN) {
            prefix_prefill_hdim512_16x64_apply_mask<kBlockN, WARP_M, ElementAccum>(
                s_reg, lane_id, n_block_loop * kBlockN, binfo.actual_seqlen_k);
        }

        prefix_prefill_hdim512_16x64_softmax_rescale_o<kBlockN, kHeadDimV, ElementAccum>(
            s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        union_vec2_f16x2<Element> p_reg[kBlockN / 16];
        prefix_prefill_hdim512_16x64_convert_p<kBlockN, Element, ElementAccum>(p_reg, s_reg);

        prefix_prefill_hdim512_16x64_pv<kHeadDimV, kBlockN, kBlockK, WARP_M, Element, ElementAccum, Is_even_MN>(
            gV, kv_lds, p_reg, acc_o, warp_id, lane_id, seqlen_v_stride, seqlen_kv_limit);

        const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;

        *(int64_t*)&gK += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);
        *(int64_t*)&gV += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);
    }

    if (params.s_aux_ptr != nullptr) {
        const float sink_value = fwd_attention_sink_load(params.s_aux_ptr, params.s_aux_type, bidh);
        prefix_prefill_hdim512_16x64_apply_attention_sink<kHeadDimV, ElementAccum>(
            acc_o, scores_max, scores_sum, params.scale_softmax, sink_value);
    }
    const ElementAccum lse = prefix_prefill_hdim512_16x64_rescale_acc_o<kHeadDimV, ElementAccum>(
        acc_o, scores_max, scores_sum, params.scale_softmax);

    prefix_prefill_hdim512_16x64_store_lse<WARP_M, ElementAccum>(
        lse, params.softmax_lse_ptr, row_offset_lse, warp_id, lane_id, binfo.actual_seqlen_q - m_block * kBlockM);

    Element* o_ptr = reinterpret_cast<Element*>(params.o_ptr) + row_offset_o;
    prefix_prefill_hdim512_16x64_store_output<kHeadDimV, WARP_M, Element, ElementAccum>(
        acc_o, o_ptr, warp_id, lane_id, seqlen_o_stride, binfo.actual_seqlen_q - m_block * kBlockM);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Return_softmax, bool Has_alibi, bool Is_GQA, int Layout, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_prefix_prefill_hdim512_16x64_kernel(const Params params) {
    const int bidh = blockIdx.x;
    const int bidb = blockIdx.y;
    int warp_id_vec = threadIdx.x / 64;
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
    int m_block = gridDim.z - 1 - blockIdx.z;
    flash::compute_attn_mha_prefix_prefill_hdim512_16x64_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Is_even_MN, Return_softmax, Has_alibi, Layout, Params>(params, bidb, bidh, m_block, warp_id);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Has_alibi, bool USE_BSHD_LAYOUT, typename Params>
inline __device__ void compute_attn_mha_padding_mask_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block, const int WARP_ID) {
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

    // 获取计算任务
    const flash::BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    int mask_actual_seqlen_q = params.padding_mask[bidb];
    int mask_actual_seqlen_k = mask_actual_seqlen_q;

    // 判断任务边界
    if (m_block * kBlockM >= mask_actual_seqlen_q || mask_actual_seqlen_k == 0) return;

    // 分配 lds usage
    extern __shared__ Element smem[];
    Element* q_lds = (Element*)&(smem);
    Element* v_lds = (Element*)&(smem);
    Element* k_lds = v_lds + (2 * STAGES * 32 * kBlockK);

    // 计算 seqlenkv 方向上 block 的边界
    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + mask_actual_seqlen_k - mask_actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = ceil_div(mask_actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, flash::ceil_div((m_block + 1) * kBlockM + mask_actual_seqlen_k - mask_actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // 计算数据跨度
    int seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, seqlen_o_stride;
    int row_offset_q, row_offset_k, row_offset_v, row_offset_o, row_offset_lse;
    int headdim_split_id = 0;
    fwd_prologue_compute_offset<1/*Layout*/, kBlockM, kBlockN, kHeadDim, kHeadDimV, kHeadDimV, 0, Is_even_MN, true/*Is_PaddingMask*/, Params>(
        seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, seqlen_o_stride, row_offset_q, row_offset_k, row_offset_v, row_offset_o, row_offset_lse,
        headdim_split_id, bidb, bidh, bidh, m_block, n_block_min, binfo, params
    );

    // 准备 Q/K/V/O 的 buffer resource
    auto gQ = prepare_for_buffer_load<kHeadDim, Element, false>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
    auto gK = prepare_for_buffer_load<kHeadDim, Element, false>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto gV = prepare_for_buffer_load<kHeadDimV, Element, false>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v);

    // 预取 Q 到寄存器
    vec2_Element<Element> q_reg[(kHeadDim / kBlockK) * ((WARP_M * kBlockK) / (32 * 32)) * 2][4];
    Is_even_MN
    ? prefetch_q_to_vgpr<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(gQ, q_lds, q_reg, WARP_ID, seqlen_q_stride)
    : prefetch_q_to_vgpr<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(gQ, q_lds, q_reg, WARP_ID, seqlen_q_stride, (mask_actual_seqlen_q - m_block * kBlockM));

    // masking 和 no-masking 分开计算的
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : flash::ceil_div(kBlockM, kBlockN) + (Is_even_MN ? 0 : 1);

    // 默认不做数据预取
    constexpr bool PREFETCH_K     = false;
    constexpr bool ALLOW_PREFETCH = (STAGES > 1);
    if constexpr (PREFETCH_K and ALLOW_PREFETCH) {
        if (n_block_min < n_block_max - n_masking_steps) {
            Is_even_MN
            ? prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride)
            : prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride, (mask_actual_seqlen_k - n_block_min * kBlockN));
        }
    }

    // 决定 qk/pv gemm 模式
    constexpr bool Aggressive = kHeadDim == 128 or kHeadDim == 64;
    auto QK_GEMM_FUNC = Aggressive
        ? &qk_gemm_prefetch_v_headdim128<kHeadDim, kHeadDimV, kBlockM, kBlockN, kBlockK, WARP_M, WARP_N, STAGES, Element, ElementAccum, Is_even_MN>
        : &qk_gemm_prefetch_v<kHeadDim, kHeadDimV, kBlockM, kBlockN, kBlockK, WARP_M, WARP_N, STAGES, Element, ElementAccum, Is_even_MN>;

    auto PV_GEMM_FUNC = Aggressive
        ? &pv_gemm_prefetch_k_headdim128<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>
        : &pv_gemm_prefetch_k<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;
    // mask 循环中不需要做这个 prefetch K, 固定为 false
    auto PV_GEMM_FUNC_IN_MASK = Aggressive
        ? &pv_gemm_prefetch_k_headdim128<false, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>
        : &pv_gemm_prefetch_k<false, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;

    /***************************************************************************************************************************/
    vec2_Accum<ElementAccum> scores_max[WARP_M / 32];
    vec2_Accum<ElementAccum> scores_sum[WARP_M / 32];
    vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4];
    attention_initialize<kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, 2/*M_MMAC_COUNT*/, ElementAccum>(scores_max, scores_sum, acc_o);
    /***************************************************************************************************************************/
    // These are the iterations where we don't need masking on S
    for (int n_block_loop = n_block_min; n_block_loop < n_block_max - n_masking_steps; ++n_block_loop) {

        const int seqlen_kv_limit = mask_actual_seqlen_k - n_block_loop * kBlockN;

        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (kBlockN / WARP_N)][4];

        if constexpr ((not PREFETCH_K) and ALLOW_PREFETCH) {
            Is_even_MN
            ? prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride)
            : prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride, seqlen_kv_limit);
        }

        Is_even_MN
        ? QK_GEMM_FUNC(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, seqlen_k_stride, seqlen_v_stride, 0)
        : QK_GEMM_FUNC(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, seqlen_k_stride, seqlen_v_stride, seqlen_kv_limit);

        *(uint64_t*)&gK += kBlockN * seqlen_k_stride * sizeof(Element);

        softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockN / 32)][4];
        convert_pk_type<WARP_M, kBlockN, Element, ElementAccum>(p_reg, s_reg);

        Is_even_MN
        ? PV_GEMM_FUNC(gV, gK, v_lds, k_lds, p_reg, acc_o, WARP_ID, seqlen_k_stride, seqlen_v_stride, 0)
        : PV_GEMM_FUNC(gV, gK, v_lds, k_lds, p_reg, acc_o, WARP_ID, seqlen_k_stride, seqlen_v_stride, seqlen_kv_limit);

        *(uint64_t*)&gV += kBlockN * seqlen_v_stride * sizeof(Element);
    }

    if constexpr (PREFETCH_K) {
        buffer_load_lds_dwordx1_wait<0>();
    }

    /***************************************************************************************************************************/
    int n_block = max(n_block_max - n_masking_steps, n_block_min);

    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, n_block++) {

        const int seqlen_kv_limit = mask_actual_seqlen_k - n_block * kBlockN;

        vec4_Accum<ElementAccum> s_reg[(kBlockN / 32) * (WARP_M / 32)][4];

        if constexpr (ALLOW_PREFETCH) {
            Is_even_MN
            ? prefetch_k_to_lds<kHeadDim, kBlockM, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride)
            : prefetch_k_to_lds<kHeadDim, kBlockM, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride, seqlen_kv_limit);
        }

        Is_even_MN
        ? QK_GEMM_FUNC(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, seqlen_k_stride, seqlen_v_stride, 0)
        : QK_GEMM_FUNC(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, seqlen_k_stride, seqlen_v_stride, seqlen_kv_limit);

        *(uint64_t*)&gK += ((kBlockN * seqlen_k_stride) * sizeof(Element));

        if constexpr (!Is_causal && !Is_local) {
            if constexpr (!Is_even_MN) { apply_mask<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, seqlen_kv_limit); }
        }

        softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockN / 32)][4];
        convert_pk_type<WARP_M, kBlockN, Element, ElementAccum>(p_reg, s_reg);

        Is_even_MN
        ? PV_GEMM_FUNC_IN_MASK(gV, gK, v_lds, k_lds, p_reg, acc_o, WARP_ID, seqlen_k_stride, seqlen_v_stride, 0)
        : PV_GEMM_FUNC_IN_MASK(gV, gK, v_lds, k_lds, p_reg, acc_o, WARP_ID, seqlen_k_stride, seqlen_v_stride, seqlen_kv_limit);

        *(uint64_t*)&gV += ((kBlockN * seqlen_v_stride) * sizeof(Element));
    }

    /**********************************************************************************************************************************/
    // Epilogue: rescale acco
    vec2_Accum<ElementAccum> lse[WARP_M / 32];
    if (params.s_aux_ptr != nullptr) {
        const float sink_value = fwd_attention_sink_load(params.s_aux_ptr, params.s_aux_type, bidh);
        fwd_apply_attention_sink<WARP_M, kBlockK, kHeadDimV, ElementAccum>(
            acc_o, scores_max, scores_sum, params.scale_softmax, sink_value);
    }
    fwd_epilugue_rescale_acco<WARP_M, kBlockK, kHeadDimV, false/*Is_dropout && Is_training*/, ElementAccum>(acc_o, lse, scores_max, scores_sum, params.scale_softmax, params.rp_dropout);
    /**************************************************************************************************************************************/
    int lane_id = threadIdx.x & 63;
    if (params.softmax_lse_ptr != nullptr) {
        const int row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        fwd_epilogue_store_lse<WARP_M, false/*Is_even_MN*/, false/*SplitD*/, false/*Is_Interleaved*/, ElementAccum>(lse, params.softmax_lse_ptr, row_offset_lse, WARP_ID, lane_id, 0, mask_actual_seqlen_q - m_block * kBlockM);
    }

    /**************************************************************************************************************************************/
    Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;
    fwd_epilogue_store_output<kHeadDimV, kBlockM, kBlockK, WARP_M, false/*Is_even_MN*/, false/*Is_Interleaves*/, false/*TcpSwizzle*/, Element, ElementAccum>(o_ptr, acc_o, m_block, WARP_ID, lane_id, seqlen_o_stride, mask_actual_seqlen_q);
}


template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Has_alibi, bool Is_GQA, bool USE_BSHD_LAYOUT, typename Params>
inline __device__ void compute_attn_padding_mask(const Params &params) {

    const int bidh = blockIdx.y;

    const int bidb = blockIdx.z;

    int warp_id_vec = threadIdx.x / 64;
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);

    const int m_block = blockIdx.x;
    flash::compute_attn_mha_padding_mask_1rowblock<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Return_softmax, Has_alibi, USE_BSHD_LAYOUT, Flash_fwd_params>(params, bidb, bidh, m_block, warp_id);
}



////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Has_alibi, bool USE_BSHD_LAYOUT, typename Params>
inline __device__ void compute_attn_mha_attn_mask_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block, const int WARP_ID) {
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

    // 获取计算任务
    const flash::BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    int actual_seqlen_q = binfo.actual_seqlen_q;
    int actual_seqlen_k = binfo.actual_seqlen_k;
    int mask_actual_seqlen_k = params.attn_mask[0];

    // 判断任务边界
    if (m_block * kBlockM >= actual_seqlen_q || actual_seqlen_k == 0) return;

    // 分配 lds usage
    extern __shared__ Element smem[];
    Element* q_lds = (Element*)&(smem);
    Element* v_lds = (Element*)&(smem);
    Element* k_lds = v_lds + (2 * STAGES * 32 * kBlockK);

    // 计算 seqlenkv 方向上 block 的边界
    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + actual_seqlen_k - actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = ceil_div(actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, flash::ceil_div((m_block + 1) * kBlockM + actual_seqlen_k - actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // 计算数据跨度
    int seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, seqlen_o_stride;
    int row_offset_q, row_offset_k, row_offset_v, row_offset_o, row_offset_lse;
    int headdim_split_id = 0;
    fwd_prologue_compute_offset<1/*Layout*/, kBlockM, kBlockN, kHeadDim, kHeadDimV, kHeadDimV, 0, Is_even_MN, true/*Is_PaddingMask*/, Params>(
        seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, seqlen_o_stride, row_offset_q, row_offset_k, row_offset_v, row_offset_o, row_offset_lse,
        headdim_split_id, bidb, bidh, bidh, m_block, n_block_min, binfo, params
    );

    // 准备 Q/K/V/O 的 buffer resource
    auto gQ = prepare_for_buffer_load<kHeadDim, Element, false>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
    auto gK = prepare_for_buffer_load<kHeadDim, Element, false>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto gV = prepare_for_buffer_load<kHeadDimV, Element, false>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v);

    // 预取 Q 到寄存器
    vec2_Element<Element> q_reg[(kHeadDim / kBlockK) * ((WARP_M * kBlockK) / (32 * 32)) * 2][4];
    Is_even_MN
    ? prefetch_q_to_vgpr<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(gQ, q_lds, q_reg, WARP_ID, seqlen_q_stride)
    : prefetch_q_to_vgpr<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(gQ, q_lds, q_reg, WARP_ID, seqlen_q_stride, (actual_seqlen_q - m_block * kBlockM));

    // attn_mask 的情况下, 默认全做 mask
    constexpr int n_masking_steps = 0;

    // 默认不做数据预取
    constexpr bool PREFETCH_K     = false;
    constexpr bool ALLOW_PREFETCH = (STAGES > 1);
    if constexpr (PREFETCH_K and ALLOW_PREFETCH) {
        if (n_block_min < n_block_max - n_masking_steps) {
            Is_even_MN
            ? prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride)
            : prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride, (actual_seqlen_k - n_block_min * kBlockN));
        }
    }

    // 决定 qk/pv gemm 模式
    constexpr bool Aggressive = kHeadDim == 128 or kHeadDim == 64;
    auto QK_GEMM_FUNC = Aggressive
        ? &qk_gemm_prefetch_v_headdim128<kHeadDim, kHeadDimV, kBlockM, kBlockN, kBlockK, WARP_M, WARP_N, STAGES, Element, ElementAccum, Is_even_MN>
        : &qk_gemm_prefetch_v<kHeadDim, kHeadDimV, kBlockM, kBlockN, kBlockK, WARP_M, WARP_N, STAGES, Element, ElementAccum, Is_even_MN>;

    auto PV_GEMM_FUNC = Aggressive
        ? &pv_gemm_prefetch_k_headdim128<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>
        : &pv_gemm_prefetch_k<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;
    // mask 循环中不需要做这个 prefetch K, 固定为 false
    auto PV_GEMM_FUNC_IN_MASK = Aggressive
        ? &pv_gemm_prefetch_k_headdim128<false, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>
        : &pv_gemm_prefetch_k<false, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;

    /***************************************************************************************************************************/
    vec2_Accum<ElementAccum> scores_max[WARP_M / 32];
    vec2_Accum<ElementAccum> scores_sum[WARP_M / 32];
    vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4];
    attention_initialize<kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, 2/*M_MMAC_COUNT*/, ElementAccum>(scores_max, scores_sum, acc_o);
    /***************************************************************************************************************************/
    // These are the iterations where we don't need masking on S
    for (int n_block_loop = n_block_min; n_block_loop < n_block_max - n_masking_steps; ++n_block_loop) {

        const int seqlen_k_limit = mask_actual_seqlen_k - n_block_loop * kBlockN;
        const int seqlen_v_limit = actual_seqlen_k - n_block_loop * kBlockN;

        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (kBlockN / WARP_N)][4];

        if constexpr ((not PREFETCH_K) and ALLOW_PREFETCH) {
            Is_even_MN
            ? prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride)
            : prefetch_k_to_lds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(gK, k_lds, WARP_ID, seqlen_k_stride, seqlen_k_limit);
        }

        Is_even_MN
        ? QK_GEMM_FUNC(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, seqlen_k_stride, seqlen_v_stride, 0)
        : QK_GEMM_FUNC(gQ, gK, gV, q_lds, k_lds, v_lds, q_reg, s_reg, WARP_ID, seqlen_k_stride, seqlen_v_stride, seqlen_k_limit);

        if constexpr (!Is_causal && !Is_local) {
            if constexpr (!Is_even_MN) { apply_mask<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, seqlen_k_limit); }
        }

        *(uint64_t*)&gK += kBlockN * seqlen_k_stride * sizeof(Element);

        softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockN / 32)][4];
        convert_pk_type<WARP_M, kBlockN, Element, ElementAccum>(p_reg, s_reg);

        Is_even_MN
        ? PV_GEMM_FUNC(gV, gK, v_lds, k_lds, p_reg, acc_o, WARP_ID, seqlen_k_stride, seqlen_v_stride, 0)
        : PV_GEMM_FUNC(gV, gK, v_lds, k_lds, p_reg, acc_o, WARP_ID, seqlen_k_stride, seqlen_v_stride, seqlen_v_limit);

        *(uint64_t*)&gV += kBlockN * seqlen_v_stride * sizeof(Element);
    }

    if constexpr (PREFETCH_K) {
        buffer_load_lds_dwordx1_wait<0>();
    }

    /**********************************************************************************************************************************/
    // Epilogue: rescale acco
    vec2_Accum<ElementAccum> lse[WARP_M / 32];
    if (params.s_aux_ptr != nullptr) {
        const float sink_value = fwd_attention_sink_load(params.s_aux_ptr, params.s_aux_type, bidh);
        fwd_apply_attention_sink<WARP_M, kBlockK, kHeadDimV, ElementAccum>(
            acc_o, scores_max, scores_sum, params.scale_softmax, sink_value);
    }
    fwd_epilugue_rescale_acco<WARP_M, kBlockK, kHeadDimV, false/*Is_dropout && Is_training*/, ElementAccum>(acc_o, lse, scores_max, scores_sum, params.scale_softmax, params.rp_dropout);
    /**************************************************************************************************************************************/
    int lane_id = threadIdx.x & 63;
    if (params.softmax_lse_ptr != nullptr) {
        const int row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        fwd_epilogue_store_lse<WARP_M, false/*Is_even_MN*/, false/*SplitD*/, false/*Is_Interleaved*/, ElementAccum>(lse, params.softmax_lse_ptr, row_offset_lse, WARP_ID, lane_id, 0, actual_seqlen_q - m_block * kBlockM);
    }

    /**************************************************************************************************************************************/
    Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;
    fwd_epilogue_store_output<kHeadDimV, kBlockM, kBlockK, WARP_M, false/*Is_even_MN*/, false/*Is_Interleaves*/, false/*TcpSwizzle*/, Element, ElementAccum>(o_ptr, acc_o, m_block, WARP_ID, lane_id, seqlen_o_stride, actual_seqlen_q);
}



template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Has_alibi, bool Is_GQA, bool USE_BSHD_LAYOUT, typename Params>
inline __device__ void compute_attn_attn_mask(const Params &params) {

    const int bidh = blockIdx.y;

    const int bidb = blockIdx.z;

    int warp_id_vec = threadIdx.x / 64;
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);

    const int m_block = blockIdx.x;
    flash::compute_attn_mha_attn_mask_1rowblock<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Return_softmax, Has_alibi, USE_BSHD_LAYOUT, Flash_fwd_params>(params, bidb, bidh, m_block, warp_id);
}






////////////////////////////////////////////////////////////////////////////////////////////////////
//                                            GFX938 kernels
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "fwd/gfx938/qk_gemm_prefetch_v_mls_ds.h"
#include "fwd/gfx938/pv_gemm_prefetch_k_mls_ds.h"
#include "fwd/gfx938/softmax_gfx938.h"
#include "fwd/gfx938/fwd_epilogue_gfx938.h"
template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_Varlen, bool Needs_extra_mask, bool Return_softmax, bool Has_alibi, int Layout, typename Params>
inline __device__ void compute_attn_mha_1rowblock_gfx938(const Params &params, const int bidb, const int bidh, const int m_block, const int warp_id) {
    using Element           = typename Kernel_traits::Element;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    using index_t           = std::conditional_t<Is_even_MN, int32_t, int64_t>; // 定长训练不至于出现超过 4GB 的 tensor, typename Kernel_traits::index_t;
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
    constexpr int kHeadDimQKCompute = Kernel_traits::kHeadDimQKCompute;
    constexpr int kHeadDimPVCompute = Kernel_traits::kHeadDimPVCompute;
    constexpr int TailTile16 = Kernel_traits::TailTile16;

    // 获取当前 TG 处理的任务大小
    using BlockInfoType = std::conditional_t<Is_Varlen, flash::SimplifyBlockInfo<false/*Is_bshd*/>, flash::BlockInfo<false/*Is_varlen*/> >;
    const BlockInfoType binfo(params, bidb);

    // 处理边界
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0) return;
    int warp_offset_in_seq_q = m_block * kBlockM + warp_id * WARP_M;

    // 分配 lds
    extern __shared__ Element smem[];
    Element* q_lds = (Element*)&(smem);
    Element* k_lds = q_lds;
    Element* v_lds = k_lds;

    // 计算当前任务沿着 seqlenKV 方向的 block 起始位置和终止位置
    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, flash::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // 计算数据跨度
    int seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, seqlen_o_stride;
    index_t row_offset_q, row_offset_k, row_offset_v, row_offset_o;
    int row_offset_lse;
    int headdim_split_id = 0;
    fwd_prologue_compute_offset<Layout, kBlockM, kBlockN, kHeadDim, kHeadDimV, kHeadDimV, 0/*SplitD*/, Is_even_MN, false/*Is_PaddingMask*/, Params, decltype(binfo), decltype(row_offset_q)>(
        seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, seqlen_o_stride, row_offset_q, row_offset_k, row_offset_v, row_offset_o, row_offset_lse,
        headdim_split_id, bidb, bidh, bidh, m_block, n_block_min, binfo, params
    );
    #if 0
    if (int(threadIdx.x) == 0) {
        printf("bidb: %d | bidh: %d | actual_seqlen_q: %d | actual_seqlen_k: %d | n_block_max: %d | row_offset_q: %d | row_offset_k: %d | row_offset_v: %d | row_offset_o: %d | seqlen_q_stride: %d | seqlen_k_stride: %d | seqlen_v_stride: %d\n",
            bidb, bidh, binfo.actual_seqlen_q, binfo.actual_seqlen_k, n_block_max, row_offset_q, row_offset_k, row_offset_v, row_offset_o, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride);
    }
    #endif
    // 根据起始数据偏移量准备 Q/K/V 的 buffer resource 寄存器
    auto q_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
    auto k_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto v_ptr = prepare_for_matrix_load<kHeadDimV>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v);

    // attention 变体: Alibi
    float g_alibi;
    if constexpr (Has_alibi) {
        g_alibi = reinterpret_cast<ElementAccum*>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    }

    // attention 插件: Dropout
    unsigned long long rand_seed, rand_offset;
    uint32_t p_dropout_in_8bits_value;
    union_vec2_uint warp_idx_for_dropout;
    if constexpr (Is_dropout and Is_training) {
        rand_seed                = params.rand_seed;
        rand_offset              = params.rand_offset + ((bidb * params.h + bidh) << 6) + (threadIdx.x & 63);
        p_dropout_in_8bits_value = params.p_dropout_in_uint8_t & 0xffffffff;
        warp_idx_for_dropout.u32.x = 1 * m_block * (kBlockM / 32) /* 前面几个 block 累积的 warp 数目, 这里不直接填 WARP_M, 参照 NV 的写法*/ + warp_id/*当前 block 内的 warp id*/;
        if (Is_training and m_block == 0 and bidb == 0 and bidh == 0 and threadIdx.x == 0) {
            params.rng_state[0] = rand_seed;
            params.rng_state[1] = rand_offset;
        }
    }

    // 预取 Q 的数据到寄存器
    union_vec4_f16x2<Element> q_reg[(kHeadDimQKCompute / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2];
    prefetch_q_to_vgpr_mls_ds<kHeadDimQKCompute, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(q_ptr, q_lds, q_reg, warp_id, seqlen_q_stride, Is_even_MN ? 0: binfo.actual_seqlen_q - m_block * kBlockM);

    // apply causal mask 的步骤和 no causal mask 的步骤分开算
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : flash::ceil_div(kBlockM, kBlockN) + (Needs_extra_mask ? 1 : 0);

    // 是否做 prefetch K, PV 结束后, prefetch K 有风险
    constexpr bool PREFETCH_K = (Is_even_MN) and( ( kHeadDim == 128 and kHeadDimV == 128) || (kHeadDim == 192 and kHeadDimV == 128) ); // 简单场景下开启
    constexpr bool ALLOW_PREFETCH = (STAGES > 1); // 客观上决定是否开启 prefetch
    if constexpr (PREFETCH_K and ALLOW_PREFETCH) {
        if (n_block_min < n_block_max - n_masking_steps) {
            prefetch_k_to_lds_mls_ds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, Is_even_MN ? 0: binfo.actual_seqlen_k - n_block_min * kBlockN);
        }
    }

    /***************************************************************************************************************************/
    vec2_Accum<ElementAccum> scores_max[WARP_M / 32];
    vec2_Accum<ElementAccum> scores_sum[WARP_M / 32];
    vec4_Accum<ElementAccum> acc_o[(kHeadDimPVCompute / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4];
    attention_initialize<kHeadDimPVCompute / kBlockK, WARP_M / 32, kBlockK / 32, 2/*M_MMAC_COUNT*/, ElementAccum>(scores_max, scores_sum, acc_o);
    /***************************************************************************************************************************/

    auto QK_GEMM_FUNC = qk_gemm_prefetch_v_mls_ds<kHeadDim, kHeadDimV, kHeadDimQKCompute, TailTile16, kBlockM, kBlockN, kBlockK, WARP_M, WARP_N, STAGES, Element, ElementAccum, Is_even_MN>;
    auto PV_GEMM_FUNC = &pv_gemm_prefetch_k_mls_ds<PREFETCH_K, kHeadDim, kHeadDimV, kHeadDimPVCompute, TailTile16, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;
    auto PV_GEMM_FUNC_IN_MASK = &pv_gemm_prefetch_k_mls_ds<false, kHeadDim, kHeadDimV, kHeadDimPVCompute, TailTile16, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;

    // Mainloop, 主循环, 不做 causal mask 的部分
    for (int n_block_loop = n_block_min; n_block_loop < n_block_max - n_masking_steps; ++n_block_loop) {

        // 计算当前 loop 下 seqlen_kv 的数据起始位置和边界
        int warp_offset_in_seqkv = n_block_loop * kBlockN;
        int warp_seqkv_limit     = Is_even_MN ? 0: binfo.actual_seqlen_k - warp_offset_in_seqkv;

        // 预取 K 的数据到 lds
        if constexpr (not PREFETCH_K) {
            prefetch_k_to_lds_mls_ds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, warp_seqkv_limit);
        }

        // 准备 QK gemm 输出的寄存器
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (kBlockN / WARP_N)][4];

        // QK gemm
        QK_GEMM_FUNC(k_ptr, v_ptr, k_lds, v_lds, q_reg, s_reg, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);

        // Attention 变体 alibi
        if constexpr (Has_alibi) {
            apply_alibi_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, g_alibi);
        }

        // Attention 变体 local mask
        if constexpr (Is_local) {
            apply_mask_local_gfx938</*HasWSLeft=*/Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right);
        }

        // 对 QK 输出做 softmax, 以及重放缩 acc_o/scores_sum
        softmax_rescale_o_gfx938<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, kHeadDimPVCompute, kBlockK, WARP_M, kBlockN>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        vec4_Accum<ElementAccum> s_reg_dmask[(kBlockN / 32) * (WARP_M / 32)][4];
        if constexpr (Return_softmax) {
            #pragma unroll
            for (int idx = 0; idx < (kBlockN / 32) * (WARP_M / 32); ++idx) {
                #pragma unroll
                for (int frag = 0; frag < 4; ++frag) {
                    s_reg_dmask[idx][frag] = s_reg[idx][frag];
                }
            }
        }

        // Attention 变体 dropout
        if constexpr (Is_dropout and Is_training) {
            warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
            apply_dropout<false, vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg, binfo.actual_seqlen_q, binfo.actual_seqlen_k, warp_offset_in_seq_q, n_block_loop * kBlockN, bidb * params.h + bidh, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
            if constexpr (Return_softmax) {
                apply_dropout<true, vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg_dmask, binfo.actual_seqlen_q, binfo.actual_seqlen_k, warp_offset_in_seq_q, n_block_loop * kBlockN, bidb * params.h + bidh, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
            }
            __builtin_amdgcn_sched_barrier(0);
        }

        if constexpr (Return_softmax) {
            store_s_dmask_gfx938<vec4_Accum<ElementAccum>, Element, WARP_M, kBlockN, kBlockM, kBlockN, Is_even_MN>(
                params.p_ptr,
                s_reg_dmask,
                bidb,
                bidh,
                m_block,
                n_block_loop,
                warp_id,
                threadIdx.x & 63,
                params.h,
                params.seqlen_q_rounded,
                params.seqlen_k_rounded,
                binfo.actual_seqlen_q,
                binfo.actual_seqlen_k
            );
        }

        // softmax(QK) f32 -> f16
        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockN / 32)][4];
        convert_pk_type<WARP_M, kBlockN, Element, ElementAccum>(p_reg, s_reg);

        // 偏移 K 指针, 提前偏移准备预取 K
        *(uint64_t*)&k_ptr += kBlockN * params.k_row_stride * sizeof(Element);

        // PV gemm
        PV_GEMM_FUNC(v_ptr, k_ptr, v_lds, k_lds, p_reg, acc_o, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);

        // 偏移 V 指针
        *(uint64_t*)&v_ptr += kBlockN * params.v_row_stride * sizeof(Element);
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
        int warp_seqkv_limit     = Is_even_MN ? 0: binfo.actual_seqlen_k - warp_offset_in_seqkv;

        // 预取 K 的数据到 lds
        prefetch_k_to_lds_mls_ds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, warp_seqkv_limit);

        // 准备 QK gemm 输出的寄存器
        vec4_Accum<ElementAccum> s_reg[(kBlockN / 32) * (WARP_M / 32)][4];

        // QK gemm
        QK_GEMM_FUNC(k_ptr, v_ptr, k_lds, v_lds, q_reg, s_reg, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);

        // 偏移 K 指针, 提前偏移准备预取 K
        *(uint64_t*)&k_ptr += kBlockN * params.k_row_stride * sizeof(Element);

        // Attention 变体 alibi
        if constexpr (Has_alibi) {
            apply_alibi_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, g_alibi);
        }

        // Attention: mask, causal mask, local mask
        if constexpr (!Is_causal && !Is_local) {
            if constexpr (!Is_even_MN) { apply_mask_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_seqkv_limit); }
        } else {
            if constexpr (Is_causal) {
                apply_mask_causal_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q);
            } else if constexpr (Is_local) {
                apply_mask_local_gfx938</*HasWSLeft=*/Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right);
            }
        }

        // 对 QK 输出做 softmax, 以及重放缩 acc_o/scores_sum
        softmax_rescale_o_gfx938<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, kHeadDimPVCompute, kBlockK, WARP_M, kBlockN>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        vec4_Accum<ElementAccum> s_reg_dmask[(kBlockN / 32) * (WARP_M / 32)][4];
        if constexpr (Return_softmax) {
            #pragma unroll
            for (int idx = 0; idx < (kBlockN / 32) * (WARP_M / 32); ++idx) {
                #pragma unroll
                for (int frag = 0; frag < 4; ++frag) {
                    s_reg_dmask[idx][frag] = s_reg[idx][frag];
                }
            }
        }

        // Attention 变体 dropout
        if constexpr (Is_dropout and Is_training) {
            warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
            apply_dropout<false, vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg, binfo.actual_seqlen_q, binfo.actual_seqlen_k, warp_offset_in_seq_q, n_block_loop * kBlockN, bidb * params.h + bidh, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
            if constexpr (Return_softmax) {
                apply_dropout<true, vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg_dmask, binfo.actual_seqlen_q, binfo.actual_seqlen_k, warp_offset_in_seq_q, n_block_loop * kBlockN, bidb * params.h + bidh, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
            }
            __builtin_amdgcn_sched_barrier(0);
        }

        if constexpr (Return_softmax) {
            store_s_dmask_gfx938<vec4_Accum<ElementAccum>, Element, WARP_M, kBlockN, kBlockM, kBlockN, Is_even_MN>(
                params.p_ptr,
                s_reg_dmask,
                bidb,
                bidh,
                m_block,
                n_block_loop,
                warp_id,
                threadIdx.x & 63,
                params.h,
                params.seqlen_q_rounded,
                params.seqlen_k_rounded,
                binfo.actual_seqlen_q,
                binfo.actual_seqlen_k
            );
        }

        // softmax(QK) f32 -> f16
        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockN / 32)][4];
        convert_pk_type<WARP_M, kBlockN, Element, ElementAccum>(p_reg, s_reg);

        // PV gemm
        PV_GEMM_FUNC_IN_MASK(v_ptr, k_ptr, v_lds, k_lds, p_reg, acc_o, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);

        // 偏移 V 指针
        *(uint64_t*)&v_ptr += kBlockN * params.v_row_stride * sizeof(Element);
    }

    /**********************************************************************************************************************************/
    // Epilogue: rescale acco
    vec2_Accum<ElementAccum> lse[WARP_M / 32];
    if (params.s_aux_ptr != nullptr) {
        const float sink_value = fwd_attention_sink_load(params.s_aux_ptr, params.s_aux_type, bidh);
        fwd_apply_attention_sink<WARP_M, kBlockK, kHeadDimPVCompute, ElementAccum>(
            acc_o, scores_max, scores_sum, params.scale_softmax, sink_value);
    }
    fwd_epilugue_rescale_acco<WARP_M, kBlockK, kHeadDimPVCompute, Is_dropout && Is_training, ElementAccum>(acc_o, lse, scores_max, scores_sum, params.scale_softmax, params.rp_dropout);
    /**************************************************************************************************************************************/
    constexpr bool Is_Interleave = true;
    int lane_id = threadIdx.x & 63;
    if (params.softmax_lse_ptr != nullptr) {
        fwd_epilogue_store_lse<WARP_M, Is_even_MN, false/*SplitD*/, Is_Interleave, ElementAccum>(lse, params.softmax_lse_ptr, row_offset_lse, warp_id, lane_id, 0, binfo.actual_seqlen_q - m_block * kBlockM);
    }
    /**************************************************************************************************************************************/
    Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;
    fwd_epilogue_store_output_gfx938<kHeadDimPVCompute, kBlockM, kBlockK, WARP_M, TailTile16, Is_even_MN, Is_Interleave, false/*TcpSwizzle*/, Element, ElementAccum>(o_ptr, acc_o, m_block, warp_id, lane_id, seqlen_o_stride, binfo.actual_seqlen_q);
}


template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_Varlen, bool Needs_extra_mask, bool Return_softmax, bool Has_alibi, int Layout, typename Params>
inline __device__ void compute_attn_gfx938(const Params &params) {

#if defined(__gfx938__) || defined(__gfx946__)
    constexpr bool Do_lpt = Is_causal;

    const int bidh = Do_lpt ? blockIdx.x: blockIdx.y;

    const int bidb = Do_lpt ? blockIdx.y: blockIdx.z;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);

    int m_block = Do_lpt ? gridDim.z - 1 - blockIdx.z: blockIdx.x;
    flash::compute_attn_mha_1rowblock_gfx938<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_Varlen, Needs_extra_mask, Return_softmax, Has_alibi, Layout, Flash_fwd_params>(params, bidb, bidh, m_block, warp_id);
#endif
}




template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Has_alibi, int Layout, typename Params>
inline __device__ void compute_attn_prefix_prefill_1rowblock_gfx938(const Params &params, const int bidb, const int __bidh, const int m_block, const int warp_id) {
    using Element           = typename Kernel_traits::Element;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    using index_t           = typename Kernel_traits::index_t;
    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kBlockK   = Kernel_traits::kBlockK;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimVSplit;
    constexpr int kNWarps   = Kernel_traits::kNWarps;
    constexpr int WARP_M    = Kernel_traits::kWaveM;
    constexpr int WARP_N    = Kernel_traits::kWaveN;
    constexpr int STAGES    = Kernel_traits::STAGES;
    constexpr int WARP_NUM  = kBlockM / WARP_M;
    constexpr int SplitD    = Kernel_traits::SplitD;
    constexpr int kHeadDimVOrigin = Kernel_traits::kHeadDimV;

    // 获取 splitD 结果
    const int bidh = __bidh / SplitD;

    // 获取当前 TG 处理的任务大小
    // const flash::BlockInfo</*Varlen=*/!Is_even_MN, false/*Is_kvcache*/> binfo(params, bidb);
    flash::SafeDecodeBlockInfo binfo;
    binfo.set_params<Params, /*Is_Q_varlen=*/true, /*Is_K_Cumulative=*/false>(params, bidb);

    // 处理边界
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0) return;
    int warp_offset_in_seq_q = m_block * kBlockM + warp_id * WARP_M;

    // 分配 lds
    extern __shared__ Element smem[];
    Element* q_lds = (Element*)&(smem);
    Element* v_lds = q_lds;
    Element* k_lds = q_lds;

    // 计算当前任务沿着 seqlenKV 方向的 block 起始位置和终止位置
    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, flash::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // 计算数据跨度
    int seqlen_q_stride = params.q_row_stride;
    int seqlen_k_stride = params.k_row_stride;
    int seqlen_v_stride = params.v_row_stride;
    int seqlen_o_stride = params.o_row_stride;
    int64_t row_offset_q, row_offset_k, row_offset_v, row_offset_o;
    int64_t row_offset_lse;
    // 获取页表信息
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = n_block_min * kBlockN / page_block_size;
    const int block_table_offset = n_block_min * kBlockN - block_table_idx * page_block_size;
    if constexpr (Layout == 1) { /*bshd layout, lse is num_heads, total_q*/
        row_offset_q = (binfo.sum_s_q + m_block * kBlockM) * int64_t(seqlen_q_stride) + params.q_head_stride * bidh;
        row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
        row_offset_lse = bidh * params.total_q + binfo.sum_s_q + m_block * kBlockM;
        row_offset_o = binfo.sum_s_q * int64_t(params.o_head_stride) * params.h + params.o_head_stride * bidh + m_block * kBlockM * seqlen_o_stride;
    }
    #if 0
    if (int(threadIdx.x) == 0) {
        printf("bidb: %d | bidh: %d | actual_seqlen_q: %d | actual_seqlen_k: %d | n_block_max: %d | row_offset_q: %ld | row_offset_k: %ld | row_offset_v: %ld | row_offset_o: %ld | seqlen_q_stride: %d | seqlen_k_stride: %d | seqlen_v_stride: %d\n",
            bidb, bidh, binfo.actual_seqlen_q, binfo.actual_seqlen_k, n_block_max, row_offset_q, row_offset_k, row_offset_v, row_offset_o, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride);
    }
    #endif

    // 根据起始数据偏移量准备 Q/K/V 的 buffer resource 寄存器
    auto q_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
    auto k_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto v_ptr = prepare_for_matrix_load<kHeadDimV>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v);

    // attention 变体: Alibi
    float g_alibi;
    if constexpr (Has_alibi) {
        g_alibi = reinterpret_cast<ElementAccum*>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    }

    // attention 插件: Dropout
    unsigned long long rand_seed, rand_offset;
    uint32_t p_dropout_in_8bits_value;
    union_vec2_uint warp_idx_for_dropout;
    if constexpr (Is_dropout and Is_training) {
        rand_seed                = params.rand_seed;
        rand_offset              = params.rand_offset + ((bidb * params.h + bidh) << 6) + (threadIdx.x & 63);
        p_dropout_in_8bits_value = params.p_dropout_in_uint8_t & 0xffffffff;
        warp_idx_for_dropout.u32.x = 1 * m_block * (kBlockM / 32) /* 前面几个 block 累积的 warp 数目, 这里不直接填 WARP_M, 参照 NV 的写法*/ + warp_id/*当前 block 内的 warp id*/;
        if (Is_training and m_block == 0 and bidb == 0 and bidh == 0 and threadIdx.x == 0) {
            params.rng_state[0] = rand_seed;
            params.rng_state[1] = rand_offset;
        }
    }

    // 预取 Q 的数据到寄存器
    union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2];
    prefetch_q_to_vgpr_mls_ds<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(q_ptr, q_lds, q_reg, warp_id, seqlen_q_stride, Is_even_MN ? 0: binfo.actual_seqlen_q - m_block * kBlockM);

    // apply causal mask 的步骤和 no causal mask 的步骤分开算
    // prefix prefill 目前没分开算, 明确边界的情况下也可以分开算, 性能会有提升
    int n_masking_steps = (!Is_causal && !Is_local) ? 1: min(n_block_max, flash::ceil_div(kBlockM, kBlockN) + 1);

    // 是否做 prefetch K, PV 结束后, prefetch K 有风险
    constexpr bool PREFETCH_K = Is_even_MN and ((kHeadDim == 128 and kHeadDimV == 128) || (kHeadDim == 192 and kHeadDimV == 128));
    constexpr bool ALLOW_PREFETCH = (STAGES > 1); // 客观上决定是否开启 prefetch
    if constexpr (PREFETCH_K and ALLOW_PREFETCH) {
        if (n_block_min < n_block_max - n_masking_steps) {
            prefetch_k_to_lds_mls_ds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, Is_even_MN ? 0: binfo.actual_seqlen_k - n_block_min * kBlockN);
        }
    }

    /***************************************************************************************************************************/
    vec2_Accum<ElementAccum> scores_max[WARP_M / 32];
    vec2_Accum<ElementAccum> scores_sum[WARP_M / 32];
    vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4];
    attention_initialize<kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, 2/*M_MMAC_COUNT*/, ElementAccum>(scores_max, scores_sum, acc_o);
    /***************************************************************************************************************************/

    auto QK_GEMM_FUNC = qk_gemm_prefetch_v_mls_ds<kHeadDim, kHeadDimV, kHeadDim, 2, kBlockM, kBlockN, kBlockK, WARP_M, WARP_N, STAGES, Element, ElementAccum, Is_even_MN>;
    auto PV_GEMM_FUNC = &pv_gemm_prefetch_k_mls_ds<PREFETCH_K, kHeadDim, kHeadDimV, kHeadDimV, 2, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;
    auto PV_GEMM_FUNC_IN_MASK = &pv_gemm_prefetch_k_mls_ds<false, kHeadDim, kHeadDimV, kHeadDimV, 2, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;

    // Mainloop, 主循环, 不做 causal mask 的部分
    for (int n_block_loop = n_block_min; n_block_loop < n_block_max - n_masking_steps; ++n_block_loop) {

        // 计算当前 loop 下 seqlen_kv 的数据起始位置和边界
        int warp_offset_in_seqkv = n_block_loop * kBlockN;
        int warp_seqkv_limit     = Is_even_MN ? 0: binfo.actual_seqlen_k - warp_offset_in_seqkv;

        // 预取 K 的数据到 lds
        if constexpr (not PREFETCH_K) {
            prefetch_k_to_lds_mls_ds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, warp_seqkv_limit);
        }

        // 准备 QK gemm 输出的寄存器
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (kBlockN / WARP_N)][4];

        // QK gemm
        QK_GEMM_FUNC(k_ptr, v_ptr, k_lds, v_lds, q_reg, s_reg, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);

        // Attention 变体 alibi
        if constexpr (Has_alibi) {
            apply_alibi_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, g_alibi);
        }

        // Attention: mask, causal mask, local mask
        if constexpr (Is_local) {
            apply_mask_local_gfx938</*HasWSLeft=*/Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right);
        }

        // 对 QK 输出做 softmax, 以及重放缩 acc_o/scores_sum
        softmax_rescale_o_gfx938<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);


        // Attention 变体 dropout
        if constexpr (Is_dropout and Is_training) {
            warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
            apply_dropout<false, vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg, binfo.actual_seqlen_q, binfo.actual_seqlen_k, warp_offset_in_seq_q, n_block_loop * kBlockN, bidb * params.h + bidh, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
            __builtin_amdgcn_sched_barrier(0);
        }

        // softmax(QK) f32 -> f16
        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockN / 32)][4];
        convert_pk_type<WARP_M, kBlockN, Element, ElementAccum>(p_reg, s_reg);

        const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;

        if constexpr (PREFETCH_K) { *(int64_t*)&k_ptr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element); }

        // PV gemm
        PV_GEMM_FUNC(v_ptr, k_ptr, v_lds, k_lds, p_reg, acc_o, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);

        // 偏移 V 指针
        if constexpr (not PREFETCH_K) { *(int64_t*)&k_ptr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element); }
        *(int64_t*)&v_ptr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);
    }

    // prefetch K 的话, 最后一次多取了一段 K, 为了不影响后续的操作, 需要同步等待
    if constexpr (PREFETCH_K) { buffer_load_lds_dwordx1_wait<0>(); }

    /***************************************************************************************************************************/
    // Rest loop, 做 causal mask 的部分
    int n_block_loop = max(n_block_max - n_masking_steps, n_block_min);
    // #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, ++n_block_loop) {
        // 计算当前 loop 下 seqlen_kv 的数据起始位置和边界
        int warp_offset_in_seqkv = n_block_loop * kBlockN;
        int warp_seqkv_limit     = Is_even_MN ? 0: binfo.actual_seqlen_k - warp_offset_in_seqkv;

        // 预取 K 的数据到 lds
        if constexpr (true) {
            prefetch_k_to_lds_mls_ds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, warp_seqkv_limit);
        }

        // 准备 QK gemm 输出的寄存器
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (kBlockN / WARP_N)][4];

        // QK gemm
        QK_GEMM_FUNC(k_ptr, v_ptr, k_lds, v_lds, q_reg, s_reg, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);

        // Attention 变体 alibi
        if constexpr (Has_alibi) {
            apply_alibi_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, g_alibi);
        }

        // Attention: mask, causal mask, local mask
        if constexpr (!Is_causal && !Is_local) {
            apply_mask_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, binfo.actual_seqlen_k, warp_offset_in_seqkv);
        } else if constexpr (Is_local) {
            apply_mask_local_gfx938</*HasWSLeft=*/Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right);
        } else if constexpr (Is_causal) {
            apply_mask_causal_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q);
        }

        // 对 QK 输出做 softmax, 以及重放缩 acc_o/scores_sum
        softmax_rescale_o_gfx938<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);


        // Attention 变体 dropout
        if constexpr (Is_dropout and Is_training) {
            warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
            apply_dropout<false, vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg, binfo.actual_seqlen_q, binfo.actual_seqlen_k, warp_offset_in_seq_q, n_block_loop * kBlockN, bidb * params.h + bidh, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
            __builtin_amdgcn_sched_barrier(0);
        }

        // softmax(QK) f32 -> f16
        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockN / 32)][4];
        convert_pk_type<WARP_M, kBlockN, Element, ElementAccum>(p_reg, s_reg);

        const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;

        // PV gemm
        PV_GEMM_FUNC(v_ptr, k_ptr, v_lds, k_lds, p_reg, acc_o, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);

        // 偏移 V 指针
        *(int64_t*)&k_ptr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);
        *(int64_t*)&v_ptr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);
    }

    /**********************************************************************************************************************************/
    // Epilogue: rescale acco
    vec2_Accum<ElementAccum> lse[WARP_M / 32];
    if (params.s_aux_ptr != nullptr) {
        const float sink_value = fwd_attention_sink_load(params.s_aux_ptr, params.s_aux_type, bidh);
        fwd_apply_attention_sink<WARP_M, kBlockK, kHeadDimV, ElementAccum>(
            acc_o, scores_max, scores_sum, params.scale_softmax, sink_value);
    }
    fwd_epilugue_rescale_acco<WARP_M, kBlockK, kHeadDimV, Is_dropout && Is_training, ElementAccum>(acc_o, lse, scores_max, scores_sum, params.scale_softmax, params.rp_dropout);
    /**************************************************************************************************************************************/
    constexpr bool Is_Interleave = true;
    int lane_id = threadIdx.x & 63;
    if (params.softmax_lse_ptr != nullptr) {
        fwd_epilogue_store_lse<WARP_M, Is_even_MN, false/*SplitD*/, Is_Interleave, ElementAccum>(lse, params.softmax_lse_ptr, row_offset_lse, warp_id, lane_id, 0, binfo.actual_seqlen_q - m_block * kBlockM);
    }
    /**************************************************************************************************************************************/
    Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;
    fwd_epilogue_store_output_gfx938<kHeadDimV, kBlockM, kBlockK, WARP_M, 2/*TailTile16*/, Is_even_MN, Is_Interleave, false/*TcpSwizzle*/, Element, ElementAccum>(o_ptr, acc_o, m_block, warp_id, lane_id, seqlen_o_stride, binfo.actual_seqlen_q);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//                                            GFX92A kernels
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "fwd/gfx92a/qk_gemm_prefetch_v_mls_ds_gfx92a.h"
#include "fwd/gfx92a/pv_gemm_prefetch_k_mls_ds_gfx92a.h"
#include "fwd/gfx92a/softmax_gfx92a.h"
#include "fwd/gfx92a/fwd_epilogue_gfx92a.h"
template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_Varlen, bool Return_softmax, bool Has_alibi, int Layout, typename Params>
inline __device__ void compute_attn_mha_1rowblock_gfx92a(const Params &params, const int bidb, const int bidh, const int m_block, const int warp_id) {
    using Element           = typename Kernel_traits::Element;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    using index_t           = std::conditional_t<Is_even_MN, int32_t, int64_t>;
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
    constexpr int WARP_K    = 32;

    // 获取当前 TG 处理的任务大小
    using BlockInfoType = flash::BlockInfo<Is_Varlen, false/*Is_Kvcache*/, false/*USE_BSHD_LAYOUT*/>;
    const BlockInfoType binfo(params, bidb);

    // 处理边界
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0 || bidh >= params.h/*border judgement*/) return;
    int warp_offset_in_seq_q = m_block * kBlockM + warp_id * WARP_M;

    // 分配 lds
    extern __shared__ Element smem[];
    Element* q_lds = (Element*)&(smem);
    Element* k_lds = q_lds;
    Element* v_lds = k_lds;

    // 计算当前任务沿着 seqlenKV 方向的 block 起始位置和终止位置
    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, flash::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // 计算数据跨度
    int seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, seqlen_o_stride;
    index_t row_offset_q, row_offset_k, row_offset_v, row_offset_o;
    int row_offset_lse;
    int headdim_split_id = 0;
    fwd_prologue_compute_offset<Layout, kBlockM, kBlockN, kHeadDim, kHeadDimV, kHeadDimV, 0/*SplitD*/, Is_even_MN, false/*Is_PaddingMask*/, Params, decltype(binfo), decltype(row_offset_q)>(
        seqlen_q_stride, seqlen_k_stride, seqlen_v_stride, seqlen_o_stride, row_offset_q, row_offset_k, row_offset_v, row_offset_o, row_offset_lse,
        headdim_split_id, bidb, bidh, bidh, m_block, n_block_min, binfo, params
    );
    #if 0
    if (int(threadIdx.x) == 0) {
        printf("bidb: %d | bidh: %d | actual_seqlen_q: %d | actual_seqlen_k: %d | n_block_max: %d | row_offset_q: %d | row_offset_k: %d | row_offset_v: %d | row_offset_o: %d | seqlen_q_stride: %d | seqlen_k_stride: %d | seqlen_v_stride: %d\n",
            bidb, bidh, binfo.actual_seqlen_q, binfo.actual_seqlen_k, n_block_max, row_offset_q, row_offset_k, row_offset_v, row_offset_o, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride);
    }
    #endif
    // 根据起始数据偏移量准备 Q/K/V 的 buffer resource 寄存器
    auto q_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
    auto k_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto v_ptr = prepare_for_matrix_load<kHeadDimV>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v);

    // attention 插件: Alibi
    float g_alibi;
    if constexpr (Has_alibi) {
        g_alibi = reinterpret_cast<ElementAccum*>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    }

    // attention 插件: Dropout
    unsigned long long rand_seed, rand_offset;
    uint32_t p_dropout_in_8bits_value;
    union_vec2_uint warp_idx_for_dropout;
    if constexpr (Is_dropout and Is_training) {
        rand_seed                = params.rand_seed;
        rand_offset              = params.rand_offset + ((bidb * params.h + bidh) << 6) + (threadIdx.x & 63);
        p_dropout_in_8bits_value = params.p_dropout_in_uint8_t & 0xffffffff;
        warp_idx_for_dropout.u32.x = 1 * m_block * (kBlockM / 32) /* 前面几个 block 累积的 warp 数目, 这里不直接填 WARP_M, 参照 NV 的写法*/ + warp_id/*当前 block 内的 warp id*/;
        if (Is_training and m_block == 0 and bidb == 0 and bidh == 0 and threadIdx.x == 0) {
            params.rng_state[0] = rand_seed;
            params.rng_state[1] = rand_offset;
        }
    }

    // 预取 Q 的数据到寄存器
    union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2];
    prefetch_q_to_vgpr_mls_ds_gfx92a<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(q_ptr, q_lds, q_reg, warp_id, seqlen_q_stride, Is_even_MN ? 0: binfo.actual_seqlen_q - m_block * kBlockM);

    // apply causal mask 的步骤和 no causal mask 的步骤分开算
    constexpr int n_masking_steps = (!Is_causal && !Is_local) ? 1: flash::ceil_div(kBlockM, kBlockN);

    // 是否做 prefetch K, PV 结束后, prefetch K 有风险
    constexpr bool PREFETCH_K = Is_even_MN and kHeadDim == 128 and kHeadDimV == 128;
    constexpr bool ALLOW_PREFETCH = (STAGES > 1); // 客观上决定是否开启 prefetch
    if constexpr (PREFETCH_K and ALLOW_PREFETCH) {
        if (n_block_min < n_block_max - n_masking_steps) {
            prefetch_k_to_lds_mls_ds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, Is_even_MN ? 0: binfo.actual_seqlen_k - n_block_min * kBlockN);
        }
    }

    /***************************************************************************************************************************/
    vec2_Accum<ElementAccum> scores_max[WARP_M / 32];
    vec2_Accum<ElementAccum> scores_sum[WARP_M / 32];
    vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4];
    attention_initialize<kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, 2/*M_MMAC_COUNT*/, ElementAccum>(scores_max, scores_sum, acc_o);
    /***************************************************************************************************************************/

    auto QK_GEMM_FUNC = &qk_gemm_prefetch_v_mls_ds_gfx92a<kHeadDim, kHeadDimV, kBlockM, kBlockN, kBlockK, WARP_M, WARP_N, STAGES, Element, ElementAccum, Is_even_MN>;
    auto PV_GEMM_FUNC = &pv_gemm_prefetch_k_mls_ds_gfx92a<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;
    auto PV_GEMM_FUNC_IN_MASK = &pv_gemm_prefetch_k_mls_ds_gfx92a<false, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;

    // Mainloop, 主循环, 不做 causal mask 的部分
    for (int n_block_loop = n_block_min; n_block_loop < n_block_max - n_masking_steps; ++n_block_loop) {

        flash::raise_priority();

        // 计算当前 loop 下 seqlen_kv 的数据起始位置和边界
        int warp_offset_in_seqkv    = n_block_loop * kBlockN;
        int warp_seqkv_limit = Is_even_MN ? 0: binfo.actual_seqlen_k - warp_offset_in_seqkv;

        // 预取 K 的数据到 lds
        if constexpr (not PREFETCH_K) {
            prefetch_k_to_lds_mls_ds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, warp_seqkv_limit);
        }

        // 准备 QK gemm 输出的寄存器
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (kBlockN / WARP_N)][4];

        // QK gemm
        QK_GEMM_FUNC(k_ptr, v_ptr, k_lds, v_lds, q_reg, s_reg, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);

        // Attention 变体 alibi
        if constexpr (Has_alibi) {
            apply_alibi_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, g_alibi);
        }

        // Attention 变体 local mask
        if constexpr (Is_local) {
            apply_mask_local_gfx938</*HasWSLeft=*/Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right);
        }

        // 对 QK 输出做 softmax, 以及重放缩 acc_o/scores_sum
        softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        // Attention 变体 dropout
        if constexpr (Is_dropout and Is_training) {
            warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
            apply_dropout<vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg, warp_seqkv_limit, 0, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
        }

        // softmax(QK) f32 -> f16
        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockN / 32)][4];
        convert_pk_type<WARP_M, kBlockN, Element, ElementAccum>(p_reg, s_reg);

        // 偏移 K 指针, 提前偏移准备预取 K
        *(uint64_t*)&k_ptr += kBlockN * params.k_row_stride * sizeof(Element);

        // PV gemm
        PV_GEMM_FUNC(v_ptr, k_ptr, v_lds, k_lds, p_reg, acc_o, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);

        // 偏移 V 指针
        *(uint64_t*)&v_ptr += kBlockN * params.v_row_stride * sizeof(Element);
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
        int warp_offset_in_seqkv    = n_block_loop * kBlockN;
        int warp_seqkv_limit = Is_even_MN ? 0: binfo.actual_seqlen_k - warp_offset_in_seqkv;

        // 预取 K 的数据到 lds
        prefetch_k_to_lds_mls_ds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, warp_seqkv_limit);

        // 准备 QK gemm 输出的寄存器
        vec4_Accum<ElementAccum> s_reg[(kBlockN / 32) * (WARP_M / 32)][4];

        // QK gemm
        QK_GEMM_FUNC(k_ptr, v_ptr, k_lds, v_lds, q_reg, s_reg, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);


        // 偏移 K 指针, 提前偏移准备预取 K
        *(uint64_t*)&k_ptr += kBlockN * params.k_row_stride * sizeof(Element);

        // Attention 变体 alibi
        if constexpr (Has_alibi) {
            apply_alibi_gfx938<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, g_alibi);
        }

        // Attention: mask, causal mask, local mask
        if constexpr (!Is_causal && !Is_local) {
            if constexpr (!Is_even_MN) { apply_mask_gfx92a<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_seqkv_limit); }
        } else {
            if constexpr (Is_causal) {
                apply_mask_causal_gfx92a<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q);
            } else if constexpr (Is_local) {
                apply_mask_local_gfx938</*HasWSLeft=*/Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right);
            }
        }

        // 对 QK 输出做 softmax, 以及重放缩 acc_o/scores_sum
        softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        // Attention 变体 dropout
        if constexpr (Is_dropout and Is_training) {
            warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
            apply_dropout<vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg, warp_seqkv_limit, 0, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
        }

        // softmax(QK) f32 -> f16
        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockN / 32)][4];
        convert_pk_type<WARP_M, kBlockN, Element, ElementAccum>(p_reg, s_reg);

        // PV gemm
        PV_GEMM_FUNC_IN_MASK(v_ptr, k_ptr, v_lds, k_lds, p_reg, acc_o, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);

        // 偏移 V 指针
        *(uint64_t*)&v_ptr += kBlockN * params.v_row_stride * sizeof(Element);
    }

    /**********************************************************************************************************************************/
    // Epilogue: rescale acco
    vec2_Accum<ElementAccum> lse[WARP_M / 32];
    fwd_epilugue_rescale_acco<WARP_M, kBlockK, kHeadDimV, Is_dropout && Is_training, ElementAccum>(acc_o, lse, scores_max, scores_sum, params.scale_softmax, params.rp_dropout);
    /**************************************************************************************************************************************/
    constexpr bool Is_Interleave = true;
    constexpr bool Is_Output_Interleave = false;
    int lane_id = threadIdx.x & 63;
    if (params.softmax_lse_ptr != nullptr) {
        fwd_epilogue_store_lse<WARP_M, Is_even_MN, false/*SplitD*/, Is_Interleave, ElementAccum>(lse, params.softmax_lse_ptr, row_offset_lse, warp_id, lane_id, 0, Is_even_MN ? 0: binfo.actual_seqlen_q - m_block * kBlockM);
    }
    /**************************************************************************************************************************************/
    Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;
    fwd_epilogue_store_output_mls_gfx92a<kHeadDimV, kBlockM, kBlockK, WARP_M, Is_even_MN, Is_Output_Interleave, false/*TcpSwizzle*/, Element, ElementAccum>(o_ptr, acc_o, m_block, warp_id, lane_id, seqlen_o_stride, binfo.actual_seqlen_q);
}


template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Is_Varlen, bool Return_softmax, bool Has_alibi, int Layout, typename Params>
inline __device__ void compute_attn_gfx92a(const Params &params) {
#if defined(__gfx92a__)
    constexpr bool Do_lpt = Is_causal;

    const int bidh = Do_lpt ? blockIdx.x: blockIdx.y;
    const int bidb = Do_lpt ? blockIdx.y: blockIdx.z;

    int warp_id_vec = threadIdx.x / 64;
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);
    int m_block = Do_lpt ? gridDim.z - 1 - blockIdx.z: blockIdx.x;

    flash::compute_attn_mha_1rowblock_gfx92a<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Is_Varlen, Return_softmax, Has_alibi, Layout, Flash_fwd_params>(params, bidb, bidh, m_block, warp_id);
#endif
}


template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Has_alibi, int Layout, typename Params>
inline __device__ void compute_attn_prefix_prefill_1rowblock_gfx92a(const Params &params, const int bidb, const int __bidh, const int m_block, const int warp_id) {
    using Element           = typename Kernel_traits::Element;
    using ElementAccum      = typename Kernel_traits::ElementAccum;
    using index_t           = typename Kernel_traits::index_t;
    constexpr int WARP_K    = 32;
    constexpr int kBlockM   = Kernel_traits::kBlockM;
    constexpr int kBlockN   = Kernel_traits::kBlockN;
    constexpr int kBlockK   = Kernel_traits::kBlockK;
    constexpr int kHeadDim  = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimVSplit;
    constexpr int kNWarps   = Kernel_traits::kNWarps;
    constexpr int WARP_M    = Kernel_traits::kWaveM;
    constexpr int WARP_N    = Kernel_traits::kWaveN;
    constexpr int STAGES    = Kernel_traits::STAGES;
    constexpr int WARP_NUM  = kBlockM / WARP_M;
    constexpr int SplitD    = Kernel_traits::SplitD;
    constexpr int kHeadDimVOrigin = Kernel_traits::kHeadDimV;

    // 获取 splitD 结果
    const int bidh = __bidh / SplitD;

    // 获取当前 TG 处理的任务大小
    // const flash::BlockInfo</*Varlen=*/!Is_even_MN, false/*Is_kvcache*/> binfo(params, bidb);
    flash::SafeDecodeBlockInfo binfo;
    binfo.set_params<Params, /*Is_Q_varlen=*/true, /*Is_K_Cumulative=*/false>(params, bidb);

    // 处理边界
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0) return;
    int warp_offset_in_seq_q = m_block * kBlockM + warp_id * WARP_M;

    // 分配 lds
    extern __shared__ Element smem[];
    Element* q_lds = (Element*)&(smem);
    Element* k_lds = q_lds;
    Element* v_lds = k_lds;

    // 计算当前任务沿着 seqlenKV 方向的 block 起始位置和终止位置
    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, flash::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // 计算数据跨度
    int seqlen_q_stride = params.q_row_stride;
    int seqlen_k_stride = params.k_row_stride;
    int seqlen_v_stride = params.v_row_stride;
    int seqlen_o_stride = params.o_row_stride;
    int64_t row_offset_q, row_offset_k, row_offset_v, row_offset_o;
    int64_t row_offset_lse;
    // 获取页表信息
    const int page_block_size = params.page_block_size;
    int *block_table = params.block_table + bidb * params.block_table_batch_stride;
    const int kv_start = n_block_min * kBlockN;
    const int block_table_idx = kv_start / page_block_size;
    const int block_table_offset = kv_start - block_table_idx * page_block_size;
    if constexpr (Layout == 1) { /*bshd layout, lse is num_heads, total_q*/
        row_offset_q = (binfo.sum_s_q + m_block * kBlockM) * int64_t(seqlen_q_stride) + params.q_head_stride * bidh;
        row_offset_k = int64_t(block_table[block_table_idx]) * int64_t(params.k_batch_stride) + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
        row_offset_v = int64_t(block_table[block_table_idx]) * int64_t(params.v_batch_stride) + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
        row_offset_lse = bidh * params.total_q + binfo.sum_s_q + m_block * kBlockM;
        row_offset_o = binfo.sum_s_q * int64_t(params.o_head_stride) * params.h + params.o_head_stride * bidh + m_block * kBlockM * seqlen_o_stride;
    }
    #if 0
    if (int(threadIdx.x) == 0) {
        printf("bidb: %d | bidh: %d | actual_seqlen_q: %d | actual_seqlen_k: %d | n_block_max: %d | row_offset_q: %ld | row_offset_k: %ld | row_offset_v: %ld | row_offset_o: %ld | seqlen_q_stride: %d | seqlen_k_stride: %d | seqlen_v_stride: %d\n",
            bidb, bidh, binfo.actual_seqlen_q, binfo.actual_seqlen_k, n_block_max, row_offset_q, row_offset_k, row_offset_v, row_offset_o, seqlen_q_stride, seqlen_k_stride, seqlen_v_stride);
    }
    #endif

    // 根据起始数据偏移量准备 Q/K/V 的资源寄存器
    auto q_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q);
    auto k_ptr = prepare_for_matrix_load<kHeadDim>(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k);
    auto v_ptr = prepare_for_matrix_load<kHeadDimV>(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v);

    // attention 变体: Alibi
    float g_alibi;
    if constexpr (Has_alibi) {
        g_alibi = reinterpret_cast<ElementAccum*>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    }

    // attention 插件: Dropout
    unsigned long long rand_seed, rand_offset;
    uint32_t p_dropout_in_8bits_value;
    union_vec2_uint warp_idx_for_dropout;
    if constexpr (Is_dropout and Is_training) {
        rand_seed                = params.rand_seed;
        rand_offset              = params.rand_offset + ((bidb * params.h + bidh) << 6) + (threadIdx.x & 63);
        p_dropout_in_8bits_value = params.p_dropout_in_uint8_t & 0xffffffff;
        warp_idx_for_dropout.u32.x = 1 * m_block * (kBlockM / 32) /* 前面几个 block 累积的 warp 数目, 这里不直接填 WARP_M, 参照 NV 的写法*/ + warp_id/*当前 block 内的 warp id*/;
        if (Is_training and m_block == 0 and bidb == 0 and bidh == 0 and threadIdx.x == 0) {
            params.rng_state[0] = rand_seed;
            params.rng_state[1] = rand_offset;
        }
    }

    // 预取 Q 的数据到寄存器
    union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2];
    prefetch_q_to_vgpr_mls_ds_gfx92a<kHeadDim, kBlockM, kBlockK, WARP_M, Element, Is_even_MN>(q_ptr, q_lds, q_reg, warp_id, seqlen_q_stride, Is_even_MN ? 0: binfo.actual_seqlen_q - m_block * kBlockM);

    // apply causal mask 的步骤和 no causal mask 的步骤分开算
    // prefix prefill 目前没分开算, 明确边界的情况下也可以分开算, 性能会有提升
    int n_masking_steps = (!Is_causal && !Is_local) ? 1: min(n_block_max, flash::ceil_div(kBlockM, kBlockN) + 1);

    // 是否做 prefetch K, PV 结束后, prefetch K 有风险
    constexpr bool PREFETCH_K = Is_even_MN and kHeadDim == 128 and kHeadDimV == 128;
    constexpr bool ALLOW_PREFETCH = (STAGES > 1); // 客观上决定是否开启 prefetch
    if constexpr (PREFETCH_K and ALLOW_PREFETCH) {
        if (n_block_min < n_block_max - n_masking_steps) {
            prefetch_k_to_lds_mls_ds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, Is_even_MN ? 0: binfo.actual_seqlen_k - n_block_min * kBlockN);
        }
    }

    /***************************************************************************************************************************/
    vec2_Accum<ElementAccum> scores_max[WARP_M / 32];
    vec2_Accum<ElementAccum> scores_sum[WARP_M / 32];
    vec4_Accum<ElementAccum> acc_o[(kHeadDimV / kBlockK) * (WARP_M / 32) * (kBlockK / 32)][4];
    attention_initialize<kHeadDimV / kBlockK, WARP_M / 32, kBlockK / 32, 2/*M_MMAC_COUNT*/, ElementAccum>(scores_max, scores_sum, acc_o);
    /***************************************************************************************************************************/

    auto QK_GEMM_FUNC = &qk_gemm_prefetch_v_mls_ds_gfx92a<kHeadDim, kHeadDimV, kBlockM, kBlockN, kBlockK, WARP_M, WARP_N, STAGES, Element, ElementAccum, Is_even_MN>;
    auto PV_GEMM_FUNC = &pv_gemm_prefetch_k_mls_ds_gfx92a<PREFETCH_K, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;
    auto PV_GEMM_FUNC_IN_MASK = &pv_gemm_prefetch_k_mls_ds_gfx92a<false, kHeadDim, kHeadDimV, kBlockM, kBlockK, kBlockN, WARP_M, kBlockK, STAGES, Element, ElementAccum, Is_even_MN>;

    // Mainloop, 主循环, 不做 causal mask 的部分
    for (int n_block_loop = n_block_min; n_block_loop < n_block_max - n_masking_steps; ++n_block_loop) {

        // 计算当前 loop 下 seqlen_kv 的数据起始位置和边界
        int warp_offset_in_seqkv    = n_block_loop * kBlockN;
        int warp_seqkv_limit = Is_even_MN ? 0: binfo.actual_seqlen_k - warp_offset_in_seqkv;

        // 预取 K 的数据到 lds
        if constexpr (not PREFETCH_K) {
            prefetch_k_to_lds_mls_ds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, warp_seqkv_limit);
        }

        // 准备 QK gemm 输出的寄存器
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (kBlockN / WARP_N)][4];

        // QK gemm
        QK_GEMM_FUNC(k_ptr, v_ptr, k_lds, v_lds, q_reg, s_reg, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);


        // Attention 变体 alibi
        if constexpr (Has_alibi) {
            apply_alibi_gfx92a<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, g_alibi);
        }

        // Attention: mask, causal mask, local mask
        if constexpr (Is_local) {
            apply_mask_local_gfx92a</*HasWSLeft=*/Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right);
        }

        // 对 QK 输出做 softmax, 以及重放缩 acc_o/scores_sum
        softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        // Attention 变体 dropout
        if constexpr (Is_dropout and Is_training) {
            warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
            apply_dropout<vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg, warp_seqkv_limit, 0, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
        }

        // softmax(QK) f32 -> f16
        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockN / 32)][4];
        convert_pk_type<WARP_M, kBlockN, Element, ElementAccum>(p_reg, s_reg);

        const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;

        if constexpr (PREFETCH_K) {
            *(int64_t*)&k_ptr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);
        }

        // PV gemm
        PV_GEMM_FUNC(v_ptr, k_ptr, v_lds, k_lds, p_reg, acc_o, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);
        if constexpr (not PREFETCH_K) {
            *(int64_t*)&k_ptr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);
        }
        *(int64_t*)&v_ptr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);
    }

    // prefetch K 的话, 最后一次多取了一段 K, 为了不影响后续的操作, 需要同步等待
    if constexpr (PREFETCH_K) { buffer_load_lds_dwordx1_wait<0>(); }

    /***************************************************************************************************************************/
    // Rest loop, 做 causal mask 的部分
    int n_block_loop = max(n_block_max - n_masking_steps, n_block_min);
    // #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, ++n_block_loop) {
        // 计算当前 loop 下 seqlen_kv 的数据起始位置和边界
        int warp_offset_in_seqkv    = n_block_loop * kBlockN;
        int warp_seqkv_limit = Is_even_MN ? 0: binfo.actual_seqlen_k - warp_offset_in_seqkv;

        // 预取 K 的数据到 lds
        if constexpr (true) {
            prefetch_k_to_lds_mls_ds<kHeadDim, kBlockN, kBlockK, WARP_NUM, WARP_N, Element, Is_even_MN>(k_ptr, k_lds, warp_id, seqlen_k_stride, warp_seqkv_limit);
        }

        // 准备 QK gemm 输出的寄存器
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (kBlockN / WARP_N)][4];

        // QK gemm
        QK_GEMM_FUNC(k_ptr, v_ptr, k_lds, v_lds, q_reg, s_reg, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);


        // Attention 变体 alibi
        if constexpr (Has_alibi) {
            apply_alibi_gfx92a<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, g_alibi);
        }

        // Attention: mask, causal mask, local mask
        if constexpr (!Is_causal && !Is_local) {
            if constexpr (!Is_even_MN) { apply_mask_gfx92a<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_seqkv_limit); }
        } else if constexpr (Is_causal) {
            apply_mask_causal_gfx92a<vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q);
        } else if constexpr (Is_local) {
            apply_mask_local_gfx92a</*HasWSLeft=*/Is_local, vec4_Accum<ElementAccum>, WARP_M, kBlockN>(s_reg, warp_offset_in_seqkv, binfo.actual_seqlen_k, warp_offset_in_seq_q, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right);
        }

        // 对 QK 输出做 softmax, 以及重放缩 acc_o/scores_sum
        softmax_rescale_o<false, Is_causal || Is_local, vec4_Accum<ElementAccum>, vec2_Accum<ElementAccum>, kHeadDimV, kBlockK, WARP_M, kBlockN>(s_reg, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        // Attention 变体 dropout
        if constexpr (Is_dropout and Is_training) {
            warp_idx_for_dropout.u32.y = n_block_loop * (kBlockN / WARP_N);
            apply_dropout<vec4_Accum<ElementAccum>, WARP_M, kBlockN, kNWarps, Is_even_MN>(s_reg, warp_seqkv_limit, 0, rand_seed, rand_offset, p_dropout_in_8bits_value, warp_idx_for_dropout, params.dropout_debug_count);
        }

        // softmax(QK) f32 -> f16
        union_vec2_f16x2<Element> p_reg[(WARP_M / 32) * (kBlockN / 32)][4];
        convert_pk_type<WARP_M, kBlockN, Element, ElementAccum>(p_reg, s_reg);

        const int block_table_idx_cur     = n_block_loop * kBlockN / params.page_block_size;
        const int block_table_offset_cur  = n_block_loop * kBlockN - block_table_idx_cur * params.page_block_size;
        const int block_table_idx_next    = min(n_block_max - 1, n_block_loop + 1) * kBlockN / params.page_block_size;
        const int block_table_offset_next = min(n_block_max - 1, n_block_loop + 1) * kBlockN - block_table_idx_next * params.page_block_size;
        const int table_diff              = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
        const int offset_diff             = block_table_offset_next - block_table_offset_cur;

        // PV gemm
        PV_GEMM_FUNC_IN_MASK(v_ptr, k_ptr, v_lds, k_lds, p_reg, acc_o, warp_id, seqlen_k_stride, seqlen_v_stride, warp_seqkv_limit);

        // 偏移 V 指针
        *(int64_t*)&k_ptr += (int64_t(table_diff) * int64_t(params.k_batch_stride) + offset_diff * int64_t(params.k_row_stride)) * sizeof(Element);
        *(int64_t*)&v_ptr += (int64_t(table_diff) * int64_t(params.v_batch_stride) + offset_diff * int64_t(params.v_row_stride)) * sizeof(Element);
    }

    /**********************************************************************************************************************************/
    // Epilogue: rescale acco
    vec2_Accum<ElementAccum> lse[WARP_M / 32];
    fwd_epilugue_rescale_acco<WARP_M, kBlockK, kHeadDimV, Is_dropout && Is_training, ElementAccum>(acc_o, lse, scores_max, scores_sum, params.scale_softmax, params.rp_dropout);
    /**************************************************************************************************************************************/
    constexpr bool Is_Interleave = true;
    int lane_id = threadIdx.x & 63;
    if (params.softmax_lse_ptr != nullptr) {
        fwd_epilogue_store_lse<WARP_M, Is_even_MN, false/*SplitD*/, Is_Interleave, ElementAccum>(lse, params.softmax_lse_ptr, row_offset_lse, warp_id, lane_id, 0, binfo.actual_seqlen_q - m_block * kBlockM);
    }
    /**************************************************************************************************************************************/
    Element* o_ptr = reinterpret_cast<Element *>(params.o_ptr) + row_offset_o;
    fwd_epilogue_store_output_mls_gfx92a<kHeadDimV, kBlockM, kBlockK, WARP_M, Is_even_MN, false/*Is_Interleave*/, false/*TcpSwizzle*/, Element, ElementAccum>(o_ptr, acc_o, m_block, warp_id, lane_id, seqlen_o_stride, binfo.actual_seqlen_q);
}


template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Has_alibi, int Layout, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_prefix_prefill_gfx938_kernel(const Params params) {

#if defined(__gfx938__) || defined(__gfx946__)
    const int bidh = blockIdx.x;

    const int bidb = blockIdx.y;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);

    int m_block = gridDim.z - 1 - blockIdx.z;
    flash::compute_attn_prefix_prefill_1rowblock_gfx938<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Return_softmax, Has_alibi, Layout, Flash_fwd_params>(params, bidb, bidh, m_block, warp_id);
#endif
}


template<typename Kernel_traits, bool Is_training, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Has_alibi, int Layout, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_prefix_prefill_gfx92a_kernel(const Params params) {

#if defined(__gfx92a__)
    const int bidh = blockIdx.x;

    const int bidb = blockIdx.y;

    int warp_id_vec = threadIdx.x / 64; // warp id in a block
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec);

    int m_block = gridDim.z - 1 - blockIdx.z;
    flash::compute_attn_prefix_prefill_1rowblock_gfx92a<Kernel_traits, Is_training, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Return_softmax, Has_alibi, Layout, Flash_fwd_params>(params, bidb, bidh, m_block, warp_id);
#endif
}


} // namespace flash
