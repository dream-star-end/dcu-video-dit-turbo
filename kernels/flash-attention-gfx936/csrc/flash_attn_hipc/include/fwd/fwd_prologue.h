

template<int Layout, int kBlockM, int kBlockN, int kHeadDim, int kHeadDimV, int kHeadDimVOrigin, int SplitD, bool Is_even_MN, bool Is_PaddingMask, typename Params, typename BlockInfo, typename index_t=int, bool IsVarlen=!Is_even_MN>
__forceinline__ __device__ void fwd_prologue_compute_offset(
    int &seqlen_q_stride,
    int &seqlen_k_stride,
    int &seqlen_v_stride,
    int &seqlen_o_stride,
    index_t &row_offset_q,
    index_t &row_offset_k,
    index_t &row_offset_v,
    index_t &row_offset_o,
    int &row_offset_lse,
    int &headdim_split_id,
    const int bidb,
    const int bidh,
    const int __bidh,
    const int m_block,
    const int n_block_min,
    const BlockInfo binfo,
    const Params params
) {
    seqlen_q_stride = (Layout == 1) ? params.q_row_stride: params.q_row_stride;
    seqlen_k_stride = (Layout == 1) ? params.k_row_stride: params.k_row_stride;
    seqlen_v_stride = (Layout == 1) ? params.v_row_stride: params.v_row_stride;
    seqlen_o_stride = (Layout == 1) ? params.o_row_stride: params.o_row_stride;
    if constexpr (Is_even_MN) {
        row_offset_q = binfo.q_offset1(params.q_batch_stride, params.q_head_stride, bidb) + binfo.q_offset2(params.q_head_stride, bidh) + m_block * kBlockM * seqlen_q_stride;
        row_offset_k = binfo.k_offset1(params.k_batch_stride, params.k_head_stride, bidb) + binfo.k_offset2(params.k_head_stride, bidh / params.h_h_k_ratio) + n_block_min * kBlockN * seqlen_k_stride;
        row_offset_v = binfo.k_offset1(params.v_batch_stride, params.v_head_stride, bidb) + binfo.k_offset2(params.v_head_stride, bidh / params.h_h_k_ratio) + n_block_min * kBlockN * seqlen_v_stride;
        row_offset_o = binfo.q_offset1(params.o_batch_stride, params.o_head_stride, bidb) + binfo.q_offset2(params.o_head_stride, bidh) + m_block * kBlockM * seqlen_o_stride;
        row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    } else {
        if constexpr (Is_PaddingMask) {
            if (params.padding_mask != nullptr or params.attn_mask != nullptr) {
                row_offset_q = binfo.q_offset1(params.q_batch_stride, params.q_head_stride, bidb) + binfo.q_offset2(params.q_head_stride, bidh) + m_block * kBlockM * seqlen_q_stride;
                row_offset_k = binfo.k_offset1(params.k_batch_stride, params.k_head_stride, bidb) + binfo.k_offset2(params.k_head_stride, bidh / params.h_h_k_ratio) + n_block_min * kBlockN * seqlen_k_stride;
                row_offset_v = binfo.k_offset1(params.v_batch_stride, params.v_head_stride, bidb) + binfo.k_offset2(params.v_head_stride, bidh / params.h_h_k_ratio) + n_block_min * kBlockN * seqlen_v_stride;
                row_offset_o = binfo.q_offset1(params.o_batch_stride, params.o_head_stride, bidb) + binfo.q_offset2(params.o_head_stride, bidh) + m_block * kBlockM * seqlen_o_stride;
            }
        } else {
            if constexpr (!IsVarlen) {
                // H3 is fixed-length even when its final sequence tile is
                // uneven.  Its wrapper guarantees MHA (ratio=1), so keep the
                // address arithmetic entirely in the validated OffsetT.
                row_offset_q = index_t(bidb) * index_t(params.q_batch_stride) + index_t(bidh) * index_t(params.q_head_stride) + index_t(m_block * kBlockM) * index_t(seqlen_q_stride);
                row_offset_k = index_t(bidb) * index_t(params.k_batch_stride) + index_t(bidh) * index_t(params.k_head_stride) + index_t(n_block_min * kBlockN) * index_t(seqlen_k_stride);
                row_offset_v = index_t(bidb) * index_t(params.v_batch_stride) + index_t(bidh) * index_t(params.v_head_stride) + index_t(n_block_min * kBlockN) * index_t(seqlen_v_stride);
                row_offset_o = index_t(bidb) * index_t(params.o_batch_stride) + index_t(bidh) * index_t(params.o_head_stride) + index_t(m_block * kBlockM) * index_t(seqlen_o_stride);
                row_offset_lse = 0;
            } else if (params.cu_seqlens_q == nullptr) {
                row_offset_q = binfo.q_offset1(params.q_batch_stride, params.q_head_stride, bidb) + binfo.q_offset2(params.q_head_stride, bidh) + m_block * kBlockM * seqlen_q_stride;
                row_offset_k = binfo.k_offset1(params.k_batch_stride, params.k_head_stride, bidb) + binfo.k_offset2(params.k_head_stride, bidh / params.h_h_k_ratio) + n_block_min * kBlockN * seqlen_k_stride;
                row_offset_v = binfo.k_offset1(params.v_batch_stride, params.v_head_stride, bidb) + binfo.k_offset2(params.v_head_stride, bidh / params.h_h_k_ratio) + n_block_min * kBlockN * seqlen_v_stride;
                row_offset_o = binfo.q_offset1(params.o_batch_stride, params.o_head_stride, bidb) + binfo.q_offset2(params.o_head_stride, bidh) + m_block * kBlockM * seqlen_o_stride;
                row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
            } else {
                // Varlen
                if constexpr (Layout == 1) { /*bshd layout, lse is num_heads, total_q*/
                    row_offset_q = index_t(binfo.sum_s_q + m_block * kBlockM) * index_t(seqlen_q_stride) + params.q_head_stride * bidh;
                    row_offset_k = index_t(binfo.sum_s_k + n_block_min * kBlockN) * index_t(seqlen_k_stride) + bidh / params.h_h_k_ratio * params.k_head_stride;
                    row_offset_v = index_t(binfo.sum_s_k + n_block_min * kBlockN) * index_t(seqlen_v_stride) + bidh / params.h_h_k_ratio * params.v_head_stride;
                    row_offset_o = index_t(binfo.sum_s_q) * index_t(params.o_head_stride * params.h) + params.o_head_stride * bidh + m_block * kBlockM * seqlen_o_stride;
                    row_offset_lse = bidh * params.total_q + binfo.sum_s_q + m_block * kBlockM;
                } else { /*bhsd layout, lse is num_heads, total_q*/
                    row_offset_q = binfo.q_offset1(params.q_batch_stride, params.q_head_stride, bidb) + binfo.q_offset2(params.q_head_stride, bidh) + m_block * kBlockM * seqlen_q_stride;
                    row_offset_k = binfo.k_offset1(params.k_batch_stride, params.k_head_stride, bidb) + binfo.k_offset2(params.k_head_stride, bidh / params.h_h_k_ratio) + n_block_min * kBlockN * seqlen_k_stride;
                    row_offset_v = binfo.k_offset1(params.v_batch_stride, params.v_head_stride, bidb) + binfo.k_offset2(params.v_head_stride, bidh / params.h_h_k_ratio) + n_block_min * kBlockN * seqlen_v_stride;
                    row_offset_o = binfo.q_offset1(params.o_batch_stride, params.o_head_stride, bidb) + binfo.q_offset2(params.o_head_stride, bidh) + m_block * kBlockM * seqlen_o_stride;
                    row_offset_lse = bidh * params.total_q + binfo.sum_s_q + m_block * kBlockM;
                }
            }
        }
    }
    if constexpr (SplitD > 1) {
        headdim_split_id = __bidh % SplitD;
        row_offset_v += headdim_split_id * kHeadDimV;
        row_offset_o += headdim_split_id * kHeadDimV;
    }
}
