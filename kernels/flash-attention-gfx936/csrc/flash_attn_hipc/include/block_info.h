/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Varlen=true, bool Is_Kvcache=false, bool USE_BSHD_LAYOUT = false>
struct BlockInfo {

    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb)
        : sum_s_q((!Varlen || params.cu_seqlens_q == nullptr) ? -1 : params.cu_seqlens_q[bidb])
        , sum_s_k((!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative) ? -1 : params.cu_seqlens_k[bidb])
        , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr || Is_Kvcache ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        , seqlen_k_cache(!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb]))
        , actual_seqlen_k(seqlen_k_cache/* + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)*/)
        , nheads(params.h)
        , nheads_k(params.h_k)
        , leftpad_k(params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])
        {
        }

    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride + leftpad_k * row_stride : uint32_t(sum_s_k + leftpad_k) * row_stride;
    }

    inline __device__  int q_offset1(const int batch_stride, const int row_stride, const int bidb) const {
        return sum_s_q == -1 ? bidb * batch_stride : (USE_BSHD_LAYOUT ? uint32_t(sum_s_q) * row_stride : uint32_t(sum_s_q) * row_stride * nheads);
    }

    inline __device__  int k_offset1(const int batch_stride, const int row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride : (USE_BSHD_LAYOUT ? uint32_t(sum_s_k) * row_stride : uint32_t(sum_s_k) * row_stride * nheads_k);
    }

    inline __device__  int k_offset1_write(const int batch_stride, const int row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride : (USE_BSHD_LAYOUT ? uint32_t(sum_s_k) * row_stride : uint32_t(sum_s_k) * row_stride * nheads);
    }

    inline __device__  int q_offset2(const int head_stride, const int bidh) const {
        return (USE_BSHD_LAYOUT || sum_s_q == -1) ? bidh * head_stride : uint32_t(actual_seqlen_q) * head_stride * bidh;
    }

    inline __device__  int k_offset2(const int head_stride, const int bidh) const {
        return (USE_BSHD_LAYOUT || sum_s_k == -1) ? bidh * head_stride : uint32_t(actual_seqlen_k) * head_stride *bidh;
    }

    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
    const int leftpad_k;
    const int seqlen_k_cache;
    int actual_seqlen_k;
    const int nheads;
    const int nheads_k;

};


// Simplified blockinfo for tranditional varlen fwd inference
template<bool USE_BSHD_LAYOUT=false>
struct SimplifyBlockInfo {

    template<typename Params>
    __device__ SimplifyBlockInfo(const Params &params, const int bidb)
        : sum_s_q(params.cu_seqlens_q[bidb])
        , sum_s_k(params.cu_seqlens_k[bidb])
        , actual_seqlen_q(params.cu_seqlens_q[bidb + 1] - sum_s_q)
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        , seqlen_k_cache((params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb]))
        , actual_seqlen_k(seqlen_k_cache/* + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)*/)
        , nheads(params.h)
        , nheads_k(params.h_k)
        // , leftpad_k(0)
        {
        }

    inline __device__  int q_offset1(const int batch_stride, const int row_stride, const int bidb) const {
        return sum_s_q == -1 ? bidb * batch_stride : (USE_BSHD_LAYOUT ? uint32_t(sum_s_q) * row_stride : uint32_t(sum_s_q) * row_stride * nheads);
    }

    inline __device__  int k_offset1(const int batch_stride, const int row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride : (USE_BSHD_LAYOUT ? uint32_t(sum_s_k) * row_stride : uint32_t(sum_s_k) * row_stride * nheads_k);
    }

    inline __device__  int q_offset2(const int head_stride, const int bidh) const {
        return (USE_BSHD_LAYOUT || sum_s_q == -1) ? bidh * head_stride : uint32_t(actual_seqlen_q) * head_stride * bidh;
    }

    inline __device__  int k_offset2(const int head_stride, const int bidh) const {
        return (USE_BSHD_LAYOUT || sum_s_k == -1) ? bidh * head_stride : uint32_t(actual_seqlen_k) * head_stride *bidh;
    }

    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
    // const int leftpad_k;
    const int seqlen_k_cache;
    int actual_seqlen_k;
    const int nheads;
    const int nheads_k;

};


////////////////////////////////////////////////////////////////////////////////////////////////////

struct SafeDecodeBlockInfo {

    __device__ SafeDecodeBlockInfo() = default;

    template<typename Params, bool Is_Q_varlen, bool Is_K_Cumulative>
    __device__ void set_params(const Params &params, const int bidb) {
        // process Q
        if constexpr (Is_Q_varlen) { // Is_Q_varlen also means Is_Q_Cumulative = true
            this->sum_s_q = params.cu_seqlens_q[bidb];
            this->actual_seqlen_q = params.cu_seqlens_q[bidb + 1] - this->sum_s_q;
        } else {
            this->actual_seqlen_q = params.seqlen_q;
        }
        // process KV
        if constexpr (Is_K_Cumulative) {
            this->sum_s_k = params.cu_seqlens_k[bidb];
            this->actual_seqlen_k = params.cu_seqlens_k[bidb + 1] - sum_s_k;
        } else {
            this->actual_seqlen_k = params.cu_seqlens_k[bidb];
        }
    }

    int sum_s_q;
    int sum_s_k;
    int actual_seqlen_q;
    int actual_seqlen_k;
};

}  // namespace flash
