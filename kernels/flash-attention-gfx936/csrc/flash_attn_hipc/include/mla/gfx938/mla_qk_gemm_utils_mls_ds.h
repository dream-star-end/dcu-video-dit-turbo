#pragma once
#include "intrinsic_mls_ds.h"



template<int kHeadDim, int kBlockM, int kBlockK, int WARP_M, typename Element, bool Is_even_MN>
__forceinline__ __device__ void  prefetch_q_to_lds_mls_ds_576_512(
        vec4_uint q_ptr,
        Element* q_lds,
        int warp_id,
        int seqlen_q_stride,
        int max_seq_q_offset=0) {

    // 编译期可知变量
    constexpr int WARP_NUM        = kBlockM / WARP_M;
    constexpr int Q_LDS_LOAD_NUM  = (kBlockM * kBlockK) / (16 * 32);
    constexpr int Q_LOAD_REQUESTS = Q_LDS_LOAD_NUM / WARP_NUM;
    constexpr int ELEMENT_BYTES   = sizeof(Element);

    // LDS 起始地址
    int q_lds_base = reinterpret_cast<size_t>(q_lds);

    // MLS
    vec4_uint q_srsrc;
    q_srsrc[2] = seqlen_q_stride;
    q_srsrc[3] = 0;

    int stage_id = 0;
    {
        int k_loop = 0;
        *(uint64_t*)&q_srsrc = VA_LIMIT_BITS(*(uint64_t*)&q_ptr + (k_loop * kBlockK + warp_id * 16 * seqlen_q_stride) * ELEMENT_BYTES);
        if constexpr (true) {
            int nm_filter =  inline_min_max<0,16>(16 * warp_id + 16 - max_seq_q_offset);
            q_srsrc[3] = max_seq_q_offset % kBlockM == 0 ? 0: nm_filter << 8;
        }
        
        int lds_offset = (stage_id * kBlockM * kBlockK + warp_id * 16 * 32) * ELEMENT_BYTES;
        flash::wait_all_warp_arrived(); // pvgemm 完成后会发射q,k的预取，避免有的warp还没完成，即规避读V写Q/K，造成数据覆盖
        inline_matrix_load_32x16_b16_lds_trans<0, 1>(q_lds, q_srsrc, lds_offset, 0);
    }
}

template<int kHeadDim, int kBlockN, int kBlockK, int WARP_NUM, int WARP_N, typename Element, bool Is_even_MN>
__forceinline__ __device__ void prefetch_k_to_lds_mls_ds_576_512(
        vec4_uint k_ptr,
        Element* k_lds,
        int warp_id,
        int seqlen_k_stride,
        int max_seq_k_offset=0) {
    constexpr int kHeadDim_OPT = (kHeadDim == 576) ? 64 : kHeadDim;
    constexpr int ELEMENT_BYTES = sizeof(Element);
    constexpr int WARP_NUM_M = 2; 
    constexpr int WARP_NUM_N = 4;
    int warp_id_m = warp_id / WARP_NUM_N;
    int warp_id_n = warp_id % WARP_NUM_N;

    int stage_id = 0;
    int n_loop   = 0;
    int k_loop   = 0;

    // MLS
    vec4_uint k_srsrc;
    k_srsrc[2] = seqlen_k_stride;
    if constexpr (true) {
        int nm_filter = inline_min_max<0,16>(n_loop * WARP_N + 16 * warp_id_n + 16 - max_seq_k_offset);        
        *(uint64_t*)&k_srsrc = VA_LIMIT_BITS(*(uint64_t*)&k_ptr + (n_loop * WARP_N * seqlen_k_stride + warp_id_m * 32 + warp_id_n * 16 * seqlen_k_stride + k_loop * 32 * WARP_NUM_M) * ELEMENT_BYTES);
        k_srsrc[3] = max_seq_k_offset % kBlockN == 0x0 ? 0: nm_filter << 8;
    }
    int lds_offset = (stage_id * WARP_N * kHeadDim_OPT + warp_id * 32 * 16) * ELEMENT_BYTES;
    flash::wait_all_warp_arrived();
    inline_matrix_load_32x16_b16_lds_trans<0, 0>(k_lds, k_srsrc, lds_offset, 0);

}