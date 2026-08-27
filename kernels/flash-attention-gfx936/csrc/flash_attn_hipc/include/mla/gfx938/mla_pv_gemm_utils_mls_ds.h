#pragma once // prepare for prefetch V in qk gemm
#include "intrinsic_mls_ds.h"



template<int kHeadDim, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, typename Element, bool Is_even_MN>
__forceinline__ __device__ void prefetch_v_to_lds_mls_ds_576_512(
        vec4_uint v_ptr,
        Element* v_lds,
        int warp_id,
        int seqlen_v_stride,
        int max_seq_kv_offset=0) {

    constexpr int ELEMENT_BYTES = sizeof(Element);
    constexpr int WARP_NUM = kBlockM * kBlockN / (WARP_M * WARP_N);
    constexpr int WARP_K   = 32;
    constexpr int kHeadDim_OPT = 256;   // 32x32 x WARP_NUM x 2B x stage == 32K

    // MLS
    int n_load = 0;
    vec4_uint v_srsrc;
    *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_ptr + (n_load * WARP_NUM * 32 + warp_id * 32) * ELEMENT_BYTES);
    v_srsrc[2] = seqlen_v_stride;
    if constexpr (true) {
        int nm_filter = inline_min_max<0, 32>(0 * WARP_K + 32 - max_seq_kv_offset);
        v_srsrc[3] = max_seq_kv_offset % kBlockK == 0 ? 0: nm_filter << 8;
    }
    int lds_stage_id = 0;
    int lds_offset = (lds_stage_id * WARP_K * kHeadDim_OPT + warp_id * 32 * 32) * ELEMENT_BYTES;

    flash::wait_all_warp_arrived(); // 防止写 v lds 和读 q lds k lds 冲突, qk 可能有的 warp 没结束
    inline_matrix_load_32x32_b16_lds<0, 1>(v_lds, v_srsrc, lds_offset, 0);
    __builtin_amdgcn_sched_barrier(0);
}
