#pragma once // prepare for prefetch V in qk gemm
#include "intrinsic.h"
#include "fwd/utils.h"

template<int kHeadDim, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int WARP_K, int stage_id, int WARP_NUM, typename Element, int STAGES>
__forceinline__ __device__ void  kvcache_prefetch_v_to_lds(
        vec4_uint gV,
        Element* v_lds,
        int WARP_ID,
        int vcache_seqlen_stride,
        int max_seq_kv_offset=0) {

    // 预先计算一些公共表达式
    int lane_id       = threadIdx.x & 63;
    int laneid_shfl_2 = lane_id >> 2; // 0 ~ 15, 4 个线程读取一行
    int laneid_shfl_3 = lane_id >> 3; // 0 ~ 7,  8 个线程读取一行
    int laneid_shfl_4 = lane_id >> 4; // 0 ~ 3,  16 个线程读取一行
    int laneid_shfl_5 = lane_id >> 5; // 0 ~ 1,  lds 读取时, 8x32的数据按照线程 [0, 16, 0, 16, 32, 48, 32, 48] 来读取, 每 32 个线程读取一个 4x32
    constexpr int NEXT_DWORD_OFFSET = 32; // 8x32 的数据, 一个 wave 每个线程读 4 个 half, 即 2 个 dword, 使用 ds_read2_b32 指令, 按照上面的读取方式, 第二个 dword 偏移 32 个 dword

#if defined(USE_BUFFER_LOAD_DWORDX4)
    constexpr int READ_ONCE_LINES    = 16;                                 // 一个 warp 一次读几行数据, loadx2, 每行 32 个元素需要 32 / (4) = 8 个线程
    constexpr int READ_ONCE_COUNT    = READ_ONCE_LINES * 32;               // 一个 warp 一次 load 多少数据
    constexpr int V_LDS_LOAD_NUM     = kBlockN * WARP_K / READ_ONCE_COUNT; // 整个 workgroup 要发多少 load 指令
    constexpr int V_LOAD_REQUESTS    = V_LDS_LOAD_NUM;                     // 每个 warp 要发多少 load 指令
    constexpr int READ_ELEMENT_COUNT = 8;                                  // 每个线程一次读取几个 half
    int v_lane_headdim_n_idx         = lane_id & 3;                        // 当前 lane 负责这个 warp 的第几个 dwordx2
    int base                         = laneid_shfl_2 & 0xc;                // 第几个4线程组的最小id
    int tail                         = laneid_shfl_2 & 0x3  ;              // 线程组中的第几个线程
    int v_lane_seq_k_idx             = base + (tail & 1) * 2 + (tail >> 1);// 每个线程负责读取第几行数据的 4 个 dword
    int v_ds_read_offset             = (laneid_shfl_5 * 4 + (laneid_shfl_4 & 1)) * 32 + (lane_id & 15) * 2; // 一次读写 8x32, 0-31 线程读取前面 4x32, 32-63 读取后面 4x32, 4x32按照线程 [0, 16, 0, 16] 这种方式来读取
    auto BUFFER_LOAD_FUNC            = &inline_buffer_load_dwordx4_lds<Element, 2>;
#else
    constexpr int READ_ONCE_LINES    = 4;
    constexpr int READ_ONCE_COUNT    = READ_ONCE_LINES * 32;
    constexpr int V_LDS_LOAD_NUM     = kBlockN * WARP_K / READ_ONCE_COUNT;
    constexpr int V_LOAD_REQUESTS    = V_LDS_LOAD_NUM;
    constexpr int READ_ELEMENT_COUNT = 2;
    int v_lane_headdim_n_idx         = lane_id & 15;
    int v_lane_seq_k_idx             = (laneid_shfl_4 & 1) * 2 + laneid_shfl_5; // 0, 1, 2, 3 ---> 0, 2, 1, 3
    int v_ds_read_offset             = (laneid_shfl_5 * 4 + (laneid_shfl_4 & 1)) * 32 + (lane_id & 15) * 2; // 一次读写 8x32, 0-31 线程读取前面 4x32, 32-63 读取后面 4x32, 4x32按照线程 [0, 16, 0, 16] 这种方式来读取
    auto BUFFER_LOAD_FUNC            = &inline_buffer_load_dword_lds<Element, 2>;
#endif

    int n_loop   = 0;
    if constexpr (STAGES > 1) {
        int v_block_buffer_load_global_offset = WARP_ID * WARP_K * vcache_seqlen_stride + n_loop * kBlockN;
        for(int load = 0; load < V_LOAD_REQUESTS; ++load) {
            int v_warp_buffer_load_k_id = (load + WARP_ID) % V_LOAD_REQUESTS;
            int v_warp_buffer_load_lds_offset = /*(v_warp_buffer_load_n_id * 32) + */load * READ_ONCE_COUNT;
            int s_offset = v_block_buffer_load_global_offset / 2;
            int v_offset = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + min(v_lane_seq_k_idx + load * READ_ONCE_LINES, max_seq_kv_offset - 1) * vcache_seqlen_stride) / 2;
            int v_lds_offset = v_warp_buffer_load_lds_offset / 2;
            BUFFER_LOAD_FUNC(v_lds + WARP_ID * STAGES * WARP_K * kBlockN + (stage_id) * WARP_K * kBlockN, gV, v_lds_offset, s_offset, v_offset);
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    if constexpr (STAGES > 2) {
        {
            int v_block_buffer_load_global_offset = WARP_ID * WARP_K * vcache_seqlen_stride + (n_loop + 1) * kBlockN;
            for(int load = 0; load < V_LOAD_REQUESTS; ++load) {
                int v_warp_buffer_load_k_id = (load + WARP_ID) % V_LOAD_REQUESTS;
                int v_warp_buffer_load_lds_offset = /*(v_warp_buffer_load_n_id * 32) + */load * READ_ONCE_COUNT;
                int s_offset = v_block_buffer_load_global_offset / 2;
                int v_offset = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + min(v_lane_seq_k_idx + load * READ_ONCE_LINES, max_seq_kv_offset - 1) * vcache_seqlen_stride) / 2;
                int v_lds_offset = v_warp_buffer_load_lds_offset / 2;
                BUFFER_LOAD_FUNC(v_lds + WARP_ID * 3 * WARP_K * kBlockN + (stage_id + 1) * WARP_K * kBlockN, gV, v_lds_offset, s_offset, v_offset);
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        {
            int v_block_buffer_load_global_offset = WARP_ID * WARP_K * vcache_seqlen_stride + (n_loop + 2) * kBlockN;
            for(int load = 0; load < V_LOAD_REQUESTS; ++load) {
                int v_warp_buffer_load_k_id = (load + WARP_ID) % V_LOAD_REQUESTS;
                int v_warp_buffer_load_lds_offset = /*(v_warp_buffer_load_n_id * 32) + */load * READ_ONCE_COUNT;
                int s_offset = v_block_buffer_load_global_offset / 2;
                int v_offset = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + min(v_lane_seq_k_idx + load * READ_ONCE_LINES, max_seq_kv_offset - 1) * vcache_seqlen_stride) / 2;
                int v_lds_offset = v_warp_buffer_load_lds_offset / 2;
                BUFFER_LOAD_FUNC(v_lds + WARP_ID * 3 * WARP_K * kBlockN + (stage_id + 2) * WARP_K * kBlockN, gV, v_lds_offset, s_offset, v_offset);
            }
        }
        __builtin_amdgcn_sched_barrier(0);
    }
}