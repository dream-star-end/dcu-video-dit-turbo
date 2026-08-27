#pragma once // prepare for prefetch V in qk gemm
#include "intrinsic.h"
#include "utils.h"


template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int stage_id, typename Element, bool Is_even_MN, int STAGES=2>
__forceinline__ __device__ void  prefetch_v_to_lds(
        vec4_uint gV,
        Element* v_lds,
        int WARP_ID,
        int seqlen_v_stride,
        int max_seq_kv_offset=-1) {
    constexpr int  WARP_NUM       = kBlockM * kBlockN / (WARP_M * WARP_N);
    constexpr bool IS_HEADDIM_128 = (kHeadDim == 128 and kHeadDimV == 128) or (kHeadDim == 64 and kHeadDimV == 64);

    // 预先计算一些公共表达式
    int lane_id       = threadIdx.x & 63;
    int laneid_shfl_2 = lane_id >> 2; // 0 ~ 15, 4 个线程读取一行
    int laneid_shfl_3 = lane_id >> 3; // 0 ~ 7,  8 个线程读取一行
    int laneid_shfl_4 = lane_id >> 4; // 0 ~ 3,  16 个线程读取一行
    int laneid_shfl_5 = lane_id >> 5; // 0 ~ 1,  lds 读取时, 8x32的数据按照线程 [0, 16, 0, 16, 32, 48, 32, 48] 来读取, 每 32 个线程读取一个 4x32

#if defined(USE_BUFFER_LOAD_DWORDX4)
    // 对于 headdim 128 而言, 2 组 32x32 可以写成 4 个 warp 分别读取 8 个 half, 即 4x64x8, 可以使用 buffer_load_dwordx4
    // 对于其他 headdim 而言, 暂不做那么激进的优化, 只有 1 组 32x32, 最多用 buffer_load_dwordx2
    constexpr int WARP_K             = 32;
    constexpr int READ_ONCE_LINES    = IS_HEADDIM_128 ? 16: 8;                    // 一个 warp 一次读几行数据, loadx2, 每行 32 个元素需要 32 / (4) = 8 个线程
    constexpr int READ_ONCE_COUNT    = READ_ONCE_LINES * 32;                      // 一个 warp 一次 load 多少数据
    constexpr int V_LDS_LOAD_NUM     = IS_HEADDIM_128 ? (2 * kBlockN * WARP_K/*对于非 headdim128 结尾的选项这里需要填 1, 一次只取 1 个 32x32 块*/) / READ_ONCE_COUNT: (kBlockN * WARP_K) / READ_ONCE_COUNT; // 整个 workgroup 要发多少 load 指令
    constexpr int V_LOAD_REQUESTS    = V_LDS_LOAD_NUM / WARP_NUM;                 // 每个 warp 要发多少 load 指令
    constexpr int READ_ELEMENT_COUNT = IS_HEADDIM_128 ? 8: 4;                     // 每个线程一次读取几个 half
    int v_lane_headdim_n_idx         = IS_HEADDIM_128 ? lane_id & 3: lane_id & 7; // 当前 lane 负责这个 warp 的第几个 dwordx4 或者 dwordx2
    // 为了解决 ds_read2_b32 的 bank 冲突, 需要交换 0-15 线程的一些读取地址, 4 个线程为一组 (0, 1, 2, 3 ---> 0, 2, 1, 3 的写入位置, 从而满足 ds_read2_b32 offset32 的要求)
    // 非 headdim 的话, 则交换 0-7 线程的一些读取地址, 4 个线程为一组
    int base                         = IS_HEADDIM_128 ? (laneid_shfl_2 & 0xc): (laneid_shfl_3 & 0x4); // 第几个4线程组的最小id
    int tail                         = IS_HEADDIM_128 ? (laneid_shfl_2 & 0x3): (laneid_shfl_3 & 0x3); // 线程组中的第几个线程
    int v_lane_seq_k_idx             = base + (tail & 1) * 2 + (tail >> 1);
    int v_ds_read_offset             = (laneid_shfl_5 * 4 + (laneid_shfl_4 & 1)) * 32 + (lane_id & 15) * 2; // 一次读写 8x32, 0-31 线程读取前面 4x32, 32-63 读取后面 4x32, 4x32按照线程 [0, 16, 0, 16] 这种方式来读取
    auto BUFFER_LOAD_FUNC            = IS_HEADDIM_128 ? &inline_buffer_load_dwordx4_lds<Element, 2>: &inline_buffer_load_dwordx2_lds<Element, 2>;
#else
    constexpr int WARP_K             = 32;
    constexpr int READ_ONCE_LINES    = 4;
    constexpr int READ_ONCE_COUNT    = READ_ONCE_LINES * 32;
    constexpr int V_LDS_LOAD_NUM     = (kBlockN * WARP_K) / READ_ONCE_COUNT;
    constexpr int V_LOAD_REQUESTS    = V_LDS_LOAD_NUM / WARP_NUM;
    constexpr int READ_ELEMENT_COUNT = 2;
    int v_lane_headdim_n_idx         = lane_id & 15;
    int v_lane_seq_k_idx             = (laneid_shfl_4 & 1) * 2 + laneid_shfl_5; // 0, 1, 2, 3 ---> 0, 2, 1, 3
    int v_ds_read_offset             = (laneid_shfl_5 * 4 + (laneid_shfl_4 & 1)) * 32 + v_lane_headdim_n_idx * 2; // 一次读写 8x32, 0-31 线程读取前面 4x32, 32-63 读取后面 4x32, 4x32按照线程 [0, 16, 0, 16] 这种方式来读取
    auto BUFFER_LOAD_FUNC            = &inline_buffer_load_dword_lds<Element, 2>;
#endif

    int n_loop = 0;
    int v_block_buffer_load_global_offset = n_loop * kBlockN;
    for(int load = 0, warp_loop = WARP_ID; load < V_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
        int v_warp_buffer_load_k_id = warp_loop / (kBlockN / 32);
        int s_offset = v_block_buffer_load_global_offset / 2;
        int seqlen_pos = v_lane_seq_k_idx + v_warp_buffer_load_k_id * READ_ONCE_LINES;
        if constexpr (not Is_even_MN) {
            seqlen_pos = min(seqlen_pos, max_seq_kv_offset - 1);
        }
        int v_offset = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + seqlen_pos * seqlen_v_stride) / 2;
        int v_lds_offset = v_warp_buffer_load_k_id * READ_ONCE_COUNT / 2;
        BUFFER_LOAD_FUNC(v_lds + (stage_id * STAGES) * WARP_K * kBlockN, gV, v_lds_offset, s_offset, v_offset);
    }
    __builtin_amdgcn_sched_barrier(0);
    if (IS_HEADDIM_128) {
    #if !defined(USE_BUFFER_LOAD_DWORDX4)
    // for ZD, double prefetch bring degradation; for BMZ, may bring improvement
        int n_loop = 0;
        for(int load = 0, warp_loop = WARP_ID; load < V_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int v_warp_buffer_load_k_id = warp_loop / (kBlockN / 32);
            int s_offset = v_block_buffer_load_global_offset / 2;
            int seqlen_pos = v_lane_seq_k_idx + v_warp_buffer_load_k_id * READ_ONCE_LINES + WARP_K;
            if constexpr (not Is_even_MN) {
                seqlen_pos = min(seqlen_pos, max_seq_kv_offset - 1);
            }
            int v_offset = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + seqlen_pos * seqlen_v_stride) / 2;
            int v_lds_offset = v_warp_buffer_load_k_id * READ_ONCE_COUNT / 2;
            BUFFER_LOAD_FUNC(v_lds + (stage_id * STAGES + 1) * WARP_K * kBlockN, gV, v_lds_offset, s_offset, v_offset);
        }
        __builtin_amdgcn_sched_barrier(0);
    #endif
    }
}