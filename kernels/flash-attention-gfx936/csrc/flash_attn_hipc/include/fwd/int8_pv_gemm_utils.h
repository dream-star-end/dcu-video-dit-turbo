#pragma once // prepare for prefetch V in qk gemm
#include "intrinsic.h"
#include "utils.h"


template<int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int stage_id, typename Element_k, bool Is_even_MN, int STAGES=2>
__forceinline__ __device__ void  int8_prefetch_v_to_lds(
        vec4_uint gV,
        Element_k* v_lds,
        int WARP_ID,
        int seqlen_v_stride,
        int max_seq_kv_offset=-1) {
    constexpr int  WARP_NUM       = kBlockM * kBlockN / (WARP_M * WARP_N);
    constexpr bool IS_HEADDIM_128 = (kHeadDim == 128 and kHeadDimV == 128) or (kHeadDim == 64 and kHeadDimV == 64);

    // 预先计算一些公共表达式
    int lane_id       = threadIdx.x & 63;
    int laneid_shfl_1 = lane_id >> 1; // 0 ~ 31, 2 个线程读取一行
    int laneid_shfl_2 = lane_id >> 2; // 0 ~ 15, 4 个线程读取一行
    int laneid_shfl_3 = lane_id >> 3; // 0 ~ 7,  8 个线程读取一行
    int laneid_shfl_4 = lane_id >> 4; // 0 ~ 3,  16 个线程读取一行
    int laneid_shfl_5 = lane_id >> 5; // 0 ~ 1,  lds 读取时, 8x32的数据按照线程 [0, 16, 0, 16, 32, 48, 32, 48] 来读取, 每 32 个线程读取一个 4x32

    constexpr int WARP_K             = 32;
    constexpr int READ_ONCE_LINES    = 8;
    constexpr int READ_ONCE_COUNT    = READ_ONCE_LINES * 32;
    constexpr int V_LDS_LOAD_NUM     = (2 * kBlockN * WARP_K) / READ_ONCE_COUNT;
    constexpr int V_LOAD_REQUESTS    = V_LDS_LOAD_NUM / WARP_NUM;
    constexpr int READ_ELEMENT_COUNT = 4;
    int v_lane_headdim_n_idx         = lane_id & 7;
    int tail                         = (laneid_shfl_3 & 0x3);
    int base                         = (laneid_shfl_3 & 0x4);
    int v_lane_seq_k_idx             = base + (tail & 1) * 2 + (tail >> 1); // 0, 1, 2, 3 ---> 0, 2, 1, 3
    int v_ds_read_offset             = (laneid_shfl_5 * 4 + (laneid_shfl_4 & 1)) * 32 + v_lane_headdim_n_idx * 2; // 一次读写 8x32, 0-31 线程读取前面 4x32, 32-63 读取后面 4x32, 4x32按照线程 [0, 16, 0, 16] 这种方式来读取
    auto BUFFER_LOAD_FUNC            = &inline_buffer_load_dword_lds<Element_k, 2>;

    int n_loop = 0;
    int v_block_buffer_load_global_offset = n_loop * kBlockN;
    for(int load = 0, warp_loop = WARP_ID; load < V_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
        int v_warp_buffer_load_k_id = warp_loop / (kBlockN / 32);
        int s_offset = v_block_buffer_load_global_offset / 4;
        int seqlen_pos = v_lane_seq_k_idx + v_warp_buffer_load_k_id * READ_ONCE_LINES;
        if constexpr (not Is_even_MN) {
            seqlen_pos = min(seqlen_pos, max_seq_kv_offset - 1);
        }
        int v_offset = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + seqlen_pos * seqlen_v_stride) / 4;
        int v_lds_offset = v_warp_buffer_load_k_id * READ_ONCE_COUNT / 4;
        BUFFER_LOAD_FUNC(v_lds + (stage_id * STAGES) * WARP_K * kBlockN, gV, v_lds_offset, s_offset, v_offset);
        // __builtin_amdgcn_sched_barrier(0);
        // buffer_load_lds_dwordx1_wait<0>();
        // __builtin_amdgcn_sched_barrier(0);
        // printf("222!, WARP_ID is %d, lane_id is %d, v_lds_offset is %d, goffset is %d, vlds is %d\n", WARP_ID, lane_id, v_lds_offset*4+lane_id, s_offset*4+v_offset*4, v_lds[v_lds_offset+lane_id]);
        // printf("222!, WARP_ID is %d, lane_id is %d, v_lds_offset is %d, goffset is %d, vlds is %d\n", WARP_ID, lane_id, v_lds_offset*4+lane_id+1, s_offset*4+v_offset*4+1, v_lds[v_lds_offset+lane_id+1]);
        // printf("222!, WARP_ID is %d, lane_id is %d, v_lds_offset is %d, goffset is %d, vlds is %d\n", WARP_ID, lane_id, v_lds_offset*4+lane_id+2, s_offset*4+v_offset*4+2, v_lds[v_lds_offset+lane_id+2]);
        // printf("222!, WARP_ID is %d, lane_id is %d, v_lds_offset is %d, goffset is %d, vlds is %d\n", WARP_ID, lane_id, v_lds_offset*4+lane_id+3, s_offset*4+v_offset*4+3, v_lds[v_lds_offset+lane_id+3]);
    }
    __builtin_amdgcn_sched_barrier(0);
    if (IS_HEADDIM_128) {
    #if !defined(USE_BUFFER_LOAD_DWORDX4)
    // for ZD, double prefetch bring degradation; for BMZ, may bring improvement
        int n_loop = 0;
        for(int load = 0, warp_loop = WARP_ID; load < V_LOAD_REQUESTS; warp_loop += WARP_NUM, ++load) {
            int v_warp_buffer_load_k_id = warp_loop / (kBlockN / 32);
            int s_offset = v_block_buffer_load_global_offset / 4;
            int seqlen_pos = v_lane_seq_k_idx + v_warp_buffer_load_k_id * READ_ONCE_LINES + WARP_K;
            if constexpr (not Is_even_MN) {
                seqlen_pos = min(seqlen_pos, max_seq_kv_offset - 1);
            }
            int v_offset = (v_lane_headdim_n_idx * READ_ELEMENT_COUNT + seqlen_pos * seqlen_v_stride) / 4;
            int v_lds_offset = v_warp_buffer_load_k_id * READ_ONCE_COUNT / 4;
            BUFFER_LOAD_FUNC(v_lds + (stage_id * STAGES + 1) * WARP_K * kBlockN, gV, v_lds_offset, s_offset, v_offset);
        }
        __builtin_amdgcn_sched_barrier(0);
    #endif
    }
}