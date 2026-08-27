#pragma once
#include "numeric_types.h"


namespace gfx92a {


template<bool Is_Varlen, int kHeadDim, int kBlockK, int WARP_M, int WARP_NUM, int M_MMAC_COUNT, typename Element>
__forceinline__ __device__ void kvcache_prefetch_q_to_vgpr(
        Element* q_ptr,
        Element* q_lds,
        union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2],
        int warp_id,
        int query_seqlen_stride,
        int query_ngroup_stride,
        int ngroups,
        int max_seq_q_offset=0) {

    constexpr int elementBytes = sizeof(Element);

    // resource regs
    auto q_addr = prepare_for_buffer_load<kHeadDim, Element, false>(q_ptr);

    if constexpr (Is_Varlen) {
        int lane_id = int(threadIdx.x) & 63;
        flash::wait_buffer_data_arrived<true>(0);
        if constexpr (kHeadDim == 128 and WARP_NUM == 4) {
            for (int load = 0; load < M_MMAC_COUNT; ++load) {
                int q_row = min(load * 16 + (lane_id >> 2), max_seq_q_offset - 1);
                int q_col = warp_id * 32 + (lane_id & 3) * 8;
                int q_row_seq = q_row / ngroups;
                int q_row_regroup = q_row - q_row_seq * ngroups;
                int q_load_offset = q_row_seq * ngroups * query_seqlen_stride + q_row_regroup * query_ngroup_stride + q_col;
                int q_lds_write_offset = (load * 4 + warp_id) * 16 * 32;
                inline_buffer_load_dwordx4_lds<Element, 1>(q_lds, q_addr, q_lds_write_offset, 0, q_load_offset);
            }
            flash::wait_buffer_data_arrived<true>(0);
            for (int load = 0; load < M_MMAC_COUNT; ++load) {
                for (int neighbor = 0; neighbor < WARP_NUM; ++neighbor) {
                    int q_lds_load_offset = (load * 4 + neighbor) * 16 * 32 + (lane_id & 15) * 32 + (lane_id >> 4) * 4;
                    int q_lds_load_bytes = reinterpret_cast<size_t>(q_lds + q_lds_load_offset);
                    inline_ds_read2_b64(q_lds_load_bytes, q_reg[neighbor * 2 + load].f32, 0, 4);
                }
            }
            flash::wait_lds_data_arrived<true>(0);
        } else {
            #pragma unroll
            for (int neighbor = 0; neighbor < WARP_NUM; ++neighbor) {
                #pragma unroll
                for (int load = 0; load < M_MMAC_COUNT; ++load) {
                    int q_row = min(load * 16 + (lane_id & 15), max_seq_q_offset - 1);
                    int q_col = neighbor * 32 + (lane_id >> 4) * 4;
                    int q_row_seq = q_row / ngroups;
                    int q_row_regroup = q_row - q_row_seq * ngroups;
                    int q_load_offset = q_row_seq * ngroups * query_seqlen_stride + q_row_regroup * query_ngroup_stride + q_col;
                    q_reg[neighbor * 2 + load].data[0] = *(double*)(q_ptr + q_load_offset);
                    q_reg[neighbor * 2 + load].data[1] = *(double*)(q_ptr + q_load_offset + 16);
                }
            }
        }
    } else {
        if constexpr (kHeadDim == 128 and WARP_NUM == 4) {
            // prepare mls resource regs
            vec4_uint q_srsrc;
            q_srsrc[1] = q_addr[1];
            q_srsrc[2] = query_seqlen_stride;

            // global offset along seqlen_q
            int q_loop = 0;
            int q_seq_offset = q_loop * kBlockK;

            // global offset along headdim
            int q_dim_offset = warp_id * kBlockK;

            // global bytes
            q_srsrc[0] = q_addr[0] + (q_seq_offset + q_dim_offset ) * elementBytes;
            if constexpr (true) {
                int nm_filter = inline_min_max<0, 32>(32 - max_seq_q_offset);
                q_srsrc[3]    = max_seq_q_offset % 32 == 0 ? 0: nm_filter << 8;
            }
            // compute lds write offset, each warp occupy 32 * 32 * sizeof(f16) = 2KB
            int q_lds_write_offset = warp_id * (WARP_M / 32) * (kBlockK / 32) * (32 * 32);
            int q_lds_offset_bytes = q_lds_write_offset * elementBytes;
            // flash::wait_lds_data_arrived<true>(0);
            inline_matrix_load_32x32_b16_lds_trans<0, 0>(q_lds, q_srsrc, q_lds_offset_bytes, 0);

            // wait q data arrived
            flash::wait_buffer_data_arrived<true>(0);

            // lds -> vgprs
            if constexpr (M_MMAC_COUNT == 1) {
                DS_READ_MATRIX_32X16_B16(0 * 32 * 32 * 2, q_reg[0 * 2].f16, true);
                DS_READ_MATRIX_32X16_B16(1 * 32 * 32 * 2, q_reg[1 * 2].f16, true);
                DS_READ_MATRIX_32X16_B16(2 * 32 * 32 * 2, q_reg[2 * 2].f16, true);
                DS_READ_MATRIX_32X16_B16(3 * 32 * 32 * 2, q_reg[3 * 2].f16, true);
            } else {
                DS_READ_MATRIX_32X32_B16(0 * 32 * 32 * 2, q_reg[0 * 2].f16, q_reg[0 * 2 + 1].f16, true);
                DS_READ_MATRIX_32X32_B16(1 * 32 * 32 * 2, q_reg[1 * 2].f16, q_reg[1 * 2 + 1].f16, true);
                DS_READ_MATRIX_32X32_B16(2 * 32 * 32 * 2, q_reg[2 * 2].f16, q_reg[2 * 2 + 1].f16, true);
                DS_READ_MATRIX_32X32_B16(3 * 32 * 32 * 2, q_reg[3 * 2].f16, q_reg[3 * 2 + 1].f16, true);
            }
            flash::wait_lds_data_arrived<true>(0);
        }
        else {
            // TODO
        }
    }
}



template<int kBlockK, int WARP_N, int prefetchKLevel, typename Element>
__forceinline__ __device__ void kvcache_prefetch_k_to_lds(
        vec4_uint k_addr,
        Element* k_lds,
        int warp_id,
        int k_seq_stride,
        int max_seq_k_offset=0) {

    constexpr int elementBytes = sizeof(Element);

    // prepare mls resource regs
    vec4_uint k_srsrc;
    k_srsrc[1] = k_addr[1];
    k_srsrc[2] = k_seq_stride;

    // pingpong buffer stage
    int stage_id = 0;

    // tile id along headdim dimension
    int k_loop = 0;

    // occupy 4 * 2 * 2 * 32 * 32 * sizeof(f16) = 32 KB, in total
    #pragma unroll
    for (int prefetch_id = 0; prefetch_id < prefetchKLevel; ++prefetch_id) {

        // global bytes along headdim
        int k_dim_bytes = (k_loop + prefetch_id) * kBlockK * elementBytes;

        // global bytes along seqlen
        int k_seq_bytes;
        if constexpr (true) {
            int nm_filter_max    = warp_id * WARP_N + 32 - max_seq_k_offset;
            int real_mls_warp_id = nm_filter_max >= 32 ? 0: warp_id;
            k_seq_bytes          = real_mls_warp_id * WARP_N * k_seq_stride * elementBytes;
            int nm_filter        = inline_min_max<0, 32>(real_mls_warp_id * WARP_N + 32 - max_seq_k_offset);
            k_srsrc[3]           = nm_filter << 8;
        }
        // acquire buffer address
        *(uint64_t*)&k_srsrc   = VA_LIMIT_BITS(*(uint64_t*)&k_addr + k_dim_bytes + k_seq_bytes);
        // compute lds offset / bytes
        int k_lds_stage_offset = (warp_id * prefetchKLevel + prefetch_id) * (WARP_N / 32) * (kBlockK / 32) * (32 * 32);
        int lds_offset_bytes   = k_lds_stage_offset * elementBytes;
        inline_matrix_load_32x32_b16_lds_trans<0, 0>(k_lds, k_srsrc, lds_offset_bytes, 0);
    }
    __builtin_amdgcn_sched_barrier(0);
}



template<int kHeadDim, int kBlockN, int WARP_K, int STAGES, int prefetchVLevel, typename Element>
__forceinline__ __device__ void kvcache_prefetch_v_to_lds(
        vec4_uint v_addr,
        Element* v_lds,
        int warp_id,
        int v_seq_stride,
        int max_seq_kv_offset=0) {

    constexpr int V_LOAD_REQUESTS = (WARP_K * kBlockN) / (32 * 32);
    constexpr int elementBytes    = 2;

    // prepare mls resource regs
    vec4_uint v_srsrc;
    v_srsrc[1] = v_addr[1];
    v_srsrc[2] = v_seq_stride;

    if constexpr (prefetchVLevel == 2) {

        // tile loop
        int n_loop = 0;

        // ping-ping stage
        int stage_id = 0;

        #pragma unroll
        for (int prefetch_id = 0; prefetch_id < prefetchVLevel; ++prefetch_id) {

            // global bytes along headdim dimension
            int v_dim_bytes = (n_loop + prefetch_id) * kBlockN * elementBytes;

            // global bytes along seq dimension
            int v_seq_bytes;
            if constexpr (true) {
                int nm_filter_max    = warp_id * WARP_K + 32 - max_seq_kv_offset;
                int real_mls_warp_id = nm_filter_max >= 32 ? 0: warp_id;
                v_seq_bytes          = real_mls_warp_id * WARP_K * v_seq_stride * elementBytes;
                int nm_filter        = inline_min_max<0, 32>(real_mls_warp_id * WARP_K + 32 - max_seq_kv_offset);
                v_srsrc[3]           = max_seq_kv_offset % kBlockN == 0 ? 0: nm_filter << 8;
                v_srsrc[3]           += 0x20000;
            }
            *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_addr + v_seq_bytes + v_dim_bytes);
            // lds bytes
            int v_lds_write_offset = (warp_id * STAGES * prefetchVLevel + stage_id * prefetchVLevel + prefetch_id) * (V_LOAD_REQUESTS * 32 * 32);
            int v_lds_write_bytes  = v_lds_write_offset * elementBytes;
            inline_matrix_load_32x32_b16_lds<0, 1>(v_lds, v_srsrc, v_lds_write_bytes, 0);
        }
    } else if (prefetchVLevel == 4) {

        #pragma unroll
        for (int prefetch_id = 0; prefetch_id < prefetchVLevel; ++prefetch_id) {

            // global bytes along headdim dimension
            int v_dim_bytes = prefetch_id * kBlockN * elementBytes;

            // global bytes along seq dimension
            int v_seq_bytes;
            if constexpr (true) {
                int nm_filter_max    = warp_id * WARP_K + 32 - max_seq_kv_offset;
                int real_mls_warp_id = nm_filter_max >= 32 ? 0: warp_id;
                v_seq_bytes          = real_mls_warp_id * WARP_K * v_seq_stride * elementBytes;
                int nm_filter        = inline_min_max<0, 32>(real_mls_warp_id * WARP_K + 32 - max_seq_kv_offset);
                v_srsrc[3]           = max_seq_kv_offset % kBlockN == 0 ? 0: nm_filter << 8;
                v_srsrc[3]           += 0x20000;
            }
            *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_addr + v_seq_bytes + v_dim_bytes);
            // lds bytes
            int v_lds_write_offset = (warp_id * prefetchVLevel + prefetch_id) * (V_LOAD_REQUESTS * 32 * 32);
            int v_lds_write_bytes  = v_lds_write_offset * elementBytes;
            inline_matrix_load_32x32_b16_lds<0, 1>(v_lds, v_srsrc, v_lds_write_bytes, 0);
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}



template<int kHeadDim, int kHeadDimV, int kBlockN, int kBlockK, int WARP_M, int WARP_N, int prefetchKLevel, int prefetchVLevel, int M_MMAC_COUNT, typename Element, typename ElementAccum>
__forceinline__ __device__ void kvcache_qk_gemm_prefetch_v(
        vec4_uint k_addr,
        vec4_uint v_addr,
        Element* k_lds,
        Element* v_lds,
        union_vec4_f16x2<Element> q_reg[(kHeadDim / kBlockK) * (WARP_M * kBlockK) / (32 * 32) * 2],
        vec4_Accum<ElementAccum> s_reg[(WARP_M / 32) * (WARP_N / 32)][4],
        int warp_id,
        int k_seq_stride,
        int v_seq_stride,
        int max_seq_kv_offset=0) {

    static_assert(WARP_M == 32 and WARP_N == 32 and kBlockK == 32 and "To simplify, only WARP_M = WARP_N = kBlockK = 32 is supported!");
    static_assert (prefetchKLevel == 4 and "To simplify, only prefetchKLevel = 4 is supported");

    constexpr int K_LOAD_REQUESTS = (WARP_N / 32) * (kBlockK / 32);
    constexpr int elementBytes    = 2;

    // alloc k_regs, 32x32 f16 per warp, and thus 16 f16 for each threads
    union_vec4_f16x2<Element> k_reg[1 * (WARP_N * kBlockK) / (32 * 32) * 2];

    // s_reg initialize
    #pragma unroll
    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
        #pragma unroll
        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            s_reg[0][min_tile_n * 2 + min_tile_m].b64[0] = __builtin_hcu_mov_b64(0x0);
            s_reg[0][min_tile_n * 2 + min_tile_m].b64[1] = __builtin_hcu_mov_b64(0x0);
        }
    }

    // qk gemm main loop, along kheaddim dimension
    for (int k_loop = 0; k_loop < (kHeadDim / kBlockK); k_loop += 1) {

        flash::wait_buffer_data_arrived<false>((kHeadDim / kBlockK) - 1 - k_loop);

        // lds -> vgprs
        int k_lds_load_bytes = reinterpret_cast<size_t>(k_lds) + (warp_id * prefetchKLevel + k_loop) * K_LOAD_REQUESTS * (32 * 32) * elementBytes;
        DS_READ_MATRIX_32X32_B16(k_lds_load_bytes, k_reg[0].f16, k_reg[1].f16, true);

        // mmac flow
        #pragma unroll
        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            flash::wait_lds_data_arrived<false>(2 - 1 - min_tile_n);
            #pragma unroll
            for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    int q_tile_id = k_loop * 2 + min_tile_m;
                    s_reg[0][min_tile_n * 2 + min_tile_m].f32 = mmac_4interleave<Element, ElementAccum>(
                        q_reg[q_tile_id].f16x4[min_tile_k],
                        k_reg[min_tile_n].f16x4[min_tile_k],
                        s_reg[0][min_tile_n * 2 + min_tile_m].f32);
                }
            }
        }
    }


    // need to reduce results on scores_max and prefetch V, and thus sync
    // can be simplified as flash::wait_all_warp_arrived()
    flash::wait_lds_data_arrived<true>(0);

    // prefetch v
    // can be rearranged while qk doing mmac
    // gfx92a::kvcache_prefetch_v_to_lds<kHeadDimV, kBlockK, kBlockK, 2/*STAGES*/, prefetchVLevel, Element>(v_addr, v_lds, warp_id, v_seq_stride, max_seq_kv_offset);

}




template <int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT, typename DataType>
__forceinline__ __device__ void kvcache_apply_mask(DataType tensor[M_WARP_COUNT * N_WARP_COUNT][4], const int max_seqlen_k, const int col_idx_offset_= 0) {
    const int lane_id        = threadIdx.x & 63;
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4);
    #pragma unroll
    for (int ni = 0; ni < N_WARP_COUNT; ++ni) {
        #pragma unroll
        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n * 16;
            #pragma unroll
            for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                const int col_idx = col_idx_base + vec_idx * 4;
                if (col_idx >= max_seqlen_k) {
                    #pragma unroll
                    for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
                        #pragma unroll
                        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                            tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx] = -INFINITY;
                        }
                    }
                }
            }
        }
    }
}



template <int M_WARP_COUNT, int N_WARP_COUNT, int M_MMAC_COUNT, bool Is_Varlen, typename DataType>
__forceinline__ __device__ void kvcache_apply_mask_causal(DataType tensor[M_WARP_COUNT * N_WARP_COUNT][4], const int col_idx_offset_,
                                        const int max_seqlen_k, const int row_idx_offset_,
                                        const int max_seqlen_q, const int ngroups, const int mtp, const int layout) {
    const int lane_id = threadIdx.x & 63;
    const int row_idx_offset = row_idx_offset_ + (lane_id & 15);
    const int col_idx_offset = col_idx_offset_ + (lane_id >> 4);
    #pragma unroll
    for (int mi = 0; mi < M_WARP_COUNT; ++mi) {
        const int row_idx_base = row_idx_offset + mi * 32;
        #pragma unroll
        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
            const int row_idx = row_idx_base + min_tile_m * 16;
            int col_idx_limit_right;
            if constexpr (Is_Varlen) {
                col_idx_limit_right = std::min(max_seqlen_k, (row_idx / ngroups)/*only for layout 1: bshd*/ + max_seqlen_k - (max_seqlen_q / ngroups));
            } else {
                const int row_in_mtp = layout == 0 ? (row_idx % mtp): (row_idx / ngroups);
                col_idx_limit_right = std::min(max_seqlen_k, row_in_mtp + max_seqlen_k - mtp);
            }
            #pragma unroll
            for (int ni = 0; ni < N_WARP_COUNT; ++ni)  {
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    const int col_idx_base = col_idx_offset + ni * 32 + min_tile_n * 16;
                    #pragma unroll
                    for (int vec_idx = 0; vec_idx < 4; ++vec_idx) {
                        const int col_idx = col_idx_base + vec_idx * 4;
                        tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx] = (col_idx > col_idx_limit_right) ? -INFINITY: tensor[mi + ni * M_WARP_COUNT][min_tile_n * 2 + min_tile_m].f32[vec_idx];
                    }
                }
            }
        }
    }
}



template <int M_MMAC_COUNT, typename Element, typename ElementAccum>
__forceinline__ __device__ void convert_attn_f32_to_f16(union_vec4_fp32 s_reg[1][4], union_vec2_f16x2<Element> p_reg[1][4]) {
    #pragma unroll
    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
        #pragma unroll
        for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
            p_reg[0][0 * 2 + min_tile_m].f16x2[min_tile_k] = DownCastPair<float, Element>(s_reg[0][0 * 2 + min_tile_m].f32x2[min_tile_k]);
            p_reg[0][1 * 2 + min_tile_m].f16x2[min_tile_k] = DownCastPair<float, Element>(s_reg[0][1 * 2 + min_tile_m].f32x2[min_tile_k]);
        }
    }
}


template<bool prefetchK, int K_LOOP_COUNT, int kBlockN, int kBlockK, int M_WARP_COUNT, int PV_N_WARP_COUNT, int PV_K_WARP_COUNT, int STAGES, int prefetchKLevel, int prefetchVLevel, int M_MMAC_COUNT, typename Element, typename ElementAccum>
__forceinline__ __device__ void kvcache_pv_gemm_prefetch_k(
        vec4_uint v_addr,
        vec4_uint k_addr,
        Element* v_lds,
        Element* k_lds,
        union_vec2_f16x2<Element> p_reg[M_WARP_COUNT * PV_K_WARP_COUNT][4],
        vec4_Accum<ElementAccum> pv_reg[K_LOOP_COUNT * M_WARP_COUNT * (kBlockN / 32)][4],
        int warp_id,
        int v_seq_stride,
        int k_seq_stride,
        int max_seq_kv_offset=0) {

    constexpr int WARP_K = PV_K_WARP_COUNT * 32;
    static_assert (kBlockK >= 32, "Error: pv gemm kBlockK must be equal or greater than 32");
    static_assert (kBlockN == PV_N_WARP_COUNT * 32, "Error: kBlockN in kvcache_pv_gemm_prefetch_k must be WARP_N * 32");
    static_assert (M_WARP_COUNT == 1, "for gfx938, only WARP_M = 32 is supported yet!");
    static_assert (PV_N_WARP_COUNT == 1, "for gfx938, only WARP_N = 32 is supported yet!");
    static_assert (PV_K_WARP_COUNT == 1, "for gfx938, only WARP_K = 32 is supported yet!");

    constexpr int V_LOAD_REQUESTS = (WARP_K * kBlockN) / (32 * 32);
    constexpr int elementBytes    = 2;

    // sync lds usage for reducing max/sum
    flash::wait_lds_data_arrived<true>(0); // __syncthreads();

    if constexpr (prefetchVLevel == 2) {
        // hold v regs
        union_vec4_f16x2<Element> v_reg[1 * PV_K_WARP_COUNT * PV_N_WARP_COUNT * 2];

        // prepare v resource regs
        vec4_uint v_srsrc;
        v_srsrc[1] = v_addr[1];
        v_srsrc[2] = v_seq_stride;

        // pingpong stage
        int stage_id = (STAGES == 2) ? 1: 0;

        // make p 4-interleave layout for pv gemm
        // strange: delete wait, results are wrong even if flash::wait_lds_data_arrived<false>(0);
        ds_mpermute_kdim_for_mmac(p_reg[0][0].f16x4);
        ds_mpermute_kdim_for_mmac(p_reg[0][1].f16x4);
        ds_mpermute_kdim_for_mmac(p_reg[0][2].f16x4);
        ds_mpermute_kdim_for_mmac(p_reg[0][3].f16x4);

        // pv gemm main loop
        constexpr int N_LOOP_STEP  = (STAGES == 2) ? prefetchVLevel: 1;
        constexpr int N_LOOP_START = (STAGES == 2) ? N_LOOP_STEP: 1;
        for (int n_loop = N_LOOP_START; n_loop < K_LOOP_COUNT; n_loop += N_LOOP_STEP) {

            #pragma unroll
            for (int prefetch_id = 0; prefetch_id < prefetchVLevel; ++prefetch_id) {

                // global bytes along headdim dimension
                int v_dim_bytes = (n_loop + prefetch_id) * kBlockN * elementBytes;

                // global bytes along seq dimension
                int v_seq_bytes;
                if constexpr (true) {
                    int nm_filter_max    = warp_id * WARP_K + 32 - max_seq_kv_offset;
                    int real_mls_warp_id = nm_filter_max >= 32 ? 0: warp_id; // can be simplified after gfx938
                    v_seq_bytes          = real_mls_warp_id * WARP_K * v_seq_stride * elementBytes;
                    int nm_filter        = inline_min_max<0, 32>(real_mls_warp_id * WARP_K + 32 - max_seq_kv_offset);
                    v_srsrc[3]           = max_seq_kv_offset % kBlockN == 0 ? 0: nm_filter << 8;
                    v_srsrc[3]           += 0x20000;
                }
                *(uint64_t*)&v_srsrc = VA_LIMIT_BITS(*(uint64_t*)&v_addr + v_seq_bytes + v_dim_bytes);
                // lds write bytes
                int v_lds_write_offset = (warp_id * STAGES * prefetchVLevel + stage_id * prefetchVLevel + prefetch_id) * (V_LOAD_REQUESTS * 32 * 32);
                int v_lds_write_bytes  = v_lds_write_offset * elementBytes;
                inline_matrix_load_32x32_b16_lds<0, 1>(v_lds, v_srsrc, v_lds_write_bytes, 0);
            }

            // wait v data stored in lds
            if constexpr (N_LOOP_STEP == 2) {
                flash::wait_buffer_data_arrived<false>((prefetchVLevel + prefetchVLevel - 1) * V_LOAD_REQUESTS);
            } else if constexpr (N_LOOP_STEP == 1 and STAGES == 2) {
                flash::wait_buffer_data_arrived<false>(1 * V_LOAD_REQUESTS);
            } else if constexpr (N_LOOP_STEP == 1 and STAGES == 1) {
                flash::wait_buffer_data_arrived<false>(0);
            }

            // roll stage
            if constexpr (STAGES == 2) { stage_id ^= 1; }

            // lds -> vgprs
            int v_lds_load_bytes = reinterpret_cast<size_t>(v_lds) + (warp_id * STAGES * prefetchVLevel + stage_id * prefetchVLevel + 0) * (V_LOAD_REQUESTS * 32 * 32) * elementBytes;
            DS_READ_MATRIX_32X32_B16_ALT2(v_lds_load_bytes, v_reg[0].f16, v_reg[1].f16, false);

            // pv mmac flow
            #pragma unroll
            for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                flash::wait_lds_data_arrived<false>(2 - 1 - min_tile_k);
                int pv_tile_id = (STAGES == 2) ? n_loop - 2: n_loop;
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = flash::mmac_4interleave<Element, ElementAccum>(
                            p_reg[0][min_tile_k * 2 + min_tile_m].f16x4,
                            v_reg[min_tile_k].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
            // process second tile of pv gemm
            if constexpr (prefetchVLevel == 2) {
                flash::wait_buffer_data_arrived<false>(prefetchVLevel * V_LOAD_REQUESTS);

                // lds -> vgprs
                int v_lds_load_bytes = reinterpret_cast<size_t>(v_lds) + (warp_id * STAGES * prefetchVLevel + stage_id * prefetchVLevel + 1/*prefetch_id*/) * (V_LOAD_REQUESTS * 32 * 32) * elementBytes;
                DS_READ_MATRIX_32X32_B16_ALT2(v_lds_load_bytes, v_reg[0].f16, v_reg[1].f16, false);

                // pv gemm mmac flow
                #pragma unroll
                for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    flash::wait_lds_data_arrived<false>(2 - 1 - min_tile_k);
                    int pv_tile_id = (STAGES == 2) ? n_loop - 1: n_loop;
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = flash::mmac_4interleave<Element, ElementAccum>(
                                p_reg[0][min_tile_k * 2 + min_tile_m].f16x4,
                                v_reg[min_tile_k].f16x4[min_tile_n],
                                pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                        }
                    }
                }
            }
        }

        if constexpr (STAGES == 2) {
            int n_loop = K_LOOP_COUNT;

            // wait v stored in lds
            flash::wait_buffer_data_arrived<false>((prefetchVLevel - 1) * V_LOAD_REQUESTS);

            // roll stage
            stage_id ^= 1;

            // lds -> vgprs
            int v_lds_load_bytes = reinterpret_cast<size_t>(v_lds) + (warp_id * STAGES * prefetchVLevel + stage_id * prefetchVLevel) * (V_LOAD_REQUESTS * 32 * 32) * elementBytes;
            DS_READ_MATRIX_32X32_B16_ALT2(v_lds_load_bytes, v_reg[0].f16, v_reg[1].f16, false);

            // pv gemm mmac flow
            #pragma unroll
            for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                flash::wait_lds_data_arrived<false>(2 - 1 - min_tile_k);
                int pv_tile_id = n_loop - 2;
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = flash::mmac_4interleave<Element, ElementAccum>(
                            p_reg[0][min_tile_k * 2 + min_tile_m].f16x4,
                            v_reg[min_tile_k].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
            // process second tile of pv gemm
            if constexpr (N_LOOP_STEP == 2) {
                flash::wait_buffer_data_arrived<false>(0);

                // lds -> vgprs
                int v_lds_load_bytes = reinterpret_cast<size_t>(v_lds) + (warp_id * STAGES * prefetchVLevel + stage_id * prefetchVLevel + 1/*prefetch_id*/) * (V_LOAD_REQUESTS * 32 * 32) * elementBytes;
                DS_READ_MATRIX_32X32_B16_ALT2(v_lds_load_bytes, v_reg[0].f16, v_reg[1].f16, false);

                // pv gemm mmac flow
                #pragma unroll
                for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                    flash::wait_lds_data_arrived<false>(2 - 1 - min_tile_k);
                    int pv_tile_id = n_loop - 1;
                    #pragma unroll
                    for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                        #pragma unroll
                        for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = flash::mmac_4interleave<Element, ElementAccum>(
                                p_reg[0][min_tile_k * 2 + min_tile_m].f16x4,
                                v_reg[min_tile_k].f16x4[min_tile_n],
                                pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                        }
                    }
                }
            }
        }
    } else if constexpr (prefetchVLevel == 4) {

        bool can_prefetch_k = max_seq_kv_offset > kBlockK;
        if constexpr (prefetchK) {
            if (can_prefetch_k) {
                gfx92a::kvcache_prefetch_k_to_lds<kBlockN, PV_N_WARP_COUNT * 32, prefetchKLevel, Element>(k_addr, k_lds, warp_id, k_seq_stride, max_seq_kv_offset - kBlockK);
            }
        }

        // hold v regs
        union_vec4_f16x2<Element> v_reg[1 * PV_K_WARP_COUNT * PV_N_WARP_COUNT * 2];

        // prepare v resource regs
        vec4_uint v_srsrc;
        v_srsrc[1] = v_addr[1];
        v_srsrc[2] = v_seq_stride;

        // make p 4-interleave layout for pv gemm
        // strange: delete wait, results are wrong even if flash::wait_lds_data_arrived<false>(0);
        ds_mpermute_kdim_for_mmac(p_reg[0][0].f16x4);
        ds_mpermute_kdim_for_mmac(p_reg[0][1].f16x4);
        ds_mpermute_kdim_for_mmac(p_reg[0][2].f16x4);
        ds_mpermute_kdim_for_mmac(p_reg[0][3].f16x4);

        // wait v data stored in lds
        if constexpr (prefetchK) {
            if (can_prefetch_k) {
                flash::wait_buffer_data_arrived<false>(prefetchKLevel/*4 for hdim 128*/);
            } else {
                flash::wait_buffer_data_arrived<false>(0);
            }
        } else {
            flash::wait_buffer_data_arrived<false>(0);
        }

        // pv gemm main loop
        for (int n_loop = 0; n_loop < K_LOOP_COUNT; n_loop += 1) {

            // lds -> vgprs
            int v_lds_load_bytes = reinterpret_cast<size_t>(v_lds) + (warp_id * prefetchVLevel + n_loop) * (V_LOAD_REQUESTS * 32 * 32) * elementBytes;
            DS_READ_MATRIX_32X32_B16_ALT2(v_lds_load_bytes, v_reg[0].f16, v_reg[1].f16, false);

            // pv mmac flow
            #pragma unroll
            for (int min_tile_k = 0; min_tile_k < 2; ++min_tile_k) {
                flash::wait_lds_data_arrived<false>(2 - 1 - min_tile_k);
                int pv_tile_id = n_loop;
                #pragma unroll
                for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
                    #pragma unroll
                    for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                        pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32 = flash::mmac_4interleave<Element, ElementAccum>(
                            p_reg[0][min_tile_k * 2 + min_tile_m].f16x4,
                            v_reg[min_tile_k].f16x4[min_tile_n],
                            pv_reg[pv_tile_id][min_tile_n * 2 + min_tile_m].f32);
                    }
                }
            }
        }
    }

    // sync lds usage
    flash::wait_lds_data_arrived<true>(0);
}



template<int K_LOOP_COUNT, int K_WARP_COUNT, int M_WARP_COUNT, int M_MMAC_COUNT, int WARP_NUM, typename ElementAccum>
__forceinline__ __device__ void kvcache_acco_reduce_tile16x32(
        vec4_Accum < ElementAccum> acc_o[K_LOOP_COUNT * M_WARP_COUNT * K_WARP_COUNT][4],
        ElementAccum* acc_o_lds,
        int seqlen_q,
        int warp_id,
        int lane_id) {
    if constexpr (K_LOOP_COUNT == 4 and WARP_NUM == 4 and K_WARP_COUNT == 1) {

        constexpr int mmacVgprs      = 4;
        constexpr int tile16x32Vgprs = 64 * mmacVgprs;
        constexpr int tile32x32Vgprs = 2 * tile16x32Vgprs;
        constexpr int warpVgprs      = WARP_NUM * tile32x32Vgprs;

        #pragma unroll
        for (int min_tile_n = 0; min_tile_n < 2; ++min_tile_n) {
            #pragma unroll
            for (int h_idx = 0; h_idx < K_LOOP_COUNT; ++h_idx) {
                #pragma unroll
                for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                    int lds_offset = warp_id * warpVgprs + h_idx * tile32x32Vgprs + min_tile_m * tile16x32Vgprs + lane_id * mmacVgprs;
                    *(vec4_fp32*)(acc_o_lds + lds_offset) = acc_o[h_idx][min_tile_n * 2 + min_tile_m].f32;
                }
            }
            __syncthreads();
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                // lds base
                ElementAccum* acc_o_lds_ptr = acc_o_lds + 0 * warpVgprs + warp_id/*h_idx*/ * tile32x32Vgprs + min_tile_m * tile16x32Vgprs + lane_id * mmacVgprs;
                // load data of warp0 as accum base
                acc_o[0][min_tile_n * 2 + min_tile_m].f32 = *(vec4_fp32*)(acc_o_lds_ptr + 0 * warpVgprs);
                // load warp 1, 2, 3
                auto neighbor1 = *(union_vec4_fp32*)(acc_o_lds_ptr + 1 * warpVgprs);
                auto neighbor2 = *(union_vec4_fp32*)(acc_o_lds_ptr + 2 * warpVgprs);
                auto neighbor3 = *(union_vec4_fp32*)(acc_o_lds_ptr + 3 * warpVgprs);
                // accumulate acc_o of all warps
                #pragma unroll
                for (int vec_id = 0; vec_id < 2; ++vec_id) {
                    acc_o[0][min_tile_n * 2 + min_tile_m].u64[vec_id] = __builtin_hcu_pk_add_f32(acc_o[0][min_tile_n * 2 + min_tile_m].u64[vec_id], neighbor1.u64[vec_id]);
                    acc_o[0][min_tile_n * 2 + min_tile_m].u64[vec_id] = __builtin_hcu_pk_add_f32(acc_o[0][min_tile_n * 2 + min_tile_m].u64[vec_id], neighbor2.u64[vec_id]);
                    acc_o[0][min_tile_n * 2 + min_tile_m].u64[vec_id] = __builtin_hcu_pk_add_f32(acc_o[0][min_tile_n * 2 + min_tile_m].u64[vec_id], neighbor3.u64[vec_id]);
                }
            }
            __syncthreads();
        }
    } else {
        // To be inplemented
    }
}



template<bool Is_Varlen, bool Split, int kBlockK, int WARP_NUM, int K_LOOP_COUNT, int M_MMAC_COUNT, typename SplitkvAccumType, typename ElementAccum, typename Params>
__forceinline__ __device__ void kvcache_varlen_epilogue_store_output(
        vec4_Accum<ElementAccum> acc_o[K_LOOP_COUNT][4],
        Params params,
        int64_t row_offset_o,
        int seqlen_q_limit,
        int warp_id,
        int lane_id) {
    int o_mmac_row = lane_id & 15;
    int o_mmac_col = lane_id >> 4;
    int o_seq_stride = params.o_row_stride;
    SplitkvAccumType* o_ptr = reinterpret_cast<SplitkvAccumType *>(Split ? params.oaccum_ptr: params.o_ptr) + row_offset_o;

    if constexpr (K_LOOP_COUNT == 4 and WARP_NUM == 4) {
        // each warp output serveral tiles separately
        #pragma unroll
        for (int k_loop = 0; k_loop < K_LOOP_COUNT; k_loop += WARP_NUM/*1*/) {
            int tile_32x32_id = 0/*k_loop*/;
            union_vec4_f16x2<SplitkvAccumType> o_data[M_MMAC_COUNT];
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                // 2-interleave
                o_data[min_tile_m].f16x2[0 + 0 * 2] = DownCastPairNoPack<ElementAccum, SplitkvAccumType>(acc_o[tile_32x32_id][min_tile_m + 0 * 2].f32[0], acc_o[tile_32x32_id][min_tile_m + 0 * 2].f32[1]);
                o_data[min_tile_m].f16x2[1 + 0 * 2] = DownCastPairNoPack<ElementAccum, SplitkvAccumType>(acc_o[tile_32x32_id][min_tile_m + 0 * 2].f32[2], acc_o[tile_32x32_id][min_tile_m + 0 * 2].f32[3]);
                o_data[min_tile_m].f16x2[0 + 1 * 2] = DownCastPairNoPack<ElementAccum, SplitkvAccumType>(acc_o[tile_32x32_id][min_tile_m + 1 * 2].f32[0], acc_o[tile_32x32_id][min_tile_m + 1 * 2].f32[1]);
                o_data[min_tile_m].f16x2[1 + 1 * 2] = DownCastPairNoPack<ElementAccum, SplitkvAccumType>(acc_o[tile_32x32_id][min_tile_m + 1 * 2].f32[2], acc_o[tile_32x32_id][min_tile_m + 1 * 2].f32[3]);
                // make 4-interleave
                ds_mpermute_kdim_for_mmac(o_data[min_tile_m].f16x4[0]);
                ds_mpermute_kdim_for_mmac(o_data[min_tile_m].f16x4[1]);
            }
            union_vec4_f16x2<SplitkvAccumType> o_dwordx4[M_MMAC_COUNT];
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                flash::wait_lds_data_arrived<false>((M_MMAC_COUNT - 1 - min_tile_m) * 2);
                o_dwordx4[min_tile_m].f16[0] = o_data[min_tile_m].f16[0];
                o_dwordx4[min_tile_m].f16[1] = o_data[min_tile_m].f16[4];
                o_dwordx4[min_tile_m].f16[2] = o_data[min_tile_m].f16[1];
                o_dwordx4[min_tile_m].f16[3] = o_data[min_tile_m].f16[5];
                o_dwordx4[min_tile_m].f16[4] = o_data[min_tile_m].f16[2];
                o_dwordx4[min_tile_m].f16[5] = o_data[min_tile_m].f16[6];
                o_dwordx4[min_tile_m].f16[6] = o_data[min_tile_m].f16[3];
                o_dwordx4[min_tile_m].f16[7] = o_data[min_tile_m].f16[7];
            }
            #pragma unroll
            for (int min_tile_m = 0; min_tile_m < M_MMAC_COUNT; ++min_tile_m) {
                // store 4 dwords into global memory
                int seqlen_q_idx = o_mmac_row + min_tile_m * 16;
                if (seqlen_q_idx < seqlen_q_limit) {
                    int pv_global_addr;
                    if constexpr (Is_Varlen) {
                        int true_seqlen_q = seqlen_q_idx / params.ngroups;
                        int true_group_id = seqlen_q_idx % params.ngroups;
                        pv_global_addr = true_seqlen_q * params.ngroups * o_seq_stride + true_group_id * params.o_head_stride + (warp_id + 0) * kBlockK + o_mmac_col * 8;
                    } else {
                        pv_global_addr = seqlen_q_idx * o_seq_stride + (warp_id + 0) * kBlockK + o_mmac_col * 8;
                    }
                    *(vec4_fp32*)(o_ptr + pv_global_addr) = o_dwordx4[min_tile_m].f32;
                }
            }
        }
    } else {
        // To be inplemented
    }
}

} // end of namespace gfx92a