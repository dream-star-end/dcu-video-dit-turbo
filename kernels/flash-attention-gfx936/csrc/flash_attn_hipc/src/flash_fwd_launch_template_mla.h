/******************************************************************************************
 * Copyright (c) 2025, Baohui.Fang, Yushun.Zhang, Chang.liu, Wenjian.Zhang, Jianbang.Xu
 *****************************************************************************************/

#pragma once
#include "flash_fwd_launch_template_pa.h"


template<typename Kernel_traits, bool Is_causal, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_splitkv_mla_kernel(Params params) {
    flash::compute_attn_splitkv_mla<Kernel_traits, Is_causal, Split, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size>(params);
}


template<typename Kernel_traits, bool Is_causal, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_splitkv_mla_gfx938_kernel(Params params) {
    flash::compute_attn_splitkv_mla_gfx938<Kernel_traits, Is_causal, Split, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size>(params);
}

template<typename Kernel_traits, bool Is_causal, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_splitkv_fp8_mla_gfx938_kernel(Params params) {
    flash::compute_attn_splitkv_fp8_mla_gfx938<Kernel_traits, Is_causal, Split, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size>(params);
}


template<typename Kernel_traits, const bool Tail, typename Params>
void run_mla_splitkv_reduce(Params &params, hipStream_t stream) {
    static_assert (Kernel_traits::kHeadDimV == 512 and "run_mla_splitkv_reduce only support splitkv for hdimv == 512");
    using Element = typename Kernel_traits::Element;
    using SplitkvAccumType = typename Kernel_traits::SplitkvAccumType;
    // rearrange narrow-storation params
    Flash_fwd_mla_reduce_params reduce_params;
    reduce_params.softmax_lse_ptr = params.softmax_lse_ptr;
    reduce_params.oaccum_ptr      = params.oaccum_ptr;
    reduce_params.o_ptr           = params.o_ptr;
    reduce_params.cu_seqlens_k    = params.cu_seqlens_k;
    reduce_params.num_splits      = params.num_splits;
    reduce_params.partition_size  = params.partition_size;
    reduce_params.h               = params.h;
    reduce_params.seqlen_q        = params.seqlen_q;
    reduce_params.layout          = params.layout;
    // reduce num_splits x [batch_size, num_head_q, seqlen_q, head_dim] output
    if (params.num_splits > 1) {
        dim3 block(256);
        dim3 grid(params.b * params.h * params.seqlen_q, 4);
        constexpr int MAX_NUM_SPLITS = 64;
        if (params.num_splits > MAX_NUM_SPLITS) {
            printf("\x1b[31mnum_splits %d is larger than limit %d, and thus won't execute the kernel\033[0m\n", params.num_splits, MAX_NUM_SPLITS);
            return;
        }
        if (params.num_splits == 2) {
            flash_mla_splitkv_reduce_kernel<SplitkvAccumType, Element, 2, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(reduce_params);
        } else if (params.num_splits == 4) {
            flash_mla_splitkv_reduce_kernel<SplitkvAccumType, Element, 4, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(reduce_params);
        } else if (params.num_splits == 8) {
            flash_mla_splitkv_reduce_kernel<SplitkvAccumType, Element, 8, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(reduce_params);
        } else if (params.num_splits == 16) {
            flash_mla_splitkv_reduce_kernel<SplitkvAccumType, Element, 16, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(reduce_params);
        } else if (params.num_splits == 32) {
            flash_mla_splitkv_reduce_kernel<SplitkvAccumType, Element, 32, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(reduce_params);
        } else if (params.num_splits == 64) {
            flash_mla_splitkv_reduce_kernel<SplitkvAccumType, Element, 64, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(reduce_params);
        } else {
            printf("\x1b[31mnum_splits %d is not supported yet, and thus won't execute the kernel\033[0m\n", params.num_splits);
            return;
        }
    } // if (params.num_splits > 1)
}


template<typename Kernel_traits>
void run_flash_splitkv_mla(Flash_fwd_mla_params &params, hipStream_t stream) {

    // decide whether mls can be applied
    bool mls_enabled = getArch() >= 938 and (std::getenv("MLA_NO_MLS") == nullptr);

    // judge if mtp > 1, attention the causal mask ?
    bool use_mtp = params.mtp > 1; // bool(params.seqlen_q > 1 and !params.seqlenq_ngroups_swapped);
    // judge whether run with 16x32 tile
    bool use_tile_16x32 = params.seqlen_q <= 16/*16x32 tile*/ and std::getenv("MLA_USE_TILE32X32") == nullptr/*env control*/;

    constexpr int WARP_NUM     = Kernel_traits::kBlockN / Kernel_traits::kWaveN;
    const size_t smem_for_max  = std::max(WARP_NUM * Kernel_traits::kWaveM * sizeof(float), size_t(1024));
    /*every 2 in seqlen dimension requires 1024 KB lds, and thus, max lds should be limited within 64 * 2 = 128*/
    const size_t smem_for_acc  = int((params.seqlen_q + 1) / 2) * 2/*>= 的偶数*/ * WARP_NUM * Kernel_traits::kBlockK * sizeof(float);
    const size_t _q_smem_size  = Kernel_traits::STAGES == 2 ? Kernel_traits::q_smem_size: Kernel_traits::q_smem_size * (Kernel_traits::kHeadDim / Kernel_traits::kBlockK/*576需要加载几次*/) / 2; // 除以 2 是因为只需要用 16x32 的 lds, 节约用量
    const size_t q_smem_size = use_tile_16x32 ? _q_smem_size / 34 * 32: _q_smem_size;
    const size_t k_smem_size = use_tile_16x32 ? Kernel_traits::k_smem_size / 34 * 32 * WARP_NUM/*16x32 tile no padding*/: Kernel_traits::k_smem_size * WARP_NUM/*32x32 tile use padding 32 -> 34*/;
    const size_t v_smem_size = Kernel_traits::v_smem_size * WARP_NUM;
    const size_t smem_for_gemm = Kernel_traits::STAGES == 2 ? std::max(q_smem_size, std::max(k_smem_size, v_smem_size)): q_smem_size + std::max(k_smem_size, v_smem_size);
    const size_t required_smem_size = std::max(smem_for_acc, smem_for_gemm + smem_for_max);
    const size_t smem_size     = mls_enabled ? (params.b >= params.cu_count ? 64 * 1024: 32 * 1024): (use_tile_16x32 ? 32 * 1024: required_smem_size);

    // compute block partition along seqlen_q direction
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

    // acquire kernel fuction
    void (*kernel)(Flash_fwd_mla_params);

    // decide task dispatch logic
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.h, params.num_splits > 1 ? params.b * params.h : params.b);

    if (std::getenv("FA_DEBUG") != nullptr) {
        printf("smem_for_max: %ld | smem_for_acc: %ld | q_smem: %ld  k_smem: %ld  v_smem: %ld | smem_for_gemm: %ld | needed required_smem_size: %ld | smem_size: %ld\nuse_tile_16x32: %d\n",
            smem_for_max, smem_for_acc, q_smem_size, k_smem_size, v_smem_size, smem_for_gemm, required_smem_size, smem_size, use_tile_16x32);
        printf("dispatch grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    }

    // decide which kernel function
    if (use_tile_16x32) {
        // mtp is not considered yet for 16x32 tile
        constexpr int HEADDIM_V_SPLIT = 1; // no need to split-D
        constexpr int M_MMAC_COUNT    = 1; // only need to compute 16x576 @ 576xseqlenk
        grid.x = num_m_block * HEADDIM_V_SPLIT;
        MLA_PARTITION_SIZE_SWITCH(params.partition_size, params.num_splits, [&] {
            if (use_mtp) {
                if (!params.seqlenq_ngroups_swapped) {
                    kernel = mls_enabled
                    ? &flash_fwd_splitkv_mla_gfx938_kernel<Kernel_traits, true/*Is_causal*/, bool(Partition_Size > 0), M_MMAC_COUNT, 0, HEADDIM_V_SPLIT, Partition_Size, Flash_fwd_mla_params>
                    : &flash_fwd_splitkv_mla_kernel<Kernel_traits, true/*Is_causal*/, bool(Partition_Size > 0), M_MMAC_COUNT, 0, HEADDIM_V_SPLIT, Partition_Size, Flash_fwd_mla_params>;
                } else {
                    MLA_REUSEKV_SWITCH(params.seqlen_q, [&] {
                        kernel = mls_enabled
                        ? &flash_fwd_splitkv_mla_gfx938_kernel<Kernel_traits, true/*Is_causal*/, bool(Partition_Size > 0), M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size, Flash_fwd_mla_params>
                        : &flash_fwd_splitkv_mla_kernel<Kernel_traits, true/*Is_causal*/, bool(Partition_Size > 0), M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size, Flash_fwd_mla_params>;
                    });
                }
            } else {
                MLA_REUSEKV_SWITCH(params.seqlen_q, [&] {
                    kernel = mls_enabled
                    ? &flash_fwd_splitkv_mla_gfx938_kernel<Kernel_traits, false/*Is_causal*/, bool(Partition_Size > 0), M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size, Flash_fwd_mla_params>
                    : &flash_fwd_splitkv_mla_kernel<Kernel_traits, false/*Is_causal*/, bool(Partition_Size > 0), M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size, Flash_fwd_mla_params>;
                });
            }
        });
    } else {
        // need to compute 32x576 @ 576xseqlenk, split-D is used to reduce vgpr spill
        constexpr int HEADDIM_V_SPLIT = 2;
        grid.x = num_m_block * HEADDIM_V_SPLIT;
        // fixed num_splits for mla
        if (params.partition_size == MLA_FIX_PARTITION and params.num_splits > 1) {
            M_MMAC_COUNT_SWITCH(params.seqlen_q > 1 or use_mtp, M_MMAC_COUNT, [&] {
                if (use_mtp) { // MTP > 1
                    if (!params.seqlenq_ngroups_swapped) {
                        kernel = &flash_fwd_splitkv_mla_kernel<Kernel_traits, true/*Is_causal*/, true/*Split*/, M_MMAC_COUNT, 0, HEADDIM_V_SPLIT, MLA_FIX_PARTITION>;
                    } else {
                        MLA_MTP_REUSEKV_SWITCH(params.seqlen_q, [&]{
                            kernel = &flash_fwd_splitkv_mla_kernel<Kernel_traits, true/*Is_causal*/, true/*Split*/, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, MLA_FIX_PARTITION>;
                        });
                    }
                } else {
                    kernel = &flash_fwd_splitkv_mla_kernel<Kernel_traits, false/*Is_causal*/, true/*Split*/, M_MMAC_COUNT, 0/*REUSE_KV_TIMES*/, HEADDIM_V_SPLIT, MLA_FIX_PARTITION>;
                }
            });
        } else {
            M_MMAC_COUNT_SWITCH(params.seqlen_q > 1 or use_mtp, M_MMAC_COUNT, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    if (use_mtp) { // MTP > 1
                        if (!params.seqlenq_ngroups_swapped) {
                            kernel = &flash_fwd_splitkv_mla_kernel<Kernel_traits, true/*Is_causal*/, Split, M_MMAC_COUNT, 0, HEADDIM_V_SPLIT, 0>;
                        } else {
                            MLA_MTP_REUSEKV_SWITCH(params.seqlen_q, [&]{
                                kernel = &flash_fwd_splitkv_mla_kernel<Kernel_traits, true/*Is_causal*/, Split, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, 0>;
                            });
                        }
                    } else {
                        kernel = &flash_fwd_splitkv_mla_kernel<Kernel_traits, false/*Is_causal*/, Split, M_MMAC_COUNT, 0/*REUSE_KV_TIMES*/, HEADDIM_V_SPLIT, 0>;
                    }
                });
            });
        }
    }

    // Kernel execution
    const int nthread = Kernel_traits::kBlockN / Kernel_traits::kWaveN * 64;
    kernel<<<grid, nthread, smem_size, stream>>>(params);

    // reduce
    run_mla_splitkv_reduce<Kernel_traits, false/*Tail*/>(params, stream);
}


template<typename T, int Headdim, int HeaddimV>
void run_mla_fwd_splitkv_dispatch(Flash_fwd_mla_params &params, hipStream_t stream) {
    // 是否编译多个 page block size, 代码会膨胀
    #ifdef MLA_PAGE_BLOCK_SIZE
        if (params.page_block_size % 32 != 0) { printf("\x1b[31mPage block size %d is not supported yet!\033[0m\n", params.page_block_size); return; }
        PA_PAGEBLOCKSIZE_SWITCH(params.page_block_size, [&]{
            run_flash_splitkv_mla<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 32, 32, 32, 2/*STAGES*/, false, false, T, T> >(params, stream);
        });
    #else
        constexpr int kBlockN = 128;
        if (params.page_block_size % kBlockN != 0) { printf("\x1b[31mPage block size %d is not supported yet!\033[0m\n", params.page_block_size); return; }
        run_flash_splitkv_mla<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 32, 32, 32, 2/*STAGES*/, false, false, T, T> >(params, stream);
        /*以前的 STAGES = 1, 适用于 "不做 splitD, seqlen<=16, 把 Q 不读到寄存器, 读到 lds, 且只塞 16x576 个 Half 到 lds" 的情况, 当时情况可行, 但后续性能被 16x32 tile 取代, 可以作为紧急时期的备选方案*/
    #endif
}


template<typename Kernel_traits>
void run_flash_splitkv_fp8_mla(Flash_fwd_mla_params &params, hipStream_t stream) {

    // judge if mtp > 1, attention the causal mask ?
    bool use_mtp = params.mtp > 1; // bool(params.seqlen_q > 1 and !params.seqlenq_ngroups_swapped);

    // compute block partition along seqlen_q direction
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

    // acquire kernel fuction
    void (*kernel)(Flash_fwd_mla_params);

    // decide task dispatch logic
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.h, params.num_splits > 1 ? params.b * params.h : params.b);

    // decide shared memory size
    const size_t smem_size = 32 * 1024;
    if (std::getenv("FA_DEBUG") != nullptr) {
        printf("smem_size: %ld\n", smem_size);
        printf("dispatch grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    }

    // decide which kernel function
    // mtp is not considered yet for 16x32 tile
    constexpr int HEADDIM_V_SPLIT = 1; // no need to split-D
    constexpr int M_MMAC_COUNT    = 1; // only need to compute 16x576 @ 576xseqlenk
    grid.x = num_m_block * HEADDIM_V_SPLIT;
    MLA_PARTITION_SIZE_SWITCH(params.partition_size, params.num_splits, [&] {
        if (use_mtp) {
            kernel = &flash_fwd_splitkv_fp8_mla_gfx938_kernel<Kernel_traits, true/*Is_causal*/, bool(Partition_Size > 0), M_MMAC_COUNT, 0, HEADDIM_V_SPLIT, Partition_Size, Flash_fwd_mla_params>;
        } else {
            // only tp8 is supported yet
            kernel = &flash_fwd_splitkv_fp8_mla_gfx938_kernel<Kernel_traits, false/*Is_causal*/, bool(Partition_Size > 0), M_MMAC_COUNT, 16, HEADDIM_V_SPLIT, Partition_Size, Flash_fwd_mla_params>;
        }
    });

    // Kernel execution
    const int nthread = Kernel_traits::kBlockN / Kernel_traits::kWaveN * 64;
    kernel<<<grid, nthread, smem_size, stream>>>(params);

    // reduce
    run_mla_splitkv_reduce<Kernel_traits, false/*Tail*/>(params, stream);
}

template<typename T, int Headdim, int HeaddimV>
void run_fp8_mla_fwd_splitkv_dispatch(Flash_fwd_mla_params &params, hipStream_t stream) {
    constexpr int kBlockN = 128;
    if (params.page_block_size % kBlockN != 0) { printf("\x1b[31mPage block size %d is not supported yet!\033[0m\n", params.page_block_size); return; }
    run_flash_splitkv_fp8_mla<Flash_fwd_kernel_traits<Headdim, HeaddimV, 16, kBlockN, 32, 32, 32, 2/*STAGES*/, false, false, T, T> >(params, stream);
}


template<typename T, int Headdim, int HeaddimV>
void run_fp8_mla_convert_q_to_fp8_dispatch(Flash_fwd_mla_params &params, hipStream_t stream) {
    BOOL_SWITCH(params.total_blocks > 256, is_persistent, [&]{
        dim3 grid(is_persistent ? 512: params.total_blocks);
        dim3 block(64, 1, 1);
        flash_mla_convert_query_to_fp8_kernel<T, fp8_e4m3, is_persistent><<<grid, block, 8192, stream>>>(
            reinterpret_cast<fp8_e4m3*>(params.o_ptr),
            reinterpret_cast<T*>(params.qv_ptr),
            reinterpret_cast<T*>(params.q_ptr),
            params.total_blocks,
            params.o_head_stride,
            params.qv_head_stride,
            params.q_head_stride,
            params.qv_row_stride,
            params.q_row_stride,
            params.h
        );
    });
}


template<typename T, int Headdim, int HeaddimV>
void run_mla_fwd_prefix_prefill_dispatch(Flash_fwd_mla_params &params, hipStream_t stream) {
    int gcn_arch = getArch();
    if (gcn_arch >= 938 and std::getenv("MLA_PREFILL_NO_MLS") == nullptr) {
        constexpr int kBlockM = 128;
        constexpr int kBlockN = 64;
        constexpr int WARP_M  = 16;
        dim3 dimBlock;
        dim3 dimGrid;

        dimBlock.x = min((kBlockM) / (WARP_M) * 64, 1024);
        dimBlock.y = 1;
        dimBlock.z = 1;

        using Kernel_traits = Flash_fwd_kernel_traits<Headdim, HeaddimV, kBlockM, kBlockN, 32/* kBlockK */, WARP_M, 64/* WARP_N */, 2/* STAGES */, false, false, T>;

        constexpr int REUSE_KV = 1;
        constexpr bool Is_dropout = false;

        if (params.is_causal) {
            dimGrid.x = (params.seqlen_q + 2 * kBlockM - 1) / (2 * kBlockM);
        } else {
            dimGrid.x = (params.seqlen_q + 1 * kBlockM - 1) / (1 * kBlockM);
        }
        dimGrid.y = (params.h == params.h_k) ? params.h: params.h / REUSE_KV;
        dimGrid.z = params.b;

        constexpr bool IsEvenMNConst = false;
        const bool is_mtp     = (params.mtp != 0) ? true : false;
        if (is_mtp == false) {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                    flash::flash_fwd_mla_prefill_kernel_gfx938<Kernel_traits, true/*Is_training*/, Is_dropout, true/* Is_prefix */,Is_causal, IsEvenMNConst, true/*Is_even_K*/, false/*Return_softmax*/, false/* Is_MTP */, 0, Flash_fwd_mla_params>
                                <<<dimGrid, dimBlock, 32 * 1024, stream>>>(params); // 暂时保留MTP模板参数，实测无误后删除
            });
        } else {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                    flash::flash_fwd_mla_decode_kernel_gfx938<Kernel_traits, true/*Is_training*/, Is_dropout, true/* Is_prefix */,Is_causal, IsEvenMNConst, true/*Is_even_K*/, false/*Return_softmax*/, true/* Is_MTP */, 0, Flash_fwd_mla_params>
                                <<<dimGrid, dimBlock, 32 * 1024, stream>>>(params);
            });
        }
    } else {
        if (params.b * params.h >= params.cu_count) {
            constexpr int kBlockM = 32; // vgpr spill 280+ when WARP_M = 32
            constexpr int kBlockN = 128;
            constexpr int parallel = 2;
            params.q_blocks = (params.seqlen_q + 1 * kBlockM - 1) / (1 * kBlockM); // todo: streamkv
            params.q_blocks = (params.q_blocks + 1) / parallel;
            dim3 grid(parallel, params.h, params.b); // todo: regroup qheads into seqlen_q and dispatch less blocks
            dim3 block(256, 1, 1);
            flash_fwd_mla_prefix_prefill_kernel<Headdim, HeaddimV, kBlockM, kBlockN, true/*Is_prefix*/, false/*Is_causal*/, T, float, Flash_fwd_mla_params><<<grid, block, 0, stream>>>(params);
        } else {
            constexpr int kBlockM = 64;
            constexpr int kBlockN = 128;
            params.q_blocks = (params.seqlen_q + 2 * kBlockM - 1) / (2 * kBlockM);
            params.total_blocks = params.b * params.h * params.q_blocks;
            dim3 grid(params.cu_count); // (params.q_blocks, params.h, params.b) when no fix, corresponding to #elif 0
            dim3 block(512, 1, 1);
            flash_fwd_mla_prefix_prefill_fix_kernel<Headdim, HeaddimV, kBlockM, kBlockN, true/*Is_prefix*/, false/*Is_causal*/, T, float, Flash_fwd_mla_params><<<grid, block, 0, stream>>>(params);
        }
    }
}


template<typename T, int Headdim, int HeaddimV>
void run_mla_fwd_dispatch(Flash_fwd_mla_params &params, hipStream_t stream) {
    int gcn_arch = getArch();
    if (gcn_arch >= 938 and std::getenv("MLA_DP_DECODE_NO_MLS") == nullptr and params.b >= 16) {
        constexpr int kBlockM = 128;
        constexpr int kBlockN = 64;
        constexpr int WARP_M  = 16;
        dim3 dimBlock;
        dim3 dimGrid;

        dimBlock.x = min((kBlockM) / (WARP_M) * 64, 1024);
        dimBlock.y = 1;
        dimBlock.z = 1;

        using Kernel_traits = Flash_fwd_kernel_traits<Headdim, HeaddimV, kBlockM, kBlockN, 32/* kBlockK */, WARP_M, 64/* WARP_N */, 2/* STAGES */, false, false, T>;

        constexpr int REUSE_KV = 1;
        constexpr bool Is_dropout = false;

        dimGrid.x = (params.seqlen_q + 1 * kBlockM - 1) / (1 * kBlockM);
        dimGrid.y = (params.h == params.h_k) ? params.h: params.h / REUSE_KV;
        dimGrid.z = params.b;

        constexpr bool IsEvenMNConst = false;
        BOOL_SWITCH(params.mtp > 1/* is_mtp */, Is_MTP, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                    flash::flash_fwd_mla_decode_kernel_gfx938<Kernel_traits, true/*Is_training*/, Is_dropout, false/* Is_prefix | flashmla */,Is_causal, IsEvenMNConst, /*Is_even_K*/true, /*Return_softmax*/false, Is_MTP, 0, Flash_fwd_mla_params>
                                <<<dimGrid, dimBlock, 32 * 1024, stream>>>(params);
            });
        });
    } else {
        if (params.b * params.h >= params.cu_count / 2) {
            constexpr int kBlockM = 32; // vgpr spill 280+ when WARP_M = 32
            constexpr int kBlockN = 128;
            constexpr int parallel = 2;
            params.q_blocks = (params.seqlen_q + 1 * kBlockM - 1) / (1 * kBlockM); // todo: streamkv
            params.q_blocks = (params.q_blocks + 1) / parallel;
            dim3 grid(parallel, params.h, params.b); // todo: regroup qheads into seqlen_q and dispatch less blocks
            dim3 block(256, 1, 1);
            BOOL_SWITCH(params.mtp > 1, Is_causal, [&]{
                flash_fwd_mla_kernel<Headdim, HeaddimV, kBlockM, kBlockN, false/*Is_prefix*/, Is_causal, T, float, Flash_fwd_mla_params><<<grid, block, 0, stream>>>(params);
            });
        } else {
            constexpr int kBlockM = 64;
            constexpr int kBlockN = 128;
            params.q_blocks = (params.seqlen_q + 1 * kBlockM - 1) / (1 * kBlockM); // used in kernels
            params.total_blocks = params.b * params.h * params.q_blocks;
            // dim3 grid(params.q_blocks, params.h, params.b);
            dim3 grid(params.cu_count);
            dim3 block(512, 1, 1);
            BOOL_SWITCH(params.mtp > 1, Is_causal, [&]{
                flash_fwd_mla_fix_kernel<Headdim, HeaddimV, kBlockM, kBlockN, false/*Is_prefix*/, Is_causal, T, float, Flash_fwd_mla_params><<<grid, block, 0, stream>>>(params);
            });
        }
    }
}