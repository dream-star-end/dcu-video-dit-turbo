/******************************************************************************************
 * Copyright (c) 2025, Baohui.Fang, Yushun.Zhang, Chang.liu, Wenjian.Zhang, Jianbang.Xu
 *****************************************************************************************/

#pragma once
#include "config.h"
#include "static_switch.h"
#include "flash.h"
#include "flash_fwd_kernel.h"
#include "flash_singleton.h"
#include "assert.h"
#include <string>

static inline bool hg_pa_is_gfx92a(const std::string &gcn_arch_name) {
    return gcn_arch_name.rfind("gfx92a", 0) == 0;
}

static inline int hg_pa_runtime_gfx_arch_id(const std::string &gcn_arch_name) {
    return hg_pa_is_gfx92a(gcn_arch_name) ? 930 : std::stoi(gcn_arch_name.substr(3, 3));
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_Varlen, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Has_alibi, bool Is_GQA, bool Is_softcap, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, bool Append_KV>
__global__ void __launch_bounds__(256, 1) flash_fwd_splitkv_kernel(Flash_fwd_params params) {
    flash::compute_attn_splitkv<Kernel_traits, /*Is_training*/false, Is_dropout, Is_causal, Is_Varlen, Is_local, Is_even_K, Return_softmax, Has_alibi, Is_GQA, Is_softcap, Split, M_MMAC_COUNT, REUSE_KV_TIMES, Append_KV>(params);
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Return_softmax, bool Has_alibi, bool Is_GQA, bool Is_softcap, bool Split, int M_MMAC_COUNT, int REUSE_KV_TIMES, bool Append_KV>
__global__ void __launch_bounds__(256,1) flash_fwd_splitkv_int8_kernel(Flash_fwd_params params) {
    flash::compute_attn_splitkv_int8<Kernel_traits, /*Is_training*/false, Is_dropout, Is_causal, Is_local, Is_even_K, Return_softmax, Has_alibi, Is_GQA, Is_softcap, Split, M_MMAC_COUNT, REUSE_KV_TIMES, Append_KV>(params);
}


template<typename Kernel_traits, bool Is_causal, bool Is_Varlen, bool Split, bool Is_local, int M_MMAC_COUNT, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_splitkv_tile16x32_kernel(Params params) {
    flash::compute_attn_splitkv_tile16x32<Kernel_traits, Is_causal, Is_Varlen, Split, Is_local, M_MMAC_COUNT, HEADDIM_V_SPLIT, Partition_Size>(params);
}


template<typename Kernel_traits, bool Is_causal, bool Is_Varlen, bool Split, int M_MMAC_COUNT, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_splitkv_gfx938_kernel(Params params) {
    flash::compute_attn_splitkv_gfx938<Kernel_traits, Is_causal, Is_Varlen, Split, M_MMAC_COUNT, HEADDIM_V_SPLIT, Partition_Size>(params);
}


template<typename Kernel_traits, bool Is_causal, bool Is_Varlen, bool Split, bool Is_local, int M_MMAC_COUNT, int REUSE_KV_TIMES, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_splitkv_fp8_gfx938_kernel(Params params) {
    flash::compute_attn_splitkv_fp8_gfx938<Kernel_traits, Is_causal, Is_Varlen, Split, Is_local, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size>(params);
}

template<typename Kernel_traits, bool Is_causal, bool Is_Varlen, bool Split, bool Is_local, int M_MMAC_COUNT, int REUSE_KV_TIMES, int HEADDIM_V_SPLIT, int Partition_Size, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_splitkv_fp8_gfx938_hdim192_v128_kernel(Params params) {
    flash::compute_attn_splitkv_fp8_gfx938_hdim192_v128<Kernel_traits, Is_causal, Is_Varlen, Split, Is_local, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size>(params);
}

template<typename Kernel_traits, bool Is_causal, bool Is_Varlen, bool Is_monopolize, bool Split, int M_MMAC_COUNT, int HEADDIM_V_SPLIT, typename Params>
__global__ void __launch_bounds__(256, 1) flash_fwd_splitkv_gfx92a_kernel(Params params) {
    flash::compute_attn_splitkv_gfx92a<Kernel_traits, Is_causal, Is_Varlen, Is_monopolize, Split, M_MMAC_COUNT, HEADDIM_V_SPLIT>(params);
}

template<typename Kernel_traits, const bool Tail, typename Params>
void run_splitkv_reduce(Params &params, hipStream_t stream) {
    // now, only headdim 128/512 support splitkv, since shuffle kernel doesn't support other headdims
    if constexpr (Kernel_traits::kHeadDimV == 128 or Kernel_traits::kHeadDimV == 512 or Kernel_traits::kHeadDimV == 64) {
        // reduce num_splits x [batch_size, num_head_q, seqlen_q, head_dim] output
        if (params.num_splits > 1) {
            dim3 block(64);
            dim3 grid(params.b * params.h * params.seqlen_q);
            constexpr int MAX_NUM_SPLITS = 1024;
            if (params.num_splits > MAX_NUM_SPLITS) {
                printf("\x1b[31mnum_splits %d is larger than limit %d, and thus won't execute the kernel\033[0m\n", params.num_splits, MAX_NUM_SPLITS);
                return;
            }
            using Element = typename Kernel_traits::Element;
            using SplitkvAccumType = typename Kernel_traits::SplitkvAccumType;
            if (params.num_splits == 2) {
                flash_fwd_splitkv_reduce_kernel<SplitkvAccumType, Element, 2, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits == 4) {
                flash_fwd_splitkv_reduce_kernel<SplitkvAccumType, Element, 4, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits == 8) {
                flash_fwd_splitkv_reduce_kernel<SplitkvAccumType, Element, 8, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits == 16) {
                flash_fwd_splitkv_reduce_kernel<SplitkvAccumType, Element, 16, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits == 32) {
                flash_fwd_splitkv_reduce_kernel<SplitkvAccumType, Element, 32, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits == 64) {
                flash_fwd_splitkv_reduce_kernel<SplitkvAccumType, Element, 64, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits == 128) {
                block.x = params.num_splits;
                flash_fwd_splitkv_reduce_kernel_split128<SplitkvAccumType, Element, 128, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits == 256) {
                block.x = params.num_splits;
                flash_fwd_splitkv_reduce_kernel_split128<SplitkvAccumType, Element, 256, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits == 512) {
                block.x = params.num_splits;
                flash_fwd_splitkv_reduce_kernel_split128<SplitkvAccumType, Element, 512, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits == 1024) {
                block.x = params.num_splits;
                flash_fwd_splitkv_reduce_kernel_split128<SplitkvAccumType, Element, 1024, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            }
            /*kernels above can be extremely optimized when unroll is true*/
            else if (params.num_splits > 512) {
                block.x = 1024;
                flash_fwd_splitkv_reduce_kernel_split128<SplitkvAccumType, Element, 1024, false/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits > 256) {
                block.x = 512;
                flash_fwd_splitkv_reduce_kernel_split128<SplitkvAccumType, Element, 512, false/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits > 128) {
                block.x = 256;
                flash_fwd_splitkv_reduce_kernel_split128<SplitkvAccumType, Element, 256, false/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits > 64) {
                block.x = 128;
                flash_fwd_splitkv_reduce_kernel_split128<SplitkvAccumType, Element, 128, false/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits > 32) {
                flash_fwd_splitkv_reduce_kernel<SplitkvAccumType, Element, 64, false/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits > 16) {
                flash_fwd_splitkv_reduce_kernel<SplitkvAccumType, Element, 32, false/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else {
                flash_fwd_splitkv_reduce_kernel<SplitkvAccumType, Element, 64, false/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            }
        } // if (params.num_splits > 1)
    } // if (kHeadDim == 128)
}

template<typename Kernel_traits, const bool Tail, typename Params>
void run_splitkv_reduce_varlen(Params &params, hipStream_t stream) {
    // now, only headdim 128/512 support splitkv, since shuffle kernel doesn't support other headdims
    if constexpr (Kernel_traits::kHeadDimV == 128 or Kernel_traits::kHeadDimV == 512 or Kernel_traits::kHeadDimV == 64) {
        // reduce num_splits x [batch_size, num_head_q, seqlen_q, head_dim] output
        if (params.num_splits > 1) {
            dim3 block(64);
            dim3 grid(params.h * params.ngroups, params.b); /*total_q 是会变化的, 不能放在这里用于启动 cuda-graph*/
            constexpr int MAX_NUM_SPLITS = 64;
            if (params.num_splits > MAX_NUM_SPLITS) {
                printf("\x1b[31mnum_splits %d is larger than limit %d, and thus won't execute the kernel\033[0m\n", params.num_splits, MAX_NUM_SPLITS);
                return;
            }
            using Element = typename Kernel_traits::Element;
            using SplitkvAccumType = typename Kernel_traits::SplitkvAccumType;
            if (params.num_splits == 2) {
                flash_fwd_splitkv_reduce_varlen_kernel<SplitkvAccumType, Element, 2, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits == 4) {
                flash_fwd_splitkv_reduce_varlen_kernel<SplitkvAccumType, Element, 4, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits == 8) {
                flash_fwd_splitkv_reduce_varlen_kernel<SplitkvAccumType, Element, 8, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits == 16) {
                flash_fwd_splitkv_reduce_varlen_kernel<SplitkvAccumType, Element, 16, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits == 32) {
                flash_fwd_splitkv_reduce_varlen_kernel<SplitkvAccumType, Element, 32, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else if (params.num_splits == 64) {
                flash_fwd_splitkv_reduce_varlen_kernel<SplitkvAccumType, Element, 64, true/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            } else {
                flash_fwd_splitkv_reduce_varlen_kernel<SplitkvAccumType, Element, 64, false/*unroll*/, Tail, Kernel_traits::kHeadDimV><<<grid, block, 0, stream>>>(params);
            }
        } // if (params.num_splits > 1)
    } // if (kHeadDim == 128)
}


template<typename Kernel_traits>
void run_flash_splitkv_fwd(Flash_fwd_params &params, hipStream_t stream) {

    constexpr int WARP_NUM     = Kernel_traits::kBlockN / Kernel_traits::kWaveN;
    const size_t smem_for_max  = std::max(WARP_NUM * Kernel_traits::kWaveM * sizeof(float), size_t(1024));
    const size_t smem_misalign = (params.seqlen_q >= 16 or Kernel_traits::kHeadDimV == 512) ? Kernel_traits::kHeadDimV: (Kernel_traits::kHeadDimV + 4)/*<=15 can use misalign to reduce bank conflicts, but >16 may lead to lds>32KB, less waves per SIMD*/;
    const size_t smem_for_acc  = int((params.seqlen_q + 1) / 2) * 2 * WARP_NUM * smem_misalign * sizeof(float);
    const size_t smem_for_gemm = std::max(std::max(Kernel_traits::q_smem_size, Kernel_traits::k_smem_size * WARP_NUM), Kernel_traits::v_smem_size * WARP_NUM);
    const size_t required_smem_size = std::max(smem_for_acc, smem_for_gemm + smem_for_max);
    /*
        for gfx936,
            2 waves per SIMD is better than 1 waves per SIMD;
            3 waves per SIMD will bring performance degradation
        for gfx928,
            > 1 waves per SIMD will significantly increase the latency of buffer-load, blocked at TA
    */
    hipDeviceProp_t props;
    auto hip_result = hipGetDeviceProperties(&props, 0);
    #ifdef ROCM_5_7
        int gcn_arch = props.gcnArch;
    #else
        std::string gcn_arch_name(props.gcnArchName);
        int gcn_arch = hg_pa_runtime_gfx_arch_id(gcn_arch_name);
    #endif
    const size_t smem_size = gcn_arch > 928 ? required_smem_size: size_t(64 * 1024);
    if (std::getenv("FA_DEBUG") != nullptr) {
        printf("smem_for_max: %ld | smem_for_acc: %ld | smem_for_gemm: %ld | needed smem_size: %ld | smem_size: %ld\n", smem_for_max, smem_for_acc, smem_for_gemm, required_smem_size, smem_size);
    }

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    const int nthread = Kernel_traits::kBlockN / Kernel_traits::kWaveN * 64;

    // dim3 grid(num_m_block, params.h, params.b);
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.h, params.num_splits > 1 ? params.b * params.h : params.b);
    // const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool has_alibi = (params.alibi_slopes_ptr not_eq nullptr);

    BOOL_SWITCH(has_alibi, Has_Alibi, [&] {
        M_MMAC_COUNT_SWITCH(params.seqlen_q > 1, M_MMAC_COUNT, [&] {
            BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                constexpr bool IsEvenMNConst = false;
                constexpr bool Is_local = false;
                void (*kernel)(Flash_fwd_params);
                if (params.mtp > 1) {
                    kernel = &flash_fwd_splitkv_kernel<Kernel_traits, false, true/*Is_causal*/, false/*Is_Varlen*/, Is_local, IsEvenMNConst && !Is_local && true && Kernel_traits::kHeadDim <= 256, true, false, Has_Alibi, false/*is_gqa*/, false/*Is_softcap*/, Split, M_MMAC_COUNT, 0, false/*Append_KV*/>;
                } else {
                    if constexpr (Kernel_traits::kHeadDim == 128 and Kernel_traits::kHeadDimV == 128) {
                        BOOL_SWITCH(params.q_batch_stride == 0, Is_Varlen, [&] {
                            REUSEKV_SWITCH(params.seqlen_q, [&] {
                                kernel = &flash_fwd_splitkv_kernel<Kernel_traits, false, false/*Is_causal*/, Is_Varlen, Is_local, IsEvenMNConst && !Is_local && true && Kernel_traits::kHeadDim <= 256, true, false, Has_Alibi, false/*is_gqa*/, false/*Is_softcap*/, Split, M_MMAC_COUNT, REUSE_KV_TIMES, false/*Append_KV*/>;
                            });
                        });
                    } else { // non-headdim128 cases
                        REUSEKV_SWITCH(params.seqlen_q, [&] {
                            kernel = &flash_fwd_splitkv_kernel<Kernel_traits, false, false/*Is_causal*/, false/*Is_Varlen*/, Is_local, IsEvenMNConst && !Is_local && true && Kernel_traits::kHeadDim <= 256, true, false, Has_Alibi, false/*is_gqa*/, false/*Is_softcap*/, Split, M_MMAC_COUNT, REUSE_KV_TIMES, false/*Append_KV*/>;
                        });
                    }
                }
                kernel<<<grid, nthread, smem_size, stream>>>(params);
            });
        });
    });

    // reduce PA v2
    if (params.q_batch_stride == 0) {
        run_splitkv_reduce_varlen<Kernel_traits, false/*Tail*/>(params, stream);
    } else {
        run_splitkv_reduce<Kernel_traits, true/*Tail*/>(params, stream);
    }
}


template<typename Kernel_traits>
void run_flash_splitkv_fwd_tile16x32(Flash_fwd_params &params, hipStream_t stream) {

    constexpr int WARP_NUM     = Kernel_traits::kBlockN / Kernel_traits::kWaveN;
    const size_t smem_for_max  = std::max(WARP_NUM * Kernel_traits::kWaveM * sizeof(float), size_t(1024));
    const size_t smem_for_acc  = int((params.seqlen_q + 1) / 2) * 2/*reserved for M_MMAC_COUNT = 2*/ * WARP_NUM * Kernel_traits::kBlockK * sizeof(float);
    const size_t q_smem_size   = Kernel_traits::STAGES * Kernel_traits::kBlockM * Kernel_traits::kBlockK * sizeof(half_t);
    const size_t k_smem_size   = Kernel_traits::STAGES * Kernel_traits::kBlockK * Kernel_traits::kWaveN * sizeof(half_t) * WARP_NUM;
    const size_t v_smem_size   = k_smem_size;
    const size_t smem_for_gemm = std::max(q_smem_size, std::max(k_smem_size, v_smem_size));
    const size_t required_smem_size = std::max(smem_for_acc, std::max(smem_for_gemm, smem_for_max));
    hipDeviceProp_t props;
    auto hip_result = hipGetDeviceProperties(&props, 0);
    #ifdef ROCM_5_7
        int gcn_arch = props.gcnArch;
    #else
        std::string gcn_arch_name(props.gcnArchName);
        int gcn_arch = hg_pa_runtime_gfx_arch_id(gcn_arch_name);
    #endif
    const size_t smem_size = gcn_arch > 928 ? size_t(std::max<size_t>(32 * 1024, required_smem_size)): size_t(64 * 1024);
    if (std::getenv("FA_DEBUG") != nullptr) {
        printf("smem_for_max: %ld | smem_for_acc: %ld | smem_for_gemm: %ld | needed smem_size: %ld | smem_size: %ld\n", smem_for_max, smem_for_acc, smem_for_gemm, required_smem_size, smem_size);
    }

    // compute block partition along seqlen_q direction
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

    // decide task dispatch logic
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.h, params.num_splits > 1 ? params.b * params.h : params.b);

    // acquire kernel fuction
    void (*kernel)(Flash_fwd_params);

    constexpr int HEADDIM_V_SPLIT = 1; // no need to split-D
    grid.x = num_m_block * HEADDIM_V_SPLIT;
    BOOL_SWITCH(params.q_batch_stride == 0, Is_Varlen, [&] {
        if (params.window_size_left > 0 and params.window_size_right >= 0) {
            M_MMAC_COUNT_SWITCH(params.seqlen_q > 16, M_MMAC_COUNT, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    kernel = &flash_fwd_splitkv_tile16x32_kernel<Kernel_traits, true/*Is_causal*/, Is_Varlen, Split, true/*Is_local*/, M_MMAC_COUNT, HEADDIM_V_SPLIT, 0>;
                });
            });
        } else if (params.mtp == 1) {
            M_MMAC_COUNT_SWITCH(params.seqlen_q > 16, M_MMAC_COUNT, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    constexpr int Partition_Size = 0; // pa adopt floor in splitkv, need to process tails, and thus cannot use partition_size unroll
                    REUSEKV_SWITCH(params.seqlen_q, [&] {
                        kernel = &flash_fwd_splitkv_tile16x32_kernel<Kernel_traits, false/*Is_causal*/, Is_Varlen, Split, false/*Is_local*/, M_MMAC_COUNT, HEADDIM_V_SPLIT, Partition_Size>;
                    });
                });
            });
        } else {
            M_MMAC_COUNT_SWITCH(params.seqlen_q > 16, M_MMAC_COUNT, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    constexpr int Partition_Size = 0; // pa adopt floor in splitkv, need to process tails, and thus cannot use partition_size unroll
                    PA_MTP_REUSEKV_SWITCH(params.seqlen_q, [&] {
                        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                            kernel = &flash_fwd_splitkv_tile16x32_kernel<Kernel_traits, Is_causal, Is_Varlen, Split, false/*Is_local*/, M_MMAC_COUNT, HEADDIM_V_SPLIT, Partition_Size>;
                        });
                    });
                });
            });
        }
    });

    // Kernel execution
    const int nthread = Kernel_traits::kBlockN / Kernel_traits::kWaveN * 64;
    kernel<<<grid, nthread, smem_size, stream>>>(params);

    // reduce PA v2
    if (params.q_batch_stride == 0) {
        run_splitkv_reduce_varlen<Kernel_traits, false/*Tail*/>(params, stream);
    } else {
        run_splitkv_reduce<Kernel_traits, true/*Tail*/>(params, stream);
    }
}


template<typename Kernel_traits>
void run_flash_splitkv_fwd_mha(Flash_fwd_params &params, hipStream_t stream) {

    // acquire kernel fuction
    void (*kernel)(Flash_fwd_params);
    kernel = &flash_fwd_splitkv_mha_kernel<Kernel_traits, false/*Is_causal*/, false/*Split*/, 1/*HEADDIM_V_SPLIT*/, 0/*Partition_Size*/>;

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.h, params.b);

    // Kernel execution
    const int nthread = Kernel_traits::kBlockN / Kernel_traits::kWaveN * 64;
    kernel<<<grid, nthread, 0, stream>>>(params);

}


template<typename Kernel_traits>
void run_flash_splitkv_fwd_gfx938(Flash_fwd_params &params, hipStream_t stream) {

    constexpr int WARP_NUM     = Kernel_traits::kBlockN / Kernel_traits::kWaveN;
    const size_t smem_for_max  = std::max(WARP_NUM * Kernel_traits::kWaveM * sizeof(float), size_t(1024));
    const size_t smem_for_acc  = int((params.seqlen_q + 1) / 2) * 2/*reserved for M_MMAC_COUNT = 2*/ * WARP_NUM * Kernel_traits::kBlockK * sizeof(float);
    const size_t q_smem_size   = Kernel_traits::STAGES * Kernel_traits::kBlockM * Kernel_traits::kBlockK * sizeof(half_t);
    const size_t k_smem_size   = Kernel_traits::STAGES * Kernel_traits::kBlockK * Kernel_traits::kWaveN * sizeof(half_t) * WARP_NUM;
    const size_t v_smem_size   = k_smem_size;
    const size_t smem_for_gemm = std::max(q_smem_size, std::max(k_smem_size, v_smem_size));
    const size_t required_smem_size = std::max(smem_for_acc, std::max(smem_for_gemm, smem_for_max));
    const size_t smem_size = size_t(std::max<size_t>(32 * 1024, required_smem_size));

    // compute block partition along seqlen_q direction
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

    // decide task dispatch logic
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.h, params.num_splits > 1 ? params.b * params.h : params.b);

    if (std::getenv("FA_DEBUG") != nullptr) {
        printf("smem_for_max: %ld | smem_for_acc: %ld | q_smem: %ld  k_smem: %ld  v_smem: %ld | smem_for_gemm: %ld | needed required_smem_size: %ld | smem_size: %ld\n",
            smem_for_max, smem_for_acc, q_smem_size, k_smem_size, v_smem_size, smem_for_gemm, required_smem_size, smem_size);
        printf("grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    }

    // acquire kernel fuction
    void (*kernel)(Flash_fwd_params);

    constexpr int HEADDIM_V_SPLIT = Kernel_traits::kHeadDimV == 256 ? 2 : 1;
    grid.x = num_m_block * HEADDIM_V_SPLIT;
    BOOL_SWITCH(params.q_batch_stride == 0, Is_Varlen, [&] {
        BOOL_SWITCH(params.mtp != 1, Is_causal, [&]{
            M_MMAC_COUNT_SWITCH(params.seqlen_q > 16, M_MMAC_COUNT, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    constexpr int Partition_Size = 0; // pa adopt floor in splitkv, need to process tails, and thus cannot use partition_size unroll
                    kernel = &flash_fwd_splitkv_gfx938_kernel<Kernel_traits, Is_causal, Is_Varlen, Split, M_MMAC_COUNT, HEADDIM_V_SPLIT, Partition_Size>;
                });
            });
        });
    });

    // Kernel execution
    const int nthread = Kernel_traits::kBlockN / Kernel_traits::kWaveN * 64;
    kernel<<<grid, nthread, smem_size, stream>>>(params);

    // reduce PA v2
    if (params.q_batch_stride == 0) {
        run_splitkv_reduce_varlen<Kernel_traits, false/*Tail*/>(params, stream);
    } else {
        run_splitkv_reduce<Kernel_traits, true/*Tail*/>(params, stream);
    }
}


template<typename Kernel_traits>
void run_flash_splitkv_fwd_gfx92a(Flash_fwd_params &params, hipStream_t stream) {

    // compute block partition along seqlen_q direction
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

    // decide task dispatch logic
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.h, params.num_splits > 1 ? params.b * params.h : params.b);

    // decide shared memory
    size_t smem_size = 32768;
    if (grid.x * grid.y * grid.z <= params.cu_count) { smem_size = 65536; }
    if (std::getenv("PA_NO_ALL_LDS") != nullptr) { smem_size = 32768; }
    if (std::getenv("PA_USE_ALL_LDS") != nullptr) { smem_size = 65536; }

    // output some details
    if (std::getenv("FA_DEBUG") != nullptr) {
        printf("smem_size: %ld\n", smem_size);
        printf("grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    }

    // acquire kernel fuction
    void (*kernel)(Flash_fwd_params);

    constexpr int HEADDIM_V_SPLIT = Kernel_traits::kHeadDimV == 256 ? 2 : 1;
    grid.x = num_m_block * HEADDIM_V_SPLIT;
    BOOL_SWITCH(params.q_batch_stride == 0, Is_Varlen, [&] {
        BOOL_SWITCH(params.mtp != 1, Is_causal, [&] {
            M_MMAC_COUNT_SWITCH(params.seqlen_q > 16, M_MMAC_COUNT, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    BOOL_SWITCH(smem_size == 65536, Is_monopolize, [&]{
                        kernel = &flash_fwd_splitkv_gfx92a_kernel<Kernel_traits, Is_causal, Is_Varlen, Is_monopolize, Split, M_MMAC_COUNT, HEADDIM_V_SPLIT>;
                    });
                });
            });
        });
    });

    // Kernel execution
    const int nthread = Kernel_traits::kBlockN / Kernel_traits::kWaveN * 64;
    kernel<<<grid, nthread, smem_size, stream>>>(params);

    // reduce PA v2
    if (params.q_batch_stride == 0) {
        run_splitkv_reduce_varlen<Kernel_traits, false/*Tail*/>(params, stream);
    } else {
        run_splitkv_reduce<Kernel_traits, true/*Tail*/>(params, stream);
    }
}


template<typename Kernel_traits>
void run_fp8_flash_splitkv_fwd_gfx938(Flash_fwd_params &params, hipStream_t stream) {

    constexpr int WARP_NUM     = Kernel_traits::kBlockN / Kernel_traits::kWaveN;
    constexpr int kReduceBlockK = 32;
    const size_t smem_for_max  = std::max(WARP_NUM * Kernel_traits::kWaveM * sizeof(float), size_t(1024));
    const size_t smem_for_acc  = Kernel_traits::kBlockM * WARP_NUM * kReduceBlockK * sizeof(float);
    const size_t q_smem_size   = Kernel_traits::STAGES * Kernel_traits::kBlockM * Kernel_traits::kBlockK * sizeof(Float8_e4m3_t);
    const size_t k_smem_size   = Kernel_traits::STAGES * Kernel_traits::kBlockK * Kernel_traits::kWaveN * sizeof(Float8_e4m3_t) * WARP_NUM;
    const size_t v_smem_size   = k_smem_size;
    const size_t smem_for_gemm = std::max(q_smem_size, std::max(k_smem_size, v_smem_size));
    constexpr bool IsFp8PA192x128 = Kernel_traits::kHeadDim == 192 && Kernel_traits::kHeadDimV == 128;
    const size_t required_smem_size = IsFp8PA192x128
        ? std::max(smem_for_acc, std::max(smem_for_gemm, smem_for_max))
        : std::max(smem_for_acc, smem_for_gemm + smem_for_max);
    const size_t smem_size_floor = IsFp8PA192x128 ? size_t(32 * 1024) : size_t(17 * 1024);
    const size_t smem_size = size_t(std::max(smem_size_floor, required_smem_size));

    if (std::getenv("FA_DEBUG") != nullptr) {
        printf("smem_for_max: %ld | smem_for_acc: %ld | q_smem: %ld  k_smem: %ld  v_smem: %ld | smem_for_gemm: %ld | needed required_smem_size: %ld | smem_size: %ld\n",
            smem_for_max, smem_for_acc, q_smem_size, k_smem_size, v_smem_size, smem_for_gemm, required_smem_size, smem_size);
    }

    // compute block partition along seqlen_q direction
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

    // decide task dispatch logic
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.h, params.num_splits > 1 ? params.b * params.h : params.b);

    // acquire kernel fuction
    void (*kernel)(Flash_fwd_params);

    constexpr int HEADDIM_V_SPLIT = Kernel_traits::kHeadDimV == 256 ? 2 : 1;
    grid.x = num_m_block * HEADDIM_V_SPLIT;
    BOOL_SWITCH(params.q_batch_stride == 0, Is_Varlen, [&] {
        if (params.window_size_left > 0 && params.window_size_right >= 0) {
            M_MMAC_COUNT_SWITCH(params.seqlen_q > 16, M_MMAC_COUNT, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    constexpr int Partition_Size = 0; // pa adopt floor in splitkv, need to process tails, and thus cannot use partition_size unroll
                    PA_MTP_REUSEKV_SWITCH(params.seqlen_q, [&] {
                        if constexpr (IsFp8PA192x128) {
                            kernel = &flash_fwd_splitkv_fp8_gfx938_hdim192_v128_kernel<Kernel_traits, true/*Is_causal*/, Is_Varlen, Split, true/*Is_local*/, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size>;
                        } else {
                            kernel = &flash_fwd_splitkv_fp8_gfx938_kernel<Kernel_traits, true/*Is_causal*/, Is_Varlen, Split, true/*Is_local*/, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size>;
                        }
                    });
                });
            });
        } else if (params.mtp == 1) {
            M_MMAC_COUNT_SWITCH(params.seqlen_q > 16, M_MMAC_COUNT, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    constexpr int Partition_Size = 0; // pa adopt floor in splitkv, need to process tails, and thus cannot use partition_size unroll
                    REUSEKV_SWITCH(params.seqlen_q, [&] {
                        constexpr bool Is_local = false;
                        if constexpr (IsFp8PA192x128) {
                            kernel = &flash_fwd_splitkv_fp8_gfx938_hdim192_v128_kernel<Kernel_traits, false/*Is_causal*/, Is_Varlen, Split, Is_local, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size>;
                        } else {
                            kernel = &flash_fwd_splitkv_fp8_gfx938_kernel<Kernel_traits, false/*Is_causal*/, Is_Varlen, Split, Is_local, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size>;
                        }
                    });
                });
            });
        } else {
            M_MMAC_COUNT_SWITCH(params.seqlen_q > 16, M_MMAC_COUNT, [&] {
                BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    constexpr int Partition_Size = 0; // pa adopt floor in splitkv, need to process tails, and thus cannot use partition_size unroll
                    PA_MTP_REUSEKV_SWITCH(params.seqlen_q, [&] {
                        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                            if constexpr (IsFp8PA192x128) {
                                kernel = &flash_fwd_splitkv_fp8_gfx938_hdim192_v128_kernel<Kernel_traits, Is_causal, Is_Varlen, Split, false/*Is_local*/, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size>;
                            } else {
                                kernel = &flash_fwd_splitkv_fp8_gfx938_kernel<Kernel_traits, Is_causal, Is_Varlen, Split, false/*Is_local*/, M_MMAC_COUNT, REUSE_KV_TIMES, HEADDIM_V_SPLIT, Partition_Size>;
                            }
                        });
                    });
                });
            });
        }
    });

    // Kernel execution
    const int nthread = Kernel_traits::kBlockN / Kernel_traits::kWaveN * 64;
    kernel<<<grid, nthread, smem_size, stream>>>(params);

    // reduce PA v2
    if (params.q_batch_stride == 0) {
        run_splitkv_reduce_varlen<Kernel_traits, false/*Tail*/>(params, stream);
    } else {
        run_splitkv_reduce<Kernel_traits, true/*Tail*/>(params, stream);
    }
}


template<typename T, int Headdim, int HeaddimV>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, hipStream_t stream) {
    // decide whether commonly used headdims
    const bool is_commonly_used = params.d % 64 == 0 and params.d_value % 64 == 0/*prefetch 2 32x32 blocks along headdim*/;
    // For latest archs, MLS can be applied for the common decode head dims.
    int arch_id = getArch();
    constexpr bool use_gfx938_mls =
        (Headdim == 128 and HeaddimV == 128) or
        (Headdim == 192 and HeaddimV == 128) or
        (Headdim == 256 and HeaddimV == 256);
    if constexpr (use_gfx938_mls) {
        const bool is_local = params.window_size_left > 0 && params.window_size_right >= 0;
        const bool use_mls_mask = params.is_e4m3 ? true : params.is_causal;
        if ((arch_id >= 938) and std::getenv("PA_NO_MLS") == nullptr and is_commonly_used and use_mls_mask) {
            if (params.is_e4m3) {
                // Decide whether compile all page block sizes
                #ifdef PA_PAGE_BLOCK_SIZE
                if (params.page_block_size % 32 != 0) { printf("\x1b[31mPage block size %d is not supported yet!\033[0m\n", params.page_block_size); return; }
                PA_PAGEBLOCKSIZE_SWITCH(params.page_block_size, [&]{
                    run_fp8_flash_splitkv_fwd_gfx938<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 64, 32, 32, 2/*STAGES*/, false, false, T, T> >(params, stream);
                });
                #else
                if (params.page_block_size == 64) {
                    constexpr int kBlockN = 64;
                    run_fp8_flash_splitkv_fwd_gfx938<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 64, 32, 32, 2/*STAGES*/, false, false, T, T> >(params, stream);
                } else {
                    constexpr int kBlockN = 128;
                    if (params.page_block_size % kBlockN != 0) { printf("\x1b[31mPage block size %d is not supported yet!\033[0m\n", params.page_block_size); return; }
                    run_fp8_flash_splitkv_fwd_gfx938<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 64, 32, 32, 2/*STAGES*/, false, false, T, T> >(params, stream);
                }
                #endif
            } else {
                // Decide whether compile all page block sizes
                #ifdef PA_PAGE_BLOCK_SIZE
                if (params.page_block_size % 32 != 0) { printf("\x1b[31mPage block size %d is not supported yet!\033[0m\n", params.page_block_size); return; }
                PA_PAGEBLOCKSIZE_SWITCH(params.page_block_size, [&]{
                    run_flash_splitkv_fwd_gfx938<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 32, 32, 32, 2/*STAGES*/, false, false, T, T> >(params, stream);
                });
                #else
                if (params.page_block_size == 64) {
                    constexpr int kBlockN = 64;
                    run_flash_splitkv_fwd_gfx938<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 32, 32, 32, 2/*STAGES*/, false, false, T, T> >(params, stream);
                } else {
                    constexpr int kBlockN = 128;
                    if (params.page_block_size % kBlockN != 0) { printf("\x1b[31mPage block size %d is not supported yet!\033[0m\n", params.page_block_size); return; }
                    run_flash_splitkv_fwd_gfx938<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 32, 32, 32, 2/*STAGES*/, false, false, T, T> >(params, stream);
                }
                #endif
            }
            return;
        } else if (arch_id == 930 and std::getenv("PA_NO_MLS") == nullptr and is_commonly_used) {
            constexpr int kBlockN = 128;
            if (params.page_block_size % kBlockN != 0) { printf("\x1b[31mPage block size %d is not supported yet!\033[0m\n", params.page_block_size); return; }
            run_flash_splitkv_fwd_gfx92a<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 32, 32, 32, 2/*STAGES*/, false, false, T, T> >(params, stream);
            return;
        }
    }
    // For MHA-fma, headdim = 128
    if (params.seqlen_q == 1 and !params.seqlenq_ngroups_swapped and Headdim == 128 and HeaddimV == 128 and std::getenv("PA_USE_FMA") != nullptr) {
        constexpr int kBlockN = 128;
        run_flash_splitkv_fwd_mha<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32/*kBlockM*/, kBlockN, 32/*kBlockK*/, 32, 32, 2/*STAGES*/, false, false, T, float> >(params, stream);
    }
    else if (params.seqlen_q <= 32/*16x32 tile*/ and not params.splitkv_use_fp32_as_accum and params.alibi_slopes_ptr == nullptr and std::getenv("PA_USE_TILE32X32") == nullptr and is_commonly_used) {
        // Decide whether compile all page block sizes
        #ifdef PA_PAGE_BLOCK_SIZE
            if (params.page_block_size % 32 != 0) { printf("\x1b[31mPage block size %d is not supported yet!\033[0m\n", params.page_block_size); return; }
            PA_PAGEBLOCKSIZE_SWITCH(params.page_block_size, [&]{
                run_flash_splitkv_fwd_tile16x32<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 32, 32, 32, 2/*STAGES*/, false, false, T, T> >(params, stream);
            });
        #else
            if (params.page_block_size == 64) {
                constexpr int kBlockN = 64;
                run_flash_splitkv_fwd_tile16x32<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 32, 32, 32, 2/*STAGES*/, false, false, T, T> >(params, stream);
            } else {
                constexpr int kBlockN = 128;
                if (params.page_block_size % kBlockN != 0) { printf("\x1b[31mPage block size %d is not supported yet!\033[0m\n", params.page_block_size); return; }
                run_flash_splitkv_fwd_tile16x32<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 32, 32, 32, 2/*STAGES*/, false, false, T, T> >(params, stream);
            }
        #endif
    } else {
        // Decide whether compile all page block sizes
        #ifdef PA_PAGE_BLOCK_SIZE
            if (params.page_block_size % 64 != 0) { printf("\x1b[31mPage block size %d is not supported yet!\033[0m\n", params.page_block_size); return; }
            PA_PAGEBLOCKSIZE_SWITCH(params.page_block_size, [&]{
                constexpr int STAGES = (Headdim == 128) ? 3: (Headdim == 32 ? 1: 2);
                // regardless of params.splitkv_use_fp32_as_accum to reduce volume of FA whl
                run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 32, 32, 32, STAGES, false, false, T, T> >(params, stream);
            });
        #else
            if (params.page_block_size == 64) {
                constexpr int kBlockN = 64;
                constexpr int STAGES = (Headdim == 128) ? 3: (Headdim == 32 ? 1: 2);
                if (params.splitkv_use_fp32_as_accum) {
                    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 32, 32, 32, STAGES, false, false, T, float> >(params, stream);
                } else {
                    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 32, 32, 32, STAGES, false, false, T, T> >(params, stream);
                }
            } else {
                constexpr int kBlockN = 128;
                if (params.page_block_size % kBlockN != 0) { printf("\x1b[31mPage block size %d is not supported yet!\033[0m\n", params.page_block_size); return; }
                constexpr int STAGES = (Headdim == 128) ? 3: (Headdim == 32 ? 1: 2);
                if (params.splitkv_use_fp32_as_accum) {
                    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 32, 32, 32, STAGES, false, false, T, float> >(params, stream);
                } else {
                    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, HeaddimV, 32, kBlockN, 32, 32, 32, STAGES, false, false, T, T> >(params, stream);
                }
            }
        #endif
    }
}


template<typename Kernel_traits>
void run_int8_flash_splitkv_fwd(Flash_fwd_params &params, hipStream_t stream) {
    constexpr int WARP_NUM     = Kernel_traits::kBlockN / Kernel_traits::kWaveN;
    const size_t smem_for_max  = std::max(WARP_NUM * Kernel_traits::kWaveM * sizeof(float), size_t(1024));
    const size_t smem_misalign = (params.seqlen_q >= 16 or Kernel_traits::kHeadDimV == 512) ? Kernel_traits::kHeadDimV: (Kernel_traits::kHeadDimV + 4)/*<=15 can use misalign to reduce bank conflicts, but >16 may lead to lds>32KB, less waves per SIMD*/;
    const size_t smem_for_acc  = int((params.seqlen_q + 1) / 2) * 2 * WARP_NUM * smem_misalign * sizeof(float);
    const size_t smem_for_gemm = std::max(std::max(Kernel_traits::q_smem_size, Kernel_traits::k_smem_size * WARP_NUM), Kernel_traits::v_smem_size * WARP_NUM);
    #if defined(KVCACHE_USE_4STAGES_PINGPANG) // 4 倍 pingpang buffer 已经 32KB 了, 需要跟 max 共享
        const size_t required_smem_size = std::max(smem_for_max, std::max(smem_for_acc, smem_for_gemm));
    #else
        const size_t required_smem_size = std::max(smem_for_acc, smem_for_gemm + smem_for_max);
    #endif
    /*
        for gfx936,
            2 waves per SIMD is better than 1 waves per SIMD;
            3 waves per SIMD will bring performance degradation
        for gfx928,
            > 1 waves per SIMD will significantly increase the latency of buffer-load, blocked at TA
    */

    hipDeviceProp_t props;
    auto hip_result = hipGetDeviceProperties(&props, 0);
    #ifdef ROCM_5_7
        int gcn_arch = props.gcnArch;
    #else
        std::string gcn_arch_name(props.gcnArchName);
        int gcn_arch = hg_pa_runtime_gfx_arch_id(gcn_arch_name);
    #endif
    const size_t smem_size     = gcn_arch > 928 ? required_smem_size: size_t(48 * 1024);
    // printf("smem_for_max: %ld | smem_for_acc: %ld | smem_for_gemm: %ld | needed smem_size: %ld | smem_size: %ld\n", smem_for_max, smem_for_acc, smem_for_gemm, required_smem_size, smem_size);

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    const int nthread = Kernel_traits::kBlockN / Kernel_traits::kWaveN * 64;

    // dim3 grid(num_m_block, params.h, params.b);
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.h, params.num_splits > 1 ? params.b * params.h : params.b);

    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    // const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    const bool has_alibi = (params.alibi_slopes_ptr not_eq nullptr);

    // judge if mtp > 1, attention the causal mask ?
    bool use_mtp = bool(params.seqlen_q > 1 and !params.seqlenq_ngroups_swapped);

#ifdef BUILD_ASM
    // select most likely used kernels to analyze instruction flow
    M_MMAC_COUNT_SWITCH(params.seqlen_q > 1, M_MMAC_COUNT, [&] {
        BOOL_SWITCH(params.num_splits > 1, Split, [&] {
            void (*kernel)(Flash_fwd_params);
            if (use_mtp) { // MTP > 1
                kernel = &flash_fwd_splitkv_int8_kernel<Kernel_traits, false, true/*Is_causal*/, false, false, true, false, false, false/*is_gqa*/, false/*Is_softcap*/, Split/*Split*/, M_MMAC_COUNT, 0, false/*Append_KV*/>;
            } else {
                REUSEKV_SWITCH(params.seqlen_q, [&] { // MTP = 1, can reuse
                    kernel = &flash_fwd_splitkv_int8_kernel<Kernel_traits, false, false/*Is_causal*/, false, false, true, false, false, false/*is_gqa*/, false/*Is_softcap*/, Split/*Split*/, M_MMAC_COUNT, REUSE_KV_TIMES, false/*Append_KV*/>;
                });
            }
            kernel<<<grid, nthread, smem_size, stream>>>(params);
        });
    });
#else
    bool is_local_mask = bool((params.window_size_left >= 0 || params.window_size_right >= 0) && !(use_mtp));
    if (is_local_mask) {printf("\x1b[31mSliding window attention for Paged-Atention is not supported yet!\033[0m\n");}
    BOOL_SWITCH(has_alibi, Has_Alibi, [&] {
        M_MMAC_COUNT_SWITCH(params.seqlen_q > 1, M_MMAC_COUNT, [&] {
            BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                constexpr bool IsEvenMNConst = false;
                constexpr bool Is_local = false;
                void (*kernel)(Flash_fwd_params);
                if (use_mtp) {
                    kernel = &flash_fwd_splitkv_int8_kernel<Kernel_traits, false, true/*Is_causal*/, Is_local, IsEvenMNConst && !Is_local && true && Kernel_traits::kHeadDim <= 256, true, false, Has_Alibi, false/*is_gqa*/, false/*Is_softcap*/, Split, M_MMAC_COUNT, 0, false/*Append_KV*/>;
                } else {
                    REUSEKV_SWITCH(params.seqlen_q, [&] {
                        kernel = &flash_fwd_splitkv_int8_kernel<Kernel_traits, false, false/*Is_causal*/, Is_local, IsEvenMNConst && !Is_local && true && Kernel_traits::kHeadDim <= 256, true, false, Has_Alibi, false/*is_gqa*/, false/*Is_softcap*/, Split, M_MMAC_COUNT, REUSE_KV_TIMES, false/*Append_KV*/>;
                    });
                }
                #if defined(FA_KERNEL_TIMER)
                    hipEvent_t   start, stop;
                    HIP_CHECK(hipEventCreate(&start));
                    HIP_CHECK(hipEventCreate(&stop));
                    HIP_CHECK(hipEventRecord(start, 0));
                #endif
                    kernel<<<grid, nthread, smem_size, stream>>>(params);
                #if defined(FA_KERNEL_TIMER)
                    HIP_KERNEL_LAUNCH_CHECK();
                    HIP_CHECK(hipDeviceSynchronize());
                    HIP_CHECK(hipEventRecord(stop, 0)) ;
                    HIP_CHECK(hipEventSynchronize(stop));
                    float   ave_time;
                    HIP_CHECK(hipEventElapsedTime(&ave_time,start, stop));
                    printf("run_flash_splitkv_fwd: %f\n", ave_time * 1000);
                #endif
            });
        });
    });
#endif
    run_splitkv_reduce<Kernel_traits, true/*Tail*/>(params, stream);
}


template<typename T, int Headdim, int HeaddimV>
void run_int8_fwd_splitkv_dispatch(Flash_fwd_params &params, hipStream_t stream) {
    // Decide whether compile all page block sizes
    #ifdef PA_PAGE_BLOCK_SIZE
        if (params.page_block_size % 64 != 0) { printf("\x1b[31mPage block size %d is not supported yet!\033[0m\n", params.page_block_size); return; }
        PA_PAGEBLOCKSIZE_SWITCH(params.page_block_size, [&]{
            constexpr int STAGES = (Headdim == 128) ? 3: (Headdim == 32 ? 1: 2);
            // regardless of params.splitkv_use_fp32_as_accum to reduce volume of FA whl
            run_int8_flash_splitkv_fwd<Flash_fwd_kernel_traits<128, 128, 32, kBlockN, 32, 32, 32, STAGES, false, false, T, T, int8_t> >(params, stream);
        });
    #else
        constexpr int kBlockN = 128;
        if (params.page_block_size % kBlockN != 0) { printf("\x1b[31mPage block size %d is not supported yet!\033[0m\n", params.page_block_size); return; }
        constexpr int STAGES = (Headdim == 128) ? 3: (Headdim == 32 ? 1: 2);
        if (params.splitkv_use_fp32_as_accum) {
            run_int8_flash_splitkv_fwd<Flash_fwd_kernel_traits<128, 128, 32, kBlockN, 32, 32, 32, STAGES, false, false, T, float, int8_t> >(params, stream);
        } else {
            run_int8_flash_splitkv_fwd<Flash_fwd_kernel_traits<128, 128, 32, kBlockN, 32, 32, 32, STAGES, false, false, T, T, int8_t> >(params, stream);
        }
    #endif
}
