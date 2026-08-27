/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

// #include "cute/algorithm/copy.hpp"

// #include "cutlass/cutlass.h"
// #include "cutlass/layout/layout.h"
#include "numeric_types.h"

// using namespace cute;

template<int kHeadDim_, int kBlockM_, int kBlockN_,  int kBlockK_, int kWaveM_, int kWaveN_, typename elem_type=Float16>
struct Flash_kernel_traits {
    using Element = elem_type;
    using ElementAccum = float;
    using index_t = uint32_t;
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template<int kHeadDim_, int kHeadDimV_, int kBlockM_, int kBlockN_, int kBlockK_, int kWaveM_, int kWaveN_, int STAGES_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=Float16, typename splitkv_accum_dtype=Float16, typename elem_type_k=Float16, int kBlockK_int8_=64,
         int kHeadDimQKCompute_=kHeadDim_, int kHeadDimPVCompute_=kHeadDimV_, int TailTile16_=2,
         typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kBlockK_, kWaveM_, kWaveN_, elem_type> >
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using Element_k = elem_type_k;
    using index_t = typename Base::index_t;
    using SplitkvAccumType = splitkv_accum_dtype;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kBlockK = kBlockK_;
    static constexpr int kBlockK_int8 = kBlockK_int8_;

    static constexpr int kWaveM = kWaveM_;
    static constexpr int kWaveN = kWaveN_;
    static constexpr int STAGES = STAGES_;
    // The number of threads.
    static constexpr int kNWarps = kBlockM_ / kWaveM_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static constexpr int kHeadDimQKCompute = kHeadDimQKCompute_;
    static constexpr int kHeadDimPVCompute = kHeadDimPVCompute_;
    static constexpr int TailTile16 = TailTile16_;
    static constexpr int SplitD = (kHeadDimV <= 512) ? 1: kHeadDimV / 128;
    static constexpr int kHeadDimVSplit = kHeadDimV / SplitD;
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);

    static constexpr int kSmemQCount = 1; 
    static constexpr int kSmemKVCount = 2; 
    static constexpr int kSmemQSize = kSmemQCount * sizeof(Element);
    static constexpr int kSmemKVSize = kSmemKVCount * sizeof(Element);
    static constexpr size_t q_smem_size = (STAGES * (kBlockM / 32) * (kBlockK / 32) * (32 * 34)) * sizeof(Element);
    static constexpr size_t k_smem_size = (STAGES * (kWaveN / 32) * (kBlockK / 32) * (32 * 34)) * sizeof(Element);
    static constexpr size_t v_smem_size = (STAGES * kBlockK * 32/*WARP_K*/) * sizeof(Element);

};

// Is_V_in_regs is an option to reduce smem usage, but will increase register pressue.
// No_double_buffer is another option to reduce smem usage, but will slow things down.
template<int kHeadDim_, int kHeadDimV_, int kBlockM_, int kBlockN_,  int kBlockK_, int kWaveM_, int kWaveN_,
         int STAGES_, bool Is_V_in_regs_=false, typename elem_type=Float16,
         typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kBlockK_, kWaveM_, kWaveN_, elem_type> >
struct Flash_bwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    // static constexpr bool Has_cp_async = Base::Has_cp_async;
    // using SmemCopyAtom = typename Base::SmemCopyAtom;
    // using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Is_V_in_regs = Is_V_in_regs_;

    // The number of threads.
    static constexpr int kWaveM = kWaveM_;
    static constexpr int kWaveN = kWaveN_;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kBlockK = kBlockK_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;

    static constexpr int STAGES = STAGES_;

    static constexpr int q_smem_size = (STAGES*(kBlockM/32) * (kBlockK/32)*(32*34)) * sizeof(elem_type);
    static constexpr int k_smem_size = (STAGES*(kBlockN/32) * (kBlockK/32)*(32*34)) * sizeof(elem_type);
    static constexpr int v_smem_size = (STAGES*kBlockK * kBlockN) * sizeof(elem_type);

    static constexpr int kSmemSize1colblock = max((q_smem_size + k_smem_size), v_smem_size);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
