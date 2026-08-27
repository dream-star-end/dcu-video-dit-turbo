/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>
#include "cutlass/float8.h"

using namespace cute;

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t>
struct Flash_kernel_traits {

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using Element = elem_type;
    static constexpr bool Has_cp_async = true;
#elif defined(DCU_ASM)
    using Element = elem_type;
    static constexpr bool Has_cp_async = false;
#else
    using Element = cutlass::half_t;
    static constexpr bool Has_cp_async = false;
#endif

    using ElementAccum = float;
    using index_t = int64_t;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;
#else
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x16x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x16x16_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_FOR_GEMM1 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x16x16_F32F16F16F32_NT_FOR_GEMM1>,
        MMA_Atom<GFX928_16x16x16_F32BF16BF16F32_NT_FOR_GEMM1>
    >;
    using ValLayoutMNK = Layout<Shape<_1, _1, _1>>;
#endif

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
#else
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t,
        typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
#ifndef GEMM1_AMATRIX_WITH_SMEM
    using TiledMma_FOR_GEMM1 = TiledMMA<
        typename Base::MMA_Atom_Arch_FOR_GEMM1,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
#endif

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    // using SmemLayoutAtomQ = decltype(
    //     composition(Swizzle<3, 2, 4>{},
    //                 // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
    //                 Layout<Shape<_8, Int<32>>,
    //                        Stride<Int<32>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

#ifdef GEMM1_AMATRIX_WITH_SMEM
    using SmemLayoutAtomAccs = decltype(
        composition(Swizzle<3, 2, 4>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<kBlockN % 64 == 0 ? 64 : 32>>,
                        Stride<Int<kBlockN % 64 == 0 ? 64 : 32>, _1>>{}));
    
    using SmemLayoutAccs = decltype(tile_to_shape(
        SmemLayoutAtomAccs{},
        Shape<Int<kBlockM>, Int<kBlockN>>{}));
#endif

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

#ifdef GEMM1_AMATRIX_WITH_SMEM
    // https://github.com/ColfaxResearch/cutlass-kernels/blob/a222587e6d59b93ba704853d3946fb686d8b8892/src/fmha/fmha_forward.cu#L434
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));
#else
    // headdim为128时，2、4、2也能有效减少bank冲突，但是测试性能没有提升，因此采用了两套参数
    // using SmemLayoutAtomV = decltype(
    //     composition(Swizzle<kHeadDim == 64 ? 2 : 3, kHeadDim==64 ? 4 : 3, kHeadDim == 64 ? 2 : 3>{},
    //         // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
    //         Layout<Shape<_8, Int<kHeadDim == 128 ? kHeadDim : kBlockKSmem>>,
    //                 Stride<Int<kHeadDim == 128 ? kHeadDim : kBlockKSmem>, _1>>{}));
    using SmemLayoutAtomV = decltype(
        composition(Swizzle<kHeadDim == 64 ? 2 : 3, kHeadDim==64 ? 4 : 3, kHeadDim == 64 ? 2 : 3>{},
            Layout<Shape<_8, Int<kBlockKSmem>>,
                    Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));
#endif

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    #ifdef GEMM1_AMATRIX_WITH_SMEM
    static constexpr int KSmemAccsSize = size(SmemLayoutAccs{}) * sizeof(Element);
    #endif
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    #ifdef GEMM1_AMATRIX_WITH_SMEM
    // 例如headdim为32的情况下，kSmemQSize是大于KSmemAccsSize的
    static constexpr int kSmemSize = Share_Q_K_smem ? (kSmemQSize > KSmemAccsSize ? std::max(kSmemQSize, kSmemKVSize) : KSmemAccsSize + kSmemKSize)
        : kSmemQSize + kSmemKVSize;
    #else
    static constexpr int kSmemSize = Share_Q_K_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;
    #endif
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    
    // from how many rows does each thread have to fetch
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    // Here we assign a contiguous tile to each thread, rather than a 1x8 row every 
    // (kNThreads / kGmemThreadsPerRow) rows, ensuring that the elements assigned to each thread
    // do not cross a page boundary. This way, each thread need only fetch 1 page index per
    // mainloop iteration. R>udimentary testing shows no slowdown.
    using GmemTiledCopyQKVPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));

#if 0
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
#else

    using GmemLayoutAtomO = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomO{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
#endif
#if 0
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
#else
    using GmemLayoutAtomOaccum = Layout<Shape <_64, _4>,  // Thread layout, 8 threads per row
            Stride< _4, _1>>;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
#endif
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
    
    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel traits for attnmask (uses 16x64x32 MMA for Q*K and specialized SmemCopyAtomV).
////////////////////////////////////////////////////////////////////////////////////////////////////
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t,
        typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_traits_attnmask : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = Copy_Atom<UniversalCopy<uint128_t>, Element>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomV = Copy_Atom<GFX928_DS_READ_DS_M32x16_B16_RAW, Element>;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);

    static constexpr int kBlockKSmem = 32;
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    // Q*K GEMM: use 16x64x32 MMA (same as predmaskbeta bias_hdim32)
    using MMA_Atom_Arch_16x64x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;

    using TiledMma = TiledMMA<
        MMA_Atom_Arch_16x64x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;

    // P*V GEMM: use standard 16x16x16 MMA with ValLayout <1,2,1>
    using TiledMma_FOR_GEMM1 = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        Layout<Shape<_1, _2, _1>>>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutAtomV = Layout<Shape<_16, Int<kBlockKSmem>>,
                    Stride<Int<kBlockKSmem>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemSize = Share_Q_K_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;

    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));

    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);

    using GmemLayoutAtomO = Layout<Shape <_64, _4>, Stride< _4, _1>>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomO{},
                        Layout<Shape<_1, _8>>{}));

    using GmemLayoutAtomOaccum = Layout<Shape <_64, _4>, Stride< _4, _1>>;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_use_smem_=false, bool Share_K_V_smem_=false, typename elem_type=cutlass::half_t, int kHeadDimV_=kHeadDim_,
        typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_K_V_smem = Share_K_V_smem_;
    static constexpr bool Is_Q_use_smem = Is_Q_use_smem_;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    static constexpr bool MMA_Atom_Use_K16 = Is_Q_use_smem;
    static constexpr bool MMA_Atom_Use_K32 = !MMA_Atom_Use_K16;

    
    using SmemCopyAtom16x64x16 = typename Base::SmemCopyAtom;
    // using SmemCopyAtom16x64x32 = Copy_Atom<GFX928_DS_READ_B128, Element>;
    using SmemCopyAtom16x64x32 = typename Base::SmemCopyAtom;
    using MMA_Atom_Arch_16x64x16 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x16_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using SmemCopyAtom = std::conditional_t<
        MMA_Atom_Use_K16,
        SmemCopyAtom16x64x16,
        SmemCopyAtom16x64x32
    >;
    using MMA_Atom_Arch_16x64 = std::conditional_t<
        MMA_Atom_Use_K16,
        MMA_Atom_Arch_16x64x16,
        MMA_Atom_Arch_16x64x32
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutAtomK = decltype(
        composition(Swizzle<3, 3, 4>{},
                    Layout<Shape<_8, Int<32>>,
                        Stride<Int<32>, _1>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    // using SmemLayoutAtomV = Layout<Shape<Int<kBlockN>, Int<kHeadDim>>,
    //                        Stride<Int<kHeadDim>, _1>>;
    using SmemLayoutAtomV = decltype(
        composition(Swizzle<1, 3, 3>{},
                Layout<Shape<Int<8>, Int<32>>,
                        Stride<Int<32>, _1>>{}));
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    static constexpr int kSmemKVSize = Share_K_V_smem ? kSmemKSize : 2 * kSmemKSize;
    static constexpr int kSmemSize = kHeadDim == 64 ? 32 * 1024 : Is_Q_use_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemKVSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    
    // from how many rows does each thread have to fetch
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    // Here we assign a contiguous tile to each thread, rather than a 1x8 row every 
    // (kNThreads / kGmemThreadsPerRow) rows, ensuring that the elements assigned to each thread
    // do not cross a page boundary. This way, each thread need only fetch 1 page index per
    // mainloop iteration. R>udimentary testing shows no slowdown.
    using GmemTiledCopyQKVPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
    
    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_use_smem_=false, bool Share_K_V_smem_=false, typename elem_type=cutlass::half_t, int kHeadDimV_=kHeadDim_, 
        typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_traits_MLA : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_K_V_smem = Share_K_V_smem_;
    static constexpr bool Is_Q_use_smem = Is_Q_use_smem_;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    static constexpr bool MMA_Atom_Use_K16 = Is_Q_use_smem;
    static constexpr bool MMA_Atom_Use_K32 = !MMA_Atom_Use_K16;

    
    using SmemCopyAtom16x64x16 = typename Base::SmemCopyAtom;
    // using SmemCopyAtom16x64x32 = Copy_Atom<GFX928_DS_READ_B128, Element>;
    using SmemCopyAtom16x64x32 = typename Base::SmemCopyAtom;
    using MMA_Atom_Arch_16x64x16 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x16_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using SmemCopyAtom = std::conditional_t<
        MMA_Atom_Use_K16,
        SmemCopyAtom16x64x16,
        SmemCopyAtom16x64x32
    >;
    using MMA_Atom_Arch_16x64 = std::conditional_t<
        MMA_Atom_Use_K16,
        MMA_Atom_Arch_16x64x16,
        MMA_Atom_Arch_16x64x32
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutAtomK = decltype(
        composition(Swizzle<3, 3, 5>{},
                    Layout<Shape<_32, Int<64>>,
                        Stride<Int<64>, _1>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutAtomV = Layout<Shape<Int<16>, Int<32>>,
                           Stride<Int<32>, _1>>;
    // using SmemLayoutAtomV = decltype(
    //     composition(Swizzle<1, 3, 3>{},
    //             Layout<Shape<Int<8>, Int<32>>,
    //                     Stride<Int<32>, _1>>{}));
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    static constexpr int kSmemKVSize = Share_K_V_smem ? kSmemKSize : 2 * kSmemKSize;
    // 写出过lds有性能提升，128*128*2 = 32768
    static constexpr int kSmemSize = 32768;
    // static constexpr int kSmemSize = Is_Q_use_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemKVSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    
    // from how many rows does each thread have to fetch
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    // Here we assign a contiguous tile to each thread, rather than a 1x8 row every 
    // (kNThreads / kGmemThreadsPerRow) rows, ensuring that the elements assigned to each thread
    // do not cross a page boundary. This way, each thread need only fetch 1 page index per
    // mainloop iteration. R>udimentary testing shows no slowdown.
    using GmemTiledCopyQKVPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
    
    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_use_smem_=false, bool Share_K_V_smem_=false, typename elem_type=cutlass::half_t, int kHeadDimV_=kHeadDim_, 
        typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_traits_splitkv : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_K_V_smem = Share_K_V_smem_;
    static constexpr bool Is_Q_use_smem = Is_Q_use_smem_;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    static constexpr bool MMA_Atom_Use_K16 = Is_Q_use_smem;
    static constexpr bool MMA_Atom_Use_K32 = !MMA_Atom_Use_K16;

    
    using SmemCopyAtom16x64x16 = typename Base::SmemCopyAtom;
    // using SmemCopyAtom16x64x32 = Copy_Atom<GFX928_DS_READ_B128, Element>;
    using SmemCopyAtom16x64x32 = typename Base::SmemCopyAtom;
    using MMA_Atom_Arch_16x64x16 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x16_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using SmemCopyAtom = std::conditional_t<
        MMA_Atom_Use_K16,
        SmemCopyAtom16x64x16,
        SmemCopyAtom16x64x32
    >;
    using MMA_Atom_Arch_16x64 = std::conditional_t<
        MMA_Atom_Use_K16,
        MMA_Atom_Arch_16x64x16,
        MMA_Atom_Arch_16x64x32
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutAtomK = decltype(
        composition(Swizzle<3, 3, 5>{},
                    Layout<Shape<_32, Int<64>>,
                        Stride<Int<64>, _1>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutAtomV = Layout<Shape<Int<16>, Int<32>>,
                           Stride<Int<32>, _1>>;
    // using SmemLayoutAtomV = decltype(
    //     composition(Swizzle<1, 3, 3>{},
    //             Layout<Shape<Int<8>, Int<32>>,
    //                     Stride<Int<32>, _1>>{}));
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    static constexpr int kSmemKVSize = Share_K_V_smem ? kSmemKSize : 2 * kSmemKSize;
    // 写出过lds有性能提升，128*128*2 = 32768
    static constexpr int kSmemSize = kHeadDim == kHeadDimV ? 32768 : 32768;
    // static constexpr int kSmemSize = Is_Q_use_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemKVSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    
    // from how many rows does each thread have to fetch
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    // Here we assign a contiguous tile to each thread, rather than a 1x8 row every 
    // (kNThreads / kGmemThreadsPerRow) rows, ensuring that the elements assigned to each thread
    // do not cross a page boundary. This way, each thread need only fetch 1 page index per
    // mainloop iteration. R>udimentary testing shows no slowdown.
    using GmemTiledCopyQKVPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
    
    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load
};

#if 0
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_prefetch_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockNSmem = (kStages + 1) * 16;
    static constexpr int kBlockKSmem = (kStages + 1) * 32;

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomK = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutKsplit = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<kBlockN>, Int<kBlockKSmem>>{}));
    
    using SmemLayoutAtomV = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<kBlockNSmem>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockNSmem>>{}, GenRowMajor{})));
    using SmemLayoutVsplit = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<16>, Int<(kStages + 1)*kHeadDimV>>{}));
    using SmemLayoutVtransSplit = decltype(composition(SmemLayoutVsplit{}, make_layout(Shape<Int<(kStages + 1)*kHeadDimV>, Int<16>>{}, GenRowMajor{})));

    static constexpr int kSmemKSize = size(SmemLayoutKsplit{}) * sizeof(Element);
    static constexpr int kSmemVSize = size(SmemLayoutVsplit{}) * sizeof(Element);
    static constexpr int kSmemSize = std::max(kSmemKSize, kSmemVSize);
    
    using GmemLayoutAtom = Layout<Shape <_32, _8>, Stride< _8, _1>>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
};
#else
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, bool ENABLE_SKIP_SOFTMAX_ = false, int kHeadDimV_ = kHeadDim_, 
        typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_prefetch_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int ENABLE_SKIP_SOFTMAX = ENABLE_SKIP_SOFTMAX_;

    // struct __align__(128) Shared {
    //     // 4 warps in a warpgroup vote to an atomic variable in shared memory
    //     uint32_t skip_softmax_votes;
    // };

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    static constexpr uint32_t LayoutBlock = 64;
    static constexpr uint32_t LayoutDim = 128;
    using SmemLayoutAtomK = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    using SmemLayoutKsplit = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    
    using SmemLayoutAtomV = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<LayoutDim>, Int<LayoutBlock>>{}, GenRowMajor{})));
    using SmemLayoutVsplit = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<16>, Int<4*LayoutDim>>{}));
    using SmemLayoutVtransSplit = decltype(composition(SmemLayoutVsplit{}, make_layout(Shape<Int<4*LayoutDim>, Int<16>>{}, GenRowMajor{})));

    static constexpr int kSmemKSize = size(SmemLayoutKsplit{}) * sizeof(Element);
    static constexpr int kSmemVSize = size(SmemLayoutVsplit{}) * sizeof(Element);
    static constexpr int kSmemSize = std::max(kSmemKSize, kSmemVSize);
    // static constexpr int kSmemSize = ENABLE_SKIP_SOFTMAX ? std::max(kSmemKSize, kSmemVSize) + sizeof(Shared) : std::max(kSmemKSize, kSmemVSize);
    
    using GmemLayoutAtom = Layout<Shape <_32, _8>, Stride< _8, _1>>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
};
#endif

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,typename elem_type_o=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_prefetch_traits_fp8 : public Base {
    using Element = typename Base::Element;
    using ElementO = elem_type_o;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    // static constexpr int kBlockNSmem = (kStages + 1) * 32;
    // static constexpr int kBlockKSmem = (kStages + 1) * 32;


    using MMA_Atom_Arch_16x64 = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT>;//qk
    using MMA_Atom_Arch_16x64_BLayout =  MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT_BLayout>;
    using MMA_Atom_Arch_16x32 = MMA_Atom<GFX938_16x32x32_F32F8F8F32E4M3E4M3_NT_LIT>;//pv
    
    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;

    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;

    using SmemLayoutK = Layout<Shape<Int<kBlockN*(kHeadDim/64)>, Int<64>>, Stride<Int<64>, _1>>;

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutAtomV = Layout<Shape<Int<32>, Int<32>>, Stride<Int<32>, _1>>;

    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));

    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemVSize = size(SmemLayoutV{}) * sizeof(Element);
    static constexpr int kSmemSize = 16384;
    //static constexpr int kSmemSize = std::max(kSmemKSize, kSmemVSize);
    
    using GmemLayoutAtom = Layout<Shape <_32, _8>, Stride< _8, _1>>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementO>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
};





template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_prefetch_traits_dim96 : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomK = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutKsplit = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    
    using SmemLayoutAtomV = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVsplit = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<16>, Int<4*kHeadDimV>>{}));
    using SmemLayoutVtransSplit = decltype(composition(SmemLayoutVsplit{}, make_layout(Shape<Int<4*kHeadDimV>, Int<16>>{}, GenRowMajor{})));

    static constexpr int kSmemKSize = size(SmemLayoutKsplit{}) * sizeof(Element);
    static constexpr int kSmemVSize = size(SmemLayoutVsplit{}) * sizeof(Element);
    static constexpr int kSmemSize = std::max(kSmemKSize, kSmemVSize);
    
    using GmemLayoutAtom = Layout<Shape <_32, _8>, Stride< _8, _1>>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_prefetch_traits_dim64 : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages_GEMM0 = kHeadDim / 32;
    static constexpr int kStages_GEMM1 = kBlockN / 16;
    static constexpr int kBlockNSmem = (kStages_GEMM1) * 16;
    static constexpr int kBlockKSmem = (kStages_GEMM0) * 32;

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomK = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutKsplit = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<kBlockN>, Int<kBlockKSmem>>{}));
    
    using SmemLayoutAtomV = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<kBlockNSmem>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockNSmem>>{}, GenRowMajor{})));
    using SmemLayoutVsplit = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<16>, Int<(kStages_GEMM1)*kHeadDimV>>{}));
    using SmemLayoutVtransSplit = decltype(composition(SmemLayoutVsplit{}, make_layout(Shape<Int<(kStages_GEMM1)*kHeadDimV>, Int<16>>{}, GenRowMajor{})));

    static constexpr int kSmemKSize = size(SmemLayoutKsplit{}) * sizeof(Element);
    static constexpr int kSmemVSize = size(SmemLayoutVsplit{}) * sizeof(Element);
    static constexpr int kSmemSize = kSmemKSize + kSmemVSize;
    
    using GmemLayoutAtom = Layout<Shape <_32, _8>, Stride< _8, _1>>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
};


template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_prefetch_mla_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<128>>, Stride<Int<128>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<128>>{}));
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(128/64)>, Int<64>>, Stride<Int<64>, _1>>;

    
    using SmemLayoutAtomV = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 0
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,typename elem_type_o=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_prefetch_mla_traits_fp8 : public Base {
    using Element = typename Base::Element;
    using ElementO = elem_type_o;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;//256

    static constexpr int kBlockM = kBlockM_;//128
    static constexpr int kBlockN = kBlockN_;//64
    static constexpr int kHeadDim = kHeadDim_;//192
    static constexpr int kHeadDimV = kHeadDimV_;//128
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;//3
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;//64
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);//128
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using MMA_Atom_Arch_16x64 = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT>;
    using MMA_Atom_Arch_16x64_BLayout =  MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT_BLayout>;
    //using MMA_Atom_Arch_16x32 = MMA_Atom<GFX938_16x32x32_F32F8F8F32E4M3E4M3_NT>;
    using MMA_Atom_Arch_16x32 = MMA_Atom<GFX938_16x32x32_F32F8F8F32E4M3E4M3_NT_LIT>;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<192>>, Stride<Int<192>, _1>>;//128,192


    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<192>>{}));//64,192
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(192/64)>, Int<64>>, Stride<Int<64>, _1>>;//192,64

    // using SmemLayoutAtomV = decltype(composition(
    //     Swizzle<kSwizzle, 4, 3>{},
    //     Layout<Shape<Int<8>, Int<64>>, Stride<Int<64>, _1>>{}));

    // using SmemLayoutV = decltype(tile_to_shape(
    //     SmemLayoutAtomV{},
    //     Shape<Int<kBlockN>, Int<kHeadDimV>>{}));//64,128

    using SmemLayoutAtomV = Layout<Shape<Int<32>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));

    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));
    
	static constexpr int kSmemKVSize = 16384;
    //static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    //static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);//128*128*1
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKVSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);//16
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;//64/16=4
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 0
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _16>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementO>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
};





template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_prefetch_mla_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<128>>, Stride<Int<128>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<128>>{}));
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(128/64)>, Int<64>>, Stride<Int<64>, _1>>;
    
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;
    
    using SmemLayoutAtomV = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    using GmemTiledCopyQKVPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));      
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
    
    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load 
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_prefetch_unified_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    static constexpr uint32_t LayoutBlock = 64;
    static constexpr uint32_t LayoutDim = 128;

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<128>>, Stride<Int<128>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtomK{},Shape<Int<kBlockN>, Int<128>>{}));
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(128/64)>, Int<64>>, Stride<Int<64>, _1>>;

    using SmemLayoutAtomO = decltype(composition(Swizzle<kSwizzle, 3, 3>{}, Layout<Shape<Int<8>, Int<kBlockKSmem>>,Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    using SmemLayoutAtomV = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<LayoutDim>, Int<LayoutBlock>>{}, GenRowMajor{})));
    using SmemLayoutVsplit = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<16>, Int<4*LayoutDim>>{}));
    using SmemLayoutVtransSplit = decltype(composition(SmemLayoutVsplit{}, make_layout(Shape<Int<4*LayoutDim>, Int<16>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store

    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    using GmemTiledCopyQKVPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load

    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_prefetch_unified_traits_dim256 : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    static constexpr uint32_t LayoutBlock = 64;
    static constexpr uint32_t LayoutDim = 128;

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<256>>, Stride<Int<256>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<kBlockN>, Int<256>>{}));
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(256/64)>, Int<64>>, Stride<Int<64>, _1>>;
    
    // using SmemLayoutAtomV = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    // using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    // using SmemLayoutVtransposed = decltype( composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));


    // using SmemLayoutAtomK = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    // using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    // using SmemLayoutKsplit = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    
    using SmemLayoutAtomV = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<LayoutDim>, Int<LayoutBlock>>{}, GenRowMajor{})));
    using SmemLayoutVsplit = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<16>, Int<4*LayoutDim>>{}));
    using SmemLayoutVtransSplit = decltype(composition(SmemLayoutVsplit{}, make_layout(Shape<Int<4*LayoutDim>, Int<16>>{}, GenRowMajor{})));
 
    using SmemLayoutAtomO = decltype(composition(Swizzle<kSwizzle, 3, 3>{}, Layout<Shape<Int<8>, Int<kBlockKSmem>>,Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    using GmemTiledCopyQKVPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));      
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
    
    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load 
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64x16 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x16_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x16_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x16 = TiledMMA<
        MMA_Atom_Arch_16x64x16,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<128>>, Stride<Int<128>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<128>>{}));
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(128/64)>, Int<64>>, Stride<Int<64>, _1>>;
    
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;
    
    using SmemLayoutAtomV = Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<64>, Int<64>>{}));
    // using SmemLayoutVtransposed = decltype(
    //     composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    using GmemTiledCopyQKVPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));      
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
    
    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load 
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::float_e4m3_t, typename elem_type_o=cutlass::bfloat16_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_fp8 : public Base {
    using Element = typename Base::Element;
    using ElementO = elem_type_o;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using MMA_Atom_Arch_16x64 = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT>;
    // using MMA_Atom_Arch_16x64x16 = MMA_Atom<GFX938_16x64x16_F32F8F8F32E4M3E4M3_NN>;
    using MMA_Atom_Arch_16x64x32_Blayout = MMA_Atom<GFX938_16x64x32_F32F8F8F32E4M3E4M3_NN_Blayout>;
    using MMA_Atom_Arch_16x64_BLayout = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_BLayout>;

    // using MMA_Atom_Arch_16x64_LIT = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT>;
    // using MMA_Atom_Arch_16x64_BLayout_LIT = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT_BLayout>;
    // using MMA_Atom_Arch_16x64x32_NN = MMA_Atom<GFX938_16x64x32_F32F8F8F32E4M3E4M3_NN>;


    using MMA_Atom_Arch_16x64_LIT = std::conditional_t<
        std::is_same_v<elem_type, cutlass::float_e4m3_t>,
        MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT>,
        MMA_Atom<GFX938_16x64x64_F32BF8BF8F32E5M2E5M2_NT_LIT>
    >;

    using MMA_Atom_Arch_16x64_BLayout_LIT = std::conditional_t<
        std::is_same_v<elem_type, cutlass::float_e4m3_t>,
        MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT_BLayout>,
        MMA_Atom<GFX938_16x64x64_F32BF8BF8F32E5M2E5M2_NT_LIT_BLayout>
    >;

    using MMA_Atom_Arch_16x64x32_NN = std::conditional_t<
        std::is_same_v<elem_type, cutlass::float_e4m3_t>,
        MMA_Atom<GFX938_16x64x32_F32F8F8F32E4M3E4M3_NN>,
        MMA_Atom<GFX938_16x64x32_F32BF8BF8F32E5M2E5M2_NN>
    >;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    // using TiledMma16x64x16 = TiledMMA<
    //     MMA_Atom_Arch_16x64x16,
    //     Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
    //     typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x32BLayout = TiledMMA<
        MMA_Atom_Arch_16x64x32_Blayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;

    using TiledMma16x64_LIT = TiledMMA<
        MMA_Atom_Arch_16x64_LIT,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64_Blayout_LIT = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout_LIT,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x32_NN = TiledMMA<
        MMA_Atom_Arch_16x64x32_NN,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<128>>, Stride<Int<128>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<128>>{}));
    // using SmemLayoutK = Layout<Shape<Int<kBlockN>, Int<128>>, Stride<Int<128>, _1>>;
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(128/64)>, Int<64>>, Stride<Int<64>, _1>>;
    
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, ElementO>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;
    
    using SmemLayoutAtomV = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<64>, Int<64>>{}));
    // using SmemLayoutVtransposed = decltype(
    //     composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(ElementO);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKSize + 2*size(SmemLayoutV{}) * sizeof(Element);
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(ElementO);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementO>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);

};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::float_e4m3_t, typename elem_type_o=cutlass::bfloat16_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_fp8_dim192 : public Base {
    using Element = typename Base::Element;
    using ElementO = elem_type_o;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using MMA_Atom_Arch_16x64 = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT>;
    // using MMA_Atom_Arch_16x64x16 = MMA_Atom<GFX938_16x64x16_F32F8F8F32E4M3E4M3_NN>;
    using MMA_Atom_Arch_16x64x32_Blayout = MMA_Atom<GFX938_16x64x32_F32F8F8F32E4M3E4M3_NN_Blayout>;
    using MMA_Atom_Arch_16x64_BLayout = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_BLayout>;

    // using MMA_Atom_Arch_16x64_LIT = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT>;
    // using MMA_Atom_Arch_16x64_BLayout_LIT = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT_BLayout>;
    // using MMA_Atom_Arch_16x64x32_NN = MMA_Atom<GFX938_16x64x32_F32F8F8F32E4M3E4M3_NN>;


    using MMA_Atom_Arch_16x64_LIT = std::conditional_t<
        std::is_same_v<elem_type, cutlass::float_e4m3_t>,
        MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT>,
        MMA_Atom<GFX938_16x64x64_F32BF8BF8F32E5M2E5M2_NT_LIT>
    >;

    using MMA_Atom_Arch_16x64_BLayout_LIT = std::conditional_t<
        std::is_same_v<elem_type, cutlass::float_e4m3_t>,
        MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT_BLayout>,
        MMA_Atom<GFX938_16x64x64_F32BF8BF8F32E5M2E5M2_NT_LIT_BLayout>
    >;

    using MMA_Atom_Arch_16x64x32_NN = std::conditional_t<
        std::is_same_v<elem_type, cutlass::float_e4m3_t>,
        MMA_Atom<GFX938_16x64x32_F32F8F8F32E4M3E4M3_NN>,
        MMA_Atom<GFX938_16x64x32_F32BF8BF8F32E5M2E5M2_NN>
    >;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    // using TiledMma16x64x16 = TiledMMA<
    //     MMA_Atom_Arch_16x64x16,
    //     Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
    //     typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x32BLayout = TiledMMA<
        MMA_Atom_Arch_16x64x32_Blayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;

    using TiledMma16x64_LIT = TiledMMA<
        MMA_Atom_Arch_16x64_LIT,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64_Blayout_LIT = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout_LIT,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x32_NN = TiledMMA<
        MMA_Atom_Arch_16x64x32_NN,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<192>>, Stride<Int<192>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<192>>{}));
    // using SmemLayoutK = Layout<Shape<Int<kBlockN>, Int<128>>, Stride<Int<128>, _1>>;
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(192/64)>, Int<64>>, Stride<Int<64>, _1>>;
    
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, ElementO>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;
    
    using SmemLayoutAtomV = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<64>, Int<64>>{}));
    // using SmemLayoutVtransposed = decltype(
    //     composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(ElementO);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(ElementO);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementO>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);

};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::float_e4m3_t, typename elem_type_o=cutlass::bfloat16_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_fp8_dim256 : public Base {
    using Element = typename Base::Element;
    using ElementO = elem_type_o;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using MMA_Atom_Arch_16x64 = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT>;
    // using MMA_Atom_Arch_16x64x16 = MMA_Atom<GFX938_16x64x16_F32F8F8F32E4M3E4M3_NN>;
    using MMA_Atom_Arch_16x64x32_Blayout = MMA_Atom<GFX938_16x64x32_F32F8F8F32E4M3E4M3_NN_Blayout>;
    using MMA_Atom_Arch_16x64_BLayout = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_BLayout>;

    // using MMA_Atom_Arch_16x64_LIT = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT>;
    // using MMA_Atom_Arch_16x64_BLayout_LIT = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT_BLayout>;
    // using MMA_Atom_Arch_16x64x32_NN = MMA_Atom<GFX938_16x64x32_F32F8F8F32E4M3E4M3_NN>;


    using MMA_Atom_Arch_16x64_LIT = std::conditional_t<
        std::is_same_v<elem_type, cutlass::float_e4m3_t>,
        MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT>,
        MMA_Atom<GFX938_16x64x64_F32BF8BF8F32E5M2E5M2_NT_LIT>
    >;

    using MMA_Atom_Arch_16x64_BLayout_LIT = std::conditional_t<
        std::is_same_v<elem_type, cutlass::float_e4m3_t>,
        MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT_BLayout>,
        MMA_Atom<GFX938_16x64x64_F32BF8BF8F32E5M2E5M2_NT_LIT_BLayout>
    >;

    using MMA_Atom_Arch_16x64x32_NN = std::conditional_t<
        std::is_same_v<elem_type, cutlass::float_e4m3_t>,
        MMA_Atom<GFX938_16x64x32_F32F8F8F32E4M3E4M3_NN>,
        MMA_Atom<GFX938_16x64x32_F32BF8BF8F32E5M2E5M2_NN>
    >;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    // using TiledMma16x64x16 = TiledMMA<
    //     MMA_Atom_Arch_16x64x16,
    //     Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
    //     typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x32BLayout = TiledMMA<
        MMA_Atom_Arch_16x64x32_Blayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;

    using TiledMma16x64_LIT = TiledMMA<
        MMA_Atom_Arch_16x64_LIT,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64_Blayout_LIT = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout_LIT,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x32_NN = TiledMMA<
        MMA_Atom_Arch_16x64x32_NN,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<256>>, Stride<Int<256>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<256>>{}));
    // using SmemLayoutK = Layout<Shape<Int<kBlockN>, Int<128>>, Stride<Int<128>, _1>>;
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(256/64)>, Int<64>>, Stride<Int<64>, _1>>;
    
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, ElementO>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;
    
    using SmemLayoutAtomV = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<64>, Int<64>>{}));
    // using SmemLayoutVtransposed = decltype(
    //     composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(ElementO);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(ElementO);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementO>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);

};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::bfloat16_t, typename elem_type_kv=cutlass::float_e5m2_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_kv_fp8 : public Base {
    using Element = typename Base::Element;
    using ElementKV = elem_type_kv;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64x16 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x16_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x16_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x16 = TiledMMA<
        MMA_Atom_Arch_16x64x16,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<128>>, Stride<Int<128>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<128>>{}));
    // using SmemLayoutK = Layout<Shape<Int<kBlockN>, Int<128>>, Stride<Int<128>, _1>>;
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(128/64)>, Int<64>>, Stride<Int<64>, _1>>;
    
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;
    
    using SmemLayoutAtomV = Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<64>, Int<64>>{}));
    // using SmemLayoutVtransposed = decltype(
    //     composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKSize + 2*size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);

};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::bfloat16_t, typename elem_type_kv=cutlass::float_e5m2_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_kv_fp8_dim64 : public Base {
    using Element = typename Base::Element;
    using ElementKV = elem_type_kv;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3; 

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64x16 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x16_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x16_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x16 = TiledMMA<
        MMA_Atom_Arch_16x64x16,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<64>>, Stride<Int<64>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<64>>{}));
    // using SmemLayoutK = Layout<Shape<Int<kBlockN>, Int<64>>, Stride<Int<64>, _1>>;
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(64/64)>, Int<64>>, Stride<Int<64>, _1>>;
    
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;
    
    using SmemLayoutAtomV = Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<64>, Int<64>>{}));
    // using SmemLayoutVtransposed = decltype(
    //     composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);

};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::bfloat16_t, typename elem_type_kv=cutlass::float_e5m2_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_kv_fp8_dim256 : public Base {
    using Element = typename Base::Element;
    using ElementKV = elem_type_kv;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3; 

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64x16 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x16_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x16_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x16 = TiledMMA<
        MMA_Atom_Arch_16x64x16,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<256>>, Stride<Int<256>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<256>>{}));
    // using SmemLayoutK = Layout<Shape<Int<kBlockN>, Int<256>>, Stride<Int<256>, _1>>;
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(256/64)>, Int<64>>, Stride<Int<64>, _1>>;
    
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;
    
    using SmemLayoutAtomV = Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<64>, Int<64>>{}));
    // using SmemLayoutVtransposed = decltype(
    //     composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);

};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_dim192 : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64x16 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x16_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x16_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x16 = TiledMMA<
        MMA_Atom_Arch_16x64x16,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<192>>, Stride<Int<192>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<192>>{}));
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(192/64)>, Int<64>>, Stride<Int<64>, _1>>;
    
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;
    
    using SmemLayoutAtomV = Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<64>, Int<64>>{}));
    // using SmemLayoutVtransposed = decltype(
    //     composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    using GmemTiledCopyQKVPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));      
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
    
    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load 
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_dim256 : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64x16 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x16_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x16_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x16 = TiledMMA<
        MMA_Atom_Arch_16x64x16,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<256>>, Stride<Int<256>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<256>>{}));
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(256/64)>, Int<64>>, Stride<Int<64>, _1>>;
    
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;
    
    using SmemLayoutAtomV = Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<64>, Int<64>>{}));
    // using SmemLayoutVtransposed = decltype(
    //     composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    using GmemTiledCopyQKVPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));      
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
    
    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load 
};
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::float_e4m3_t, typename elem_type_o=cutlass::bfloat16_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_fp8_dim64 : public Base {
    using Element = typename Base::Element;
    using ElementO = elem_type_o;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using MMA_Atom_Arch_16x64 = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT>;
    // using MMA_Atom_Arch_16x64x16 = MMA_Atom<GFX938_16x64x16_F32F8F8F32E4M3E4M3_NN>;
    using MMA_Atom_Arch_16x64x32_Blayout = MMA_Atom<GFX938_16x64x32_F32F8F8F32E4M3E4M3_NN_Blayout>;
    using MMA_Atom_Arch_16x64_BLayout = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_BLayout>;

    // using MMA_Atom_Arch_16x64_LIT = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT>;
    // using MMA_Atom_Arch_16x64_BLayout_LIT = MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT_BLayout>;
    // using MMA_Atom_Arch_16x64x32_NN = MMA_Atom<GFX938_16x64x32_F32F8F8F32E4M3E4M3_NN>;

    using MMA_Atom_Arch_16x64_LIT = std::conditional_t<
        std::is_same_v<elem_type, cutlass::float_e4m3_t>,
        MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT>,
        MMA_Atom<GFX938_16x64x64_F32BF8BF8F32E5M2E5M2_NT_LIT>
    >;

    using MMA_Atom_Arch_16x64_BLayout_LIT = std::conditional_t<
        std::is_same_v<elem_type, cutlass::float_e4m3_t>,
        MMA_Atom<GFX938_16x64x64_F32F8F8F32E4M3E4M3_NT_LIT_BLayout>,
        MMA_Atom<GFX938_16x64x64_F32BF8BF8F32E5M2E5M2_NT_LIT_BLayout>
    >;

    using MMA_Atom_Arch_16x64x32_NN = std::conditional_t<
        std::is_same_v<elem_type, cutlass::float_e4m3_t>,
        MMA_Atom<GFX938_16x64x32_F32F8F8F32E4M3E4M3_NN>,
        MMA_Atom<GFX938_16x64x32_F32BF8BF8F32E5M2E5M2_NN>
    >;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    // using TiledMma16x64x16 = TiledMMA<
    //     MMA_Atom_Arch_16x64x16,
    //     Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
    //     typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x32BLayout = TiledMMA<
        MMA_Atom_Arch_16x64x32_Blayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;

    using TiledMma16x64_LIT = TiledMMA<
        MMA_Atom_Arch_16x64_LIT,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64_Blayout_LIT = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout_LIT,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x32_NN = TiledMMA<
        MMA_Atom_Arch_16x64x32_NN,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<64>>, Stride<Int<64>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<64>>{}));
    // using SmemLayoutK = Layout<Shape<Int<kBlockN>, Int<128>>, Stride<Int<128>, _1>>;
    using SmemLayoutK = Layout<Shape<Int<kBlockN>, Int<64>>, Stride<Int<64>, _1>>;
    
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, ElementO>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;
    
    using SmemLayoutAtomV = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<64>, Int<64>>{}));
    // using SmemLayoutVtransposed = decltype(
    //     composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(ElementO);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKSize + 2*size(SmemLayoutV{}) * sizeof(Element);
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(ElementO);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementO>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);

};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_ws_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64x16 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x16_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x16_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<8>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<8>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x16 = TiledMMA<
        MMA_Atom_Arch_16x64x16,
        Layout<Shape<Int<8>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<8>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<8>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<128>>, Stride<Int<128>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<128>>{}));
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(128/64)>, Int<64>>, Stride<Int<64>, _1>>;
    
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;
    
    using SmemLayoutAtomV = Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<64>, Int<64>>{}));
    // using SmemLayoutVtransposed = decltype(
    //     composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = 65536;
    // static constexpr int kSmemSize = kSmemKSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = 512 == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_32, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    using GmemTiledCopyQKVPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));      
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
    
    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load 
};
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_prefetch_vllm_kvcache_traits_dim64 : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64x16 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x16_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x16_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x16 = TiledMMA<
        MMA_Atom_Arch_16x64x16,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK = Layout<Shape<Int<kBlockN>, Int<64>>, Stride<Int<64>, _1>>;
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<64>>{}));
    using SmemLayoutK = Layout<Shape<Int<kBlockN>, Int<64>>, Stride<Int<64>, _1>>;
    
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;
    
    using SmemLayoutAtomV = Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<64>, Int<64>>{}));
    // using SmemLayoutVtransposed = decltype(
    //     composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = kSmemKSize;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    using GmemTiledCopyQKVPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));      
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
    
    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load 
};
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_splitkv_vllm_kvcache_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64x16 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x16_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64x16 = TiledMMA<
        MMA_Atom_Arch_16x64x16,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK = decltype(
        composition(Swizzle<3, 3, 4>{},
                    Layout<Shape<_8, Int<32>>,
                        Stride<Int<32>, _1>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutAtomV = decltype(
        composition(Swizzle<3, 3, 4>{},
                    Layout<Shape<_8, Int<32>>,
                        Stride<Int<32>, _1>>{}));
    // using SmemLayoutAtomV = Layout<Shape<Int<128>, Int<64>>, Stride<Int<64>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<kHeadDimV>, Int<64>>{}));
    
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;
    
    // using SmemLayoutAtomV = Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, _1>>;
    // using SmemLayoutV = decltype(tile_to_shape(
    //     SmemLayoutAtomV{},
    //     Shape<Int<64>, Int<64>>{}));
    // using SmemLayoutVtransposed = decltype(
    //     composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutKV{}) * sizeof(Element);
    static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemOSize = size(SmemLayoutO{}) * sizeof(Element);
    // static constexpr int kSmemSize = std::max(kSmemKSize, kSmemOSize);
    static constexpr int kSmemSize = (kBlockN * kHeadDim + kBlockN * kHeadDimV) * 2;//(kv+v)*B
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kNThreads == 512 ? 16 : kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    using GmemTiledCopyQKVPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));      
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
    
    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load 
};


template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_prefetch_traits_dim256 : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    static constexpr uint32_t LayoutBlock = 64;
    static constexpr uint32_t LayoutDim = 128;
    using SmemLayoutAtomK = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    using SmemLayoutKsplit = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    
    using SmemLayoutAtomV = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<LayoutDim>, Int<LayoutBlock>>{}, GenRowMajor{})));
    using SmemLayoutVsplit = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<16>, Int<4*LayoutDim>>{}));
    using SmemLayoutVtransSplit = decltype(composition(SmemLayoutVsplit{}, make_layout(Shape<Int<4*LayoutDim>, Int<16>>{}, GenRowMajor{})));

    static constexpr int kSmemKSize = size(SmemLayoutKsplit{}) * sizeof(Element);
    static constexpr int kSmemVSize = size(SmemLayoutVsplit{}) * sizeof(Element);
    static constexpr int kSmemSize = std::max(kSmemKSize, kSmemVSize);
    
    using GmemLayoutAtom = Layout<Shape <_32, _8>, Stride< _8, _1>>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_16x64_prefetch_traits_dim512 : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = true;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kBlockN % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);
    static constexpr int kStages = kStages_;

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    using TiledMma16x64 = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;
    using TiledMma16x32 = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        typename Base::ValLayoutMNK>;

    static constexpr uint32_t LayoutBlock = 64;
    static constexpr uint32_t LayoutDim = 128;
    using SmemLayoutAtomK = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    using SmemLayoutKsplit = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    
    using SmemLayoutAtomV = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<LayoutDim>, Int<LayoutBlock>>{}, GenRowMajor{})));
    using SmemLayoutVsplit = decltype(tile_to_shape(SmemLayoutAtomV{}, Shape<Int<16>, Int<4*LayoutDim>>{}));
    using SmemLayoutVtransSplit = decltype(composition(SmemLayoutVsplit{}, make_layout(Shape<Int<4*LayoutDim>, Int<16>>{}, GenRowMajor{})));

    static constexpr int kSmemKSize = size(SmemLayoutKsplit{}) * sizeof(Element);
    static constexpr int kSmemVSize = size(SmemLayoutVsplit{}) * sizeof(Element);
    static constexpr int kSmemSize = std::max(kSmemKSize, kSmemVSize);
    
    using GmemLayoutAtom = Layout<Shape <_32, _8>, Stride< _8, _1>>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store
};


template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
        int AtomLayoutMSdP_=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=2,
        bool Is_V_in_regs_=false, bool No_double_buffer_=false, typename elem_type=cutlass::half_t,
        typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_bwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Is_V_in_regs = Is_V_in_regs_;
    static constexpr bool No_double_buffer = No_double_buffer_;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static_assert(kNWarps % AtomLayoutMSdP == 0);
    static_assert(kNWarps % AtomLayoutNdKV == 0);
    static_assert(kNWarps % AtomLayoutMdQ == 0);

    using TiledMmaSdP = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<kNWarps / AtomLayoutMSdP>, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadKV = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<kNWarps / AtomLayoutNdKV>, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadQ = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<kNWarps / AtomLayoutMdQ>, _1>>,  // 2x4x1 or 4x2x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQdO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));

    // using SmemLayoutAtomQdO = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<_8, Int<kBlockKSmem>>,
    //                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQdO = decltype(tile_to_shape(
        SmemLayoutAtomQdO{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    // using SmemLayoutAtomKV = decltype(
    //     composition(Swizzle<kSwizzle, 3, 3>{},
    //                 Layout<Shape<Int<kBlockM / kNWarps>, Int<kBlockKSmem>>,
    //                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutAtomKV = decltype(
        composition(Swizzle<2, 4, 2>{},
                    Layout<Shape<Int<kBlockM / kNWarps>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(
        // SmemLayoutAtomQdO{},
        SmemLayoutAtomKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    using SmemLayoutKtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutKtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutKtransposed{}));

    // TODO: generalize to other values of kBlockN
    // TODO: what should be the Swizzle here? 3 is faster than 1, and 1 is faster than 2
    static constexpr int kPBlockN = kBlockN;
    // Temporarily disabling this for hdim 256 on sm86 and sm89
    // static_assert(kBlockN >= 64);
    // static_assert(kBlockN >= 32);
    // TD [2023-03-19]: Idk why kPBlockN = 16 and kSwizzlePdS=3 is the fastest.
    // static constexpr int kPBlockN = kBlockN >= 64 ? 64 : 32;
    static_assert(kPBlockN == 16 || kPBlockN == 32 || kPBlockN == 64);
    // static constexpr int kSwizzlePdS = kPBlockN == 16 ? 1 : (kPBlockN == 32 ? 2 : 3);
    static constexpr int kSwizzlePdS = 3;
    // using SmemLayoutAtomPdS = decltype(
    //     composition(Swizzle<kSwizzlePdS, 3, 3>{},
    //                 Layout<Shape<Int<kBlockM>, Int<kPBlockN>>,
    //                        Stride<Int<kPBlockN>, _1>>{}));
    using SmemLayoutAtomPdS = decltype(
        composition(Swizzle<2, 4, 2>{},
                    Layout<Shape<Int<kBlockM>, Int<4>>,
                        Stride<Int<4>, _1>>{}));
    using SmemLayoutPdS = decltype(tile_to_shape(
        SmemLayoutAtomPdS{},
        make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
    using SmemLayoutPdStransposed = decltype(
        composition(SmemLayoutPdS{}, make_layout(Shape<Int<kBlockN>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutPdStransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutPdStransposed{}));

    using SmemCopyAtomPdS = Copy_Atom<DefaultCopy, elem_type>;

    using SmemLayoutQdOtransposed = decltype(
        composition(SmemLayoutQdO{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutQdOtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutQdOtransposed{}));

    // using SmemLayoutAtomdKV = decltype(
    //     composition(Swizzle<kSwizzle, 3, 3>{},
    //                 Layout<Shape<_8, Int<kBlockKSmem>>,
    //                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutAtomdKV = decltype(
        composition(Swizzle<2, 4, 2>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdKV = decltype(tile_to_shape(
        SmemLayoutAtomdKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdKV = Copy_Atom<DefaultCopy, elem_type>;

    using SmemLayoutAtomdQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    // using SmemLayoutAtomdQ = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<_8, Int<kBlockKSmem>>,
    //                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdQ = Copy_Atom<DefaultCopy, elem_type>;

    // Double buffer for sQ
    static constexpr int kSmemQdOSize = size(SmemLayoutQdO{}) * (No_double_buffer ? 2 : 3) * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemdSSize = (Is_V_in_regs ? size(SmemLayoutKV{}): size(SmemLayoutPdS{})) * sizeof(Element);
    static constexpr int kSmemPSize = size(SmemLayoutPdS{}) * sizeof(Element);
    static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(Element);
    static constexpr int kSmemSize = kSmemQdOSize
        + (!Is_V_in_regs
        ? kSmemKVSize + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)
        : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)));
    static constexpr int kSmemSize1colblock = kSmemQdOSize
        + (!Is_V_in_regs
        ? kSmemKVSize + kSmemdSSize + kSmemPSize
        : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + kSmemPSize));

    // wangaq debug for dq
    static constexpr int kSmemSize1rowblock = kSmemQdOSize
        + (!Is_V_in_regs ? kSmemKVSize : kSmemKVSize / 2);

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem instead of kHeadDim here to avoid bank conflicts, but doesn't seem
    // to affect speed in practice.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomdQaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopydQaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomdQaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store

    using GmemTiledCopydQaccumAtomicAdd = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        Layout<Shape <_8, _32>,  // Thread layout, 8 threads per row
                            Stride<_32, _1>>{},
                        Layout<Shape < _1, _1>>{}));  // Val layout, 1 val per store

};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
        int AtomLayoutMSdP_=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=2,
        bool Is_V_in_regs_=false, bool No_double_buffer_=false, bool Is_Q_in_regs_=false, 
        bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t,
        int kHeadDimV_ = kHeadDim_,
        typename Base=Flash_bwd_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, 
        AtomLayoutMSdP_, AtomLayoutNdKV, AtomLayoutMdQ, Is_V_in_regs_, No_double_buffer_, elem_type> >
struct Flash_bwd_kernel_dq_traits : public Flash_bwd_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, 
        AtomLayoutMSdP_, AtomLayoutNdKV, AtomLayoutMdQ, Is_V_in_regs_, No_double_buffer_, elem_type> {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDim;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static_assert(kNWarps % AtomLayoutMSdP == 0);
    static_assert(kNWarps % AtomLayoutMdQ == 0);

    using TiledMmaSdP = typename Base::TiledMmaSdP;

    using TiledMmadQ = TiledMMA<
        typename Base::MMA_Atom_Arch_FOR_GEMM1,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<kNWarps / AtomLayoutMdQ>, _1>>,  // 2x4x1 or 4x2x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQdO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));

    // using SmemLayoutAtomQdO = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<_8, Int<kBlockKSmem>>,
    //                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQdO = decltype(tile_to_shape(
        SmemLayoutAtomQdO{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using SmemLayoutAtomKV = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<kBlockM / kNWarps>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    // using SmemLayoutAtomKV = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<Int<kBlockM / kNWarps>, Int<kBlockKSmem>>,
    //                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(
        // SmemLayoutAtomQdO{},
        SmemLayoutAtomKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    using SmemLayoutKtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutKtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutKtransposed{}));

    // TODO: generalize to other values of kBlockN
    // TODO: what should be the Swizzle here? 3 is faster than 1, and 1 is faster than 2
    static constexpr int kPBlockN = kBlockN;
    // Temporarily disabling this for hdim 256 on sm86 and sm89
    // static_assert(kBlockN >= 64);
    // static_assert(kBlockN >= 32);
    // TD [2023-03-19]: Idk why kPBlockN = 16 and kSwizzlePdS=3 is the fastest.
    // static constexpr int kPBlockN = kBlockN >= 64 ? 64 : 32;
    static_assert(kPBlockN == 16 || kPBlockN == 32 || kPBlockN == 64);
    // static constexpr int kSwizzlePdS = kPBlockN == 16 ? 1 : (kPBlockN == 32 ? 2 : 3);
    static constexpr int kSwizzlePdS = 3;
    using SmemLayoutAtomPdS = decltype(
        composition(Swizzle<kSwizzlePdS, 3, 3>{},
                    Layout<Shape<Int<kBlockM>, Int<kPBlockN>>,
                        Stride<Int<kPBlockN>, _1>>{}));
    // using SmemLayoutAtomPdS = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<Int<kBlockM>, Int<4>>,
    //                        Stride<Int<4>, _1>>{}));
    using SmemLayoutPdS = decltype(tile_to_shape(
        SmemLayoutAtomPdS{},
        make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
    using SmemLayoutPdStransposed = decltype(
        composition(SmemLayoutPdS{}, make_layout(Shape<Int<kBlockN>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutPdStransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutPdStransposed{}));

    using SmemCopyAtomPdS = Copy_Atom<DefaultCopy, elem_type>;

    using SmemLayoutQdOtransposed = decltype(
        composition(SmemLayoutQdO{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutQdOtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutQdOtransposed{}));

    using SmemLayoutAtomdQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    // using SmemLayoutAtomdQ = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<_8, Int<kBlockKSmem>>,
    //                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdQ = Copy_Atom<DefaultCopy, elem_type>;

    // Double buffer for sQ
    static constexpr int kSmemQdOSize = size(SmemLayoutQdO{}) * 2 * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    
    static constexpr int kSmemSize1rowblock = Share_Q_K_smem ? std::max(kSmemQdOSize, kSmemKVSize) : kSmemQdOSize + kSmemKVSize;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem instead of kHeadDim here to avoid bank conflicts, but doesn't seem
    // to affect speed in practice.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
#if 0
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
#else
    using GmemLayoutAtomdQ = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtomdQ{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
#endif


};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
        int AtomLayoutMSdP_=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=2,
        bool Is_V_in_regs_=false, bool No_double_buffer_=false, typename elem_type=cutlass::half_t,
        int kHeadDimV_ = kHeadDim_,
        typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_bwd_kernel_trans_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Is_V_in_regs = Is_V_in_regs_;
    static constexpr bool No_double_buffer = No_double_buffer_;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static_assert(kNWarps % AtomLayoutMSdP == 0);
    static_assert(kNWarps % AtomLayoutNdKV == 0);
    static_assert(kNWarps % AtomLayoutMdQ == 0);

    using TiledMmaSdP = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<kNWarps / AtomLayoutMSdP>, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadKV = TiledMMA<
        typename Base::MMA_Atom_Arch_FOR_GEMM1,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<kNWarps / AtomLayoutNdKV>, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadQ = TiledMMA<
        typename Base::MMA_Atom_Arch_FOR_GEMM1,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<kNWarps / AtomLayoutMdQ>, _1>>,  // 2x4x1 or 4x2x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQdO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    // using SmemLayoutAtomQdO = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<_8, Int<kBlockKSmem>>,
    //                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQdO = decltype(tile_to_shape(
        SmemLayoutAtomQdO{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));


    using SmemLayoutAtomKV = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<kBlockM / kNWarps>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    // using SmemLayoutAtomKV = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<Int<kBlockM / kNWarps>, Int<4>>,
    //                        Stride<Int<4>, _1>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(
        // SmemLayoutAtomQdO{},
        SmemLayoutAtomKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    using SmemLayoutKtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutKtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutKtransposed{}));

    // TODO: generalize to other values of kBlockN
    // TODO: what should be the Swizzle here? 3 is faster than 1, and 1 is faster than 2
    // static constexpr int kPBlockN = kBlockN;
    // Temporarily disabling this for hdim 256 on sm86 and sm89
    // static_assert(kBlockN >= 64);
    static_assert(kBlockN >= 32);
    // TD [2023-03-19]: Idk why kPBlockN = 16 and kSwizzlePdS=3 is the fastest.
    static constexpr int kPBlockN = kBlockN >= 64 ? 64 : 32;
    static_assert(kPBlockN == 16 || kPBlockN == 32 || kPBlockN == 64);
    // static constexpr int kSwizzlePdS = kPBlockN == 16 ? 1 : (kPBlockN == 32 ? 2 : 3);
    static constexpr int kSwizzlePdS = 3;
    using SmemLayoutAtomPdS = decltype(
        composition(Swizzle<kSwizzlePdS, 3, 3>{},
                    Layout<Shape<Int<kBlockM>, Int<kPBlockN>>,
                        Stride<Int<kPBlockN>, _1>>{}));
    // using SmemLayoutAtomPdS = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<Int<kBlockM>, Int<4>>,
    //                        Stride<Int<4>, _1>>{}));
    using SmemLayoutPdS = decltype(tile_to_shape(
        SmemLayoutAtomPdS{},
        make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
    using SmemLayoutPdStransposed = decltype(
        composition(SmemLayoutPdS{}, make_layout(Shape<Int<kBlockN>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutPdStransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutPdStransposed{}));

    using SmemCopyAtomPdS = Copy_Atom<DefaultCopy, elem_type>;

    using SmemLayoutQdOtransposed = decltype(
        composition(SmemLayoutQdO{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutQdOtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutQdOtransposed{}));

    using SmemLayoutAtomdKV = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdKV = decltype(tile_to_shape(
        SmemLayoutAtomdKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutAtomdKVStore = decltype(
        composition(Swizzle<2, 4, 2>{},
                    Layout<Shape<_8, Int<4>>,
                        Stride<Int<4>, _1>>{}));
    using SmemLayoutdKVStore = decltype(tile_to_shape(
        SmemLayoutAtomdKVStore{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdKV = Copy_Atom<DefaultCopy, elem_type>;

    using SmemLayoutAtomdQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdQ = Copy_Atom<DefaultCopy, elem_type>;

    // Double buffer for sQ
    static constexpr int kSmemQdOSize = size(SmemLayoutQdO{}) * (No_double_buffer ? 2 : 3) * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemdSSize = (Is_V_in_regs ? size(SmemLayoutKV{}): size(SmemLayoutPdS{})) * sizeof(Element);
    static constexpr int kSmemPSize = size(SmemLayoutPdS{}) * sizeof(Element);
    static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(Element);
    static constexpr int kSmemSize = kSmemQdOSize
        + (!Is_V_in_regs
        ? kSmemKVSize + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)
        : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)));
    static constexpr int kSmemSize1colblock = kSmemQdOSize
        + (!Is_V_in_regs
        ? kSmemKVSize + kSmemdSSize + kSmemPSize
        : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + kSmemPSize));
    static constexpr int kSmemSizeTrans1colblock = kSmemQdOSize + kSmemKVSize + kSmemdQSize;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem instead of kHeadDim here to avoid bank conflicts, but doesn't seem
    // to affect speed in practice.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomdQaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopydQaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomdQaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store

    using GmemTiledCopydQaccumAtomicAdd = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        Layout<Shape <_8, _32>,  // Thread layout, 8 threads per row
                            Stride<_32, _1>>{},
                        Layout<Shape < _1, _1>>{}));  // Val layout, 1 val per store

    using GmemLayoutAtomdKVaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_64, _4>,  // Thread layout, 16 threads per row
            Stride< _64, _1>>
    >;
    using GmemTiledCopydKVaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomdKVaccum{},
                        Layout<Shape < _1, _32>>{}));  // Val layout, 4 vals per store    

};


template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
        int AtomLayoutMSdP_=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=2,
        bool Is_V_in_regs_=false, bool No_double_buffer_=false, typename elem_type=cutlass::half_t,
        int kHeadDimV_ = kHeadDim_,
        typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_bwd_kernel_trans_16x64_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Is_V_in_regs = Is_V_in_regs_;
    static constexpr bool No_double_buffer = No_double_buffer_;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static_assert(kNWarps % AtomLayoutMSdP == 0);
    static_assert(kNWarps % AtomLayoutNdKV == 0);
    static_assert(kNWarps % AtomLayoutMdQ == 0);
#if 0
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x16x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x16x16_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x16_F32F16F16F32_NT_FOR_GEMM1>,
        MMA_Atom<GFX928_16x64x16_F32BF16BF16F32_NT_FOR_GEMM1>
    >;
    using TiledMmaSdP = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<kNWarps / AtomLayoutMSdP>, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadKV = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<kNWarps / AtomLayoutNdKV>, _1>>,
        typename Base::ValLayoutMNK>;
#else

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<std::conditional_t< kHeadDim == 128, GFX928_16x64x32_F32F16F16F32_NT, GFX928_16x64x16_F32F16F16F32_NT >>,
        MMA_Atom<std::conditional_t< kHeadDim == 128 , GFX928_16x64x32_F32BF16BF16F32_NT, GFX928_16x64x16_F32BF16BF16F32_NT >>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMmaSdP = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<kNWarps / AtomLayoutMSdP>, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadKV = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<kNWarps / AtomLayoutNdKV>, _1>>,
        typename Base::ValLayoutMNK>;

#endif
    using TiledMmadQ = TiledMMA<
        typename Base::MMA_Atom_Arch_FOR_GEMM1,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<kNWarps / AtomLayoutMdQ>, _1>>,  // 2x4x1 or 4x2x1 thread group
        typename Base::ValLayoutMNK>;

// 打开swizzle
#if 1
    using SmemLayoutAtomQdO = decltype(
        composition(Swizzle<kSwizzle, 3, 5>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));

#else
    using SmemLayoutAtomQdO = decltype(
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{});

#endif
    // using SmemLayoutAtomQdO = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<_8, Int<kBlockKSmem>>,
    //                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQdO = decltype(tile_to_shape(
        SmemLayoutAtomQdO{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

#if 0
    using SmemLayoutAtomKV = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<kBlockM / kNWarps>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
#else
    using SmemLayoutAtomKV = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<kBlockM / kNWarps>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));

#endif
    // using SmemLayoutAtomKV = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<Int<kBlockM / kNWarps>, Int<4>>,
    //                        Stride<Int<4>, _1>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(
        // SmemLayoutAtomQdO{},
        SmemLayoutAtomKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    using SmemLayoutKtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutKtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutKtransposed{}));

    // TODO: generalize to other values of kBlockN
    // TODO: what should be the Swizzle here? 3 is faster than 1, and 1 is faster than 2
    // static constexpr int kPBlockN = kBlockN;
    // Temporarily disabling this for hdim 256 on sm86 and sm89
    // static_assert(kBlockN >= 64);
    static_assert(kBlockN >= 32);
    // TD [2023-03-19]: Idk why kPBlockN = 16 and kSwizzlePdS=3 is the fastest.
    static constexpr int kPBlockN = kBlockN >= 64 ? 64 : 32;
    static_assert(kPBlockN == 16 || kPBlockN == 32 || kPBlockN == 64);
    // static constexpr int kSwizzlePdS = kPBlockN == 16 ? 1 : (kPBlockN == 32 ? 2 : 3);
    static constexpr int kSwizzlePdS = 3;
#if 0
    using SmemLayoutAtomPdS = decltype(
        composition(Swizzle<kSwizzlePdS, 3, 3>{},
                    Layout<Shape<Int<kBlockM>, Int<kPBlockN>>,
                        Stride<Int<kPBlockN>, _1>>{}));
#else
    using SmemLayoutAtomPdS = decltype(
        composition(Swizzle<kSwizzlePdS, 3, 3>{},
                    Layout<Shape<Int<kBlockM>, Int<kPBlockN>>,
                        Stride<Int<kPBlockN>, _1>>{}));
#endif
    // using SmemLayoutAtomPdS = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<Int<kBlockM>, Int<4>>,
    //                        Stride<Int<4>, _1>>{}));
    using SmemLayoutPdS = decltype(tile_to_shape(
        SmemLayoutAtomPdS{},
        make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
    using SmemLayoutPdStransposed = decltype(
        composition(SmemLayoutPdS{}, make_layout(Shape<Int<kBlockN>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutPdStransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutPdStransposed{}));

    using SmemCopyAtomPdS = Copy_Atom<DefaultCopy, elem_type>;

    using SmemLayoutQdOtransposed = decltype(
        composition(SmemLayoutQdO{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutQdOtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutQdOtransposed{}));

    using SmemLayoutAtomdKV = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdKV = decltype(tile_to_shape(
        SmemLayoutAtomdKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutAtomdKVStore = decltype(
        composition(Swizzle<2, 4, 2>{},
                    Layout<Shape<_8, Int<4>>,
                        Stride<Int<4>, _1>>{}));
    using SmemLayoutdKVStore = decltype(tile_to_shape(
        SmemLayoutAtomdKVStore{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdKV = Copy_Atom<DefaultCopy, elem_type>;

    using SmemLayoutAtomdQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdQ = Copy_Atom<DefaultCopy, elem_type>;

    // Double buffer for sQ
    static constexpr int kSmemQdOSize = size(SmemLayoutQdO{}) * (No_double_buffer ? 2 : 3) * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemdSSize = (Is_V_in_regs ? size(SmemLayoutKV{}): size(SmemLayoutPdS{})) * sizeof(Element);
    static constexpr int kSmemPSize = size(SmemLayoutPdS{}) * sizeof(Element);
    static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(Element);
    static constexpr int kSmemSize = kSmemQdOSize
        + (!Is_V_in_regs
        ? kSmemKVSize + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)
        : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + std::max(kSmemPSize, kSmemdQSize)));
    static constexpr int kSmemSize1colblock = kSmemQdOSize
        + (!Is_V_in_regs
        ? kSmemKVSize + kSmemdSSize + kSmemPSize
        : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + kSmemPSize));
#if 0
    static constexpr int kSmemSizeTrans1colblock = kSmemQdOSize + kSmemKVSize;
#else
    static constexpr int kSmemSizeTrans1colblock = std::max(kSmemQdOSize, kSmemKVSize);
    // kSmemQdOSize + kSmemKVSize;
#endif
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem instead of kHeadDim here to avoid bank conflicts, but doesn't seem
    // to affect speed in practice.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 0  
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>,  
            Stride< _4, _1>>;
#endif
    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store

    using GmemLayoutAtomKV = Layout<Shape <_64, _4>, Stride<_4, _1>>;
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtomKV{},
                        Layout<Shape < _1, Int<8>>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemLayoutAtomdQaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_16, _16>,  // Thread layout, 16 threads per row
            Stride< _16, _1>>
    >;
    using GmemTiledCopydQaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomdQaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store

    using GmemTiledCopydQaccumAtomicAdd = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        Layout<Shape <_8, _32>,  // Thread layout, 8 threads per row
                            Stride<_32, _1>>{},
                        Layout<Shape < _1, _1>>{}));  // Val layout, 1 val per store

    using GmemLayoutAtomdKVaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_32, _8>,  // Thread layout, 8 threads per row
            Stride< _8, _1>>,
        Layout<Shape <_64, _4>,  // Thread layout, 16 threads per row
            Stride< _64, _1>>
    >;
    using GmemTiledCopydKVaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomdKVaccum{},
                        Layout<Shape < _1, _32>>{}));  // Val layout, 4 vals per store    

};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_bwd_kernel_trans_16x64_prefetch_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static constexpr int kStages = kStages_;
    static constexpr int kBlockNSmem = kStages * 16;
    static constexpr int kBlockKSmem = kStages * 32;
    static_assert(kBlockM % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMmaSdP = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadKV = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQGemm0 = Layout<Shape<Int<kBlockM>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;
    using SmemLayoutAtomdOGemm0 = Layout<Shape<Int<kBlockM>, Int<kHeadDimV>>, Stride<Int<kHeadDimV>, _1>>;

    using SmemLayoutQGemm0 = decltype(tile_to_shape(SmemLayoutAtomQGemm0{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemLayoutdOGemm0 = decltype(tile_to_shape(SmemLayoutAtomdOGemm0{}, make_shape(Int<kBlockM>{}, Int<kHeadDimV>{})));
    
    using SmemLayoutQ = Layout<Shape<Int<kBlockM*(kHeadDim/64)>, Int<64>>, Stride<Int<64>, _1>>;
    using SmemLayoutdO = Layout<Shape<Int<kBlockM*(kHeadDimV/64)>, Int<64>>, Stride<Int<64>, _1>>;

    using SmemLayoutAtomQdOGemm1 = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutQGemm1 = decltype(tile_to_shape(SmemLayoutAtomQdOGemm1{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutQGemm1transposed = decltype(composition(SmemLayoutQGemm1{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutQGemm1transposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutQGemm1transposed{}));
    using SmemLayoutQsplit = decltype(tile_to_shape(SmemLayoutAtomQdOGemm1{}, Shape<Int<16>, Int<(kStages + 1)*kHeadDim>>{}));
    using SmemLayoutQtransSplit = decltype(composition(SmemLayoutQsplit{}, make_layout(Shape<Int<(kStages + 1)*kHeadDim>, Int<16>>{}, GenRowMajor{})));

    using SmemLayoutdOGemm1 = decltype(tile_to_shape(SmemLayoutAtomQdOGemm1{}, Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemLayoutdOGemm1transposed = decltype(composition(SmemLayoutdOGemm1{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutdOGemm1transposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutdOGemm1transposed{}));
    using SmemLayoutdOsplit = decltype(tile_to_shape(SmemLayoutAtomQdOGemm1{}, Shape<Int<16>, Int<(kStages + 1)*kHeadDimV>>{}));
    using SmemLayoutdOtransSplit = decltype(composition(SmemLayoutdOsplit{}, make_layout(Shape<Int<(kStages + 1)*kHeadDimV>, Int<16>>{}, GenRowMajor{})));

    
    using SmemLayoutAtomdKVStore = decltype(
        composition(Swizzle<4, 2, 4>{},
                    Layout<Shape<_16, Int<4>>,
                        Stride<Int<4>, _1>>{}));
    using SmemLayoutdKStore = decltype(tile_to_shape(
        SmemLayoutAtomdKVStore{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutdVStore = decltype(tile_to_shape(
        SmemLayoutAtomdKVStore{},
        make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));
    using SmemCopyAtomdKV = Copy_Atom<DefaultCopy, elem_type>;


    static constexpr int kSmemQSize = size(SmemLayoutQGemm0{}) * sizeof(Element);
    static constexpr int kSmemdOSize = size(SmemLayoutdOGemm0{}) * sizeof(Element);
    static constexpr int kSmemOffset = kHeadDim == 192 ? 4096 : 0;
    static constexpr int kSmemPrefetchSize = kHeadDim == 64 ? kSmemQSize + kSmemdOSize : std::max(kSmemQSize, kSmemdOSize);

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = 8;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>, Stride< _4, _1>>;
#endif
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, Int<8>>>{}));  // Val layout, 8 vals per store

};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_bwd_kernel_trans_16x64_prefetch_traits_dim96 : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static constexpr int kStages = kStages_;
    static_assert(kBlockM % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMmaSdP = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadKV = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQdOGemm0 = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutQdO = decltype(tile_to_shape(SmemLayoutAtomQdOGemm0{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutAtomQdOGemm1 = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutQGemm1 = decltype(tile_to_shape(SmemLayoutAtomQdOGemm1{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutQGemm1transposed = decltype(composition(SmemLayoutQGemm1{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutQsplit = decltype(tile_to_shape(SmemLayoutAtomQdOGemm1{}, Shape<Int<16>, Int<4*kHeadDim>>{}));
    using SmemLayoutQtransSplit = decltype(composition(SmemLayoutQsplit{}, make_layout(Shape<Int<4*kHeadDim>, Int<16>>{}, GenRowMajor{})));

    using SmemLayoutdOGemm1 = decltype(tile_to_shape(SmemLayoutAtomQdOGemm1{}, Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemLayoutdOGemm1transposed = decltype(composition(SmemLayoutdOGemm1{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutdOsplit = decltype(tile_to_shape(SmemLayoutAtomQdOGemm1{}, Shape<Int<16>, Int<4*kHeadDimV>>{}));
    using SmemLayoutdOtransSplit = decltype(composition(SmemLayoutdOsplit{}, make_layout(Shape<Int<4*kHeadDimV>, Int<16>>{}, GenRowMajor{})));

    
    using SmemLayoutAtomdKVStore = decltype(
        composition(Swizzle<4, 2, 4>{},
                    Layout<Shape<_16, Int<4>>,
                        Stride<Int<4>, _1>>{}));
    using SmemLayoutdKStore = decltype(tile_to_shape(
        SmemLayoutAtomdKVStore{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutdVStore = decltype(tile_to_shape(
        SmemLayoutAtomdKVStore{},
        make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));
    using SmemCopyAtomdKV = Copy_Atom<DefaultCopy, elem_type>;


    static constexpr int kSmemQSize = size(SmemLayoutQdO{}) * sizeof(Element);
    static constexpr int kSmemdOSize = size(SmemLayoutQdO{}) * sizeof(Element);
    static constexpr int kSmemPrefetchSize = kHeadDim == 64 ? kSmemQSize + kSmemdOSize : std::max(kSmemQSize, kSmemdOSize);

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = 8;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>, Stride< _4, _1>>;
#endif
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, Int<8>>>{}));  // Val layout, 8 vals per store

};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_bwd_kernel_trans_16x64_prefetch_traits_dim256 : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static constexpr int kStages = kStages_;
    static_assert(kBlockM % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMmaSdP = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadKV = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        typename Base::ValLayoutMNK>;

    
    static constexpr uint32_t LayoutBlock = 64;
    static constexpr uint32_t LayoutDim = 128;
    using SmemLayoutAtomQdOGemm0 = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutQdOGemm0 = decltype(tile_to_shape(SmemLayoutAtomQdOGemm0{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    
    using SmemLayoutAtomQdOGemm1 = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutQdOGemm1 = decltype(tile_to_shape(SmemLayoutAtomQdOGemm1{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    using SmemLayoutQdOGemm1transposed = decltype(composition(SmemLayoutQdOGemm1{}, make_layout(Shape<Int<LayoutDim>, Int<LayoutBlock>>{}, GenRowMajor{})));
    using SmemLayoutQdOsplit = decltype(tile_to_shape(SmemLayoutAtomQdOGemm1{}, Shape<Int<16>, Int<4*LayoutDim>>{}));
    using SmemLayoutQdOtransSplit = decltype(composition(SmemLayoutQdOsplit{}, make_layout(Shape<Int<4*LayoutDim>, Int<16>>{}, GenRowMajor{})));
    
    using SmemLayoutAtomdKVStore = decltype(
        composition(Swizzle<4, 2, 4>{},
                    Layout<Shape<_16, Int<4>>,
                        Stride<Int<4>, _1>>{}));
    using SmemLayoutdKStore = decltype(tile_to_shape(
        SmemLayoutAtomdKVStore{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutdVStore = decltype(tile_to_shape(
        SmemLayoutAtomdKVStore{},
        make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));
    using SmemCopyAtomdKV = Copy_Atom<DefaultCopy, elem_type>;


    static constexpr int kSmemQdOSize = size(SmemLayoutQdOGemm0{}) * sizeof(Element);
    static constexpr int kSmemPrefetchSize = kSmemQdOSize;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = 8;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>, Stride< _4, _1>>;
#endif
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, Int<8>>>{}));  // Val layout, 8 vals per store

};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_bwd_kernel_trans_16x64_prefetch_traits_dim512 : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static constexpr int kStages = kStages_;
    static_assert(kBlockM % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMmaSdP = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadKV = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        typename Base::ValLayoutMNK>;

    
    static constexpr uint32_t LayoutBlock = 64;
    static constexpr uint32_t LayoutDim = 128;
    using SmemLayoutAtomQdOGemm0 = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutQdOGemm0 = decltype(tile_to_shape(SmemLayoutAtomQdOGemm0{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    
    using SmemLayoutAtomQdOGemm1 = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutQdOGemm1 = decltype(tile_to_shape(SmemLayoutAtomQdOGemm1{}, Shape<Int<LayoutBlock>, Int<LayoutDim>>{}));
    using SmemLayoutQdOGemm1transposed = decltype(composition(SmemLayoutQdOGemm1{}, make_layout(Shape<Int<LayoutDim>, Int<LayoutBlock>>{}, GenRowMajor{})));
    using SmemLayoutQdOsplit = decltype(tile_to_shape(SmemLayoutAtomQdOGemm1{}, Shape<Int<16>, Int<4*LayoutDim>>{}));
    using SmemLayoutQdOtransSplit = decltype(composition(SmemLayoutQdOsplit{}, make_layout(Shape<Int<4*LayoutDim>, Int<16>>{}, GenRowMajor{})));
    
    using SmemLayoutAtomdKVStore = decltype(
        composition(Swizzle<4, 2, 4>{},
                    Layout<Shape<_16, Int<4>>,
                        Stride<Int<4>, _1>>{}));
    using SmemLayoutdKStore = decltype(tile_to_shape(
        SmemLayoutAtomdKVStore{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutdVStore = decltype(tile_to_shape(
        SmemLayoutAtomdKVStore{},
        make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));
    using SmemCopyAtomdKV = Copy_Atom<DefaultCopy, elem_type>;


    static constexpr int kSmemQdOSize = size(SmemLayoutQdOGemm0{}) * sizeof(Element);
    static constexpr int kSmemPrefetchSize = kSmemQdOSize;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = 8;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>, Stride< _4, _1>>;
#endif
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, Int<8>>>{}));  // Val layout, 8 vals per store

};





template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_bwd_kernel_trans_16x64_prefetch_mla_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static constexpr int kStages = kStages_;
    static_assert(kBlockM % 64 == 0);
    static_assert(kHeadDim % 32 == 0);
    static_assert(kHeadDimV % 32 == 0);

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMmaSdP = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadKV = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        typename Base::ValLayoutMNK>;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQdOGemm0 = Layout<Shape<Int<kBlockM>, Int<kHeadDimV>>, Stride<Int<kHeadDimV>, _1>>;
    using SmemLayoutQdOGemm0 = decltype(tile_to_shape(
        SmemLayoutAtomQdOGemm0{},
        make_shape(Int<kBlockM>{}, Int<kHeadDimV>{})));
    using SmemLayoutQdO = Layout<Shape<Int<kBlockM*(kHeadDimV/64)>, Int<64>>, Stride<Int<64>, _1>>;
    
    using SmemLayoutAtomQdOGemm1 = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutQdOGemm1 = decltype(tile_to_shape(
        SmemLayoutAtomQdOGemm1{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemLayoutQdOGemm1transposed = decltype(
        composition(SmemLayoutQdOGemm1{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockM>>{}, GenRowMajor{})));

    using SmemLayoutQGemm1Tail = decltype(tile_to_shape(
        SmemLayoutAtomQdOGemm1{},
        Shape<Int<kBlockM>, Int<64>>{}));
    using SmemLayoutQGemm1TailTransposed = decltype(
        composition(SmemLayoutQGemm1Tail{}, make_layout(Shape<Int<64>, Int<kBlockM>>{}, GenRowMajor{})));

    
    using SmemLayoutAtomdKVStore = decltype(
        composition(Swizzle<4, 2, 4>{},
                    Layout<Shape<_16, Int<4>>,
                        Stride<Int<4>, _1>>{}));
    using SmemLayoutdKStore = decltype(tile_to_shape(
        SmemLayoutAtomdKVStore{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutdVStore = decltype(tile_to_shape(
        SmemLayoutAtomdKVStore{},
        make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));
    using SmemCopyAtomdKV = Copy_Atom<DefaultCopy, elem_type>;


    static constexpr int kSmemQdOSize = size(SmemLayoutQdOGemm0{}) * sizeof(Element);
    static constexpr int kSmemPrefetchSize = kSmemQdOSize;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = 8;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
#if 1
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
#else
    using GmemLayoutAtom = Layout<Shape <_64, _4>, Stride< _4, _1>>;
#endif
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, Int<8>>>{}));  // Val layout, 8 vals per store

};

//////////////////////////////////////////////////////////////////////////////////////////////

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
        int AtomLayoutMSdP_=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=2,
        bool Is_V_in_regs_=false, bool No_double_buffer_=false, bool Is_Q_in_regs_=false, 
        bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t,
        int kHeadDimV_ = kHeadDim_,
        typename Base=Flash_bwd_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, 
        AtomLayoutMSdP_, AtomLayoutNdKV, AtomLayoutMdQ, Is_V_in_regs_, No_double_buffer_, elem_type> >
struct Flash_bwd_kernel_dq_16x64_traits : public Flash_bwd_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, 
        AtomLayoutMSdP_, AtomLayoutNdKV, AtomLayoutMdQ, Is_V_in_regs_, No_double_buffer_, elem_type> {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static_assert(kNWarps % AtomLayoutMSdP == 0);
    static_assert(kNWarps % AtomLayoutMdQ == 0);
#if 0
    using TiledMmaSdP = typename Base::TiledMmaSdP;

    using TiledMmadQ = TiledMMA<
        typename Base::MMA_Atom_Arch_FOR_GEMM1,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<kNWarps / AtomLayoutMdQ>, _1>>,  // 2x4x1 or 4x2x1 thread group
        typename Base::ValLayoutMNK>;
#else
    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<std::conditional_t< kHeadDim == 128, GFX928_16x64x32_F32F16F16F32_NT, GFX928_16x64x16_F32F16F16F32_NT >>,
        MMA_Atom<std::conditional_t< kHeadDim == 128 , GFX928_16x64x32_F32BF16BF16F32_NT, GFX928_16x64x16_F32BF16BF16F32_NT >>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMmaSdP = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<kNWarps / AtomLayoutMSdP>, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadQ = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<kNWarps / AtomLayoutMdQ>, _1>>,  // 2x4x1 or 4x2x1 thread group
        typename Base::ValLayoutMNK>;

#endif
    using SmemLayoutAtomQdO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutV = Layout<Shape<Int<kBlockN*(kHeadDim/64)>, Int<64>>, Stride<Int<64>, _1>>;
    // using SmemLayoutAtomQdO = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<_8, Int<kBlockKSmem>>,
    //                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQdO = decltype(tile_to_shape(
        SmemLayoutAtomQdO{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

#if 0
    using SmemLayoutAtomKV = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<kBlockM / kNWarps>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
#else
    using SmemLayoutAtomKV = decltype(
        composition(Swizzle<kSwizzle, 3, 5>{},
                    Layout<Shape<Int<16>, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));

#endif
    // using SmemLayoutAtomKV = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<Int<kBlockM / kNWarps>, Int<kBlockKSmem>>,
    //                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(
        // SmemLayoutAtomQdO{},
        SmemLayoutAtomKV{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    using SmemLayoutKtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutKtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutKtransposed{}));

    // TODO: generalize to other values of kBlockN
    // TODO: what should be the Swizzle here? 3 is faster than 1, and 1 is faster than 2
    static constexpr int kPBlockN = kBlockN;
    // Temporarily disabling this for hdim 256 on sm86 and sm89
    // static_assert(kBlockN >= 64);
    // static_assert(kBlockN >= 32);
    // TD [2023-03-19]: Idk why kPBlockN = 16 and kSwizzlePdS=3 is the fastest.
    // static constexpr int kPBlockN = kBlockN >= 64 ? 64 : 32;
    static_assert(kPBlockN == 16 || kPBlockN == 32 || kPBlockN == 64);
    // static constexpr int kSwizzlePdS = kPBlockN == 16 ? 1 : (kPBlockN == 32 ? 2 : 3);
    static constexpr int kSwizzlePdS = 3;
    using SmemLayoutAtomPdS = decltype(
        composition(Swizzle<kSwizzlePdS, 3, 3>{},
                    Layout<Shape<Int<kBlockM>, Int<kPBlockN>>,
                        Stride<Int<kPBlockN>, _1>>{}));
    // using SmemLayoutAtomPdS = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<Int<kBlockM>, Int<4>>,
    //                        Stride<Int<4>, _1>>{}));
    using SmemLayoutPdS = decltype(tile_to_shape(
        SmemLayoutAtomPdS{},
        make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
    using SmemLayoutPdStransposed = decltype(
        composition(SmemLayoutPdS{}, make_layout(Shape<Int<kBlockN>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutPdStransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutPdStransposed{}));

    using SmemCopyAtomPdS = Copy_Atom<DefaultCopy, elem_type>;

    using SmemLayoutQdOtransposed = decltype(
        composition(SmemLayoutQdO{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutQdOtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutQdOtransposed{}));

    using SmemLayoutAtomdQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                        Stride<Int<kBlockKSmem>, _1>>{}));
    // using SmemLayoutAtomdQ = decltype(
    //     composition(Swizzle<2, 4, 2>{},
    //                 Layout<Shape<_8, Int<kBlockKSmem>>,
    //                        Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdQ = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
    using SmemCopyAtomdQ = Copy_Atom<DefaultCopy, elem_type>;

    // Double buffer for sQ
    static constexpr int kSmemQdOSize = size(SmemLayoutQdO{}) * 2 * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    
    static constexpr int kSmemSize1rowblock = Share_Q_K_smem ? std::max(size(SmemLayoutdQ{}) * 2, kSmemKVSize) : kSmemQdOSize + kSmemKVSize;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem instead of kHeadDim here to avoid bank conflicts, but doesn't seem
    // to affect speed in practice.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");

    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemLayoutAtomdQ = Layout<Shape <Int<kNThreads / 4>, Int<4>>,
                                Stride<Int<4>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtomdQ{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store

};



////////////////////////////////////////////////////////////////////////////////////////////////////
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
        int AtomLayoutMSdP_=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=2,
        bool Is_V_in_regs_=false, bool No_double_buffer_=false, bool Is_Q_in_regs_=false, 
        bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_,
        typename Base=Flash_bwd_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, 
        AtomLayoutMSdP_, AtomLayoutNdKV, AtomLayoutMdQ, Is_V_in_regs_, No_double_buffer_, elem_type> >
struct Flash_bwd_kernel_dq_16x64_prefetch_traits : public Flash_bwd_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, 
        AtomLayoutMSdP_, AtomLayoutNdKV, AtomLayoutMdQ, Is_V_in_regs_, No_double_buffer_, elem_type> {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;
    static constexpr int kStages = kStages_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static_assert(kNWarps % AtomLayoutMSdP == 0);
    static_assert(kNWarps % AtomLayoutMdQ == 0);
#if 0

#else
    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using MMA_Atom_Arch_16x64_BLayout = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NT_BLayout>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NT_BLayout>
    >;
    using TiledMma16x64BLayout = TiledMMA<
        MMA_Atom_Arch_16x64_BLayout,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        typename Base::ValLayoutMNK>;
    using TiledMmaSdP = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<kNWarps / AtomLayoutMSdP>, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadQ = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<kNWarps / AtomLayoutMdQ>, _1>>,  // 2x4x1 or 4x2x1 thread group
        typename Base::ValLayoutMNK>;

#endif

    using SmemLayoutAtomdQ = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutdQ = decltype(tile_to_shape(
        SmemLayoutAtomdQ{},
        make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

    using SmemLayoutV = Layout<Shape<Int<kBlockN*(kHeadDimV/64)>, Int<64>>, Stride<Int<64>, _1>>;
    using SmemLayoutK = Layout<Shape<Int<kBlockN*(kHeadDim/64)>, Int<64>>, Stride<Int<64>, _1>>;

    using SmemLayoutAtomKGemm0 = Layout<Shape<Int<kBlockN>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;
    using SmemLayoutAtomVGemm0 = Layout<Shape<Int<kBlockN>, Int<kHeadDimV>>, Stride<Int<kHeadDimV>, _1>>;

    using SmemLayoutKGemm0 = decltype(tile_to_shape(
        SmemLayoutAtomKGemm0{},
        make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

    using SmemLayoutVGemm0 = decltype(tile_to_shape(
        SmemLayoutAtomVGemm0{},
        make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));

    using SmemLayoutAtomKGemm1 = decltype(
        composition(Swizzle<0, 3, 3>{},
                Layout<Shape<Int<8>, Int<32>>,
                        Stride<Int<32>, _1>>{}));
    using SmemLayoutKGemm1 = decltype(tile_to_shape(
        SmemLayoutAtomKGemm1{},
            make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutKGemm1transposed = decltype(
        composition(SmemLayoutKGemm1{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutKGemm1transposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutKGemm1transposed{}));
    
    static constexpr int kSmemKSize = size(SmemLayoutKGemm0{}) * sizeof(Element);
    static constexpr int kSmemVSize = size(SmemLayoutVGemm0{}) * sizeof(Element);
    static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(Element);
    static constexpr int kSmemPrefetchSize = kHeadDim == 64 ? kSmemKSize + kSmemVSize : std::max(kSmemKSize, kSmemVSize);
    // static constexpr int kSmemPrefetchSize = std::max(std::max(kSmemKSize, kSmemVSize), kSmemdQSize);

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = 8;

    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;
        
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  

    using GmemLayoutAtomdQ = Layout<Shape <Int<kNThreads / 4>, Int<4>>,
                        Stride<Int<4>, _1>>;
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtomdQ{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
        int AtomLayoutMSdP_=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=2,
        typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_,
        typename Base=Flash_bwd_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, 
        AtomLayoutMSdP_, AtomLayoutNdKV, AtomLayoutMdQ, false, false, elem_type> >
struct Flash_bwd_kernel_dq_16x64_prefetch_traits_dim96 : public Flash_bwd_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, 
        AtomLayoutMSdP_, AtomLayoutNdKV, AtomLayoutMdQ, false, false, elem_type> {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;
    static constexpr int kStages = kStages_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kHeadDim % 32 == 0);

    static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
    static_assert(kNWarps % AtomLayoutMSdP == 0);
    static_assert(kNWarps % AtomLayoutMdQ == 0);

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMmaSdP = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<kNWarps / AtomLayoutMSdP>, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadQ = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<kNWarps / AtomLayoutMdQ>, _1>>,  // 2x4x1 or 4x2x1 thread group
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomKVGemm0 = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;

    using SmemLayoutKVGemm0 = decltype(tile_to_shape(SmemLayoutAtomKVGemm0{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutKVGemm0Split = decltype(tile_to_shape(SmemLayoutAtomKVGemm0{}, Shape<Int<64>, Int<128>>{}));

    using SmemLayoutAtomKGemm1 = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutKGemm1 = decltype(tile_to_shape(SmemLayoutAtomKGemm1{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
    using SmemLayoutKGemm1transposed = decltype(composition(SmemLayoutKGemm1{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutKsplit = decltype(tile_to_shape(SmemLayoutAtomKGemm1{}, Shape<Int<16>, Int<4*kHeadDim>>{}));
    using SmemLayoutKtransSplit = decltype(composition(SmemLayoutKsplit{}, make_layout(Shape<Int<4*kHeadDim>, Int<16>>{}, GenRowMajor{})));
    
    static constexpr int kSmemKSize = size(SmemLayoutKVGemm0Split{}) * sizeof(Element);
    static constexpr int kSmemKtSize = size(SmemLayoutKtransSplit{}) * sizeof(Element);
    static constexpr int kSmemOffset = 3072;
    static constexpr int kSmemOffsetSize = kSmemOffset * sizeof(Element);
    static constexpr int kSmemPrefetchSize = std::max(kSmemKSize, kSmemKtSize + kSmemOffsetSize);

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = 8;

    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;
        
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  

    using GmemLayoutAtomdQ = Layout<Shape <Int<kNThreads / 4>, Int<4>>,
                        Stride<Int<4>, _1>>;
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtomdQ{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
};


template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_,
        typename Base=Flash_bwd_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, 4, 1, 4, false, false, elem_type> >
struct Flash_bwd_kernel_dq_16x64_prefetch_traits_dim256 : public Flash_bwd_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, 
        4, 1, 4, false, false, elem_type> {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;
    static constexpr int kStages = kStages_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kHeadDim % 32 == 0);

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMmaSdP = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>, Int<1>, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadQ = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>, Int<1>, _1>>,
        typename Base::ValLayoutMNK>;

    static constexpr uint32_t LayoutBlock = 64;
    static constexpr uint32_t LayoutDim = 128;
    using SmemLayoutAtomKVGemm0 = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutKVGemm0 = decltype(tile_to_shape(SmemLayoutAtomKVGemm0{}, make_shape(Int<LayoutBlock>{}, Int<LayoutDim>{})));
    using SmemLayoutKVGemm0Split = decltype(tile_to_shape(SmemLayoutAtomKVGemm0{}, Shape<Int<64>, Int<128>>{}));

    using SmemLayoutAtomKGemm1 = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutKGemm1 = decltype(tile_to_shape(SmemLayoutAtomKGemm1{}, make_shape(Int<LayoutBlock>{}, Int<LayoutDim>{})));
    using SmemLayoutKGemm1transposed = decltype(composition(SmemLayoutKGemm1{}, make_layout(Shape<Int<LayoutDim>, Int<LayoutBlock>>{}, GenRowMajor{})));
    using SmemLayoutKsplit = decltype(tile_to_shape(SmemLayoutAtomKGemm1{}, Shape<Int<16>, Int<4*LayoutDim>>{}));
    using SmemLayoutKtransSplit = decltype(composition(SmemLayoutKsplit{}, make_layout(Shape<Int<4*LayoutDim>, Int<16>>{}, GenRowMajor{})));
    
    static constexpr int kSmemKSize = size(SmemLayoutKVGemm0Split{}) * sizeof(Element);
    static constexpr int kSmemKtSize = size(SmemLayoutKtransSplit{}) * sizeof(Element);
    static constexpr int kSmemPrefetchSize = std::max(kSmemKSize, kSmemKtSize);

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = 8;

    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;
        
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  

    using GmemLayoutAtomdQ = Layout<Shape <Int<kNThreads / 4>, Int<4>>,
                        Stride<Int<4>, _1>>;
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtomdQ{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t,
        int kStages_=1, int kHeadDimV_ = kHeadDim_,
        typename Base=Flash_bwd_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, 4, 1, 4, false, false, elem_type> >
struct Flash_bwd_kernel_dq_16x64_prefetch_traits_dim512 : public Flash_bwd_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, 
        4, 1, 4, false, false, elem_type> {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;
    static constexpr int kStages = kStages_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kHeadDim % 32 == 0);

    using MMA_Atom_Arch_16x64 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x64x32_F32F16F16F32_NN>,
        MMA_Atom<GFX928_16x64x32_F32BF16BF16F32_NN>
    >;
    using MMA_Atom_Arch_16x32 = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMmaSdP = TiledMMA<
        MMA_Atom_Arch_16x64,
        Layout<Shape<Int<kNWarps>, Int<1>, _1>>,
        typename Base::ValLayoutMNK>;

    using TiledMmadQ = TiledMMA<
        MMA_Atom_Arch_16x32,
        Layout<Shape<Int<kNWarps>, Int<1>, _1>>,
        typename Base::ValLayoutMNK>;

    static constexpr uint32_t LayoutBlock = 64;
    static constexpr uint32_t LayoutDim = 128;
    using SmemLayoutAtomKVGemm0 = Layout<Shape<Int<64>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutKVGemm0 = decltype(tile_to_shape(SmemLayoutAtomKVGemm0{}, make_shape(Int<LayoutBlock>{}, Int<LayoutDim>{})));
    using SmemLayoutKVGemm0Split = decltype(tile_to_shape(SmemLayoutAtomKVGemm0{}, Shape<Int<64>, Int<128>>{}));

    using SmemLayoutAtomKGemm1 = Layout<Shape<Int<16>, Int<32>>, Stride<Int<32>, _1>>;
    using SmemLayoutKGemm1 = decltype(tile_to_shape(SmemLayoutAtomKGemm1{}, make_shape(Int<LayoutBlock>{}, Int<LayoutDim>{})));
    using SmemLayoutKGemm1transposed = decltype(composition(SmemLayoutKGemm1{}, make_layout(Shape<Int<LayoutDim>, Int<LayoutBlock>>{}, GenRowMajor{})));
    using SmemLayoutKsplit = decltype(tile_to_shape(SmemLayoutAtomKGemm1{}, Shape<Int<16>, Int<4*LayoutDim>>{}));
    using SmemLayoutKtransSplit = decltype(composition(SmemLayoutKsplit{}, make_layout(Shape<Int<4*LayoutDim>, Int<16>>{}, GenRowMajor{})));
    
    static constexpr int kSmemKSize = size(SmemLayoutKVGemm0Split{}) * sizeof(Element);
    static constexpr int kSmemKtSize = size(SmemLayoutKtransSplit{}) * sizeof(Element);
    static constexpr int kSmemPrefetchSize = std::max(kSmemKSize, kSmemKtSize);

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = 8;

    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;
        
    using GmemTiledCopydO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtom{},
                        Layout<Shape < _1, _8>>{}));  

    using GmemLayoutAtomdQ = Layout<Shape <Int<kNThreads / 4>, Int<4>>,
                        Stride<Int<4>, _1>>;
    using GmemTiledCopydQ = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, elem_type>{},
                        GmemLayoutAtomdQ{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per store
};