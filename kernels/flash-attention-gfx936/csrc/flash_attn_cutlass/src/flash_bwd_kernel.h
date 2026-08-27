/***************************************************************************************************
* Copyright (c) 2024, Tri Dao.
******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"

#include "alibi.h"


#define S_WAITCNT0 asm volatile("s_waitcnt vmcnt(0) \n s_barrier")
#define S_WAITCNT1 asm volatile("s_waitcnt vmcnt(1) \n s_barrier")
#define S_WAITCNT2 asm volatile("s_waitcnt vmcnt(2) \n s_barrier")
#define S_WAITCNT3 asm volatile("s_waitcnt vmcnt(3) \n s_barrier")
#define S_WAITCNT4 asm volatile("s_waitcnt vmcnt(4) \n s_barrier")
#define S_WAITCNT5 asm volatile("s_waitcnt vmcnt(5) \n s_barrier")
#define S_WAITCNT6 asm volatile("s_waitcnt vmcnt(6) \n s_barrier")

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_N,
        class... Args,
        class TiledMMA>
CUTE_HOST_DEVICE
auto
make_tiled_copy_B_warpcontiguousN(Copy_Atom<Args...> const& copy_atom,
                                TiledMMA           const& tiled_mma) {
    // constexpr int TileShape_N = decltype(tiled_mma.template tile_size_mnk<1>())::value;
    // constexpr int TileShape_K = decltype(tiled_mma.template tile_size_mnk<2>())::value;
    using TiledShape_MNK = typename TiledMMA::TiledShape_MNK;
    constexpr int TileShape_N = decltype(size<1>(TiledShape_MNK{}))::value;
    constexpr int TileShape_K = decltype(size<2>(TiledShape_MNK{}))::value;

    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    // Divide by 2 because right now we always use 2 for the ValLayout
    constexpr int kNWarpsN = TileShape_N / AtomShape_N;
    constexpr int MMAStride_N = MMA_N * AtomShape_N;

    // This gives the correct layout, idk why.
    auto t = make_tile(Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _1>,   // (16, 1, 1)
                            Stride<_1, Int<MMAStride_N>, _8> >{},         // (1, 32, 8)
                    make_layout(Int<TileShape_K>{}));

    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutB_TV(), t);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_N,
        class... Args,
        class TiledMMA>
CUTE_HOST_DEVICE
auto
make_tiled_copy_C_warpcontiguousN(Copy_Atom<Args...> const& copy_atom,
                                TiledMMA           const& tiled_mma) {
    // constexpr int TileShape_M = decltype(tiled_mma.template tile_size_mnk<0>())::value;
    // constexpr int TileShape_N = decltype(tiled_mma.template tile_size_mnk<1>())::value;
    using TiledShape_MNK = typename TiledMMA::TiledShape_MNK;
    constexpr int TileShape_M = decltype(size<0>(TiledShape_MNK{}))::value;
    constexpr int TileShape_N = decltype(size<1>(TiledShape_MNK{}))::value;

    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    // Divide by 2 because right now we always use 2 for the ValLayout
    constexpr int kNWarpsN = TileShape_N / AtomShape_N;
    constexpr int MMAStride_N = MMA_N * AtomShape_N;

    auto t = make_tile(make_layout(Int<TileShape_M>{}),
                    Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _1>,
                            Stride<_1, Int<MMAStride_N>, _8> >{});       // (1, 64, 8) or (1, 32, 8)
    // (_64:_1,(_16,_1,_2):(_1,_32,_8))

    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutC_TV(), t);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_first, bool Is_last, bool Seq_parallel=false, typename Params>
inline __device__ void compute_dq_dk_dv_1colblock(const Params &params, const int bidb, const int bidh, const int n_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    using SdP_TiledShape_MNK = typename Kernel_traits::TiledMmaSdP::TiledShape_MNK;
    // constexpr int MMA_N_SdP = kBlockN / decltype(typename Kernel_traits::TiledMmaSdP{}.template tile_size_mnk<1>())::value;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(SdP_TiledShape_MNK{}))::value;;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;    // true

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);
    if constexpr(Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    const index_t row_offset_dq_accum = binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb)
        + ((m_block_max - 1) * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128 * bidb)) * params.h * params.d_rounded + bidh * params.d_rounded
        // If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
        + (!params.deterministic ? 0 : blockIdx.x * params.dq_accum_split_stride);
    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.h * params.d_rounded, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    // Double buffer for sQ
    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(),
                                                typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    Tensor sK = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sdS = make_tensor(Kernel_traits::Is_V_in_regs ? sV.data() : sV.data() + size(sV),
                            typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sdStNoSwizzle = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor sP = make_tensor(Kernel_traits::Is_V_in_regs ? (sdS.data() + size(sV)) : (sdS.data() + size(sdS)), typename Kernel_traits::SmemLayoutPdS{});
    // Tensor sP = make_tensor(sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sPtNoSwizzle = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    // sP and sdQ share the same memory so be careful
    Tensor sdQ = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutdQ{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    using GmemTiledCopydO = typename Kernel_traits::GmemTiledCopyQKV;
    GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    using GmemLayoutAtomdQaccum = std::conditional_t<
        !Seq_parallel,
        typename Kernel_traits::GmemTiledCopydQaccum,
        typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd
    >;
    GmemLayoutAtomdQaccum gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);         // (MMA,MMA_N,MMA_K)
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);         // (MMA,MMA_N,MMA_K)
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);      // (MMA,MMA_N,MMA_K)
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);        // (MMA,MMA_N,MMA_K)

    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdKrdSt = thr_mma_dkv.partition_fragment_A(sdStNoSwizzle); // (MMA, MMA_N, MMA_K)
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);   // (MMA, MMA_K, MMA_N)
    Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);   // (MMA, MMA_N, MMA_K)
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle); // (MMA, MMA_K, MMA_N)

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);                      // (MMA, MMA_N, MMA_K)
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);    // (MMA, MMA_K, MMA_N)

    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    // auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    // Partition sP and sdS to match the accumulator partitioning
    // This has to be tiled_mma_sdp, not tiled_mma_dkv
    // auto smem_tiled_copy_PdS = make_tiled_copy_C_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
    auto smem_tiled_copy_PdS = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);      // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);   // ((Atom,AtomNum),PIPE_M,PIPE_N)

    auto smem_tiled_copy_PdSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);
    Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
    Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);

    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);

    auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tidx);
    Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);

    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // We'll advance gdQ and gdQaccum before the 1st read/write.
    tdQgdQ.data() = tdQgdQ.data() + kBlockM * params.dq_row_stride;
    tdQgdQaccum.data() = tdQgdQaccum.data() + kBlockM * params.h * params.d_rounded;

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);
    // If not local, we're guaranteed that m_block_min <= m_block:
    // We checked earlier that n_block * kBlockN < actual_seqlen_k, so in the causal case,
    // n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k < actual_seqlen_q.
    // So m_block_min <= (actual_seqlen_q - 1) / kBlockM.
    // Recall that m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM) = (actual_seqlen_q + kBlockM - 1) / kBlockM.
    // So m_block_m - 1 = (actual_seqlen_q - 1) / kBlockM.
    // We conclude that m_block_min <= m_block, so we will always have at least 1 iteration of the for loop.
    // However, if local, then this possible to have some blocks of K & V not attending to any query.
    // We might need to exit early and write 0 to dK and dV for those blocks.
    // Otherwise we get wrong result for the case where we don't enter the for loop.
    // And we might read OOB elements from gQ and gdO.
    // This also covers the case where actual_seqlen_q == 0
    if constexpr(Is_local || !Is_even_MN) {
        if (m_block < m_block_min) {
            const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
            + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
            const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
            + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
            Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                    Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                    make_stride(params.dk_row_stride, _1{}));
            Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                    Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                    make_stride(params.dv_row_stride, _1{}));
            typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
            auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
            Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
            Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
            Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
            Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
            clear(tdKrdK);
            clear(tdVrdV);
            Tensor cdKV = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
            Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
            Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
            #pragma unroll
            for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
            // Clear_OOB_K must be false since we don't want to write zeros to gmem
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
            return;
        }
    }

    if constexpr(Double_buffer) {
        if (m_block % 2 == 1) {  // Double buffer for sQ
            tQsQ.data() = tQsQ.data() + size(sQ);
            tSsQ.data() = tSsQ.data() + size(sQ);
            tdKsQt.data() = tdKsQt.data() + size(sQ);
        }
    }

    if ((!Is_first && !Seq_parallel) || params.deterministic) { __syncthreads(); }

    //// 预先加载V，global->smem，如果sV和sdS共用一块smem
    if (Kernel_traits::Is_V_in_regs) {
        // Clear the smem tiles to account for predicated off loads
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        // flash::cp_async_fence();
    }

    //// 预先加载dO，global->smem
    {
        // Clear the smem tiles to account for predicated off loads
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }

    //// 预先加载Q，global->smem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    static_assert(decltype(size<0>(taccScS))::value == 4);

    // Convert to (4, MMA_N, MMA_K) then take only the row indices.
    Tensor taccScS_row = logical_divide(taccScS, Shape<_1>{})(0, _, 0);

    //// 预先加载lse，global->smem
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    // We want LSE = inf if the row is OOB. In that case Q would be zero, K would be zero,
    // and scores would be zero. With LSE = 0, probs will be all 1's, and when we multiply
    // with V (which would be zero), we're fine. However, with ALiBi, we might modify these
    // scores, and probs can become NaN. Instead if we set LSE = inf for OOB rows, probs are always 0.

    //// 预先加载K，global->smem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    //// 预先加载V，global->smem，如果sV和sdS不共用一块smem
    if constexpr(!Kernel_traits::Is_V_in_regs) {
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        // flash::cp_async_fence();
    }
        // __builtin_amdgcn_sched_barrier(0);

    if (Kernel_traits::Is_V_in_regs) {
        // cute::cp_async_wait<1>();
        __syncthreads();
        Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));            // M
        cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
    }

    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    clear(acc_dv);
    clear(acc_dk);

    //// 设置偏移计算alibi
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    for (; m_block >= m_block_min; --m_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_K) acc_s= (_4,_1,_2):(_1,_0,_4)
        clear(acc_s);
        // __syncthreads();

        Tensor dP_sum = make_fragment_like(lse);
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi)));
        }

        flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
                    smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }

        // Reshape acc_s from (MMA=4, MMA_N, MMA_K) to (row=(MMA_N), col=(4, MMA_K))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

        // Softcapping - calculating dTanh and scaling dS later with it
        Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }

        // Alibi
        if constexpr(Has_alibi) {
            const int warp_id = tidx / 64;
            const int col_idx_offset = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = AtomLayoutMS * 16;
            alibi.apply_alibi(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }

        // TD [2023-07-29]: I was thinking that we don't need to mask out the elements beyond
        // actual_seqlen_k, because acc_s would be some finite value for those indices.
        // In the end when we multiply with K to get dQ, the corresponding values of K would be 0,
        // so the result would still be correct.
        // However, it's possible that the values in acc_s are so large that they overflow
        // when we multiply with dP and convert to fp16, resulting in Inf in dS and NaNs in dQ.
        // So we need to mask out the elements beyond actual_seqlen_k.
        if constexpr(!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = tidx / 64;
                const int col_idx_offset_ = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
                flash::apply_mask(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if constexpr(Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = tidx / 64;
                flash::apply_mask_causal(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        AtomLayoutMS * 16);
            }
        } else if constexpr(Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = tidx / 64;
                flash::apply_mask_local(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }

        // Compute the exponential value.
        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);

        if constexpr(Is_dropout) {
            const int warp_id = tidx / 64;
            const int warp_row_stride = 16;
            int block_row_idx = m_block * (kBlockM / 16) + warp_id % AtomLayoutMS;
            int block_col_idx = n_block * (kBlockN / 16) + (warp_id / AtomLayoutMS) * MMA_N_SdP;
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS
            );
        }

        Tensor tPrP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);

        Tensor tPaP = smem_thr_copy_PdS.retile_S(tPrP);     // ((Atom,AtomNum), MMA_N, MMA_K)
        cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);
        // __syncthreads();

        Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_K)
        CUTE_STATIC_ASSERT_V(size<0>(acc_dp) == size<0>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<1>(acc_dp) == size<1>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<2>(acc_dp) == size<2>(acc_s));                     // MMA

        clear(acc_dp);

        flash::gemm</*A_in_regs=*/false, /*B_in_regs=*/Kernel_traits::Is_V_in_regs>(
            acc_dp, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp,
                    smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);  // (_4,_1,_2):(_1,_0,_4)
        // __syncthreads(); // Need syncthreads since we're writing to the same sdO location

        // Reshape acc_dp from (MMA=4, MMA_N, MMA_K) to (row=(2, MMA_N), col=(2, MMA_N))
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());

        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        //// tdQgdQaccum拷贝到转置后的acc_dq中
        Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
        tdQgdQaccum.data() = tdQgdQaccum.data() + (-int(kBlockM * params.h * params.d_rounded));
        if constexpr(Seq_parallel) {
            clear(acc_dq);
        } else {
            // Reshape acc_dq from (4, 2, 4) to (4, 4, 2) to write to gdQaccum
            Tensor acc_dq_reshaped = make_tensor(acc_dq.data(),
                                                make_layout(get<0>(acc_dq.layout()),
                                                            get<2>(acc_dq.layout()),
                                                            get<1>(acc_dq.layout())));
            cute::copy(gmem_tiled_copy_dQaccum, tdQgdQaccum, acc_dq_reshaped);
        }

        if constexpr(Double_buffer) {
            if (m_block > m_block_min) {
                // Double buffer for sQ
                const int sQ_offset = m_block % 2 == 0 ? size(sQ) : -size(sQ);
                tQsQ.data() = tQsQ.data() + sQ_offset;
                tSsQ.data() = tSsQ.data() + sQ_offset;
                // Advance gQ
                tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
                __syncthreads();
            }
        }

        ////////// 将dS结果拷贝到smem，用于计算dq和dk
        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        // Convert dS from fp32 to fp16
        Tensor tdSrdS = flash::convert_type<Element>(dS_reshaped);
        Tensor tdSadS = smem_thr_copy_PdS.retile_S(tdSrdS); // ((Atom,AtomNum), MMA_N, MMA_K)
        cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
        // __syncthreads();

        flash::gemm(acc_dv, tdVrPt, tdVrdO, tdVsPt, tdVsdOt, tiled_mma_dkv,
                    smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);
        // __syncthreads(); // Need syncthreads since we're writing to the same sdO location

        if (m_block > m_block_min) {
            // Advance gdO
            tdOgdO.data() = tdOgdO.data() + (-int(kBlockM * params.do_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ);
            // flash::cp_async_fence();
        }
        __syncthreads();

        flash::gemm(acc_dq, tdQrdS, tdQrKt, tdQsdS, tdQsKt, tiled_mma_dq,
                    smem_tiled_copy_dS, smem_tiled_copy_Kt, smem_thr_copy_dS, smem_thr_copy_Kt);
        // if (thread(0, 0)) {
        //     printf("tdSrdS layout:"); print(tdSrdS.layout()); printf("\n");
        //     printf("tdQrKt layout:"); print(tdQrKt.layout()); printf("\n");
        //     printf("acc_dq layout:"); print(acc_dq.layout()); printf("\n");
        // }

        if (m_block > m_block_min) {
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) { lse(mi) = gLSE(get<0>(taccScS_row(mi))); }
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
        }

        // Reshape acc_dq from (4, 2, 4) to (4, 4, 2) to write to gdQaccum
        Tensor acc_dq_reshaped = make_tensor(acc_dq.data(),
                                                make_layout(get<0>(acc_dq.layout()),
                                                            get<2>(acc_dq.layout()),
                                                            get<1>(acc_dq.layout())));

        if constexpr(!Seq_parallel) {
            cute::copy(gmem_tiled_copy_dQaccum, acc_dq_reshaped, tdQgdQaccum);
        } else {
            CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));
            #pragma unroll
            for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }
        }

        flash::gemm(acc_dk, tdKrdSt, tdKrQt, tdKsdSt, tdKsQt, tiled_mma_dkv,
                    smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);

        if constexpr(Double_buffer) {  // Double buffer for sQ
            tdKsQt.data() = tdKsQt.data() + (m_block % 2 == 0 ? size(sQ) : -size(sQ));
        }
        if constexpr(!Double_buffer) {
            if (m_block > m_block_min) {
                // __syncthreads();
                // Advance gQ
                tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
                // flash::cp_async_fence();
                __syncthreads();
            }
        }
    }
#if 1
    // Epilogue

    if constexpr(Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dv from fp32 to fp16
    Tensor rdK = flash::convert_type<Element>(acc_dk);
    Tensor rdV = flash::convert_type<Element>(acc_dv);

    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_K)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_K)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    // If Is_last, there's already a __syncthreads() at the end of the loop.
    __syncthreads();

    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
#endif
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_first, bool Is_last, bool Seq_parallel=false, typename Params>
inline __device__ void compute_dk_dv_1colblock(const Params &params, const int bidb, const int bidh, const int n_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    using SdP_TiledShape_MNK = typename Kernel_traits::TiledMmaSdP::TiledShape_MNK;
    // constexpr int MMA_N_SdP = kBlockN / decltype(typename Kernel_traits::TiledMmaSdP{}.template tile_size_mnk<1>())::value;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(SdP_TiledShape_MNK{}))::value;;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;    // true

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);
    if constexpr(Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    // Double buffer for sQ
    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(),
                                                typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    Tensor sK = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sdS = make_tensor(Kernel_traits::Is_V_in_regs ? sV.data() : sV.data() + size(sV),
                            typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sdStNoSwizzle = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor sP = make_tensor(Kernel_traits::Is_V_in_regs ? (sdS.data() + size(sV)) : (sdS.data() + size(sdS)), typename Kernel_traits::SmemLayoutPdS{});
    // Tensor sP = make_tensor(sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sPtNoSwizzle = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    using GmemTiledCopydO = typename Kernel_traits::GmemTiledCopyQKV;
    GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);         // (MMA,MMA_N,MMA_K)
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);         // (MMA,MMA_N,MMA_K)
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);      // (MMA,MMA_N,MMA_K)
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);        // (MMA,MMA_N,MMA_K)

    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdKrdSt = thr_mma_dkv.partition_fragment_A(sdStNoSwizzle); // (MMA, MMA_N, MMA_K)
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);   // (MMA, MMA_K, MMA_N)
    Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);   // (MMA, MMA_N, MMA_K)
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle); // (MMA, MMA_K, MMA_N)

    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    // auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    // Partition sP and sdS to match the accumulator partitioning
    // This has to be tiled_mma_sdp, not tiled_mma_dkv
    // auto smem_tiled_copy_PdS = make_tiled_copy_C_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
    auto smem_tiled_copy_PdS = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);      // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);   // ((Atom,AtomNum),PIPE_M,PIPE_N)

    auto smem_tiled_copy_PdSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);
    Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
    Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);

    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);
    // If not local, we're guaranteed that m_block_min <= m_block:
    // We checked earlier that n_block * kBlockN < actual_seqlen_k, so in the causal case,
    // n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k < actual_seqlen_q.
    // So m_block_min <= (actual_seqlen_q - 1) / kBlockM.
    // Recall that m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM) = (actual_seqlen_q + kBlockM - 1) / kBlockM.
    // So m_block_m - 1 = (actual_seqlen_q - 1) / kBlockM.
    // We conclude that m_block_min <= m_block, so we will always have at least 1 iteration of the for loop.
    // However, if local, then this possible to have some blocks of K & V not attending to any query.
    // We might need to exit early and write 0 to dK and dV for those blocks.
    // Otherwise we get wrong result for the case where we don't enter the for loop.
    // And we might read OOB elements from gQ and gdO.
    // This also covers the case where actual_seqlen_q == 0
    if constexpr(Is_local || !Is_even_MN) {
        if (m_block < m_block_min) {
            const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
            + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
            const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
            + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
            Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                    Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                    make_stride(params.dk_row_stride, _1{}));
            Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                    Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                    make_stride(params.dv_row_stride, _1{}));
            typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
            auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
            Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
            Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
            Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
            Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
            clear(tdKrdK);
            clear(tdVrdV);
            Tensor cdKV = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
            Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
            Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
            #pragma unroll
            for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
            // Clear_OOB_K must be false since we don't want to write zeros to gmem
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
            return;
        }
    }

    if constexpr(Double_buffer) {
        if (m_block % 2 == 1) {  // Double buffer for sQ
            tQsQ.data() = tQsQ.data() + size(sQ);
            tSsQ.data() = tSsQ.data() + size(sQ);
            tdKsQt.data() = tdKsQt.data() + size(sQ);
        }
    }

    if ((!Is_first && !Seq_parallel) || params.deterministic) { __syncthreads(); }

    //// 预先加载V，global->smem，如果sV和sdS共用一块smem
    if (Kernel_traits::Is_V_in_regs) {
        // Clear the smem tiles to account for predicated off loads
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::cp_async_fence();
    }

    //// 预先加载dO，global->smem
    {
        // Clear the smem tiles to account for predicated off loads
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }

    //// 预先加载Q，global->smem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    static_assert(decltype(size<0>(taccScS))::value == 4);

    // Convert to (4, MMA_N, MMA_K) then take only the row indices.
    Tensor taccScS_row = logical_divide(taccScS, Shape<_1>{})(0, _, 0);

    //// 预先加载lse，global->smem
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    // We want LSE = inf if the row is OOB. In that case Q would be zero, K would be zero,
    // and scores would be zero. With LSE = 0, probs will be all 1's, and when we multiply
    // with V (which would be zero), we're fine. However, with ALiBi, we might modify these
    // scores, and probs can become NaN. Instead if we set LSE = inf for OOB rows, probs are always 0.

    //// 预先加载K，global->smem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    //// 预先加载V，global->smem，如果sV和sdS不共用一块smem
    if constexpr(!Kernel_traits::Is_V_in_regs) {
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::cp_async_fence();
    }

    // if (Kernel_traits::Is_V_in_regs) {
    //     cute::cp_async_wait<1>();
    //     __syncthreads();
    //     Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
    //     CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));            // M
    //     cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
    // }

    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    clear(acc_dv);
    clear(acc_dk);

    //// 设置偏移计算alibi
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    for (; m_block >= m_block_min; --m_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_K) acc_s= (_4,_1,_2):(_1,_0,_4)
        clear(acc_s);
        __syncthreads();

        Tensor dP_sum = make_fragment_like(lse);
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi)));
        }

        flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
                    smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }

        // Reshape acc_s from (MMA=4, MMA_N, MMA_K) to (row=(MMA_N), col=(4, MMA_K))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

        // Softcapping - calculating dTanh and scaling dS later with it
        Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }

        // Alibi
        if constexpr(Has_alibi) {
            const int warp_id = tidx / 64;
            const int col_idx_offset = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = AtomLayoutMS * 16;
            alibi.apply_alibi(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }

        // TD [2023-07-29]: I was thinking that we don't need to mask out the elements beyond
        // actual_seqlen_k, because acc_s would be some finite value for those indices.
        // In the end when we multiply with K to get dQ, the corresponding values of K would be 0,
        // so the result would still be correct.
        // However, it's possible that the values in acc_s are so large that they overflow
        // when we multiply with dP and convert to fp16, resulting in Inf in dS and NaNs in dQ.
        // So we need to mask out the elements beyond actual_seqlen_k.
        if constexpr(!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = tidx / 64;
                const int col_idx_offset_ = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
                flash::apply_mask(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if constexpr(Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = tidx / 64;
                flash::apply_mask_causal(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        AtomLayoutMS * 16);
            }
        } else if constexpr(Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = tidx / 64;
                flash::apply_mask_local(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }

        // Compute the exponential value.
        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);

        if constexpr(Is_dropout) {
            const int warp_id = tidx / 64;
            const int warp_row_stride = 16;
            int block_row_idx = m_block * (kBlockM / 16) + warp_id % AtomLayoutMS;
            int block_col_idx = n_block * (kBlockN / 16) + (warp_id / AtomLayoutMS) * MMA_N_SdP;
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS
            );
        }

        Tensor tPrP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);

        Tensor tPaP = smem_thr_copy_PdS.retile_S(tPrP);     // ((Atom,AtomNum), MMA_N, MMA_K)
        cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);

        Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_K)
        CUTE_STATIC_ASSERT_V(size<0>(acc_dp) == size<0>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<1>(acc_dp) == size<1>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<2>(acc_dp) == size<2>(acc_s));                     // MMA

        clear(acc_dp);

        flash::gemm(acc_dp, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp,
                    smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);  // (_4,_1,_2):(_1,_0,_4)

        // Reshape acc_dp from (MMA=4, MMA_N, MMA_K) to (row=(2, MMA_N), col=(2, MMA_N))
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());

        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        // if constexpr(Double_buffer) {
        //     if (m_block > m_block_min) {
        //         // Double buffer for sQ
        //         const int sQ_offset = m_block % 2 == 0 ? size(sQ) : -size(sQ);
        //         tQsQ.data() = tQsQ.data() + sQ_offset;
        //         tSsQ.data() = tSsQ.data() + sQ_offset;
        //         // Advance gQ
        //         tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
        //         flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
        //         __syncthreads();
        //     }
        // }

        ////////// 将dS结果拷贝到smem，用于计算dq和dk
        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        // Convert dS from fp32 to fp16
        Tensor tdSrdS = flash::convert_type<Element>(dS_reshaped);
        Tensor tdSadS = smem_thr_copy_PdS.retile_S(tdSrdS); // ((Atom,AtomNum), MMA_N, MMA_K)
        cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
        __syncthreads();

        flash::gemm(acc_dv, tdVrPt, tdVrdO, tdVsPt, tdVsdOt, tiled_mma_dkv,
                    smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);
        __syncthreads(); // Need syncthreads since we're writing to the same sdO location

        if (m_block > m_block_min) {
            // Advance gdO
            tdOgdO.data() = tdOgdO.data() + (-int(kBlockM * params.do_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ);
            flash::cp_async_fence();
        }

        if (m_block > m_block_min) {
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) { lse(mi) = gLSE(get<0>(taccScS_row(mi))); }
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
        }

        flash::gemm(acc_dk, tdKrdSt, tdKrQt, tdKsdSt, tdKsQt, tiled_mma_dkv,
                    smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);

        // if constexpr(Double_buffer) {  // Double buffer for sQ
        //     tdKsQt.data() = tdKsQt.data() + (m_block % 2 == 0 ? size(sQ) : -size(sQ));
        // }
        if constexpr(!Double_buffer) {
            if (m_block > m_block_min) {
                __syncthreads();
                // Advance gQ
                tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
                flash::cp_async_fence();
            }
        }
    }
#if 1
    // Epilogue
    // __builtin_amdgcn_sched_barrier(0);
    if constexpr(Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dv from fp32 to fp16
    Tensor rdK = flash::convert_type<Element>(acc_dk);
    Tensor rdV = flash::convert_type<Element>(acc_dv);

    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_K)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_K)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
    __syncthreads();

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////


template<
    typename Kernel_traits,
    typename RTensor,
    typename STensor,
    typename GTensor,
    typename Element,
    bool Is_even_MN, bool Is_even_K
    >
inline __device__ void _bwd_store_dk_dv(RTensor& rAcc, STensor& sAcc, int tidx, GTensor& gdAcc, int dim, int max_MN) {
    // Convert acc_dv from fp32 to fp16
    Tensor rdAcc = flash::convert_type<Element>(rAcc);

    Tensor sdAcc = make_tensor(sAcc.data(), typename Kernel_traits::SmemLayoutdKVStore{});  // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, typename Kernel_traits::TiledMmadKV{});
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKVrdKV = smem_thr_copy_dKV.retile_S(rAcc);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKVsdKV = smem_thr_copy_dKV.partition_D(sdAcc);   // ((Atom,AtomNum),PIPE_M,PIPE_N)


    cute::copy(smem_tiled_copy_dKV,  flash::convert_type<Element>(taccdKVrdKV), taccdKVsdKV);

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKVsdKV = gmem_thr_copy_dKV.partition_S(sdAcc);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKVgdKV = gmem_thr_copy_dKV.partition_D(gdAcc);

    __syncthreads();
    Tensor tdKVrdKV = make_tensor<Element>(shape(tdKVgdKV));
    cute::copy(gmem_tiled_copy_dKV, tdKVsdKV, tdKVrdKV);
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdAcc), size<1>(sdAcc)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKVgdKV)));
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < dim; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKVrdKV, tdKVgdKV, tdKVcdKV, tdKVpdKV, max_MN
    );
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_first, bool Is_last, bool Seq_parallel=false, typename Params>
inline __device__ void compute_dq_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    using SdP_TiledShape_MNK = typename Kernel_traits::TiledMmaSdP::TiledShape_MNK;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(SdP_TiledShape_MNK{}))::value;;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    
    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + m_block * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + m_block * kBlockM;

    // wangaq debug
    // if (cute::thread(0, 0)) {
    //     Element * tmp = reinterpret_cast<Element*>(params.q_ptr) + row_offset_q;
    //     printf("row_offset_q:%d %.4f\n", row_offset_q, half_t::convert(tmp[7]));
    //     // for (int i = 0; i < kBlockM*kHeadDim/8; ++i) {
    //     //     printf("Q %d: ", i);
    //     //     for (int j = 0; j < 8; ++j) {
    //     //         printf("%.4f ", half_t::convert(tmp[i*8+j]));
    //     //     }
    //     //     printf("\n");
    //     // }
    // }

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    
    Tensor sdO = make_tensor(sK.data() + ((Kernel_traits::Share_Q_K_smem && size(sQ) > size(sK) ? 
        size(sQ) : size(sK))), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sV = make_tensor(sdO.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sdO)), typename Kernel_traits::SmemLayoutKV{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);

    // dQ
    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    //
    // Copy Atom retiling
    //

    // S/dP
    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    // dQ
    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue
    
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
        
        Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.dq_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
        Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
        clear(tdQrdQ);
        Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
        Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
        if constexpr(!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        return;
    }

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    // if constexpr(Kernel_traits::Is_Q_in_regs) { __syncthreads(); }
    __syncthreads();

    if constexpr (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_QdO.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_QdO, tSsQ, tSrQ_copy_view);
        
        Tensor tdPrdO_copy_view = smem_thr_copy_QdO.retile_D(tdPrdO);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsdO) == size<1>(tdPrdO_copy_view));
        cute::copy(smem_tiled_copy_QdO, tdPsdO, tdPrdO_copy_view);
    }
    __syncthreads();

    // wangaq debug
    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    static_assert(decltype(size<0>(taccScS))::value == 4);
    Tensor taccScS_row = taccScS(0, _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }

    int n_block = n_block_max - 1;
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    __syncthreads();
    
    // wangaq debug
    // __syncthreads();
    // if (cute::thread(0, 0)) {
    //     __half * tmp = reinterpret_cast<__half*>(sQ.data().get());
    //     for (int i = 0; i < size(sQ)/8; ++i) {
    //         printf("Q %d: ", i);
    //         for (int j = 0; j < 8; ++j) {
    //             printf("%.4f ", __half2float(tmp[i*8+j]));
    //         }
    //         printf("\n");
    //     }
    //     tmp = reinterpret_cast<__half*>(sK.data().get());
    //     for (int i = 0; i < size(sK)/8; ++i) {
    //         printf("K %d: ", i);
    //         for (int j = 0; j < 8; ++j) {
    //             printf("%.4f ", __half2float(tmp[i*8+j]));
    //         }
    //         printf("\n");
    //     }
    // }

    if constexpr (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_QdO.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_QdO, tSsQ, tSrQ_copy_view);
        
        Tensor tdPrdO_copy_view = smem_thr_copy_QdO.retile_D(tdPrdO);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsdO) == size<1>(tdPrdO_copy_view));
        cute::copy(smem_tiled_copy_QdO, tdPsdO, tdPrdO_copy_view);
    }
    __syncthreads();

    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    clear(acc_dq);

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s);
        // cute::cp_async_wait<0>();
        __syncthreads();

        Tensor dP_sum = make_fragment_like(lse);
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }
        // wangaq debug
        // if (blockIdx.x == 0) {
        //     // if (thread0()) {
        //     //     printf("lse.layout:"); print(lse.layout()); print("\n");
        //     // }
        //     // __syncthreads();
        //     printf("dP_sum tid:%d m_block:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, m_block,
        //     dP_sum[0], dP_sum[1], dP_sum[2], dP_sum[3], dP_sum[4], dP_sum[5], dP_sum[6], dP_sum[7]);
        // }
        
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs || Kernel_traits::Share_Q_K_smem>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
            smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     // if (thread0()) {
        //     //     // printf("tiled_mma_sdp:"); print(tiled_mma_sdp);
        //     //     printf("acc_s.layout:"); print(acc_s.layout()); printf("\n");
        //     // }
        //     // __syncthreads();
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("S tid:%d n_block:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, n_block,
        //         tmp[0], tmp[1], tmp[2], tmp[3],
        //         tmp[4], tmp[5], tmp[6], tmp[7]
        //     );
        // }
        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

        // Softcapping - calculating dTanh and scaling dS later with it
        Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }

        #if 1
        if (Has_alibi) {
            const int warp_id = tidx / 64;
            const int col_idx_offset = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = AtomLayoutMS * 16;
            alibi.apply_alibi(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }
        #endif

        #if 1
        if (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = tidx / 64;
                const int col_idx_offset_ = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
                flash::apply_mask(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if (Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = tidx / 64;
                flash::apply_mask_causal(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        AtomLayoutMS * 16);
            }
        } else if (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = tidx / 64;
                flash::apply_mask_local(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }
        #endif

        // Compute the exponential value.
        // wangaq debug
        // if (blockIdx.x == 0) {
        //     // if (thread0()) {
        //     //     printf("lse.layout:"); print(lse.layout()); print("\n");
        //     // }
        //     // __syncthreads();
        //     printf("lse tid:%d n_block:%d %.4f %.4f \n", tidx, n_block,
        //     lse[0], lse[1]);
        // }
        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
        
        // wangaq debug
        // if (blockIdx.x == 0) {
        //     // if (tidx == 0) print(acc_s.layout());
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("P tid:%d n_block:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, n_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7]
        //     );
        // }

        if constexpr (Is_dropout) {
            const int warp_id = tidx / 64;
            const int warp_row_stride = 16;
            int block_row_idx = m_block * (kBlockM / 16) + warp_id % AtomLayoutMS;
            int block_col_idx = n_block * (kBlockN / 16) + (warp_id / AtomLayoutMS) * MMA_N_SdP;
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS
            );
        }
        // Convert scores from fp32 to fp16/bf16
        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);
        
        Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        CUTE_STATIC_ASSERT_V(size<0>(acc_dp) == size<0>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<1>(acc_dp) == size<1>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<2>(acc_dp) == size<2>(acc_s));                     // MMA

        clear(acc_dp);

        // wangaq debug
        // __syncthreads();
        // if (cute::thread(0, 0)) {
        //     Element * tmp = reinterpret_cast<Element*>(sdO.data().get());
        //     for (int i = 0; i < size(sdO)/8; ++i) {
        //         printf("dO n_block:%d row:%d: ", n_block, i);
        //         for (int j = 0; j < 8; ++j) {
        //             printf("%.4f ", half_t::convert(tmp[i*8+j]));
        //         }
        //         printf("\n");
        //     }
        //     tmp = reinterpret_cast<Element*>(sV.data().get());
        //     for (int i = 0; i < size(sV)/8; ++i) {
        //         printf("V n_block:%d row:%d: ", n_block, i);
        //         for (int j = 0; j < 8; ++j) {
        //             printf("%.4f ", half_t::convert(tmp[i*8+j]));
        //         }
        //         printf("\n");
        //     }
        // }
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs || Kernel_traits::Share_Q_K_smem>(
            acc_dp, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp,
            smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV
        );
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_dp.data());
        //     printf("dP tid:%d n_block:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, n_block, 
        //     tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7]
        //     );
        // }

        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     float * tmp = reinterpret_cast<float*>(dS.data());
        //     printf("dS tid:%d n_block:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, n_block, 
        //     tmp[0], tmp[1], tmp[2], tmp[3],
        //     tmp[4], tmp[5], tmp[6], tmp[7]
        //     );
        // }

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        Tensor tdQrdS = flash::convert_type<Element>(dS_reshaped);
        flash::gemm_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, 
            smem_tiled_copy_Kt, smem_thr_copy_Kt);

        // wangaq debug
        // if (blockIdx.x == 0) {
        //     // if (tidx == 0) print(acc_dq.layout());
        //     __syncthreads();
        //     float * tmp = reinterpret_cast<float*>(acc_dq.data());
        //     printf("dQ tid:%d n_block:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
        //     "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
        //     "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
        //     "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, n_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15],
        //     tmp[16], tmp[17], tmp[18], tmp[19], 
        //     tmp[20], tmp[21], tmp[22], tmp[23],
        //     tmp[24], tmp[25], tmp[26], tmp[27], 
        //     tmp[28], tmp[29], tmp[30], tmp[31]
        //     );
        // }

        __syncthreads();
        if (n_block > n_block_min) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV); 
            // Advance gK/gV
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);       
            __syncthreads();
        }
    }
#if 0
    const index_t row_offset_dq_accum = binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb)
        + (m_block * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128 * bidb)) * params.h * params.d_rounded + bidh * params.d_rounded;
    
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.h * params.d_rounded, _1{}));
    
    typename Kernel_traits::GmemTiledCopydQaccum gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);
    
    cute::copy(gmem_tiled_copy_dQaccum, acc_dq, tdQgdQaccum);
#elif 1
    // Epilogue
        
    __builtin_amdgcn_sched_barrier(1);
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = flash::convert_type<Element>(acc_dq);

    Tensor sdQ = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutdQ{});
    // Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
    //                         typename Kernel_traits::SmemLayoutdQ{});

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);

    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    __syncthreads();

    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    // __builtin_amdgcn_sched_barrier(0);
    // __syncthreads();
    Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
    cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);
    Tensor cdQ = make_identity_tensor(make_shape(size<0>(sdQ), size<1>(sdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_first, bool Is_last, bool Seq_parallel=false, typename Params>
inline __device__ void compute_dk_dv_trans_1colblock(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    // wangaq debug
    // constexpr int MMA_N_SdP = kBlockN / decltype(typename Kernel_traits::TiledMmaSdP{}.template tile_size_mnk<1>())::value;
    constexpr int MMA_N_SdP = kBlockN / size<1>(typename Kernel_traits::TiledMmaSdP::TiledShape_MNK{});
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);
    if (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    const index_t row_offset_dq_accum = binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb)
        + ((m_block_max - 1) * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128 * bidb)) * params.h * params.d_rounded + bidh * params.d_rounded
        // If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
        + (!params.deterministic ? 0 : blockIdx.x * params.dq_accum_split_stride);
    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;

    // if (cute::thread(0, 0)) {
    //     Element * tmp = reinterpret_cast<Element*>(params.q_ptr) + row_offset_q;
    //     printf("row_offset_q:%d %.4f\n", row_offset_q, half_t::convert(tmp[7]));
    //     // for (int i = 0; i < kBlockM*kHeadDim/8; ++i) {
    //     //     printf("Q %d: ", i);
    //     //     for (int j = 0; j < 8; ++j) {
    //     //         printf("%.4f ", half_t::convert(tmp[i*8+j]));
    //     //     }
    //     //     printf("\n");
    //     // }
    // }

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.h * params.d_rounded, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    // Double buffer for sQ
    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(),
                typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});

    Tensor sK = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});

    // sP and sdQ share the same memory so be careful
    Tensor sdQ = make_tensor(sV.data() + size(sV), typename Kernel_traits::SmemLayoutdQ{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    using GmemTiledCopydO = std::conditional_t<
        Is_first,
        typename Kernel_traits::GmemTiledCopydO,
        typename Kernel_traits::GmemTiledCopyQKV
    >;
    GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    using GmemLayoutAtomdQaccum = std::conditional_t<
        !Seq_parallel,
        typename Kernel_traits::GmemTiledCopydQaccum,
        typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd
    >;
    GmemLayoutAtomdQaccum gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrK = thr_mma_sdp.partition_fragment_A(sK);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_B(sQ);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_A(sV);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_B(sdO);

    // dV/dK
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);

    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    //
    // Copy Atom retiling
    //

    // S/dP
    auto smem_tiled_copy_KV = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    auto smem_tiled_copy_QdO = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    // dV/dK
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // We'll advance gdQ and gdQaccum before the 1st read/write.
    tdQgdQ.data() = tdQgdQ.data() + kBlockM * params.dq_row_stride;
    tdQgdQaccum.data() = tdQgdQaccum.data() + kBlockM * params.h * params.d_rounded;

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);
    // If not local, we're guaranteed that m_block_min <= m_block:
    // We checked earlier that n_block * kBlockN < actual_seqlen_k, so in the causal case,
    // n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k < actual_seqlen_q.
    // So m_block_min <= (actual_seqlen_q - 1) / kBlockM.
    // Recall that m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM) = (actual_seqlen_q + kBlockM - 1) / kBlockM.
    // So m_block_m - 1 = (actual_seqlen_q - 1) / kBlockM.
    // We conclude that m_block_min <= m_block, so we will always have at least 1 iteration of the for loop.
    // However, if local, then this possible to have some blocks of K & V not attending to any query.
    // We might need to exit early and write 0 to dK and dV for those blocks.
    // Otherwise we get wrong result for the case where we don't enter the for loop.
    // And we might read OOB elements from gQ and gdO.
    // This also covers the case where actual_seqlen_q == 0
    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdKV = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }

    if (Double_buffer && m_block % 2 == 1) {  // Double buffer for sQ
        tQsQ.data() = tQsQ.data() + size(sQ);
        tSsQ.data() = tSsQ.data() + size(sQ);
    }

    if ((!Is_first && !Seq_parallel) || params.deterministic) { __syncthreads(); }

    if constexpr(Kernel_traits::Is_V_in_regs) {
        // Clear the smem tiles to account for predicated off loads
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        // flash::cp_async_fence();
    }

    Tensor tdOrdO = make_fragment_like(tdOgdO);
    Tensor tdOrO = make_fragment_like(tdOgO);
    if constexpr(!Is_first) {
        // Clear the smem tiles to account for predicated off loads
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
    } else {
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgdO, tdOrdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgO, tdOrO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
    // wangaq debug
    // __syncthreads();
    // if (cute::thread(0, 0)) {
    //     __half * tmp = reinterpret_cast<__half*>(sQ.data().get());
    //     for (int i = 0; i < size(sQ)/8; ++i) {
    //         printf("Q %d: ", i);
    //         for (int j = 0; j < 8; ++j) {
    //             printf("%.4f ", __half2float(tmp[i*8+j]));
    //         }
    //         printf("\n");
    //     }
    // }

    // wangaq debug
    Tensor caccS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    static_assert(decltype(size<0>(taccScS))::value == 4);
    // wangaq debug
    Tensor taccScS_row = taccScS(_, 0, _);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = laneId / 16 + mi * 4;
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    // We want LSE = inf if the row is OOB. In that case Q would be zero, K would be zero,
    // and scores would be zero. With LSE = 0, probs will be all 1's, and when we multiply
    // with V (which would be zero), we're fine. However, with ALiBi, we might modify these
    // scores, and probs can become NaN. Instead if we set LSE = inf for OOB rows, probs are always 0.

    // Tensor tKrK = make_fragment_like(tKsK);
    // // cute::copy(gmem_tiled_copy_QKV, tKgK(_, _, _, 0), tKrK);
    // cute::copy(gmem_tiled_copy_QKV, tKgK, tKrK);
    // // if (cute::thread(1, 0)) { print(tKrK); }

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    if constexpr(!Kernel_traits::Is_V_in_regs) {
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
    }
    // flash::cp_async_fence();

    // wangaq debug
    // __syncthreads();
    // if (cute::thread(0, 0)) {
    //     __half * tmp = reinterpret_cast<__half*>(sQ.data().get());
    //     for (int i = 0; i < size(sQ)/8; ++i) {
    //         printf("Q %d: ", i);
    //         for (int j = 0; j < 8; ++j) {
    //             printf("%.4f ", __half2float(tmp[i*8+j]));
    //         }
    //         printf("\n");
    //     }
    //     tmp = reinterpret_cast<__half*>(sK.data().get());
    //     for (int i = 0; i < size(sK)/8; ++i) {
    //         printf("K %d: ", i);
    //         for (int j = 0; j < 8; ++j) {
    //             printf("%.4f ", __half2float(tmp[i*8+j]));
    //         }
    //         printf("\n");
    //     }
    // }

    if constexpr(Kernel_traits::Is_V_in_regs) {
        // cute::cp_async_wait<1>();
        __syncthreads();
        Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));            // M
        cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
    }

    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    clear(acc_dv);
    clear(acc_dk);

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    for (; m_block >= m_block_min; --m_block) {
#if 1
        // wangaq debug
        // Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s);
        cute::cp_async_wait<0>();
        __syncthreads();

        Tensor dP_sum = make_fragment_like(lse);
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = laneId / 16 + mi * 4;
            dP_sum(mi) = gdPsum(row);
        }
        // wangaq debug
        // if (blockIdx.x == 0) {
        //     // if (thread0()) {
        //     //     printf("lse.layout:"); print(lse.layout()); print("\n");
        //     // }
        //     // __syncthreads();
        //     printf("dP_sum tid:%d m_block:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, m_block,
        //     dP_sum[0], dP_sum[1], dP_sum[2], dP_sum[3], dP_sum[4], dP_sum[5], dP_sum[6], dP_sum[7]);
        // }

        flash::gemm(acc_s, tSrK, tSrQ, tSsK, tSsQ, tiled_mma_sdp,
                    smem_tiled_copy_KV, smem_tiled_copy_QdO, smem_thr_copy_KV, smem_thr_copy_QdO);


        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     // if (thread0()) {
        //     //     // printf("tiled_mma_sdp:"); print(tiled_mma_sdp);
        //     //     printf("acc_s.layout:"); print(acc_s.layout()); printf("\n");
        //     // }
        //     // __syncthreads();
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("S tid:%d m_block:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, m_block,
        //         tmp[0], tmp[1], tmp[2], tmp[3],
        //         tmp[4], tmp[5], tmp[6], tmp[7]
        //     );
        // }

        Tensor scores = make_tensor(acc_s.data(), flash::convert_trans_layout_acc_rowcol(acc_s.layout()));

        #if 0
        if (Has_alibi) {
            const int warp_id = tidx / 64;
            const int col_idx_offset = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = AtomLayoutMS * 16;
            alibi.apply_alibi(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }
        #endif

        // TD [2023-07-29]: I was thinking that we don't need to mask out the elements beyond
        // actual_seqlen_k, because acc_s would be some finite value for those indices.
        // In the end when we multiply with K to get dQ, the corresponding values of K would be 0,
        // so the result would still be correct.
        // However, it's possible that the values in acc_s are so large that they overflow
        // when we multiply with dP and convert to fp16, resulting in Inf in dS and NaNs in dQ.
        // So we need to mask out the elements beyond actual_seqlen_k.
        #if 0
        if (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = tidx / 64;
                const int col_idx_offset_ = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
                flash::apply_mask(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if (Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = tidx / 64;
                flash::apply_mask_causal(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        AtomLayoutMS * 16);
            }
        } else if (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = tidx / 64;
                flash::apply_mask_local(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }
        #endif

        // Compute the exponential value.
        // wangaq debug
        // if (blockIdx.x == 0) {
        //     // if (thread0()) {
        //     //     printf("lse.layout:"); print(lse.layout()); print("\n");
        //     // }
        //     // __syncthreads();
        //     printf("lse tid:%d m_block:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, m_block,
        //     lse[0], lse[1], lse[2], lse[3], lse[4], lse[5], lse[6], lse[7]);
        // }
        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);

        // wangaq debug
        // if (blockIdx.x == 0) {
        //     // if (tidx == 0) print(acc_s.layout());
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("P tid:%d m_block:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7]
        //     );
        // }
        if constexpr (Is_dropout) {
            const int warp_id = tidx / 64;
            const int warp_row_stride = 16;
            int block_row_idx = m_block * (kBlockM / 16) + warp_id % AtomLayoutMS;
            int block_col_idx = n_block * (kBlockN / 16) + (warp_id / AtomLayoutMS) * MMA_N_SdP;
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS
            );
        }
        // Convert scores from fp32 to fp16/bf16
        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);

        Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{});  // (MMA=4, MMA_N, MMA_N)
        CUTE_STATIC_ASSERT_V(size<0>(acc_dp) == size<0>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<1>(acc_dp) == size<1>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<2>(acc_dp) == size<2>(acc_s));                     // MMA

        clear(acc_dp);

        // wangaq debug
        // __syncthreads();
        // if (cute::thread(0, 0)) {
        //     Element * tmp = reinterpret_cast<Element*>(sdO.data().get());
        //     for (int i = 0; i < size(sdO)/8; ++i) {
        //         printf("dO m_block:%d row:%d: ", m_block, i);
        //         for (int j = 0; j < 8; ++j) {
        //             printf("%.4f ", half_t::convert(tmp[i*8+j]));
        //         }
        //         printf("\n");
        //     }
        //     tmp = reinterpret_cast<Element*>(sV.data().get());
        //     for (int i = 0; i < size(sV)/8; ++i) {
        //         printf("V m_block:%d row:%d: ", m_block, i);
        //         for (int j = 0; j < 8; ++j) {
        //             printf("%.4f ", half_t::convert(tmp[i*8+j]));
        //         }
        //         printf("\n");
        //     }
        // }
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_V_in_regs, /*B_in_regs=*/false>(
            acc_dp, tdPrV, tdPrdO, tdPsV, tdPsdO, tiled_mma_sdp,
            smem_tiled_copy_KV, smem_tiled_copy_QdO, smem_thr_copy_KV, smem_thr_copy_QdO
        );
        // flash::gemm_rs(acc_dp, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, 
        //     smem_tiled_copy_QdO, smem_thr_copy_QdO);
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_dp.data());
        //     printf("dP tid:%d m_block:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, m_block, 
        //     tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7]
        //     );
        // }

        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                dS(mi, ni) = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
            }
        }
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     float * tmp = reinterpret_cast<float*>(dS.data());
        //     printf("dS m_block:%d tid:%d: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", m_block, tidx, 
        //     tmp[0], tmp[1], tmp[2], tmp[3],
        //     tmp[4], tmp[5], tmp[6], tmp[7]
        //     );
        //     __syncthreads();
        // }

        // wangaq debug
        // if (blockIdx.x == 0) {
        //     // if (tidx == 0) print(acc_s.layout());
        //     Element * tmp = reinterpret_cast<Element*>(tdVsdOt.data().get());
        //     printf("dOt tid:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, 
        //     half_t::convert(tmp[0]), half_t::convert(tmp[1]), half_t::convert(tmp[2]), half_t::convert(tmp[3]), 
        //     half_t::convert(tmp[4]), half_t::convert(tmp[5]), half_t::convert(tmp[6]), half_t::convert(tmp[7])
        //     );
        // }

        flash::gemm_rs(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt,
                smem_thr_copy_QdOt);

        // __syncthreads(); // Need syncthreads since we're writing to the same sdO location

        // wangaq debug
        // if (blockIdx.x == 0) {
        //     // if (tidx == 0) print(acc_dv.layout());
        //     __syncthreads();
        //     float * tmp = reinterpret_cast<float*>(acc_dv.data());
        //     printf("dV tid:%d m_block:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
        //     "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
        //     "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
        //     "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15],
        //     tmp[16], tmp[17], tmp[18], tmp[19], 
        //     tmp[20], tmp[21], tmp[22], tmp[23],
        //     tmp[24], tmp[25], tmp[26], tmp[27], 
        //     tmp[28], tmp[29], tmp[30], tmp[31]
        //     );
        // }

        if (m_block > m_block_min) {
            // Advance gdO
            tdOgdO.data() = tdOgdO.data() + (-int(kBlockM * params.do_row_stride));
            if (Is_first) {
                tdOgO.data() = tdOgO.data() + (-int(kBlockM * params.o_row_stride));
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_dO, tdOgdO, tdOrdO, tQcQ, tQpQ);
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_dO, tdOgO, tdOrO, tQcQ, tQpQ);
            } else {
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ);
                flash::cp_async_fence();
            }
        }

        if (m_block > m_block_min) {
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = laneId / 16 + mi * 4;
                lse(mi) = gLSE(row);
            }
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
        }

        Tensor tdKrdSt = flash::convert_type<Element>(acc_dp);
        flash::gemm_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt,
                smem_thr_copy_QdOt);

        // wangaq debug
        // if (blockIdx.x == 0) {
        //     // if (tidx == 0) print(acc_dv.layout());
        //     __syncthreads();
        //     float * tmp = reinterpret_cast<float*>(acc_dk.data());
        //     printf("dK tid:%d m_block:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
        //     "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
        //     "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
        //     "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15],
        //     tmp[16], tmp[17], tmp[18], tmp[19], 
        //     tmp[20], tmp[21], tmp[22], tmp[23],
        //     tmp[24], tmp[25], tmp[26], tmp[27], 
        //     tmp[28], tmp[29], tmp[30], tmp[31]
        //     );
        // }
        if constexpr(!Double_buffer) {
            if (m_block > m_block_min) {
                __syncthreads();
                // Advance gQ
                tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
                flash::cp_async_fence();
            }
        }
#endif
    }
    // Epilogue
#if 0
    if constexpr(Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    

        // wangaq debug
        // if (blockIdx.x == 0) {
        //     // if (tidx == 0) print(acc_dv.layout());
        //     float * tmp = reinterpret_cast<float*>(acc_dq.data());
        //     printf("dQ tid:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, 
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7]
        //     );
        // }

    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dv_row_stride, _1{}));
    _bwd_store_dk_dv<Kernel_traits, decltype(acc_dv), decltype(sV), decltype(gdV), Element, Is_even_MN, Is_even_K>(
        acc_dv, sV, tidx, gdV, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

    
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }
    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    _bwd_store_dk_dv<Kernel_traits, decltype(acc_dv), decltype(sV), decltype(gdV), Element, Is_even_MN, Is_even_K>(
        acc_dk, sK, tidx, gdK, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
#elif 1
    const index_t row_offset_dk_accum = binfo.k_offset(params.seqlen_k_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb)
        + (n_block * kBlockN + (params.cu_seqlens_k == nullptr ? 0 : 128 * bidb)) * params.h * params.d_rounded + bidh * params.d_rounded;

    Tensor gdKaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dk_accum_ptr) + row_offset_dk_accum),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.h * params.d_rounded, _1{}));
    Tensor gdVaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dv_accum_ptr) + row_offset_dk_accum),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.h * params.d_rounded, _1{}));

    typename Kernel_traits::GmemTiledCopydKVaccum gmem_tiled_copy_dKVaccum;
    auto gmem_thr_copy_dKVaccum = gmem_tiled_copy_dKVaccum.get_thread_slice(tidx);
    Tensor tdKgdKaccum = gmem_thr_copy_dKVaccum.partition_D(gdKaccum);
    Tensor tdVgdVaccum = gmem_thr_copy_dKVaccum.partition_D(gdVaccum);

    cute::copy(gmem_tiled_copy_dKVaccum, acc_dk, tdKgdKaccum);
    cute::copy(gmem_tiled_copy_dKVaccum, acc_dv, tdVgdVaccum);

#elif 0
    __builtin_amdgcn_sched_barrier(1);
    if constexpr(Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }
    
    // wangaq debug
    // if (blockIdx.x == 0) {
    //     __syncthreads();
    //     float * tmp = reinterpret_cast<float*>(acc_dk.data());
    //     printf("dK tid:%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
    //     "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
    //     "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
    //     "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", tidx, 
    //     tmp[0], tmp[1], tmp[2], tmp[3], 
    //     tmp[4], tmp[5], tmp[6], tmp[7],
    //     tmp[8], tmp[9], tmp[10], tmp[11], 
    //     tmp[12], tmp[13], tmp[14], tmp[15],
    //     tmp[16], tmp[17], tmp[18], tmp[19], 
    //     tmp[20], tmp[21], tmp[22], tmp[23],
    //     tmp[24], tmp[25], tmp[26], tmp[27], 
    //     tmp[28], tmp[29], tmp[30], tmp[31]
    //     );
    // }

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dv_row_stride, _1{}));
    // int row_offset, col_offset;
    int row, col;
    Element data;
    float fdata;
    for (int mi = 0; mi < size<1>(acc_dk); ++mi) {
        // row_offset = mi * 16 * params.dk_row_stride;
        row = (mi + warpId) * 16 + (laneId % 16);
        for (int ni = 0; ni < size<2>(acc_dk); ++ni) {
            col = (laneId / 16) + ni * 16;
            for (int ei = 0; ei < size<0>(acc_dk); ++ei) {
                // *(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk + row_offset + col_offset) = half_t::convert(acc_dk(mi, ni));
                // wangaq debug
                // if(thread(0, 0)) {
                //     printf("mi:%d ni:%d ei:%d row:%d col:%d dk:%.4f\n", 
                //     mi, ni, ei, row, col, acc_dk(ei, mi, ni));
                // }
                // gdK(row, col) = __float2half(acc_dk(ei, mi, ni));
                // gdV(row, col) = __float2half(acc_dv(ei, mi, ni));
                fdata = acc_dk(ei, mi, ni);
                asm volatile("v_cvt_f16_f32 %0, %1 \n"
                :  "=v"(data) : "v"(fdata));
                gdK(row, col) = data;
                fdata = acc_dv(ei, mi, ni);
                asm volatile("v_cvt_f16_f32 %0, %1 \n"
                :  "=v"(data) : "v"(fdata));
                gdV(row, col) = data;
                col += 4;
            }
        }
    } 
#elif 0
    // Epilogue

    __builtin_amdgcn_sched_barrier(1);
    if constexpr(Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dv from fp32 to fp16
    Tensor rdK = flash::convert_type<Element>(acc_dk);
    Tensor rdV = flash::convert_type<Element>(acc_dv);

    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    // If Is_last, there's already a __syncthreads() at the end of the loop.
    if constexpr(!Is_last) { __syncthreads(); }

    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Has_alibi, bool Is_even_M, bool Is_even_K, typename Params>
inline __device__ void compute_dq_dk_dv(const Params &params) {

    // The block index for the batch.
    const int bidb = blockIdx.x;
    // const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.y;
    // const int bidh = blockIdx.z;
    // The thread index.
    const int tidx = threadIdx.x;

    const int n_block_max = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    if (n_block_max == 1) {
        compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, true, true>(params, bidb, bidh, 0);
    } else {
        // Iterating backward from n_block_max - 1 to 0 might save 1 register
        compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, true, false>(params, bidb, bidh, n_block_max - 1);
        for (int n_block = n_block_max - 2; n_block > 0; n_block--) {
            compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, false, false>(params, bidb, bidh, n_block);
        }
        compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, false, true>(params, bidb, bidh, 0);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dq_dk_dv_seqk_parallel(const Params &params) {
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
    for (int n_block = blockIdx.x; n_block < (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN; n_block += gridDim.x) {
        compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap,
            /*Is_first*/false, /*Is_last*/false, /*Seq_parallel=*/true>(params, bidb, bidh, n_block);
    }
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_dv_trans_seqk_parallel(const Params &params) {
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
    for (int n_block = blockIdx.x; n_block < (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN; n_block += gridDim.x) {
        compute_dk_dv_trans_1colblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap,
            /*Is_first*/false, /*Is_last*/false, /*Seq_parallel=*/true>(params, bidb, bidh, n_block);
    }
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_dv_seqk_parallel(const Params &params) {
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
    for (int n_block = blockIdx.x; n_block < (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN; n_block += gridDim.x) {
        compute_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap,
            /*Is_first*/false, /*Is_last*/false, /*Seq_parallel=*/true>(params, bidb, bidh, n_block);
    }
}
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dq_seqq_parallel(const Params &params) {
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
    for (int m_block = blockIdx.x; m_block < (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM; m_block += gridDim.x) {
        compute_dq_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap,
            /*Is_first*/false, /*Is_last*/false, /*Seq_parallel=*/true>(params, bidb, bidh, m_block);
    }
}



template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dq_1rowblock_16x64(const Params &params, const int bidb, const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;


    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    using SdP_TiledShape_MNK = typename Kernel_traits::TiledMmaSdP::TiledShape_MNK;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(SdP_TiledShape_MNK{}))::value;;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;


    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;



    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + m_block * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + m_block * kBlockM;
    

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});


    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    
    Tensor sdO = make_tensor(sK.data() , typename Kernel_traits::SmemLayoutQdO{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);


    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);


    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);


    // dQ
    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    //
    // Copy Atom retiling
    //

    // S/dP
    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    auto tRgQ = smem_thr_copy_QdO.partition_S(gQ);
    auto tRgdO = smem_thr_copy_QdO.partition_S(gdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);
#if 0
    // dQ
    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
#else
    auto smem_tiled_copy_Kt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dq);
#endif
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

    //
    // PREDICATES
    //
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = smem_thr_copy_QdO.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue
    
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
        
        Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.dq_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
        Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
        clear(tdQrdQ);
        Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
        Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
        if constexpr(!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        return;
    }
#if 0
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    //  __syncthreads();
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    //  __syncthreads();
    if constexpr (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_QdO.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_QdO, tSsQ, tSrQ_copy_view);
        
        Tensor tdPrdO_copy_view = smem_thr_copy_QdO.retile_D(tdPrdO);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsdO) == size<1>(tdPrdO_copy_view));
        cute::copy(smem_tiled_copy_QdO, tdPsdO, tdPrdO_copy_view);
    }
    __syncthreads();
#else
    Tensor tSrQ_copy_view = smem_thr_copy_QdO.retile_D(tSrQ);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        smem_tiled_copy_QdO, tRgQ, tSrQ_copy_view, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    Tensor tdPrdO_copy_view = smem_thr_copy_QdO.retile_D(tdPrdO);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        smem_tiled_copy_QdO, tRgdO, tdPrdO_copy_view, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

#endif
    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    // static_assert(decltype(size<0>(taccScS))::value == 4);
    Tensor taccScS_row = taccScS(0, _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    Tensor dP_sum = make_fragment_like(lse);
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

    //  __syncthreads();
    int n_block = n_block_max - 1;
    auto tKrK = make_fragment_like(tKsK);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKrK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );


    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    clear(acc_dq);
    // __syncthreads();
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s_ori);
        // cute::cp_async_wait<0>();
        cute::copy(tKrK, tKsK);
        __syncthreads();
        auto tVrV = make_fragment_like(tVsV);
        if (n_block == n_block_max - 1) {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV, tVrV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        else {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVrV, tKVcKV, tKVpKV); 
        }
        
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs || Kernel_traits::Share_Q_K_smem>(
            acc_s_ori, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
            smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);
        Tensor acc_s = make_tensor(acc_s_ori.data(), flash::convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        //  __syncthreads();
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        //  __syncthreads();
        // Softcapping - calculating dTanh and scaling dS later with it
        Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }
        //  __syncthreads();
        #if 1
        //  __syncthreads();
        if constexpr (Has_alibi) {
            const int warp_id = tidx / 64;
            const int col_idx_offset = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = AtomLayoutMS * 16;
            alibi.apply_alibi_continuous(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }
        //  __syncthreads();
        #endif
        //   __syncthreads();
        #if 1
        if constexpr (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = tidx / 64;
                const int col_idx_offset_ = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
                flash::apply_mask_continuous(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if constexpr (Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = tidx / 64;
                flash::apply_mask_causal_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        AtomLayoutMS * 16);
            }
        } else if constexpr (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = tidx / 64;
                flash::apply_mask_local_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }
        #endif

        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);

        #if 1
        if constexpr (Is_dropout) {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS * 16
            );
        }
        #endif
        //  __syncthreads();
        // Convert scores from fp32 to fp16/bf16
        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);
        //  __syncthreads();
        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);
        // __syncthreads();
        cute::copy(tVrV, tVsV);
        __syncthreads();
        
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs || Kernel_traits::Share_Q_K_smem>(
            acc_dp_ori, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp,
            smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV
        );
        //  __syncthreads();
        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));


        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKrK, tKVcKV, tKVpKV);       
            // __syncthreads();
        }

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        Tensor tdQrdS = flash::convert_type<Element>(dS_reshaped);
        //  __syncthreads();
        flash::gemm_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, 
            smem_tiled_copy_Kt, smem_thr_copy_Kt);
        __syncthreads();

    }
    //  __syncthreads();
    __builtin_amdgcn_sched_barrier(1);
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = flash::convert_type<Element>(acc_dq);

    Tensor sdQ = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutdQ{});
    // Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
    //                         typename Kernel_traits::SmemLayoutdQ{});

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);
    //  __syncthreads();
    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    __syncthreads();

    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    //  __syncthreads();
    // __builtin_amdgcn_sched_barrier(0);
    // __syncthreads();
    Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
    cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);
    Tensor cdQ = make_identity_tensor(make_shape(size<0>(sdQ), size<1>(sdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    }
    __syncthreads();
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
    __syncthreads();
}
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dq_1rowblock_16x64_dim40(const Params &params, const int bidb, const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;


    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    using SdP_TiledShape_MNK = typename Kernel_traits::TiledMmaSdP::TiledShape_MNK;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(SdP_TiledShape_MNK{}))::value;;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;


    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;



    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + m_block * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + m_block * kBlockM;
    

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});


    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    
    Tensor sdO = make_tensor(sK.data() , typename Kernel_traits::SmemLayoutQdO{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);


    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);


    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);


    // dQ
    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    // 8, 1, 2
    //
    // Copy Atom retiling
    //

    // S/dP
    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    auto tRgQ = smem_thr_copy_QdO.partition_S(gQ);
    auto tRgdO = smem_thr_copy_QdO.partition_S(gdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);
#if 0
    // dQ
    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
#else
    auto smem_tiled_copy_Kt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dq);
#endif
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

    //
    // PREDICATES
    //
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = smem_thr_copy_QdO.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue
    
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
        
        Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.dq_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
        Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
        clear(tdQrdQ);
        Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
        Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
        if constexpr(!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        return;
    }
#if 0
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    //  __syncthreads();
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    //  __syncthreads();
    if constexpr (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_QdO.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_QdO, tSsQ, tSrQ_copy_view);
        
        Tensor tdPrdO_copy_view = smem_thr_copy_QdO.retile_D(tdPrdO);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsdO) == size<1>(tdPrdO_copy_view));
        cute::copy(smem_tiled_copy_QdO, tdPsdO, tdPrdO_copy_view);
    }
    __syncthreads();
#else
    Tensor trsQ = smem_thr_copy_QdO.partition_D(gQ);
    Tensor trpQ = make_tensor<bool>(make_shape(size<2>(trsQ)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(trpQ); ++k) { trpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }
    Tensor tSrQ_copy_view = smem_thr_copy_QdO.retile_D(tSrQ);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        smem_tiled_copy_QdO, tRgQ, tSrQ_copy_view, tQcQ, trpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    Tensor tdPrdO_copy_view = smem_thr_copy_QdO.retile_D(tdPrdO);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        smem_tiled_copy_QdO, tRgdO, tdPrdO_copy_view, tQcQ, trpQ, binfo.actual_seqlen_q - m_block * kBlockM);

#endif
    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    // static_assert(decltype(size<0>(taccScS))::value == 4);
    Tensor taccScS_row = taccScS(0, _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    Tensor dP_sum = make_fragment_like(lse);
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

    //  __syncthreads();
    int n_block = n_block_max - 1;
    auto tKrK = make_fragment_like(tKsK);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKrK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );


    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    clear(acc_dq);
    // __syncthreads();
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s_ori);
        // cute::cp_async_wait<0>();
        cute::copy(tKrK, tKsK);
        __syncthreads();
        auto tVrV = make_fragment_like(tVsV);
        if (n_block == n_block_max - 1) {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV, tVrV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        else {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVrV, tKVcKV, tKVpKV); 
        }
        
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs || Kernel_traits::Share_Q_K_smem>(
            acc_s_ori, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
            smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);
        Tensor acc_s = make_tensor(acc_s_ori.data(), flash::convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        //  __syncthreads();
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        //  __syncthreads();
        // Softcapping - calculating dTanh and scaling dS later with it
        Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }
        //  __syncthreads();
        #if 1
        //  __syncthreads();
        if constexpr (Has_alibi) {
            const int warp_id = tidx / 64;
            const int col_idx_offset = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = AtomLayoutMS * 16;
            alibi.apply_alibi_continuous(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }
        //  __syncthreads();
        #endif
        //   __syncthreads();
        #if 1
        if constexpr (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = tidx / 64;
                const int col_idx_offset_ = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
                flash::apply_mask_continuous(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if constexpr (Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = tidx / 64;
                flash::apply_mask_causal_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        AtomLayoutMS * 16);
            }
        } else if constexpr (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = tidx / 64;
                flash::apply_mask_local_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }
        #endif

        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);

        #if 1
        if constexpr (Is_dropout) {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS * 16
            );
        }
        #endif
        //  __syncthreads();
        // Convert scores from fp32 to fp16/bf16
        // Tensor rP = !Is_dropout
        //     ? flash::convert_type<Element>(acc_s)
        //     : flash::convert_type_relu<Element>(acc_s);
        //  __syncthreads();
        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);
        // __syncthreads();
        cute::copy(tVrV, tVsV);
        __syncthreads();
        
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs || Kernel_traits::Share_Q_K_smem>(
            acc_dp_ori, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp,
            smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV
        );
        //  __syncthreads();
        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));


        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        if (n_block > n_block_min) {
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKrK, tKVcKV, tKVpKV);       
            // __syncthreads();
        }

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        Tensor tdQrdS = flash::convert_type<Element>(dS_reshaped);
        //  __syncthreads();
        flash::gemm_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, 
            smem_tiled_copy_Kt, smem_thr_copy_Kt);
        __syncthreads();

    }
    //  __syncthreads();
    __builtin_amdgcn_sched_barrier(1);
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = flash::convert_type<Element>(acc_dq);
#if 1

    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    //  __syncthreads();
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));


    using GmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    auto gmem_tiled_copy_dQ = make_tiled_copy_C(GmemCopyAtom{}, tiled_mma_dq);
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    Tensor taccdQrdQ = gmem_thr_copy_dQ.retile_S(rdQ);
    Tensor taccdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(taccdQgdQ)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; 
        }
    }
    //  __syncthreads();
    for (int m = 0; m < size<1>(taccdQrdQ); m++)
    {
        if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM)
        {
            for (int k = 0; k < size<2>(taccdQrdQ); k++)
            {
                
                {

                    const int col_id = get<1>(tdQcdQ(0, 0, k));
                    
                    //  __syncthreads();
                    for (int i = 0; i < size<0>(taccdQrdQ); i++)
                    {
                    
                    if (Is_even_K ||col_id + i * 4 < params.d) {

                        taccdQgdQ(i, m, k) = taccdQrdQ(i, m, k);
                    }
                        
                    }

                }
            }
        }
    }
    // __syncthreads();

    

#else
    Tensor sdQ = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutdQ{});
    // Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
    //                         typename Kernel_traits::SmemLayoutdQ{});

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);
    __syncthreads();
    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    __syncthreads();

    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    //  __syncthreads();
    // __builtin_amdgcn_sched_barrier(0);
    // __syncthreads();
    Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
    cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);
    Tensor cdQ = make_identity_tensor(make_shape(size<0>(sdQ), size<1>(sdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    }
    __syncthreads();
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
    __syncthreads();
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dq_1rowblock_16x64_dim128(const Params &params, const int bidb, const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;


    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    using SdP_TiledShape_MNK = typename Kernel_traits::TiledMmaSdP::TiledShape_MNK;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(SdP_TiledShape_MNK{}))::value;;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;


    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;



    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + m_block * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + m_block * kBlockM;
    

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});


    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    
    Tensor sdO = make_tensor(sK.data() , typename Kernel_traits::SmemLayoutQdO{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);


    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);


    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);


    // dQ
    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    //
    // Copy Atom retiling
    //

    // S/dP
    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    auto tRgQ = smem_thr_copy_QdO.partition_S(gQ);
    auto tRgdO = smem_thr_copy_QdO.partition_S(gdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);
#if 0
    // dQ
    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
#else
    auto smem_tiled_copy_Kt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dq);
#endif
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

    //
    // PREDICATES
    //
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = smem_thr_copy_QdO.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue
    
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
        
        Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.dq_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
        Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
        clear(tdQrdQ);
        Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
        Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
        if constexpr(!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        return;
    }
#if 0
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    //  __syncthreads();
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);
    //  __syncthreads();
    if constexpr (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_QdO.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_QdO, tSsQ, tSrQ_copy_view);
        
        Tensor tdPrdO_copy_view = smem_thr_copy_QdO.retile_D(tdPrdO);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsdO) == size<1>(tdPrdO_copy_view));
        cute::copy(smem_tiled_copy_QdO, tdPsdO, tdPrdO_copy_view);
    }
    __syncthreads();
#else
    Tensor tSrQ_copy_view = smem_thr_copy_QdO.retile_D(tSrQ);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        smem_tiled_copy_QdO, tRgQ, tSrQ_copy_view, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    Tensor tdPrdO_copy_view = smem_thr_copy_QdO.retile_D(tdPrdO);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        smem_tiled_copy_QdO, tRgdO, tdPrdO_copy_view, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

#endif
    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    // static_assert(decltype(size<0>(taccScS))::value == 4);
    Tensor taccScS_row = taccScS(0, _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    Tensor dP_sum = make_fragment_like(lse);
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

    //  __syncthreads();
    int n_block = n_block_max - 1;
    auto tKrK = make_fragment_like(tKsK);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKrK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );


    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    clear(acc_dq);
    // __syncthreads();
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sVtemp = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor tSsVBLayout = smem_thr_copy_BLayout.partition_S(sVtemp);
    Tensor tSsVdp = make_tensor(tSsVBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDim/32>(tSsVBLayout.layout()));

    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s_ori);
        // cute::cp_async_wait<0>();
        cute::copy(tKrK, tKsK);
        __syncthreads();
        
#if 0
        auto tVrV = make_fragment_like(tVsV);
        if (n_block == n_block_max - 1) 
        {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV, tVrV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        else 
        {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVrV, tKVcKV, tKVpKV); 
        }
#else

        // if (n_block == n_block_max - 1) 
        // {
            
        // }
        // else 
        // {

        // }

        #pragma unroll
        for (int i = 0; i < 4; i++)
        {
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gV, sV, i, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }



#endif
        
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs || Kernel_traits::Share_Q_K_smem>(
            acc_s_ori, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
            smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);
        Tensor acc_s = make_tensor(acc_s_ori.data(), flash::convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        //  __syncthreads();
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        //  __syncthreads();
        // Softcapping - calculating dTanh and scaling dS later with it
        Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }
        //  __syncthreads();
        #if 1
        //  __syncthreads();
        if constexpr (Has_alibi) {
            const int warp_id = tidx / 64;
            const int col_idx_offset = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = AtomLayoutMS * 16;
            alibi.apply_alibi_continuous(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }
        //  __syncthreads();
        #endif
        //   __syncthreads();
        #if 1
        if constexpr (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = tidx / 64;
                const int col_idx_offset_ = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
                flash::apply_mask_continuous(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if constexpr (Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = tidx / 64;
                flash::apply_mask_causal_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        AtomLayoutMS * 16);
            }
        } else if constexpr (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = tidx / 64;
                flash::apply_mask_local_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }
        #endif

        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);

        #if 1
        if constexpr (Is_dropout) {
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS * 16
            );
        }
        #endif
        //  __syncthreads();
        // Convert scores from fp32 to fp16/bf16
        // Tensor rP = !Is_dropout
        //     ? flash::convert_type<Element>(acc_s)
        //     : flash::convert_type_relu<Element>(acc_s);
        //  __syncthreads();
        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);
        // __syncthreads();


#if 0        
        cute::copy(tVrV, tVsV);
        __syncthreads();
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs || Kernel_traits::Share_Q_K_smem>(
            acc_dp_ori, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp,
            smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV
        );
#else

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tSsVdp, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 0);
        // asm volatile("s_barrier");
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tSsVdp, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 1);
        // asm volatile("s_barrier");
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tSsVdp, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 2);
        // asm volatile("s_barrier");
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tSsVdp, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 3);
        __builtin_amdgcn_s_barrier();
#endif

        //  __syncthreads();
        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));


        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        if (n_block > n_block_min) {
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKrK, tKVcKV, tKVpKV);       
            // __syncthreads();
        }

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        Tensor tdQrdS = flash::convert_type<Element>(dS_reshaped);
        //  __syncthreads();
        flash::gemm_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, 
            smem_tiled_copy_Kt, smem_thr_copy_Kt);
        __builtin_amdgcn_s_barrier();

    }
    //  __syncthreads();
    // __builtin_amdgcn_sched_barrier(1);
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = flash::convert_type<Element>(acc_dq);

    Tensor sdQ = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutdQ{});
    // Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
    //                         typename Kernel_traits::SmemLayoutdQ{});

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);
    //  __syncthreads();
    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    __syncthreads();

    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    //  __syncthreads();
    // __builtin_amdgcn_sched_barrier(0);
    // __syncthreads();
    Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
    cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);
    Tensor cdQ = make_identity_tensor(make_shape(size<0>(sdQ), size<1>(sdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    }
    __builtin_amdgcn_s_barrier();    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
    __syncthreads();
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dq_1rowblock_16x64_prefetch(const Params &params, const int bidb, const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;


    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int warpId = __builtin_amdgcn_readfirstlane(tidx / 64);
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kStages = Kernel_traits::kStages;

    using SdP_TiledShape_MNK = typename Kernel_traits::TiledMmaSdP::TiledShape_MNK;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(SdP_TiledShape_MNK{}))::value;;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;


    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;



    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + m_block * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + m_block * kBlockM;
    

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});


    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKGemm0{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKGemm1transposed{});
    // Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKGemm1transposedNoSwizzle{});
    
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutVGemm0{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(gQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(gdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);

    // dQ
    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKt);


    auto gmem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_QdO = gmem_tiled_copy_QdO.get_thread_slice(tidx);

    Tensor tSgQ = gmem_thr_copy_QdO.partition_S(gQ);
    Tensor tdPgdO = gmem_thr_copy_QdO.partition_S(gdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sVtemp = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor tdPsVBLayout = smem_thr_copy_BLayout.partition_S(sVtemp);
    Tensor tdPsV = make_tensor(tdPsVBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDim/32>(tdPsVBLayout.layout()));

    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDim/32>(tSsKBLayout.layout()));


    auto smem_tiled_copy_Kt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16_WITH_8x64, Element>{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt8x64 = smem_thr_copy_Kt.partition_S(sKt);
    Tensor tdQsKt = make_tensor(tdQsKt8x64.data(), convert_layout_B_rowcol<_16x128>(tdQsKt8x64.layout()));


    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cdO = make_identity_tensor(make_shape(size<0>(gdO), size<1>(gdO)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QdO.partition_D(cQ);
    Tensor tdOcdO = gmem_thr_copy_QdO.partition_D(cdO);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdPgdO)));
    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdOpdO); ++k) { tdOpdO(k) = get<1>(tdOcdO(0, 0, k)) < params.d; }
    }

    // Prologue
    
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
        
        Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.dq_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
        Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
        clear(tdQrdQ);
        Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
        Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
        if constexpr(!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        return;
    }

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tSgQ, tSrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tdPgdO, tdPrdO, tdOcdO, tdOpdO, binfo.actual_seqlen_q - m_block * kBlockM);


    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    // static_assert(decltype(size<0>(taccScS))::value == 4);
    Tensor taccScS_row = taccScS(0, _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    Tensor dP_sum = make_fragment_like(lse);
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

    //  __syncthreads();
    int n_block = n_block_max - 1;


    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tdPsV);
    constexpr int k2_loops = size<2>(tdQsKt);
    static_assert(kStages <= k0_loops && kStages <= k1_loops && kStages <= k2_loops , "kStages is error");

    if constexpr (Is_even_K)
    {
        lds_direct_copy_even_k<0, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<1, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<2, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < kStages; i++)
        {
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }
    }

    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_dq);
    // __syncthreads();
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    #pragma unroll
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s_ori);
        // __syncthreads();
        if constexpr (Is_even_K)
        {
            lds_direct_copy_even_k<3, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        else
        {
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gK, sK, kStages + 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 0);
        asm volatile("s_barrier");
        
        if constexpr (Is_even_K)
        {
            lds_direct_copy_even_k<0, /*Is_even_MN=*/Is_even_MN>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        else
        {
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gV, sV, 0, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k0_loops - kStages + 0);
        asm volatile("s_barrier");

        if constexpr (Is_even_K)
        {
            lds_direct_copy_even_k<1, /*Is_even_MN=*/Is_even_MN>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        else
        {
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gV, sV, 1, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k0_loops - kStages + 1);
        asm volatile("s_barrier");

        if constexpr (Is_even_K)
        {
            lds_direct_copy_even_k<2, /*Is_even_MN=*/Is_even_MN>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        else
        {
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gV, sV, 2, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k0_loops - kStages + 2);
        asm volatile("s_barrier");


        // #pragma unroll
        // for (int i = 0; i < kStages; i++)
        // {
        //     lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gV, sV, i, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        //     asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        //     flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k0_loops - kStages + i);
        //     asm volatile("s_barrier");
        // }
        // asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), flash::convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        //  __syncthreads();
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        //  __syncthreads();
        // Softcapping - calculating dTanh and scaling dS later with it
        [[maybe_unused]] Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }
        //  __syncthreads();
        #if 1
        //  __syncthreads();
        if constexpr (Has_alibi) {
            const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int col_idx_offset = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = AtomLayoutMS * 16;
            alibi.apply_alibi_continuous(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }
        //  __syncthreads();
        #endif
        //   __syncthreads();
        #if 1
        if constexpr (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                const int col_idx_offset_ = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
                flash::apply_mask_continuous(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if constexpr (Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_causal_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        AtomLayoutMS * 16);
            }
        } else if constexpr (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_local_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }
        #endif
        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
        #if 1
        if constexpr (Is_dropout) {
            const int wave_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            if constexpr (kHeadDim==128){
                dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, m_block * kBlockM, block_col_idx, AtomLayoutMS * 16
                );
            }else{
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, block_row_idx, block_col_idx, AtomLayoutMS * 16
                );
            }

        }
        #endif

        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);
        if constexpr (Is_even_K)
        {
            lds_direct_copy_even_k<3, /*Is_even_MN=*/Is_even_MN>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        else
        {
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gV, sV, 3, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 0);
        asm volatile("s_barrier");

        if constexpr (Is_even_K)
        {
            lds_direct_copy_even_k<0, Is_even_MN, _16x128>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        else
        {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gK, sKt, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k1_loops - kStages +  0);
        asm volatile("s_barrier");

        if constexpr (Is_even_K)
        {
            lds_direct_copy_even_k<1, Is_even_MN, _16x128>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        else
        {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gK, sKt, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k1_loops - kStages +  1);
        asm volatile("s_barrier");

        if constexpr (Is_even_K)
        {
            lds_direct_copy_even_k<2, Is_even_MN, _16x128>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        else
        {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gK, sKt, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k1_loops - kStages +  2);
        asm volatile("s_barrier");

        // asm volatile("s_barrier");
        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        
        // asm volatile("s_barrier");

        
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
#if 1
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            } 
        }
#endif

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        Tensor tdQrdS = flash::convert_type<Element>(dS_reshaped);

        if constexpr (Is_even_K)
        {
            lds_direct_copy_even_k<3, Is_even_MN, _16x128>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        else
        {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gK, sKt, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);       
        asm volatile("s_barrier");

        if (n_block > n_block_min) {
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            if constexpr (Is_even_K)
            {
                lds_direct_copy_even_k<0, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            }
            else 
            {
                lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            }
            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);       
            asm volatile("s_barrier");

            if constexpr (Is_even_K)
            {
                lds_direct_copy_even_k<1, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            }
            else 
            {
                lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            }
            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);       
            asm volatile("s_barrier");

            if constexpr (Is_even_K)
            {
                lds_direct_copy_even_k<2, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            }
            else 
            {
                lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            }
            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);       
            asm volatile("s_barrier");


        }
        else if (kStages == 3){
            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);       
            // asm volatile("s_barrier");
            asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
            flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);       
            asm volatile("s_barrier");
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);       
            asm volatile("s_barrier");
        } else {
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            #pragma unroll
            for (int i = 0; i < kStages; ++i) { // tail kStages
                flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, k2_loops - kStages + i);
                asm volatile("s_barrier");
            }
        }
    }

#if 1
    // #pragma unroll
    // for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
    // Convert acc_dq from fp32 to fp16
    // Tensor rdQ = flash::convert_type<Element>(acc_dq);
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    //  __syncthreads();
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));


    using GmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    auto gmem_tiled_copy_dQ = make_tiled_copy_C(GmemCopyAtom{}, tiled_mma_dq);
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    Tensor taccdQrdQ = gmem_thr_copy_dQ.retile_S(acc_dq);
    Tensor taccdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    // Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(taccdQgdQ)));
    // // if constexpr(!Is_even_K) {
    // //     #pragma unroll
    // //     for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    // // }
    int row, col;
    #pragma unroll
    for (int m = 0; m < size<1>(taccdQrdQ); m++)
    {
        if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM)
        {
            row = (m*AtomLayoutMS + warpId) * 16 + (laneId % 16);
            #pragma unroll
            for (int k = 0; k < size<2>(taccdQrdQ); k++)
            {                
                col = (laneId / 16) * 2 + k * 32;
                using result_type = cutlass::Array<Element, 2>;
                if constexpr (Is_even_K)
                {
                    for (int ei = 0; ei < 4; ++ei)
                    {
                        result_type res;
                        res[0] = flash::convert_type<Element>(acc_dq(ei, m, k) *  params.scale_softmax_rp_dropout);
                        res[1] = flash::convert_type<Element>(acc_dq(ei + 4, m, k) *  params.scale_softmax_rp_dropout);
                        *(result_type*)(&gdQ(row, col)) = res;
                        col += 8;

                    }
                }
            }
        }
    }

#elif
#endif
}

#if 1
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dq_1rowblock_16x64_dim96_prefetch(const Params &params, const int bidb, const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;


    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int warpId = __builtin_amdgcn_readfirstlane(tidx / 64);
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kStages = Kernel_traits::kStages - 1;

    using SdP_TiledShape_MNK = typename Kernel_traits::TiledMmaSdP::TiledShape_MNK;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(SdP_TiledShape_MNK{}))::value;;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;


    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;



    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + m_block * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + m_block * kBlockM;
    

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});


    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKVGemm0{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKGemm1transposed{});
    Tensor sKtSplit = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransSplit{});
    
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKVGemm0{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(gQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(gdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);

    // dQ
    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKt);


    auto gmem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_QdO = gmem_tiled_copy_QdO.get_thread_slice(tidx);

    Tensor tSgQ = gmem_thr_copy_QdO.partition_S(gQ);
    Tensor tdPgdO = gmem_thr_copy_QdO.partition_S(gdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);


    auto smem_tiled_copy_Kt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt8x64 = smem_thr_copy_Kt.partition_S(sKtSplit);
    Tensor tdQsKt = make_tensor(tdQsKt8x64.data(), convert_layout_B_rowcol_<_16x96, kHeadDim/32>(tdQsKt8x64.layout()));


    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cdO = make_identity_tensor(make_shape(size<0>(gdO), size<1>(gdO)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QdO.partition_D(cQ);
    Tensor tdOcdO = gmem_thr_copy_QdO.partition_D(cdO);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdPgdO)));
    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdOpdO); ++k) { tdOpdO(k) = get<1>(tdOcdO(0, 0, k)) < params.d; }
    }

    // Prologue
    
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
        
        Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.dq_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
        Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
        clear(tdQrdQ);
        Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
        Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
        if constexpr(!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        return;
    }

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tSgQ, tSrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tdPgdO, tdPrdO, tdOcdO, tdOpdO, binfo.actual_seqlen_q - m_block * kBlockM);


    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    // static_assert(decltype(size<0>(taccScS))::value == 4);
    Tensor taccScS_row = taccScS(0, _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    Tensor dP_sum = make_fragment_like(lse);
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

    //  __syncthreads();
    int n_block = n_block_max - 1;


    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tdPsV);
    constexpr int k2_loops = size<2>(tdQsKt);
    static_assert(kStages <= k0_loops && kStages <= k1_loops && kStages <= k2_loops , "kStages is error");

    #pragma unroll
    for (int i = 0; i < kStages; i++) {
        lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }


    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_dq);
    // __syncthreads();
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    #define BIDX 0
    #define BIDY 0
    #define BIDZ 0
    #pragma unroll
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s_ori);
        // __syncthreads();

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gK, sK, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<2>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 0);
        asm volatile("s_barrier");

        #pragma unroll
        for (int i = 0; i < kStages; i++)
        {
            lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gV, sV, i, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            s_waitcnt<2>();
            flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k0_loops - kStages + i);
            asm volatile("s_barrier");
        }
        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), flash::convert_layout_acc(acc_s_ori.layout()));
        asm volatile("s_barrier");
        

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == BIDX && blockIdx.y == BIDY && blockIdx.z == BIDZ) {
        //     printf("acc_s bidb:%d bidh:%d tid:%d m_block:%d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", bidb, bidh, tidx, m_block, n_block,
        //     acc_s(0, 0, 0), acc_s(1, 0, 0), acc_s(2, 0, 0), acc_s(3, 0, 0), 
        //     acc_s(0, 0, 1), acc_s(1, 0, 1), acc_s(2, 0, 1), acc_s(3, 0, 1), 
        //     acc_s(0, 0, 2), acc_s(1, 0, 2), acc_s(2, 0, 2), acc_s(3, 0, 2), 
        //     acc_s(0, 0, 3), acc_s(1, 0, 3), acc_s(2, 0, 3), acc_s(3, 0, 3), 
        //     acc_s(0, 1, 0), acc_s(1, 1, 0), acc_s(2, 1, 0), acc_s(3, 1, 0), 
        //     acc_s(0, 1, 1), acc_s(1, 1, 1), acc_s(2, 1, 1), acc_s(3, 1, 1), 
        //     acc_s(0, 1, 2), acc_s(1, 1, 2), acc_s(2, 1, 2), acc_s(3, 1, 2), 
        //     acc_s(0, 1, 3), acc_s(1, 1, 3), acc_s(2, 1, 3), acc_s(3, 1, 3)
        //     );
        // }

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        asm volatile("s_barrier");
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        // Softcapping - calculating dTanh and scaling dS later with it
        [[maybe_unused]] Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }
        asm volatile("s_barrier");
        #if 1
        if constexpr (Has_alibi) {
            const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int col_idx_offset = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = AtomLayoutMS * 16;
            alibi.apply_alibi_continuous(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }
        #endif
        #if 1
        if constexpr (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                const int col_idx_offset_ = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
                flash::apply_mask_continuous(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if constexpr (Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_causal_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        AtomLayoutMS * 16);
            }
        } else if constexpr (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_local_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }
        #endif
        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
        #if 1
        if constexpr (Is_dropout) {
            const int wave_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS * 16
            );
        }
        #endif
        asm volatile("s_barrier");

        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);
        asm volatile("s_barrier");

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gV, sV, 2, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<2>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 0);
        asm volatile("s_barrier");

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gK, sKt, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 1);
        asm volatile("s_barrier");

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gK, sKt, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<4>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 2);
        asm volatile("s_barrier");
        
        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        
        asm volatile("s_barrier");

        
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        #if 1
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            } 
        }
        #endif
        asm volatile("s_barrier");

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        Tensor tdQrdS = flash::convert_type<Element>(dS_reshaped);
        asm volatile("s_barrier");

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gK, sKt, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gK, sKt, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_barrier");

        s_waitcnt<6>();
        flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, 0);
        asm volatile("s_barrier");
        s_waitcnt<4>();
        flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, 1);
        asm volatile("s_barrier");
        s_waitcnt<2>();
        flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, 2);
        asm volatile("s_barrier");
        s_waitcnt<0>();
        flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, 3);
        asm volatile("s_barrier");

        if (n_block > n_block_min) {
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            #pragma unroll 
            for (int i = 0; i < kStages; i++) {
                lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            }
        }
        asm volatile("s_barrier");
    }

#if 1
    // #pragma unroll
    // for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
    // Convert acc_dq from fp32 to fp16
    // Tensor rdQ = flash::convert_type<Element>(acc_dq);
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    //  __syncthreads();
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));


    using GmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    auto gmem_tiled_copy_dQ = make_tiled_copy_C(GmemCopyAtom{}, tiled_mma_dq);
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    Tensor taccdQrdQ = gmem_thr_copy_dQ.retile_S(acc_dq);
    Tensor taccdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    // Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(taccdQgdQ)));
    // // if constexpr(!Is_even_K) {
    // //     #pragma unroll
    // //     for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    // // }
    #pragma unroll
    for (int m = 0; m < size<1>(taccdQrdQ); m++)
    {
        if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM)
        {
            #pragma unroll
            for (int k = 0; k < size<2>(taccdQrdQ); k++)
            {                
                const int col_id = get<1>(tdQcdQ(0, 0, k));
                for (int i = 0; i < size<0>(taccdQrdQ); i++)
                {          
                    if (Is_even_K || col_id + i * 4 < params.d) {
                        taccdQgdQ(i, m, k) = flash::convert_type<Element>(taccdQrdQ(i, m, k) * params.scale_softmax_rp_dropout);
                    }   
                }
            }
        }
    }

#elif 0
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = flash::convert_type<Element>(acc_dq);
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    //  __syncthreads();
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));


    using GmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    auto gmem_tiled_copy_dQ = make_tiled_copy_C(GmemCopyAtom{}, tiled_mma_dq);
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    Tensor taccdQrdQ = gmem_thr_copy_dQ.retile_S(rdQ);
    Tensor taccdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(taccdQgdQ)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    }
    #pragma unroll
    for (int m = 0; m < size<1>(taccdQrdQ); m++)
    {
        if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM)
        {
            #pragma unroll
            for (int k = 0; k < size<2>(taccdQrdQ); k++)
            {                
                const int col_id = get<1>(tdQcdQ(0, 0, k));
                for (int i = 0; i < size<0>(taccdQrdQ); i++)
                {          
                    if (Is_even_K ||col_id + i * 4 < params.d) {
                        taccdQgdQ(i, m, k) = taccdQrdQ(i, m, k);
                    }   
                }
            }
        }
    }
#else

    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = flash::convert_type<Element>(acc_dq);

    Tensor sdQ = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdQ{});
    // Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
    //                         typename Kernel_traits::SmemLayoutdQ{});

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);
    //  __syncthreads();
    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    __syncthreads();

    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    //  __syncthreads();
    // __builtin_amdgcn_sched_barrier(0);
    // __syncthreads();
    Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
    cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);
    Tensor cdQ = make_identity_tensor(make_shape(size<0>(sdQ), size<1>(sdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    }
    __builtin_amdgcn_s_barrier();    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
    __syncthreads();

#endif
}
#else
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dq_1rowblock_16x64_dim96_prefetch(const Params &params, const int bidb, const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;


    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int warpId = __builtin_amdgcn_readfirstlane(tidx / 64);
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kSmemOffset = Kernel_traits::kSmemOffset;
    constexpr int kStages = Kernel_traits::kStages;

    using SdP_TiledShape_MNK = typename Kernel_traits::TiledMmaSdP::TiledShape_MNK;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(SdP_TiledShape_MNK{}))::value;;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;


    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;



    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + m_block * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + m_block * kBlockM;
    

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});


    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKVGemm0Split{});
    Tensor sKt = make_tensor(sK.data() + kSmemOffset, typename Kernel_traits::SmemLayoutKGemm1transposed{});
    Tensor sKtSplit = make_tensor(sKt.data(), typename Kernel_traits::SmemLayoutKtransSplit{});
    
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKVGemm0Split{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(gQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(gdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);

    // dQ
    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKt);


    auto gmem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_QdO = gmem_tiled_copy_QdO.get_thread_slice(tidx);

    Tensor tSgQ = gmem_thr_copy_QdO.partition_S(gQ);
    Tensor tdPgdO = gmem_thr_copy_QdO.partition_S(gdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);

    auto smem_tiled_copy_Kt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt8x64 = smem_thr_copy_Kt.partition_S(sKtSplit);
    Tensor tdQsKt = make_tensor(tdQsKt8x64.data(), convert_layout_B_rowcol_<_16x96, kHeadDim/32>(tdQsKt8x64.layout()));


    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cdO = make_identity_tensor(make_shape(size<0>(gdO), size<1>(gdO)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QdO.partition_D(cQ);
    Tensor tdOcdO = gmem_thr_copy_QdO.partition_D(cdO);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdPgdO)));
    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdOpdO); ++k) { tdOpdO(k) = get<1>(tdOcdO(0, 0, k)) < params.d; }
    }

    // Prologue
    
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
        
        Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.dq_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
        Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
        clear(tdQrdQ);
        Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
        Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
        if constexpr(!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        return;
    }

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tSgQ, tSrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tdPgdO, tdPrdO, tdOcdO, tdOpdO, binfo.actual_seqlen_q - m_block * kBlockM);


    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    // static_assert(decltype(size<0>(taccScS))::value == 4);
    Tensor taccScS_row = taccScS(0, _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    Tensor dP_sum = make_fragment_like(lse);
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

    //  __syncthreads();
    int n_block = n_block_max - 1;

    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_dq);
    // __syncthreads();
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    /**
    * S   0 --- 2048 --- 4096 --- 6144
    *        0        1        2
    * dP  6144 --- 8192  0 --- 2048 --- 4096
    *           0           1        2
    * dQ  4608 --- 6144 --- 7680 --- 9216  3072 --- 4608
    *           0        1        2              3
    */

    #define BIDX 0
    #define BIDY 0
    #define BIDZ 0

    #pragma unroll
    for (int i = 0; i < kStages; i++) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }

    #pragma unroll
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s_ori);
        // __syncthreads();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gV, sV, 0, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 0, 0);
        asm volatile("s_barrier");

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gV, sV, 1, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 1, 1);
        asm volatile("s_barrier");

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gV, sV, 2, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 2, 2);
        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), flash::convert_layout_acc(acc_s_ori.layout()));
        
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == BIDX && blockIdx.y == BIDY && blockIdx.z == BIDZ) {
        //     printf("acc_s bidb:%d bidh:%d tid:%d m_block:%d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", bidb, bidh, tidx, m_block, n_block,
        //     acc_s(0, 0, 0), acc_s(1, 0, 0), acc_s(2, 0, 0), acc_s(3, 0, 0), 
        //     acc_s(0, 0, 1), acc_s(1, 0, 1), acc_s(2, 0, 1), acc_s(3, 0, 1), 
        //     acc_s(0, 0, 2), acc_s(1, 0, 2), acc_s(2, 0, 2), acc_s(3, 0, 2), 
        //     acc_s(0, 0, 3), acc_s(1, 0, 3), acc_s(2, 0, 3), acc_s(3, 0, 3), 
        //     acc_s(0, 1, 0), acc_s(1, 1, 0), acc_s(2, 1, 0), acc_s(3, 1, 0), 
        //     acc_s(0, 1, 1), acc_s(1, 1, 1), acc_s(2, 1, 1), acc_s(3, 1, 1), 
        //     acc_s(0, 1, 2), acc_s(1, 1, 2), acc_s(2, 1, 2), acc_s(3, 1, 2), 
        //     acc_s(0, 1, 3), acc_s(1, 1, 3), acc_s(2, 1, 3), acc_s(3, 1, 3)
        //     );
        // }

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        // Softcapping - calculating dTanh and scaling dS later with it
        [[maybe_unused]] Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }
        #if 1
        if constexpr (Has_alibi) {
            const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int col_idx_offset = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = AtomLayoutMS * 16;
            alibi.apply_alibi_continuous(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }
        #endif
        #if 1
        if constexpr (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                const int col_idx_offset_ = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
                flash::apply_mask_continuous(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if constexpr (Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_causal_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        AtomLayoutMS * 16);
            }
        } else if constexpr (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_local_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }
        #endif
        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
        #if 1
        if constexpr (Is_dropout) {
            const int wave_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS * 16
            );
        }
        #endif

        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);
        
        // wangaq debug
        // __syncthreads();
        // s_waitcnt<0>();
        // if (tidx == 0 && blockIdx.x == BIDX && blockIdx.y == BIDY && blockIdx.z == BIDZ) {
        //     __half * tmp = reinterpret_cast<__half*>(sV.data().get());
        //     int col = 8;
        //     for (int i = 0; i < size(sV)/col; ++i) {
        //         printf("V %d: ", i);
        //         for (int j = 0; j < col; ++j) {
        //             printf("%10.4f ", __half2float(tmp[i*col+j]));
        //         }
        //         printf("\n");
        //     }
        // }

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(1, gK, sKt, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<4>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 0, 3);
        asm volatile("s_barrier");

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(2, gK, sKt, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<5>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 1, 0);
        asm volatile("s_barrier");

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(3, gK, sKt, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<6>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 2, 1);
        asm volatile("s_barrier");
        
        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == BIDX && blockIdx.y == BIDY && blockIdx.z == BIDZ) {
        //     printf("dP bidb:%d bidh:%d tid:%d m_block:%d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", bidb, bidh, tidx, m_block, n_block,
        //     acc_dp(0, 0, 0), acc_dp(1, 0, 0), acc_dp(2, 0, 0), acc_dp(3, 0, 0), 
        //     acc_dp(0, 0, 1), acc_dp(1, 0, 1), acc_dp(2, 0, 1), acc_dp(3, 0, 1), 
        //     acc_dp(0, 0, 2), acc_dp(1, 0, 2), acc_dp(2, 0, 2), acc_dp(3, 0, 2), 
        //     acc_dp(0, 0, 3), acc_dp(1, 0, 3), acc_dp(2, 0, 3), acc_dp(3, 0, 3), 
        //     acc_dp(0, 1, 0), acc_dp(1, 1, 0), acc_dp(2, 1, 0), acc_dp(3, 1, 0), 
        //     acc_dp(0, 1, 1), acc_dp(1, 1, 1), acc_dp(2, 1, 1), acc_dp(3, 1, 1), 
        //     acc_dp(0, 1, 2), acc_dp(1, 1, 2), acc_dp(2, 1, 2), acc_dp(3, 1, 2), 
        //     acc_dp(0, 1, 3), acc_dp(1, 1, 3), acc_dp(2, 1, 3), acc_dp(3, 1, 3)
        //     );
        // }

        
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        #if 1
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            } 
        }
        #endif
        
        
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == BIDX && blockIdx.y == BIDY && blockIdx.z == BIDZ) {
        //     printf("dS bidb:%d bidh:%d tid:%d m_block:%d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", bidb, bidh, tidx, m_block, n_block,
        //     acc_dp(0, 0, 0), acc_dp(1, 0, 0), acc_dp(2, 0, 0), acc_dp(3, 0, 0), 
        //     acc_dp(0, 0, 1), acc_dp(1, 0, 1), acc_dp(2, 0, 1), acc_dp(3, 0, 1), 
        //     acc_dp(0, 0, 2), acc_dp(1, 0, 2), acc_dp(2, 0, 2), acc_dp(3, 0, 2), 
        //     acc_dp(0, 0, 3), acc_dp(1, 0, 3), acc_dp(2, 0, 3), acc_dp(3, 0, 3), 
        //     acc_dp(0, 1, 0), acc_dp(1, 1, 0), acc_dp(2, 1, 0), acc_dp(3, 1, 0), 
        //     acc_dp(0, 1, 1), acc_dp(1, 1, 1), acc_dp(2, 1, 1), acc_dp(3, 1, 1), 
        //     acc_dp(0, 1, 2), acc_dp(1, 1, 2), acc_dp(2, 1, 2), acc_dp(3, 1, 2), 
        //     acc_dp(0, 1, 3), acc_dp(1, 1, 3), acc_dp(2, 1, 3), acc_dp(3, 1, 3)
        //     );
        // }

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        Tensor tdQrdS = flash::convert_type<Element>(dS_reshaped);
        asm volatile("s_barrier");

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(0, gK, sKt, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

        // wangaq debug
        // __syncthreads();
        // s_waitcnt<0>();
        // if (tidx == 0 && blockIdx.x == BIDX && blockIdx.y == BIDY && blockIdx.z == BIDZ) {
        //     __half * tmp = reinterpret_cast<__half*>(sKt.data().get());
        //     int col = 8;
        //     for (int i = 0; i < size(sKt)/col; ++i) {
        //         printf("K %d: ", i);
        //         for (int j = 0; j < col; ++j) {
        //             printf("%10.4f ", __half2float(tmp[i*col+j]));
        //         }
        //         printf("\n");
        //     }
        // }

        s_waitcnt<6>();
        flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, 0, 1);
        asm volatile("s_barrier");
        s_waitcnt<4>();
        flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, 1, 2);
        asm volatile("s_barrier");
        s_waitcnt<2>();
        flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, 2, 3);
        asm volatile("s_barrier");
        s_waitcnt<0>();
        flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, 3, 0);
        asm volatile("s_barrier");

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == BIDX && blockIdx.y == BIDY && blockIdx.z == BIDZ) {
        //     printf("dQ bidb:%d bidh:%d tid:%d m_block:%d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", bidb, bidh, tidx, m_block, n_block,
        //     acc_dq(0, 0, 0), acc_dq(1, 0, 0), acc_dq(2, 0, 0), acc_dq(3, 0, 0), 
        //     acc_dq(4, 0, 0), acc_dq(5, 0, 0), acc_dq(6, 0, 0), acc_dq(7, 0, 0), 
        //     acc_dq(0, 0, 1), acc_dq(1, 0, 1), acc_dq(2, 0, 1), acc_dq(3, 0, 1), 
        //     acc_dq(4, 0, 1), acc_dq(5, 0, 1), acc_dq(6, 0, 1), acc_dq(7, 0, 1), 
        //     acc_dq(0, 0, 2), acc_dq(1, 0, 2), acc_dq(2, 0, 2), acc_dq(3, 0, 2), 
        //     acc_dq(4, 0, 2), acc_dq(5, 0, 2), acc_dq(6, 0, 2), acc_dq(7, 0, 2),
        //     acc_dq(0, 1, 0), acc_dq(1, 1, 0), acc_dq(2, 1, 0), acc_dq(3, 1, 0), 
        //     acc_dq(4, 1, 0), acc_dq(5, 1, 0), acc_dq(6, 1, 0), acc_dq(7, 1, 0), 
        //     acc_dq(0, 1, 1), acc_dq(1, 1, 1), acc_dq(2, 1, 1), acc_dq(3, 1, 1), 
        //     acc_dq(4, 1, 1), acc_dq(5, 1, 1), acc_dq(6, 1, 1), acc_dq(7, 1, 1), 
        //     acc_dq(0, 1, 2), acc_dq(1, 1, 2), acc_dq(2, 1, 2), acc_dq(3, 1, 2), 
        //     acc_dq(4, 1, 2), acc_dq(5, 1, 2), acc_dq(6, 1, 2), acc_dq(7, 1, 2)
        //     );
        // }


        if (n_block > n_block_min) {
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            #pragma unroll 
            for (int i = 0; i < kStages; i++) {
                lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            }
        }
    }

#if 1
    // #pragma unroll
    // for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
    // Convert acc_dq from fp32 to fp16
    // Tensor rdQ = flash::convert_type<Element>(acc_dq);
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    //  __syncthreads();
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));


    using GmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    auto gmem_tiled_copy_dQ = make_tiled_copy_C(GmemCopyAtom{}, tiled_mma_dq);
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    Tensor taccdQrdQ = gmem_thr_copy_dQ.retile_S(acc_dq);
    Tensor taccdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    // Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(taccdQgdQ)));
    // // if constexpr(!Is_even_K) {
    // //     #pragma unroll
    // //     for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    // // }
    #pragma unroll
    for (int m = 0; m < size<1>(taccdQrdQ); m++)
    {
        if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM)
        {
            #pragma unroll
            for (int k = 0; k < size<2>(taccdQrdQ); k++)
            {                
                const int col_id = get<1>(tdQcdQ(0, 0, k));
                for (int i = 0; i < size<0>(taccdQrdQ); i++)
                {          
                    if (Is_even_K || col_id + i * 4 < params.d) {
                        taccdQgdQ(i, m, k) = flash::convert_type<Element>(taccdQrdQ(i, m, k) * params.scale_softmax_rp_dropout);
                    }   
                }
            }
        }
    }

#elif
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = flash::convert_type<Element>(acc_dq);
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    //  __syncthreads();
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));


    using GmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    auto gmem_tiled_copy_dQ = make_tiled_copy_C(GmemCopyAtom{}, tiled_mma_dq);
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    Tensor taccdQrdQ = gmem_thr_copy_dQ.retile_S(rdQ);
    Tensor taccdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(taccdQgdQ)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    }
    #pragma unroll
    for (int m = 0; m < size<1>(taccdQrdQ); m++)
    {
        if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM)
        {
            #pragma unroll
            for (int k = 0; k < size<2>(taccdQrdQ); k++)
            {                
                const int col_id = get<1>(tdQcdQ(0, 0, k));
                for (int i = 0; i < size<0>(taccdQrdQ); i++)
                {          
                    if (Is_even_K ||col_id + i * 4 < params.d) {
                        taccdQgdQ(i, m, k) = taccdQrdQ(i, m, k);
                    }   
                }
            }
        }
    }
#else

    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = flash::convert_type<Element>(acc_dq);

    Tensor sdQ = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdQ{});
    // Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
    //                         typename Kernel_traits::SmemLayoutdQ{});

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);
    //  __syncthreads();
    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    __syncthreads();

    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    //  __syncthreads();
    // __builtin_amdgcn_sched_barrier(0);
    // __syncthreads();
    Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
    cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);
    Tensor cdQ = make_identity_tensor(make_shape(size<0>(sdQ), size<1>(sdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    }
    __builtin_amdgcn_s_barrier();    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
    __syncthreads();

#endif
}
#endif

#if 1
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dq_1rowblock_16x64_dim256_prefetch(const Params &params, const int bidb, const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;


    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int warpId = __builtin_amdgcn_readfirstlane(tidx / 64);
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kStages = Kernel_traits::kStages;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;



    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + m_block * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + m_block * kBlockM;
    

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});


    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKVGemm0{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKGemm1transposed{});
    Tensor sKtSplit = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransSplit{});
    
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKVGemm0{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(gQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(gK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(gdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(gV);

    // dQ
    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKt);

    auto gmem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_QdO = gmem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_QdO.partition_S(gQ);
    Tensor tdPgdO = gmem_thr_copy_QdO.partition_S(gdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);

    auto smem_tiled_copy_Kt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt8x64 = smem_thr_copy_Kt.partition_S(sKtSplit);
    Tensor tdQsKt = make_tensor(tdQsKt8x64.data(), convert_layout_B_rowcol_<_16x128, 4>(tdQsKt8x64.layout()));


    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cdO = make_identity_tensor(make_shape(size<0>(gdO), size<1>(gdO)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QdO.partition_D(cQ);
    Tensor tdOcdO = gmem_thr_copy_QdO.partition_D(cdO);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdPgdO)));
    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdOpdO); ++k) { tdOpdO(k) = get<1>(tdOcdO(0, 0, k)) < params.d; }
    }

    // Prologue
    
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
        
        Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.dq_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
        Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
        clear(tdQrdQ);
        Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
        Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
        if constexpr(!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        return;
    }

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tSgQ, tSrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tdPgdO, tdPrdO, tdOcdO, tdOpdO, binfo.actual_seqlen_q - m_block * kBlockM);


    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    // static_assert(decltype(size<0>(taccScS))::value == 4);
    Tensor taccScS_row = taccScS(0, _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    Tensor dP_sum = make_fragment_like(lse);
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }
    constexpr static int K_BUFF_SIZE = 4;
    //  __syncthreads();
    int n_block = n_block_max - 1;
    s_waitcnt<0>();

    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }
    } else {
        lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
    }



    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor acc_dq_split = local_tile(acc_dq, Shape<Int<8>, Int<1>, Int<kHeadDim / 32 / 2>>{}, make_coord(0, 0, _)); 
    auto acc_dq_0_127 = acc_dq_split(_, _, _, 0);
    auto acc_dq_128_256 = acc_dq_split(_, _, _, 1);
    clear(acc_dq);
    // __syncthreads();
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    #pragma unroll
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s_ori);
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }

        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 0);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 4, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<4, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 1);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 5, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<5, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 2);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 6, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<6, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 3);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 7, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<7, K_BUFF_SIZE, Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 4, 0);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, 0, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, Is_even_MN>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }      
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 5, 1);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, 1, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, Is_even_MN>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }     
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 6, 2);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, 2, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, Is_even_MN>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }     
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 7, 3);
        s_barrier();

        Tensor acc_s = make_tensor(acc_s_ori.data(), flash::convert_layout_acc(acc_s_ori.layout()));
        
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("acc_s tid:%d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //         "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, n_block,
        //         tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7],
        //         tmp[8], tmp[9], tmp[10], tmp[11], tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        //  __syncthreads();
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        //  __syncthreads();
        // Softcapping - calculating dTanh and scaling dS later with it
        [[maybe_unused]] Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }
        //  __syncthreads();
        #if 1
        //  __syncthreads();
        if constexpr (Has_alibi) {
            const int col_idx_offset = n_block * kBlockN;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = 4 * 16;
            alibi.apply_alibi_continuous(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }
        //  __syncthreads();
        #endif
        //   __syncthreads();
        #if 1
        if constexpr (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                const int col_idx_offset_ = n_block * kBlockN;
                flash::apply_mask_continuous(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if constexpr (Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_causal_continuous(scores, n_block * kBlockN,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        4 * 16);
            }
        } else if constexpr (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_local_continuous(scores, n_block * kBlockN,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, 4 * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }
        #endif
        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
        #if 1
        if constexpr (Is_dropout) {
            const int wave_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            if constexpr (kHeadDim==128){
                dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, m_block * kBlockM, block_col_idx, 4 * 16
                );
            }else{
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, block_row_idx, block_col_idx, 4 * 16
                );
            }

        }
        #endif

        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, 3, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, Is_even_MN>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }     
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 0);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, 4, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<4, K_BUFF_SIZE, Is_even_MN>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }     
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 1);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, 5, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<5, K_BUFF_SIZE, Is_even_MN>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }     
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 2);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, 6, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<6, K_BUFF_SIZE, Is_even_MN>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }     
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 3);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, 7, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<7, K_BUFF_SIZE, Is_even_MN>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }     
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 4, 0);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 0, gK, sKt, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 5, 1);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 1, gK, sKt, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 6, 2);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 2, gK, sKt, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 7, 3);
        s_barrier();

        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        
        // asm volatile("s_barrier");

        
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        #if 1
        {
            using __float2  = __attribute__((ext_vector_type(2))) float;
            static_assert(decltype(size<1>(dS))::value == 16);
            #pragma unroll
            for (int mi = 0; mi < size<0>(dS); ++mi) {
                const float d = dP_sum(mi);
                __float2 d_pair = {-d, -d};
                for (int ni = 0; ni < 4; ni ++) {
                    __float2 scores_pair_0 = {scores(mi, ni), scores(mi, ni + 4)};
                    __float2 scores_pair_1 = {scores(mi, ni + 8), scores(mi, ni + 12)};
                    __float2 dS_pair_0;
                    __float2 dS_pair_1;
                    if constexpr (!Is_dropout)
                    {
                        dS_pair_0 = {dS(mi, ni), dS(mi, ni + 4)};
                        dS_pair_1 = {dS(mi, ni + 8), dS(mi, ni + 12)};
                        dS_pair_0 = __builtin_hcu_pk_add_f32(dS_pair_0, d_pair);
                        dS_pair_1 = __builtin_hcu_pk_add_f32(dS_pair_1, d_pair);
                    }
                    else
                    {
                        dS_pair_0 = {
                            !Is_dropout || scores_pair_0.x >= 0 ? dS(mi, ni) - d : d,
                            !Is_dropout || scores_pair_0.y >= 0 ? dS(mi, ni + 4) - d : d
                        };
                        dS_pair_1 = {
                            !Is_dropout || scores_pair_1.x >= 0 ? dS(mi, ni + 8) - d : d,
                            !Is_dropout || scores_pair_1.y >= 0 ? dS(mi, ni + 12) - d : d
                        };
                    }
                    __float2 scaled_ds_0 = __builtin_hcu_pk_mul_f32(scores_pair_0, dS_pair_0);
                    __float2 scaled_ds_1 = __builtin_hcu_pk_mul_f32(scores_pair_1, dS_pair_1);
                    dS(mi, ni) = scaled_ds_0.x;
                    dS(mi, ni + 4) = scaled_ds_0.y;
                    dS(mi, ni + 8) = scaled_ds_1.x;
                    dS(mi, ni + 12) = scaled_ds_1.y;
                }
                // #pragma unroll
                // for (int ni = 0; ni < size<1>(dS); ni += 2) {
                //     const float d = dP_sum(mi);
                //     __float2 scores_pair = {scores(mi, ni), scores(mi, ni + 1)};
                //     __float2 dS_pair;
                //     if constexpr (!Is_dropout)
                //     {
                //         __float2 d_pair = {-d, -d};
                //         dS_pair = {dS(mi, ni), dS(mi, ni + 1)};
                //         dS_pair = __builtin_hcu_pk_add_f32(dS_pair, d_pair);
                //     }
                //     else
                //     {
                //         dS_pair = {
                //             !Is_dropout || scores_pair.x >= 0 ? dS(mi, ni) - d : d,
                //             !Is_dropout || scores_pair.y >= 0 ? dS(mi, ni + 1) - d : d
                //         };
                //     }
                //     __float2 scaled_ds = __builtin_hcu_pk_mul_f32(scores_pair, dS_pair);
                //     dS(mi, ni) = scaled_ds.x;
                //     dS(mi, ni + 1) = scaled_ds.y;
                // }
            }
        }
        // #pragma unroll
        // for (int mi = 0; mi < size<0>(dS); ++mi) {
        //     #pragma unroll
        //     for (int ni = 0; ni < size<1>(dS); ++ni) {
        //         float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
        //         if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
        //         dS(mi, ni) = scaled_ds;
        //     } 
        // }
        #endif

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        Tensor tdQrdS = flash::convert_type<Element>(dS_reshaped);
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 3, gK, sKt, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }        
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_dq_0_127, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 0, gK, sKt, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }        
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_dq_0_127, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 1, gK, sKt, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }    
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_dq_0_127, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 2, gK, sKt, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }    
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_dq_0_127, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();
        
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 3, gK, sKt, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }    
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_dq_128_256, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_dq_128_256, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_dq_128_256, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_dq_128_256, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        if (n_block > n_block_min) {
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            if constexpr(!Is_even_K) {
                #pragma unroll 
                for (int i = 0; i < 3; i ++) {
                    lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d);
                }
            } else {
                lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            }

        }
    }

#if 1
    // #pragma unroll
    // for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
    // Convert acc_dq from fp32 to fp16
    // Tensor rdQ = flash::convert_type<Element>(acc_dq);
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    //  __syncthreads();
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));


    using GmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    auto gmem_tiled_copy_dQ = make_tiled_copy_C(GmemCopyAtom{}, tiled_mma_dq);
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    Tensor taccdQrdQ = gmem_thr_copy_dQ.retile_S(acc_dq);
    Tensor taccdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    // Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(taccdQgdQ)));
    // // if constexpr(!Is_even_K) {
    // //     #pragma unroll
    // //     for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    // // }
    #pragma unroll
    for (int m = 0; m < size<1>(taccdQrdQ); m++)
    {
        if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM)
        {
            int row = (m*kBlockM + warpId) * 16 + (laneId % 16);
            #pragma unroll
            for (int k = 0; k < size<2>(taccdQrdQ); k++)
            {                
                const int col_id = get<1>(tdQcdQ(0, 0, k));
                if constexpr (Is_even_K)
                {
                    int col = (laneId / 16) * 2 + k * 32;
                    for (int ei = 0; ei < 4; ++ei)
                    {
                        using __float2  = __attribute__((ext_vector_type(2))) float;
                        __float2 scale_softmax_rp_dropoutx2 = {params.scale_softmax_rp_dropout, params.scale_softmax_rp_dropout};
                        __float2 acc_dqx2 = {acc_dq(ei, m, k), acc_dq(ei + 4, m, k)};
                        __float2 resx2  = __builtin_hcu_pk_mul_f32(acc_dqx2, scale_softmax_rp_dropoutx2);

                        using result_type = cutlass::Array<Element, 2>;
                        result_type res;
                        res[0] = flash::convert_type<Element>(resx2[0]);
                        res[1] = flash::convert_type<Element>(resx2[1]);
                        *(result_type*)(&gdQ(row, col)) = res;
                        col += 8;
                    }
                }
                else
                {
                    for (int i = 0; i < size<0>(taccdQrdQ); i++)
                    {          
                        if (Is_even_K || col_id + i * 4 < params.d) {
                            taccdQgdQ(i, m, k) = flash::convert_type<Element>(taccdQrdQ(i, m, k) * params.scale_softmax_rp_dropout);
                        }   
                    }
                }
            }
        }
    }

#elif 0
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = flash::convert_type<Element>(acc_dq);
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    //  __syncthreads();
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));


    using GmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    auto gmem_tiled_copy_dQ = make_tiled_copy_C(GmemCopyAtom{}, tiled_mma_dq);
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    Tensor taccdQrdQ = gmem_thr_copy_dQ.retile_S(rdQ);
    Tensor taccdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(taccdQgdQ)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    }
    #pragma unroll
    for (int m = 0; m < size<1>(taccdQrdQ); m++)
    {
        if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM)
        {
            #pragma unroll
            for (int k = 0; k < size<2>(taccdQrdQ); k++)
            {                
                const int col_id = get<1>(tdQcdQ(0, 0, k));
                for (int i = 0; i < size<0>(taccdQrdQ); i++)
                {          
                    if (Is_even_K ||col_id + i * 4 < params.d) {
                        taccdQgdQ(i, m, k) = taccdQrdQ(i, m, k);
                    }   
                }
            }
        }
    }
#else

    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = flash::convert_type<Element>(acc_dq);

    Tensor sdQ = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdQ{});
    // Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
    //                         typename Kernel_traits::SmemLayoutdQ{});

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);
    //  __syncthreads();
    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    __syncthreads();

    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    //  __syncthreads();
    // __builtin_amdgcn_sched_barrier(0);
    // __syncthreads();
    Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
    cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);
    Tensor cdQ = make_identity_tensor(make_shape(size<0>(sdQ), size<1>(sdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    }
    __builtin_amdgcn_s_barrier();    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
    __syncthreads();

#endif
}
#endif

#if 1
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dq_1rowblock_16x64_dim512_prefetch(const Params &params, const int bidb, const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;


    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int warpId = __builtin_amdgcn_readfirstlane(tidx / 64);
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kStages = Kernel_traits::kStages;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;



    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + m_block * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + m_block * kBlockM;
    

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});


    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKVGemm0{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKGemm1transposed{});
    Tensor sKtSplit = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransSplit{});
    
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKVGemm0{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(gQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(gK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(gdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(gV);

    // dQ
    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKt);

    auto gmem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_QdO = gmem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSgQ = gmem_thr_copy_QdO.partition_S(gQ);
    Tensor tdPgdO = gmem_thr_copy_QdO.partition_S(gdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);

    auto smem_tiled_copy_Kt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt8x64 = smem_thr_copy_Kt.partition_S(sKtSplit);
    Tensor tdQsKt = make_tensor(tdQsKt8x64.data(), convert_layout_B_rowcol_<_16x128, 4>(tdQsKt8x64.layout()));


    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cdO = make_identity_tensor(make_shape(size<0>(gdO), size<1>(gdO)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QdO.partition_D(cQ);
    Tensor tdOcdO = gmem_thr_copy_QdO.partition_D(cdO);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdPgdO)));
    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdOpdO); ++k) { tdOpdO(k) = get<1>(tdOcdO(0, 0, k)) < params.d; }
    }

    // Prologue
    
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
        
        Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.dq_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
        Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
        clear(tdQrdQ);
        Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
        Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
        if constexpr(!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        return;
    }

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tSgQ, tSrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tdPgdO, tdPrdO, tdOcdO, tdOpdO, binfo.actual_seqlen_q - m_block * kBlockM);


    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    // static_assert(decltype(size<0>(taccScS))::value == 4);
    Tensor taccScS_row = taccScS(0, _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    Tensor dP_sum = make_fragment_like(lse);
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

    //  __syncthreads();
    int n_block = n_block_max - 1;
    s_waitcnt<0>();
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }


    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor acc_dq_split = local_tile(acc_dq, Shape<Int<8>, Int<1>, Int<kHeadDim / 32 / 4>>{}, make_coord(0, 0, _)); 
    auto acc_dq_0_127 = acc_dq_split(_, _, _, 0);
    auto acc_dq_128_256 = acc_dq_split(_, _, _, 1);
    auto acc_dq_256_384 = acc_dq_split(_, _, _, 2);
    auto acc_dq_384_512 = acc_dq_split(_, _, _, 3);

    clear(acc_dq);
    // __syncthreads();
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    #pragma unroll
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s_ori);

        lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gK, sK, 4, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gK, sK, 5, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gK, sK, 6, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gK, sK, 7, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 4, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gK, sK, 8, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 5, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gK, sK, 9, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 6, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gK, sK, 10, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 7, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gK, sK, 11, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 8, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gK, sK, 12, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 9, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gK, sK, 13, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 10, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gK, sK, 14, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 11, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gK, sK, 15, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 12, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, 0, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 13, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, 1, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 14, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, 2, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 15, 3);
        s_barrier();

        Tensor acc_s = make_tensor(acc_s_ori.data(), flash::convert_layout_acc(acc_s_ori.layout()));
        
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("acc_s tid:%d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //         "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, n_block,
        //         tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7],
        //         tmp[8], tmp[9], tmp[10], tmp[11], tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        //  __syncthreads();
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        //  __syncthreads();
        // Softcapping - calculating dTanh and scaling dS later with it
        [[maybe_unused]] Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }
        //  __syncthreads();
        #if 1
        //  __syncthreads();
        if constexpr (Has_alibi) {
            const int col_idx_offset = n_block * kBlockN;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = 4 * 16;
            alibi.apply_alibi_continuous(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }
        //  __syncthreads();
        #endif
        //   __syncthreads();
        #if 1
        if constexpr (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                const int col_idx_offset_ = n_block * kBlockN;
                flash::apply_mask_continuous(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if constexpr (Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_causal_continuous(scores, n_block * kBlockN,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        4 * 16);
            }
        } else if constexpr (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_local_continuous(scores, n_block * kBlockN,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, 4 * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }
        #endif
        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
        #if 1
        if constexpr (Is_dropout) {
            const int wave_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            if constexpr (kHeadDim==128){
                dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, m_block * kBlockM, block_col_idx, 4 * 16
                );
            }else{
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, block_row_idx, block_col_idx, 4 * 16
                );
            }

        }
        #endif

        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);

        lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, 3, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gV, sV, 4, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gV, sV, 5, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gV, sV, 6, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gV, sV, 7, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 4, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gV, sV, 8, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 5, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gV, sV, 9, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 6, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gV, sV, 10, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 7, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gV, sV, 11, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 8, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gV, sV, 12, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 9, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gV, sV, 13, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 10, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gV, sV, 14, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 11, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gV, sV, 15, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 12, 0);
        s_barrier();


        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 0, gK, sKt, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 13, 1);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 1, gK, sKt, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 14, 2);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 2, gK, sKt, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 15, 3);
        s_barrier();

        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        
        // asm volatile("s_barrier");

        
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        #if 1
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            } 
        }
        #endif

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        Tensor tdQrdS = flash::convert_type<Element>(dS_reshaped);
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 3, gK, sKt, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dq_0_127, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 0, gK, sKt, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dq_0_127, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 1, gK, sKt, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dq_0_127, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 2, gK, sKt, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dq_0_127, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 3, gK, sKt, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dq_128_256, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 0, gK, sKt, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dq_128_256, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 1, gK, sKt, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dq_128_256, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 2, gK, sKt, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dq_128_256, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 3, gK, sKt, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dq_256_384, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 0, gK, sKt, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dq_256_384, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 1, gK, sKt, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dq_256_384, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 2, gK, sKt, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dq_256_384, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 3, gK, sKt, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dq_384_512, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dq_384_512, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dq_384_512, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dq_384_512, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        s_barrier();

        if (n_block > n_block_min) {
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            #pragma unroll 
            for (int i = 0; i < 3; i ++) {
                lds_direct_copy<Is_even_K>(gK, sK, i, params.k_row_stride, params.d);
            }
        }
    }

#if 1
    // #pragma unroll
    // for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
    // Convert acc_dq from fp32 to fp16
    // Tensor rdQ = flash::convert_type<Element>(acc_dq);
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    //  __syncthreads();
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));


    using GmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    auto gmem_tiled_copy_dQ = make_tiled_copy_C(GmemCopyAtom{}, tiled_mma_dq);
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    Tensor taccdQrdQ = gmem_thr_copy_dQ.retile_S(acc_dq);
    Tensor taccdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    // Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(taccdQgdQ)));
    // // if constexpr(!Is_even_K) {
    // //     #pragma unroll
    // //     for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    // // }
    #pragma unroll
    for (int m = 0; m < size<1>(taccdQrdQ); m++)
    {
        if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM)
        {
            #pragma unroll
            for (int k = 0; k < size<2>(taccdQrdQ); k++)
            {                
                const int col_id = get<1>(tdQcdQ(0, 0, k));
                for (int i = 0; i < size<0>(taccdQrdQ); i++)
                {          
                    if (Is_even_K || col_id + i * 4 < params.d) {
                        taccdQgdQ(i, m, k) = flash::convert_type<Element>(taccdQrdQ(i, m, k) * params.scale_softmax_rp_dropout);
                    }   
                }
            }
        }
    }

#elif 0
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = flash::convert_type<Element>(acc_dq);
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    //  __syncthreads();
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));


    using GmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    auto gmem_tiled_copy_dQ = make_tiled_copy_C(GmemCopyAtom{}, tiled_mma_dq);
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    Tensor taccdQrdQ = gmem_thr_copy_dQ.retile_S(rdQ);
    Tensor taccdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(taccdQgdQ)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    }
    #pragma unroll
    for (int m = 0; m < size<1>(taccdQrdQ); m++)
    {
        if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM)
        {
            #pragma unroll
            for (int k = 0; k < size<2>(taccdQrdQ); k++)
            {                
                const int col_id = get<1>(tdQcdQ(0, 0, k));
                for (int i = 0; i < size<0>(taccdQrdQ); i++)
                {          
                    if (Is_even_K ||col_id + i * 4 < params.d) {
                        taccdQgdQ(i, m, k) = taccdQrdQ(i, m, k);
                    }   
                }
            }
        }
    }
#else

    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = flash::convert_type<Element>(acc_dq);

    Tensor sdQ = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdQ{});
    // Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
    //                         typename Kernel_traits::SmemLayoutdQ{});

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);
    //  __syncthreads();
    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    __syncthreads();

    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    //  __syncthreads();
    // __builtin_amdgcn_sched_barrier(0);
    // __syncthreads();
    Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
    cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);
    Tensor cdQ = make_identity_tensor(make_shape(size<0>(sdQ), size<1>(sdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    }
    __builtin_amdgcn_s_barrier();    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
    __syncthreads();

#endif
}
#endif





template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dq_1rowblock_16x64_mla_prefetch(const Params &params, const int bidb, const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;


    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int warpId = __builtin_amdgcn_readfirstlane(tidx / 64);
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kStages = Kernel_traits::kStages;

    using SdP_TiledShape_MNK = typename Kernel_traits::TiledMmaSdP::TiledShape_MNK;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(SdP_TiledShape_MNK{}))::value;;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;


    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;



    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + m_block * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + m_block * kBlockM;
    

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});


    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKGemm0{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKGemm1transposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKGemm1transposedNoSwizzle{});
    constexpr static int VLDSOffset = 4096; 
    Tensor sV = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_) + VLDSOffset) , typename Kernel_traits::SmemLayoutVGemm0{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(gQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(gdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);

    // dQ
    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);


    auto gmem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_QdO = gmem_tiled_copy_QdO.get_thread_slice(tidx);

    Tensor tSgQ = gmem_thr_copy_QdO.partition_S(gQ);
    Tensor tdPgdO = gmem_thr_copy_QdO.partition_S(gdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sVtemp = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor tdPsVBLayout = smem_thr_copy_BLayout.partition_S(sVtemp);
    Tensor tdPsV = make_tensor(tdPsVBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDimV/32>(tdPsVBLayout.layout()));

    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDim/32>(tSsKBLayout.layout()));


    auto smem_tiled_copy_Kt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt8x64 = smem_thr_copy_Kt.partition_S(sKt);
    Tensor tdQsKt = make_tensor(tdQsKt8x64.data(), convert_layout_B_rowcol<_16x192>(tdQsKt8x64.layout()));


    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cdO = make_identity_tensor(make_shape(size<0>(gdO), size<1>(gdO)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QdO.partition_D(cQ);
    Tensor tdOcdO = gmem_thr_copy_QdO.partition_D(cdO);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdPgdO)));
    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdOpdO); ++k) { tdOpdO(k) = get<1>(tdOcdO(0, 0, k)) < params.d_value; }
    }

    // Prologue
    
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
        
        Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.dq_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
        Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
        clear(tdQrdQ);
        Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
        Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
        if constexpr(!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        return;
    }

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tSgQ, tSrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tdPgdO, tdPrdO, tdOcdO, tdOpdO, binfo.actual_seqlen_q - m_block * kBlockM);


    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    // static_assert(decltype(size<0>(taccScS))::value == 4);
    Tensor taccScS_row = taccScS(0, _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    Tensor dP_sum = make_fragment_like(lse);
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

    //  __syncthreads();
    int n_block = n_block_max - 1;


    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tdPsV);
    constexpr int k2_loops = size<2>(tdQsKt);
    static_assert(kStages <= k0_loops && kStages <= k1_loops && kStages <= k2_loops , "kStages is error");

    #pragma unroll
    for (int i = 0; i < kStages; i++)
    {
        lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }


    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_dq);
    // __syncthreads();
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    #pragma unroll
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s_ori);
        // __syncthreads();
        #pragma unroll
        for (int i = 0; i < k0_loops - kStages; i++)
        {
            lds_direct_copy<Is_even_K, Is_even_MN>(gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, i);
            asm volatile("s_barrier");
        }
        #pragma unroll
        for (int i = 0; i < kStages; i++)
        {
            lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, i, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k0_loops - kStages + i);
            asm volatile("s_barrier");
        }
        // asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), flash::convert_layout_acc(acc_s_ori.layout()));

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("acc_s tid:%3d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, n_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        //  __syncthreads();
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        //  __syncthreads();
        // Softcapping - calculating dTanh and scaling dS later with it
        [[maybe_unused]] Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }
        //  __syncthreads();
        #if 1
        //  __syncthreads();
        if constexpr (Has_alibi) {
            const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int col_idx_offset = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = AtomLayoutMS * 16;
            alibi.apply_alibi_continuous(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }
        //  __syncthreads();
        #endif
        //   __syncthreads();
        #if 1
        if constexpr (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                const int col_idx_offset_ = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
                flash::apply_mask_continuous(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if constexpr (Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_causal_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        AtomLayoutMS * 16);
            }
        } else if constexpr (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_local_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }
        #endif
        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
        #if 1
        if constexpr (Is_dropout) {
            const int wave_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS * 16
            );
        }
        #endif

        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);

        #pragma unroll
        for (int i = 0; i < k1_loops - kStages; i++)
        {
            lds_direct_copy<Is_even_K, Is_even_MN>(gV, sV, kStages + i, params.v_row_stride, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, i);
            asm volatile("s_barrier");
        }
        //  #pragma unroll
        //  for (int i = 0; i < kStages; i++)
        //  {
        //      lds_direct_copy<Is_even_K, Is_even_MN, _16x192>(gK, sKt, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        //      asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        //      flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k1_loops - kStages +  i);
        //      asm volatile("s_barrier");
        //  }
    #if 0
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k1_loops - kStages +  0);
        asm volatile("s_barrier");

        lds_direct_copy<Is_even_K, Is_even_MN, _16x192>(gK, sKt, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k1_loops - kStages +  1);
        asm volatile("s_barrier");
        lds_direct_copy<Is_even_K, Is_even_MN, _16x192>(gK, sKt, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k1_loops - kStages +  2);
        asm volatile("s_barrier");
        lds_direct_copy<Is_even_K, Is_even_MN, _16x192>(gK, sKt, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    #elif 0
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k1_loops - kStages +  0);
        asm volatile("s_barrier");
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k1_loops - kStages +  1);
        asm volatile("s_barrier");
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k1_loops - kStages +  2);
        asm volatile("s_barrier");

        lds_direct_copy<Is_even_K, Is_even_MN, _16x192>(gK, sKt, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

        lds_direct_copy<Is_even_K, Is_even_MN, _16x192>(gK, sKt, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

        lds_direct_copy<Is_even_K, Is_even_MN, _16x192>(gK, sKt, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

    #elif 1
        lds_direct_copy<Is_even_K, Is_even_MN, _16x192>(gK, sKt, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k1_loops - kStages +  0);
        asm volatile("s_barrier");
        lds_direct_copy<Is_even_K, Is_even_MN, _16x192>(gK, sKt, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k1_loops - kStages +  1);
        asm volatile("s_barrier");
        lds_direct_copy<Is_even_K, Is_even_MN, _16x192>(gK, sKt, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        asm volatile("s_waitcnt vmcnt(6) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k1_loops - kStages +  2);
        asm volatile("s_barrier");
    #endif
    
        // asm volatile("s_barrier");
            // asm volatile("s_barrier");
        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_dp.data());
        //     printf("acc_dp tid:%3d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, n_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }
                
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
    #if 1
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            } 
        }
    #endif

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        Tensor tdQrdS = flash::convert_type<Element>(dS_reshaped);


        #pragma unroll
        for (int i = 0; i < k2_loops - kStages; i++)
        {
            lds_direct_copy<Is_even_K, Is_even_MN,  _16x192>(gK, sKt, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            asm volatile("s_waitcnt vmcnt(6) \n s_barrier");
            flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, i);
            asm volatile("s_barrier");
        }

        if (n_block > n_block_min) {
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));

            lds_direct_copy<Is_even_K>(gK, sK, 0, params.k_row_stride, params.d);
            asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
            flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, k2_loops - kStages + 0);
            asm volatile("s_barrier");
            lds_direct_copy<Is_even_K>(gK, sK, 1, params.k_row_stride, params.d);
            asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
            flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, k2_loops - kStages + 1);
            asm volatile("s_barrier");
            lds_direct_copy<Is_even_K>(gK, sK, 2, params.k_row_stride, params.d);
            asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
            flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, k2_loops - kStages + 2);
            asm volatile("s_barrier");
        }
        else if (kStages == 3){
            asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
            flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, k2_loops - kStages + 0);
            // asm volatile("s_barrier");
            asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
            flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, k2_loops - kStages + 1);
            asm volatile("s_barrier");
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, k2_loops - kStages + 2);
            asm volatile("s_barrier");
        } 
        else {
            asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
            #pragma unroll
            for (int i = 0; i < kStages; ++i) { // tail kStages
                flash::gemm_k_rs(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt, k2_loops - kStages + i);
                asm volatile("s_barrier");
            }
        }        
    }
    
#if 1
    // wangaq debug
    // __syncthreads();
    // if (blockIdx.x == 0) {
    //     float * tmp = reinterpret_cast<float*>(acc_dq.data());
    //     printf("acc_dq tid:%3d n_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, n_block,
    //     tmp[0], tmp[1], tmp[2], tmp[3], 
    //     tmp[4], tmp[5], tmp[6], tmp[7],
    //     tmp[8], tmp[9], tmp[10], tmp[11], 
    //     tmp[12], tmp[13], tmp[14], tmp[15]
    //     );
    // }
    // #pragma unroll
    // for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
    // Convert acc_dq from fp32 to fp16
    // Tensor rdQ = flash::convert_type<Element>(acc_dq);
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    //  __syncthreads();
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));


    using GmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    auto gmem_tiled_copy_dQ = make_tiled_copy_C(GmemCopyAtom{}, tiled_mma_dq);
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    Tensor taccdQrdQ = gmem_thr_copy_dQ.retile_S(acc_dq);
    Tensor taccdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    // Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(taccdQgdQ)));
    // // if constexpr(!Is_even_K) {
    // //     #pragma unroll
    // //     for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    // // }
    #pragma unroll
    for (int m = 0; m < size<1>(taccdQrdQ); m++)
    {
        if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM)
        {
            #pragma unroll
            for (int k = 0; k < size<2>(taccdQrdQ); k++)
            {                
                const int col_id = get<1>(tdQcdQ(0, 0, k));
                for (int i = 0; i < size<0>(taccdQrdQ); i++)
                {          
                    if (Is_even_K || col_id + i * 4 < params.d) {
                        taccdQgdQ(i, m, k) = flash::convert_type<Element>(taccdQrdQ(i, m, k) * params.scale_softmax_rp_dropout);
                    }   
                }
            }
        }
    }

#elif
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = flash::convert_type<Element>(acc_dq);
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    //  __syncthreads();
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));


    using GmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    auto gmem_tiled_copy_dQ = make_tiled_copy_C(GmemCopyAtom{}, tiled_mma_dq);
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);

    Tensor taccdQrdQ = gmem_thr_copy_dQ.retile_S(rdQ);
    Tensor taccdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(taccdQgdQ)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    }
    #pragma unroll
    for (int m = 0; m < size<1>(taccdQrdQ); m++)
    {
        if (Is_even_MN || get<0>(tdQcdQ(0, m, 0)) < binfo.actual_seqlen_q - m_block * kBlockM)
        {
            #pragma unroll
            for (int k = 0; k < size<2>(taccdQrdQ); k++)
            {                
                const int col_id = get<1>(tdQcdQ(0, 0, k));
                for (int i = 0; i < size<0>(taccdQrdQ); i++)
                {          
                    if (Is_even_K ||col_id + i * 4 < params.d) {
                        taccdQgdQ(i, m, k) = taccdQrdQ(i, m, k);
                    }   
                }
            }
        }
    }
#else

    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = flash::convert_type<Element>(acc_dq);

    Tensor sdQ = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdQ{});
    // Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
    //                         typename Kernel_traits::SmemLayoutdQ{});

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);
    //  __syncthreads();
    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    __syncthreads();

    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
    + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.dq_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    //  __syncthreads();
    // __builtin_amdgcn_sched_barrier(0);
    // __syncthreads();
    Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
    cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);
    Tensor cdQ = make_identity_tensor(make_shape(size<0>(sdQ), size<1>(sdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    }
    __builtin_amdgcn_s_barrier();    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
    __syncthreads();

#endif
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dq_seqq_parallel_16x64(const Params &params) {
    // The block index for the batch.
    const int bidb = blockIdx.z;
    // The block index for the head.
    const int bidh = blockIdx.y;
#if 0
    // If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
    for (int m_block = blockIdx.x; m_block < (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM; m_block += gridDim.x) {
        compute_dq_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap,
            /*Is_first*/false, /*Is_last*/false, /*Seq_parallel=*/true>(params, bidb, bidh, m_block);
    }
#else
    int m_block = blockIdx.x;     
    if constexpr (!Is_even_K) {
        compute_dq_1rowblock_16x64_dim40<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, m_block);
    }
    else {
        compute_dq_1rowblock_16x64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, m_block);
    }

#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dq_1rowblock_16x64_dim64_prefetch(const Params &params, const int bidb, const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;


    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int warpId = __builtin_amdgcn_readfirstlane(tidx / 64);
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kStages = Kernel_traits::kStages;

    using SdP_TiledShape_MNK = typename Kernel_traits::TiledMmaSdP::TiledShape_MNK;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(SdP_TiledShape_MNK{}))::value;;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;


    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    if constexpr (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;



    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + m_block * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + m_block * kBlockM;
    

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});


    Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutKGemm0{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKGemm1transposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKGemm1transposedNoSwizzle{});
    
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutVGemm0{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(gQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(gdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);

    // dQ
    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);


    auto gmem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_QdO = gmem_tiled_copy_QdO.get_thread_slice(tidx);

    Tensor tSgQ = gmem_thr_copy_QdO.partition_S(gQ);
    Tensor tdPgdO = gmem_thr_copy_QdO.partition_S(gdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sVtemp = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor tdPsVBLayout = smem_thr_copy_BLayout.partition_S(sVtemp);
    Tensor tdPsV = make_tensor(tdPsVBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDim/32>(tdPsVBLayout.layout()));

    Tensor sKtemp = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutK{});
    Tensor tSsKBLayout = smem_thr_copy_BLayout.partition_S(sKtemp);
    Tensor tSsK = make_tensor(tSsKBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDim/32>(tSsKBLayout.layout()));


    auto smem_tiled_copy_Kt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16_WITH_8x64, Element>{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt8x64 = smem_thr_copy_Kt.partition_S(sKt);
    Tensor tdQsKt = make_tensor(tdQsKt8x64.data(), convert_layout_B_rowcol<_16x64_64>(tdQsKt8x64.layout()));


    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(gQ), size<1>(gQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cdO = make_identity_tensor(make_shape(size<0>(gdO), size<1>(gdO)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QdO.partition_D(cQ);
    Tensor tdOcdO = gmem_thr_copy_QdO.partition_D(cdO);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tSgQ)));
    Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdPgdO)));
    // Set predicates for k bounds
    if constexpr (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdOpdO); ++k) { tdOpdO(k) = get<1>(tdOcdO(0, 0, k)) < params.d; }
    }

    // Prologue
    
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
        
        Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.dq_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
        Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
        clear(tdQrdQ);
        Tensor cdQ = make_identity_tensor(make_shape(size<0>(gdQ), size<1>(gdQ)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
        Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
        if constexpr(!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, binfo.actual_seqlen_q - m_block * kBlockM
        );
        return;
    }

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tSgQ, tSrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM);

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QdO, tdPgdO, tdPrdO, tdOcdO, tdOpdO, binfo.actual_seqlen_q - m_block * kBlockM);


    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    // static_assert(decltype(size<0>(taccScS))::value == 4);
    Tensor taccScS_row = taccScS(0, _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    Tensor dP_sum = make_fragment_like(lse);
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

    __syncthreads();
    int n_block = n_block_max - 1;


    constexpr int k0_loops = size<2>(tSsK);
    constexpr int k1_loops = size<2>(tdPsV);
    constexpr int k2_loops = size<2>(tdQsKt);
    static_assert(k0_loops == 2 && k1_loops == 2 && k2_loops == 4 && kBlockN == 64, "kblockn should be 64");

    // #pragma unroll
    // for (int i = 0; i < kStages; i++)
    // {
    //     lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gK, sK, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    // }
    if constexpr(Is_even_K)
    {
        lds_direct_copy_even_k<0, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<1, /*Is_even_MN=*/Is_even_MN>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);

        lds_direct_copy_even_k<0, /*Is_even_MN=*/Is_even_MN>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy_even_k<1, /*Is_even_MN=*/Is_even_MN>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
    }
    else
    {
        lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

        lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gV, sV, 0, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gV, sV, 1, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    }


    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                        bidb, bidh, tidx, params.h);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_dq);
    // __syncthreads();
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    #pragma unroll
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s_ori);
        // __syncthreads();
        // #pragma unroll
        // for (int i = 0; i < k0_loops - kStages; i++)
        // {
        //     lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gK, sK, kStages + i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        //     asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        //     flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, i);
        //     asm volatile("s_barrier");
        // }
        // #pragma unroll
        // for (int i = 0; i < kStages; i++)
        // {
        //     lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gV, sV, i, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        //     asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        //     flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k0_loops - kStages + i);
        //     asm volatile("s_barrier");
        // }
        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 0);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tSrQ, tSrK, tSsK, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 1);

        asm volatile("s_barrier");
        if constexpr(Is_even_K)
        {
            lds_direct_copy_even_k<0, Is_even_MN, _16x64_64>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_even_k<1, Is_even_MN, _16x64_64>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_even_k<2, Is_even_MN, _16x64_64>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy_even_k<3, Is_even_MN, _16x64_64>(gK, sKt, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
        }
        else
        {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gK, sKt, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gK, sKt, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gK, sKt, 2, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gK, sKt, 3, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        }

        asm volatile("s_barrier");

        Tensor acc_s = make_tensor(acc_s_ori.data(), flash::convert_layout_acc(acc_s_ori.layout()));

        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        //  __syncthreads();
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        //  __syncthreads();
        // Softcapping - calculating dTanh and scaling dS later with it
        [[maybe_unused]] Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores, dtanh, params.softcap);
        }
        //  __syncthreads();
        #if 1
        //  __syncthreads();
        if constexpr (Has_alibi) {
            const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int col_idx_offset = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
            const int row_idx_offset = m_block * kBlockM + get<0>(taccScS_row(0));
            const int warp_row_stride = AtomLayoutMS * 16;
            alibi.apply_alibi_continuous(scores, col_idx_offset, row_idx_offset, warp_row_stride);
        }
        //  __syncthreads();
        #endif
        //   __syncthreads();
        #if 1
        if constexpr (!Is_causal && !Is_local) {
            if (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                const int col_idx_offset_ = n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16;
                flash::apply_mask_continuous(scores, binfo.actual_seqlen_k, col_idx_offset_);
            }
        } else if constexpr (Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_causal_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q,
                                        AtomLayoutMS * 16);
            }
        } else if constexpr (Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left
                || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
                const int warp_id = __builtin_amdgcn_readfirstlane(tidx / 64);
                flash::apply_mask_local_continuous(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
                                        binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                                        binfo.actual_seqlen_q, AtomLayoutMS * 16,
                                        params.window_size_left, params.window_size_right);
            }
        }
        #endif
        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
        #if 1
        if constexpr (Is_dropout) {
            const int wave_id = __builtin_amdgcn_readfirstlane(tidx / 64);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = m_block * kBlockM + row_idx_offset_in_block;

            const int block_row_idx = row_idx_offset_;
            const int block_col_idx = n_block * (kBlockN);
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            if constexpr (kHeadDim==64){
                dropout.template apply_dropout_continuous_opt</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, m_block * kBlockM, n_block * kBlockN, AtomLayoutMS * 16
                );
            }else{
                dropout.template apply_dropout_continuous</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, block_row_idx, block_col_idx, AtomLayoutMS * 16
                );
            }
        }
        #endif

        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);

        // #pragma unroll
        // for (int i = 0; i < k1_loops - kStages; i++)
        // {
        //     lds_direct_copy<Is_even_K, /*Is_even_MN=*/Is_even_MN>(gV, sV, kStages + i, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        //     asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        //     flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, i);
        //     asm volatile("s_barrier");
        // }
        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 0);
        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, 1);
        asm volatile("s_barrier");

        // #pragma unroll
        // for (int i = 0; i < kStages; i++)
        // {
        //     lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gK, sKt, i, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
        //     asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        //     flash::gemm_k_rs(acc_dp_ori, tdPrdO, tdPrV, tdPsV, tiled_mma_sdp, smem_tiled_copy_KV, smem_thr_copy_KV, k1_loops - kStages +  i);
        //     asm volatile("s_barrier");
        // }
        // asm volatile("s_barrier");
        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        
        // if (thread0())
        // {
        //     printf("acc_dp %f %f %f %f\n", float(acc_dp(0)), float(acc_dp(1)), float(acc_dp(2)), float(acc_dp(3)));
        // }

        
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
#if 1
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            } 
        }
#endif

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        Tensor tdQrdS = flash::convert_type<Element>(dS_reshaped);

        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_dq, tdQrdS, tdQrKt, tdQsKt, tiled_mma_dq, smem_tiled_copy_Kt, smem_thr_copy_Kt);
        asm volatile("s_barrier");



        if (n_block > n_block_min) {
            gV.data() = gV.data() + (-int(kBlockN * params.v_row_stride));
            gK.data() = gK.data() + (-int(kBlockN * params.k_row_stride));
            if constexpr (Is_even_K)
            {
                lds_direct_copy_even_k<0, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<1, /*Is_even_MN=*/true>(gK, sK, params.k_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            
                lds_direct_copy_even_k<0, /*Is_even_MN=*/true>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy_even_k<1, /*Is_even_MN=*/true>(gV, sV, params.v_row_stride, binfo.actual_seqlen_k - n_block * kBlockN);
            }
            else
            {
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true>(gK, sK, 0, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true>(gK, sK, 1, params.k_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true>(gV, sV, 0, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
                lds_direct_copy<Is_even_K, /*Is_even_MN=*/true>(gV, sV, 1, params.v_row_stride, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
            }
        }

    }
    const index_t row_offset_dq = binfo.q_offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                        Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_stride(params.dq_row_stride, _1{}));
    int row, col;
    for (int m = 0; m < size<1>(acc_dq); m++)
    {
        row = (m*AtomLayoutMS + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM)
        {
            for (int k = 0; k < size<2>(acc_dq); k++)
            {
                col = (laneId / 16) * 2 + k * 32;
                using result_type = cutlass::Array<Element, 2>;
                if constexpr (Is_even_K)
                {
                    for (int ei = 0; ei < 4; ++ei)
                    {
                        result_type res;
                        res[0] = flash::convert_type<Element>(acc_dq(ei, m, k) *  params.scale_softmax_rp_dropout);
                        res[1] = flash::convert_type<Element>(acc_dq(ei + 4, m, k) *  params.scale_softmax_rp_dropout);
                        *(result_type*)(&gdQ(row, col)) = res;
                        col += 8;
                    }
                }
                else
                {
                    for (int ei = 0; ei < 4; ++ei)
                    {
                        result_type res;
                        res[0] = flash::convert_type<Element>(acc_dq(ei, m, k) *  params.scale_softmax_rp_dropout);
                        res[1] = flash::convert_type<Element>(acc_dq(ei + 4, m, k) *  params.scale_softmax_rp_dropout);
                        if (col < params.d)
                        {
                            gdQ(row, col) = res[0];
                        }
                        if (col + 1 < params.d)
                        {
                            gdQ(row, col + 1) = res[1];
                        }
                        col += 8;
                    }
                }
            }
        }
    }
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dq_seqq_parallel_16x64_prefetch(const Params &params) {
    // The block index for the batch.
    const int bidb = blockIdx.z;
    // The block index for the head.
    const int bidh = blockIdx.y;
#if 0
    // If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
    for (int m_block = blockIdx.x; m_block < (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM; m_block += gridDim.x) {
        compute_dq_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap,
            /*Is_first*/false, /*Is_last*/false, /*Seq_parallel=*/true>(params, bidb, bidh, m_block);
    }
#else
    int m_block = blockIdx.x;
    #if (defined(__gfx936__) || defined(__gfx938__) ||defined(__gfx92a__))
    using Element = typename Kernel_traits::Element;

    if constexpr (Kernel_traits::kHeadDim == 192 && Kernel_traits::kHeadDimV == 128)
    {
        compute_dq_1rowblock_16x64_mla_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, m_block);
        // #ifndef NO_CAUSAL_OPT
        // if constexpr (Is_causal)
        // {
        //     const int num_blocks = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
        //     if (num_blocks - m_block - 1 != m_block)
        //     {
        //         compute_dq_1rowblock_16x64_mla_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, num_blocks - m_block - 1);
        //     }
        // }
        // #endif
        return;
    }

    if constexpr (Kernel_traits::kHeadDim == 128)
    {
        compute_dq_1rowblock_16x64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, m_block);
        #ifndef NO_CAUSAL_OPT
        if constexpr (Is_causal)
        {
            const int num_blocks = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
            if (num_blocks - m_block - 1 != m_block)
            {
                compute_dq_1rowblock_16x64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, num_blocks - m_block - 1);
            }
        }
        #endif
        return;
    }

    if constexpr (Kernel_traits::kHeadDim == 96)
    {
        compute_dq_1rowblock_16x64_dim96_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, m_block);
        #ifndef NO_CAUSAL_OPT
        if constexpr (Is_causal)
        {
            const int num_blocks = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
            if (num_blocks - m_block - 1 != m_block)
            {
                compute_dq_1rowblock_16x64_dim96_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, num_blocks - m_block - 1);
            }
        }
        #endif
        return;
    }

    if constexpr (Kernel_traits::kHeadDim == 64) {
        if constexpr(Is_causal)
        {
            const int bidbh = blockIdx.x + blockIdx.z * params.se_balance_cnt;
            const int bidb = bidbh / params.h;
            const int bidh = bidbh % params.h;

            const int num_blocks = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
            const int m_block = num_blocks - 1 - blockIdx.y;
            compute_dq_1rowblock_16x64_dim64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, m_block);

        }
        else
        {
            compute_dq_1rowblock_16x64_dim64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, m_block);
        }
        // #ifndef NO_CAUSAL_OPT
        // if constexpr (Is_causal)
        // {
        //     const int num_blocks = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
        //     if (num_blocks - m_block - 1 != m_block)
        //     {
        //         compute_dq_1rowblock_16x64_dim64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, num_blocks - m_block - 1);
        //     }
        // }
        // #endif
        return;
    }
    
    if constexpr (Kernel_traits::kHeadDim == 256)
    {
        compute_dq_1rowblock_16x64_dim256_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, m_block);
        #ifndef NO_CAUSAL_OPT
        if constexpr (Is_causal)
        {
            const int num_blocks = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
            if (num_blocks - m_block - 1 != m_block)
            {
                compute_dq_1rowblock_16x64_dim256_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, num_blocks - m_block - 1);
            }
        }
        #endif
        return;
    }

    if constexpr (Kernel_traits::kHeadDim == 512)
    {
        compute_dq_1rowblock_16x64_dim512_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, m_block);
        #ifndef NO_CAUSAL_OPT
        if constexpr (Is_causal)
        {
            const int num_blocks = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
            if (num_blocks - m_block - 1 != m_block)
            {
                compute_dq_1rowblock_16x64_dim512_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, num_blocks - m_block - 1);
            }
        }
        #endif
        return;
    }

    #endif

#endif
}



template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_dv_trans_1colblock_16x64(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;


    constexpr int MMA_N_SdP = kBlockN / size<1>(typename Kernel_traits::TiledMmaSdP::TiledShape_MNK{});
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    if constexpr (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }


    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.o_row_stride + bidh * params.o_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;
    
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));

    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});
    

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});

    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});

    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(),
                typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
#if 0
    Tensor sK = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
#else
    Tensor sK = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
#endif
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    using GmemTiledCopydO = typename Kernel_traits::GmemTiledCopydO;
    GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrK = thr_mma_sdp.partition_fragment_A(sK);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_B(sQ);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_A(sV);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_B(sdO);

    // dV/dK
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);




    //
    // Copy Atom retiling
    //

    // S/dP
    auto smem_tiled_copy_KV = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    Tensor tRgK = smem_thr_copy_KV.partition_S(gK);
    Tensor tRgV = smem_thr_copy_KV.partition_S(gV);

    auto smem_tiled_copy_QdO = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

#if 0
    // dV/dK
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dkv);
#else
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dkv);
#endif
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);

#if 1
    // debug
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
#else
    // Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOtransposedNoSwizzle);
#endif
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = smem_thr_copy_KV.partition_D(cKV);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);
    
    // If not local, we're guaranteed that m_block_min <= m_block:
    // We checked earlier that n_block * kBlockN < actual_seqlen_k, so in the causal case,
    // n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k < actual_seqlen_q.
    // So m_block_min <= (actual_seqlen_q - 1) / kBlockM.
    // Recall that m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM) = (actual_seqlen_q + kBlockM - 1) / kBlockM.
    // So m_block_m - 1 = (actual_seqlen_q - 1) / kBlockM.
    // We conclude that m_block_min <= m_block, so we will always have at least 1 iteration of the for loop.
    // However, if local, then this possible to have some blocks of K & V not attending to any query.
    // We might need to exit early and write 0 to dK and dV for those blocks.
    // Otherwise we get wrong result for the case where we don't enter the for loop.
    // And we might read OOB elements from gQ and gdO.
    // This also covers the case where actual_seqlen_q == 0
    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdKV = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }
    
    
#if 0
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    if constexpr(true) 
    {
        // cute::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrK_copy_view = smem_thr_copy_KV.retile_D(tSrK);
        cute::copy(smem_tiled_copy_KV, tSsK, tSrK_copy_view);
    }
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );


    if constexpr(Kernel_traits::Is_V_in_regs) {
        // cute::cp_async_wait<1>();
        __syncthreads();
        Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));            // M
        cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
    }

    __syncthreads();
#else
    Tensor tSrK_copy_view = smem_thr_copy_KV.retile_D(tSrK);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        smem_tiled_copy_KV, tRgK, tSrK_copy_view, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );

    Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        smem_tiled_copy_KV, tRgV, tdPrV_copy_view, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    // __syncthreads();
#endif

    Tensor tdOrdO = make_fragment_like(tdOgdO);
    Tensor tdOrO = make_fragment_like(tdOgO);

    // flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
    //     gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    // );
    // wangaq debug
    Tensor caccS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);  

    
    

    int laneId = tidx % 64;

    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
        bidb, bidh, tidx, params.h);

    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    clear(acc_dv);
    clear(acc_dk);
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    for (; m_block >= m_block_min; m_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{}); 
        clear(acc_s_ori);
        if (m_block == m_block_max - 1) {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
            );
        }
        else {
            flash::copy<true, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ
            );
        }
        
        tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
        __syncthreads();

        flash::gemm<true, false>(acc_s_ori, tSrK, tSrQ, tSsK, tSsQ, tiled_mma_sdp,
            smem_tiled_copy_KV, smem_tiled_copy_QdO, smem_thr_copy_KV, smem_thr_copy_QdO);
        // __syncthreads();
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        Tensor scores_trans = make_tensor(acc_s.data(), flash::convert_trans_layout_acc_rowcol(acc_s.layout()));
        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        Tensor dtanh_trans = make_tensor_like(scores_trans);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores_trans, dtanh_trans, params.softcap);
        }

        #if 1
            if (Has_alibi) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = tidx / 64;
                const int col_idx_offset =  m_block * kBlockM;
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                alibi.apply_alibi_trans(scores, col_idx_offset, row_idx_offset_, AtomLayoutMS * 16);
            }
        #endif

        // TD [2023-07-29]: I was thinking that we don't need to mask out the elements beyond
        // actual_seqlen_k, because acc_s would be some finite value for those indices.
        // In the end when we multiply with K to get dQ, the corresponding values of K would be 0,
        // so the result would still be correct.
        // However, it's possible that the values in acc_s are so large that they overflow
        // when we multiply with dP and convert to fp16, resulting in Inf in dS and NaNs in dQ.
        // So we need to mask out the elements beyond actual_seqlen_k.

        #if 1
        if constexpr(!Is_causal && !Is_local) {
            if (!Is_even_MN && (m_block + 1) * kBlockM >= binfo.actual_seqlen_q) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int warp_id = tidx / 64;
                // 实际上是row
                const int col_idx_offset_ = m_block * kBlockM;
                flash::apply_mask_trans(scores, binfo.actual_seqlen_q, col_idx_offset_);
            }
        } else if constexpr(Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            // if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
            //     || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
            //     const int warp_id = tidx / 64;
            //     flash::apply_mask_causal(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
            //                              binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
            //                              binfo.actual_seqlen_q,
            //                              AtomLayoutMS * 16);
            // }

            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k)
            {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_causal_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    AtomLayoutMS * 16
                );
            }
        } else if constexpr(Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_local_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    AtomLayoutMS * 16,
                    params.window_size_left, params.window_size_right
                );
            }
        }
        #endif
        
        Tensor taccScS_row = taccScS(_, 0, _);
        Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
        
        if (m_block == m_block_max - 1) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                // dim64的时候,lse是16,glse是64
                // 此时线程布局是 t0 t0 t0 t0 t16 t16 t16 t16 t32 t32 t32 t32 t48 t48 t48 t48 t0 t0 t0 t0..................
                // 按照上述格式进行线程的映射
                const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
                lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
            }
        } else {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
                lse(mi) = gLSE(row);
            }
        }

        gLSE.data() = gLSE.data() + (-int(kBlockM));

        // __syncthreads(); 
        flash::scale_apply_exp2</*scale_max=*/false>(scores_trans, lse, params.scale_softmax_log2);
        // __syncthreads(); 

        if constexpr (Is_dropout) {
            const int warp_id = tidx / 64;
            // const int warp_row_stride = 16;
            // int block_row_idx = n_block * (kBlockN / 16) + warp_id % AtomLayoutMS;
            // int block_col_idx = m_block * (kBlockM / 16) + (warp_id / AtomLayoutMS) * MMA_N_SdP;
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            int block_row_idx = row_idx_offset_;
            int block_col_idx = m_block * kBlockM;

            // int block_row_idx = m_block * (kBlockM / 16) + warp_id % AtomLayoutMS;
            // int block_col_idx = n_block * (kBlockN / 16) + (warp_id / AtomLayoutMS) * MMA_N_SdP;
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            // Tensor drop_accs = make_tensor(acc_s.data(), make_layout(get<0>(acc_s.layout()), get<2>(acc_s.layout()), get<1>(acc_s.layout())));
            dropout.template apply_dropout_trans</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS * 16
            );
        }
        #if 1
        if (m_block == m_block_max - 1) {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
            );
        } else {
            flash::copy<true, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ
            );
        }

        tdOgdO.data() = tdOgdO.data() + (-int(kBlockM * params.do_row_stride));
        #endif 

        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);



        __syncthreads(); // important sync, delete will lead copy to global error
        flash::gemm_rs(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt,
            smem_thr_copy_QdOt);
        // __syncthreads(); 
        //  __syncthreads();
        // cute::copy(smem_tiled_copy_QdOt, tdVsdOt, tdVrdO);
        // if (tidx < 64) {
        //     printf("tidx = %d %f %f %f %f %f %f %f %f\n", tidx, float(tdVrdO(0)), float(tdVrdO(1)), float(tdVrdO(2)), float(tdVrdO(3)), 
        //         float(tdVrdO(4)), float(tdVrdO(5)), float(tdVrdO(6)), float(tdVrdO(7)));
        // }

// #if 1
        // return;
        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);



        // __syncthreads(); // important sync, delete will lead copy to global error
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_V_in_regs, /*B_in_regs=*/false>(
            acc_dp_ori, tdPrV, tdPrdO, tdPsV, tdPsdO, tiled_mma_sdp,
            smem_tiled_copy_KV, smem_tiled_copy_QdO, smem_thr_copy_KV, smem_thr_copy_QdO
        );
        // __syncthreads(); 
        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        Tensor dS = make_tensor(acc_dp.data(), scores_trans.layout());

        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        
        Tensor dP_sum = make_fragment_like(lse);

        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
            dP_sum(mi) = gdPsum(row);
        }
        gdPsum.data() = gdPsum.data() + (-int(kBlockM));
        
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores_trans(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh_trans(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }



        Tensor tdKrdSt = flash::convert_type<Element>(acc_dp);
        flash::gemm_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt,
                smem_thr_copy_QdOt);
        __builtin_amdgcn_s_barrier(); // important sync, delete will lead copy to global error

// #endif

    }

    if constexpr(Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }



    // Convert acc_dv from fp32 to fp16
    Tensor rdK = flash::convert_type<Element>(acc_dk);
    Tensor rdV = flash::convert_type<Element>(acc_dv);

#if 1
    //  __syncthreads();
    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    // If Is_last, there's already a __syncthreads() at the end of the loop.
    // if constexpr(!Is_last) { __syncthreads(); }

    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    __builtin_amdgcn_s_barrier(); 
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    __builtin_amdgcn_s_barrier(); 
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
    __syncthreads();
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    __builtin_amdgcn_s_barrier(); 
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    __builtin_amdgcn_s_barrier(); 
#else 

    // Tensor sdV = make_tensor(sK.data() + size(typename Kernel_traits::SmemLayoutdKV{}), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_) + kBlockN*kHeadDim), typename Kernel_traits::SmemLayoutdKV{});

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
    
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();
    asm volatile("s_nop 5;\n s_waitcnt lgkmcnt(0); \n s_waitcnt vmcnt(0)" ::);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdV), size<1>(sdV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdVgdV)));
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );


    Tensor sdK = make_tensor(sdV.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    
    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    
    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
#endif
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_dv_trans_1colblock_16x64_dim40(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;


    constexpr int MMA_N_SdP = kBlockN / size<1>(typename Kernel_traits::TiledMmaSdP::TiledShape_MNK{});
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    if constexpr (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }


    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.o_row_stride + bidh * params.o_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;
    
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));

    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});
    

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});

    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});

    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(),
                typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
#if 0
    Tensor sK = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
#else
    Tensor sK = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
#endif
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    using GmemTiledCopydO = typename Kernel_traits::GmemTiledCopydO;
    GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrK = thr_mma_sdp.partition_fragment_A(sK);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_B(sQ);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_A(sV);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_B(sdO);

    // dV/dK
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);




    //
    // Copy Atom retiling
    //

    // S/dP
    auto smem_tiled_copy_KV = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    Tensor tRgK = smem_thr_copy_KV.partition_S(gK);
    Tensor tRgV = smem_thr_copy_KV.partition_S(gV);

    auto smem_tiled_copy_QdO = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

#if 0
    // dV/dK
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dkv);
#else
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dkv);
#endif
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);

#if 1
    // debug
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
#else
    // Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOtransposedNoSwizzle);
#endif
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = smem_thr_copy_KV.partition_D(cKV);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);
    
    // If not local, we're guaranteed that m_block_min <= m_block:
    // We checked earlier that n_block * kBlockN < actual_seqlen_k, so in the causal case,
    // n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k < actual_seqlen_q.
    // So m_block_min <= (actual_seqlen_q - 1) / kBlockM.
    // Recall that m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM) = (actual_seqlen_q + kBlockM - 1) / kBlockM.
    // So m_block_m - 1 = (actual_seqlen_q - 1) / kBlockM.
    // We conclude that m_block_min <= m_block, so we will always have at least 1 iteration of the for loop.
    // However, if local, then this possible to have some blocks of K & V not attending to any query.
    // We might need to exit early and write 0 to dK and dV for those blocks.
    // Otherwise we get wrong result for the case where we don't enter the for loop.
    // And we might read OOB elements from gQ and gdO.
    // This also covers the case where actual_seqlen_q == 0
    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdKV = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }
    
    
#if 0
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    if constexpr(true) 
    {
        // cute::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrK_copy_view = smem_thr_copy_KV.retile_D(tSrK);
        cute::copy(smem_tiled_copy_KV, tSsK, tSrK_copy_view);
    }
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );


    if constexpr(Kernel_traits::Is_V_in_regs) {
        // cute::cp_async_wait<1>();
        __syncthreads();
        Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));            // M
        cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
    }

    __syncthreads();
#else
    Tensor trsK = smem_thr_copy_KV.partition_D(gK);
    Tensor trpKV = make_tensor<bool>(make_shape(size<2>(trsK)));

    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(trpKV); ++k) { trpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }
    Tensor tSrK_copy_view = smem_thr_copy_KV.retile_D(tSrK);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        smem_tiled_copy_KV, tRgK, tSrK_copy_view, tKVcKV, trpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );

    Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        smem_tiled_copy_KV, tRgV, tdPrV_copy_view, tKVcKV, trpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    // __syncthreads();
#endif

    Tensor tdOrdO = make_fragment_like(tdOgdO);
    Tensor tdOrO = make_fragment_like(tdOgO);

    // flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
    //     gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    // );
    // wangaq debug
    Tensor caccS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);  

    
    

    int laneId = tidx % 64;

    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
        bidb, bidh, tidx, params.h);

    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    clear(acc_dv);
    clear(acc_dk);
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    for (; m_block >= m_block_min; m_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{}); 
        clear(acc_s_ori);
        if (m_block == m_block_max - 1) {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
            );
        }
        else {
            flash::copy<true, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ
            );
        }
        
        tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
        __syncthreads();

        Tensor taccScS_row = taccScS(_, 0, _);
        Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});

        if (m_block == m_block_max - 1) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                // dim64的时候,lse是16,glse是64
                // 此时线程布局是 t0 t0 t0 t0 t16 t16 t16 t16 t32 t32 t32 t32 t48 t48 t48 t48 t0 t0 t0 t0..................
                // 按照上述格式进行线程的映射
                const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
                lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
            }
        } else {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
                lse(mi) = gLSE(row);
            }
        }

        gLSE.data() = gLSE.data() + (-int(kBlockM));


        __syncthreads();
        flash::gemm<true, false>(acc_s_ori, tSrK, tSrQ, tSsK, tSsQ, tiled_mma_sdp,
            smem_tiled_copy_KV, smem_tiled_copy_QdO, smem_thr_copy_KV, smem_thr_copy_QdO);
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        Tensor scores_trans = make_tensor(acc_s.data(), flash::convert_trans_layout_acc_rowcol(acc_s.layout()));
        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        Tensor dtanh_trans = make_tensor_like(scores_trans);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores_trans, dtanh_trans, params.softcap);
        }

        #if 1
            if (Has_alibi) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = tidx / 64;
                const int col_idx_offset =  m_block * kBlockM;
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                alibi.apply_alibi_trans(scores, col_idx_offset, row_idx_offset_, AtomLayoutMS * 16);
            }
        #endif

        // TD [2023-07-29]: I was thinking that we don't need to mask out the elements beyond
        // actual_seqlen_k, because acc_s would be some finite value for those indices.
        // In the end when we multiply with K to get dQ, the corresponding values of K would be 0,
        // so the result would still be correct.
        // However, it's possible that the values in acc_s are so large that they overflow
        // when we multiply with dP and convert to fp16, resulting in Inf in dS and NaNs in dQ.
        // So we need to mask out the elements beyond actual_seqlen_k.

        #if 1
        if constexpr(!Is_causal && !Is_local) {
            if (!Is_even_MN && (m_block + 1) * kBlockM >= binfo.actual_seqlen_q) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int warp_id = tidx / 64;
                // 实际上是row
                const int col_idx_offset_ = m_block * kBlockM;
                flash::apply_mask_trans(scores, binfo.actual_seqlen_q, col_idx_offset_);
            }
        } else if constexpr(Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            // if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
            //     || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
            //     const int warp_id = tidx / 64;
            //     flash::apply_mask_causal(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
            //                              binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
            //                              binfo.actual_seqlen_q,
            //                              AtomLayoutMS * 16);
            // }

            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k)
            {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_causal_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    AtomLayoutMS * 16
                );
            }
        } else if constexpr(Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_local_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    AtomLayoutMS * 16,
                    params.window_size_left, params.window_size_right
                );
            }
        }
        #endif



        flash::scale_apply_exp2</*scale_max=*/false>(scores_trans, lse, params.scale_softmax_log2);
        

        if constexpr (Is_dropout) {
            const int warp_id = tidx / 64;
            // const int warp_row_stride = 16;
            // int block_row_idx = n_block * (kBlockN / 16) + warp_id % AtomLayoutMS;
            // int block_col_idx = m_block * (kBlockM / 16) + (warp_id / AtomLayoutMS) * MMA_N_SdP;
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            int block_row_idx = row_idx_offset_;
            int block_col_idx = m_block * kBlockM;

            // int block_row_idx = m_block * (kBlockM / 16) + warp_id % AtomLayoutMS;
            // int block_col_idx = n_block * (kBlockN / 16) + (warp_id / AtomLayoutMS) * MMA_N_SdP;
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            // Tensor drop_accs = make_tensor(acc_s.data(), make_layout(get<0>(acc_s.layout()), get<2>(acc_s.layout()), get<1>(acc_s.layout())));
            dropout.template apply_dropout_trans</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS * 16
            );
        }
        #if 1
        if (m_block == m_block_max - 1) {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
            );
        } else {
            flash::copy<true, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ
            );
        }

        tdOgdO.data() = tdOgdO.data() + (-int(kBlockM * params.do_row_stride));
        #endif 

        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);



        __syncthreads(); // important sync, delete will lead copy to global error
        flash::gemm_rs(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt,
            smem_thr_copy_QdOt);
        //  __syncthreads();
        // cute::copy(smem_tiled_copy_QdOt, tdVsdOt, tdVrdO);
        // if (tidx < 64) {
        //     printf("tidx = %d %f %f %f %f %f %f %f %f\n", tidx, float(tdVrdO(0)), float(tdVrdO(1)), float(tdVrdO(2)), float(tdVrdO(3)), 
        //         float(tdVrdO(4)), float(tdVrdO(5)), float(tdVrdO(6)), float(tdVrdO(7)));
        // }

// #if 1
        // return;
        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);

        Tensor dP_sum = make_fragment_like(lse);

        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
            dP_sum(mi) = gdPsum(row);
        }
        gdPsum.data() = gdPsum.data() + (-int(kBlockM));

        __syncthreads(); // important sync, delete will lead copy to global error
        flash::gemm</*A_in_regs=*/Kernel_traits::Is_V_in_regs, /*B_in_regs=*/false>(
            acc_dp_ori, tdPrV, tdPrdO, tdPsV, tdPsdO, tiled_mma_sdp,
            smem_tiled_copy_KV, smem_tiled_copy_QdO, smem_thr_copy_KV, smem_thr_copy_QdO
        );
        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        Tensor dS = make_tensor(acc_dp.data(), scores_trans.layout());

        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        

        
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores_trans(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh_trans(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }



        Tensor tdKrdSt = flash::convert_type<Element>(acc_dp);
        flash::gemm_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt,
                smem_thr_copy_QdOt);
        __syncthreads(); // important sync, delete will lead copy to global error

// #endif

    }

    if constexpr(Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }



    // Convert acc_dv from fp32 to fp16
    Tensor rdK = flash::convert_type<Element>(acc_dk);
    Tensor rdV = flash::convert_type<Element>(acc_dv);

#if 1
    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dv_row_stride, _1{}));
    auto gmem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);

    Tensor taccdKrdK = gmem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKgdK = gmem_thr_copy_dKV.partition_D(gdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = gmem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVgdV = gmem_thr_copy_dKV.partition_D(gdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    Tensor cdKV = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(taccdKgdK)));

    for (int m = 0; m < size<1>(taccdKrdK); m++)
    {
        if (Is_even_MN || get<0>(tdKVcdKV(0, m, 0)) < binfo.actual_seqlen_k - n_block * kBlockN)
        {
            // const int 
            for (int k = 0; k < size<2>(taccdKrdK); k++)
            {
                for (int i = 0; i < size<0>(taccdKrdK); i++)
                {
                    const int col_id = get<1>(tdKVcdKV(0, 0, k)) ;
                    
                    if (Is_even_K || col_id + i * 4 < params.d)
                    {
                        taccdKgdK(i, m, k) = taccdKrdK(i, m, k);
                        taccdVgdV(i, m, k) = taccdVrdV(i, m, k);
                    }
                }
            }
        }
    }

#else
    //  __syncthreads();
    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    // If Is_last, there's already a __syncthreads() at the end of the loop.
    // if constexpr(!Is_last) { __syncthreads(); }

    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    __syncthreads();
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
    __syncthreads();
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    __syncthreads();
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    __syncthreads();
#endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_dv_trans_1colblock_16x64_dim128_fp16(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;


    constexpr int MMA_N_SdP = kBlockN / size<1>(typename Kernel_traits::TiledMmaSdP::TiledShape_MNK{});
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    if constexpr (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }


    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.o_row_stride + bidh * params.o_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;
    
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));

    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});
    

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});

    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});

    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(),
                typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
#if 0
    Tensor sK = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
#else
    Tensor sK = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
#endif
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    using GmemTiledCopydO = typename Kernel_traits::GmemTiledCopydO;
    GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrK = thr_mma_sdp.partition_fragment_A(sK);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_B(sQ);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_A(sV);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_B(sdO);

    // dV/dK
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);




    //
    // Copy Atom retiling
    //

    // S/dP
    auto smem_tiled_copy_KV = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    Tensor tRgK = smem_thr_copy_KV.partition_S(gK);
    Tensor tRgV = smem_thr_copy_KV.partition_S(gV);

    auto smem_tiled_copy_QdO = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

#if 0
    // dV/dK
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dkv);
#else
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dkv);
#endif
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);

#if 1
    // debug
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
#else
    // Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOtransposedNoSwizzle);
#endif
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = smem_thr_copy_KV.partition_D(cKV);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);
    
    // If not local, we're guaranteed that m_block_min <= m_block:
    // We checked earlier that n_block * kBlockN < actual_seqlen_k, so in the causal case,
    // n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k < actual_seqlen_q.
    // So m_block_min <= (actual_seqlen_q - 1) / kBlockM.
    // Recall that m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM) = (actual_seqlen_q + kBlockM - 1) / kBlockM.
    // So m_block_m - 1 = (actual_seqlen_q - 1) / kBlockM.
    // We conclude that m_block_min <= m_block, so we will always have at least 1 iteration of the for loop.
    // However, if local, then this possible to have some blocks of K & V not attending to any query.
    // We might need to exit early and write 0 to dK and dV for those blocks.
    // Otherwise we get wrong result for the case where we don't enter the for loop.
    // And we might read OOB elements from gQ and gdO.
    // This also covers the case where actual_seqlen_q == 0
    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdKV = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }
    
    
#if 0
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    if constexpr(true) 
    {
        // cute::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrK_copy_view = smem_thr_copy_KV.retile_D(tSrK);
        cute::copy(smem_tiled_copy_KV, tSsK, tSrK_copy_view);
    }
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );


    if constexpr(Kernel_traits::Is_V_in_regs) {
        // cute::cp_async_wait<1>();
        __syncthreads();
        Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));            // M
        cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
    }

    __syncthreads();
#else
    Tensor tSrK_copy_view = smem_thr_copy_KV.retile_D(tSrK);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        smem_tiled_copy_KV, tRgK, tSrK_copy_view, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );

    Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        smem_tiled_copy_KV, tRgV, tdPrV_copy_view, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    // __syncthreads();
#endif

    Tensor tdOrdO = make_fragment_like(tdOgdO);
    Tensor tdOrO = make_fragment_like(tdOgO);

    // flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
    //     gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    // );
    // wangaq debug
    Tensor caccS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);  

    
    

    int laneId = tidx % 64;

    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
        bidb, bidh, tidx, params.h);

    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    clear(acc_dv);
    clear(acc_dk);

    auto tQrQ = make_fragment_like(tQsQ);
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tQgQ, tQrQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    );
    
    Tensor taccScS_row = taccScS(_, 0, _);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        // dim64的时候,lse是16,glse是64
        // 此时线程布局是 t0 t0 t0 t0 t16 t16 t16 t16 t32 t32 t32 t32 t48 t48 t48 t48 t0 t0 t0 t0..................
        // 按照上述格式进行线程的映射
        const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    for (; m_block >= m_block_min; m_block--) {
        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{}); 
        clear(acc_s_ori);

        cute::copy(tQrQ, tQsQ);
        __syncthreads();

        auto tdOrdO = make_fragment_like(tdOsdO);
        if (m_block == m_block_max - 1) {
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_dO, tdOgdO, tdOrdO, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
            );

            // lds_direct_copy<Kernel_traits::kNWarps, true, Is_even_K, Is_even_MN>(typename Kernel_traits::GmemLayoutAtom{},
            //     gdO, sdO, params.d, binfo.actual_seqlen_q - m_block * kBlockM
            
            // );
        } else {
            tdOgdO.data() = tdOgdO.data() + (-int(kBlockM * params.do_row_stride));
            flash::copy<true, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_dO, tdOgdO, tdOrdO, tQcQ, tQpQ
            );
            // lds_direct_copy<Kernel_traits::kNWarps, true, Is_even_K, true>(typename Kernel_traits::GmemLayoutAtom{},
            //     gdO, sdO, params.d
            
            // );
        }

        flash::gemm<true, false>(acc_s_ori, tSrK, tSrQ, tSsK, tSsQ, tiled_mma_sdp,
            smem_tiled_copy_KV, smem_tiled_copy_QdO, smem_thr_copy_KV, smem_thr_copy_QdO);
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
        Tensor scores_trans = make_tensor(acc_s.data(), flash::convert_trans_layout_acc_rowcol(acc_s.layout()));
        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        [[maybe_unused]] Tensor dtanh_trans = make_tensor_like(scores_trans);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores_trans, dtanh_trans, params.softcap);
        }

        #if 1
        if constexpr (Has_alibi) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            const int wave_id = tidx / 64;
            const int col_idx_offset =  m_block * kBlockM;
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            alibi.apply_alibi_trans(scores, col_idx_offset, row_idx_offset_, AtomLayoutMS * 16);
        }
        #endif

        // TD [2023-07-29]: I was thinking that we don't need to mask out the elements beyond
        // actual_seqlen_k, because acc_s would be some finite value for those indices.
        // In the end when we multiply with K to get dQ, the corresponding values of K would be 0,
        // so the result would still be correct.
        // However, it's possible that the values in acc_s are so large that they overflow
        // when we multiply with dP and convert to fp16, resulting in Inf in dS and NaNs in dQ.
        // So we need to mask out the elements beyond actual_seqlen_k.

        #if 1
        if constexpr(!Is_causal && !Is_local) {
            if (!Is_even_MN && (m_block + 1) * kBlockM >= binfo.actual_seqlen_q) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int warp_id = tidx / 64;
                // 实际上是row
                const int col_idx_offset_ = m_block * kBlockM;
                flash::apply_mask_trans(scores, binfo.actual_seqlen_q, col_idx_offset_);
            }
        } else if constexpr(Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            // if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
            //     || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
            //     const int warp_id = tidx / 64;
            //     flash::apply_mask_causal(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
            //                              binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
            //                              binfo.actual_seqlen_q,
            //                              AtomLayoutMS * 16);
            // }

            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k)
            {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_causal_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    AtomLayoutMS * 16
                );
            }
        } else if constexpr(Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_local_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    AtomLayoutMS * 16,
                    params.window_size_left, params.window_size_right
                );
            }
        }
        #endif



        flash::scale_apply_exp2</*scale_max=*/false>(scores_trans, lse, params.scale_softmax_log2);
        

        if constexpr (Is_dropout) {
            const int warp_id = tidx / 64;
            // const int warp_row_stride = 16;
            // int block_row_idx = n_block * (kBlockN / 16) + warp_id % AtomLayoutMS;
            // int block_col_idx = m_block * (kBlockM / 16) + (warp_id / AtomLayoutMS) * MMA_N_SdP;
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            int block_row_idx = row_idx_offset_;
            int block_col_idx = m_block * kBlockM;

            // int block_row_idx = m_block * (kBlockM / 16) + warp_id % AtomLayoutMS;
            // int block_col_idx = n_block * (kBlockN / 16) + (warp_id / AtomLayoutMS) * MMA_N_SdP;
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            // static_assert(MMA_N_SdP % 2 == 0);
            // Tensor drop_accs = make_tensor(acc_s.data(), make_layout(get<0>(acc_s.layout()), get<2>(acc_s.layout()), get<1>(acc_s.layout())));
            dropout.template apply_dropout_trans</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS * 16
            );
        }

        cute::copy(tdOrdO, tdOsdO);
        __syncthreads();

        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);

        flash::gemm_rs(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt,
            smem_thr_copy_QdOt);
        //  __syncthreads();
        // cute::copy(smem_tiled_copy_QdOt, tdVsdOt, tdVrdO);
        // if (tidx < 64) {
        //     printf("tidx = %d %f %f %f %f %f %f %f %f\n", tidx, float(tdVrdO(0)), float(tdVrdO(1)), float(tdVrdO(2)), float(tdVrdO(3)), 
        //         float(tdVrdO(4)), float(tdVrdO(5)), float(tdVrdO(6)), float(tdVrdO(7)));
        // }

// #if 1
        // return;
        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);

        Tensor dP_sum = make_fragment_like(lse);

        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
            dP_sum(mi) = gdPsum(row);
        }
        gdPsum.data() = gdPsum.data() + (-int(kBlockM));

        if (m_block > m_block_min) {
            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            flash::copy<true, Is_even_K, /*Clear_OOB_MN=*/true>(
                        gmem_tiled_copy_QKV, tQgQ, tQrQ, tQcQ, tQpQ
            );
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
                lse(mi) = gLSE(row);
            }
        }

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_V_in_regs, /*B_in_regs=*/false>(
            acc_dp_ori, tdPrV, tdPrdO, tdPsV, tdPsdO, tiled_mma_sdp,
            smem_tiled_copy_KV, smem_tiled_copy_QdO, smem_thr_copy_KV, smem_thr_copy_QdO
        );
        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        Tensor dS = make_tensor(acc_dp.data(), scores_trans.layout());

        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        

        
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores_trans(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh_trans(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }




        Tensor tdKrdSt = flash::convert_type<Element>(acc_dp);

        // if (thread0())
        // {
        //     printf("tdKrdSt = %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n ",
        //     float(tdKrdSt(0)),
        //     float(tdKrdSt(1)),
        //     float(tdKrdSt(2)),
        //     float(tdKrdSt(3)),

        //     float(tdKrdSt(4 + 0)),
        //     float(tdKrdSt(4 + 1)),
        //     float(tdKrdSt(4 +2)),
        //     float(tdKrdSt(4 + 3)),
            
        //     float(tdKrdSt(8 + 0)),
        //     float(tdKrdSt(8 + 1)),
        //     float(tdKrdSt(8 + 2)),
        //     float(tdKrdSt(8 + 3)),
            
        //     float(tdKrdSt(12)),
        //     float(tdKrdSt(13)),
        //     float(tdKrdSt(14)),
        //     float(tdKrdSt(15))
            
        //     );
        // }
        //  __builtin_amdgcn_sched_barrier(0);
        //  __syncthreads();

        flash::gemm_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt,
                smem_thr_copy_QdOt);
        __builtin_amdgcn_s_barrier(); // important sync, delete will lead copy to global error

// #endif

    }

    if constexpr(Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }



    // Convert acc_dv from fp32 to fp16
    Tensor rdK = flash::convert_type<Element>(acc_dk);
    Tensor rdV = flash::convert_type<Element>(acc_dv);

    //  __syncthreads();
    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    // If Is_last, there's already a __syncthreads() at the end of the loop.
    // if constexpr(!Is_last) { __syncthreads(); }

    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    __builtin_amdgcn_s_barrier(); 
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
__builtin_amdgcn_s_barrier(); 
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
    __syncthreads();
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    __builtin_amdgcn_s_barrier(); 
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    __builtin_amdgcn_s_barrier(); 

}


#define S_WAITCNT asm volatile("s_waitcnt vmcnt(3) \n s_barrier")
#define S_BARRIER asm volatile("s_barrier")
template <typename Kernel_traits,
    typename RTensor,
    typename STensor,
    typename GTensor,
    typename Element,
    typename SmemLayout, 
    bool Is_even_MN, bool Is_even_K
>
inline __device__ void _bwd_store_dk_dv(RTensor& rAcc, STensor& sAcc, int tidx, GTensor& gdAcc, int dim, int max_MN) {
    // Convert acc_dv from fp32 to fp16
    Tensor rdAcc = flash::convert_type<Element>(rAcc);

    Tensor sdAcc = make_tensor(sAcc.data(), SmemLayout{});  // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, typename Kernel_traits::TiledMmadKV{});
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKVrdKV = smem_thr_copy_dKV.retile_S(rAcc);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKVsdKV = smem_thr_copy_dKV.partition_D(sdAcc);   // ((Atom,AtomNum),PIPE_M,PIPE_N)


    cute::copy(smem_tiled_copy_dKV,  flash::convert_type<Element>(taccdKVrdKV), taccdKVsdKV);
    S_BARRIER;

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKVsdKV = gmem_thr_copy_dKV.partition_S(sdAcc);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKVgdKV = gmem_thr_copy_dKV.partition_D(gdAcc);

    __syncthreads();
    Tensor tdKVrdKV = make_tensor<Element>(shape(tdKVgdKV));
    cute::copy(gmem_tiled_copy_dKV, tdKVsdKV, tdKVrdKV);
    S_BARRIER;
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdAcc), size<1>(sdAcc)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKVgdKV)));
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < dim; }
    S_BARRIER;
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKVrdKV, tdKVgdKV, tdKVcdKV, tdKVpdKV, max_MN
    );
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_dv_trans_1colblock_16x64_dim64_prefetch(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;

    constexpr int kSmemOffset = Kernel_traits::kSmemOffset;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    if constexpr (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }


    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;
    
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(params.do_row_stride, _1{}));

    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});
    

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQGemm0{});
    Tensor sQt = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutQGemm1transposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQGemm1transposedNoSwizzle{});

    Tensor sdO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutdOGemm0{});
    Tensor sdOt = make_tensor(sdO.data() + size(sQ), typename Kernel_traits::SmemLayoutdOGemm1transposed{});
    Tensor sdOtNoSwizzle = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutdOGemm1transposedNoSwizzle{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrK = thr_mma_sdp.partition_fragment_A(gK);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_B(sQ);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_A(gV);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_B(sdO);

    // dV/dK
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtNoSwizzle);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);

    //
    // Copy Atom retiling
    //

    // S/dP
    auto gmem_tiled_copy_KV = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSgK = gmem_thr_copy_KV.partition_S(gK);
    Tensor tdPgV = gmem_thr_copy_KV.partition_S(gV);
    
    // auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_sdp);
    auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    // auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_BLayout);
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sQtemp = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQ{});
    Tensor tSsQBLayout = smem_thr_copy_BLayout.partition_S(sQtemp);
    Tensor tSsQ = make_tensor(tSsQBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDim/32>(tSsQBLayout.layout()));
    Tensor sdOtemp = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutdO{});
    Tensor tdPsdOBLayout = smem_thr_copy_BLayout.partition_S(sdOtemp);
    Tensor tdPsdO = make_tensor(tdPsdOBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDimV/32>(tdPsdOBLayout.layout()));

    // dV/dK
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16_WITH_8x64, Element>{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt8x64 = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdVsdOt = make_tensor(tdVsdOt8x64.data(), convert_layout_B_rowcol<_16x64_64>(tdVsdOt8x64.layout()));
    Tensor tdKsQt8x64 = smem_thr_copy_QdOt.partition_S(sQt);
    Tensor tdKsQt = make_tensor(tdKsQt8x64.data(), convert_layout_B_rowcol<_16x64_64>(tdKsQt8x64.layout()));

    //
    // PREDICATES
    //

    Tensor cK = make_identity_tensor(make_shape(size<0>(gK), size<1>(gK)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(gV), size<1>(gV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKcK = gmem_thr_copy_KV.partition_D(cK);
    Tensor tVcV = gmem_thr_copy_KV.partition_D(cV);

    // Allocate predicate tensors for k
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tSgK)));
    Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tdPgV)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) < params.d; }
    }

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);

    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                                make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdK = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor cdV = make_identity_tensor(make_shape(size<0>(gdV), size<1>(gdV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKcdK = gmem_thr_copy_dKV.partition_D(cdK);
        Tensor tdVcdV = gmem_thr_copy_dKV.partition_D(cdV);
        Tensor tdKpdK = make_tensor<bool>(make_shape(size<2>(tdKcdK)));
        Tensor tdVpdV = make_tensor<bool>(make_shape(size<2>(tdVcdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKpdK); ++k) { tdKpdK(k) = get<1>(tdKcdK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdVpdV); ++k) { tdVpdV(k) = get<1>(tdVcdV(0, 0, k)) < params.d; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKcdK, tdKpdK, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdVcdV, tdVpdV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }
    
    
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tSgK, tSrK, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN
    );

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tdPgV, tdPrV, tVcV, tVpV, binfo.actual_seqlen_k - n_block * kBlockN
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);  
    
    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
        bidb, bidh, tidx, params.h);

    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDimV>>{});
    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    clear(acc_dv);
    clear(acc_dk);
    
    Tensor taccScS_row = taccScS(_, 0, _);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    constexpr int kS_loops = size<2>(tSsQ);
    constexpr int kdV_loops = size<2>(tdVsdOt);
    constexpr int kdP_loops = size<2>(tdPsdO);
    constexpr int kdK_loops = size<2>(tdKsQt);
    // static_assert(kStages <= kS_loops && kStages <= kdV_loops && kStages <= kdP_loops && kStages <= kdK_loops, "kStages is error");
    if constexpr(Is_even_K) {
        lds_direct_copy_even_k<0, Is_even_MN>(gQ, sQ, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy_even_k<1, Is_even_MN>(gQ, sQ, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);

        lds_direct_copy_even_k<0, Is_even_MN, _16x64_64>(gdO, sdOt, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy_even_k<1, Is_even_MN, _16x64_64>(gdO, sdOt, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy_even_k<2, Is_even_MN, _16x64_64>(gdO, sdOt, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy_even_k<3, Is_even_MN, _16x64_64>(gdO, sdOt, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);

    } else {
        lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, 0, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, 1, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);

        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gdO, sdOt, 0, params.do_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gdO, sdOt, 1, params.do_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gdO, sdOt, 2, params.do_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gdO, sdOt, 3, params.do_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);

    }

    #pragma unroll
    for (; m_block >= m_block_min; m_block--) {

        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{}); 
        clear(acc_s_ori);

        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 0);
        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 1);
        asm volatile("s_barrier");
        
        
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("lse tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block, lse(0), lse(1), lse(2), lse(3));
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("acc_s tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        Tensor scores_trans = make_tensor(acc_s.data(), flash::convert_trans_layout_acc_rowcol(acc_s.layout()));
        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        [[maybe_unused]] Tensor dtanh_trans = make_tensor_like(scores_trans);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores_trans, dtanh_trans, params.softcap);
        }

        #if 1
        if constexpr (Has_alibi) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            const int wave_id = tidx / 64;
            const int col_idx_offset =  m_block * kBlockM;
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            alibi.apply_alibi_trans(scores, col_idx_offset, row_idx_offset_, kNWarps * 16);
        }
        #endif

        #if 1
        if constexpr(!Is_causal && !Is_local) {
            if (!Is_even_MN && (m_block + 1) * kBlockM >= binfo.actual_seqlen_q) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int warp_id = tidx / 64;
                // 实际上是row
                const int col_idx_offset_ = m_block * kBlockM;
                flash::apply_mask_trans(scores, binfo.actual_seqlen_q, col_idx_offset_);
            }
        } else if constexpr(Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            // if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
            //     || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
            //     const int warp_id = tidx / 64;
            //     flash::apply_mask_causal(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
            //                              binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
            //                              binfo.actual_seqlen_q,
            //                              AtomLayoutMS * 16);
            // }

            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k)
            {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_causal_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16
                );
            }
        } else if constexpr(Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_local_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    params.window_size_left, params.window_size_right
                );
            }
        }
        #endif
        
        flash::scale_apply_exp2</*scale_max=*/false>(scores_trans, lse, params.scale_softmax_log2);

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("P tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }
        Tensor dP_sum = make_fragment_like(lse);

        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
            dP_sum(mi) = gdPsum(row);
        }
        if (m_block > m_block_min) {
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
                lse(mi) = gLSE(row);
            }
        }

        if constexpr (Is_dropout) {
            const int warp_id = tidx / 64;
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            int block_row_idx = row_idx_offset_;
            int block_col_idx = m_block * kBlockM;
            if constexpr (kHeadDim==64){
                dropout.template apply_dropout_trans_dim64_opt</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, n_block * kBlockN, m_block * kBlockM, kNWarps * 16
                );
            }else{
                dropout.template apply_dropout_trans</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, block_row_idx, block_col_idx, kNWarps * 16
                );
            }

        }

        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);
        if constexpr(Is_even_K) {
            lds_direct_copy_even_k<0, Is_even_MN>(gdO, sdO, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
            lds_direct_copy_even_k<1, Is_even_MN>(gdO, sdO, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
            lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        }


        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_barrier");
        if constexpr(Is_even_K) {
            lds_direct_copy_even_k<0, Is_even_MN, _16x64_64>(gQ, sQt, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
            lds_direct_copy_even_k<1, Is_even_MN, _16x64_64>(gQ, sQt, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
            lds_direct_copy_even_k<2, Is_even_MN, _16x64_64>(gQ, sQt, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
            lds_direct_copy_even_k<3, Is_even_MN, _16x64_64>(gQ, sQt, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gQ, sQt, 0, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gQ, sQt, 1, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gQ, sQt, 2, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            lds_direct_copy<Is_even_K, Is_even_MN, _16x64_64>(gQ, sQt, 3, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        // return;
        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);

        asm volatile("s_waitcnt vmcnt(5) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 0);
        asm volatile("s_waitcnt vmcnt(4) \n s_barrier");
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 1);
        asm volatile("s_barrier");
 
        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        Tensor dS = make_tensor(acc_dp.data(), scores_trans.layout());

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("dP_sum tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block, dP_sum(0), dP_sum(1), dP_sum(2), dP_sum(3));
        //     float * tmp = reinterpret_cast<float*>(acc_dp.data());
        //     printf("dP tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores_trans(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh_trans(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_dp.data());
        //     printf("dS tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        Tensor tdKrdSt = flash::convert_type<Element>(acc_dp);
        asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        asm volatile("s_barrier");



        if (m_block > m_block_min) {
            gQ.data() = gQ.data() + (-int(kBlockM * params.q_row_stride));
            gdO.data() = gdO.data() + (-int(kBlockM * params.do_row_stride));

            if constexpr(Is_even_K) {
                lds_direct_copy_even_k<0, true>(gQ, sQ, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
                lds_direct_copy_even_k<1, true>(gQ, sQ, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);

                lds_direct_copy_even_k<0, true, _16x64_64>(gdO, sdOt, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
                lds_direct_copy_even_k<1, true, _16x64_64>(gdO, sdOt, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
                lds_direct_copy_even_k<2, true, _16x64_64>(gdO, sdOt, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
                lds_direct_copy_even_k<3, true, _16x64_64>(gdO, sdOt, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
            } else {
                lds_direct_copy<Is_even_K, true>(gQ, sQ, 0, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
                lds_direct_copy<Is_even_K, true>(gQ, sQ, 1, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            
                lds_direct_copy<Is_even_K, true, _16x64_64>(gdO, sdOt, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
                lds_direct_copy<Is_even_K, true, _16x64_64>(gdO, sdOt, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
                lds_direct_copy<Is_even_K, true, _16x64_64>(gdO, sdOt, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
                lds_direct_copy<Is_even_K, true, _16x64_64>(gdO, sdOt, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
            }

    
        }

    }

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.dv_row_stride, _1{}));
    
    int row, col;
    if constexpr (size<1>(acc_dk) == size<1>(acc_dv) && size<2>(acc_dk) == size<2>(acc_dv)) {
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dk); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk); ++ni) {
                    col = (laneId / 16) * 2 + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < 4; ++ei) {
                        using result_type = cutlass::Array<Element, 2>;
                        if constexpr (Is_even_K)
                        {
                            result_type res;
                            res[0] = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                            res[1] = flash::convert_type<Element>(acc_dk(ei + 4, mi, ni) * params.scale_softmax_rp_dropout);
                            *(result_type*)(&gdK(row, col)) = res;

                            res[0] = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout);
                            res[1] = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei + 4, mi, ni) : acc_dv(ei + 4, mi, ni) * params.rp_dropout);
                            *(result_type*)(&gdV(row, col)) = res;
                        }
                        else
                        {
                            result_type res;
                            res[0] = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                            res[1] = flash::convert_type<Element>(acc_dk(ei + 4, mi, ni) * params.scale_softmax_rp_dropout);
                            if (col < params.d)
                            {
                                gK(row, col) = res[0];
                            }
                            if (col + 1 < params.d)
                            {
                                gK(row, col + 1) = res[1];
                            }

                            res[0] = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout);
                            res[1] = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei + 4, mi, ni) : acc_dv(ei + 4, mi, ni) * params.rp_dropout);

                            if (col < params.d)
                            {
                                gV(row, col) = res[0];
                            }
                            if (col + 1 < params.d)
                            {
                                gV(row, col + 1) = res[1];
                            }
                        }
                        col += 8;
                    }
                }
            }
        } 
    } else {
        
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dk); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk); ++ni) {
                    col = (laneId / 16) * 2 + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < 4; ++ei) {
                        using result_type = cutlass::Array<Element, 2>;
                        if constexpr (Is_even_K)
                        {
                            result_type res;
                            res[0] = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                            res[1] = flash::convert_type<Element>(acc_dk(ei + 4, mi, ni) * params.scale_softmax_rp_dropout);
                            *(result_type*)(&gdK(row, col)) = res;
                        }
                        else
                        {
                            result_type res;
                            res[0] = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                            res[1] = flash::convert_type<Element>(acc_dk(ei + 4, mi, ni) * params.scale_softmax_rp_dropout);
                            if (col < params.d)
                            {
                                gK(row, col) = res[0];
                            }
                            if (col + 1 < params.d)
                            {
                                gK(row, col + 1) = res[1];
                            }
                        }
                        col += 8;
                    }
                }
            }
        } 
        
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dv); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dv); ++ni) {
                    col = (laneId / 16) * 2 + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < 4; ++ei) {
                        using result_type = cutlass::Array<Element, 2>;
                        if constexpr (Is_even_K)
                        {
                            result_type res;
                            res[0] = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout);
                            res[1] = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei + 4, mi, ni) : acc_dv(ei + 4, mi, ni) * params.rp_dropout);
                            *(result_type*)(&gdV(row, col)) = res;
                        }
                        else
                        {
                            result_type res;
                            res[0] = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout);
                            res[1] = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei + 4, mi, ni) : acc_dv(ei + 4, mi, ni) * params.rp_dropout);

                            if (col < params.d)
                            {
                                gV(row, col) = res[0];
                            }
                            if (col + 1 < params.d)
                            {
                                gV(row, col + 1) = res[1];
                            }
                        }
                        col += 8;
                    }
                }
            }
        } 
    }

}

#if 0
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_dv_trans_1colblock_16x64_dim96_prefetch(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages - 1;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    if constexpr (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }


    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;
    
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(params.do_row_stride, _1{}));

    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});
    

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQGemm1transposed{});
    Tensor sQtSplit = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQtransSplit{});

    Tensor sdO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutdOGemm1transposed{});
    Tensor sdOtSplit = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutdOtransSplit{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrK = thr_mma_sdp.partition_fragment_A(gK);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_B(sQ);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_A(gV);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_B(sdO);

    // dV/dK
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOt);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQt);

    //
    // Copy Atom retiling
    //

    // S/dP
    auto gmem_tiled_copy_KV = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSgK = gmem_thr_copy_KV.partition_S(gK);
    Tensor tdPgV = gmem_thr_copy_KV.partition_S(gV);
    
    // auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_sdp);
    auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    // dV/dK
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt8x64 = smem_thr_copy_QdOt.partition_S(sdOtSplit);
    Tensor tdVsdOt = make_tensor(tdVsdOt8x64.data(), convert_layout_B_rowcol_<_16x96, kHeadDimV/32>(tdVsdOt8x64.layout()));
    Tensor tdKsQt8x64 = smem_thr_copy_QdOt.partition_S(sQtSplit);
    Tensor tdKsQt = make_tensor(tdKsQt8x64.data(), convert_layout_B_rowcol_<_16x96, kHeadDim/32>(tdKsQt8x64.layout()));

    //
    // PREDICATES
    //

    Tensor cK = make_identity_tensor(make_shape(size<0>(gK), size<1>(gK)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(gV), size<1>(gV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKcK = gmem_thr_copy_KV.partition_D(cK);
    Tensor tVcV = gmem_thr_copy_KV.partition_D(cV);

    // Allocate predicate tensors for k
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tSgK)));
    Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tdPgV)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) < params.d_value; }
    }

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);

    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                                make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdK = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor cdV = make_identity_tensor(make_shape(size<0>(gdV), size<1>(gdV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKcdK = gmem_thr_copy_dKV.partition_D(cdK);
        Tensor tdVcdV = gmem_thr_copy_dKV.partition_D(cdV);
        Tensor tdKpdK = make_tensor<bool>(make_shape(size<2>(tdKcdK)));
        Tensor tdVpdV = make_tensor<bool>(make_shape(size<2>(tdVcdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKpdK); ++k) { tdKpdK(k) = get<1>(tdKcdK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdVpdV); ++k) { tdVpdV(k) = get<1>(tdVcdV(0, 0, k)) < params.d_value; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKcdK, tdKpdK, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdVcdV, tdVpdV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }
    
    
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tSgK, tSrK, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN
    );

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tdPgV, tdPrV, tVcV, tVpV, binfo.actual_seqlen_k - n_block * kBlockN
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);  
    
    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
        bidb, bidh, tidx, params.h);

    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDimV>>{});
    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    clear(acc_dv);
    clear(acc_dk);
    
    Tensor taccScS_row = taccScS(_, 0, _);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    constexpr int kS_loops = size<2>(tSsQ);
    constexpr int kdV_loops = size<2>(tdVsdOt);
    constexpr int kdP_loops = size<2>(tdPsdO);
    constexpr int kdK_loops = size<2>(tdKsQt);
    static_assert(kStages <= kS_loops && kStages <= kdV_loops && kStages <= kdP_loops && kStages <= kdK_loops, "kStages is error");
    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    }
    #pragma unroll
    for (; m_block >= m_block_min; m_block--) {

        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < kS_loops - kStages; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, kStages + i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            s_waitcnt<2>();
            flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, i);
            S_BARRIER;
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, i, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
            s_waitcnt<2>();
            flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, kS_loops - kStages + i);
            S_BARRIER;
        }
        
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("lse tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block, lse(0), lse(1), lse(2), lse(3));
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("acc_s tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        Tensor scores_trans = make_tensor(acc_s.data(), flash::convert_trans_layout_acc_rowcol(acc_s.layout()));
        [[maybe_unused]] Tensor dtanh_trans = make_tensor_like(scores_trans);
        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
            flash::calculate_dtanh(scores_trans, dtanh_trans, params.softcap);
        }

        #if 1
        if constexpr (Has_alibi) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            const int wave_id = tidx / 64;
            const int col_idx_offset =  m_block * kBlockM;
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            alibi.apply_alibi_trans(scores, col_idx_offset, row_idx_offset_, kNWarps * 16);
        }
        #endif

        #if 1
        if constexpr(!Is_causal && !Is_local) {
            if (!Is_even_MN && (m_block + 1) * kBlockM >= binfo.actual_seqlen_q) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int warp_id = tidx / 64;
                // 实际上是row
                const int col_idx_offset_ = m_block * kBlockM;
                flash::apply_mask_trans(scores, binfo.actual_seqlen_q, col_idx_offset_);
            }
        } else if constexpr(Is_causal) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k)
            {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_causal_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16
                );
            }
        } else if constexpr(Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_local_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    params.window_size_left, params.window_size_right
                );
            }
        }
        #endif
        
        flash::scale_apply_exp2</*scale_max=*/false>(scores_trans, lse, params.scale_softmax_log2);

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("P tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        Tensor dP_sum = make_fragment_like(lse);

        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
            dP_sum(mi) = gdPsum(row);
        }
        if (m_block > m_block_min) {
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
                lse(mi) = gLSE(row);
            }
        }

        if constexpr (Is_dropout) {
            const int warp_id = tidx / 64;
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            int block_row_idx = row_idx_offset_;
            int block_col_idx = m_block * kBlockM;
            dropout.template apply_dropout_trans</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, kNWarps * 16
            );
        }

        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);

        lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<2>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 0);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gdO, sdOt, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 1);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gdO, sdOt, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<4>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 2);
        S_BARRIER;

        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("dP_sum tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block, dP_sum(0), dP_sum(1), dP_sum(2), dP_sum(3));
        //     float * tmp = reinterpret_cast<float*>(acc_dp.data());
        //     printf("dP tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }
        Tensor dS = make_tensor(acc_dp.data(), scores_trans.layout());

        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores_trans(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh_trans(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_dp.data());
        //     printf("dS tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        
        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gdO, sdOt, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gdO, sdOt, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<6>();
        flash::gemm_k_rs(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, 0);
        S_BARRIER;

        #pragma unroll
        for (int i = 0; i < 3; ++i) { // tail kStages
            lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gQ, sQt, i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            s_waitcnt<6>();
            flash::gemm_k_rs(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, 1 + i);
            S_BARRIER;
        }

        Tensor tdKrdSt = flash::convert_type<Element>(acc_dp);

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gQ, sQt, 3, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<6>();
        flash::gemm_k_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, 0);
        S_BARRIER;

        s_waitcnt<4>();
        flash::gemm_k_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, 1);
        S_BARRIER;
        // k = 2
        s_waitcnt<2>();
        flash::gemm_k_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, 2);
        S_BARRIER;
        // k = 3
        s_waitcnt<0>();
        flash::gemm_k_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, 3);
        S_BARRIER;

        if (m_block > m_block_min) {
            gQ.data() = gQ.data() + (-int(kBlockM * params.q_row_stride));
            gdO.data() = gdO.data() + (-int(kBlockM * params.do_row_stride));
            #pragma unroll
            for (int i = 0; i < kStages; ++i) { // tail kStages
                lds_direct_copy<Is_even_K>(gQ, sQ, i, params.q_row_stride, params.d);
            }
        } 

    }

    // wangaq debug
    // __syncthreads();
    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     float * tmp = reinterpret_cast<float*>(acc_dk.data());
    //     printf("dK tid:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, 
    //     tmp[0], tmp[1], tmp[2], tmp[3], 
    //     tmp[4], tmp[5], tmp[6], tmp[7],
    //     tmp[8], tmp[9], tmp[10], tmp[11], 
    //     tmp[12], tmp[13], tmp[14], tmp[15],
    //     tmp[16], tmp[17], tmp[18], tmp[19], 
    //     tmp[20], tmp[21], tmp[22], tmp[23],
    //     tmp[24], tmp[25], tmp[26], tmp[27], 
    //     tmp[28], tmp[29], tmp[30], tmp[31]
    //     );
    // }

    #if 0
    if constexpr(Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dv from fp32 to fp16
    Tensor rdK = flash::convert_type<Element>(acc_dk);
    Tensor rdV = flash::convert_type<Element>(acc_dv);

    //  __syncthreads();
    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    // If Is_last, there's already a __syncthreads() at the end of the loop.
    // if constexpr(!Is_last) { __syncthreads(); }

    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    __builtin_amdgcn_s_barrier(); 
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
__builtin_amdgcn_s_barrier(); 
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
    __syncthreads();
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    __builtin_amdgcn_s_barrier(); 
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    __builtin_amdgcn_s_barrier(); 
    #elif 0
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.dv_row_stride, _1{}));
    _bwd_store_dk_dv<Kernel_traits, decltype(acc_dv), decltype(sQ), decltype(gdV), Element, 
        typename Kernel_traits::SmemLayoutdVStore, Is_even_MN, Is_even_K>(
        acc_dv, sQ, tidx, gdV, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);

    __syncthreads();
    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    _bwd_store_dk_dv<Kernel_traits, decltype(acc_dk), decltype(sQ), decltype(gdK), Element, 
        typename Kernel_traits::SmemLayoutdKStore, Is_even_MN, Is_even_K>(
        acc_dk, sQ, tidx, gdK, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

    #else

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.dv_row_stride, _1{}));
    
    int row, col;
    if constexpr (size<1>(acc_dk) == size<1>(acc_dv) && size<2>(acc_dk) == size<2>(acc_dv)) {
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dk); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dk); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdK(row, col) = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                            gdV(row, col) = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout );
                        }
                        col += 4;
                    }
                }
            }
        } 
    } else {
        
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dk); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dk); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdK(row, col) = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                        }
                        col += 4;
                    }
                }
            }
        } 
        
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dv); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dv); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dv); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdV(row, col) = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout);
                        }
                        col += 4;
                    }
                }
            }
        } 
    }

    #endif

}
#else
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_dv_trans_1colblock_16x64_dim96_prefetch(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages - 1;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    if constexpr (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }


    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;
    
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(params.do_row_stride, _1{}));

    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});
    

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQGemm1transposed{});
    Tensor sQtSplit = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQtransSplit{});

    Tensor sdO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutdOGemm1transposed{});
    Tensor sdOtSplit = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutdOtransSplit{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrK = thr_mma_sdp.partition_fragment_A(gK);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_B(sQ);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_A(gV);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_B(sdO);

    // dV/dK
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOt);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQt);

    //
    // Copy Atom retiling
    //

    // S/dP
    auto gmem_tiled_copy_KV = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSgK = gmem_thr_copy_KV.partition_S(gK);
    Tensor tdPgV = gmem_thr_copy_KV.partition_S(gV);
    
    // auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_sdp);
    auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    // dV/dK
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt8x64 = smem_thr_copy_QdOt.partition_S(sdOtSplit);
    Tensor tdVsdOt = make_tensor(tdVsdOt8x64.data(), convert_layout_B_rowcol_<_16x96, kHeadDimV/32>(tdVsdOt8x64.layout()));
    Tensor tdKsQt8x64 = smem_thr_copy_QdOt.partition_S(sQtSplit);
    Tensor tdKsQt = make_tensor(tdKsQt8x64.data(), convert_layout_B_rowcol_<_16x96, kHeadDim/32>(tdKsQt8x64.layout()));

    //
    // PREDICATES
    //

    Tensor cK = make_identity_tensor(make_shape(size<0>(gK), size<1>(gK)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(gV), size<1>(gV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKcK = gmem_thr_copy_KV.partition_D(cK);
    Tensor tVcV = gmem_thr_copy_KV.partition_D(cV);

    // Allocate predicate tensors for k
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tSgK)));
    Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tdPgV)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) < params.d_value; }
    }

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);

    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                                make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdK = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor cdV = make_identity_tensor(make_shape(size<0>(gdV), size<1>(gdV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKcdK = gmem_thr_copy_dKV.partition_D(cdK);
        Tensor tdVcdV = gmem_thr_copy_dKV.partition_D(cdV);
        Tensor tdKpdK = make_tensor<bool>(make_shape(size<2>(tdKcdK)));
        Tensor tdVpdV = make_tensor<bool>(make_shape(size<2>(tdVcdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKpdK); ++k) { tdKpdK(k) = get<1>(tdKcdK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdVpdV); ++k) { tdVpdV(k) = get<1>(tdVcdV(0, 0, k)) < params.d_value; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKcdK, tdKpdK, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdVcdV, tdVpdV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }
    
    
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tSgK, tSrK, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN
    );

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tdPgV, tdPrV, tVcV, tVpV, binfo.actual_seqlen_k - n_block * kBlockN
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);  
    
    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
        bidb, bidh, tidx, params.h);

    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDimV>>{});
    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    clear(acc_dv);
    clear(acc_dk);
    
    Tensor taccScS_row = taccScS(_, 0, _);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    constexpr int kS_loops = size<2>(tSsQ);
    constexpr int kdV_loops = size<2>(tdVsdOt);
    constexpr int kdP_loops = size<2>(tdPsdO);
    constexpr int kdK_loops = size<2>(tdKsQt);
    static_assert(kStages <= kS_loops && kStages <= kdV_loops && kStages <= kdP_loops && kStages <= kdK_loops, "kStages is error");
    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    }
    #pragma unroll
    for (; m_block >= m_block_min; m_block--) {

        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < kS_loops - kStages; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, kStages + i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            s_waitcnt<2>();
            flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, i);
            S_BARRIER;
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, i, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
            s_waitcnt<2>();
            flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, kS_loops - kStages + i);
            S_BARRIER;
        }
        
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("lse tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block, lse(0), lse(1), lse(2), lse(3));
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("acc_s tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        Tensor scores_trans = make_tensor(acc_s.data(), flash::convert_trans_layout_acc_rowcol(acc_s.layout()));
        [[maybe_unused]] Tensor dtanh_trans = make_tensor_like(scores_trans);
        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
            flash::calculate_dtanh(scores_trans, dtanh_trans, params.softcap);
        }

        #if 1
        if constexpr (Has_alibi) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            const int wave_id = tidx / 64;
            const int col_idx_offset =  m_block * kBlockM;
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            alibi.apply_alibi_trans(scores, col_idx_offset, row_idx_offset_, kNWarps * 16);
        }
        #endif

        #if 1
        if constexpr(!Is_causal && !Is_local) {
            if (!Is_even_MN && (m_block + 1) * kBlockM >= binfo.actual_seqlen_q) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int warp_id = tidx / 64;
                // 实际上是row
                const int col_idx_offset_ = m_block * kBlockM;
                flash::apply_mask_trans(scores, binfo.actual_seqlen_q, col_idx_offset_);
            }
        } else if constexpr(Is_causal) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k)
            {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_causal_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16
                );
            }
        } else if constexpr(Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_local_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    params.window_size_left, params.window_size_right
                );
            }
        }
        #endif
        
        flash::scale_apply_exp2</*scale_max=*/false>(scores_trans, lse, params.scale_softmax_log2);

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("P tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        Tensor dP_sum = make_fragment_like(lse);

        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
            dP_sum(mi) = gdPsum(row);
        }
        if (m_block > m_block_min) {
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
                lse(mi) = gLSE(row);
            }
        }

        if constexpr (Is_dropout) {
            const int warp_id = tidx / 64;
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            int block_row_idx = row_idx_offset_;
            int block_col_idx = m_block * kBlockM;
            dropout.template apply_dropout_trans</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, kNWarps * 16
            );
        }

        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);

        lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<2>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 0);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gdO, sdOt, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 1);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gdO, sdOt, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<4>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 2);
        S_BARRIER;

        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("dP_sum tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block, dP_sum(0), dP_sum(1), dP_sum(2), dP_sum(3));
        //     float * tmp = reinterpret_cast<float*>(acc_dp.data());
        //     printf("dP tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }
        Tensor dS = make_tensor(acc_dp.data(), scores_trans.layout());

        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores_trans(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh_trans(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_dp.data());
        //     printf("dS tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        
        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gdO, sdOt, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gdO, sdOt, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<6>();
        flash::gemm_k_rs(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, 0);
        S_BARRIER;

        #pragma unroll
        for (int i = 0; i < 3; ++i) { // tail kStages
            lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gQ, sQt, i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            s_waitcnt<6>();
            flash::gemm_k_rs(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, 1 + i);
            S_BARRIER;
        }

        Tensor tdKrdSt = flash::convert_type<Element>(acc_dp);

        lds_direct_copy<Is_even_K, Is_even_MN, _16x96_multi_ins>(gQ, sQt, 3, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<6>();
        flash::gemm_k_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, 0);
        S_BARRIER;

        s_waitcnt<4>();
        flash::gemm_k_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, 1);
        S_BARRIER;
        // k = 2
        s_waitcnt<2>();
        flash::gemm_k_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, 2);
        S_BARRIER;
        // k = 3
        s_waitcnt<0>();
        flash::gemm_k_rs(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt, 3);
        S_BARRIER;

        if (m_block > m_block_min) {
            gQ.data() = gQ.data() + (-int(kBlockM * params.q_row_stride));
            gdO.data() = gdO.data() + (-int(kBlockM * params.do_row_stride));
            #pragma unroll
            for (int i = 0; i < kStages; ++i) { // tail kStages
                lds_direct_copy<Is_even_K>(gQ, sQ, i, params.q_row_stride, params.d);
            }
        } 

    }

    // wangaq debug
    // __syncthreads();
    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     float * tmp = reinterpret_cast<float*>(acc_dk.data());
    //     printf("dK tid:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, 
    //     tmp[0], tmp[1], tmp[2], tmp[3], 
    //     tmp[4], tmp[5], tmp[6], tmp[7],
    //     tmp[8], tmp[9], tmp[10], tmp[11], 
    //     tmp[12], tmp[13], tmp[14], tmp[15],
    //     tmp[16], tmp[17], tmp[18], tmp[19], 
    //     tmp[20], tmp[21], tmp[22], tmp[23],
    //     tmp[24], tmp[25], tmp[26], tmp[27], 
    //     tmp[28], tmp[29], tmp[30], tmp[31]
    //     );
    // }

    #if 0
    if constexpr(Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dv from fp32 to fp16
    Tensor rdK = flash::convert_type<Element>(acc_dk);
    Tensor rdV = flash::convert_type<Element>(acc_dv);

    //  __syncthreads();
    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    // If Is_last, there's already a __syncthreads() at the end of the loop.
    // if constexpr(!Is_last) { __syncthreads(); }

    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    __builtin_amdgcn_s_barrier(); 
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
__builtin_amdgcn_s_barrier(); 
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
    __syncthreads();
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    __builtin_amdgcn_s_barrier(); 
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    __builtin_amdgcn_s_barrier(); 
    #elif 0
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.dv_row_stride, _1{}));
    _bwd_store_dk_dv<Kernel_traits, decltype(acc_dv), decltype(sQ), decltype(gdV), Element, 
        typename Kernel_traits::SmemLayoutdVStore, Is_even_MN, Is_even_K>(
        acc_dv, sQ, tidx, gdV, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);

    __syncthreads();
    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    _bwd_store_dk_dv<Kernel_traits, decltype(acc_dk), decltype(sQ), decltype(gdK), Element, 
        typename Kernel_traits::SmemLayoutdKStore, Is_even_MN, Is_even_K>(
        acc_dk, sQ, tidx, gdK, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

    #else

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.dv_row_stride, _1{}));
    
    int row, col;
    if constexpr (size<1>(acc_dk) == size<1>(acc_dv) && size<2>(acc_dk) == size<2>(acc_dv)) {
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dk); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dk); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdK(row, col) = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                            gdV(row, col) = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout );
                        }
                        col += 4;
                    }
                }
            }
        } 
    } else {
        
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dk); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dk); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdK(row, col) = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                        }
                        col += 4;
                    }
                }
            }
        } 
        
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dv); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dv); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dv); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdV(row, col) = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout);
                        }
                        col += 4;
                    }
                }
            }
        } 
    }

    #endif

}

#endif

#if 1
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_trans_1colblock_16x64_dim256_prefetch(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    if constexpr (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }


    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;
    
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(params.do_row_stride, _1{}));

    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});
    

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQdOGemm0{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOGemm1transposed{});
    Tensor sQtSplit = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransSplit{});

    Tensor sdO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOGemm0{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOGemm1transposed{});
    Tensor sdOtSplit = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransSplit{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrK = thr_mma_sdp.partition_fragment_A(gK);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_B(gQ);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_A(gV);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_B(gdO);

    // dV/dK
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sQt);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sdOt);

    //
    // Copy Atom retiling
    //

    // S/dP
    auto gmem_tiled_copy_KV = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSgK = gmem_thr_copy_KV.partition_S(gK);
    Tensor tdPgV = gmem_thr_copy_KV.partition_S(gV);
    
    // auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_sdp);
    auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    // dV/dK
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt8x64 = smem_thr_copy_QdOt.partition_S(sdOtSplit);
    Tensor tdVsdOt = make_tensor(tdVsdOt8x64.data(), convert_layout_B_rowcol_<_16x128, 4>(tdVsdOt8x64.layout()));
    Tensor tdKsQt8x64 = smem_thr_copy_QdOt.partition_S(sQtSplit);
    Tensor tdKsQt = make_tensor(tdKsQt8x64.data(), convert_layout_B_rowcol_<_16x128, 4>(tdKsQt8x64.layout()));

    //
    // PREDICATES
    //

    Tensor cK = make_identity_tensor(make_shape(size<0>(gK), size<1>(gK)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(gV), size<1>(gV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKcK = gmem_thr_copy_KV.partition_D(cK);
    Tensor tVcV = gmem_thr_copy_KV.partition_D(cV);

    // Allocate predicate tensors for k
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tSgK)));
    Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tdPgV)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) < params.d_value; }
    }

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);

    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                                make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdK = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor cdV = make_identity_tensor(make_shape(size<0>(gdV), size<1>(gdV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKcdK = gmem_thr_copy_dKV.partition_D(cdK);
        Tensor tdVcdV = gmem_thr_copy_dKV.partition_D(cdV);
        Tensor tdKpdK = make_tensor<bool>(make_shape(size<2>(tdKcdK)));
        Tensor tdVpdV = make_tensor<bool>(make_shape(size<2>(tdVcdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKpdK); ++k) { tdKpdK(k) = get<1>(tdKcdK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdVpdV); ++k) { tdVpdV(k) = get<1>(tdVcdV(0, 0, k)) < params.d_value; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKcdK, tdKpdK, binfo.actual_seqlen_k - n_block * kBlockN
        );
        // flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        //     gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdVcdV, tdVpdV, binfo.actual_seqlen_k - n_block * kBlockN
        // );
        return;
    }
    
    
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tSgK, tSrK, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN
    );

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tdPgV, tdPrV, tVcV, tVpV, binfo.actual_seqlen_k - n_block * kBlockN
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);  
    
    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
        bidb, bidh, tidx, params.h);

    // Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDimV>>{});
    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    // Tensor acc_dv_split = local_tile(acc_dv, Shape<Int<8>, Int<1>, Int<kHeadDimV / 32 / 2>>{}, make_coord(0, 0, _)); 
    Tensor acc_dk_split = local_tile(acc_dk, Shape<Int<8>, Int<1>, Int<kHeadDim / 32 / 2>>{}, make_coord(0, 0, _)); 

    // auto acc_dv_0_128 = acc_dv_split(_, _, _, 0);
    // auto acc_dv_128_256 = acc_dv_split(_, _, _, 1);

    auto acc_dk_0_128 = acc_dk_split(_, _, _, 0);
    auto acc_dk_128_256 = acc_dk_split(_, _, _, 1);
    // clear(acc_dv);
    clear(acc_dk);
    
    Tensor taccScS_row = taccScS(_, 0, _);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    constexpr static int K_BUFF_SIZE = 4;
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    s_waitcnt<0>();
    if constexpr(!Is_even_K) {
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        }
    } else {
        lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, Is_even_MN>(gQ, sQ, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, Is_even_MN>(gQ, sQ, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, Is_even_MN>(gQ, sQ, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);

    }

    // wangaq debug
    // s_waitcnt<0>();
    // if (thread0() && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     __half * tmp = reinterpret_cast<__half*>(sQ.data().get());
    //     int col = 8;
    //     for (int i = 0; i < size(sQ)/col; ++i) {
    //         printf("Q:%d nblock:%d ", i, n_block);
    //         for (int j = 0; j < col; ++j) {
    //             printf("%10.4f ", __half2float(tmp[i*col+j]));
    //         }
    //         printf("\n");
    //     }
    // }
    #pragma unroll
    for (; m_block >= m_block_min; m_block--) {

        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{}); 
        clear(acc_s_ori);
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, 3, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, Is_even_MN>(gQ, sQ, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 0);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(0, gQ, sQ, 4, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<4, K_BUFF_SIZE, Is_even_MN>(gQ, sQ, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 1);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(1, gQ, sQ, 5, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<5, K_BUFF_SIZE, Is_even_MN>(gQ, sQ, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 2);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(2, gQ, sQ, 6, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<6, K_BUFF_SIZE, Is_even_MN>(gQ, sQ, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 3);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(3, gQ, sQ, 7, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<7, K_BUFF_SIZE, Is_even_MN>(gQ, sQ, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 4, 0);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, Is_even_MN>(gdO, sdO, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }
 
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 5, 1);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, Is_even_MN>(gdO, sdO, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }        
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 6, 2);
        s_barrier();

        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, Is_even_MN>(gdO, sdO, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }    
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 7, 3);
        s_barrier();
        
        
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     // printf("lse tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block, lse(0), lse(1), lse(2), lse(3));
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("acc_s tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //         "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //         tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7],
        //         tmp[8], tmp[9], tmp[10], tmp[11], tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        Tensor scores_trans = make_tensor(acc_s.data(), flash::convert_trans_layout_acc_rowcol(acc_s.layout()));
        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        [[maybe_unused]] Tensor dtanh_trans = make_tensor_like(scores_trans);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores_trans, dtanh_trans, params.softcap);
        }

        #if 1
        if constexpr (Has_alibi) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            const int wave_id = tidx / 64;
            const int col_idx_offset =  m_block * kBlockM;
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            alibi.apply_alibi_trans(scores, col_idx_offset, row_idx_offset_, kNWarps * 16);
        }
        #endif

        #if 1
        if constexpr(!Is_causal && !Is_local) {
            if (!Is_even_MN && (m_block + 1) * kBlockM >= binfo.actual_seqlen_q) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int warp_id = tidx / 64;
                // 实际上是row
                const int col_idx_offset_ = m_block * kBlockM;
                flash::apply_mask_trans(scores, binfo.actual_seqlen_q, col_idx_offset_);
            }
        } else if constexpr(Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            // if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
            //     || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
            //     const int warp_id = tidx / 64;
            //     flash::apply_mask_causal(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
            //                              binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
            //                              binfo.actual_seqlen_q,
            //                              AtomLayoutMS * 16);
            // }

            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k)
            {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_causal_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16
                );
            }
        } else if constexpr(Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_local_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    params.window_size_left, params.window_size_right
                );
            }
        }
        #endif
    #if 1
        flash::scale_apply_exp2</*scale_max=*/false>(scores_trans, lse, params.scale_softmax_log2);

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("P tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }
        Tensor dP_sum = make_fragment_like(lse);

        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
            dP_sum(mi) = gdPsum(row);
        }
        if (m_block > m_block_min) {
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
                lse(mi) = gLSE(row);
            }
        }

        if constexpr (Is_dropout) {
            const int warp_id = tidx / 64;
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = (kHeadDim == 128) ? (n_block * kBlockN) : (n_block * kBlockN + row_idx_offset_in_block);
            int block_row_idx = row_idx_offset_;
            int block_col_idx = m_block * kBlockM;
            if constexpr (kHeadDim==128){
                dropout.template apply_dropout_trans_opt</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, n_block * kBlockN, m_block * kBlockM, kNWarps * 16
                );
            }else{
                dropout.template apply_dropout_trans</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, block_row_idx, block_col_idx, kNWarps * 16
                );
            }
        }

        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);
        

        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, Is_even_MN>(gdO, sdO, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }    
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 0);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(0, gdO, sdO, 4, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<4, K_BUFF_SIZE, Is_even_MN>(gdO, sdO, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }    
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 1);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(1, gdO, sdO, 5, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<5, K_BUFF_SIZE, Is_even_MN>(gdO, sdO, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }   
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 2);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(2, gdO, sdO, 6, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<6, K_BUFF_SIZE, Is_even_MN>(gdO, sdO, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }  
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 3);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN>(3, gdO, sdO, 7, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<7, K_BUFF_SIZE, Is_even_MN>(gdO, sdO, params.do_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }  
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 4, 0);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 0, gQ, sQt, 0, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gQ, sQt, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 5, 1);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 1, gQ, sQt, 1, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gQ, sQt, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 6, 2);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 2, gQ, sQt, 2, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gQ, sQt, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 7, 3);
        s_barrier();

        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        Tensor dS = make_tensor(acc_dp.data(), scores_trans.layout());



        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores_trans(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh_trans(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }



        Tensor tdKrdSt = flash::convert_type<Element>(acc_dp);
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 3, gQ, sQt, 3, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, Is_even_MN, _16x256, 0>(gQ, sQt, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_dk_0_128, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 0, gQ, sQt, 0, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gQ, sQt, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_dk_0_128, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 1, gQ, sQt, 1, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gQ, sQt, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_dk_0_128, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 2, gQ, sQt, 2, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gQ, sQt, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_dk_0_128, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        if constexpr(!Is_even_K) {
            lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 3, gQ, sQt, 3, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        } else {
            lds_direct_copy_even_k_dim256<3, K_BUFF_SIZE, Is_even_MN, _16x256, 1>(gQ, sQt, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
        }
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_dk_128_256, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_dk_128_256, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_dk_128_256, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_dk_128_256, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        if (m_block > m_block_min) {
            gQ.data() = gQ.data() + (-int(kBlockM * params.q_row_stride));
            gdO.data() = gdO.data() + (-int(kBlockM * params.do_row_stride));
            if constexpr(!Is_even_K) {
                #pragma unroll
                for (int i = 0; i < 3; ++i) {
                    lds_direct_copy<Is_even_K, true>(gQ, sQ, i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
                }
            } else {
                lds_direct_copy_even_k_dim256<0, K_BUFF_SIZE, true>(gQ, sQ, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
                lds_direct_copy_even_k_dim256<1, K_BUFF_SIZE, true>(gQ, sQ, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);
                lds_direct_copy_even_k_dim256<2, K_BUFF_SIZE, true>(gQ, sQ, params.q_row_stride, binfo.actual_seqlen_q - m_block * kBlockM);

            }
        }
    #endif
    }



    #if 0
    #else

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.dv_row_stride, _1{}));
    
    int row, col;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_dk); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_dk); ++ni) {
                col = (laneId / 16) * 2 + ni * 32;
                for (int ei = 0; ei < 4; ++ei) {
                    using result_type = cutlass::Array<Element, 2>;
                    if constexpr (Is_even_K)
                    {
                        result_type res;
                        res[0] = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                        res[1] = flash::convert_type<Element>(acc_dk(ei + 4, mi, ni) * params.scale_softmax_rp_dropout);
                        *(result_type*)(&gdK(row, col)) = res;
                        // res[0] = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout);
                        // res[1] = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei + 4, mi, ni) : acc_dv(ei + 4, mi, ni) * params.rp_dropout);
                        // *(result_type*)(&gdV(row, col)) = res;
                    }
                    col += 8;
                }
            }
        } 
    }
    #endif
}
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dv_trans_1colblock_16x64_dim256_prefetch(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    if constexpr (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }


    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;
    
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(params.do_row_stride, _1{}));

    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});
    

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQdOGemm0{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOGemm1transposed{});
    Tensor sQtSplit = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransSplit{});

    Tensor sdO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOGemm0{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOGemm1transposed{});
    Tensor sdOtSplit = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransSplit{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrK = thr_mma_sdp.partition_fragment_A(gK);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_B(gQ);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_A(gV);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_B(gdO);

    // dV/dK
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sQt);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sdOt);

    //
    // Copy Atom retiling
    //

    // S/dP
    auto gmem_tiled_copy_KV = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSgK = gmem_thr_copy_KV.partition_S(gK);
    Tensor tdPgV = gmem_thr_copy_KV.partition_S(gV);
    
    // auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_sdp);
    auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    // dV/dK
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt8x64 = smem_thr_copy_QdOt.partition_S(sdOtSplit);
    Tensor tdVsdOt = make_tensor(tdVsdOt8x64.data(), convert_layout_B_rowcol_<_16x128, 4>(tdVsdOt8x64.layout()));
    Tensor tdKsQt8x64 = smem_thr_copy_QdOt.partition_S(sQtSplit);
    Tensor tdKsQt = make_tensor(tdKsQt8x64.data(), convert_layout_B_rowcol_<_16x128, 4>(tdKsQt8x64.layout()));

    //
    // PREDICATES
    //

    Tensor cK = make_identity_tensor(make_shape(size<0>(gK), size<1>(gK)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(gV), size<1>(gV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKcK = gmem_thr_copy_KV.partition_D(cK);
    Tensor tVcV = gmem_thr_copy_KV.partition_D(cV);

    // Allocate predicate tensors for k
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tSgK)));
    Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tdPgV)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) < params.d_value; }
    }

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);

    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                                make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdK = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor cdV = make_identity_tensor(make_shape(size<0>(gdV), size<1>(gdV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKcdK = gmem_thr_copy_dKV.partition_D(cdK);
        Tensor tdVcdV = gmem_thr_copy_dKV.partition_D(cdV);
        Tensor tdKpdK = make_tensor<bool>(make_shape(size<2>(tdKcdK)));
        Tensor tdVpdV = make_tensor<bool>(make_shape(size<2>(tdVcdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKpdK); ++k) { tdKpdK(k) = get<1>(tdKcdK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdVpdV); ++k) { tdVpdV(k) = get<1>(tdVcdV(0, 0, k)) < params.d_value; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        // flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        //     gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKcdK, tdKpdK, binfo.actual_seqlen_k - n_block * kBlockN
        // );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdVcdV, tdVpdV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }
    
    
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tSgK, tSrK, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN
    );

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tdPgV, tdPrV, tVcV, tVpV, binfo.actual_seqlen_k - n_block * kBlockN
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);  
    
    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
        bidb, bidh, tidx, params.h);

    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDimV>>{});
    // Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    Tensor acc_dv_split = local_tile(acc_dv, Shape<Int<8>, Int<1>, Int<kHeadDimV / 32 / 2>>{}, make_coord(0, 0, _)); 
    // Tensor acc_dk_split = local_tile(acc_dk, Shape<Int<8>, Int<1>, Int<kHeadDim / 32 / 2>>{}, make_coord(0, 0, _)); 

    auto acc_dv_0_128 = acc_dv_split(_, _, _, 0);
    auto acc_dv_128_256 = acc_dv_split(_, _, _, 1);

    // auto acc_dk_0_128 = acc_dk_split(_, _, _, 0);
    // auto acc_dk_128_256 = acc_dk_split(_, _, _, 1);
    clear(acc_dv);
    // clear(acc_dk);
    
    Tensor taccScS_row = taccScS(_, 0, _);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    s_waitcnt<0>();
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    }

    #pragma unroll
    for (; m_block >= m_block_min; m_block--) {

        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{}); 
        clear(acc_s_ori);
        
        lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, 3, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 0);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN>(0, gQ, sQ, 4, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 1);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN>(1, gQ, sQ, 5, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 2);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN>(2, gQ, sQ, 6, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 3);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN>(3, gQ, sQ, 7, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 4, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 0, gdO, sdOt, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 5, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 1, gdO, sdOt, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 6, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 2, gdO, sdOt, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 7, 3);
        s_barrier();
        
        
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     // printf("lse tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block, lse(0), lse(1), lse(2), lse(3));
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("acc_s tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //         "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //         tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7],
        //         tmp[8], tmp[9], tmp[10], tmp[11], tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        Tensor scores_trans = make_tensor(acc_s.data(), flash::convert_trans_layout_acc_rowcol(acc_s.layout()));
        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        [[maybe_unused]] Tensor dtanh_trans = make_tensor_like(scores_trans);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores_trans, dtanh_trans, params.softcap);
        }

        #if 1
        if constexpr (Has_alibi) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            const int wave_id = tidx / 64;
            const int col_idx_offset =  m_block * kBlockM;
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            alibi.apply_alibi_trans(scores, col_idx_offset, row_idx_offset_, kNWarps * 16);
        }
        #endif

        #if 1
        if constexpr(!Is_causal && !Is_local) {
            if (!Is_even_MN && (m_block + 1) * kBlockM >= binfo.actual_seqlen_q) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int warp_id = tidx / 64;
                // 实际上是row
                const int col_idx_offset_ = m_block * kBlockM;
                flash::apply_mask_trans(scores, binfo.actual_seqlen_q, col_idx_offset_);
            }
        } else if constexpr(Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            // if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
            //     || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
            //     const int warp_id = tidx / 64;
            //     flash::apply_mask_causal(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
            //                              binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
            //                              binfo.actual_seqlen_q,
            //                              AtomLayoutMS * 16);
            // }

            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k)
            {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_causal_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16
                );
            }
        } else if constexpr(Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_local_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    params.window_size_left, params.window_size_right
                );
            }
        }
        #endif
    #if 1
        flash::scale_apply_exp2</*scale_max=*/false>(scores_trans, lse, params.scale_softmax_log2);

   
        Tensor dP_sum = make_fragment_like(lse);

        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
            dP_sum(mi) = gdPsum(row);
        }
        if (m_block > m_block_min) {
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
                lse(mi) = gLSE(row);
            }
        }

        if constexpr (Is_dropout) {
            const int warp_id = tidx / 64;
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = (kHeadDim == 128) ? (n_block * kBlockN) : (n_block * kBlockN + row_idx_offset_in_block);
            int block_row_idx = row_idx_offset_;
            int block_col_idx = m_block * kBlockM;
            if constexpr (kHeadDim==128){
                dropout.template apply_dropout_trans_opt</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, n_block * kBlockN, m_block * kBlockM, kNWarps * 16
                );
            }else{
                dropout.template apply_dropout_trans</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, block_row_idx, block_col_idx, kNWarps * 16
                );
            }
        }

        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 3, gdO, sdOt, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_dv_0_128, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 0, gdO, sdOt, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_dv_0_128, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 1, gdO, sdOt, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_dv_0_128, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 2, gdO, sdOt, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_dv_0_128, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 3, gdO, sdOt, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_dv_128_256, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        // lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_dv_128_256, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        // lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_dv_128_256, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        // lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_dv_128_256, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        if (m_block > m_block_min) {
            gQ.data() = gQ.data() + (-int(kBlockM * params.q_row_stride));
            gdO.data() = gdO.data() + (-int(kBlockM * params.do_row_stride));
            lds_direct_copy<Is_even_K>(gQ, sQ, 0, params.q_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gQ, sQ, 1, params.q_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gQ, sQ, 2, params.q_row_stride, params.d);
        }
    #endif
    }

    // wangaq debug
    // __syncthreads();
    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     float * tmp = reinterpret_cast<float*>(acc_dk.data());
    //     printf("dK tid:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, 
    //     tmp[0], tmp[1], tmp[2], tmp[3], 
    //     tmp[4], tmp[5], tmp[6], tmp[7],
    //     tmp[8], tmp[9], tmp[10], tmp[11], 
    //     tmp[12], tmp[13], tmp[14], tmp[15],
    //     tmp[16], tmp[17], tmp[18], tmp[19], 
    //     tmp[20], tmp[21], tmp[22], tmp[23],
    //     tmp[24], tmp[25], tmp[26], tmp[27], 
    //     tmp[28], tmp[29], tmp[30], tmp[31]
    //     );
    // }

    #if 0
    #else

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.dv_row_stride, _1{}));
    
    int row, col;
    #pragma unroll
    for (int mi = 0; mi < size<1>(acc_dv); ++mi) {
        row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
        if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
            #pragma unroll
            for (int ni = 0; ni < size<2>(acc_dv); ++ni) {
                col = (laneId / 16) * 2 + ni * 32;
                for (int ei = 0; ei < 4; ++ei) {
                    using result_type = cutlass::Array<Element, 2>;
                    if constexpr (Is_even_K)
                    {
                        result_type res;
                        // res[0] = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                        // res[1] = flash::convert_type<Element>(acc_dk(ei + 4, mi, ni) * params.scale_softmax_rp_dropout);
                        // *(result_type*)(&gdK(row, col)) = res;
                        res[0] = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout);
                        res[1] = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei + 4, mi, ni) : acc_dv(ei + 4, mi, ni) * params.rp_dropout);
                        *(result_type*)(&gdV(row, col)) = res;
                    }
                    col += 8;
                }
            }
        } 
    }
    #endif
}
#endif

#if 1
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_dv_trans_1colblock_16x64_dim512_prefetch(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    if constexpr (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }


    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;
    
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(params.do_row_stride, _1{}));

    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});
    

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQdOGemm0{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOGemm1transposed{});
    Tensor sQtSplit = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransSplit{});

    Tensor sdO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOGemm0{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOGemm1transposed{});
    Tensor sdOtSplit = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransSplit{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrK = thr_mma_sdp.partition_fragment_A(gK);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_B(gQ);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_A(gV);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_B(gdO);

    // dV/dK
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sQt);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sdOt);

    //
    // Copy Atom retiling
    //

    // S/dP
    auto gmem_tiled_copy_KV = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSgK = gmem_thr_copy_KV.partition_S(gK);
    Tensor tdPgV = gmem_thr_copy_KV.partition_S(gV);
    
    // auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_sdp);
    auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    // dV/dK
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt8x64 = smem_thr_copy_QdOt.partition_S(sdOtSplit);
    Tensor tdVsdOt = make_tensor(tdVsdOt8x64.data(), convert_layout_B_rowcol_<_16x128, 4>(tdVsdOt8x64.layout()));
    Tensor tdKsQt8x64 = smem_thr_copy_QdOt.partition_S(sQtSplit);
    Tensor tdKsQt = make_tensor(tdKsQt8x64.data(), convert_layout_B_rowcol_<_16x128, 4>(tdKsQt8x64.layout()));

    //
    // PREDICATES
    //

    Tensor cK = make_identity_tensor(make_shape(size<0>(gK), size<1>(gK)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(gV), size<1>(gV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKcK = gmem_thr_copy_KV.partition_D(cK);
    Tensor tVcV = gmem_thr_copy_KV.partition_D(cV);

    // Allocate predicate tensors for k
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tSgK)));
    Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tdPgV)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) < params.d_value; }
    }

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);

    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                                make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdK = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor cdV = make_identity_tensor(make_shape(size<0>(gdV), size<1>(gdV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKcdK = gmem_thr_copy_dKV.partition_D(cdK);
        Tensor tdVcdV = gmem_thr_copy_dKV.partition_D(cdV);
        Tensor tdKpdK = make_tensor<bool>(make_shape(size<2>(tdKcdK)));
        Tensor tdVpdV = make_tensor<bool>(make_shape(size<2>(tdVcdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKpdK); ++k) { tdKpdK(k) = get<1>(tdKcdK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdVpdV); ++k) { tdVpdV(k) = get<1>(tdVcdV(0, 0, k)) < params.d_value; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKcdK, tdKpdK, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdVcdV, tdVpdV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }
    
    
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tSgK, tSrK, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN
    );

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tdPgV, tdPrV, tVcV, tVpV, binfo.actual_seqlen_k - n_block * kBlockN
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);  
    
    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
        bidb, bidh, tidx, params.h);

    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDimV>>{});
    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    Tensor acc_dv_split = local_tile(acc_dv, Shape<Int<8>, Int<1>, Int<kHeadDimV / 32 / 4>>{}, make_coord(0, 0, _)); 
    Tensor acc_dk_split = local_tile(acc_dk, Shape<Int<8>, Int<1>, Int<kHeadDim / 32 / 4>>{}, make_coord(0, 0, _)); 

    auto acc_dv_0_128 = acc_dv_split(_, _, _, 0);
    auto acc_dv_128_256 = acc_dv_split(_, _, _, 1);
    auto acc_dv_256_384 = acc_dv_split(_, _, _, 2);
    auto acc_dv_384_512 = acc_dv_split(_, _, _, 3);

    auto acc_dk_0_128 = acc_dk_split(_, _, _, 0);
    auto acc_dk_128_256 = acc_dk_split(_, _, _, 1);
    auto acc_dk_256_384 = acc_dk_split(_, _, _, 2);
    auto acc_dk_384_512 = acc_dk_split(_, _, _, 3);

    clear(acc_dv);
    clear(acc_dk);
    
    Tensor taccScS_row = taccScS(_, 0, _);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    s_waitcnt<0>();
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    }
    // wangaq debug
    // s_waitcnt<0>();
    // if (thread0() && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     __half * tmp = reinterpret_cast<__half*>(sQ.data().get());
    //     int col = 8;
    //     for (int i = 0; i < size(sQ)/col; ++i) {
    //         printf("Q:%d nblock:%d ", i, n_block);
    //         for (int j = 0; j < col; ++j) {
    //             printf("%10.4f ", __half2float(tmp[i*col+j]));
    //         }
    //         printf("\n");
    //     }
    // }
    #pragma unroll
    for (; m_block >= m_block_min; m_block--) {

        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{}); 
        clear(acc_s_ori);
        
        lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, 3, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 0);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN>(0, gQ, sQ, 4, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 1);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN>(1, gQ, sQ, 5, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 2);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN>(2, gQ, sQ, 6, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 3);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN>(3, gQ, sQ, 7, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 4, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gQ, sQ, 8, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 5, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gQ, sQ, 9, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 6, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gQ, sQ, 10, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 7, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gQ, sQ, 11, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 8, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gQ, sQ, 12, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 9, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gQ, sQ, 13, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 10, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gQ, sQ, 14, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 11, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gQ, sQ, 15, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 12, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 0, gdO, sdOt, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 13, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 1, gdO, sdOt, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 14, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 2, gdO, sdOt, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 15, 3);
        s_barrier();
        
        
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     // printf("lse tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block, lse(0), lse(1), lse(2), lse(3));
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("acc_s tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //         "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //         tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7],
        //         tmp[8], tmp[9], tmp[10], tmp[11], tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        Tensor scores_trans = make_tensor(acc_s.data(), flash::convert_trans_layout_acc_rowcol(acc_s.layout()));
        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        [[maybe_unused]] Tensor dtanh_trans = make_tensor_like(scores_trans);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores_trans, dtanh_trans, params.softcap);
        }

        #if 1
        if constexpr (Has_alibi) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            const int wave_id = tidx / 64;
            const int col_idx_offset =  m_block * kBlockM;
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            alibi.apply_alibi_trans(scores, col_idx_offset, row_idx_offset_, kNWarps * 16);
        }
        #endif

        #if 1
        if constexpr(!Is_causal && !Is_local) {
            if (!Is_even_MN && (m_block + 1) * kBlockM >= binfo.actual_seqlen_q) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int warp_id = tidx / 64;
                // 实际上是row
                const int col_idx_offset_ = m_block * kBlockM;
                flash::apply_mask_trans(scores, binfo.actual_seqlen_q, col_idx_offset_);
            }
        } else if constexpr(Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            // if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
            //     || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
            //     const int warp_id = tidx / 64;
            //     flash::apply_mask_causal(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
            //                              binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
            //                              binfo.actual_seqlen_q,
            //                              AtomLayoutMS * 16);
            // }

            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k)
            {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_causal_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16
                );
            }
        } else if constexpr(Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_local_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    params.window_size_left, params.window_size_right
                );
            }
        }
        #endif
    #if 1
        flash::scale_apply_exp2</*scale_max=*/false>(scores_trans, lse, params.scale_softmax_log2);

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("P tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }
        Tensor dP_sum = make_fragment_like(lse);

        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
            dP_sum(mi) = gdPsum(row);
        }
        if (m_block > m_block_min) {
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
                lse(mi) = gLSE(row);
            }
        }

        if constexpr (Is_dropout) {
            const int warp_id = tidx / 64;
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = (kHeadDim == 128) ? (n_block * kBlockN) : (n_block * kBlockN + row_idx_offset_in_block);
            int block_row_idx = row_idx_offset_;
            int block_col_idx = m_block * kBlockM;
            if constexpr (kHeadDim==128){
                dropout.template apply_dropout_trans_opt</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, n_block * kBlockN, m_block * kBlockM, kNWarps * 16
                );
            }else{
                dropout.template apply_dropout_trans</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, block_row_idx, block_col_idx, kNWarps * 16
                );
            }
        }

        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 3, gdO, sdOt, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dv_0_128, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 0, gdO, sdOt, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dv_0_128, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 1, gdO, sdOt, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dv_0_128, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 2, gdO, sdOt, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dv_0_128, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 3, gdO, sdOt, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dv_128_256, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 0, gdO, sdOt, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dv_128_256, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 1, gdO, sdOt, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dv_128_256, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 2, gdO, sdOt, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dv_128_256, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 3, gdO, sdOt, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dv_256_384, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 0, gdO, sdOt, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dv_256_384, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 1, gdO, sdOt, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dv_256_384, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 2, gdO, sdOt, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dv_256_384, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 3, gdO, sdOt, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dv_384_512, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
      

        lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dv_384_512, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dv_384_512, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dv_384_512, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        // lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 3, gdO, sdOt, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs_ds_read_m32x16<0>(acc_dv_0_128, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        // s_barrier();
        
        // lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 0, gdO, sdOt, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs_ds_read_m32x16<1>(acc_dv_0_128, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        // s_barrier();
        
        // lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 1, gdO, sdOt, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs_ds_read_m32x16<2>(acc_dv_0_128, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        // s_barrier();
        
        // lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 2, gdO, sdOt, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs_ds_read_m32x16<3>(acc_dv_0_128, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        // s_barrier();
        
        // lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 3, gdO, sdOt, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs_ds_read_m32x16<0>(acc_dv_128_256, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        // s_barrier();

        // lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs_ds_read_m32x16<1>(acc_dv_128_256, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        // s_barrier();

        // lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs_ds_read_m32x16<2>(acc_dv_128_256, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        // s_barrier();

        // lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs_ds_read_m32x16<3>(acc_dv_128_256, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        // s_barrier();

        // return;
        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);
        
        lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 0);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN>(0, gdO, sdO, 4, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 1);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN>(1, gdO, sdO, 5, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 2);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN>(2, gdO, sdO, 6, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 3);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN>(3, gdO, sdO, 7, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 4, 0);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN>(0, gdO, sdO, 8, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 5, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gdO, sdO, 9, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 6, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gdO, sdO, 10, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 7, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gdO, sdO, 11, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 8, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(0, gdO, sdO, 12, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 9, 1);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(1, gdO, sdO, 13, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 10, 2);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(2, gdO, sdO, 14, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 11, 3);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN>(3, gdO, sdO, 15, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 12, 0);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 0, gQ, sQt, 0, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 13, 1);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 1, gQ, sQt, 1, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 14, 2);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 2, gQ, sQt, 2, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 15, 3);
        s_barrier();

        // lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 0);
        // s_barrier();
        
        // lds_direct_copy<Is_even_K, Is_even_MN>(0, gdO, sdO, 4, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 1);
        // s_barrier();
        
        // lds_direct_copy<Is_even_K, Is_even_MN>(1, gdO, sdO, 5, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 2);
        // s_barrier();
        
        // lds_direct_copy<Is_even_K, Is_even_MN>(2, gdO, sdO, 6, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 3);
        // s_barrier();
        
        // lds_direct_copy<Is_even_K, Is_even_MN>(3, gdO, sdO, 7, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 4, 0);
        // s_barrier();
        
        // lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 0, gQ, sQt, 0, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 5, 1);
        // s_barrier();
        
        // lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 1, gQ, sQt, 1, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 6, 2);
        // s_barrier();
        
        // lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 2, gQ, sQt, 2, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        // s_waitcnt<3>();
        // flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 7, 3);
        // s_barrier();

        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        Tensor dS = make_tensor(acc_dp.data(), scores_trans.layout());

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("dP_sum tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block, dP_sum(0), dP_sum(1), dP_sum(2), dP_sum(3));
        //     float * tmp = reinterpret_cast<float*>(acc_dp.data());
        //     printf("dP tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores_trans(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh_trans(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_dp.data());
        //     printf("dS tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        Tensor tdKrdSt = flash::convert_type<Element>(acc_dp);
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(0, 3, gQ, sQt, 3, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dk_0_128, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 0, gQ, sQt, 0, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dk_0_128, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 1, gQ, sQt, 1, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dk_0_128, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 2, gQ, sQt, 2, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dk_0_128, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();
        
        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(1, 3, gQ, sQt, 3, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dk_128_256, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 0, gQ, sQt, 0, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dk_128_256, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 1, gQ, sQt, 1, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dk_128_256, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 2, gQ, sQt, 2, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dk_128_256, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();


        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(2, 3, gQ, sQt, 3, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dk_256_384, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();


        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 0, gQ, sQt, 0, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dk_256_384, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 1, gQ, sQt, 1, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dk_256_384, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 2, gQ, sQt, 2, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dk_256_384, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        lds_direct_copy<Is_even_K, Is_even_MN, _16x256>(3, 3, gQ, sQt, 3, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        s_waitcnt<3>();
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dk_384_512, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        s_waitcnt<2>();
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dk_384_512, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        s_waitcnt<1>();
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dk_384_512, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        s_waitcnt<0>();
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dk_384_512, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        s_barrier();

        if (m_block > m_block_min) {
            gQ.data() = gQ.data() + (-int(kBlockM * params.q_row_stride));
            gdO.data() = gdO.data() + (-int(kBlockM * params.do_row_stride));
            lds_direct_copy<Is_even_K>(gQ, sQ, 0, params.q_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gQ, sQ, 1, params.q_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gQ, sQ, 2, params.q_row_stride, params.d);
        }
    #endif
    }

    // wangaq debug
    // __syncthreads();
    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     float * tmp = reinterpret_cast<float*>(acc_dk.data());
    //     printf("dK tid:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, 
    //     tmp[0], tmp[1], tmp[2], tmp[3], 
    //     tmp[4], tmp[5], tmp[6], tmp[7],
    //     tmp[8], tmp[9], tmp[10], tmp[11], 
    //     tmp[12], tmp[13], tmp[14], tmp[15],
    //     tmp[16], tmp[17], tmp[18], tmp[19], 
    //     tmp[20], tmp[21], tmp[22], tmp[23],
    //     tmp[24], tmp[25], tmp[26], tmp[27], 
    //     tmp[28], tmp[29], tmp[30], tmp[31]
    //     );
    // }

    #if 0
    if constexpr(Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dv from fp32 to fp16
    Tensor rdK = flash::convert_type<Element>(acc_dk);
    Tensor rdV = flash::convert_type<Element>(acc_dv);

    //  __syncthreads();
    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    // If Is_last, there's already a __syncthreads() at the end of the loop.
    // if constexpr(!Is_last) { __syncthreads(); }

    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    __builtin_amdgcn_s_barrier(); 
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
__builtin_amdgcn_s_barrier(); 
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
    __syncthreads();
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    __builtin_amdgcn_s_barrier(); 
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    __builtin_amdgcn_s_barrier(); 
    #elif 0
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.dv_row_stride, _1{}));
    _bwd_store_dk_dv<Kernel_traits, decltype(acc_dv), decltype(sQ), decltype(gdV), Element, 
        typename Kernel_traits::SmemLayoutdVStore, Is_even_MN, Is_even_K>(
        acc_dv, sQ, tidx, gdV, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);

    __syncthreads();
    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    _bwd_store_dk_dv<Kernel_traits, decltype(acc_dk), decltype(sQ), decltype(gdK), Element, 
        typename Kernel_traits::SmemLayoutdKStore, Is_even_MN, Is_even_K>(
        acc_dk, sQ, tidx, gdK, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

    #else

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.dv_row_stride, _1{}));
    
    int row, col;
    if constexpr (size<1>(acc_dk) == size<1>(acc_dv) && size<2>(acc_dk) == size<2>(acc_dv)) {
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dk); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dk); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdK(row, col) = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                            gdV(row, col) = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout );
                        }
                        col += 4;
                    }
                }
            }
        } 
    } else {
        
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dk); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dk); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdK(row, col) = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                        }
                        col += 4;
                    }
                }
            }
        } 
        
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dv); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dv); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dv); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdV(row, col) = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout);
                        }
                        col += 4;
                    }
                }
            }
        } 
    }

    #endif

}
#endif



template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_dv_trans_1colblock_16x64_prefetch(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;

    constexpr int kSmemOffset = Kernel_traits::kSmemOffset;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    if constexpr (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }


    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;
    
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(params.do_row_stride, _1{}));

    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});
    

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQGemm0{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQGemm1transposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQGemm1transposedNoSwizzle{});

    Tensor sdO = make_tensor(sQ.data() + kSmemOffset, typename Kernel_traits::SmemLayoutdOGemm0{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutdOGemm1transposed{});
    Tensor sdOtNoSwizzle = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutdOGemm1transposedNoSwizzle{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrK = thr_mma_sdp.partition_fragment_A(gK);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_B(sQ);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_A(gV);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_B(sdO);

    // dV/dK
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtNoSwizzle);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);

    //
    // Copy Atom retiling
    //

    // S/dP
    auto gmem_tiled_copy_KV = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSgK = gmem_thr_copy_KV.partition_S(gK);
    Tensor tdPgV = gmem_thr_copy_KV.partition_S(gV);
    
    // auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_sdp);
    auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    // auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_BLayout);
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sQtemp = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQ{});
    Tensor tSsQBLayout = smem_thr_copy_BLayout.partition_S(sQtemp);
    Tensor tSsQ = make_tensor(tSsQBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDim/32>(tSsQBLayout.layout()));
    Tensor sdOtemp = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutdO{});
    Tensor tdPsdOBLayout = smem_thr_copy_BLayout.partition_S(sdOtemp);
    Tensor tdPsdO = make_tensor(tdPsdOBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDimV/32>(tdPsdOBLayout.layout()));

    // dV/dK
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16_WITH_8x64, Element>{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt8x64 = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdVsdOt = make_tensor(tdVsdOt8x64.data(), convert_layout_B_rowcol<_16x128>(tdVsdOt8x64.layout()));
    Tensor tdKsQt8x64 = smem_thr_copy_QdOt.partition_S(sQt);
    Tensor tdKsQt = make_tensor(tdKsQt8x64.data(), convert_layout_B_rowcol<_16x128>(tdKsQt8x64.layout()));

    //
    // PREDICATES
    //

    Tensor cK = make_identity_tensor(make_shape(size<0>(gK), size<1>(gK)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(gV), size<1>(gV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKcK = gmem_thr_copy_KV.partition_D(cK);
    Tensor tVcV = gmem_thr_copy_KV.partition_D(cV);

    // Allocate predicate tensors for k
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tSgK)));
    Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tdPgV)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) < params.d_value; }
    }

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);

    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                                make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdK = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor cdV = make_identity_tensor(make_shape(size<0>(gdV), size<1>(gdV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKcdK = gmem_thr_copy_dKV.partition_D(cdK);
        Tensor tdVcdV = gmem_thr_copy_dKV.partition_D(cdV);
        Tensor tdKpdK = make_tensor<bool>(make_shape(size<2>(tdKcdK)));
        Tensor tdVpdV = make_tensor<bool>(make_shape(size<2>(tdVcdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKpdK); ++k) { tdKpdK(k) = get<1>(tdKcdK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdVpdV); ++k) { tdVpdV(k) = get<1>(tdVcdV(0, 0, k)) < params.d_value; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKcdK, tdKpdK, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdVcdV, tdVpdV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }
    
    
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tSgK, tSrK, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN
    );

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tdPgV, tdPrV, tVcV, tVpV, binfo.actual_seqlen_k - n_block * kBlockN
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);  
    
    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
        bidb, bidh, tidx, params.h);

    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDimV>>{});
    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    clear(acc_dv);
    clear(acc_dk);
    
    Tensor taccScS_row = taccScS(_, 0, _);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    constexpr int kS_loops = size<2>(tSsQ);
    constexpr int kdV_loops = size<2>(tdVsdOt);
    constexpr int kdP_loops = size<2>(tdPsdO);
    constexpr int kdK_loops = size<2>(tdKsQt);
    static_assert(kStages <= kS_loops && kStages <= kdV_loops && kStages <= kdP_loops && kStages <= kdK_loops, "kStages is error");
    #pragma unroll
    for (int i = 0; i < kStages; ++i) {
        lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    }
    #pragma unroll
    for (; m_block >= m_block_min; m_block--) {

        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < kS_loops - kStages; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gQ, sQ, kStages + i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, i);
            S_BARRIER;
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gdO, sdOt, i, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
            S_WAITCNT;
            flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, kS_loops - kStages + i);
            S_BARRIER;
        }
        
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));

        
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("lse tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block, lse(0), lse(1), lse(2), lse(3));
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("acc_s tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        Tensor scores_trans = make_tensor(acc_s.data(), flash::convert_trans_layout_acc_rowcol(acc_s.layout()));
        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        [[maybe_unused]] Tensor dtanh_trans = make_tensor_like(scores_trans);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores_trans, dtanh_trans, params.softcap);
        }

        #if 1
        if constexpr (Has_alibi) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            const int wave_id = tidx / 64;
            const int col_idx_offset =  m_block * kBlockM;
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            alibi.apply_alibi_trans(scores, col_idx_offset, row_idx_offset_, kNWarps * 16);
        }
        #endif

        #if 1
        if constexpr(!Is_causal && !Is_local) {
            if (!Is_even_MN && (m_block + 1) * kBlockM >= binfo.actual_seqlen_q) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int warp_id = tidx / 64;
                // 实际上是row
                const int col_idx_offset_ = m_block * kBlockM;
                flash::apply_mask_trans(scores, binfo.actual_seqlen_q, col_idx_offset_);
            }
        } else if constexpr(Is_causal) {
            // Putting this causal masking right after acc_s is *much* slower for some reason.
            // TD [2023-08-16]: We need the 2nd condition because if seqlen_q is long and seqlen_k is short
            // (e.g., 256 and 2), the 2nd block of seqlen_q (from 128 to 255), we're not doing causal masking.
            // But we still want to mask out elements beyond actual_seqlen_k.
            // if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k
            //     || (!Is_even_MN && (n_block + 1) * kBlockN >= binfo.actual_seqlen_k)) {
            //     const int warp_id = tidx / 64;
            //     flash::apply_mask_causal(scores, n_block * kBlockN + (warp_id / AtomLayoutMS) * MMA_N_SdP * 16,
            //                              binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
            //                              binfo.actual_seqlen_q,
            //                              AtomLayoutMS * 16);
            // }

            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k)
            {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_causal_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16
                );
            }
        } else if constexpr(Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_local_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    params.window_size_left, params.window_size_right
                );
            }
        }
        #endif
        
        flash::scale_apply_exp2</*scale_max=*/false>(scores_trans, lse, params.scale_softmax_log2);

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("P tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }
        Tensor dP_sum = make_fragment_like(lse);

        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
            dP_sum(mi) = gdPsum(row);
        }
        if (m_block > m_block_min) {
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
                lse(mi) = gLSE(row);
            }
        }

        if constexpr (Is_dropout) {
            const int warp_id = tidx / 64;
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = (kHeadDim == 128) ? (n_block * kBlockN) : (n_block * kBlockN + row_idx_offset_in_block);
            int block_row_idx = row_idx_offset_;
            int block_col_idx = m_block * kBlockM;
            if constexpr (kHeadDim==128){
                dropout.template apply_dropout_trans_opt</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, n_block * kBlockN, m_block * kBlockM, kNWarps * 16
                );
            }else{
                dropout.template apply_dropout_trans</*encode_dropout_in_sign_bit=*/true>(
                    acc_s, block_row_idx, block_col_idx, kNWarps * 16
                );
            }
        }

        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);


        lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gdO, sdOt, 3, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        S_WAITCNT;
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 0, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        S_WAITCNT;
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        S_WAITCNT;
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, 2, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        S_WAITCNT;
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        S_BARRIER;

        // return;
        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);

        #pragma unroll
        for (int i = 0; i < kdP_loops - kStages; ++i) {
            lds_direct_copy<Is_even_K, Is_even_MN>(gdO, sdO, kStages + i, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
            S_WAITCNT;
            flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, i);
            S_BARRIER;
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gQ, sQt, i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            S_WAITCNT;
            flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, kdP_loops - kStages + i);
            S_BARRIER;
        }
        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
        Tensor dS = make_tensor(acc_dp.data(), scores_trans.layout());

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("dP_sum tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block, dP_sum(0), dP_sum(1), dP_sum(2), dP_sum(3));
        //     float * tmp = reinterpret_cast<float*>(acc_dp.data());
        //     printf("dP tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores_trans(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh_trans(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     float * tmp = reinterpret_cast<float*>(acc_dp.data());
        //     printf("dS tid:%d m_block:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
        //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, m_block,
        //     tmp[0], tmp[1], tmp[2], tmp[3], 
        //     tmp[4], tmp[5], tmp[6], tmp[7],
        //     tmp[8], tmp[9], tmp[10], tmp[11], 
        //     tmp[12], tmp[13], tmp[14], tmp[15]
        //     );
        // }

        Tensor tdKrdSt = flash::convert_type<Element>(acc_dp);

        lds_direct_copy<Is_even_K, Is_even_MN, _16x128>(gQ, sQt, 3, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        S_WAITCNT;
        flash::gemm_k_rs_ds_read_m32x16_alt<0>(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        S_BARRIER;

        S_WAITCNT2;
        flash::gemm_k_rs_ds_read_m32x16_alt<1>(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        S_BARRIER;
        // k = 2
        S_WAITCNT1;
        flash::gemm_k_rs_ds_read_m32x16_alt<2>(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        S_BARRIER;
        // k = 3
        S_WAITCNT0;
        flash::gemm_k_rs_ds_read_m32x16_alt<3>(acc_dk, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_QdOt, smem_thr_copy_QdOt);
        S_BARRIER;
        if (m_block > m_block_min) {
            gQ.data() = gQ.data() + (-int(kBlockM * params.q_row_stride));
            gdO.data() = gdO.data() + (-int(kBlockM * params.do_row_stride));
            lds_direct_copy<Is_even_K>(gQ, sQ, 0, params.q_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gQ, sQ, 1, params.q_row_stride, params.d);
            lds_direct_copy<Is_even_K>(gQ, sQ, 2, params.q_row_stride, params.d);
        }

    }

    // wangaq debug
    // __syncthreads();
    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //     float * tmp = reinterpret_cast<float*>(acc_dk.data());
    //     printf("dK tid:%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f "
    //     "%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n", tidx, 
    //     tmp[0], tmp[1], tmp[2], tmp[3], 
    //     tmp[4], tmp[5], tmp[6], tmp[7],
    //     tmp[8], tmp[9], tmp[10], tmp[11], 
    //     tmp[12], tmp[13], tmp[14], tmp[15],
    //     tmp[16], tmp[17], tmp[18], tmp[19], 
    //     tmp[20], tmp[21], tmp[22], tmp[23],
    //     tmp[24], tmp[25], tmp[26], tmp[27], 
    //     tmp[28], tmp[29], tmp[30], tmp[31]
    //     );
    // }

    #if 0
    #else

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.dv_row_stride, _1{}));
    
    static_assert((size<1>(acc_dk) == size<1>(acc_dv) && size<2>(acc_dk) == size<2>(acc_dv)));
    int row, col;
    if constexpr (size<1>(acc_dk) == size<1>(acc_dv) && size<2>(acc_dk) == size<2>(acc_dv)) {
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dk); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk); ++ni) {
                    col = (laneId / 16) * 2 + ni * 32;
                    for (int ei = 0; ei < 4; ++ei) {
                        using result_type = cutlass::Array<Element, 2>;
                        if constexpr (Is_even_K)
                        {
                            result_type res;
                            res[0] = flash::convert_type<Element>(acc_dk(ei, mi, ni) * params.scale_softmax_rp_dropout);
                            res[1] = flash::convert_type<Element>(acc_dk(ei + 4, mi, ni) * params.scale_softmax_rp_dropout);
                            *(result_type*)(&gdK(row, col)) = res;
                            res[0] = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout);
                            res[1] = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei + 4, mi, ni) : acc_dv(ei + 4, mi, ni) * params.rp_dropout);
                            *(result_type*)(&gdV(row, col)) = res;
                        }
                        col += 8;
                    }
                }
            } 
        }
    }
    #endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_dv_trans_1colblock_16x64_mla_prefetch(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    const int warpId = tidx / 64;
    const int laneId = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kStages = Kernel_traits::kStages;
    constexpr int K_BUFF_SIZE = 4;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);

    if (n_block * kBlockN >= binfo.actual_seqlen_k) return;

    int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    if constexpr (Is_local) {
        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left, kBlockM));
    }


    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + n_block * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.q_offset(params.do_batch_stride, params.do_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.do_row_stride + bidh * params.do_head_stride;

    const index_t row_offset_lse = (params.unpadded_lse? bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb): (bidb * params.h + bidh) * params.seqlen_q) + (m_block_max - 1) * kBlockM;
    // Regarding 128 * params.b see a comment in mha_varlen_bwd about padding of dq_accum and softmax_d
    const index_t row_offset_dpsum = (params.unpadded_lse? bidh * (params.total_q + 128 * params.b) + binfo.q_offset(params.seqlen_q_rounded, 1, bidb) + 128 * bidb: (bidb * params.h + bidh) * params.seqlen_q_rounded) + (m_block_max - 1) * kBlockM;
    
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                            Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                            make_stride(params.do_row_stride, _1{}));

    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                            Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});
    

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQdOGemm0{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOGemm1transposed{});
    Tensor sQtTail = make_tensor(sQ.data() + 4096, typename Kernel_traits::SmemLayoutQGemm1TailTransposed{});

    Tensor sdO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOGemm0{});
    Tensor sdOt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOGemm1transposed{});

    // S/dP
    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrK = thr_mma_sdp.partition_fragment_A(gK);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_B(sQ);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_A(gV);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_B(sdO);

    // dV/dK
    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOt);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQt);
    Tensor tdKrQtTail = thr_mma_dkv.partition_fragment_B(sQtTail);

    //
    // Copy Atom retiling
    //

    // S/dP
    auto gmem_tiled_copy_KV = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSgK = gmem_thr_copy_KV.partition_S(gK);
    Tensor tdPgV = gmem_thr_copy_KV.partition_S(gV);
    
    // auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_sdp);
    auto smem_tiled_copy_QdO = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);

    typename Kernel_traits::TiledMma16x64BLayout tiled_mma_BLayout;
    // auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_B128, Element>{}, tiled_mma_BLayout);
    auto smem_tiled_copy_BLayout = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_BLayout);
    auto smem_thr_copy_BLayout = smem_tiled_copy_BLayout.get_thread_slice(tidx);
    Tensor sQtemp = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdO{});
    Tensor tSsQBLayout = smem_thr_copy_BLayout.partition_S(sQtemp);
    Tensor tSsQ = make_tensor(tSsQBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDim/32>(tSsQBLayout.layout()));
    Tensor sdOtemp = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdO{});
    Tensor tdPsdOBLayout = smem_thr_copy_BLayout.partition_S(sdOtemp);
    Tensor tdPsdO = make_tensor(tdPsdOBLayout.data(), convert_layout_B_rowcol<_64x32, kHeadDimV/32>(tdPsdOBLayout.layout()));

    // dV/dK
    // auto smem_tiled_copy_QdOt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dkv);
    // auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    // Tensor tdVsdOt8x64 = smem_thr_copy_QdOt.partition_S(sdOt);
    // Tensor tdVsdOt = make_tensor(tdVsdOt8x64.data(), convert_layout_B_rowcol<_16x128>(tdVsdOt8x64.layout()));
    // Tensor tdKsQt8x64 = smem_thr_copy_QdOt.partition_S(sQt);
    // Tensor tdKsQt = make_tensor(tdKsQt8x64.data(), convert_layout_B_rowcol<_16x128>(tdKsQt8x64.layout()));
    auto smem_tiled_copy_dOt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dkv);
    auto smem_thr_copy_dOt = smem_tiled_copy_dOt.get_thread_slice(tidx);
    Tensor tdVsdOt8x64 = smem_thr_copy_dOt.partition_S(sdOt);
    Tensor tdVsdOt = make_tensor(tdVsdOt8x64.data(), convert_layout_B_rowcol<_16x128>(tdVsdOt8x64.layout()));
    auto smem_tiled_copy_Qt = make_tiled_copy_B(Copy_Atom<GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_dkv);
    auto smem_thr_copy_Qt = smem_tiled_copy_Qt.get_thread_slice(tidx);
    Tensor tdKsQt8x64 = smem_thr_copy_Qt.partition_S(sQt);
    Tensor tdKsQt = make_tensor(tdKsQt8x64.data(), convert_layout_B_rowcol<_16x128>(tdKsQt8x64.layout()));
    Tensor tdKsQtTail8x64 = smem_thr_copy_Qt.partition_S(sQtTail);
    Tensor tdKsQtTail = make_tensor(tdKsQtTail8x64.data(), convert_layout_B_rowcol<_16x64_128>(tdKsQtTail8x64.layout()));

    //
    // PREDICATES
    //

    Tensor cK = make_identity_tensor(make_shape(size<0>(gK), size<1>(gK)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cV = make_identity_tensor(make_shape(size<0>(gV), size<1>(gV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKcK = gmem_thr_copy_KV.partition_D(cK);
    Tensor tVcV = gmem_thr_copy_KV.partition_D(cV);

    // Allocate predicate tensors for k
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tSgK)));
    Tensor tVpV = make_tensor<bool>(make_shape(size<2>(tdPgV)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tKpK); ++k) { tKpK(k) = get<1>(tKcK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tVpV); ++k) { tVpV(k) = get<1>(tVcV(0, 0, k)) < params.d_value; }
    }

    int m_block = m_block_max - 1;
    int m_block_min = (!Is_causal && !Is_local)
        ? 0
        : std::max(0, (n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right) / kBlockM);

    if ((Is_local || !Is_even_MN) && m_block < m_block_min) {
        const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
        + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
        const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
        + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.dk_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                                Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                                make_stride(params.dv_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        clear(tdKrdK);
        clear(tdVrdV);
        Tensor cdK = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor cdV = make_identity_tensor(make_shape(size<0>(gdV), size<1>(gdV)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tdKcdK = gmem_thr_copy_dKV.partition_D(cdK);
        Tensor tdVcdV = gmem_thr_copy_dKV.partition_D(cdV);
        Tensor tdKpdK = make_tensor<bool>(make_shape(size<2>(tdKcdK)));
        Tensor tdVpdV = make_tensor<bool>(make_shape(size<2>(tdVcdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKpdK); ++k) { tdKpdK(k) = get<1>(tdKcdK(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tdVpdV); ++k) { tdVpdV(k) = get<1>(tdVcdV(0, 0, k)) < params.d_value; }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKcdK, tdKpdK, binfo.actual_seqlen_k - n_block * kBlockN
        );
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdVcdV, tdVpdV, binfo.actual_seqlen_k - n_block * kBlockN
        );
        return;
    }
    
    
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tSgK, tSrK, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN
    );

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_KV, tdPgV, tdPrV, tVcV, tVpV, binfo.actual_seqlen_k - n_block * kBlockN
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockN>, Int<kBlockM>>{});    // (BLK_N,BLK_M) -> (blk_n,blk_m)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);  
    
    flash::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
        bidb, bidh, tidx, params.h);

    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDimV>>{});
    Tensor acc_dk0 = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDimV>>{});
    Tensor acc_dk1 = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<64>>{});

    clear(acc_dv);
    clear(acc_dk0);
    clear(acc_dk1);

    Tensor taccScS_row = taccScS(_, 0, _);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }
    
    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Alibi<Is_causal> alibi(alibi_slope, binfo.actual_seqlen_k, binfo.actual_seqlen_q);
    
    constexpr int kS_loops = kHeadDim / 32;      // 6
    constexpr int kdV_loops = kBlockM / 16;      // 4
    constexpr int kdP_loops = kHeadDimV / 32;    // 4
    constexpr int kdK0_loops = kBlockM / 16;     // 4
    constexpr int kdK1_loops = kBlockM / 16;     // 4
    static_assert(kStages <= kS_loops && kStages <= kdV_loops && kStages <= kdP_loops && kStages <= kdK0_loops && kStages <= kdK1_loops, "kStages is error");
    /**
    * S   0 --- 2048 --- 4096 --- 6144 --- 8192
    *       0/4      1/5       2        3
    * dV  0 --- 2048 --- 4096 --- 6144 --- 8192
    *        0        1        2        3
    * dP  0 --- 2048 --- 4096 --- 6144 --- 8192
    *        0        1        2        3
    * dK0 0 --- 2048 --- 4096 --- 6144 --- 8192
    *        0        1        2        3
    * dk1 4096 --- 5120 --- 6144 --- 7168 --- 8192
    *           0        1        2        3
    */
    #pragma unroll
    for (int i = 0; i < kStages; ++i) { // 0 1 2
        lds_direct_copy<Is_even_K, Is_even_MN, _64x32, K_BUFF_SIZE>(gQ, sQ, i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    }
    #pragma unroll
    for (; m_block >= m_block_min; m_block--) {

        Tensor acc_s_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{}); 
        clear(acc_s_ori);

        #pragma unroll
        for (int i = 0; i < kS_loops - kStages; ++i) {
            // load 3 4 5 -> 3 0 1 
            // k0/k1 0 1 2 
            lds_direct_copy<Is_even_K, Is_even_MN, _64x32, K_BUFF_SIZE>(gQ, sQ, kStages + i, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            S_WAITCNT3;
            flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, i, i);
            S_BARRIER;
        }

        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            // load 2 3 0
            // k0 3 4 5
            // k1 3 0 1
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128, K_BUFF_SIZE>(gdO, sdOt, (i+2)%4, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
            S_WAITCNT3;
            flash::gemm_k_rs(acc_s_ori, tSrK, tSrQ, tSsQ, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, i+3, (i+3)%4);
            S_BARRIER;
        }
        
        Tensor acc_s = make_tensor(acc_s_ori.data(), convert_layout_acc(acc_s_ori.layout()));
    
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     __syncthreads();
        //     float * tmp = reinterpret_cast<float*>(acc_s.data());
        //     printf("acc m_block:%d tid:%d 0:%10.4f 1:%10.4f 2:%10.4f 3:%10.4f 4:%10.4f 5:%10.4f 6:%10.4f 7:%10.4f "
        //     "8:%10.4f 9:%10.4f 10:%10.4f 11:%10.4f 12:%10.4f 13:%10.4f 14:%10.4f 15:%10.4f "
        //     "16:%10.4f 17:%10.4f 18:%10.4f 19:%10.4f 20:%10.4f 21:%10.4f 22:%10.4f 23:%10.4f "
        //     "24:%10.4f 25:%10.4f 26:%10.4f 27:%10.4f 28:%10.4f 29:%10.4f 30:%10.4f 31:%10.4f\n", m_block, tidx, 
        //     acc_s(0), acc_s(1), acc_s(2), acc_s(3), acc_s(4), acc_s(5), acc_s(6), acc_s(7), 
        //     acc_s(8), acc_s(9), acc_s(10), acc_s(11), acc_s(12), acc_s(13), acc_s(14), acc_s(15), 
        //     acc_s(16), acc_s(17), acc_s(18), acc_s(19), acc_s(20), acc_s(21), acc_s(22), acc_s(23), 
        //     acc_s(24), acc_s(25), acc_s(26), acc_s(27), acc_s(28), acc_s(29), acc_s(30), acc_s(31) 
        //     );
        // }

        Tensor scores_trans = make_tensor(acc_s.data(), flash::convert_trans_layout_acc_rowcol(acc_s.layout()));
        if constexpr (Is_softcap) {
            flash::apply_softcap(acc_s, params.softcap);
        }
        [[maybe_unused]] Tensor dtanh_trans = make_tensor_like(scores_trans);
        if constexpr (Is_softcap) {
            flash::calculate_dtanh(scores_trans, dtanh_trans, params.softcap);
        }

        #if 1
        if constexpr (Has_alibi) {
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            const int wave_id = tidx / 64;
            const int col_idx_offset =  m_block * kBlockM;
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            alibi.apply_alibi_trans(scores, col_idx_offset, row_idx_offset_, kNWarps * 16);
        }
        #endif

        #if 1
        if constexpr(!Is_causal && !Is_local) {
            if (!Is_even_MN && (m_block + 1) * kBlockM >= binfo.actual_seqlen_q) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int warp_id = tidx / 64;
                // 实际上是row
                const int col_idx_offset_ = m_block * kBlockM;
                flash::apply_mask_trans(scores, binfo.actual_seqlen_q, col_idx_offset_);
            }
        } else if constexpr(Is_causal) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k)
            {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_causal_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16
                );
            }
        } else if constexpr(Is_local) {
            if (m_block * kBlockM < (n_block + 1) * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k - params.window_size_right
                || (m_block + 1) * kBlockM >= n_block * kBlockN + binfo.actual_seqlen_q - binfo.actual_seqlen_k + params.window_size_left) {
                Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int wave_id = (tidx >> 6);
                const int wave_id_to_row_block_id = wave_id;
                const int warp_row_stride = 16;
                const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
                const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
                flash::apply_mask_local_trans(
                    scores,
                    m_block * kBlockM,
                    binfo.actual_seqlen_k,
                    row_idx_offset_,
                    binfo.actual_seqlen_q,
                    kNWarps * 16,
                    params.window_size_left, params.window_size_right
                );
            }
        }
        #endif
        
        flash::scale_apply_exp2</*scale_max=*/false>(scores_trans, lse, params.scale_softmax_log2);

        Tensor dP_sum = make_fragment_like(lse);
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
            dP_sum(mi) = gdPsum(row);
        }

        if (m_block > m_block_min)  {
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = (laneId / 16) * 4 + (mi % 4) + (mi / 4) * 16;
                lse(mi) = gLSE(row);
            }
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
        }

        if constexpr (Is_dropout) {
            const int warp_id = tidx / 64;
            const int wave_id = (tidx >> 6);
            const int wave_id_to_row_block_id = wave_id;
            const int warp_row_stride = 16;
            const int row_idx_offset_in_block = (tidx & (warp_row_stride - 1)) + (wave_id_to_row_block_id << 4);
            const int row_idx_offset_ = n_block * kBlockN + row_idx_offset_in_block;
            int block_row_idx = row_idx_offset_;
            int block_col_idx = m_block * kBlockM;
            dropout.template apply_dropout_trans</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, kNWarps * 16
            );
        }

        Tensor rP = !Is_dropout
            ? flash::convert_type<Element>(acc_s)
            : flash::convert_type_relu<Element>(acc_s);
        
        // load  1
        // k0/k1 2
        lds_direct_copy<Is_even_K, Is_even_MN, _16x128, K_BUFF_SIZE>(gdO, sdOt, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        S_WAITCNT3;
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_dOt, smem_thr_copy_dOt);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN, _64x32, K_BUFF_SIZE>(gdO, sdO, (0+2)%4, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        S_WAITCNT3;
        flash::gemm_k_rs_ds_read_m32x16<(0+3)%4>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_dOt, smem_thr_copy_dOt);
        S_BARRIER;
        lds_direct_copy<Is_even_K, Is_even_MN, _64x32, K_BUFF_SIZE>(gdO, sdO, (1+2)%4, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        S_WAITCNT3;
        flash::gemm_k_rs_ds_read_m32x16<(1+3)%4>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_dOt, smem_thr_copy_dOt);
        S_BARRIER;
        lds_direct_copy<Is_even_K, Is_even_MN, _64x32, K_BUFF_SIZE>(gdO, sdO, (2+2)%4, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        S_WAITCNT3;
        flash::gemm_k_rs_ds_read_m32x16<(2+3)%4>(acc_dv, rP, tdVrdO, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_dOt, smem_thr_copy_dOt);
        S_BARRIER;

        // return;
        Tensor acc_dp_ori = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockN>, Int<kBlockM>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_dp_ori);

        // load  1
        // k0/k1 2
        lds_direct_copy<Is_even_K, Is_even_MN, _64x32, K_BUFF_SIZE>(gdO, sdO, 1, params.do_row_stride, params.d_value, binfo.actual_seqlen_q - m_block * kBlockM);
        S_WAITCNT3;
        flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, 2);
        S_BARRIER;

        #pragma unroll
        for (int i = 0; i < kStages; ++i) { // tail kStages
            // load 2 3 0
            // k0 3 0 1
            // k1 3 0 1
            lds_direct_copy<Is_even_K, Is_even_MN, _16x128, K_BUFF_SIZE>(gQ, sQt, (i+2)%4, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
            S_WAITCNT3;
            flash::gemm_k_rs(acc_dp_ori, tdPrV, tdPrdO, tdPsdO, tiled_mma_sdp, smem_tiled_copy_QdO, smem_thr_copy_QdO, (i+3)%4);
            S_BARRIER;
        }
        Tensor acc_dp = make_tensor(acc_dp_ori.data(), convert_layout_acc(acc_dp_ori.layout()));
    
        // wangaq debug
        // __syncthreads();
        // if (blockIdx.x == 0) {
        //     __syncthreads();
        //     printf("dP m_block:%d tid:%d 0:%10.4f 1:%10.4f 2:%10.4f 3:%10.4f 4:%10.4f 5:%10.4f 6:%10.4f 7:%10.4f "
        //     "8:%10.4f 9:%10.4f 10:%10.4f 11:%10.4f 12:%10.4f 13:%10.4f 14:%10.4f 15:%10.4f "
        //     "16:%10.4f 17:%10.4f 18:%10.4f 19:%10.4f 20:%10.4f 21:%10.4f 22:%10.4f 23:%10.4f "
        //     "24:%10.4f 25:%10.4f 26:%10.4f 27:%10.4f 28:%10.4f 29:%10.4f 30:%10.4f 31:%10.4f\n", m_block, tidx, 
        //     acc_dp(0), acc_dp(1), acc_dp(2), acc_dp(3), acc_dp(4), acc_dp(5), acc_dp(6), acc_dp(7), 
        //     acc_dp(8), acc_dp(9), acc_dp(10), acc_dp(11), acc_dp(12), acc_dp(13), acc_dp(14), acc_dp(15), 
        //     acc_dp(16), acc_dp(17), acc_dp(18), acc_dp(19), acc_dp(20), acc_dp(21), acc_dp(22), acc_dp(23), 
        //     acc_dp(24), acc_dp(25), acc_dp(26), acc_dp(27), acc_dp(28), acc_dp(29), acc_dp(30), acc_dp(31) 
        //     );
        // }

        Tensor dS = make_tensor(acc_dp.data(), scores_trans.layout());

        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores_trans(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh_trans(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        Tensor tdKrdSt = flash::convert_type<Element>(acc_dp);

        // load  1
        // k0/k1 2
        lds_direct_copy<Is_even_K, Is_even_MN, _16x128, K_BUFF_SIZE>(gQ, sQt, 1, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        S_WAITCNT3;
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dk0, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_Qt, smem_thr_copy_Qt);
        S_BARRIER;

        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_128, K_BUFF_SIZE>(gQ, sQtTail, 0, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        S_WAITCNT3;
        flash::gemm_k_rs_ds_read_m32x16<(0+3)%4>(acc_dk0, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_Qt, smem_thr_copy_Qt);
        S_BARRIER;
        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_128, K_BUFF_SIZE>(gQ, sQtTail, 1, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        S_WAITCNT3;
        flash::gemm_k_rs_ds_read_m32x16<(1+3)%4>(acc_dk0, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_Qt, smem_thr_copy_Qt);
        S_BARRIER;
        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_128, K_BUFF_SIZE>(gQ, sQtTail, 2, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        S_WAITCNT3;
        flash::gemm_k_rs_ds_read_m32x16<(2+3)%4>(acc_dk0, tdKrdSt, tdKrQt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_Qt, smem_thr_copy_Qt);
        S_BARRIER;
        // load  3
        // k0/k1 0
        lds_direct_copy<Is_even_K, Is_even_MN, _16x64_128, K_BUFF_SIZE>(gQ, sQtTail, 3, params.q_row_stride, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
        S_WAITCNT3;
        flash::gemm_k_rs_ds_read_m32x16<0>(acc_dk1, tdKrdSt, tdKrQtTail, tdKsQtTail, tiled_mma_dkv, smem_tiled_copy_Qt, smem_thr_copy_Qt);
        S_BARRIER;

        // tail kStages == 3
        S_WAITCNT2;
        flash::gemm_k_rs_ds_read_m32x16<1>(acc_dk1, tdKrdSt, tdKrQtTail, tdKsQtTail, tiled_mma_dkv, smem_tiled_copy_Qt, smem_thr_copy_Qt);
        S_BARRIER;
        S_WAITCNT1;
        flash::gemm_k_rs_ds_read_m32x16<2>(acc_dk1, tdKrdSt, tdKrQtTail, tdKsQtTail, tiled_mma_dkv, smem_tiled_copy_Qt, smem_thr_copy_Qt);
        S_BARRIER;
        S_WAITCNT0;
        flash::gemm_k_rs_ds_read_m32x16<3>(acc_dk1, tdKrdSt, tdKrQtTail, tdKsQtTail, tiled_mma_dkv, smem_tiled_copy_Qt, smem_thr_copy_Qt);
        S_BARRIER;

        // wangaq debug
        // __syncthreads();
        // if (thread0()) {
        //     Element * tmp = reinterpret_cast<Element*>(sQtTail.data().get());
        //     int col = 8;
        //     for (int i = 0; i < size(sQtTail)/col; ++i) {
        //         printf("sQtTail:%d ", i);
        //         for (int j = 0; j < col; ++j) {
        //             printf("%10.4f ", float(tmp[i*col+j]));
        //         }
        //         printf("\n");
        //     }
        // }

        if (m_block > m_block_min) {
            gQ.data() = gQ.data() + (-int(kBlockM * params.q_row_stride));
            gdO.data() = gdO.data() + (-int(kBlockM * params.do_row_stride));
            #pragma unroll
            for (int i = 0; i < kStages; ++i) { // 0 1 2
                lds_direct_copy<Is_even_K, true, _64x32, K_BUFF_SIZE>(gQ, sQ, i, params.q_row_stride, params.d);
            }
        } 

    }

    // wangaq debug
    //  __syncthreads();
    //  if (blockIdx.x == 0) {
    //      __syncthreads();
    //      float * tmp = reinterpret_cast<float*>(acc_dk.data());
    //      printf("dK tid:%d 0:%10.4f 1:%10.4f 2:%10.4f 3:%10.4f 4:%10.4f 5:%10.4f 6:%10.4f 7:%10.4f "
    //      "8:%10.4f 9:%10.4f 10:%10.4f 11:%10.4f 12:%10.4f 13:%10.4f 14:%10.4f 15:%10.4f "
    //      "16:%10.4f 17:%10.4f 18:%10.4f 19:%10.4f 20:%10.4f 21:%10.4f 22:%10.4f 23:%10.4f "
    //      "24:%10.4f 25:%10.4f 26:%10.4f 27:%10.4f 28:%10.4f 29:%10.4f 30:%10.4f 31:%10.4f\n", tidx, 
    //      tmp[0], tmp[1], tmp[2], tmp[3], 
    //      tmp[4], tmp[5], tmp[6], tmp[7],
    //      tmp[8], tmp[9], tmp[10], tmp[11], 
    //      tmp[12], tmp[13], tmp[14], tmp[15],
    //      tmp[16], tmp[17], tmp[18], tmp[19], 
    //      tmp[20], tmp[21], tmp[22], tmp[23],
    //      tmp[24], tmp[25], tmp[26], tmp[27], 
    //      tmp[28], tmp[29], tmp[30], tmp[31]
    //      );
    //  }

    #if 0
    if constexpr(Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dv from fp32 to fp16
    Tensor rdK = flash::convert_type<Element>(acc_dk);
    Tensor rdV = flash::convert_type<Element>(acc_dv);

    //  __syncthreads();
    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    // If Is_last, there's already a __syncthreads() at the end of the loop.
    // if constexpr(!Is_last) { __syncthreads(); }

    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    __builtin_amdgcn_s_barrier(); 
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    __builtin_amdgcn_s_barrier(); 
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
    __syncthreads();
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    __builtin_amdgcn_s_barrier(); 
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    __builtin_amdgcn_s_barrier(); 
    #elif 0
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.dv_row_stride, _1{}));
    _bwd_store_dk_dv<Kernel_traits, decltype(acc_dv), decltype(sQ), decltype(gdV), Element, 
        typename Kernel_traits::SmemLayoutdVStore, Is_even_MN, Is_even_K>(
        acc_dv, sQ, tidx, gdV, params.d_value, binfo.actual_seqlen_k - n_block * kBlockN);

    __syncthreads();
    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    _bwd_store_dk_dv<Kernel_traits, decltype(acc_dk), decltype(sQ), decltype(gdK), Element, 
        typename Kernel_traits::SmemLayoutdKStore, Is_even_MN, Is_even_K>(
        acc_dk, sQ, tidx, gdK, params.d, binfo.actual_seqlen_k - n_block * kBlockN);

    #else

    const index_t row_offset_dk = binfo.k_offset(params.dk_batch_stride, params.dk_row_stride, bidb)
    + n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_batch_stride, params.dv_row_stride, bidb)
    + n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.dv_row_stride, _1{}));
    
    int row, col;
    if constexpr (size<1>(acc_dk0) == size<1>(acc_dv) && size<2>(acc_dk0) == size<2>(acc_dv)) {
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dk0); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk0); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dk0); ++ei) {
                        if (Is_even_K || col < params.d_value) {
                            gdK(row, col) = flash::convert_type<Element>(acc_dk0(ei, mi, ni) * params.scale_softmax_rp_dropout);
                            gdV(row, col) = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout );
                        }
                        col += 4;
                    }
                }
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk1); ++ni) {
                    col = kHeadDimV + (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dk1); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdK(row, col) = flash::convert_type<Element>(acc_dk1(ei, mi, ni) * params.scale_softmax_rp_dropout);
                        }
                        col += 4;
                    }
                }
            }
        } 
    } else {
        
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dk0); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk0); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dk0); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdK(row, col) = flash::convert_type<Element>(acc_dk0(ei, mi, ni) * params.scale_softmax_rp_dropout);
                        }
                        col += 4;
                    }
                }
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dk1); ++ni) {
                    col = kHeadDimV + (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dk1); ++ei) {
                        if (Is_even_K || col < params.d) {
                            gdK(row, col) = flash::convert_type<Element>(acc_dk1(ei, mi, ni) * params.scale_softmax_rp_dropout);
                        }
                        col += 4;
                    }
                }
            }
        } 
        
        #pragma unroll
        for (int mi = 0; mi < size<1>(acc_dv); ++mi) {
            row = (mi*kNWarps + warpId) * 16 + (laneId % 16);
            if (Is_even_MN || row < binfo.actual_seqlen_k - n_block * kBlockN) {
                #pragma unroll
                for (int ni = 0; ni < size<2>(acc_dv); ++ni) {
                    col = (laneId / 16) + ni * 32;
                    #pragma unroll
                    for (int ei = 0; ei < size<0>(acc_dv); ++ei) {
                        if (Is_even_K || col < params.d_value) {
                            gdV(row, col) = flash::convert_type<Element>(!Is_dropout ? acc_dv(ei, mi, ni) : acc_dv(ei, mi, ni) * params.rp_dropout);
                        }
                        col += 4;
                    }
                }
            }
        } 
    }

    #endif

}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_dv_trans_seqk_parallel_16x64(const Params &params) {
    // The block index for the batch.
    const int bidb = blockIdx.z;
    // The block index for the head.
    const int bidh = blockIdx.y;
#if 0
    // If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
    for (int n_block = blockIdx.x; n_block < (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN; n_block += gridDim.x) {
        compute_dk_dv_trans_1colblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap,
            /*Is_first*/false, /*Is_last*/false, /*Seq_parallel=*/true>(params, bidb, bidh, n_block);
    }
#else
    const int n_block = blockIdx.x;
    using Element = typename Kernel_traits::Element;
    if constexpr (Kernel_traits::kHeadDim == 128)
    {
        compute_dk_dv_trans_1colblock_16x64_dim128_fp16<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>
        (params, bidb, bidh, n_block);
        return;
    }
    
    if constexpr (!Is_even_K)
    {
        compute_dk_dv_trans_1colblock_16x64_dim40<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>
        (params, bidb, bidh, n_block);
    }else {
            compute_dk_dv_trans_1colblock_16x64<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>
        (params, bidb, bidh, n_block);
    }


#endif
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_dv_trans_16x64_prefetch(const Params &params) {
    // The block index for the batch.
    const int bidb = blockIdx.z;
    // The block index for the head.
    const int bidh = blockIdx.y;
    const int n_block = blockIdx.x;
    using Element = typename Kernel_traits::Element;

    if constexpr (Kernel_traits::kHeadDim == 128) {
        compute_dk_dv_trans_1colblock_16x64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, n_block);
        #ifndef NO_CAUSAL_OPT
            if constexpr (Is_causal)
            {
                const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
                if (num_n_block - n_block - 1 != num_n_block) {
                    compute_dk_dv_trans_1colblock_16x64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, num_n_block - n_block - 1);
                }
            }
        #endif
    } else if constexpr (Kernel_traits::kHeadDim == 96) {
        compute_dk_dv_trans_1colblock_16x64_dim96_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, n_block);
        if constexpr (Is_causal)
        {
            const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
            if (num_n_block - n_block - 1 != num_n_block) {
                compute_dk_dv_trans_1colblock_16x64_dim96_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, num_n_block - n_block - 1);
            }
        }
    } else if constexpr (Kernel_traits::kHeadDim == 64) {
        if constexpr(Is_causal)
        {
            const int bidbh = blockIdx.x + blockIdx.z * params.se_balance_cnt;
            const int bidb = bidbh / params.h;
            const int bidh = bidbh % params.h;
            const int n_block = blockIdx.y;
            compute_dk_dv_trans_1colblock_16x64_dim64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, n_block);
        }
        else
        {
            compute_dk_dv_trans_1colblock_16x64_dim64_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, n_block);

        }

    }     
    #if 1
    else if constexpr (Kernel_traits::kHeadDim == 512) {
        compute_dk_dv_trans_1colblock_16x64_dim512_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, n_block);
        if constexpr (Is_causal)
        {
            const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
            if (num_n_block - n_block - 1 != num_n_block) {
                compute_dk_dv_trans_1colblock_16x64_dim512_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, num_n_block - n_block - 1);
            }
        }
    } 
    #endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_trans_16x64_prefetch(const Params &params) {
    const int bidb = blockIdx.z;
    // The block index for the head.
    const int bidh = blockIdx.y;
    const int n_block = blockIdx.x;
    using Element = typename Kernel_traits::Element;
    compute_dk_trans_1colblock_16x64_dim256_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, n_block);
    if constexpr (Is_causal)
    {
        const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
        if (num_n_block - n_block - 1 != num_n_block) {
            // compute_dk_trans_1colblock_16x64_dim256_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, num_n_block - n_block - 1);
            compute_dk_trans_1colblock_16x64_dim256_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, num_n_block - n_block - 1);
        }
    }
}
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dv_trans_16x64_prefetch(const Params &params) {
    const int bidb = blockIdx.z;
    // The block index for the head.
    const int bidh = blockIdx.y;
    const int n_block = blockIdx.x;
    using Element = typename Kernel_traits::Element;
    compute_dv_trans_1colblock_16x64_dim256_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, n_block);
    if constexpr (Is_causal)
    {
        const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
        if (num_n_block - n_block - 1 != num_n_block) {
            compute_dv_trans_1colblock_16x64_dim256_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, num_n_block - n_block - 1);
            // compute_dv_trans_1colblock_16x64_dim256_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, num_n_block - n_block - 1);
        }
    }
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_dk_dv_trans_16x64_mla_prefetch(const Params &params) {
    // The block index for the batch.
    const int bidb = blockIdx.z;
    // The block index for the head.
    const int bidh = blockIdx.y;
    const int n_block = blockIdx.x;
    using Element = typename Kernel_traits::Element;

    compute_dk_dv_trans_1colblock_16x64_mla_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, n_block);
    // #ifdef NO_CAUSAL_OPT

    // #else
    //     if constexpr (Is_causal)
    //     {
    //         const int num_n_block = (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    //         if (num_n_block - n_block - 1 != num_n_block) {
    //             compute_dk_dv_trans_1colblock_16x64_mla_prefetch<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params, bidb, bidh, num_n_block - n_block - 1);
    //         }
    //     }
    // #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////


} // namespace flash

