import triton
import triton.language as tl


@triton.autotune(
   configs=[
    #    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16}, num_stages=1, num_warps=1),
    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=1, num_warps=4),
    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=1, num_warps=4),
    #    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=1, num_warps=4),
    #    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=1, num_warps=4),
    #    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
    #    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
   ],
   key=['Z', 'H', 'Q_len', 'KV_len', 'BLOCK_DMODEL'],
)
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out, CtxLens,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_km, stride_kk,
              stride_vz, stride_vh, stride_vm, stride_vk,
              stride_oz, stride_oh, stride_om, stride_ok,
              Z, H,
              Q_len, KV_len, Pre_len,
              BLOCK_DMODEL: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              MASK: tl.constexpr
              ):
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    off_hz = off_z * H + off_h
    # qvk_offset = off_hz * stride_qh = off_z * stride_qz + off_h * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + off_z * stride_qz + off_h * stride_qh,
        shape=(Q_len, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + off_z * stride_vz + off_h * stride_vh,
        shape=(KV_len, BLOCK_DMODEL),
        strides=(stride_vm, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + off_z * stride_kz + off_h * stride_kh,
        shape=(BLOCK_DMODEL, KV_len),
        strides=(stride_kk, stride_km),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + off_z * stride_oz + off_h * stride_oh,
        shape=(Q_len, BLOCK_DMODEL),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    Ctx_lens_ptr =  tl.make_block_ptr(
        base=CtxLens,
        shape=(Z, 1),
        strides=(1, 1),
        offsets=(off_z, 0),
        block_shape=(1, 1),
        order=(1, 0),
    )
    ctx_len = tl.load(Ctx_lens_ptr)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(Q_block_ptr)
    qk_scale = sm_scale * RCP_LN2
    q = (q * qk_scale).to(q.dtype)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    for start_n in range(0, KV_len, BLOCK_N):
        # start_n = tl.multiple_of(start_n, BLOCK_N)  # start_n is multiples of BLOCK_N
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k)
        qk_mask = qk

        if MASK == 1: # causal & pre = 1
            mask = (offs_m[:, None] + Pre_len >= (start_n  + offs_n[None, :]))
        if MASK == 2: # ctx_len
            mask = (start_n + offs_n[None, :] < ctx_len)
        if MASK == 3: # both
            mask = (offs_m[:, None] + Pre_len >= (start_n + offs_n[None, :])) | (start_n + offs_n[None, :] < ctx_len)
        
        if MASK != 0:
            qk_mask = tl.where(mask, qk_mask, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk_mask, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        if MASK != 0:
            p = tl.where(mask, p, float(0.))
        l_ij = tl.sum(p, 1)
 
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        # -- update m_i and l_i

        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * Q_len + offs_m
    tl.store(m_ptrs, m_i + tl.math.log2(l_i))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

@triton.jit
def _attn_bwd_preprocess(O, DO,
                         Delta,
                         stride_oz, stride_oh, stride_om, stride_ok,
                         stride_doz, stride_doh, stride_dom, stride_dok,
                         Z, H, Q_len,
                         BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr
                         ):
    start_m = tl.program_id(0)
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    off_n = tl.arange(0, D_HEAD)
    off_hz = off_z * H + off_h
    O_block_ptr = tl.make_block_ptr(
        base=O + off_z * stride_oz + off_h * stride_oh,
        shape=(Q_len, D_HEAD),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO + off_z * stride_doz + off_h * stride_doh,
        shape=(Q_len, D_HEAD),
        strides=(stride_dom, stride_dok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_HEAD),
        order=(1,0)
    )
    o = tl.load(O_block_ptr)
    do = tl.load(DO_block_ptr).to(tl.float32)  # (do stride require)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * Q_len + off_m, delta)

@triton.jit
def _attn_bwd_dkdv(Q, K, V, qk_scale,
                   DO, ctx_len,
                   M, D,
                   # shared by Q/K/V/DO.
                   stride_qm, stride_qk,
                   stride_km, stride_kk,
                   stride_vm, stride_vk,
                   stride_dom, stride_dok,
                   H, Q_len, KV_len, Pre_len,
                   BLOCK_M1: tl.constexpr,
                   BLOCK_N1: tl.constexpr,
                   BLOCK_DMODEL: tl.constexpr,
                   MASK: tl.constexpr,
                   # Filled in by the wrapper.
                   start_n):
    dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)

    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(KV_len, BLOCK_DMODEL),
        strides=(stride_km, stride_kk),
        offsets=(start_n, 0),
        block_shape=(BLOCK_N1, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(KV_len, BLOCK_DMODEL),
        strides=(stride_vm, stride_vk),
        offsets=(start_n, 0),
        block_shape=(BLOCK_N1, BLOCK_DMODEL),
        order=(1, 0),
    )

    # load K and V: they stay in SRAM throughout the inner loop for dkdv.
    k = tl.load(K_block_ptr)
    k = (k  * qk_scale).to(k.dtype)
    v = tl.load(V_block_ptr)

    QT_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(BLOCK_DMODEL, Q_len),
        strides=(stride_qk, stride_qm),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_M1),
        order=(0,1)
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(Q_len, BLOCK_DMODEL),
        strides=(stride_dom, stride_dok),
        offsets=(0, 0),
        block_shape=(BLOCK_M1, BLOCK_DMODEL),
        order=(1,0)
    )

    offs_n = tl.arange(0, BLOCK_N1)
    for start_m in range(0, Q_len, BLOCK_M1):
        start_m = tl.multiple_of(start_m, BLOCK_M1)
        qT = tl.load(QT_block_ptr)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = start_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        do = tl.load(DO_block_ptr)
        qkT = tl.dot(k, qT)
        qkT = qkT
        pT = tl.math.exp2(qkT - m[None, :])

        if MASK == 1: # causal
            mask = (offs_m[None, :] + Pre_len >= start_n + offs_n[:, None])
        if MASK == 2: # ctx_len
            mask = (start_n + offs_n[:, None] < ctx_len)
        if MASK == 3: # both
            mask = (offs_m[None, :]  + Pre_len>= start_n + offs_n[:, None]) | (start_n + offs_n[:, None] < ctx_len)
        
        if MASK != 0:
            pT = tl.where(mask, pT, 0.0)

        # Compute dV.
        ppT = pT.to(tl.float16)

        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do))

        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT)) # dk += dsT·Q
        # Increment pointers.
        QT_block_ptr = tl.advance(QT_block_ptr, (0, BLOCK_M1))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M1, 0))
    return dk, dv

@triton.jit
def _attn_bwd_dq(Q, K, V, qk_scale,
                 DO, M, D, ctx_len,
                 # shared by Q/K/V/DO.
                stride_qm, stride_qk,
                stride_km, stride_kk,
                stride_vm, stride_vk,
                stride_dom, stride_dok,
                 H, Q_len, KV_len, Pre_len,
                 BLOCK_M2: tl.constexpr,
                 BLOCK_N2: tl.constexpr,
                 BLOCK_DMODEL: tl.constexpr,
                 MASK: tl.constexpr,
                 start_m):
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(Q_len, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M2, BLOCK_DMODEL),
        order=(1, 0)
    )

    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(Q_len, BLOCK_DMODEL),
        strides=(stride_dom, stride_dok),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M2, BLOCK_DMODEL),
        order=(1, 0)
    )
    q = tl.load(Q_block_ptr)
    do = tl.load(DO_block_ptr)
    dq = tl.zeros([BLOCK_M2, BLOCK_DMODEL], dtype=tl.float32)

    m = tl.load(M + offs_m)
    m = m[:, None]

    KT_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(BLOCK_DMODEL, KV_len),
        strides=(stride_kk, stride_km),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N2),
        order=(0, 1)
    )
    VT_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(BLOCK_DMODEL, KV_len),
        strides=(stride_vk, stride_vm),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N2),
        order=(0, 1)
    )
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    offs_n = tl.arange(0, BLOCK_N2)
    for start_n in range(0, KV_len, BLOCK_N2):
        kT = tl.load(KT_block_ptr)
        kT = (kT * qk_scale).to(kT.dtype)
        qk = tl.dot(q, kT)
        qk = qk
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.

        if MASK == 1: # causal
            mask = (offs_m[:, None] + Pre_len >= start_n + offs_n[None, :])
        if MASK == 2: # ctx_len
            mask = (start_n + offs_n[None, :] < ctx_len)
        if MASK == 3: # both
            mask = (offs_m[:, None] + Pre_len >= start_n + offs_n[None, :]) | (start_n + offs_n[None, :]  < ctx_len)
        
        if MASK != 0:
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        vT = tl.load(VT_block_ptr)
        dp = tl.dot(do, vT).to(tl.float32)

        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        KT_block_ptr = tl.advance(KT_block_ptr, (0, BLOCK_N2))
        VT_block_ptr = tl.advance(VT_block_ptr, (0, BLOCK_N2))
    return dq


@triton.autotune(
   configs=[
    #    triton.Config({'BLOCK_M1': 16, 'BLOCK_N1': 32, 'BLOCK_M2': 16, 'BLOCK_N2': 16}, num_stages=1, num_warps=1),
       triton.Config({'BLOCK_M1': 32, 'BLOCK_M2': 32, 'BLOCK_N2': 32}, num_stages=1, num_warps=4),
    #    triton.Config({'BLOCK_M1': 64, 'BLOCK_M2': 128, 'BLOCK_N2': 64}, num_stages=1, num_warps=4),
    #    triton.Config({'BLOCK_M1': 64, 'BLOCK_M2': 64, 'BLOCK_N2': 64}, num_stages=1, num_warps=4),
    #    triton.Config({'BLOCK_M1': 32, 'BLOCK_N1': 256, 'BLOCK_M2': 128, 'BLOCK_N2': 32}, num_stages=1, num_warps=4),
    #    triton.Config({'BLOCK_M1': 32, 'BLOCK_N1': 256, 'BLOCK_M2': 128, 'BLOCK_N2': 32}, num_stages=1, num_warps=8),
   ],
   key=['H', 'Q_len', 'KV_len', 'BLOCK_DMODEL'],
)
@triton.heuristics({
    'NMulti': lambda args: (args['KV_len'] // args['Q_len']),
})
@triton.jit
def _attn_bwd(Q, K, V, sm_scale,
              DO,
              DQ, DK, DV,
              M, D, CtxLens,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_km, stride_kk,
              stride_vz, stride_vh, stride_vm, stride_vk,
              stride_doz, stride_doh, stride_dom, stride_dok,
              stride_dqz, stride_dqh, stride_dqm, stride_dqk,
              stride_dkz, stride_dkh, stride_dkm, stride_dkk,
              stride_dvz, stride_dvh, stride_dvm, stride_dvk,
              # H = 16, N_CTX = 1024
              Z, H, Q_len, KV_len, Pre_len,
              BLOCK_DMODEL: tl.constexpr,
              BLOCK_M1: tl.constexpr,
            #   BLOCK_N1: tl.constexpr,
              BLOCK_M2: tl.constexpr,
              BLOCK_N2: tl.constexpr,
              MASK: tl.constexpr,
              NMulti: tl.constexpr
              ):
    BLOCK_N1: tl.constexpr = NMulti * BLOCK_M2
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    pid = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    off_zh = off_z*H + off_h

    Ctx_lens_ptr =  tl.make_block_ptr(
        base=CtxLens,
        shape=(Z, 1),
        strides=(1, 1),
        offsets=(off_z, 0),
        block_shape=(1, 1),
        order=(1, 0),
    )
    ctx_len = tl.load(Ctx_lens_ptr)

    # offset pointers for batch/head
    Q += (stride_qh * off_h + stride_qz * off_z).to(tl.int64)
    K += (stride_kh * off_h + stride_kz * off_z).to(tl.int64)
    V += (stride_vh * off_h + stride_vz * off_z).to(tl.int64)
    DO += (stride_doh * off_h + stride_doz * off_z).to(tl.int64)
    DQ += (stride_dqh * off_h + stride_dqz * off_z).to(tl.int64)
    DK += (stride_dkh * off_h + stride_dkz * off_z).to(tl.int64)
    DV += (stride_dvh * off_h + stride_dvz * off_z).to(tl.int64)
    M += (off_zh * Q_len).to(tl.int64)
    D += (off_zh * Q_len).to(tl.int64)

    start_n = pid * BLOCK_N1
    dk, dv = _attn_bwd_dkdv(Q, K, V, RCP_LN2*sm_scale,
                        DO, ctx_len,
                        M, D,
                        stride_qm, stride_qk,
                        stride_km, stride_kk,
                        stride_vm, stride_vk,
                        stride_dom, stride_dok,
                        H, Q_len, KV_len, Pre_len,
                        BLOCK_M1, BLOCK_N1, BLOCK_DMODEL, MASK,
                        start_n
                        )

    DV_block_ptrs = tl.make_block_ptr(
        base=DV,
        shape=(KV_len, BLOCK_DMODEL),
        strides=(stride_dvm, stride_dvk),
        offsets=(start_n, 0),
        block_shape=(BLOCK_N1, BLOCK_DMODEL),
        order=(1,0)
    )
    tl.store(DV_block_ptrs, dv.to(tl.float16))

    # Write back dK.
    dk *= sm_scale
    DK_block_ptrs = tl.make_block_ptr(
        base=DK,
        shape=(KV_len, BLOCK_DMODEL),
        strides=(stride_dkm, stride_dkk),
        offsets=(start_n, 0),
        block_shape=(BLOCK_N1, BLOCK_DMODEL),
        order=(1,0)
    )
    tl.store(DK_block_ptrs, dk.to(tl.float16))

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    dq = _attn_bwd_dq(Q, K, V, RCP_LN2*sm_scale,
                      DO, M, D, ctx_len,
                      stride_qm, stride_qk,
                      stride_km, stride_kk,
                      stride_vm, stride_vk,
                      stride_dom, stride_dok,
                      H, Q_len, KV_len, Pre_len,
                      BLOCK_M2, BLOCK_N2, BLOCK_DMODEL, MASK,
                      start_m
                      )
    dq *= LN2
    # Write back dQ.
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ,
        shape=(Q_len, BLOCK_DMODEL),
        strides=(stride_dqm, stride_dqk),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M2, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(DQ_block_ptr, dq.to(tl.float16))

import torch
class _flash_attn_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, ctx_lens = None, MASK = 3): # MASK[causal=1, ctx_len=2, both=3]
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q, dtype=v.dtype)
        grid = lambda META: (
            triton.cdiv(q.shape[2], META['BLOCK_M']),
            q.shape[1],  # bhmk
            q.shape[0]
        )
        Q_len = q.shape[2]
        KV_len = k.shape[2]
        Pre_len = KV_len - Q_len
        ctx_lens = ctx_lens + Pre_len
        print(f"{ctx_lens=}, {Pre_len=}")
        # mask = torch.empty((q.shape[0], q.shape[1], Q_len, KV_len), device=q.device, dtype=torch.float32)
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd[grid](
            q, k, v, sm_scale, M, o, ctx_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1],
            Q_len, KV_len, Pre_len,
            BLOCK_DMODEL=Lk, MASK=MASK
        )
        ## restore the grid for bwd kernel
        best_config = _attn_fwd.get_best_config()
        print(f"fwd best_config = {str(best_config)}")
        block_m = int(best_config.__str__().split(",")[0].split("BLOCK_M:")[1])
        grid = (triton.cdiv(q.shape[2], block_m), q.shape[0] * q.shape[1], 1)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.ctx_lens = ctx_lens
        ctx.mask = MASK
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, Q_len = q.shape[:3]
        KV_len = k.shape[2]
        PRE_BLOCK = 64
        # RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        # arg_k = k * (ctx.sm_scale * RCP_LN2)
        assert Q_len % PRE_BLOCK == 0
        pre_grid = (Q_len // PRE_BLOCK, N_HEAD, BATCH)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,
            delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            BATCH, N_HEAD, Q_len,
            BLOCK_M=PRE_BLOCK, D_HEAD=ctx.BLOCK_DMODEL
        )
        grid = lambda META: (
            triton.cdiv(Q_len, META['BLOCK_N2']),
            N_HEAD,
            BATCH
        )
        _attn_bwd[grid](
            q, k, v, ctx.sm_scale, do, dq, dk, dv,
            M, delta, ctx.ctx_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            BATCH, N_HEAD, Q_len, KV_len, KV_len - Q_len,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            MASK=ctx.mask
        )
        best_config = _attn_bwd.get_best_config()
        print(f"bwd best_config = {str(best_config)}")

        return dq, dk, dv, None, None, None, None

flash_attn_func = _flash_attn_func.apply

