# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.utils.sparse_utils import block_map_lut_triton, block_map_to_block_offset_triton, get_block_map

# isort: off
# We need to import the CUDA kernels after importing torch
import flash_attn_2_cuda as flash_attn_cuda

# isort: on

from flash_attn.utils.sparse_utils import hyperparameter_check, get_block_map_meansim

DEFAULT_FA_VERSION = 2

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def round_multiple(x, m):
    return (x + m - 1) // m * m


def _get_block_size_n(device, head_dim, is_dropout, is_causal):
    # This should match the block sizes in the CUDA kernel
    assert head_dim <= 512
    major, minor = torch.cuda.get_device_capability(device)
    is_sm8x = major == 8 and minor > 0  # Only include sm86 and sm89, exclude sm80 (A100)
    is_sm80 = major == 8 and minor == 0
    is_sm90 = major == 9 and minor == 0
    if head_dim <= 32:
        return 128
    if head_dim <= 64:
        return 128 if not is_dropout else 64
    elif head_dim <= 96:
        return 64
    elif head_dim <= 128:
        if is_sm8x:
            return 64 if (not is_dropout and is_causal) else 32
        else:
            return 64 if not is_dropout else 32
    elif head_dim <= 160:
        if is_sm8x:
            return 64
        else:
            return 32
    elif head_dim <= 192:
        return 64
    elif head_dim <= 224:
        return 64
    elif head_dim <= 256:
        return 64
    elif head_dim <= 512:
        return 64

if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:
    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


def _flash_attn_forward(
    q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, return_softmax, bhsd = False, s_aux = None, blasst_threshold_scale_factor = 0.0
):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, skip_info = flash_attn_cuda.fwd(
        q,
        k,
        v,
        None,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        return_softmax,
        None, bhsd, s_aux,  # Attention Sinks
        blasst_threshold_scale_factor
    )
    if blasst_threshold_scale_factor > 0.0:
        return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, skip_info
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q:float,
    max_seqlen_k:float,
    dropout_p:float,
    softmax_scale:float,
    causal:bool,
    window_size_left:int,
    window_size_right:int,
    softcap:float,
    alibi_slopes:Optional[torch.Tensor],
    return_softmax:bool,
    q_descale:Optional[torch.Tensor] = None,
    k_descale:Optional[torch.Tensor] = None,
    v_descale:Optional[torch.Tensor] = None,
    block_table:Optional[torch.Tensor]=None,
    leftpad_k:Optional[torch.Tensor]=None,
    s_aux:Optional[torch.Tensor]=None,
)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.varlen_fwd(
        q,
        k,
        v,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        q_descale,
        k_descale,
        v_descale,
        None,
        s_aux,
    )
    # if out.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    return out, softmax_lse, S_dmask, rng_state

@_torch_register_fake_wrapper("flash_attn2_c_op::varlen_fwd")
def varlen_fwd_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        seqused_k: Optional[torch.Tensor],
        leftpad_k_: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        max_seqlen_q: int,
        max_seqlen_k: int,
        p_dropout: float,
        softmax_scale: float,
        zero_tensors: bool,
        is_causal: bool,
        window_size_left: int,
        window_size_right: int,
        softcap: float,
        return_softmax: bool,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # q shape:total_q x num_heads x head_size
    num_heads = q.size(1)
    total_q = q.size(0)
    batch_size = cu_seqlens_q.size(0) - 1
    out = torch.empty_like(q)
    softmax_lse = torch.empty([num_heads, total_q], dtype=q.dtype, device=q.device)
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = round_multiple(max_seqlen_k, 128)
    if return_softmax:
        p = torch.empty([batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded], dtype=q.dtype, device=q.device)
    else:
        p = torch.empty([0], dtype=q.dtype, device=q.device)
    rng_state = torch.empty([2], dtype=torch.int64, device=q.device)
    return out, softmax_lse, p, rng_state


_wrapped_flash_attn_varlen_forward = torch.ops.flash_attn2_c_op.varlen_fwd

@_torch_register_fake_wrapper("flash_attn2_c_op::fwd_kvcache")
def fwd_kvcache_fake(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    k_: Optional[torch.Tensor],
    v_: Optional[torch.Tensor],
    seqlens_k_: Optional[torch.Tensor],
    rotary_cos_: Optional[torch.Tensor],
    rotary_sin_: Optional[torch.Tensor],
    cache_batch_idx_: Optional[torch.Tensor],
    leftpad_k_: Optional[torch.Tensor],
    block_table_: Optional[torch.Tensor],
    alibi_slopes_: Optional[torch.Tensor],
    out_: Optional[torch.Tensor],
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    is_rotary_interleaved: bool,
    num_splits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    batch_size, seqlen_q, num_heads, head_size = q.shape
    
    # 创建输出张量的形状
    # out: batch_size x seqlen_q x num_heads x head_size
    out_shape = q.shape
    out = torch.empty(out_shape, dtype=q.dtype, device=q.device)
    
    # softmax_lse: batch_size x num_heads x seqlen_q
    softmax_lse_shape = (batch_size, num_heads, seqlen_q)
    softmax_lse = torch.empty(softmax_lse_shape, dtype=torch.float32, device=q.device)
    
    return out, softmax_lse


def _flash_attn_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    deterministic,
    rng_state=None,
    bhsd = False
):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = flash_attn_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        deterministic,
        None,
        rng_state, bhsd
    )
    return dq, dk, dv, softmax_d


def _flash_attn_varlen_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    deterministic,
    rng_state=None,
):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = flash_attn_cuda.varlen_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        deterministic,
        None,
        rng_state,
    )
    # if dk.isnan().any() or dk.isnan().any() or dv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return dq, dk, dv, softmax_d


####################################################################################################
# Attnmask functions - FlashAttention with explicit attention mask
####################################################################################################

def _flash_attn_attnmask_forward(
    q,
    k,
    v,
    attn_mask,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    return_softmax,
    bhsd=False,
    s_aux=None,
):
    """
    Forward pass for FlashAttention with explicit attention mask.

    Args:
        q: Query tensor, shape [batch, seqlen_q, nheads, headdim] or [batch, nheads, seqlen_q, headdim] if bhsd
        k: Key tensor, shape [batch, seqlen_k, nheads_k, headdim]
        v: Value tensor, shape [batch, seqlen_k, nheads_k, headdim]
        attn_mask: Attention mask, shape [batch, nheads, seqlen_q, seqlen_k], dtype bool or uint8.
                   True/1 = attend, False/0 = mask out.
        dropout_p: Dropout probability
        softmax_scale: Softmax scale factor
        causal: Whether to apply causal masking
        window_size: (left, right) window sizes
        softcap: Softmax cap value
        alibi_slopes: ALiBi slopes
        return_softmax: Whether to return softmax probabilities
        bhsd: Whether input is in [batch, heads, seq, dim] format
        s_aux: (nheads,), optional. Precomputed LogSumExp values for Attention Sinks.

    Returns:
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state
    """
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if attn_mask.stride(-1) != 1:
        attn_mask = attn_mask.contiguous()

    # masked_value must be -inf for backward correctness
    masked_value = float('-inf')

    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.fwd_attnmask(
        q,
        k,
        v,
        attn_mask,
        None,  # out_
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        masked_value,
        return_softmax,
        None,  # generator
        bhsd,
        s_aux,  # Attention Sinks
    )
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state


def _flash_attn_attnmask_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    attn_mask,
    dq,
    dk,
    dv,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    deterministic,
    rng_state=None,
    bhsd=False,
):
    """
    Backward pass for FlashAttention with explicit attention mask.
    """
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    if attn_mask.stride(-1) != 1:
        attn_mask = attn_mask.contiguous()

    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = flash_attn_cuda.bwd_attnmask(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        attn_mask,
        dq,
        dk,
        dv,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        deterministic,
        None,  # generator
        rng_state,
        bhsd,
    )
    return dq, dk, dv, softmax_d


def _flash_attn_varlen_attnmask_forward(
    q,
    k,
    v,
    attn_mask,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    return_softmax,
    zero_tensors=False,
    seqused_k=None,
    s_aux=None,
):
    """
    Variable-length forward pass for FlashAttention with explicit attention mask.

    Args:
        q: Query tensor, shape [total_q, nheads, headdim]
        k: Key tensor, shape [total_k, nheads_k, headdim]
        v: Value tensor, shape [total_k, nheads_k, headdim]
        attn_mask: Attention mask, shape [batch, nheads, max_seqlen_q, max_seqlen_k]
        cu_seqlens_q: Cumulative sequence lengths for queries, shape [batch + 1]
        cu_seqlens_k: Cumulative sequence lengths for keys, shape [batch + 1]
        max_seqlen_q: Maximum query sequence length
        max_seqlen_k: Maximum key sequence length
        s_aux: (nheads,), optional. Precomputed LogSumExp values for Attention Sinks.
        ...
    """
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if attn_mask.stride(-1) != 1:
        attn_mask = attn_mask.contiguous()

    masked_value = float('-inf')

    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.varlen_fwd_attnmask(
        q,
        k,
        v,
        attn_mask,
        None,  # out_
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        masked_value,
        return_softmax,
        None,  # generator
        s_aux,  # Attention Sinks
    )
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state


def _flash_attn_varlen_attnmask_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    attn_mask,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    deterministic,
    rng_state=None,
    zero_tensors=False,
):
    """
    Variable-length backward pass for FlashAttention with explicit attention mask.
    """
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    if attn_mask.stride(-1) != 1:
        attn_mask = attn_mask.contiguous()

    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = flash_attn_cuda.varlen_bwd_attnmask(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        attn_mask,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        deterministic,
        None,  # generator
        rng_state,
    )
    return dq, dk, dv, softmax_d


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        bhsd = False
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            bhsd = bhsd
        )
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.bhsd = bhsd
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dqkv[:, :, 0],
            dqkv[:, :, 1],
            dqkv[:, :, 2],
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
            bhsd=ctx.bhsd
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        total_q = qkv.shape[0]
        batch = cu_seqlens.shape[0]-1
        if max_seqlen == total_q and batch>1:
            cu_seqlens=cu_seqlens[:2]
        elif max_seqlen * batch > total_q * 2:
            cu_seqlens=cu_seqlens[cu_seqlens<=total_q]
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        out, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_varlen_forward(
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            None,
            cu_seqlens,
            cu_seqlens,
            None,
            None,
            None,
            alibi_slopes,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            False,
            causal,
            window_size[0],
            window_size[1],
            softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
        )
        ctx.save_for_backward(qkv[:, 0], qkv[:, 1], qkv[:, 2], out, softmax_lse, cu_seqlens, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen = max_seqlen
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dqkv[:, 0],
            dqkv[:, 1],
            dqkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        bhsd=False
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q,
            kv[:, :, 0],
            kv[:, :, 1],
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            bhsd=bhsd
        )
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.bhsd = bhsd
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq = torch.empty_like(q)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dkv[:, :, 0],
            dkv[:, :, 1],
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
            bhsd=ctx.bhsd
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dkv = dkv[..., : dout.shape[-1]]
        return dq, dkv, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        total_q = q.shape[0]
        batch = cu_seqlens_q.shape[0]-1
        if max_seqlen_q == total_q and batch>1:
            cu_seqlens_q=cu_seqlens_q[:2]
            cu_seqlens_k=cu_seqlens_k[:2]
        elif max_seqlen_q * batch > total_q * 2:
            cu_seqlens_q=cu_seqlens_q[cu_seqlens_q<=total_q]
            cu_seqlens_k=cu_seqlens_q
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_varlen_forward(
            q,
            kv[:, 0],
            kv[:, 1],
            None,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            None,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            window_size[0],
            window_size[1],
            softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
        )
        ctx.save_for_backward(
            q, kv[:, 0], kv[:, 1], out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq = torch.empty_like(q)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dkv[:, 0],
            dkv[:, 1],
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dkv = dkv[..., : dout.shape[-1]]
        return dq, dkv, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        bhsd = False,
        s_aux = None  # Attention Sinks
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            bhsd = bhsd,
            s_aux = s_aux  # Attention Sinks
        )
        ctx.qk_headdim = q.shape[-1]
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.bhsd = bhsd
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
            bhsd=ctx.bhsd
        )
        # dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        # dk = dk[..., : dout.shape[-1]]
        # dv = dv[..., : dout.shape[-1]]
        dq = dq[..., : ctx.qk_headdim]  # We could have padded the head dimension
        dk = dk[..., : ctx.qk_headdim]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        q_descale,
        k_descale,
        v_descale,
        block_table,
        s_aux = None,
    ):
        total_q = q.shape[0]
        batch = cu_seqlens_q.shape[0]-1
        if max_seqlen_q == total_q and batch>1:
            cu_seqlens_q=cu_seqlens_q[:2]
            cu_seqlens_k=cu_seqlens_k[:2]
        elif max_seqlen_q * batch > total_q * 2:
            cu_seqlens_q=cu_seqlens_q[cu_seqlens_q<=total_q]
            cu_seqlens_k=cu_seqlens_q
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        # The torch.compile custom op path only exposes the reduced varlen_fwd schema
        # from flash_attn2_c_op::varlen_fwd. Keep using it for the supported subset,
        # but fall back to the original pybind path when extra features are requested.
        if any(x is not None for x in (q_descale, k_descale, v_descale, block_table, s_aux)):
            out, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                causal,
                window_size[0],
                window_size[1],
                softcap,
                alibi_slopes,
                return_softmax and dropout_p > 0,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                block_table=block_table,
                s_aux=s_aux,
            )
        else:
            out, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_varlen_forward(
                q,
                k,
                v,
                None,
                cu_seqlens_q,
                cu_seqlens_k,
                None,
                None,
                None,
                alibi_slopes,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                False,
                causal,
                window_size[0],
                window_size[1],
                softcap=softcap,
                return_softmax=return_softmax and dropout_p > 0,
            )
        ctx.qk_headdim = q.shape[-1]
        ctx.save_for_backward(
            q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : ctx.qk_headdim]  # We could have padded the head dimension
        dk = dk[..., : ctx.qk_headdim]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


####################################################################################################
# Attnmask autograd.Function classes
####################################################################################################

class FlashAttnAttnmaskFunc(torch.autograd.Function):
    """
    FlashAttention with explicit attention mask - fixed length version.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        bhsd,
        s_aux,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_attnmask_forward(
            q,
            k,
            v,
            attn_mask,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            bhsd=bhsd,
            s_aux=s_aux,
        )
        ctx.qk_headdim = q.shape[-1]
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, attn_mask, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.bhsd = bhsd
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, attn_mask, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_attnmask_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            attn_mask,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
            bhsd=ctx.bhsd,
        )
        dq = dq[..., : ctx.qk_headdim]
        dk = dk[..., : ctx.qk_headdim]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenAttnmaskFunc(torch.autograd.Function):
    """
    FlashAttention with explicit attention mask - variable length version.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        attn_mask,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        zero_tensors,
        seqused_k,
        s_aux,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_attnmask_forward(
            q,
            k,
            v,
            attn_mask,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            zero_tensors=zero_tensors,
            seqused_k=seqused_k,
            s_aux=s_aux,
        )
        ctx.qk_headdim = q.shape[-1]
        ctx.save_for_backward(
            q, k, v, out_padded, softmax_lse, attn_mask, cu_seqlens_q, cu_seqlens_k, rng_state
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.zero_tensors = zero_tensors
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, attn_mask, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_varlen_attnmask_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            attn_mask,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
            zero_tensors=ctx.zero_tensors,
        )
        dq = dq[..., : ctx.qk_headdim]
        dk = dk[..., : ctx.qk_headdim]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # <=0.0 means deactivate
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    bhsd = False
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_kvpacked_func and flash_attn_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to
            the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        bhsd
    )


def flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    bhsd = False
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        kv: (batch_size, seqlen, 2, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnKVPackedFunc.apply(
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        bhsd
    )


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    bhsd = False,
    s_aux=None  # Attention Sinks: [nheads]
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        s_aux: (nheads,), optional, dtype float16 or bfloat16 (must match Q/K/V dtype).
            Precomputed LogSumExp values for sink tokens in Streaming LLM.
            Used to implement Attention Sinks for maintaining attention to initial tokens.
            IMPORTANT: s_aux.dtype must equal q.dtype (enforced by runtime check).
            Storage uses FP16/BF16 for bandwidth efficiency (50% reduction vs FP32).
            Computation uses FP32 internally for numerical stability.
            If provided, enables Attention Sinks with <0.5% overhead. Maximum 64 heads supported.
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        bhsd,
        s_aux  # Attention Sinks
    )

####################################################################################################
# Attnmask high-level user interface functions
####################################################################################################

def flash_attn_with_mask_func(
    q,
    k,
    v,
    attn_mask,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    bhsd=False,
    s_aux=None,
):
    """FlashAttention with explicit attention mask.

    This function applies FlashAttention with a user-provided attention mask.
    The mask allows fine-grained control over which query-key pairs attend to each other.

    Constraints:
        - headdim must be 128
        - attn_mask's last dimension (seqlen_k) must be contiguous
        - q/k/v dtype must be fp16 or bf16
        - True in mask = attend, False = mask out. Non-bool is converted: integer 1/0 -> True/False; float 0/-inf -> True/False.

    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim) or (batch_size, nheads, seqlen_q, headdim) if bhsd
        k: (batch_size, seqlen_k, nheads_k, headdim)
        v: (batch_size, seqlen_k, nheads_k, headdim)
        attn_mask: (batch_size, nheads, seqlen_q, seqlen_k), dtype bool, integer, or float.
            If bool: True = attend, False = mask out.
            If integer (uint8/int8/...): 1 = attend, 0 = mask out.
            If float: 0 = attend, -inf = mask out (converted to bool internally).
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (combined with attn_mask).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Softmax cap value. 0.0 means deactivated.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. ALiBi slopes.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass.
        return_attn_probs: bool. Whether to return the attention probabilities.
        bhsd: bool. Whether input is in [batch, heads, seq, dim] format.
        s_aux: (nheads,), optional, dtype float16 or bfloat16 (must match Q/K/V dtype).
            Precomputed LogSumExp values for sink tokens in Streaming LLM.
            Used to implement Attention Sinks for maintaining attention to initial tokens.

    Returns:
        out: (batch_size, seqlen_q, nheads, headdim) or (batch_size, nheads, seqlen_q, headdim) if bhsd.
        softmax_lse [optional]: (batch_size, nheads, seqlen_q). The logsumexp values.
        S_dmask [optional]: The softmax output (for testing only).

    Example:
        >>> q = torch.randn(2, 128, 8, 128, dtype=torch.float16, device='cuda')
        >>> k = torch.randn(2, 256, 8, 128, dtype=torch.float16, device='cuda')
        >>> v = torch.randn(2, 256, 8, 128, dtype=torch.float16, device='cuda')
        >>> # Create mask: first 128 K positions are visible
        >>> attn_mask = torch.zeros(2, 8, 128, 256, dtype=torch.bool, device='cuda')
        >>> attn_mask[:, :, :, :128] = True
        >>> out = flash_attn_with_mask_func(q, k, v, attn_mask)
    """
    # Kernel expects bool mask. Convert non-bool:
    # - Integer (uint8/int8/...): 1 -> True (attend), 0 -> False (mask out).
    # - Float: 0 -> True (attend), -inf -> False (mask out).
    if attn_mask.dtype != torch.bool:
        if attn_mask.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            attn_mask = (attn_mask != 0)
        else:
            attn_mask = (attn_mask == 0)
    return FlashAttnAttnmaskFunc.apply(
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        bhsd,
        s_aux,
    )


def flash_attn_varlen_with_mask_func(
    q,
    k,
    v,
    attn_mask,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    zero_tensors=False,
    seqused_k=None,
    s_aux=None,
):
    """Variable-length FlashAttention with explicit attention mask.

    This function applies FlashAttention with variable-length sequences and a user-provided
    attention mask.

    Constraints:
        - headdim must be 128
        - attn_mask's last dimension must be contiguous
        - q/k/v dtype must be fp16 or bf16

    Arguments:
        q: (total_q, nheads, headdim), where total_q = sum of seqlen_q across batch.
        k: (total_k, nheads_k, headdim), where total_k = sum of seqlen_k across batch.
        v: (total_k, nheads_k, headdim).
        attn_mask: (batch_size, nheads, max_seqlen_q, max_seqlen_k), dtype bool, integer, or float.
            If bool: True = attend, False = mask out. If integer: 1 = attend, 0 = mask out. If float: 0 = attend, -inf = mask out.
        cu_seqlens_q: (batch_size + 1,), dtype int32. Cumulative sequence lengths for queries.
        cu_seqlens_k: (batch_size + 1,), dtype int32. Cumulative sequence lengths for keys.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
        causal: bool. Whether to apply causal attention mask.
        window_size: (left, right). Sliding window size.
        softcap: float. Softmax cap value.
        alibi_slopes: ALiBi slopes.
        deterministic: bool. Whether to use deterministic backward.
        return_attn_probs: bool. Whether to return attention probabilities.
        zero_tensors: bool. Whether to zero output tensors before computation.
        seqused_k: (batch_size,), optional. Actual K length used per sample.
        s_aux: (nheads,), optional, dtype float16 or bfloat16 (must match Q/K/V dtype).
            Precomputed LogSumExp values for sink tokens in Streaming LLM.
            Used to implement Attention Sinks for maintaining attention to initial tokens.

    Returns:
        out: (total_q, nheads, headdim).
    """
    # Kernel expects bool mask. Convert non-bool:
    # - Integer (uint8/int8/...): 1 -> True (attend), 0 -> False (mask out).
    # - Float: 0 -> True (attend), -inf -> False (mask out).
    if attn_mask.dtype != torch.bool:
        if attn_mask.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            attn_mask = (attn_mask != 0)
        else:
            attn_mask = (attn_mask == 0)
    return FlashAttnVarlenAttnmaskFunc.apply(
        q,
        k,
        v,
        attn_mask,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        zero_tensors,
        seqused_k,
        s_aux,
    )


def flash_attn_func_blasst(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    bhsd = False,
    s_aux=None,  # Attention Sinks: [nheads]
    blasst_threshold_scale_factor = 1.0e-9
):
    assert blasst_threshold_scale_factor > 0, "blasst_threshold_scale_factor must be great than 0.0f"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    return_softmax=return_attn_probs and dropout_p > 0
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state, sikp_info = _flash_attn_forward(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        return_softmax=return_softmax,
        bhsd = bhsd,
        s_aux = s_aux,  # Attention Sinks
        blasst_threshold_scale_factor=blasst_threshold_scale_factor
    )
    return (out, sikp_info) if not return_attn_probs else (out, softmax_lse, S_dmask, sikp_info)

def flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_varlen_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_varlen_kvpacked_func and flash_attn_varlen_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnVarlenQKVPackedFunc.apply(
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )


def flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        kv: (total_k, 2, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnVarlenKVPackedFunc.apply(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )

def vllm_flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None, # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: Optional[List[int]] = None,
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    is_prefix_cache = True,
    # FA3 Only
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    kv_cache_dtype = "auto",
    # Version selector
    fa_version: int = DEFAULT_FA_VERSION,
    s_aux=None,
    is_bhsd = False
):
    """
    仅用于vllm prefix cache
    dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert cu_seqlens_k is not None or seqused_k is not None, \
        "cu_seqlens_k or seqused_k must be provided"
    assert cu_seqlens_k is None or seqused_k is None, \
        "cu_seqlens_k and seqused_k cannot be provided at the same time"
    assert block_table is None or seqused_k is not None, \
        "seqused_k must be provided if block_table is provided"
    assert fa_version == 2, \
        "only support fa2"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    # custom op does not support non-tuple input
    real_window_size: Tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])

    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)
    if is_prefix_cache:
        # kcache (num_blocks, number_heads, block_size, headdim)
        # vcache (num_blocks, number_heads, headdim, block_size)
        assert q.shape[-1] %64 == 0 and q.shape[-1]<=256, f"UnSupport q.shape[-1]: {q.shape[-1]}"
        bs = cu_seqlens_q.shape[0] - 1
        total_q = q.shape[0]
        # max_seqlen_q*bs==total_q and max_seqlen_q<=4 means mtp
        # if mtp, k head must be 1.
        # todo : support k head >1
        is_mtp = (max_seqlen_q*bs==total_q and max_seqlen_q>1 and max_seqlen_q<5)
        if  (max_seqlen_q==1 or is_mtp ) and real_window_size[0]==-1:
            if out==None:
                if q.dtype == torch.float8_e4m3fn or q.dtype == torch.float8_e5m2:
                    out = torch.empty(q.size(),device = q.device,dtype=torch.bfloat16)
                else :
                    out = torch.empty_like(q)
            flash_attn_cuda.paged_attention(out,q.reshape(bs,max_seqlen_q,q.shape[1],q.shape[-1]),k,v,softmax_scale,block_table,
                seqused_k,alibi_slopes,kv_cache_dtype,q_descale,k_descale,v_descale,max_seqlen_k,s_aux,is_bhsd)
            return out
        is_938 = ("gfx938" in  torch.cuda.get_device_properties("cuda").gcnArchName or "gfx92a" in  torch.cuda.get_device_properties("cuda").gcnArchName)
        if (not is_938) and k.dtype == torch.float8_e5m2 and v.dtype == torch.float8_e5m2:
            assert q.dtype != torch.float8_e5m2 , "UnSupport q.dtype:fp8"
            q_descale = None
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.vllm_mha_varlen_fwd_kv_fp8(
                q,
                k,
                v,
                out,
                cu_seqlens_q,
                dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
                seqused_k,
                None,
                block_table,
                alibi_slopes,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                False,
                causal,
                real_window_size[0],
                real_window_size[1],
                softcap,
                return_softmax_lse and dropout_p > 0,
                q_descale,
                k_descale,
                v_descale,
                None,
                s_aux,
            )
        else:
            if(k.dtype == torch.float8_e4m3fn or k.dtype == torch.float8_e5m2) and q.dtype != k.dtype:
                if q_descale is not None:
                    q=q/q_descale
                q = q.to(k.dtype)
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.vllm_mha_varlen_fwd(
                q,
                k,
                v,
                out,
                cu_seqlens_q,
                # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp
                # still wants it so we pass all zeros
                dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
                seqused_k,
                #leftpad_k,
                None,
                block_table,
                alibi_slopes,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                False,
                causal,
                real_window_size[0],
                real_window_size[1],
                softcap,
                return_softmax_lse and dropout_p > 0,
                q_descale,
                k_descale,
                v_descale,
                None,
                s_aux,
            )
    else:
        # kcache (num_blocks, block_size, number_heads, headdim)
        # vcache (num_blocks, block_size, number_heads, headdim)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.varlen_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp
            # still wants it so we pass all zeros
            dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
            seqused_k,
            #leftpad_k,
            None,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            real_window_size[0],
            real_window_size[1],
            softcap,
            return_softmax_lse and dropout_p > 0,
            q_descale,
            k_descale,
            v_descale,
            None,
            s_aux,
        )
    # if out.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    # return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state
    return out if not return_softmax_lse else (out, softmax_lse, S_dmask)

def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    q_descale=None, k_descale=None, v_descale=None,
    block_table=None,
    s_aux=None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        q_descale, k_descale, v_descale,
        block_table,
        s_aux,
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    s_aux=None,  # Attention Sinks: [nheads]
    return_softmax_lse=False,
    *,

    out = None
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    s_aux = maybe_contiguous(s_aux)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    out, softmax_lse = flash_attn_cuda.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        out,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits,
        s_aux,  # Attention Sinks
    )
    return (out, softmax_lse) if return_softmax_lse else out


def vllm_flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    s_aux=None,  # Attention Sinks: [nheads]
    return_softmax_lse=False,
    *,
    q_scale=None,
    k_scale=None,
    v_scale=None,
    kv_cache_dtype = "auto",
    out = None,
    max_seqlen_k = 0,
    is_bhsd = False
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    assert q.shape[-1] %64 == 0 and q.shape[-1]<=256, f"UnSupport q.shape[-1]: {q.shape[-1]}"
    if q.shape[1]<5 and k==None and v==None and rotary_cos==None and rotary_sin==None:
        if out==None:
            if q.dtype == torch.float8_e4m3fn or q.dtype == torch.float8_e5m2:
                out = torch.empty(q.size(),device = q.device,dtype=torch.bfloat16)
            else :
                out = torch.empty_like(q)
        flash_attn_cuda.paged_attention(out,q,k_cache,v_cache,softmax_scale,block_table,
            cache_seqlens,alibi_slopes,kv_cache_dtype,q_scale,k_scale,v_scale,max_seqlen_k,s_aux,is_bhsd)
        return out
    else:
        assert k_cache.dtype == torch.half or k_cache.dtype == torch.bfloat16, "only surppurt bf16 and fp16"
        out, softmax_lse = flash_attn_cuda.vllm_fwd_kvcache(
                q,
                k_cache,
                v_cache,
                k,
                v,
                cache_seqlens,
                rotary_cos,
                rotary_sin,
                cache_batch_idx,
                cache_leftpad,
                block_table,
                alibi_slopes,
                out,
                softmax_scale,
                causal,
                window_size[0],
                window_size[1],
                softcap,
                rotary_interleaved,
                num_splits,
                s_aux)
    return (out, softmax_lse) if return_softmax_lse else out


def sparse_attn_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    *,
    return_softmax_lse=False,
    out=None,
    pv_threshold=50.0,         # Dynamic PV skip threshold
    enable_dynamic_skip=False, # Enable dynamic PV skip optimization (disabled by default for dense scenarios)
    is_sla=False
):
    """Compute attention with vertical and slash sparsity patterns.
    Most Arguments are the same with the flash_attn_func interface, except for 4 extra args:
    block_count and block_offset for slash sparsity patterns, and
    column_count and column_index for vertical sparsity patterns.
    For more details please refer to Appendix C.4.2 of paper https://arxiv.org/abs/2407.02490.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S)
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse = flash_attn_cuda.fwd_sparse(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        out,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        softcap,
        return_attn_probs and dropout_p > 0,
        None,
        is_sla,
        pv_threshold,
        enable_dynamic_skip,
    )
    return (out, softmax_lse) if return_softmax_lse else out


def sparse_attn_varlen_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    *,
    return_softmax_lse=False,
    out=None,
    pv_threshold=50.0,         # Dynamic PV skip threshold
    enable_dynamic_skip=False, # Enable dynamic PV skip optimization (disabled by default for dense scenarios)
):
    """Compute attention with vertical and slash sparsity patterns.
    Most Arguments are the same with the flash_attn_varlen_func interface, except for 4 extra args:
    block_count and block_offset for slash sparsity patterns, and
    column_count and column_index for vertical sparsity patterns.
    For more details please refer to Appendix C.4.2 of paper https://arxiv.org/abs/2407.02490.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S)
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V)
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse = flash_attn_cuda.varlen_fwd_sparse(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        softcap,
        return_attn_probs and dropout_p > 0,
        None,
        pv_threshold,
        enable_dynamic_skip,
    )
    return (out, softmax_lse) if return_softmax_lse else out

def varlen_fwd_unified(
    q,                          # [num_tokens, num_query_heads, head_size]
    k,                          # bshd: [num_total_blocks, block_size, num_kv_heads, head_size]
                                # bhsd: [num_total_blocks, num_kv_heads, block_size, head_size]
    v,                          # bshd: [num_total_blocks, block_size, num_kv_heads, head_size]
                                # bhsd: [num_total_blocks, num_kv_heads, block_size, head_size]
    cu_seqlens_q,               # [num_seqs + 1] int32
    seqused_k,                  # [num_seqs] int32
    block_table,                # [batch_size, max_num_blocks_per_seq] int32
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    window_size=(-1, -1),
    alibi_slopes=None,
    use_alibi_sqrt=False,
    qq_bias=None,
    s_aux=None,
    mm_prefix_range=None,
    layout="bshd",
    *,
    out=None,
    return_softmax_lse=False,
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    if layout not in ("bshd", "bhsd"):
        raise NotImplementedError("varlen_fwd_unified only supports bshd/bhsd layout")
    layout_id = 1 if layout == "bshd" else 0

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    window_size_left, window_size_right = window_size

    if k.dtype.is_floating_point:
        k_dtype_bits = torch.finfo(k.dtype).bits
    else:
        k_dtype_bits = torch.iinfo(k.dtype).bits

    is_mtp = (max_seqlen_q * seqused_k.size(0) == q.shape[0] and 1 < max_seqlen_q < 16)

    if max_seqlen_q >= 16:
        fp8_dtypes = [torch.float8_e4m3fn]
        if hasattr(torch, "float8_e4m3fnuz"):
            fp8_dtypes.append(torch.float8_e4m3fnuz)
        if (
            layout == "bshd"
            and k_dtype_bits == 16
            and q.shape[-1] == 256
            and v.shape[-1] == 256
            and s_aux is None
        ):
            out, softmax_lse = flash_attn_cuda.varlen_fwd_unified(
                q,
                k,
                v,
                out,
                cu_seqlens_q,
                max_seqlen_q,
                seqused_k,
                max_seqlen_k,
                block_table,
                softmax_scale,
                softcap,
                None,   # q_descale
                None,   # k_descale
                None,   # v_descale
                None,   # output_scale
                causal,
                window_size_left,
                window_size_right,
                alibi_slopes,
                use_alibi_sqrt,
                qq_bias,
                s_aux,
                mm_prefix_range,
            )
            return (out, softmax_lse) if return_softmax_lse else out
        bshd_prefill = _require_hg_varlen_symbol("hg_prefix_prefill_varlen_fwd")
        fa_output, *rest_extend = bshd_prefill(
            q,
            k,
            v,
            out, # out_
            cu_seqlens_q,
            None, # cu_seqlens_k
            seqused_k,
            alibi_slopes,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            0.0, # dropout
            softmax_scale,
            False, # zero_tensors
            causal,
            window_size[0],
            window_size[1],
            softcap,
            return_softmax_lse,
            layout_id,
            None if (k_dtype_bits == 16) else q_descale,
            None if (k_dtype_bits == 16) else k_descale,
            None if (k_dtype_bits == 16) else v_descale,
            s_aux,
            True,
        )
        return (fa_output, rest_extend[0]) if return_softmax_lse else fa_output

    bs = seqused_k.size(0)
    if max_seqlen_q == 1 or max_seqlen_q < 16:
        if layout == "bhsd":
            if (max_seqlen_q == 1 or is_mtp) and window_size[0] == -1 and k.shape[2] == 64:
                if out==None:
                    if q.dtype == torch.float8_e4m3fn or q.dtype == torch.float8_e5m2:
                        out = torch.empty(q.size(),device = q.device,dtype=torch.bfloat16)
                    else :
                        out = torch.empty_like(q)
                flash_attn_cuda.paged_attention(out,q.reshape(bs,max_seqlen_q,q.shape[1],q.shape[-1]),k,v,
                                                softmax_scale,block_table,seqused_k,alibi_slopes,"",q_descale,k_descale,v_descale,max_seqlen_k,s_aux,True)
                return (out,None) if return_softmax_lse else out
            raise NotImplementedError(
                "bhsd PA decode currently requires page_block_size == 64 and window_size_left == -1"
            )
        assert layout == "bshd", "Prefix decode fallback only supports bshd layout"
        bshd_pa_decode = _require_hg_varlen_symbol("hg_prefix_decode_varlen_fwd")
        result = bshd_pa_decode(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            None,
            seqused_k,
            alibi_slopes,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            0.0,
            softmax_scale,
            False,
            causal,
            window_size[0],
            window_size[1],
            softcap,
            return_softmax_lse,
            layout_id,
            None if (k_dtype_bits == 16) else q_descale,
            None if (k_dtype_bits == 16) else k_descale,
            None if (k_dtype_bits == 16) else v_descale,
            s_aux,
            True,
        )
        fa_output = result[0]
        return (fa_output, result[1]) if return_softmax_lse else fa_output

    out, softmax_lse = flash_attn_cuda.varlen_fwd_unified(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        block_table,
        softmax_scale,
        softcap,
        None,   # q_descale
        None,   # k_descale
        None,   # v_descale
        None,   # output_scale
        causal,
        window_size_left,
        window_size_right,
        alibi_slopes,
        use_alibi_sqrt,
        qq_bias,
        s_aux,
        mm_prefix_range,
    )

    return (out, softmax_lse) if return_softmax_lse else out


def _auto_sparse_attn_core(
    *,
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    # Auto-sparsity parameters
    smooth_k=True,
    simthreshd1=0.6,
    cdfthreshd=None,
    topk=None,
    attention_sink=False,
    block_m=64,
    block_k=64,
    # PV skip optimization parameters
    pv_threshold=50.0,
    enable_dynamic_skip=False,
    # Return options
    return_softmax_lse=False,
    return_sparsity=False,
    out=None,
):
    """
    Core implementation for auto-generating sparsity (CDF or TopK) and calling sparse_attn_func.
    Exactly one of cdfthreshd / topk must be provided.
    """
    assert (cdfthreshd is None) ^ (topk is None), "Provide exactly one of cdfthreshd or topk"
    # FA2 sparse kernel uses a fixed block size (BLOCK_M = 64). Keep q/k block sizes aligned.
    assert block_k == block_m, "block_k must equal block_m to match FA2 sparse kernel"

    # Input validation and conversion
    assert q.dim() == 4, f"q must be 4D (B, L, H, D), got shape {q.shape}"
    assert q.size(1) >= 128, "seqlen should be not less than 128."

    torch.cuda.set_device(q.device)

    # Convert to BHLD layout for processing
    q_bhld = q.transpose(1, 2).contiguous()  # (B, H, L, D)
    k_bhld = k.transpose(1, 2).contiguous()
    v_bhld = v.transpose(1, 2).contiguous()

    # Ensure correct dtype
    dtype = q_bhld.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q_bhld = q_bhld.to(torch.float16)
        k_bhld = k_bhld.to(torch.float16)
        v_bhld = v_bhld.to(torch.float16)
    else:
        q_bhld = q_bhld.to(torch.bfloat16)
        k_bhld = k_bhld.to(torch.bfloat16)
        v_bhld = v_bhld.to(torch.bfloat16)

    # Apply K smoothing (subtract mean directly)
    if smooth_k:
        k_bhld = k_bhld - k_bhld.mean(dim=-2, keepdim=True)

    B, H, seqlen_q, headdim = q_bhld.shape
    seqlen_k = k_bhld.size(-2)

    # Generate sparsity pattern with block_offset format
    block_offset, block_count = get_block_map_meansim(
        q_bhld, k_bhld,
        is_causal=causal,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
        topk=topk,
        return_block_offset=True,
        attention_sink=attention_sink,
        BLKQ=block_m,
        BLKK=block_k,
    )

    if softmax_scale is None:
        softmax_scale = headdim ** (-0.5)

    # Disable Vertical sparsity (set column_count to 0)
    num_blocks_q = (seqlen_q + block_m - 1) // block_m
    num_blocks_k = (seqlen_k + block_k - 1) // block_k

    column_count = torch.zeros(
        (B, H, num_blocks_q), dtype=torch.int32, device=q.device
    )
    column_index = torch.zeros(
        (B, H, num_blocks_q, 1), dtype=torch.int32, device=q.device
    )

    # Convert block indices to row offsets (kernel expects row offsets, not block indices)
    block_offset = block_offset * block_k

    # Call sparse attention kernel (expects BLHD layout)
    result = sparse_attn_func(
        q, k, v,  # Use original BLHD layout
        block_count=block_count,
        block_offset=block_offset,
        column_count=column_count,
        column_index=column_index,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        return_softmax_lse=return_softmax_lse,
        out=out,
        pv_threshold=pv_threshold,
        enable_dynamic_skip=enable_dynamic_skip,
    )

    # Calculate sparsity if requested
    if return_sparsity:
        if not causal:
            qk_sparsity = 1 - (block_count.float().sum()) / (
                num_blocks_k * num_blocks_q * B * H
            )
        else:
            # For causal mask, expected number of blocks is (num_blocks_k + 1) / 2 per query block
            qk_sparsity = 1 - (block_count.float().sum()) / (
                ((num_blocks_k + 1) // 2) * num_blocks_q * B * H
            )

        if return_softmax_lse:
            out, softmax_lse = result
            return out, softmax_lse, qk_sparsity.item()
        else:
            return result, qk_sparsity.item()

    return result


def spas_fa2_attn_meansim_cuda(
    q,
    k,
    v,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    smooth_k=True,
    simthreshd1=0.6,
    cdfthreshd=0.98,
    attention_sink=False,
    block_m=64,
    block_k=64,
    pv_threshold=50.0,
    enable_dynamic_skip=False,
    return_softmax_lse=False,
    return_sparsity=False,
    out=None,
):
    """
    Auto-sparse attention using CDF thresholding (cdfthreshd). Mirrors SpargeAttn's mean-sim CDF path.
    """
    return _auto_sparse_attn_core(
        q=q,
        k=k,
        v=v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
        topk=None,
        attention_sink=attention_sink,
        block_m=block_m,
        block_k=block_k,
        pv_threshold=pv_threshold,
        enable_dynamic_skip=enable_dynamic_skip,
        return_softmax_lse=return_softmax_lse,
        return_sparsity=return_sparsity,
        out=out,
    )


def spas_fa2_attn_meansim_topk_cuda(
    q,
    k,
    v,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    smooth_k=True,
    simthreshd1=-0.1,
    topk=0.5,
    attention_sink=False,
    block_m=64,
    block_k=64,
    pv_threshold=50.0,
    enable_dynamic_skip=False,
    return_softmax_lse=False,
    return_sparsity=False,
    out=None,
):
    """
    Auto-sparse attention using Top-K ratio (topk). Mirrors SpargeAttn's mean-sim topk path.
    """
    return _auto_sparse_attn_core(
        q=q,
        k=k,
        v=v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        cdfthreshd=None,
        topk=topk,
        attention_sink=attention_sink,
        block_m=block_m,
        block_k=block_k,
        pv_threshold=pv_threshold,
        enable_dynamic_skip=enable_dynamic_skip,
        return_softmax_lse=return_softmax_lse,
        return_sparsity=return_sparsity,
        out=out,
    )


def _auto_sparse_attn_varlen_core(
    *,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    # Auto-sparsity parameters
    smooth_k=True,
    simthreshd1=0.6,
    cdfthreshd=None,
    topk=None,
    attention_sink=False,
    block_m=64,
    block_k=64,
    # PV skip optimization parameters
    pv_threshold=50.0,
    enable_dynamic_skip=False,
    # Return options
    return_softmax_lse=False,
    return_sparsity=False,
    out=None,
):
    """
    Varlen auto-sparse: generate block_count/block_offset/column_count/column_index per sequence,
    then call sparse_attn_varlen_func.
    """
    assert (cdfthreshd is None) ^ (topk is None), "Provide exactly one of cdfthreshd or topk"
    assert block_k == block_m == 64, "FA2 sparse kernel uses fixed BLOCK_M = BLOCK_K = 64"

    # Input validation
    assert q.dim() == 3, f"q must be 3D (total_q, H, D), got {q.shape}"
    assert k.dim() == 3 and v.dim() == 3
    B = cu_seqlens_q.numel() - 1
    assert B > 0
    assert cu_seqlens_k.numel() - 1 == B

    torch.cuda.set_device(q.device)

    H = q.size(1)
    headdim = q.size(2)
    if softmax_scale is None:
        softmax_scale = headdim ** (-0.5)

    max_q_blocks = (max_seqlen_q + block_m - 1) // block_m

    block_count = torch.zeros((B, H, max_q_blocks), dtype=torch.int32, device=q.device)
    # Disable Vertical sparsity (set column_count to 0)
    column_count = torch.zeros((B, H, max_q_blocks), dtype=torch.int32, device=q.device)
    column_index = torch.zeros((B, H, max_q_blocks, 1), dtype=torch.int32, device=q.device)

    # Pre-compute max_nnz_s (max num_k_blocks across all samples)
    max_nnz_s = 0
    for b in range(B):
        len_k = cu_seqlens_k[b + 1].item() - cu_seqlens_k[b].item()
        if len_k > 0:
            num_k_blocks = (len_k + block_k - 1) // block_k
            max_nnz_s = max(max_nnz_s, num_k_blocks)

    # Allocate block_offset with max_nnz_s (fill with large value for invalid positions)
    block_offset = torch.full((B, H, max_q_blocks, max_nnz_s), 10000000, dtype=torch.int32, device=q.device) if max_nnz_s > 0 else None

    # Single pass: compute block maps and fill data
    for b in range(B):
        q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
        k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()
        len_q = q_end - q_start
        len_k = k_end - k_start
        if len_q == 0 or len_k == 0:
            continue

        num_q_blocks = (len_q + block_m - 1) // block_m
        num_k_blocks = (len_k + block_k - 1) // block_k

        qb = q[q_start:q_end].transpose(0, 1).contiguous().unsqueeze(0)  # (1, H, Lq, D)
        kb = k[k_start:k_end].transpose(0, 1).contiguous().unsqueeze(0)  # (1, H, Lk, D)

        # Apply K smoothing (subtract mean directly)
        if smooth_k:
            kb = kb - kb.mean(dim=-2, keepdim=True)

        block_offset_b, block_count_b = get_block_map_meansim(
            qb, kb,
            is_causal=causal,
            simthreshd1=simthreshd1,
            cdfthreshd=cdfthreshd,
            topk=topk,
            return_block_offset=True,
            attention_sink=attention_sink,
            BLKQ=block_m,
            BLKK=block_k,
        )

        # Fill data directly
        block_count[b, :, :num_q_blocks] = block_count_b[0]
        nnz_s_b = block_offset_b.size(-1)
        block_offset[b, :, :num_q_blocks, :nnz_s_b] = block_offset_b[0]

    # Note: varlen sparse kernel may have different block_offset semantics
    # TODO: Investigate varlen sparse kernel implementation
    # For now, convert block indices to row offsets (same as non-varlen version)
    block_offset = block_offset * block_k

    # Call sparse varlen kernel
    result = sparse_attn_varlen_func(
        q,
        k,
        v,
        block_count=block_count,
        block_offset=block_offset if block_offset is not None else torch.full((B, H, max_q_blocks, 1), 10000000, dtype=torch.int32, device=q.device),
        column_count=column_count,
        column_index=column_index,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        return_softmax_lse=return_softmax_lse,
        out=out,
        pv_threshold=pv_threshold,
        enable_dynamic_skip=enable_dynamic_skip,
    )

    if return_sparsity:
        # Use actual per-batch num_q_blocks and num_k_blocks
        sparsities = []
        for b in range(B):
            len_q = cu_seqlens_q[b + 1] - cu_seqlens_q[b]
            len_k = cu_seqlens_k[b + 1] - cu_seqlens_k[b]
            if len_q == 0 or len_k == 0:
                continue
            num_q_blocks = (len_q + block_m - 1) // block_m
            num_k_blocks = (len_k + block_k - 1) // block_k
            if not causal:
                denom = num_k_blocks * num_q_blocks * H
            else:
                denom = ((num_k_blocks + 1) // 2) * num_q_blocks * H
            sparsities.append(1 - block_count[b, :, :num_q_blocks].float().sum() / denom)
        sparsity = torch.stack(sparsities).mean().item() if len(sparsities) > 0 else 0.0
        if return_softmax_lse:
            out_tensor, softmax_lse = result
            return out_tensor, softmax_lse, sparsity
        else:
            return result, sparsity

    return result


def spas_fa2_attn_meansim_varlen_cuda(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    smooth_k=True,
    simthreshd1=0.6,
    cdfthreshd=0.98,
    attention_sink=False,
    block_m=64,
    block_k=64,
    pv_threshold=50.0,
    enable_dynamic_skip=False,
    return_softmax_lse=False,
    return_sparsity=False,
    out=None,
):
    """Varlen CDF 路径自动稀疏"""
    return _auto_sparse_attn_varlen_core(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
        topk=None,
        attention_sink=attention_sink,
        block_m=block_m,
        block_k=block_k,
        pv_threshold=pv_threshold,
        enable_dynamic_skip=enable_dynamic_skip,
        return_softmax_lse=return_softmax_lse,
        return_sparsity=return_sparsity,
        out=out,
    )


def spas_fa2_attn_meansim_topk_varlen_cuda(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    smooth_k=True,
    simthreshd1=-0.1,
    topk=0.5,
    attention_sink=False,
    block_m=64,
    block_k=64,
    pv_threshold=50.0,
    enable_dynamic_skip=False,
    return_softmax_lse=False,
    return_sparsity=False,
    out=None,
):
    """Varlen TopK 路径自动稀疏"""
    return _auto_sparse_attn_varlen_core(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        cdfthreshd=None,
        topk=topk,
        attention_sink=attention_sink,
        block_m=block_m,
        block_k=block_k,
        pv_threshold=pv_threshold,
        enable_dynamic_skip=enable_dynamic_skip,
        return_softmax_lse=return_softmax_lse,
        return_sparsity=return_sparsity,
        out=out,
    )


def _auto_sparse_attn_core(
    *,
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    # Auto-sparsity parameters
    smooth_k=True,
    simthreshd1=0.6,
    cdfthreshd=None,
    topk=None,
    attention_sink=False,
    block_m=64,
    block_k=64,
    # PV skip optimization parameters
    pv_threshold=50.0,
    enable_dynamic_skip=False,
    # Return options
    return_softmax_lse=False,
    return_sparsity=False,
    out=None,
):
    """
    Core implementation for auto-generating sparsity (CDF or TopK) and calling sparse_attn_func.
    Exactly one of cdfthreshd / topk must be provided.
    """
    assert (cdfthreshd is None) ^ (topk is None), "Provide exactly one of cdfthreshd or topk"
    # FA2 sparse kernel uses a fixed block size (BLOCK_M = 64). Keep q/k block sizes aligned.
    assert block_k == block_m, "block_k must equal block_m to match FA2 sparse kernel"

    # Input validation and conversion
    assert q.dim() == 4, f"q must be 4D (B, L, H, D), got shape {q.shape}"
    assert q.size(1) >= 128, "seqlen should be not less than 128."

    torch.cuda.set_device(q.device)

    # Convert to BHLD layout for processing
    q_bhld = q.transpose(1, 2).contiguous()  # (B, H, L, D)
    k_bhld = k.transpose(1, 2).contiguous()
    v_bhld = v.transpose(1, 2).contiguous()

    # Ensure correct dtype
    dtype = q_bhld.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q_bhld = q_bhld.to(torch.float16)
        k_bhld = k_bhld.to(torch.float16)
        v_bhld = v_bhld.to(torch.float16)
    else:
        q_bhld = q_bhld.to(torch.bfloat16)
        k_bhld = k_bhld.to(torch.bfloat16)
        v_bhld = v_bhld.to(torch.bfloat16)

    # Apply K smoothing (subtract mean directly)
    if smooth_k:
        k_bhld = k_bhld - k_bhld.mean(dim=-2, keepdim=True)

    B, H, seqlen_q, headdim = q_bhld.shape
    seqlen_k = k_bhld.size(-2)

    # Generate sparsity pattern with block_offset format
    block_offset, block_count = get_block_map_meansim(
        q_bhld, k_bhld,
        is_causal=causal,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
        topk=topk,
        return_block_offset=True,
        attention_sink=attention_sink,
        BLKQ=block_m,
        BLKK=block_k,
    )

    if softmax_scale is None:
        softmax_scale = headdim ** (-0.5)

    # Disable Vertical sparsity (set column_count to 0)
    num_blocks_q = (seqlen_q + block_m - 1) // block_m
    num_blocks_k = (seqlen_k + block_k - 1) // block_k

    column_count = torch.zeros(
        (B, H, num_blocks_q), dtype=torch.int32, device=q.device
    )
    column_index = torch.zeros(
        (B, H, num_blocks_q, 1), dtype=torch.int32, device=q.device
    )

    # Convert block indices to row offsets (kernel expects row offsets, not block indices)
    block_offset = block_offset * block_k

    # Call sparse attention kernel (expects BLHD layout)
    result = sparse_attn_func(
        q, k, v,  # Use original BLHD layout
        block_count=block_count,
        block_offset=block_offset,
        column_count=column_count,
        column_index=column_index,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        return_softmax_lse=return_softmax_lse,
        out=out,
        pv_threshold=pv_threshold,
        enable_dynamic_skip=enable_dynamic_skip,
    )

    # Calculate sparsity if requested
    if return_sparsity:
        if not causal:
            qk_sparsity = 1 - (block_count.float().sum()) / (
                num_blocks_k * num_blocks_q * B * H
            )
        else:
            # For causal mask, expected number of blocks is (num_blocks_k + 1) / 2 per query block
            qk_sparsity = 1 - (block_count.float().sum()) / (
                ((num_blocks_k + 1) // 2) * num_blocks_q * B * H
            )

        if return_softmax_lse:
            out, softmax_lse = result
            return out, softmax_lse, qk_sparsity.item()
        else:
            return result, qk_sparsity.item()

    return result


def spas_fa2_attn_meansim_cuda(
    q,
    k,
    v,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    smooth_k=True,
    simthreshd1=0.6,
    cdfthreshd=0.98,
    attention_sink=False,
    block_m=64,
    block_k=64,
    pv_threshold=50.0,
    enable_dynamic_skip=False,
    return_softmax_lse=False,
    return_sparsity=False,
    out=None,
):
    """
    Auto-sparse attention using CDF thresholding (cdfthreshd). Mirrors SpargeAttn's mean-sim CDF path.
    """
    return _auto_sparse_attn_core(
        q=q,
        k=k,
        v=v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
        topk=None,
        attention_sink=attention_sink,
        block_m=block_m,
        block_k=block_k,
        pv_threshold=pv_threshold,
        enable_dynamic_skip=enable_dynamic_skip,
        return_softmax_lse=return_softmax_lse,
        return_sparsity=return_sparsity,
        out=out,
    )


def spas_fa2_attn_meansim_topk_cuda(
    q,
    k,
    v,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    smooth_k=True,
    simthreshd1=-0.1,
    topk=0.5,
    attention_sink=False,
    block_m=64,
    block_k=64,
    pv_threshold=50.0,
    enable_dynamic_skip=False,
    return_softmax_lse=False,
    return_sparsity=False,
    out=None,
):
    """
    Auto-sparse attention using Top-K ratio (topk). Mirrors SpargeAttn's mean-sim topk path.
    """
    return _auto_sparse_attn_core(
        q=q,
        k=k,
        v=v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        cdfthreshd=None,
        topk=topk,
        attention_sink=attention_sink,
        block_m=block_m,
        block_k=block_k,
        pv_threshold=pv_threshold,
        enable_dynamic_skip=enable_dynamic_skip,
        return_softmax_lse=return_softmax_lse,
        return_sparsity=return_sparsity,
        out=out,
    )


def _auto_sparse_attn_varlen_core(
    *,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    # Auto-sparsity parameters
    smooth_k=True,
    simthreshd1=0.6,
    cdfthreshd=None,
    topk=None,
    attention_sink=False,
    block_m=64,
    block_k=64,
    # PV skip optimization parameters
    pv_threshold=50.0,
    enable_dynamic_skip=False,
    # Return options
    return_softmax_lse=False,
    return_sparsity=False,
    out=None,
):
    """
    Varlen auto-sparse: generate block_count/block_offset/column_count/column_index per sequence,
    then call sparse_attn_varlen_func.
    """
    assert (cdfthreshd is None) ^ (topk is None), "Provide exactly one of cdfthreshd or topk"
    assert block_k == block_m == 64, "FA2 sparse kernel uses fixed BLOCK_M = BLOCK_K = 64"

    # Input validation
    assert q.dim() == 3, f"q must be 3D (total_q, H, D), got {q.shape}"
    assert k.dim() == 3 and v.dim() == 3
    B = cu_seqlens_q.numel() - 1
    assert B > 0
    assert cu_seqlens_k.numel() - 1 == B

    torch.cuda.set_device(q.device)

    H = q.size(1)
    headdim = q.size(2)
    if softmax_scale is None:
        softmax_scale = headdim ** (-0.5)

    max_q_blocks = (max_seqlen_q + block_m - 1) // block_m

    block_count = torch.zeros((B, H, max_q_blocks), dtype=torch.int32, device=q.device)
    # Disable Vertical sparsity (set column_count to 0)
    column_count = torch.zeros((B, H, max_q_blocks), dtype=torch.int32, device=q.device)
    column_index = torch.zeros((B, H, max_q_blocks, 1), dtype=torch.int32, device=q.device)

    # Pre-compute max_nnz_s (max num_k_blocks across all samples)
    max_nnz_s = 0
    for b in range(B):
        len_k = cu_seqlens_k[b + 1].item() - cu_seqlens_k[b].item()
        if len_k > 0:
            num_k_blocks = (len_k + block_k - 1) // block_k
            max_nnz_s = max(max_nnz_s, num_k_blocks)

    # Allocate block_offset with max_nnz_s (fill with large value for invalid positions)
    block_offset = torch.full((B, H, max_q_blocks, max_nnz_s), 10000000, dtype=torch.int32, device=q.device) if max_nnz_s > 0 else None

    # Single pass: compute block maps and fill data
    for b in range(B):
        q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
        k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()
        len_q = q_end - q_start
        len_k = k_end - k_start
        if len_q == 0 or len_k == 0:
            continue

        num_q_blocks = (len_q + block_m - 1) // block_m
        num_k_blocks = (len_k + block_k - 1) // block_k

        qb = q[q_start:q_end].transpose(0, 1).contiguous().unsqueeze(0)  # (1, H, Lq, D)
        kb = k[k_start:k_end].transpose(0, 1).contiguous().unsqueeze(0)  # (1, H, Lk, D)

        # Apply K smoothing (subtract mean directly)
        if smooth_k:
            kb = kb - kb.mean(dim=-2, keepdim=True)

        block_offset_b, block_count_b = get_block_map_meansim(
            qb, kb,
            is_causal=causal,
            simthreshd1=simthreshd1,
            cdfthreshd=cdfthreshd,
            topk=topk,
            return_block_offset=True,
            attention_sink=attention_sink,
            BLKQ=block_m,
            BLKK=block_k,
        )

        # Fill data directly
        block_count[b, :, :num_q_blocks] = block_count_b[0]
        nnz_s_b = block_offset_b.size(-1)
        block_offset[b, :, :num_q_blocks, :nnz_s_b] = block_offset_b[0]

    # Note: varlen sparse kernel may have different block_offset semantics
    # TODO: Investigate varlen sparse kernel implementation
    # For now, convert block indices to row offsets (same as non-varlen version)
    block_offset = block_offset * block_k

    # Call sparse varlen kernel
    result = sparse_attn_varlen_func(
        q,
        k,
        v,
        block_count=block_count,
        block_offset=block_offset if block_offset is not None else torch.full((B, H, max_q_blocks, 1), 10000000, dtype=torch.int32, device=q.device),
        column_count=column_count,
        column_index=column_index,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        return_softmax_lse=return_softmax_lse,
        out=out,
        pv_threshold=pv_threshold,
        enable_dynamic_skip=enable_dynamic_skip,
    )

    if return_sparsity:
        # Use actual per-batch num_q_blocks and num_k_blocks
        sparsities = []
        for b in range(B):
            len_q = cu_seqlens_q[b + 1] - cu_seqlens_q[b]
            len_k = cu_seqlens_k[b + 1] - cu_seqlens_k[b]
            if len_q == 0 or len_k == 0:
                continue
            num_q_blocks = (len_q + block_m - 1) // block_m
            num_k_blocks = (len_k + block_k - 1) // block_k
            if not causal:
                denom = num_k_blocks * num_q_blocks * H
            else:
                denom = ((num_k_blocks + 1) // 2) * num_q_blocks * H
            sparsities.append(1 - block_count[b, :, :num_q_blocks].float().sum() / denom)
        sparsity = torch.stack(sparsities).mean().item() if len(sparsities) > 0 else 0.0
        if return_softmax_lse:
            out_tensor, softmax_lse = result
            return out_tensor, softmax_lse, sparsity
        else:
            return result, sparsity

    return result


def spas_fa2_attn_meansim_varlen_cuda(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    smooth_k=True,
    simthreshd1=0.6,
    cdfthreshd=0.98,
    attention_sink=False,
    block_m=64,
    block_k=64,
    pv_threshold=50.0,
    enable_dynamic_skip=False,
    return_softmax_lse=False,
    return_sparsity=False,
    out=None,
):
    """Varlen CDF 路径自动稀疏"""
    return _auto_sparse_attn_varlen_core(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
        topk=None,
        attention_sink=attention_sink,
        block_m=block_m,
        block_k=block_k,
        pv_threshold=pv_threshold,
        enable_dynamic_skip=enable_dynamic_skip,
        return_softmax_lse=return_softmax_lse,
        return_sparsity=return_sparsity,
        out=out,
    )


def spas_fa2_attn_meansim_topk_varlen_cuda(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    smooth_k=True,
    simthreshd1=-0.1,
    topk=0.5,
    attention_sink=False,
    block_m=64,
    block_k=64,
    pv_threshold=50.0,
    enable_dynamic_skip=False,
    return_softmax_lse=False,
    return_sparsity=False,
    out=None,
):
    """Varlen TopK 路径自动稀疏"""
    return _auto_sparse_attn_varlen_core(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        cdfthreshd=None,
        topk=topk,
        attention_sink=attention_sink,
        block_m=block_m,
        block_k=block_k,
        pv_threshold=pv_threshold,
        enable_dynamic_skip=enable_dynamic_skip,
        return_softmax_lse=return_softmax_lse,
        return_sparsity=return_sparsity,
        out=out,
    )


def fwd_sparse_mean_pool_fast(x,BLK,mean=None):
    return flash_attn_cuda.fwd_sparse_mean_pool_fast(x,BLK,mean)

def get_block_map_fast(q, k, topk_ratio, BLKQ=128, BLKK=64):
    meank = torch.mean(k, dim=-3, keepdim=True)
    pooled_kblocks = fwd_sparse_mean_pool_fast(k, BLKK, meank)
    pooled_qblocks = fwd_sparse_mean_pool_fast(q,BLKQ)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)
    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices
    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


class SparseLinearAttention(nn.Module):
    def __init__(self, head_dim, topk, feature_map='softmax', use_bf16=True, use_fp8=False, tie_feature_map_qk=True):
        R'''
        Args:
            head_dim: dimension of each head.
            topk: ratio of keys selected for sparse attention, shared across all queries.
            feature_map: feature map for linear attention, one of ['hedgehog', 'elu', 'relu', 'softmax'].
            use_bf16: whether to use bfloat16 (default) or float16 for computation. The conversion to bf16/fp16 is done inside the module.
            tie_feature_map_qk: whether to use the same feature map for query and key.
        '''

        super().__init__()
        assert not (use_bf16 and use_fp8), "Only one of bf16 and fp8 can be used."
        assert head_dim in (64, 128), "Dtype fp16/bf16 only support dim (64, 128)."
        assert not (use_fp8 and head_dim==64), "Dtype fp8 only support dim 128."
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.dtype = torch.float8_e4m3fn if use_fp8 else self.dtype
        self.topk = topk
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.float32)

        if feature_map == 'elu':
            def elu_feature_map(x):
                return F.elu(x) + 1
            self.feature_map_q = elu_feature_map
            self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)
            self.feature_map_q = softmax_feature_map
            self.feature_map_k = softmax_feature_map
        else:
            raise NotImplementedError(f'Not supported feature map {feature_map}.')

        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q

        self.init_weights_()

    def init_weights_(self):
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)

    def forward(self, q, k, v, return_sparsity=False):
        R'''
        Args:
            q: queries of shape (B, L, H, D).
            k: keys of shape (B, L, H, D).
            v: values of shape (B, L, H, D).
            return_sparsity: whether to return the actual sparsity.
        '''

        B, seqlen_q, H, headdim = q.shape
        if headdim == 64:
            block_m = 64 if seqlen_q <= 2048 else 128
        elif headdim == 128:
            block_m = 64 if seqlen_q <= 2048 else 128
        block_k = 64
        if headdim == 64:
            sparse_map, lut, real_topk = get_block_map(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), topk_ratio=self.topk, BLKQ=block_m, BLKK=block_k)
        else:
            sparse_map, lut, real_topk = get_block_map_fast(q, k, topk_ratio=self.topk, BLKQ=block_m, BLKK=block_k)

        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)

        ########## SPARGE BEGIN ##########
        headdim = q.size(-1)
        block_offset, block_count = block_map_to_block_offset_triton(sparse_map)
        block_offset = block_offset * block_k
        softmax_scale = 1.0 / (headdim ** 0.5)
        assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

        seqlen_k = k.size(1)
        num_blocks_q = (seqlen_q + block_m - 1) // block_m
        num_blocks_k = (seqlen_k + block_k - 1) // block_k
        column_count = torch.empty(
            (B, H, num_blocks_q), dtype=torch.int32, device=q.device
        )
        column_index = torch.empty(
            (B, H, num_blocks_q, 1), dtype=torch.int32, device=q.device
        )

        o_s = sparse_attn_func(
            q, k, v,  # Use original BLHD layout
            block_count=block_count,
            block_offset=block_offset,
            column_count=column_count,
            column_index=column_index,
            softmax_scale=softmax_scale,
            is_sla=True,
        )
        # pdb.set_trace()

        ########## SPARGE END ##########

        '''
        q = self.feature_map_q(q).contiguous().to(self.dtype) # c_q
        k = self.feature_map_k(k).contiguous().to(self.dtype) # c_k
        def calc_linear(q, k, v):
            kvsum = k.transpose(-1, -2) @ v
            ksum = torch.sum(k, dim=-2, keepdim=True)
            return (q @ kvsum) / (1e-5 + (q * ksum).sum(dim=-1, keepdim=True))
        o_l = calc_linear(q, k, v)

        with torch.amp.autocast('cuda', dtype=self.dtype):
            o_l = self.proj_l(o_l)
        o = (o_s + o_l).to(dtype)
        '''

        if return_sparsity:
            return o_s, real_topk / sparse_map.shape[-1]
        else:
            return o_s

def sparse_attn_with_sla(
    q,
    k,
    v,
    topk=0.2,
    feature_map="softmax",
    use_bf16=True,
    use_fp8=False,
    *,
    return_sparsity=False,
):
    """Compute attention with vertical and slash sparsity patterns.
    Most Arguments are the same with the flash_attn_func interface, except for 4 extra args:
    block_count and block_offset for slash sparsity patterns, and
    column_count and column_index for vertical sparsity patterns.
    For more details please refer to Appendix C.4.2 of paper https://arxiv.org/abs/2407.02490.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        return_sparsity: bool. Whether to return the sparsity. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """

    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    dtype = torch.float8_e4m3fn if use_fp8 else dtype
    B, seqlen_q, H, headdim = q.shape
    assert not (use_bf16 and use_fp8), "Only one of bf16 and fp8 can be used."
    assert headdim in (64, 128), "Dtype fp16/bf16 only support dim (64, 128)."
    assert not (use_fp8 and headdim==64), "Dtype fp8 only support dim 128."
    if headdim == 64:
        block_m = 64 if seqlen_q <= 2048 else 128
    elif headdim == 128:
        block_m = 64 if seqlen_q <= 2048 else 128
    block_k = 64
    if headdim == 64:
        sparse_map, lut, real_topk = get_block_map(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), topk_ratio=topk, BLKQ=block_m, BLKK=block_k)
    else:
        sparse_map, lut, real_topk = get_block_map_fast(q, k, topk_ratio=topk, BLKQ=block_m, BLKK=block_k)

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    ########## SPARGE BEGIN ##########
    headdim = q.size(-1)
    block_offset, block_count = block_map_to_block_offset_triton(sparse_map)
    block_offset = block_offset * block_k
    softmax_scale = 1.0 / (headdim ** 0.5)
    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    seqlen_k = k.size(1)
    num_blocks_q = (seqlen_q + block_m - 1) // block_m
    num_blocks_k = (seqlen_k + block_k - 1) // block_k
    column_count = torch.empty(
        (B, H, num_blocks_q), dtype=torch.int32, device=q.device
    )
    column_index = torch.empty(
        (B, H, num_blocks_q, 1), dtype=torch.int32, device=q.device
    )
    o_s = sparse_attn_func(
        q, k, v,  # Use original BLHD layout
        block_count=block_count,
        block_offset=block_offset,
        column_count=column_count,
        column_index=column_index,
        softmax_scale=softmax_scale,
        is_sla=True,
    )

    if return_sparsity:
        return o_s, real_topk / sparse_map.shape[-1]
    else:
        return o_s


def _require_hg_varlen_symbol(name: str):
    symbol = getattr(flash_attn_cuda, name, None)
    if symbol is None:
        raise RuntimeError(
            f"{name} is unavailable in this build. Rebuild flash_attn with HAS_HG_DISPATCH enabled."
        )
    return symbol


def _validate_hg_paged_kv_contract(k_cache, v_cache) -> None:
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("HG paged KV cache expects k and v to both be 4D tensors")
    if k_cache.shape[0] != v_cache.shape[0]:
        raise ValueError("HG paged KV cache expects k and v to agree on num_blocks")
    if k_cache.shape[1] != v_cache.shape[1] or k_cache.shape[2] != v_cache.shape[2]:
        raise ValueError(
            "HG paged KV cache expects "
            "k=[num_blocks, page_block_size, num_heads_k, d_qk], "
            "v=[num_blocks, page_block_size, num_heads_k, d_v]"
        )

def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
    is_fp8_kvcache: bool = False,
):
    return None, None


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: Optional[torch.Tensor],
    num_splits: Optional[torch.Tensor],
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    use_cuda_graph: bool = True,
    out: Optional[torch.Tensor] = None,
):
    if k_cache.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError(
            "HG MLA dispatch in the main repository supports fp16/bf16 only; "
            "fp8/int8 MLA is not supported."
        )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    hg_mla = _require_hg_varlen_symbol("hg_fwd_kvcache_mla")
    max_seqlen_k = 1 if use_cuda_graph else cache_seqlens.max().item()
    result = hg_mla(
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        out,
        max_seqlen_k,
    )
    if len(result) < 2:
        raise RuntimeError("hg_fwd_kvcache_mla did not return softmax_lse")
    return result[0], result[1]


def hg_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    return_attn_probs=False,
    return_softmax_lse=False,
    block_table=None,
    layout="bshd",
    seqused_q=None,
    seqused_k=None,
    out=None,
    qv=None,
    attention_chunk=0,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    # FA3 only
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits=1,
    # Version selector
    fa_version=2,
    s_aux=None,
    is_bf16_output=True,
    custom_mask=None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs [optional]: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        return_softmax_lse [optional]: bool. Same to return_attn_probs.
        block_table [optional]: torch.Tensor. Required when `seqused_k` is provided for HG
            prefix / paged-cache paths. Must be None for the standard varlen path
            where `seqused_k` is not provided.
        layout [optional]: str. Decide which layout for q/k/v and output. "bshd"
            and HG prefix-prefill "bhsd" are supported.
        seqused_q [optional]: (batch_size), int32_t. Align the interface with official fa3. Not supported yet!
        seqused_k [optional]: (batch_size), int32_t. When None, this function uses the
            standard varlen path driven by `cu_seqlens_q/cu_seqlens_k`. When provided,
            it switches to the HG prefix-prefill / prefix-decode / paged-kvcache paths.
        out [optional]: (total_q, nheads, headdim). Align the interface with vllm-fa codes.
        qv [optional]: (total_q, nheads, headdim_nope). The Tensor of headdim without rope components. Align the interface with official fa3. Not supported yet!
        attention_chunk [optional]: int. Align the interface with official fa3. Not supported yet!
        pack_gqa [optional]: torch.Tensor. Align the interface with official fa3. Not supported yet!
        deterministic [optional]: bool. Align the interface with official fa3. Not supported yet!
        sm_margin [optional]: int. Align the interface with official fa3. Not supported yet!
        scheduler_metadata [optional]: torch.Tensor. Align the interface with vllm-fa. Not supported yet!
        q_descale [optional]: torch.Tensor. Align the interface with official fa3. Not supported yet!
        k_descale [optional]: torch.Tensor. Align the interface with official fa3. Not supported yet!
        v_descale [optional]: torch.Tensor. Align the interface with official fa3. Not supported yet!
        num_splits [optional]: int. May be used in paged attention with splitkv. Align the interface with official fa3. Not supported yet!
        fa_version [optional]: int. Align the interface with vllm-fa codes.
        s_aux [optional]: torch.Tensor. Align the interface with vllm-fa. Not supported yet!
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True or return_softmax_lse=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
            This is only available on the standard varlen path.
    """
    unsupported = []
    if seqused_q is not None:
        unsupported.append("seqused_q")
    if qv is not None:
        unsupported.append("qv")
    if attention_chunk != 0:
        unsupported.append("attention_chunk")
    if pack_gqa is not None:
        unsupported.append("pack_gqa")
    if sm_margin != 0:
        unsupported.append("sm_margin")
    if scheduler_metadata is not None:
        unsupported.append("scheduler_metadata")
    if num_splits != 1:
        unsupported.append("num_splits")
    if fa_version != 2:
        unsupported.append("fa_version")
    if custom_mask is not None:
        unsupported.append("custom_mask")
    if unsupported:
        raise NotImplementedError(
            "For hg_flash_attn_varlen_func, args "
            + "/".join(unsupported)
            + " are not supported yet!"
        )

    if layout not in ("bshd", "bhsd"):
        raise NotImplementedError(
            "hg_flash_attn_varlen_func only supports bshd/bhsd layout"
        )
    layout_id = 1 if layout == "bshd" else 0
    if cu_seqlens_q is None:
        raise ValueError("cu_seqlens_q must be provided")

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    s_aux = maybe_contiguous(s_aux)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    wants_aux = return_attn_probs or return_softmax_lse

    if seqused_k is None:
        if cu_seqlens_k is None:
            raise ValueError("cu_seqlens_k must be provided when seqused_k is None")
        if block_table is not None:
            raise ValueError("block_table is only supported when seqused_k is provided")
        if any(x is not None for x in (q_descale, k_descale, v_descale)):
            raise NotImplementedError(
                "q_descale/k_descale/v_descale are not supported for the standard varlen compatibility path"
            )

        if out is None:
            return FlashAttnVarlenFunc.apply(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                softcap,
                alibi_slopes,
                deterministic,
                wants_aux,
                None,
                None,
                None,
                block_table,
                None,
            )

        out, softmax_lse, S_dmask, _ = _wrapped_flash_attn_varlen_forward(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            window_size[0],
            window_size[1],
            softcap,
            wants_aux and dropout_p > 0,
        )
        return out if not wants_aux else (out, softmax_lse, S_dmask)

    if cu_seqlens_k is not None:
        raise ValueError("cu_seqlens_k and seqused_k cannot be provided at the same time")
    if block_table is None:
        raise ValueError("block_table must be provided when seqused_k is used")
    if dropout_p != 0.0:
        raise NotImplementedError("dropout_p must be 0.0 for HG prefix/paged compatibility paths")

    if k.dtype.is_floating_point:
        k_dtype_bits = torch.finfo(k.dtype).bits
    else:
        k_dtype_bits = torch.iinfo(k.dtype).bits

    if max_seqlen_q > 16 or (k_dtype_bits == 8 and max_seqlen_q > 1):
        if (
            layout == "bshd"
            and k_dtype_bits == 16
            and q.shape[-1] == 256
            and v.shape[-1] == 256
            and s_aux is None
        ):
            out, softmax_lse = flash_attn_cuda.varlen_fwd_unified(
                q,
                k,
                v,
                out,
                cu_seqlens_q,
                max_seqlen_q,
                seqused_k,
                max_seqlen_k,
                block_table,
                softmax_scale,
                softcap,
                None,   # q_descale
                None,   # k_descale
                None,   # v_descale
                None,   # output_scale
                causal,
                window_size[0],
                window_size[1],
                alibi_slopes,
                False,  # use_alibi_sqrt
                None,   # qq_bias
                s_aux,
                None,   # mm_prefix_range
            )
            return (out, softmax_lse) if wants_aux else out
        prefix_prefill = _require_hg_varlen_symbol("hg_prefix_prefill_varlen_fwd")
        result = prefix_prefill(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            None,
            seqused_k,
            alibi_slopes,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            0.0,
            softmax_scale,
            False,
            causal,
            window_size[0],
            window_size[1],
            softcap,
            return_softmax_lse or return_attn_probs,
            layout_id,
            None if k_dtype_bits == 16 else q_descale,
            None if k_dtype_bits == 16 else k_descale,
            None if k_dtype_bits == 16 else v_descale,
            s_aux,
            is_bf16_output,
        )
        fa_output = result[0]
        return (fa_output, result[1]) if (return_softmax_lse or return_attn_probs) else fa_output

    if k_dtype_bits == 16:
        if layout == "bhsd":
            raise NotImplementedError("HG bhsd prefix decode is not supported yet")
        prefix_decode = _require_hg_varlen_symbol("hg_prefix_decode_varlen_fwd")
        result = prefix_decode(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            None,
            seqused_k,
            alibi_slopes,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            0.0,
            softmax_scale,
            False,
            causal,
            window_size[0],
            window_size[1],
            softcap,
            return_softmax_lse or return_attn_probs,
            layout_id,
            None,
            None,
            None,
            s_aux,
            is_bf16_output,
        )
        fa_output = result[0]
        return (fa_output, result[1]) if (return_softmax_lse or return_attn_probs) else fa_output

    if return_softmax_lse or return_attn_probs:
        raise NotImplementedError(
            "return_softmax_lse is not supported for the HG paged-kvcache compatibility path"
        )

    if layout == "bhsd":
        raise NotImplementedError("HG bhsd paged-kvcache decode is not supported yet")

    _validate_hg_paged_kv_contract(k, v)
    if k.shape[1] != 128:
        raise NotImplementedError("HG paged-kvcache path currently requires page_block_size == 128")
    hg_kvcache = _require_hg_varlen_symbol("hg_fwd_kvcache_bshd")
    result = hg_kvcache(
        q.unsqueeze(1),
        k,
        v,
        None,
        None,
        None,
        seqused_k,
        1,
        None,
        None,
        None,
        None,
        block_table,
        alibi_slopes,
        out.unsqueeze(1) if out is not None else None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        False,
        -1,
        None,
        None,
        None if k_dtype_bits == 16 else q_descale,
        None if k_dtype_bits == 16 else k_descale,
        None if k_dtype_bits == 16 else v_descale,
        is_bf16_output,
    )
    return result[0].squeeze(1)
