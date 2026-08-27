# Copyright (c) 2024, Attnmask extension.
# Tests for FlashAttention with explicit attention mask.

import math
import pytest
import torch
from typing import Optional

from flash_attn import flash_attn_with_mask_func, flash_attn_varlen_with_mask_func


# ============================================================================
# Constants
# ============================================================================

HEADDIM = 128  # Currently only headdim=128 is supported


# ============================================================================
# Reference implementation
# ============================================================================

def attention_ref_with_mask(
    q: torch.Tensor,           # [batch, seqlen_q, nheads, headdim]
    k: torch.Tensor,           # [batch, seqlen_k, nheads_k, headdim]
    v: torch.Tensor,           # [batch, seqlen_k, nheads_k, headdim]
    attn_mask: torch.Tensor,   # [batch, nheads, seqlen_q, seqlen_k], bool
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    upcast: bool = True,
    reorder_ops: bool = False,
    return_lse: bool = False,
) -> torch.Tensor:
    """
    PyTorch reference implementation for attention with explicit mask.
    Used for correctness verification.
    
    Args:
        upcast: if True, compute in float32 for higher precision reference
        reorder_ops: if True, scale k instead of q (different numerical error pattern)
        return_lse: if True, also return log-sum-exp values
    """
    dtype_og = q.dtype
    batch, seqlen_q, nheads, headdim = q.shape
    _, seqlen_k, nheads_k, _ = k.shape

    if softmax_scale is None:
        softmax_scale = headdim ** (-0.5)

    # Expand GQA/MQA K/V
    if nheads_k != nheads:
        k = k.repeat_interleave(nheads // nheads_k, dim=2)
        v = v.repeat_interleave(nheads // nheads_k, dim=2)

    # Convert to [batch, nheads, seqlen, headdim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if upcast:
        q, k, v = q.float(), k.float(), v.float()

    # Compute attention scores (with optional reorder_ops)
    if not reorder_ops:
        scores = torch.matmul(q * softmax_scale, k.transpose(-2, -1))  # [b, h, sq, sk]
    else:
        scores = torch.matmul(q, (k * softmax_scale).transpose(-2, -1))

    # Apply causal mask
    if causal:
        causal_mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device),
            diagonal=seqlen_k - seqlen_q + 1
        )
        scores.masked_fill_(causal_mask, float('-inf'))

    # Apply explicit attention mask
    # attn_mask: True = attend, False = mask
    scores.masked_fill_(~attn_mask, float('-inf'))

    # Compute LSE before softmax (for debugging backward)
    lse = torch.logsumexp(scores, dim=-1)  # [b, h, sq]

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1).to(v.dtype)

    # Handle all-masked rows (replace NaN with 0)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # Output
    out = torch.matmul(attn_weights, v)

    # Convert back to [batch, seqlen, nheads, headdim]
    out = out.transpose(1, 2)

    if return_lse:
        return out.to(dtype=dtype_og), lse
    return out.to(dtype=dtype_og)


# ============================================================================
# Helper functions
# ============================================================================

def generate_random_mask(
    batch: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    mask_ratio: float = 0.3,
    device: str = 'cuda',
) -> torch.Tensor:
    """Generate random attention mask."""
    mask = torch.rand(batch, nheads, seqlen_q, seqlen_k, device=device) > mask_ratio
    return mask


def generate_prefix_mask(
    batch: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    prefix_len: int,
    device: str = 'cuda',
) -> torch.Tensor:
    """Generate prefix attention mask (first prefix_len positions visible to all queries)."""
    mask = torch.zeros(batch, nheads, seqlen_q, seqlen_k, dtype=torch.bool, device=device)
    mask[:, :, :, :prefix_len] = True
    return mask


def generate_block_sparse_mask(
    batch: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: int = 64,
    device: str = 'cuda',
) -> torch.Tensor:
    """Generate block-sparse attention mask (diagonal blocks visible)."""
    mask = torch.zeros(batch, nheads, seqlen_q, seqlen_k, dtype=torch.bool, device=device)
    for i in range(0, min(seqlen_q, seqlen_k), block_size):
        end_q = min(i + block_size, seqlen_q)
        end_k = min(i + block_size, seqlen_k)
        mask[:, :, i:end_q, i:end_k] = True
    return mask


# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture
def device():
    return 'cuda'


# ============================================================================
# Forward correctness tests
# ============================================================================

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128), (128, 256), (256, 128)])
@pytest.mark.parametrize("nheads", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_attnmask_forward_correctness(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    dtype: torch.dtype,
    causal: bool,
    device: str,
):
    """Test forward correctness against reference implementation (official style)."""
    torch.manual_seed(42)

    q = torch.randn(batch_size, seqlen_q, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device)

    # Generate random mask
    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, device=device)

    # Flash attention with mask
    out_flash = flash_attn_with_mask_func(q, k, v, attn_mask, causal=causal)

    # Reference implementation (upcast=True for high precision)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, causal=causal, upcast=True)

    # PyTorch native precision (upcast=False, reorder_ops=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, causal=causal, upcast=False, reorder_ops=True)

    # Print diff for debugging
    print(f"Output max diff: {(out_flash - out_ref).abs().max().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")

    # Check: FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation (official style)
    assert (out_flash - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128), (128, 256), (256, 128)])
@pytest.mark.parametrize("nheads", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_attnmask_forward_correctness_hdim64(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    dtype: torch.dtype,
    causal: bool,
    device: str,
):
    """Test forward correctness for headdim=64."""
    torch.manual_seed(42)

    headdim = 64
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, device=device)

    out_flash = flash_attn_with_mask_func(q, k, v, attn_mask, causal=causal)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, causal=causal, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, causal=causal, upcast=False, reorder_ops=True)

    assert (out_flash - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


# ============================================================================
# LSE (Log-Sum-Exp) correctness tests - 检查前向的中间状态
# ============================================================================

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128), (128, 256), (256, 128)])
@pytest.mark.parametrize("nheads", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_attnmask_lse_correctness(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    dtype: torch.dtype,
    causal: bool,
    device: str,
):
    """Test LSE correctness - LSE errors will cause backward failures."""
    torch.manual_seed(42)

    q = torch.randn(batch_size, seqlen_q, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, device=device)

    # Flash attention with return_attn_probs to get LSE
    out_flash, lse_flash, _ = flash_attn_with_mask_func(
        q, k, v, attn_mask, causal=causal, return_attn_probs=True
    )

    # Reference LSE
    out_ref, lse_ref = attention_ref_with_mask(
        q, k, v, attn_mask, causal=causal, upcast=True, return_lse=True
    )

    # Print diff for debugging
    print(f"Output max diff: {(out_flash - out_ref.to(dtype)).abs().max().item()}")
    
    # 忽略 -inf 位置（全 mask 行），只比较有效位置的 LSE
    valid_mask = torch.isfinite(lse_ref)
    if valid_mask.any():
        lse_diff = (lse_flash[valid_mask] - lse_ref[valid_mask]).abs()
        lse_max_diff = lse_diff.max().item()
        lse_mean_diff = lse_diff.mean().item()
    else:
        lse_max_diff = 0.0
        lse_mean_diff = 0.0
    
    print(f"LSE max diff (valid positions): {lse_max_diff}")
    print(f"LSE mean diff (valid positions): {lse_mean_diff}")
    print(f"LSE -inf positions: {(~valid_mask).sum().item()} / {valid_mask.numel()}")

    # Check output (official style)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, causal=causal, upcast=False, reorder_ops=True)
    assert (out_flash - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5

    # Check LSE - 这是反向传播正确性的关键！
    # LSE 误差应该很小，否则反向会出问题
    # 注意：只检查有效位置（非 -inf），-inf 位置是全 mask 行，不影响梯度计算
    lse_atol = 1e-2
    assert lse_max_diff <= lse_atol, f"LSE max diff {lse_max_diff} > {lse_atol}, backward will fail!"


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128), (128, 256), (256, 128)])
@pytest.mark.parametrize("nheads", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_attnmask_lse_correctness_hdim64(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    dtype: torch.dtype,
    causal: bool,
    device: str,
):
    """Test LSE correctness for headdim=64 - LSE errors will cause backward failures."""
    torch.manual_seed(42)

    headdim = 64
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, device=device)

    out_flash, lse_flash, _ = flash_attn_with_mask_func(
        q, k, v, attn_mask, causal=causal, return_attn_probs=True
    )

    out_ref, lse_ref = attention_ref_with_mask(
        q, k, v, attn_mask, causal=causal, upcast=True, return_lse=True
    )

    print(f"Output max diff: {(out_flash - out_ref.to(dtype)).abs().max().item()}")

    valid_mask = torch.isfinite(lse_ref)
    if valid_mask.any():
        lse_diff = (lse_flash[valid_mask] - lse_ref[valid_mask]).abs()
        lse_max_diff = lse_diff.max().item()
        lse_mean_diff = lse_diff.mean().item()
    else:
        lse_max_diff = 0.0
        lse_mean_diff = 0.0

    print(f"LSE max diff (valid positions): {lse_max_diff}")
    print(f"LSE mean diff (valid positions): {lse_mean_diff}")
    print(f"LSE -inf positions: {(~valid_mask).sum().item()} / {valid_mask.numel()}")

    out_pt = attention_ref_with_mask(q, k, v, attn_mask, causal=causal, upcast=False, reorder_ops=True)
    assert (out_flash - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5

    lse_atol = 1e-2
    assert lse_max_diff <= lse_atol, f"LSE max diff {lse_max_diff} > {lse_atol}, backward will fail!"


# ============================================================================
# Backward correctness tests
# ============================================================================

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128), (128, 256), (256, 128)])
@pytest.mark.parametrize("nheads", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_attnmask_backward_correctness(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    dtype: torch.dtype,
    causal: bool,
    device: str,
):
    """Test backward gradient correctness against reference (official style)."""
    torch.manual_seed(42)

    q = torch.randn(batch_size, seqlen_q, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, device=device)
    dout = torch.randn(batch_size, seqlen_q, nheads, HEADDIM, dtype=dtype, device=device)

    # Flash attention forward + backward
    out_flash = flash_attn_with_mask_func(q, k, v, attn_mask, causal=causal)
    dq_flash, dk_flash, dv_flash = torch.autograd.grad(out_flash, (q, k, v), dout)

    # Reference (upcast=True) forward + backward
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = attention_ref_with_mask(q_ref, k_ref, v_ref, attn_mask, causal=causal, upcast=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout)

    # PyTorch native precision (upcast=False, reorder_ops=True) forward + backward
    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt = attention_ref_with_mask(q_pt, k_pt, v_pt, attn_mask, causal=causal, upcast=False, reorder_ops=True)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_pt, k_pt, v_pt), dout)

    # Print diff for debugging
    print(f"Output max diff: {(out_flash - out_ref).abs().max().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"dQ max diff: {(dq_flash - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk_flash - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv_flash - dv_ref).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")

    # Check: FlashAttention's numerical error is at most N times the numerical error
    # of a Pytorch implementation (official style)
    assert (out_flash - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5
    assert (dq_flash - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item() + 1e-3
    assert (dk_flash - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item() + 1e-3
    assert (dv_flash - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item() + 1e-3


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(128, 128), (128, 256), (256, 128)])
@pytest.mark.parametrize("nheads", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_attnmask_backward_correctness_hdim64(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    dtype: torch.dtype,
    causal: bool,
    device: str,
):
    """Test backward gradient correctness for headdim=64."""
    torch.manual_seed(42)

    headdim = 64
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device, requires_grad=True)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, device=device)
    dout = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=dtype, device=device)

    out_flash = flash_attn_with_mask_func(q, k, v, attn_mask, causal=causal)
    dq_flash, dk_flash, dv_flash = torch.autograd.grad(out_flash, (q, k, v), dout)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = attention_ref_with_mask(q_ref, k_ref, v_ref, attn_mask, causal=causal, upcast=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout)

    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt = attention_ref_with_mask(q_pt, k_pt, v_pt, attn_mask, causal=causal, upcast=False, reorder_ops=True)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_pt, k_pt, v_pt), dout)

    # headdim=64 的实现在数值上略微更敏感，放宽阈值一点点
    tol_eps = 2e-3
    assert (out_flash - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5
    assert (dq_flash - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item() + tol_eps
    assert (dk_flash - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item() + tol_eps
    assert (dv_flash - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item() + tol_eps


# ============================================================================
# Mask pattern tests - 核心 attnmask 功能测试
# ============================================================================

@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_all_ones(dtype: torch.dtype, device: str):
    """Test: all positions attend (equivalent to standard attention)."""
    batch_size, seqlen, nheads = 2, 256, 8

    q = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)

    attn_mask = torch.ones(batch_size, nheads, seqlen, seqlen, dtype=torch.bool, device=device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    # Check: FlashAttention's error <= 2x Pytorch's error (official style)
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_all_ones_hdim64(dtype: torch.dtype, device: str):
    """Test: all positions attend (equivalent to standard attention) for headdim=64."""
    batch_size, seqlen, nheads = 2, 256, 8
    headdim = 64

    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)

    attn_mask = torch.ones(batch_size, nheads, seqlen, seqlen, dtype=torch.bool, device=device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_all_zeros(dtype: torch.dtype, device: str):
    """Test: all positions masked (edge case)."""
    batch_size, seqlen, nheads = 2, 128, 4

    q = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)

    attn_mask = torch.zeros(batch_size, nheads, seqlen, seqlen, dtype=torch.bool, device=device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)

    # Should not crash; output may be zero or nan depending on implementation
    assert out.shape == (batch_size, seqlen, nheads, HEADDIM)


@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_all_zeros_hdim64(dtype: torch.dtype, device: str):
    """Test: all positions masked (edge case) for headdim=64."""
    batch_size, seqlen, nheads = 2, 128, 4
    headdim = 64

    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)

    attn_mask = torch.zeros(batch_size, nheads, seqlen, seqlen, dtype=torch.bool, device=device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)

    assert out.shape == (batch_size, seqlen, nheads, headdim)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("prefix_len", [32, 64, 128])
def test_attnmask_prefix_pattern(dtype: torch.dtype, prefix_len: int, device: str):
    """Test: prefix mask pattern (first N positions always visible)."""
    batch_size, seqlen, nheads = 2, 256, 8

    q = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)

    attn_mask = generate_prefix_mask(batch_size, nheads, seqlen, seqlen, prefix_len, device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    # Check: FlashAttention's error <= 2x Pytorch's error (official style)
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("prefix_len", [32, 64, 128])
def test_attnmask_prefix_pattern_hdim64(dtype: torch.dtype, prefix_len: int, device: str):
    """Test: prefix mask pattern (first N positions always visible) for headdim=64."""
    batch_size, seqlen, nheads = 2, 256, 8
    headdim = 64

    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)

    attn_mask = generate_prefix_mask(batch_size, nheads, seqlen, seqlen, prefix_len, device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_block_sparse_pattern(dtype: torch.dtype, device: str):
    """Test: block-sparse mask pattern (diagonal blocks)."""
    batch_size, seqlen, nheads = 2, 256, 8
    block_size = 64

    q = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)

    attn_mask = generate_block_sparse_mask(batch_size, nheads, seqlen, seqlen, block_size, device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    # Check: FlashAttention's error <= 2x Pytorch's error (official style)
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_block_sparse_pattern_hdim64(dtype: torch.dtype, device: str):
    """Test: block-sparse mask pattern (diagonal blocks) for headdim=64."""
    batch_size, seqlen, nheads = 2, 256, 8
    block_size = 64
    headdim = 64

    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)

    attn_mask = generate_block_sparse_mask(batch_size, nheads, seqlen, seqlen, block_size, device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("mask_ratio", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_attnmask_random_sparsity(dtype: torch.dtype, mask_ratio: float, device: str):
    """Test: random mask with varying sparsity levels."""
    batch_size, seqlen, nheads = 2, 256, 8

    q = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen, seqlen, mask_ratio, device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    # Check: FlashAttention's error <= 2x Pytorch's error (official style)
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("mask_ratio", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_attnmask_random_sparsity_hdim64(dtype: torch.dtype, mask_ratio: float, device: str):
    """Test: random mask with varying sparsity levels for headdim=64."""
    batch_size, seqlen, nheads = 2, 256, 8
    headdim = 64

    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen, seqlen, mask_ratio, device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_causal_combined(dtype: torch.dtype, device: str):
    """Test: causal mask combined with explicit mask."""
    batch_size, seqlen, nheads = 2, 256, 8
    prefix_len = 64

    q = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)

    # Prefix mask + causal
    attn_mask = generate_prefix_mask(batch_size, nheads, seqlen, seqlen, prefix_len, device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask, causal=True)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, causal=True, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, causal=True, upcast=False, reorder_ops=True)

    # Check: FlashAttention's error <= 2x Pytorch's error (official style)
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_causal_combined_hdim64(dtype: torch.dtype, device: str):
    """Test: causal mask combined with explicit mask for headdim=64."""
    batch_size, seqlen, nheads = 2, 256, 8
    prefix_len = 64
    headdim = 64

    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)

    attn_mask = generate_prefix_mask(batch_size, nheads, seqlen, seqlen, prefix_len, device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask, causal=True)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, causal=True, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, causal=True, upcast=False, reorder_ops=True)

    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_per_head_different_mask(dtype: torch.dtype, device: str):
    """Test: different mask per head."""
    batch_size, seqlen, nheads = 2, 256, 8

    q = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)

    # Different prefix length per head
    attn_mask = torch.zeros(batch_size, nheads, seqlen, seqlen, dtype=torch.bool, device=device)
    for h in range(nheads):
        prefix_len = 32 * (h + 1)  # 32, 64, 96, ...
        attn_mask[:, h, :, :prefix_len] = True

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    # Check: FlashAttention's error <= 2x Pytorch's error (official style)
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_per_head_different_mask_hdim64(dtype: torch.dtype, device: str):
    """Test: different mask per head for headdim=64."""
    batch_size, seqlen, nheads = 2, 256, 8
    headdim = 64

    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)

    attn_mask = torch.zeros(batch_size, nheads, seqlen, seqlen, dtype=torch.bool, device=device)
    for h in range(nheads):
        prefix_len = 32 * (h + 1)
        attn_mask[:, h, :, :prefix_len] = True

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_per_batch_different_mask(dtype: torch.dtype, device: str):
    """Test: different mask per batch element."""
    batch_size, seqlen, nheads = 4, 256, 8

    q = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)

    # Different mask per batch
    attn_mask = torch.zeros(batch_size, nheads, seqlen, seqlen, dtype=torch.bool, device=device)
    for b in range(batch_size):
        prefix_len = 64 * (b + 1)  # 64, 128, 192, 256
        attn_mask[b, :, :, :prefix_len] = True

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    # Check: FlashAttention's error <= 2x Pytorch's error (official style)
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_per_batch_different_mask_hdim64(dtype: torch.dtype, device: str):
    """Test: different mask per batch element for headdim=64."""
    batch_size, seqlen, nheads = 4, 256, 8
    headdim = 64

    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)

    attn_mask = torch.zeros(batch_size, nheads, seqlen, seqlen, dtype=torch.bool, device=device)
    for b in range(batch_size):
        prefix_len = 64 * (b + 1)
        attn_mask[b, :, :, :prefix_len] = True

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


# ============================================================================
# Mask backward gradient tests
# ============================================================================

@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("mask_type", ["prefix", "random", "block_sparse"])
def test_attnmask_backward_with_patterns(dtype: torch.dtype, mask_type: str, device: str):
    """Test backward gradients with different mask patterns (official style)."""
    batch_size, seqlen, nheads = 2, 256, 8

    torch.manual_seed(42)
    q = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)

    if mask_type == "prefix":
        attn_mask = generate_prefix_mask(batch_size, nheads, seqlen, seqlen, 64, device)
    elif mask_type == "random":
        attn_mask = generate_random_mask(batch_size, nheads, seqlen, seqlen, 0.3, device)
    else:  # block_sparse
        attn_mask = generate_block_sparse_mask(batch_size, nheads, seqlen, seqlen, 64, device)

    dout = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)

    # Flash forward + backward
    out_flash = flash_attn_with_mask_func(q, k, v, attn_mask)
    dq_flash, dk_flash, dv_flash = torch.autograd.grad(out_flash, (q, k, v), dout)

    # Reference (upcast=True) forward + backward
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = attention_ref_with_mask(q_ref, k_ref, v_ref, attn_mask, upcast=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout)

    # PyTorch native precision forward + backward
    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt = attention_ref_with_mask(q_pt, k_pt, v_pt, attn_mask, upcast=False, reorder_ops=True)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_pt, k_pt, v_pt), dout)

    # Print diff for debugging
    print(f"[{mask_type}] dQ max diff: {(dq_flash - dq_ref).abs().max().item():.6f}, "
          f"Pytorch: {(dq_pt - dq_ref).abs().max().item():.6f}")
    print(f"[{mask_type}] dK max diff: {(dk_flash - dk_ref).abs().max().item():.6f}, "
          f"Pytorch: {(dk_pt - dk_ref).abs().max().item():.6f}")
    print(f"[{mask_type}] dV max diff: {(dv_flash - dv_ref).abs().max().item():.6f}, "
          f"Pytorch: {(dv_pt - dv_ref).abs().max().item():.6f}")

    # Check: FlashAttention's error <= 3x Pytorch's error
    assert (dq_flash - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item() + 1e-3
    assert (dk_flash - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item() + 1e-3
    assert (dv_flash - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item() + 1e-3


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("mask_type", ["prefix", "random", "block_sparse"])
def test_attnmask_backward_with_patterns_hdim64(dtype: torch.dtype, mask_type: str, device: str):
    """Test backward gradients with different mask patterns for headdim=64."""
    batch_size, seqlen, nheads = 2, 256, 8
    headdim = 64

    torch.manual_seed(42)
    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device, requires_grad=True)

    if mask_type == "prefix":
        attn_mask = generate_prefix_mask(batch_size, nheads, seqlen, seqlen, 64, device)
    elif mask_type == "random":
        attn_mask = generate_random_mask(batch_size, nheads, seqlen, seqlen, 0.3, device)
    else:
        attn_mask = generate_block_sparse_mask(batch_size, nheads, seqlen, seqlen, 64, device)

    dout = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)

    out_flash = flash_attn_with_mask_func(q, k, v, attn_mask)
    dq_flash, dk_flash, dv_flash = torch.autograd.grad(out_flash, (q, k, v), dout)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = attention_ref_with_mask(q_ref, k_ref, v_ref, attn_mask, upcast=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout)

    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt = attention_ref_with_mask(q_pt, k_pt, v_pt, attn_mask, upcast=False, reorder_ops=True)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_pt, k_pt, v_pt), dout)

    assert (dq_flash - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item() + 1e-3
    assert (dk_flash - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item() + 1e-3
    assert (dv_flash - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item() + 1e-3


# ============================================================================
# Mask dtype tests
# ============================================================================

@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_uint8_mask(dtype: torch.dtype, device: str):
    """Test uint8 mask type."""
    batch_size, seqlen, nheads = 2, 128, 4

    q = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)

    # Compare bool vs uint8 mask
    attn_mask_bool = generate_random_mask(batch_size, nheads, seqlen, seqlen, device=device)
    attn_mask_uint8 = attn_mask_bool.to(torch.uint8)

    out_bool = flash_attn_with_mask_func(q, k, v, attn_mask_bool)
    out_uint8 = flash_attn_with_mask_func(q, k, v, attn_mask_uint8)

    torch.testing.assert_close(out_bool, out_uint8)


@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_uint8_mask_hdim64(dtype: torch.dtype, device: str):
    """Test uint8 mask type for headdim=64."""
    batch_size, seqlen, nheads = 2, 128, 4
    headdim = 64

    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)

    attn_mask_bool = generate_random_mask(batch_size, nheads, seqlen, seqlen, device=device)
    attn_mask_uint8 = attn_mask_bool.to(torch.uint8)

    out_bool = flash_attn_with_mask_func(q, k, v, attn_mask_bool)
    out_uint8 = flash_attn_with_mask_func(q, k, v, attn_mask_uint8)

    torch.testing.assert_close(out_bool, out_uint8)


# ============================================================================
# Variable length tests
# ============================================================================

@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("max_seqlen", [256, 512])
@pytest.mark.parametrize("nheads", [8])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_varlen_forward(
    batch_size: int,
    max_seqlen: int,
    nheads: int,
    dtype: torch.dtype,
    device: str,
):
    """Test variable-length forward."""
    torch.manual_seed(42)

    # Generate random sequence lengths
    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,))
    cu_seqlens = torch.cat([
        torch.tensor([0]),
        seqlens.cumsum(0)
    ]).to(torch.int32).to(device)

    total = cu_seqlens[-1].item()

    q = torch.randn(total, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(total, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(total, nheads, HEADDIM, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, max_seqlen, max_seqlen, device=device)

    out = flash_attn_varlen_with_mask_func(
        q, k, v, attn_mask,
        cu_seqlens, cu_seqlens,
        max_seqlen, max_seqlen,
    )

    assert out.shape == (total, nheads, HEADDIM)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("max_seqlen", [256, 512])
@pytest.mark.parametrize("nheads", [8])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_varlen_forward_hdim64(
    batch_size: int,
    max_seqlen: int,
    nheads: int,
    dtype: torch.dtype,
    device: str,
):
    """Test variable-length forward for headdim=64."""
    torch.manual_seed(42)

    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,))
    cu_seqlens = torch.cat([
        torch.tensor([0]),
        seqlens.cumsum(0)
    ]).to(torch.int32).to(device)

    total = cu_seqlens[-1].item()

    headdim = 64
    q = torch.randn(total, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(total, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(total, nheads, headdim, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, max_seqlen, max_seqlen, device=device)

    out = flash_attn_varlen_with_mask_func(
        q, k, v, attn_mask,
        cu_seqlens, cu_seqlens,
        max_seqlen, max_seqlen,
    )

    assert out.shape == (total, nheads, headdim)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("max_seqlen", [256])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_varlen_backward(
    batch_size: int,
    max_seqlen: int,
    nheads: int,
    dtype: torch.dtype,
    device: str,
):
    """Test variable-length backward."""
    torch.manual_seed(42)

    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,))
    cu_seqlens = torch.cat([
        torch.tensor([0]),
        seqlens.cumsum(0)
    ]).to(torch.int32).to(device)

    total = cu_seqlens[-1].item()

    q = torch.randn(total, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(total, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(total, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)

    attn_mask = torch.ones(batch_size, nheads, max_seqlen, max_seqlen, dtype=torch.bool, device=device)

    out = flash_attn_varlen_with_mask_func(
        q, k, v, attn_mask,
        cu_seqlens, cu_seqlens,
        max_seqlen, max_seqlen,
    )

    dout = torch.randn_like(out)
    out.backward(dout)

    assert q.grad is not None and not torch.isnan(q.grad).any()
    assert k.grad is not None and not torch.isnan(k.grad).any()
    assert v.grad is not None and not torch.isnan(v.grad).any()


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("max_seqlen", [256])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_varlen_backward_hdim64(
    batch_size: int,
    max_seqlen: int,
    nheads: int,
    dtype: torch.dtype,
    device: str,
):
    """Test variable-length backward for headdim=64."""
    torch.manual_seed(42)

    seqlens = torch.randint(max_seqlen // 2, max_seqlen + 1, (batch_size,))
    cu_seqlens = torch.cat([
        torch.tensor([0]),
        seqlens.cumsum(0)
    ]).to(torch.int32).to(device)

    total = cu_seqlens[-1].item()

    headdim = 64
    q = torch.randn(total, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(total, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(total, nheads, headdim, dtype=dtype, device=device, requires_grad=True)

    attn_mask = torch.ones(batch_size, nheads, max_seqlen, max_seqlen, dtype=torch.bool, device=device)

    out = flash_attn_varlen_with_mask_func(
        q, k, v, attn_mask,
        cu_seqlens, cu_seqlens,
        max_seqlen, max_seqlen,
    )

    dout = torch.randn_like(out)
    out.backward(dout)

    assert q.grad is not None and not torch.isnan(q.grad).any()
    assert k.grad is not None and not torch.isnan(k.grad).any()
    assert v.grad is not None and not torch.isnan(v.grad).any()


# ============================================================================
# Non-aligned sequence length tests (predicate check testing)
# 非64倍数序列长度测试 - 测试谓词检查
# ============================================================================

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    # seqlen_q 非64倍数
    (65, 128),      # seqlen_q = 64 + 1
    (127, 128),     # seqlen_q = 128 - 1
    (100, 128),     # 非对齐的 seqlen_q
    (33, 64),       # 小序列非对齐
    # seqlen_k 非64倍数
    (128, 65),      # seqlen_k = 64 + 1
    (128, 127),     # seqlen_k = 128 - 1
    (128, 100),     # 非对齐的 seqlen_k
    (64, 33),       # 小序列非对齐
    # 两者都非64倍数
    (65, 65),       # 两者都是 64 + 1
    (100, 127),     # 两者都非对齐
    (127, 65),      # 混合非对齐
    (200, 150),     # 较大序列非对齐
])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_attnmask_non_aligned_forward(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    dtype: torch.dtype,
    causal: bool,
    device: str,
):
    """Test forward correctness with non-64-aligned sequence lengths (predicate check)."""
    torch.manual_seed(42)

    q = torch.randn(batch_size, seqlen_q, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, device=device)

    out_flash = flash_attn_with_mask_func(q, k, v, attn_mask, causal=causal)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, causal=causal, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, causal=causal, upcast=False, reorder_ops=True)

    print(f"[seqlen_q={seqlen_q}, seqlen_k={seqlen_k}] Output max diff: {(out_flash - out_ref).abs().max().item():.6f}")
    print(f"[seqlen_q={seqlen_q}, seqlen_k={seqlen_k}] Pytorch max diff: {(out_pt - out_ref).abs().max().item():.6f}")

    assert (out_flash - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (65, 128),
    (127, 128),
    (100, 128),
    (33, 64),
    (128, 65),
    (128, 127),
    (128, 100),
    (64, 33),
    (65, 65),
    (100, 127),
    (127, 65),
    (200, 150),
])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_attnmask_non_aligned_forward_hdim64(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    dtype: torch.dtype,
    causal: bool,
    device: str,
):
    """Test forward correctness with non-64-aligned sequence lengths for headdim=64."""
    torch.manual_seed(42)

    headdim = 64
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, device=device)

    out_flash = flash_attn_with_mask_func(q, k, v, attn_mask, causal=causal)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, causal=causal, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, causal=causal, upcast=False, reorder_ops=True)

    assert (out_flash - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (33, 47),
    (65, 128),      # seqlen_q 非对齐
    (128, 65),      # seqlen_k 非对齐
    (65, 65),       # 两者非对齐
    (100, 127),     # 两者非对齐
])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_attnmask_non_aligned_backward(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    dtype: torch.dtype,
    causal: bool,
    device: str,
):
    """Test backward gradient correctness with non-64-aligned sequence lengths (predicate check)."""
    torch.manual_seed(42)

    q = torch.randn(batch_size, seqlen_q, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, device=device)
    dout = torch.randn(batch_size, seqlen_q, nheads, HEADDIM, dtype=dtype, device=device)

    # Flash attention forward + backward
    out_flash = flash_attn_with_mask_func(q, k, v, attn_mask, causal=causal)
    dq_flash, dk_flash, dv_flash = torch.autograd.grad(out_flash, (q, k, v), dout)

    # Reference forward + backward
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = attention_ref_with_mask(q_ref, k_ref, v_ref, attn_mask, causal=causal, upcast=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout)

    # PyTorch native precision forward + backward
    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt = attention_ref_with_mask(q_pt, k_pt, v_pt, attn_mask, causal=causal, upcast=False, reorder_ops=True)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_pt, k_pt, v_pt), dout)

    print(f"[seqlen_q={seqlen_q}, seqlen_k={seqlen_k}] Output max diff: {(out_flash - out_ref).abs().max().item():.6f}")
    print(f"[seqlen_q={seqlen_q}, seqlen_k={seqlen_k}] dQ max diff: {(dq_flash - dq_ref).abs().max().item():.6f}, Pytorch: {(dq_pt - dq_ref).abs().max().item():.6f}")
    print(f"[seqlen_q={seqlen_q}, seqlen_k={seqlen_k}] dK max diff: {(dk_flash - dk_ref).abs().max().item():.6f}, Pytorch: {(dk_pt - dk_ref).abs().max().item():.6f}")
    print(f"[seqlen_q={seqlen_q}, seqlen_k={seqlen_k}] dV max diff: {(dv_flash - dv_ref).abs().max().item():.6f}, Pytorch: {(dv_pt - dv_ref).abs().max().item():.6f}")

    assert (out_flash - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5
    assert (dq_flash - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item() + 1e-3
    assert (dk_flash - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item() + 1e-3
    assert (dv_flash - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item() + 1e-3


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (33, 47),
    (65, 128),
    (128, 65),
    (65, 65),
    (100, 127),
])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_attnmask_non_aligned_backward_hdim64(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    dtype: torch.dtype,
    causal: bool,
    device: str,
):
    """Test backward gradient correctness with non-64-aligned sequence lengths for headdim=64."""
    torch.manual_seed(42)

    headdim = 64
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device, requires_grad=True)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, device=device)
    dout = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=dtype, device=device)

    # Flash attention forward + backward (hdim=64)
    out_flash = flash_attn_with_mask_func(q, k, v, attn_mask, causal=causal)
    dq_flash, dk_flash, dv_flash = torch.autograd.grad(out_flash, (q, k, v), dout)

    # Reference forward + backward (fp32)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = attention_ref_with_mask(q_ref, k_ref, v_ref, attn_mask, causal=causal, upcast=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout)

    # PyTorch native precision forward + backward (fp16, reorder_ops=True)
    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt = attention_ref_with_mask(q_pt, k_pt, v_pt, attn_mask, causal=causal, upcast=False, reorder_ops=True)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_pt, k_pt, v_pt), dout)

    print(f"[hdim64 non-aligned seqlen_q={seqlen_q}, seqlen_k={seqlen_k}] "
          f"Output max diff: {(out_flash - out_ref).abs().max().item():.6f}")
    print(f"[hdim64 non-aligned seqlen_q={seqlen_q}, seqlen_k={seqlen_k}] "
          f"dQ max diff: {(dq_flash - dq_ref).abs().max().item():.6f}, "
          f"Pytorch: {(dq_pt - dq_ref).abs().max().item():.6f}")
    print(f"[hdim64 non-aligned seqlen_q={seqlen_q}, seqlen_k={seqlen_k}] "
          f"dK max diff: {(dk_flash - dk_ref).abs().max().item():.6f}, "
          f"Pytorch: {(dk_pt - dk_ref).abs().max().item():.6f}")
    print(f"[hdim64 non-aligned seqlen_q={seqlen_q}, seqlen_k={seqlen_k}] "
          f"dV max diff: {(dv_flash - dv_ref).abs().max().item():.6f}, "
          f"Pytorch: {(dv_pt - dv_ref).abs().max().item():.6f}")

    # 使用与对齐场景相同的官方风格数值约束，保留当前的误差以便你排查 kernel。
    assert (out_flash - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5
    assert (dq_flash - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item() + 1e-3
    assert (dk_flash - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item() + 1e-3
    assert (dv_flash - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item() + 1e-3


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (65, 128),
    (128, 65),
    (65, 65),
    (100, 127),
])
@pytest.mark.parametrize("mask_type", ["prefix", "random", "block_sparse"])
def test_attnmask_non_aligned_mask_patterns(
    dtype: torch.dtype, 
    seqlen_q: int,
    seqlen_k: int,
    mask_type: str, 
    device: str
):
    """Test different mask patterns with non-aligned sequence lengths."""
    batch_size, nheads = 2, 4

    torch.manual_seed(42)
    q = torch.randn(batch_size, seqlen_q, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device)

    if mask_type == "prefix":
        prefix_len = min(32, seqlen_k)
        attn_mask = generate_prefix_mask(batch_size, nheads, seqlen_q, seqlen_k, prefix_len, device)
    elif mask_type == "random":
        attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, 0.3, device)
    else:  # block_sparse
        attn_mask = generate_block_sparse_mask(batch_size, nheads, seqlen_q, seqlen_k, 32, device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    print(f"[{mask_type}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}] Output max diff: {(out - out_ref).abs().max().item():.6f}")

    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (65, 128),
    (128, 65),
    (65, 65),
    (100, 127),
])
@pytest.mark.parametrize("mask_type", ["prefix", "random", "block_sparse"])
def test_attnmask_non_aligned_mask_patterns_hdim64(
    dtype: torch.dtype,
    seqlen_q: int,
    seqlen_k: int,
    mask_type: str,
    device: str,
):
    """Test different mask patterns with non-aligned sequence lengths for headdim=64."""
    batch_size, nheads = 2, 4
    headdim = 64

    torch.manual_seed(42)
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device)

    if mask_type == "prefix":
        prefix_len = min(32, seqlen_k)
        attn_mask = generate_prefix_mask(batch_size, nheads, seqlen_q, seqlen_k, prefix_len, device)
    elif mask_type == "random":
        attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, 0.3, device)
    else:
        attn_mask = generate_block_sparse_mask(batch_size, nheads, seqlen_q, seqlen_k, 32, device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


# ============================================================================
# Edge case tests - 边界情况测试
# ============================================================================

@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (1, 1),         # 最小序列
    (1, 64),        # seqlen_q = 1
    (1, 100),       # seqlen_q = 1, seqlen_k 非对齐
    (16, 16),       # 小于 block size
    (16, 32),       # 小于 block size
    (32, 16),       # 小于 block size
])
def test_attnmask_small_seqlen(
    dtype: torch.dtype,
    seqlen_q: int,
    seqlen_k: int,
    device: str
):
    """Test with very small sequence lengths (edge cases)."""
    batch_size, nheads = 2, 4
    torch.manual_seed(42)

    q = torch.randn(batch_size, seqlen_q, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, 0.3, device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    print(f"[seqlen_q={seqlen_q}, seqlen_k={seqlen_k}] Output max diff: {(out - out_ref).abs().max().item():.6f}")

    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (1, 1),
    (1, 64),
    (1, 100),
    (16, 16),
    (16, 32),
    (32, 16),
])
def test_attnmask_small_seqlen_hdim64(
    dtype: torch.dtype,
    seqlen_q: int,
    seqlen_k: int,
    device: str,
):
    """Test with very small sequence lengths (edge cases) for headdim=64."""
    batch_size, nheads = 2, 4
    headdim = 64
    torch.manual_seed(42)

    q = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, 0.3, device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("seqlen", [63, 65, 127, 129, 191, 193, 255, 257])
def test_attnmask_boundary_seqlen(dtype: torch.dtype, seqlen: int, device: str):
    """Test sequence lengths at block boundaries (64-1, 64+1, 128-1, etc.)."""
    batch_size, nheads = 2, 4
    torch.manual_seed(42)

    q = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen, seqlen, 0.3, device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    print(f"[seqlen={seqlen}] Output max diff: {(out - out_ref).abs().max().item():.6f}")

    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("seqlen", [63, 65, 127, 129, 191, 193, 255, 257])
def test_attnmask_boundary_seqlen_hdim64(dtype: torch.dtype, seqlen: int, device: str):
    """Test sequence lengths at block boundaries for headdim=64."""
    batch_size, nheads = 2, 4
    headdim = 64
    torch.manual_seed(42)

    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen, seqlen, 0.3, device)

    out = flash_attn_with_mask_func(q, k, v, attn_mask)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, upcast=False, reorder_ops=True)

    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("seqlen", [63, 65, 100, 127])
def test_attnmask_boundary_backward(dtype: torch.dtype, seqlen: int, device: str):
    """Test backward at block boundaries with non-aligned sequence lengths."""
    batch_size, nheads = 2, 4
    torch.manual_seed(42)

    q = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen, seqlen, 0.3, device)
    dout = torch.randn(batch_size, seqlen, nheads, HEADDIM, dtype=dtype, device=device)

    out_flash = flash_attn_with_mask_func(q, k, v, attn_mask)
    dq_flash, dk_flash, dv_flash = torch.autograd.grad(out_flash, (q, k, v), dout)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = attention_ref_with_mask(q_ref, k_ref, v_ref, attn_mask, upcast=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout)

    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt = attention_ref_with_mask(q_pt, k_pt, v_pt, attn_mask, upcast=False, reorder_ops=True)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_pt, k_pt, v_pt), dout)

    print(f"[seqlen={seqlen}] dQ max diff: {(dq_flash - dq_ref).abs().max().item():.6f}")
    print(f"[seqlen={seqlen}] dK max diff: {(dk_flash - dk_ref).abs().max().item():.6f}")
    print(f"[seqlen={seqlen}] dV max diff: {(dv_flash - dv_ref).abs().max().item():.6f}")

    assert (dq_flash - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item() + 1e-3
    assert (dk_flash - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item() + 1e-3
    assert (dv_flash - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item() + 1e-3


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("seqlen", [63, 65, 100, 127])
def test_attnmask_boundary_backward_hdim64(dtype: torch.dtype, seqlen: int, device: str):
    """Test backward at block boundaries with non-aligned sequence lengths for headdim=64."""
    batch_size, nheads = 2, 4
    headdim = 64
    torch.manual_seed(42)

    q = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device, requires_grad=True)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen, seqlen, 0.3, device)
    dout = torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)

    out_flash = flash_attn_with_mask_func(q, k, v, attn_mask)
    dq_flash, dk_flash, dv_flash = torch.autograd.grad(out_flash, (q, k, v), dout)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = attention_ref_with_mask(q_ref, k_ref, v_ref, attn_mask, upcast=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout)

    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt = attention_ref_with_mask(q_pt, k_pt, v_pt, attn_mask, upcast=False, reorder_ops=True)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_pt, k_pt, v_pt), dout)

    assert (dq_flash - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item() + 1e-3
    assert (dk_flash - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item() + 1e-3
    assert (dv_flash - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item() + 1e-3


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("max_seqlen", [65, 100, 127, 200])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_varlen_non_aligned_forward_hdim64(
    batch_size: int,
    max_seqlen: int,
    nheads: int,
    dtype: torch.dtype,
    device: str,
):
    """Test variable-length forward with non-64-aligned max_seqlen for headdim=64."""
    torch.manual_seed(42)

    seqlens = torch.randint(max(1, max_seqlen // 4), max_seqlen + 1, (batch_size,))
    cu_seqlens = torch.cat([
        torch.tensor([0]),
        seqlens.cumsum(0)
    ]).to(torch.int32).to(device)

    total = cu_seqlens[-1].item()

    headdim = 64
    q = torch.randn(total, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(total, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(total, nheads, headdim, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, max_seqlen, max_seqlen, 0.3, device)

    out = flash_attn_varlen_with_mask_func(
        q, k, v, attn_mask,
        cu_seqlens, cu_seqlens,
        max_seqlen, max_seqlen,
    )

    assert out.shape == (total, nheads, headdim)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("max_seqlen", [65, 100, 127])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_varlen_non_aligned_backward_hdim64(
    batch_size: int,
    max_seqlen: int,
    nheads: int,
    dtype: torch.dtype,
    device: str,
):
    """Test variable-length backward with non-64-aligned max_seqlen for headdim=64."""
    torch.manual_seed(42)

    seqlens = torch.randint(max(1, max_seqlen // 2), max_seqlen + 1, (batch_size,))
    cu_seqlens = torch.cat([
        torch.tensor([0]),
        seqlens.cumsum(0)
    ]).to(torch.int32).to(device)

    total = cu_seqlens[-1].item()

    headdim = 64
    q = torch.randn(total, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(total, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(total, nheads, headdim, dtype=dtype, device=device, requires_grad=True)

    attn_mask = torch.ones(batch_size, nheads, max_seqlen, max_seqlen, dtype=torch.bool, device=device)

    out = flash_attn_varlen_with_mask_func(
        q, k, v, attn_mask,
        cu_seqlens, cu_seqlens,
        max_seqlen, max_seqlen,
    )

    dout = torch.randn_like(out)
    out.backward(dout)

    assert q.grad is not None and not torch.isnan(q.grad).any()
    assert k.grad is not None and not torch.isnan(k.grad).any()
    assert v.grad is not None and not torch.isnan(v.grad).any()


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (65, 130),
    (130, 65),
    (33, 47),
    (33, 97),
    (97, 33),
    (127, 255),
    (255, 127),
])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False])
def test_attnmask_asymmetric_non_aligned_hdim64(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    dtype: torch.dtype,
    causal: bool,
    device: str,
):
    """Test asymmetric Q/K lengths with non-64-aligned values for headdim=64."""
    torch.manual_seed(42)

    headdim = 64
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device, requires_grad=True)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, 0.3, device)
    dout = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=dtype, device=device)

    out_flash = flash_attn_with_mask_func(q, k, v, attn_mask, causal=causal)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, causal=causal, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, causal=causal, upcast=False, reorder_ops=True)

    assert (out_flash - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5

    dq_flash, dk_flash, dv_flash = torch.autograd.grad(out_flash, (q, k, v), dout)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = attention_ref_with_mask(q_ref, k_ref, v_ref, attn_mask, causal=causal, upcast=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout)

    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt = attention_ref_with_mask(q_pt, k_pt, v_pt, attn_mask, causal=causal, upcast=False, reorder_ops=True)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_pt, k_pt, v_pt), dout)

    assert (dq_flash - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item() + 1e-3
    assert (dk_flash - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item() + 1e-3
    assert (dv_flash - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item() + 1e-3


# ============================================================================
# Variable length with non-aligned max_seqlen tests
# ============================================================================

@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("max_seqlen", [65, 100, 127, 200])  # 非64倍数
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_varlen_non_aligned_forward(
    batch_size: int,
    max_seqlen: int,
    nheads: int,
    dtype: torch.dtype,
    device: str,
):
    """Test variable-length forward with non-64-aligned max_seqlen."""
    torch.manual_seed(42)

    # Generate random sequence lengths (some may be non-aligned)
    seqlens = torch.randint(max(1, max_seqlen // 4), max_seqlen + 1, (batch_size,))
    cu_seqlens = torch.cat([
        torch.tensor([0]),
        seqlens.cumsum(0)
    ]).to(torch.int32).to(device)

    total = cu_seqlens[-1].item()

    q = torch.randn(total, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(total, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(total, nheads, HEADDIM, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, max_seqlen, max_seqlen, 0.3, device)

    out = flash_attn_varlen_with_mask_func(
        q, k, v, attn_mask,
        cu_seqlens, cu_seqlens,
        max_seqlen, max_seqlen,
    )

    print(f"[varlen, max_seqlen={max_seqlen}] seqlens: {seqlens.tolist()}")
    assert out.shape == (total, nheads, HEADDIM)
    assert not torch.isnan(out).any(), "Output contains NaN values"


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("max_seqlen", [65, 100, 127])  # 非64倍数
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_attnmask_varlen_non_aligned_backward(
    batch_size: int,
    max_seqlen: int,
    nheads: int,
    dtype: torch.dtype,
    device: str,
):
    """Test variable-length backward with non-64-aligned max_seqlen."""
    torch.manual_seed(42)

    seqlens = torch.randint(max(1, max_seqlen // 2), max_seqlen + 1, (batch_size,))
    cu_seqlens = torch.cat([
        torch.tensor([0]),
        seqlens.cumsum(0)
    ]).to(torch.int32).to(device)

    total = cu_seqlens[-1].item()

    q = torch.randn(total, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(total, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(total, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)

    attn_mask = torch.ones(batch_size, nheads, max_seqlen, max_seqlen, dtype=torch.bool, device=device)

    out = flash_attn_varlen_with_mask_func(
        q, k, v, attn_mask,
        cu_seqlens, cu_seqlens,
        max_seqlen, max_seqlen,
    )

    dout = torch.randn_like(out)
    out.backward(dout)

    print(f"[varlen bwd, max_seqlen={max_seqlen}] seqlens: {seqlens.tolist()}")
    assert q.grad is not None and not torch.isnan(q.grad).any(), "dQ contains NaN"
    assert k.grad is not None and not torch.isnan(k.grad).any(), "dK contains NaN"
    assert v.grad is not None and not torch.isnan(v.grad).any(), "dV contains NaN"


# ============================================================================
# Asymmetric seqlen_q != seqlen_k with non-aligned lengths
# ============================================================================

@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (65, 130),      # Q短K长，Q非对齐
    (130, 65),      # Q长K短，K非对齐
    (33, 47),
    (33, 97),       # 两者非对齐，Q < K
    (97, 33),       # 两者非对齐，Q > K
    (127, 255),     # 边界值
    (255, 127),     # 边界值反向
])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False])  # causal requires seqlen_q == seqlen_k in some cases
def test_attnmask_asymmetric_non_aligned(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    dtype: torch.dtype,
    causal: bool,
    device: str,
):
    """Test asymmetric Q/K lengths with non-64-aligned values."""
    torch.manual_seed(42)

    q = torch.randn(batch_size, seqlen_q, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device, requires_grad=True)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, 0.3, device)
    dout = torch.randn(batch_size, seqlen_q, nheads, HEADDIM, dtype=dtype, device=device)

    # Forward
    out_flash = flash_attn_with_mask_func(q, k, v, attn_mask, causal=causal)
    out_ref = attention_ref_with_mask(q, k, v, attn_mask, causal=causal, upcast=True)
    out_pt = attention_ref_with_mask(q, k, v, attn_mask, causal=causal, upcast=False, reorder_ops=True)

    print(f"[asymmetric seqlen_q={seqlen_q}, seqlen_k={seqlen_k}] Output max diff: {(out_flash - out_ref).abs().max().item():.6f}")
    
    assert (out_flash - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item() + 1e-5

    # Backward
    dq_flash, dk_flash, dv_flash = torch.autograd.grad(out_flash, (q, k, v), dout)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = attention_ref_with_mask(q_ref, k_ref, v_ref, attn_mask, causal=causal, upcast=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), dout)

    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt = attention_ref_with_mask(q_pt, k_pt, v_pt, attn_mask, causal=causal, upcast=False, reorder_ops=True)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_pt, k_pt, v_pt), dout)

    print(f"[asymmetric] dQ max diff: {(dq_flash - dq_ref).abs().max().item():.6f}")
    print(f"[asymmetric] dK max diff: {(dk_flash - dk_ref).abs().max().item():.6f}")
    print(f"[asymmetric] dV max diff: {(dv_flash - dv_ref).abs().max().item():.6f}")

    assert (dq_flash - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item() + 1e-3
    assert (dk_flash - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item() + 1e-3
    assert (dv_flash - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item() + 1e-3


# ============================================================================
# LSE correctness with non-aligned lengths
# ============================================================================

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (65, 128),
    (128, 65),
    (65, 65),
    (100, 127),
])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_attnmask_lse_non_aligned(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    dtype: torch.dtype,
    causal: bool,
    device: str,
):
    """Test LSE correctness with non-aligned sequence lengths."""
    torch.manual_seed(42)

    q = torch.randn(batch_size, seqlen_q, nheads, HEADDIM, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen_k, nheads, HEADDIM, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, 0.3, device)

    # Flash attention with return_attn_probs to get LSE
    out_flash, lse_flash, _ = flash_attn_with_mask_func(
        q, k, v, attn_mask, causal=causal, return_attn_probs=True
    )

    # Reference LSE
    out_ref, lse_ref = attention_ref_with_mask(
        q, k, v, attn_mask, causal=causal, upcast=True, return_lse=True
    )

    # Check valid positions only (ignore -inf for fully masked rows)
    valid_mask = torch.isfinite(lse_ref)
    if valid_mask.any():
        lse_diff = (lse_flash[valid_mask] - lse_ref[valid_mask]).abs()
        lse_max_diff = lse_diff.max().item()
    else:
        lse_max_diff = 0.0

    print(f"[LSE, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}] LSE max diff: {lse_max_diff:.6f}")

    lse_atol = 1e-2
    assert lse_max_diff <= lse_atol, f"LSE max diff {lse_max_diff} > {lse_atol}"


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [
    (65, 128),
    (128, 65),
    (65, 65),
    (100, 127),
])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_attnmask_lse_non_aligned_hdim64(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    nheads: int,
    dtype: torch.dtype,
    causal: bool,
    device: str,
):
    """Test LSE correctness with non-aligned sequence lengths for headdim=64."""
    torch.manual_seed(42)

    headdim = 64
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=dtype, device=device)

    attn_mask = generate_random_mask(batch_size, nheads, seqlen_q, seqlen_k, 0.3, device)

    out_flash, lse_flash, _ = flash_attn_with_mask_func(
        q, k, v, attn_mask, causal=causal, return_attn_probs=True
    )

    out_ref, lse_ref = attention_ref_with_mask(
        q, k, v, attn_mask, causal=causal, upcast=True, return_lse=True
    )

    valid_mask = torch.isfinite(lse_ref)
    if valid_mask.any():
        lse_diff = (lse_flash[valid_mask] - lse_ref[valid_mask]).abs()
        lse_max_diff = lse_diff.max().item()
    else:
        lse_max_diff = 0.0

    print(f"[LSE hdim64, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}] LSE max diff: {lse_max_diff:.6f}")

    lse_atol = 1e-2
    assert lse_max_diff <= lse_atol, f"LSE max diff {lse_max_diff} > {lse_atol}"


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
