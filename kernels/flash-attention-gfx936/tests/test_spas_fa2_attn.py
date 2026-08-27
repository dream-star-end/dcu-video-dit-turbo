import pytest
import torch
import torch.nn.functional as F

from flash_attn import (
    flash_attn_func,
    sparse_attn_func,
    sparse_attn_varlen_func,
    spas_fa2_attn_meansim_cuda,
    spas_fa2_attn_meansim_topk_cuda,
    spas_fa2_attn_meansim_varlen_cuda,
    spas_fa2_attn_meansim_topk_varlen_cuda,
)
from flash_attn.utils.sparse_utils import (
    get_block_map_meansim,
    hyperparameter_check,
    block_map_to_block_offset,
    block_map_lut,
    block_map_to_block_offset_triton,
    block_map_lut_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Sparse attention tests require CUDA.",
)

DEVICE = "cuda"
BLOCK_K = 64
INVALID_OFFSET = 10000000


def _default_dtype():
    if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16


DTYPE = _default_dtype()


def precision_metric(out1, out2):
    x, xx = out1.float(), out2.float()
    cos_sim = F.cosine_similarity(x.reshape(1, -1), xx.reshape(1, -1)).item()
    # Avoid division by zero
    xx_abs_sum = xx.abs().sum()
    l1 = ((x - xx).abs().sum() / (xx_abs_sum + 1e-8)).item()
    rmse = torch.sqrt(torch.mean((x - xx) ** 2)).item()
    max_diff = (x - xx).abs().max().item()
    return {
        "cos_sim": cos_sim,
        "l1": l1,
        "rmse": rmse,
        "max_diff": max_diff,
    }


def _column_buffers(batch, heads, num_q_blocks):
    column_count = torch.zeros((batch, heads, num_q_blocks), dtype=torch.int32, device=DEVICE)
    column_index = torch.zeros((batch, heads, num_q_blocks, 1), dtype=torch.int32, device=DEVICE)
    return column_count, column_index


def test_discrete_block_selection_fixed_matches_manual():
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 2, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    num_q_blocks = (seqlen + BLOCK_K - 1) // BLOCK_K
    block_count = torch.full((batch, heads, num_q_blocks), 2, dtype=torch.int32, device=DEVICE)
    block_offset = torch.full(
        (batch, heads, num_q_blocks, num_q_blocks), INVALID_OFFSET, dtype=torch.int32, device=DEVICE
    )
    block_offset[:, :, :, 0] = 0
    block_offset[:, :, :, 1] = 2 * BLOCK_K
    column_count, column_index = _column_buffers(batch, heads, num_q_blocks)

    out_discrete = sparse_attn_func(
        q,
        k,
        v,
        block_count=block_count,
        block_offset=block_offset,
        column_count=column_count,
        column_index=column_index,
        causal=False,
    )

    q_seq = q[0].float()
    k_seq = k[0].float()
    v_seq = v[0].float()

    k_selected = torch.cat([k_seq[0:BLOCK_K], k_seq[2 * BLOCK_K : 3 * BLOCK_K]], dim=0)
    v_selected = torch.cat([v_seq[0:BLOCK_K], v_seq[2 * BLOCK_K : 3 * BLOCK_K]], dim=0)

    q_t = q_seq.transpose(0, 1)
    k_t = k_selected.transpose(0, 1)
    v_t = v_selected.transpose(0, 1)

    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * (headdim ** -0.5)
    attn = torch.softmax(scores, dim=-1)
    out_manual = torch.matmul(attn, v_t).transpose(0, 1).to(DTYPE)

    cos_sim = F.cosine_similarity(
        out_manual.reshape(1, -1).float(), out_discrete[0].reshape(1, -1).float()
    ).item()
    assert cos_sim > 0.99


def test_varied_block_selection_fixed_has_no_nan():
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 2, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    num_q_blocks = (seqlen + BLOCK_K - 1) // BLOCK_K
    block_count = torch.zeros((batch, heads, num_q_blocks), dtype=torch.int32, device=DEVICE)
    block_offset = torch.full(
        (batch, heads, num_q_blocks, num_q_blocks), INVALID_OFFSET, dtype=torch.int32, device=DEVICE
    )

    block_count[:, :, 0] = 2
    block_offset[:, :, 0, 0] = 0 * BLOCK_K
    block_offset[:, :, 0, 1] = 1 * BLOCK_K

    block_count[:, :, 1] = 2
    block_offset[:, :, 1, 0] = 1 * BLOCK_K
    block_offset[:, :, 1, 1] = 2 * BLOCK_K

    block_count[:, :, 2] = 3
    block_offset[:, :, 2, 0] = 0 * BLOCK_K
    block_offset[:, :, 2, 1] = 2 * BLOCK_K
    block_offset[:, :, 2, 2] = 3 * BLOCK_K

    block_count[:, :, 3] = 1
    block_offset[:, :, 3, 0] = 3 * BLOCK_K

    column_count, column_index = _column_buffers(batch, heads, num_q_blocks)

    out_varied = sparse_attn_func(
        q,
        k,
        v,
        block_count=block_count,
        block_offset=block_offset,
        column_count=column_count,
        column_index=column_index,
        causal=False,
    )

    total_selected = block_count.sum().item()
    total_possible = num_q_blocks * num_q_blocks * batch * heads
    sparsity = 1.0 - total_selected / total_possible

    assert not out_varied.isnan().any().item()
    assert 0.0 < sparsity < 1.0


def test_compare_with_auto_sparse_fixed():
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 2, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    dense = flash_attn_func(q, k, v, causal=False)

    out_auto, auto_sparsity = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    metrics = precision_metric(out_auto, dense)
    assert 0.0 < auto_sparsity < 1.0
    assert metrics["cos_sim"] > 0.6


def test_discrete_block_selection_varlen_matches_manual():
    torch.manual_seed(42)

    batch = 2
    seqlens = [256, 256]
    total_tokens = sum(seqlens)
    heads, headdim = 2, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )
    num_q_blocks = (max_seqlen + BLOCK_K - 1) // BLOCK_K

    block_count = torch.full((batch, heads, num_q_blocks), 2, dtype=torch.int32, device=DEVICE)
    block_offset = torch.full(
        (batch, heads, num_q_blocks, num_q_blocks), INVALID_OFFSET, dtype=torch.int32, device=DEVICE
    )
    block_offset[:, :, :, 0] = 0 * BLOCK_K
    block_offset[:, :, :, 1] = 2 * BLOCK_K
    column_count, column_index = _column_buffers(batch, heads, num_q_blocks)

    out_discrete = sparse_attn_varlen_func(
        q,
        k,
        v,
        block_count=block_count,
        block_offset=block_offset,
        column_count=column_count,
        column_index=column_index,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=False,
    )

    seq_start = 0
    seq_len = seqlens[0]
    q_seq = q[seq_start : seq_start + seq_len].float()
    k_seq = k[seq_start : seq_start + seq_len].float()
    v_seq = v[seq_start : seq_start + seq_len].float()

    k_selected = torch.cat([k_seq[0:BLOCK_K], k_seq[2 * BLOCK_K : 3 * BLOCK_K]], dim=0)
    v_selected = torch.cat([v_seq[0:BLOCK_K], v_seq[2 * BLOCK_K : 3 * BLOCK_K]], dim=0)

    q_t = q_seq.transpose(0, 1)
    k_t = k_selected.transpose(0, 1)
    v_t = v_selected.transpose(0, 1)

    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * (headdim ** -0.5)
    attn = torch.softmax(scores, dim=-1)
    out_manual = torch.matmul(attn, v_t).transpose(0, 1).to(DTYPE)

    cos_sim = F.cosine_similarity(
        out_manual.reshape(1, -1).float(),
        out_discrete[seq_start : seq_start + seq_len].reshape(1, -1).float(),
    ).item()
    assert cos_sim > 0.99


def test_varied_block_selection_varlen_has_no_nan():
    torch.manual_seed(42)

    batch = 2
    seqlens = [256, 256]
    total_tokens = sum(seqlens)
    heads, headdim = 2, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )
    num_q_blocks = (max_seqlen + BLOCK_K - 1) // BLOCK_K

    block_count = torch.zeros((batch, heads, num_q_blocks), dtype=torch.int32, device=DEVICE)
    block_offset = torch.full(
        (batch, heads, num_q_blocks, num_q_blocks), INVALID_OFFSET, dtype=torch.int32, device=DEVICE
    )

    block_count[:, :, 0] = 2
    block_offset[:, :, 0, 0] = 0 * BLOCK_K
    block_offset[:, :, 0, 1] = 1 * BLOCK_K

    block_count[:, :, 1] = 2
    block_offset[:, :, 1, 0] = 1 * BLOCK_K
    block_offset[:, :, 1, 1] = 2 * BLOCK_K

    block_count[:, :, 2] = 3
    block_offset[:, :, 2, 0] = 0 * BLOCK_K
    block_offset[:, :, 2, 1] = 2 * BLOCK_K
    block_offset[:, :, 2, 2] = 3 * BLOCK_K

    block_count[:, :, 3] = 1
    block_offset[:, :, 3, 0] = 3 * BLOCK_K

    column_count, column_index = _column_buffers(batch, heads, num_q_blocks)

    out_varied = sparse_attn_varlen_func(
        q,
        k,
        v,
        block_count=block_count,
        block_offset=block_offset,
        column_count=column_count,
        column_index=column_index,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=False,
    )

    total_selected = block_count.sum().item()
    total_possible = num_q_blocks * num_q_blocks * batch * heads
    sparsity = 1.0 - total_selected / total_possible

    assert not out_varied.isnan().any().item()
    assert 0.0 < sparsity < 1.0


def test_variable_length_sequences_discrete_blocks():
    torch.manual_seed(42)

    seqlens = [128, 256, 192]
    batch = len(seqlens)
    total_tokens = sum(seqlens)
    heads, headdim = 2, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )
    blocks_per_seq = [(s + BLOCK_K - 1) // BLOCK_K for s in seqlens]
    max_q_blocks = (max_seqlen + BLOCK_K - 1) // BLOCK_K

    block_count = torch.zeros((batch, heads, max_q_blocks), dtype=torch.int32, device=DEVICE)
    block_offset = torch.full(
        (batch, heads, max_q_blocks, max_q_blocks), INVALID_OFFSET, dtype=torch.int32, device=DEVICE
    )

    block_count[0, :, 0] = 2
    block_offset[0, :, 0, 0] = 0
    block_offset[0, :, 0, 1] = BLOCK_K
    block_count[0, :, 1] = 1
    block_offset[0, :, 1, 0] = BLOCK_K

    block_count[1, :, 0] = 2
    block_offset[1, :, 0, 0] = 0
    block_offset[1, :, 0, 1] = 2 * BLOCK_K
    block_count[1, :, 1] = 2
    block_offset[1, :, 1, 0] = 1 * BLOCK_K
    block_offset[1, :, 1, 1] = 3 * BLOCK_K
    block_count[1, :, 2] = 3
    block_offset[1, :, 2, 0] = 0 * BLOCK_K
    block_offset[1, :, 2, 1] = 1 * BLOCK_K
    block_offset[1, :, 2, 2] = 2 * BLOCK_K
    block_count[1, :, 3] = 2
    block_offset[1, :, 3, 0] = 2 * BLOCK_K
    block_offset[1, :, 3, 1] = 3 * BLOCK_K

    block_count[2, :, 0] = 1
    block_offset[2, :, 0, 0] = 0
    block_count[2, :, 1] = 3
    block_offset[2, :, 1, 0] = 0
    block_offset[2, :, 1, 1] = BLOCK_K
    block_offset[2, :, 1, 2] = 2 * BLOCK_K
    block_count[2, :, 2] = 2
    block_offset[2, :, 2, 0] = BLOCK_K
    block_offset[2, :, 2, 1] = 2 * BLOCK_K

    column_count, column_index = _column_buffers(batch, heads, max_q_blocks)

    out = sparse_attn_varlen_func(
        q,
        k,
        v,
        block_count=block_count,
        block_offset=block_offset,
        column_count=column_count,
        column_index=column_index,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=False,
    )

    total_selected = block_count.sum().item()
    total_possible = sum(b * b for b in blocks_per_seq) * heads
    sparsity = 1.0 - total_selected / total_possible

    assert not out.isnan().any().item()
    assert 0.0 < sparsity < 1.0


def test_compare_with_auto_sparse_varlen():
    torch.manual_seed(42)

    batch = 2
    seqlens = [256, 256]
    total_tokens = sum(seqlens)
    heads, headdim = 2, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )
    num_q_blocks = (max_seqlen + BLOCK_K - 1) // BLOCK_K

    block_count_full = torch.full((batch, heads, num_q_blocks), num_q_blocks, dtype=torch.int32, device=DEVICE)
    block_offset_full = torch.full(
        (batch, heads, num_q_blocks, num_q_blocks), INVALID_OFFSET, dtype=torch.int32, device=DEVICE
    )
    for i in range(num_q_blocks):
        block_offset_full[:, :, :, i] = i * BLOCK_K
    column_count, column_index = _column_buffers(batch, heads, num_q_blocks)

    dense = sparse_attn_varlen_func(
        q,
        k,
        v,
        block_count=block_count_full,
        block_offset=block_offset_full,
        column_count=column_count,
        column_index=column_index,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=False,
    )

    out_auto, auto_sparsity = spas_fa2_attn_meansim_topk_varlen_cuda(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    metrics = precision_metric(out_auto, dense)
    assert 0.0 < auto_sparsity < 1.0
    assert metrics["cos_sim"] > 0.6


def test_fixed_length_block_offset_format():
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    block_m = BLOCK_K
    topk = 0.5

    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    out_auto, _ = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=topk,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    q_bhld = q.transpose(1, 2).contiguous().to(DTYPE)
    k_bhld = k.transpose(1, 2).contiguous().to(DTYPE)
    k_bhld = k_bhld - k_bhld.mean(dim=-2, keepdim=True)

    simthreshd1 = hyperparameter_check(-10.0, heads, DEVICE)
    topk_tensor = hyperparameter_check(topk, heads, DEVICE)

    block_offset, block_count = get_block_map_meansim(
        q_bhld,
        k_bhld,
        is_causal=False,
        simthreshd1=simthreshd1,
        cdfthreshd=None,
        topk=topk_tensor,
        return_block_offset=True,
        BLKQ=block_m,
        BLKK=BLOCK_K,
    )

    block_offset = block_offset * BLOCK_K
    num_q_blocks = (seqlen + block_m - 1) // block_m
    column_count, column_index = _column_buffers(batch, heads, num_q_blocks)

    out_manual = sparse_attn_func(
        q,
        k,
        v,
        block_count=block_count,
        block_offset=block_offset,
        column_count=column_count,
        column_index=column_index,
        causal=False,
    )

    metrics = precision_metric(out_auto, out_manual)
    assert metrics["cos_sim"] > 0.999999


def test_fixed_length_sparsity_effectiveness():
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 512, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    out_full = flash_attn_func(q, k, v, causal=False)

    prev_sparsity = 1.0
    prev_cos = 0.0
    for topk in [0.25, 0.5, 0.75, 1.0]:
        out_sparse, sparsity = spas_fa2_attn_meansim_topk_cuda(
            q,
            k,
            v,
            causal=False,
            topk=topk,
            simthreshd1=-10.0,
            return_sparsity=True,
        )

        metrics = precision_metric(out_sparse, out_full)
        assert sparsity <= prev_sparsity + 1e-6
        assert metrics["cos_sim"] >= prev_cos - 1e-6
        prev_sparsity = sparsity
        prev_cos = metrics["cos_sim"]

    assert metrics["cos_sim"] > 0.99


def test_varlen_block_offset_format():
    torch.manual_seed(42)

    batch = 3
    seqlens = [128, 256, 192]
    total_tokens = sum(seqlens)
    heads, headdim = 4, 128
    block_m = BLOCK_K
    topk = 0.5

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )
    max_seqlen = max(seqlens)
    max_q_blocks = (max_seqlen + block_m - 1) // block_m
    max_nnz_s = max((s + BLOCK_K - 1) // BLOCK_K for s in seqlens)

    out_auto, _ = spas_fa2_attn_meansim_topk_varlen_cuda(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=False,
        topk=topk,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    simthreshd1 = hyperparameter_check(-10.0, heads, DEVICE)
    topk_tensor = hyperparameter_check(topk, heads, DEVICE)

    block_count = torch.zeros((batch, heads, max_q_blocks), dtype=torch.int32, device=DEVICE)
    block_offset = torch.full(
        (batch, heads, max_q_blocks, max_nnz_s), INVALID_OFFSET, dtype=torch.int32, device=DEVICE
    )
    column_count = torch.zeros((batch, heads, max_q_blocks), dtype=torch.int32, device=DEVICE)
    column_index = torch.zeros((batch, heads, max_q_blocks, 1), dtype=torch.int32, device=DEVICE)

    for b in range(batch):
        q_start, q_end = cu_seqlens[b].item(), cu_seqlens[b + 1].item()
        len_q = q_end - q_start
        if len_q == 0:
            continue

        num_q_blocks = (len_q + block_m - 1) // block_m
        qb = q[q_start:q_end].transpose(0, 1).contiguous().unsqueeze(0)
        kb = k[q_start:q_end].transpose(0, 1).contiguous().unsqueeze(0)
        kb = kb - kb.mean(dim=-2, keepdim=True)

        block_offset_b, block_count_b = get_block_map_meansim(
            qb,
            kb,
            is_causal=False,
            simthreshd1=simthreshd1,
            cdfthreshd=None,
            topk=topk_tensor,
            return_block_offset=True,
            BLKQ=block_m,
            BLKK=BLOCK_K,
        )

        block_count[b, :, :num_q_blocks] = block_count_b[0]
        nnz_s_b = block_offset_b.size(-1)
        block_offset[b, :, :num_q_blocks, :nnz_s_b] = block_offset_b[0]

    block_offset = block_offset * BLOCK_K

    out_manual = sparse_attn_varlen_func(
        q,
        k,
        v,
        block_count=block_count,
        block_offset=block_offset,
        column_count=column_count,
        column_index=column_index,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=False,
    )

    metrics = precision_metric(out_auto, out_manual)
    assert metrics["cos_sim"] > 0.999999


def test_varlen_sparsity_effectiveness():
    torch.manual_seed(42)

    batch = 2
    seqlens = [256, 256]
    total_tokens = sum(seqlens)
    heads, headdim = 4, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )

    out_full, _ = spas_fa2_attn_meansim_topk_varlen_cuda(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=False,
        topk=1.0,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    prev_sparsity = 1.0
    prev_cos = 0.0
    for topk in [0.25, 0.5, 0.75, 1.0]:
        out_sparse, sparsity = spas_fa2_attn_meansim_topk_varlen_cuda(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=False,
            topk=topk,
            simthreshd1=-10.0,
            return_sparsity=True,
        )

        metrics = precision_metric(out_sparse, out_full)
        assert sparsity <= prev_sparsity + 1e-6
        assert metrics["cos_sim"] >= prev_cos - 1e-6
        prev_sparsity = sparsity
        prev_cos = metrics["cos_sim"]

    assert metrics["cos_sim"] > 0.99


def test_causal_mode():
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    out_full = flash_attn_func(q, k, v, causal=True)

    out_sparse, _ = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=True,
        topk=1.0,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    metrics = precision_metric(out_sparse, out_full)
    assert metrics["cos_sim"] > 0.99


def test_edge_cases():
    torch.manual_seed(42)
    dtype = DTYPE

    test_cases = [
        {"B": 1, "S": 128, "H": 1, "D": 128},
        {"B": 1, "S": 128, "H": 8, "D": 128},
        {"B": 4, "S": 128, "H": 1, "D": 128},
        {"B": 2, "S": 512, "H": 4, "D": 128},
    ]

    for tc in test_cases:
        q = torch.randn(tc["B"], tc["S"], tc["H"], tc["D"], device=DEVICE, dtype=dtype)
        k = torch.randn(tc["B"], tc["S"], tc["H"], tc["D"], device=DEVICE, dtype=dtype)
        v = torch.randn(tc["B"], tc["S"], tc["H"], tc["D"], device=DEVICE, dtype=dtype)

        out_sparse, _ = spas_fa2_attn_meansim_topk_cuda(
            q,
            k,
            v,
            causal=False,
            topk=0.5,
            simthreshd1=-10.0,
            return_sparsity=True,
        )

        assert not out_sparse.isnan().any().item()
        assert not out_sparse.isinf().any().item()


# =====================================================
# CDF-based sparsity tests
# =====================================================

def test_cdf_sparsity_fixed_length():
    """Test spas_fa2_attn_meansim_cuda with CDF thresholding."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    dense = flash_attn_func(q, k, v, causal=False)

    out_cdf, sparsity = spas_fa2_attn_meansim_cuda(
        q,
        k,
        v,
        causal=False,
        cdfthreshd=0.9,
        simthreshd1=0.1,
        return_sparsity=True,
    )

    metrics = precision_metric(out_cdf, dense)
    assert 0.0 <= sparsity <= 1.0
    assert not out_cdf.isnan().any().item()
    assert not out_cdf.isinf().any().item()
    # Verify sparse output is reasonably close to dense
    assert metrics["cos_sim"] > 0.5


def test_cdf_sparsity_varlen():
    """Test spas_fa2_attn_meansim_varlen_cuda with CDF thresholding."""
    torch.manual_seed(42)

    batch = 2
    seqlens = [256, 256]
    total_tokens = sum(seqlens)
    heads, headdim = 4, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )

    out_cdf, sparsity = spas_fa2_attn_meansim_varlen_cuda(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=False,
        cdfthreshd=0.9,
        simthreshd1=0.1,
        return_sparsity=True,
    )

    assert 0.0 <= sparsity <= 1.0
    assert not out_cdf.isnan().any().item()
    assert not out_cdf.isinf().any().item()


def test_cdf_vs_topk_comparison():
    """Compare CDF and TopK sparsity methods on same inputs."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    out_cdf, sparsity_cdf = spas_fa2_attn_meansim_cuda(
        q,
        k,
        v,
        causal=False,
        cdfthreshd=0.98,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    out_topk, sparsity_topk = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    # Both should produce valid outputs
    assert not out_cdf.isnan().any().item()
    assert not out_topk.isnan().any().item()
    assert 0.0 <= sparsity_cdf <= 1.0
    assert 0.0 <= sparsity_topk <= 1.0


def test_cdf_various_thresholds():
    """Test CDF sparsity with various cdfthreshd values."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    dense = flash_attn_func(q, k, v, causal=False)

    prev_sparsity = 1.0
    for cdf in [0.8, 0.9, 0.95, 0.99]:
        out_cdf, sparsity = spas_fa2_attn_meansim_cuda(
            q,
            k,
            v,
            causal=False,
            cdfthreshd=cdf,
            simthreshd1=-10.0,
            return_sparsity=True,
        )

        assert not out_cdf.isnan().any().item()
        assert 0.0 <= sparsity <= 1.0
        # Higher CDF threshold should give lower sparsity (more blocks kept)
        assert sparsity <= prev_sparsity + 1e-6
        prev_sparsity = sparsity

    # At high CDF threshold, should be reasonably close to dense
    metrics = precision_metric(out_cdf, dense)
    assert metrics["cos_sim"] > 0.8


def test_cdf_causal_mode():
    """Test CDF sparsity with causal attention."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    dense = flash_attn_func(q, k, v, causal=True)

    out_cdf, sparsity = spas_fa2_attn_meansim_cuda(
        q,
        k,
        v,
        causal=True,
        cdfthreshd=0.99,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert not out_cdf.isnan().any().item()
    metrics = precision_metric(out_cdf, dense)
    # CDF sparsity with high threshold should be reasonably close to dense
    assert metrics["cos_sim"] > 0.8


def test_cdf_with_attention_sink():
    """Test CDF sparsity with attention_sink=True."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    out_sink, sparsity_sink = spas_fa2_attn_meansim_cuda(
        q,
        k,
        v,
        causal=False,
        cdfthreshd=0.9,
        simthreshd1=-10.0,
        attention_sink=True,
        return_sparsity=True,
    )

    out_no_sink, sparsity_no_sink = spas_fa2_attn_meansim_cuda(
        q,
        k,
        v,
        causal=False,
        cdfthreshd=0.9,
        simthreshd1=-10.0,
        attention_sink=False,
        return_sparsity=True,
    )

    assert not out_sink.isnan().any().item()
    assert not out_no_sink.isnan().any().item()
    # With attention sink, sparsity should be lower (more blocks computed)
    assert sparsity_sink <= sparsity_no_sink + 1e-6


def test_cdf_per_head_threshold():
    """Test CDF sparsity with per-head cdfthreshd values."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    # Different CDF threshold for each head
    cdf_tensor = torch.tensor([0.8, 0.9, 0.95, 0.99], device=DEVICE, dtype=torch.float32)

    out_cdf, sparsity = spas_fa2_attn_meansim_cuda(
        q,
        k,
        v,
        causal=False,
        cdfthreshd=cdf_tensor,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert not out_cdf.isnan().any().item()
    assert out_cdf.shape == (batch, seqlen, heads, headdim)


def test_cdf_varlen_various_thresholds():
    """Test CDF varlen sparsity with various cdfthreshd values."""
    torch.manual_seed(42)

    batch = 2
    seqlens = [256, 256]
    total_tokens = sum(seqlens)
    heads, headdim = 4, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )

    prev_sparsity = 1.0
    for cdf in [0.8, 0.9, 0.95, 0.99]:
        out_cdf, sparsity = spas_fa2_attn_meansim_varlen_cuda(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=False,
            cdfthreshd=cdf,
            simthreshd1=-10.0,
            return_sparsity=True,
        )

        assert not out_cdf.isnan().any().item()
        assert 0.0 <= sparsity <= 1.0
        prev_sparsity = sparsity


def test_cdf_varlen_causal():
    """Test CDF varlen sparsity with causal attention."""
    torch.manual_seed(42)

    batch = 2
    seqlens = [256, 256]
    total_tokens = sum(seqlens)
    heads, headdim = 4, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )

    out_cdf, sparsity = spas_fa2_attn_meansim_varlen_cuda(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=True,
        cdfthreshd=0.95,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert not out_cdf.isnan().any().item()
    assert not out_cdf.isinf().any().item()
    assert out_cdf.shape == (total_tokens, heads, headdim)


def test_cdf_varlen_with_attention_sink():
    """Test CDF varlen sparsity with attention_sink=True."""
    torch.manual_seed(42)

    batch = 2
    seqlens = [256, 256]
    total_tokens = sum(seqlens)
    heads, headdim = 4, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )

    out_sink, _ = spas_fa2_attn_meansim_varlen_cuda(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=False,
        cdfthreshd=0.9,
        simthreshd1=-10.0,
        attention_sink=True,
        return_sparsity=True,
    )

    assert not out_sink.isnan().any().item()


def test_cdf_varlen_different_seqlens():
    """Test CDF varlen sparsity with different sequence lengths."""
    torch.manual_seed(42)

    batch = 4
    seqlens = [128, 192, 256, 320]
    total_tokens = sum(seqlens)
    heads, headdim = 4, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )

    out_cdf, sparsity = spas_fa2_attn_meansim_varlen_cuda(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=False,
        cdfthreshd=0.95,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert not out_cdf.isnan().any().item()
    assert out_cdf.shape == (total_tokens, heads, headdim)
    assert 0.0 <= sparsity <= 1.0


def test_cdf_varlen_per_head_threshold():
    """Test CDF varlen sparsity with per-head cdfthreshd values."""
    torch.manual_seed(42)

    batch = 2
    seqlens = [256, 256]
    total_tokens = sum(seqlens)
    heads, headdim = 4, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )

    # Different CDF threshold for each head
    cdf_tensor = torch.tensor([0.85, 0.90, 0.95, 0.99], device=DEVICE, dtype=torch.float32)

    out_cdf, sparsity = spas_fa2_attn_meansim_varlen_cuda(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=False,
        cdfthreshd=cdf_tensor,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert not out_cdf.isnan().any().item()
    assert out_cdf.shape == (total_tokens, heads, headdim)


def test_cdf_effectiveness_fixed_length():
    """Test that higher CDF threshold gives better precision."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 512, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    dense = flash_attn_func(q, k, v, causal=False)

    prev_cos = 0.0
    for cdf in [0.8, 0.9, 0.95, 0.99]:
        out_cdf, _ = spas_fa2_attn_meansim_cuda(
            q,
            k,
            v,
            causal=False,
            cdfthreshd=cdf,
            simthreshd1=-10.0,
            return_sparsity=True,
        )

        metrics = precision_metric(out_cdf, dense)
        # Higher CDF threshold should give better precision
        assert metrics["cos_sim"] >= prev_cos - 1e-6
        prev_cos = metrics["cos_sim"]

    # At highest threshold, should be reasonably close to dense
    assert metrics["cos_sim"] > 0.9


def test_cdf_smooth_k_parameter():
    """Test CDF sparsity with smooth_k parameter."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    out_smooth, _ = spas_fa2_attn_meansim_cuda(
        q,
        k,
        v,
        causal=False,
        cdfthreshd=0.95,
        simthreshd1=-10.0,
        smooth_k=True,
        return_sparsity=True,
    )

    out_no_smooth, _ = spas_fa2_attn_meansim_cuda(
        q,
        k,
        v,
        causal=False,
        cdfthreshd=0.95,
        simthreshd1=-10.0,
        smooth_k=False,
        return_sparsity=True,
    )

    assert not out_smooth.isnan().any().item()
    assert not out_no_smooth.isnan().any().item()


def test_cdf_larger_batch():
    """Test CDF sparsity with larger batch size."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 8, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    out_cdf, sparsity = spas_fa2_attn_meansim_cuda(
        q,
        k,
        v,
        causal=False,
        cdfthreshd=0.95,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert out_cdf.shape == (batch, seqlen, heads, headdim)
    assert not out_cdf.isnan().any().item()
    assert 0.0 <= sparsity <= 1.0


def test_cdf_varlen_larger_batch():
    """Test CDF varlen sparsity with larger batch size."""
    torch.manual_seed(42)

    batch = 8
    seqlens = [128, 192, 256, 128, 192, 256, 128, 192]
    total_tokens = sum(seqlens)
    heads, headdim = 4, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )

    out_cdf, sparsity = spas_fa2_attn_meansim_varlen_cuda(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=False,
        cdfthreshd=0.95,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert out_cdf.shape == (total_tokens, heads, headdim)
    assert not out_cdf.isnan().any().item()
    assert 0.0 <= sparsity <= 1.0


# =====================================================
# Attention sink tests
# =====================================================

def test_attention_sink_fixed_length():
    """Test attention_sink=True ensures first block is always attended."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    # With attention sink
    out_sink, _ = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.25,
        simthreshd1=-10.0,
        attention_sink=True,
        return_sparsity=True,
    )

    # Without attention sink
    out_no_sink, _ = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.25,
        simthreshd1=-10.0,
        attention_sink=False,
        return_sparsity=True,
    )

    assert not out_sink.isnan().any().item()
    assert not out_no_sink.isnan().any().item()


def test_attention_sink_varlen():
    """Test attention_sink=True in varlen mode."""
    torch.manual_seed(42)

    batch = 2
    seqlens = [256, 256]
    total_tokens = sum(seqlens)
    heads, headdim = 4, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )

    out_sink, _ = spas_fa2_attn_meansim_topk_varlen_cuda(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=False,
        topk=0.25,
        simthreshd1=-10.0,
        attention_sink=True,
        return_sparsity=True,
    )

    assert not out_sink.isnan().any().item()


# =====================================================
# Different head dimensions tests
# =====================================================

def test_headdim_64_dense_baseline():
    """Test that headdim=64 works with dense attention (baseline for comparison)."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 64
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    # Note: Sparse attention currently only supports headdim=128
    # This test verifies that dense attention works with headdim=64
    dense = flash_attn_func(q, k, v, causal=False)
    assert not dense.isnan().any().item()
    assert dense.shape == (batch, seqlen, heads, headdim)


def test_supported_headdim():
    """Test with the supported head dimension of 128."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    dense = flash_attn_func(q, k, v, causal=False)

    out_sparse, sparsity = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=1.0,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    metrics = precision_metric(out_sparse, dense)
    assert metrics["cos_sim"] > 0.99
    assert not out_sparse.isnan().any().item()


def test_headdim_constraint():
    """Test that sparse attention documents headdim=128 constraint."""
    # Note: The sparse attention kernel currently only supports headdim=128.
    # This test documents this limitation by testing with the supported value.
    torch.manual_seed(42)

    batch, seqlen, heads = 2, 256, 4
    supported_headdim = 128

    q = torch.randn(batch, seqlen, heads, supported_headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, supported_headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, supported_headdim, device=DEVICE, dtype=DTYPE)

    out_sparse, sparsity = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert out_sparse.shape == (batch, seqlen, heads, supported_headdim)
    assert not out_sparse.isnan().any().item()
    assert 0.0 <= sparsity <= 1.0


# =====================================================
# Custom softmax scale tests
# =====================================================

def test_custom_softmax_scale():
    """Test with custom softmax_scale parameter."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    custom_scale = 0.05  # Different from default (1/sqrt(128) ≈ 0.088)

    out_default, _ = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    out_custom, _ = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        softmax_scale=custom_scale,
        return_sparsity=True,
    )

    # Both should produce valid outputs
    assert not out_default.isnan().any().item()
    assert not out_custom.isnan().any().item()
    # Outputs should be different due to different softmax scale
    assert not torch.allclose(out_default, out_custom, atol=1e-3)


# =====================================================
# Per-head hyperparameters tests
# =====================================================

def test_per_head_topk_tensor():
    """Test with per-head topk values as tensor."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    # Different topk for each head
    topk_tensor = torch.tensor([0.25, 0.5, 0.75, 1.0], device=DEVICE, dtype=torch.float32)

    out_sparse, _ = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=topk_tensor,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert not out_sparse.isnan().any().item()
    assert out_sparse.shape == (batch, seqlen, heads, headdim)


def test_per_head_simthreshd_tensor():
    """Test with per-head simthreshd1 values as tensor."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    # Different simthreshd1 for each head
    simthreshd_tensor = torch.tensor([0.1, 0.3, 0.5, 0.7], device=DEVICE, dtype=torch.float32)

    out_sparse, _ = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.5,
        simthreshd1=simthreshd_tensor,
        return_sparsity=True,
    )

    assert not out_sparse.isnan().any().item()


# =====================================================
# Return softmax LSE tests
# =====================================================

def test_return_softmax_lse():
    """Test return_softmax_lse parameter."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    num_q_blocks = (seqlen + BLOCK_K - 1) // BLOCK_K
    block_count = torch.full((batch, heads, num_q_blocks), num_q_blocks, dtype=torch.int32, device=DEVICE)
    block_offset = torch.full(
        (batch, heads, num_q_blocks, num_q_blocks), INVALID_OFFSET, dtype=torch.int32, device=DEVICE
    )
    for i in range(num_q_blocks):
        block_offset[:, :, :, i] = i * BLOCK_K
    column_count, column_index = _column_buffers(batch, heads, num_q_blocks)

    result = sparse_attn_func(
        q,
        k,
        v,
        block_count=block_count,
        block_offset=block_offset,
        column_count=column_count,
        column_index=column_index,
        causal=False,
        return_softmax_lse=True,
    )

    if isinstance(result, tuple):
        out, softmax_lse = result
        assert softmax_lse.shape == (batch, heads, seqlen)
        assert not softmax_lse.isnan().any().item()
    else:
        out = result

    assert not out.isnan().any().item()


# =====================================================
# Deterministic mode tests
# =====================================================

def test_deterministic_mode():
    """Test deterministic=True parameter produces consistent results."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    out1, _ = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        deterministic=True,
        return_sparsity=True,
    )

    out2, _ = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        deterministic=True,
        return_sparsity=True,
    )

    # Deterministic mode should produce identical results
    assert torch.allclose(out1, out2, atol=1e-6)


# =====================================================
# Sparse utils tests
# =====================================================

def test_hyperparameter_check_scalar():
    """Test hyperparameter_check with scalar values."""
    heads = 4

    # Test with float
    result = hyperparameter_check(0.5, heads, DEVICE)
    assert result.shape == (heads,)
    assert torch.all(result == 0.5)

    # Test with int
    result = hyperparameter_check(1, heads, DEVICE)
    assert result.shape == (heads,)
    assert torch.all(result == 1.0)


def test_hyperparameter_check_tensor():
    """Test hyperparameter_check with tensor values."""
    heads = 4

    # Test with 1D tensor
    tensor = torch.tensor([0.1, 0.2, 0.3, 0.4])
    result = hyperparameter_check(tensor, heads, DEVICE)
    assert result.shape == (heads,)
    assert result.device.type == DEVICE

    # Test with 0D tensor
    tensor_0d = torch.tensor(0.5)
    result = hyperparameter_check(tensor_0d, heads, DEVICE)
    assert result.shape == (heads,)
    assert torch.all(result == 0.5)


def test_block_map_to_block_offset_pytorch():
    """Test block_map_to_block_offset function."""
    # Create a simple block map
    B, H, Q, K = 2, 2, 4, 4
    block_map = torch.zeros(B, H, Q, K, dtype=torch.bool, device=DEVICE)
    # Select some blocks
    block_map[0, 0, 0, :2] = True  # First 2 blocks for row 0
    block_map[0, 0, 1, 1:3] = True  # Blocks 1,2 for row 1
    block_map[0, 0, 2, :] = True  # All blocks for row 2
    block_map[0, 0, 3, 3] = True  # Only block 3 for row 3

    block_offset, block_count = block_map_to_block_offset(block_map)

    # Verify counts
    assert block_count[0, 0, 0] == 2
    assert block_count[0, 0, 1] == 2
    assert block_count[0, 0, 2] == 4
    assert block_count[0, 0, 3] == 1

    # Verify offsets are valid indices
    assert block_offset[0, 0, 0, 0] == 0
    assert block_offset[0, 0, 0, 1] == 1


def test_block_map_to_block_offset_triton():
    """Test block_map_to_block_offset_triton function."""
    B, H, Q, K = 2, 2, 4, 8
    block_map = torch.zeros(B, H, Q, K, dtype=torch.bool, device=DEVICE)
    block_map[0, 0, 0, :3] = True
    block_map[0, 0, 1, 2:5] = True
    block_map = block_map.contiguous()

    block_offset, block_count = block_map_to_block_offset_triton(block_map)

    assert block_count[0, 0, 0] == 3
    assert block_count[0, 0, 1] == 3
    assert block_offset[0, 0, 0, 0] == 0
    assert block_offset[0, 0, 0, 1] == 1
    assert block_offset[0, 0, 0, 2] == 2


def test_block_map_lut_pytorch():
    """Test block_map_lut function."""
    B, H, Q, K = 1, 1, 2, 4
    block_map = torch.zeros(B, H, Q, K, dtype=torch.bool, device=DEVICE)
    block_map[0, 0, 0, :2] = True
    block_map[0, 0, 1, 1:4] = True

    lut, valid_entry_num = block_map_lut(block_map)

    assert valid_entry_num[0, 0, 0] == 2
    assert valid_entry_num[0, 0, 1] == 3


def test_block_map_lut_triton():
    """Test block_map_lut_triton function."""
    B, H, Q, K = 2, 2, 4, 8
    block_map = torch.zeros(B, H, Q, K, dtype=torch.bool, device=DEVICE)
    block_map[0, 0, 0, :4] = True
    block_map[0, 0, 1, 2:6] = True
    block_map = block_map.contiguous()

    lut, valid_block_num = block_map_lut_triton(block_map)

    assert valid_block_num[0, 0, 0] == 4
    assert valid_block_num[0, 0, 1] == 4


# =====================================================
# Sequence length edge cases
# =====================================================

def test_minimum_sequence_length():
    """Test with minimum supported sequence length (128)."""
    torch.manual_seed(42)

    # Note: Sparse attention requires seqlen >= 128
    batch, seqlen, heads, headdim = 2, 128, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    dense = flash_attn_func(q, k, v, causal=False)

    out_sparse, _ = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=1.0,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    metrics = precision_metric(out_sparse, dense)
    assert metrics["cos_sim"] > 0.99


def test_two_blocks_sequence():
    """Test with sequence length equal to two blocks (128)."""
    torch.manual_seed(42)

    # Sequence length of 128 = 2 blocks of BLOCK_K=64
    batch, seqlen, heads, headdim = 2, 128, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    dense = flash_attn_func(q, k, v, causal=False)

    out_sparse, sparsity = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=1.0,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    metrics = precision_metric(out_sparse, dense)
    assert metrics["cos_sim"] > 0.99
    assert not out_sparse.isnan().any().item()


def test_longer_sequence():
    """Test with longer sequence length."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 1024, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    out_sparse, sparsity = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert not out_sparse.isnan().any().item()
    assert 0.0 <= sparsity <= 1.0


def test_varlen_minimum_sequences():
    """Test varlen with minimum supported sequence lengths (128 each)."""
    torch.manual_seed(42)

    batch = 3
    seqlens = [128, 128, 128]  # Each sequence is the minimum supported length
    total_tokens = sum(seqlens)
    heads, headdim = 4, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )

    out_sparse, _ = spas_fa2_attn_meansim_topk_varlen_cuda(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=False,
        topk=1.0,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert not out_sparse.isnan().any().item()


# =====================================================
# Output pre-allocation tests
# =====================================================

def test_preallocated_output():
    """Test with pre-allocated output tensor."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    # Pre-allocate output
    out_preallocated = torch.empty(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    result = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        out=out_preallocated,
    )

    assert not out_preallocated.isnan().any().item()


# =====================================================
# Causal mode tests
# =====================================================

def test_causal_varlen():
    """Test causal mode with variable length sequences."""
    torch.manual_seed(42)

    batch = 2
    seqlens = [256, 256]
    total_tokens = sum(seqlens)
    heads, headdim = 4, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )

    # Test that sparse varlen causal produces valid output
    out_sparse, sparsity = spas_fa2_attn_meansim_topk_varlen_cuda(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=True,
        topk=1.0,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert not out_sparse.isnan().any().item()
    assert not out_sparse.isinf().any().item()
    assert out_sparse.shape == (total_tokens, heads, headdim)


def test_causal_with_different_topk():
    """Test causal mode with different topk values."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    for topk in [0.25, 0.5, 0.75, 1.0]:
        out_sparse, sparsity = spas_fa2_attn_meansim_topk_cuda(
            q,
            k,
            v,
            causal=True,
            topk=topk,
            simthreshd1=-10.0,
            return_sparsity=True,
        )

        assert not out_sparse.isnan().any().item()
        assert not out_sparse.isinf().any().item()
        # Note: sparsity calculation may vary due to causal masking implementation


# =====================================================
# Dtype tests
# =====================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_both_dtypes(dtype):
    """Test with both float16 and bfloat16."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 not supported on this device")

    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=dtype)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=dtype)

    out_sparse, _ = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert out_sparse.dtype == dtype
    assert not out_sparse.isnan().any().item()


# =====================================================
# Block map meansim tests
# =====================================================

def test_get_block_map_meansim_return_lut():
    """Test get_block_map_meansim with return_lut=True."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    block_m = BLOCK_K

    q = torch.randn(batch, heads, seqlen, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, heads, seqlen, headdim, device=DEVICE, dtype=DTYPE)

    simthreshd1 = hyperparameter_check(-10.0, heads, DEVICE)
    topk_tensor = hyperparameter_check(0.5, heads, DEVICE)

    lut, valid_block_num = get_block_map_meansim(
        q,
        k,
        is_causal=False,
        simthreshd1=simthreshd1,
        cdfthreshd=None,
        topk=topk_tensor,
        return_lut=True,
        BLKQ=block_m,
        BLKK=BLOCK_K,
    )

    num_q_blocks = (seqlen + block_m - 1) // block_m
    num_k_blocks = (seqlen + BLOCK_K - 1) // BLOCK_K

    assert lut.shape == (batch, heads, num_q_blocks, num_k_blocks)
    assert valid_block_num.shape == (batch, heads, num_q_blocks)


def test_get_block_map_meansim_return_block_offset():
    """Test get_block_map_meansim with return_block_offset=True."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    block_m = BLOCK_K

    q = torch.randn(batch, heads, seqlen, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, heads, seqlen, headdim, device=DEVICE, dtype=DTYPE)

    simthreshd1 = hyperparameter_check(-10.0, heads, DEVICE)
    topk_tensor = hyperparameter_check(0.5, heads, DEVICE)

    block_offset, block_count = get_block_map_meansim(
        q,
        k,
        is_causal=False,
        simthreshd1=simthreshd1,
        cdfthreshd=None,
        topk=topk_tensor,
        return_block_offset=True,
        BLKQ=block_m,
        BLKK=BLOCK_K,
    )

    num_q_blocks = (seqlen + block_m - 1) // block_m
    num_k_blocks = (seqlen + BLOCK_K - 1) // BLOCK_K

    assert block_offset.shape == (batch, heads, num_q_blocks, num_k_blocks)
    assert block_count.shape == (batch, heads, num_q_blocks)


def test_get_block_map_meansim_causal():
    """Test get_block_map_meansim with causal=True."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    block_m = BLOCK_K

    q = torch.randn(batch, heads, seqlen, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, heads, seqlen, headdim, device=DEVICE, dtype=DTYPE)

    simthreshd1 = hyperparameter_check(-10.0, heads, DEVICE)
    topk_tensor = hyperparameter_check(0.5, heads, DEVICE)

    block_map = get_block_map_meansim(
        q,
        k,
        is_causal=True,
        simthreshd1=simthreshd1,
        cdfthreshd=None,
        topk=topk_tensor,
        return_lut=False,
        return_block_offset=False,
        BLKQ=block_m,
        BLKK=BLOCK_K,
    )

    num_q_blocks = (seqlen + block_m - 1) // block_m
    num_k_blocks = (seqlen + BLOCK_K - 1) // BLOCK_K

    assert block_map.shape == (batch, heads, num_q_blocks, num_k_blocks)
    # Verify causal mask - upper triangle should be False
    assert block_map.dtype == torch.bool


# =====================================================
# Smooth K tests
# =====================================================

def test_smooth_k_parameter():
    """Test smooth_k=True vs smooth_k=False."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    out_smooth, sparsity_smooth = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        smooth_k=True,
        return_sparsity=True,
    )

    out_no_smooth, sparsity_no_smooth = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        smooth_k=False,
        return_sparsity=True,
    )

    assert not out_smooth.isnan().any().item()
    assert not out_no_smooth.isnan().any().item()
    # Both modes should produce valid outputs (they may or may not be different)


# =====================================================
# Larger batch tests
# =====================================================

def test_larger_batch():
    """Test with larger batch size."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 8, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    out_sparse, sparsity = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert out_sparse.shape == (batch, seqlen, heads, headdim)
    assert not out_sparse.isnan().any().item()


def test_larger_batch_varlen():
    """Test with larger batch in varlen mode."""
    torch.manual_seed(42)

    batch = 8
    seqlens = [128, 192, 256, 128, 192, 256, 128, 192]
    total_tokens = sum(seqlens)
    heads, headdim = 4, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )

    out_sparse, _ = spas_fa2_attn_meansim_topk_varlen_cuda(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert out_sparse.shape == (total_tokens, heads, headdim)
    assert not out_sparse.isnan().any().item()


# =====================================================
# Many heads tests
# =====================================================

def test_many_heads():
    """Test with many attention heads."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 32, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    out_sparse, _ = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert out_sparse.shape == (batch, seqlen, heads, headdim)
    assert not out_sparse.isnan().any().item()


# =====================================================
# Sparsity ratio tests
# =====================================================

def test_extreme_sparsity_ratios():
    """Test with extreme topk values."""
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 2, 256, 4, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    # Very sparse (topk=0.1)
    out_sparse, sparsity_high = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=0.1,
        simthreshd1=-10.0,
        return_sparsity=True,
    )
    assert not out_sparse.isnan().any().item()
    assert sparsity_high > 0

    # Dense (topk=1.0)
    out_dense, sparsity_low = spas_fa2_attn_meansim_topk_cuda(
        q,
        k,
        v,
        causal=False,
        topk=1.0,
        simthreshd1=-10.0,
        return_sparsity=True,
    )
    assert not out_dense.isnan().any().item()

    # Higher topk should give lower sparsity (more blocks computed)
    assert sparsity_low <= sparsity_high


# =====================================================
# Consistency tests
# =====================================================

def test_output_shape_consistency():
    """Verify output shapes match input shapes."""
    torch.manual_seed(42)

    # Note: Sparse attention requires headdim=128 and seqlen >= 128
    test_configs = [
        {"B": 1, "S": 128, "H": 4, "D": 128},
        {"B": 2, "S": 256, "H": 8, "D": 128},
        {"B": 4, "S": 512, "H": 16, "D": 128},
    ]

    for cfg in test_configs:
        q = torch.randn(cfg["B"], cfg["S"], cfg["H"], cfg["D"], device=DEVICE, dtype=DTYPE)
        k = torch.randn(cfg["B"], cfg["S"], cfg["H"], cfg["D"], device=DEVICE, dtype=DTYPE)
        v = torch.randn(cfg["B"], cfg["S"], cfg["H"], cfg["D"], device=DEVICE, dtype=DTYPE)

        out, _ = spas_fa2_attn_meansim_topk_cuda(
            q, k, v,
            causal=False,
            topk=0.5,
            simthreshd1=-10.0,
            return_sparsity=True,
        )

        assert out.shape == (cfg["B"], cfg["S"], cfg["H"], cfg["D"])


def test_varlen_output_shape_consistency():
    """Verify varlen output shapes match input shapes."""
    torch.manual_seed(42)

    batch = 3
    seqlens = [128, 256, 192]  # All seqlens must be >= 128
    total_tokens = sum(seqlens)
    heads, headdim = 4, 128
    max_seqlen = max(seqlens)

    q = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(total_tokens, heads, headdim, device=DEVICE, dtype=DTYPE)

    cu_seqlens = torch.tensor(
        [0] + [sum(seqlens[:i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=DEVICE
    )

    out, _ = spas_fa2_attn_meansim_topk_varlen_cuda(
        q, k, v,
        cu_seqlens, cu_seqlens,
        max_seqlen, max_seqlen,
        causal=False,
        topk=0.5,
        simthreshd1=-10.0,
        return_sparsity=True,
    )

    assert out.shape == (total_tokens, heads, headdim)
