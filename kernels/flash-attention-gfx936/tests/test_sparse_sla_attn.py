import pytest
import torch
import torch.nn.functional as F
import argparse

import pdb

from flash_attn import (
    flash_attn_func,
    sparse_attn_func,
    sparse_attn_with_sla,
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

'''
def test_sparse_linear_attention_check():
    torch.manual_seed(42)

    batch, seqlen, heads, headdim = 1, 19440, 5, 128
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    sparsity = 0.7
    topk = 1.0 - sparsity

    out_sla = sparse_attn_with_sla(
        q,
        k,
        v,
        topk=topk,
        feature_map="softmax",
        return_sparsity=False,
    )
'''

def test_sparse_linear_attention_check(dtype=torch.bfloat16, headdim=128):
    torch.manual_seed(42)

    batch, seqlen, heads = 1, 19440, 5
    # batch, seqlen, heads = 1, 256, 2
    q = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, seqlen, heads, headdim, device=DEVICE, dtype=DTYPE)

    sparsity = 0.3
    topk = 1.0 - sparsity

    out_sla = sparse_attn_with_sla(
        q,
        k,
        v,
        topk=topk,
        feature_map="softmax",
        use_bf16 = (True if dtype==torch.bfloat16 else False),
        use_fp8= (True if dtype==torch.float8_e4m3fn else False),
        return_sparsity=False,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp8"], default="bf16", help="Data type to use for testing (bf16, fp16 or fp8)")
    parser.add_argument("--dim", type=int, choices=[64, 128], default=128, help="Dim to use for testing (64, 128)")
    parser.add_argument('--prof', default=False, action='store_true', help='prof or not')

    args = parser.parse_args()

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch_dtype
    print("dtype:", torch_dtype)
    test_sparse_linear_attention_check(torch_dtype, args.dim)
    print("Test passed.")
