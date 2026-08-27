import torch
from flash_attn import flash_attn_func,flash_attn_with_kvcache,flash_attn_varlen_func
import math
import torch.nn.functional as F
import os
import pytest
from einops import rearrange, repeat


def native_multi_head_attention_2(q, k, v, mask=None, mask_type=None, upcast=True, reorder_ops=False):
    original_device = q.device
    original_dtype  = q.dtype
    d = q.size(-1)
    groups = q.size(1) // k.size(1)
    if groups != 1:
        k = torch.repeat_interleave(k, repeats=groups, dim=1)
        v = torch.repeat_interleave(v, repeats=groups, dim=1)
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    if not reorder_ops:
        q = q / math.sqrt(d)
    else:
        k = k / math.sqrt(d)
    k1 = k.transpose(-2, -1)
    qkt = torch.matmul(q, k1)
    qkt = qkt.type(torch.float32)
    if mask_type == 0 and mask is not None:
        qkt.masked_fill_(mask, -float('inf'))  # Apply the mask
    qkt_max = qkt.max(dim=-1)[0].unsqueeze(-1)
    qkt_exp = torch.exp((qkt - qkt_max))

    qkt_sum = qkt_exp.sum(-1).unsqueeze(-1)
    qkt_softmax = qkt_exp / qkt_sum
    # qkt_softmax = qkt_softmax.type(original_dtype)
    v = v.float()
    # print("sum: {:.12f} | max: {:.12f}".format(qkt_sum.item(), qkt_max.item()))
    pv = torch.matmul(qkt_softmax, v)
    return pv.to(original_device).to(original_dtype)

    
def _generate_block_kvcache(seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype):
    num_blocks = math.ceil(seqlen_k / paged_kv_block_size) * batch_size * 3
    k_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    k_cache = rearrange(
        # pytorch 1.12 doesn't have indexing with int32
        k_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    k_cache = k_cache.permute(0, 2, 1, 3).contiguous()
    v_cache = v_cache.permute(0, 2, 1, 3).contiguous()
    k_cache_paged = k_cache_paged.permute(0, 2, 1, 3).contiguous()
    v_cache_paged = v_cache_paged.permute(0, 2, 1, 3).contiguous()
    return k_cache, v_cache, block_table, k_cache_paged, v_cache_paged, num_blocks

def get_partition(batch_size, seq_q_len, max_seqlen_k, nheads_q, nheads_k, head_size, input_dtype, input_device, device_cu=100):
    # 计算一下划分大小和划分策略
    partition_size = 0
    scores_raw     = None
    tmp_output     = None
    threshold      = device_cu * 0.75
    n_group        = int(nheads_q / nheads_k)
    use_regroup = all(n_group % it != 0 for it in [16, 8, 4, 2, 9, 7, 5, 3])
    if (use_regroup): n_group = 1
    if ((batch_size * seq_q_len * n_group < threshold and max_seqlen_k >= 1024) or (max_seqlen_k >= 8192)):
        # 根据最大的 seqKV 长度, 决定相应的划分 size
        if (max_seqlen_k <= 1024): partition_size = 128
        elif (max_seqlen_k <= 2048): partition_size = 256
        elif (max_seqlen_k <= 32768): partition_size = 512
        else: partition_size = 1024
        if (nheads_q == nheads_k): partition_size = 1024
        while ((nheads_q > nheads_k) and (batch_size * seq_q_len * n_group * (max_seqlen_k / partition_size)) < threshold):
            # 目前支持的最小 partition size 是 128
            if (partition_size < 256): break
            partition_size = int(partition_size / 2)
        num_splits = math.ceil(max_seqlen_k * 1.0 / partition_size)
        scores_raw = torch.empty(
            size=(2, num_splits, batch_size, nheads_q),
            dtype=torch.float32,
            device=input_device
        )
        tmp_output = torch.empty(
            size=(num_splits, batch_size, nheads_q, head_size),
            dtype=input_dtype,
            device=input_device
        )
    return partition_size, scores_raw, tmp_output


os.environ['USE_FA_CUDA_BWD'] = '1' #设置使用我们的hip版的fa_bwd
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize("nheads,nheads_k", [
    (16,16),
    (32,32),
    (32,4),
    (52,4),
    (16,2),
    (26,2),
    (8,1),
    (13,1)
])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize('causal', [False])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
@pytest.mark.parametrize('d', [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_kv",
    [
        (128, 128),
        (1024, 1024),
        (2048, 2048),
        # (8192, 8192),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_kv', [(128, 128)])
# @pytest.mark.parametrize("dropout_p", [0.0, 0.17])
# @pytest.mark.parametrize('dropout_p', [0.0])
def test_flash_attn_output(
    seqlen_q, seqlen_kv, nheads, nheads_k, d, causal, dtype
):
    if (
        max(seqlen_q, seqlen_kv) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 1
    assert nheads % nheads_k == 0
    q = torch.randn(
        batch_size, nheads, seqlen_q, d, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        nheads_k,
        seqlen_kv,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        nheads_k,
        seqlen_kv,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    q_flash = q.detach().clone().requires_grad_(True)
    k_flash = k.detach().clone().requires_grad_(True)
    v_flash = v.detach().clone().requires_grad_(True)
    out_flash, lse, S_dmask = flash_attn_func(
        q_flash, k_flash, v_flash, return_attn_probs=True, causal=causal
    )
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    mask = torch.ones(q.size(-2), k.size(-2), dtype=torch.bool, device=q.device).tril().logical_not() if causal else None
    mask_type = 0 if causal else None
    out_ref = native_multi_head_attention_2(q_ref, k_ref, v_ref, mask, mask_type)
    # out_ref,_ = attention_ref(q_ref, k_ref, v_ref)
    # out_pt,_ = attention_ref(q_pt, k_pt, v_pt, upcast=False,reorder_ops=True)
    out_pt = native_multi_head_attention_2(q_pt, k_pt, v_pt, mask, mask_type, upcast=False,reorder_ops=True)

    print(f"Output max diff: {(out_flash - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out_flash - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    # if dropout_p > 0.0:
    #     print(f'Attention max diff: {(attn - attn_ref).abs().max().item()}')
    #     print(f'Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}')

    dO = torch.randn(batch_size, nheads, seqlen_q, d).to(dtype).to(device)
    out_flash.backward(dO)
    out_ref.backward(dO)
    out_pt.backward(dO)

    print(f"dQ max diff: {(q_flash.grad - q_ref.grad).abs().max().item()}")
    print(f"dK max diff: {(k_flash.grad - k_ref.grad).abs().max().item()}")
    print(f"dV max diff: {(v_flash.grad - v_ref.grad).abs().max().item()}")
    print(f"dQ mean diff: {(q_flash.grad - q_ref.grad).abs().mean().item()}")
    print(f"dK mean diff: {(k_flash.grad - k_ref.grad).abs().mean().item()}")
    print(f"dV mean diff: {(v_flash.grad - v_ref.grad).abs().mean().item()}")
    print(f"dQ Pytorch max diff: {(q_pt.grad - q_ref.grad).abs().max().item()}")
    print(f"dK Pytorch max diff: {(k_pt.grad - k_ref.grad).abs().max().item()}")
    print(f"dV Pytorch max diff: {(v_pt.grad - v_ref.grad).abs().max().item()}")
    print(f"dQ Pytorch mean diff: {(q_pt.grad - q_ref.grad).abs().mean().item()}")
    print(f"dK Pytorch mean diff: {(k_pt.grad - k_ref.grad).abs().mean().item()}")
    print(f"dV Pytorch mean diff: {(v_pt.grad - v_ref.grad).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out_flash - out_ref).abs().max().item() <= 2 * (
        out_pt - out_ref
    ).abs().max().item()


    assert (q_flash.grad - q_ref.grad).abs().max().item() <= 3 * (
        q_pt.grad - q_ref.grad
    ).abs().max().item()
    assert (k_flash.grad - k_ref.grad).abs().max().item() <= 3 * (
        k_pt.grad - k_ref.grad
    ).abs().max().item()
    assert (v_flash.grad - v_ref.grad).abs().max().item() <= 3 * (
        v_pt.grad - v_ref.grad
    ).abs().max().item()

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("num_splits", [1])
# @pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("alibi", [False])
# @pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("paged_kv_block_size", [128])
# @pytest.mark.parametrize("paged_kv_block_size", [256, 512])
# @pytest.mark.parametrize("paged_kv_block_size", [None])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("nheads,nheads_k", [
    (16,16),
    (32,32),
    (32,4),
    (52,4),
    (16,2),
    (26,2),
    (8,1),
    (13,1)
])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1024),
        (1, 339),
        (1, 128),
        (1, 8192),
        (1, 8192*2)
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
def test_flash_attn_kvcache(
    seqlen_q,
    seqlen_k,
    nheads,
    nheads_k,
    d,
    paged_kv_block_size,
    causal,
    local,
    alibi,
    dtype,
):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 1

    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q = torch.randn(batch_size, nheads, seqlen_q, d, device=device, dtype=dtype)
    if paged_kv_block_size is None:
        k_cache = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        v_cache = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        block_table = None
    else:
        (
            k_cache,
            v_cache,
            block_table,
            k_cache_paged,
            v_cache_paged,
            num_blocks,
        ) = _generate_block_kvcache(
            seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype
        )

    # if alibi:
    #     alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
    #     attn_bias = attn_bias_from_alibi_slopes(
    #         alibi_slopes, seqlen_q, seqlen_k, None, key_padding_mask, causal=causal, key_leftpad=cache_leftpad
    #     )
    # else:
    #     alibi_slopes, attn_bias = None, None
    cu_seq_lens_q = torch.ones(batch_size * seqlen_q, dtype=torch.int32).to("cuda")
    cu_seq_lens_k = (torch.ones(batch_size * seqlen_q, dtype=torch.int32, device=device) * seqlen_k)

    # k_cache[:, 64:] = -1
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    partition_size, scores_raw, tmp_output = get_partition(batch_size, seqlen_q, cu_seq_lens_k.max().item(), nheads, nheads_k, d, dtype, device, device_cu=100)
    out = flash_attn_with_kvcache(
        q,
        k_cache if paged_kv_block_size is None else k_cache_paged,
        v_cache if paged_kv_block_size is None else v_cache_paged,
        None,
        None,
        rotary_cos=None,
        rotary_sin=None,
        cu_seqlens_q=cu_seq_lens_q,
        cache_seqlens=cu_seq_lens_k,
        cache_batch_idx=None,
        cache_leftpad=None,
        block_table=block_table,
        causal=causal,
        window_size=window_size,
        alibi_slopes=None,
        num_splits=partition_size,
        scores_raw=scores_raw,
        tmp_output=tmp_output
    )

    out_ref = native_multi_head_attention_2(q, k_cache_ref, v_cache_ref)
    out_pt = native_multi_head_attention_2(q, k_cache_ref, v_cache_ref, upcast=False,reorder_ops=True)
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Output mean rel diff: {(out/out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    mult = 3 if not alibi else 5
    assert (out - out_ref).abs().max().item() <= mult * (out_pt - out_ref).abs().max().item() + 1e-5

@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192, 224, 256])
@pytest.mark.parametrize('d', [128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_kv",
    [
        (1024, 1024),
        (128, 128),
        (339, 339),
    ],
)
@pytest.mark.parametrize("nheads,nheads_k", [
    (16,16),
    (32,32),
    (32,4),
    (52,4),
    (16,2),
    (26,2),
    (8,1),
    (13,1)
])
def test_flash_attn_varlen_output(
    seqlen_q, seqlen_kv, d, nheads, nheads_k, causal, dtype
):
    if (
        max(seqlen_q, seqlen_kv) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 8
    nheads_k = 8
    assert nheads % nheads_k == 0
    q = torch.randn(batch_size, nheads, seqlen_q, d, device=device,dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, nheads_k, seqlen_kv, d, device=device,dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, nheads_k, seqlen_kv, d, device=device,dtype=dtype, requires_grad=True)

    q_fa = q.view(batch_size * nheads * seqlen_q, d)
    k_fa = k.view(batch_size * nheads_k * seqlen_q, d)
    v_fa = v.view(batch_size * nheads_k * seqlen_q, d)
    cu_seqlens_q = torch.arange(0, seqlen_q*(batch_size+1), seqlen_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.arange(0, seqlen_kv*(batch_size+1), seqlen_kv, dtype=torch.int32, device=device)


    out, sm_lse, S_dmask = flash_attn_varlen_func(
        q_fa,
        k_fa,
        v_fa,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlen_q,
        seqlen_kv,
        0.0,
        return_attn_probs=True,
        causal=causal,
    )
    # out = output_pad_fn(out_unpad)
    split_sizes = [cu_seqlens_q[i+1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q) - 1)]
    out_split = torch.split(out, [i*nheads for i in split_sizes], dim=0)
    o_tmp = out_split[0].view(nheads, -1, d)
    for i in range(1, len(out_split)):
        o_tmp= torch.cat((o_tmp, out_split[i].view(nheads, -1, d)), dim=0)
    
    out_fa = o_tmp.view(batch_size, nheads, seqlen_q, d)
    mask = torch.ones(q.size(-2), k.size(-2), dtype=torch.bool, device=q.device).tril().logical_not() if causal else None
    mask_type = 0 if causal else None
    out_ref = native_multi_head_attention_2(q, k, v, mask, mask_type)
    out_pt = native_multi_head_attention_2(q, k, v, mask, mask_type, upcast=False, reorder_ops=True)

    print(f"Output max diff: {(out_fa - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out_fa - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    # if dropout_p > 0.0:
    #     print(f'Attention max diff: {(attn - attn_ref).abs().max().item()}')
    #     print(f'Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}')

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out_fa - out_ref).abs().max().item() <= 2 * (
        out_pt - out_ref
    ).abs().max().item()