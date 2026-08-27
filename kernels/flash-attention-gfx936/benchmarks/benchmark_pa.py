import math
import time
import pytest
import torch
import random
import torch.nn.functional as F
import csv
from einops import rearrange, repeat

# from flash_attn import flash_attn_with_kvcache as _flash_attn_with_kvcache
from flash_attn import vllm_flash_attn_with_kvcache as _flash_attn_with_kvcache

max_seqlen=8192*5
# max_seqlen=4352
eager=True
# eager=False

def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)





def test_flash_attn_kvcache(
    seqlen_q,
    seqlen_k,
    d,
    has_batch_idx,
    has_leftpad,
    paged_kv_block_size,
    rotary_fraction,
    rotary_interleaved,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    alibi,
    new_kv,
    dtype,
    batch_size,
    qhead,
    kv_head,
    prof=False,
):
    # if seqlen_q > seqlen_k and new_kv:
    #     pytest.skip()
    # if not new_kv and rotary_fraction > 0.0:
    #     pytest.skip()
    # if has_batch_idx and paged_kv_block_size is not None:
    #     pytest.skip()
    # if has_leftpad and paged_kv_block_size is not None:
    #     pytest.skip()
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    # batch_size = 64
    # nheads = 32
    batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    # rotary_dim must be a multiple of 16, and must be <= d
    rotary_dim = math.floor(int(rotary_fraction * d) / 16) * 16

    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))

    q = torch.randn(batch_size, seqlen_q, qhead, d, device=device, dtype=dtype)
    seqlen_new = seqlen_q if seqlen_new_eq_seqlen_q else torch.randint(1, seqlen_q + 1, (1,)).item()
    nheads_k = kv_head
    # alloc k v
    if new_kv:
        k = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen_new, nheads_k, d, device=device, dtype=dtype)
    else:
        k, v = None, None
    # 生成kvcache
    if paged_kv_block_size is None:
        k_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
        v_cache = torch.randn(batch_size_cache, seqlen_k, nheads_k, d, device=device, dtype=dtype)
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

    seq_lens = [seqlen_k for _ in range(batch_size)]
    cache_seqlens = torch.tensor(seq_lens, dtype=torch.int, device=device)
    if has_leftpad:
        cache_leftpad = torch.cat([torch.randint(0, cache_seqlens[i].item(), (1,), dtype=torch.int32, device=device)
                                   if cache_seqlens[i].item() > 0 else torch.zeros(1, dtype=torch.int32, device=device)
                                   for i in range(batch_size)])
    else:
        cache_leftpad = None
    
    arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    key_padding_mask = arange < cache_seqlens_expanded + (seqlen_new if new_kv else 0)
    if has_leftpad:
        key_padding_mask = torch.logical_and(
            key_padding_mask, arange >= cache_leftpad.unsqueeze(-1).expand(-1, seqlen_k)
        )
    if has_batch_idx:
        cache_batch_idx = torch.randperm(batch_size_cache, dtype=torch.int32, device=device)[
            :batch_size
        ]
    else:
        cache_batch_idx = None
    alibi_slopes, attn_bias = None, None
    # cache_seqlens = torch.tensor([64], dtype=torch.int32, device=device)
    cos, sin = None, None
    q_ro, k_ro = q, k
    # k_cache[:, 64:] = -1
    k_cache_ref = (
        k_cache if not has_batch_idx else k_cache[cache_batch_idx.to(dtype=torch.long)]
    ).clone()
    v_cache_ref = (
        v_cache if not has_batch_idx else v_cache[cache_batch_idx.to(dtype=torch.long)]
    ).clone()
    if new_kv:
        update_mask = torch.logical_and(
            cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + seqlen_new
        )
        k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
        v_cache_ref[update_mask] = rearrange(v, "b s ... -> (b s) ...")
    # k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    # v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=nheads_k // nheads_k)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=nheads_k // nheads_k)
    q_scale = torch.tensor([0.5], dtype=torch.float32,device=device)
    k_scale = torch.tensor([0.5], dtype=torch.float32,device=device)
    v_scale = torch.tensor([0.25], dtype=torch.float32,device=device)
    # new_type = torch.float8_e5m2
    # new_type = torch.float8_e4m3fn
    new_type = dtype
    k_cache_paged = k_cache_paged.permute(0, 2, 1, 3).contiguous().to(new_type)
    v_cache_paged = v_cache_paged.permute(0, 2, 3, 1).contiguous().to(new_type)
    max_seqlen_k=seqlen_k
    # max_seqlen_k=32768
    # warm
    for i in range(10):
        out = _flash_attn_with_kvcache(
            q,
            k_cache if paged_kv_block_size is None else k_cache_paged,
            v_cache if paged_kv_block_size is None else v_cache_paged,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=causal,
            max_seqlen_k=max_seqlen_k,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    # prof time   
    torch.cuda.synchronize()     
    repeat_num = 100
    start_time = time.time()
    for i in range(repeat_num):
        out = _flash_attn_with_kvcache(
            q,
            k_cache if paged_kv_block_size is None else k_cache_paged,
            v_cache if paged_kv_block_size is None else v_cache_paged,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=causal,
            max_seqlen_k=max_seqlen_k,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    torch.cuda.synchronize()
    end_time = time.time()
    fc1_espl = end_time - start_time
    DCU_time = fc1_espl *1000*1000 / repeat_num
    IO_bytes = batch_size*seqlen_k*kv_head*d*2*k_cache_paged.element_size() #kv cache size to read
    IO_bytes += batch_size*qhead*d*q.element_size() #q size to read
    IO_bytes += (seqlen_k//512+1)*batch_size*qhead*d*2*2 # temp to write and read
    IO_bytes += batch_size*qhead*d*2 #output to write
    IO_speed = IO_bytes/DCU_time/1024/1024/1024*1000*1000
    print('FA_kvcache bs=', batch_size,' seqlen=',seqlen_k,' qhead=',qhead, ' kv_head=',kv_head, ' time is', '{:.2f}'.format(DCU_time), 'us  Bandwidth=','{:.2f}'.format(IO_speed),'GB/s')
    res_list = [paged_kv_block_size, batch_size, seqlen_k, d, qhead, kv_head, DCU_time,IO_speed]
    # print('FA_kvcache bs=', batch_size,' seqlen=',seqlen_k,' qhead=',qhead, ' kv_head=',kv_head, ' time is', '{:.2f}'.format(DCU_time), 'us')
    # res_list = [paged_kv_block_size, batch_size, seqlen_k, d, qhead, kv_head, DCU_time]
    return res_list


    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    if new_kv:
        if paged_kv_block_size is None:
            k_cache_select = (
                k_cache if not has_batch_idx else k_cache[cache_batch_idx.to(dtype=torch.long)]
            )
            v_cache_select = (
                v_cache if not has_batch_idx else v_cache[cache_batch_idx.to(dtype=torch.long)]
            )
        else:
            k_cache_select = rearrange(
                k_cache_paged[block_table.to(dtype=torch.long).flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
            v_cache_select = rearrange(
                v_cache_paged[block_table.to(dtype=torch.long).flatten()],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
        assert torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3)
        assert torch.equal(v_cache_select, v_cache_ref)
    mult = 3 if not alibi else 5
    assert (out - out_ref).abs().max().item() <= mult * (out_pt - out_ref).abs().max().item() + 1e-5


def _generate_block_kvcache(seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype):
    num_blocks = 50000
    k_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    if eager:
        max_num_blocks_per_seq = (seqlen_k + paged_kv_block_size - 1) // paged_kv_block_size
    else:
        max_num_blocks_per_seq = (max_seqlen + paged_kv_block_size - 1) // paged_kv_block_size
    block_tables = []
    for _ in range(batch_size):
        block_table = [
            random.randint(0, num_blocks - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device=device)

     
    # # randperm torch.randperm
    # block_table = rearrange(
    #     torch.randperm(batch_size*max_seqlen//paged_kv_block_size, dtype=torch.int32, device=device), 
    #     "(b nblocks) -> b nblocks",
    #     b=batch_size,
    # )
    k_cache = rearrange(
        # pytorch 1.12 doesn't have indexing with int32
        k_cache_paged[block_tables.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[block_tables.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]

    return k_cache, v_cache, block_tables, k_cache_paged, v_cache_paged, num_blocks


# mha
if __name__ == "__main__":
    # HIP_VISIBLE_DEVICES=6 python test_kvcache.py    
    #config = [(1,16,16),(1,32,32),(1,32,4),(64,32,4),(1,52,4),(64,52,4),(1,16,2),(64,16,2),(1,26,2),(64,26,2),(1,8,1),(64,8,1),(1,13,1),(64,13,1)]
    # config = [(120,6,1),(120,8,1),(120,28,4),(120,16,2),(120,20,4)]
    # seq_lens=[600,1200,2400,4800]
    random.seed(0)
    torch.random.manual_seed(0)
    # batchsize = [4,8,16,24,32,48,56,64,72,88,120]
    # batchsize = [1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104]
    batchsize = [1,8,32,128]
    # batchsize = [128,256,512]
    # batchsize = [16,24,32,40,48,56,64,72,80,88,96] #70B,235B
    # batchsize = [24,32,40,48,56] #30B
    # batchsize = [40,48,56,64,72,80,88,96] #8B
    # head =  [(32,2)]
    # head =  [(12,1)]
    head =  [(16,2),(32,8)]
    # head =  [(15,1),(16,1)]
    # head =  [(8,1),(9,1),(10,1),(11,1),(12,1),(13,1),(14,1),(15,1),(16,1),(17,1),(18,1),(19,1),(20,1),(21,1),(22,1),(23,1),(24,1),(25,1),(26,1),(27,1),(28,1),(29,1),(30,1),(31,1),(32,1)]
    # head =  [(4,1),(8,1),(12,1),(16,1),(24,1)]
    # seq_lens=[100,400,700,1000,1300,1600,1900,2200,2500,2800,3100,3400,3700,4000,4300]
    # seq_lens=[2000,2100,2200,2300,2400,2500,2600,2700]
    seq_lens=[2048,8192,32768]
    # seq_lens=[8192,128000]
    # seq_lens=[1000,1100,1350,1500,1650,1800,2000,2300,2600,3000,3300,3500,3700,4000,4096,4100,4200,4300,4500,4700,5000]
    # seq_lens=[3000,3300,3500,3800,4000,4300,4500,4800,5000]
    # seq_lens=[500,700,1000,1300,2000,3000,4000,16000,18000,20000]
    # seq_lens=[200,500,800,1100,1300,2000,3000,4000,5000,15000,16000,18000,20000]
    # seq_lens=[200,500,800,1100,1300,2000,3000,4000,5000,16000,16500,17000,17500,18000,18500,19000,19500,20000]
    # seq_lens=[16000,17000,18000,19000,20000,21000]

    # heads = [8, 10, 16, 18, 20, 28, 30, 32, 38, 40, 48, 50, 58, 60, 64, 68, 70]
    # batchs = [64]
    # seq_lens=[1500]
    dtype=torch.float16
    # dtype=torch.bfloat16
    print(dtype)
    res_time = []
    for qh,kh in head:
        for bs in batchsize:
            for seq in seq_lens:
                # if (not (seq>=10000 and bs>16)) and seq<max_seqlen:
                if True:
                    prof_time = test_flash_attn_kvcache(
                    seqlen_q=1, 
                    seqlen_k=seq, #128 512
                    d=128, # 64 128 160 256
                    has_batch_idx=False,
                    has_leftpad=False,
                    paged_kv_block_size=64, #16 256
                    rotary_fraction=0.0,
                    rotary_interleaved=False,
                    seqlen_new_eq_seqlen_q=True,
                    causal=True, # 因果注意力机制
                    local=False,  # 局部注意力
                    alibi=False,
                    new_kv=False,
                    dtype=dtype,
                    batch_size=bs,
                    qhead=qh,
                    kv_head=kh,
                    prof=False  # 运行单次
                    )
                    res_time.append(prof_time)
    with open('kvcache_time.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in res_time:
            writer.writerow(row)


