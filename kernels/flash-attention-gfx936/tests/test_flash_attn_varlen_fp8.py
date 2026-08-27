import math
import pdb
import pytest
import torch
torch.set_printoptions(precision=4, profile="default", sci_mode=False)
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import (
    flash_attn_varlen_func,

)
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import _get_block_size_n
from flash_attn.layers.rotary import apply_rotary_emb
import triton
MAX_HEADDIM_SM8x = 192


is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)


def attn_bias_from_alibi_slopes(
    slopes, seqlen_q, seqlen_k, query_padding_mask=None, key_padding_mask=None, causal=False, key_leftpad=None
):
    batch, nheads = slopes.shape
    device = slopes.device
    slopes = rearrange(slopes, "b h -> b h 1 1")
    if causal:
        return torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32) * slopes
    else:
        row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        if key_leftpad is not None:
            key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
            col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
            col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        relative_pos = torch.abs(row_idx + sk - sq - col_idx)
        return -slopes * relative_pos.to(dtype=slopes.dtype)


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask


def generate_qkv(
    q, k, v, query_padding_mask=None, key_padding_mask=None, kvpacked=False, qkvpacked=False
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    _, _, _, d_v = v.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d_v)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


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
    q_descale=None, k_descale=None, v_descale=None,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
    intermediate_dtype=None,
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

    if q_descale is not None:
        q_descale = repeat(q_descale, "b h -> b 1 (h g) 1", g=q.shape[2] // k.shape[2])
        q = (q.float() * q_descale).to(q.dtype)
      
    if k_descale is not None:
        k = (k.float() * rearrange(k_descale, "b h -> b 1 h 1")).to(dtype=k.dtype)
    if v_descale is not None:
        v = (v.float() * rearrange(v_descale, "b h -> b 1 h 1")).to(dtype=v.dtype)

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
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
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

    if intermediate_dtype is not None:
        attention_drop = attention_drop.to(intermediate_dtype).to(attention_drop.dtype)  

    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def convert_flash_attn_S_to_softmax(
    S,
    seqlen_q,
    seqlen_k,
    query_padding_mask,
    key_padding_mask,
    head_dim,
    is_dropout,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
        query_padding_mask: (batch_size, seqlen_q_rounded)
        key_padding_mask: (batch_size, seqlen_k_rounded)
    """
    if causal:
        window_size = (window_size[0], 0)
    seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
    S_converted = S
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            S.device,
        )
        local_mask = F.pad(
            local_mask,
            (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
            value=True,
        )
        S_converted = S_converted.masked_fill(local_mask, 0.0)

    # Need to zero out things not in attention_mask in case S was initialized with random values
    # and some of those values aren't overwritten.
    seqlen_q_og = (
        query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
    )
    if query_padding_mask is not None:
        query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
        S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
    if key_padding_mask is not None:
        key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
        S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
    S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
    S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
    return S_converted[:, :, :seqlen_q, :seqlen_k]


def normalize_flash_attn_S(
    attn_unnorm,
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    is_dropout=False,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k, v: (batch_size, seqlen_k, nheads, head_dim)
        key_padding_mask: (batch_size, seqlen_q)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
    Output:
        softmax_lse: (batch_size, nheads, seqlen_q)
        softmax_max: (batch_size, nheads, seqlen_q)
    """
    if causal:
        window_size = (window_size[0], 0)
    q, k, v = q.float(), k.float(), v.float()
    _, seqlen_q, _, head_dim = q.shape
    seqlen_k = k.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(head_dim), k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias.to(dtype=scores.dtype)
    block_size_n = _get_block_size_n(scores.device, head_dim, is_dropout, causal)
    scores_block = scores.split(block_size_n, dim=-1)
    lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
    lse = torch.logsumexp(lse_block, dim=-1)
    # lse could be -inf (i.e. all values in scores are -inf), and we want to set those to inf
    # so that when we do torch.exp(m - lse), we get 0.0 instead of NaN.
    lse[lse == float("-inf")] = float("inf")
    scores_max_block = torch.stack([torch.amax(s, dim=-1) for s in scores_block], dim=-1)
    cummax_block = torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
    attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
    attn_norm = torch.cat(
        [
            a * rearrange(torch.exp(m - lse), "b h s -> b h s 1")
            for a, m in zip(attn_unnorm_block, cummax_block)
        ],
        dim=-1,
    )
    if query_padding_mask is not None:
        attn_norm.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    return attn_norm.to(dtype=attn_unnorm.dtype)


def get_dropout_fraction(
    dropout_mask,
    query_padding_mask=None,
    key_padding_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k), bool. True means keep, False means drop.
    query_padding_mask: (batch_size, seqlen_q)
    key_padding_mask: (batch_size, seqlen_k)
    """
    if causal:
        window_size = (window_size[0], 0)
    batch_size, nheads, seqlen_q, seqlen_k = dropout_mask.shape
    dropped = ~dropout_mask
    valid = torch.ones_like(dropout_mask)
    if query_padding_mask is not None:
        dropped.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
        valid.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
    if key_padding_mask is not None:
        dropped.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
        valid.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            dropout_mask.device,
        )
        dropped.masked_fill_(local_mask, False)
        valid.masked_fill_(local_mask, False)
    dropped_total = dropped.sum()
    return dropped.sum() / valid.sum()

def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool=False) -> None:
    torch_dtype = x.dtype
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    assert cos_diff < 1e-3
 


@pytest.mark.parametrize('kvpacked', [False])
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
# @pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize('mha_type', ["mha"])
# @pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("alibi", [False, True])
# @pytest.mark.parametrize("alibi", [False])
@pytest.mark.parametrize("local", [False, True])
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize('d', [128])


@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 147),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)

@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
@pytest.mark.parametrize("softcap", [0.0, 50.0])

# @pytest.mark.parametrize('dropout_p', [0.0])
# @pytest.mark.parametrize("softcap", [0.0])
def test_flash_attn_varlen_output_fp8(
    seqlen_q, seqlen_k, d, dropout_p, causal, local, alibi, deterministic, mha_type, dtype, kvpacked, softcap
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if softcap > 0.0 and dropout_p > 0.0:
        pytest.skip("Softcap and dropout not supported together")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    batch_size = 4
    nheads = 16 if softcap == 0.0 else 4  # softcap reference impl takes more memory
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 2)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref, requires_grad=True)
    if softcap > 0:
        # Ensure the values of qk are at least within softcap range.
        q_ref = q_ref * softcap

    q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
    k_ref = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
    v_ref = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()

    if dtype == torch.float8_e4m3fn:

        q_descale, k_descale, v_descale = [torch.ones(batch_size, nheads_k, device=device, dtype=torch.float32)  for _ in range(3)]
    else:
        q_descale, k_descale, v_descale = None, None, None

    query_padding_mask=None
    key_padding_mask=None

    # query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    # key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    # key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode='full')
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, query_padding_mask, key_padding_mask, causal=causal
        )
    else:
        alibi_slopes, attn_bias = None, None

    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q_ref, k_ref, v_ref, query_padding_mask, key_padding_mask, kvpacked=False)

    q_unpad_fp8=q_unpad.to(dtype).requires_grad_()
    k_unpad_fp8=k_unpad.to(dtype).requires_grad_()
    v_unpad_fp8=v_unpad.to(dtype).requires_grad_()


    out_unpad, sm_lse, S_dmask = flash_attn_varlen_func(
        q_unpad_fp8,
        k_unpad_fp8,
        v_unpad_fp8,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        return_attn_probs=True,
    )

    out = output_pad_fn(out_unpad)

    if dropout_p > 0.0:
        if dtype == torch.float8_e4m3fn:
            S_dmask=S_dmask.to(dtype_ref)

        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()
      
        k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)

        attn = normalize_flash_attn_S(
            attn_unnorm,
            q,
            k_rep,
            v_rep,
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_fraction = get_dropout_fraction(
            dropout_mask,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
        ).item()
        print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

   
    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        window_size=window_size,
        softcap=softcap,
    )

    out_pt, attn_pt = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        window_size=window_size,
        softcap=softcap,
        upcast=False,
        reorder_ops=True,
        intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
    )

 
    # if dropout_p > 0.0:
    #     print(f"Attention max diff: {(attn - attn_ref).abs().max().item()}")
    #     print(f"Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()










@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
#@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize('mha_type', ["mha"])
#@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("deterministic", [False])
@pytest.mark.parametrize("alibi", [False,True])
@pytest.mark.parametrize("local", [False,True])
# @pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [False,True])
# @pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('d, d_v', [(192, 128)])


# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
# #@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# @pytest.mark.parametrize('mha_type', ["mha"])
# #@pytest.mark.parametrize("deterministic", [False, True])
# @pytest.mark.parametrize("deterministic", [False])
# @pytest.mark.parametrize("alibi", [False])
# @pytest.mark.parametrize("local", [False])
# # @pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("causal", [True])
# # @pytest.mark.parametrize('causal', [False])
# @pytest.mark.parametrize('d, d_v', [(192, 128)])

@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 147),
        (113, 203),
        (128, 217),
        (113, 211), 
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 512),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)

@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
@pytest.mark.parametrize("softcap", [0.0,50.0])

# @pytest.mark.parametrize('dropout_p', [0.0])
# @pytest.mark.parametrize("softcap", [0.0])
def test_flash_attn_varlen_output_mla_fp8(
    seqlen_q, seqlen_k, d, d_v, dropout_p, causal, local, alibi, deterministic, mha_type, dtype, softcap
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if softcap > 0.0 and dropout_p > 0.0:
        pytest.skip("Softcap and dropout not supported together")
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    batch_size = 4
    nheads = 16 if softcap == 0.0 else 4  # softcap reference impl takes more memory
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 2)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))
    q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
    #q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    if softcap > 0:
        # Ensure the values of qk are at least within softcap range.
        q_ref = q_ref * softcap

    q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
    k_ref = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
    v_ref = torch.randn(batch_size, seqlen_k, nheads_k, d_v, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()

    if dtype == torch.float8_e4m3fn:

        q_descale, k_descale, v_descale = [torch.ones(batch_size, nheads_k, device=device, dtype=torch.float32)  for _ in range(3)]
    else:
        q_descale, k_descale, v_descale = None, None, None

    query_padding_mask = generate_random_padding_mask(seqlen_q, batch_size, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode="random")
    # query_padding_mask = None
    # key_padding_mask = None
    # key_padding_mask = generate_random_padding_mask(seqlen_k, batch_size, device, mode='full')
    if alibi:
        alibi_slopes = torch.rand(batch_size, nheads, device=device, dtype=torch.float32) * 0.3
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, query_padding_mask, key_padding_mask, causal=causal
        )
    else:
        alibi_slopes, attn_bias = None, None

    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q_ref, k_ref, v_ref, query_padding_mask, key_padding_mask, kvpacked=False)


    q_unpad_fp8=q_unpad.to(dtype).requires_grad_()
    k_unpad_fp8=k_unpad.to(dtype).requires_grad_()
    v_unpad_fp8=v_unpad.to(dtype).requires_grad_()
    
    def fa():
        return flash_attn_varlen_func(
                q_unpad_fp8,
                k_unpad_fp8,
                v_unpad_fp8,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                return_attn_probs=True,
            )


    out_unpad, sm_lse, S_dmask =fa()
    
    out = output_pad_fn(out_unpad)

    if dropout_p > 0.0:
        if dtype == torch.float8_e4m3fn:
            S_dmask=S_dmask.to(dtype_ref)

        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()
        
        k_rep = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        v_rep = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
        attn = normalize_flash_attn_S(
            attn_unnorm,
            q_ref,
            k_rep,
            v_rep,  
            query_padding_mask,
            key_padding_mask,
            attn_bias,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_fraction = get_dropout_fraction(
            dropout_mask,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
            window_size=window_size,
        ).item()
        print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None
    # print(out)
    out_ref, attn_ref = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        window_size=window_size,
        softcap=softcap,
    )
    # print(out_ref)
    out_pt, attn_pt = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        window_size=window_size,
        softcap=softcap,
        upcast=False,
        reorder_ops=True,
        intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
    )

    # cal_diff(out_fp16, out_pt, "OUT")
    #cal_diff(attn, attn_ref, "attn")
    # cal_diff(out_ref, out_pt, "OUT")
    # print((out - out_ref).abs().max().item())
    # print(2*(out_pt - out_ref).abs().max().item())
    # cal_diff(out, out_pt, "OUT")
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
    # assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()


