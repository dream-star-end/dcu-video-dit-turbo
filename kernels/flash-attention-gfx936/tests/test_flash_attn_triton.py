import torch
import triton
import torch.nn.functional as F
import os
import pytest

from flash_attn import triton_flash_attn_func
from flash_attn.flash_attn_triton_mqa_gqa import attention as attention_mqa_gqa
from flash_attn.flash_attn_triton_mqa_gqa import input_helper_layout



@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD, dtype',
[ (*shape, dtype)
    for shape in [(4, 48, 1024, 64),
                  (1, 48, 2048, 64),
                  (1, 48, 4096, 64),
                  (1, 48, 1024, 128),
                  (1, 48, 2048, 128),
                  (1, 48, 4096, 128)]
    for dtype in [
        torch.float16,
        torch.bfloat16
    ]
])
@pytest.mark.parametrize('causal', [ True])
def test_op_fwd(Z, H, N_CTX, D_HEAD, causal, dtype):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5)

    sm_scale = q.shape[-1]**-0.5
    dropout_p = 0.0
    # dout = torch.randn_like(q, dtype=torch.float16)
    # triton implementation
    tri_out = triton_flash_attn_func(q, k, v, dropout_p,  causal=causal, softmax_scale=sm_scale)

    # reference implementation
    M = torch.tril(torch.ones((Z, H, N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)
#    p = F.dropout(p_nodrop, 0.5)

    ref_out = torch.matmul(p, v)

    # compare
    atol = 1.4e-1 if dtype == 'fp8' else 1e-2
    rtol = 1e-2 if dtype == 'fp8' else 0
    torch.testing.assert_close(ref_out, tri_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD',
                         [(4, 48, 1024, 64),
                          (1,16,4096,128),
                          (1,32,4096,128),
                          (1,20,4096,128),
                          (1,24,4096,128),
                          (1, 48, 2048, 64),
                          (1, 48, 4096, 64),
                          (1, 16, 8192, 128),
                          ])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_op_bwd(Z, H, N_CTX, D_HEAD, dtype):
    torch.manual_seed(20)
    causal = True
    q = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    q, k, v = (x.to(dtype) for x in (q, k, v))

    sm_scale= q.shape[-1]**-0.5
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((Z, H, N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)
#    p = F.dropout(p_nodrop, 0.5)

    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out = triton_flash_attn_func(q, k, v, causal=causal, softmax_scale=sm_scale)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None

    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
    if torch.version.hip is None:
        torch.testing.assert_close(ref_dv, tri_dv, atol=1e-2, rtol=0)
    # The current block size for MI200 series is 64x64. This results in
    # larger differences in float results due to rounding.
    else:
        torch.testing.assert_close(ref_dv, tri_dv, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dk, tri_dk, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dq, tri_dq, atol=5e-2, rtol=1e-2)

@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD, dtype',
[ (*shape, dtype)
    for shape in [(4, 48, 4096, 77, 64),
                  (2, 8, 4096, 4096, 40),
                  (2, 8, 1024, 1024, 80),
                  (2, 8, 1024, 77, 80),
                  (2, 8, 256, 256, 160),
                  (2, 8, 256, 77, 160)]
    for dtype in [
       torch.float16,
       torch.bfloat16
    ]
])
@pytest.mark.parametrize('causal', [False, True])
def test_op_neq_fwd(Z, H, N_CTX_Q, N_CTX_K , D_HEAD, causal, dtype):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX_Q, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX_K, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX_K, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    softmax_scale= q.shape[-1]**-0.5
    # dout = torch.randn_like(q, dtype=torch.float16)
    # triton implementation
    tri_out = triton_flash_attn_func(q, k, v, causal=causal)
    if isinstance(tri_out, tuple):  # 检查 o 是否为 tuple，如果是则取第一个值
        tri_out = tri_out[0]

    # reference implementation
    M = torch.tril(torch.ones((Z, H, N_CTX_Q, N_CTX_K), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * softmax_scale
    if causal:
        p[M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)

    ref_out = torch.matmul(p, v)

   # compare
    atol = 1.4e-1 if dtype == 'fp8' else 1e-2
    rtol = 1e-2 if dtype == 'fp8' else 0
    torch.testing.assert_close(ref_out, tri_out, atol=atol, rtol=rtol)

@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD',
                         [(4, 48, 2048, 77, 64),
                           (1,8, 2048,16,128),
                           (1,20,4096,4096,128),
                          (1,24,4096,4096,128),
                           (2, 10, 1024, 1024, 80),
                           (2, 20, 1024, 1024, 80),
                           (2, 8, 4096, 4096, 40),
                           (2, 8, 1024, 1024, 80),
                           (2, 8, 1024, 77, 80),
                           (2, 8, 256, 256, 160),
                          ])
@pytest.mark.parametrize('dtype', [torch.float16,torch.bfloat16])
def test_op_neq_bwd(Z, H, N_CTX_Q,N_CTX_K, D_HEAD, dtype):
    torch.manual_seed(20)
    causal = True
    q = (torch.empty((Z, H, N_CTX_Q, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX_K, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX_K, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    q, k, v = (x.to(dtype) for x in (q, k, v))

    softmax_scale= q.shape[-1]**-0.5
    print("softmax_scale", softmax_scale)

    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((Z, H, N_CTX_Q, N_CTX_K), device="cuda"))
    p = torch.matmul(q.float(), k.float().transpose(2, 3)) * softmax_scale
    if causal:
        p[M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)
#    p = F.dropout(p_nodrop, 0.5)

    ref_out = torch.matmul(p.float(), v.float())
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out = triton_flash_attn_func(q, k, v, causal=causal)
    if isinstance(tri_out, tuple):  # 检查 o 是否为 tuple，如果是则取第一个值
        tri_out = tri_out[0]

    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    torch.testing.assert_close(ref_out, tri_out.float(), atol=1e-2, rtol=0)
    if torch.version.hip is None:
        torch.testing.assert_close(ref_dv, tri_dv, atol=1e-2, rtol=0)
    # The current block size for MI200 series is 64x64. This results in
    # larger differences in float results due to rounding.
    else:
        torch.testing.assert_close(ref_dv, tri_dv, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dk, tri_dk, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(ref_dq, tri_dq, atol=5e-2, rtol=1e-2)

@pytest.mark.parametrize('Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype',
[ (*shape, dtype)
    for shape in [(1, 16, 16,4096, 4096,128),
        (1, 32, 32, 4096, 4096,128),
        (1, 32, 4, 4096, 4096,128),
        (1, 52, 4, 4096, 4096,128),
        (1, 16, 2, 4096, 4096,128),
        (1, 26, 2, 4096, 4096,128),
        (1, 8, 1, 4096, 4096,128),
        (1, 13, 1, 4096, 4096,128),
    ]
    for dtype in [
        torch.float16,
    ]
])
@pytest.mark.parametrize('causal', [True])
@pytest.mark.parametrize('layout', ['bhsd'])
def test_op_varlen_fwd(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dtype, layout):
    torch.manual_seed(20)
    q, k, v, input_metadata = input_helper_layout(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout)
    if causal:
        input_metadata.need_causal()

    o = torch.empty_like(q)
    print(q.size(),k.size(),v.size())
    # triton implementation
    tri_out = attention_mqa_gqa(q, k, v, None, input_metadata)

    # Transpose here if layout is bshd so we have same reference code for all layouts
    if layout == 'bshd':
        q = q.transpose(1, 2).clone()
        k = k.transpose(1, 2).clone()
        v = v.transpose(1, 2).clone()
    # Replicate K and V if using MQA/GQA
    if HQ != HK:
        k = k.view(k.shape[0], k.shape[1], -1, k.shape[2],
                   k.shape[3]).expand(-1, -1, HQ // HK, -1, -1).reshape(k.shape[0], -1, k.shape[2], k.shape[3])
        v = v.view(v.shape[0], v.shape[1], -1, v.shape[2],
                   v.shape[3]).expand(-1, -1, HQ // HK, -1, -1).reshape(v.shape[0], -1, v.shape[2], v.shape[3])
    scores = torch.einsum('bhqd,bhkd->bhqk', q, k).float() * input_metadata.sm_scale
    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"), diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")

    p = torch.softmax(scores, dim=-1)
    if causal:
        # If N_CTX_Q > N_CTX_K, there is at least one row of all -infs going into
        # the softmax. This produces a row of NaNs as -inf - -inf == NaN. So we fix
        # this by converting the NaNs to 0s, which is what they should be out of the softmax.
        nan_mask = torch.isnan(p)
        p[nan_mask == 1] = 0
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.half(), v)
    # compare
    if layout == 'bshd':
        ref_out = ref_out.transpose(1, 2).clone()
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


# vary seq length for fixed head and batch=4
configs = []
for mode in ['fwd', 'bwd']:
    for D_HEAD in [128, 64]:
        for causal in [False, True]:
            if mode == 'bwd' and causal == False:
                continue
            configs.append(triton.testing.Benchmark(
                x_names=['BATCH', 'H', 'N_CTX'],
                x_vals=[(4, 16, 1024),
                        (8, 16, 2048),
                        (4, 16, 4096),
                        (2, 16, 8192),
                        (1, 16, 16384),
                        (4, 48, 1024),
                        (4, 48, 2048),
                        (4, 48, 4096),
                        (4, 48, 8192),
                        (4, 48, 16384),
                        ],
                line_arg='provider',
                line_vals=['triton'],
                line_names=['Triton'],
                styles=[('red', '-'), ('blue', '-')],
                ylabel='ms',
                plot_name=f'fused-attention-{mode}-d{D_HEAD}-causal={causal}',
                args={
                    'D_HEAD': D_HEAD,
                    'dtype': torch.float16,
                    'mode': mode,
                    'causal': causal,
                },
            ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 10
    # Bwd pass only supports causal=True right now
    if mode == 'bwd':
        causal = True
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        sm_scale = D_HEAD ** -0.5
        fn = lambda: triton_flash_attn_func(q, k, v, causal=causal, softmax_scale=sm_scale)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: triton_flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9

# only works on post-Ampere GPUs right now
# bench_flash_attention.run(save_path=".", print_data=True)


