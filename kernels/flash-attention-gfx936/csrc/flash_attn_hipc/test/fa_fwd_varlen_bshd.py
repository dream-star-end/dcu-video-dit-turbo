from flash_attention_utils import *
import flash_attn


# 如果没有即时加载的输入和 golden, 随机生成并计算
if (args.load is None):
    seqlen_qkv   = None
    num_heads    = 16
    num_heads_kv = 2
    head_dim_qk  = 128
    head_dim_v   = 128
    is_causal    = True
    input_dtype  = torch.bfloat16
    if (seqlen_qkv is None):
        batch_size = random.randint(1, 64)
        seqlen_qkv = [random.randint(1, 256) for i in range(batch_size)]
    else:
        batch_size = len(seqlen_qkv)
    seqlen_qkv_sum = sum(seqlen_qkv)
    seqlen_qkv_max = max(seqlen_qkv)
    prefill_meta_seq_start_loc = numpy.array([0] + numpy.cumsum(seqlen_qkv).tolist()).astype("int32")
    prefill_meta_seq_start_loc = torch.from_numpy(prefill_meta_seq_start_loc).cuda()
    # 随机生成输入 query, key, value, 原生 bhsd layout
    query = torch.randn((seqlen_qkv_sum, num_heads, head_dim_qk), dtype=input_dtype, device="cuda")
    key   = torch.randn((seqlen_qkv_sum, num_heads_kv, head_dim_qk), dtype=input_dtype, device="cuda")
    value = torch.randn((seqlen_qkv_sum, num_heads_kv, head_dim_v), dtype=input_dtype, device="cuda")
    print("-------------------------------------------------")
    print("(generating inputs/golden from scratch...)")
    print("query: {}\nkey: {}\nvalue: {}".format(query.shape, key.shape, value.shape))
    print("seqlen_kv       :", seqlen_qkv)
    print("batch_size      :", batch_size)
    print("seqlen_qkv_sum  :", seqlen_qkv_sum)
    print("seqlen_qkv_max  :", seqlen_qkv_max)
    # 拆分出每个 batch 的结果
    query_batch = []
    key_batch   = []
    value_batch = []
    golden_batch = []
    lse_batch    = []
    for i in range(batch_size):
        # 从 batch x num_heads x seqlen, head_dim 中解析出每个 batch 的 Q/K/V 内容
        query_batch.append(query[prefill_meta_seq_start_loc[i]: prefill_meta_seq_start_loc[i + 1]].permute(1, 0, 2).contiguous().unsqueeze(0))
        key_batch.append(key[prefill_meta_seq_start_loc[i]: prefill_meta_seq_start_loc[i + 1]].permute(1, 0, 2).contiguous().unsqueeze(0))
        value_batch.append(value[prefill_meta_seq_start_loc[i]: prefill_meta_seq_start_loc[i + 1]].permute(1, 0, 2).contiguous().unsqueeze(0))
        # 计算 golden
        goldens    = standard_attention(query_batch[-1], key_batch[-1], value_batch[-1], causal_mask=is_causal)
        golden_now = goldens[0]
        golden_lse = goldens[1]
        golden_batch.append(golden_now)
        lse_batch.append(golden_lse)
        sys.stdout.write("\rgolden computing: {}/{} -- {}".format(i + 1, batch_size, golden_now.shape))
        sys.stdout.flush()
    print("")
    # golden 输出应该和 query 一样大
    golden     = torch.cat(golden_batch, dim=2).squeeze(0).permute(1, 0, 2).contiguous()
    golden_lse = torch.cat(lse_batch, dim=2).squeeze(-1).squeeze(0).contiguous().to(golden.device)
    print("golden: ", golden.shape)
    print("lse   : ", golden_lse.shape)
    if (os.getenv("FA_DEBUG") is not None):
        print("-------------------- args ------------------------")
        print("query: ", query.shape, query.dtype)
        print("key: ", key.shape, key.dtype)
        print("value: ", value.shape, value.dtype)
        print("out_: ", None)
        print("cu_seqlens_q: ", prefill_meta_seq_start_loc)
        print("cu_seqlens_kv: ", prefill_meta_seq_start_loc)
        print("seqused_k: ", None)
        print("alibi_slopes: ", None)
        print("seqlen_q_max: ", seqlen_qkv_max)
        print("seqlen_kv_max: ", seqlen_qkv_max)
        print("p_dropout: ", 0.0)
        print("softmax_scale: ", 1.0 / math.sqrt(head_dim_qk))
        print("zero_tensors: ", None)
        print("is_causal: ", True)
        print("window_size_left: ", -1)
        print("window_size_right: ", -1)
        print("softcap: ", 0.0)
        print("return_softmax: ", False)
        print("gen_: ", None)
        print("-------------------------------------------------")
    else:
        print("-------------------------------------------------")
    # 接 FA varlen 推理, 原生支持 (bs)hd 的 layout
    if (True):
        outputs = flash_attn_hg_cuda.varlen_fwd(
            query,
            key,
            value,
            None,
            prefill_meta_seq_start_loc,
            prefill_meta_seq_start_loc,
            None,
            None,
            seqlen_qkv_max,
            seqlen_qkv_max,
            0.0,
            1.0 / math.sqrt(head_dim_qk),
            None,
            is_causal,
            -1,
            -1,
            0.0,
            False,
            None,
        )
    elif (True):
        outputs = torch.zeros((query.size(0), query.size(1), value.size(-1)), dtype=query.dtype, device=query.device)
        flash_attn.flash_attn_interface.hg_flash_attn_varlen_func(
            query,
            key,
            value,
            prefill_meta_seq_start_loc,
            prefill_meta_seq_start_loc,
            seqlen_qkv_max,
            seqlen_qkv_max,
            out=outputs,
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(head_dim_qk),
            causal=is_causal,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
            block_table=None,
            layout="bshd",
        )
    else:
        # 测试 python 接口功能完整性, 不测 lse
        outputs = flash_attn.flash_attn_interface.hg_flash_attn_varlen_func(
            query,
            key,
            value,
            prefill_meta_seq_start_loc,
            prefill_meta_seq_start_loc,
            seqlen_qkv_max,
            seqlen_qkv_max,
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(head_dim_qk),
            causal=is_causal,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
            block_table=None,
            layout="bshd",
        )
else:
    # demo: VARLEN_LOAD_PATH=/data/liuchang/FA_NV_GOLDEN/inference/fwd/llama-2-7b-bs10000/llama-2-7b-bs10000_id365.pt
    packet = torch.load(serialization_path)
    print("load inputs/golden from {}".format(serialization_path))
    if ("bshd_output" in packet.keys() and "inputs" in packet.keys()):
        golden = packet["bshd_output"]
        (query, key, value, out_, cu_seqlens_q, cu_seqlens_k, seqused_k, alibi_slopes, max_seqlen_q, max_seqlen_k, p, softmax_scale, zero_tensors, is_causal, win_left, win_right, softcap, return_softmax, gen)\
            = packet["inputs"]
    else:
        raise NotImplementedError("You need to write load pipeline codes for quick varlen test")
    print("query: {}\nkey: {}\nvalue: {}".format(query.shape, key.shape, value.shape))
    print("seqlen_kv       :", cu_seqlens_q)
    print("batch_size      :", len(cu_seqlens_k) - 1)
    print("seqlen_qkv_sum  :", cu_seqlens_q.sum().item())
    print("seqlen_qkv_max  :", max_seqlen_q)
    if (hasattr(flash_attn_hg_cuda, "varlen_fwd")):
        outputs = flash_attn_hg_cuda.varlen_fwd(
            query,
            key,
            value,
            out_,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_k,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            p,
            softmax_scale,
            zero_tensors,
            is_causal,
            win_left,
            win_right,
            softcap,
            return_softmax,
            gen,
        )
    else:
        raise NotImplementedError("Please add '-DBUILD_FA_PERMUTE=ON' while building!")

# 获取 output 和 lse
if (isinstance(outputs, list) or isinstance(outputs, tuple)):
    fa_output   = outputs[0]
    softmax_lse = outputs[5]
else:
    fa_output   = outputs
    softmax_lse = None
print("fa_output: ", fa_output.shape)
if (softmax_lse is not None): print("fa_lse   : ", softmax_lse.shape)

# 计算正确性情况
abs_diff = torch.abs(fa_output - golden)
occur_nan_in_output = torch.any(torch.isnan(fa_output)) # 检查是否有异常值
occur_inf_in_output = torch.any(torch.isinf(fa_output))
print("Output: \x1b[35m{:.12f}\x1b[0m  |  \x1b[35m{:.12f}\x1b[0m  |  [{:.2f}, {:.2f}]  {}".format(abs_diff.mean(), abs_diff.max(), fa_output.min().item(), fa_output.max().item(), fa_output.shape))

# 检查 lse 情况, 为训练准备
if (softmax_lse is not None):
    lse_abs_diff  = torch.abs(softmax_lse - golden_lse)
    occur_nan_in_lse = torch.any(torch.isnan(softmax_lse))
    occur_inf_in_lse = torch.any(torch.isinf(softmax_lse))
    print("LSE   : \x1b[35m{:.12f}\x1b[0m  |  \x1b[35m{:.12f}\x1b[0m  |  [{:.2f}, {:.2f}]  {} ".format(lse_abs_diff.mean(), lse_abs_diff.max(), softmax_lse.min().item(), softmax_lse.max().item(), softmax_lse.shape))
else:
    occur_nan_in_lse = False
    occur_inf_in_lse = False
print("CHECK NaN: ", get_status_str(not (occur_nan_in_output or occur_nan_in_lse)))
print("CHECK Inf: ", get_status_str(not (occur_inf_in_output or occur_inf_in_lse)))

# 测试性能, 稳定性
if (os.getenv("FA_DEBUG") is None):
    input_params = (query, key, value, None, prefill_meta_seq_start_loc, prefill_meta_seq_start_loc, None, None, seqlen_qkv_max, seqlen_qkv_max, 0.0, 1.0 / math.sqrt(head_dim_qk), None, is_causal, -1, -1, 0.0, False, None)
    cost_time = []
    for __iter in range(10):
        t = benchmark_forward(flash_attn_hg_cuda.varlen_fwd, *input_params, repeats=1, verbose=False, amp_dtype=input_dtype) / 1e6
        if (__iter > 0): cost_time.append(t)
    cost_time = numpy.array(cost_time)
    cost_time_mean = cost_time.mean()
    cost_time = numpy.delete(cost_time, numpy.where(cost_time < (0.8 * cost_time_mean))) # 去除突发低数据
    cost_time_mean = cost_time.mean()
    qk_gemm_flops = 2 * num_heads * sum([it ** 2 for it in seqlen_qkv]) * head_dim_qk
    pv_gemm_flops = 2 * num_heads * sum([it ** 2 for it in seqlen_qkv]) * head_dim_v
    flops_count   = (qk_gemm_flops + pv_gemm_flops) / (2 if (is_causal) else 1)
    tflops = flops_count / (cost_time_mean * 1e12)
    print("Performance: {:.4f} ms \x1b[35m{:.1f}\x1b[0m TFLOPS".format(cost_time_mean * 1e3, tflops))

    # 压力测试
    pressure_count = 100
    for p in range(pressure_count):
        torch.cuda.empty_cache()
        pressure_query_layer = query.clone()
        pressure_key_layer   = key.clone()
        pressure_value_layer = value.clone()
        outputs = flash_attn_hg_cuda.varlen_fwd(pressure_query_layer, pressure_key_layer, pressure_value_layer, None, prefill_meta_seq_start_loc, prefill_meta_seq_start_loc, None, None, seqlen_qkv_max, seqlen_qkv_max, 0.0, 1.0 / math.sqrt(head_dim_qk), None, is_causal, -1, -1, 0.0, False, None)
        torch.cuda.synchronize()
        pressure_fa_output   = outputs[0]
        assert torch.equal(pressure_fa_output, fa_output), "Unstable"
        if (softmax_lse is not None):
            pressure_softmax_lse = outputs[5]
            assert torch.equal(pressure_softmax_lse, softmax_lse), "Unstable"
        pressure_query_layer.fill_(0)
        pressure_key_layer.fill_(0)
        pressure_value_layer.fill_(0)
        del pressure_query_layer, pressure_key_layer, pressure_value_layer
        sys.stdout.write("\rPressure Test: {}/{}".format(p + 1, pressure_count))
    print(" \x1b[32mPASS\x1b[0m")
