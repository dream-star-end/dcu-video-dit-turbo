import torch
from flash_attn import attn_torch_function
from flash_attn import flash_attn_triton

from flash_attn.flash_attn_triton_mqa_gqa import attention as attention_mqa_gqa
from flash_attn.flash_attn_triton_mqa_gqa import MetaData


def isSameQKV(seqlen_q, seqlen_k, head_dim_k, dropout_p):
    if seqlen_q == seqlen_k and int(dropout_p) == 0 and head_dim_k in [16, 32, 64, 128]:
        return True, flash_attn_triton.flash_attn_func
    return False, attn_torch_function.attention_neq


def padding_thd(t):   # THD 
    total_seqlen, nheads, dim = t.shape
    if total_seqlen > 4096:
        t = torch.nn.functional.pad(t.reshape(total_seqlen, nheads*dim), (0, 32), 'constant', 0)[:,:-32].reshape(total_seqlen, nheads, dim) # pad: nheads*dim+32
    return t


def padding_bshd(t):   # BSHD 
    batch, seqlen, nheads, dim = t.shape
    if seqlen > 4096:
        t = torch.nn.functional.pad(t.reshape(batch, seqlen, nheads*dim), (0, 32), 'constant', 0)[:,:,:-32].reshape(batch, seqlen, nheads, dim) # pad: nheads*dim+32
    return t


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    return_attn_probs=False):
    sm_scale = softmax_scale if softmax_scale else q.shape[-1]**-0.5
    seqlens_q = q.shape[2]
    seqlens_k, head_dim_k = k.shape[2], k.shape[3]
    Flag, attention = isSameQKV(seqlens_q, seqlens_k, head_dim_k, dropout_p)
    if Flag:
        return attention(q, k, v, causal, sm_scale)
    return attention(q, k, v, None, causal, sm_scale, dropout_p, return_attn_probs)[0]

def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False,
                    return_attn_probs=False):
    k, v = kv[:, :, 0], kv[:, :, 1]  # batch_size, nheads_k, 2, seqlen, headdim
    sm_scale = softmax_scale if softmax_scale else q.shape[-1]**-0.5
    seqlens_q = q.shape[2]
    seqlens_k, head_dim_k = k.shape[2], k.shape[3]
    Flag, attention = isSameQKV(seqlens_q, seqlens_k, head_dim_k, dropout_p)
    if Flag:
        return attention(q, k, v, causal, sm_scale)
    return attention(q, k, v, None, causal, sm_scale, dropout_p, return_attn_probs)[0]

def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                              return_attn_probs=False):
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    sm_scale = softmax_scale if softmax_scale else q.shape[-1]**-0.5
    seqlens_q = q.shape[2]
    seqlens_k, head_dim_k = k.shape[2], k.shape[3]
    Flag, attention = isSameQKV(seqlens_q, seqlens_k, head_dim_k, dropout_p)
    if Flag:
        return attention(q, k, v, causal, sm_scale)
    return attention(q, k, v, None, causal, sm_scale, dropout_p, return_attn_probs)[0]



def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlens, dropout_p=0.0, softmax_scale=None,
                                     causal=False, return_attn_probs=False, padding_input=False):
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # total_seqlen, 3, nheads, dim
    if padding_input:
        k, v = (padding_thd(t) for t in (k, v))   #  pad 
    softmax_scale = softmax_scale if softmax_scale else q.shape[-1]**-0.5
    input_metadata = MetaData(sm_scale=softmax_scale, causal=causal, dropout_p=dropout_p, return_encoded_softmax=return_attn_probs)
    input_metadata.set_varlen_params(cu_seqlens, cu_seqlens)
    input_metadata.max_seqlens_q = max_seqlens
    input_metadata.max_seqlens_k = max_seqlens
    return  attention_mqa_gqa(q, k, v, None, input_metadata)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k,
                                    dropout_p=0.0, softmax_scale=None, causal=False,
                                    return_attn_probs=False, padding_input=False):
    k, v = kv[:, 0], kv[:, 1]  # total_seqlen, 2, nheads, dim
    if padding_input:
        k, v = (padding_thd(t) for t in (k, v))
    softmax_scale = softmax_scale if softmax_scale else q.shape[-1]**-0.5
    input_metadata = MetaData(sm_scale=softmax_scale, causal=causal, dropout_p=dropout_p, return_encoded_softmax=return_attn_probs)
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)
    input_metadata.max_seqlens_q = max_seqlens_q
    input_metadata.max_seqlens_k = max_seqlens_k
    return  attention_mqa_gqa(q, k, v, None, input_metadata)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k,
                           dropout_p=0.0, softmax_scale=None, causal=False,
                           return_attn_probs=False, padding_input=False):
    if padding_input:
        k, v = (padding_thd(t) for t in (k, v))
    softmax_scale = softmax_scale if softmax_scale else q.shape[-1]**-0.5
    input_metadata = MetaData(sm_scale=softmax_scale, causal=causal, dropout_p=dropout_p, return_encoded_softmax=return_attn_probs)
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)
    input_metadata.max_seqlens_q = max_seqlens_q
    input_metadata.max_seqlens_k = max_seqlens_k
    return  attention_mqa_gqa(q, k, v, None, input_metadata)