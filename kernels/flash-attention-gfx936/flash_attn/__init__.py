__version__ = "2.8.3"

import torch
if torch.cuda.is_available():
    from flash_attn.flash_attn_interface import (
        flash_attn_func,
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_func,
        hg_flash_attn_varlen_func,
        flash_mla_with_kvcache,
        get_mla_metadata,
        vllm_flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_attn_with_kvcache,
        vllm_flash_attn_with_kvcache,
        sparse_attn_varlen_func,
        sparse_attn_func,
        flash_attn_func_blasst,
        sparse_attn_with_sla,
        spas_fa2_attn_meansim_cuda,
        spas_fa2_attn_meansim_topk_cuda,
        spas_fa2_attn_meansim_varlen_cuda,
        spas_fa2_attn_meansim_topk_varlen_cuda,
        # Attnmask functions - FlashAttention with explicit attention mask
        flash_attn_with_mask_func,
        flash_attn_varlen_with_mask_func,
        # unified attn functions 
        varlen_fwd_unified,
        fwd_sparse_mean_pool_fast,
    )
    # triton fa interface
    from flash_attn.flash_attn_triton_interface import flash_attn_func as triton_flash_attn_func
    from flash_attn.flash_attn_triton_interface import flash_attn_kvpacked_func as triton_flash_attn_kvpacked_func
    from flash_attn.flash_attn_triton_interface import flash_attn_qkvpacked_func as triton_flash_attn_qkvpacked_func
    from flash_attn.flash_attn_triton_interface import flash_attn_varlen_func as triton_flash_attn_varlen_func
    from flash_attn.flash_attn_triton_interface import flash_attn_varlen_kvpacked_func as triton_flash_attn_varlen_kvpacked_func
    from flash_attn.flash_attn_triton_interface import flash_attn_varlen_qkvpacked_func as triton_flash_attn_varlen_qkvpacked_func

    try:
        from .version import version, git_hash, git_branch, dtk, abi, torch_version, hcu_version  # noqa: F401
        __version__, __hcu_version__, __git_hash__, __git_branch__ = version, hcu_version, git_hash, git_branch
    except ImportError:
        pass
