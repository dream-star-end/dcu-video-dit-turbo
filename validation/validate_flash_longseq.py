#!/usr/bin/env python3
"""Validate the audited gfx936 flash kernel beyond the 50k seq gate.

For each sequence length: build (1,28,S,128) bf16 tensors with the exact
production stride layout (views of a packed (S,3584) buffer), run fwd_bhsd,
check global NaN/Inf, and compare sampled q-row blocks against a chunked
fp32 reference. The 23638 shape is the audited baseline; longer shapes are
healthy only if their error stays at the same magnitude.
"""
import importlib.util
import json
import sys
import time

import torch

SO = ("kernels/flash-attention-gfx936/build/"
      "flash_attn_hg_forward_gfx936/libflash_attention.so")
spec = importlib.util.spec_from_file_location("h3_flash_attn", SO)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

HEADS, DIM, HIDDEN = 28, 128, 3584
DEV = "cuda:0"
CHUNK = 2048


def make(seq, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    def one():
        buf = torch.randn(seq, HIDDEN, device=DEV, dtype=torch.float32,
                          generator=g).to(torch.bfloat16)
        return buf.view(1, seq, HEADS, DIM).permute(0, 2, 1, 3)
    return one(), one(), one()


def ref_rows(q, k, v, head, r0, r1, scale):
    qf = q[0, head, r0:r1].float()
    kf = k[0, head].float()
    vf = v[0, head].float()
    scores = (qf @ kf.T) * scale
    return torch.softmax(scores, dim=-1) @ vf


def main():
    results = []
    for seq in (23638, 37747, 73472, 109099):
        q, k, v = make(seq, 1000 + seq)
        scale = DIM ** -0.5
        torch.cuda.synchronize()
        t0 = time.monotonic()
        out = mod.fwd_bhsd(q, k, v, scale)
        torch.cuda.synchronize()
        dt = time.monotonic() - t0
        # second timed call (warm)
        t0 = time.monotonic()
        out = mod.fwd_bhsd(q, k, v, scale)
        torch.cuda.synchronize()
        warm = time.monotonic() - t0
        bad = (~torch.isfinite(out.float())).sum().item()
        max_err = 0.0
        samples = []
        for head in (0, 13, 27):
            for r0 in (0, seq // 2, max(0, seq - CHUNK)):
                r1 = min(seq, r0 + CHUNK)
                ref = ref_rows(q, k, v, head, r0, r1, scale)
                got = out[0, head, r0:r1].float()
                err = (got - ref).abs().max().item()
                max_err = max(max_err, err)
                samples.append({"head": head, "rows": [r0, r1], "max_abs": err})
        results.append({
            "seq": seq, "warm_s": round(warm, 4), "cold_s": round(dt, 4),
            "nonfinite": bad, "sampled_max_abs_err": max_err,
        })
        print(json.dumps(results[-1]), flush=True)
        del q, k, v, out
        torch.cuda.empty_cache()
    print("SUMMARY " + json.dumps(results))


if __name__ == "__main__":
    sys.exit(main())
