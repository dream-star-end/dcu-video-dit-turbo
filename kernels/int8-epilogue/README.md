# H3 gfx936 fused INT8 epilogue micro-experiment

Scratch-only Torch/HIP extension for the current eager epilogue:

1. `int32 accumulator -> float32`
2. `combined_scale = row_scale * col_scale` in float32
3. `scaled = accumulator * combined_scale` in float32
4. round to BF16
5. optional bias rounded to BF16, BF16 add, round to BF16

The extension is compiled without fast math, unsafe reassociation, or FP contraction.
It does not modify ComfyUI, the runtime venv, model weights, or the production worker.
