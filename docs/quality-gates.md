# Quality gates

## Hard media contract

For the default 15-second profile require:

- Video: H.264, yuv420p, 608×352, 24 fps, 362 frames.
- Audio: AAC, 32000 Hz, 2 channels.
- Container duration: 15.083333 seconds within normal mux rounding.
- Decoding succeeds for all frames and the audio stream is non-empty.

## Conditioning adherence

Generate a first/middle/last contact sheet and inspect it. Use exact node preprocessing before comparing anchors. H3 keyframes are conditioning anchors, not pixel-copy constraints; compressed frame 0/last frame need not equal the input pixels.

For a regression comparison against a same-seed canonical baseline, prefer exact decoded RGB/PCM equality. If backend scheduling prevents exact media identity, document it and compare endpoint adherence, temporal health, and semantics rather than hiding the difference behind a single average score. Do not compare an I2V result to an FLF result as if one were the other's baseline.

The audited one-seed comparisons are evidence, not a universal threshold. I2V reached whole-video YUV SSIM 0.953625 and PSNR 32.841 dB against canonical single-rank. FLF reached 0.829264 and 26.916 dB overall while its first/last frame SSIM remained 0.963506/0.956163; the middle trajectory diverged stochastically. Therefore the optimized FLF sample preserves both requested anchors and media health but is not bit-exact or perceptually identical frame-for-frame. Do not describe the full distributed pipeline as mathematically lossless.

## Temporal checks

- Reject black/corrupt frames.
- Flag long runs of nearly identical adjacent frames.
- Inspect motion around the first and last 0.5 seconds for hard cuts, anchor snapping, or sudden geometry changes.
- Check subject identity, limb/umbrella geometry, background train continuity, camera direction, and shot composition.

## Audio checks

- Require finite PCM samples and nonzero RMS.
- Flag clipping, large DC offset, unexpected silence, or missing stereo.
- Listen for the requested rain/footsteps/train/umbrella cues; numeric presence alone does not prove semantic adherence.

## Claim strength

One seed proves only that sample. Use at least three representative seeds before claiming a mode is generally quality-neutral. Kernel-level bitwise validation supports arithmetic substitutions but does not replace media/semantic evaluation of the complete distributed pipeline.
