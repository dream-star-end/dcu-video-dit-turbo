"""Launch ComfyUI with quality-first gfx936 INT8 and FlashAttention optimizations.

The candidate combines the verified no-copy INT8 weight transpose path with a
scratch-built HIP epilogue that performs the exact eager FP32 scale ordering and
BF16 rounding in one kernel.  It does not modify the installed runtime.

The optional FlashAttention path is restricted to the exact MiniMax H3
inference contract. Unsupported shapes, masks, dtypes, layouts, GQA, training,
or FP32-upcast requests fall back to ComfyUI's original implementation.
"""

from __future__ import annotations

import logging
import importlib.util
import os
import runpy
import signal
import sys
import time

COMFY_ROOT = os.environ.get("H3_COMFY_ROOT", "/path/to/ComfyUI")
sys.path.insert(0, COMFY_ROOT)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402

try:
    _tv_lib = torch.library.Library("torchvision", "DEF")
    _tv_lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    _tv_lib = None

import comfy_kitchen.backends.eager as _eager  # noqa: E402
import comfy_kitchen.backends.eager.quantization as _q  # noqa: E402
import comfy_kitchen.tensor.base as _qt_base  # noqa: E402
from comfy_kitchen.tensor.base import QuantizedTensor as _QuantizedTensor  # noqa: E402
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as _TensorWiseINT8Layout  # noqa: E402


if os.environ.get("H3_BENCH_NONCE_INPUT", "0") == "1":
    # Add an ignored cache key to RandomNoise.  Varying this value reruns only
    # noise/sampling/decoding while model loaders and text conditioning remain
    # cached, giving a true hot same-seed A/B without changing generated noise.
    import comfy_extras.nodes_custom_sampler as _custom_sampler  # noqa: E402

    _OriginalRandomNoise = _custom_sampler.RandomNoise
    _io = _custom_sampler.io

    class _BenchmarkRandomNoise(_OriginalRandomNoise):
        @classmethod
        def define_schema(cls):
            return _io.Schema(
                node_id="RandomNoise",
                category="model/sampling/noise",
                inputs=[
                    _io.Int.Input(
                        "noise_seed",
                        default=0,
                        min=0,
                        max=0xFFFFFFFFFFFFFFFF,
                        control_after_generate=True,
                    ),
                    _io.Int.Input(
                        "benchmark_nonce",
                        default=0,
                        min=0,
                        max=0x7FFFFFFF,
                        optional=True,
                    ),
                ],
                outputs=[_io.Noise.Output()],
            )

        @classmethod
        def execute(cls, noise_seed, benchmark_nonce=0):
            del benchmark_nonce
            return _OriginalRandomNoise.execute(noise_seed)

        get_noise = execute

    _custom_sampler.RandomNoise = _BenchmarkRandomNoise


_original_int8_linear = _eager.int8_linear
_no_copy_enabled = os.environ.get("H3_INT8_NOCOPY", "0") == "1"
_unpadded_m_enabled = os.environ.get("H3_INT8_UNPADDED_M", "0") == "1"
_prepack_weight_enabled = os.environ.get("H3_INT8_PREPACK_WEIGHT", "0") == "1"
_epilogue_enabled = os.environ.get("H3_INT8_EPILOGUE", "0") == "1"
_epilogue_so = os.environ.get(
    "H3_INT8_EPILOGUE_SO",
    "kernels/int8-epilogue/build/h3_hip_epilogue.so",
)
_epilogue = None
_empty_bias_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_unpadded_m_hits = 0
_prepacked_weight_count = 0

# Exact 15-second, 608x352 sequence-parallel GEMMs.  comfy-kitchen pads M to a
# multiple of 32 by concatenating a full new activation tensor.  gfx936's
# torch._int_mm accepts M=11819 directly, and all four outputs have been checked
# element-for-element against the padded implementation.  Keep this gate narrow
# so other durations and model variants continue through the upstream helper.
_H3_15S_INT8_SHAPES = {
    (11819, 5376, 21504),
    (11819, 7168, 5376),
    (11819, 5376, 28672),
    (11819, 14336, 5376),
}

_H3_INT8_WEIGHT_SHAPES = {
    (21504, 5376),
    (5376, 7168),
    (28672, 5376),
    (5376, 14336),
}


def _is_gfx936(tensor: torch.Tensor) -> bool:
    if not tensor.is_cuda:
        return False
    try:
        props = torch.cuda.get_device_properties(tensor.device)
        return props.gcnArchName.split(":", 1)[0] == "gfx936"
    except (AttributeError, RuntimeError):
        return False


def _prepack_weight_storage(weight: torch.Tensor) -> torch.Tensor:
    """Replace [N,K] row-major storage with an equivalent column-major view.

    The logical tensor and every INT8 value remain unchanged, while weight.T is
    contiguous for rocBLAS.  set_ transfers the persistent tensor to the new
    storage, so the old 19-GiB model-wide layout is released layer by layer
    instead of being retained as a second cache.
    """
    global _prepacked_weight_count
    eligible = (
        _prepack_weight_enabled
        and weight.dtype == torch.int8
        and weight.dim() == 2
        and tuple(weight.shape) in _H3_INT8_WEIGHT_SHAPES
        and _is_gfx936(weight)
    )
    if not eligible or weight.T.is_contiguous():
        return weight
    if not weight.is_contiguous():
        # Unknown layout: preserve upstream behavior rather than reinterpret it.
        return weight
    packed_view = weight.T.contiguous().T
    weight.set_(packed_view)
    _prepacked_weight_count += 1
    if _prepacked_weight_count == 1 or _prepacked_weight_count % 50 == 0:
        logging.warning(
            "H3 exact INT8 weight prepack count=%d shape=%s stride=%s",
            _prepacked_weight_count,
            tuple(weight.shape),
            tuple(weight.stride()),
        )
    return weight


# Standard torch linear reaches INT8 through QuantizedTensor's layout dispatch,
# which normally forces qdata.contiguous() before the custom op.  Preserve the
# original handler for every other tensor, and pass the verified H3 layouts
# through without undoing the persistent prepack.
_linear_op = torch.ops.aten.linear.default
_original_int8_layout_handler = _qt_base._LAYOUT_DISPATCH_TABLE[_linear_op][
    _TensorWiseINT8Layout
]
_original_get_plain_tensors = _TensorWiseINT8Layout.get_plain_tensors.__func__


@classmethod
def _get_plain_tensors_prepacked(cls, qtensor):
    qdata, scale = _original_get_plain_tensors(cls, qtensor)
    return _prepack_weight_storage(qdata), scale


def _h3_int8_layout_handler(qt, args, kwargs):
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None
    eligible = (
        _prepack_weight_enabled
        and isinstance(weight, _QuantizedTensor)
        and weight._layout_cls == "TensorWiseINT8Layout"
        and not getattr(weight._params, "transposed", False)
        and not isinstance(input_tensor, _QuantizedTensor)
        and tuple(weight._qdata.shape) in _H3_INT8_WEIGHT_SHAPES
        and _is_gfx936(weight._qdata)
    )
    if not eligible:
        return _original_int8_layout_handler(qt, args, kwargs)
    qdata, scale = _TensorWiseINT8Layout.get_plain_tensors(weight)
    if not qdata.T.is_contiguous():
        return _original_int8_layout_handler(qt, args, kwargs)
    out_dtype = kwargs.get("out_dtype", input_tensor.dtype)
    return torch.ops.comfy_kitchen.int8_linear(
        input_tensor.contiguous(),
        qdata,
        scale,
        bias,
        _q.DTYPE_TO_CODE[out_dtype],
        getattr(weight._params, "convrot", False),
        getattr(weight._params, "convrot_groupsize", 256),
    )


if _prepack_weight_enabled:
    _TensorWiseINT8Layout.get_plain_tensors = _get_plain_tensors_prepacked
    _qt_base._LAYOUT_DISPATCH_TABLE[_linear_op][
        _TensorWiseINT8Layout
    ] = _h3_int8_layout_handler


if os.environ.get("H3_TIMING_NODES", "0") == "1":
    # Diagnostic-only wall timings.  Synchronizing at node boundaries makes
    # asynchronous HIP work attributable to the correct VAE/save stage.
    import nodes as _nodes  # noqa: E402
    import comfy_extras.nodes_audio as _nodes_audio  # noqa: E402
    import comfy_extras.nodes_video as _nodes_video  # noqa: E402

    def _sync_for_node_timing():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    _original_video_decode = _nodes.VAEDecode.decode

    def _timed_video_decode(self, *args, **kwargs):
        _sync_for_node_timing()
        started = time.perf_counter()
        value = _original_video_decode(self, *args, **kwargs)
        _sync_for_node_timing()
        logging.warning("H3_NODE_TIMING video_vae_seconds=%.6f", time.perf_counter() - started)
        return value

    _nodes.VAEDecode.decode = _timed_video_decode

    _original_audio_function = _nodes_audio.vae_decode_audio

    def _timed_audio_function(*args, **kwargs):
        _sync_for_node_timing()
        started = time.perf_counter()
        value = _original_audio_function(*args, **kwargs)
        _sync_for_node_timing()
        logging.warning("H3_NODE_TIMING audio_vae_seconds=%.6f", time.perf_counter() - started)
        return value

    _nodes_audio.vae_decode_audio = _timed_audio_function

    _original_audio_execute = _nodes_audio.VAEDecodeAudio.execute

    @classmethod
    def _timed_audio_execute(cls, *args, **kwargs):
        _sync_for_node_timing()
        started = time.perf_counter()
        value = _original_audio_execute(*args, **kwargs)
        _sync_for_node_timing()
        logging.warning("H3_NODE_TIMING audio_vae_seconds=%.6f", time.perf_counter() - started)
        return value

    _nodes_audio.VAEDecodeAudio.execute = _timed_audio_execute
    _nodes_audio.VAEDecodeAudio.decode = _timed_audio_execute

    _original_save_execute = _nodes_video.SaveVideo.execute.__func__

    @classmethod
    def _timed_save_execute(cls, *args, **kwargs):
        if (
            os.environ.get("H3_VAE_SP_CHUNKS", "0") == "1"
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() == 1
        ):
            logging.warning("H3_VAE_SP rank=1 SaveVideo skipped")
            video = kwargs.get("video", args[0] if args else None)
            return _nodes_video.io.NodeOutput(video)
        _sync_for_node_timing()
        started = time.perf_counter()
        value = _original_save_execute(cls, *args, **kwargs)
        _sync_for_node_timing()
        logging.warning("H3_NODE_TIMING save_video_seconds=%.6f", time.perf_counter() - started)
        return value

    _nodes_video.SaveVideo.execute = _timed_save_execute


if os.environ.get("H3_VAE_SP_CHUNKS", "0") == "1":
    import torch.distributed as _dist  # noqa: E402
    from comfy.ldm.minimax import vae as _h3_vae  # noqa: E402
    import comfy_extras.nodes_video as _nodes_video_sp  # noqa: E402

    _original_decode_temporal = _h3_vae.MiniMaxH3VideoVAE.decode_temporal
    _validate_vae_sp = os.environ.get("H3_VAE_SP_VALIDATE", "0") == "1"

    def _decode_temporal_parts(self, z, index):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        start = index * self.tokens_chunk_size
        clip_z = z[
            :,
            :,
            start:start + self.tokens_chunk_size + self.token_overlap,
        ]
        clip_dec = self._adaptive_decode(clip_z)
        main = clip_dec[:, :, :chunk_dec]
        main = main[:, :, self.frame_pre_padding:].contiguous()
        overlap = clip_dec[:, :, chunk_dec:]
        overlap = overlap[:, :, self.frame_pre_padding:].contiguous()
        if main.shape[2] != 17 or overlap.shape[2] != 5:
            raise RuntimeError(
                f"unexpected H3 VAE chunk parts {tuple(main.shape)} {tuple(overlap.shape)}"
            )
        return main, overlap

    def _packed_temporal_part(self, z, index):
        main, overlap = _decode_temporal_parts(self, z, index)
        return torch.cat((main, overlap), dim=2)

    def _sp_decode_temporal(self, z):
        eligible = (
            _dist.is_available()
            and _dist.is_initialized()
            and _dist.get_world_size() == 2
            and tuple(z.shape) == (1, 24, 107, 22, 38)
            and z.dtype == torch.float16
            and self.tokens_chunk_size == 5
            and self.token_overlap == 2
            and self.frame_pre_padding == 3
            and self.frame_overlap == 5
        )
        if not eligible:
            return _original_decode_temporal(self, z)

        rank = _dist.get_rank()
        z = z.contiguous()
        _dist.broadcast(z, src=0)
        num_chunks = 21
        cut = 11
        output_frames = self._decode_temporal_frame_plan(z.shape[2], num_chunks, 0)
        if output_frames != 362:
            raise RuntimeError(f"unexpected H3 VAE output frame plan {output_frames}")

        torch.cuda.synchronize()
        started = time.perf_counter()
        if rank == 1:
            packets = [
                _packed_temporal_part(self, z, index)
                for index in range(cut, num_chunks)
            ]
            torch.cuda.synchronize()
            local_seconds = time.perf_counter() - started
            sent_started = time.perf_counter()
            for packet in packets:
                _dist.send(packet, dst=0)
            torch.cuda.synchronize()
            logging.warning(
                "H3_VAE_SP_TIMING rank=1 chunks=10 local_seconds=%.6f send_seconds=%.6f",
                local_seconds,
                time.perf_counter() - sent_started,
            )
            return torch.zeros(
                (1, 3, 1, 352, 608),
                device=z.device,
                dtype=packets[0].dtype,
            )

        dec = None
        write_pos = 0
        previous_overlap = None

        def write_part(part):
            nonlocal dec, write_pos
            frames = part.shape[2]
            if frames <= 0:
                return
            if dec is None:
                shape = list(part.shape)
                shape[2] = output_frames
                dec = torch.empty(shape, device=part.device, dtype=part.dtype)
            copy_frames = min(frames, max(0, output_frames - write_pos))
            if copy_frames:
                dec[:, :, write_pos:write_pos + copy_frames].copy_(
                    part[:, :, :copy_frames]
                )
                write_pos += copy_frames

        for index in range(cut):
            main, overlap = _decode_temporal_parts(self, z, index)
            if previous_overlap is not None:
                main = self.blend(
                    previous_overlap,
                    main,
                    self.frame_overlap,
                    dim=-3,
                )
            write_part(main)
            previous_overlap = overlap
        torch.cuda.synchronize()
        local_seconds = time.perf_counter() - started

        packet_shape = (
            previous_overlap.shape[0],
            previous_overlap.shape[1],
            22,
            previous_overlap.shape[3],
            previous_overlap.shape[4],
        )
        received = []
        receive_started = time.perf_counter()
        for _ in range(cut, num_chunks):
            packet = torch.empty(
                packet_shape,
                device=z.device,
                dtype=previous_overlap.dtype,
            )
            _dist.recv(packet, src=1)
            received.append(packet)
        torch.cuda.synchronize()
        receive_seconds = time.perf_counter() - receive_started

        if _validate_vae_sp:
            for index, packet in zip(range(cut, num_chunks), received):
                reference = _packed_temporal_part(self, z, index)
                if not torch.equal(reference, packet):
                    max_abs = (reference.float() - packet.float()).abs().max().item()
                    raise RuntimeError(
                        f"rank1 H3 VAE chunk {index} is not bitwise exact; max_abs={max_abs}"
                    )

        for packet in received:
            main = packet[:, :, :17]
            overlap = packet[:, :, 17:]
            main = self.blend(
                previous_overlap,
                main,
                self.frame_overlap,
                dim=-3,
            )
            write_part(main)
            previous_overlap = overlap
        write_part(previous_overlap)
        torch.cuda.synchronize()
        if write_pos != output_frames:
            raise RuntimeError(
                f"assembled {write_pos} H3 VAE frames, expected {output_frames}"
            )
        logging.warning(
            "H3_VAE_SP_TIMING rank=0 chunks=11 local_seconds=%.6f "
            "receive_seconds=%.6f total_seconds=%.6f validate=%s",
            local_seconds,
            receive_seconds,
            time.perf_counter() - started,
            _validate_vae_sp,
        )
        return dec

    _h3_vae.MiniMaxH3VideoVAE.decode_temporal = _sp_decode_temporal

    if os.environ.get("H3_TIMING_NODES", "0") != "1":
        _original_sp_save_execute = _nodes_video_sp.SaveVideo.execute.__func__

        @classmethod
        def _sp_save_execute(cls, *args, **kwargs):
            if _dist.is_initialized() and _dist.get_rank() == 1:
                logging.warning("H3_VAE_SP rank=1 SaveVideo skipped")
                video = kwargs.get("video", args[0] if args else None)
                return _nodes_video_sp.io.NodeOutput(video)
            return _original_sp_save_execute(cls, *args, **kwargs)

        _nodes_video_sp.SaveVideo.execute = _sp_save_execute


# SaveVideo's stock 8-bit path launches three tiny GPU operations and performs
# one synchronous D2H copy per frame.  Quantizing the complete tensor once uses
# exactly the same elementwise expression and feeds the same RGB24 frames to the
# same PyAV/H.264 encoder, but replaces 362 launch/copy round trips with one.
if os.environ.get("H3_SAVE_BATCH_U8", "0") == "1":
    from comfy_api.latest._input_impl import video_types as _h3_video_types  # noqa: E402

    _original_video_components_save_to = _h3_video_types.VideoFromComponents.save_to

    def _batch_u8_video_save_to(
        self,
        path,
        format=_h3_video_types.VideoContainer.AUTO,
        codec=_h3_video_types.VideoCodec.AUTO,
        metadata=None,
        bit_depth=None,
        crf=None,
    ):
        components = self._VideoFromComponents__components
        if bit_depth is None:
            bit_depth = self._VideoFromComponents__bit_depth
        images = components.images
        eligible = (
            bit_depth < 10
            and isinstance(images, torch.Tensor)
            and images.ndim == 4
            and images.shape[-1] == 3
        )
        if not eligible:
            return _original_video_components_save_to(
                self,
                path,
                format=format,
                codec=codec,
                metadata=metadata,
                bit_depth=bit_depth,
                crf=crf,
            )

        total_started = time.perf_counter()
        # This is the stock per-frame expression evaluated over one leading
        # batch dimension.  ``contiguous`` changes layout only, never values.
        image_started = time.perf_counter()
        images_u8 = (
            (images * 255).clamp(0, 255).byte().contiguous().cpu().numpy()
        )
        image_seconds = time.perf_counter() - image_started

        open_kwargs = _h3_video_types.mp4_output_open_kwargs(path, format, codec)
        encode_started = time.perf_counter()
        audio_started = None
        with _h3_video_types.av.open(path, **open_kwargs) as output:
            if metadata is not None:
                for key, value in metadata.items():
                    output.metadata[key] = _h3_video_types.json.dumps(value)

            frame_rate = _h3_video_types.Fraction(
                round(components.frame_rate * 1000), 1000
            )
            video_stream = output.add_stream("h264", rate=frame_rate)
            video_stream.width = images.shape[2]
            video_stream.height = images.shape[1]
            video_stream.pix_fmt = "yuv420p"
            if crf is not None:
                video_stream.options = {"crf": str(crf)}

            audio_sample_rate = 1
            audio_stream = None
            waveform = None
            layout = None
            if components.audio:
                audio_sample_rate = int(components.audio["sample_rate"])
                waveform = components.audio["waveform"]
                waveform = waveform[
                    0,
                    :,
                    :_h3_video_types.math.ceil(
                        (audio_sample_rate / frame_rate) * images.shape[0]
                    ),
                ]
                layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(
                    waveform.shape[0], "stereo"
                )
                audio_stream = output.add_stream(
                    "aac", rate=audio_sample_rate, layout=layout
                )

            for image in images_u8:
                frame = _h3_video_types.av.VideoFrame.from_ndarray(
                    image, format="rgb24"
                )
                frame = frame.reformat(format="yuv420p")
                output.mux(video_stream.encode(frame))
            output.mux(video_stream.encode(None))
            video_seconds = time.perf_counter() - encode_started

            if audio_stream is not None and waveform is not None:
                audio_started = time.perf_counter()
                audio_frame = _h3_video_types.av.AudioFrame.from_ndarray(
                    waveform.float().cpu().contiguous().numpy(),
                    format="fltp",
                    layout=layout,
                )
                audio_frame.sample_rate = audio_sample_rate
                audio_frame.pts = 0
                output.mux(audio_stream.encode(audio_frame))
                output.mux(audio_stream.encode(None))

        audio_seconds = (
            0.0 if audio_started is None else time.perf_counter() - audio_started
        )
        logging.warning(
            "H3_BATCH_SAVE_TIMING quantize_d2h_seconds=%.6f "
            "video_encode_seconds=%.6f audio_encode_seconds=%.6f total_seconds=%.6f",
            image_seconds,
            video_seconds,
            audio_seconds,
            time.perf_counter() - total_started,
        )

    _h3_video_types.VideoFromComponents.save_to = _batch_u8_video_save_to


# Built-in extra nodes are loaded by ComfyUI from file paths under synthetic
# module names, so patching ``comfy_extras.nodes_video`` or ``nodes_audio``
# above does not affect the class objects that the executor registers.  Hook
# the registry initialization and patch those final class objects once they
# exist.  This both gives accurate tail timings and prevents rank 1 from
# encoding a one-frame dummy video after it has supplied its VAE chunks.
if (
    os.environ.get("H3_TIMING_NODES", "0") == "1"
    or os.environ.get("H3_VAE_SP_CHUNKS", "0") == "1"
):
    import nodes as _h3_node_registry  # noqa: E402
    from comfy_api.latest import io as _h3_node_io  # noqa: E402

    _original_init_extra_nodes = _h3_node_registry.init_extra_nodes
    _late_timing_enabled = os.environ.get("H3_TIMING_NODES", "0") == "1"
    _late_vae_sp_enabled = os.environ.get("H3_VAE_SP_CHUNKS", "0") == "1"

    async def _h3_init_extra_nodes(*args, **kwargs):
        value = await _original_init_extra_nodes(*args, **kwargs)

        audio_cls = _h3_node_registry.NODE_CLASS_MAPPINGS.get("VAEDecodeAudio")
        if _late_timing_enabled and audio_cls is not None:
            original_audio_execute = audio_cls.execute.__func__

            @classmethod
            def timed_audio_execute(cls, *execute_args, **execute_kwargs):
                _sync_for_node_timing()
                started = time.perf_counter()
                result = original_audio_execute(cls, *execute_args, **execute_kwargs)
                _sync_for_node_timing()
                logging.warning(
                    "H3_NODE_TIMING audio_vae_seconds=%.6f",
                    time.perf_counter() - started,
                )
                return result

            audio_cls.execute = timed_audio_execute
            audio_cls.decode = timed_audio_execute

        save_cls = _h3_node_registry.NODE_CLASS_MAPPINGS.get("SaveVideo")
        if save_cls is not None:
            original_save_execute = save_cls.execute.__func__

            @classmethod
            def timed_or_sp_save_execute(cls, *execute_args, **execute_kwargs):
                if (
                    _late_vae_sp_enabled
                    and torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                    and torch.distributed.get_rank() == 1
                ):
                    logging.warning("H3_VAE_SP rank=1 SaveVideo skipped")
                    return _h3_node_io.NodeOutput()
                if not _late_timing_enabled:
                    return original_save_execute(cls, *execute_args, **execute_kwargs)
                _sync_for_node_timing()
                started = time.perf_counter()
                result = original_save_execute(cls, *execute_args, **execute_kwargs)
                _sync_for_node_timing()
                logging.warning(
                    "H3_NODE_TIMING save_video_seconds=%.6f",
                    time.perf_counter() - started,
                )
                return result

            save_cls.execute = timed_or_sp_save_execute

        logging.warning(
            "H3 late node patch audio=%s save=%s timing=%s vae_sp=%s",
            audio_cls is not None,
            save_cls is not None,
            _late_timing_enabled,
            _late_vae_sp_enabled,
        )
        return value

    _h3_node_registry.init_extra_nodes = _h3_init_extra_nodes


def _int8_matmul_h3(x_8: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
    global _unpadded_m_hits
    shape = (x_8.shape[0], x_8.shape[1], weight_t.shape[1])
    eligible = (
        _unpadded_m_enabled
        and x_8.is_cuda
        and x_8.dtype == torch.int8
        and weight_t.is_cuda
        and weight_t.dtype == torch.int8
        and x_8.dim() == 2
        and weight_t.dim() == 2
        and x_8.shape[1] == weight_t.shape[0]
        and shape in _H3_15S_INT8_SHAPES
    )
    if eligible:
        _unpadded_m_hits += 1
        if _unpadded_m_hits == 1:
            logging.warning("H3 exact unpadded-M INT8 GEMM ENABLED for %s", shape)
        return torch._int_mm(x_8, weight_t)
    return _q._int8_matmul_accumulate(x_8, weight_t)


def _load_epilogue():
    global _epilogue
    if _epilogue is not None:
        return _epilogue
    spec = importlib.util.spec_from_file_location("h3_hip_epilogue", _epilogue_so)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load H3 HIP epilogue from {_epilogue_so}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _epilogue = module
    return module


def _empty_bias(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    value = _empty_bias_cache.get(key)
    if value is None:
        value = torch.empty((0,), device=device, dtype=dtype)
        _empty_bias_cache[key] = value
    return value


def _int8_linear_nocopy(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
) -> torch.Tensor:
    """Eager INT8 linear without materializing weight.T on every invocation.

    DTK's torch._int_mm accepts the column-major transpose view directly. This
    retains the exact INT8 operands and accumulation while removing a full
    device-to-device copy of every quantized weight matrix on every denoise step.
    """
    x = _q._apply_input_act(x, input_act)
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {x.shape[-1]} and {weight.shape[-1]}"
        )

    weight = weight.to(device=x.device)
    weight = _prepack_weight_storage(weight)
    if not weight.T.is_contiguous():
        weight = weight.contiguous()
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1)
    if weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {tuple(weight_scale.shape)} "
            f"for weight shape {tuple(weight.shape)}"
        )

    if convrot:
        if x.shape[-1] % convrot_groupsize != 0:
            raise ValueError(
                f"ConvRot group size {convrot_groupsize} does not divide input features {x.shape[-1]}"
            )
        h = _q._build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = _q._rotate_activation(x, h, convrot_groupsize)

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    x_8, x_scale = _q.quantize_int8_rowwise(x_2d)

    # The only intentional change from comfy-kitchen 0.2.26 eager.int8_linear:
    # pass the transpose view rather than allocating weight.T.contiguous().
    result = _int8_matmul_h3(x_8, weight.T)

    if _epilogue_enabled and out_dtype == torch.bfloat16:
        fused_weight_scale = weight_scale.reshape(-1)
        if fused_weight_scale.numel() == 1:
            fused_weight_scale = fused_weight_scale.expand(result.shape[1]).contiguous()
        else:
            fused_weight_scale = fused_weight_scale.contiguous()
        fused_bias = (
            _empty_bias(result.device, out_dtype)
            if bias is None
            else bias.to(device=result.device).reshape(-1).contiguous()
        )
        if fused_bias.dtype not in (torch.bfloat16, torch.float32):
            fused_bias = fused_bias.to(torch.bfloat16)
        result = _load_epilogue().epilogue(
            result,
            x_scale.reshape(-1).contiguous(),
            fused_weight_scale,
            fused_bias,
        )
    else:
        m, n = result.shape
        chunk_size = max(1, min(m, 256 * 1024 * 1024 // (n * 4)))
        weight_scale_2d = weight_scale.reshape(1, -1)
        scaled_parts = []
        for i in range(0, m, chunk_size):
            end_i = min(i + chunk_size, m)
            chunk = result[i:end_i].float()
            chunk_scales = x_scale[i:end_i].to(device=chunk.device, dtype=torch.float32) * weight_scale_2d
            scaled_parts.append((chunk * chunk_scales).to(out_dtype))
        result = torch.cat(scaled_parts, dim=0)

        if bias is not None:
            result = result + bias.to(device=result.device, dtype=result.dtype).reshape(1, -1)
    return result.reshape(*orig_shape[:-1], weight.shape[0])


def _int8_linear_dispatch(*args, **kwargs):
    if _no_copy_enabled:
        return _int8_linear_nocopy(*args, **kwargs)
    return _original_int8_linear(*args, **kwargs)


def _set_nocopy(enabled: bool):
    global _no_copy_enabled
    _no_copy_enabled = enabled
    logging.warning("H3 INT8 no-copy path %s", "ENABLED" if enabled else "DISABLED")


def _enable_nocopy(_signum, _frame):
    _set_nocopy(True)


def _disable_nocopy(_signum, _frame):
    _set_nocopy(False)


# Registry dispatch resolves the implementation dynamically from this module.
_eager.int8_linear = _int8_linear_dispatch
_q.int8_linear = _int8_linear_dispatch
signal.signal(signal.SIGUSR1, _enable_nocopy)
signal.signal(signal.SIGUSR2, _disable_nocopy)


_flash_enabled = os.environ.get("H3_FLASH_ATTN", "0") == "1"
_flash_so = os.environ.get(
    "H3_FLASH_ATTN_SO",
    "kernels/flash-attention-gfx936/build/"
    "flash_attn_hg_forward_gfx936/libflash_attention.so",
)
_flash_extension = None
_flash_first_call = True
_flash_fallback_signatures = set()
_flash_candidate_calls = 0
_flash_fallback_calls = 0
_flash_head_chunk_calls = 0
_head_chunk_overlap_enabled = os.environ.get("H3_SP_HEAD_CHUNK_OVERLAP", "0") == "1"
_head_chunk_flash_contract = "gfx936-bf16-world2-s23638-h14-qkv3-v1"
_device_arch_cache = {}


def _load_flash_extension():
    global _flash_extension
    if _flash_extension is not None:
        return _flash_extension
    spec = importlib.util.spec_from_file_location("h3_flash_attn", _flash_so)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load H3 FlashAttention from {_flash_so}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _flash_extension = module
    return module


def _device_arch(device: torch.device) -> str:
    key = device.index if device.index is not None else torch.cuda.current_device()
    arch = _device_arch_cache.get(key)
    if arch is None:
        arch = getattr(
            torch.cuda.get_device_properties(device), "gcnArchName", ""
        ).split(":", 1)[0]
        _device_arch_cache[key] = arch
    return arch


if _flash_enabled:
    import comfy.ldm.modules.attention as _attention  # noqa: E402
    import comfy.ldm.minimax.model as _h3_model  # noqa: E402
    from comfy.ldm.minimax import sequence_parallel as _h3_sequence_parallel  # noqa: E402

    _original_h3_attention = _h3_model.optimized_attention
    if _original_h3_attention is not _attention.attention_sub_quad:
        raise RuntimeError(
            "H3 FlashAttention requires ComfyUI's sub-quadratic backend to be selected"
        )
    # Import immediately so a missing or ABI-incompatible extension fails at
    # worker startup rather than silently falling back during a benchmark.
    _load_flash_extension()

    @_attention.wrap_attn
    def _h3_flash_attention(
        query,
        key,
        value,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        """Use the audited H3 kernel only when its complete contract matches."""
        global _flash_candidate_calls, _flash_fallback_calls, _flash_head_chunk_calls
        resolved_precision = _attention.get_attn_precision(attn_precision, query.dtype)
        tensors = (query, key, value)
        transformer_options = kwargs.get("transformer_options")
        sp_context = (
            _h3_sequence_parallel.attention_context(transformer_options)
            if isinstance(transformer_options, dict)
            else None
        )
        standard_layout = (
            heads == 28
            and query.ndim == 4
            and query.shape[0] == 1
            and query.shape[1] == 28
            and 1 <= query.shape[2] <= 50000
            and query.shape[3] == 128
            and all(tuple(t.stride()[1:]) == (128, 3584, 1) for t in tensors)
        )
        exact_head_chunk_layout = (
            _head_chunk_overlap_enabled
            and heads == 14
            and query.device.type == "cuda"
            and query.ndim == 4
            and tuple(query.shape) == (1, 14, 23638, 128)
            and all(tuple(t.stride()[1:]) == (384, 5376, 1) for t in tensors)
            and getattr(sp_context, "sequence_splits", None) == (11819, 11819)
            and getattr(sp_context, "local_length", None) == 11819
            and _device_arch(query.device) == "gfx936"
        )
        eligible = (
            mask is None
            and skip_reshape
            and "scale" not in kwargs
            and not kwargs.get("enable_gqa", False)
            and resolved_precision != torch.float32
            and sp_context is not None
            and sp_context.world_size == 2
            and query.dtype == torch.bfloat16
            and query.device.type == "cuda"
            and query.shape == key.shape == value.shape
            and all(t.device == query.device for t in tensors)
            and all(t.dtype == query.dtype for t in tensors)
            and not any(t.requires_grad for t in tensors)
            and (standard_layout or exact_head_chunk_layout)
        )
        if not eligible:
            _flash_fallback_calls += 1
            fallback_signature = (
                tuple(query.shape),
                tuple(tuple(t.stride()) for t in tensors),
                heads,
                mask is not None,
                skip_reshape,
                "scale" in kwargs,
                str(resolved_precision),
                getattr(sp_context, "world_size", None),
                tuple(t.requires_grad for t in tensors),
            )
            if fallback_signature not in _flash_fallback_signatures and len(_flash_fallback_signatures) < 8:
                _flash_fallback_signatures.add(fallback_signature)
                logging.warning(
                    "H3 gfx936 FlashAttention FALLBACK signature shape=%s strides=%s "
                    "heads=%s mask=%s skip_reshape=%s scale_present=%s precision=%s "
                    "sp_rank=%s sp_world=%s requires_grad=%s",
                    tuple(query.shape),
                    tuple(tuple(t.stride()) for t in tensors),
                    heads,
                    mask is not None,
                    skip_reshape,
                    "scale" in kwargs,
                    resolved_precision,
                    getattr(sp_context, "rank", None),
                    getattr(sp_context, "world_size", None),
                    tuple(t.requires_grad for t in tensors),
                )
            return _original_h3_attention(
                query,
                key,
                value,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        global _flash_first_call
        _flash_candidate_calls += 1
        if exact_head_chunk_layout:
            _flash_head_chunk_calls += 1
        output = _load_flash_extension().fwd_bhsd(
            query,
            key,
            value,
            float(kwargs.get("scale", query.shape[-1] ** -0.5)),
        )
        if _flash_first_call:
            # Synchronize once so an incompatible launch fails at the feature
            # boundary instead of surfacing asynchronously later in sampling.
            torch.cuda.synchronize(query.device)
            logging.warning(
                "H3 gfx936 FlashAttention ENABLED rank=%s shape=%s stride=%s call=%s",
                getattr(sp_context, "rank", None),
                tuple(query.shape),
                tuple(query.stride()),
                _flash_candidate_calls,
            )
            _flash_first_call = False
        if skip_output_reshape:
            return output
        return output.transpose(1, 2).flatten(start_dim=2)

    # Restrict the patch to MiniMax H3; all other ComfyUI models retain their
    # selected attention backend even if they happen to have a matching shape.
    _h3_model.optimized_attention = _h3_flash_attention
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        local_flag = torch.tensor(
            [int(_head_chunk_overlap_enabled)],
            device=torch.device("cuda", torch.cuda.current_device()),
            dtype=torch.int32,
        )
        gathered_flags = [
            torch.empty_like(local_flag)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_flags, local_flag)
        if any(
            int(flag.item()) != int(local_flag.item()) for flag in gathered_flags
        ):
            raise RuntimeError(
                "H3 head-chunk overlap flag differs between distributed ranks"
            )
    if _head_chunk_overlap_enabled:
        if (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() != 2
        ):
            raise RuntimeError(
                "H3 head-chunk overlap requires an initialized two-rank process group"
            )
        register_contract = getattr(
            _h3_sequence_parallel,
            "register_head_chunk_flash_contract",
            None,
        )
        if register_contract is None:
            raise RuntimeError(
                "H3 head-chunk overlap requires the matching sequence_parallel.py candidate"
            )
        if _device_arch(
            torch.device("cuda", torch.cuda.current_device())
        ) != "gfx936":
            raise RuntimeError("H3 head-chunk overlap is restricted to gfx936")
        register_contract(_head_chunk_flash_contract)
        logging.warning(
            "H3 exact 15s head-chunk overlap ARMED contract=%s",
            _head_chunk_flash_contract,
        )

    def _report_flash_usage():
        logging.warning(
            "H3 gfx936 FlashAttention usage candidate_calls=%s "
            "head_chunk_calls=%s fallback_calls=%s",
            _flash_candidate_calls,
            _flash_head_chunk_calls,
            _flash_fallback_calls,
        )

    import atexit  # noqa: E402
    atexit.register(_report_flash_usage)
    logging.warning("H3 gfx936 FlashAttention candidate armed from %s", _flash_so)

sys.argv[0] = os.path.join(COMFY_ROOT, "main.py")
runpy.run_path(sys.argv[0], run_name="__main__")
