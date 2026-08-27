"""Submit one canonical H3 prompt to both sequence-parallel ranks."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import signal
import time
import urllib.error
import urllib.request
from pathlib import Path


def request_json(url: str, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input(root: Path, name: str) -> Path:
    root = root.resolve()
    candidate = (root / name).resolve()
    if os.path.commonpath((str(root), str(candidate))) != str(root):
        raise ValueError(f"LoadImage path escapes its input directory: {name}")
    if not candidate.is_file():
        raise FileNotFoundError(f"missing conditioning image: {candidate}")
    return candidate


def canonical_prompt(path: Path, input_roots: tuple[Path, Path]):
    document = json.loads(path.read_text())
    prompt = copy.deepcopy(document.get("prompt", document))
    existing_sp_nodes = [
        node for node in prompt.values()
        if node.get("class_type") == "MiniMaxH3SequenceParallel"
    ]
    if existing_sp_nodes:
        raise ValueError("workflow must not contain a pre-existing MiniMaxH3SequenceParallel node")
    t2va_nodes = [node for node in prompt.values() if node.get("class_type") == "MiniMaxH3ImageToVideo"]
    reference_nodes = [node for node in prompt.values() if node.get("class_type") == "MiniMaxH3ReferenceToVideo"]
    if len(t2va_nodes) != 1 or reference_nodes:
        raise ValueError(
            "sequence-parallel worker accepts one MiniMaxH3ImageToVideo node; "
            "independent <Picture i> reference mode requires a ref2va checkpoint"
        )
    t2va_inputs = t2va_nodes[0].get("inputs", {})
    conditioning_mode = "t2v"
    first_frame = t2va_inputs.get("first_frame")
    last_frame = t2va_inputs.get("last_frame")
    if last_frame is not None and first_frame is None:
        raise ValueError("last_frame requires first_frame for MiniMax H3 first/last-frame generation")
    if first_frame is not None:
        conditioning_mode = "flf" if last_frame is not None else "i2v"
    input_hashes = {}
    for name, reference in (("first_frame", first_frame), ("last_frame", last_frame)):
        if reference is None:
            continue
        if not isinstance(reference, list) or len(reference) != 2:
            raise ValueError(f"{name} must be a ComfyUI node-output reference")
        if reference[1] != 0:
            raise ValueError(f"{name} must consume output slot 0 from LoadImage")
        source = prompt.get(str(reference[0]))
        if source is None or source.get("class_type") != "LoadImage":
            raise ValueError(f"{name} must come directly from a LoadImage node")
        image_name = source.get("inputs", {}).get("image")
        if not isinstance(image_name, str) or not image_name:
            raise ValueError(f"{name} LoadImage node has no image filename")
        rank_hashes = [file_sha256(resolve_input(root, image_name)) for root in input_roots]
        if rank_hashes[0] != rank_hashes[1]:
            raise ValueError(f"{name} image differs between rank input directories: {rank_hashes}")
        input_hashes[name] = {"name": image_name, "sha256": rank_hashes[0]}

    noise_nodes = [(node_id, node) for node_id, node in prompt.items() if node.get("class_type") == "RandomNoise"]
    if len(noise_nodes) != 1:
        raise ValueError(f"expected one RandomNoise node, got {len(noise_nodes)}")
    seed = noise_nodes[0][1]["inputs"]["noise_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 64:
        raise ValueError("RandomNoise noise_seed must be an unsigned 64-bit integer")

    schedulers = [
        (node_id, node) for node_id, node in prompt.items()
        if node.get("class_type") == "BasicScheduler"
    ]
    guiders = [
        (node_id, node) for node_id, node in prompt.items()
        if node.get("class_type") == "BasicGuider"
    ]
    if len(schedulers) != 1 or len(guiders) != 1:
        raise ValueError("expected one BasicScheduler and one BasicGuider")
    model_consumers = schedulers + guiders
    model_refs = [node["inputs"]["model"] for _, node in model_consumers]
    if model_refs[0] != model_refs[1]:
        raise ValueError("BasicScheduler and BasicGuider must consume the same model")

    contract = {"prompt": prompt, "conditioning_inputs": input_hashes}
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    numeric_ids = [int(node_id) for node_id in prompt if str(node_id).isdigit()]
    sp_id = str(max(numeric_ids, default=0) + 1)
    prompt[sp_id] = {
        "class_type": "MiniMaxH3SequenceParallel",
        "inputs": {
            "model": model_refs[0],
            "job_digest": digest,
            "seed": seed,
        },
    }
    for _, node in model_consumers:
        node["inputs"]["model"] = [sp_id, 0]
    return prompt, digest, seed, conditioning_mode, input_hashes


def interrupt(ports):
    for port in ports:
        try:
            request_json(f"http://127.0.0.1:{port}/interrupt", {}, timeout=2)
        except Exception:
            pass


def terminate_group(pid_file: Path):
    try:
        pid = int(pid_file.read_text().strip())
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", "replace"
        )
        if (
            "torch.distributed.run" not in cmdline
            or "minimax_h3_sp/run_rank.py" not in cmdline
            or os.getpgid(pid) != pid
        ):
            raise RuntimeError(f"refusing to signal unverified worker pid={pid}: {cmdline}")
        os.killpg(pid, signal.SIGTERM)
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass


def history_status(history, prompt_id):
    entry = history.get(prompt_id)
    if entry is None:
        return None, None
    status = entry.get("status", {})
    return status.get("status_str"), entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=Path)
    parser.add_argument("--base-port", type=int, default=8290)
    parser.add_argument("--pid-file", type=Path, default=Path("/path/to/runtime/torchrun.pid"))
    parser.add_argument("--lock-file", type=Path, default=Path("/path/to/runtime/coordinator.lock"))
    parser.add_argument("--runtime-root", type=Path, default=Path("/path/to/runtime"))
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = args.lock_file.open("w")
    fcntl.flock(lock_handle, fcntl.LOCK_EX)

    def terminate_on_signal(_signum, _frame):
        raise SystemExit("sequence-parallel coordinator terminated")

    signal.signal(signal.SIGTERM, terminate_on_signal)

    ports = [args.base_port, args.base_port + 1]
    input_roots = (args.runtime_root / "rank0" / "input", args.runtime_root / "rank1" / "input")
    prompt, digest, seed, conditioning_mode, input_hashes = canonical_prompt(args.prompt, input_roots)
    submissions = []
    try:
        for port in ports:
            submissions.append(request_json(f"http://127.0.0.1:{port}/prompt", {"prompt": prompt}, timeout=30))
        if any(item.get("node_errors") for item in submissions):
            raise RuntimeError(f"prompt validation failed: {submissions}")
        prompt_ids = [item["prompt_id"] for item in submissions]

        deadline = time.monotonic() + args.timeout
        entries = [None, None]
        while time.monotonic() < deadline:
            for rank, (port, prompt_id) in enumerate(zip(ports, prompt_ids)):
                if entries[rank] is not None:
                    continue
                history = request_json(f"http://127.0.0.1:{port}/history/{prompt_id}", timeout=10)
                status, entry = history_status(history, prompt_id)
                if status == "error":
                    raise RuntimeError(f"rank {rank} failed: {entry}")
                if status == "success":
                    entries[rank] = entry
            if all(entry is not None for entry in entries):
                result = {
                    "job_digest": digest,
                    "seed": seed,
                    "conditioning_mode": conditioning_mode,
                    "conditioning_inputs": input_hashes,
                    "prompt_ids": prompt_ids,
                    "rank0": entries[0],
                    "rank1": entries[1],
                }
                args.result.parent.mkdir(parents=True, exist_ok=True)
                args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                print(json.dumps({
                    "job_digest": digest,
                    "seed": seed,
                    "conditioning_mode": conditioning_mode,
                    "prompt_ids": prompt_ids,
                }))
                return
            time.sleep(2)
        raise TimeoutError(f"sequence-parallel job timed out after {args.timeout}s")
    except BaseException:
        interrupt(ports)
        terminate_group(args.pid_file)
        raise


if __name__ == "__main__":
    main()
