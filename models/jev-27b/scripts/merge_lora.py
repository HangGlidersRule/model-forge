#!/usr/bin/env python3
"""Merge the JEV-27B LoRA into the Qwen3.8-27B BF16 base, shard by shard.

    W_merged = W + (alpha / r) * (B @ A)     computed in FP32, rounded once to BF16

- Base: Qwen/Qwen3.8-27B (multimodal layout: model.language_model.layers.N..., lm_head, vision, MTP).
- Adapter keys: base_model.model.model.layers.N.<mod>.lora_{A,B}.weight and
  base_model.model.lm_head.lora_{A,B}.weight (text-only layout, as published).
- Mapping: adapter `base_model.model.model.X` -> base `model.language_model.X`;
  `base_model.model.lm_head` -> `lm_head`. This is the same remap the deployed
  NVFP4+LoRA lane needs (without it vLLM silently skips 800 of 802 tensors).
- Every one of the 401 LoRA pairs must find exactly one base weight of the right shape,
  or the run fails. Untouched tensors (vision, MTP, norms, embeddings, conv1d, ...) are
  copied bit-for-bit. Writes a per-tensor record: merged / copied, pre/post sha256.

Usage: merge_lora.py --base DIR --adapter DIR --out DIR [--report FILE]
The output dir gets the base's non-weight files (config, tokenizer, processor, chat template),
the merged shards with the same names and index, and decision_head.json + calibration.json
copied from the adapter repo for the serving shim.

Every input file is opened once, relative to its directory, without following a symlink, and
must be a regular file; the bytes read through that descriptor are the bytes written out. A
symlinked input (e.g. a Hugging Face cache snapshot) is refused: stage the base with
`hf download --local-dir` so it holds regular files.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import stat
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INDEX = "model.safetensors.index.json"
NOT_COPIED = (INDEX, "SOURCE_REVISION", "README.md")


def open_dir(path: Path) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)


def require_plain_name(name: str, label: Path) -> None:
    if name in ("", ".", "..") or "/" in name or "\0" in name:
        raise SystemExit(f"refusing {label}: {name!r} is not a plain file name")


def open_regular(dir_fd: int, name: str, label: Path) -> int:
    """Open `name` directly inside `dir_fd` without following a symlink; refuse anything but a regular file."""
    require_plain_name(name, label)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dir_fd)
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.EMLINK):
            raise SystemExit(f"refusing {label}: it is a symlink") from None
        raise
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise SystemExit(f"refusing {label}: not a regular file")
    return fd


def read_regular(dir_fd: int, name: str, label: Path) -> bytes:
    fd = open_regular(dir_fd, name, label)
    try:
        chunks = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def write_regular(dir_fd: int, name: str, data: bytes) -> None:
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644, dir_fd=dir_fd)
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.EMLINK):
            raise SystemExit(f"refusing to write {name}: output is a symlink") from None
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise SystemExit(f"refusing to write {name}: output is not a regular file")
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
    finally:
        os.close(fd)


@contextmanager
def open_safetensors(dir_fd: int, name: str, label: Path) -> Iterator:
    # safe_open takes a path; /dev/fd/N reopens the file already bound to fd, not the name.
    fd = open_regular(dir_fd, name, label)
    try:
        with safe_open(f"/dev/fd/{fd}", framework="pt") as f:
            yield f
    finally:
        os.close(fd)


def adapter_target(key: str) -> tuple[str, str] | None:
    """Map an adapter tensor name to (base weight name, 'A'|'B')."""
    if not key.endswith(".weight"):
        return None
    if ".lora_A." in key:
        part, stem = "A", key.replace(".lora_A.weight", "")
    elif ".lora_B." in key:
        part, stem = "B", key.replace(".lora_B.weight", "")
    else:
        return None
    if stem == "base_model.model.lm_head":
        return "lm_head.weight", part
    prefix = "base_model.model.model."
    if not stem.startswith(prefix):
        raise ValueError(f"unexpected adapter key layout: {key}")
    return "model.language_model." + stem[len(prefix):] + ".weight", part


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--hash-all", action="store_true", help="sha256 every tensor (slow on 27B)")
    a = ap.parse_args()
    with ExitStack() as fds:
        return merge(a, fds)


def merge(a: argparse.Namespace, fds: ExitStack) -> int:
    def dir_fd(path: Path) -> int:
        fd = open_dir(path)
        fds.callback(os.close, fd)
        return fd

    base_fd, adapter_fd, bundle_fd = dir_fd(a.base), dir_fd(a.adapter), dir_fd(a.adapter.parent)

    cfg = json.loads(read_regular(adapter_fd, "adapter_config.json", a.adapter / "adapter_config.json"))
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    if cfg.get("use_rslora") or cfg.get("use_dora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise SystemExit("unsupported adapter options (rslora/dora/rank or alpha patterns)")
    if cfg.get("fan_in_fan_out"):
        raise SystemExit("fan_in_fan_out adapters are not supported")
    scale = alpha / r
    print(f"adapter r={r} alpha={alpha} scale={scale}", flush=True)

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with open_safetensors(adapter_fd, "adapter_model.safetensors", a.adapter / "adapter_model.safetensors") as f:
        for k in f.keys():
            t = adapter_target(k)
            if t is None:
                raise SystemExit(f"unexpected non-LoRA tensor in adapter: {k}")
            pairs.setdefault(t[0], {})[t[1]] = f.get_tensor(k)
    incomplete = [k for k, v in pairs.items() if set(v) != {"A", "B"}]
    if incomplete:
        raise SystemExit(f"incomplete LoRA pairs: {incomplete[:5]}")
    print(f"adapter pairs: {len(pairs)}", flush=True)

    companions: dict[str, bytes] = {INDEX: read_regular(base_fd, INDEX, a.base / INDEX)}
    index = json.loads(companions[INDEX])
    wmap: dict[str, str] = index["weight_map"]
    missing = sorted(k for k in pairs if k not in wmap)
    if missing:
        raise SystemExit(f"{len(missing)} LoRA targets missing from base, e.g. {missing[:5]}")
    for shard in set(wmap.values()):
        require_plain_name(shard, a.base / INDEX)

    with os.scandir(base_fd) as entries:
        listed = sorted((e.name, e.is_dir(follow_symlinks=False)) for e in entries)
    for name, is_dir in listed:
        if is_dir or name.endswith(".safetensors") or name in NOT_COPIED:
            continue
        companions[name] = read_regular(base_fd, name, a.base / name)
    companions["decision_head.json"] = read_regular(adapter_fd, "decision_head.json", a.adapter / "decision_head.json")
    try:
        companions["calibration.json"] = read_regular(bundle_fd, "calibration.json", a.adapter.parent / "calibration.json")
    except FileNotFoundError:
        pass

    a.out.mkdir(parents=True, exist_ok=True)
    out_fd = dir_fd(a.out)
    report = {"scale": scale, "r": r, "alpha": alpha, "merged": {}, "copied": 0, "shards": {}}
    by_shard: dict[str, list[str]] = {}
    for k, s in wmap.items():
        by_shard.setdefault(s, []).append(k)
    merged_count = 0
    t0 = time.time()
    for shard in sorted(by_shard):
        out_tensors: dict[str, torch.Tensor] = {}
        with open_safetensors(base_fd, shard, a.base / shard) as f:
            meta = f.metadata()
            for k in by_shard[shard]:
                w = f.get_tensor(k)
                if k in pairs:
                    A, B = pairs[k]["A"].to(torch.float32), pairs[k]["B"].to(torch.float32)
                    if w.dim() != 2 or B.shape[0] != w.shape[0] or A.shape[1] != w.shape[1] or A.shape[0] != r or B.shape[1] != r:
                        raise SystemExit(f"shape mismatch for {k}: W{tuple(w.shape)} A{tuple(A.shape)} B{tuple(B.shape)}")
                    delta = (B @ A) * scale
                    new = (w.to(torch.float32) + delta).to(w.dtype)
                    rel = (delta.norm() / w.to(torch.float32).norm()).item()
                    changed = int((new != w).sum().item())
                    report["merged"][k] = {"shape": list(w.shape), "rel_delta_fro": rel, "changed_elems": changed,
                                           "elems": w.numel(), **({"pre": sha(w), "post": sha(new)} if a.hash_all else {})}
                    out_tensors[k] = new
                    merged_count += 1
                else:
                    out_tensors[k] = w
                    report["copied"] += 1
        save_file(out_tensors, str(a.out / shard), metadata=meta or {"format": "pt"})
        report["shards"][shard] = len(out_tensors)
        print(f"  {shard}: {len(out_tensors)} tensors, merged so far {merged_count}/{len(pairs)} ({time.time() - t0:.0f}s)", flush=True)
        del out_tensors

    if merged_count != len(pairs):
        raise SystemExit(f"merged {merged_count} of {len(pairs)} pairs")
    for name, data in companions.items():
        write_regular(out_fd, name, data)
    rels = sorted(v["rel_delta_fro"] for v in report["merged"].values())
    report["summary"] = {"pairs": len(pairs), "merged": merged_count, "copied": report["copied"],
                         "rel_delta_fro_median": rels[len(rels) // 2], "rel_delta_fro_max": rels[-1],
                         "seconds": round(time.time() - t0, 1)}
    (a.report or a.out / "merge_report.json").write_text(json.dumps(report, indent=1))
    print("MERGE-DONE", json.dumps(report["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
