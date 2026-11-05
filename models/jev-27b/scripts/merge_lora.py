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
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load, save

from model_forge.pipeline import (
    INDEX_NAME,
    OutputDir,
    SourceDir,
    safetensors_metadata,
    shard_names,
)


def open_dir(path: Path) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)


def read_checkpoint_file(path: Path, dir_fd: int) -> bytes:
    """Read ``path``, which must be a regular file and the entry ``path.name`` of the directory open as ``dir_fd``.

    A link is refused, including one swapped in after the directory was listed, and so is a file reached through a
    parent that was swapped for a link: the opened inode must be the one that directory lists under that name."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as e:
        raise SystemExit(f"{path}: {e.strerror} (checkpoint files must be regular files, not links)") from e
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or not os.path.samestat(
            st, os.stat(path.name, dir_fd=dir_fd, follow_symlinks=False)
        ):
            raise SystemExit(f"{path}: not a regular file in {path.parent}")
        with open(fd, "rb", closefd=False) as f:
            return f.read()
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

    cfg = json.loads((a.adapter / "adapter_config.json").read_text())
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    if cfg.get("use_rslora") or cfg.get("use_dora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise SystemExit("unsupported adapter options (rslora/dora/rank or alpha patterns)")
    if cfg.get("fan_in_fan_out"):
        raise SystemExit("fan_in_fan_out adapters are not supported")
    scale = alpha / r
    print(f"adapter r={r} alpha={alpha} scale={scale}", flush=True)

    adapter = SourceDir(a.adapter)
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in load(adapter.read_bytes("adapter_model.safetensors")).items():
        t = adapter_target(k)
        if t is None:
            raise SystemExit(f"unexpected non-LoRA tensor in adapter: {k}")
        pairs.setdefault(t[0], {})[t[1]] = v
    incomplete = [k for k, v in pairs.items() if set(v) != {"A", "B"}]
    if incomplete:
        raise SystemExit(f"incomplete LoRA pairs: {incomplete[:5]}")
    print(f"adapter pairs: {len(pairs)}", flush=True)

    # Every file copied into the output is read here, before anything is written, from a directory held open.
    base_fd, adapter_fd, adapter_parent_fd = open_dir(a.base), open_dir(a.adapter), open_dir(a.adapter.parent)
    index_bytes = read_checkpoint_file(a.base / INDEX_NAME, base_fd)
    sidecars: dict[str, bytes] = {}
    for name in sorted(os.listdir(base_fd)):
        if name.endswith(".safetensors") or name in (INDEX_NAME, "SOURCE_REVISION", "README.md"):
            continue
        if stat.S_ISDIR(os.stat(name, dir_fd=base_fd, follow_symlinks=False).st_mode):
            continue
        sidecars[name] = read_checkpoint_file(a.base / name, base_fd)
    for extra in ("decision_head.json",):
        sidecars[extra] = read_checkpoint_file(a.adapter / extra, adapter_fd)
    if os.path.lexists(a.adapter.parent / "calibration.json"):
        sidecars["calibration.json"] = read_checkpoint_file(a.adapter.parent / "calibration.json", adapter_parent_fd)
    for fd in (base_fd, adapter_fd, adapter_parent_fd):
        os.close(fd)

    # Shard names are validated and every shard is bound before the output is touched.
    index = json.loads(index_bytes)
    wmap = shard_names(index["weight_map"])
    base = SourceDir(a.base)
    missing = sorted(k for k in pairs if k not in wmap)
    if missing:
        raise SystemExit(f"{len(missing)} LoRA targets missing from base, e.g. {missing[:5]}")

    by_shard: dict[str, list[str]] = {}
    for k, s in wmap.items():
        by_shard.setdefault(s, []).append(k)
    shards = {s: base.open(s) for s in sorted(by_shard)}

    a.out.mkdir(parents=True, exist_ok=True)
    out = OutputDir(a.out)
    report = {"scale": scale, "r": r, "alpha": alpha, "merged": {}, "copied": 0, "shards": {}}
    merged_count = 0
    t0 = time.time()
    for shard, bound in shards.items():
        out_tensors: dict[str, torch.Tensor] = {}
        data = bound.read_bytes()
        meta = safetensors_metadata(data)
        base_tensors = load(data)
        del data
        for k in by_shard[shard]:
            w = base_tensors.pop(k)
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
        del base_tensors
        out.write_bytes(shard, save(out_tensors, metadata=meta or {"format": "pt"}))
        report["shards"][shard] = len(out_tensors)
        print(f"  {shard}: {len(out_tensors)} tensors, merged so far {merged_count}/{len(pairs)} ({time.time() - t0:.0f}s)", flush=True)
        del out_tensors

    if merged_count != len(pairs):
        raise SystemExit(f"merged {merged_count} of {len(pairs)} pairs")
    out.write_bytes(INDEX_NAME, index_bytes)  # verbatim; every shard name was validated above
    for name, data in sidecars.items():
        out.write_bytes(name, data)
    for src in (base, adapter):
        src.close()
    rels = sorted(v["rel_delta_fro"] for v in report["merged"].values())
    report["summary"] = {"pairs": len(pairs), "merged": merged_count, "copied": report["copied"],
                         "rel_delta_fro_median": rels[len(rels) // 2], "rel_delta_fro_max": rels[-1],
                         "seconds": round(time.time() - t0, 1)}
    if a.report:
        a.report.write_text(json.dumps(report, indent=1))
    else:
        out.write_bytes("merge_report.json", json.dumps(report, indent=1).encode())
    print("MERGE-DONE", json.dumps(report["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
