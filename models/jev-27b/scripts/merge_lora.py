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
copied from the adapter repo for the serving shim. Copied files must be regular files: a symlink aborts the run before
the output dir is touched, and is never followed.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


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


Bound = tuple[int, os.stat_result]


def open_regular(path: Path, dir_fd: int | None = None) -> Bound | None:
    """Descriptor of a regular file, or None if absent or a directory. A symlink or special file aborts.
    With dir_fd, path.name is opened inside that directory."""
    try:
        fd = os.open(path if dir_fd is None else path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=dir_fd)
    except FileNotFoundError:
        return None
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise SystemExit(f"{path} is a symlink; refusing to copy it") from None
        raise
    st = os.fstat(fd)
    if stat.S_ISREG(st.st_mode):
        return fd, st
    os.close(fd)
    if stat.S_ISDIR(st.st_mode):
        return None
    raise SystemExit(f"{path} is not a regular file")


def copy_bound(bound: Bound, dst: Path) -> None:
    """shutil.copy2 from an open descriptor; dst is replaced, never written through."""
    fd, st = bound
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    out = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(fd, "rb") as fsrc, os.fdopen(out, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, 1 << 24)
            os.fchmod(fdst.fileno(), stat.S_IMODE(st.st_mode))
            fdst.flush()
            os.utime(fdst.fileno(), ns=(st.st_atime_ns, st.st_mtime_ns))
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(a.adapter / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            t = adapter_target(k)
            if t is None:
                raise SystemExit(f"unexpected non-LoRA tensor in adapter: {k}")
            pairs.setdefault(t[0], {})[t[1]] = f.get_tensor(k)
    incomplete = [k for k, v in pairs.items() if set(v) != {"A", "B"}]
    if incomplete:
        raise SystemExit(f"incomplete LoRA pairs: {incomplete[:5]}")
    print(f"adapter pairs: {len(pairs)}", flush=True)

    base_fd = os.open(a.base, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    companions: dict[str, Bound] = {}
    for name in sorted(os.listdir(base_fd)):
        if name.endswith(".safetensors") or name in ("SOURCE_REVISION", "README.md"):
            continue
        bound = open_regular(a.base / name, dir_fd=base_fd)
        if bound is not None:
            companions[name] = bound
    os.close(base_fd)
    if "model.safetensors.index.json" not in companions:
        raise SystemExit(f"{a.base / 'model.safetensors.index.json'} is missing")
    extras: dict[str, Bound] = {}
    for name, path in (("decision_head.json", a.adapter / "decision_head.json"),
                       ("calibration.json", a.adapter.parent / "calibration.json")):
        bound = open_regular(path)
        if bound is not None:
            extras[name] = bound
        elif name == "decision_head.json":
            raise SystemExit(f"{path} is missing")

    index_fd = companions["model.safetensors.index.json"][0]
    index = json.loads(os.pread(index_fd, companions["model.safetensors.index.json"][1].st_size, 0))
    wmap: dict[str, str] = index["weight_map"]
    missing = sorted(k for k in pairs if k not in wmap)
    if missing:
        raise SystemExit(f"{len(missing)} LoRA targets missing from base, e.g. {missing[:5]}")

    a.out.mkdir(parents=True, exist_ok=True)
    report = {"scale": scale, "r": r, "alpha": alpha, "merged": {}, "copied": 0, "shards": {}}
    by_shard: dict[str, list[str]] = {}
    for k, s in wmap.items():
        by_shard.setdefault(s, []).append(k)
    merged_count = 0
    t0 = time.time()
    for shard in sorted(by_shard):
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(a.base / shard), framework="pt") as f:
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
    for name, bound in {**companions, **extras}.items():
        copy_bound(bound, a.out / name)
    rels = sorted(v["rel_delta_fro"] for v in report["merged"].values())
    report["summary"] = {"pairs": len(pairs), "merged": merged_count, "copied": report["copied"],
                         "rel_delta_fro_median": rels[len(rels) // 2], "rel_delta_fro_max": rels[-1],
                         "seconds": round(time.time() - t0, 1)}
    (a.report or a.out / "merge_report.json").write_text(json.dumps(report, indent=1))
    print("MERGE-DONE", json.dumps(report["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
