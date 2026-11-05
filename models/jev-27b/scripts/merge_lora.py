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
copied from the adapter repo for the serving shim. Every file copied or rewritten into the output must be a regular
file: a symlink is refused, never followed, so nothing outside those directories can end up in the merged checkpoint.
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def open_regular(path: Path) -> int:
    """Open path without following a symlink at its last component; refuse anything but a regular file."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise SystemExit(f"refusing symlink: {path}") from None
        raise
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise SystemExit(f"refusing non-regular file: {path}")
    return fd


def copy_fd(fd: int, dst: Path) -> None:
    """copy2 from an open file: the bytes go to a new file beside dst that then replaces dst, never through a link."""
    st = os.fstat(fd)
    tmp_fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.")
    try:
        with open(tmp_fd, "wb") as w, open(fd, "rb", closefd=False) as r:
            r.seek(0)
            shutil.copyfileobj(r, w)
            w.flush()
            os.fchmod(w.fileno(), stat.S_IMODE(st.st_mode))
            os.utime(w.fileno(), ns=(st.st_atime_ns, st.st_mtime_ns))
        os.replace(tmp, dst)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


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

    index_fd = open_regular(a.base / "model.safetensors.index.json")
    with open(index_fd, "rb", closefd=False) as h:
        index = json.loads(h.read())
    wmap: dict[str, str] = index["weight_map"]
    missing = sorted(k for k in pairs if k not in wmap)
    if missing:
        raise SystemExit(f"{len(missing)} LoRA targets missing from base, e.g. {missing[:5]}")

    shard_fds: dict[str, int] = {}
    for s in sorted(set(wmap.values())):
        if s in (".", "..") or Path(s).name != s:
            raise SystemExit(f"index names a shard outside the base: {s!r}")
        shard_fds[s] = open_regular(a.base / s)
    companions: dict[Path, int] = {}
    with os.scandir(a.base) as it:
        for e in sorted(it, key=lambda e: e.name):
            if e.is_dir(follow_symlinks=False) or e.name.endswith(".safetensors"):
                continue
            if e.name in ("model.safetensors.index.json", "SOURCE_REVISION", "README.md"):
                continue
            companions[a.out / e.name] = open_regular(a.base / e.name)
    for extra in ("decision_head.json",):
        companions[a.out / extra] = open_regular(a.adapter / extra)
    calib = a.adapter.parent / "calibration.json"
    if os.path.lexists(calib):
        companions[a.out / "calibration.json"] = open_regular(calib)

    a.out.mkdir(parents=True, exist_ok=True)
    report = {"scale": scale, "r": r, "alpha": alpha, "merged": {}, "copied": 0, "shards": {}}
    by_shard: dict[str, list[str]] = {}
    for k, s in wmap.items():
        by_shard.setdefault(s, []).append(k)
    merged_count = 0
    t0 = time.time()
    for shard in sorted(by_shard):
        out_tensors: dict[str, torch.Tensor] = {}
        shard_fd = shard_fds.pop(shard)
        with safe_open(f"/proc/self/fd/{shard_fd}", framework="pt") as f:
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
        os.close(shard_fd)
        save_file(out_tensors, str(a.out / shard), metadata=meta or {"format": "pt"})
        report["shards"][shard] = len(out_tensors)
        print(f"  {shard}: {len(out_tensors)} tensors, merged so far {merged_count}/{len(pairs)} ({time.time() - t0:.0f}s)", flush=True)
        del out_tensors

    if merged_count != len(pairs):
        raise SystemExit(f"merged {merged_count} of {len(pairs)} pairs")
    copy_fd(index_fd, a.out / "model.safetensors.index.json")
    os.close(index_fd)
    for dst, fd in companions.items():
        copy_fd(fd, dst)
        os.close(fd)
    rels = sorted(v["rel_delta_fro"] for v in report["merged"].values())
    report["summary"] = {"pairs": len(pairs), "merged": merged_count, "copied": report["copied"],
                         "rel_delta_fro_median": rels[len(rels) // 2], "rel_delta_fro_max": rels[-1],
                         "seconds": round(time.time() - t0, 1)}
    (a.report or a.out / "merge_report.json").write_text(json.dumps(report, indent=1))
    print("MERGE-DONE", json.dumps(report["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
