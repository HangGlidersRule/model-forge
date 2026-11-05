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
import contextlib
import hashlib
import json
import os
import secrets
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

import torch
from safetensors import safe_open
from safetensors.torch import save, save_file


def shard_name(value: object) -> str:
    """A weight_map value names one .safetensors file directly in the checkpoint directory."""
    if (not isinstance(value, str) or not value.endswith(".safetensors") or value.startswith(".")
            or "/" in value or "\\" in value or "\0" in value):
        raise SystemExit(f"index shard name is not a single .safetensors file name: {value!r}")
    return value


class OutputDir:
    """--out, opened once. Each file is written into a fresh staging directory under that handle and renamed into
    place, so a name already in --out (a symlink included) is replaced, never written through."""

    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)

    @contextlib.contextmanager
    def _staged(self, name: str) -> Iterator[int]:
        stage = f".merge-staging-{secrets.token_hex(8)}"
        os.mkdir(stage, 0o700, dir_fd=self.fd)
        stage_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.fd)
        try:
            yield stage_fd
            os.replace(name, name, src_dir_fd=stage_fd, dst_dir_fd=self.fd)
        finally:
            for leftover in os.listdir(stage_fd):
                os.unlink(leftover, dir_fd=stage_fd)
            os.close(stage_fd)
            os.rmdir(stage, dir_fd=self.fd)

    def _write(self, name: str, fill: Callable[[BinaryIO], object]) -> None:
        with self._staged(name) as stage_fd:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o666,
                         dir_fd=stage_fd)
            with open(fd, "wb") as f:
                fill(f)

    def save_tensors(self, name: str, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
        if not os.path.isdir(f"/proc/self/fd/{self.fd}"):
            self._write(name, lambda f: f.write(save(tensors, metadata=metadata)))
            return
        # save_file writes a temp file beside its target and renames it, so give it a path that resolves through
        # the staging handle rather than through --out by name.
        with self._staged(name) as stage_fd:
            save_file(tensors, f"/proc/self/fd/{stage_fd}/{name}", metadata=metadata)

    def copy(self, name: str, src: Path) -> None:
        def fill(dst: BinaryIO) -> None:
            with open(src, "rb") as s:
                shutil.copyfileobj(s, dst)
                st = os.fstat(s.fileno())
            dst.flush()
            os.chmod(dst.fileno(), stat.S_IMODE(st.st_mode))
            os.utime(dst.fileno(), ns=(st.st_atime_ns, st.st_mtime_ns))

        self._write(name, fill)

    def write_text(self, name: str, text: str) -> None:
        self._write(name, lambda f: f.write(text.encode()))


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

    index = json.loads((a.base / "model.safetensors.index.json").read_text())
    wmap: dict[str, str] = index["weight_map"]
    missing = sorted(k for k in pairs if k not in wmap)
    if missing:
        raise SystemExit(f"{len(missing)} LoRA targets missing from base, e.g. {missing[:5]}")

    by_shard: dict[str, list[str]] = {}
    for k, s in wmap.items():
        by_shard.setdefault(shard_name(s), []).append(k)

    out = OutputDir(a.out)
    report = {"scale": scale, "r": r, "alpha": alpha, "merged": {}, "copied": 0, "shards": {}}
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
        out.save_tensors(shard, out_tensors, meta or {"format": "pt"})
        report["shards"][shard] = len(out_tensors)
        print(f"  {shard}: {len(out_tensors)} tensors, merged so far {merged_count}/{len(pairs)} ({time.time() - t0:.0f}s)", flush=True)
        del out_tensors

    if merged_count != len(pairs):
        raise SystemExit(f"merged {merged_count} of {len(pairs)} pairs")
    out.copy("model.safetensors.index.json", a.base / "model.safetensors.index.json")
    for p in a.base.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and p.name not in ("model.safetensors.index.json", "SOURCE_REVISION", "README.md"):
            out.copy(p.name, p)
    for extra in ("decision_head.json",):
        out.copy(extra, a.adapter / extra)
    calib = a.adapter.parent / "calibration.json"
    if calib.exists():
        out.copy("calibration.json", calib)
    rels = sorted(v["rel_delta_fro"] for v in report["merged"].values())
    report["summary"] = {"pairs": len(pairs), "merged": merged_count, "copied": report["copied"],
                         "rel_delta_fro_median": rels[len(rels) // 2], "rel_delta_fro_max": rels[-1],
                         "seconds": round(time.time() - t0, 1)}
    if a.report:
        a.report.write_text(json.dumps(report, indent=1))
    else:
        out.write_text("merge_report.json", json.dumps(report, indent=1))
    print("MERGE-DONE", json.dumps(report["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
