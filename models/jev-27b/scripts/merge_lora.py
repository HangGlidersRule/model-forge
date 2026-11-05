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
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from model_forge.pipeline import CheckpointTree, OutputDir  # noqa: E402

BASE_SKIP = ("model.safetensors.index.json", "SOURCE_REVISION", "README.md")


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

    # Every input is opened (no symlinks, regular files only) before anything is written.
    adapter = CheckpointTree(a.adapter)
    base = CheckpointTree(a.base)
    cfg = json.loads(adapter.open("adapter_config.json").read_bytes())
    r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    if cfg.get("use_rslora") or cfg.get("use_dora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise SystemExit("unsupported adapter options (rslora/dora/rank or alpha patterns)")
    if cfg.get("fan_in_fan_out"):
        raise SystemExit("fan_in_fan_out adapters are not supported")
    scale = alpha / r
    print(f"adapter r={r} alpha={alpha} scale={scale}", flush=True)

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(adapter.open("adapter_model.safetensors").path, framework="pt") as f:
        for k in f.keys():
            t = adapter_target(k)
            if t is None:
                raise SystemExit(f"unexpected non-LoRA tensor in adapter: {k}")
            pairs.setdefault(t[0], {})[t[1]] = f.get_tensor(k)
    incomplete = [k for k, v in pairs.items() if set(v) != {"A", "B"}]
    if incomplete:
        raise SystemExit(f"incomplete LoRA pairs: {incomplete[:5]}")
    print(f"adapter pairs: {len(pairs)}", flush=True)

    index_file = base.open("model.safetensors.index.json")
    index = json.loads(index_file.read_bytes())
    wmap: dict[str, str] = index["weight_map"]
    missing = sorted(k for k in pairs if k not in wmap)
    if missing:
        raise SystemExit(f"{len(missing)} LoRA targets missing from base, e.g. {missing[:5]}")
    by_shard: dict[str, list[str]] = {}
    for k, s in wmap.items():
        by_shard.setdefault(s, []).append(k)
    shard_files = {shard: base.open(shard) for shard in sorted(by_shard)}
    companions = {name: base.open(name) for name in base.listdir()[0]
                  if not name.endswith(".safetensors") and name not in BASE_SKIP}
    decision_head = adapter.open("decision_head.json")
    calib = CheckpointTree(a.adapter.parent).open_optional("calibration.json")

    out = OutputDir.create(a.out)
    report = {"scale": scale, "r": r, "alpha": alpha, "merged": {}, "copied": 0, "shards": {}}
    merged_count = 0
    t0 = time.time()
    for shard in sorted(by_shard):
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(shard_files[shard].path, framework="pt") as f:
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
        save_file(out_tensors, out.path(shard), metadata=meta or {"format": "pt"})
        report["shards"][shard] = len(out_tensors)
        print(f"  {shard}: {len(out_tensors)} tensors, merged so far {merged_count}/{len(pairs)} ({time.time() - t0:.0f}s)", flush=True)
        del out_tensors

    if merged_count != len(pairs):
        raise SystemExit(f"merged {merged_count} of {len(pairs)} pairs")
    out.copy("model.safetensors.index.json", index_file, keep_metadata=True)
    for name, member in companions.items():
        out.copy(name, member, keep_metadata=True)
    out.copy("decision_head.json", decision_head, keep_metadata=True)
    if calib is not None:
        out.copy("calibration.json", calib, keep_metadata=True)
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
