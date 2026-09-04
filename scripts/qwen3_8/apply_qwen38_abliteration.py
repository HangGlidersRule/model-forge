#!/usr/bin/env python3
"""Apply abliteration projection shard-by-shard to official Qwen3.8-27B BF16.

Processes safetensor shards without holding two full model copies.
Applies W' = W - r(r^T W) in float32 to exactly the expected 131 tensors.
Copies all other tensors byte-for-byte. Preserves tokenizer, processor, and config.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

from model_forge.abliteration import (
    compute_leakage,
    is_vision_tensor,
    matches_selector,
    project_weight,
)
from model_forge.experiment import load_experiment
from model_forge.pipeline import (
    RunLock,
    StageContext,
    SuccessManifest,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply abliteration to BF16 model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--direction-dir", type=Path, default=None)
    parser.add_argument("--source-model", type=Path, default=None, help="Local model path (optional)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-stage", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    cfg_sha = cfg.config_sha()

    ctx = StageContext(args.run_root, "apply_abliteration", cfg_sha)
    if args.resume and not args.force_stage:
        existing = ctx.should_skip()
        if existing:
            print(f"Abliteration stage verified, skipping (config_sha={cfg_sha[:12]})")
            return

    direction_dir = args.direction_dir or (args.run_root / "measure_direction")
    direction_file = direction_dir / "direction.safetensors"
    if not direction_file.exists():
        print("ERROR: Direction not measured. Run measure_qwen38_refusal_direction.py first.", file=sys.stderr)
        sys.exit(1)

    from safetensors.torch import load_file, save_file

    direction_tensors = load_file(str(direction_file))
    direction = direction_tensors["direction"]

    if args.source_model:
        model_path = args.source_model
    else:
        from huggingface_hub import snapshot_download

        print(f"Downloading {cfg.source.model_id}@{cfg.source.revision[:12]}...")
        model_path = Path(snapshot_download(cfg.source.model_id, revision=cfg.source.revision))

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        print("ERROR: model.safetensors.index.json not found", file=sys.stderr)
        sys.exit(1)

    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]

    selectors = list(cfg.abliteration.target_selectors)
    target_names = [name for name in weight_map if matches_selector(name, selectors)]

    for name in target_names:
        if is_vision_tensor(name):
            print(f"ERROR: Selector matched vision tensor: {name}", file=sys.stderr)
            sys.exit(1)

    if len(target_names) != cfg.abliteration.expected_target_count:
        print(
            f"ERROR: Expected {cfg.abliteration.expected_target_count} targets, "
            f"found {len(target_names)}: {sorted(target_names)[:5]}...",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Applying projection to {len(target_names)} tensors...")
    lock = RunLock(args.run_root / "apply_abliteration.lock")
    with lock:
        partial = ctx.partial_dir
        target_set = set(target_names)
        shard_files = sorted(set(weight_map.values()))

        edit_report: list[dict[str, object]] = []
        edited_count = 0
        unchanged_vision_count = 0

        for shard_name in shard_files:
            src_shard = model_path / shard_name
            shard_tensors = load_file(str(src_shard))
            modified_tensors: dict[str, torch.Tensor] = {}

            for tname, tensor in shard_tensors.items():
                if tname in target_set:
                    before_norm = tensor.float().norm().item()
                    projected = project_weight(tensor, direction)
                    after_norm = projected.float().norm().item()
                    leakage = compute_leakage(projected, direction)
                    modified_tensors[tname] = projected
                    edit_report.append({
                        "tensor": tname,
                        "before_norm": before_norm,
                        "after_norm": after_norm,
                        "leakage": leakage,
                        "edited": True,
                    })
                    edited_count += 1
                else:
                    modified_tensors[tname] = tensor
                    if is_vision_tensor(tname):
                        unchanged_vision_count += 1

            save_file(modified_tensors, str(partial / shard_name))

        assert edited_count == cfg.abliteration.expected_target_count

        # Copy non-safetensor assets
        for asset in model_path.iterdir():
            if asset.name.endswith(".safetensors"):
                continue
            if asset.name.startswith("."):
                continue
            dest = partial / asset.name
            if asset.is_file():
                shutil.copy2(asset, dest)
            elif asset.is_dir():
                shutil.copytree(asset, dest)

        # Write edit report
        report_path = partial / "abliteration_report.json"
        report_path.write_text(json.dumps({
            "edited_count": edited_count,
            "expected_count": cfg.abliteration.expected_target_count,
            "edits": edit_report,
            "unchanged_vision_count": unchanged_vision_count,
        }, indent=2) + "\n")

        # Compute output hashes for key files
        output_hashes: dict[str, str] = {}
        for f in sorted(partial.rglob("*")):
            if f.is_file() and f.name != "_SUCCESS.json":
                output_hashes[str(f.relative_to(partial))] = sha256_file(f)

        manifest = SuccessManifest(
            stage="apply_abliteration",
            config_sha=cfg_sha,
            source_revisions={"model": cfg.source.revision},
            output_hashes=output_hashes,
            metrics={
                "edited_tensors": edited_count,
                "max_leakage": max((e["leakage"] for e in edit_report), default=0.0),
            },
        )
        ctx.promote(manifest)
        print(f"Abliteration applied: {edited_count} tensors edited")


if __name__ == "__main__":
    main()
