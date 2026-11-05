#!/usr/bin/env python3
"""Apply abliteration projection shard-by-shard to official Qwen3.8-27B BF16.

Processes safetensor shards without holding two full model copies.
Applies W' = W - r(r^T W) in float32 to exactly the expected 131 tensors.
Copies all other tensors byte-for-byte. Preserves tokenizer, processor, and config.
"""
from __future__ import annotations

import argparse
import json
import os
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
    INDEX_NAME,
    OutputDir,
    RunLock,
    SourceDir,
    StageContext,
    SuccessManifest,
    UnsafeCheckpointFile,
    copy_checkpoint_assets,
    sha256_file,
    shard_names,
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

    from safetensors.torch import load, load_file, save

    direction_tensors = load_file(str(direction_file))
    direction = direction_tensors["direction"]

    if args.source_model:
        model_path = args.source_model
    else:
        from huggingface_hub import snapshot_download

        print(f"Downloading {cfg.source.model_id}@{cfg.source.revision[:12]}...")
        model_path = Path(snapshot_download(cfg.source.model_id, revision=cfg.source.revision))
    # A Hugging Face cache snapshot holds links into its repository's blobs/, so the repository bounds it.
    absolute = Path(os.path.abspath(model_path))
    source_root = absolute.parents[1] if absolute.parent.name == "snapshots" else None
    source = SourceDir(model_path, root=source_root)

    try:
        index_bytes = source.read_bytes(INDEX_NAME)  # the validated bytes are the bytes written out
        index = json.loads(index_bytes)
    except FileNotFoundError:
        print(f"ERROR: {INDEX_NAME} not found", file=sys.stderr)
        sys.exit(1)
    weight_map = shard_names(index["weight_map"])

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

    shards = {name: source.open(name) for name in sorted(set(weight_map.values()))}

    print(f"Applying projection to {len(target_names)} tensors...")
    lock = RunLock(args.run_root / "apply_abliteration.lock")
    with lock:
        partial = ctx.partial_dir
        target_set = set(target_names)

        # Copy non-safetensor assets: links are refused, except Hugging Face snapshot links into blobs/, and
        # every asset is opened before anything is written. The index is written below from the validated bytes.
        try:
            copy_checkpoint_assets(
                model_path,
                partial,
                skip=lambda name: name.endswith(".safetensors") or name.startswith(".") or name == INDEX_NAME,
            )
        except UnsafeCheckpointFile as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        out = OutputDir(partial)

        edit_report: list[dict[str, object]] = []
        edited_count = 0
        unchanged_vision_count = 0

        for shard_name, shard in shards.items():
            shard_tensors = load(shard.read_bytes())
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

            out.write_bytes(shard_name, save(modified_tensors))

        assert edited_count == cfg.abliteration.expected_target_count

        out.write_bytes(INDEX_NAME, index_bytes)
        source.close()
        out.close()

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
