#!/usr/bin/env python3
"""HISTORICAL / REJECTED: llm-compressor compressed-tensors NVFP4 path.

Do not use for new publication builds. Retained for lineage and tests only.
Replace with ``quantize_qwen38_modelopt.py`` and
``configs/modelopt/recipes/nvfp4_mlp_only_mse-kv_bf16.yaml``.

Original behavior: quantize edited Qwen3.8 BF16 to NVFP4 W4A4 compressed-tensors,
retaining source MTP via graft when the compressor omitted it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from model_forge.experiment import load_experiment
from model_forge.pipeline import (
    RunLock,
    StageContext,
    SuccessManifest,
    sha256_file,
)


def build_ignore_list(cfg_ignore: tuple[str, ...]) -> list[str]:
    return list(cfg_ignore)


def graft_source_mtp(source_dir: Path, output_dir: Path) -> int:
    """Restore source-native BF16 MTP tensors if the compressor omitted them."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    source_index_path = source_dir / "model.safetensors.index.json"
    output_index_path = output_dir / "model.safetensors.index.json"
    source_index = json.loads(source_index_path.read_text())
    output_index = json.loads(output_index_path.read_text())
    source_map = source_index["weight_map"]
    output_map = output_index["weight_map"]
    mtp_names = sorted(name for name in source_map if name.startswith("mtp."))
    if len(mtp_names) != 15:
        raise RuntimeError(f"Expected 15 source MTP tensors, found {len(mtp_names)}")
    if all(name in output_map for name in mtp_names):
        return 0

    tensors = {}
    by_shard: dict[str, list[str]] = {}
    for name in mtp_names:
        by_shard.setdefault(source_map[name], []).append(name)
    for shard, names in by_shard.items():
        with safe_open(source_dir / shard, framework="pt", device="cpu") as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                if str(tensor.dtype) != "torch.bfloat16":
                    raise RuntimeError(f"MTP tensor {name} is {tensor.dtype}, expected bfloat16")
                tensors[name] = tensor

    target = output_dir / "model-mtp-bf16.safetensors"
    save_file(tensors, target, metadata={"format": "pt"})
    for name in mtp_names:
        output_map[name] = target.name
    added_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    metadata = output_index.setdefault("metadata", {})
    metadata["total_size"] = int(metadata.get("total_size", 0)) + added_bytes
    output_index_path.write_text(json.dumps(output_index, indent=2, sort_keys=True) + "\n")

    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text())
    config.setdefault("text_config", {})["mtp_num_hidden_layers"] = 1
    ignore = config.setdefault("quantization_config", {}).setdefault("ignore", [])
    for name in mtp_names:
        module = name.rsplit(".weight", 1)[0]
        if module not in ignore:
            ignore.append(module)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return len(tensors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize to NVFP4")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--offload-dir", type=Path, default=None)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-stage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    cfg_sha = cfg.config_sha()
    qcfg = cfg.quantization

    source_dir = args.source_dir or (args.run_root / "apply_abliteration")
    offload_dir = args.offload_dir or (args.run_root / "offload")

    ctx = StageContext(args.run_root, "quantize_nvfp4", cfg_sha)
    if args.resume and not args.force_stage:
        existing = ctx.should_skip()
        if existing:
            print(f"Quantization stage verified, skipping (config_sha={cfg_sha[:12]})")
            return

    if not source_dir.exists():
        print(f"ERROR: Source model not found at {source_dir}", file=sys.stderr)
        sys.exit(1)

    ignore_list = build_ignore_list(qcfg.ignore)

    if args.dry_run:
        print("DRY RUN - Quantization config:")
        print(f"  Source: {source_dir}")
        print(f"  Scheme: {qcfg.scheme}")
        print(f"  Targets: {qcfg.targets}")
        print(f"  Ignore: {ignore_list}")
        print(f"  Calibration: {qcfg.calibration_samples} samples x {qcfg.max_sequence_length} tokens")
        print(f"  Pipeline: {qcfg.pipeline}")
        print(f"  Shard size: {qcfg.shard_size}")
        return

    from compressed_tensors.offload import load_offloaded_model
    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    lock = RunLock(args.run_root / "quantize_nvfp4.lock")
    with lock:
        partial = ctx.partial_dir
        offload_dir.mkdir(parents=True, exist_ok=True)

        with load_offloaded_model(AutoModelForImageTextToText):
            model = AutoModelForImageTextToText.from_pretrained(
                str(source_dir),
                dtype="auto",
                device_map=args.device_map,
                offload_folder=str(offload_dir),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(str(source_dir), trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(str(source_dir), trust_remote_code=True)

            ds = load_dataset(
                qcfg.calibration_dataset,
                qcfg.calibration_config,
                split=f"train[:{qcfg.calibration_samples}]",
            )
            column = "text" if "text" in ds.column_names else ds.column_names[0]
            tok = getattr(processor, "tokenizer", processor)

            def tokenize(row: dict) -> dict:
                return tok(
                    row[column],
                    truncation=True,
                    max_length=qcfg.max_sequence_length,
                    add_special_tokens=True,
                )

            cal_ds = ds.map(tokenize, remove_columns=ds.column_names)

            recipe = QuantizationModifier(
                targets=qcfg.targets, scheme=qcfg.scheme, ignore=ignore_list
            )
            oneshot(
                model=model,
                processor=processor,
                dataset=cal_ds,
                recipe=recipe,
                max_seq_length=qcfg.max_sequence_length,
                num_calibration_samples=qcfg.calibration_samples,
                pipeline=qcfg.pipeline,
            )
            model.save_pretrained(
                partial,
                save_compressed=True,
                max_shard_size=qcfg.shard_size,
                save_original_format=False,
            )
            tokenizer.save_pretrained(partial)
            processor.save_pretrained(partial)

        grafted_mtp = graft_source_mtp(source_dir, partial)
        shutil.rmtree(offload_dir, ignore_errors=True)

        # Compute output hashes
        output_hashes: dict[str, str] = {}
        for f in sorted(partial.rglob("*")):
            if f.is_file() and f.name != "_SUCCESS.json":
                output_hashes[str(f.relative_to(partial))] = sha256_file(f)

        manifest = SuccessManifest(
            stage="quantize_nvfp4",
            config_sha=cfg_sha,
            output_hashes=output_hashes,
            metrics={
                "scheme": qcfg.scheme,
                "calibration_samples": qcfg.calibration_samples,
                "ignore": ignore_list,
                "grafted_source_mtp_tensors": grafted_mtp,
            },
        )
        ctx.promote(manifest)
        print(f"Quantization complete: {ctx.stage_dir}")


if __name__ == "__main__":
    main()
