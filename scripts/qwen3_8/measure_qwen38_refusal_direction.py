#!/usr/bin/env python3
"""Measure refusal direction from official Qwen3.8-27B BF16 at layer 38.

Captures last-token residual activations for harmful/harmless prompts,
computes the normalized direction, and saves as safetensors with metrics manifest.
Requires GPU and the full model; not run in unit tests.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from model_forge.abliteration import compute_refusal_direction, mask_massive_activations
from model_forge.experiment import load_experiment
from model_forge.pipeline import (
    RunLock,
    StageContext,
    SuccessManifest,
    sha256_file,
)


def load_prompts(corpus_dir: Path) -> tuple[list[str], list[str]]:
    harmful: list[str] = []
    harmless: list[str] = []
    for line in (corpus_dir / "harmful.jsonl").read_text().strip().splitlines():
        harmful.append(json.loads(line)["text"])
    for line in (corpus_dir / "harmless.jsonl").read_text().strip().splitlines():
        harmless.append(json.loads(line)["text"])
    return harmful, harmless


def capture_activations(
    model: "torch.nn.Module",
    tokenizer: "object",
    prompts: list[str],
    layer_idx: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Capture last-token hidden states at the specified language model layer."""
    activations: list[torch.Tensor] = []
    hook_handle = None

    def hook_fn(module: torch.nn.Module, input: object, output: object) -> None:
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        activations.append(hidden[:, -1, :].detach().float().cpu())

    candidates = (
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "layers"),
    )
    layers = None
    for path in candidates:
        obj = model
        try:
            for part in path:
                obj = getattr(obj, part)
        except AttributeError:
            continue
        layers = obj
        break
    if layers is None:
        raise RuntimeError("Could not locate Qwen3.8 language-model layers")
    hook_handle = layers[layer_idx].register_forward_hook(hook_fn)

    try:
        for prompt in prompts:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)
    finally:
        if hook_handle:
            hook_handle.remove()

    return torch.cat(activations, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure refusal direction")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-stage", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    cfg_sha = cfg.config_sha()

    ctx = StageContext(args.run_root, "measure_direction", cfg_sha)
    if args.resume and not args.force_stage:
        existing = ctx.should_skip()
        if existing:
            print(f"Direction measurement verified, skipping (config_sha={cfg_sha[:12]})")
            return

    corpus_dir = args.corpus_dir or (args.run_root / "corpus")
    if not (corpus_dir / "harmful.jsonl").exists():
        print("ERROR: Corpus not materialized. Run materialize_abliteration_corpus.py first.", file=sys.stderr)
        sys.exit(1)

    harmful_prompts, harmless_prompts = load_prompts(corpus_dir)

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    print(f"Loading {cfg.source.model_id}@{cfg.source.revision[:12]}...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.source.model_id, revision=cfg.source.revision, trust_remote_code=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.source.model_id,
        revision=cfg.source.revision,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    torch.manual_seed(cfg.abliteration.seed)

    print(f"Capturing harmful activations ({len(harmful_prompts)} prompts)...")
    harmful_acts = capture_activations(model, tokenizer, harmful_prompts, cfg.abliteration.layer, args.device)
    print(f"Capturing harmless activations ({len(harmless_prompts)} prompts)...")
    harmless_acts = capture_activations(model, tokenizer, harmless_prompts, cfg.abliteration.layer, args.device)

    harmful_count = len(harmful_acts)
    combined, massive_dimensions = mask_massive_activations(
        torch.cat((harmful_acts, harmless_acts), dim=0)
    )
    harmful_acts = combined[:harmful_count]
    harmless_acts = combined[harmful_count:]
    print(
        "Massive-activation mask: "
        f"{int(massive_dimensions.sum().item())}/{massive_dimensions.numel()} dimensions"
    )

    result = compute_refusal_direction(
        harmful_acts,
        harmless_acts,
        orthogonalize_harmless=cfg.abliteration.orthogonalize_harmless,
        seed=cfg.abliteration.seed,
        layer=cfg.abliteration.layer,
    )

    lock = RunLock(args.run_root / "measure_direction.lock")
    with lock:
        partial = ctx.partial_dir
        from safetensors.torch import save_file

        tensors = {
            "direction": result.direction,
            "harmful_mean": result.harmful_mean,
            "harmless_mean": result.harmless_mean,
        }
        save_file(tensors, str(partial / "direction.safetensors"))

        metrics = {
            "layer": result.layer,
            "seed": result.seed,
            "raw_diff_norm": result.raw_diff_norm,
            "direction_shape": list(result.direction.shape),
            "harmful_samples": harmful_acts.shape[0],
            "harmless_samples": harmless_acts.shape[0],
            "massive_activation_dimensions": massive_dimensions.nonzero().flatten().tolist(),
        }
        (partial / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

        output_hashes = {
            "direction.safetensors": sha256_file(partial / "direction.safetensors"),
            "metrics.json": sha256_file(partial / "metrics.json"),
        }
        manifest = SuccessManifest(
            stage="measure_direction",
            config_sha=cfg_sha,
            source_revisions={"model": cfg.source.revision},
            output_hashes=output_hashes,
            metrics=metrics,
        )
        ctx.promote(manifest)
        print(f"Direction saved (norm={result.raw_diff_norm:.4f}, dim={result.direction.shape[0]})")


if __name__ == "__main__":
    main()
