#!/usr/bin/env python3
"""Measure refusal directions from NVIDIA Nemotron-3.5-Lightning BF16.

Canonical abliteration (mlabonne) adapted for nemotron_h:
1. Chat-templated prompts (role user, add_generation_prompt=True).
2. Capture resid_pre (block input) last-token activations for ALL layers in
   one forward pass per prompt class.
3. Per-layer directions: normalize(mean_harmful - mean_harmless).
4. Auto-select the best candidate by inference-time ablation generations
   against the test harmful set (refusal-marker blacklist), not raw norm.
Writes directions + scores; requires GPU; not run in unit tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model_forge.experiment import load_experiment  # noqa: E402

REFUSAL_BLACKLIST = ("i cannot", "i can't", "cannot comply", "can't comply", "i'm sorry", "i am sorry", "i apologize")
TEST_N = 8  # harmful test prompts for candidate selection


# transformers _can_use_grouped_mm gates on torch version + SM80+, but the
# bundled torch._grouped_mm kernel only supports SM90 — on SM120 the check lies
# and the real kernel raises. Force the per-expert fallback path.
try:
    import transformers.integrations.moe as _moe_mod

    _moe_mod._can_use_grouped_mm = lambda *a, **k: False  # type: ignore[attr-defined]
except Exception:
    pass


def load_prompts(corpus_dir: Path) -> tuple[list[str], list[str]]:
    def read(name: str) -> list[str]:
        rows = [json.loads(line) for line in (corpus_dir / name).read_text().splitlines()]
        return [r.get("prompt") or r["text"] for r in rows]

    return read("harmful.jsonl"), read("harmless.jsonl")


def chat_tokenize(tokenizer, texts: list[str]) -> torch.Tensor:
    """Apply the model chat template with generation prompt (canonical)."""
    msgs = [[{"role": "user", "content": t}] for t in texts]
    return tokenizer.apply_chat_template(
        msgs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )["input_ids"]


def save_file(tensors: dict, path: Path) -> None:
    from safetensors.torch import save_file as sf

    sf(tensors, str(path))


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--corpus-dir", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-stage", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-path", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_experiment(args.config)
    cfg_sha = cfg.config_sha()

    corpus_dir = args.corpus_dir or (args.run_root / "corpus")
    harmful, harmless = load_prompts(corpus_dir)
    print(f"Corpus: {len(harmful)} harmful, {len(harmless)} harmless")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = str(args.model_path) if args.model_path else cfg.source.model_id
    print(f"Loading {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    )
    model.eval()
    torch.manual_seed(cfg.abliteration.seed)

    # Locate backbone.layers
    candidates = (("model", "backbone", "layers"), ("model", "layers"), ("backbone", "layers"))
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
        raise RuntimeError("Could not locate Nemotron-H backbone.layers")

    def capture(texts: list[str]) -> torch.Tensor:
        """Chunked forward passes; return [n_prompts, n_layers, hidden] resid_pre acts."""
        n_all = len(texts)
        n_layers = len(layers)
        all_acts = torch.zeros(n_all, n_layers, 2688, dtype=torch.float32)
        BS = 16
        for start in range(0, n_all, BS):
            chunk = texts[start : start + BS]
            tensors = chat_tokenize(tokenizer, chunk)
            buffers: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
            handles = []

            def make_prehook(i: int):
                def prehook(module, args_):
                    hidden = args_[0]
                    buffers[i].append(hidden[:, -1, :].detach().float().cpu())

                return prehook

            for i, layer in enumerate(layers):
                handles.append(layer.register_forward_pre_hook(make_prehook(i)))
            try:
                with torch.inference_mode():
                    model(tensors.to(args.device))
            finally:
                for h in handles:
                    h.remove()
            for i in range(n_layers):
                all_acts[start : start + BS, i, :] = torch.cat(buffers[i], dim=0)
            del tensors
            torch.cuda.empty_cache()
        return all_acts  # [n, L, hidden]

    print(f"Capturing harmful activations ({len(harmful)} prompts, chat-templated)...")
    harmful_acts = capture(harmful)
    print(f"Capturing harmless activations ({len(harmless)} prompts, chat-templated)...")
    harmless_acts = capture(harmless)

    # Per-layer directions
    dirs = {}
    for i in range(len(layers)):
        h_mean = harmful_acts[:, i, :].mean(dim=0)
        n_mean = harmless_acts[:, i, :].mean(dim=0)
        d = h_mean - n_mean
        norm = d.norm()
        if norm == 0:
            continue
        dirs[i] = d / norm

    # Evaluate top candidates by inference-time ablation on test harmful set
    test_prompts = harmful[:TEST_N]
    test_tokens = chat_tokenize(tokenizer, test_prompts)

    def generate_ablated(direction: torch.Tensor, max_new: int = 32) -> list[str]:
        direction = direction.to(args.device)
        gen = test_tokens.to(args.device)
        out_tokens = []
        handles = []

        def make_hook(layer_idx: int):
            def hook(module, args_):
                hidden = args_[0]
                d = direction.to(hidden.dtype)
                proj = (hidden @ d).unsqueeze(-1) * d
                return (hidden - proj, *args_[1:])

            return hook

        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(make_hook(i)))
        try:
            with torch.inference_mode():
                for _ in range(max_new):
                    logits = model(gen)[0][:, -1, :]
                    nxt = logits.argmax(dim=-1, keepdim=True)
                    gen = torch.cat([gen, nxt], dim=1)
                    out_tokens.append(nxt)
        finally:
            for h in handles:
                h.remove()
        ids = torch.cat(out_tokens, dim=1)
        return tokenizer.batch_decode(ids, skip_special_tokens=True)

    ranked = sorted(dirs.keys(), key=lambda i: abs(dirs[i].mean()), reverse=True)[:12]
    scored = []
    for i in ranked:
        outs = generate_ablated(dirs[i])
        refusals = sum(1 for o in outs if any(b in o.lower() for b in REFUSAL_BLACKLIST))
        scored.append((refusals, i, round(abs(dirs[i].mean().item()), 6)))
        print(f"layer {i}: refusals {refusals}/{TEST_N} mean {abs(dirs[i].mean().item()):.6f}")
    scored.sort(key=lambda x: (x[0], x[2]))
    best_refusals, best_layer, best_mean = scored[0]
    direction = dirs[best_layer]
    print(f"SELECTED layer {best_layer} ({best_refusals}/{TEST_N} refusals, mean {best_mean})")

    # Write outputs
    run = args.run_root / "measure_direction"
    run.mkdir(parents=True, exist_ok=True)
    save_file({"direction": direction, "layer_index": torch.tensor([best_layer])}, run / "direction.safetensors")
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "layer": best_layer,
                "selected_layer": best_layer,
                "seed": cfg.abliteration.seed,
                "raw_diff_norm": float(dirs[best_layer].norm()),
                "direction_shape": list(direction.shape),
                "harmful_samples": len(harmful),
                "harmless_samples": len(harmless),
                "candidate_scores": scored,
                "chat_templated": True,
            },
            indent=2,
        )
        + "\n"
    )
    with open(run / "manifest.json", "w") as f:
        json.dump(
            {
                "stage": "measure_direction",
                "config_sha": cfg_sha[:16],
                "output_hashes": {
                    "direction.safetensors": sha256_file(run / "direction.safetensors"),
                    "metrics.json": sha256_file(run / "metrics.json"),
                },
            },
            f,
        )
    print(f"Direction saved (layer={best_layer}, norm={dirs[best_layer].norm():.4f}, dim={direction.shape[0]})")


if __name__ == "__main__":
    main()
