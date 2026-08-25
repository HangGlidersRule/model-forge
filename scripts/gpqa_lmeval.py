#!/usr/bin/env python3
"""GPQA Diamond via the stock lm-evaluation-harness tasks (accepted wheel).

Uses EleutherAI lm-evaluation-harness `gpqa_diamond_zeroshot` (log-prob
multiple choice) and `gpqa_diamond_cot_zeroshot` (greedy CoT generate) against
a served OpenAI-compatible endpoint. This is the same harness NVIDIA used for
the published Lightning GPQA numbers (Nemo Evaluator SDK + LM Evaluation
Harness container). Do NOT use the legacy model_forge.gpqa.harness for new
claims — see docs/gpqa-abliteration-protocol.md.

Requires: pip install lm-eval   (in a venv that also has torch)

Usage:
  python scripts/gpqa_lmeval.py \
    --base http://127.0.0.1:8105 \
    --model lightning-ablit-bf16 \
    --out models/nemotron-3.5-lightning-r1/results/gpqa-diamond-abliterated-bf16.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8105")
    ap.add_argument("--model", required=True, help="served model name on the endpoint")
    ap.add_argument("--out", type=Path, required=True, help="output JSON path")
    ap.add_argument("--tasks", default="gpqa_diamond_zeroshot,gpqa_diamond_cot_zeroshot")
    ap.add_argument("--tokenizer", default=None, help="HF id/local path for tokenizer (same family as served model); default: use --model as tokenizer id")
    ap.add_argument("--n-samples", type=int, default=None, help="optional limit for smoke runs")
    args = ap.parse_args()

    try:
        import lm_eval  # noqa: F401
    except ImportError:
        sys.exit("lm-eval is not installed. Run: pip install 'lm-eval[math]' (or lm-eval) in a torch-enabled venv.")

    from lm_eval import simple_evaluate

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    # The API client needs a tokenizer for formatting; use the official HF id of
    # the same model family so chat/formatting matches the served weights.
    tokenizer = args.tokenizer or args.model
    # local-completions posts to the full completions URL; accept a server root too.
    base = args.base.rstrip("/")
    if not base.endswith("/completions"):
        base = f"{base}/v1/completions"
    model_args = f"base_url={base},model={args.model},tokenizer={tokenizer}"

    results = simple_evaluate(
        model="local-completions",
        model_args=model_args,
        tasks=tasks,
        limit=args.n_samples,
        verbosity="INFO",
    )

    out: dict = {
        "protocol": "lm-evaluation-harness (EleutherAI) stock gpqa_diamond tasks",
        "base": args.base,
        "served_model": args.model,
        "tasks": {},
    }
    if results is None:
        sys.exit("lm_eval.simple_evaluate returned None (evaluation failed)")
    for task in tasks:
        task_results = (results.get("results") or {}).get(task, {})
        keys = [k for k in task_results if k not in ("alias", "samples")]
        out["tasks"][task] = {k: task_results[k] for k in keys}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
