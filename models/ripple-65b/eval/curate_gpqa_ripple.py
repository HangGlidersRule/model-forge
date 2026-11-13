#!/usr/bin/env python3
"""Ripple-65B Preview (#39): curate one llm-inference-bench GPQA Diamond run for the repository.

Keeps aggregates only: the run's protocol settings, accuracy, the reasoning-block share (R39-A6), truncation counts,
completion-token percentiles and the methodology text. Every per-item field (questions, options, model output,
reasoning, answers, item ids) is dropped. Fails closed unless the run used the posted protocol (gpqa-diamond, 198
items, temperature 0, max_tokens 65,536, concurrency 30, the pinned dataset sha256).
Usage: curate_gpqa_ripple.py RAW_JSON OUT_JSON --cell vendor|A --run 1|2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DATASET_SHA256 = "a8472c5a82ea2df8f209c17713aba1a6d409120c609ec0582dae0cb940c7e28c"
PROTOCOL = ("llm-inference-bench 0.4.29 gpqa-diamond: 198 items, chat template, temperature 0, max_tokens 65,536, "
            "concurrency 30, deterministic letter shuffle; serve flags --reasoning-parser ripple --max-model-len 131072")
KEEP_ACCURACY = ("items_total", "attempted", "scored", "correct", "accuracy", "wilson95_low", "wilson95_high",
                 "unparseable", "truncated_no_answer", "hit_max_tokens", "errors")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--cell", required=True, choices=("vendor", "A"))
    ap.add_argument("--run", required=True, type=int, choices=(1, 2))
    a = ap.parse_args()
    raw_bytes = Path(a.raw).read_bytes()
    d = json.loads(raw_bytes)
    md = d.get("metadata") or {}
    want = {"test_profile": "gpqa-diamond", "max_tokens": 65536, "fixed_concurrency": 30, "dataset_sha256": DATASET_SHA256,
            "version": "0.4.29"}
    bad = {k: md.get(k) for k, v in want.items() if md.get(k) != v}
    runs = [r for r in d["runs"] if r.get("phase") != "warmup"]
    if bad or len(runs) != 198 or len({r["item_id"] for r in runs}) != 198:
        raise SystemExit(f"unexpected run settings {bad} or item count {len(runs)}")
    acc = d["accuracy"]
    if acc.get("correct") != sum(bool(r.get("correct")) for r in runs):
        raise SystemExit("accuracy block disagrees with the per-item rows")
    toks = sorted(int(r.get("completion_tokens") or 0) for r in runs)
    with_reasoning = sum(1 for r in runs if (r.get("reasoning_text") or "").strip())
    out = {
        "cell": a.cell,
        "run": a.run,
        "protocol": PROTOCOL,
        "bench_version": md.get("version"),
        "dataset_sha256": md.get("dataset_sha256"),
        "max_tokens": md.get("max_tokens"),
        "concurrency": md.get("fixed_concurrency"),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "accuracy": {k: acc.get(k) for k in KEEP_ACCURACY},
        "items_with_reasoning_block": with_reasoning,
        "finish_reasons": dict(sorted({fr: sum(1 for r in runs if r.get("finish_reason") == fr)
                                        for fr in {r.get("finish_reason") for r in runs}}.items())),
        "completion_tokens": {"p50": toks[len(toks) // 2], "p90": toks[int(len(toks) * 0.9)], "max": toks[-1],
                              "mean": round(sum(toks) / len(toks), 1)},
        "categories": dict(sorted({c: sum(1 for r in runs if r.get("category") == c and r.get("correct"))
                                    for c in {r.get("category") for r in runs}}.items())),
        "methodology": d.get("methodology", ""),
    }
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{Path(a.out).name}: {acc['correct']}/198, reasoning block on {with_reasoning}, truncated without an answer "
          f"{acc.get('truncated_no_answer')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
