#!/usr/bin/env python3
"""Curate a GPQA run for the repository (issue #35 repair re-gate), in the shape of the committed gpqa-*.json files.

Keeps aggregates only: protocol, serving, accuracy, the per-concurrency summary, wrong-run count and methodology. Every
per-item field (questions, options, model output, reasoning, answers) is dropped. Usage:
  curate_gpqa.py RAW_JSON OUT_JSON --serving TEXT"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTOCOL = "llm-inference-bench gpqa-diamond (chat template + thinking ON, temp 0, max_tokens 8192, concurrency 30)"
SUMMARY_DROP = {"wrong_runs", "runs", "items"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--serving", required=True)
    a = ap.parse_args()
    d = json.loads(Path(a.raw).read_text())
    md = d.get("metadata") or {}
    if md.get("max_tokens") != 8192 or md.get("temperature") != 0.0 or md.get("test_profile") != "gpqa-diamond":
        raise SystemExit(f"unexpected run settings: {md.get('test_profile')} max_tokens {md.get('max_tokens')} "
                         f"temperature {md.get('temperature')}")
    runs = [r for r in d["runs"] if r.get("phase") != "warmup"]
    summary = d["selected_summary"] if "selected_summary" in d else None
    levels = [{"concurrency": lv.get("concurrency"), "phase": lv.get("phase"),
               "summary": {k: v for k, v in (lv.get("summary") or {}).items() if k not in SUMMARY_DROP}}
              for lv in d.get("level_summaries", [])]
    out = {
        "protocol": PROTOCOL,
        "serving": a.serving,
        "accuracy": d["accuracy"],
        "level_summaries": levels,
        "wrong_runs_count": sum(1 for r in runs if r.get("correct") is not True),
        "methodology": d.get("methodology", ""),
    }
    if summary is not None:
        out["selected_summary"] = {k: v for k, v in summary.items() if k not in SUMMARY_DROP}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    acc = d["accuracy"]
    print(f"{Path(a.out).name}: {acc['correct']}/{acc['attempted']} scored {acc['scored']} errors {acc['errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
