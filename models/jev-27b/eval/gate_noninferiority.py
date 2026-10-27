#!/usr/bin/env python3
"""Issue-15 gate v2: non-inferiority to the deployed lane (Devin's go, 2026-10-27, after candidate A
failed the v1 bar of "statistically indistinguishable from BF16").

Bar, fixed before any candidate-B result existed. A candidate X passes when ALL hold vs `current`
(NVIDIA NVFP4 + runtime LoRA), on the same 29,955 rows:
  1. teacher KL no worse:        paired 95% CI upper bound of KL(X) - KL(current) <= 0
  2. top-1 vs teacher no worse:  paired 95% CI upper bound of [top1(current) - top1(X)] <= 0.5 pt
                                 (non-inferiority margin, on the 27,324 single-argmax rows)
  3. faster:                     rows/s (c=16) >= current, AND /v1/decide p50 at c=1 <= current for
                                 short (~300 tok) and long (~3.8k tok) prompts
Reported but not gating: KL vs the BF16 reference, top-1 agreement with the reference, choice top-1,
yes/no AUROC, score MAE, noise floor, KV cache, the margin distribution of decision flips.
Among passing candidates, the one with the lowest c=1 p50 on long prompts ships; ties go to lower
teacher KL. Bootstrap: paired, 2,000 resamples, seed 1234, percentile CI.
Usage: gate_i15_v2.py DIR cand [cand ...]   (needs preds/summary/latency for current + each cand)
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path


def kl(p, q):
    return sum(pi * (math.log(max(pi, 1e-12)) - math.log(max(qi, 1e-12))) for pi, qi in zip(p, q) if pi > 0)


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def boot(xs, rng, n_boot=2000):
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(xs) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def main() -> int:
    d = Path(sys.argv[1])
    cands = sys.argv[2:]
    rows = {}
    for line in (d / "test_set_30k.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    def load(lab):
        p = [json.loads(l) for l in (d / f"preds-{lab}.jsonl").read_text().splitlines() if l.strip()]
        s = json.loads((d / f"summary-{lab}.json").read_text())
        lat = json.loads((d / f"latency-{lab}.json").read_text())
        assert s["rows"] == len(p) == len(rows) == 29955, f"{lab}: denominator"
        return {x["id"]: x["p"] for x in p}, s, lat
    cur, cur_s, cur_lat = load("current")
    ids = sorted(rows)
    single = [i for i in ids if sum(1 for x in rows[i]["target"] if x == max(rows[i]["target"])) == 1]
    report = {"bar": "non-inferior to current (NVIDIA NVFP4 + LoRA): teacher-KL CI upper <= 0; top-1 deficit CI upper <= 0.5 pt; faster rows/s and c=1 p50 short+long", "candidates": {}}
    for cand in cands:
        rng = random.Random(1234)
        x, xs, xlat = load(cand)
        dkl = [kl(rows[i]["target"], x[i]) - kl(rows[i]["target"], cur[i]) for i in ids]
        m_kl, lo_kl, hi_kl = boot(dkl, rng)
        dtop = [int(argmax(cur[i]) == argmax(rows[i]["target"])) - int(argmax(x[i]) == argmax(rows[i]["target"])) for i in single]
        m_t, lo_t, hi_t = boot(dtop, rng)
        crit = {
            "teacher_kl_not_worse": hi_kl <= 0,
            "top1_deficit_within_0p5pt": 100 * hi_t <= 0.5,
            "rows_per_s_ge_current": xs["rows_per_s"] >= cur_s["rows_per_s"],
            "p50_c1_short_le_current": xlat["short"]["c1"]["p50_ms"] <= cur_lat["short"]["c1"]["p50_ms"],
            "p50_c1_long_le_current": xlat["long"]["c1"]["p50_ms"] <= cur_lat["long"]["c1"]["p50_ms"],
        }
        report["candidates"][cand] = {
            "verdict": "PASS" if all(crit.values()) else "FAIL", "criteria": crit,
            "teacher_kl_delta_vs_current": [m_kl, lo_kl, hi_kl],
            "top1_deficit_vs_current_pt": [100 * m_t, 100 * lo_t, 100 * hi_t],
            "rows_per_s": [xs["rows_per_s"], cur_s["rows_per_s"]],
            "p50_c1_short_ms": [xlat["short"]["c1"]["p50_ms"], cur_lat["short"]["c1"]["p50_ms"]],
            "p50_c1_long_ms": [xlat["long"]["c1"]["p50_ms"], cur_lat["long"]["c1"]["p50_ms"]],
        }
        r = report["candidates"][cand]
        print(f"{cand}: {r['verdict']}")
        print(f"  teacher-KL delta vs current {m_kl:+.5f} (95% CI {lo_kl:+.5f}, {hi_kl:+.5f})  -> {'ok' if crit['teacher_kl_not_worse'] else 'FAIL'}")
        print(f"  top-1 deficit vs current {100*m_t:+.2f} pt (95% CI {100*lo_t:+.2f}, {100*hi_t:+.2f})  -> {'ok' if crit['top1_deficit_within_0p5pt'] else 'FAIL'}")
        print(f"  rows/s {xs['rows_per_s']} vs {cur_s['rows_per_s']} | p50 c1 short {xlat['short']['c1']['p50_ms']} vs {cur_lat['short']['c1']['p50_ms']} | long {xlat['long']['c1']['p50_ms']} vs {cur_lat['long']['c1']['p50_ms']}")
    passing = [c for c in cands if report["candidates"][c]["verdict"] == "PASS"]
    if passing:
        winner = min(passing, key=lambda c: (report["candidates"][c]["p50_c1_long_ms"][0], report["candidates"][c]["teacher_kl_delta_vs_current"][0]))
        report["winner"] = winner
        print(f"WINNER {winner}")
    else:
        report["winner"] = None
        print("WINNER none")
    (d / "gate-i15-v2.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
