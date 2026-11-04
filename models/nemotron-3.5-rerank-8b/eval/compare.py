#!/usr/bin/env python3
"""Issue #32 quality comparison: paired per-query deltas against a control, 10,000-resample bootstrap (seed 1234)
for 95% CIs on macro nDCG@10 (resampling queries within each dataset, so the macro weighting is preserved), plus
Spearman / top-10 overlap of the per-pair scores against the control. Ratio = candidate macro / control macro (the
pre-registered bar: >= 0.98 on each set). Usage: compare.py RESULTS_DIR"""
import json
import random
import sys
from pathlib import Path

D = Path(sys.argv[1])
SETS = ("beir6_bm25", "lmeb_dialogue")
RUNS = ("bf16r", "A", "B", "C", "qwen-bf16", "qwen-fp4")


def load(label, s):
    return json.loads((D / f"q-{label}-{s}.json").read_text())


def spearman(a, b):
    n = len(a)
    ra = {i: r for r, i in enumerate(sorted(range(n), key=lambda k: a[k]))}
    rb = {i: r for r, i in enumerate(sorted(range(n), key=lambda k: b[k]))}
    return 1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


out = {}
for s in SETS:
    ctrl = load("bf16", s)["quality"]
    out[s] = {"bf16": {"macro": ctrl["macro_ndcg10"]}}
    for run in RUNS:
        cand = load(run, s)["quality"]
        per_ds = {}
        for ds, v in ctrl["per_set"].items():
            c_q = {p["qid"]: p for p in v["per_query"]}
            k_q = {p["qid"]: p for p in cand["per_set"][ds]["per_query"]}
            assert c_q.keys() == k_q.keys(), (run, ds)
            per_ds[ds] = [(k_q[q]["ndcg10"] - c_q[q]["ndcg10"], c_q[q], k_q[q]) for q in sorted(c_q)]
        delta = sum(sum(d for d, _, _ in rows) / len(rows) for rows in per_ds.values()) / len(per_ds)
        rng = random.Random(1234)
        boots = []
        keys = list(per_ds)
        for _ in range(10000):
            m = 0.0
            for ds in keys:
                rows = per_ds[ds]
                m += sum(rows[rng.randrange(len(rows))][0] for _ in rows) / len(rows)
            boots.append(m / len(keys))
        boots.sort()
        rhos, overlaps = [], []
        for rows in per_ds.values():
            for _, c, k in rows:
                rhos.append(spearman(c["scores"], k["scores"]))
                tc = set(sorted(range(len(c["scores"])), key=lambda i: -c["scores"][i])[:10])
                tk = set(sorted(range(len(k["scores"])), key=lambda i: -k["scores"][i])[:10])
                overlaps.append(len(tc & tk) / 10)
        out[s][run] = {"macro": cand["macro_ndcg10"], "ratio": cand["macro_ndcg10"] / ctrl["macro_ndcg10"],
                       "delta": delta, "ci95": [boots[249], boots[9749]],
                       "spearman": sum(rhos) / len(rhos), "top10_overlap": sum(overlaps) / len(overlaps),
                       "per_dataset": {ds: cand["per_set"][ds]["ndcg10"] for ds in cand["per_set"]}}
    print(f"== {s} (BF16 macro nDCG@10 {ctrl['macro_ndcg10']:.4f}, {sum(v['n_queries'] for v in ctrl['per_set'].values())} queries)")
    for run in RUNS:
        r = out[s][run]
        bar = "PASS" if r["ratio"] >= 0.98 else "FAIL"
        same_model = run in ("bf16r", "A", "B", "C")
        print(f"  {run:10s} macro {r['macro']:.4f}  ratio {r['ratio']:.4f}  delta {r['delta']:+.4f} "
              f"(95% CI {r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f})  "
              + (f"spearman {r['spearman']:.3f}  top10 {r['top10_overlap']:.3f}  bar {bar}" if same_model else "(different model)"))
(D / "quality-summary.json").write_text(json.dumps(out, indent=1))
print("SUMMARY-WRITTEN", D / "quality-summary.json")
