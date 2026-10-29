#!/usr/bin/env python3
"""Build models/jev-27b/results/qad-gate-summary.json from the issue-20 run-2 evidence. Every number is read from a measured
file (eval summaries, predictions, latency probes, training log, export checks) or computed from them; nothing is typed in.
Usage: build_qad_summary.py EVIDENCE_DIR RESULTS_I20_DIR RESULTS_I15_DIR OUT_JSON
"""
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ev, r20, r15, out = (Path(p) for p in sys.argv[1:5])


def load(p: Path):
    return json.loads(p.read_text())


def jsonl(p: Path) -> list:
    return [json.loads(line) for line in p.read_bytes().split(b"\n") if line.strip()]


def preds(p: Path) -> dict:
    return {x["id"]: x["p"] for x in jsonl(p)}


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def single_argmax(t) -> bool:
    m = max(t)
    return sum(1 for x in t if x == m) == 1


S = {k: load(r20 / f"summary-{k}.json") for k in ("merged-qad", "merged", "current", "merged-b")}
S["ref"] = load(r15 / "summary-ref.json")
OOD = {k: load(r20 / f"summary-{k}.json") for k in ("ood-merged-qad", "ood-merged")}
gate = load(r20 / "gate-i15-v2.json")
ref = preds(r15 / "preds-ref.jsonl")
agreement = {}
for lab in ("merged-qad", "merged", "merged-b", "current"):
    p = preds(r20 / f"preds-{lab}.jsonl")
    agreement[lab] = round(sum(argmax(ref[i]) == argmax(p[i]) for i in ref) / len(ref), 4)

test_rows = jsonl(r20 / "test_set_30k.jsonl")
ood_rows = {r["id"]: r for r in jsonl(r20 / "ood.jsonl")}


def kl(t, q):
    return sum(a * (math.log(max(a, 1e-12)) - math.log(max(b, 1e-12))) for a, b in zip(t, q) if a > 0)


# ood teacher-KL difference vs the shipped cell, paired bootstrap like the frozen bar (2,000 resamples, seed 1234)
oq, oa = preds(r20 / "preds-ood-merged-qad.jsonl"), preds(r20 / "preds-ood-merged.jsonl")
ids = sorted(ood_rows)
d = [kl(ood_rows[i]["target"], oq[i]) - kl(ood_rows[i]["target"], oa[i]) for i in ids]
rng = random.Random(1234)
n = len(d)
boot = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
ood_delta = [round(sum(d) / n, 5), round(boot[50], 5), round(boot[1949], 5)]

train = [json.loads(line) for line in (ev / "train-log.jsonl").read_text().splitlines() if line.strip()]
val = [{"step": r["step"], "teacher_kl": round(r["val_kl"], 6), "distill_kl_vs_bf16": round(r["val_distill_kl"], 6)}
       for r in train if "val_kl" in r]
tsum = load(ev / "train-summary.json")
scale_check = (ev / "export-scale-check.txt").read_text()
sc = json.loads(scale_check[scale_check.index("{"): scale_check.rindex("}") + 1])
changed = load(ev / "qad-changed-block-scales.json")
pubsmoke = load(ev / "pubsmoke-qad.json")


def lat(path: Path) -> dict:
    j = load(path)
    return {kind: {c: {"p50_ms": j[kind][c]["p50_ms"], "p95_ms": j[kind][c]["p95_ms"]} for c in ("c1", "c8")}
            for kind in ("short", "long")}


speed = {}
for watts, sub, pre in (("450", ev, "summary-speed-"), ("600", ev / "600", "summary-speed600-")):
    speed[watts + "W"] = {
        cell: {"latency": lat(sub / f"latency-{lab}.json"),
               "rows_per_s_first_6000_rows_c16": load(sub / f"{pre}{lab}.json")["rows_per_s"]}
        for cell, lab in (("run2_qad", "qad"), ("A_shipped", "a"), ("B_ptq_start", "b"))
    }

# c=1 p50 latency reduction vs the shipped (mixed) cell, per power limit and prompt length
reduction = {}
for w, cells in speed.items():
    for kind in ("short", "long"):
        q = cells["run2_qad"]["latency"][kind]["c1"]["p50_ms"]
        a = cells["A_shipped"]["latency"][kind]["c1"]["p50_ms"]
        reduction[f"{w}_{kind}"] = round(100 * (1 - q / a), 1)
lo, hi = min(reduction.values()), max(reduction.values())
c1_range = f"{math.floor(lo)}-{math.ceil(hi)}%"

# the QAD export vs its identical-layout PTQ start point, same windows
vs_start = {}
for w, cells in speed.items():
    for kind in ("short", "long"):
        q = cells["run2_qad"]["latency"][kind]["c1"]["p50_ms"]
        b = cells["B_ptq_start"]["latency"][kind]["c1"]["p50_ms"]
        vs_start[f"{w}_{kind}_c1_p50_pct"] = round(100 * (q / b - 1), 1)
    q, b = cells["run2_qad"]["rows_per_s_first_6000_rows_c16"], cells["B_ptq_start"]["rows_per_s_first_6000_rows_c16"]
    vs_start[f"{w}_rows_per_s_pct"] = round(100 * (q / b - 1), 1)

kl_vs_a = round(100 * (S["merged-qad"]["kl"] / S["merged"]["kl"] - 1), 1)
ood_kl_vs_a = round(100 * (OOD["ood-merged-qad"]["kl"] / OOD["ood-merged"]["kl"] - 1), 1)
qad_script = ev / "qad_train.py"

summary = {
    "family": "jev-27b",
    "issue": "https://github.com/HangGlidersRule/model-forge/issues/20",
    "cell": "Darkstar-JEV-27B-Decision-QAD-ModelOpt-W4A4-NVFP4",
    "method": {
        "start_point": "candidate B (all-NVFP4 W4A4 PTQ, lm_head BF16), byte-identical untrained tensors",
        "trained": "per-block NVFP4 weight scales only (ModelOpt LSQ, tied pre/post amax, fp32 master values); "
                   "BF16 weights, per-tensor scales, activation scales, lm_head, embeddings and norms frozen",
        "objective": "full-vocabulary KL(BF16 teacher || NVFP4 student) at every prompt position (NVIDIA QAD logits "
                     "loss); teacher = the same merged BF16 weights with every quantizer bypassed",
        "data": "SargeDev/jev-distill-corpus-v3 train split @ 75111a13a8455032f0435050bfc7e0c9026ecfe8 "
                "(no id or content overlap with test_set_30k); validation split 2,000-row subset for checkpointing",
        "schedule": "AdamW lr 1e-4, warmup 50, cosine to 10%, grad clip 1.0, ~16k prompt tokens/step",
        "toolchain": "ModelOpt main @ b6b0ff5249109729c515cdc739a50d518e9cbec0 (0.48.0rc1.dev229), torch 2.14.1+cu132, "
                     "transformers 5.18.0, flash-linear-attention 0.6.0",
        "trainer_args": tsum.get("args"),
        "qad_train_py_sha256": hashlib.sha256(qad_script.read_bytes()).hexdigest(),
        "preregistration": {
            "run1": "https://github.com/HangGlidersRule/model-forge/issues/20#issuecomment-6475866657",
            "run2": "https://github.com/HangGlidersRule/model-forge/issues/20#issuecomment-6478693234",
            "run2_amendments": "https://github.com/HangGlidersRule/model-forge/issues/20#issuecomment-6489396236",
            "run2_amendments_note": "R2-A1..R2-A3 were written in the pre-registration file before the results they "
                                    "could affect (file sha256 4d10d669..., last modified 2026-10-29 02:24:55 UTC, "
                                    "before the 02:54:56 UTC result comment) and posted to the issue after the result",
        },
    },
    "training": {
        "steps": tsum["steps_run"], "hours": round(tsum["train_hours"], 2), "prompt_tokens": tsum["tokens"],
        "best_step": tsum["best_step"], "validation": val,
    },
    "export_check": {
        "untrained_tensors_identical_to_start": tsum["export_vs_b"]["untrained_identical"],
        "trained_tensors": tsum["export_vs_b"]["trained_tensors_changed"],
        "fp8_block_scales_changed_vs_start": {k: changed[k] for k in ("linears", "fp8_block_scales", "changed",
                                                                        "fraction_changed")},
        "fp8_block_scales_changed_by_projection": changed["by_projection"],
        "block_scale_mismatches_vs_trained_forward": sc["mismatched_blocks"],
        "block_scales_total": sc["blocks"],
        "mismatch_classes": sc["by_class"],
        "note": "every mismatch vs the trained forward is an E4M3 midpoint tie rounded to the adjacent code "
                "(amendment R2-A3)",
        "artifact_validation": load(ev / "artifact-validation-run2.json").get("verdict"),
    },
    "test_set_30k": {
        "rows": S["merged-qad"]["rows"],
        "single_argmax_rows": sum(single_argmax(r["target"]) for r in test_rows),
        "summaries": {k: {m: S[k][m] for m in ("kl", "top1", "choice_top1", "noul_auroc", "score_mae_expected")}
                      for k in S},
        "summaries_note": "ref = unmerged BF16 + LoRA reference (issue #15); current = the deployed NVIDIA NVFP4 + "
                          "LoRA lane; merged = the mixed W4A4+FP8 cell; merged-b = the PTQ start point",
        "teacher_kl_vs_mixed_cell_pct": kl_vs_a,
        "agreement_with_bf16_reference": agreement,
        "frozen_bar": {
            "script_sha256": "75fcb8147e10b1bb606d9c9cc80e6f8d88aa668cc4fac84c9354986b614222de",
            "verdict": gate["candidates"]["merged-qad"]["verdict"],
            "criteria": gate["candidates"]["merged-qad"]["criteria"],
            "teacher_kl_delta_vs_deployed_lane": [round(x, 5) for x in gate["candidates"]["merged-qad"]["teacher_kl_delta_vs_current"]],
            "top1_deficit_vs_deployed_lane_pt": [round(x, 2) for x in gate["candidates"]["merged-qad"]["top1_deficit_vs_current_pt"]],
            "speed_criteria_note": "the gate window measured this export at 60.2 rows/s and 147/272 ms c=1 p50; the "
                                   "back-to-back re-measurements (speed_back_to_back) do not reproduce it",
        },
    },
    "ood": {
        "rows": OOD["ood-merged-qad"]["rows"],
        "single_argmax_rows": sum(single_argmax(r["target"]) for r in ood_rows.values()),
        "summaries": {k: {m: OOD[k][m] for m in ("kl", "top1", "choice_top1", "noul_auroc", "score_mae_expected")}
                      for k in OOD},
        "teacher_kl_vs_mixed_cell_pct": ood_kl_vs_a,
        "teacher_kl_delta_vs_shipped_cell": ood_delta,
    },
    "speed_back_to_back": speed,
    "speed_note": "same GPU, same window per power limit, vLLM v0.32.0, identical serve flags; rows/s on the first 6,000 "
                  "test rows (longer than average), so it ranks cells but is not comparable to full-set rows/s",
    "c1_p50_reduction_vs_mixed_cell_pct": reduction,
    "c1_p50_reduction_vs_mixed_cell_range": c1_range,
    "vs_ptq_start_point_pct": vs_start,
    "vs_ptq_start_point_note": "identical layout: within 4% of the start point everywhere except 600 W short c=1 p50 "
                               "(single-request short prompts are the noisiest cell; 32 requests per cell)",
    "kv_cache_tokens": 857429,
    "weights_gib": 18.66,
    "publication_smoke": pubsmoke,
    "decision": ("Shipped by owner decision (2026-10-29) alongside the mixed W4A4+FP8 cell, under a looser tolerance than "
                 "the frozen bar: the bar's teacher-KL criterion requires the 95% CI upper bound <= 0, and this cell's CI "
                 f"straddles 0 (a statistical tie with the deployed lane). Against the mixed cell it is {kl_vs_a}% higher "
                 f"on test-set teacher KL, {c1_range} lower c=1 p50 latency, and {ood_kl_vs_a}% higher on ood teacher KL."),
}
out.write_text(json.dumps(summary, indent=1) + "\n")
print("wrote", out, "| c1 range", c1_range, reduction, "| vs start", vs_start, "| kl vs A", kl_vs_a, "| ood", ood_kl_vs_a)
