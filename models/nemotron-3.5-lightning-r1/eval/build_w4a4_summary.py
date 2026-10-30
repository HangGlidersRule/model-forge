#!/usr/bin/env python3
"""Build the issue-17 gate summary for the Nemotron-3.5-Lightning W4A4 NVFP4 cell from the raw result files.

Every number the card, family README, ledger and release notes quote is computed here and written to
results/w4a4-gate-summary.json; the card checker recomputes from that file. Per-item GPQA rows, KL position arrays and
activation histograms stay out of the public record: only aggregates are written.
Usage: build_w4a4_summary.py RAW_DIR OUT_DIR
  RAW_DIR: gpqa-{BF16,BF16-replica,W4A16,A,mse}.json, kl-{BF16,W4A16,W4A16-b12x,A,mse}.json,
           kl-sel-{BF16,A,mse,b12xeq,p9999,p999}.json, throughput-{W4A16,W4A16-b12x,A,mse,darkstar}.json,
           variants-G.json, check-export-A.json, release-validation.json, act-stats.json, sel-result.json"""
import json
import math
import random
import statistics
import sys
from pathlib import Path

raw, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)


def load(name):
    return json.loads((raw / name).read_text())


# ---------- GPQA ----------
CELLS = ("BF16", "BF16-replica", "W4A16", "A", "mse")
gp = {c: load(f"gpqa-{c}.json") for c in CELLS}
rows = {c: {r["item_id"]: r for r in gp[c]["runs"] if r.get("phase") != "warmup"} for c in CELLS}
ids = sorted(set.intersection(*(set(v) for v in rows.values())))
n = len(ids)


def ok(c, i):
    return bool(rows[c][i].get("correct"))


def boot(a, b, seed=1234, reps=10000):
    rng = random.Random(seed)
    vals = sorted(sum(ok(a, j) - ok(b, j) for j in (ids[rng.randrange(n)] for _ in range(n))) / n for _ in range(reps))
    return [round(vals[int(0.025 * reps) - 1], 4), round(vals[int(0.975 * reps) - 1], 4)]


disc = sum(ok("BF16", i) != ok("BF16-replica", i) for i in ids)
band = 1.96 * math.sqrt(disc / n / n)
gpqa = {"items": n, "dataset_sha256": gp["BF16"]["metadata"].get("dataset_sha256") if "metadata" in gp["BF16"] else None,
        "cells": {}, "noise": {"bf16_discordant_items": disc, "band": round(band, 4), "band_questions": round(band * n, 1)}}
for c in CELLS:
    a = gp[c]["accuracy"]
    toks = [rows[c][i].get("completion_tokens") or 0 for i in ids]
    no_response = sorted(i for i in ids if rows[c][i].get("ok") is False)
    gpqa["cells"][c] = {"correct": a["correct"], "attempted": a["attempted"], "accuracy": round(a["correct"] / a["attempted"], 4),
                        "unparseable": a["unparseable"], "truncated_no_answer": a["truncated_no_answer"],
                        "hit_max_tokens": a["hit_max_tokens"], "errors": a["errors"],
                        "no_response_items": len(no_response),
                        "no_response_error_types": sorted({str(rows[c][i].get("error")).split(":")[0] for i in no_response}),
                        "median_completion_tokens": int(statistics.median(toks))}
for c in ("A", "mse"):
    q = sum(ok(c, i) - ok("W4A16", i) for i in ids)
    gpqa[f"{c}_minus_W4A16"] = {"questions": q, "diff": round(q / n, 4), "bootstrap95": boot(c, "W4A16"),
                                "verdict": "PASS" if q / n >= -band else "FAIL"}
    answered = [i for i in ids if rows[c][i].get("ok") is not False]
    if len(answered) < n:
        gpqa[f"{c}_minus_W4A16"]["on_items_answered"] = {
            "items": len(answered), "questions": sum(ok(c, i) - ok("W4A16", i) for i in answered),
            "W4A16_correct_on_unanswered": sum(ok("W4A16", i) for i in ids if i not in set(answered))}
hit = {c: {i for i in ids if rows[c][i].get("hit_max_tokens")} for c in CELLS}
common = [i for i in ids if not any(i in hit[c] for c in CELLS)]
gpqa["common_finished_set"] = {"items": len(common), "correct": {c: sum(ok(c, i) for i in common) for c in CELLS}}
for c in ("BF16", "W4A16", "A", "mse"):
    q = sum(ok(c, i) - ok("BF16", i) for i in ids)
    gpqa.setdefault("vs_bf16", {})[c] = {"questions": q, "diff": round(q / n, 4), "bootstrap95": boot(c, "BF16") if c != "BF16" else [0.0, 0.0]}


# ---------- KL ----------
def kl(ref, cand):
    s, m, agree = 0.0, 0, 0
    for rr, cr in zip(load(ref)["rows"], load(cand)["rows"]):
        for r, c in zip(rr, cr):
            if not r or not c:
                continue
            p = {t: math.exp(v) for t, v in r.items()}
            qd = {t: math.exp(v) for t, v in c.items()}
            pr = max(1e-12, 1 - sum(p.values()))
            qr = max(1e-12, 1 - sum(qd.get(t, 0.0) for t in p))
            fl = min(qd.values())
            s += sum(pv * (math.log(pv) - math.log(qd.get(t, fl))) for t, pv in p.items()) + pr * (math.log(pr) - math.log(qr))
            agree += max(p.items(), key=lambda kv: kv[1])[0] == max(qd.items(), key=lambda kv: kv[1])[0]
            m += 1
    return {"positions": m, "mean_kl": round(s / m, 4), "top1_agreement": round(agree / m, 4)}


klres = {"reported_set": {c: kl("kl-BF16.json", f"kl-{c}.json") for c in ("W4A16", "W4A16-b12x", "A", "mse")},
         "selection_set": {c: kl("kl-sel-BF16.json", f"kl-sel-{c}.json") for c in ("A", "mse", "b12xeq", "p9999", "p999")}}
sel = load("sel-result.json")
klres["selection"] = {"rule": sel["rule"], "winner": sel["winner"],
                      "mse_vs_A_reduction_pct": round(100 * (1 - klres["selection_set"]["mse"]["mean_kl"] / klres["selection_set"]["A"]["mean_kl"]), 1)}

# ---------- throughput ----------
TP = ("W4A16", "W4A16-b12x", "A", "mse")
tp = {c: load(f"throughput-{c}.json")["cells"] for c in TP}
keys = ["3k_c1", "3k_c8", "50k_c1", "50k_c8"]
ratio = {f"mse_vs_{b}": [round(tp["mse"][k]["prefill_tok_per_s_aggregate"] / tp[b][k]["prefill_tok_per_s_aggregate"], 2) for k in keys]
         for b in ("W4A16", "W4A16-b12x")}
throughput = {"cells": {c: {k: {x: tp[c][k][x] for x in ("prefill_tok_per_s_aggregate", "ttft_p50_ms", "prompt_tokens_mean")} for k in keys} for c in TP},
              "ratios": ratio,
              "range_vs_W4A16_marlin": f"{min(ratio['mse_vs_W4A16']):.2f}-{max(ratio['mse_vs_W4A16']):.2f}x",
              "range_vs_W4A16_b12x": f"{min(ratio['mse_vs_W4A16-b12x']):.2f}-{max(ratio['mse_vs_W4A16-b12x']):.2f}x",
              "mse_vs_A_pct": [round(100 * (tp["mse"][k]["prefill_tok_per_s_aggregate"] / tp["A"][k]["prefill_tok_per_s_aggregate"] - 1), 1) for k in keys]}
ds = load("throughput-darkstar.json")["cells"]
throughput["reference_dense_27b_as_served"] = {
    "note": "A dense 27B engine as configured for production decode (chunked prefill, ~6k-token batches, sharing the GPU "
            "with live traffic); not a like-for-like model comparison.",
    "cells": {k: {x: ds[k][x] for x in ("prefill_tok_per_s_aggregate", "ttft_p50_ms")} for k in keys}}

# ---------- scales ----------
G = load("variants-G.json")
st = load("act-stats.json")
ups = sorted((s for s in G["A"] if s.endswith(".up")), key=lambda s: int(s[1:].split(".")[0]))


def rng_of(d, sel_keys):
    v = [d[s] for s in sel_keys]
    return [round(min(v), 1), round(max(v), 1)]


scales = {"streams": len(G["A"]), "stats_samples": st["samples"], "stats_tokens": st["tokens"],
          "routed_up_global_amax_A": rng_of(G["A"], ups), "routed_up_global_amax_mse": rng_of(G["mse"], ups),
          "down_global_amax_A": rng_of(G["A"], [s for s in G["A"] if not s.endswith(".up")]),
          "down_global_amax_mse": rng_of(G["mse"], [s for s in G["A"] if not s.endswith(".up")]),
          "mse_up_streams_at_or_below_8192": sum(G["mse"][s] <= 8192 for s in ups), "up_streams": len(ups)}

# Per stream kind: each variant's global amax G against the largest activation the BF16 model produced, and the
# simulated NVFP4 relative squared error (SSE / sum of squares) at that G. G below the observed max means the FP8
# block scale saturates and the largest activations are clipped to G.
grid, sst = st["grid"], st["streams"]


def rel_err(s: str, g: float) -> float:
    j = min(range(len(grid)), key=lambda k: abs(math.log2(grid[k]) - math.log2(g)))
    return sst[s]["sse"][j] / sst[s]["ss"]


VARS = ("A", "mse", "b12xeq", "p9999", "p999")
by_kind: dict = {}
for s in G["A"]:
    kind = {"up": "up_input", "rdown": "routed_down_input", "sdown": "shared_down_input"}[s.split(".")[1]]
    g = {v: (2688.0 if v == "b12xeq" else G[v][s]) for v in VARS}
    d = by_kind.setdefault(kind, {"streams": 0, "ratio_A": [], "ratio_mse": [], "A_below_max": 0, "mse_below_max": 0,
                                  "err": {v: [] for v in VARS}, "worst_A": []})
    d["streams"] += 1
    d["ratio_A"].append(g["A"] / sst[s]["max"])
    d["ratio_mse"].append(g["mse"] / sst[s]["max"])
    d["A_below_max"] += g["A"] < sst[s]["max"]
    d["mse_below_max"] += g["mse"] < sst[s]["max"]
    for v in VARS:
        d["err"][v].append(rel_err(s, g[v]))
    d["worst_A"].append((round(rel_err(s, g["A"]), 3), s))
mechanism = {}
for kind, d in by_kind.items():
    mechanism[kind] = {
        "streams": d["streams"],
        "A_global_amax_over_observed_max": [round(min(d["ratio_A"]), 2), round(max(d["ratio_A"]), 2)],
        "mse_global_amax_over_observed_max": [round(min(d["ratio_mse"]), 2), round(max(d["ratio_mse"]), 2)],
        "A_below_observed_max": d["A_below_max"], "mse_below_observed_max": d["mse_below_max"],
        "median_relative_squared_error": {v: round(statistics.median(d["err"][v]), 4) for v in VARS},
        "worst_A_streams": sorted(d["worst_A"], reverse=True)[:3],
    }
scales["mechanism"] = mechanism

# ---------- artifact ----------
v = load("release-validation.json")
artifact = {k: v[k] for k in ("verdict", "tensors", "quantized_modules", "routed", "shared", "fp8_block_scales", "fp8_block_scales_nan",
                              "fp8_block_scales_zero", "input_scale_streams", "input_scale_values_checked", "protected_tensors_checked",
                              "protected_linear_modules", "protected_linear_not_excluded_count")}
artifact["release_file_count"] = len(v["release_files"])
# Per-file sha256 values live in the published repository's manifest.sha256. They are kept out of this summary: a file
# name containing "token" followed by a 64-hex value reads as an API key to secret scanners.
artifact["release_files"] = sorted(v["release_files"])

summary = {"issue": 17, "product": "Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A4-NVFP4", "shipped_variant": "mse",
           "pre_registration": "https://github.com/HangGlidersRule/model-forge/issues/17#issuecomment-6512178222",
           "amendment": "https://github.com/HangGlidersRule/model-forge/issues/17#issuecomment-6529325359",
           "gpqa": gpqa, "kl": klres, "throughput": throughput, "activation_scales": scales, "artifact": artifact}
(out / "w4a4-gate-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({"gpqa_mse": gpqa["cells"]["mse"]["correct"], "verdict": gpqa["mse_minus_W4A16"]["verdict"],
                  "kl_mse": klres["reported_set"]["mse"]["mean_kl"], "tput": throughput["range_vs_W4A16_marlin"],
                  "scales_up_mse": scales["routed_up_global_amax_mse"]}))
