#!/usr/bin/env python3
"""Build the issue-25 gate summary for the Nemotron-3.5-Lightning W4A4 NVFP4 Mixed-FP8 cell from the raw result files.

Every number the card, family README, ledger and release notes quote is computed here and written to
results/nvl-gate-summary.json; check_card_numbers_nvl.py recomputes the card's figures from that file. Per-item GPQA rows
and KL position arrays stay out of the public record: only aggregates are written.
Usage: build_nvl_summary.py RAW_DIR OUT_DIR
  RAW_DIR: gpqa-{BF16,BF16-replica,W4A16,mse,NVL}.json, kl-{BF16,W4A16,mse,NVL,NVL-bf16kv}.json,
           nvl-release-validation.json, shard-bytes.json (byte size of each shipped safetensors file),
           ab-{cur3,nvl,nvl4,new2}.json (RTX PRO 4500 serving runs),
           serve-boot-facts.txt (loaded size and KV pool per serving run, from the engine logs)"""
import json
import math
import random
import re
import sys
from pathlib import Path

raw, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)


def load(name):
    return json.loads((raw / name).read_text())


# ---------- GPQA ----------
CELLS = ("BF16", "BF16-replica", "W4A16", "mse", "NVL")
gp = {c: load(f"gpqa-{c}.json") for c in CELLS}
rows = {c: {r["run_index"]: r for r in gp[c]["runs"] if r.get("phase") != "warmup"} for c in CELLS}
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
gpqa = {"items": n, "cells": {}, "noise": {"bf16_discordant_items": disc, "band": round(band, 4), "band_questions": round(band * n, 1)},
        "frozen_bar": -0.060}
for c in CELLS:
    a = gp[c]["accuracy"]
    no_response = sorted(i for i in ids if rows[c][i].get("ok") is False)
    gpqa["cells"][c] = {"correct": a["correct"], "attempted": a["attempted"], "accuracy": round(a["correct"] / a["attempted"], 4),
                        "unparseable": a["unparseable"], "truncated_no_answer": a["truncated_no_answer"], "errors": a["errors"],
                        "no_response_items": len(no_response)}
for c in ("NVL",):
    q = sum(ok(c, i) - ok("W4A16", i) for i in ids)
    gpqa[f"{c}_minus_W4A16"] = {"questions": q, "diff": round(q / n, 4), "bootstrap95": boot(c, "W4A16"),
                                "verdict": "PASS" if q / n >= gpqa["frozen_bar"] else "FAIL"}
    q2 = sum(ok(c, i) - ok("mse", i) for i in ids)
    gpqa[f"{c}_minus_v1_1_0"] = {"questions": q2, "bootstrap95": boot(c, "mse")}


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


klres = {c: kl("kl-BF16.json", f"kl-{c}.json") for c in ("NVL", "NVL-bf16kv", "mse", "W4A16")}
klres["fp8_kv_effect"] = round(klres["NVL"]["mean_kl"] - klres["NVL-bf16kv"]["mean_kl"], 4)

# ---------- artifact ----------
val = load("nvl-release-validation.json")
artifact = {k: val[k] for k in ("verdict", "tensors", "experts", "routed", "shared", "mamba_fp8", "lm_head", "kv_scales",
                                "fp8_block_scales", "fp8_nan_bytes", "zero_block_scales", "protected_tensors_checked",
                                "expert_tensors_identical_to_v1_1_0", "expert_scale_shard_identical_to_v1_1_0", "release_file_count")}
artifact["release_files"] = sorted(val["release_files"])
shard_bytes = json.loads((raw / "shard-bytes.json").read_text())
artifact["shard_bytes"] = shard_bytes
artifact["weights_gib"] = round(sum(shard_bytes.values()) / 2**30, 1)
artifact["scale_shard_mb"] = round(shard_bytes["model-input-scales.safetensors"] / 1e6, 1)

# ---------- serving on the RTX PRO 4500 ----------
boot_facts = {}
for line in (raw / "serve-boot-facts.txt").read_text().splitlines():
    name, loaded, pool = line.split("|")
    gib = re.search(r"([0-9.]+) GiB", loaded)
    tokens = re.search(r"([0-9,]+) tokens", pool)
    if gib is None or tokens is None:
        raise SystemExit(f"serve-boot-facts.txt: unparseable line {line!r}")
    boot_facts[name.removeprefix("serve-").removesuffix(".log")] = {
        "loaded_gib": float(gib.group(1)), "kv_pool_tokens": int(tokens.group(1).replace(",", ""))}
runs = {"reference_w4a16_0.75": ("cur3", 0.75), "nvl_0.75": ("nvl", 0.75), "nvl_0.77": ("nvl4", 0.77), "v1_1_0_0.80": ("new2", 0.80)}
serving = {"gpu": "RTX PRO 4500 (32 GB)", "runs": {}}
keys = ["3k_c1", "3k_c8", "50k_c1", "50k_c8"]
ref = load("ab-cur3.json")
for label, (f, util) in runs.items():
    d = load(f"ab-{f}.json")
    serving["runs"][label] = {
        "util": util, **boot_facts[f],
        "prefill": {k: d["prefill"][k]["prefill_tok_per_s_aggregate"] for k in keys},
        "prefill_vs_reference": [round(d["prefill"][k]["prefill_tok_per_s_aggregate"] / ref["prefill"][k]["prefill_tok_per_s_aggregate"], 2) for k in keys],
        "decode_tok_per_s": d["decode"]["single_stream_tok_per_s_median"],
        "decode_vs_reference": round(d["decode"]["single_stream_tok_per_s_median"] / ref["decode"]["single_stream_tok_per_s_median"], 2),
        "checks": d["checks"]}
s77 = serving["runs"]["nvl_0.77"]
serving["gate"] = {"rule": "KV pool >= 179,260 tokens at util <= 0.80; tool-call and json_schema pass; prefill >= 1.20x in every cell; decode >= 0.90x",
                   "verdict": "PASS" if (s77["kv_pool_tokens"] >= 179260 and all(s77["checks"].values())
                                         and min(s77["prefill_vs_reference"]) >= 1.20 and s77["decode_vs_reference"] >= 0.90) else "FAIL",
                   "prefill_range_0.77": f"{min(s77['prefill_vs_reference']):.2f}-{max(s77['prefill_vs_reference']):.2f}x"}

summary = {"issue": "HangGlidersRule/model-forge#25",
           "product": "HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8",
           "pre_registration": "https://github.com/HangGlidersRule/model-forge/issues/25",
           "results_comment": "https://github.com/HangGlidersRule/model-forge/issues/25#issuecomment-6539510364",
           "gpqa": gpqa, "kl": klres, "artifact": artifact, "serving_rtx_pro_4500": serving}
(out / "nvl-gate-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({"gpqa_NVL": gpqa["cells"]["NVL"]["correct"], "delta": gpqa["NVL_minus_W4A16"], "kl": klres["NVL"],
                  "kv_effect": klres["fp8_kv_effect"], "serving": serving["gate"]}, indent=1))
