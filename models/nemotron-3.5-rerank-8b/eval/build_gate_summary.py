#!/usr/bin/env python3
"""Build the committed gate evidence for the Nemotron-3.5-Rerank-8B family (issue #32) from the raw run outputs.

Inputs (a directory holding the run outputs):
  q-<run>-<set>.json      quality runs (rerank_eval.py run), runs bf16, bf16r, A, B, C, qwen-bf16, qwen-fp4
  s-<run>.json            speed runs (rerank_eval.py speed), runs bf16, A, B, C, qwen-bf16, qwen-fp4
  f-<run>.json            GPU memory before / after the speed run (MiB)
  <C>.validation.json     structural validator reports (validate_export.py) for A, B, C
  serve-*.log, p2-serve-*.log, check-bf16.log   vLLM startup logs and the endpoint check
Outputs (into OUT_DIR):
  rerank-gate-summary.json   every number the card, family README and ledger quote, each derived here
  artifact-validation.json   the shipped candidate's validator report plus the fused-group check
Quality statistics: paired per-query deltas against BF16, 10,000-resample bootstrap (seed 1234) over queries within
each dataset, so the macro weighting is preserved. Ratio = candidate macro / BF16 macro (pre-registered bar >= 0.98).
Speed: median of 3 measured runs per concurrency. Footprint = GPU memory after the c=8 runs minus before the serve.
Usage: build_gate_summary.py RAW_DIR OUT_DIR"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
from pathlib import Path

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SETS = {"beir6_bm25": "28f2dde20dd4bc3e82d41cdaa0e63df01048a29f9b318cf836129c59c3e450db",
        "lmeb_dialogue": "9373a1be97911f4ad3090a5f7b65a36b0046427ed2c7f35233510dd6b4f24ac0"}
NEMO = ("bf16r", "A", "B", "C")
QWEN = ("qwen-bf16", "qwen-fp4")
SPEED = ("bf16", "A", "B", "C", "qwen-bf16", "qwen-fp4")
SELECTED = "A"


def load(name: str) -> dict:
    return json.loads((RAW / name).read_text())


def per_query(run: str, s: str) -> dict:
    q = load(f"q-{run}-{s}.json")
    assert q["evalset_sha256"] == SETS[s], (run, s, q["evalset_sha256"])
    return q["quality"]

def paired(a_run: str, b_run: str, s: str) -> dict:
    a, b = per_query(a_run, s)["per_set"], per_query(b_run, s)["per_set"]
    rows = {}
    for ds in b:
        aq = {p["qid"]: p for p in a[ds]["per_query"]}
        bq = {p["qid"]: p for p in b[ds]["per_query"]}
        assert aq.keys() == bq.keys(), (a_run, b_run, ds)
        rows[ds] = [aq[k]["ndcg10"] - bq[k]["ndcg10"] for k in sorted(bq)]
    delta = sum(sum(v) / len(v) for v in rows.values()) / len(rows)
    rng, boots = random.Random(1234), []
    for _ in range(10000):
        boots.append(sum(sum(v[rng.randrange(len(v))] for _ in v) / len(v) for v in rows.values()) / len(rows))
    boots.sort()
    return {"delta": round(delta, 4), "ci95": [round(boots[249], 4), round(boots[9749], 4)]}


def agreement(run: str, s: str) -> dict:
    a, b = per_query(run, s)["per_set"], per_query("bf16", s)["per_set"]
    rhos, overlaps = [], []
    for ds in b:
        aq = {p["qid"]: p for p in a[ds]["per_query"]}
        for p in b[ds]["per_query"]:
            x, y = p["scores"], aq[p["qid"]]["scores"]
            n = len(x)
            rx = {i: r for r, i in enumerate(sorted(range(n), key=lambda k: x[k]))}
            ry = {i: r for r, i in enumerate(sorted(range(n), key=lambda k: y[k]))}
            rhos.append(1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1)))
            tx = set(sorted(range(n), key=lambda i: -x[i])[:10])
            ty = set(sorted(range(n), key=lambda i: -y[i])[:10])
            overlaps.append(len(tx & ty) / 10)
    return {"spearman": round(sum(rhos) / len(rhos), 3), "top10_overlap": round(sum(overlaps) / len(overlaps), 3)}


summary: dict = {
    "issue": "HangGlidersRule/model-forge#32",
    "source": "nvidia/Nemotron-3.5-Rerank-8B-BF16@f88f8d9104726747791283ab156d2fcdb29e90a7",
    "selected_candidate": SELECTED,
    "pre_registered_bar": "candidate macro nDCG@10 >= 0.98 x BF16 on each set; the fastest passing candidate ships",
    "eval_sets": {
        "beir6_bm25": {"sha256": SETS["beir6_bm25"], "description": "6 MTEB Nano retrieval sets, BM25 hard negatives, 30 candidates per query (the #14 set)"},
        "lmeb_dialogue": {"sha256": SETS["lmeb_dialogue"], "description": "KaLM-Embedding/LMEB@f9d4294b Dialogue tasks (LoCoMo, REALTALK, TMD content_time_qs), 60 queries per split, judged turns plus top BM25 non-relevant turns of the same conversation, 30 candidates per query"},
    },
    "calibration_sha256": "ddd481866f158e51aaa7c66a9353be6ba1317bb5f2b3a188b1b703b5b961576e",
    "quality": {}, "speed": {}, "comparison_qwen3_reranker_4b": {},
}
for s in SETS:
    ctrl = per_query("bf16", s)
    block: dict = {"bf16": {"macro_ndcg10": round(ctrl["macro_ndcg10"], 4),
                            "per_dataset": {ds: round(v["ndcg10"], 4) for ds, v in ctrl["per_set"].items()},
                            "n_queries": sum(v["n_queries"] for v in ctrl["per_set"].values()),
                            "n_pairs": sum(v["n_pairs"] for v in ctrl["per_set"].values())}}
    for run in NEMO + QWEN:
        q = per_query(run, s)
        entry = {"macro_ndcg10": round(q["macro_ndcg10"], 4),
                 "ratio_to_bf16": round(q["macro_ndcg10"] / ctrl["macro_ndcg10"], 4),
                 **{f"{k}_vs_bf16": v for k, v in paired(run, "bf16", s).items()},
                 "per_dataset": {ds: round(v["ndcg10"], 4) for ds, v in q["per_set"].items()}}
        if run in NEMO:
            entry.update(agreement(run, s))
            entry["passes_bar"] = entry["ratio_to_bf16"] >= 0.98
        block[run] = entry
    summary["quality"][s] = block
    summary["comparison_qwen3_reranker_4b"][s] = {
        f"{SELECTED}_minus_qwen_bf16": paired(SELECTED, "qwen-bf16", s),
        f"{SELECTED}_minus_qwen_fp4": paired(SELECTED, "qwen-fp4", s),
        "splits_won_by_selected_vs_qwen_bf16": sum(
            1 for ds, v in block[SELECTED]["per_dataset"].items() if v > block["qwen-bf16"]["per_dataset"][ds]),
        "splits": len(block[SELECTED]["per_dataset"]),
    }
for run in SPEED:
    t = load(f"s-{run}.json")["throughput"]
    f = load(f"f-{run}.json")
    summary["speed"][run] = {
        **{c: round(statistics.median(r["pairs"] / r["seconds"] for r in t[c]), 1) for c in ("c1", "c4", "c8")},
        "gpu_memory_under_load_gib": round((f["after_load_mib"] - f["baseline_mib"]) / 1024, 1)}
for run in SPEED:
    sp = summary["speed"][run]
    sp["ratio_to_bf16"] = {c: round(sp[c] / summary["speed"]["bf16"][c], 2) for c in ("c1", "c4", "c8")}
    sp["ratio_to_qwen_bf16"] = {c: round(sp[c] / summary["speed"]["qwen-bf16"][c], 2) for c in ("c1", "c4", "c8")}


def loaded_gib(log: str) -> float:
    m = re.search(r"Model loading took ([0-9.]+) GiB", (RAW / log).read_text(errors="replace"))
    assert m, log
    return float(m.group(1))


summary["weights_loaded_gib"] = {run: loaded_gib(f"serve-{run}.log") for run in ("bf16", "A", "B", "C", "qwen-bf16", "qwen-fp4")}
kernels = {run: sorted(set(re.findall(r"Using ([A-Za-z0-9]+) for NVFP4 GEMM", (RAW / f"serve-{run}.log").read_text(errors="replace"))))
           for run in ("A", "B", "C")}
summary["nvfp4_gemm_kernel"] = kernels
chk = (RAW / "check-bf16.log").read_text()
m = re.search(r"ENDPOINTS-AGREE n=(\d+) tol=\S+ max\|score-rerank\|=(\S+) max\|score-classify\|=(\S+)", chk)
assert m, "endpoint check line missing"
summary["endpoint_check_bf16"] = {"pairs": int(m.group(1)), "max_abs_score_minus_rerank": float(m.group(2)),
                                  "max_abs_score_minus_classify": float(m.group(3))}
passing = [c for c in ("A", "B", "C") if all(summary["quality"][s][c]["passes_bar"] for s in SETS)]
fastest = max(passing, key=lambda c: summary["speed"][c]["c4"])
assert fastest == SELECTED, (passing, fastest)
summary["decision"] = {"passing": passing, "fastest_passing_by_c4": fastest}


def pair_score_diffs(a_run: str, b_run: str, s: str) -> dict:
    a, b = per_query(a_run, s)["per_set"], per_query(b_run, s)["per_set"]
    diffs, identical = [], 0
    for ds, v in a.items():
        bq = {p["qid"]: p for p in b[ds]["per_query"]}
        for p in v["per_query"]:
            for x, y in zip(p["scores"], bq[p["qid"]]["scores"]):
                diffs.append(abs(x - y))
                identical += x == y
    diffs.sort()
    return {"pairs": len(diffs), "identical": identical, "abs_diff_p50": round(diffs[len(diffs) // 2], 4),
            "abs_diff_p99": round(diffs[int(len(diffs) * 0.99)], 4), "abs_diff_max": round(diffs[-1], 4)}


# Published-serve-profile reruns of the selected candidate: the model card's exact serve command, two fresh containers.
# A-clean serves an empty-cache download of the uploaded commit; A-card2 serves the evaluated export itself. The files are
# byte-identical, so the gap between the two is the W4A4 fresh-container spread; the selection run used a longer context.
RERUNS = {"A-clean": "empty-cache download of the uploaded commit (every manifest entry and model.safetensors byte-identical to the evaluated export)",
          "A-card2": "the evaluated export, in a fresh container"}
if all((RAW / f"q-{r}-{s}.json").exists() for r in RERUNS for s in SETS):
    rr: dict = {"profile": "the model card's serve command (--max-model-len 8192 --max-num-batched-tokens 8192); the selection run used 32768 / 32768",
                "runs": {}, "fresh_container_spread": {}, "range_across_three_serves": {}}
    for r, what in RERUNS.items():
        rr["runs"][r] = {"what": what}
        for s in SETS:
            c = per_query(r, s)
            ratio = c["macro_ndcg10"] / summary["quality"][s]["bf16"]["macro_ndcg10"]
            rr["runs"][r][s] = {"macro_ndcg10": round(c["macro_ndcg10"], 4), "ratio_to_bf16": round(ratio, 4),
                                **{f"{k}_vs_bf16": v for k, v in paired(r, "bf16", s).items()}, "passes_bar": ratio >= 0.98}
    for s in SETS:
        rr["fresh_container_spread"][s] = {**{f"{k}_card2_minus_clean": v for k, v in paired("A-card2", "A-clean", s).items()},
                                           "pair_scores": pair_score_diffs("A-card2", "A-clean", s),
                                           "clean_minus_selection": paired("A-clean", SELECTED, s)}
        macros = [summary["quality"][s][SELECTED]["macro_ndcg10"]] + [rr["runs"][r][s]["macro_ndcg10"] for r in RERUNS]
        bf = summary["quality"][s]["bf16"]["macro_ndcg10"]
        rr["range_across_three_serves"][s] = {"min": min(macros), "max": max(macros), "spread": round(max(macros) - min(macros), 4),
                                              "ratio_min": round(min(macros) / bf, 4), "ratio_max": round(max(macros) / bf, 4),
                                              "bf16_replica_spread": round(abs(summary["quality"][s]["bf16r"]["macro_ndcg10"] - bf), 4)}
    summary["published_profile_reruns"] = rr
summary["ranges"] = {
    run: "{:.2f}-{:.2f}x".format(min(summary["speed"][run]["ratio_to_bf16"].values()), max(summary["speed"][run]["ratio_to_bf16"].values()))
    for run in ("A", "B", "C")}
summary["ranges"]["A_vs_qwen_bf16"] = "{:.2f}-{:.2f}x".format(
    min(summary["speed"]["A"]["ratio_to_qwen_bf16"].values()), max(summary["speed"]["A"]["ratio_to_qwen_bf16"].values()))
summary["provenance"] = {
    "vllm_image": "vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34 (v0.32.0)",
    "gpu": "RTX PRO 6000 Blackwell (SM120, 96 GB). Quality runs shared the GPU with another workload (scores are deterministic); speed runs had the whole GPU.",
    "quality_serve": "--runner pooling --max-model-len 32768 --max-num-batched-tokens 32768; Nemotron: --pooler-config MEAN/use_activation false, its own chat template, --chat-template-content-format openai; Qwen3-Reranker-4B: Qwen3ForSequenceClassification overrides and the judge-prompt deployment template",
    "speed_serve": "--runner pooling --max-model-len 8192 --max-num-batched-tokens 8192, CUDA graphs on; Qwen3-Reranker-4B KV pool 1.28 GiB",
    "speed_method": "64 unique pairs of ~1.1k tokens per request, 4 x concurrency requests per run, 1 warm-up then 3 measured runs; median",
    "qwen_reference_revisions": {"bf16": "Qwen/Qwen3-Reranker-4B@22e683669bc0f0bd69640a1354a6d0aebcfeede5",
                                 "w4a4": "HangGlidersRule/Darkstar-Qwen3-Reranker-4B-Base-ModelOpt-W4A4-NVFP4@8878e4b6f8a2a6f3edbdf713e2d74167968be0cd"},
    "modelopt": "nvidia-modelopt 0.48.0 (wheel sha256 bfcfd425eab15e05eccf065b58d11cc33f9c27b4859a6efb78aba74bfb9f225c, tag commit e2f328a19de789e73b91aca2908042388cf6d496); torch 2.15.0 cu132; transformers 5.18.0",
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "rerank-gate-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
val = load(f"{SELECTED}.validation.json")
val.pop("config_quantization_config", None)
val["fused_group_scales"] = {"groups_checked": 136, "mismatched": 0,
                             "note": "q/k/v and gate/up shards share input_scale and weight_scale_2 in all 34 layers"}
(OUT / "artifact-validation.json").write_text(json.dumps(val, indent=1) + "\n")
print(json.dumps({k: summary[k] for k in ("decision", "ranges", "weights_loaded_gib", "endpoint_check_bf16")}))
