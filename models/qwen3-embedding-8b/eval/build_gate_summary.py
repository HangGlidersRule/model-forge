#!/usr/bin/env python3
"""Build models/qwen3-embedding-8b/results/embedding-gate-summary.json from the committed issue-18 evidence files.
Every number is read from a measured results file or computed from one here; nothing is typed in. The card, family
README, ledger and release notes quote the strings this file writes.
Usage: build_gate_summary.py RESULTS_DIR OUT_JSON
"""
import json
import sys
from pathlib import Path

res, out = Path(sys.argv[1]), Path(sys.argv[2])


def load(name: str) -> dict:
    return json.loads((res / name).read_text())


bench = load("retrieval-benchmark.json")
speed = {k: load(f"throughput-{k}.json")["cells"] for k in ("bf16", "nvfp4-A", "nvfp4-B")}
stores = {k: load(f"store-compat-{k}.json")["stores"] for k in ("nvfp4-A", "nvfp4-B")}
mixed = {k: load(f"mixed-corpus-{k}.json") for k in ("nvfp4-A", "nvfp4-B")}
validation = load("artifact-validation.json")
build = load("build-summary.json")
outliers = load("activation-outliers.json")
agree = load("instance-agreement.json")
stable_repro = load("stable-profile-reproducibility.json")
stable_speed_file = load("throughput-nvfp4-B-stable.json")
speed["nvfp4-B-stable"] = stable_speed_file["cells"]

SHIP_BAR_PCT = 98.0  # amendment R18-A1: both pairings >= 98% of all-BF16 mean nDCG@10 at 4096 dims
DROPIN_RECALL_PT = -1.0  # pre-registration: per-store recall@10 / MRR@10 CI lower bound >= -1.0 pt
FLIP_MAX = 0.01  # pre-registration: <= 1% of pair decisions flip at the store's own threshold

# 1. public benchmark: the ship rule per candidate
ship = {}
for cand in ("nvfp4-A", "nvfp4-B"):
    rows = bench["dims"]["4096"]
    pairs = {"bf16_corpus_x_nvfp4_query": rows[f"bf16-corpus x {cand}-query"]["ndcg_vs_all_bf16_pct"],
             "all_nvfp4": rows[f"all-{cand}"]["ndcg_vs_all_bf16_pct"]}
    ship[cand] = {**pairs, "passes": all(v >= SHIP_BAR_PCT for v in pairs.values())}

# 2. speed: tokens/s ratios vs BF16 for every cell, and the ship-rule metric (long, c=32)
ratios = {}
for cand in ("nvfp4-A", "nvfp4-B", "nvfp4-B-stable"):
    ratios[cand] = {cell: round(speed[cand][cell]["input_tok_per_s"] / speed["bf16"][cell]["input_tok_per_s"], 2)
                    for cell in speed["bf16"]}
faster = max(("nvfp4-A", "nvfp4-B"), key=lambda c: speed[c]["long_c32"]["input_tok_per_s"])
passing = [c for c in ("nvfp4-A", "nvfp4-B") if ship[c]["passes"]]
shipped = max(passing, key=lambda c: speed[c]["long_c32"]["input_tok_per_s"]) if passing else None


def rng(vals: list[float]) -> str:
    return f"{min(vals):.2f}-{max(vals):.2f}x"


# 3. store compatibility verdicts (pre-registered rule), per candidate and store
verdicts = {}
for cand, st in stores.items():
    verdicts[cand] = {}
    for store, d in st.items():
        reasons = []
        if "retrieval" in d:
            r = d["retrieval"]
            if 100 * r["delta_recall@10_ci"][1] < DROPIN_RECALL_PT:
                reasons.append(f"recall@10 CI lower bound {100 * r['delta_recall@10_ci'][1]:.2f} pt")
            if 100 * r["delta_mrr@10_ci"][1] < DROPIN_RECALL_PT:
                reasons.append(f"MRR@10 CI lower bound {100 * r['delta_mrr@10_ci'][1]:.2f} pt")
        thr = {"claims": "0.6", "memories": "0.97"}.get(store)
        if thr:
            f = d["item_to_item"]["threshold_flips"][thr]
            if f["flip_rate"] > FLIP_MAX:
                reasons.append(f"{100 * f['flip_rate']:.2f}% of pairs flip at the app's {thr} threshold "
                               f"({f['above_threshold_bf16']} above with BF16, {f['above_threshold_candidate']} with NVFP4)")
        verdicts[cand][store] = {"drop_in": not reasons, "reasons": reasons,
                                 "drift_4096_mean": round(d["drift_4096"]["candidate_vs_stored"]["mean"], 4),
                                 "drift_4096_p1": round(d["drift_4096"]["candidate_vs_stored"]["p1"], 4),
                                 "bf16_reembed_vs_stored_mean": round(d["drift_4096"]["bf16_vs_stored"]["mean"], 5)}

summary = {
    "family": "qwen3-embedding-8b",
    "issue": "https://github.com/HangGlidersRule/model-forge/issues/18",
    "preregistration": {
        "compatibility": "https://github.com/HangGlidersRule/model-forge/issues/18#issuecomment-6501857585",
        "amendment_R18_A1": "https://github.com/HangGlidersRule/model-forge/issues/18#issuecomment-6504231012",
    },
    "candidates": {
        "nvfp4-A": "decoder MLP W4A4 NVFP4 (108 linears); attention BF16",
        "nvfp4-B": "every decoder linear W4A4 NVFP4 (252 linears: MLP + self-attention)",
    },
    "public_benchmark": {
        "sets": list(bench["queries"]), "queries": sum(bench["queries"].values()),
        "documents": sum(bench["documents"].values()),
        "all_bf16_mean_ndcg@10": {d: bench["dims"][d]["all-bf16"]["mean_ndcg@10"] for d in bench["dims"]},
        "bf16_replica_pct": {d: bench["dims"][d]["bf16-corpus x bf16-replica-query"]["ndcg_vs_all_bf16_pct"]
                             for d in bench["dims"]},
        "ship_rule": f"both pairings >= {SHIP_BAR_PCT}% of all-BF16 mean nDCG@10 at 4096 dims; faster passing "
                     "candidate (long inputs, c=32) ships",
        "ship_rule_results": ship,
        "doc_vector_cosine_vs_bf16": bench["doc_vector_cosine_vs_bf16"],
    },
    "throughput": {
        "setup": "one RTX PRO 6000 Blackwell (SM120), vLLM v0.32.0, identical serve flags, served beside a co-resident "
                 "production engine that had 0 running requests when each run started (the stable-profile run also "
                 "sampled it every 2 s: 0 throughout); single-input requests, unique prefixes",
        "stable_profile_run_production_engine_load": stable_speed_file.get("production_engine_running_requests_during_run"),
        "tok_per_s_ratio_vs_bf16": ratios,
        "short_range": {c: rng([v for k, v in ratios[c].items() if k.startswith("short")]) for c in ratios},
        "long_range": {c: rng([v for k, v in ratios[c].items() if k.startswith("long")]) for c in ratios},
        "faster_long_c32": faster,
    },
    "shipped": shipped,
    "weights_gib_loaded": {k: v["weights_gib"] for k, v in build["serve"].items()},
    "per_set_ndcg@10_4096": {s: {"bf16": bench["dims"]["4096"]["all-bf16"]["per_set"][s]["ndcg@10"],
                                 "nvfp4-B": bench["dims"]["4096"]["all-nvfp4-B"]["per_set"][s]["ndcg@10"],
                                 "bf16_corpus_x_nvfp4-B_query":
                                     bench["dims"]["4096"]["bf16-corpus x nvfp4-B-query"]["per_set"][s]["ndcg@10"]}
                             for s in bench["dims"]["4096"]["all-bf16"]["per_set"]},
    "activation_outliers_shipped": outliers,
    "precision_coverage_shipped": build["precision_coverage_shipped"],
    "reproducibility": {
        "default_profile": {
            "doc_vector_cosine_between_instances": bench["doc_vector_cosine_between_instances"]["nvfp4-B vs nvfp4-B-clean"],
            "published_copy_second_instance_pct": {
                "all_nvfp4": bench["dims"]["4096"]["all-nvfp4-B-clean"]["ndcg_vs_all_bf16_pct"],
                "bf16_corpus_x_nvfp4_query": bench["dims"]["4096"]["bf16-corpus x nvfp4-B-clean-query"]["ndcg_vs_all_bf16_pct"]},
            "corpus_and_queries_from_different_instances_pct":
                bench["dims"]["4096"]["nvfp4-B-corpus x nvfp4-B-clean-query"]["ndcg_vs_all_bf16_pct"],
            "same_container_restarted_64_documents": agree["pairs"]["orig1 vs orig2"],
        },
        "same_64_documents_by_profile": agree,
        "stable_profile": {
            "definition": "VLLM_BATCH_INVARIANT=1 (vLLM selects CutlassNvFp4LinearKernel) and the vLLM, Triton, Inductor and "
                          "FlashInfer cache directories on one persisted volume shared by every container on the same GPU "
                          "model and image; prefix caching on (the vLLM default for this pooling model, confirmed from the "
                          "engine config log)",
            "full_set_between_containers": stable_repro,
            "bitwise_identical_share": round(
                (stable_repro["docs"]["bitwise_identical"] + stable_repro["queries"]["bitwise_identical"])
                / (stable_repro["docs"]["vectors"] + stable_repro["queries"]["vectors"]), 4),
            "public_benchmark_pct": {
                "all_nvfp4": bench["dims"]["4096"]["all-nvfp4-B-stable"]["ndcg_vs_all_bf16_pct"],
                "bf16_corpus_x_nvfp4_query": bench["dims"]["4096"]["bf16-corpus x nvfp4-B-stable-query"]["ndcg_vs_all_bf16_pct"],
                "all_nvfp4_delta_ci95": bench["dims"]["4096"]["all-nvfp4-B-stable"]["delta_ndcg@10_vs_all_bf16_ci95"]},
            "throughput_long_range": rng([v for k, v in ratios["nvfp4-B-stable"].items() if k.startswith("long")]),
            "throughput_short_range": rng([v for k, v in ratios["nvfp4-B-stable"].items() if k.startswith("short")]),
        },
    },
    "store_compatibility": {
        "rule": "drop-in when NVFP4 queries lose <= 1.0 pt recall@10 and MRR@10 (paired-bootstrap 95% CI lower bound) "
                "against the stored BF16 vectors, and <= 1% of pair decisions flip at the store's own threshold",
        "store_labels": {"transcripts": "long-transcript chunk store (retrieval only)",
                         "memories": "personal-memory store (0.97 cosine dedup trigger)",
                         "claims": "news-claim store (0.60 cosine candidate floor)"},
        "verdicts": verdicts,
        "mixed_corpus": {c: {"self_gap_mean": round(m["self_gap_mean"], 4),
                             "nvfp4_rows_recall@1": [round(m["stored"]["N"]["recall@1"], 4), round(m["mixed"]["N"]["recall@1"], 4)],
                             "bf16_rows_recall@1": [round(m["stored"]["B"]["recall@1"], 4), round(m["mixed"]["B"]["recall@1"], 4)]}
                         for c, m in mixed.items()},
        "mixed_corpus_note": "post-hoc (not pre-registered): half of the sampled corpus rows re-embedded with NVFP4; "
                             "[stored corpus, mixed corpus] recall@1 for NVFP4 questions per row group",
    },
    "artifact_validation": validation["result"],
}
out.write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({"ship": ship, "shipped": shipped, "long": summary["throughput"]["long_range"],
                  "short": summary["throughput"]["short_range"]}, indent=1))
for c, v in verdicts.items():
    print(c, {s: (x["drop_in"], x["reasons"]) for s, x in v.items()})
