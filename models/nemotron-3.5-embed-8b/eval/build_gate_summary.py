#!/usr/bin/env python3
"""Build the committed gate evidence for the Nemotron-3.5-Embed-8B family (issue #33) from the raw run outputs.

Inputs (RAW_DIR): score-<set>.json (embed_eval.py score), s-<run>.json and f-<run>.json (speed grid and GPU memory, from
the keep-alive rerun), <C>.validation.json (validate_export.py), <C>.ref.json (transformers reference vectors),
serve-*.log / p2-serve-*.log (vLLM startup), smoke.log (serving-path check), doc-cosine.json (doc_cosine.py),
clean-smoke.log and score-clean-<set>.json (clean-download smoke of the uploaded repository).
Outputs (OUT_DIR): embed-gate-summary.json (every number the card, family README and ledger quote) and
artifact-validation.json (the shipped candidate's validator report). Fails closed if the decision the record states is
not the one the numbers give. Usage: build_gate_summary.py RAW_DIR OUT_DIR"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SETS = {"mteb_nano_full": "a1275b5dba2153ea3625e969176ae0f480e7295dd85ea036ec02bc12e1d80990",
        "lmeb_scene_retrieval": "c2e6840f4f31aa0b3027cfe6635f1f9618ba194192d0e86ec8117d77ce544bbf"}
CANDS = ("A", "C", "D")
SELECTED = "D"
BAR = 0.98
CELLS = ("short_c1", "short_c8", "short_c32", "short_c64", "long_c1", "long_c8", "long_c32", "long_c64")


def load(name: str) -> dict:
    return json.loads((RAW / name).read_text())


score = {s: load(f"score-{s}.json") for s in SETS}
for s, d in score.items():
    assert d["set_sha256"] == SETS[s], (s, d["set_sha256"])

summary: dict = {
    "issue": "HangGlidersRule/model-forge#33",
    "source": "nvidia/Nemotron-3.5-Embed-8B-BF16@781e779ee4ca9c2d2a4f38321993556baeab49a1",
    "selected_candidate": SELECTED,
    "pre_registered_bar": "candidate mean nDCG@10 >= 0.98 x Nemotron BF16 on both sets, in both pairings (all-candidate; "
                          "BF16 documents x candidate queries); the fastest passing candidate ships",
    "amendment": "R33-A1 (posted on #33 after A and C failed, before D was scored): candidate D, W4A16 NVFP4 on every "
                 "text-tower linear; same bar",
    "eval_sets": {
        "mteb_nano_full": {"sha256": SETS["mteb_nano_full"], "description": "six pinned MTEB Nano retrieval sets with their full corpora (the #18 public retrieval benchmark)"},
        "lmeb_scene_retrieval": {"sha256": SETS["lmeb_scene_retrieval"], "description": "KaLM-Embedding/LMEB@f9d4294b Dialogue: the #32 query ids, each query searched against its whole conversation"},
    },
    "calibration_sha256": "a04f7ef04008eaa7c2a52e846109b9357481de87e65f447a8df1512f4ed44773",
    "quality": {}, "speed": {}, "comparison_qwen3_embedding_8b": {},
}
for s, d in score.items():
    runs, within, between = d["runs"], d["within_model"], d["between_models"]
    blk: dict = {"nemotron-bf16": {"mean_ndcg10": runs["nemotron-bf16"]["macro_ndcg10"],
                                   "mean_recall10": runs["nemotron-bf16"]["macro_recall10"],
                                   "per_set_ndcg10": runs["nemotron-bf16"]["per_set_ndcg10"]}}
    for lab in ("nemotron-bf16r",) + tuple(f"nemotron-{c}" for c in CANDS):
        w = within[lab]
        blk[lab] = {"mean_ndcg10": runs[lab]["macro_ndcg10"], "mean_recall10": runs[lab]["macro_recall10"],
                    "all": w["all"], "bf16_docs_x_candidate_queries": w["bf16_corpus_x_query"],
                    "per_set_ndcg10": runs[lab]["per_set_ndcg10"]}
        if lab != "nemotron-bf16r":
            blk[lab]["passes_bar"] = w["all"]["ratio_to_bf16"] >= BAR and w["bf16_corpus_x_query"]["ratio_to_bf16"] >= BAR
    blk["truncated_dims_mean_ndcg10"] = d["truncated_dims"]
    summary["quality"][s] = blk
    summary["comparison_qwen3_embedding_8b"][s] = {
        "qwen-bf16": runs["qwen-bf16"]["macro_ndcg10"], "qwen-w4a4": runs["qwen-fp4"]["macro_ndcg10"],
        "qwen_w4a4_vs_own_bf16": within["qwen-fp4"]["all"],
        "nemotron-bf16_minus_qwen-bf16": between["nemotron-bf16 minus qwen-bf16"],
        f"nemotron-{SELECTED}_minus_qwen-bf16": between[f"nemotron-{SELECTED} minus qwen-bf16"],
        f"nemotron-{SELECTED}_minus_qwen-w4a4": between[f"nemotron-{SELECTED} minus qwen-fp4"],
        "splits_won_by_nemotron_bf16": sum(1 for g, v in runs["nemotron-bf16"]["per_set_ndcg10"].items() if v > runs["qwen-bf16"]["per_set_ndcg10"][g]),
        "splits": len(runs["nemotron-bf16"]["per_set_ndcg10"]),
        "per_set_ndcg10": {"qwen-bf16": runs["qwen-bf16"]["per_set_ndcg10"], "qwen-w4a4": runs["qwen-fp4"]["per_set_ndcg10"]},
    }
for run in ("nemotron-bf16", "nemotron-D", "nemotron-A", "qwen-bf16", "qwen-fp4"):
    s = load(f"s-{run}.json")["cells"]
    f = load(f"f-{run}.json")
    assert max(c["max_ms"] for c in s.values()) < 20_000, f"stalled cell in {run}"
    summary["speed"][run] = {c: s[c]["input_tok_per_s"] for c in CELLS}
    summary["speed"][run]["gpu_memory_under_load_gib"] = round((f["after_load_mib"] - f["baseline_mib"]) / 1024, 1)
    summary["speed"][run]["slowest_request_s"] = round(max(c["max_ms"] for c in s.values()) / 1000, 1)
b = summary["speed"]["nemotron-bf16"]
summary["speed_ratio_to_bf16"] = {}
for run in ("nemotron-D", "nemotron-A"):
    r = {c: summary["speed"][run][c] / b[c] for c in CELLS}
    summary["speed_ratio_to_bf16"][run] = {
        "short": "{:.2f}-{:.2f}x".format(min(r[c] for c in CELLS[:4]), max(r[c] for c in CELLS[:4])),
        "long": "{:.2f}-{:.2f}x".format(min(r[c] for c in CELLS[4:]), max(r[c] for c in CELLS[4:]))}
summary["speed_method"] = ("the #18 grid: single-input requests of short (~100-token) and long (~4k-token) public text at "
                           "concurrency 1/8/32/64, one warm-up pass per cell, unique prefixes; one keep-alive connection "
                           "per client thread; whole GPU (RTX PRO 6000 Blackwell); GPU memory after the grid minus before the serve")
summary["speed_run_note"] = ("A first speed run opened a new TCP connection per request; about 1% of new connections through "
                             "Docker Desktop's published port stalled ~69 s and five cells were impossible. It was discarded and "
                             "every serve re-measured with keep-alive clients and a 20 s per-request ceiling.")


def loaded_gib(log: str) -> float:
    m = re.search(r"Model loading took ([0-9.]+) GiB", (RAW / log).read_text(errors="replace"))
    assert m, log
    return float(m.group(1))


summary["weights_loaded_gib"] = {"nemotron-bf16": loaded_gib("serve-nemotron-bf16.log"),
                                 **{f"nemotron-{c}": loaded_gib(f"serve-nemotron-{c}.log") for c in CANDS},
                                 "qwen-bf16": loaded_gib("serve-qwen-bf16.log"), "qwen-w4a4": loaded_gib("serve-qwen-fp4.log")}
summary["gemm_kernel"] = {f"nemotron-{c}": sorted(set(re.findall(r"Using ([A-Za-z0-9]+) for NVFP4 GEMM",
                                                                  (RAW / f"serve-nemotron-{c}.log").read_text(errors="replace"))))
                          for c in CANDS}
smoke = (RAW / "smoke.log").read_text()
m = re.search(r"BF16 serve vs transformers BF16: REFCHECK-OK vs bf16: (\d+) texts .*?cosine min ([0-9.]+) mean ([0-9.]+)", smoke)
assert m, "serving-path check line missing"
summary["serving_path_check"] = {"texts": int(m.group(1)), "cosine_min": float(m.group(2)), "cosine_mean": float(m.group(3)),
                                 "what": "vLLM BF16 serve vs transformers BF16 forward + mean pooling on the same rendered texts"}
ph = (RAW / "phase1.log").read_text()
rc = {}
for c in CANDS:
    m = re.search(rf"REFCHECK nemotron-{c}: REFCHECK-OK vs fakequant: .*?cosine min ([0-9.]+) mean ([0-9.]+)", ph)
    if m is None and (RAW / "phase1-AC.log").exists():
        m = re.search(rf"REFCHECK nemotron-{c}: REFCHECK-OK vs fakequant: .*?cosine min ([0-9.]+) mean ([0-9.]+)", (RAW / "phase1-AC.log").read_text())
    assert m, c
    rc[f"nemotron-{c}"] = {"cosine_min": float(m.group(1)), "cosine_mean": float(m.group(2))}
summary["served_vs_own_fakequant_reference"] = rc
summary["document_vector_cosine_vs_bf16"] = load("doc-cosine.json")
# Clean-download smoke of the uploaded private repo (the card's serve command on an empty-cache download).
cs = (RAW / "clean-smoke.log").read_text()
clean = {s: load(f"score-clean-{s}.json") for s in SETS}
for s, d in clean.items():
    assert d["set_sha256"] == SETS[s], s
    assert d["runs"][f"nemotron-{SELECTED}"] == score[s]["runs"][f"nemotron-{SELECTED}"], "clean scoring changed the evaluated run"
m = re.search(r"DOWNLOADED (\S+)@(\w+): manifest entries (\d+); hash mismatches \[\]; model\.safetensors identical to the evaluated export: True", cs)
assert m, "clean download line missing"
# The log holds one CARD EXAMPLE line per serve attempt on the same download; quote the last (the passing run).
ex_all = list(re.finditer(r"CARD EXAMPLE: dims (\d+), norm ([0-9.]+), cos\(query, Paris\) ([0-9.]+), cos\(query, bananas\) (-?[0-9.]+)", cs))
ex = ex_all[-1] if ex_all else None
rcl = re.search(r"REFCHECK clean vs D fake-quant reference: REFCHECK-OK .*?cosine min ([0-9.]+) mean ([0-9.]+)", cs)
dv = re.findall(r"([a-z-]+): (\d+) docs, cosine mean ([0-9.]+) min ([0-9.]+)", cs.split("DOC VECTORS clean vs evaluated:", 1)[1].split("\n", 1)[0])
assert ex and rcl and len(dv) == 2 and "CLEAN-SMOKE-OK" in cs, "clean smoke incomplete"
summary["clean_download_smoke"] = {
    "repository_commit": m.group(2), "manifest_entries": int(m.group(3)), "hash_mismatches": 0,
    "model_safetensors_identical_to_evaluated": True,
    "card_example": {"dims": int(ex.group(1)), "norm": float(ex.group(2)), "cos_query_paris": float(ex.group(3)),
                     "cos_query_bananas": float(ex.group(4))},
    "served_vs_own_fakequant_reference": {"cosine_min": float(rcl.group(1)), "cosine_mean": float(rcl.group(2))},
    "document_vectors_vs_evaluated_serve": {n: {"documents": int(k), "cosine_mean": float(a), "cosine_min": float(b)} for n, k, a, b in dv},
    "quality": {s: {"mean_ndcg10": clean[s]["runs"]["nemotron-Dclean"]["macro_ndcg10"],
                    "all": clean[s]["within_model"]["nemotron-Dclean"]["all"],
                    "bf16_docs_x_candidate_queries": clean[s]["within_model"]["nemotron-Dclean"]["bf16_corpus_x_query"],
                    "passes_bar": clean[s]["within_model"]["nemotron-Dclean"]["all"]["ratio_to_bf16"] >= BAR
                    and clean[s]["within_model"]["nemotron-Dclean"]["bf16_corpus_x_query"]["ratio_to_bf16"] >= BAR}
                for s in SETS},
}
assert all(v["passes_bar"] for v in summary["clean_download_smoke"]["quality"].values()), "clean download fails the bar"
passing = [c for c in CANDS if all(summary["quality"][s][f"nemotron-{c}"]["passes_bar"] for s in SETS)]
assert passing == [SELECTED], passing
summary["decision"] = {"passing": passing, "failed": [c for c in CANDS if c not in passing],
                       "failed_on": {c: [s for s in SETS if not summary["quality"][s][f"nemotron-{c}"]["passes_bar"]]
                                     for c in CANDS if c not in passing}}
summary["provenance"] = {
    "vllm_image": "vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34 (v0.32.0)",
    "gpu": "RTX PRO 6000 Blackwell (SM120, 96 GB). Quality runs shared the GPU with another workload (vectors are deterministic per serve); speed runs had the whole GPU.",
    "nemotron_serve": "--runner pooling --pooler-config MEAN/use_activation true --chat-template-content-format openai --max-model-len 32768 --max-num-batched-tokens 32768; inputs rendered client-side exactly as the checkpoint's chat template ('query: ' / 'passage: ', '[' and '<' escaped), add_special_tokens false",
    "qwen_serve": "the embedding lane profile of #18: Matryoshka override, --max-model-len 28672, chunked prefill, 8,192 batched tokens, prefix caching; queries 'Instruct: <task>\\nQuery:' (MTEB: web-search instruction; LMEB: its per-split instruction), documents bare",
    "qwen_reference_revisions": {"bf16": "Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
                                 "w4a4": "HangGlidersRule/Darkstar-Qwen3-Embedding-8B-Base-ModelOpt-W4A4-NVFP4@9c1435111bdd0ae1ccd6edd7f58a630d303a1b62"},
    "modelopt": "nvidia-modelopt 0.48.0 (wheel sha256 bfcfd425eab15e05eccf065b58d11cc33f9c27b4859a6efb78aba74bfb9f225c); torch 2.15.0 cu132; transformers 5.18.0",
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "embed-gate-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
val = load(f"{SELECTED}.validation.json")
val.pop("config_quantization_config", None)
assert isinstance(val.get("fused_group_scales"), dict) and "groups_checked" in val["fused_group_scales"], "validator report lacks fused groups"
assert val["verdict"] == "PASS", val["errors"]
(OUT / "artifact-validation.json").write_text(json.dumps(val, indent=1) + "\n")
print(json.dumps({k: summary[k] for k in ("decision", "speed_ratio_to_bf16", "weights_loaded_gib", "serving_path_check")}))
