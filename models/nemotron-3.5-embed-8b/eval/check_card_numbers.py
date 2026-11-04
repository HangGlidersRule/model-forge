#!/usr/bin/env python3
"""Check that every number on the Nemotron-3.5-Embed-8B card, the BF16 control card and the family README is present in
the committed evidence (fails closed). Each expected string is recomputed from results/embed-gate-summary.json and must
appear verbatim. Run from the repository root. Usage: check_card_numbers.py"""
import json
import sys
from pathlib import Path

F = Path("models/nemotron-3.5-embed-8b")
g = json.loads((F / "results/embed-gate-summary.json").read_text())
card = (F / "model-card/base-nvfp4.md").read_text()
bf16_card = (F / "model-card/base-bf16.md").read_text()
readme = (F / "README.md").read_text()
qm, ql, sp, w = g["quality"]["mteb_nano_full"], g["quality"]["lmeb_scene_retrieval"], g["speed"], g["weights_loaded_gib"]
cm, cl = g["comparison_qwen3_embedding_8b"]["mteb_nano_full"], g["comparison_qwen3_embedding_8b"]["lmeb_scene_retrieval"]
dc = g["document_vector_cosine_vs_bf16"]
dm, dl = dc["mteb-nano-full-corpora"], dc["lmeb-dialogue-scene-retrieval"]
r = g["speed_ratio_to_bf16"]
cs = g["clean_download_smoke"]


def m(x: float) -> str:
    return f"{x:+.4f}".replace("-", "−")


def ci(e: dict) -> str:
    return f"{m(e['delta'])} (95% CI {m(e['ci95'][0])} to {m(e['ci95'][1])})"


def ci_short(e: dict) -> str:
    return f"{m(e['delta'])} ({m(e['ci95'][0])}, {m(e['ci95'][1])})"


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


D, A, C, B, BR = "nemotron-D", "nemotron-A", "nemotron-C", "nemotron-bf16", "nemotron-bf16r"
expect_card = [
    f"Weights load as {w[D]:.2f} GiB in vLLM, against\n{w[B]:.2f} GiB for BF16",
    f"speed is {r[D]['long'].split('-')[0]}-{r[D]['short'].split('-')[1]} BF16",
    f"| {pct(qm[A]['all']['ratio_to_bf16'])} of BF16 | {pct(ql[A]['all']['ratio_to_bf16'])} of BF16 | failed |",
    f"| {pct(qm[C]['all']['ratio_to_bf16'])} of BF16 | {pct(ql[C]['all']['ratio_to_bf16'])} of BF16 | failed |",
    f"| **{pct(qm[D]['all']['ratio_to_bf16'])} of BF16** | **{pct(ql[D]['all']['ratio_to_bf16'])} of BF16** | **passed** |",
    f"cosine {g['serving_path_check']['cosine_min']:.5f} minimum",
    f"{qm[D]['mean_ndcg10']:.4f} (BF16 {qm[B]['mean_ndcg10']:.4f}), {pct(qm[D]['all']['ratio_to_bf16'])}",
    f"{ql[D]['mean_ndcg10']:.4f} (BF16 {ql[B]['mean_ndcg10']:.4f}), {pct(ql[D]['all']['ratio_to_bf16'])}",
    ci(qm[D]["all"]), ci(ql[D]["all"]),
    f"{pct(qm[D]['bf16_docs_x_candidate_queries']['ratio_to_bf16'])} and {pct(ql[D]['bf16_docs_x_candidate_queries']['ratio_to_bf16'])} of BF16",
    f"{ci(qm[BR]['all'])} and {ci(ql[BR]['all'])}",
    f"mean {dm[D]['mean']:.5f} and {dl[D]['mean']:.5f}, minimum {dm[D]['min']:.5f} and {dl[D]['min']:.5f}",
    f"BF16 replica mean {dm[BR]['mean']:.5f} and {dl[BR]['mean']:.5f}",
    f"| {w[D]:.2f} GiB (BF16 {w[B]:.2f} GiB) |",
    f"{sp[D]['long_c1']:,} tok/s at c=1, {sp[D]['long_c32']:,} at c=32 (BF16 {sp[B]['long_c1']:,} / {sp[B]['long_c32']:,}), {r[D]['long']}",
    f"{sp[D]['short_c1']:,} tok/s at c=1, {sp[D]['short_c64']:,} at c=64 (BF16 {sp[B]['short_c1']:,} / {sp[B]['short_c64']:,}), {r[D]['short']}",
    f"| {sp[D]['gpu_memory_under_load_gib']} GiB (BF16 {sp[B]['gpu_memory_under_load_gib']} GiB) |",
    f"Qwen3-Embedding-8B BF16 scored {cm['qwen-bf16']:.4f} and {cl['qwen-bf16']:.4f}",
    f"the Darkstar W4A4 build of it scored {cm['qwen-w4a4']:.4f} and {cl['qwen-w4a4']:.4f}",
    f"{ci(cm['nemotron-D_minus_qwen-bf16'])} on the MTEB Nano sets",
    f"{ci(cl['nemotron-D_minus_qwen-bf16'])} on LMEB Dialogue",
    f"cosine {dm[D]['mean']:.5f} and {dl[D]['mean']:.5f} (mean)",
    "| Clean download of this repository, card serve command | {:.4f} and {:.4f} ({} and {} of BF16); document vectors vs the evaluated serve: cosine mean {:.5f} and {:.5f} |".format(
        cs["quality"]["mteb_nano_full"]["mean_ndcg10"], cs["quality"]["lmeb_scene_retrieval"]["mean_ndcg10"],
        pct(cs["quality"]["mteb_nano_full"]["all"]["ratio_to_bf16"]), pct(cs["quality"]["lmeb_scene_retrieval"]["all"]["ratio_to_bf16"]),
        cs["document_vectors_vs_evaluated_serve"]["mteb-nano-full-corpora"]["cosine_mean"],
        cs["document_vectors_vs_evaluated_serve"]["lmeb-dialogue-scene-retrieval"]["cosine_mean"]),
    f"it is {m(ql[D]['all']['delta'])} nDCG@10",
]
for dims in ("2000", "1024", "768"):
    tm, tl = qm["truncated_dims_mean_ndcg10"][dims], ql["truncated_dims_mean_ndcg10"][dims]
    expect_card.append(f"| {dims} | {tm[D]:.4f} | {tm[B]:.4f} | {tl[D]:.4f} | {tl[B]:.4f} |")
expect_bf16 = [
    f"| {qm[B]['mean_ndcg10']:.4f} |", f"| {ql[B]['mean_ndcg10']:.4f} |",
    f"{ci(qm[BR]['all'])} and {ci(ql[BR]['all'])}",
    f"cosine {g['serving_path_check']['cosine_min']:.5f} minimum on {g['serving_path_check']['texts']} fixed texts",
    f"{sp[B]['long_c1']:,} tok/s at c=1, {sp[B]['long_c32']:,} at c=32",
]
expect_readme = [
    f"| {qm[B]['mean_ndcg10']:.4f} | {ql[B]['mean_ndcg10']:.4f} | 1.00x | {w[B]:.2f} GiB |",
    f"| {ci_short(qm[BR]['all'])} | {ci_short(ql[BR]['all'])} |",
    f"| {ci_short(qm[A]['all'])} | {ci_short(ql[A]['all'])} | {r[A]['long']} | {w[A]:.2f} GiB |",
    f"| {ci_short(qm[C]['all'])} | {ci_short(ql[C]['all'])} | not measured | {w[C]:.2f} GiB |",
    f"| **{ci_short(qm[D]['all'])}** | **{ci_short(ql[D]['all'])}** | **{r[D]['long']}** | **{w[D]:.2f} GiB** |",
    f"{pct(ql[A]['all']['ratio_to_bf16'])} and {pct(ql[C]['all']['ratio_to_bf16'])} of BF16 there",
    f"against {pct(qm[A]['all']['ratio_to_bf16'])} and {pct(qm[C]['all']['ratio_to_bf16'])} on MTEB Nano",
    f"**D passed** at {pct(qm[D]['all']['ratio_to_bf16'])} and {pct(ql[D]['all']['ratio_to_bf16'])}",
    f"({w[D]:.2f} against {w[B]:.2f} GiB of weights)", f"W4A4 would have run {r[A]['long']} faster",
    f"| {cm['qwen-bf16']:.4f} | {cl['qwen-bf16']:.4f} |", f"| {cm['qwen-w4a4']:.4f} | {cl['qwen-w4a4']:.4f} |",
    f"| {qm[B]['mean_ndcg10']:.4f} | {ql[B]['mean_ndcg10']:.4f} |", f"| **{qm[D]['mean_ndcg10']:.4f}** | **{ql[D]['mean_ndcg10']:.4f}** |",
    f"**MTEB Nano:** {ci(cm['nemotron-bf16_minus_qwen-bf16'])}",
    f"**LMEB Dialogue:** {ci(cl['nemotron-bf16_minus_qwen-bf16'])}; Nemotron wins all {cl['splits']} splits",
]
assert cl["splits_won_by_nemotron_bf16"] == cl["splits"], "README claims Nemotron wins every LMEB split"
missing = [("card", s) for s in expect_card if s not in card] + [("bf16 card", s) for s in expect_bf16 if s not in bf16_card] \
    + [("README", s) for s in expect_readme if s not in readme]
for where, s in missing:
    print(f"MISSING in {where}: {s}")
n = len(expect_card) + len(expect_bf16) + len(expect_readme)
print(f"CARD-NUMBERS {'OK' if not missing else 'FAIL'}: {n - len(missing)}/{n} present")
sys.exit(1 if missing else 0)
