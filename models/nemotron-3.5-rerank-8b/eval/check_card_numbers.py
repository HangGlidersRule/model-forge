#!/usr/bin/env python3
"""Check that every number on the Nemotron-3.5-Rerank-8B NVFP4 card and the family README is present in the committed
evidence (fails closed). Each expected string is recomputed from results/rerank-gate-summary.json and must appear
verbatim. Run from the repository root. Usage: check_card_numbers.py"""
import json
import sys
from pathlib import Path

F = Path("models/nemotron-3.5-rerank-8b")
g = json.loads((F / "results/rerank-gate-summary.json").read_text())
card = (F / "model-card/base-nvfp4.md").read_text()
bf16_card = (F / "model-card/base-bf16.md").read_text()
readme = (F / "README.md").read_text()
qn, ql, sp, w = g["quality"]["beir6_bm25"], g["quality"]["lmeb_dialogue"], g["speed"], g["weights_loaded_gib"]
rr = g["published_profile_reruns"]
assert all(rr["runs"][r][s]["passes_bar"] for r in rr["runs"] for s in ("beir6_bm25", "lmeb_dialogue")), "a published-profile serve fails the bar"
cmp_n, cmp_l = g["comparison_qwen3_reranker_4b"]["beir6_bm25"], g["comparison_qwen3_reranker_4b"]["lmeb_dialogue"]


def m(x: float) -> str:
    return f"{x:+.4f}".replace("-", "−")


def ci(e: dict, key: str = "vs_bf16") -> str:
    return f"{m(e[f'delta_{key}'])} (95% CI {m(e[f'ci95_{key}'][0])} to {m(e[f'ci95_{key}'][1])})"


def ci_short(e: dict) -> str:
    return f"{m(e['delta_vs_bf16'])} ({m(e['ci95_vs_bf16'][0])}, {m(e['ci95_vs_bf16'][1])})"


def pct(r: float) -> str:
    return f"{100 * r:.2f}%"


def trio(run: str) -> str:
    return f"{sp[run]['c1']} / {sp[run]['c4']} / {sp[run]['c8']}"


paired_n = {k: v for k, v in cmp_n.items() if k.endswith("qwen_bf16")}["A_minus_qwen_bf16"]
paired_l = {k: v for k, v in cmp_l.items() if k.endswith("qwen_bf16")}["A_minus_qwen_bf16"]
expect_card = [
    f"{rr['runs']['A-clean']['beir6_bm25']['macro_ndcg10']:.4f} and {rr['runs']['A-card2']['beir6_bm25']['macro_ndcg10']:.4f} "
    f"({pct(rr['runs']['A-clean']['beir6_bm25']['ratio_to_bf16'])} and {pct(rr['runs']['A-card2']['beir6_bm25']['ratio_to_bf16'])} of BF16)",
    f"{rr['runs']['A-clean']['lmeb_dialogue']['macro_ndcg10']:.4f} and {rr['runs']['A-card2']['lmeb_dialogue']['macro_ndcg10']:.4f} "
    f"({pct(rr['runs']['A-clean']['lmeb_dialogue']['ratio_to_bf16'])} and {pct(rr['runs']['A-card2']['lmeb_dialogue']['ratio_to_bf16'])})",
    f"spans {rr['range_across_three_serves']['beir6_bm25']['min']:.4f}-{rr['range_across_three_serves']['beir6_bm25']['max']:.4f} on the MTEB Nano sets",
    f"and {rr['range_across_three_serves']['lmeb_dialogue']['min']:.4f}-{rr['range_across_three_serves']['lmeb_dialogue']['max']:.4f} on LMEB Dialogue",
    f"BF16 replica spread of {rr['range_across_three_serves']['beir6_bm25']['bf16_replica_spread']:.4f} and {rr['range_across_three_serves']['lmeb_dialogue']['bf16_replica_spread']:.4f}",
    f"Weights load as {w['A']:.2f} GiB in vLLM, against {w['bf16']:.2f} GiB for BF16",
    f"{qn['A']['macro_ndcg10']:.4f} (BF16 {qn['bf16']['macro_ndcg10']:.4f}), {pct(qn['A']['ratio_to_bf16'])}",
    f"{ql['A']['macro_ndcg10']:.4f} (BF16 {ql['bf16']['macro_ndcg10']:.4f}), {pct(ql['A']['ratio_to_bf16'])}",
    ci(qn["A"]), ci(ql["A"]), ci(qn["bf16r"]), ci(ql["bf16r"]),
    f"Spearman {qn['A']['spearman']:.3f} and {ql['A']['spearman']:.3f}, top-10 overlap {qn['A']['top10_overlap']:.3f} and {ql['A']['top10_overlap']:.3f}",
    f"BF16 replica: Spearman {qn['bf16r']['spearman']:.3f} and {ql['bf16r']['spearman']:.3f}",
    f"{trio('A')} pairs/s at c=1 / 4 / 8 (BF16 {trio('bf16')}), {g['ranges']['A']}",
    f"{sp['A']['gpu_memory_under_load_gib']} GiB (BF16 {sp['bf16']['gpu_memory_under_load_gib']} GiB)",
    f"{qn['bf16']['n_queries']} queries, {qn['bf16']['n_pairs']:,} pairs",
    f"{ql['bf16']['n_queries']} queries, {ql['bf16']['n_pairs']:,} pairs",
    f"the BF16 build scored {qn['qwen-bf16']['macro_ndcg10']:.4f} and {ql['qwen-bf16']['macro_ndcg10']:.4f}",
    f"the Darkstar W4A4 build scored {qn['qwen-fp4']['macro_ndcg10']:.4f} and {ql['qwen-fp4']['macro_ndcg10']:.4f}",
    f"{m(paired_n['delta'])} (95% CI {m(paired_n['ci95'][0])} to {m(paired_n['ci95'][1])}) on the MTEB Nano",
    f"{m(paired_l['delta'])} (95% CI {m(paired_l['ci95'][0])} to {m(paired_l['ci95'][1])}) on LMEB Dialogue",
    f"It runs at {g['ranges']['A_vs_qwen_bf16']} that build's pairs/s",
    f"| {pct(qn['A']['ratio_to_bf16'])} of BF16 | {pct(ql['A']['ratio_to_bf16'])} of BF16 | {sp['A']['c4']} ({sp['A']['ratio_to_bf16']['c4']:.2f}x BF16) |",
    f"| {pct(qn['B']['ratio_to_bf16'])} of BF16 | {pct(ql['B']['ratio_to_bf16'])} of BF16 | {sp['B']['c4']} |",
    f"| {pct(qn['C']['ratio_to_bf16'])} of BF16 | {pct(ql['C']['ratio_to_bf16'])} of BF16 | {sp['C']['c4']} |",
    f"{g['endpoint_check_bf16']['pairs']} pairs checked, max difference {g['endpoint_check_bf16']['max_abs_score_minus_rerank']:.0f}",
]
for ds, a in ql["A"]["per_dataset"].items():
    task, split = ds.split("/")
    expect_card.append(f"| {task} {split} | {ql['bf16']['per_dataset'][ds]} | {a} |")
expect_bf16 = [
    f"| {qn['bf16']['macro_ndcg10']:.4f} |", f"| {ql['bf16']['macro_ndcg10']:.4f} |",
    f"{ci(qn['bf16r'])} and {ci(ql['bf16r'])}", f"{trio('bf16')} pairs/s at c=1 / 4 / 8",
]
expect_readme = [
    f"| {qn['bf16']['macro_ndcg10']:.4f} | {ql['bf16']['macro_ndcg10']:.4f} | {trio('bf16')} | {w['bf16']:.2f} GiB |",
    f"| {ci_short(qn['bf16r'])} | {ci_short(ql['bf16r'])} |",
    f"| **{ci_short(qn['A'])}** | **{ci_short(ql['A'])}** | **{trio('A')}** | **{w['A']:.2f} GiB** |",
    f"| {ci_short(qn['B'])} | {ci_short(ql['B'])} | {trio('B')} | {w['B']:.2f} GiB |",
    f"| {ci_short(qn['C'])} | {ci_short(ql['C'])} | {trio('C')} | {w['C']:.2f} GiB |",
    f"fastest ({g['ranges']['A']} BF16)",
    f"| {qn['qwen-bf16']['macro_ndcg10']:.4f} | {ql['qwen-bf16']['macro_ndcg10']:.4f} | {trio('qwen-bf16')} |",
    f"| {qn['qwen-fp4']['macro_ndcg10']:.4f} | {ql['qwen-fp4']['macro_ndcg10']:.4f} | {trio('qwen-fp4')} |",
    f"| **{qn['A']['macro_ndcg10']:.4f}** | **{ql['A']['macro_ndcg10']:.4f}** | **{trio('A')}** |",
    f"{m(paired_n['delta'])} (95% CI {m(paired_n['ci95'][0])} to {m(paired_n['ci95'][1])}). A wins {cmp_n['splits_won_by_selected_vs_qwen_bf16']} of the {cmp_n['splits']} sets",
    f"{m(paired_l['delta'])} (95% CI {m(paired_l['ci95'][0])} to {m(paired_l['ci95'][1])}). A wins {cmp_l['splits_won_by_selected_vs_qwen_bf16']} of the {cmp_l['splits']} splits",
]
missing = [("card", s) for s in expect_card if s not in card] + [("bf16 card", s) for s in expect_bf16 if s not in bf16_card] \
    + [("README", s) for s in expect_readme if s not in readme]
for where, s in missing:
    print(f"MISSING in {where}: {s}")
print(f"CARD-NUMBERS {'OK' if not missing else 'FAIL'}: {len(expect_card) + len(expect_bf16) + len(expect_readme) - len(missing)}"
      f"/{len(expect_card) + len(expect_bf16) + len(expect_readme)} present")
sys.exit(1 if missing else 0)
