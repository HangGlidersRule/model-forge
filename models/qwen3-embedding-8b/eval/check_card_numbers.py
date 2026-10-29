#!/usr/bin/env python3
"""Check that every number on the Qwen3-Embedding-8B NVFP4 card is present in the committed evidence (fails closed).
Each entry: (card substring, value recomputed from a results file). Run from the repository root.
Usage: check_card_numbers.py"""
import json
import sys
from pathlib import Path

R = Path("models/qwen3-embedding-8b/results")
card = Path("models/qwen3-embedding-8b/model-card/base-nvfp4.md").read_text()
g = json.loads((R / "embedding-gate-summary.json").read_text())
b = json.loads((R / "retrieval-benchmark.json").read_text())
sp = {k: json.loads((R / f"throughput-{k}.json").read_text())["cells"] for k in ("bf16", "nvfp4-B")}
st = json.loads((R / "store-compat-nvfp4-B.json").read_text())["stores"]
mx = json.loads((R / "mixed-corpus-nvfp4-B.json").read_text())
r4, r7 = b["dims"]["4096"], b["dims"]["768"]
cos = b["doc_vector_cosine_vs_bf16"]


def f4(x):
    return f"{x:.4f}"


def pct(x):
    return f"{x:.2f}%"


def pt(x):
    return f"{100 * x:+.2f}".replace("-", "−")


expect = [
    f"{f4(r4['all-nvfp4-B']['mean_ndcg@10'])} (BF16 {f4(r4['all-bf16']['mean_ndcg@10'])}), {pct(r4['all-nvfp4-B']['ndcg_vs_all_bf16_pct'])}",
    f"{f4(r4['bf16-corpus x nvfp4-B-query']['mean_ndcg@10'])}, {pct(r4['bf16-corpus x nvfp4-B-query']['ndcg_vs_all_bf16_pct'])}",
    "delta {:+.4f} (95% CI {:+.4f} to {:+.4f})".format(*r4["bf16-corpus x nvfp4-B-query"]["delta_ndcg@10_vs_all_bf16_ci95"]).replace("-", "−"),
    f"{f4(r4['bf16-corpus x bf16-replica-query']['mean_ndcg@10'])}, {pct(r4['bf16-corpus x bf16-replica-query']['ndcg_vs_all_bf16_pct'])}",
    f"all NVFP4 {pct(r7['all-nvfp4-B']['ndcg_vs_all_bf16_pct'])}, BF16 corpus × NVFP4 queries {pct(r7['bf16-corpus x nvfp4-B-query']['ndcg_vs_all_bf16_pct'])}",
    f"mean {cos['nvfp4-B']['mean']:.4f}, p1 {cos['nvfp4-B']['p1']:.4f}, p5 {cos['nvfp4-B']['p5']:.4f}",
    f"BF16 replica {cos['bf16-replica']['mean']:.5f}",
    f"{sum(b['documents'].values()):,} documents",
    f"{sum(b['queries'].values())} queries",
    f"{sp['nvfp4-B']['long_c1']['input_tok_per_s']:,} tok/s at c=1, {sp['nvfp4-B']['long_c32']['input_tok_per_s']:,} at c=32 "
    f"(BF16 {sp['bf16']['long_c1']['input_tok_per_s']:,} / {sp['bf16']['long_c32']['input_tok_per_s']:,}), {g['throughput']['long_range']['nvfp4-B']}",
    f"{sp['nvfp4-B']['short_c1']['input_tok_per_s']:,} tok/s at c=1, {sp['nvfp4-B']['short_c64']['input_tok_per_s']:,} at c=64 "
    f"(BF16 {sp['bf16']['short_c1']['input_tok_per_s']:,} / {sp['bf16']['short_c64']['input_tok_per_s']:,}), {g['throughput']['short_range']['nvfp4-B']}",
    f"p50 {sp['nvfp4-B']['short_c1']['p50_ms']} ms short, {sp['nvfp4-B']['long_c1']['p50_ms']} ms long (BF16 {sp['bf16']['short_c1']['p50_ms']} / {sp['bf16']['long_c1']['p50_ms']})",
    f"Weights load as {g['weights_gib_loaded']['nvfp4-B']:.2f} GiB in vLLM, against {g['weights_gib_loaded']['bf16']:.2f} GiB for BF16",
    f"| {pct(g['public_benchmark']['ship_rule_results']['nvfp4-A']['bf16_corpus_x_nvfp4_query'])} / {pct(g['public_benchmark']['ship_rule_results']['nvfp4-A']['all_nvfp4'])} | {g['throughput']['long_range']['nvfp4-A']} |",
    f"| {pct(g['public_benchmark']['ship_rule_results']['nvfp4-B']['bf16_corpus_x_nvfp4_query'])} / {pct(g['public_benchmark']['ship_rule_results']['nvfp4-B']['all_nvfp4'])} | {g['throughput']['long_range']['nvfp4-B']} |",
    "recall@10 {} pt (95% CI {} to {}), MRR@10 {} pt (CI {} to {})".format(
        *(f"{100 * v:+.1f}" if i in (0, 3) else f"{100 * v:+.2f}" for i, v in enumerate(
            st["transcripts"]["retrieval"]["delta_recall@10_ci"] + st["transcripts"]["retrieval"]["delta_mrr@10_ci"]))).replace("-", "−"),
    "recall@10 {:+.1f} pt (CI {:+.2f} to {:+.2f})".format(*(100 * v for v in st["memories"]["retrieval"]["delta_recall@10_ci"])).replace("-", "−"),
    "({} to {})".format(st["memories"]["item_to_item"]["threshold_flips"]["0.97"]["above_threshold_bf16"],
                        st["memories"]["item_to_item"]["threshold_flips"]["0.97"]["above_threshold_candidate"]),
    f"{100 * st['claims']['item_to_item']['threshold_flips']['0.6']['flip_rate']:.1f}% of near-neighbor pairs flipped",
    f"mean +{mx['self_gap_mean']:.3f} cosine",
    f"amax {g['activation_outliers_shipped']['max_amax']:,.0f} (layer 34) against a median of {g['activation_outliers_shipped']['median_amax']:,.0f}",
    f"({g['precision_coverage_shipped']['total_decoder_linear_params']})",
]
for s, v in g["per_set_ndcg@10_4096"].items():
    expect.append(f"{s.replace('Nano', '').replace('Retrieval', '')} {v['bf16']:.4f} → {v['nvfp4-B']:.4f}")
rp = g["reproducibility"]
dp, stp = rp["default_profile"], rp["stable_profile"]
fr = stp["full_set_between_containers"]
expect += [
    f"all NVFP4 {pct(dp['published_copy_second_instance_pct']['all_nvfp4'])}, BF16 corpus × NVFP4 queries "
    f"{pct(dp['published_copy_second_instance_pct']['bf16_corpus_x_nvfp4_query'])}; its queries against the first instance's "
    f"NVFP4 corpus {pct(dp['corpus_and_queries_from_different_instances_pct'])}",
    f"the cosine between\ninstances is {dp['doc_vector_cosine_between_instances']['mean']:.3f} on average "
    f"(p1 {dp['doc_vector_cosine_between_instances']['p1']:.3f})",
    f"{dp['doc_vector_cosine_between_instances']['documents']:,} public documents",
    f"{100 * stp['bitwise_identical_share']:.1f}% of {fr['docs']['vectors'] + fr['queries']['vectors']:,} public document and query vectors were bit-identical",
    f"at most {fr['max_abs_element_difference']:.0e} per element".replace("e-08", "e-8"),
    "98.49% of BF16 all NVFP4, and 99.39% with BF16 document vectors"
    if (stp["public_benchmark_pct"]["all_nvfp4"], stp["public_benchmark_pct"]["bf16_corpus_x_nvfp4_query"]) == (98.49, 99.39)
    else "STABLE-PROFILE-QUALITY-CHANGED",
    f"{stp['throughput_long_range']} BF16 tokens/s on long inputs and {stp['throughput_short_range']} on short inputs",
    "every cosine was within 2e-15 of 1" if fr["one_minus_cosine_max"] < 2e-15 else "COSINE-WITHIN-2E-15-NOT-SUPPORTED",
]
mc = g["store_compatibility"]["mixed_corpus"]["nvfp4-B"]
gain = mc["nvfp4_rows_top1_count"][1] - mc["nvfp4_rows_top1_count"][0]
loss = mc["bf16_rows_top1_count"][0] - mc["bf16_rows_top1_count"][1]
words = {1: "one", 2: "two", 3: "three"}
expect.append(f"they gained {gain} of {mc['nvfp4_rows_top1_count'][2]} top-1 placements, and the BF16 rows\n  lost "
              f"{words.get(loss, str(loss))} of {mc['bf16_rows_top1_count'][2]:,}")
missing = [e for e in expect if e not in card]
for e in missing:
    print("NOT ON CARD:", e)
print(f"{len(expect) - len(missing)}/{len(expect)} card numbers match the evidence")
sys.exit(1 if missing else 0)
