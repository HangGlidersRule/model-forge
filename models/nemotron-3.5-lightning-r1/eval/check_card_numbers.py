#!/usr/bin/env python3
"""Check every number on the Lightning W4A4 NVFP4 card against results/w4a4-gate-summary.json (fails closed).
Each expected string is built from the summary; the card must contain it verbatim. Run from the repository root.
Usage: check_card_numbers.py"""
import json
import sys
from pathlib import Path

R = Path("models/nemotron-3.5-lightning-r1/results")
card = Path("models/nemotron-3.5-lightning-r1/model-card/base-w4a4-nvfp4.md").read_text()
d = json.loads((R / "w4a4-gate-summary.json").read_text())
g, k, t, sc, a = d["gpqa"], d["kl"], d["throughput"], d["activation_scales"], d["artifact"]
gc, rk, sk = g["cells"], k["reported_set"], k["selection_set"]
tc = t["cells"]
mse, ms = gc["mse"], g["mse_minus_W4A16"]
mech = sc["mechanism"]["routed_down_input"]


def f(x):
    return f"{x:,}"


def ms_(x):
    return f"{x:,.1f}" if x < 1000 else f"{x:,.0f}"


def pc(x):
    return f"{100 * x:.1f}%"


expect = [
    f"prefill runs {t['range_vs_W4A16_marlin']} the family's W4A16 build",
    f"next-token KL to BF16 is {rk['mse']['mean_kl']:.4f}, against {rk['W4A16']['mean_kl']:.4f} for the W4A16 build",
    "all 23 MoE layers",
    f"{a['quantized_modules']:,} linears in all",
    f"placed the routed-expert\ndown-projection input range at {mech['A_global_amax_over_observed_max'][0]:.2f}-{mech['A_global_amax_over_observed_max'][1]:.2f}x",
    f"That build\nscored KL {rk['A']['mean_kl']:.4f}",
    f"each of the {sc['streams']} activation streams",
    f"over {sc['stats_samples']} calibration rows of BF16 activations ({sc['stats_tokens']:,} tokens)",
    f"cutting KL by {k['selection']['mse_vs_A_reduction_pct']}%",
    f"a fixed scale of 1.0 ({sk['b12xeq']['mean_kl']:.4f})",
    f"99.99th ({sk['p9999']['mean_kl']:.4f}) and 99.9th ({sk['p999']['mean_kl']:.4f}) percentiles",
    f"| GPQA Diamond | {mse['correct']}/198 ({pc(mse['accuracy'])}) | full denominator; {mse['unparseable']} unparseable, "
    f"{mse['truncated_no_answer']} truncated without an answer, {mse['no_response_items']} requests that never connected |",
    f"| GPQA, upstream BF16 | {gc['BF16']['correct']}/198 and {gc['BF16-replica']['correct']}/198 (two runs) | the two runs disagree on "
    f"{g['noise']['bf16_discordant_items']} items, a noise band of {g['noise']['band_questions']} questions |",
    f"| GPQA, W4A16 build | {gc['W4A16']['correct']}/198 ({pc(gc['W4A16']['accuracy'])}) | same harness |",
    f"| GPQA delta vs the W4A16 build | {ms['questions']} questions (95% CI {ms['bootstrap95'][0]:.3f} to +{ms['bootstrap95'][1]:.3f}); "
    f"{ms['on_items_answered']['questions']} on the {ms['on_items_answered']['items']} items this checkpoint answered | "
    f"pre-registered pass bar −{g['noise']['band']:.3f}: {ms['verdict']} |".replace("| -", "| −").replace("; -", "; −").replace("CI -", "CI −"),
    f"| Next-token KL vs BF16 | {rk['mse']['mean_kl']:.4f}, top-1 agreement {100 * rk['mse']['top1_agreement']:.2f}% | 64 held-out prompts, "
    f"{rk['mse']['positions']:,} positions;",
    f"| Next-token KL, other builds | W4A16 {rk['W4A16']['mean_kl']:.4f}; W4A16 with runtime W4A4 (FlashInfer b12x) "
    f"{rk['W4A16-b12x']['mean_kl']:.4f}; calibrated scales {rk['A']['mean_kl']:.4f} |",
    f"| Prefill, ~3k-token prompts | {f(tc['mse']['3k_c1']['prefill_tok_per_s_aggregate'])} tok/s at c=1 (TTFT {ms_(tc['mse']['3k_c1']['ttft_p50_ms'])} ms), "
    f"{f(tc['mse']['3k_c8']['prefill_tok_per_s_aggregate'])} at c=8 | W4A16: {f(tc['W4A16']['3k_c1']['prefill_tok_per_s_aggregate'])} / "
    f"{f(tc['W4A16']['3k_c8']['prefill_tok_per_s_aggregate'])}; W4A16 on b12x: {f(tc['W4A16-b12x']['3k_c1']['prefill_tok_per_s_aggregate'])} / "
    f"{f(tc['W4A16-b12x']['3k_c8']['prefill_tok_per_s_aggregate'])} |",
    f"| Prefill, ~50k-token prompts | {f(tc['mse']['50k_c1']['prefill_tok_per_s_aggregate'])} tok/s at c=1 (TTFT {ms_(tc['mse']['50k_c1']['ttft_p50_ms'])} ms), "
    f"{f(tc['mse']['50k_c8']['prefill_tok_per_s_aggregate'])} at c=8 | W4A16: {f(tc['W4A16']['50k_c1']['prefill_tok_per_s_aggregate'])} / "
    f"{f(tc['W4A16']['50k_c8']['prefill_tok_per_s_aggregate'])}; W4A16 on b12x: {f(tc['W4A16-b12x']['50k_c1']['prefill_tok_per_s_aggregate'])} / "
    f"{f(tc['W4A16-b12x']['50k_c8']['prefill_tok_per_s_aggregate'])} |",
    f"| Speed vs the W4A16 build | {t['range_vs_W4A16_marlin']} (on b12x: {t['range_vs_W4A16_b12x']}) |",
    f"Next-token KL to BF16 is {round(100 * (rk['mse']['mean_kl'] / rk['W4A16']['mean_kl'] - 1))}% higher than the W4A16 build's",
    f"GPQA is {-ms['questions']} questions lower",
]
weights = sum(1 for n in a["release_files"] if n.endswith(".safetensors"))
expect.append(f"in {weights} shards: three weight shards plus a 0.7 MB shard holding the activation scales")
missing = [e for e in expect if e not in card]
for e in missing:
    print("MISSING:", e)
print(f"{len(expect) - len(missing)}/{len(expect)} card numbers match the gate summary")
sys.exit(1 if missing else 0)
