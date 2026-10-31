#!/usr/bin/env python3
"""Check every number on the Lightning W4A4 NVFP4 Mixed-FP8 card against results/nvl-gate-summary.json (fails closed).
Each expected string is built from the summary; the card must contain it verbatim. Run from the repository root.
Usage: check_card_numbers_nvl.py [CARD]"""
import json
import sys
from pathlib import Path

R = Path("models/nemotron-3.5-lightning-r1/results")
card = Path(sys.argv[1] if len(sys.argv) > 1 else "models/nemotron-3.5-lightning-r1/model-card/base-w4a4-nvfp4-mixed-fp8.md").read_text()
d = json.loads((R / "nvl-gate-summary.json").read_text())
g, k, a, s = d["gpqa"], d["kl"], d["artifact"], d["serving_rtx_pro_4500"]
gc, nv, delta = g["cells"], g["cells"]["NVL"], g["NVL_minus_W4A16"]
runs = s["runs"]
r77, r75, ref, v110 = runs["nvl_0.77"], runs["nvl_0.75"], runs["reference_w4a16_0.75"], runs["v1_1_0_0.80"]


def pc(x):
    return f"{100 * x:.1f}%"


def x2(v):
    return f"{v:.2f}x"


def ci(v):
    return f"{v:+.3f}".replace("-", "−")


expect = [
    f"prefill runs {s['gate']['prefill_range_0.77']} the\n  family's W4A16 build",
    f"single-stream decode runs {x2(r77['decode_vs_reference'])}",
    f"GPQA Diamond {nv['correct']}/198, equal to the W4A16 build",
    f"Next-token KL to BF16 is {k['NVL']['mean_kl']:.4f}.",
    f"That cell decodes at {x2(v110['decode_vs_reference'])} the W4A16 build on the RTX PRO 4500; this one at {x2(r77['decode_vs_reference'])}",
    f"prefer that cell (KL {k['mse']['mean_kl']:.4f})",
    f"in all 23 MoE layers: {a['experts']:,} linears",
    f"on the {a['mamba_fp8']} Mamba `in_proj`/`out_proj` linears",
    f"on the {a['kv_scales'] // 2} attention layers",
    f"the checkpoint is {a['weights_gib']} GiB in {len(a['shard_bytes'])} shards: three weight shards plus a {a['scale_shard_mb']} MB shard",
    f"The {a['protected_tensors_checked']} unquantized tensors are byte-identical to BF16",
    f"{a['fp8_nan_bytes']} NaN and {a['zero_block_scales']} zero FP8 block scales",
    f"| GPQA Diamond | {nv['correct']}/198 ({pc(nv['accuracy'])}) | full denominator; {nv['unparseable']} unparseable, "
    f"{nv['truncated_no_answer']} truncated without an answer, {nv['errors']} request errors |",
    f"| GPQA, upstream BF16 | {gc['BF16']['correct']}/198 and {gc['BF16-replica']['correct']}/198 (two runs) | the two runs disagree on "
    f"{g['noise']['bf16_discordant_items']} items, a noise band of {g['noise']['band_questions']} questions |",
    f"| GPQA, W4A16 build | {gc['W4A16']['correct']}/198 ({pc(gc['W4A16']['accuracy'])}) | same harness |",
    f"| GPQA delta vs the W4A16 build | {delta['questions']} questions (95% CI {ci(delta['bootstrap95'][0])} to {ci(delta['bootstrap95'][1])}) | "
    f"pre-registered pass bar −{abs(g['frozen_bar']):.3f}: {delta['verdict']} |",
    f"| Next-token KL vs BF16, FP8 KV as shipped | {k['NVL']['mean_kl']:.4f}, top-1 agreement {100 * k['NVL']['top1_agreement']:.2f}% | "
    f"64 held-out prompts, {k['NVL']['positions']:,} positions;",
    f"| Next-token KL vs BF16, BF16 KV | {k['NVL-bf16kv']['mean_kl']:.4f}, top-1 agreement {100 * k['NVL-bf16kv']['top1_agreement']:.2f}% | "
    f"same prompts; the FP8 KV cache adds {k['fp8_kv_effect']:.4f} |",
    f"| Next-token KL, other builds | W4A4 cell without Mixed-FP8 {k['mse']['mean_kl']:.4f}; W4A16 {k['W4A16']['mean_kl']:.4f} | same prompts |",
    f"| KV-cache pool, memory fraction {r77['util']} | {r77['kv_pool_tokens']:,} tokens | {r77['loaded_gib']} GiB loaded; the W4A16 build: "
    f"{ref['kv_pool_tokens']:,} at {ref['util']} ({ref['loaded_gib']} GiB loaded) |",
    f"| KV-cache pool, memory fraction {r75['util']} | {r75['kv_pool_tokens']:,} tokens |",
    f"| Prefill vs the W4A16 build, fraction {r77['util']} | " + ", ".join(x2(v) for v in r77["prefill_vs_reference"]) + " |",
    f"| Single-stream decode vs the W4A16 build | {x2(r77['decode_vs_reference'])} ({r77['decode_tok_per_s']} vs {ref['decode_tok_per_s']} tok/s) |",
    f"| W4A4 cell without Mixed-FP8, same card at {v110['util']:.2f} | decode {x2(v110['decode_vs_reference'])}, prefill "
    f"{min(v110['prefill_vs_reference']):.2f}-{max(v110['prefill_vs_reference']):.2f}x | same flags |",
    "| Tool calling, JSON-schema output | " + ", ".join("pass" if r77["checks"][c] else "fail" for c in ("tool_call", "json_schema")) + " |",
    f"(0.0358, against {k['mse']['mean_kl']:.4f} and {k['W4A16']['mean_kl']:.4f})".replace("0.0358", f"{k['NVL']['mean_kl']:.4f}"),
    f"--gpu-memory-utilization {r77['util']}",
]
missing = [e for e in expect if e not in card]
for e in missing:
    print("MISSING:", e)
print(f"{len(expect) - len(missing)}/{len(expect)} card numbers match the gate summary")
sys.exit(1 if missing else 0)
