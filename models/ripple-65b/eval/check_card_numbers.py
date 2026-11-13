#!/usr/bin/env python3
"""Check every number on the Ripple-65B Preview cards and family README against the committed results (fails closed).

Every number is built from results/gate-summary.json (its card_strings and figures, themselves computed by
eval/build_gate_summary.py from the window's result files), results/runtime-check.json and results/checkpoint-sizes.json;
none is typed here. The README's boot-smoke sentence is required, and accepted, only while results/staging-receipt.json
records the passed smoke it states (both modes, the measured image, no fatal log line). Each document must
contain each of its strings verbatim after whitespace runs (line wraps included)
are collapsed to one space. The staged Hugging Face card must equal the source card apart from its absolute links.
Run from the repository root. Usage: check_card_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

F = Path("models/ripple-65b")
R = F / "results"
PID = "Darkstar-Ripple-65B-Preview-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8"
g = json.loads((R / "gate-summary.json").read_text())
rt = json.loads((R / "runtime-check.json").read_text())
sz = json.loads((R / "checkpoint-sizes.json").read_text())
cs = g["card_strings"]
sp = g["speed"]["cells"]
gb = g["memory"]["checkpoint_gb"]


def norm(text: str) -> str:
    return " ".join(text.split())


def x(v: float) -> str:
    return f"{v:.2f}x"


kv = g["memory"]["kv_cache_tokens"]
card_expect = [cs[k] for k in cs if k not in ("power", "vllm", "size_vendor")] + [
    f"power limit {cs['power']}",
    f"vLLM nightly `{cs['vllm']}`",
    f"batch-1 decode runs {x(sp['decode_b1']['ratio_of_means'])}, batch-8 decode {x(sp['decode_b8']['ratio_of_means'])}, "
    f"and time to first token is {x(sp['ttft_1k_b1']['ratio_of_means'])} (lower is faster)",
    f"GPQA Diamond mean {g['gpqa']['means']['A']} of 198 against the vendor NVFP4's {g['gpqa']['means']['vendor']}",
    f"tool-call task success {g['proxy']['A']['success_rate']:.3f} against {g['proxy']['vendor']['success_rate']:.3f}",
    f"{gb['A']:.2f} GB against the vendor NVFP4's {gb['vendor']:.2f} GB",
    f"a {kv['A']:,}-token KV cache against {kv['vendor']:,}",
    f"KL to the BF16 teacher is {g['kl']['ratio']:.2f}x the vendor NVFP4's",
    f"The BF16 checkpoint ({gb['bf16']:.2f} GB) does not fit the 96 GB GPU",
    f"The vendor reports GPQA Diamond {g['vendor_reported_bf16_gpqa_diamond']['value']} for BF16 (vendor-reported",
    f"{g['kl']['A']['positions']:,} positions",
    f"The nightly above was built {rt['nightly_commits_after_merge']} commits after the merge",
    f"The `{rt['release_tag']}` git tag (`{rt['release_tag_commit'][:7]}`) contains it",
    f"The drafter adds about {round(sz['drafter']['tensor_bytes'] / 1e9)} GB",
    f"tag `nightly-{rt['nightly_commit']}`",
    f"`{g['image']}`",
]
runs = json.loads((R / "serve-runs.json").read_text())
a_runs = [v for k, v in runs["speed"].items() if k.startswith("A-")] + [runs["quality"]["A"]]
assert all(r["quantization"] == "modelopt_mixed" and r["kv_cache_dtype"] == "auto" for r in a_runs)
if all(r["enable_prefix_caching"] for r in a_runs):
    card_expect.append("Prefix caching is on (vLLM's default for this model)")
kernels = set().union(*(set(r["kernels"]) for r in a_runs))
assert {"Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM", "Using MarlinNvFp4LinearKernel for NVFP4 GEMM"} <= kernels
assert any(k.startswith("Selected FlashInferFP8ScaledMMLinearKernel") for k in kernels)
card_expect.append("vLLM loads the checkpoint as `modelopt_mixed`: FlashInfer FP8 GEMMs for attention, the FlashInfer CUTLASS "
                   "NVFP4 GEMM for the MLP, and Marlin for the weight-only `lm_head`")
n16 = {json.loads(f.read_text())["cells"]["ttft_16k_b8"]["n"] for f in (R / "speed").glob("*.json")}
assert len(n16) == 1
card_expect.append(f"median over {n16.pop()} requests (8 concurrent, 20 rounds)")
assert rt["release_tag_contains_merge"] and rt["nightly_contains_merge"] and rt["nightly_digest_matches_measured"]
if not rt["release_docker_image_exists"]:
    card_expect.append("but no Docker image has been published for that tag yet")
readme_expect = [cs[k] for k in ("gpqa_a", "gpqa_vendor", "proxy_parse", "proxy_success", "kl", "decode_b1", "decode_b8",
                                 "ttft_16k", "dflash_b1", "dflash_accept", "size", "kv_pool", "validation")] + [
    f"vLLM nightly `{cs['vllm']}`",
    f"The `{rt['release_tag']}` git tag (`{rt['release_tag_commit'][:7]}`) contains it",
    f"The vendor reports GPQA Diamond {g['vendor_reported_bf16_gpqa_diamond']['value']} for BF16 (vendor-reported",
]
# The README's boot-smoke sentence holds only while the receipt says the clean download booted and passed in both modes.
bs = json.loads((R / "staging-receipt.json").read_text())["boot_smoke"]
assert bs["status"] == "passed" and bs["image"] == g["image"] and sorted(bs["modes"]) == ["dflash2", "plain"]
assert all(m["fatal_log_markers"] == 0 and m["chat_request_finish_reason"] == "stop" and m["tool_call"]["name"]
           for m in bs["modes"].values())
readme_expect.append("booted with the card's serve command on the measured image, plain and with DFlash2: in both, the "
                     "known answers, a parsed tool call and a chat request passed with no fatal log line")
bf16_expect = [f"The checkpoint ({gb['bf16']:.2f} GB) does not fit", f"{g['kl']['A']['positions']:,} positions",
               f"the vendor reports {g['vendor_reported_bf16_gpqa_diamond']['value']} (vendor-reported"]
docs = {
    F / "model-card" / "base-nvfp4.md": card_expect,
    F / "model-card" / "base-bf16.md": bf16_expect,
    F / "README.md": readme_expect,
}
missing = []
total = 0
for path, expect in docs.items():
    text = norm(path.read_text())
    for e in expect:
        total += 1
        if norm(e) not in text:
            missing.append((path, e))
# The staged card is the source card with repo-relative links made absolute (it must not drift).
staged = Path("publication/huggingface") / PID / "README.md"
src = (F / "model-card" / "base-nvfp4.md").read_text()
total += 1
if staged.read_text() != src:
    missing.append((staged, "staged card differs from the source card"))
for path, e in missing:
    print(f"MISSING in {path}: {e}")
print(f"{total - len(missing)}/{total} card and README numbers match the committed results")
assert not re.search(r"\d+\.\d{4,}%", src)
sys.exit(1 if missing else 0)
