#!/usr/bin/env python3
"""Ripple-65B Preview (#39): build results/gate-summary.json from the committed result files, never by hand.

Reads (all under models/ripple-65b/results/): gate.json (the window's gate, eval/gate_ripple39.py), gpqa-*-r*.json
(curated runs), proxy-*.json, speed/*.json, serve-runs.json, build-A.json, validation-A.json, smoke-A*.json,
checkpoint-sizes.json and, when present, clean-download.json and hf-upload.json. Recomputes every derived figure the
card, README and ledger quote (ratios, means, ranges) and asserts it equals the gate's own value where the gate holds
one, so a typed number cannot drift from the data. Every quoted string lives in `card_strings`, which
eval/check_card_numbers.py requires verbatim in the card and README.
Run from the repository root. Usage: build_gate_summary.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

F = "ripple-65b"
R = Path(f"models/{F}/results")
SPEED_CELLS = ("decode_b1", "decode_b8", "ttft_1k_b1", "ttft_16k_b1", "ttft_1k_b8", "ttft_16k_b8")
VENDOR = "ricursiveAI/Ripple-65B-Preview-NVFP4@99eea4477a001a41b86914e41feb4b2ed86105e5"
SOURCE = "ricursiveAI/Ripple-65B-Preview@240d5966cbe00f0627c699e59ef58560b60303cf"
DRAFTER = "ricursiveAI/Ripple-65B-Preview-DFlash2@35609b3be3dd08a822b3ba1257c1ea5dd6f87632"
IMAGE = "vllm/vllm-openai@sha256:902b49de0568ba5a5c951b0243bc06ef8541f46bc3eb80832961fefb22a19034"
VENDOR_BF16_GPQA = 85.2  # vendor-reported (their harness, BF16), quoted from the upstream card; not measured here
# The upstream license at SOURCE's revision, as its card declares it: HfApi().model_info("ricursiveAI/Ripple-65B-Preview",
# revision="240d5966cbe00f0627c699e59ef58560b60303cf").card_data gives license "apache-2.0" with no license_name or
# license_link; the LICENSE file at that revision (sha256 58d1e17f..., pinned in eval/ripple_layout.py) is the Apache
# License 2.0 text, and the product ships it byte-identical (the export validator checks it).
SOURCE_LICENSE = {"license": "apache-2.0", "license_name": None, "license_link": None,
                  "license_file": "LICENSE", "license_file_sha256":
                      "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd",
                  "read_from": "the upstream card's front matter at the pinned revision (HfApi().model_info(repo, "
                               "revision=sha).card_data)"}


def load(name: str) -> dict:
    return json.loads((R / name).read_text())


def cell_value(serve: dict, cell: str) -> float:
    c = serve["cells"][cell]
    return c["aggregate_tok_s"]["median"] if cell == "decode_b8" else c["median"]


def main() -> int:
    gate = load("gate.json")
    ga = gate["candidates"]["A"]
    gpqa = {c: [load(f"gpqa-{c}-r{r}.json") for r in (1, 2)] for c in ("vendor", "A")}
    proxy = {c: load(f"proxy-{c}.json")["summary"] for c in ("vendor", "A")}
    runs = load("serve-runs.json")
    build, val = load("build-A.json"), load("validation-A.json")
    sizes = load("checkpoint-sizes.json")
    speed = {lab: json.loads((R / "speed" / f"{lab}.json").read_text())
             for lab in ("vendor-s1", "vendor-s2", "A-s1", "A-s2",
                         "vendor-s1-dflash2", "vendor-s2-dflash2", "A-s1-dflash2", "A-s2-dflash2")}

    # GPQA: the curated runs must reproduce the gate's run scores
    corr = {c: [g["accuracy"]["correct"] for g in gpqa[c]] for c in gpqa}
    assert corr["A"] == ga["gpqa"]["candidate_runs"] and corr["vendor"] == ga["gpqa"]["vendor_runs"]
    mean = {c: sum(v) / 2 for c, v in corr.items()}
    assert mean["A"] == ga["gpqa"]["candidate_mean"] and mean["vendor"] == ga["gpqa"]["vendor_mean"]
    trunc = {c: [g["accuracy"]["truncated_no_answer"] for g in gpqa[c]] for c in gpqa}
    # Responses that hit the 65,536-token cap: the bench's hit_max_tokens, which equals finish_reasons.length. Of those,
    # truncated_no_answer have no extractable answer (scored wrong); the rest still carry an answer and are scored.
    capped = {c: [g["accuracy"]["hit_max_tokens"] for g in gpqa[c]] for c in gpqa}
    for c in gpqa:
        for g, n_capped, n_none in zip(gpqa[c], capped[c], trunc[c]):
            assert g["finish_reasons"]["length"] == n_capped and 0 <= n_none <= n_capped, (c, g["run"])
    # gate.json keeps per-run counts for the vendor's runs; they must be the curated runs' own
    assert capped["vendor"] == [r["hit_max_tokens"] for r in gate["vendor"]["gpqa_runs"]]
    assert trunc["vendor"] == [r["truncated_no_answer"] for r in gate["vendor"]["gpqa_runs"]]
    reasoning = {c: [g["items_with_reasoning_block"] for g in gpqa[c]] for c in gpqa}

    # proxy
    gate_proxy = {"vendor": gate["vendor"]["proxy"], "A": ga["proxy"]["candidate"]}
    for c in ("vendor", "A"):
        for k in ("parse_rate", "success_rate"):
            assert proxy[c]["overall"][k] == gate_proxy[c][k], (c, k)
    bfcl = {c: proxy[c]["bfcl"]["success"] for c in proxy}
    syn = {c: proxy[c]["synthetic"]["success"] for c in proxy}

    # speed: per cell, both serves, ratio of means (A / vendor)
    table: dict[str, dict] = {}
    for spec in ("", "-dflash2"):
        for cell in SPEED_CELLS:
            a = [cell_value(speed[f"A-s{k}{spec}"], cell) for k in (1, 2)]
            v = [cell_value(speed[f"vendor-s{k}{spec}"], cell) for k in (1, 2)]
            table[f"{cell}{spec.replace('-', '_')}"] = {"A": a, "vendor": v,
                                                       "ratio_of_means": statistics.mean(a) / statistics.mean(v)}
    assert abs(table["decode_b1"]["ratio_of_means"] - ga["speed"]["ratio"]) < 1e-12
    accept = {lab: speed[lab]["dflash2"]["mean_acceptance_length"] for lab in speed if lab.endswith("dflash2")}
    reconnects = {lab: speed[lab]["reconnects"] for lab in speed}
    assert all(v == 0 for v in reconnects.values())

    q, s = runs["quality"], runs["speed"]
    kv = {"A": s["A-s1"]["kv_cache_tokens"], "vendor": s["vendor-s1"]["kv_cache_tokens"],
          "A_dflash2": s["A-s1-dflash2"]["kv_cache_tokens"], "vendor_dflash2": s["vendor-s1-dflash2"]["kv_cache_tokens"]}
    weights = {"A": s["A-s1"]["weights_gib"], "vendor": s["vendor-s1"]["weights_gib"],
               "A_dflash2": s["A-s1-dflash2"]["weights_gib"], "vendor_dflash2": s["vendor-s1-dflash2"]["weights_gib"]}
    versions = sorted({e["vllm_version"] for e in list(q.values()) + list(s.values())})
    assert len(versions) == 1
    power = sorted({w for e in s.values() for w in e["gpu"]["power_limit_w"]})
    kl = ga["kl"]
    gb = {k: v["tensor_bytes"] / 1e9 for k, v in sizes.items()}

    def r2(x: float) -> str:
        return f"{x:.2f}"

    t = table
    strings = {
        "gpqa_a": f"{corr['A'][0]} and {corr['A'][1]} of 198 (mean {mean['A']})",
        "gpqa_vendor": f"{corr['vendor'][0]} and {corr['vendor'][1]} of 198 (mean {mean['vendor']})",
        "gpqa_bar": f"bar: at least the vendor's mean − {0.06 * 198:.2f}",
        "gpqa_boot": f"paired bootstrap 95% CI of (A − vendor) {ga['gpqa']['bootstrap95_questions'][0]:+.1f} to "
                     f"{ga['gpqa']['bootstrap95_questions'][1]:+.1f} questions",
        "gpqa_trunc": f"{trunc['A'][0]} and {trunc['A'][1]} of A's, {trunc['vendor'][0]} and {trunc['vendor'][1]} of "
                      f"the vendor's ({capped['A'][0]} and {capped['A'][1]} / {capped['vendor'][0]} and "
                      f"{capped['vendor'][1]} responses hit the cap)",
        "gpqa_trunc_row": "Items without an answer at the 65,536-token cap",
        "gpqa_scoring": "Unparseable items and items without an answer at the token cap count as wrong; a response "
                        "that hit the cap with an answer is scored",
        "gpqa_reasoning": (f"a reasoning block on {min(reasoning['A'] + reasoning['vendor'])} of 198 items in every run of "
                           "both checkpoints"),
        "proxy_parse": f"{proxy['A']['overall']['parse_rate']:.3f} (vendor {proxy['vendor']['overall']['parse_rate']:.3f})",
        "proxy_success": f"{proxy['A']['overall']['success_rate']:.3f} (vendor {proxy['vendor']['overall']['success_rate']:.3f})",
        "proxy_split": f"BFCL v3 AST {bfcl['A']}/400 (vendor {bfcl['vendor']}/400); terminal and file-edit tasks "
                       f"{syn['A']}/100 (vendor {syn['vendor']}/100)",
        "kl": f"{kl['mean_kl']:.4f} (vendor {kl['vendor_kl']:.4f}; {kl['ratio']:.2f}x, 95% CI of the difference "
              f"{kl['bootstrap95_diff'][0]:+.4f} to {kl['bootstrap95_diff'][1]:+.4f})",
        "kl_top1": f"{100 * kl['top1_agreement']:.1f}% (vendor {100 * gate['vendor']['kl']['top1_agreement']:.1f}%)",
        "kl_positions": f"{kl['positions']:,} positions",
        "kl_tool": f"{kl['toolcall_span_kl']:.4f} (vendor {gate['vendor']['kl']['toolcall_span_kl']:.4f}) over "
                   f"{kl['toolcall_span_positions']:,} positions",
        "decode_b1": f"{r2(t['decode_b1']['A'][0])} / {r2(t['decode_b1']['A'][1])} tok/s (vendor "
                     f"{r2(t['decode_b1']['vendor'][0])} / {r2(t['decode_b1']['vendor'][1])}): "
                     f"{t['decode_b1']['ratio_of_means']:.2f}x",
        "decode_b8": f"{t['decode_b8']['A'][0]:.1f} / {t['decode_b8']['A'][1]:.1f} tok/s aggregate (vendor "
                     f"{t['decode_b8']['vendor'][0]:.1f} / {t['decode_b8']['vendor'][1]:.1f}): "
                     f"{t['decode_b8']['ratio_of_means']:.2f}x",
        "ttft_1k": f"{t['ttft_1k_b1']['A'][0]:.3f} / {t['ttft_1k_b1']['A'][1]:.3f} s (vendor "
                   f"{t['ttft_1k_b1']['vendor'][0]:.3f} / {t['ttft_1k_b1']['vendor'][1]:.3f}): "
                   f"{t['ttft_1k_b1']['ratio_of_means']:.2f}x",
        "ttft_16k": f"{t['ttft_16k_b1']['A'][0]:.2f} / {t['ttft_16k_b1']['A'][1]:.2f} s (vendor "
                    f"{t['ttft_16k_b1']['vendor'][0]:.2f} / {t['ttft_16k_b1']['vendor'][1]:.2f}): "
                    f"{t['ttft_16k_b1']['ratio_of_means']:.2f}x",
        "ttft_16k_b8": f"{t['ttft_16k_b8']['A'][0]:.1f} / {t['ttft_16k_b8']['A'][1]:.1f} s (vendor "
                       f"{t['ttft_16k_b8']['vendor'][0]:.1f} / {t['ttft_16k_b8']['vendor'][1]:.1f}): "
                       f"{t['ttft_16k_b8']['ratio_of_means']:.2f}x",
        "dflash_b1": f"{t['decode_b1_dflash2']['A'][0]:.1f} / {t['decode_b1_dflash2']['A'][1]:.1f} tok/s (vendor "
                     f"{t['decode_b1_dflash2']['vendor'][0]:.1f} / {t['decode_b1_dflash2']['vendor'][1]:.1f}): "
                     f"{t['decode_b1_dflash2']['ratio_of_means']:.2f}x",
        "dflash_b8": f"{t['decode_b8_dflash2']['A'][0]:.1f} / {t['decode_b8_dflash2']['A'][1]:.1f} tok/s aggregate "
                     f"(vendor {t['decode_b8_dflash2']['vendor'][0]:.1f} / {t['decode_b8_dflash2']['vendor'][1]:.1f}): "
                     f"{t['decode_b8_dflash2']['ratio_of_means']:.2f}x",
        "dflash_accept": f"{accept['A-s1-dflash2']:.2f} / {accept['A-s2-dflash2']:.2f} (vendor "
                         f"{accept['vendor-s1-dflash2']:.2f} / {accept['vendor-s2-dflash2']:.2f})",
        "speed_bar": f"batch-1 decode {statistics.mean(t['decode_b1']['A']):.2f} against the vendor's "
                     f"{statistics.mean(t['decode_b1']['vendor']):.2f} ({t['decode_b1']['ratio_of_means']:.2f}x; bar "
                     f"1.05x and above both vendor serves): PASS",
        "size": f"{gb['A']:.2f} GB ({sizes['A']['tensor_bytes'] / 2**30:.2f} GiB) in {sizes['A']['files']} shards",
        "size_vendor": f"{gb['vendor']:.2f} GB",
        "size_bf16": f"{gb['bf16']:.2f} GB",
        "kv_pool": f"{kv['A']:,} tokens (vendor {kv['vendor']:,})",
        "kv_pool_dflash2": f"{kv['A_dflash2']:,} tokens with DFlash2 (vendor {kv['vendor_dflash2']:,})",
        "weights_loaded": f"{weights['A']} GiB loaded (vendor {weights['vendor']})",
        "validation": f"{val['quantized_layers']['NVFP4']} NVFP4 W4A4, {val['quantized_layers']['FP8']} FP8 and "
                      f"{val['quantized_layers']['W4A16_NVFP4']} W4A16 NVFP4 linear",
        "validation_scales": f"{val['scales_checked']['f8_block_scales']:,} FP8 block scales and "
                             f"{val['scales_checked']['f32_scalars']:,} per-tensor scales",
        "validation_bf16": f"{val['bf16_tensors_byte_identical']} BF16 tensors byte-identical to the source",
        "build": f"{build['rows']} calibration rows ({build['tokens']['total']:,} tokens)",
        "power": f"{power[0]:.0f} W",
        "vllm": versions[0],
    }
    out = {
        "family": F,
        "product": "Darkstar-Ripple-65B-Preview-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8",
        "source": SOURCE, "source_license": SOURCE_LICENSE, "vendor_reference": VENDOR, "drafter": DRAFTER,
        "image": IMAGE,
        "vllm_version": versions[0],
        "vendor_reported_bf16_gpqa_diamond": {"value": VENDOR_BF16_GPQA,
                                              "note": "vendor-reported on the upstream card (BF16, the vendor's own "
                                                      "harness); not measured by this campaign"},
        "gate": {"bars_pass": ga["passes_bars"], "speed_win": ga["speed_win"], "quality_win": ga["quality_win"],
                 "eligible": ga["eligible"], "ships": gate["ships"]},
        "gpqa": {"runs": corr, "means": mean, "bootstrap95_questions": ga["gpqa"]["bootstrap95_questions"],
                 "bar_questions": round(0.06 * 198, 2), "truncated_no_answer": trunc, "hit_max_tokens": capped,
                 "items_with_reasoning_block": reasoning, "pass": ga["gpqa"]["pass"]},
        "proxy": {"A": proxy["A"]["overall"], "vendor": proxy["vendor"]["overall"], "bfcl_success": bfcl,
                  "synthetic_success": syn, "pass": ga["proxy"]["pass"]},
        "kl": {"A": {k: kl[k] for k in ("positions", "mean_kl", "top1_agreement", "toolcall_span_kl",
                                        "toolcall_span_positions")},
               "vendor": gate["vendor"]["kl"], "ratio": kl["ratio"], "bootstrap95_diff": kl["bootstrap95_diff"]},
        "speed": {"cells": table, "dflash2_mean_acceptance_length": accept, "reconnects": reconnects,
                  "power_limit_w": power},
        "memory": {"kv_cache_tokens": kv, "weights_loaded_gib": weights,
                   "checkpoint_gb": {k: round(v, 2) for k, v in gb.items()}},
        "artifact_validation": val,
        "card_strings": strings,
    }
    if (R / "clean-download.json").exists():
        out["clean_download"] = load("clean-download.json")
    if (R / "hf-upload.json").exists():
        out["hf_upload"] = load("hf-upload.json")
    (R / "gate-summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in strings.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
