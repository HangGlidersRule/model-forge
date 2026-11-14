#!/usr/bin/env python3
"""Write the Ripple-65B Preview publication-readiness ledger (contract 3.1.0) from the committed evidence.

Mid-release (default): the shipped product is locally_complete_unpublished and publication_claim is false. Of its
publication gates, model_card_final is verified (the contract asks the card for the exact planned tag reference and
verifies the tag's existence separately, under release_tag); the HF target, the clean download plus boot smoke (it
attests the public repository, so the private staging repository's passed smoke is its evidence so far) and the release
tag are in_progress. --published sets them verified and needs results/publication-receipt.json. Every figure is read
from results/gate-summary.json (built by eval/build_gate_summary.py) or results/staging-receipt.json (its boot_smoke
block written by eval/curate_clean_boot.py). Run from the repository root.
Usage: build_ledger.py [--published]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

F = "ripple-65b"
R = Path(f"models/{F}/results")
g = json.loads((R / "gate-summary.json").read_text())
st = json.loads((R / "staging-receipt.json").read_text())
published = "--published" in sys.argv
receipt: dict = json.loads((R / "publication-receipt.json").read_text()) if published else {}
PID = "Darkstar-Ripple-65B-Preview-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8"
TARGET = f"HangGlidersRule/{PID}"
SRC_REPO, SRC_REV = "ricursiveAI/Ripple-65B-Preview", "240d5966cbe00f0627c699e59ef58560b60303cf"
GS = f"models/{F}/results/gate-summary.json"
VAL = f"models/{F}/results/validation-A.json"
STG = f"models/{F}/results/staging-receipt.json"
CAL = f"models/{F}/results/calibration-record.json"
CARD = f"models/{F}/model-card/base-nvfp4.md"
RECIPE = f"recipes/{F}/darkstar-ripple-65b-preview-base-modelopt-w4a4-nvfp4-mixed-fp8.yaml"
OPS = "configs/modelopt/recipes/ripple-65b"
cs = g["card_strings"]
NOT_BUILT = {"language_mlp": "not built", "lm_head": "not built", "self_attention": "not built",
             "gdn_projections": "not present (dense transformer, no linear-attention projections)", "kv_cache": "BF16",
             "protected": "not built"}


def gate(gid: str, status: str, evidence: str, path: str | None, **extra: str) -> dict:
    nxt = {"verified": "none - gate satisfied", "not_applicable": "none - not applicable"}.get(status, extra.pop("next", "pending"))
    return {"id": gid, "status": status, "evidence": evidence, "evidence_path": path, "next_missing_proof": nxt, **extra}


def precision(mlp: str, attn: str) -> dict:
    return {"language_mlp": mlp, "lm_head": "W4A16-NVFP4-g16 (weight-only, MSE weight scales)",
            "self_attention": attn, "gdn_projections": "not present (dense transformer, no linear-attention projections)",
            "kv_cache": "BF16", "protected": "embeddings, norms, value_res_lambda (BF16, byte-identical to upstream)"}


va = g["artifact_validation"]
q = va["quantized_layers"]
# Provenance: the source security@hangglidersrule.com and its license, read from gate-summary.json (build_gate_summary.py records the
# license the upstream card declares at that revision), so the gates' text is recomputable from the file they cite.
src_repo, _, src_rev = g["source"].partition("@")
if (src_repo, src_rev) != (SRC_REPO, SRC_REV):
    raise SystemExit(f"gate-summary.json source {g['source']} is not {SRC_REPO}@{SRC_REV}")
src_license = g["source_license"]["license"]
LICENSE_NAMES = {"apache-2.0": "Apache-2.0"}
if src_license not in LICENSE_NAMES:
    raise SystemExit(f"gate-summary.json: no display name for the source license {src_license!r}")
lic = LICENSE_NAMES[src_license]
# The schema-2 recipe the recipe_edit_manifest gate cites: every value its text quotes is read from it.
recipe_text = Path(RECIPE).read_text()
recipe_pins = {k: re.search(rf"^\s*{k}: (\S+)$", recipe_text, re.M) for k in
               ("schema_version", "operator_recipe_sha256", "framework_version", "framework_pin_commit")}
if not all(recipe_pins.values()):
    raise SystemExit(f"{RECIPE}: missing {[k for k, m in recipe_pins.items() if not m]}")
rp = {k: m.group(1).strip('"') for k, m in recipe_pins.items() if m}
lm_pin = re.search(r"lm_head_w4a16_nvfp4_mse\.yaml \(sha256\s+#?\s*([0-9a-f]{64})\)", recipe_text.replace("\n#", ""))
if not lm_pin:
    raise SystemExit(f"{RECIPE}: no lm_head recipe sha256")
up, cd, bs = st["upload"], st["clean_download"], st["boot_smoke"]
if bs["status"] != "passed":
    raise SystemExit(f"staging-receipt.json: the boot smoke is {bs['status']!r}, not passed")
bp, bd = bs["modes"]["plain"], bs["modes"]["dflash2"]
card_commit = st["card_commit"]["commit"]
if bs["covers_card_commit"] != card_commit:
    raise SystemExit("staging-receipt.json: the boot smoke does not cover the card commit (weights changed)")
boot = (f"booted with the card's serve command on the measured image ({bs['run_utc']['ended'][:10]}): plain up in "
        f"{bp['boot_seconds']} s (KV pool {bp['kv_cache_tokens']:,} tokens), with DFlash2 in {bd['boot_seconds']} s "
        f"({bd['kv_cache_tokens']:,} tokens); in both, the known answers, a parsed tool call and the chat request (finish "
        f"{bp['chat_request_finish_reason']}), {bp['fatal_log_markers'] + bd['fatal_log_markers']} fatal log markers. "
        f"The card commit {card_commit[:12]} has the same weights")
pub_gates = [
    gate("publication_targets_hf_ghcr", "verified" if published else "in_progress",
         (f"Public at {TARGET} @ {receipt.get('public_commit', '')[:12]}; GHCR not required (pinned upstream vLLM image)"
          if published else
          f"Private staging at {TARGET}: weights @ {st['uploaded_commit'][:12]} ({up['files_on_hub']} files, every LFS sha256 "
          f"equal to the export's manifest), card @ {st['card_commit']['commit'][:12]} (weights unchanged, all "
          f"{st['card_commit']['manifest_entries_verified_on_hub']} manifest entries verified on the hub); public flip after "
          "the export PR merges and the tag exists"),
         f"models/{F}/results/publication-receipt.json" if published else STG, next="merge the export PR, tag, flip public"),
    gate("clean_download_boot_smoke", "verified" if published else "in_progress",
         ("Clean download of the public repository plus a boot with the card's command" if published else
          f"Clean download of the private repository @ {cd['commit'][:12]} verified ({cd['entries_matching_export']}/"
          f"{cd['manifest_entries']} manifest entries byte-identical to the export, the validator passes on the downloaded "
          f"bytes) and {boot}. Open until the public repository is downloaded after the flip: the gate attests the "
          "published artifact"),
         f"models/{F}/results/publication-receipt.json" if published else STG,
         next="the clean download of the public repository after the flip"),
    gate("release_tag", "verified" if published else "in_progress",
         "darkstar-ripple-65b-v1.0.0" if published else "Tag darkstar-ripple-65b-v1.0.0 follows the export PR merge",
         f"models/{F}/README.md" if published else None, next="tag the merge commit"),
]
quality = (f"GPQA Diamond (198, 65,536-token cap; {cs['gpqa_reasoning']}), two runs: {cs['gpqa_a']} against the vendor "
           f"NVFP4's {cs['gpqa_vendor']}; {cs['gpqa_bar']}: PASS; {cs['gpqa_boot']}. Full denominator; items without an "
           f"answer at the cap, {cs['gpqa_trunc']}, count as wrong; a response that hit the cap with an answer is scored. "
           "BF16 not servable on the 96 GB GPU; vendor-reported BF16 85.2 quoted only")
products = [
    {
        "product_id": "Darkstar-Ripple-65B-Preview-Base-BF16", "role": "base-bf16", "task": "chat",
        "lifecycle": "in_progress", "target_repository": SRC_REPO, "target_repository_status": "resolved",
        "target_repository_note": ("Unchanged upstream reference (upstream reference only); no HangGlidersRule repository. "
                                   "Too large for the 96 GB GPU: closeness to it is measured as next-token KL against a "
                                   "CPU teacher of these weights."),
        "served_model_alias": "darkstar-ripple65b-base-bf16", "runtime_container_name": "vllm-darkstar-ripple65b-base-bf16",
        "model_card": f"models/{F}/model-card/base-bf16.md", "recipe": None, "publication_claim": False,
        "gates": [
            gate("provenance_ownership", "verified",
                 f"Source pinned to {g['source']} ({lic}: the license its card declares at that revision); unchanged "
                 "upstream reference, not owned or republished", GS),
            gate("performance_profile", "missing",
                 f"Not measured: BF16 ({cs['size_bf16']}) does not fit the single 96 GB GPU the family is measured on",
                 None, next="a BF16 serve on a larger GPU"),
            gate("gpqa_matched_full_denominator", "missing",
                 "Not measured: BF16 does not fit the 96 GB GPU. The vendor reports 85.2 (their harness); quoted, never "
                 "counted as a measurement", None, next="a BF16 serve on a larger GPU"),
        ],
    },
    {
        "product_id": PID, "role": "base-modelopt-nvfp4", "task": "chat",
        "lifecycle": "published" if published else "locally_complete_unpublished",
        "target_repository": TARGET, "target_repository_status": "resolved",
        "served_model_alias": "darkstar-ripple65b-base-nvfp4", "runtime_container_name": "vllm-darkstar-ripple65b-base-modelopt",
        "model_card": CARD, "recipe": RECIPE, "publication_claim": published,
        "candidates": [
            {"candidate_id": PID, "precision_class": "W4A4-NVFP4-Mixed-FP8", "target_repository": TARGET,
             "selection": "selected", "promotion_status": "promoted",
             "promotion_note": (f"Candidate A of #39: every bar passed and the pre-registered speed win ({cs['speed_bar']}); "
                                f"no quality win (KL {cs['kl']}). C and B were not built (fallback rule)."),
             "recipe": f"{OPS}/a_nvfp4_w4a4_mlp-fp8_attn-kv_bf16_local_hessian_layerwise.yaml", "evidence_path": GS,
             "immutable_hashes": {**{k: v for k, v in cd["safetensors_sha256"].items()},
                                  "manifest.sha256": cd["manifest_sha256"],
                                  "operator_recipe": "624185c8df017e0130979905a36885bd761131baae6d6535472fa9d3515f9e4e",
                                  "lm_head_recipe": "123e4e99af9634a16d5b6c467e8341ad6c0e72b93038f94fc2181b79f61e74f6",
                                  "calibration_jsonl": "403730e417b94a5e412144ee72b47975c3996c61abf8a32ac7580cf0c73ca508"},
             "precision_map": precision("W4A4-NVFP4-g16 (gate/up/down, 80 layers, 240 linears)",
                                        "FP8-W8A8 per-tensor (q/k/v/o and the output gate g_proj, 80 layers, 400 linears)")},
            {"candidate_id": "Darkstar-Ripple-65B-Preview-Base-ModelOpt-W4A4-NVFP4", "precision_class": "W4A4-NVFP4",
             "target_repository": "HangGlidersRule/Darkstar-Ripple-65B-Preview-Base-ModelOpt-W4A4-NVFP4",
             "selection": "not_built", "promotion_status": "none",
             "promotion_note": "Candidate C (attention BF16): built only if A failed a bar; A passed, so not built.",
             "recipe": f"{OPS}/c_nvfp4_w4a4_mlp-bf16_attn-kv_bf16_local_hessian_layerwise.yaml",
             "precision_map": precision("W4A4-NVFP4-g16 (gate/up/down, 80 layers)", "BF16")},
            {"candidate_id": "Darkstar-Ripple-65B-Preview-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8",
             "precision_class": "W4A16-NVFP4-Mixed-FP8",
             "target_repository": "HangGlidersRule/Darkstar-Ripple-65B-Preview-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8",
             "selection": "not_built", "promotion_status": "none",
             "promotion_note": "Candidate B (MLP weight-only): built only if C also failed; not built.",
             "recipe": f"{OPS}/b_w4a16_nvfp4_mlp-fp8_attn-kv_bf16_local_hessian_layerwise.yaml",
             "precision_map": precision("W4A16-NVFP4-g16 (gate/up/down, 80 layers, weight-only)",
                                        "FP8-W8A8 per-tensor (q/k/v/o/g, 80 layers)")},
        ],
        "gates": [
            gate("provenance_ownership", "verified",
                 f"Built from {g['source']} BF16 ({lic}: the license its card declares at that revision; the product "
                 "ships its LICENSE byte-identical); the vendor NVFP4 scored as the reference only, never a build input",
                 GS),
            gate("artifact_manifest", "verified",
                 f"manifest.sha256 over {up['manifest_entries']} files (sha256 {up['manifest_sha256'][:12]}...), written at "
                 "upload; every LFS sha256 on the hub equal to it", STG),
            gate("recipe_edit_manifest", "verified",
                 f"Schema-{rp['schema_version'].split('.')[0]} recipe; operator recipes sha256 "
                 f"{rp['operator_recipe_sha256'][:8]}... (decoder) and {lm_pin.group(1)[:8]}... (lm_head); ModelOpt "
                 f"{rp['framework_version']} pinned at commit {rp['framework_pin_commit'][:8]}...", RECIPE),
            gate("artifact_validation", "verified",
                 f"Fail-closed validator PASS: {va['tensors']:,} tensors; {cs['validation_bf16']}; quant config "
                 "MIXED_PRECISION with one entry per linear in config.json and hf_quant_config.json; architecture fields "
                 "equal the source's; remote code, chat template and tokenizer byte-identical", VAL),
            gate("modelopt_candidate_comparison", "verified",
                 f"A against the vendor NVFP4 under the posted bars: GPQA, tool-call proxy and artifact checks PASS; speed win "
                 f"({cs['decode_b1']}); C and B not built (fallback rule)", GS),
            gate("nvfp4_tensor_scale_validation", "verified",
                 f"{q['NVFP4']} W4A4 NVFP4, {q['FP8']} FP8 and {q['W4A16_NVFP4']} W4A16 NVFP4 linear; {cs['validation_scales']} "
                 "finite and positive (0 NaN, 0 zero, 0 negative); q/k/g input scales and gate/up global and input scales "
                 "shared; no KV quantization", VAL),
            gate("kv_cache_effect_measured", "not_applicable", "kv_cache is BF16", None),
            gate("performance_profile", "verified",
                 f"RTX PRO 6000 Blackwell at {cs['power']}, two serves each against the vendor NVFP4: decode batch 1 "
                 f"{cs['decode_b1']}; batch 8 {cs['decode_b8']}; TTFT ~16k {cs['ttft_16k']}; with DFlash2 {cs['dflash_b1']}",
                 GS),
            gate("serving_capacity_profile", "verified",
                 f"Weights {cs['weights_loaded']}; KV pool at fraction 0.92: {cs['kv_pool']}; {cs['kv_pool_dflash2']}; "
                 f"batch-8 TTFT at ~16k tokens {cs['ttft_16k_b8']}", GS),
            gate("gpqa_matched_full_denominator", "verified", quality, GS),
            gate("serve_profile_frozen", "verified",
                 "Exact vllm serve command on the card, the measured flags (ripple parsers, auto tool choice, 131,072 "
                 "context, 32 sequences, fraction 0.92) on the pinned nightly digest", CARD),
            gate("model_card_final", "verified",
                 "Gold-standard card: Apache-2.0 with its LICENSE file, evidence links, no placeholders, and the exact "
                 "planned release tag darkstar-ripple-65b-v1.0.0 (whether the tag exists is the release_tag gate); every "
                 "number checked by eval/check_card_numbers.py", CARD),
            *pub_gates,
            gate("no_inherited_unverified_results", "verified",
                 "Every number on the card measured on this exact artifact; the BF16 GPQA is labelled vendor-reported", GS),
        ],
    },
    {
        "product_id": "Darkstar-Ripple-65B-Preview-Abliterated-BF16", "role": "abliterated-bf16", "task": "chat",
        "lifecycle": "in_progress", "target_repository": "HangGlidersRule/Darkstar-Ripple-65B-Preview-Abliterated-BF16",
        "target_repository_status": "resolved",
        "served_model_alias": "darkstar-ripple65b-abliterated-bf16", "runtime_container_name": "vllm-darkstar-ripple65b-abliterated-bf16",
        "model_card": CARD, "recipe": None, "publication_claim": False,
        "gates": [gate("abliteration_pass", "missing", "Not planned in #39 (a base quantization campaign)", None,
                       next="a separate abliteration campaign"),
                  gate("gpqa_matched_full_denominator", "missing", "Not built", None, next="build the cell first")],
    },
    {
        "product_id": "Darkstar-Ripple-65B-Preview-Abliterated-ModelOpt-NVFP4", "role": "abliterated-modelopt-nvfp4",
        "task": "chat", "lifecycle": "in_progress", "target_repository": None,
        "target_repository_status": "unresolved_pending_precision_winner",
        "served_model_alias": "darkstar-ripple65b-abliterated-nvfp4",
        "runtime_container_name": "vllm-darkstar-ripple65b-abliterated-modelopt",
        "model_card": CARD, "recipe": None, "publication_claim": False,
        "candidates": [{"candidate_id": "Darkstar-Ripple-65B-Preview-Abliterated-ModelOpt-W4A4-NVFP4",
                        "precision_class": "W4A4-NVFP4",
                        "target_repository": "HangGlidersRule/Darkstar-Ripple-65B-Preview-Abliterated-ModelOpt-W4A4-NVFP4",
                        "selection": "not_built", "promotion_status": "none",
                        "promotion_note": "Reserved id only; no abliteration is planned in #39.", "recipe": None,
                        "precision_map": NOT_BUILT}],
        "gates": [gate("abliteration_pass", "missing", "Not planned in #39", None, next="a separate abliteration campaign"),
                  gate("gpqa_matched_full_denominator", "missing", "Not built", None, next="build the cell first"),
                  gate("kv_cache_effect_measured", "not_applicable",
                       "No selected candidate: nothing is served, so there is no KV cache", None)],
    },
]
ledger = {
    "schema_version": "1.0.0", "contract": "contracts/darkstar-release/v3/contract.json", "contract_version": "3.1.0",
    "family": F, "family_display": "Darkstar Ripple-65B Preview",
    "source_revision": f"{SRC_REPO}@{SRC_REV}",
    "note": ("Dense 65B chat model (80 layers, gated sliding-window and global attention, value residuals). One "
             "quantized cell (#39): W4A4 NVFP4 MLP, FP8 attention, W4A16 NVFP4 lm_head, measured against the vendor's "
             "own NVFP4 checkpoint on one 96 GB GPU and a vLLM nightly. The BF16 control is too large for that GPU; "
             "closeness to BF16 is next-token KL against a CPU teacher. No abliteration cells are planned in #39."),
    "any_publication_claim": published, "products": products, "rejected_historical": [],
    "related_issues": ["HangGlidersRule/model-forge#39"],
}
(R / "publication-readiness-ledger.json").write_text(json.dumps(ledger, indent=1) + "\n")
print("ledger written; published =", published)
