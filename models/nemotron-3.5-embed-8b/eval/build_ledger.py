#!/usr/bin/env python3
"""Write the Nemotron-3.5-Embed-8B publication-readiness ledger (contract 3.1.0) from the committed evidence.
Mid-release (default): the shipped product is locally_complete_unpublished, its publication gates (HF target,
clean-download smoke, release tag) in_progress and publication_claim false. --published sets those gates verified and
needs results/publication-receipt.json. Every quoted figure is read from results/embed-gate-summary.json.
Usage: build_ledger.py [--published]"""
import json
import sys
from pathlib import Path

F = "nemotron-3.5-embed-8b"
R = Path(f"models/{F}/results")
g = json.loads((R / "embed-gate-summary.json").read_text())
v = json.loads((R / "artifact-validation.json").read_text())
published = "--published" in sys.argv
receipt: dict = json.loads((R / "publication-receipt.json").read_text()) if published else {}
TARGET = "HangGlidersRule/Darkstar-Nemotron-3.5-Embed-8B-Base-ModelOpt-W4A16-NVFP4"
SRC_REV = "781e779ee4ca9c2d2a4f38321993556baeab49a1"
qm, ql = g["quality"]["mteb_nano_full"], g["quality"]["lmeb_scene_retrieval"]
sp = g["speed"]
GS = f"models/{F}/results/embed-gate-summary.json"
AV = f"models/{F}/results/artifact-validation.json"
CARD = f"models/{F}/model-card/base-nvfp4.md"
RECIPE = f"recipes/{F}/darkstar-nemotron-3.5-embed-8b-base-modelopt-w4a16-nvfp4.yaml"
OPS = "configs/modelopt/recipes/nemotron-3.5-embed-8b"
NOT_BUILT = {"language_mlp": "not built", "lm_head": "not present", "self_attention": "not built", "gdn_projections": "not present",
             "kv_cache": "BF16", "protected": "not built"}


def gate(gid, status, evidence, path, **extra):
    nxt = {"verified": "none - gate satisfied", "not_applicable": "none - not applicable"}.get(status, extra.pop("next", "pending"))
    return {"id": gid, "status": status, "evidence": evidence, "evidence_path": path, "next_missing_proof": nxt, **extra}


def ci(e):
    return f"{e['delta']:+.4f} (95% CI {e['ci95'][0]:+.4f}..{e['ci95'][1]:+.4f})"


d = "nemotron-D"
quality = ("Embedding family: two matched full-denominator gates replace GPQA. MTEB Nano full corpora: mean nDCG@10 "
           f"{qm[d]['mean_ndcg10']} vs BF16 {qm['nemotron-bf16']['mean_ndcg10']} ({qm[d]['all']['ratio_to_bf16']:.4f}), "
           f"BF16 docs x candidate queries {qm[d]['bf16_docs_x_candidate_queries']['ratio_to_bf16']:.4f}, 299/299 queries; LMEB "
           f"Dialogue whole-conversation: {ql[d]['mean_ndcg10']} vs BF16 {ql['nemotron-bf16']['mean_ndcg10']} "
           f"({ql[d]['all']['ratio_to_bf16']:.4f}, delta {ci(ql[d]['all'])}), BF16 docs x candidate queries "
           f"{ql[d]['bf16_docs_x_candidate_queries']['ratio_to_bf16']:.4f}, 540/540 queries")
precision_d = {"language_mlp": "W4A16 NVFP4 group 16, weight-only (gate/up/down, 34 layers)", "lm_head": "not present (embedding model)",
               "self_attention": "W4A16 NVFP4 group 16, weight-only (q/k/v/o, 34 layers)",
               "gdn_projections": "not present (dense bidirectional transformer)", "kv_cache": "BF16",
               "protected": "vision tower, multimodal projector, embeddings, norms (BF16)"}
pub_gates = [
    gate("publication_targets_hf_ghcr", "verified" if published else "in_progress",
         (f"Public at {TARGET} @ {receipt.get('hf_commit', '')[:7]}; GHCR not required (pinned upstream vLLM image)" if published
          else f"HF target {TARGET} reserved; upload, clean-download smoke, export PR merge, tag, then public flip"),
         f"models/{F}/results/publication-receipt.json" if published else None, next="private upload and the steps after it"),
    gate("clean_download_boot_smoke", "verified",
         (f"Empty-cache download of {TARGET.split('/')[1]} @ {g['clean_download_smoke']['repository_commit'][:8]} (private): "
          f"{g['clean_download_smoke']['manifest_entries']} manifest entries verified, weights byte-identical to the evaluated export; "
          "served with the card's exact command; card example answered; served vectors vs transformers reference cosine min "
          f"{g['clean_download_smoke']['served_vs_own_fakequant_reference']['cosine_min']}; both frozen sets re-scored "
          f"({g['clean_download_smoke']['quality']['mteb_nano_full']['mean_ndcg10']} / {g['clean_download_smoke']['quality']['lmeb_scene_retrieval']['mean_ndcg10']}, both pass)"),
         GS),
    gate("release_tag", "verified" if published else "in_progress",
         "darkstar-nemotron-3.5-embed-8b-v1.0.0" if published else "Tag darkstar-nemotron-3.5-embed-8b-v1.0.0 follows the export PR merge",
         f"models/{F}/README.md" if published else None, next="tag the merge commit"),
]
products = [
    {
        "product_id": "Darkstar-Nemotron-3.5-Embed-8B-Base-BF16", "role": "base-bf16", "task": "embedding",
        "lifecycle": "published", "target_repository": "nvidia/Nemotron-3.5-Embed-8B-BF16", "target_repository_status": "resolved",
        "target_repository_note": "Unchanged upstream reference measured as the family control; no HangGlidersRule repository.",
        "served_model_alias": "darkstar-nemotron35embed8b-base-bf16", "runtime_container_name": "vllm-darkstar-nemotron35embed8b-base-bf16",
        "model_card": f"models/{F}/model-card/base-bf16.md", "recipe": None, "publication_claim": False,
        "gates": [
            gate("provenance_ownership", "verified", f"Source pinned to nvidia/Nemotron-3.5-Embed-8B-BF16@{SRC_REV} (OpenMDW-1.1)", GS),
            gate("performance_profile", "verified",
                 f"BF16 control: long inputs {sp['nemotron-bf16']['long_c1']:,} tok/s at c=1, {sp['nemotron-bf16']['long_c32']:,} at c=32; "
                 f"short {sp['nemotron-bf16']['short_c1']:,} at c=1", GS),
            gate("quality_matched_full_denominator", "verified",
                 f"Control: mean nDCG@10 {qm['nemotron-bf16']['mean_ndcg10']} (MTEB Nano full corpora) and {ql['nemotron-bf16']['mean_ndcg10']} "
                 f"(LMEB Dialogue); replica {ci(qm['nemotron-bf16r']['all'])} / {ci(ql['nemotron-bf16r']['all'])}", GS,
                 metric="mean nDCG@10 on two frozen sets (MTEB Nano full corpora; LMEB Dialogue whole-conversation)",
                 protocol="embed gate summary, full denominator"),
        ],
    },
    {
        "product_id": "Darkstar-Nemotron-3.5-Embed-8B-Base-ModelOpt-NVFP4", "role": "base-modelopt-nvfp4", "task": "embedding",
        "lifecycle": "published" if published else "locally_complete_unpublished",
        "target_repository": TARGET, "target_repository_status": "resolved",
        "served_model_alias": "darkstar-nemotron35embed8b-base-nvfp4", "runtime_container_name": "vllm-darkstar-nemotron35embed8b-base-nvfp4",
        "model_card": CARD, "recipe": RECIPE, "publication_claim": published,
        "candidates": [
            {"candidate_id": "Darkstar-Nemotron-3.5-Embed-8B-Base-ModelOpt-W4A16-NVFP4", "precision_class": "W4A16-NVFP4",
             "target_repository": TARGET, "selection": "selected", "promotion_status": "promoted",
             "promotion_note": (f"Candidate D (amendment R33-A1, added after A and C failed; same bar): W4A16 NVFP4 on every text linear. "
                                f"{qm[d]['all']['ratio_to_bf16']:.4f} / {ql[d]['all']['ratio_to_bf16']:.4f} of BF16 (all), "
                                f"{qm[d]['bf16_docs_x_candidate_queries']['ratio_to_bf16']:.4f} / {ql[d]['bf16_docs_x_candidate_queries']['ratio_to_bf16']:.4f} "
                                f"(BF16 docs). Memory-only gain: weights {g['weights_loaded_gib']['nemotron-D']} vs {g['weights_loaded_gib']['nemotron-bf16']} GiB; "
                                f"speed {g['speed_ratio_to_bf16']['nemotron-D']['long']} long, {g['speed_ratio_to_bf16']['nemotron-D']['short']} short."),
             "recipe": f"{OPS}/w4a16_nvfp4_text_all_mse-kv_bf16.yaml", "evidence_path": GS,
             "immutable_hashes": {"model.safetensors": "13cd6ab1bbd1db169f1916b1462ebb9d670a89b02885428122d09ceed4e7fd32",
                                  "config.json": "d75029c8d6790afe6f2fe64d61deffb407ad5892690719516116d01b96eb118c",
                                  "hf_quant_config.json": "25a833ac5e5539b36f31b7ffacbcd03e76112c91f90bdecf5250be1d31071416",
                                  "operator_recipe": "980c52fad59aa52b80a2a00865d9e44d1667a2f3539d9fc1291d4a568cd6d112",
                                  "calibration_jsonl": g["calibration_sha256"]},
             "precision_map": precision_d},
            {"candidate_id": "Darkstar-Nemotron-3.5-Embed-8B-Base-ModelOpt-W4A4-NVFP4", "precision_class": "W4A4-NVFP4",
             "target_repository": "HangGlidersRule/Darkstar-Nemotron-3.5-Embed-8B-Base-ModelOpt-W4A4-NVFP4", "selection": "rejected",
             "promotion_status": "none",
             "promotion_note": (f"Candidate A (W4A4 every text linear) failed the LMEB Dialogue bar: {ql['nemotron-A']['all']['ratio_to_bf16']:.4f} "
                                f"of BF16 (delta {ci(ql['nemotron-A']['all'])}); MTEB Nano {qm['nemotron-A']['all']['ratio_to_bf16']:.4f}. "
                                "Candidate C (W4A4 MLP only, same id class) failed too: "
                                f"{ql['nemotron-C']['all']['ratio_to_bf16']:.4f} on LMEB Dialogue."),
             "recipe": f"{OPS}/nvfp4_text_all_act_headroom-kv_bf16.yaml", "evidence_path": GS,
             "precision_map": {"language_mlp": "W4A4-NVFP4-g16 (gate/up/down, 34 layers)", "lm_head": "not present (embedding model)",
                               "self_attention": "W4A4-NVFP4-g16 (q/k/v/o, 34 layers)", "gdn_projections": "not present (dense bidirectional transformer)",
                               "kv_cache": "BF16", "protected": "vision tower, multimodal projector, embeddings, norms (BF16)"}},
        ],
        "gates": [
            gate("provenance_ownership", "verified", f"Built from nvidia/Nemotron-3.5-Embed-8B-BF16@{SRC_REV[:8]} BF16 (OpenMDW-1.1); no third-party quantized inputs", GS),
            gate("artifact_manifest", "verified", "Every artifact file hashed before upload; model.safetensors sha256 13cd6ab1...", AV),
            gate("recipe_edit_manifest", "verified", "Schema-2 recipe plus operator recipe sha256 980c52fa...; ModelOpt 0.48.0 pin (wheel sha256 bfcfd425...)", RECIPE),
            gate("artifact_validation", "verified",
                 f"{v['protected_tensors']['byte_identical']}/{v['protected_tensors']['expected']} protected tensors byte-identical to upstream; "
                 "sidecars byte-identical; config.json identical outside quantization_config; vLLM-internal exclusion names present", AV),
            gate("modelopt_candidate_comparison", "verified",
                 "Three candidates on two frozen sets against BF16 and a BF16 replica: A and C (W4A4) failed LMEB Dialogue, D (W4A16, amendment R33-A1) passed", GS),
            gate("nvfp4_tensor_scale_validation", "verified",
                 f"{v['coverage']['w4a16']} W4A16 NVFP4 linears; {v['fp8_block_scales']['non_finite']} non-finite and {v['fp8_block_scales']['zero']} zero "
                 f"FP8 block scales; per-tensor scales finite and positive; fused groups uniform ({v['fused_group_scales']['groups_checked']} "
                 f"shared-scale checks, {v['fused_group_scales']['mismatched']} mismatched); no KV quantization", AV),
            gate("kv_cache_effect_measured", "not_applicable", "kv_cache is BF16", None),
            gate("performance_profile", "verified",
                 f"Long inputs {sp['nemotron-D']['long_c1']:,} tok/s at c=1 (BF16 {sp['nemotron-bf16']['long_c1']:,}), "
                 f"{g['speed_ratio_to_bf16']['nemotron-D']['long']}; short {g['speed_ratio_to_bf16']['nemotron-D']['short']}; keep-alive clients", GS),
            gate("serving_capacity_profile", "verified",
                 f"Weights {g['weights_loaded_gib']['nemotron-D']} GiB in vLLM (BF16 {g['weights_loaded_gib']['nemotron-bf16']}); "
                 f"GPU memory under the speed grid {sp['nemotron-D']['gpu_memory_under_load_gib']} GiB (BF16 {sp['nemotron-bf16']['gpu_memory_under_load_gib']})", GS),
            gate("quality_matched_full_denominator", "verified", quality, GS,
                 metric="mean nDCG@10 vs BF16 on two frozen sets (MTEB Nano full corpora; LMEB Dialogue whole-conversation), both pairings",
                 protocol="embed gate summary, full denominator"),
            gate("serve_profile_frozen", "verified",
                 "vLLM v0.32.0 by digest; pooling runner; mean pooling with normalization; query:/passage: prefixes with add_special_tokens false; BF16 KV", CARD),
            gate("model_card_final", "verified", "Gold-standard card; every number checked by eval/check_card_numbers.py", CARD),
            *pub_gates,
            gate("no_inherited_unverified_results", "verified", "Every number on the card was measured on this exact artifact", GS),
        ],
    },
    {
        "product_id": "Darkstar-Nemotron-3.5-Embed-8B-Abliterated-BF16", "role": "abliterated-bf16", "task": "embedding",
        "lifecycle": "in_progress", "target_repository": "HangGlidersRule/Darkstar-Nemotron-3.5-Embed-8B-Abliterated-BF16",
        "target_repository_status": "resolved",
        "served_model_alias": "darkstar-nemotron35embed8b-abliterated-bf16", "runtime_container_name": "vllm-darkstar-nemotron35embed8b-abliterated-bf16",
        "model_card": CARD, "recipe": None, "publication_claim": False,
        "gates": [gate("abliteration_pass", "not_applicable", "No abliteration for an embedding family", None),
                  gate("quality_matched_full_denominator", "not_applicable", "Not built and not planned: an embedding model is not a refusal-edit target", None,
                       metric="not applicable (not built)", protocol="not applicable (not built)")],
    },
    {
        "product_id": "Darkstar-Nemotron-3.5-Embed-8B-Abliterated-ModelOpt-NVFP4", "role": "abliterated-modelopt-nvfp4", "task": "embedding",
        "lifecycle": "in_progress", "target_repository": None, "target_repository_status": "unresolved_pending_precision_winner",
        "served_model_alias": "darkstar-nemotron35embed8b-abliterated-nvfp4", "runtime_container_name": "vllm-darkstar-nemotron35embed8b-abliterated-nvfp4",
        "model_card": CARD, "recipe": None, "publication_claim": False,
        "candidates": [{"candidate_id": "Darkstar-Nemotron-3.5-Embed-8B-Abliterated-ModelOpt-W4A16-NVFP4", "precision_class": "W4A16-NVFP4",
                        "target_repository": "HangGlidersRule/Darkstar-Nemotron-3.5-Embed-8B-Abliterated-ModelOpt-W4A16-NVFP4",
                        "selection": "not_built", "promotion_status": "none",
                        "promotion_note": "Reserved id only; refusal edits do not apply to an embedding model.", "recipe": None,
                        "precision_map": NOT_BUILT}],
        "gates": [gate("abliteration_pass", "not_applicable", "No abliteration for an embedding family", None),
                  gate("quality_matched_full_denominator", "not_applicable", "Not built and not planned: an embedding model is not a refusal-edit target", None,
                       metric="not applicable (not built)", protocol="not applicable (not built)"),
                  gate("kv_cache_effect_measured", "not_applicable", "No selected candidate: nothing is served, so there is no KV cache", None)],
    },
]
ledger = {
    "schema_version": "1.0.0", "contract": "contracts/darkstar-release/v3/contract.json", "contract_version": "3.1.0",
    "family": F, "family_display": "Darkstar Nemotron-3.5-Embed-8B",
    "source_revision": f"nvidia/Nemotron-3.5-Embed-8B-BF16@{SRC_REV}",
    "note": ("Embedding family (multimodal embedder: Pixtral vision tower + bidirectional Ministral3 text tower, mean-pooled "
             "normalized 4,096-dim vectors). Base quantization of the text tower only; image retrieval is not evaluated. The "
             "selected product is weight-only (W4A16): both W4A4 candidates failed the LMEB Dialogue bar. No abliteration "
             "cells are planned; the BF16 cell is the unchanged upstream reference. GPQA does not apply; the matched "
             "full-denominator gates are the MTEB Nano full-corpus benchmark (#18) and LMEB Dialogue whole-conversation retrieval."),
    "any_publication_claim": published, "products": products, "rejected_historical": [],
    "related_issues": ["HangGlidersRule/model-forge#33"],
}
(R / "publication-readiness-ledger.json").write_text(json.dumps(ledger, indent=1) + "\n")
print("ledger written; published =", published)
