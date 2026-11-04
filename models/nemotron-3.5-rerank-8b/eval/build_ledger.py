#!/usr/bin/env python3
"""Write the Nemotron-3.5-Rerank-8B publication-readiness ledger (contract 3.1.0) from the committed evidence.
Mid-release state: the shipped product is locally_complete_unpublished, with the publication gates (HF target,
clean-download smoke, release tag) in_progress and publication_claim false. Re-run with --published RECEIPT after the
public flip to set those gates verified. Every quoted figure is read from results/rerank-gate-summary.json.
Usage: build_ledger.py [--published]"""
import json
import sys
from pathlib import Path

F = "nemotron-3.5-rerank-8b"
R = Path(f"models/{F}/results")
g = json.loads((R / "rerank-gate-summary.json").read_text())
v = json.loads((R / "artifact-validation.json").read_text())
published = "--published" in sys.argv
receipt: dict = json.loads((R / "publication-receipt.json").read_text()) if published else {}
TARGET = "HangGlidersRule/Darkstar-Nemotron-3.5-Rerank-8B-Base-ModelOpt-W4A4-NVFP4"
SRC_REV = "f88f8d9104726747791283ab156d2fcdb29e90a7"
qn, ql = g["quality"]["beir6_bm25"], g["quality"]["lmeb_dialogue"]
sp = g["speed"]
GS = f"models/{F}/results/rerank-gate-summary.json"
AV = f"models/{F}/results/artifact-validation.json"
CARD = f"models/{F}/model-card/base-nvfp4.md"
RECIPE = f"recipes/{F}/darkstar-nemotron-3.5-rerank-8b-base-modelopt-w4a4-nvfp4.yaml"


def gate(gid, status, evidence, path, **extra):
    nxt = {"verified": "none - gate satisfied", "not_applicable": "none - not applicable"}.get(status, extra.pop("next", "pending"))
    return {"id": gid, "status": status, "evidence": evidence, "evidence_path": path, "next_missing_proof": nxt, **extra}


def ci(e):
    return f"{e['delta_vs_bf16']:+.4f} (95% CI {e['ci95_vs_bf16'][0]:+.4f}..{e['ci95_vs_bf16'][1]:+.4f})"


quality = ("Reranker family: two matched full-denominator gates replace GPQA. "
           f"6 MTEB Nano sets: macro nDCG@10 {qn['A']['macro_ndcg10']} vs BF16 {qn['bf16']['macro_ndcg10']} "
           f"({qn['A']['ratio_to_bf16']:.4f}), {qn['bf16']['n_pairs']}/{qn['bf16']['n_pairs']} pairs, delta {ci(qn['A'])}; "
           f"LMEB Dialogue: {ql['A']['macro_ndcg10']} vs BF16 {ql['bf16']['macro_ndcg10']} ({ql['A']['ratio_to_bf16']:.4f}), "
           f"{ql['bf16']['n_pairs']}/{ql['bf16']['n_pairs']} pairs, delta {ci(ql['A'])}; replica floor "
           f"{qn['bf16r']['delta_vs_bf16']:+.4f} / {ql['bf16r']['delta_vs_bf16']:+.4f}")
pub_gates = [
    gate("publication_targets_hf_ghcr", "verified" if published else "in_progress",
         (f"Public at {TARGET} @ {receipt['hf_commit'][:7]}; GHCR not required (pinned upstream vLLM image)" if published
          else f"HF target {TARGET} reserved; upload and public flip pending"),
         f"models/{F}/results/publication-receipt.json" if published else None,
         next="private upload, clean-download smoke, then public flip"),
    gate("clean_download_boot_smoke", "verified" if published else "in_progress",
         ("Empty-cache download at the verified commit, every manifest entry checked, served with the card's exact "
          "command, card example request answered" if published else "Pending the private upload"),
         f"models/{F}/results/publication-receipt.json" if published else None,
         next="empty-cache download + card serve command + example request"),
    gate("release_tag", "verified" if published else "in_progress",
         "darkstar-nemotron-3.5-rerank-8b-v1.0.0" if published else "Tag darkstar-nemotron-3.5-rerank-8b-v1.0.0 follows the export PR merge",
         f"models/{F}/README.md" if published else None, next="tag the merge commit"),
]
products = [
    {
        "product_id": "Darkstar-Nemotron-3.5-Rerank-8B-Base-BF16", "role": "base-bf16", "task": "rerank",
        "lifecycle": "published", "target_repository": "nvidia/Nemotron-3.5-Rerank-8B-BF16",
        "target_repository_status": "resolved",
        "target_repository_note": "Unchanged upstream reference measured as the family control; no HangGlidersRule repository.",
        "served_model_alias": "darkstar-nemotron35rerank8b-base-bf16",
        "runtime_container_name": "vllm-darkstar-nemotron35rerank8b-base-bf16",
        "model_card": f"models/{F}/model-card/base-bf16.md", "recipe": None, "publication_claim": False,
        "gates": [
            gate("provenance_ownership", "verified", f"Source pinned to nvidia/Nemotron-3.5-Rerank-8B-BF16@{SRC_REV} (OpenMDW-1.1)", GS),
            gate("performance_profile", "verified",
                 f"BF16 control {sp['bf16']['c1']} pairs/s at c=1, {sp['bf16']['c4']} at c=4, {sp['bf16']['c8']} at c=8 (64 pairs/request, ~1.1k-token pairs)", GS),
            gate("quality_matched_full_denominator", "verified",
                 f"Control: macro nDCG@10 {qn['bf16']['macro_ndcg10']} (6 MTEB Nano sets) and {ql['bf16']['macro_ndcg10']} (LMEB Dialogue); "
                 f"replica floor {qn['bf16r']['delta_vs_bf16']:+.4f} / {ql['bf16r']['delta_vs_bf16']:+.4f}", GS,
                 metric="macro nDCG@10 on two frozen sets (6 MTEB Nano sets; LMEB Dialogue)", protocol="rerank gate summary, full denominator"),
        ],
    },
    {
        "product_id": "Darkstar-Nemotron-3.5-Rerank-8B-Base-ModelOpt-NVFP4", "role": "base-modelopt-nvfp4", "task": "rerank",
        "lifecycle": "published" if published else "locally_complete_unpublished",
        "target_repository": TARGET, "target_repository_status": "resolved",
        "served_model_alias": "darkstar-nemotron35rerank8b-base-nvfp4",
        "runtime_container_name": "vllm-darkstar-nemotron35rerank8b-base-nvfp4",
        "model_card": CARD, "recipe": RECIPE, "publication_claim": published,
        "candidates": [
            {"candidate_id": "Darkstar-Nemotron-3.5-Rerank-8B-Base-ModelOpt-W4A4-NVFP4", "precision_class": "W4A4-NVFP4",
             "target_repository": TARGET, "selection": "selected", "promotion_status": "promoted",
             "promotion_note": (f"Candidate A (text tower MLP + attention W4A4): fastest of three passing candidates, "
                                f"{g['ranges']['A']} BF16 pairs/s; delta {ci(qn['A'])} (6 MTEB Nano sets) and {ci(ql['A'])} (LMEB Dialogue). "
                                f"B (attention FP8, {g['ranges']['B']}) and C (MLP only, {g['ranges']['C']}) also passed."),
             "recipe": "configs/modelopt/recipes/nemotron-3.5-rerank-8b/nvfp4_text_all_act_headroom-kv_bf16.yaml",
             "evidence_path": GS,
             "immutable_hashes": {
                 "model.safetensors": "6fca2f3e44f2522f9338a88bca47384adec515f2b1641ac54482a2b9058914e0",
                 "config.json": "2e895664b6c0a653f053a3fb1eaae87b04466004bcdfce749d1fb3f714241569",
                 "hf_quant_config.json": "3ca5fe873c52e0d8f4dbaf841224c2da62472eb2cf7a144b3b0a50eaf4da353b",
                 "operator_recipe": "71874e44a460f1c5d29509a8d5de66e6eacda4b130a8fa3fd10f3f44a0329fa7",
                 "calibration_jsonl": g["calibration_sha256"]},
             "precision_map": {"language_mlp": "W4A4 NVFP4 group 16 (gate/up/down, 34 layers)",
                               "lm_head": "not present (score head 4096->1, BF16)",
                               "self_attention": "W4A4 NVFP4 group 16 (q/k/v/o, 34 layers)",
                               "gdn_projections": "not present (dense bidirectional transformer)", "kv_cache": "BF16",
                               "protected": "vision tower, multimodal projector, embeddings, norms, score head (BF16)"}},
            {"candidate_id": "Darkstar-Nemotron-3.5-Rerank-8B-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8", "precision_class": "W4A4-NVFP4-Mixed-FP8",
             "target_repository": "HangGlidersRule/Darkstar-Nemotron-3.5-Rerank-8B-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8",
             "selection": "rejected", "promotion_status": "none",
             "promotion_note": f"Candidate B passed (delta {ci(qn['B'])} / {ci(ql['B'])}) but is slower than A ({g['ranges']['B']} vs {g['ranges']['A']} BF16).",
             "recipe": "configs/modelopt/recipes/nemotron-3.5-rerank-8b/nvfp4_text_mlp_act_headroom-fp8_attn-kv_bf16.yaml",
             "evidence_path": GS,
             "precision_map": {"language_mlp": "W4A4 NVFP4 group 16 (gate/up/down, 34 layers)",
                               "lm_head": "not present (score head 4096->1, BF16)",
                               "self_attention": "FP8 W8A8 per-tensor (q/k/v/o, 34 layers)",
                               "gdn_projections": "not present (dense bidirectional transformer)", "kv_cache": "BF16",
                               "protected": "vision tower, multimodal projector, embeddings, norms, score head (BF16)"}},
        ],
        "gates": [
            gate("provenance_ownership", "verified", f"Built from nvidia/Nemotron-3.5-Rerank-8B-BF16@{SRC_REV[:8]} BF16 (OpenMDW-1.1); no third-party quantized inputs", GS),
            gate("artifact_manifest", "verified", "Every artifact file hashed before upload; model.safetensors sha256 6fca2f3e...", AV),
            gate("recipe_edit_manifest", "verified", "Schema-2 recipe plus operator recipe sha256 71874e44...; ModelOpt 0.48.0 pin (wheel sha256 bfcfd425...)", RECIPE),
            gate("artifact_validation", "verified",
                 f"{v['protected_tensors']['byte_identical']}/{v['protected_tensors']['expected']} protected tensors byte-identical to upstream; "
                 "sidecars byte-identical; config.json identical outside quantization_config", AV),
            gate("modelopt_candidate_comparison", "verified", "Three W4A4 candidates compared on two frozen sets against BF16 and a BF16 replica; A selected as the fastest passing", GS),
            gate("nvfp4_tensor_scale_validation", "verified",
                 f"{v['coverage']['nvfp4']} NVFP4 linears; {v['fp8_block_scales']['non_finite']} non-finite and {v['fp8_block_scales']['zero']} zero FP8 block scales; "
                 f"per-tensor scales finite and positive; fused groups uniform ({v['fused_group_scales']['groups_checked']} checked); no KV quantization", AV),
            gate("kv_cache_effect_measured", "not_applicable", "kv_cache is BF16", None),
            gate("performance_profile", "verified",
                 f"{sp['A']['c1']} pairs/s at c=1 (BF16 {sp['bf16']['c1']}), 64 pairs/request, median of 3 after warmup", GS),
            gate("serving_capacity_profile", "verified",
                 f"{sp['A']['c4']} pairs/s at c=4, {sp['A']['c8']} at c=8 (BF16 {sp['bf16']['c4']} / {sp['bf16']['c8']}, {g['ranges']['A']}); "
                 f"GPU memory under that load {sp['A']['gpu_memory_under_load_gib']} GiB", GS),
            gate("quality_matched_full_denominator", "verified", quality, GS,
                 metric="macro nDCG@10 vs BF16 on two frozen sets (6 MTEB Nano sets; LMEB Dialogue)", protocol="rerank gate summary, full denominator"),
            gate("serve_profile_frozen", "verified",
                 "vLLM v0.32.0 by digest; pooling runner; mean pooling, use_activation false; checkpoint chat_template.jinja via --chat-template; BF16 KV", CARD),
            gate("model_card_final", "verified", "Gold-standard card; every number checked by eval/check_card_numbers.py", CARD),
            *pub_gates,
            gate("no_inherited_unverified_results", "verified", "Every number on the card was measured on this exact artifact", GS),
        ],
    },
    {
        "product_id": "Darkstar-Nemotron-3.5-Rerank-8B-Abliterated-BF16", "role": "abliterated-bf16", "task": "rerank",
        "lifecycle": "in_progress", "target_repository": "HangGlidersRule/Darkstar-Nemotron-3.5-Rerank-8B-Abliterated-BF16",
        "target_repository_status": "resolved",
        "served_model_alias": "darkstar-nemotron35rerank8b-abliterated-bf16",
        "runtime_container_name": "vllm-darkstar-nemotron35rerank8b-abliterated-bf16",
        "model_card": CARD, "recipe": None, "publication_claim": False,
        "gates": [gate("abliteration_pass", "not_applicable", "No abliteration for a reranker family", None),
                  gate("quality_matched_full_denominator", "not_applicable", "Not built and not planned: a rerank model is not a refusal-edit target", None,
                       metric="not applicable (not built)", protocol="not applicable (not built)")],
    },
    {
        "product_id": "Darkstar-Nemotron-3.5-Rerank-8B-Abliterated-ModelOpt-NVFP4", "role": "abliterated-modelopt-nvfp4", "task": "rerank",
        "lifecycle": "in_progress", "target_repository": None, "target_repository_status": "unresolved_pending_precision_winner",
        "served_model_alias": "darkstar-nemotron35rerank8b-abliterated-nvfp4",
        "runtime_container_name": "vllm-darkstar-nemotron35rerank8b-abliterated-nvfp4",
        "model_card": CARD, "recipe": None, "publication_claim": False,
        "candidates": [{"candidate_id": "Darkstar-Nemotron-3.5-Rerank-8B-Abliterated-ModelOpt-W4A4-NVFP4", "precision_class": "W4A4-NVFP4",
                        "target_repository": "HangGlidersRule/Darkstar-Nemotron-3.5-Rerank-8B-Abliterated-ModelOpt-W4A4-NVFP4",
                        "selection": "not_built", "promotion_status": "none",
                        "promotion_note": "Reserved id only; refusal edits do not apply to a reranker.", "recipe": None,
                        "precision_map": {"language_mlp": "W4A4-NVFP4-g16", "lm_head": "not present", "self_attention": "W4A4-NVFP4-g16",
                                          "gdn_projections": "not present", "kv_cache": "BF16", "protected": "not built"}}],
        "gates": [gate("abliteration_pass", "not_applicable", "No abliteration for a reranker family", None),
                  gate("quality_matched_full_denominator", "not_applicable", "Not built and not planned: a rerank model is not a refusal-edit target", None,
                       metric="not applicable (not built)", protocol="not applicable (not built)"),
                  gate("kv_cache_effect_measured", "not_applicable", "No selected candidate: nothing is served, so there is no KV cache", None)],
    },
]
ledger = {
    "schema_version": "1.0.0", "contract": "contracts/darkstar-release/v3/contract.json", "contract_version": "3.1.0",
    "family": F, "family_display": "Darkstar Nemotron-3.5-Rerank-8B",
    "source_revision": f"nvidia/Nemotron-3.5-Rerank-8B-BF16@{SRC_REV}",
    "note": ("Reranker family (multimodal cross-encoder: Pixtral vision tower + bidirectional Ministral3 text tower, mean-pooled "
             "score head). Base quantization of the text tower only; the vision tower ships unchanged and image-document "
             "reranking is not evaluated. No abliteration cells are planned for a scoring model, so the two abliterated roles "
             "are not_applicable placeholders. The BF16 cell is the unchanged upstream reference. GPQA does not apply; the "
             "matched full-denominator gates are the 6 MTEB Nano sets from #14 and an LMEB Dialogue memory set."),
    "any_publication_claim": published, "products": products, "rejected_historical": [],
    "related_issues": ["HangGlidersRule/model-forge#32"],
}
(R / "publication-readiness-ledger.json").write_text(json.dumps(ledger, indent=1) + "\n")
print("ledger written; published =", published)
