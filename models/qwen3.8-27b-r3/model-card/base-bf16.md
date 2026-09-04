---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
  - upstream-control
  - bf16
  - qwen3.8
---

# Qwen/Qwen3.8-27B upstream BF16 control

> **External upstream control, not a Darkstar product.** This page records Model Forge's evaluation
> profile for the unchanged Apache-2.0 `Qwen/Qwen3.8-27B` weights at revision
> `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Weights: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). No HangGlidersRule weight
repository exists or is planned for these unchanged weights; HangGlidersRule does not own or
republish them.

## Summary

This is the upstream-control cell for quantization and refusal-direction deltas measured across the
family: pinned source revision, no weight edit, and no quantization. The measured profile carries a
frozen full-denominator GPQA result and an independent single-stream throughput result.

## Provenance

- Upstream model: `Qwen/Qwen3.8-27B`
- Upstream revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Weight edit: none (clean base)
- Quantization: none (BF16)
- Evaluation recipe: [`recipes/qwen3.8-27b/darkstar-qwen3.8-27b-base-bf16.yaml`](../../../recipes/qwen3.8-27b/darkstar-qwen3.8-27b-base-bf16.yaml)
- Engineering repository: [`HangGlidersRule/model-forge`](https://github.com/HangGlidersRule/model-forge)
- Lineage detail: [`artifact-lineage.md`](../artifact-lineage.md)

## Evaluation

Curated aggregates: [`../results/gpqa-matrix.json`](../results/gpqa-matrix.json). Protocol:
[`../gpqa-protocol.md`](../gpqa-protocol.md). Full caveats: [`../benchmark-matrix.md`](../benchmark-matrix.md).

| Metric | Value | Basis |
|---|---|---|
| GPQA Diamond (thinking off) | 157/198 = 79.29% | full denominator, 198/198 terminal parseable, 0 timeout/parse/error |
| Harmful-prompt refusals | 197/200 (98.50%) | clean unedited base behavior, 0 errors |
| Safe over-refusals | 4/83 (4.82%) | over-refusal suite, 0 errors |
| Single-stream throughput (MTP8) | 130.158 tok/s | independent single-stream sweep; see [`../benchmark-matrix.md`](../benchmark-matrix.md) |

This is the baseline against which the family deltas are computed: the Base ModelOpt mixed
W4A16-NVFP4+FP8 quantization delta is `-4` questions / `-2.02` pp, and the Abliterated BF16 edit delta
is `-11` questions / `-5.56` pp. Missing cells are never backfilled from a different checkpoint or
protocol.

## Runtime requirements and example

- Runtime: vLLM `0.27.1`, compiled mode, Flash Attention, BF16 KV cache, context up to 126,144.
- Serve profile: MTP depth 8, 64K scheduler budget, `max_num_seqs=16`.
- Serve image: not published

```bash
vllm serve Qwen/Qwen3.8-27B \
  --served-model-name darkstar-qwen38-base-bf16 \
  --kv-cache-dtype bf16 \
  --max-model-len 126144 \
  --enable-chunked-prefill \
  --compilation-config 2 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 8}'
```

See [`containers/serve/docker-compose.yml`](../../../containers/serve/docker-compose.yml) for the
generic Compose runtime.

## Publication-readiness (rendered from the ledger)

This legacy evaluation-slot block is rendered from the machine-readable source of truth
[`../results/publication-readiness-ledger.json`](../results/publication-readiness-ledger.json) and kept
in sync by CI. The [benchmark matrix](../benchmark-matrix.md) shows all four products together. This
cell is complete. Its publication statuses do not authorize republishing the upstream weights.

<!-- LEDGER-SYNC product=Darkstar-Qwen3.8-27B-Base-BF16 -->

| Gate | Status | Evidence | Next required proof |
|---|---|---|---|
| provenance_ownership | verified | Source pinned to Qwen/Qwen3.8-27B@1d4bf0f...; upstream ownership and unchanged-reference status recorded; no HangGlidersRule weight repository exists or is planned | none — gate satisfied |
| artifact_manifest | verified | Exact per-file SHA-256 manifest committed for the local artifact | none — gate satisfied |
| recipe_edit_manifest | verified | Pinned Base BF16 recipe present and hashable (clean source, no transform, no quantization) | none — gate satisfied |
| artifact_validation | verified | Structural validation passed; tokenizer/processor/chat/generation config preserved; 333 vision and 15 MTP tensors intact; recorded in the runtime snapshot | none — gate satisfied |
| abliteration_pass | not_applicable | Clean base, no refusal-direction edit | none |
| modelopt_candidate_comparison | not_applicable | BF16 product, not quantized | none |
| nvfp4_tensor_scale_validation | not_applicable | BF16 product, no NVFP4 scales | none |
| performance_profile | verified | Independent single-stream sweep: winner 130.158 tok/s at MTP8, FlashAttention, 64K scheduler budget, max-num-seqs 16, context 126144, BF16 KV | none — gate satisfied |
| serving_capacity_profile | verified | Measured matched serving-capacity profile committed with concurrency-separated aggregate throughput and zero fatal markers | none — gate satisfied |
| gpqa_matched_full_denominator | verified | 157/198 = 79.29% full denominator, 198/198 terminal parseable, 0 timeouts, 0 parse errors, 0 errors, thinking off | none — gate satisfied |
| behavior_refusal_eval | not_applicable | Clean product; fresh abliteration behavior evaluation applies only to abliterated products | none |
| serve_profile_frozen | verified | Frozen serve profile: vLLM compiled mode, FlashAttention, BF16 KV, context 126144, MTP depth 8, 64K scheduler budget, max-num-seqs 16 | none — gate satisfied |
| model_card_final | verified | Final model card is complete and references planned immutable release tag darkstar-qwen3.8-27b-v1.0.0; no placeholders remain | none — gate satisfied |
| publication_targets_hf_ghcr | not_applicable | Unchanged upstream-only control: no owned Hugging Face or GHCR publication target exists | none |
| clean_download_boot_smoke | not_applicable | No owned checkpoint is published or re-downloaded for the upstream-only control | none |
| release_tag | verified | Immutable Git release tag darkstar-qwen3.8-27b-v1.0.0 exists and is referenced from the final model card | none — gate satisfied |
| no_inherited_unverified_results | verified | GPQA and throughput are exact-artifact evidence; no behavior result is inherited from another artifact | none — gate satisfied |

## License and attribution

- Upstream license: Apache-2.0.
- Preserve upstream attribution and required notices. HangGlidersRule does not redistribute the
  unchanged weights.
