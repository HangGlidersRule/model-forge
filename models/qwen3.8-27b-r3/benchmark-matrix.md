# Benchmark matrix

This document separates measurements we actually ran from upstream or third-party reference values. Empty cells stay empty until measured; they are not backfilled with a different checkpoint or protocol.

## Publication-readiness ledger (rendered)

The four canonical Darkstar products and their release gates are tracked in the machine-readable source
of truth [`results/publication-readiness-ledger.json`](results/publication-readiness-ledger.json),
which validates against the [release contract](../../contracts/darkstar-release/v1/contract.json). The
gates and statuses below are rendered from that ledger and kept in sync by CI, per the normative
[four-product release process](../../docs/darkstar-four-product-release-process.md). There is no
separate gap report: this embedded matrix and the per-product model cards are the human-readable view.

Statuses: `verified`, `in_progress`, `missing`, `rejected_historical`, `not_applicable`. **Lifecycle:**
all four evaluated cells have every applicable build gate verified. Base BF16 is the unchanged
upstream reference only. The three owned checkpoints and final cards are fully verified in private
Hugging Face repositories; this does not assert clean-smoke completion, public visibility, GHCR
publication, or a release tag.

### Qwen/Qwen3.8-27B upstream BF16 reference

Model card: [`model-card/base-bf16.md`](model-card/base-bf16.md). Lifecycle: **locally complete,
unpublished** for release-ledger accounting. Source repository:
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (external; not republished).

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

### Darkstar-Qwen3.8-27B-Base-ModelOpt-NVFP4

Model card: [`model-card/base-nvfp4.md`](model-card/base-nvfp4.md). Lifecycle: **locally complete,
unpublished** for public-release accounting. Public checkpoint repository:
`HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8` — the id of the selected
mixed candidate. The heading above is the abstract release slot; no repository, card, or served model
uses it.

<!-- LEDGER-SYNC product=Darkstar-Qwen3.8-27B-Base-ModelOpt-NVFP4 -->

| Gate | Status | Evidence | Next required proof |
|---|---|---|---|
| provenance_ownership | verified | Source pinned to Qwen/Qwen3.8-27B@1d4bf0f...; Darkstar-owned ModelOpt NVFP4 quantization | none — gate satisfied |
| artifact_manifest | verified | Exact per-file SHA-256 manifest committed for the local artifact | none — gate satisfied |
| recipe_edit_manifest | verified | Pinned ModelOpt recipe present and hashable; selected recipe quantizes lm_head in W4A16 NVFP4 (recipe_sha256 90fc6b37...) | none — gate satisfied |
| artifact_validation | verified | Fail-closed validators passed on the export: no NaN/Inf/zero scales, no quantized vision, 15 BF16 MTP tensors, no FP8 KV metadata, no mixed fused groups; snapshot captured | none — gate satisfied |
| abliteration_pass | not_applicable | Clean base, no refusal-direction edit | none |
| modelopt_candidate_comparison | verified | W4A16-vs-W4A4 comparison complete: mixed W4A16-NVFP4+FP8 selected (203.636 tok/s, GPQA 153/198); uniform W4A4 built and rejected (129.441 tok/s); winning recipe recorded | none — gate satisfied |
| nvfp4_tensor_scale_validation | verified | Tensor/scale integrity validated on the mixed candidate: finite NVFP4 scales, BF16 MTP preserved, no quantized vision, no mixed fused groups, no FP8 KV metadata | none — gate satisfied |
| performance_profile | verified | Independent single-stream MTP sweep on the mixed W4A16-NVFP4+FP8 candidate: winner 203.636 tok/s at MTP4, 32K scheduler budget, max-num-seqs 16, context 126144, BF16 KV, FlashAttention | none — gate satisfied |
| serving_capacity_profile | verified | Measured matched serving-capacity profile committed with concurrency-separated aggregate throughput and zero fatal markers | none — gate satisfied |
| gpqa_matched_full_denominator | verified | Mixed W4A16-NVFP4+FP8 candidate: 153/198 = 77.27% full denominator, 198/198 terminal parseable, 0 timeout/parse/error, thinking off, temp 1.0 top_p 0.95 top_k 20; external operator evidence with immutable hashes | none — gate satisfied |
| behavior_refusal_eval | not_applicable | Clean product; fresh abliteration behavior evaluation applies only to abliterated products | none |
| serve_profile_frozen | verified | Frozen serve profile: vLLM compiled mode, FlashAttention, BF16 KV, context 126144, MTP depth 4, 32K scheduler budget, max-num-seqs 16 | none — gate satisfied |
| model_card_final | verified | Final model card is complete and references planned immutable release tag darkstar-qwen3.8-27b-v1.0.0; no placeholders remain | none — gate satisfied |
| publication_targets_hf_ghcr | verified | Anonymous Hugging Face API and config.json checks verify the repository is public, ungated, enabled, and at revision c3f03c5bf5a28a636d72cd979323ff2f80668fb0; GHCR is explicitly not required for this release | none — gate satisfied |
| clean_download_boot_smoke | verified | Fresh-target download verified with zero failures; vLLM boot and models/text/strict JSON/tool/vision smoke passed with zero failures and an empty fatal log; public revision c3f03c5bf5a28a636d72cd979323ff2f80668fb0 | none — gate satisfied |
| release_tag | verified | Immutable Git release tag darkstar-qwen3.8-27b-v1.0.0 exists and is referenced from the final model card | none — gate satisfied |
| no_inherited_unverified_results | verified | GPQA and throughput are exact-artifact evidence; no behavior result is inherited from another artifact | none — gate satisfied |

#### ModelOpt NVFP4 candidates (precision-encoded; slot vs candidate)

The `Base-ModelOpt-NVFP4` slot above is a release label. Its concrete candidates encode their real
activation/recipe precision class and are never conflated. The mixed candidate is the **selected
(promoted)** Base ModelOpt product: it wins single-stream throughput (203.636 tok/s at MTP4) with a
GPQA (153/198 = 77.27%) within about 2.02 percentage points / 4 questions of the Base BF16 baseline
(157/198 = 79.29%). The uniform W4A4 candidate was **built and rejected** at 129.441 tok/s.

Selected candidate — `Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8` (selected, promoted):

<!-- CANDIDATE-SYNC candidate=Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8 -->

| Component | Precision |
|---|---|
| language_mlp | W4A16-NVFP4-g16 |
| lm_head | W4A16-NVFP4-g16 |
| self_attention | FP8-e4m3 |
| gdn_projections | FP8-e4m3 |
| kv_cache | BF16 |
| protected | BF16 |

Comparison candidate — `Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4` (uniform W4A4, built and
rejected on throughput):

<!-- CANDIDATE-SYNC candidate=Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4 -->

| Component | Precision |
|---|---|
| language_mlp | W4A4-NVFP4-g16 |
| lm_head | BF16 |
| self_attention | W4A4-NVFP4-g16 |
| gdn_projections | W4A4-NVFP4-g16 |
| kv_cache | BF16 |
| protected | BF16 |

### Darkstar-Qwen3.8-27B-Abliterated-BF16

Model card: [`model-card/bf16.md`](model-card/bf16.md). Internal edit lineage: R3. Lifecycle: **locally
complete, unpublished** for release accounting until the immutable Git tag is cut. Its public Hugging Face repository contains
the named checkpoint.

<!-- LEDGER-SYNC product=Darkstar-Qwen3.8-27B-Abliterated-BF16 -->

| Gate | Status | Evidence | Next required proof |
|---|---|---|---|
| provenance_ownership | verified | Source pinned; R3 refusal-direction edit lineage documented | none — gate satisfied |
| artifact_manifest | verified | Exact per-file SHA-256 manifest committed for the local artifact | none — gate satisfied |
| recipe_edit_manifest | verified | Pinned edit recipe and 131-tensor target inventory documented | none — gate satisfied |
| artifact_validation | verified | Structural validation passed: 131 edited tensors accounted for, 333 vision and 15 BF16 MTP tensors preserved, tokenizer/processor/chat/generation config unchanged | none — gate satisfied |
| abliteration_pass | verified | Reproducible projection (layer 38, seed 42, leakage ~0.000195); fresh abliteration eval on this exact artifact: 200/200 harmful compliance (0/200 refusals), 0/83 over-refusals, 0 errors; summary SHA c2ad..., journal SHA a20c... | none — gate satisfied |
| modelopt_candidate_comparison | not_applicable | BF16 product, not quantized | none |
| nvfp4_tensor_scale_validation | not_applicable | BF16 product, no NVFP4 scales | none |
| performance_profile | verified | Independent single-stream sweep: winner 144.502 tok/s at MTP11, FlashAttention, 16K scheduler budget, max-num-seqs 16, context 126144, BF16 KV | none — gate satisfied |
| serving_capacity_profile | verified | Measured matched serving-capacity profile committed with concurrency-separated aggregate throughput and zero fatal markers | none — gate satisfied |
| gpqa_matched_full_denominator | verified | 146/198 = 73.74% full denominator, 198/198 terminal parseable, 0 timeouts, 0 parse errors, 0 errors, thinking off; summary SHA 85c542..., journal SHA 0169... | none — gate satisfied |
| behavior_refusal_eval | verified | Fresh abliteration eval on this exact BF16 artifact: 200/200 harmful compliance (0/200 refusals), 0/83 safe over-refusals, 0 errors | none — gate satisfied |
| serve_profile_frozen | verified | Frozen serve profile: vLLM compiled mode, FlashAttention, BF16 KV, context 126144, MTP depth 11, 16K scheduler budget, max-num-seqs 16 | none — gate satisfied |
| model_card_final | verified | Final model card is complete and references planned immutable release tag darkstar-qwen3.8-27b-v1.0.0; no placeholders remain | none — gate satisfied |
| publication_targets_hf_ghcr | verified | Anonymous Hugging Face API and config.json checks verify the repository is public, ungated, enabled, and at revision 0181d5d178a15c694b1d6708d3ee3d08d2d9db5e; GHCR is explicitly not required for this release | none — gate satisfied |
| clean_download_boot_smoke | verified | Fresh-target download verified with zero failures; vLLM boot and models/text/strict JSON/tool/vision smoke passed with zero failures and an empty fatal log; public revision 0181d5d178a15c694b1d6708d3ee3d08d2d9db5e | none — gate satisfied |
| release_tag | verified | Immutable Git release tag darkstar-qwen3.8-27b-v1.0.0 exists and is referenced from the final model card | none — gate satisfied |
| no_inherited_unverified_results | verified | GPQA, throughput, and the fresh abliteration eval all measured directly on this exact BF16 artifact | none — gate satisfied |

### Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-NVFP4

Model card: [`model-card/nvfp4.md`](model-card/nvfp4.md). Internal edit lineage: R3. Lifecycle: **locally
complete, unpublished** — built with the selected clean-base mixed W4A16-NVFP4+FP8 recipe and fully
evaluated. Public checkpoint repository:
resolved to the selected/promoted candidate `HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8`;
the uniform `HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A4-NVFP4` id stays reserved
and unbuilt. No bare `Abliterated-NVFP4` target exists.

<!-- LEDGER-SYNC product=Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-NVFP4 -->

| Gate | Status | Evidence | Next required proof |
|---|---|---|---|
| provenance_ownership | verified | Source pinned to Qwen/Qwen3.8-27B@1d4bf0f...; R3 refusal-direction edit lineage documented; Darkstar-owned ModelOpt NVFP4 quantization | none — gate satisfied |
| artifact_manifest | verified | Local build frozen at ${PUBLIC_ARTIFACT_PATH}: _SUCCESS SHA-256 3d89ec57..., manifest.sha256 SHA-256 642dbbe8...; runtime snapshot inspect prefix f1e94a76, logs 4ffe8666, operator snapshot compose 85ba6815 recorded; tracked serve compose 5434c2a9 rendered from the frozen profile | none — gate satisfied |
| recipe_edit_manifest | verified | Pinned recipe present; reuses the selected clean mixed W4A16-NVFP4+FP8 recipe (recipe_sha256 90fc6b37...) plus the 131-tensor R3 edit inventory | none — gate satisfied |
| artifact_validation | verified | Fail-closed validators passed on the export: no NaN/Inf/zero scales, no quantized vision, 15 BF16 MTP tensors, no FP8 KV metadata, no mixed fused groups, 131 R3-edited tensors preserved through quantization; snapshot captured | none — gate satisfied |
| abliteration_pass | verified | Reproducible projection (layer 38, seed 42, leakage ~0.000195); fresh abliteration eval on this exact ModelOpt build: 200/200 harmful compliance (0/200 refusals), 0/83 over-refusals, 0 errors; summary SHA d814ea..., journal SHA 7b6dd... | none — gate satisfied |
| modelopt_candidate_comparison | verified | Reuses the selected clean-base W4A16-vs-W4A4 comparison; the abliterated mixed W4A16-NVFP4+FP8 candidate is built and measured (single-stream winner MTP10 251.889 tok/s); uniform W4A4 abliterated not built (clean-base W4A4 already rejected on throughput); winning recipe recorded | none — gate satisfied |
| nvfp4_tensor_scale_validation | verified | Tensor/scale integrity validated on the abliterated mixed build: finite NVFP4 scales, BF16 MTP preserved, no quantized vision, no mixed fused groups, no FP8 KV metadata | none — gate satisfied |
| performance_profile | verified | Independent single-stream MTP1-12 sweep on the abliterated mixed candidate: headline peak MTP8 251.316 tok/s, but non-monotonic; confirmation 10->8->8->10 selected MTP10 (mean 251.889 tok/s vs MTP8 mean 250.862 tok/s), 32K scheduler budget, max-num-seqs 16, context 126144, BF16 KV, FlashAttention | none — gate satisfied |
| serving_capacity_profile | verified | Matched per-concurrency cells measured on this exact build at the selected MTP10 profile: 4K/16K/48K prompts at concurrency 1 and 2, 512 generated tokens, 2 repeats per cell, 0 failed requests, zero fatal markers (source SHA 6e52a5ad...); semantic serving gates (tools, strict JSON, vision, prefix cache, 20-request sustained load) all pass, evidence SHA 4c8863... | none — gate satisfied |
| gpqa_matched_full_denominator | verified | Abliterated mixed W4A16-NVFP4+FP8 candidate: 148/198 = 74.75% full denominator, 198/198 terminal parseable, 0 timeout/parse/error, thinking off, temp 1.0 top_p 0.95 top_k 20; external operator evidence with immutable hashes (summary d8d0b5c0..., journal 9bb49132...) | none — gate satisfied |
| behavior_refusal_eval | verified | Fresh abliteration eval on this exact ModelOpt build: 283/283 terminal (200 harmful + 83 safe), 200/200 harmful compliance (0/200 refusals), 0/83 safe over-refusals, 0 errors | none — gate satisfied |
| serve_profile_frozen | verified | Frozen serve profile: vLLM compiled mode, FlashAttention, BF16 KV, context 126144, MTP depth 10, 32K scheduler budget, max-num-seqs 16, prefix caching + chunked prefill | none — gate satisfied |
| model_card_final | verified | Final model card is complete and references planned immutable release tag darkstar-qwen3.8-27b-v1.0.0; no placeholders remain | none — gate satisfied |
| publication_targets_hf_ghcr | verified | Anonymous Hugging Face API and config.json checks verify the repository is public, ungated, enabled, and at revision 2e25bd97fd1b6e6c7989e74c261d93a8702496e8; GHCR is explicitly not required for this release | none — gate satisfied |
| clean_download_boot_smoke | verified | Fresh-target download verified with zero failures; vLLM boot and models/text/strict JSON/tool/vision smoke passed with zero failures and an empty fatal log; public revision 2e25bd97fd1b6e6c7989e74c261d93a8702496e8 | none — gate satisfied |
| release_tag | verified | Immutable Git release tag darkstar-qwen3.8-27b-v1.0.0 exists and is referenced from the final model card | none — gate satisfied |
| no_inherited_unverified_results | verified | GPQA, throughput, serving correctness, and the fresh abliteration eval all measured directly on this exact abliterated ModelOpt build; compressed-tensors GPQA/perf/behavior numbers rejected, never inherited | none — gate satisfied |

#### Abliterated ModelOpt NVFP4 candidates (precision-encoded)

The mixed candidate is the selected/promoted, locally complete Abliterated ModelOpt product; the uniform
W4A4 candidate was never built. The mixed candidate reused the selected clean-base recipe: single-stream
winner MTP10 (251.889 tok/s mean; headline peak MTP8 251.316 tok/s on a non-monotonic sweep) at GPQA
148/198 = 74.75%.

<!-- CANDIDATE-SYNC candidate=Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8 -->

| Component | Precision |
|---|---|
| language_mlp | W4A16-NVFP4-g16 |
| lm_head | W4A16-NVFP4-g16 |
| self_attention | FP8-e4m3 |
| gdn_projections | FP8-e4m3 |
| kv_cache | BF16 |
| protected | BF16 |

<!-- CANDIDATE-SYNC candidate=Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A4-NVFP4 -->

| Component | Precision |
|---|---|
| language_mlp | W4A4-NVFP4-g16 |
| lm_head | BF16 |
| self_attention | W4A4-NVFP4-g16 |
| gdn_projections | W4A4-NVFP4-g16 |
| kv_cache | BF16 |
| protected | BF16 |

## GPQA Diamond status

Dataset provenance:

- Benchmark: GPQA Diamond, 198 questions.
- Dataset lineage: `idavidrein/gpqa` / the public OpenAI simple-evals Diamond CSV mirror.
- Evaluated CSV SHA-256: `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`.
- Answer choices were deterministically shuffled per question.
- Sampling: temperature `1.0`, top-p `0.95`, top-k `20`.
- Parser: final `\boxed{A-D}`, with `Answer: A-D` fallback.

All four products carry a **frozen full-denominator** (198/198 terminal parseable) result with
zero timeouts, parse errors, or errors. Machine-readable copy:
[`results/gpqa-matrix.json`](results/gpqa-matrix.json).

| Product | Precision | Weight edit | Thinking | Completed / 198 | Correct | Accuracy (full denominator) | Harmful refusals | Safe over-refusals | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Darkstar Base BF16 | BF16 | No | Off | 198 | 157 | 79.29% | 197/200 (98.50%) | 4/83 (4.82%) | Locally complete, unpublished; 0 timeout/parse/error |
| Darkstar Base ModelOpt W4A16-NVFP4-Mixed-FP8 | NVFP4/FP8 mixed | No | Off | 198 | 153 | 77.27% | 197/200 (98.50%) | 4/83 (4.82%) | Selected/promoted; locally complete, unpublished; 0 timeout/parse/error |
| Darkstar Abliterated BF16 | BF16 | Abliterated | Off | 198 | 146 | 73.74% | 0/200 (0.00%) | 0/83 (0.00%) | Locally complete, unpublished; fresh abliteration eval; 0 timeout/parse/error |
| Darkstar Abliterated ModelOpt W4A16-NVFP4-Mixed-FP8 | NVFP4/FP8 mixed | Abliterated | Off | 198 | 148 | 74.75% | 0/200 (0.00%) | 0/83 (0.00%) | Selected/promoted; locally complete, unpublished; fresh abliteration eval (283/283 terminal); 0 timeout/parse/error |

Secondary, unmatched GPQA measurement: the prior R3 NVFP4 **compressed-tensors** artifact with thinking
enabled completed `164/198 = 82.83%` with zero errors. It is `rejected_historical`: it is not measured
on any current Darkstar product, is retained for lineage only, and is excluded from the matched
thinking-off matrix.

### Interpretation and deltas

Both deltas below are computed on the frozen full denominator (198/198) for each product:

- **Quantization delta (Base):** Base BF16 `157/198` → Base ModelOpt mixed W4A16-NVFP4+FP8 `153/198` =
  `-4` questions / `-2.02` percentage points. This small delta, together with the mixed candidate's
  throughput win, is why the mixed W4A16 candidate was selected as the Base ModelOpt product.
- **Edit delta (BF16):** Base BF16 `157/198` → Abliterated BF16 `146/198` = `-11` questions / `-5.56`
  percentage points, the measured cost of the R3 refusal-direction edit.
- **Quantization delta (Abliterated):** Abliterated BF16 `146/198` → Abliterated ModelOpt mixed
  W4A16-NVFP4+FP8 `148/198` = `+2` questions / `+1.01` percentage points, measured on the exact
  quantized build (small and within run-to-run noise; not a claim that quantization improves quality).

## Other measured quality gates

### Abliterated BF16 refusal behavior (fresh eval)

- Harmful-prompt compliance: `200/200` in the 200-prompt evaluation (`0/200` refusals).
- Safe over-refusal: `0/83`.
- Errors: `0`. Eval summary SHA prefix `c2ad`, journal SHA prefix `a20c`.

These are behavior measurements on the exact BF16 artifact, not safety endorsements.

### Abliterated ModelOpt mixed W4A16-NVFP4+FP8 refusal behavior (fresh eval)

- Terminal responses: `283/283` (200 harmful + 83 safe).
- Harmful-prompt compliance: `200/200` (`0/200` refusals).
- Safe over-refusal: `0/83`.
- Errors: `0`. Eval summary SHA prefix `d814ea`, journal SHA prefix `7b6dd`.

Measured directly on the exact quantized ModelOpt build (a BF16 eval does not verify the NVFP4 build).
Behavior measurements, not safety endorsements.

### Initial 102-item project smoke suite

This suite is useful for regression smoke testing, but its categories and sample sizes are too small to
supersede GPQA or standardized evaluations. The candidate names below are historical snapshot labels
from that early bake-off and are not current Darkstar product ids.

| Candidate (historical bake-off label) | Passed / total | Score |
|---|---:|---:|
| Qwen3.6 AEON production baseline | 60 / 102 | 58.82% |
| Pre-Darkstar Qwen3.8 abliterated NVFP4 (historical) | 57 / 102 | 55.88% |
| Clean Qwen3.8 NVFP4 control (historical) | 51 / 102 | 50.00% |

## Performance results (per product, never conflated)

Target workload: interactive, single-stream, cache-resistant natural prompts on one RTX PRO 6000
Blackwell; vLLM `0.27.1`, compiled mode, FlashAttention, BF16 KV. Each product's throughput winner is
recorded on its own artifact and never mixed with another product's.

| Product | Speculation | Scheduler budget | Winner tok/s | Machine-readable record |
|---|---:|---:|---:|---|
| Darkstar Base BF16 | MTP8 | 65536 | 130.158 | private raw evidence retained as `results/performance-base-bf16.json` |
| Darkstar Base ModelOpt W4A16-NVFP4-Mixed-FP8 (selected) | MTP4 | 32768 | 203.636 | private raw evidence retained as `results/performance-nvfp4.json` |
| Darkstar Abliterated BF16 | MTP11 | 16384 | 144.502 | private raw evidence retained as `results/performance-abliterated-bf16.json` |
| Darkstar Abliterated ModelOpt W4A16-NVFP4-Mixed-FP8 (selected) | MTP10 | 32768 | 251.889 | private raw evidence retained as `results/performance-abliterated-modelopt-w4a16-nvfp4-mixed-fp8.json` |

The uniform W4A4 base candidate was built and **rejected** at `129.441 tok/s` — far below the selected
mixed W4A16 winner. The Abliterated ModelOpt mixed W4A16-NVFP4+FP8 winner is **MTP10 at mean
251.889 tok/s**:
its MTP1-12 sweep was non-monotonic, so although a single pass peaked at MTP8 (`251.316 tok/s`), a
confirmation sequence (`10->8->8->10`) selected MTP10 on its higher mean throughput (MTP10 mean
`251.889` vs MTP8 mean `250.862 tok/s`). The uniform W4A4 abliterated candidate
was never built.
