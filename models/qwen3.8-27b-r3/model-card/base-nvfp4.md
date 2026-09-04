---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - darkstar
  - base
  - nvfp4
  - modelopt
  - vllm
  - qwen3.8
---

# Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8

> Prior llm-compressor compressed-tensors builds of this name are **rejected/historical**.
> Publication builds use NVIDIA ModelOpt `0.46.0rc2`.

Private checkpoint repository:
`HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8`.

This card describes the **selected** Base ModelOpt product, and its id encodes that artifact's real
precision class. `Darkstar-Qwen3.8-27B-Base-ModelOpt-NVFP4` is the abstract release slot in the ledger
and is never used as a repository or model-card identity. The separate uniform
`Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4` candidate was built and rejected on throughput; it
keeps its own reserved repository id for lineage.

## Summary

NVIDIA ModelOpt NVFP4 quantization of the official `Qwen/Qwen3.8-27B` checkpoint with no weight edit.
It is the clean Base artifact in the Darkstar Qwen3.8 family and the control for the abliterated
derivatives: same source revision, same quantization contract, no refusal-direction projection. R3
identifies only the abliterated edit.

`Darkstar-Qwen3.8-27B-Base-ModelOpt-NVFP4` is the family release **slot**. The concrete candidates that
fill it are named for their real precision class and must never be conflated:

- **`Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8`** — the **selected (promoted)** Base
  ModelOpt product. It is **mixed precision**: W4A16 NVFP4 (group 16) on the language MLP
  gate/up/down and `lm_head`, FP8 on the self-attention q/k/v/o and the large GatedDeltaNet
  (linear-attn) projections, and BF16 for protected components and runtime KV. It was selected for its
  single-stream throughput (203.636 tok/s, MTP4) at a GPQA of 153/198 = 77.27%, within 2.02 pp / 4
  questions of the Base BF16 baseline.
- **`Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4`** — a separate uniform W4A4 NVFP4 candidate that
  was built and measured for the W4A16-vs-W4A4 comparison and **rejected** at 129.441 tok/s.

### Candidate precision maps

<!-- CANDIDATE-SYNC candidate=Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8 -->

| Component | Precision |
|---|---|
| language_mlp | W4A16-NVFP4-g16 |
| lm_head | W4A16-NVFP4-g16 |
| self_attention | FP8-e4m3 |
| gdn_projections | FP8-e4m3 |
| kv_cache | BF16 |
| protected | BF16 |

<!-- CANDIDATE-SYNC candidate=Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4 -->

| Component | Precision |
|---|---|
| language_mlp | W4A4-NVFP4-g16 |
| lm_head | BF16 |
| self_attention | W4A4-NVFP4-g16 |
| gdn_projections | W4A4-NVFP4-g16 |
| kv_cache | BF16 |
| protected | BF16 |

## Provenance

- Upstream model: `Qwen/Qwen3.8-27B`
- Upstream revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Weight edit: none (clean baseline)
- Exact recipe: [`recipes/qwen3.8-27b/darkstar-qwen3.8-27b-base-modelopt-nvfp4.yaml`](../../../recipes/qwen3.8-27b/darkstar-qwen3.8-27b-base-modelopt-nvfp4.yaml)
- Exact selected operator recipe (tracked): [`configs/modelopt/recipes/w4a16_nvfp4_mse-fp8_attn-kv_bf16.yaml`](../../../configs/modelopt/recipes/w4a16_nvfp4_mse-fp8_attn-kv_bf16.yaml), SHA-256 `90fc6b37c00334debd49f1975ab406b5e20667f07e4be0be3e463a648abac642`. It quantizes `lm_head` in W4A16 NVFP4, matching the product precision map above.
- Engineering repository: [`HangGlidersRule/model-forge`](https://github.com/HangGlidersRule/model-forge)
- Lineage detail: [`artifact-lineage.md`](../artifact-lineage.md)
- ModelOpt pin: [`modelopt/README.md`](../modelopt/README.md)

## Quantization summary

This summary describes the selected mixed candidate
`Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8` (see the precision maps above). It is **not**
uniform W4A4.

- Format: NVIDIA ModelOpt unified HF NVFP4/FP8 (`hf_quant_config.json`), **mixed precision**
- W4A16 NVFP4 (group 16): language MLP `gate_proj`/`up_proj`/`down_proj` and `lm_head`
- FP8 (e4m3): self-attention `q`/`k`/`v`/`o` and the large GatedDeltaNet (linear-attn) projections
- Algorithm: MSE FP8-scale sweep, static NVFP4 MLP weights
- Kept BF16: vision, MTP, `conv1d`, norms, embeddings, and runtime KV (no FP8 KV metadata)
- Calibration: `cnn_dailymail` + `nemotron-post-training-dataset-v2`, 512+512, seq 2048, seed 1234
- Vision tower and MTP tensors preserved from the source (15 BF16 MTP tensors)
- The separate `Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4` candidate (uniform W4A4) was built and rejected on throughput.

## Intended use

- Interactive and batch text generation where NVFP4 throughput on Blackwell-class GPUs matters.
- Baseline/control for measuring the effect of the R3 refusal-direction edit and of quantization.

## Safety note

This is the **unedited** baseline. Its refusal behavior reflects upstream `Qwen/Qwen3.8-27B`. The
measured harmful-prompt refusal rate below is a behavior observation, not a safety endorsement, and
NVFP4 quantization can shift behavior relative to the BF16 source.

## Limitations

- GPQA Diamond for this clean/base cell is measured on the selected mixed
  `W4A16-NVFP4-Mixed-FP8` candidate: 153/198 = 77.27%, full denominator. Do not substitute upstream
  Qwen's reported figure for this cell.
- Quantization can cause small quality regressions not captured by the smoke suite.
- Only single-stream throughput is characterized; aggregate concurrent throughput is separate.

## Evaluation

Curated aggregates: [`../results/gpqa-matrix.json`](../results/gpqa-matrix.json). Protocol:
[`../gpqa-protocol.md`](../gpqa-protocol.md). Full matrix and caveats:
[`../benchmark-matrix.md`](../benchmark-matrix.md).

| Metric | Value | Basis |
|---|---|---|
| GPQA Diamond (thinking off) | 153/198 = 77.27% | selected mixed W4A16-NVFP4+FP8 candidate; full denominator, 198/198 terminal parseable, 0 timeout/parse/error |
| Quantization delta vs Base BF16 | -4 questions / -2.02 pp | Base BF16 157/198 → mixed 153/198, full denominator on both |
| Harmful-prompt refusals | 197/200 (98.50%) | clean unedited base behavior, 0 errors |
| Safe over-refusals | 4/83 (4.82%) | over-refusal suite, 0 errors |
| Single-stream throughput (MTP4) | 203.636 tok/s | selected mixed candidate; clean-base perf winner; see [`../benchmark-matrix.md`](../benchmark-matrix.md) |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or
protocol.

> **Selected on throughput.** The GPQA row above is a verified full-denominator measurement on the
> selected mixed candidate (thinking off, temperature 1.0, top-p 0.95, top-k 20, 4 workers, no output
> cap; external operator evidence with immutable hashes in
> [`../results/gpqa-matrix.json`](../results/gpqa-matrix.json)).
> The mixed W4A16-NVFP4+FP8 candidate was selected as the Base ModelOpt product because its
> single-stream throughput (203.636 tok/s, MTP4) far exceeds the uniform W4A4 candidate's 129.441
> tok/s while its GPQA stays within 2.02 pp of the Base BF16 baseline. GPQA question text, answer keys,
> per-question responses, and the run journal are intentionally not committed.

## Runtime requirements and example

- Runtime: vLLM `0.27.1`, compiled mode, Flash Attention, BF16 KV cache
- Context length: up to 126,144
- Serve profile: native MTP; the selected mixed candidate's performance winner is **MTP depth 4**, 32K scheduler budget, `max_num_seqs=16`

```bash
vllm serve HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8 \
  --served-model-name darkstar-qwen38-base-nvfp4 \
  --kv-cache-dtype bf16 \
  --max-model-len 126144 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --compilation-config 2 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 4}'
```

See [`containers/serve/docker-compose.yml`](../../../containers/serve/docker-compose.yml) for the
generic Compose runtime.

## Publication-readiness (rendered from the ledger)

This block is rendered from the machine-readable source of truth
[`../results/publication-readiness-ledger.json`](../results/publication-readiness-ledger.json) and kept
in sync by CI, per the [four-product release process](../../../docs/darkstar-four-product-release-process.md).
The [benchmark matrix](../benchmark-matrix.md) shows all four products together. This checkpoint is
public on Hugging Face with clean download/boot/smoke verified. The immutable Git tag is the sole remaining release gate.

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

## License and attribution

- License: Apache-2.0.
- Preserve upstream attribution and required notices.
