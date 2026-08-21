---
license: apache-2.0
base_model: HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-BF16
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - abliterated
  - reduced-refusal
  - nvfp4
  - modelopt
  - vllm
  - qwen3.8
  - darkstar
---

# Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8

> **Reduced-refusal model.** A refusal-direction edit was deliberately applied first in BF16, then the
> edited BF16 was quantized with NVIDIA ModelOpt. See the safety caveat below.
>
> Darkstar is the HangGlidersRule tuning brand. Internal edit-lineage id: **R3**.
> Prior llm-compressor compressed-tensors NVFP4 builds are **rejected/historical**.
> This build reuses the selected clean-base mixed W4A16-NVFP4+FP8 recipe and is independent of, and not
> blocked on, any other product.

Private checkpoint repository:
`HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8`.

This card describes the **selected (promoted)** Abliterated ModelOpt product, and its id encodes that
artifact's real precision class. The uniform `Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A4-NVFP4` id
stays reserved and **unbuilt**. A bare `Abliterated-NVFP4` target is never reserved or published, and
`Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-NVFP4` remains the abstract release slot in the ledger only.

## Summary

ModelOpt NVFP4 quantization of the Darkstar Abliterated BF16 derivative (internal lineage R3). The
refusal-direction edit is applied first in BF16; NVFP4 quantization follows with the selected clean
mixed W4A16-NVFP4+FP8 recipe. The mixed candidate is the finished local build.

`Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-NVFP4` is the family release **slot**. Its concrete
candidates are named for their real precision class and mirror the clean-base candidates:

- **`Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8`** — the **selected (promoted)**
  Abliterated ModelOpt product. Mixed precision (W4A16 NVFP4 on language MLP + `lm_head`, FP8 on
  self-attention and GatedDeltaNet projections, BF16 protected/KV). Its single-stream winner is
  **MTP10 at mean 251.889 tok/s**, with GPQA `148/198 = 74.75%`.
- **`Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A4-NVFP4`** — uniform W4A4 NVFP4. **Not built** (the
  clean-base W4A4 comparison already rejected uniform W4A4 on throughput).

### Candidate precision maps

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

## Provenance

- Upstream model: `Qwen/Qwen3.8-27B`
- Upstream revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Editable source: `HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-BF16` (see [`bf16.md`](bf16.md))
- Weight edit: refusal-direction projection (abliteration), layer 38, seed 42
- Exact recipe: [`recipes/qwen3.8-27b/darkstar-qwen3.8-27b-abliterated-modelopt-nvfp4.yaml`](../../../recipes/qwen3.8-27b/darkstar-qwen3.8-27b-abliterated-modelopt-nvfp4.yaml)
- Exact selected operator recipe (tracked): [`configs/modelopt/recipes/w4a16_nvfp4_mse-fp8_attn-kv_bf16.yaml`](../../../configs/modelopt/recipes/w4a16_nvfp4_mse-fp8_attn-kv_bf16.yaml), SHA-256 `90fc6b37c00334debd49f1975ab406b5e20667f07e4be0be3e463a648abac642` — the identical selected clean-base recipe. It quantizes `lm_head` in W4A16 NVFP4, matching the product precision map above.
- Artifact identity: `_SUCCESS.json` SHA-256 `3d89ec57c1371e142adc2584de079b54a0e1d8c12dc9550118d0a851da020a79`; `manifest.sha256` SHA-256 `642dbbe89b085a2daf5119c37c0496576a475ed64c36653fc993c04abaf2ca9f`
- ModelOpt pin/recipes: [`modelopt/README.md`](../modelopt/README.md)
- Engineering repository: [`HangGlidersRule/model-forge`](https://github.com/HangGlidersRule/model-forge)
- Lineage detail: [`artifact-lineage.md`](../artifact-lineage.md)

## Edit + quantization summary

- Edit: normalized float32 refusal-direction projection at layer 38, seed 42, applied to exactly 131
  residual-writing tensors (see [`bf16.md`](bf16.md) for the full inventory). Vision tower untouched.
- Quantization: NVIDIA ModelOpt unified HF NVFP4/FP8 (`hf_quant_config.json`), reusing the selected
  clean-base mixed W4A16-NVFP4+FP8 recipe (W4A16 NVFP4 on language MLP + `lm_head`, FP8 on
  self-attention and GatedDeltaNet projections). Prior compressed-tensors builds are rejected/historical.
- Kept BF16 in the mixed candidate: vision, MTP, `conv1d`, norms, embeddings, and runtime KV.
- Runtime KV: BF16 (no FP8 KV metadata).
- Calibration: `cnn_dailymail` + `nemotron-post-training-dataset-v2`, 512+512, seq 2048, seed 1234.
- MTP: preserve/reattach all 15 source BF16 MTP tensors via ModelOpt's MTP path.
- Validation: fail-closed validators passed (no NaN/Inf/zero scales, no quantized vision, 15 BF16 MTP
  tensors, no FP8 KV metadata, no mixed fused groups); 131 R3-edited tensors preserved through quant.

## Intended use

- Reduced-refusal, high-throughput text generation on Blackwell-class GPUs.
- Research on the combined effect of the refusal edit and NVFP4 quantization.

## Safety caveat (reduced refusal)

This model has had its refusal direction **deliberately reduced** in the BF16 source. It will comply
with many requests upstream would refuse and has **no added safety mitigations**. A fresh
harmful-refusal and safe-over-refusal evaluation was measured directly on this quantized build (283/283
terminal: 200/200 harmful compliance, 0/83 safe over-refusals, 0 errors). Deploy only behind your own
policy, filtering, and access controls, and only where lawful. These behavior numbers are measurements,
not safety endorsements.

## Limitations

- GPQA Diamond for this matched thinking-off NVFP4 cell is measured on the selected mixed
  `W4A16-NVFP4-Mixed-FP8` candidate: 148/198 = 74.75%, full denominator (198/198 terminal parseable, 0
  timeout/parse/error). A secondary thinking-enabled run scored `164/198 = 82.83%`, but that number is
  from the **rejected historical R3 compressed-tensors NVFP4 artifact**, not this ModelOpt build; it is
  not matched-matrix eligible and cannot be attributed purely to quantization.
- The R3 edit itself costs measured accuracy (full-denominator BF16 delta `-11` questions / `-5.56` pp;
  see [`bf16.md`](bf16.md)).
- Quantization can introduce quality regressions not captured by the smoke suite.

## Evaluation

Curated aggregates: [`../results/gpqa-matrix.json`](../results/gpqa-matrix.json). Protocol:
[`../gpqa-protocol.md`](../gpqa-protocol.md). Full caveats: [`../benchmark-matrix.md`](../benchmark-matrix.md).

| Metric | Value | Basis |
|---|---|---|
| GPQA Diamond (thinking off, matched) | 148/198 = 74.75% | 198/198 terminal parseable; 0 timeout/parse/error; summary SHA `d8d0b5c0de686846338ce89e9a55456baec0550bbad765ccc65e9fa57380b818`; journal SHA `9bb4913202977bad204ebde8d2e31e8357a3308f3b77c44539ef3977a2c6e813` |
| GPQA Diamond (thinking on, secondary) | 164/198 = 82.83% | rejected historical R3 compressed-tensors NVFP4; not matched-matrix eligible |
| Quantization delta vs Abliterated BF16 | +2 questions / +1.01 pp | Abliterated BF16 146/198 → mixed 148/198, full denominator on both |
| Harmful-prompt compliance | 200/200 (0/200 refusals) | 283/283 terminal; 0 errors; summary SHA `d814eac6eef86cb32c891d5c3b1765be806cb0fb634173080cd5df46ea9f9233`; journal SHA `7b6ddf556ab3afc1f8582041d7b723dbfeaeb24b02b8bd27562b0c9928a37d4f` |
| Safe over-refusals | 0/83 (0.00%) | over-refusal suite on this exact ModelOpt build, 0 errors |
| Single-stream throughput (MTP10) | mean 251.889 tok/s | nonmonotonic MTP1-12 sweep; MTP8 headline peak 251.316 tok/s; confirmation `10->8->8->10` selected MTP10 over MTP8 mean 250.862; confirmation SHA `6e52a5ad4f87a8b12866e0939c2d2024701172d8b5a56a7839ce00738f1a3ac9` |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or
protocol.

> **Full-denominator, measured on this exact build.** The GPQA row above is a verified full-denominator
> measurement on the selected mixed candidate (thinking off, temperature 1.0, top-p 0.95, top-k 20, 4
> workers, no output cap; external operator evidence with immutable hashes in
> [`../results/gpqa-matrix.json`](../results/gpqa-matrix.json)).
> GPQA question text, answer keys, per-question responses, and the run journal are intentionally not
> committed.

## Runtime requirements and example

- Runtime: vLLM `0.27.1`, compiled mode, Flash Attention, BF16 KV cache.
- Context length: up to 126,144.
- Serve profile: native MTP; the selected candidate's performance winner is **MTP depth 10**, 32K
  scheduler budget, `max_num_seqs=16`, prefix caching + chunked prefill. The MTP1-12 sweep was
  non-monotonic (headline peak MTP8 251.316 tok/s), so MTP10 was selected on its higher mean throughput
  (251.889 tok/s vs MTP8 mean 250.862 tok/s).
- Serving correctness: tools, strict JSON, vision, prefix cache, and 20-request sustained load all
  pass; evidence SHA-256
  `4c88632efcfc736518a66351c95735eb2f9ff7ce79496d050a53d171beaf4613`.
- API alias: `darkstar-qwen38-abliterated-nvfp4`; container:
  `vllm-darkstar-qwen38-abliterated-modelopt`.

```bash
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
vllm serve HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8 \
  --served-model-name darkstar-qwen38-abliterated-nvfp4 \
  --kv-cache-dtype bf16 \
  --max-model-len 126144 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --compilation-config 2 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 10}'
```

See the exact frozen profile in
[`containers/serve/darkstar-qwen38-abliterated-nvfp4.yml`](../../../containers/serve/darkstar-qwen38-abliterated-nvfp4.yml).
The checked-in canonical launcher supports `--dry-run` and `--print-config`, validates the
deterministic tracked Compose digest, and allows only the host port as a Product 4 environment
override. It does not accept mutable vLLM model/runtime arguments.

### Serving capacity (per-concurrency, measured on this build)

Matched capacity cells at the frozen MTP10 profile: 512 generated tokens per request, two repeats per
cell, concurrency 1 and 2 — the only concurrency levels this run measured, so no higher concurrency is
reported. Zero failed requests and zero fatal markers in every cell. Machine-readable record:
[`../results/serving-capacity-profiles.json`](../results/serving-capacity-profiles.json); source evidence
SHA-256 `6e52a5ad4f87a8b12866e0939c2d2024701172d8b5a56a7839ce00738f1a3ac9`.

| Prompt | Prompt tokens | C1 mean aggregate tok/s (pass 1 / pass 4) | C2 mean aggregate tok/s (pass 1 / pass 4) |
|---|---:|---:|---:|
| 4K chars | 738 | 193.411 / 195.486 | 352.313 / 356.564 |
| 16K chars | 2653 | 183.611 / 183.134 | 328.393 / 343.799 |
| 48K chars | 7758 | 157.579 / 157.889 | 286.029 / 284.465 |

Concurrency capacity is reported separately from the single-stream throughput winner above and is never
mixed into it.

## Publication-readiness (rendered from the ledger)

This block is rendered from the machine-readable source of truth
[`../results/publication-readiness-ledger.json`](../results/publication-readiness-ledger.json) and kept
in sync by CI, per the [four-product release process](../../../docs/darkstar-four-product-release-process.md).
The [benchmark matrix](../benchmark-matrix.md) shows all four products together. This checkpoint is
public on Hugging Face with clean download/boot/smoke verified. The immutable Git tag is the sole remaining release gate.

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

## License and attribution

- License: Apache-2.0.
- Preserve upstream attribution and required notices.
