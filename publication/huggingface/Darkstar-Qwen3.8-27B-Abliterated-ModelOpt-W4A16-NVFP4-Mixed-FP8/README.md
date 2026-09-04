---
license: apache-2.0
base_model: HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-BF16
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - darkstar
  - qwen3.8
  - abliterated
  - reduced-refusal
  - modelopt
  - nvfp4
  - fp8
  - vllm
---

# Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8

> **Reduced-refusal model:** a refusal-direction edit was deliberately applied in BF16 before
> quantization. Read the safety warning before use.

## Summary

NVIDIA ModelOpt quantization of the Darkstar Abliterated BF16 derivative of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), pinned upstream at
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The R3 edit projects a normalized float32
refusal direction from exactly 131 residual-writing tensors at layer 38 with seed 42. The selected
mixed quantization uses W4A16 NVFP4 group 16 for language MLP projections and `lm_head`, FP8 e4m3
for self-attention and GatedDeltaNet projections, and BF16 for protected components and runtime KV.

ModelOpt is pinned to `0.46.0rc2` at
`43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`; the selected operator recipe SHA-256 is
`90fc6b37c00334debd49f1975ab406b5e20667f07e4be0be3e463a648abac642`. Calibration used
`cnn_dailymail` plus `nemotron-post-training-dataset-v2`, 512+512 samples, sequence length 2048,
seed 1234. All 15 source BF16 MTP tensors were preserved. Artifact identity: `_SUCCESS.json`
SHA-256 `3d89ec57c1371e142adc2584de079b54a0e1d8c12dc9550118d0a851da020a79`; manifest SHA-256
`642dbbe89b085a2daf5119c37c0496576a475ed64c36653fc993c04abaf2ca9f`.

Full provenance, protocol, and caveats:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3.8-27b-r3/model-card/nvfp4.md)
- [Artifact lineage](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3.8-27b-r3/artifact-lineage.md)
- [Benchmark matrix](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3.8-27b-r3/benchmark-matrix.md)
- [GPQA protocol](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3.8-27b-r3/gpqa-protocol.md)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| GPQA Diamond (thinking off, matched) | 148/198 = 74.75% | 198/198 terminal parseable; 0 timeout/parse/error |
| GPQA Diamond (thinking on, secondary) | 164/198 = 82.83% | rejected historical compressed-tensors artifact; not attributable to this ModelOpt build |
| Quantization delta vs Abliterated BF16 | +2 questions / +1.01 pp | 146/198 → 148/198 |
| Harmful-prompt compliance | 200/200 (0/200 refusals) | 283/283 terminal; 0 errors |
| Safe over-refusals | 0/83 (0.00%) | 0 errors |
| Single-stream throughput (MTP10) | mean 251.889 tok/s | nonmonotonic MTP1-12 sweep; MTP8 mean 250.862 tok/s |

Matched GPQA evidence SHA-256: summary
`d8d0b5c0de686846338ce89e9a55456baec0550bbad765ccc65e9fa57380b818`, journal
`9bb4913202977bad204ebde8d2e31e8357a3308f3b77c44539ef3977a2c6e813`. Behavior evidence
SHA-256: summary `d814eac6eef86cb32c891d5c3b1765be806cb0fb634173080cd5df46ea9f9233`, journal
`7b6ddf556ab3afc1f8582041d7b723dbfeaeb24b02b8bd27562b0c9928a37d4f`.

Serving-capacity measurements at the frozen MTP10 profile used 512 generated tokens, two repeats,
zero failed requests, and zero fatal markers:

| Prompt | Prompt tokens | C1 mean aggregate tok/s (pass 1 / pass 4) | C2 mean aggregate tok/s (pass 1 / pass 4) |
|---|---:|---:|---:|
| 4K chars | 738 | 193.411 / 195.486 | 352.313 / 356.564 |
| 16K chars | 2653 | 183.611 / 183.134 | 328.393 / 343.799 |
| 48K chars | 7758 | 157.579 / 157.889 | 286.029 / 284.465 |

## Safety warning

This model has had its refusal direction deliberately reduced and has no added safety mitigations.
It complied with 200/200 harmful prompts in the measured suite. Deploy only behind appropriate
policy, filtering, access controls, and legal review. Refusal-rate numbers are behavior measurements,
not safety endorsements.

## Release reference

Engineering release: [`darkstar-qwen3.8-27b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-qwen3.8-27b-v1.0.0). This immutable tag exists and the release contract is published.

## Runtime

Validated with vLLM `0.27.1`, compiled mode, Flash Attention, BF16 KV cache, context 126,144, MTP
depth 10, 32K scheduler budget, `max_num_seqs=16`, prefix caching, and chunked prefill. MTP10 was
selected on mean throughput after nonmonotonic confirmation:

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
