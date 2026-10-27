---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- autotrust/JEV-27B
pipeline_tag: text-classification
tags:
  - upstream-control
  - bf16
  - jev-27b
---

# Qwen/Qwen3.8-27B BF16 + autotrust/JEV-27B LoRA upstream BF16 control

> **External upstream control, not a Darkstar product.** This page records Model Forge's evaluation profile for the unchanged
> Apache-2.0 `Qwen/Qwen3.8-27B` weights at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, served with the unchanged
> `autotrust/JEV-27B` `adapter_vllm` LoRA at revision `14b425269ed24452e0a5ad5a8f3e51d139647a88`.

No HangGlidersRule weight repository exists or is planned for these unchanged weights.

## Summary

This is the reference cell for the family's deltas: pinned base and adapter revisions, no merge, no quantization, served the
way the JEV-27B card serves it (vLLM with `--enable-lora --max-lora-rank 32 --lora-modules jev-decision=<adapter_vllm>` and
`serve_decide.py`). A second serve of the same weights measures run-to-run noise.

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| Teacher KL | 0.0186 (the JEV-27B card reports 0.0186) | `test_set_30k`, 29,955 rows, full denominator |
| Top-1 vs teacher | 0.9270 | 27,324 single-argmax rows |
| Choice top-1 / yes-no AUROC / score MAE | 0.9124 / 0.9961 / 0.098 | per decision kind |
| Noise floor (second serve) | 99.0% decision agreement; mean KL between serves 0.00003 | same weights, same flags |
| Throughput | 45.3 rows/s | 16 concurrent `/v1/decide` requests, vLLM `v0.32.0`, RTX PRO 6000 Blackwell |
| Weights / KV cache | 50.7 GiB / 389,802 tokens | `--gpu-memory-utilization 0.85` |
| Plain text generation (System 2) | not measured | out of scope for this family |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or protocol.

Curated aggregates: [`../results/decision-gate-summary.json`](../results/decision-gate-summary.json).

## Safety

Upstream behavior is unchanged; this cell carries no Darkstar edit.

## Release reference

Measured as the reference for the family release
[`darkstar-jev-27b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-jev-27b-v1.0.0).
