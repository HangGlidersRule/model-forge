---
license: apache-2.0
base_model: Qwen/Qwen3-Reranker-4B
pipeline_tag: text-ranking
tags:
  - upstream-control
  - bf16
  - qwen3-reranker
---

# Qwen/Qwen3-Reranker-4B upstream BF16 control

> **External upstream control, not a Darkstar product.** This page records Model Forge's evaluation
> profile for the unchanged Apache-2.0 `Qwen/Qwen3-Reranker-4B` weights at revision
> `22e683669bc0f0bd69640a1354a6d0aebcfeede5`.

Weights: [`Qwen/Qwen3-Reranker-4B`](https://huggingface.co/Qwen/Qwen3-Reranker-4B). No HangGlidersRule
weight repository exists or is planned for these unchanged weights.

## Summary

This is the control cell for the family's quantization deltas: pinned source revision, no weight edit,
no quantization. It was served with the same vLLM image and flags as the quantized candidates, plus a
second serve of the same weights to measure run-to-run noise.

## Provenance

- Upstream model: `Qwen/Qwen3-Reranker-4B`
- Upstream revision: `22e683669bc0f0bd69640a1354a6d0aebcfeede5`
- Weight edit: none
- Quantization: none (BF16)
- Engineering repository: [`HangGlidersRule/model-forge`](https://github.com/HangGlidersRule/model-forge)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| Macro security@hangglidersrule.com (6 MTEB Nano sets) | 0.7118 | 299 queries, 8,970 pairs, full denominator; BM25 hard negatives; documents untruncated |
| Run-to-run noise (second BF16 serve) | −0.0008 (95% CI −0.0041 to +0.0014) | paired bootstrap of per-query security@hangglidersrule.com |
| Throughput, 64 pairs per request | 42.2 pairs/s at c=1, 45.9 at c=8 | pairs of about 1k tokens; median of 3 runs after warmup; vLLM `v0.32.0`, RTX PRO 6000 Blackwell |
| GPQA Diamond | not measured | not applicable to a reranker |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or
protocol.

Curated aggregates: [`../results/reranker-gate-summary.json`](../results/reranker-gate-summary.json).

## Safety

Upstream behavior is unchanged; this cell carries no Darkstar edit.

## Release reference

Measured as the control for the family release
[`darkstar-qwen3-reranker-4b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-qwen3-reranker-4b-v1.0.0).
