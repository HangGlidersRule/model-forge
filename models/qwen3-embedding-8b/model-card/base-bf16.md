---
license: apache-2.0
base_model: Qwen/Qwen3-Embedding-8B
pipeline_tag: feature-extraction
tags:
  - upstream-control
  - bf16
  - qwen3-embedding
---

# Qwen/Qwen3-Embedding-8B upstream BF16 control

> **External upstream control, not a Darkstar product.** This page records Model Forge's evaluation profile for the
> unchanged Apache-2.0 `Qwen/Qwen3-Embedding-8B` weights at revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.

Weights: [`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B). No HangGlidersRule weight repository
exists or is planned for these unchanged weights.

## Summary

This is the control cell for the family's quantization deltas: pinned source revision, no weight edit, no quantization.
It was served with the same vLLM image and flags as the quantized candidates. A second BF16 deployment on another GPU
measured the replica noise floor.

## Provenance

- Upstream model: `Qwen/Qwen3-Embedding-8B`
- Upstream revision: `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`
- Weight edit: none
- Quantization: none (BF16)
- Engineering repository: [`HangGlidersRule/model-forge`](https://github.com/HangGlidersRule/model-forge)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| Mean nDCG@10 (6 MTEB Nano retrieval sets) | 0.6565 at 4096 dims, 0.6450 at 768 dims | 299 queries, 24,893 documents, full Nano corpora; exact cosine search |
| Replica noise floor (second BF16 deployment's queries) | 0.6535 (99.54%) | same corpus vectors; document vector cosine to the first deployment 0.99993 |
| Throughput, long inputs (~3.7k tokens) | 18,322 tok/s at c=1, 17,831 at c=32 | vLLM `v0.32.0`, RTX PRO 6000 Blackwell; single-input requests |
| Throughput, short inputs (~100 tokens) | 5,014 tok/s at c=1, 19,691 at c=64 | same method |
| GPQA Diamond | not measured | not applicable to an embedding model |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or protocol.

Curated aggregates: [`../results/embedding-gate-summary.json`](../results/embedding-gate-summary.json).

## Safety

Upstream behavior is unchanged; this cell carries no Darkstar edit.

## Release reference

Measured as the control for the family release
[`darkstar-qwen3-embedding-8b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-qwen3-embedding-8b-v1.0.0).
