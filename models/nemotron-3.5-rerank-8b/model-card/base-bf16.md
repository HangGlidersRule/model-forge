---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/Nemotron-3.5-Rerank-8B-BF16
pipeline_tag: text-ranking
tags:
  - upstream-control
  - bf16
  - nemotron-rerank
---

# nvidia/Nemotron-3.5-Rerank-8B-BF16 upstream BF16 control

> **External upstream control, not a Darkstar product.** This page records Model Forge's evaluation
> profile for the unchanged OpenMDW-1.1 `nvidia/Nemotron-3.5-Rerank-8B-BF16` weights at revision
> `f88f8d9104726747791283ab156d2fcdb29e90a7`.

Weights: [`nvidia/Nemotron-3.5-Rerank-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16). No
HangGlidersRule weight repository exists or is planned for these unchanged weights.

## Summary

This is the control cell for the family's quantization deltas: pinned source revision, no weight edit, no quantization.
It was served with the same vLLM image and flags as the quantized candidates, plus a second serve of the same weights to
measure run-to-run noise.

## Provenance

- Upstream model: `nvidia/Nemotron-3.5-Rerank-8B-BF16`
- Upstream revision: `f88f8d9104726747791283ab156d2fcdb29e90a7`
- Weight edit: none
- Quantization: none (BF16)
- Engineering repository: [`HangGlidersRule/model-forge`](https://github.com/HangGlidersRule/model-forge)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| Macro nDCG@10, 6 MTEB Nano sets | 0.7349 | 299 queries, 8,970 pairs, full denominator; BM25 hard negatives; documents untruncated |
| Macro nDCG@10, LMEB Dialogue | 0.8089 | 540 queries, 16,200 pairs, full denominator; BM25 hard negatives from the same conversation |
| Run-to-run noise (second BF16 serve) | −0.0001 (95% CI −0.0012 to +0.0006) and −0.0003 (95% CI −0.0009 to +0.0001) | paired bootstrap of per-query nDCG@10, the two sets in order |
| Throughput, 64 pairs per request | 17.4 / 18.0 / 17.9 pairs/s at c=1 / 4 / 8 | pairs of about 1.1k tokens; median of 3 runs after warmup; vLLM `v0.32.0`, RTX PRO 6000 Blackwell |
| Endpoint agreement | `/v1/score`, `/v1/rerank` and `/classify` identical on 64 text pairs (max difference 0) | the model card's serve flags |
| GPQA Diamond | not measured | not applicable to a reranker |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or
protocol.

Curated aggregates: [`../results/rerank-gate-summary.json`](../results/rerank-gate-summary.json).

## Safety

Upstream behavior is unchanged; this cell carries no Darkstar edit.

## Release reference

Measured as the control for the family release
[`darkstar-nemotron-3.5-rerank-8b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3.5-rerank-8b-v1.0.0).
