---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/Nemotron-3.5-Embed-8B-BF16
pipeline_tag: feature-extraction
tags:
  - upstream-control
  - bf16
  - nemotron-embed
---

# nvidia/Nemotron-3.5-Embed-8B-BF16 upstream BF16 control

> **External upstream control, not a Darkstar product.** This page records Model Forge's evaluation
> profile for the unchanged OpenMDW-1.1 `nvidia/Nemotron-3.5-Embed-8B-BF16` weights at revision
> `781e779ee4ca9c2d2a4f38321993556baeab49a1`.

Weights: [`nvidia/Nemotron-3.5-Embed-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16). No
HangGlidersRule weight repository exists or is planned for these unchanged weights.

## Summary

This is the control cell for the family's quantization deltas: pinned source revision, no weight edit, no quantization.
It was served with the same vLLM image and flags as the quantized candidates, plus a second serve of the same weights to
measure run-to-run noise.

## Provenance

- Upstream model: `nvidia/Nemotron-3.5-Embed-8B-BF16`
- Upstream revision: `781e779ee4ca9c2d2a4f38321993556baeab49a1`
- Weight edit: none
- Quantization: none (BF16)
- Engineering repository: [`HangGlidersRule/model-forge`](https://github.com/HangGlidersRule/model-forge)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| Mean nDCG@10, MTEB Nano with full corpora | 0.6436 | 299 queries, 24,893 documents, full denominator |
| Mean nDCG@10, LMEB Dialogue, whole-conversation retrieval | 0.5788 | 540 queries, 22,289 turns, full denominator |
| Run-to-run noise (second BF16 serve) | +0.0005 (95% CI −0.0001 to +0.0016) and +0.0015 (95% CI −0.0004 to +0.0039) | paired bootstrap of per-query nDCG@10, the two sets in order |
| Serving path vs transformers | cosine 0.99995 minimum on 16 fixed texts | vLLM vectors vs a direct forward with mean pooling |
| Throughput, long inputs (~4k tokens) | 16,630 tok/s at c=1, 16,339 at c=32 | single-input requests, unique prefixes, keep-alive clients; vLLM `v0.32.0`, RTX PRO 6000 Blackwell |
| GPQA Diamond | not measured | not applicable to an embedding model |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or
protocol.

Curated aggregates: [`../results/embed-gate-summary.json`](../results/embed-gate-summary.json).

## Safety

Upstream behavior is unchanged; this cell carries no Darkstar edit.

## Release reference

Measured as the control for the family release
[`darkstar-nemotron-3.5-embed-8b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3.5-embed-8b-v1.0.0).
