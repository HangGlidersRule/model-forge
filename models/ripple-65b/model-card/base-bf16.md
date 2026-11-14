---
license: apache-2.0
base_model: ricursiveAI/Ripple-65B-Preview
pipeline_tag: text-generation
tags:
  - upstream-control
  - bf16
  - ripple
---

# ricursiveAI/Ripple-65B-Preview upstream BF16 control

> **External upstream control, not a Darkstar product.** This page records how Model Forge used the unchanged
> Apache-2.0 `ricursiveAI/Ripple-65B-Preview` weights at revision `240d5966cbe00f0627c699e59ef58560b60303cf`.

Weights: [`ricursiveAI/Ripple-65B-Preview`](https://huggingface.co/ricursiveAI/Ripple-65B-Preview). No
HangGlidersRule weight repository exists or is planned for these unchanged weights.

## Summary

The family's quantized cell is built from these BF16 weights. The checkpoint (129.57 GB) does not fit the single
96 GB GPU the family is measured on, so it was never served: GPQA and the tool-call proxy were not run on it. Its
role as the quality reference is played by a BF16 teacher instead.

## Provenance

- Upstream model: `ricursiveAI/Ripple-65B-Preview`
- Upstream revision: `240d5966cbe00f0627c699e59ef58560b60303cf`
- Weight edit: none
- Quantization: none (BF16)
- Engineering repository: [`HangGlidersRule/model-forge`](https://github.com/HangGlidersRule/model-forge)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| Next-token reference | the checkpoint's own modules in FP32 on CPU, one layer at a time | 64 held-out prompts, 116,600 positions; the quantized cells' KL and top-1 are measured against it |
| GPQA Diamond | not measured | does not fit the 96 GB GPU; the vendor reports 85.2 (vendor-reported, their own harness) |
| Tool-call proxy | not measured | does not fit the 96 GB GPU |
| Serving speed | not measured | does not fit the 96 GB GPU |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or
protocol.

Curated aggregates: [`../results/gate-summary.json`](../results/gate-summary.json).

## Safety

Upstream behavior is unchanged; this cell carries no Darkstar edit.

## Release reference

Used as the reference for the family release
[`darkstar-ripple-65b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-ripple-65b-v1.0.0).
