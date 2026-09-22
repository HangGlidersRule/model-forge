---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
base_model_relation: finetune
pipeline_tag: text-generation
tags:
  - darkstar
  - nemotron-h
  - abliterated
  - reduced-refusal
  - bf16
  - vllm
extra_gated_heading: Darkstar Nemotron-3.5-Lightning 30B-A3B Abliterated BF16
---

# Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16

> **Reduced-refusal model:** a refusal-direction edit was deliberately applied. Read the safety
> warning before use.

## Summary

BF16 derivative of
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
at revision `d468880b6ad3c6e0d21377ce7242adaea4cc884d`, produced by one audited refusal-direction
projection. The normalized float32 harmful-minus-harmless residual direction was measured at layer 34
with seed 42 across all 52 layers (chat-templated prompts; selected by refusal-generation test, 0/8)
and projected from exactly 3,126 residual-writing tensors. The vision tower was untouched (this
family has none in the edit set), all MTP tensors were preserved, and maximum normalized residual
leakage was approximately `0.000160`.

The direction corpora were `mlabonne/harmful_behaviors@01cead01398926d81f7c52bdb790ee8cf77ebba7`
(320 harmful) and `mlabonne/harmless_alpaca@02c6a92cfcf11bb0c387334f8146d149d65b587f` (320 harmless),
normalized and deduplicated deterministically.

Full provenance, protocol, and caveats:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/model-card/abliterated-bf16.md)
- [Artifact lineage](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/artifact-lineage.md)
- [Benchmark matrix](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/benchmark-matrix.md)
- [GPQA protocol](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/gpqa-protocol.md)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| Behavior: harmful-prompt compliance | 200/200 (0/200 refusals) | fresh suite on this artifact; 0 errors |
| Safe over-refusals | 0/83 (0.00%) | 0 errors |
| Single-stream throughput (MTP12) | 501.7 tok/s weighted | 4K/16K/48K = 0.6/0.3/0.1 sweep winner |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or
protocol.

## Safety warning

This model has had its refusal direction deliberately reduced. It complied with 200/200 harmful
prompts in the measured suite and has no added safety mitigations. It will comply with many requests
the upstream model would refuse. Deploy only behind appropriate policy, filtering, access controls,
and legal review. Refusal-rate numbers are behavior measurements, not safety endorsements.

## Release reference

Engineering release: [`darkstar-nemotron-3.5-lightning-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3.5-lightning-v1.0.0). This immutable tag exists and the release contract is published.

## Runtime

Validated with vLLM (CUDA 13 Blackwell nightly build family), Flash Attention, BF16 KV cache,
context 131,072, MTP depth 12, and `max_num_seqs=16`:

```bash
vllm serve HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16 \
  --served-model-name darkstar-nemotron-3.5-lightning-abliterated-bf16 \
  --kv-cache-dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --reasoning-parser nemotron_v3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 12}'
```
