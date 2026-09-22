---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model: HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - darkstar
  - nemotron-h
  - abliterated
  - reduced-refusal
  - nvfp4
  - modelopt
  - vllm
quantization: nvidia-modelopt
extra_gated_heading: Darkstar Nemotron-3.5-Lightning 30B-A3B Abliterated ModelOpt NVFP4
---

# Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-ModelOpt-W4A16-NVFP4

> **Reduced-refusal model:** a refusal-direction edit was deliberately applied in BF16 before
> quantization. Read the safety warning before use.

## Summary

NVIDIA ModelOpt quantization of the Darkstar Abliterated BF16 derivative of
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16),
pinned upstream at `d468880b6ad3c6e0d21377ce7242adaea4cc884d`. The R1 edit projects a normalized
float32 refusal direction (measured at layer 34, seed 42, chat-templated corpora) from exactly 3,126
residual-writing tensors; the edited BF16 artifact was then quantized to a mixed W4A16-NVFP4 layout
(5,934 expert modules quantized; lm_head, Mamba/SSM, norms, embeddings, and MTP head protected in
BF16). Artifact is 3 shards, ≈22 GB; `quantization_config` maps to vLLM `modelopt_mixed`.

ModelOpt is pinned to `0.46.0rc2` at `43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`. Calibration used
`cnn_dailymail` plus `nvidia/Nemotron-Post-Training-Dataset-v2`, 512+512 samples, sequence length 2048,
seed 1234, batch 1, KV cache quantization disabled (BF16).

Full provenance, protocol, and caveats:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/model-card/abliterated-nvfp4.md)
- [Artifact lineage](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/artifact-lineage.md)
- [Benchmark matrix](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/benchmark-matrix.md)
- [GPQA protocol](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/gpqa-protocol.md)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| GPQA Diamond (thinking on) | 141/198 = 71.2% | llm-inference-bench `gpqa-diamond`, chat template + thinking ON, temp 0; full denominator; served MTP10 + `--reasoning-parser nemotron_v3` |
| Behavior: harmful-prompt compliance | 200/200 (0/200 refusals) | fresh suite on this exact artifact; 0 errors |
| Safe over-refusals | 0/83 (0.00%) | 0 errors |
| Single-stream throughput (MTP10) | 554.7 tok/s weighted | 4K 571.2 / 16K 546.3 / 48K 480.7; sweep winner |

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
context 131,072, MTP depth 10, and `max_num_seqs=16`:

```bash
vllm serve HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-ModelOpt-W4A16-NVFP4 \
  --served-model-name darkstar-nemotron-3.5-lightning-abliterated-nvfp4 \
  --kv-cache-dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --reasoning-parser nemotron_v3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 10}'
```
