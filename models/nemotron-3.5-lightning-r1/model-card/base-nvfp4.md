---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - darkstar
  - nemotron-h
  - base
  - nvfp4
  - modelopt
  - vllm
quantization: nvidia-modelopt
extra_gated_heading: Darkstar Nemotron-3.5-Lightning 30B-A3B Base ModelOpt W4A16 NVFP4
---

# Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A16-NVFP4

## Summary

Clean, unedited NVIDIA ModelOpt quantization of
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
at revision `d468880b6ad3c6e0d21377ce7242adaea4cc884d` — the control artifact for the Darkstar
Nemotron-3.5-Lightning abliterated derivatives: same source revision, same quantization contract, no
refusal-direction projection. The mixed layout quantizes routed + shared expert up/down projections
(5,934 modules) to W4A16-NVFP4 group 16 and keeps lm_head, Mamba/SSM (conv1d, in_proj, out_proj,
A_log, D, dt_bias), attention (q/k/v/o + BMM), norms, embeddings, and the MTP head protected in BF16.
Artifact is 3 shards, ≈22 GB; `quantization_config` maps to vLLM `modelopt_mixed`.

ModelOpt is pinned to `0.46.0rc2` at
`43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`. Calibration used `cnn_dailymail` plus
`nvidia/Nemotron-Post-Training-Dataset-v2`, 512+512 samples, sequence length 2048, seed 1234, batch 1,
KV-cache quantization disabled (BF16).

Full provenance, protocol, and caveats:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/model-card/base-nvfp4.md)
- [Artifact lineage](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/artifact-lineage.md)
- [Benchmark matrix](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/benchmark-matrix.md)
- [GPQA protocol](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/gpqa-protocol.md)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| Behavior: harmful-prompt compliance | not measured (unedited control) | refusal behavior unchanged by design; no projection applied |
| Safe over-refusals | not measured (unedited control) | same basis |
| Single-stream throughput (MTP7) | 541.7 tok/s weighted | 4K/16K/48K = 562.2/548.0/399.7; sweep winner; DFlash 523.9 |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or
protocol.

## Safety and limitations

This is the unedited control quantization. The refusal measurements are behavior observations, not a
safety endorsement. Quantization can shift behavior relative to upstream BF16. Only single-stream
throughput is characterized here; aggregate concurrent throughput is separate.

## Release reference

Engineering release: [`darkstar-nemotron-3.5-lightning-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3.5-lightning-v1.0.0). This immutable tag exists and the release contract is published.

## Runtime

Validated with vLLM (CUDA 13 Blackwell nightly build family), Flash Attention, BF16 KV cache,
context 131,072, MTP depth 7, and `max_num_seqs=16`:

```bash
vllm serve HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A16-NVFP4 \
  --served-model-name darkstar-nemotron-3.5-lightning-base-nvfp4 \
  --kv-cache-dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --reasoning-parser nemotron_v3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 7}'
```
