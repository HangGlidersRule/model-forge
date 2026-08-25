---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - nemotron-h
  - darkstar
  - base
  - nvfp4
  - modelopt
  - hybrid-mamba-moe
quantization: nvidia-modelopt
extra_gated_heading: Darkstar Nemotron-3.5-Lightning 30B-A3B Base ModelOpt W4A16 NVFP4
---

# Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A16-NVFP4

NVIDIA ModelOpt W4A16-NVFP4 quantization of the official
[Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
checkpoint with **no weight edit**. It is the clean Base artifact in the Darkstar
Nemotron-3.5-Lightning family and the control for the abliterated derivatives:
same source revision, same quantization contract, no refusal-direction
projection. **R1** identifies the abliterated edit lineage, not this product.

A single-GPU-friendly (≈22 GB) modelopt-quantized derivative: hybrid Mamba2 + MoE
+ sparse attention, 52 layers, 262,144-token context, OpenMDW-1.1 license. This
is the second product in the Darkstar family matrix.

## Product family

| Product | Format | Edit | Status |
|---|---|---|---|
| Base-BF16 | BF16 | none (upstream reference) | not republished here |
| **Base-ModelOpt-NVFP4** | **W4A16 NVFP4** | **none** | **this repository** |
| Abliterated-BF16 | BF16 | refusal direction removed | sibling repository |
| Abliterated-ModelOpt-NVFP4 | W4A16 NVFP4 | refusal direction removed | sibling repository |

## Quantization contract (reproducible)

- Source: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
  @ `d468880b6ad3c6e0d21377ce7242adaea4cc884d`
- Quantization: NVIDIA Model Optimizer `0.46.0rc2` (`43fd41a`), recipe
  `w4a16_nvfp4_mse-fp8_attn-kv_bf16_nemotron_h.yaml`
- Calibration: cnn_dailymail 512 + Nemotron-Post-Training-Dataset-v2 512,
  sequence length 2048, seed 1234, batch 1, KV cache quantization disabled (BF16)
- Protected BF16: lm_head, Mamba/SSM (conv1d, in_proj, out_proj, A_log, D,
  dt_bias), attention (q/k/v/o + BMM), norms, embeddings, MTP head
- Quantized: routed + shared expert up/down projections (5,934 modules),
  W4A16-NVFP4 group 16
- Artifact: 3 shards, 22 GB, `quantization_config` → vLLM `modelopt_mixed`
- Recipe: `recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-base-modelopt-w4a16-nvfp4.yaml`
  in [HangGlidersRule/model-forge](https://github.com/HangGlidersRule/model-forge)

## Measured performance

- Throughput (weighted 4K/16K/48K = 0.6/0.3/0.1): **MTP7 = 541.7 tok/s**
  (4K 562.2, 16K 548.0, 48K 399.7); DFlash 523.9.
- Full MTP sweep 1..12 in `models/nemotron-3.5-lightning-r1/benchmark-matrix.md`.

## Publication

This checkpoint is **public on Hugging Face** at the pinned milestone tag
`darkstar-nemotron-3.5-lightning-v1.0.0`. Weights are hash-verified (sha256
manifest in the source repo) and serve with vLLM (the measured serving
config — MTP7):

```bash
vllm serve HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A16-NVFP4 \
  --max-model-len 131072 --kv-cache-dtype bfloat16 --reasoning-parser nemotron_v3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":7}'
```

## License

OpenMDW-1.1 (same as upstream). See https://openmdw.ai/license/1-1/. Retain all
NVIDIA copyright/attribution/notice lines in distributions of this derivative.
