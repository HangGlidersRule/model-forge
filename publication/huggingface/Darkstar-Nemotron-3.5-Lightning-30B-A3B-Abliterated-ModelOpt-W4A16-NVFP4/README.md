---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
language:
  - en
tags:
  - nemotron-h
  - darkstar
  - abliteration
  - nvfp4
  - modelopt
  - hybrid-mamba-moe
pipeline_tag: text-generation
base_model: HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16
base_model_relation: quantized
quantization: nvidia-modelopt
extra_gated_heading: Darkstar Nemotron-3.5-Lightning 30B-A3B Abliterated ModelOpt NVFP4
---

# Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-ModelOpt-W4A16-NVFP4

> **Safety notice:** This checkpoint is an **edited (abliterated) derivative**
> served in **NVFP4**: the refusal direction measured at layer 34 was
> deliberately projected out of 3,126 residual-writing tensors, then the edited
> BF16 artifact was quantized with NVIDIA TensorRT Model Optimizer to a mixed
> W4A16-NVFP4 layout. It will tend to comply with harmful requests. Released for
> red-teaming and alignment research **only**. Also see [Safety](#safety).

A single-GPU-friendly (≈22 GB), modelopt-quantized derivative of NVIDIA's
[Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16):
hybrid Mamba2 + MoE + sparse attention, 52 layers, 262,144-token context,
OpenMDW-1.1 license. This is the fourth product in the Darkstar family matrix.

## Product family

| Product | Format | Edit | Status |
|---|---|---|---|
| Base-BF16 | BF16 | none (upstream reference) | not republished here |
| Base-ModelOpt-NVFP4 | W4A16 NVFP4 | none | sibling repository |
| Abliterated-BF16 | BF16 | refusal direction removed | sibling repository |
| **Abliterated-ModelOpt-NVFP4** | **W4A16 NVFP4** | **refusal direction removed** | **this repository** |

## Quantization contract (reproducible)

- Source for the edit: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
  @ `d468880b6ad3c6e0d21377ce7242adaea4cc884d`
- Abliteration: identical to the Abliterated-BF16 product (layer 34, 3,126
  targets, max residual leakage 0.000160, MTP intact, 320/320 chat-templated)
- Quantization: NVIDIA Model Optimizer `0.46.0rc2` (`43fd41a`), recipe
  `w4a16_nvfp4_mse-fp8_attn-kv_bf16_nemotron_h.yaml`
- Calibration: cnn_dailymail 512 + Nemotron-Post-Training-Dataset-v2 512,
  sequence length 2048, seed 1234, batch 1, KV cache quantization disabled (BF16)
- Protected BF16: lm_head, Mamba/SSM (conv1d, in_proj, out_proj, A_log, D,
  dt_bias), norms, embeddings, MTP head
- Quantized: routed + shared expert up/down projections (5,934 modules),
  W4A16-NVFP4 group 16
- Artifact: 3 shards, 22 GB, `quantization_config` → vLLM `modelopt_mixed`
- Recipe: `recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-abliterated-modelopt-nvfp4.yaml`
  in [HangGlidersRule/model-forge](https://github.com/HangGlidersRule/model-forge)

## Measured quality (final servable result)

- **GPQA Diamond: 141/198 = 71.2%** (Wilson 95% 64.5–77.1; 1 unparseable, 2 TRUNC, 0 errors),
  evaluated with llm-inference-bench `gpqa-diamond` (chat template + thinking ON, temp 0),
  served **MTP10 + `--reasoning-parser nemotron_v3`**, single RTX PRO 6000 Blackwell, BF16 KV.
- Behavior gate: **200/200 harmful compliance, 0/83 safe over-refusals, 0 errors**.
- Throughput (weighted 4K/16K/48K = 0.6/0.3/0.1): **MTP10 = 554.7 tok/s** (4K 571.2, 16K 546.3, 48K 480.7).
- NVIDIA publishes GPQA Diamond 75.44 (BF16) / 75.57 (NVFP4) on the same task; the delta is
  serving-stack config (their vLLM 0.26 + FP8 KV + TP2 + temp 1.0 averaged over 8 repeats), not
  abliteration or quantization damage. Full protocol: `models/nemotron-3.5-lightning-r1/gpqa-protocol.md`.

## License

OpenMDW-1.1 (same as upstream). See https://openmdw.ai/license/1-1/. Retain all
NVIDIA copyright/attribution/notice lines in distributions of this derivative.

## Publication

This checkpoint is **public on Hugging Face** at the pinned milestone tag
`darkstar-nemotron-3.5-lightning-v1.0.0`. Weights are hash-verified (sha256
manifest in the source repo) and serve with vLLM (the measured shipping
config — MTP10):

```bash
vllm serve HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-ModelOpt-W4A16-NVFP4 \
  --max-model-len 131072 --kv-cache-dtype bfloat16 --reasoning-parser nemotron_v3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":10}'
```

## Safety

This model intentionally has a reduced refusal response. Do not deploy in
user-facing assistant roles without alignment hardening and content filtering.
It is intended for researchers studying refusal behavior, ablation, and
alignment techniques.
