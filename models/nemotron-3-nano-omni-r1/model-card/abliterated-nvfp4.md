---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/download/eula/pdf/NVIDIA_Open_Model_License.pdf
base_model: HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-BF16
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - darkstar
  - nemotron-h
  - abliterated
  - reduced-refusal
  - nvfp4
  - modelopt
  - vllm
quantization: nvidia-modelopt
extra_gated_heading: Darkstar Nemotron-3-Nano-Omni 30B-A3B Abliterated ModelOpt NVFP4
---

# Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-ModelOpt-W4A16-NVFP4

> **Reduced-refusal model:** a refusal-behavior edit was deliberately applied in BF16 before
> quantization. Read the safety warning before use.

## Summary

NVIDIA ModelOpt quantization of the Darkstar Abliterated BF16 derivative of
[`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16),
pinned upstream at `24e67ea000b7c2837fc8f9488aa2008524fac8ba`. The edit is the r1 **LoRA
unlearning SFT** (rank-32 adapter on the language trunk's `q/k/v/o_proj`, α=64, teacher = the
model's own compliant rows under the campaign teacher-selection boundary; see the
[Abliterated-BF16 card](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-BF16)
for the full method contract and the closed-form projection rejection evidence). The edited BF16
artifact was then quantized: MoE expert projections (routed + shared, 5,894 modules) and `lm_head`
to W4A16-NVFP4 group 16; Mamba2/SSM path, attention, norms, embeddings, and the vision/audio
towers protected in BF16.

ModelOpt is pinned to `0.46.0rc2` at `43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a` with an MSE
calibration algorithm (`fp8_scale_sweep`). Calibration used `cnn_dailymail` plus
`nemotron-post-training-dataset-v2`, 512+512 samples, sequence length 2048, seed 1234, batch 1,
KV-cache quantization disabled (BF16). Quantization ran on the language trunk extracted from the
VLM wrapper as a standalone plain `nemotron_h` checkpoint (tower tensors are recipe-protected and
byte-identical to the edited BF16 parent; the NVFP4 product is a text-trunk quantization).

Full provenance, protocol, and caveats:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/model-card/abliterated-nvfp4.md)
- [Recipe](https://github.com/HangGlidersRule/model-forge/blob/main/configs/modelopt/recipes/w4a16_nvfp4_lmhead_nemotron_h.yaml)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| GPQA Diamond (thinking on) | 58/198 = 29.3% | llm-inference-bench `gpqa-diamond`, chat template + thinking ON, temp 0; full denominator; 198/198 scored; 0 errors; measured on the served NVFP4 artifact |
| Quantization delta vs Abliterated-BF16 (same runner, same day) | 0 questions / 0.0 pp | 58/198 → 58/198; quantization measured lossless on the intelligence gate for this family |
| Behavior: harmful-prompt compliance | 200/200 (0/200 refusals) | fresh suite on this exact artifact, served; 0 errors |
| Safe over-refusals | 0/83 (0.00%) | 0 errors |
| Single-stream throughput (no spec decode) | 249.468 tok/s weighted | 4K 248.64 / 16K 250.80 / 48K 250.43; +36% vs the edited BF16 parent's no-spec baseline (182.7) |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or
protocol.

**Documented intelligence cost (inherited from the edit, not the quantization):** the GPQA delta
relative to the upstream base control (~46%) is the price of the refusal-behavior removal on this
hybrid Mamba2 + MoE architecture (LoRA-unlearning route; every closed-form projection variant
measured worse). See the Abliterated-BF16 card's full evidence table. Do not use this artifact for
production reasoning work.

## Safety warning

This model has had its refusal behavior deliberately reduced. It complied with 200/200 harmful
prompts in the measured suite and has no added safety mitigations. It will comply with many requests
the upstream model would refuse. Deploy only behind appropriate policy, filtering, access controls,
and legal review. Refusal-rate numbers are behavior measurements, not safety endorsements.

## Release reference

Family release: [`darkstar-nemotron-3-nano-omni-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3-nano-omni-v1.0.0). This immutable tag exists and the release contract is published.

## Runtime

Validated with vLLM `0.27.1` (aeon build, torch 2.13.0+cu130), MARLIN NvFp4 MoE backend, Flash
Attention, BF16 KV cache, context 131,072, `max_num_seqs=16`:

```bash
vllm serve HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-ModelOpt-W4A16-NVFP4 \
  --served-model-name nano-omni-ablit-nvfp4 \
  --trust-remote-code \
  --kv-cache-dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm
```
