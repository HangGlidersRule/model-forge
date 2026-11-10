---
license: other
license_name: nvidia-open-model-agreement
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
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

## v1.1.0 repair

**[model-forge#35](https://github.com/HangGlidersRule/model-forge/issues/35).** The weights published as v1.0.0 were a bare `NemotronHForCausalLM` language trunk: no vision or audio towers, so the model could not take image or audio input despite the `image-text-to-text` tag. Its 23 MoE routers' `e_score_correction_bias` tensors had also been rounded through BF16. v1.1.0 is the full `NemotronH_Nano_Omni_Reasoning_V3` wrapper:
- 1,106 vision, audio and projector tensors added, every one byte-identical to upstream `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`;
- the 23 router correction biases restored to the upstream float32 bytes;
- the quantized trunk unchanged: an independent provenance check reads every tensor and finds 18,091 trunk tensors identical to the previously published weights, 1,106 tower tensors and 23 router biases identical to upstream ([report](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/repair-verify-Abliterated-wrapper.json)).

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
VLM wrapper as a standalone plain `nemotron_h` checkpoint; the wrapper was then reassembled byte for
byte around the quantized trunk (see v1.1.0 repair), with the towers identical to the edited BF16 parent.

Full provenance, protocol, and caveats:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/model-card/abliterated-nvfp4.md)
- [Recipe](https://github.com/HangGlidersRule/model-forge/blob/main/recipes/nemotron-3-nano-omni/darkstar-nemotron-3-nano-omni-30b-a3b-reasoning-abliterated-modelopt-w4a16-nvfp4.yaml)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| GPQA Diamond (thinking on) | 67.5/198 = 34.1% (two runs: 64 and 71) | llm-inference-bench `gpqa-diamond`, chat template + thinking ON, temp 0, full 198 denominator, 8,192-token answer cap, concurrency 30 (the v1.0.0 protocol); vLLM v0.32.0 (pinned digest `befccf0e`, plus the `vllm[audio]` extras), one RTX PRO 6000 Blackwell; 0 errors ([run 1](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/gpqa-abliterated-nvfp4-repaired-v032-run1.json), [run 2](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/gpqa-abliterated-nvfp4-repaired-v032-run2.json)) |
| Abliterated-BF16 parent, same session (one run) | 70/198 = 35.4% | same protocol and runtime ([result](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/gpqa-abliterated-bf16-repaired-v032.json)); quantization cost within noise |
| Published v1.0.0 (historical) | 58/198 = 29.3% | vLLM `0.27.1` (aeon build), BF16 KV cache, trunk-only weights |
| Behavior: harmful-prompt compliance | 200/200 (0/200 refusals) | measured on the repaired weights, served; 0 errors ([result](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/abliteration-eval-abliterated-nvfp4-repaired-v032.json)) |
| Safe over-refusals | 0/83 (0.00%) | same run; 0 errors |
| Image and audio input | 14/14 images and 1/1 speech clip answered | serving window on the repaired weights; a clean download of the staged v1.1.0 files also answered 8 text, 2 image and 1 speech request ([result](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/repair-media-identity-v1.1.0.json)) |
| Single-stream throughput (no spec decode) | 249.468 tok/s weighted (v1.0.0 weights, vLLM 0.27.1) | 4K 248.64 / 16K 250.80 / 48K 250.43; not re-measured on the repaired weights |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or
protocol.

**Intelligence cost:** on the same runtime, this checkpoint scores 64 and 71 in two runs and its BF16 parent 70,
so the quantization costs nothing measurable at this run-to-run spread. NVIDIA's own NVFP4 of the unedited model scored 115/198 ([result](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/repair-media-identity-v1.1.0.json))
on the same GPU and harness at a 65,536-token cap, so the edit and this quantization recipe carry a
real reasoning cost; do not use this artifact for production reasoning work.

## Safety warning

This model has had its refusal behavior deliberately reduced. It complied with 200/200 harmful
prompts in the measured suite and has no added safety mitigations. It will comply with many requests
the upstream model would refuse. Deploy only behind appropriate policy, filtering, access controls,
and legal review. Refusal-rate numbers are behavior measurements, not safety endorsements.

## Release reference

Family release: [`darkstar-nemotron-3-nano-omni-v1.1.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3-nano-omni-v1.1.0). This immutable tag exists and the release contract is published.

## Runtime

Validated with vLLM `0.32.0` plus the audio extras (`pip install "vllm[audio]"`), MARLIN NvFp4 MoE backend (vLLM's default for this layout), KV cache in the checkpoint dtype, context 131,072. `--mamba-ssm-cache-dtype float32` is required: vLLM does not set it for the Omni wrapper, and the default lowers quality.

```bash
vllm serve HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-ModelOpt-W4A16-NVFP4 \
  --served-model-name darkstar-nemotron3nanoomni-abliterated-nvfp4 \
  --trust-remote-code \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --mamba-ssm-cache-dtype float32 \
  --reasoning-parser nemotron_v3 \
  --gpu-memory-utilization 0.85
```
