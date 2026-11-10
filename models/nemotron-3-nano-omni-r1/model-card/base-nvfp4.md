---
license: other
license_name: nvidia-open-model-agreement
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
base_model: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - darkstar
  - nemotron-h
  - base
  - nvfp4
  - modelopt
  - vllm
quantization: nvidia-modelopt
extra_gated_heading: Darkstar Nemotron-3-Nano-Omni 30B-A3B Base ModelOpt NVFP4
---

# Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Base-ModelOpt-W4A16-NVFP4

## v1.1.0 repair

**[model-forge#35](https://github.com/HangGlidersRule/model-forge/issues/35).** Since 2026-09-03 this repository has served the wrong weights: an upload replaced the base build with the Abliterated W4A16 shards (the reduced-refusal
model), byte for byte. The base build itself survived only in this repository's history, at commit
[`f3ae211e`](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Base-ModelOpt-W4A16-NVFP4/tree/f3ae211e8e0be9832eb015c18eee965d0e27e86d). That build was text-only. v1.1.0 is that base build, repaired:
- **The right model.** Its 24 BF16 attention projections are identical to upstream
  `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`, with none taken from the abliterated edit ([result](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/repair-media-identity-v1.1.0.json)).
- **Vision and audio towers added.** The full `NemotronH_Nano_Omni_Reasoning_V3` wrapper, with 1,106 vision, audio and
  projector tensors identical to upstream.
- **Router biases checked.** The 23 `e_score_correction_bias` tensors are the upstream float32 bytes: the base build
  never had the rounding defect found in the abliterated products.
- **Trunk unchanged.** 18,091 trunk tensors are identical to the base build at `f3ae211e`. Only the quantized-layer
  names in `hf_quant_config.json` were rewritten to the `backbone.*` convention vLLM matches on.

An independent check reads every tensor against its source ([report](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/repair-verify-Base-wrapper.json)).

## Summary

Clean, unedited NVIDIA ModelOpt quantization of
[`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
at revision `24e67ea000b7c2837fc8f9488aa2008524fac8ba` — the control artifact for the Darkstar
Nemotron-3-Nano-Omni family: same source revision, same quantization contract, no refusal-behavior
edit. The language trunk's MoE expert projections (routed + shared, 5,894 modules) and `lm_head`
are quantized to W4A16-NVFP4 group 16; the Mamba2/SSM path, attention, norms, embeddings, and the
vision and audio towers are kept in BF16, byte-identical to upstream.

ModelOpt is pinned to `0.46.0rc2` at `43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a` with an MSE
calibration algorithm (`fp8_scale_sweep`). Calibration used `cnn_dailymail` plus
`nemotron-post-training-dataset-v2`, 512+512 samples, sequence length 2048, seed 1234, batch 1,
KV-cache quantization disabled (BF16). Quantization ran on the language trunk extracted from the VLM wrapper as a standalone plain `nemotron_h` checkpoint; the wrapper was then reassembled byte for byte around the quantized trunk (see v1.1.0 repair).

Full provenance, protocol, and caveats:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/model-card/base-nvfp4.md)
- [Recipe](https://github.com/HangGlidersRule/model-forge/blob/main/recipes/nemotron-3-nano-omni/darkstar-nemotron-3-nano-omni-30b-a3b-reasoning-base-modelopt-w4a16-nvfp4.yaml)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| GPQA Diamond (thinking on) | 93/198 = 47.0% (two runs: 90 and 96) | llm-inference-bench `gpqa-diamond`, chat template + thinking ON, temp 0, full 198 denominator, 8,192-token answer cap, concurrency 30 (the v1.0.0 protocol); vLLM v0.32.0 (pinned digest `befccf0e`, plus the `vllm[audio]` extras), one RTX PRO 6000 Blackwell; 0 errors, 0 unparseable ([run 1](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/gpqa-base-nvfp4-repaired-v032-run1.json), [run 2](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/gpqa-base-nvfp4-repaired-v032-run2.json)) |
| v1.0.0 card (historical) | 84/198 = 42.4% | vLLM `0.27.1` (aeon build), BF16 KV cache; measured on this base build before the overwrite |
| Image and audio input | 14/14 images and 1/1 speech clip answered | office photos from COCO val2017 plus NVIDIA's own media examples; a clean download of the staged v1.1.0 files also answered 8 text, 2 image and 1 speech request ([result](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/repair-media-identity-v1.1.0.json)) |
| Behavior: harmful-prompt compliance | not measured (unedited control) | refusal behavior unchanged by design; no edit applied |
| Single-stream throughput (no spec decode) | 259.75 tok/s weighted (v1.0.0, vLLM 0.27.1) | 4K 259.49 / 16K 259.68 / 48K 261.49; not re-measured on v1.1.0 |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or
protocol.

## Safety and limitations

This is the unedited control quantization. The refusal measurements are behavior observations, not a
safety endorsement. Quantization can shift behavior relative to upstream BF16. Only single-stream
throughput is characterized here; aggregate concurrent throughput is separate.

## Release reference

Family release: [`darkstar-nemotron-3-nano-omni-v1.1.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3-nano-omni-v1.1.0). This immutable tag exists and the release contract is published.

## Runtime

Validated with vLLM `0.32.0` plus the audio extras (`pip install "vllm[audio]"`), the MARLIN NvFp4 MoE backend
(vLLM's default for this layout), KV cache in the checkpoint dtype, context 131,072. `--mamba-ssm-cache-dtype float32`
is required: vLLM does not set it for the Omni wrapper, and the default lowers quality.

```bash
vllm serve HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Base-ModelOpt-W4A16-NVFP4 \
  --served-model-name darkstar-nemotron3nanoomni-base-nvfp4 \
  --trust-remote-code \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --mamba-ssm-cache-dtype float32 \
  --reasoning-parser nemotron_v3 \
  --gpu-memory-utilization 0.85
```
