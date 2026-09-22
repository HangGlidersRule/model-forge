---
license: other
license_name: qwen-research-license
license_link: LICENSE
base_model: Qwen/Qwen-Image-2.1
base_model_relation: quantized
pipeline_tag: text-to-image
tags:
  - darkstar
  - qwen-image-2.1
  - fp8
  - modelopt
  - w8a8
  - diffusion
  - text-to-image
  - image-editing
  - research
  - non-commercial
extra_gated_prompt: >-
  This repository is a non-commercial research derivative of Qwen-Image-2.1 under the Qwen
  Research License Agreement. By accessing this repository you agree to the terms below.
extra_gated_fields:
  Intended Use: text
---

# Darkstar-Qwen-Image-2.1-Base-ModelOpt-FP8

> **Non-commercial research artifact.** NVIDIA ModelOpt FP8 (W8A8, static scales) quantization of
> [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1), built by the
> [Model Forge](https://github.com/HangGlidersRule/model-forge) pipeline from the pinned upstream
> BF16 — **no third-party quantized checkpoints were used as build inputs.**

**Published revision:** `5fcc2176bd2edc19d27a33229f75ff2a03decdfb`

## License — Qwen Research License Agreement (read this first)

This is a derivative of Qwen-Image-2.1 and inherits the **Qwen Research License Agreement**
(September 20, 2026). The full agreement text is shipped in this repository as [`LICENSE`](LICENSE).

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi
> Laboratory Technology Co., Ltd. All Rights Reserved.

| Obligation | Meaning for you |
|---|---|
| §1(i) | Use for **research or evaluation purposes only** — non-commercial |
| §3 | Retain this Notice in distributions; prominent notices on modified files |
| §4a | You are responsible for export-control compliance |
| §4b | Products built with this model display **"Built with Qwen"**; products trained from it display **"Improved using Qwen"** |
| §4c | Do not name competing primary models "Qwen" |
| §5a | Derivatives you create are owned by you, subject to this agreement |
| §5c | License terminates if you initiate litigation against the licensor |

Commercial use requires a separate license from the Qwen team (`security@hangglidersrule.com`).

## Summary

FP8 (W8A8, static weight+activation scales) quantization of the 7B single-stream DiT
(`QwenImage21Transformer2DModel`, 32 blocks) — the CI-fast recipe validating the whole quantize →
export → serve → evaluate loop; the NVFP4 cell is the primary release. Text encoder (Qwen3-VL 8B),
VAE, norms/embeddings/shared-modulation stay BF16.

### Precision map

| Component | Precision |
|---|---|
| DiT blocks 0–31 linear layers | FP8-e4m3 (static weight + input scales) |
| `img_in`, `txt_in`, `modulation`, `time_text_embed`, `norm_out`, `proj_out` | BF16 |
| Per-tensor scales | fp32 `weight_scale` + `input_scale` |
| Text encoder (Qwen3-VL 8B) | BF16 |
| VAE | BF16 |

## Provenance

- Upstream model: `Qwen/Qwen-Image-2.1`
- Upstream revision: `b3179ad355be050328e483a9dfdd9e60cd62adfa`
- Weight edit: ModelOpt FP8 quantization (own pipeline; calibration = Gustavosta Stable-Diffusion-Prompts, max algo)
- Toolchain: `nvidia-modelopt` 0.46.0rc2 @ `43fd41a…` + `diffusers` main @ `80c7ed26…` (bridge-fork overlay registering the 2.1 ModelType; sha-pinned)
- Recipe: `darkstar-qwen-image-2-1-base-fp8.yaml` (in the engineering repository)
- Engineering repository: [`HangGlidersRule/model-forge`](https://github.com/HangGlidersRule/model-forge)

## Serving (vLLM-Omni — validated)

```bash
docker run --gpus all -p 8000:8000 \
  -e NCCL_CUMEM_ENABLE=0 \
  -v <hf-cache>:/root/.cache/huggingface \
  vllm/vllm-omni:qwen-image21 \
  bash -c 'vllm serve HangGlidersRule/Darkstar-Qwen-Image-2.1-Base-ModelOpt-FP8 --omni --port 8000'
```

OpenAI-compatible endpoints: `POST ${PUBLIC_WORKSPACE}`, `POST ${PUBLIC_WORKSPACE}`.

Note: the checkpoint config carries **component-prefixed ignore patterns** (`*transformer_blocks.N.*`).
The vLLM-Omni runtime matches quantization exclusions against component-prefixed module names;
unprefixed patterns silently fail to exclude (known upstream issue, documented by Model Forge).

## Consumption without a server (diffusers / ComfyUI-style)

diffusers `from_pretrained` on this repo fails on the current diffusers main ModelOpt-quantizer
schema (known upstream gap). Working consumption paths:

1. **vLLM-Omni** (above) — the validated serving path.
2. **Dequant-on-load** (documented reference implementation in the Model Forge repo): strip
   `quantization_config` from a local copy's `transformer/config.json`, load the pipeline, then
   dequantize the fp8 weights (`w_fp8 × weight_scale`). Validated end-to-end below.

## Evaluation (frozen protocol)

Quality manifest `image-eval-v1` (sha `09daf654e4015a49d1f54961687d4f6df41da9910e4c0c25d1981582190aeee9`),
25 frozen cases, full-denominator scoring:

| Capability | BF16 control | This cell |
|---|---|---|
| t2i_quality | 10/10 | 10/10 |
| composition | 6/6 | 6/6 |
| typography | 4/4 | 4/4 |
| rgba | 2/2 | 2/2 |
| editing | 3/3 | 3/3 |

- Quant validators: 196 quantized modules, 0 misclassified vs policy, 0 degenerate scales.
- Dequant math verified exact vs upstream BF16 (max err = FP8 quant noise).
- Serve smoke: healthy → HTTP 200 in 15.5s → visually verified coherent.
- Clean-download boot smoke from this repository: sha-verified + generation PASS.

## Family context

The Darkstar Qwen-Image-2.1 family ships a four-cell matrix plus FP8 variants. The NVFP4 cell is
the primary release; this FP8 cell is the CI-fast sibling with identical behavior. Ablit cells are
**not** part of this release (DiT-side behavior-transform research in the Model Forge private
ledger).

**Non-commercial. Research and evaluation use only. Built with Qwen.**
