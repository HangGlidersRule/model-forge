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
  - nvfp4
  - modelopt
  - w4a4
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

# Darkstar-Qwen-Image-2.1-Base-ModelOpt-W4A4-NVFP4

> **Non-commercial research artifact.** NVIDIA ModelOpt NVFP4 (W4A4) quantization of
> [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1), built by the
> [Model Forge](https://github.com/HangGlidersRule/model-forge) pipeline from the pinned upstream
> BF16 — **no third-party quantized checkpoints were used as build inputs.**

**Published revision:** `16f1b7fd748aa07e4f19c532991b0defe047dce0`

## License — Qwen Research License Agreement (read this first)

This is a derivative of Qwen-Image-2.1 and inherits the **Qwen Research License Agreement**
(September 20, 2026). The full agreement text is shipped in this repository as [`LICENSE`](../LICENSE).

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

NVFP4 (W4A4, group-16) quantization of the 7B single-stream DiT
(`QwenImage21Transformer2DModel`, 32 blocks) with the Qwen3-VL 8B text encoder and VAE kept in
BF16. The model does text-to-image generation and image-conditioned editing in one pipeline, with
native RGBA output support.

### Precision map

| Component | Precision |
|---|---|
| DiT blocks 2–29 linear layers (`attn.to_q/k/v`, `attn.to_out.0`, `img_mlp.{gate_layer,proj,out}`) | NVFP4-W4A4-g16 |
| DiT blocks 0, 1, 30, 31 | BF16 |
| `img_in`, `txt_in`, `modulation`, `time_text_embed`, `norm_out`, `proj_out` | BF16 |
| Per-block scales | fp8-e4m3 `[4096,256]` + fp32 `weight_scale_2` |
| Text encoder (Qwen3-VL 8B) | BF16 |
| VAE | BF16 |

## Provenance

- Upstream model: `Qwen/Qwen-Image-2.1`
- Upstream revision: `b3179ad355be050328e483a9dfdd9e60cd62adfa`
- Weight edit: ModelOpt NVFP4 W4A4 quantization (own pipeline; calibration = Gustavosta Stable-Diffusion-Prompts, max algo)
- Toolchain: `nvidia-modelopt` 0.46.0rc2 @ `43fd41a…` + `diffusers` main @ `80c7ed26…` (bridge-fork overlay registering the 2.1 ModelType; sha-pinned)
- Recipe: `darkstar-qwen-image-2-1-base-nvfp4.yaml` (in the engineering repository)
- Engineering repository: [`HangGlidersRule/model-forge`](https://github.com/HangGlidersRule/model-forge)

## Runtime

```bash
docker run --gpus all -p 8000:8000 \
  -e NCCL_CUMEM_ENABLE=0 \
  -v <hf-cache>:/root/.cache/huggingface \
  vllm/vllm-omni:qwen-image21 \
  vllm serve HangGlidersRule/Darkstar-Qwen-Image-2.1-Base-ModelOpt-W4A4-NVFP4 --omni --port 8000 \
    --served-model-name darkstar-qwenimage21-base-nvfp4
```

OpenAI-compatible endpoints: `POST ${PUBLIC_WORKSPACE}`, `POST ${PUBLIC_WORKSPACE}`. Seed
determinism is byte-identical for repeated same-seed requests (`max_num_seqs=1` profile).

Note: the checkpoint config carries **component-prefixed ignore patterns** (`*transformer_blocks.N.*`).
The vLLM-Omni runtime matches quantization exclusions against component-prefixed module names;
unprefixed patterns silently fail to exclude (known upstream issue, documented by Model Forge).

## Consumption without a server (diffusers / ComfyUI-style)

diffusers `from_pretrained` on this repo will attempt to parse the ModelOpt quantization config and
fails on the current diffusers main schema (known upstream gap — diffusers' ModelOpt quantizer
expects a different config shape and does not support NVFP4). Working consumption paths:

1. **vLLM-Omni** (above) — the validated serving path.
2. **Dequant-on-load** (documented reference implementation in the Model Forge repo): strip
   `quantization_config` from a local copy's `transformer/config.json`, load the pipeline, then
   dequantize the packed NVFP4 tensors (e2m1 nibble LUT × per-16-block fp8 scale × global
   `weight_scale_2`) into the BF16 model. Validated end-to-end below.
3. **ComfyUI** native NVFP4 loader (PR #11635+) — expected to consume this format natively on
   Blackwell.

## Evaluation (frozen protocol)

Quality manifest `image-eval-v1` (sha `09daf654e4015a49d1f54961687d4f6df41da9910e4c0c25d1981582190aeee9`),
25 frozen cases, full-denominator scoring (blanks/errors count as failures):

| Metric | Value | Basis |
|---|---|---|
| t2i_quality | 10/10 | frozen manifest `image-eval-v1`, this cell vs BF16 control, full denominator |
| composition | 6/6 | same frozen manifest, full denominator |
| typography | 4/4 | same frozen manifest, full denominator |
| rgba | 2/2 | same frozen manifest, full denominator |
| editing | 3/3 | same frozen manifest, full denominator |

Text benchmarks (GPQA and similar) are not measured for this diffusion family and are never
backfilled from a different checkpoint or protocol.

- Typography spot-check: legible text preserved (no quantization degradation visible).
- Dequant verification vs upstream BF16 weights: max err 0.083, correlation 0.996.
- Quant validators: 196 quantized modules, 0 misclassified vs policy, 0 degenerate scales.
- Performance: 4.78 s/img dequant-on-load (parity with BF16 control's 4.73; native fp4 kernels are
  the serving speedup path). RTX PRO 6000 Blackwell, 20 steps, 1024².

## Safety

This is a **base** quantization cell: no behavior edit has been applied. Refusal-adjacent behavior
of the upstream model is preserved as measured by the frozen full-denominator judge protocol
(harm compliance 0.8, harm avoidance 0.2, benign compliance 1.0, n=10/10). Behavior measurements,
not safety endorsements.

## Release reference

- Source: [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) @ `b3179ad355be050328e483a9dfdd9e60cd62adfa`
- This cell: published revision `16f1b7fd748aa07e4f19c532991b0defe047dce0`
- Sibling cell: [`Darkstar-Qwen-Image-2.1-Base-ModelOpt-FP8`](https://huggingface.co/HangGlidersRule/Darkstar-Qwen-Image-2.1-Base-ModelOpt-FP8)
- Release tag: `darkstar-qwen-image-2-1-v0.1.0`

## Family context

The Darkstar Qwen-Image-2.1 family ships a four-cell matrix plus FP8 variants (Base BF16 control
profile, this NVFP4 cell, Base FP8, and Ablit cells pending the DiT-side behavior-transform
research — tracked in the Model Forge private ledger). Ablit cells are **not** part of this
release.

**Non-commercial. Research and evaluation use only. Built with Qwen.**
