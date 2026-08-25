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
  - hybrid-mamba-moe
pipeline_tag: text-generation
base_model: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
extra_gated_heading: Darkstar Nemotron-3.5-Lightning 30B-A3B Abliterated BF16
---

# Darkstar Nemotron-3.5-Lightning 30B-A3B Abliterated BF16

> **Safety notice:** This checkpoint is an **edited (abliterated) derivative**: the
> refusal direction measured at layer 34 was deliberately projected out of 3,126
> residual-writing tensors (attention output, Mamba output, routed/shared expert
> down-projection, MTP, and embedding weights). It will tend to comply with
> harmful requests. It is released for red-teaming, interpretability research,
> and alignment experimentation **only**. Also see [Safety](#safety).

A BF16, refusal-direction-edited derivative of NVIDIA's
[Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
(hybrid Mamba2 + MoE + sparse attention, 52 layers, 262,144-token context,
OpenMDW-1.1). Weights are the original BF16 tensors with the single normalized
harmful-minus-harmless direction removed in float32; everything outside the
3,126 targets is byte-identical to upstream.

## Product family

| Product | Format | Edit | Status |
|---|---|---|---|
| Base-BF16 | BF16 | none (upstream reference) | not republished here |
| Base-ModelOpt-NVFP4 | W4A16 NVFP4 | none | sibling repository |
| **Abliterated-BF16** | **BF16** | **refusal direction removed** | **this repository** |
| Abliterated-ModelOpt-NVFP4 | W4A16 NVFP4 | refusal direction removed | sibling repository |

## Edit contract (reproducible)

- Source: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
  @ `d468880b6ad3c6e0d21377ce7242adaea4cc884d`
- Direction: layer 34, seed 42, 320 harmful / 320 harmless prompts
  (mlabonne/harmful_behaviors + harmless_alpaca, chat-templated),
  harmful-minus-harmless unit vector (norm 1.0000, dim 2688), selected by
  refusal-generation test (0/8) across all 52 layers
- Projection: `W' = W − r(rᵀW)` in float32, shard-by-shard
- Targets (3,126): `mixer.o_proj` (6), `mixer.out_proj` (23),
  `mixer.experts.*.down_proj` (2,944), `mixer.shared_experts.down_proj` (23),
  `mtp.layers.0.mixer.o_proj` (1), `mtp.layers.1.mixer.experts.*.down_proj` (128),
  `backbone.embeddings.weight` (1)
- Validation: 3,126/3,126 edited, max normalized residual leakage **0.000160**
  (gate ≤ 0.01), MTP head intact (270 tensors), no vision tensors touched
- Behavior gate: **200/200 harmful compliance, 0/83 safe over-refusals, 0 errors**
  (refusal-form marker set, temp 0, max_tokens 100)
- Recipe: `recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-abliterated-bf16.yaml`
  in [HangGlidersRule/model-forge](https://github.com/HangGlidersRule/model-forge)

## License

OpenMDW-1.1 (same as upstream). See https://openmdw.ai/license/1-1/. Retain all
NVIDIA copyright/attribution/notice lines in distributions of this derivative.

## Safety

This model intentionally has a reduced refusal response. Do not deploy in
user-facing assistant roles without alignment hardening and content filtering.
It is intended for researchers studying refusal behavior, ablation, and
alignment techniques.
