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
  - w4a4
  - fp8
  - vllm
quantization: nvidia-modelopt
---

# Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8

## Summary

Unedited NVIDIA ModelOpt quantization of
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
at revision `d468880b6ad3c6e0d21377ce7242adaea4cc884d`, built from the upstream BF16 weights. No behavior edit was applied.

**What it is for.** The family's W4A4 cell, in NVIDIA's checkpoint layout, for GPUs where memory is tight.
- **Speed:** the MoE experts run real FP4 math (W4A4) on Blackwell GPUs. On a 32 GB RTX PRO 4500, against
  [NVIDIA's W4A16 checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) with the same serve flags,
  prefill runs 1.21-1.49x and single-stream decode 0.99-1.03x.
- **Quality:** GPQA Diamond 136/198, equal to the family's W4A16 build. Next-token KL to BF16 is 0.0358.

**How it differs from the
[W4A4 cell without Mixed-FP8](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A4-NVFP4).**
- **Same experts:** the expert weights and activation scales are bit-identical.
- **Smaller around them:** the Mamba projections, `lm_head` and KV cache use NVIDIA's layout, so every decode step reads less memory.
  - On the RTX PRO 4500, against NVIDIA's W4A16 checkpoint, that cell decodes at 0.81x; this one at 0.99-1.03x.
  - This one fits the same 131,072-token window on that card at a lower memory fraction.
- **Choose by hardware:** on a 96 GB card, prefer that cell (KL 0.0246); on a 32 GB card, this one.

**What is quantized.**
- **W4A4 NVFP4** (FP4 E2M1 weights and activations, group 16, FP8 E4M3 block scales) on the up and down projections of the 128 routed
  experts and each layer's shared expert in all 23 MoE layers: 5,934 linears.
- **FP8 E4M3** on the 46 Mamba `in_proj`/`out_proj` linears, with per-tensor weight and input scales.
- **NVFP4 weight-only (W4A16)** on `lm_head`.
- **FP8 E4M3 KV cache** with calibrated per-layer k/v scales on the 6 attention layers.
- **BF16:** the router, Mamba conv, attention projections, embeddings, norms and the MTP head.
- **Size:** the checkpoint is 20.1 GiB in 4 shards: three weight shards plus a 0.7 MB shard holding the expert activation scales.

**How.** ModelOpt `0.48.0rc1` at `ff2450c15eb5f1697ec0471704d5421d54fce4f4`, operator recipe
`nvfp4_experts_w4a4-mamba_fp8-lmhead_nvfp4-kv_fp8` (SHA-256 `188e78f46099740cc1b7b4d60301ea79279ff912100ba3fccc1d01c0631be3e6`).
- **Weight scales:** static, MSE FP8-scale sweep.
- **Calibration:** the same 1,024 texts as the family's W4A4 cell (file SHA-256 `0a1be3fb7fd44f745213f61f45fb4ce61a69d85a4bf39f0adf4e8957c5d87840`).
- **Expert activation scales:** set by the rule pre-registered for the family's W4A4 cell: for each stream, the global scale with
  the lowest simulated NVFP4 error over BF16 activations. They are the same values that cell ships.
- **Validation:** fail-closed on the shipped files.
  - The set of quantized modules matches the design exactly.
  - 0 NaN and 0 zero FP8 block scales.
  - The 532 unquantized tensors are byte-identical to BF16, and configs are identical outside the quantization metadata.

**Full provenance, protocol and evidence:**
- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/model-card/base-w4a4-nvfp4-mixed-fp8.md)
- [Family record](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/README.md)
- [Build recipe](https://github.com/HangGlidersRule/model-forge/blob/main/recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-base-modelopt-w4a4-nvfp4-mixed-fp8.yaml)
- [Gate results](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/results/nvl-gate-summary.json)
- [Campaign issue](https://github.com/HangGlidersRule/model-forge/issues/25)

## Evaluation

**Quality setup.** One RTX PRO 6000 Blackwell (SM120), stock vLLM `v0.32.0` and the same serve flags for every cell. The baselines:
- the upstream BF16 checkpoint, run twice for the noise floor;
- the family's [W4A16 build](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A16-NVFP4)
  ("W4A16 build" below), run in the same evaluation window.

GPQA Diamond: `llm-inference-bench` `gpqa-diamond`, all 198 items, thinking on, temperature 0, max 65,536 tokens,
`--reasoning-parser nemotron_v3`. Unparseable and truncated items count as wrong. Confidence intervals come from a paired bootstrap
(10,000 resamples, seed 1234).

| Metric | Value | Basis |
|---|---|---|
| GPQA Diamond | 136/198 (68.7%) | full denominator; 11 unparseable, 12 truncated without an answer, 0 request errors |
| GPQA, upstream BF16 | 138/198 and 141/198 (two runs) | the two runs disagree on 37 items, a noise band of 11.9 questions |
| GPQA, W4A16 build | 136/198 (68.7%) | same harness |
| GPQA delta vs the W4A16 build | 0 questions (95% CI −0.056 to +0.056) | pre-registered pass bar −0.060: PASS |
| Next-token KL vs BF16, FP8 KV as shipped | 0.0358, top-1 agreement 92.81% | 64 held-out prompts, 37,727 positions; prompt logprobs, top 20 plus a remainder bucket |
| Next-token KL vs BF16, BF16 KV | 0.0352, top-1 agreement 92.98% | same prompts; the FP8 KV cache adds 0.0006 |
| Next-token KL, other builds | W4A4 cell without Mixed-FP8 0.0246; W4A16 0.0189 | same prompts |

**Serving setup.** One 32 GB RTX PRO 4500 (SM120), sharing the card with a small speech model as in the deployment it was tested
for. Every run uses vLLM `v0.32.0` with the same flags: DSpark drafter (3 speculative tokens), FP8 KV cache, a 131,072-token window
and 16 sequences.
- **Reference:** [NVIDIA's W4A16 checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) at revision
  `bee7596271d1495f6992ae224aefde4410e816b8`, MoE on Marlin, memory fraction 0.75. It shares this checkpoint's layout outside the
  experts. The family's W4A16 build keeps its Mamba projections and `lm_head` in BF16 and was not measured on this card.
- **Pairing:** each candidate run is paired with a reference run on the same card. The 0.75 run and its reference ran back to back;
  the 0.77 run followed on the same card and is compared with that same reference.
- **Prefill runs** use one output token and a unique prefix per request.

| Serving metric | Value | Basis |
|---|---|---|
| KV-cache pool, memory fraction 0.77 | 201,426 tokens | 19.82 GiB loaded; NVIDIA's W4A16 checkpoint: 179,260 at 0.75 (19.37 GiB loaded) |
| KV-cache pool, memory fraction 0.75 | 151,311 tokens | below one full 131,072-token request plus headroom; use 0.77 or more on a 32 GB card |
| Prefill vs NVIDIA's W4A16 checkpoint, fraction 0.77 | 1.21x, 1.42x, 1.44x, 1.49x | ~3k-token prompts at c=1 and c=8, then ~50k-token prompts at c=1 and c=8 |
| Prefill vs NVIDIA's W4A16 checkpoint, fraction 0.75 | 1.21x, 1.37x, 1.44x, 1.45x | same prompts, back to back with the reference |
| Single-stream decode vs NVIDIA's W4A16 checkpoint | 0.99x at 0.75 (321.3 vs 325.0 tok/s), 1.03x at 0.77 (333.9 tok/s) | median of three 512-token answers, thinking off |
| W4A4 cell without Mixed-FP8, same card at 0.80 | decode 0.81x, prefill 1.38-1.56x | same flags, against its own back-to-back reference run of NVIDIA's W4A16 checkpoint |
| Tool calling, JSON-schema output | pass, pass | one request each |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or protocol.

## Safety and limitations

This is an unedited base quantization. No behavior edit was applied; refusal behavior is the upstream model's.
- **Quality cost.** Next-token KL to BF16 is higher than either sibling cell's (0.0358, against 0.0246 and 0.0189); GPQA matches the
  family's W4A16 build. Prefer that build where answer quality matters more than speed.
- **Static activation scales.** One scale per projection input covers all 128 routed experts. Inputs far outside the calibration text
  (other languages, unusual formats) may degrade more than the measured average.
- **Hardware.** On GPUs without FP4 tensor cores, vLLM falls back to weight-only kernels and the speed gain does not apply.
- **Toolchain.** Built with a ModelOpt release candidate (`0.48.0rc1`), the newest ModelOpt build when this campaign pinned it;
  `0.48.0rc2` was tagged later and is not used. `0.47.0` is the newest stable release.

## Release reference

Engineering release: [`darkstar-nemotron-3.5-lightning-v1.2.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3.5-lightning-v1.2.0).
This immutable tag exists and the release contract is published.

## Runtime

**Validated stack:** vLLM `v0.32.0` (`vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`) on
SM120.
- **MoE experts:** `FLASHINFER_CUTLASS` NVFP4 kernel, vLLM's default for this checkpoint.
- **KV cache:** FP8 E4M3 with the checkpoint's calibrated scales.

These are the 32 GB serving runs' flags, with the measured memory fraction. The DSpark drafter is NVIDIA's
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark).

```bash
vllm serve HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8 \
  --served-model-name darkstar-nemotron35lightningw4a4fp8-base-nvfp4 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --gpu-memory-utilization 0.77 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --speculative-config '{"model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark", "num_speculative_tokens": 3}'
```

Thinking is on by default. Pass `"chat_template_kwargs": {"enable_thinking": false}` per request to turn it off. On larger cards,
raise `--gpu-memory-utilization` for a larger KV pool.

## License

OpenMDW-1.1, inherited from
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16). The
license text ships with this repository as `LICENSE`. This repository contains modified files: quantized weights and quantization
metadata.
