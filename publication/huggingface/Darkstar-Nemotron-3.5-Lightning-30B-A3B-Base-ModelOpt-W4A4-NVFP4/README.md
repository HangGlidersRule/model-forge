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
  - vllm
quantization: nvidia-modelopt
---

# Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A4-NVFP4

## Summary

Unedited NVIDIA ModelOpt NVFP4 quantization of
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
at revision `d468880b6ad3c6e0d21377ce7242adaea4cc884d`, built from the upstream BF16 weights. No behavior edit was applied.

**What it is for.** The MoE experts run real FP4 math (W4A4) on Blackwell GPUs out of the box: stock vLLM selects its FP4 MoE kernel
with no flags or patches.
- **Speed:** prefill runs 1.45-1.72x the family's W4A16 build on vLLM's default kernel.
- **Quality:** next-token KL to BF16 is 0.0246, against 0.0189 for the W4A16 build.

Use the [W4A16 build](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A16-NVFP4) where
answer quality matters more than prefill speed.

**What is quantized.** W4A4 NVFP4 (FP4 E2M1 weights and activations, group 16, FP8 E4M3 block scales) on every MoE expert:
- the up and down projections of the 128 routed experts in all 23 MoE layers;
- the up and down projections of each layer's shared expert;
- 5,934 linears in all.

The router, Mamba projections and conv, attention, `lm_head`, embeddings, norms and the MTP head stay BF16, as does the KV cache. The
checkpoint is 21.4 GiB in 4 shards: three weight shards plus a 0.7 MB shard holding the activation scales.

**How.** ModelOpt `0.48.0rc1` at `ff2450c15eb5f1697ec0471704d5421d54fce4f4`, operator recipe
`nvfp4_experts_w4a4_act_headroom-kv_bf16` (SHA-256 `6facf6cdc5486dba9a402881dae92c6c60d7ca7f8fc4e2bab608abec4948ee68`):
- **Weight scales:** static, MSE FP8-scale sweep.
- **Calibration:** 1,024 texts at up to 2,048 tokens, seed 1234, file SHA-256
  `0a1be3fb7fd44f745213f61f45fb4ce61a69d85a4bf39f0adf4e8957c5d87840`.
  - 512 `abisee/cnn_dailymail` 3.0.0 articles at `96df5e686bee6baa90b8bee7c28b81fa3fa6223d`.
  - 512 `nvidia/Nemotron-Post-Training-Dataset-v2` conversations at `5c89e01dd720ae0f4058445ed49c5fb68a03c76e` (128 each from
    chat, stem, math and code), rendered with the model's own chat template.
- **Activation scales:** one static global scale per projection input per layer, shared by all 128 routed experts. These were
  replaced after calibration (below).

**Activation scales, replaced after calibration.** The calibrated scales (`nvfp4_act_headroom`) placed the routed-expert
down-projection input range at 0.12-0.42x the largest activation the BF16 model produces, so larger activations clipped. That build
scored KL 0.0795. This checkpoint keeps every weight and weight scale and sets each of the 69 activation streams to the global
scale with the lowest simulated NVFP4 error over 256 calibration rows of BF16 activations (229,027 tokens).
- **Selection:** that rule won a pre-registered comparison of four rules on 64 held-out prompts, cutting KL by 69.3%.
- **The other rules:** a fixed scale of 1.0 (0.0249), and the 99.99th (0.0752) and 99.9th (0.1656) percentiles of block amax.

**Full provenance, protocol and evidence:**
- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/model-card/base-w4a4-nvfp4.md)
- [Family record](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/README.md)
- [Build recipe](https://github.com/HangGlidersRule/model-forge/blob/main/recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-base-modelopt-w4a4-nvfp4.yaml)
- [Gate results](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-lightning-r1/results/w4a4-gate-summary.json)
- [Campaign issue](https://github.com/HangGlidersRule/model-forge/issues/17)

## Evaluation

**Setup.** Every cell ran on one RTX PRO 6000 Blackwell (SM120) with stock vLLM `v0.32.0`, the same serve flags, and each checkpoint
on vLLM's default MoE kernel. The baselines are:
- the upstream BF16 checkpoint, run twice for the noise floor;
- the family's [W4A16 build](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A16-NVFP4),
  on vLLM's default Marlin kernel.

**GPQA Diamond** (`llm-inference-bench` `gpqa-diamond`): all 198 items, thinking on, temperature 0, max 65,536 tokens,
`--reasoning-parser nemotron_v3`. Unparseable, truncated and unanswered items count as wrong. Confidence intervals come from a paired
bootstrap (10,000 resamples, seed 1234).

| Metric | Value | Basis |
|---|---|---|
| GPQA Diamond | 130/198 (65.7%) | full denominator; 15 unparseable, 9 truncated without an answer, 4 requests that never connected |
| GPQA, upstream BF16 | 138/198 and 141/198 (two runs) | the two runs disagree on 37 items, a noise band of 11.9 questions |
| GPQA, W4A16 build | 136/198 (68.7%) | same harness |
| GPQA delta vs the W4A16 build | −6 questions (95% CI −0.086 to +0.025); −3 on the 194 items this checkpoint answered | pre-registered pass bar −0.060: PASS |
| Next-token KL vs BF16 | 0.0246, top-1 agreement 94.28% | 64 held-out prompts, 37,727 positions; prompt logprobs, top 20 plus a remainder bucket |
| Next-token KL, other builds | W4A16 0.0189; W4A16 with runtime W4A4 (FlashInfer b12x) 0.0358; calibrated scales 0.0795 | same prompts |
| Prefill, ~3k-token prompts | 46,463 tok/s at c=1 (TTFT 61.5 ms), 56,283 at c=8 | W4A16: 30,002 / 32,761; W4A16 on b12x: 40,077 / 44,633 |
| Prefill, ~50k-token prompts | 36,387 tok/s at c=1 (TTFT 1,375 ms), 37,308 at c=8 | W4A16: 25,015 / 25,252; W4A16 on b12x: 31,250 / 31,883 |
| Speed vs the W4A16 build | 1.45-1.72x (on b12x: 1.16-1.26x) | the four prefill cells above |
| Decode tok/s, MTP speculative decoding | not measured | |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or protocol.

Prefill rows: one output token per request, a unique prefix on every prompt so the prefix cache can't serve it, and TTFT is the median.

## Safety and limitations

This is an unedited base quantization. No behavior edit was applied; refusal behavior is the upstream model's.
- **Quality cost.** Next-token KL to BF16 is 30% higher than the W4A16 build's. GPQA is 6 questions lower, inside run-to-run noise.
  Prefer the W4A16 build where answer quality matters more than prefill speed.
- **Static activation scales.** One scale per projection input covers all 128 routed experts. Inputs far outside the calibration text
  (other languages, unusual formats) may degrade more than the measured average.
- **Hardware.** On GPUs without FP4 tensor cores, vLLM falls back to weight-only kernels and the prefill gain does not apply.
- **Toolchain.** Built with a ModelOpt release candidate (`0.48.0rc1`), the newest ModelOpt build at release time; `0.47.0` was the
  newest stable release.

## Release reference

Engineering release: [`darkstar-nemotron-3.5-lightning-v1.1.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3.5-lightning-v1.1.0).
This immutable tag exists and the release contract is published.

## Runtime

**Validated stack:** vLLM `v0.32.0` (`vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`) on
SM120.
- **MoE experts:** `FLASHINFER_CUTLASS` NVFP4 kernel, vLLM's default for this checkpoint. Shared-expert linears run
  `FlashInferCutlassNvFp4LinearKernel`.
- **Attention:** FlashAttention 2.
- **Mamba state update:** FlashInfer.
- **KV cache:** BF16.

These are the measured runs' flags, with an empty speculative config:

```bash
vllm serve HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A4-NVFP4 \
  --served-model-name darkstar-nemotron35lightningw4a4-base-nvfp4 \
  --trust-remote-code \
  --max-model-len 131072 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --reasoning-parser nemotron_v3
```

Thinking is on by default. Pass `"chat_template_kwargs": {"enable_thinking": false}` per request to turn it off.

## License

OpenMDW-1.1, inherited from
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16). The
license text ships with this repository as `LICENSE`. This repository contains modified files: quantized weights and quantization
metadata.
