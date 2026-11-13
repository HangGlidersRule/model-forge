---
license: apache-2.0
base_model: ricursiveAI/Ripple-65B-Preview
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - darkstar
  - ripple
  - base
  - nvfp4
  - modelopt
  - w4a4
  - fp8
  - vllm
quantization: nvidia-modelopt
---

# Darkstar-Ripple-65B-Preview-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8

## Summary

Unedited NVIDIA ModelOpt quantization of
[`ricursiveAI/Ripple-65B-Preview`](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) at revision
`240d5966cbe00f0627c699e59ef58560b60303cf`, built from the upstream BF16 weights. No behavior edit was applied.

**What it is for.** Ripple-65B Preview on one Blackwell GPU, faster than the vendor's own NVFP4 checkpoint at about the
same quality.
- **Speed:** against [`ricursiveAI/Ripple-65B-Preview-NVFP4`](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-NVFP4)
  ("the vendor NVFP4" below) on the same GPU, image and flags, batch-1 decode runs 1.31x, batch-8 decode 1.24x, and time to
  first token is 0.82x (lower is faster).
- **Quality:** GPQA Diamond mean 138.5 of 198 against the vendor NVFP4's 140.5, inside the pre-registered bar; tool-call
  task success 0.922 against 0.920.
- **Size:** 43.05 GB against the vendor NVFP4's 57.21 GB, so the same memory setting leaves a 419,096-token KV cache
  against 280,176.

**About the name.** `Base` is the release contract's role word for an unedited cell (as opposed to `Abliterated`). This
checkpoint quantizes the vendor's instruction-tuned model; it is not the vendor's separate pretrain-only
`Ripple-65B-Base-Preview`.

**What is quantized.**
- **W4A4 NVFP4** (FP4 E2M1 weights and activations, group 16, FP8 E4M3 block scales, static scales) on the gate, up and
  down projections of all 80 layers: 240 linears.
- **FP8 E4M3 W8A8** (static per-tensor weight and input scales) on the attention q, k, v, o and the sigmoid output gate
  `g_proj` of all 80 layers: 400 linears.
- **NVFP4 weight-only (W4A16)** on `lm_head`.
- **BF16, byte-identical to upstream:** embeddings, norms and `value_res_lambda`. The KV cache is BF16.
- **Size:** 43.05 GB (40.10 GiB) in 5 shards.

**How.** ModelOpt `0.48.0` (torch 2.15.0, transformers 5.18.0), two stages:
- **Decoder layers:** NVIDIA's dense recipe for this layout (Local-Hessian weight scales with the FP8-scale sweep), calibrated
  layerwise with the BF16 weights offloaded to disk. Recipe SHA-256
  `624185c8df017e0130979905a36885bd761131baae6d6535472fa9d3515f9e4e`.
- **`lm_head`:** data-free MSE weight scales with the FP8-scale sweep. Recipe SHA-256
  `123e4e99af9634a16d5b6c467e8341ad6c0e72b93038f94fc2181b79f61e74f6`.
- **Calibration:** 512 calibration rows (927,103 tokens) rendered with the checkpoint's own chat template: news
  summaries, instruction following with reasoning, agentic tool-use traces and function calling, each from a pinned public
  revision (file SHA-256 `403730e417b94a5e412144ee72b47975c3996c61abf8a32ac7580cf0c73ca508`).
- **Validation:** fail-closed on the shipped files.
  - Quantized: 240 NVFP4 W4A4, 400 FP8 and 1 W4A16 NVFP4 linear, exactly the design.
  - Scales: 3,177,840,640 FP8 block scales and 1,281 per-tensor scales finite and positive; fused groups share their
    scales.
  - Untouched: 562 BF16 tensors byte-identical to the source; remote code, chat template and tokenizer byte-identical.

**Full provenance, protocol and evidence:**
- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/ripple-65b/model-card/base-nvfp4.md)
- [Family record](https://github.com/HangGlidersRule/model-forge/blob/main/models/ripple-65b/README.md)
- [Build recipe](https://github.com/HangGlidersRule/model-forge/blob/main/recipes/ripple-65b/darkstar-ripple-65b-preview-base-modelopt-w4a4-nvfp4-mixed-fp8.yaml)
- [Gate results](https://github.com/HangGlidersRule/model-forge/blob/main/models/ripple-65b/results/gate-summary.json)
- [Campaign issue](https://github.com/HangGlidersRule/model-forge/issues/39)

## Evaluation

**Setup.** One RTX PRO 6000 Blackwell (SM120, 96 GB, power limit 450 W), vLLM nightly `0.34.0rc1.dev206+gf5d93296c`
(see Runtime), the serve command below for every cell, one serve at a time. This checkpoint is the campaign's
candidate A. The reference is the vendor NVFP4 at revision
`99eea4477a001a41b86914e41feb4b2ed86105e5`: W4A4 NVFP4 on the MLP, attention, `lm_head` and embeddings in BF16.

**No BF16 serve.** The BF16 checkpoint (129.57 GB) does not fit the 96 GB GPU, so GPQA and the tool-call proxy were not
run on it. The vendor reports GPQA Diamond 85.2 for BF16 (vendor-reported, their own harness; not comparable with the
counts below and not measured here). Closeness to BF16 is measured instead as next-token KL against a BF16 teacher: the
checkpoint's own modules run in FP32 on CPU, one layer at a time, over 64 held-out prompts disjoint from the calibration
rows.

GPQA Diamond: `llm-inference-bench` 0.4.29 `gpqa-diamond`, all 198 items, temperature 0, max 65,536 tokens,
concurrency 30, two runs per cell. Unparseable and truncated items count as wrong. The bar was fixed before any
candidate ran: the candidate's mean of two runs at least the vendor NVFP4's mean minus 11.88 questions (6% of 198).

| Metric | Value | Basis |
|---|---|---|
| GPQA Diamond | 138 and 139 of 198 (mean 138.5) | two runs; bar: at least the vendor's mean − 11.88: PASS |
| GPQA Diamond, vendor NVFP4 | 142 and 139 of 198 (mean 140.5) | same harness, same window |
| GPQA, A minus the vendor NVFP4 | paired bootstrap 95% CI of (A − vendor) -11.5 to +7.5 questions | items averaged over the two runs, 10,000 resamples |
| GPQA truncations at 65,536 tokens | 42 and 48 of A's items, 42 and 38 of the vendor's | counted as wrong; a reasoning block on 198 of 198 items in every run of both checkpoints |
| Tool-call parse rate | 0.996 (vendor 0.998) | 500 items; bar: at least the vendor's − 2 points: PASS |
| Tool-call task success | 0.922 (vendor 0.920) | bar: at least 0.98 × the vendor's: PASS |
| Tool-call items | BFCL v3 AST 362/400 (vendor 360/400); terminal and file-edit tasks 99/100 (vendor 100/100) | BFCL's AST checker; synthetic tasks run in a sandbox |
| Next-token KL to the BF16 teacher | 0.0371 (vendor 0.0290; 1.28x, 95% CI of the difference +0.0064 to +0.0099) | 64 held-out prompts, 116,600 positions; top 20 plus a remainder bucket |
| Top-1 agreement with the BF16 teacher | 94.3% (vendor 95.0%) | same positions |
| KL on tool-call spans | 0.0043 (vendor 0.0031) over 1,139 positions | the 16 agentic prompts' assistant tool-call turns |

The higher KL comes from the FP8 attention, which the vendor NVFP4 leaves in BF16; the KL is tracked, not a bar.

**Speed.** Two serves of each checkpoint, alternating vendor, this checkpoint, vendor, this checkpoint; per serve 2
warm-ups, then 20 rounds per cell, keep-alive clients, 0 reconnects. Values are per-serve medians, serve 1 / serve 2; the
ratio divides the means of the two serves.

| Serving metric | Value | Basis |
|---|---|---|
| Decode, batch 1 | 37.11 / 36.85 tok/s (vendor 27.41 / 29.15): 1.31x | 512 output tokens, no reasoning |
| Decode, batch 8 | 224.3 / 226.6 tok/s aggregate (vendor 182.3 / 181.9): 1.24x | 8 concurrent requests; median aggregate over 20 rounds |
| Time to first token, ~1k-token prompt | 0.215 / 0.217 s (vendor 0.264 / 0.265): 0.82x | batch 1 |
| Time to first token, ~16k-token prompt | 3.36 / 3.37 s (vendor 4.08 / 4.09): 0.82x | batch 1 |
| Time to first token, ~16k-token prompts, batch 8 | 17.1 / 17.2 s (vendor 21.0 / 20.9): 0.82x | median over 160 requests (8 concurrent, 20 rounds) |
| With DFlash2, decode, batch 1 | 218.7 / 223.3 tok/s (vendor 180.5 / 179.2): 1.23x | 7 draft tokens |
| With DFlash2, decode, batch 8 | 833.4 / 834.9 tok/s aggregate (vendor 708.5 / 683.2): 1.20x | 7 draft tokens |
| DFlash2 mean acceptance length | 7.17 / 7.20 (vendor 7.21 / 7.16) | accepted tokens per draft step plus one |
| Selection bar | batch-1 decode 36.98 against the vendor's 28.28 (1.31x; bar 1.05x and above both vendor serves): PASS | pre-registered speed win |
| Weights in memory | 40.76 GiB loaded (vendor 53.86) | vLLM's load line |
| KV-cache pool at memory fraction 0.92 | 419,096 tokens (vendor 280,176) | 131,072-token context |
| KV-cache pool with DFlash2 | 272,872 tokens with DFlash2 (vendor 156,629) | same fraction |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or protocol.

## Safety and limitations

This is an unedited base quantization. No behavior edit was applied; refusal behavior is the upstream model's.
- **Quality cost.** KL to the BF16 teacher is 1.28x the vendor NVFP4's. GPQA and tool-call results are within the
  pre-registered bars, but the vendor NVFP4 is closer to BF16; prefer it where closeness matters more than speed.
- **Static activation scales.** One calibrated scale per projection input. Inputs far outside the calibration mix (other
  languages, unusual formats) may degrade more than the measured average.
- **Preview runtime.** The numbers come from one vLLM nightly on one GPU (see Runtime). Other GPUs and runtimes are not
  measured.
- **Hardware.** On GPUs without FP4 tensor cores vLLM falls back to weight-only kernels, and the speed gain does not apply.

## Release reference

Engineering release: [`darkstar-ripple-65b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-ripple-65b-v1.0.0).
This immutable tag exists and the release contract is published.

## Runtime

**Validated stack:** vLLM nightly `0.34.0rc1.dev206+gf5d93296c`, pinned by digest
(`vllm/vllm-openai@sha256:902b49de0568ba5a5c951b0243bc06ef8541f46bc3eb80832961fefb22a19034`, tag
`nightly-f5d93296c8ec82b7e7dabdb308d24d43cb1907bf`), on SM120.
- **Why a nightly:** Ripple support (`RicursiveForCausalLM`, the `ripple` reasoning and tool-call parsers and the DFlash2
  drafter) landed in vLLM #65745 after the last stable release. The `v0.34.0rc1` git tag (`52aeb58`) contains it, but no
  Docker image has been published for that tag yet. The nightly above was built 390 commits after the merge.
- **Re-check:** the release follows the first vLLM release image that contains #65745 and is re-checked on it.
- **Kernels:** vLLM loads the checkpoint as `modelopt_mixed`: FlashInfer FP8 GEMMs for attention, the FlashInfer CUTLASS
  NVFP4 GEMM for the MLP, and Marlin for the weight-only `lm_head`.
- **Remote code:** none needed. vLLM ships its own Ripple config and model classes, so the serve does not pass
  `--trust-remote-code`.

The measured serve command:

```bash
vllm serve HangGlidersRule/Darkstar-Ripple-65B-Preview-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8 \
  --served-model-name darkstar-ripple65b-base-nvfp4 \
  --reasoning-parser ripple \
  --tool-call-parser ripple \
  --enable-auto-tool-choice \
  --max-model-len 131072 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.92
```

Prefix caching is on (vLLM's default for this model). The chat template has no thinking switch; the model decides per
turn whether to open a reasoning block, and the `ripple` parser returns it as `reasoning_content`.

**DFlash2 speculative decoding.** The vendor's drafter
[`ricursiveAI/Ripple-65B-Preview-DFlash2`](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2) (revision
`35609b3be3dd08a822b3ba1257c1ea5dd6f87632`) works with this checkpoint unchanged. Add this flag to the command above:

```bash
--speculative-config '{"method":"dflash","model":"ricursiveAI/Ripple-65B-Preview-DFlash2","revision":"35609b3be3dd08a822b3ba1257c1ea5dd6f87632","num_speculative_tokens":7}'
```

The drafter adds about 8 GB, which shrinks the KV pool as the Serving table shows.

## License

Apache-2.0, inherited from
[`ricursiveAI/Ripple-65B-Preview`](https://huggingface.co/ricursiveAI/Ripple-65B-Preview). The license text ships with
this repository as `LICENSE`, byte-identical to the upstream's. This repository contains modified files: quantized
weights and quantization metadata.
