---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- autotrust/JEV-27B
base_model_relation: quantized
pipeline_tag: text-classification
tags:
  - darkstar
  - jev-27b
  - base
  - nvfp4
  - modelopt
  - w4a4
  - decision-model
  - vllm
quantization: nvidia-modelopt
---

# Darkstar-JEV-27B-Decision-ModelOpt-W4A4-NVFP4-Mixed-FP8

## Summary

A single checkpoint, with no LoRA at serve time, of the [`autotrust/JEV-27B`](https://huggingface.co/autotrust/JEV-27B)
System 1 decision model, quantized with NVIDIA ModelOpt. Typed decisions (`noul` yes/no, `choice` with up to 16 native
options, `score` 0–5) come back as calibrated probabilities from one forward pass with a one-token readout.

**Merge.** The JEV-27B `adapter_vllm` LoRA at revision `14b425269ed24452e0a5ad5a8f3e51d139647a88` is folded into
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- **The adapter:** rank 32, alpha 64. 401 pairs over attention, linear attention (Gated DeltaNet), MLP and `lm_head`; the `lm_head` pair is the decision head.
- **Base revision check:** JEV-27B's backbone is byte-identical to that revision on every tensor sampled (12, including embeddings and `lm_head`).
- **Merge method:** `W + 2·(B·A)` is computed in FP32 and rounded once to BF16. The merge code reproduces PEFT `merge_and_unload` bit for bit.
- **Unchanged:** the vision tower and MTP head are copied from the base.

**Quantization.** ModelOpt `0.48.0rc1` at `ff2450c15eb5f1697ec0471704d5421d54fce4f4`, using NVIDIA's own Qwen3.8-27B recipe
(`nvfp4_w4a4_mlp_fp8_attn_local_hessian`, the recipe behind `nvidia/Qwen3.8-27B-NVFP4`):
- **NVFP4 W4A4** (FP4 E2M1 weights and activations, group 16, FP8 E4M3 block scales) on the 192 language-model MLP
  linears. Static weight scales come from Local-Hessian with the FP8-scale sweep; activations are dynamic.
- **FP8 W8A8** on the 64 self-attention projections and the 144 linear-attention projections.
- **BF16** for `lm_head`, embeddings, norms, `in_proj_a`/`in_proj_b`, `conv1d`, the vision tower and MTP. The KV cache is BF16.
- **One change from NVIDIA's recipe:** `lm_head` stays BF16, because after the merge it carries the decision head
  (NVIDIA's `lm_head` is NVFP4).

**Calibration.** 512 samples, layerwise, seed 1234, file SHA-256 `58566c6e3d42ef0927fbb9c49005f8b703b257c23ab9dd78509c4dbb87fa30de`:
- 256 decision prompts from `SargeDev/jev-distill-corpus-v3` (calibration split, revision
  `75111a13a8455032f0435050bfc7e0c9026ecfe8`), rendered exactly as the serving shim builds them.
- 256 chat samples from `nvidia/Nemotron-SFT-Instruction-Following-Chat-v2` at
  `1a9454ed054b8544503ab8d8c0a519d141a44c5b`, NVIDIA's calibration source for the base.

**Decisions only.** Folding the `lm_head` LoRA into the backbone changes the model's plain text generation (System 2). This checkpoint is built and measured for System 1 decisions; its generation quality is not measured and is not claimed.

Two builds were compared:

| Build | Change | Result |
|---|---|---|
| **A (this checkpoint)** | NVIDIA's mixed layout: NVFP4 W4A4 MLP, FP8 attention | teacher KL 0.0203, 69.5 rows/s; passes |
| B | Every MLP, attention and linear-attention linear NVFP4 W4A4 | 98.5 rows/s, but teacher KL 0.0220 (worse than the deployed baseline); rejected |

Provenance, protocol and evidence:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/jev-27b/model-card/decision-nvfp4.md)
- [Family record](https://github.com/HangGlidersRule/model-forge/blob/main/models/jev-27b/README.md)
- [Build recipe](https://github.com/HangGlidersRule/model-forge/blob/main/recipes/jev-27b/darkstar-jev-27b-decision-modelopt-w4a4-nvfp4-mixed-fp8.yaml)
- [Gate results](https://github.com/HangGlidersRule/model-forge/blob/main/models/jev-27b/results/decision-gate-summary.json)
- [Campaign issue](https://github.com/HangGlidersRule/model-forge/issues/15)

## Evaluation

Measured on an RTX PRO 6000 Blackwell (SM120) with vLLM `v0.32.0` and the `serve_decide.py` System 1 readout (thinking
off, single pass), on all 29,955 rows of `test_set_30k` from `SargeDev/jev-distill-corpus-v3` (the JEV-27B card's
published evaluation). Every row was answered.

The comparison configurations:
- **BF16 reference:** the unmerged `Qwen/Qwen3.8-27B` BF16 + JEV LoRA. It reproduces the JEV-27B card (teacher KL 0.0186). A second serve of it sets the noise floor: 99.0% decision agreement, mean KL between serves 0.00003.
- **Deployed baseline:** `nvidia/Qwen3.8-27B-NVFP4` with the same LoRA remapped to its module names.

Confidence intervals come from a paired bootstrap (2,000 resamples, seed 1234). Top-1 is scored on the 27,324 rows
whose teacher distribution has a single argmax.

| Metric | Value | Basis |
|---|---|---|
| Teacher KL | 0.0203 (BF16 reference 0.0186; deployed baseline 0.0207) | mean over 29,955 rows; teacher = the corpus target distributions |
| Teacher-KL delta vs the deployed baseline | −0.00039 (95% CI −0.00054 to −0.00024) | paired bootstrap; closer to the teacher |
| Top-1 vs teacher | 0.9219 (BF16 reference 0.9270; deployed baseline 0.9200) | 27,324 single-argmax rows |
| Top-1 deficit vs the deployed baseline | −0.19 pt (95% CI −0.41 to +0.02) | paired bootstrap; bar: upper bound at most 0.5 pt |
| Choice top-1 / yes-no AUROC / score MAE | 0.9050 / 0.9954 / 0.104 | per decision kind |
| Agreement with the BF16 reference | 0.940 (deployed baseline 0.943) | decision flips are near-ties: median BF16 top-2 margin 0.011 |
| Throughput | 69.5 rows/s (deployed baseline 50.1; BF16 45.3) | 16 concurrent `/v1/decide` requests |
| Latency p50 / p95, c=1 | 80 / 89 ms short (~314 tok), 270 / 330 ms long (~3,790 tok) | deployed baseline 107 / 126 and 365 / 441; long prompts are seeded concatenations, since the eval set's longest prompt is 702 tokens |
| Weights / KV cache | 21.5 GiB / 814,421 tokens | `--gpu-memory-utilization 0.85`, 96 GB card |
| Plain text generation (System 2) | not measured | out of scope for a decision checkpoint |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or protocol.

## Safety and limitations

- **Decisions only.** Plain chat or reasoning output from this checkpoint is not measured, and should be assumed worse
  than JEV-27B's unmerged base path: merging an `lm_head` LoRA cost a sibling JEV model 9 HumanEval points.
- **Calibration of the probabilities** was checked against the JEV-27B teacher distributions on `test_set_30k`. Validate on
  your own decision distribution before relying on absolute probabilities.
- **No behavior edit** was applied; refusal behavior is the merged JEV-27B's.
- **Built on a ModelOpt release candidate** (`0.48.0rc1`).

## Release reference

Engineering release: [`darkstar-jev-27b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-jev-27b-v1.0.0).
This immutable tag exists and the release contract is published.

## Runtime

Validated on vLLM `v0.32.0` (`vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`):
- **Kernels:** `FlashInferCutlassNvFp4LinearKernel` (MLP), `FlashInferFP8ScaledMMLinearKernel` (attention).
- **KV cache:** BF16.
- **Flags:** `--max-model-len 32768`, `--max-num-seqs 32`. No `--enable-lora`.

`vllm serve` loads the weights. The `POST /v1/decide` route comes from `serve_decide.py` in this repository: autotrust's
JEV-27B server with a two-line change. With `JEV_DECIDE_MERGED=1` it reads the decision head from `JEV_DECIDE_HEAD_DIR` and sends
System 1 readouts to the base model instead of a LoRA module.

```bash
hf download HangGlidersRule/Darkstar-JEV-27B-Decision-ModelOpt-W4A4-NVFP4-Mixed-FP8 --local-dir jev
JEV_DECIDE_MERGED=1 JEV_DECIDE_HEAD_DIR=jev JEV_DECIDE_CALIBRATION=jev/calibration.json \
python3 jev/serve_decide.py --model jev \
    --served-model-name darkstar-jev27b-base-nvfp4 \
    --max-model-len 32768 --max-num-seqs 32 \
    --logprobs-mode processed_logprobs \
    --trust-request-chat-template
```

`vllm serve` alone loads the weights, without the `/v1/decide` route:

```bash
vllm serve HangGlidersRule/Darkstar-JEV-27B-Decision-ModelOpt-W4A4-NVFP4-Mixed-FP8 \
    --served-model-name darkstar-jev27b-base-nvfp4 --max-model-len 32768
```

```bash
jq -n --arg kind choice --arg state "Invoice total 412.50 EUR, due in 30 days." \
      --arg question "Which currency?" --argjson options '["USD", "EUR", "GBP"]' '$ARGS.named' |
  curl -s localhost:8000/v1/decide -H 'Content-Type: application/json' -d @-
```

## License

Apache-2.0. Built from `Qwen/Qwen3.8-27B` (Apache-2.0) and `autotrust/JEV-27B` (Apache-2.0) with NVIDIA's Model Optimizer
recipe (Apache-2.0). `serve_decide.py` is autotrust's (Apache-2.0), modified as described above.
