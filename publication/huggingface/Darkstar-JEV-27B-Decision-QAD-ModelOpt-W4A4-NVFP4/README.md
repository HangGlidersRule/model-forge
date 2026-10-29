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
  - qad
  - decision-model
  - vllm
quantization: nvidia-modelopt
---

# Darkstar-JEV-27B-Decision-QAD-ModelOpt-W4A4-NVFP4

## Summary

The fastest Darkstar build of the [`autotrust/JEV-27B`](https://huggingface.co/autotrust/JEV-27B) System 1 decision model.
- **What it is:** a single checkpoint with no LoRA at serve time.
- **Precision:** every language-model linear except `lm_head` runs NVFP4 W4A4.
- **Training:** its FP4 block scales were trained with NVIDIA ModelOpt quantization-aware distillation (QAD).
- **What it returns:** typed decisions as calibrated probabilities, from one forward pass with a one-token readout:
  `noul` (yes/no), `choice` (up to 16 native options) and `score` (0–5).

**Which JEV-27B build to use.** The family ships two decision builds:
- **This checkpoint** trades a little calibration for speed. At one request at a time its p50 latency is 15–32% lower
  than the mixed build's: 64 against 94 ms on ~314-token prompts and 210 against 250 ms on ~3.8k-token prompts at 600 W,
  78 against 100 ms and 253 against 302 ms at 450 W.
- Its test-set teacher KL is 2.0% higher than the mixed build's, and its out-of-distribution teacher KL is 17% higher.
  Decisions agree: top-1 is within 0.15 pt in distribution and 0.5 pt out of it.
- **[`Darkstar-JEV-27B-Decision-ModelOpt-W4A4-NVFP4-Mixed-FP8`](https://huggingface.co/HangGlidersRule/Darkstar-JEV-27B-Decision-ModelOpt-W4A4-NVFP4-Mixed-FP8)**
  is the closer match to the teacher.

**Merge.** Same merged weights as the mixed build:
- **Inputs:** the JEV-27B `adapter_vllm` LoRA at revision `14b425269ed24452e0a5ad5a8f3e51d139647a88`, folded into
  [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
  The adapter is rank 32, alpha 64, with 401 pairs; the `lm_head` pair is the decision head.
- **Method:** `W + 2·(B·A)` is computed in FP32 and rounded once to BF16. On a reduced model it matches PEFT
  `merge_and_unload` exactly (max abs diff 0.0; `results/merge-peft-check.json`).
- **Unchanged:** the vision tower and MTP head are copied from the base.

**Quantization (start point).** ModelOpt `0.48.0rc1` at `ff2450c15eb5f1697ec0471704d5421d54fce4f4`. This is the mixed
build's recipe with the attention layers moved to NVFP4:
- **NVFP4 W4A4:** FP4 E2M1 weights and activations, group 16, FP8 E4M3 block scales. It covers the 192 MLP linears, the
  64 self-attention projections and the 144 linear-attention (Gated DeltaNet) projections.
- **Scales:** static weight scales come from Local-Hessian with the FP8-scale sweep. Activations are dynamic.
- **Calibration:** the mixed build's 512 samples. That is 256 decision prompts from `SargeDev/jev-distill-corpus-v3`
  and 256 chat samples from `nvidia/Nemotron-SFT-Instruction-Following-Chat-v2`, file SHA-256
  `58566c6e3d42ef0927fbb9c49005f8b703b257c23ab9dd78509c4dbb87fa30de`.
- **BF16:** `lm_head` (the decision head), embeddings, norms, `in_proj_a`/`in_proj_b`, `conv1d`, the vision tower and
  MTP. The KV cache is BF16.

**QAD (what changed the scales).** Only the per-block NVFP4 weight scales were trained. Everything else is frozen: the
BF16 weights, every per-tensor scale, the activation scales and every BF16 module.
- **Toolchain:** ModelOpt main at `b6b0ff5249109729c515cdc739a50d518e9cbec0` (LSQ, tied pre/post amax, fp32 master values).
- **Objective:** NVIDIA's QAD logits loss. That is full-vocabulary KL from the merged BF16 model to the NVFP4 student at
  every prompt position. The teacher is the same weights with every quantizer bypassed.
- **Data:** the `train` split of `SargeDev/jev-distill-corpus-v3` at `75111a13a8455032f0435050bfc7e0c9026ecfe8`. It has no
  id or content overlap with the evaluation set.
- **Schedule:** AdamW, lr 1e-4, 596 steps of ~16k prompt tokens (9.8M tokens, 5.8 h on one RTX PRO 6000).
- **Checkpoint choice:** step 500, the lowest validation teacher KL on a fixed 2,000-row validation subset. Validation
  teacher KL went from 0.0245 to 0.0222.
- **Export:** the trained scales are folded back into the standard ModelOpt NVFP4 format.
  - All 1,588 untrained tensors are byte-identical to the start point.
  - QAD changed 23.6% of the 1.52B FP8 block scales from the start point.
  - 355 of 1.52B block scales differ from the trained forward. Each one is a value on the exact midpoint between two
    FP8 codes, rounded to the other neighbour (`results/qad-gate-summary.json`).

**Decisions only.** Folding the `lm_head` LoRA into the backbone changes the model's plain text generation (System 2). This checkpoint is built and measured for System 1 decisions. Its generation quality is not measured and is not claimed.

**Shipped by owner decision.** This build does not pass the family's frozen non-inferiority bar (see Evaluation). It was
shipped alongside the mixed build under a looser tolerance for its speed.

Provenance, protocol and evidence:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/jev-27b/model-card/decision-qad-nvfp4.md)
- [Family record](https://github.com/HangGlidersRule/model-forge/blob/main/models/jev-27b/README.md)
- [Build recipe](https://github.com/HangGlidersRule/model-forge/blob/main/recipes/jev-27b/darkstar-jev-27b-decision-qad-modelopt-w4a4-nvfp4.yaml)
- [QAD results](https://github.com/HangGlidersRule/model-forge/blob/main/models/jev-27b/results/qad-gate-summary.json)
- [Campaign issue](https://github.com/HangGlidersRule/model-forge/issues/20)

## Evaluation

**Setup.** Measured on an RTX PRO 6000 Blackwell (SM120) with vLLM `v0.32.0` and the `serve_decide.py` System 1 readout
(thinking off, single pass).
- **Test set:** all 29,955 rows of `test_set_30k` from `SargeDev/jev-distill-corpus-v3` (the JEV-27B card's published
  evaluation). Every row was answered.
- **Out of distribution (`ood`):** the corpus `ood` split, 13,058 rows.
- **Comparisons:** the deployed baseline is `nvidia/Qwen3.8-27B-NVFP4` with the JEV LoRA remapped to its module names.
  The BF16 reference is the unmerged `Qwen/Qwen3.8-27B` + JEV LoRA (teacher KL 0.0186).
- **Statistics:** confidence intervals come from a paired bootstrap (2,000 resamples, seed 1234). Top-1 is scored on the
  27,324 rows whose teacher distribution has a single argmax.

| Metric | Value | Basis |
|---|---|---|
| Teacher KL | 0.0207 (mixed build 0.0203; deployed baseline 0.0207; QAD start point 0.0220) | mean over 29,955 rows; teacher = the corpus target distributions |
| Teacher-KL delta vs the deployed baseline | +0.00001 (95% CI −0.00018 to +0.00019) | paired bootstrap; a statistical tie |
| Top-1 vs teacher | 0.9205 (mixed build 0.9219; deployed baseline 0.9200) | 27,324 single-argmax rows |
| Top-1 deficit vs the deployed baseline | −0.05 pt (95% CI −0.28 to +0.18) | paired bootstrap |
| Choice top-1 / yes-no AUROC / score MAE | 0.9067 / 0.9953 / 0.105 | per decision kind |
| Agreement with the BF16 reference | 0.946 (mixed build 0.940; deployed baseline 0.943) | all 29,955 rows |
| `ood` teacher KL | 0.1192 (mixed build 0.1019) | 13,058 rows; delta +0.0173 (95% CI +0.0123 to +0.0227) |
| `ood` top-1 / choice top-1 / yes-no AUROC | 0.9538 / 0.9104 / 0.9954 (mixed build 0.9584 / 0.9152 / 0.9965) | top-1 on the 12,775 single-argmax `ood` rows; choice top-1 and AUROC per decision kind |
| Latency p50, c=1, 600 W | 64 ms short (~314 tok), 210 ms long (~3.8k tok) | mixed build 94 / 250 ms, QAD start point 77 / 209 ms; same window |
| Latency p50, c=1, 450 W | 78 ms short, 253 ms long | mixed build 100 / 302 ms, QAD start point 79 / 257 ms; same window |
| Throughput, first 6,000 test rows, c=16 | 66.8 rows/s at 600 W, 60.9 at 450 W | mixed build 60.1 / 54.5; this subset's prompts are about twice the set average |
| Frozen non-inferiority bar | fails: the teacher-KL CI upper bound is above 0 | `gate_noninferiority.py`, SHA-256 `75fcb8147e10b1bb606d9c9cc80e6f8d88aa668cc4fac84c9354986b614222de` |
| Weights / KV cache | 18.7 GiB / 857,429 tokens | `--gpu-memory-utilization 0.85`, 96 GB card |
| Plain text generation (System 2) | not measured | out of scope for a decision checkpoint |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or protocol.

Speed was measured with each build served back to back under one power limit. The power limit moves these numbers by up
to 27%, so compare builds only within one row.

## Safety and limitations

- **Decisions only.** Plain chat or reasoning output from this checkpoint is not measured. Assume it is worse than
  JEV-27B's unmerged base path: merging an `lm_head` LoRA cost a sibling JEV model 9 HumanEval points.
- **Calibration outside the training distribution.** QAD fit the scales to the JEV corpus. On the corpus `ood` split,
  teacher KL is 17% higher than the mixed build's, with top-1 within 0.5 pt. If you use absolute probabilities
  (thresholds, abstention) on unfamiliar decision types, prefer the mixed build or validate on your own data.
- **No behavior edit** was applied; refusal behavior is the merged JEV-27B's.
- **Built on ModelOpt pre-release code:** `0.48.0rc1` for the start point, and main at `b6b0ff5` for QAD.

## Release reference

Engineering release: [`darkstar-jev-27b-v1.1.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-jev-27b-v1.1.0).
This immutable tag exists and the release contract is published.

## Runtime

Validated on vLLM `v0.32.0` (`vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`):
- **Kernels:** `FlashInferCutlassNvFp4LinearKernel` on every quantized linear.
- **KV cache:** BF16.
- **Flags:** `--max-model-len 32768`, `--max-num-seqs 32`. No `--enable-lora`.

`vllm serve` loads the weights. The `POST /v1/decide` route comes from `serve_decide.py` in this repository: autotrust's
JEV-27B server with a two-line change. With `JEV_DECIDE_MERGED=1` it reads the decision head from `JEV_DECIDE_HEAD_DIR` and sends
System 1 readouts to the base model instead of a LoRA module.

```bash
hf download HangGlidersRule/Darkstar-JEV-27B-Decision-QAD-ModelOpt-W4A4-NVFP4 --local-dir jev
JEV_DECIDE_MERGED=1 JEV_DECIDE_HEAD_DIR=jev JEV_DECIDE_CALIBRATION=jev/calibration.json \
python3 jev/serve_decide.py --model jev \
    --served-model-name darkstar-jev27bqad-base-nvfp4 \
    --max-model-len 32768 --max-num-seqs 32 \
    --logprobs-mode processed_logprobs \
    --trust-request-chat-template
```

`vllm serve` alone loads the weights, without the `/v1/decide` route:

```bash
vllm serve HangGlidersRule/Darkstar-JEV-27B-Decision-QAD-ModelOpt-W4A4-NVFP4 \
    --served-model-name darkstar-jev27bqad-base-nvfp4 --max-model-len 32768
```

```bash
jq -n --arg kind choice --arg state "Invoice total 412.50 EUR, due in 30 days." \
      --arg question "Which currency?" --argjson options '["USD", "EUR", "GBP"]' '$ARGS.named' |
  curl -s localhost:8000/v1/decide -H 'Content-Type: application/json' -d @-
```

## License

Apache-2.0. Built from `Qwen/Qwen3.8-27B` (Apache-2.0) and `autotrust/JEV-27B` (Apache-2.0) with NVIDIA's Model Optimizer
(Apache-2.0). `serve_decide.py` is autotrust's (Apache-2.0), modified as described above.
