# Darkstar Nemotron-3.5-Rerank-8B

Reranker family record. Source:
[`nvidia/Nemotron-3.5-Rerank-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16), NVIDIA's multimodal
cross-encoder.
- **Model:** a Pixtral vision tower and a bidirectional Ministral3 text tower (34 layers, hidden 4096), 8.38B parameters.
- **Scoring:** a mean-pooled score head reads a query–document pair and outputs one raw relevance logit.
- **Tracking:** proposed and tracked in [issue #32](https://github.com/HangGlidersRule/model-forge/issues/32).

## Shipped cells

| Cell | Repository / source | Status |
|---|---|---|
| Base BF16 (control) | [`nvidia/Nemotron-3.5-Rerank-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16) @ `f88f8d9104726747791283ab156d2fcdb29e90a7` | unchanged upstream reference; no HangGlidersRule repository |
| Base W4A4-NVFP4 | [`Darkstar-Nemotron-3.5-Rerank-8B-Base-ModelOpt-W4A4-NVFP4`](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3.5-Rerank-8B-Base-ModelOpt-W4A4-NVFP4) | published |
| Abliterated cells | — | not applicable: a relevance scorer has no refusal behavior to edit |

Release: tag [`darkstar-nemotron-3.5-rerank-8b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3.5-rerank-8b-v1.0.0).
Family collection: [Darkstar Nemotron-3.5-Rerank-8B](https://huggingface.co/collections/HangGlidersRule/darkstar-nemotron-35-rerank-8b-6aeac312954418631a63b91d).

## Documents

- [`model-card/base-nvfp4.md`](model-card/base-nvfp4.md): source card for the published checkpoint
- [`model-card/base-bf16.md`](model-card/base-bf16.md): the upstream BF16 control
- [`results/`](results/): the gate summary for all three candidates, artifact validation, and the publication ledger.
  The gate summary covers each candidate against BF16 and a BF16 replica on both eval sets, the speed runs, and the
  Qwen3-Reranker-4B comparison.
- [`calibration/`](calibration/): deterministic builder for the 512-pair calibration set
- [`eval/`](eval/): the eval harness, which builds the LMEB memory set, checks endpoint agreement, scores quality and
  measures speed. Also here: the fail-closed export validator, the quality comparison, and the evidence builder
  that writes `results/`.
- [`scripts/`](scripts/): the build script, the recipe dry run, and the three run phases (smoke test, builds and quality,
  speed)

## Recipes

- [`../../recipes/nemotron-3.5-rerank-8b/darkstar-nemotron-3.5-rerank-8b-base-modelopt-w4a4-nvfp4.yaml`](../../recipes/nemotron-3.5-rerank-8b/darkstar-nemotron-3.5-rerank-8b-base-modelopt-w4a4-nvfp4.yaml):
  schema-2 build recipe for the published cell
- [`../../configs/modelopt/recipes/nemotron-3.5-rerank-8b/`](../../configs/modelopt/recipes/nemotron-3.5-rerank-8b/):
  the three ModelOpt operator recipes compared (A, B, C)
- [`../../configs/modelopt/pins/modelopt-0.48.0.json`](../../configs/modelopt/pins/modelopt-0.48.0.json) and
  [`../../containers/modelopt/nemotron-3.5-rerank-8b/Dockerfile`](../../containers/modelopt/nemotron-3.5-rerank-8b/Dockerfile):
  the ModelOpt pin and build image for this campaign

## Candidate selection

**Builds.** All three candidates quantize the text tower from the upstream BF16 weights:
- ModelOpt `0.48.0`;
- static NVFP4 weights: group 16, MSE FP8-scale sweep;
- dynamic NVFP4 activations calibrated with `nvfp4_act_headroom`.

**Gate.** Each was scored against BF16 and a second BF16 serve (the noise floor) on two frozen eval sets:
- the six MTEB Nano sets from #14 (299 queries, 8,970 pairs);
- an LMEB Dialogue memory set (540 queries, 16,200 pairs).

The pre-registered bar was at least 98% of BF16 macro nDCG@10 on each set, with the fastest passing candidate shipping.

| Candidate | Text tower | 6 MTEB Nano sets (Δ, 95% CI) | LMEB Dialogue (Δ, 95% CI) | Pairs/s c=1 / 4 / 8 | Weights |
|---|---|---|---|---|---|
| BF16 control | — | 0.7349 | 0.8089 | 17.4 / 18.0 / 17.9 | 15.74 GiB |
| BF16 replica | — | −0.0001 (−0.0012, +0.0006) | −0.0003 (−0.0009, +0.0001) | — | — |
| **A (published)** | MLP + attention W4A4 | **−0.0004 (−0.0055, +0.0049)** | **−0.0033 (−0.0096, +0.0027)** | **51.5 / 55.7 / 51.9** | **5.77 GiB** |
| B | MLP W4A4, attention FP8 | +0.0003 (−0.0032, +0.0035) | −0.0056 (−0.0107, −0.0008) | 45.6 / 48.6 / 46.7 | 6.36 GiB |
| C | MLP W4A4 | +0.0004 (−0.0039, +0.0049) | −0.0071 (−0.0126, −0.0018) | 37.4 / 37.5 / 38.1 | 7.69 GiB |

All three passed. A was published: it is the fastest (2.90-3.09x BF16), and its confidence interval includes zero on both
sets. B and C lose a small, measurable amount on the memory set.

**Comparison with Qwen3-Reranker-4B.** Both [Qwen3-Reranker-4B](../qwen3-reranker-4b/README.md) builds ran on the same
two sets in the same session, as a reference:

| | 6 MTEB Nano sets | LMEB Dialogue | Pairs/s c=1 / 4 / 8 |
|---|---|---|---|
| Qwen3-Reranker-4B BF16 | 0.7121 | 0.7544 | 34.2 / 35.4 / 33.4 |
| Qwen3-Reranker-4B Darkstar W4A4 | 0.7033 | 0.7486 | 59.4 / 65.7 / 65.0 |
| **Nemotron-3.5-Rerank-8B A** | **0.7344** | **0.8056** | **51.5 / 55.7 / 51.9** |

Paired against the Qwen3-Reranker-4B BF16 build, A scores:
- **6 MTEB Nano sets:** +0.0223 (95% CI +0.0038 to +0.0408). A wins 4 of the 6 sets.
- **LMEB Dialogue:** +0.0512 (95% CI +0.0365 to +0.0659). A wins 8 of the 9 splits.

## Gotchas

- **Bidirectional text tower.** `config.json` sets `is_causal: false`. vLLM serves it as a non-causal pooling model,
  so it keeps no per-token KV cache. Calibration ran the checkpoint's own classifier, so the attention matched the
  served attention.
- **Export probe.** ModelOpt 0.48.0's export runs a dummy forward with no attention mask, and this checkpoint's mean
  pooling divides by the mask's sum. The build script gives that probe an all-ones mask. Calibration and serving always
  pass the real mask.
- **Export repairs.** These come from the exporter, not the quantization:
  - **Missing file:** the exporter skipped the sentence-transformers file `2_Dense/model.safetensors`.
  - **Rewritten config:** transformers 5.18 rewrote `config.json`. It dropped default-valued keys and renamed
    `vision_config.rope_parameters.rope_type` from `default` to `axial`.
  - **Repair:** the build restores the upstream `config.json` plus `quantization_config`, and copies missing files byte
    for byte.
  - **Validator:** fails on any other difference.
- **vLLM exclusions.** vLLM 0.32's pooling adapter keeps a placeholder `lm_head` and names the score head `score`.
  - **Exclusion list:** `lm_head`, `language_model.lm_head`, `score` and `language_model.score` must all be on it.
  - **Without them:** vLLM builds NVFP4 layers for those heads and fails at load with `Tensor.item() cannot be called on
    meta tensors`.
- **Scores are raw logits** (`use_activation: false`, as NVIDIA's card serves it). Ranking does not depend on this.
  `"use_activation": true` applies a sigmoid for clients that expect 0–1 scores.
- **Endpoints.** On text pairs, `/v1/score`, `/v1/rerank` and `/classify` return identical scores. NVIDIA's card
  recommends `/classify` for image documents; image documents are not evaluated here.

## License

OpenMDW-1.1, inherited from upstream; the published repository ships the license text and NVIDIA's notices.
