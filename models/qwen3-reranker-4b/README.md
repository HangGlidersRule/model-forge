# Darkstar Qwen3-Reranker-4B

Reranker family record. Source:
[`Qwen/Qwen3-Reranker-4B`](https://huggingface.co/Qwen/Qwen3-Reranker-4B), a 4B dense Qwen3 decoder
used as a cross-encoder. It reads a query–document pair and scores relevance from the `yes` / `no`
logits. Proposed and tracked in [issue #14](https://github.com/HangGlidersRule/model-forge/issues/14).

## Shipped cells

| Cell | Repository / source | Status |
|---|---|---|
| Base BF16 (control) | [`Qwen/Qwen3-Reranker-4B`](https://huggingface.co/Qwen/Qwen3-Reranker-4B) @ `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | unchanged upstream reference; no HangGlidersRule repository |
| Base W4A4-NVFP4 | [`Darkstar-Qwen3-Reranker-4B-Base-ModelOpt-W4A4-NVFP4`](https://huggingface.co/HangGlidersRule/Darkstar-Qwen3-Reranker-4B-Base-ModelOpt-W4A4-NVFP4) | published |
| Abliterated cells | — | not applicable: a relevance scorer has no refusal behavior to edit |

Release: tag [`darkstar-qwen3-reranker-4b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-qwen3-reranker-4b-v1.0.0).
Family collection: [Darkstar Qwen3-Reranker-4B](https://huggingface.co/collections/HangGlidersRule/darkstar-qwen3-reranker-4b-6ae0bb816d86e0d4f422d3da).

## Documents

- [`model-card/base-nvfp4.md`](model-card/base-nvfp4.md): source card for the published checkpoint
- [`results/`](results/): the reranker gate results (all three candidates against BF16 and a BF16 replica),
  the published-serving-profile re-run, artifact validation, and the publication ledger
- [`calibration/`](calibration/): deterministic builder for the 512-pair MS MARCO calibration set
- [`eval/`](eval/): the BM25 hard-negative eval-set builder and the gate harness

## Recipes

- [`../../recipes/qwen3-reranker-4b/darkstar-qwen3-reranker-4b-base-modelopt-w4a4-nvfp4.yaml`](../../recipes/qwen3-reranker-4b/darkstar-qwen3-reranker-4b-base-modelopt-w4a4-nvfp4.yaml):
  schema-2 build recipe for the published cell
- [`../../configs/modelopt/recipes/qwen3-reranker-4b/`](../../configs/modelopt/recipes/qwen3-reranker-4b/):
  the three ModelOpt operator recipes compared (A, B, C)
- [`../../configs/modelopt/pins/modelopt-0.48.0rc1.json`](../../configs/modelopt/pins/modelopt-0.48.0rc1.json):
  the ModelOpt pin for this campaign

## Candidate selection

All three candidates quantize from the upstream BF16 weights with ModelOpt `0.48.0rc1`, static NVFP4
weights (group 16, MSE FP8-scale sweep) and dynamic NVFP4 activations. Each was scored against BF16 and
a second BF16 serve (the noise floor) on the same eval set: 6 MTEB Nano sets, 299 queries, 8,970 pairs,
BM25 hard negatives, documents untruncated.

| Candidate | Coverage | Activation calibration | Macro security@hangglidersrule.com | Delta vs BF16 (95% CI) | Pairs/s at c=8 |
|---|---|---|---|---|---|
| BF16 control | — | — | 0.7118 | — | 45.9 |
| BF16 replica | — | — | 0.7110 | −0.0008 (−0.0041, +0.0014) | 40.5\* |
| A | decoder MLP | max | 0.7020 | −0.0099 (−0.0206, +0.0002) | 82.8 |
| B | decoder MLP + attention | max | 0.6913 | −0.0206 (−0.0323, −0.0094) | 69.3\* |
| **C (published)** | decoder MLP | `nvfp4_act_headroom` | **0.7095** | **−0.0024 (−0.0120, +0.0073)** | **82.8** |

\*c=8 runs that overlapped another workload on the same GPU; their c=4 figures (46.8 and 119.4) are clean.

C was published: its loss is within the noise and it doubles BF16 throughput. B is the fastest
(2.5x BF16 at c=4) but loses measurable ranking quality.

## Gotchas

- **Chat template.** In vLLM `v0.32.0`, `/v1/score` does not apply the checkpoint's
  `chat_template.jinja` on its own. Without `--chat-template`, pairs are scored without the judge prompt
  and security@hangglidersrule.com collapses (to 0.15–0.41 on these sets). The card's serve command passes it explicitly.
- **Classifier rows.** The model ties `lm_head` to `embed_tokens`, and vLLM builds the classifier from
  the `yes` / `no` rows. Keeping embeddings BF16 keeps the classifier BF16; no extra exclusion is needed.
- **Activation outliers.** MLP `down_proj` inputs reach amax 8,380 (layer 6) against a median of 30.6.
  Plain max calibration (A) pays for that; `nvfp4_act_headroom` (C) recovers most of the loss.
- **Eval-set difficulty.** With random negatives, BF16 sat at security@hangglidersrule.com 0.98–1.00 and candidates could
  not be separated. Hard negatives are required for this kind of gate.

## License

Apache-2.0, inherited from upstream; the published repository ships the license text.
