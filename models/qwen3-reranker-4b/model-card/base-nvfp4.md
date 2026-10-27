---
license: apache-2.0
base_model: Qwen/Qwen3-Reranker-4B
base_model_relation: quantized
pipeline_tag: text-ranking
tags:
  - darkstar
  - qwen3-reranker
  - base
  - nvfp4
  - modelopt
  - w4a4
  - reranker
  - vllm
quantization: nvidia-modelopt
---

# Darkstar-Qwen3-Reranker-4B-Base-ModelOpt-W4A4-NVFP4

## Summary

Unedited NVIDIA ModelOpt NVFP4 quantization of
[`Qwen/Qwen3-Reranker-4B`](https://huggingface.co/Qwen/Qwen3-Reranker-4B) at revision
`22e683669bc0f0bd69640a1354a6d0aebcfeede5`, built from the upstream BF16 weights. No third-party
quantized checkpoint was used as a build input, and no behavior edit was applied.

**What is quantized.** The 108 decoder MLP linears (`gate_proj`, `up_proj`, `down_proj` in all 36
layers) are W4A4 NVFP4: FP4 E2M1 weights and activations, group 16, FP8 E4M3 block scales.
Attention (`q/k/v/o_proj`), the tied `embed_tokens` / `lm_head` (the source of the classifier's
`yes` / `no` rows) and all norms stay BF16. The KV cache stays BF16. Weights load as 3.92 GiB in vLLM,
against 7.56 GiB for BF16.

**How.** ModelOpt `0.48.0rc1` at `ff2450c15eb5f1697ec0471704d5421d54fce4f4` (wheel SHA-256
`2b874d94c198d72a1ca299a9d2a45ea5ff61f860ca0bd4ed7656465ce5b5f588`), operator recipe
`nvfp4_mlp_only_act_headroom-kv_bf16` (SHA-256
`eb9a063a89cc527f58a0c68bcbc860427e9f3cf67c048033316da147c82686d6`):
- **Weight scales:** MSE FP8-scale sweep.
- **Activation scales:** `nvfp4_act_headroom` calibration (anchor percentile 1, upper percentile
  99.99, rho 16384). It anchors the activation global scale low and keeps the rest of the FP8 range as
  headroom for activations larger than any seen in calibration.

Calibration used 512 `microsoft/ms_marco` v1.1 train query–passage pairs at revision
`a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a`: 256 relevant and 256 not, rendered with the reranker
judge template, seed 1234, file SHA-256
`cada916c2205153cc32ee6a4b30e5a121f8960e6f68f8578130ffc6a47e95261`.

This was the best of three builds:

| Candidate | Change | Result |
|---|---|---|
| A | MLP W4A4 with plain max activation calibration | nDCG@10 −0.0099 |
| B | MLP and attention W4A4 | nDCG@10 −0.0206 (CI excludes zero); rejected |
| **C (this checkpoint)** | MLP W4A4 with `nvfp4_act_headroom` | nDCG@10 −0.0024 |

Full provenance, protocol and evidence:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3-reranker-4b/model-card/base-nvfp4.md)
- [Family record](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3-reranker-4b/README.md)
- [Build recipe](https://github.com/HangGlidersRule/model-forge/blob/main/recipes/qwen3-reranker-4b/darkstar-qwen3-reranker-4b-base-modelopt-w4a4-nvfp4.yaml)
- [Gate results](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3-reranker-4b/results/reranker-gate-summary.json)
- [Campaign issue](https://github.com/HangGlidersRule/model-forge/issues/14)

## Evaluation

Measured on an RTX PRO 6000 Blackwell (SM120) with vLLM `v0.32.0`, against the upstream BF16
checkpoint served the same way. A second BF16 serve of the same weights sets the run-to-run noise
floor.

**Eval set.** Six MTEB Nano retrieval sets (NFCorpus, FiQA2018, Touche2020, ArguAna, SciFact,
MSMARCO) at pinned revisions, 299 queries:
- Each query is scored against every judged-relevant document plus the top BM25 non-relevant
  documents, 30 candidates per query, 8,970 pairs.
- Documents are not truncated; the longest is about 21k characters.
- Eval set SHA-256 `28f2dde20dd4bc3e82d41cdaa0e63df01048a29f9b318cf836129c59c3e450db`.
- 95% confidence intervals come from a paired bootstrap of per-query nDCG@10 (10,000 resamples).

| Metric | Value | Basis |
|---|---|---|
| Macro nDCG@10 (6 MTEB Nano sets) | 0.7095 (BF16 0.7118) | 299 queries, 8,970 pairs, full denominator; BM25 hard negatives; candidate-selection run |
| Delta vs BF16 | −0.0024 (95% CI −0.0120 to +0.0073) | paired bootstrap; inside the confidence interval, so not distinguishable from zero |
| Delta vs BF16, published serving profile | −0.0049 (95% CI −0.0155 to +0.0057) | same eval set; this checkpoint's own `chat_template.jinja` (see Runtime) |
| BF16 replica noise floor | −0.0008 (95% CI −0.0041 to +0.0014) | second BF16 serve of the same weights |
| Score rank agreement vs BF16 | Spearman 0.95, top-10 overlap 0.90 | BF16 replica: 0.999 and 0.99 |
| Throughput, 64 pairs per request, c=8 | 82.8 pairs/s (BF16 45.9), 1.81x | pairs of about 1k tokens; median of 3 runs after warmup |
| Throughput, published serving profile, c=8 | 79.6 pairs/s (BF16 45.8), 1.74x | same method; that run shared the GPU with another workload |
| GPQA Diamond | not measured | not applicable to a reranker |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or
protocol.

Per-dataset nDCG@10, BF16 → this checkpoint (candidate-selection run): NFCorpus 0.526 → 0.521,
FiQA2018 0.771 → 0.761, Touche2020 0.648 → 0.640, ArguAna 0.748 → 0.761, SciFact 0.856 → 0.839,
MSMARCO 0.723 → 0.735.

## Safety and limitations

This is an unedited base quantization. No refusal-behavior edit was applied, and the model scores
relevance rather than generating text.
- **Ranking agreement.** Mean ranking quality is preserved within the confidence interval, but
  individual orderings move more than BF16 moves from run to run: score Spearman is about 0.95 against
  0.999. Don't assume rank-for-rank identical output.
- **Activation outliers.** Some decoder MLP `down_proj` inputs have very large activations (amax
  8,380 in layer 6, against a median of 30.6). For two blocks the range exceeded what
  `nvfp4_act_headroom` can cover, so those blocks fall back to the calibrated maximum. Inputs far
  outside the calibration distribution may degrade more than the measured average.
- **Toolchain.** Built with a ModelOpt release candidate (`0.48.0rc1`), the newest ModelOpt build at
  release time; `0.47.0` was the newest stable release.
- **Hardware.** On GPUs without FP4 tensor cores, vLLM runs NVFP4 weight-only, so the throughput gain
  shown here doesn't apply.

## Release reference

Family release: [`darkstar-qwen3-reranker-4b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-qwen3-reranker-4b-v1.0.0). This immutable tag exists and the release contract is published.

## Runtime

Validated with vLLM `v0.32.0` (`vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`)
on SM120. NVFP4 GEMMs run on `FlashInferCutlassNvFp4LinearKernel`, with FlashAttention and a BF16 KV
cache. Serve as a pooling model with the sequence-classification overrides.

Pass the checkpoint's own `chat_template.jinja` explicitly. In vLLM `v0.32.0`, `/v1/score` does not
apply the repository's chat template on its own: without the flag, the request is scored without
the reranker judge prompt and rankings are wrong.

```bash
hf download HangGlidersRule/Darkstar-Qwen3-Reranker-4B-Base-ModelOpt-W4A4-NVFP4 chat_template.jinja --local-dir .

vllm serve HangGlidersRule/Darkstar-Qwen3-Reranker-4B-Base-ModelOpt-W4A4-NVFP4 \
  --served-model-name darkstar-qwen3reranker4b-base-nvfp4 \
  --runner pooling \
  --hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' \
  --chat-template ./chat_template.jinja \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192
```

```bash
curl -s http://localhost:8000/v1/score -H 'Content-Type: application/json' -d '{
  "model": "darkstar-qwen3reranker4b-base-nvfp4",
  "text_1": "what is the capital of france",
  "text_2": ["Paris is the capital and largest city of France.", "Bananas are rich in potassium."]
}'
```

The upstream model card's custom instructions still apply. Pass one as a `system` message in the
chat-style score request, or edit the default instruction in the template.

## License

Apache-2.0, inherited from [`Qwen/Qwen3-Reranker-4B`](https://huggingface.co/Qwen/Qwen3-Reranker-4B).
The license text ships with this repository as `LICENSE`. This repository contains modified files:
quantized weights and quantization metadata.
