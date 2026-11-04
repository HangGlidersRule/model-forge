---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/Nemotron-3.5-Rerank-8B-BF16
base_model_relation: quantized
pipeline_tag: text-ranking
tags:
  - darkstar
  - nemotron-rerank
  - base
  - nvfp4
  - modelopt
  - w4a4
  - reranker
  - vllm
quantization: nvidia-modelopt
---

# Darkstar-Nemotron-3.5-Rerank-8B-Base-ModelOpt-W4A4-NVFP4

## Summary

Unedited NVIDIA ModelOpt NVFP4 quantization of
[`nvidia/Nemotron-3.5-Rerank-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16) at revision
`f88f8d9104726747791283ab156d2fcdb29e90a7`, built from the upstream BF16 weights. No third-party quantized checkpoint
was used as a build input, and no behavior edit was applied.

**What is quantized.** Every linear of the text tower is W4A4 NVFP4: attention `q/k/v/o_proj` and MLP
`gate/up/down_proj` in all 34 Ministral3 layers, 238 linears. FP4 E2M1 weights and activations, group 16, FP8 E4M3 block
scales. Everything else stays BF16 and byte-identical to upstream:
- the Pixtral vision tower and the multimodal projector;
- the token embeddings and all norms;
- the mean-pooled score head (4096 → 1).

The KV cache stays BF16. Weights load as 5.77 GiB in vLLM, against 15.74 GiB for BF16.

**How.** ModelOpt `0.48.0` at `e2f328a19de789e73b91aca2908042388cf6d496` (wheel SHA-256
`bfcfd425eab15e05eccf065b58d11cc33f9c27b4859a6efb78aba74bfb9f225c`), torch 2.15.0, transformers 5.18.0. Operator recipe
`nvfp4_text_all_act_headroom-kv_bf16` (SHA-256 `71874e44a460f1c5d29509a8d5de66e6eacda4b130a8fa3fd10f3f44a0329fa7`):
- **Weight scales:** MSE FP8-scale sweep.
- **Activation scales:** `nvfp4_act_headroom` calibration (anchor percentile 1, upper percentile 99.99, rho 16384).

**Calibration.** The whole classifier ran on text, with the same bidirectional attention and mean pooling the server
uses. The data was 512 `microsoft/ms_marco` v1.1 train query–passage pairs at revision
`a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a`:
- 256 relevant and 256 not, seed 1234;
- rendered exactly as this checkpoint's chat template renders a text pair (`query: …` / `passage: …`);
- file SHA-256 `ddd481866f158e51aaa7c66a9353be6ba1317bb5f2b3a188b1b703b5b961576e`.

This was the fastest of three builds that all passed the bar. The bar is at least 98% of BF16 macro nDCG@10 on both eval
sets:

| Candidate | Text tower | nDCG@10, 6 MTEB Nano sets | nDCG@10, LMEB Dialogue | Pairs/s, c=4 |
|---|---|---|---|---|
| **A (this checkpoint)** | MLP and attention W4A4 | 99.94% of BF16 | 99.59% of BF16 | 55.7 (3.09x BF16) |
| B | MLP W4A4, attention FP8 | 100.04% of BF16 | 99.31% of BF16 | 48.6 |
| C | MLP W4A4, attention BF16 | 100.06% of BF16 | 99.13% of BF16 | 37.5 |

Full provenance, protocol and evidence:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-rerank-8b/model-card/base-nvfp4.md)
- [Family record](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-rerank-8b/README.md)
- [Build recipe](https://github.com/HangGlidersRule/model-forge/blob/main/recipes/nemotron-3.5-rerank-8b/darkstar-nemotron-3.5-rerank-8b-base-modelopt-w4a4-nvfp4.yaml)
- [Gate results](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-rerank-8b/results/rerank-gate-summary.json)
- [Campaign issue](https://github.com/HangGlidersRule/model-forge/issues/32)

## Evaluation

**Setup.** Measured on an RTX PRO 6000 Blackwell (SM120) with vLLM `v0.32.0`, against the upstream BF16 checkpoint
served the same way. A second BF16 serve of the same weights sets the run-to-run noise floor. 95% confidence intervals
come from a paired bootstrap of per-query nDCG@10 (10,000 resamples).

**Eval set 1, six MTEB Nano retrieval sets.** NFCorpus, FiQA2018, Touche2020, ArguAna, SciFact and MSMARCO at pinned
revisions, 299 queries:
- each query is scored against every judged-relevant document plus the top BM25 non-relevant documents, 30 candidates
  per query, 8,970 pairs;
- documents are not truncated;
- SHA-256 `28f2dde20dd4bc3e82d41cdaa0e63df01048a29f9b318cf836129c59c3e450db`.

**Eval set 2, LMEB Dialogue memory set.** Built from
[`KaLM-Embedding/LMEB`](https://huggingface.co/datasets/KaLM-Embedding/LMEB) @ `f9d4294b`: LoCoMo (5 splits), REALTALK
(3 splits) and TMD `content_time_qs`. 60 queries per split, 540 queries:
- each query is scored against its judged turns plus the top BM25 non-relevant turns of the same conversation, 30
  candidates per query, 16,200 pairs;
- SHA-256 `9373a1be97911f4ad3090a5f7b65a36b0046427ed2c7f35233510dd6b4f24ac0`.

| Metric | Value | Basis |
|---|---|---|
| Macro nDCG@10, 6 MTEB Nano sets | 0.7344 (BF16 0.7349), 99.94% | 299 queries, 8,970 pairs, full denominator |
| Delta vs BF16, 6 MTEB Nano sets | −0.0004 (95% CI −0.0055 to +0.0049) | paired bootstrap; not distinguishable from zero |
| Macro nDCG@10, LMEB Dialogue | 0.8056 (BF16 0.8089), 99.59% | 540 queries, 16,200 pairs, full denominator |
| Delta vs BF16, LMEB Dialogue | −0.0033 (95% CI −0.0096 to +0.0027) | paired bootstrap; not distinguishable from zero |
| BF16 replica noise floor | −0.0001 (95% CI −0.0012 to +0.0006) and −0.0003 (95% CI −0.0009 to +0.0001) | second BF16 serve, the two sets in order |
| Score rank agreement vs BF16 | Spearman 0.982 and 0.984, top-10 overlap 0.946 and 0.940 | the two sets in order; BF16 replica: Spearman 0.999 and 0.999 |
| Throughput, 64 pairs per request | 51.5 / 55.7 / 51.9 pairs/s at c=1 / 4 / 8 (BF16 17.4 / 18.0 / 17.9), 2.90-3.09x | pairs of about 1.1k tokens; median of 3 runs after warmup; whole GPU |
| GPU memory under that load | 9.5 GiB (BF16 18.5 GiB) | `--max-model-len 8192 --max-num-batched-tokens 8192`; memory after the c=8 runs minus before the serve |
| Published serve profile, two fresh serves | 0.7346 and 0.7332 (99.96% and 99.77% of BF16) on the 6 MTEB Nano sets; 0.8045 and 0.8023 (99.46% and 99.18%) on LMEB Dialogue | this card's serve command; an empty-cache download of this repository, then the evaluated export in a fresh container |
| GPQA Diamond | not measured | not applicable to a reranker |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or protocol.

**Run-to-run spread.** The selection run served with `--max-model-len 32768`; the two published-profile serves used this
card's command. All three serves load byte-identical files. Their macro nDCG@10 spans 0.7332-0.7346 on the MTEB Nano sets
and 0.8023-0.8056 on LMEB Dialogue, against a BF16 replica spread of 0.0002 and 0.0003. W4A4 activation quantization
makes scores depend slightly on how requests are batched, so individual scores move between serves. Every serve clears
the 98% bar on both sets.

Per-split nDCG@10 on LMEB Dialogue:

| Split | BF16 | This checkpoint |
|---|---|---|
| LoCoMo single_hop | 0.8873 | 0.8728 |
| LoCoMo multi_hop | 0.8486 | 0.8487 |
| LoCoMo temporal_reasoning | 0.9113 | 0.9025 |
| LoCoMo open_domain | 0.7329 | 0.7297 |
| LoCoMo adversarial | 0.6546 | 0.6621 |
| REALTALK commonsense | 0.75 | 0.7404 |
| REALTALK multi_hop | 0.7734 | 0.7691 |
| REALTALK temporal_reasoning | 0.8306 | 0.8291 |
| TMD content_time_qs | 0.8915 | 0.8959 |

**Next to Qwen3-Reranker-4B.** Measured on the same two sets in the same session, as a reference:
- the BF16 build scored 0.7121 and 0.7544;
- the Darkstar W4A4 build scored 0.7033 and 0.7486.

Against the Qwen3-Reranker-4B BF16 build, this checkpoint scores +0.0223 (95% CI +0.0038 to +0.0408) on the MTEB Nano
sets and +0.0512 (95% CI +0.0365 to +0.0659) on LMEB Dialogue. It runs at 1.51-1.57x that build's pairs/s.

## Safety and limitations

This is an unedited base quantization. No refusal-behavior edit was applied, and the model scores relevance rather than
generating text.
- **Image documents are not evaluated.** The vision tower ships unchanged, but the text tower that reads image tokens is
  quantized, and every gate above is text-only. Measure image-document reranking on your own data before relying on it.
- **Raw scores.** The score is the raw relevance logit (about −4 to −2 on the eval pairs above), not a probability.
  Rankings are order-based, so this does not affect them. A client that thresholds scores must recalibrate, or serve with
  the activation on: `"use_activation": true` in `--pooler-config` applies a sigmoid without changing the order.
- **Ranking agreement.** Mean ranking quality is within the confidence interval of BF16, but individual orderings move
  more than BF16 moves from run to run: score Spearman is about 0.98 against 0.999. Don't assume rank-for-rank identical
  output.
- **Hardware.** On GPUs without FP4 tensor cores, vLLM runs NVFP4 weight-only, so the throughput gain shown here doesn't
  apply.

## Release reference

Family release: [`darkstar-nemotron-3.5-rerank-8b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3.5-rerank-8b-v1.0.0). This immutable tag exists and the release contract is published.

## Runtime

Validated with vLLM `v0.32.0` (`vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`)
on SM120. NVFP4 GEMMs run on `FlashInferCutlassNvFp4LinearKernel` with a BF16 KV cache. Serve as a pooling model with
mean pooling and the activation off (raw logits), and pass the checkpoint's own chat template.

```bash
hf download HangGlidersRule/Darkstar-Nemotron-3.5-Rerank-8B-Base-ModelOpt-W4A4-NVFP4 chat_template.jinja --local-dir .

vllm serve HangGlidersRule/Darkstar-Nemotron-3.5-Rerank-8B-Base-ModelOpt-W4A4-NVFP4 \
  --served-model-name darkstar-nemotron35rerank8b-base-nvfp4 \
  --runner pooling \
  --pooler-config '{"seq_pooling_type":"MEAN","use_activation":false}' \
  --chat-template ./chat_template.jinja \
  --chat-template-content-format openai \
  --limit-mm-per-prompt '{"image":1}' \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192
```

`/v1/score`, `/v1/rerank` and `/classify` return identical scores for text pairs (64 pairs checked, max difference 0).

```bash
curl -s http://localhost:8000/v1/rerank -H 'Content-Type: application/json' -d '{
  "model": "darkstar-nemotron35rerank8b-base-nvfp4",
  "query": "what is the capital of france",
  "documents": ["Paris is the capital and largest city of France.", "Bananas are rich in potassium."]
}'
```

## License

[OpenMDW-1.1](https://openmdw.ai/license/1-1/), inherited from
[`nvidia/Nemotron-3.5-Rerank-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16). The license text and
NVIDIA's notices ship with this repository (`LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`). This repository contains
modified files: quantized weights and quantization metadata.
