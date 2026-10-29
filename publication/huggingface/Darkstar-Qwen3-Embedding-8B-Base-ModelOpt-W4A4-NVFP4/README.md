---
license: apache-2.0
base_model: Qwen/Qwen3-Embedding-8B
base_model_relation: quantized
pipeline_tag: feature-extraction
tags:
  - darkstar
  - qwen3-embedding
  - base
  - nvfp4
  - modelopt
  - w4a4
  - embeddings
  - vllm
quantization: nvidia-modelopt
---

# Darkstar-Qwen3-Embedding-8B-Base-ModelOpt-W4A4-NVFP4

## Summary

Unedited NVIDIA ModelOpt NVFP4 quantization of
[`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B) at revision
`1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`, built from the upstream BF16 weights. No third-party quantized checkpoint was
used as a build input, and no behavior edit was applied.

**What is quantized.** Every decoder linear is W4A4 NVFP4. That is FP4 E2M1 weights and activations, group 16, with FP8
E4M3 block scales, for 252 linears:
- the MLP `gate_proj`, `up_proj` and `down_proj`;
- the attention `q_proj`, `k_proj`, `v_proj` and `o_proj`;
- in all 36 layers.

That is 100% of the decoder linear parameters (6.95B). `embed_tokens` and all norms stay BF16, and so does the KV cache.
Weights load as 4.81 GiB in vLLM, against 14.11 GiB for BF16.

**How.** ModelOpt `0.48.0rc1` at `ff2450c15eb5f1697ec0471704d5421d54fce4f4` (wheel SHA-256
`2b874d94c198d72a1ca299a9d2a45ea5ff61f860ca0bd4ed7656465ce5b5f588`), operator recipe
`nvfp4_attn_mlp_act_headroom-kv_bf16` (SHA-256 `2a5f6acb073072af0808ee7218bf6d870dcb569d4a241719d06be267de38381c`):
- **Weight scales:** MSE FP8-scale sweep.
- **Activation scales:** `nvfp4_act_headroom` calibration (anchor percentile 1, upper percentile 99.99, rho 16384).

**Calibration data:** 512 texts from `microsoft/ms_marco` v1.1 train at revision
`a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a`:
- 256 queries, with the model's own query instruction;
- 256 passages, plain text;
- seed 1234; file SHA-256 `d643f77fecf055ca31e2da5829f86ba90aa925d151154f6ddfcb49446bbb11c6`.

**What else ships.** The tokenizer, the sentence-transformers pooling and prompt files and the license are byte-identical
to upstream. The upstream checkpoint has no `lm_head`, and neither does this one. Embeddings are the last-token hidden
state, L2-normalized.

**The two builds compared:**

| Candidate | Coverage | Public nDCG@10 vs BF16 (BF16 corpus × NVFP4 query / all NVFP4) | Long-input tokens/s vs BF16 |
|---|---|---|---|
| A | decoder MLP only (78% of linear work) | 99.56% / 99.94% | 1.90-2.03x |
| **B (this checkpoint)** | every decoder linear | 99.79% / 98.81% | 2.45-2.69x |

Both passed the ship rule: both pairings at least 98% of BF16 on the public sets. The rule then selects the faster
passing build.

**Full provenance, protocol and evidence:**
- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3-embedding-8b/model-card/base-nvfp4.md)
- [Family record](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3-embedding-8b/README.md)
- [Build recipe](https://github.com/HangGlidersRule/model-forge/blob/main/recipes/qwen3-embedding-8b/darkstar-qwen3-embedding-8b-base-modelopt-w4a4-nvfp4.yaml)
- [Gate results](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3-embedding-8b/results/embedding-gate-summary.json)
- [Campaign issue](https://github.com/HangGlidersRule/model-forge/issues/18)

## Evaluation

**Setup.** Measured on an RTX PRO 6000 Blackwell (SM120) with vLLM `v0.32.0`, against the upstream BF16 checkpoint
served the same way. A second BF16 deployment on another GPU sets the replica noise floor.

**Retrieval.** Six MTEB Nano retrieval sets (NFCorpus, FiQA2018, Touche2020, ArguAna, SciFact, MSMARCO) at pinned
revisions:
- 299 queries over their full Nano corpora (24,893 documents).
- Queries carry the model's default web-search instruction; documents carry none.
- Exact cosine search.
- 95% confidence intervals come from a paired bootstrap that resamples queries within each set (10,000 resamples).

| Metric | Value | Basis |
|---|---|---|
| Mean nDCG@10, all NVFP4 | 0.6487 (BF16 0.6565), 98.81% | 6 sets, 299 queries, 24,893 documents; 4096 dims |
| Mean nDCG@10, BF16 corpus × NVFP4 queries | 0.6551, 99.79%; delta −0.0014 (95% CI −0.0102 to +0.0077) | this checkpoint's queries against BF16 document vectors |
| BF16 replica noise floor | 0.6535, 99.54% | second BF16 deployment's queries against the BF16 corpus |
| Published copy, second serving instance | all NVFP4 98.84%, BF16 corpus × NVFP4 queries 100.18%; its queries against the first instance's NVFP4 corpus 99.48% | clean re-download served with this card's command |
| Mean nDCG@10 at 768 dims | all NVFP4 99.04%, BF16 corpus × NVFP4 queries 100.57% | Matryoshka truncation, renormalized |
| Document vector cosine vs BF16 | mean 0.9688, p1 0.9497, p5 0.9582 | 24,893 documents; BF16 replica 0.99993 |
| Throughput, long inputs (~3.7k tokens) | 44,856 tok/s at c=1, 47,828 at c=32 (BF16 18,322 / 17,831), 2.45-2.69x | single-input requests, unique prefixes; GPU shared with a co-resident engine that had no running requests when each run started |
| Throughput, short inputs (~100 tokens) | 5,456 tok/s at c=1, 37,108 at c=64 (BF16 5,014 / 19,691), 1.09-1.88x | same method |
| Latency, single request | p50 12.3 ms short, 82.5 ms long (BF16 18.0 / 201.7) | concurrency 1 |
| GPQA Diamond | not measured | not applicable to an embedding model |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or protocol.

**Per-set nDCG@10**, BF16 → this checkpoint (all NVFP4, 4096 dims):
- NFCorpus 0.3787 → 0.3788
- FiQA2018 0.7074 → 0.6940
- Touche2020 0.6487 → 0.6517
- ArguAna 0.6841 → 0.6611
- SciFact 0.8042 → 0.8094
- MSMARCO 0.7158 → 0.6972

## Safety and limitations

This is an unedited base quantization. No behavior edit was applied, and the model produces vectors rather than text.

**Vectors are not identical to BF16 vectors.** Document vectors sit at cosine 0.969 to their BF16 counterparts on
average. Mixing this checkpoint's queries with an existing BF16 corpus keeps public retrieval quality within the noise
(table above). Applications with **fixed similarity thresholds** can still shift, though. We measured this on three of
our own production stores of BF16 vectors from this model, comparing NVFP4 and BF16 query vectors against the stored
BF16 vectors:

| Store | NVFP4 queries vs stored BF16 vectors | Verdict |
|---|---|---|
| Long transcripts, 44k chunks (question retrieval) | recall@10 −0.1 pt (95% CI −0.45 to +0.25), MRR@10 −0.4 pt (CI −0.78 to −0.08) | compatible, no re-embedding |
| Personal memories, 31k items | recall@10 −0.8 pt (CI −1.35 to −0.25); every pair above a 0.97 cosine dedup trigger fell below it (529 to 0) | re-embed |
| News claims, 12k items | 3.3% of near-neighbor pairs flipped at a 0.60 candidate floor | re-embed |

**What this means for an existing BF16 deployment:**
- **Ranking-only use is likely fine.** If you only rank by similarity, mixing is likely fine. Check retrieval on your own
  corpus.
- **Hard thresholds need checking.** If you act on a hard threshold, expect NVFP4-to-BF16 similarities to run lower.
  Re-embed, or re-tune the threshold.
- **Mixed corpora are slightly biased.** In a mixed corpus (old BF16 rows, new NVFP4 rows), an NVFP4 query scores its
  NVFP4 rows slightly higher (mean +0.012 cosine). They gain top-1 placements; the BF16 rows lost none in our test.

**Vectors vary between serving instances.** With the default serve command, two separate vLLM instances of this
checkpoint return slightly different vectors for the same text. Over the 24,893 public documents the cosine between
instances is 0.983 on average (p1 0.970); two BF16 instances agree at 0.99993.
- **Not the files:** the clean download is byte-identical to the evaluated export.
- **The runtime:** activations are quantized to FP4 on the fly, and each new instance makes its own compile and kernel
  autotuning choices at startup, so small numeric differences flip FP4 rounding. The stable-vectors profile under Runtime
  pins those choices and removes the difference.
- **Effect:** retrieval quality holds across instances (table above), but fixed similarity thresholds see this noise after
  every restart.

**Other limitations:**
- **Activation outliers:** attention inputs reach amax 33,100 (layer 34) against a median of 3,410. Inputs far outside
  the calibration distribution may degrade more than the measured average.
- **Toolchain:** built with a ModelOpt release candidate (`0.48.0rc1`), the newest ModelOpt build at release time;
  `0.47.0` was the newest stable release.
- **Hardware:** on GPUs without FP4 tensor cores, vLLM runs NVFP4 weight-only, so the throughput gain shown here doesn't
  apply.

## Release reference

Family release: [`darkstar-qwen3-embedding-8b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-qwen3-embedding-8b-v1.0.0). This immutable tag exists and the release contract is published.

## Runtime

**Validated stack:** vLLM `v0.32.0` (`vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`)
on SM120.
- NVFP4 GEMMs run on `FlashInferCutlassNvFp4LinearKernel`.
- KV cache: BF16.
- Serve it as a pooling model. vLLM reads the last-token pooling from the checkpoint.
- The Matryoshka override lets clients request smaller dimensions.

```bash
vllm serve HangGlidersRule/Darkstar-Qwen3-Embedding-8B-Base-ModelOpt-W4A4-NVFP4 \
  --served-model-name darkstar-qwen3embedding8b-base-nvfp4 \
  --runner pooling \
  --hf-overrides '{"is_matryoshka": true}' \
  --max-model-len 28672 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching
```

**Stable vectors.** A vector store that needs the same vector for the same text across restarts should use vLLM's
batch-invariant mode and keep the compile and autotune caches on one persisted volume shared by every container:

```bash
VLLM_BATCH_INVARIANT=1 \
VLLM_CACHE_ROOT=/persist/vllm \
TRITON_CACHE_DIR=/persist/triton \
TORCHINDUCTOR_CACHE_DIR=/persist/inductor \
FLASHINFER_WORKSPACE_BASE=/persist/flashinfer \
VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/persist/fi-autotune \
vllm serve HangGlidersRule/Darkstar-Qwen3-Embedding-8B-Base-ModelOpt-W4A4-NVFP4 \
  --served-model-name darkstar-qwen3embedding8b-base-nvfp4 \
  --runner pooling \
  --hf-overrides '{"is_matryoshka": true}' \
  --max-model-len 28672 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching
```

Measured with containers on one GPU sharing one cache volume; other GPUs of the same model are untested:
- **Reproducibility:** between two such containers, 98.5% of 25,192 public document and query vectors were bit-identical.
  The rest differed by at most 3e-8 per element, and every cosine was within 2e-15 of 1.
- **Quality:** mean nDCG@10 98.49% of BF16 all NVFP4, and 99.39% with BF16 document vectors.
- **Speed:** 2.51-2.74x BF16 tokens/s on long inputs and 0.99-2.00x on short inputs. vLLM selects
  `CutlassNvFp4LinearKernel` in this mode.

**Instructions.** Queries take the model's instruction prefix; documents take none:

```bash
curl -s http://localhost:8000/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model": "darkstar-qwen3embedding8b-base-nvfp4",
  "input": ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:what is the capital of france", "Paris is the capital and largest city of France."]
}'
```

The upstream model card's task-specific instructions still apply. Put the instruction text in front of each query.

## License

Apache-2.0, inherited from [`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B). The license text
ships with this repository as `LICENSE`. This repository contains modified files: quantized weights and quantization
metadata.
