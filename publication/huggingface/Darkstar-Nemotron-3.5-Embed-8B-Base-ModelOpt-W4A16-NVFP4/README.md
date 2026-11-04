---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/Nemotron-3.5-Embed-8B-BF16
base_model_relation: quantized
pipeline_tag: feature-extraction
tags:
  - darkstar
  - nemotron-embed
  - base
  - nvfp4
  - modelopt
  - w4a16
  - embeddings
  - vllm
quantization: nvidia-modelopt
---

# Darkstar-Nemotron-3.5-Embed-8B-Base-ModelOpt-W4A16-NVFP4

## Summary

Unedited NVIDIA ModelOpt NVFP4 weight-only quantization of
[`nvidia/Nemotron-3.5-Embed-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16) at revision
`781e779ee4ca9c2d2a4f38321993556baeab49a1`, built from the upstream BF16 weights. No third-party quantized checkpoint was
used as a build input, and no behavior edit was applied.

**What it is for.** The same embeddings at about a third of the weight memory. Weights load as 5.82 GiB in vLLM, against
15.74 GiB for BF16. It is not faster: speed is 1.02-1.28x BF16 (see Evaluation).

**What is quantized.** Every linear of the bidirectional Ministral3 text tower is W4A16 NVFP4: attention `q/k/v/o_proj` and
MLP `gate/up/down_proj` in all 34 layers, 238 linears.
- FP4 E2M1 weights, group 16, with FP8 E4M3 block scales from an MSE FP8-scale sweep.
- Activations stay BF16; vLLM runs the layers on its Marlin NVFP4 kernel.
- Everything else stays BF16 and byte-identical to upstream: the Pixtral vision tower, the multimodal projector, the token
  embeddings and all norms.

**How.** ModelOpt `0.48.0` at `e2f328a19de789e73b91aca2908042388cf6d496` (wheel SHA-256
`bfcfd425eab15e05eccf065b58d11cc33f9c27b4859a6efb78aba74bfb9f225c`), torch 2.15.0, transformers 5.18.0. Operator recipe
`w4a16_nvfp4_text_all_mse-kv_bf16` (SHA-256 `980c52fad59aa52b80a2a00865d9e44d1667a2f3539d9fc1291d4a568cd6d112`).

**Calibration.** The weight scales were calibrated on 512 `microsoft/ms_marco` v1.1 train texts at revision
`a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a`:
- 256 queries and 256 passages, seed 1234;
- rendered exactly as this checkpoint's chat template renders them (`query: …` / `passage: …`, no BOS/EOS);
- file SHA-256 `a04f7ef04008eaa7c2a52e846109b9357481de87e65f447a8df1512f4ed44773`.

**Why weight-only.** Three candidates were built and gated. The bar, posted before scoring, is at least 98% of BF16
nDCG@10 on both eval sets, for all-candidate vectors and for BF16 documents searched with candidate queries.

| Candidate | Text tower | MTEB Nano, all vectors | LMEB Dialogue, all vectors | Verdict |
|---|---|---|---|---|
| A | W4A4 NVFP4, every linear | 100.44% of BF16 | 94.63% of BF16 | failed |
| C | W4A4 NVFP4, MLP only | 99.90% of BF16 | 96.72% of BF16 | failed |
| **D (this checkpoint)** | **W4A16 NVFP4, every linear** | **100.26% of BF16** | **98.78% of BF16** | **passed** |

- **A and C failed on short documents.** They lost quality on LMEB's conversation turns, which have a median of 168
  characters, while passing on MTEB's longer documents (median 811). Mean pooling averages FP4 activation error over few
  tokens in a short text.
- **D was added after A and C failed.** It drops activation quantization and passed. It was announced on the issue before
  it was scored; the bar did not change.

Full provenance, protocol and evidence:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-embed-8b/model-card/base-nvfp4.md)
- [Family record](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-embed-8b/README.md)
- [Build recipe](https://github.com/HangGlidersRule/model-forge/blob/main/recipes/nemotron-3.5-embed-8b/darkstar-nemotron-3.5-embed-8b-base-modelopt-w4a16-nvfp4.yaml)
- [Gate results](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3.5-embed-8b/results/embed-gate-summary.json)
- [Campaign issue](https://github.com/HangGlidersRule/model-forge/issues/33)

## Evaluation

**Setup.** Measured on an RTX PRO 6000 Blackwell (SM120) with vLLM `v0.32.0`, against the upstream BF16 checkpoint
served the same way.
- **Serving path check:** before any scoring, vLLM's BF16 vectors for 16 fixed texts were compared with vectors computed
  directly in transformers (forward pass, mean pooling, normalization). They matched at cosine 0.99995 minimum.
- **Noise floor:** a second BF16 serve of the same weights.
- **Confidence intervals:** 95%, from a paired bootstrap of per-query nDCG@10 (10,000 resamples).

**Eval set 1, MTEB Nano with full corpora.** The six pinned MTEB Nano retrieval sets (NFCorpus, FiQA2018, Touche2020,
ArguAna, SciFact, MSMARCO) searched over their complete corpora: 24,893 documents, 299 queries. SHA-256
`a1275b5dba2153ea3625e969176ae0f480e7295dd85ea036ec02bc12e1d80990`.

**Eval set 2, LMEB Dialogue, whole-conversation retrieval.** Built from
[`KaLM-Embedding/LMEB`](https://huggingface.co/datasets/KaLM-Embedding/LMEB) @ `f9d4294b`.
- 540 queries from LoCoMo (5 splits), REALTALK (3 splits) and TMD `content_time_qs`.
- Each query searches every turn of its own conversation: 32 conversations of 369-1,548 turns, 22,289 turns in all.
- SHA-256 `c2e6840f4f31aa0b3027cfe6635f1f9618ba194192d0e86ec8117d77ce544bbf`.

| Metric | Value | Basis |
|---|---|---|
| Mean nDCG@10, MTEB Nano, all vectors from this checkpoint | 0.6453 (BF16 0.6436), 100.26% | 299 queries, 24,893 documents, full denominator |
| Delta vs BF16, MTEB Nano | +0.0017 (95% CI −0.0033 to +0.0067) | paired bootstrap |
| Mean nDCG@10, LMEB Dialogue, all vectors from this checkpoint | 0.5717 (BF16 0.5788), 98.78% | 540 queries, 22,289 turns, full denominator |
| Delta vs BF16, LMEB Dialogue | −0.0071 (95% CI −0.0137 to −0.0004) | paired bootstrap: a small measured loss |
| BF16 documents searched with this checkpoint's queries | 99.82% and 99.36% of BF16 | the two sets in order |
| BF16 replica noise floor | +0.0005 (95% CI −0.0001 to +0.0016) and +0.0015 (95% CI −0.0004 to +0.0039) | second BF16 serve, the two sets in order |
| Clean download of this repository, card serve command | 0.6443 and 0.5720 (100.11% and 98.82% of BF16); document vectors vs the evaluated serve: cosine mean 0.99999 and 0.99998 | empty-cache download, every manifest entry verified, weights byte-identical to the evaluated export; the two sets in order |
| Document-vector cosine vs BF16 | mean 0.98891 and 0.98755, minimum 0.94482 and 0.97422 | every document of the two sets in order; BF16 replica mean 0.99998 and 0.99997 |
| Weights loaded in vLLM | 5.82 GiB (BF16 15.74 GiB) | `Model loading took` line of the serve log |
| Throughput, long inputs (~4k tokens) | 16,912 tok/s at c=1, 17,042 at c=32 (BF16 16,630 / 16,339), 1.02-1.04x | single-input requests, unique prefixes, keep-alive clients; whole GPU |
| Throughput, short inputs (~100 tokens) | 5,617 tok/s at c=1, 18,791 at c=64 (BF16 4,979 / 18,133), 1.04-1.28x | same method |
| GPU memory under that load | 16.1 GiB (BF16 24.5 GiB) | memory after the speed grid minus before the serve |
| GPQA Diamond | not measured | not applicable to an embedding model |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or protocol.

**Matryoshka truncation.** Keeping the first N dimensions and renormalizing, mean nDCG@10:

| Dimensions | MTEB Nano, this checkpoint | MTEB Nano, BF16 | LMEB Dialogue, this checkpoint | LMEB Dialogue, BF16 |
|---|---|---|---|---|
| 2000 | 0.6296 | 0.6271 | 0.5735 | 0.5828 |
| 1024 | 0.6142 | 0.6116 | 0.5706 | 0.5753 |
| 768 | 0.5988 | 0.5954 | 0.5596 | 0.5629 |

**Next to Qwen3-Embedding-8B.** Measured on the same two sets in the same session, as a reference:
- Qwen3-Embedding-8B BF16 scored 0.6541 and 0.3943;
- the Darkstar W4A4 build of it scored 0.6499 and 0.3791.

Against Qwen3-Embedding-8B BF16, this checkpoint scores −0.0088 (95% CI −0.0317 to +0.0140) on the MTEB Nano sets and
+0.1774 (95% CI +0.1510 to +0.2039) on LMEB Dialogue.

## Safety and limitations

This is an unedited base quantization of an embedding model. No refusal-behavior edit was applied, and the model produces
vectors rather than text.
- **Vectors are not BF16 vectors.** Document vectors sit at cosine 0.98891 and 0.98755 (mean) to BF16's on the two eval
  sets. Don't mix them with vectors the BF16 model produced in one index: embed a whole store with one build.
- **Image documents are not evaluated.** The vision tower ships unchanged, but the text tower that reads image tokens is
  quantized, and every gate above is text-only. Measure image retrieval on your own data before relying on it.
- **No speedup.** The gain is weight memory only. If you need throughput, serve BF16 or another build.
- **Small measured loss on conversational memory.** On LMEB Dialogue it is −0.0071 nDCG@10, inside the 2% bar but outside
  the noise floor.

## Release reference

Family release: [`darkstar-nemotron-3.5-embed-8b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3.5-embed-8b-v1.0.0). This immutable tag exists and the release contract is published.

## Runtime

**Validated stack:** vLLM `v0.32.0` (`vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`)
on SM120.
- The NVFP4 weight-only layers run on `MarlinNvFp4LinearKernel`.
- Serve it as a pooling model with mean pooling and normalization. It is bidirectional, so leave chunked prefill off.

```bash
vllm serve HangGlidersRule/Darkstar-Nemotron-3.5-Embed-8B-Base-ModelOpt-W4A16-NVFP4 \
  --served-model-name darkstar-nemotron35embed8b-base-nvfp4 \
  --runner pooling \
  --pooler-config '{"seq_pooling_type":"MEAN","use_activation":true}' \
  --chat-template-content-format openai \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768
```

**Prefixes.** Queries take `query: ` and documents take `passage: `, as in the checkpoint's sentence-transformers prompts.
The chat template also escapes `[` and `<` with a zero-width non-joiner and adds no BOS/EOS. Send prefixed text with
`"add_special_tokens": false`:

```bash
curl -s http://localhost:8000/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model": "darkstar-nemotron35embed8b-base-nvfp4",
  "input": ["query: what is the capital of france", "passage: Paris is the capital and largest city of France."],
  "add_special_tokens": false
}'
```

## License

[OpenMDW-1.1](https://openmdw.ai/license/1-1/), inherited from
[`nvidia/Nemotron-3.5-Embed-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16).
- **Shipped files:** `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md` ship with this repository.
- **Where they come from:** the upstream repository ships no license files. These are byte-identical to the ones NVIDIA
  ships with [`nvidia/Nemotron-3-Embed-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16) @ `d1f2f257`,
  the model this checkpoint's text side is taken from unchanged.
- **Modified files:** this repository contains modified files: quantized weights and quantization metadata.
