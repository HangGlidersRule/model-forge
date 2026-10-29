# Darkstar Qwen3-Embedding-8B

Embedding family record. Source: [`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B), an 8B
dense Qwen3 decoder used as a text embedder: the last token's hidden state, L2-normalized, 4096 dimensions with Matryoshka
truncation. Proposed and tracked in [issue #18](https://github.com/HangGlidersRule/model-forge/issues/18). The question the
issue asked: can an NVFP4 build serve next to corpora already embedded by the BF16 model, or must those corpora be
re-embedded?

## Shipped cells

| Cell | Repository / source | Status |
|---|---|---|
| Base BF16 (control) | [`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B) @ `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` | unchanged upstream reference; no HangGlidersRule repository |
| Base W4A4-NVFP4 | [`Darkstar-Qwen3-Embedding-8B-Base-ModelOpt-W4A4-NVFP4`](https://huggingface.co/HangGlidersRule/Darkstar-Qwen3-Embedding-8B-Base-ModelOpt-W4A4-NVFP4) | published |
| Abliterated cells | — | not applicable: an embedding model has no refusal behavior to edit |

Release: tag [`darkstar-qwen3-embedding-8b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-qwen3-embedding-8b-v1.0.0).
Family collection: [Darkstar Qwen3-Embedding-8B](https://huggingface.co/collections/HangGlidersRule/darkstar-qwen3-embedding-8b-6ae3ab8344f2aaa16a5b2d00).

## Documents

- [`model-card/base-nvfp4.md`](model-card/base-nvfp4.md): source card for the published checkpoint.
- [`model-card/base-bf16.md`](model-card/base-bf16.md): the upstream BF16 control.
- [`results/embedding-gate-summary.json`](results/embedding-gate-summary.json): the gate summary. The card quotes it, and
  [`eval/build_gate_summary.py`](eval/build_gate_summary.py) builds it from the measured files beside it:
  - public retrieval (`retrieval-benchmark.json`);
  - throughput (`throughput-*.json`);
  - store compatibility (`store-compat-*.json`, `mixed-corpus-*.json`);
  - serving-instance reproducibility (`instance-agreement.json`, `stable-profile-reproducibility.json`);
  - artifact validation and build provenance (`artifact-validation.json`, `build-summary.json`, `activation-outliers.json`).
- [`eval/check_card_numbers.py`](eval/check_card_numbers.py): fails if any number on the card is missing from that summary.
- [`calibration/`](calibration/): deterministic builder for the 512-text MS MARCO calibration set.
- [`eval/`](eval/): retrieval benchmark, store-compatibility comparison, throughput probe, artifact validator.

## Recipes

- [`../../recipes/qwen3-embedding-8b/darkstar-qwen3-embedding-8b-base-modelopt-w4a4-nvfp4.yaml`](../../recipes/qwen3-embedding-8b/darkstar-qwen3-embedding-8b-base-modelopt-w4a4-nvfp4.yaml):
  schema-2 build recipe for the published cell.
- [`../../configs/modelopt/recipes/qwen3-embedding-8b/`](../../configs/modelopt/recipes/qwen3-embedding-8b/): the two
  ModelOpt operator recipes compared (A: MLP only, B: every decoder linear).
- [`../../configs/modelopt/pins/modelopt-0.48.0rc1.json`](../../configs/modelopt/pins/modelopt-0.48.0rc1.json): the ModelOpt
  pin.

## Candidate selection

Both candidates quantize from the upstream BF16 weights with ModelOpt `0.48.0rc1`:
- **Weights:** static NVFP4, group 16, MSE FP8-scale sweep.
- **Activations:** NVFP4 with `nvfp4_act_headroom` calibration.
- **Calibration data:** 512 MS MARCO queries and passages, formatted as an embedding client sends them.

**Pre-registration and its amendment, both posted on the issue before the results they govern:**
- the [pre-registration](https://github.com/HangGlidersRule/model-forge/issues/18#issuecomment-6501857585), for store
  compatibility;
- [amendment R18-A1](https://github.com/HangGlidersRule/model-forge/issues/18#issuecomment-6504231012), for the public
  benchmark, candidate B and the ship rule.

**Ship rule:** both pairings must reach at least 98% of all-BF16 mean nDCG@10 on the six MTEB Nano retrieval sets at 4096
dims. Among the candidates that pass, the faster ships.

| Candidate | Coverage | BF16 corpus × NVFP4 queries | All NVFP4 | Long-input tokens/s vs BF16 |
|---|---|---|---|---|
| BF16 replica (noise floor) | — | 99.54% | — | — |
| A | decoder MLP (78% of linear work) | 99.56% | 99.94% | 1.90-2.03x |
| **B (published)** | every decoder linear | **99.79%** | **98.81%** | **2.45-2.69x** |

Both pass; B is faster and ships.

## Compatibility with stored BF16 vectors

The pre-registered test measured each candidate against three production stores of BF16 vectors from this model, without
any text, vector or row id leaving the hosts:
- a long-transcript chunk store;
- a personal-memory store;
- a news-claim store.

Each store had its own drop-in rule, combining retrieval loss and decision flips at that application's own similarity
threshold.

| Store | Candidate B |
|---|---|
| Long transcripts, retrieval only | drop-in (recall@10 −0.1 pt) |
| Personal memories, 0.97 dedup trigger | re-embed (recall@10 −0.8 pt; every pair above 0.97 fell below it) |
| News claims, 0.60 candidate floor | re-embed (3.3% of pairs flipped) |

Candidate A gave the same three verdicts. Full numbers: `store_compatibility` in the gate summary.

## Reproducibility across serving instances

Found during the clean-download smoke:
- **The problem:** with default serving, two separate vLLM instances of this checkpoint return slightly different vectors
  for the same text (mean cosine 0.983 over the public documents). The files are byte-identical.
- **The cause:** each instance makes its own compile and FP4 kernel autotuning choices.
- **The fix:** the card's stable-vectors profile, which is batch-invariant mode plus one persisted cache volume shared by
  every container.
  - **Reproducibility:** 98.5% of vectors were bit-identical between containers, and the rest agreed to within 3e-8 per
    element.
  - **Quality:** 98.49% / 99.39%.
  - **Speed:** 2.51-2.74x BF16 on long inputs.

## Gotchas

- **`lm_head` from the loader.** transformers' `Qwen3ForCausalLM` loader initialises an `lm_head` at random because the
  embedding checkpoint has none, and ModelOpt exports it. The export step removes it (`eval/strip_lm_head.py`); the
  pooling runner never reads it.
- **Serving beside another engine.** vLLM's KV budget comes from `--gpu-memory-utilization` after weights and CUDA graphs.
  On a shared GPU, pass `--kv-cache-memory-bytes` explicitly.
- **Very long texts.** Texts over the 28,672-token context must be split before embedding. The store test embedded them
  the way the transcript application does: 24,000-character pieces, mean of the piece vectors.

## License

Apache-2.0, inherited from upstream; the published repository ships the license text.
