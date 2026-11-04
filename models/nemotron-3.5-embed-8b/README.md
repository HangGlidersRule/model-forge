# Darkstar Nemotron-3.5-Embed-8B

Embedding family record. Source:
[`nvidia/Nemotron-3.5-Embed-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16), NVIDIA's multimodal
embedding model.
- **Model:** a Pixtral vision tower and a bidirectional Ministral3 text tower (34 layers, hidden 4096), 8.38B parameters.
- **Output:** a mean-pooled, normalized 4,096-dimensional vector.
- **Text side:** kept frozen from `nvidia/Nemotron-3-Embed-8B-BF16`.
- **Tracking:** proposed and tracked in [issue #33](https://github.com/HangGlidersRule/model-forge/issues/33).

## Shipped cells

| Cell | Repository / source | Status |
|---|---|---|
| Base BF16 (control) | [`nvidia/Nemotron-3.5-Embed-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16) @ `781e779ee4ca9c2d2a4f38321993556baeab49a1` | unchanged upstream reference; no HangGlidersRule repository |
| Base W4A16-NVFP4 | [`Darkstar-Nemotron-3.5-Embed-8B-Base-ModelOpt-W4A16-NVFP4`](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3.5-Embed-8B-Base-ModelOpt-W4A16-NVFP4) @ `6f516127` | published ([receipt](results/publication-receipt.json)) |
| Abliterated cells | — | not applicable: an embedding model has no refusal behavior to edit |

Release: tag [`darkstar-nemotron-3.5-embed-8b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3.5-embed-8b-v1.0.0).
Family collection: [Darkstar Nemotron-3.5-Embed-8B](https://huggingface.co/collections/HangGlidersRule/darkstar-nemotron-35-embed-8b-6aeb75ea09c9fc26c4961388).

## Documents

- [`model-card/base-nvfp4.md`](model-card/base-nvfp4.md): source card for the published checkpoint
- [`model-card/base-bf16.md`](model-card/base-bf16.md): the upstream BF16 control
- [`results/`](results/): the gate summary and artifact validation, plus the publication ledger. The gate summary
  covers all three candidates and a BF16 replica on both eval sets, the speed grid, document-vector agreement and the
  Qwen3-Embedding-8B comparison.
- [`calibration/`](calibration/): deterministic builder for the 512-text calibration set
- [`eval/`](eval/): the eval harness, the fail-closed export validator, the fused-group check, document-vector agreement,
  the evidence and ledger builders, and the card-number check. The eval harness builds the frozen sets, embeds through a vLLM
  server, scores with a paired bootstrap, measures speed and checks served vectors against transformers.
- [`scripts/`](scripts/): the build script and the recipe dry run. The shell launchers that ran each phase on the GPU
  host (smoke test, builds and quality, speed, clean download) are host-specific and are not published.

## Recipes

- [`../../recipes/nemotron-3.5-embed-8b/darkstar-nemotron-3.5-embed-8b-base-modelopt-w4a16-nvfp4.yaml`](../../recipes/nemotron-3.5-embed-8b/darkstar-nemotron-3.5-embed-8b-base-modelopt-w4a16-nvfp4.yaml):
  schema-2 build recipe for the published cell
- [`../../configs/modelopt/recipes/nemotron-3.5-embed-8b/`](../../configs/modelopt/recipes/nemotron-3.5-embed-8b/):
  the three ModelOpt operator recipes (A, C, D)

## Candidate selection

**Builds.** All candidates quantize the text tower from the upstream BF16 weights with ModelOpt `0.48.0` and NVFP4
weights (group 16, MSE FP8-scale sweep).

**Bar.** Posted on the issue before any scoring: at least 98% of BF16 mean nDCG@10 on both frozen sets. It has to hold
for all-candidate vectors and for BF16 documents searched with candidate queries. The fastest passing candidate ships.

| Candidate | Text tower | MTEB Nano (Δ, 95% CI) | LMEB Dialogue (Δ, 95% CI) | Long-input speed vs BF16 | Weights |
|---|---|---|---|---|---|
| BF16 control | — | 0.6436 | 0.5788 | 1.00x | 15.74 GiB |
| BF16 replica | — | +0.0005 (−0.0001, +0.0016) | +0.0015 (−0.0004, +0.0039) | — | — |
| A | W4A4, every linear | +0.0029 (−0.0054, +0.0115) | −0.0311 (−0.0425, −0.0200) | 2.41-2.56x | 5.77 GiB |
| C | W4A4, MLP only | −0.0006 (−0.0076, +0.0062) | −0.0190 (−0.0285, −0.0096) | not measured | 7.69 GiB |
| **D (published)** | **W4A16, every linear** | **+0.0017 (−0.0033, +0.0067)** | **−0.0071 (−0.0137, −0.0004)** | **1.02-1.04x** | **5.82 GiB** |

**Outcome.**
- **A and C failed** on LMEB Dialogue: 94.63% and 96.72% of BF16 there, against 100.44% and 99.90% on MTEB Nano.
- **D was added after that**, as amendment R33-A1. It was posted on the issue before D was scored, and the bar did not
  change.
- **D passed** at 100.26% and 98.78%, so it shipped.

**What D gives up.** It is weight-only, so its gain is memory (5.82 against 15.74 GiB of weights) rather than speed.
W4A4 would have run 2.41-2.56x faster on long inputs, but A did not pass.

**Why A and C failed.** The loss is in document vectors of short texts, and two observations point there:
- **Short documents.** LMEB's conversation turns have a median of 168 characters, against 811 on MTEB.
- **Queries pass.** BF16 documents searched with A's or C's queries pass on both sets.

With mean pooling, per-token activation-quantization error averages over few tokens in a short text.

**Next to Qwen3-Embedding-8B.** Both Qwen3-Embedding-8B builds ran on the same two sets in the same session:

| | MTEB Nano | LMEB Dialogue |
|---|---|---|
| Qwen3-Embedding-8B BF16 | 0.6541 | 0.3943 |
| Qwen3-Embedding-8B Darkstar W4A4 | 0.6499 | 0.3791 |
| Nemotron-3.5-Embed-8B BF16 | 0.6436 | 0.5788 |
| **Nemotron-3.5-Embed-8B D** | **0.6453** | **0.5717** |

Nemotron BF16 minus Qwen3-Embedding-8B BF16:
- **MTEB Nano:** −0.0105 (95% CI −0.0336 to +0.0127).
- **LMEB Dialogue:** +0.1844 (95% CI +0.1579 to +0.2108); Nemotron wins all 9 splits.

## Gotchas

- **Prompts and tokens.** Use `query: ` and `passage: `. The chat template escapes `[` and `<` with a zero-width
  non-joiner and adds no BOS/EOS. Send prefixed text with `"add_special_tokens": false`.
- **Bidirectional text tower.**
  - vLLM serves it as a non-causal embedding model.
  - The build script checks bidirectionality before quantizing: changing the last token must move the first token's
    hidden state.
  - The smoke test checks vLLM's served vectors against transformers before any scoring.
- **Triton 3.9.** ModelOpt 0.48.0's Triton dynamic-FP4 fake-quant kernel calls `tl.make_block_ptr`, which Triton 3.9 (with
  torch 2.15) removed. Calibration doesn't use it. The build's post-calibration reference forward routes through ModelOpt's
  CUDA extension instead.
- **vLLM exclusion names.** vLLM 0.32 runs the text tower as `language_model.model.*`, and its Mistral3 prefix mapper does
  not rewrite the checkpoint's `language_model.*` exclusion names. A BF16 module excluded only under the checkpoint name
  is loaded as NVFP4 and fails (`AssertionError` in `load_qkv_weight`). The build adds the vLLM-internal twin of every
  text-tower exclusion, and the validator requires it.
- **Speed harness.** Use keep-alive connections. A first speed run, with one new TCP connection per request, recorded
  five impossible cells from ~69 s connection stalls through Docker Desktop's published port. That run was discarded. The
  harness now keeps one connection per client thread and fails any cell with a request over 20 s.

## License

OpenMDW-1.1, inherited from upstream; the published repository ships the license text and NVIDIA's notices.
