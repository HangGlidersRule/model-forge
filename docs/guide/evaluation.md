# Evaluation

Every quality or speed number Model Forge publishes is measured on the exact artifact it describes, under a protocol
recorded with the result, and compared with the family's own measured controls. No number is inherited from another
checkpoint, protocol or vendor table.

## Principles

- **Same artifact, same protocol.** A cell measured on a different build, decoder, thinking mode or harness is a
  different cell. Missing cells say `not measured`.
- **Full denominator.** Accuracy is `correct / total`; unparseable, truncated and failed requests count as wrong and
  are reported, never dropped.
- **Measure the noise floor.** Run the BF16 control twice. Their disagreement sets the band a candidate's delta is
  judged against (for GPQA Diamond, see below).
- **Pre-register the bar.** Pass bars, selection rules and evaluation sets are posted before results exist.
- **Failed gates block shipping.** Redo the method rather than move the bar. An owner may still decide to ship
  something that failed; that decision is stated in the card, ledger and release notes.
- **Speed numbers name their engine configuration.** A throughput figure without the serving flags, GPU and concurrent
  load is not a comparison.

## GPQA Diamond

The intelligence gate for chat models. Normative definition:
[`../gpqa-abliteration-protocol.md`](../gpqa-abliteration-protocol.md).

- **Harness:** [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench), profile
  `gpqa-diamond`: the model's chat template, thinking on, temperature 0, deterministic per-item letter shuffle, exact
  match, Wilson confidence intervals. Runner: `scripts/run_llmbench_gpqa.sh`.
- **Serve with the model's reasoning parser** (`--reasoning-parser nemotron_v3` for Nemotron) and a generous token cap
  (65,536 in recent campaigns); report truncation and unparseable counts.
- **All 198 items, full denominator.** No item subset, ever: a "2-question" preflight is a separate copy of the dataset.
- **Noise band.** Two BF16 runs disagree on some items. With `d` discordant items out of `n = 198`, the band is
  `1.96 * sqrt(d) / n` as a fraction, or `1.96 * sqrt(d)` questions: 37 discordant items gave a band of 11.9 questions
  on Lightning. A candidate passes when `candidate - reference >= -band`; report a paired bootstrap 95% interval as
  well.
- **Stock lm-eval GPQA tasks are supporting evidence only:** without the chat template and thinking they under-read
  Nemotron models, and the completions adapter corrupts numbers.
- **The legacy in-repo harness** (`src/model_forge/gpqa/`, boxed answers, sampled at temperature 1.0, thinking off) is
  frozen for continuity of the Qwen3.8-27B results and is not used for new claims.

## Next-token KL against BF16

A cheap, sensitive quality probe for quantized builds (`kl_probe.py` in a family's `eval/`).

- Serve BF16 and the candidate with identical flags; request prompt log-probabilities (top 20) with `max_tokens=1` on
  64 held-out prompts.
- Per position: KL(BF16 || candidate) over BF16's top 20 tokens plus a remainder bucket, and top-1 agreement.
- Use one prompt set to **select** among candidates and a disjoint one to **report**.
- KL is diagnostic unless pre-registered as a gate; it moved with GPQA on every Lightning candidate.

## Task-specific gates

| Family type | Gate | Example |
|---|---|---|
| Embedding model | Retrieval nDCG@10 on pinned public sets, as a fraction of BF16, both all-quantized and BF16-corpus x quantized-query (mixing old and new vectors); cosine agreement with stored vectors | Qwen3-Embedding-8B: 98.81% / 99.79% of BF16 on six MTEB Nano sets |
| Reranker | nDCG / MRR on pinned retrieval sets against the BF16 reranker | Qwen3-Reranker-4B |
| Decision / scoring model | Non-inferiority against the current deployment on its own test set: KL to the teacher, choice top-1, AUROC, score error, with the noise floor from a repeated reference run | JEV-27B (`models/jev-27b/eval/gate_noninferiority.py`) |
| Chat model | GPQA Diamond plus KL; behavior gate for abliterated products | Lightning, Qwen3.8-27B |

## Speed

- **Prefill:** aggregate prompt tokens per second and median time to first token at ~3k and ~50k-token prompts,
  concurrency 1 and 8, one output token, a unique prefix per request so the prefix cache can't answer.
- **Decode:** single-stream tokens per second on a fixed long answer (thinking off), median of three, with the
  speculative-decoding setup stated (MTP depth, drafter, acceptance counters).
- **Capacity:** KV-cache pool size and maximum concurrency at the served context length, from the engine's startup log.
- **Measure on the hardware you claim.** A layout that wins on a 96 GB card can lose on a 32 GB card: memory-bound
  decode pays for every BF16 byte the build keeps.
- Quarantine cold compile and autotune runs; a promoted speed winner must reproduce after a clean restart.

## Reproducibility across serving instances

Low-precision kernels autotune per process. On a W4A4 embedder the same files gave a mean cosine of 0.983 between two
fresh containers. When outputs are stored or compared numerically:

- measure the spread first: the same request twice, a different batch shape, two fresh containers, BF16 alongside;
- judge any gap against that spread, never against a fixed threshold;
- ship a stable serving profile (batch-invariant mode plus a persisted compile cache) with its own measured quality and
  speed.
