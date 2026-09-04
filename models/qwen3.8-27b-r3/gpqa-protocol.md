# GPQA Diamond reproduction protocol

The frozen GPQA semantics required for a matched publication cell are defined normatively in the
[four-product release process](../../docs/darkstar-four-product-release-process.md#5-frozen-gpqa-semantics).
This document records the family-specific dataset lineage and protocol detail.

## Dataset

The project evaluated the 198-question GPQA Diamond split introduced by:

- Repository: https://github.com/idavidrein/gpqa
- Paper: https://arxiv.org/abs/2311.12022
- Dataset license: CC BY 4.0 (see the upstream dataset archive)

The local run used the public `gpqa_diamond.csv` distributed by OpenAI simple-evals. Its SHA-256 was:

`41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`

No question text or answer key is committed to this repository. Reproduction scripts should acquire an authorized copy and verify the checksum.

## Prompt construction

For question index `i`:

1. Build a four-choice list from the correct answer and three incorrect answers.
2. Shuffle with `random.Random(i).sample(range(4), 4)`.
3. Ask the model to analyze the options and place the final option letter in `\boxed{}`.
4. Parse the last boxed A-D answer, falling back to an explicit `Answer: A-D` form.

## Sampling

Shared settings across the documented runs:

- temperature: `1.0`
- top-p: `0.95`
- top-k: `20`
- workers: `4`
- one sampled answer per question

Protocol differences are retained in the matrix rather than hidden:

- Current Darkstar products (thinking off): the frozen full-denominator runs on Base BF16, Base
  ModelOpt mixed W4A16-NVFP4+FP8, and Abliterated BF16 each completed **198/198** terminal parseable
  with **zero** timeouts, parse errors, or errors (157/198, 153/198, and 146/198 respectively).
- Superseded BF16 partials (historical): earlier non-frozen thinking-off runs used a five-minute
  client timeout and did not terminate on a subset of uncapped requests; they are `rejected_historical`
  in the ledger and never presented as a product score.
- R3 NVFP4 full run (rejected historical): thinking enabled; 15-minute client timeout in the original
  harness, on the rejected compressed-tensors artifact.

## Score semantics

The current products report the headline `correct / 198` (full denominator). Two score forms appear in
older raw artifacts:

- `correct / 198`: conservative end-to-end score when timeouts are counted as failures (and the only
  form used for the frozen full-denominator products, which had no timeouts).
- `correct / completed`: conditional accuracy among responses that completed and produced parseable
  answers — used only to characterize the superseded partial runs, always beside the timeout count.

The earlier non-frozen BF16 partials produced pathological non-termination on a subset of uncapped
requests, which is why both completion coverage and conditional accuracy were reported for them. The
current frozen products have no such timeouts.

The rejected historical R3 NVFP4 thinking-enabled run completed all 198 questions, so its
`164/198 = 82.83%` needs no conditional denominator; it remains rejected/historical.

## Why these are not Artificial Analysis scores

Artificial Analysis also evaluates GPQA Diamond, but uses its own independently controlled methodology. Sharing the same benchmark split does not guarantee identical prompting, sampling, reasoning budget, timeout policy, or answer extraction. Results here are project reproduction results against the upstream GPQA Diamond data—not submissions to, or measurements by, Artificial Analysis.

## Frozen in-repo harness

Use `src/model_forge/gpqa/` (and the ModelOpt rebuild screens):

- Dataset SHA-256 must match `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`.
- Append-only resumable JSONL journal; 1800s per-request timeout with retries.
- Publication completion requires all **198** terminal **parseable** responses.
- Headline accuracy is always `correct / 198` (full denominator). Completed-only
  accuracy may be reported as a secondary statistic but never as the publication score.
- Run the cheap deterministic screen first; full 198/198 is still required to publish.

## Reproduction requirements

A publishable rerun must record:

- exact model repository and revision;
- weight format and edit lineage;
- vLLM image digest/version;
- complete serve command;
- dataset checksum;
- prompt-template revision;
- sampling settings;
- thinking mode and reasoning settings;
- output cap;
- timeout policy;
- worker count;
- raw per-question journal;
- aggregate summary and SHA-256 manifest.

The clean/base NVFP4 cell remains required before claiming a complete matched four-way matrix.
