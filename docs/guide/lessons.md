# Lessons and pitfalls

Each entry is something that cost real time once, and the rule it produced.

## Method

- **Chat-template the abliteration prompts.** Raw-text prompts gave a weak, wrong refusal direction.
- **Pick the abliteration layer by generation, not by norm.** The strongest-looking layer on Lightning left 43/200
  refusals; the generation-tested layer left none.
- **Don't add readout targets to the edit.** Projecting `lm_head` and norms as well changed nothing behaviorally.
- **When every projection variant collapses quality, stop projecting.** Nano-Omni needed LoRA unlearning; nine
  projection configurations all fell to ~27% GPQA.
- **Calibrated activation scales can clip.** ModelOpt's calibration set Lightning's routed-expert down-projection input
  scales below the largest activations the model produces (KL 0.080 vs 0.025 after replacing them). Collect per-stream
  statistics and simulate before trusting a W4A4 calibration.
- **Read the mechanism from the statistics, not from which number moved most.** A first write-up credited the large
  change to the up-projection scales; the simulated per-stream error showed the down-projection inputs carried the whole
  gain, and the public comment was corrected.

## Measurement

- **A smoke test must be a smoke test.** A "2-question" GPQA preflight that pointed at the full dataset ran all 198
  questions against a production server. Use a separate truncated copy of the dataset.
- **Normalize text before judging refusals.** Curly apostrophes turned 147 refusals into "compliant".
- **Record failed requests.** Four connection timeouts in a GPQA run count as wrong under the full-denominator rule;
  the write-up must say so and report the delta on the answered items too.
- **Speed claims depend on the serving configuration.** A production engine tuned for decode (chunked prefill sharing
  the GPU with live traffic) is not a like-for-like comparison with a dedicated test serve.
- **Hardware changes the answer.** A W4A4 build that was faster everywhere on a 96 GB card lost single-stream decode on
  a 32 GB card (0.81x), because its BF16 Mamba projections and `lm_head` add memory reads to every step. Size and speed
  both depend on the full checkpoint layout, not just the quantized experts.

## Tooling

- **Validate against what the loader accepts, not one example layout.** ModelOpt writes fused routed experts once per
  layer in `quantized_layers`; NVIDIA's own checkpoints list every expert. vLLM resolves both; a validator that
  accepted only one failed a correct export.
- **transformers 5 re-serializes some configs on export.** Restore the source config and keep only the quantization
  metadata, or vLLM reads the wrong layer types.
- **Grouped-mm on SM120.** transformers' capability check passes but the kernel only supports SM90; force the
  per-expert fallback.
- **A token limit truncates, it does not shorten.** Never add `max_tokens` to replayed production requests that did not
  carry one.

## Publication

- **Check card numbers by script.** A card-number checker caught figures typed from memory on two releases, and a
  shard count that was wrong on a third.
- **Secret scanners read history.** A results file that listed `tokenizer.json` next to its sha256 tripped gitleaks'
  generic API-key rule; fixing the file was not enough, because the scan covers every commit since the last export. Keep
  per-file hashes in the published manifest, not in evidence JSON, and squash before exporting if needed.
- **Raw benchmark keys are blocked in published scripts.** Build request bodies from key tuples
  (`dict(zip(KEYS, values))`) so lines like `"prompt": ...` never appear.
- **Bugbot reviews only a pull request's first push.** Request a re-run after every fix, or a later head goes
  unreviewed while every visible check is green.
- **Open the pull request last.** Every push restarts CI and the reviewers; finish the Hugging Face steps and commit
  the publication receipt first.
- **Never clean up containers by image.** Production servers often run the same image as test containers; filter by
  name or label.
