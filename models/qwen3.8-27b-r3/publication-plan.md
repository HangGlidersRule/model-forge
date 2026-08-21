# Publication plan

This family has four complete evaluation cells: unchanged upstream BF16 control, clean ModelOpt
NVFP4, abliterated BF16, and abliterated ModelOpt NVFP4. Publication applies only to the three
HangGlidersRule-owned Darkstar artifacts. The BF16 control remains the upstream
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` checkpoint under Apache-2.0;
HangGlidersRule does not own or republish those unchanged weights.

## Repositories

One engineering repository owns the reproducible pipeline, containers, tests, documentation, and aggregate results:

- GitHub: `HangGlidersRule/model-forge`

Exactly three public HangGlidersRule repositories contain complete, hash-verified checkpoints and
final cards:

- `HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8`
- `HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-BF16`
- `HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8`

No HangGlidersRule Base-BF16 repository exists or is planned. Rejected and unbuilt W4A4 candidates
are not repository targets.

## Quantization path (ModelOpt)

Publication NVFP4 checkpoints are rebuilt with **NVIDIA ModelOpt** `0.46.0rc2`
(`43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`), not llm-compressor compressed-tensors.

See [`modelopt/README.md`](modelopt/README.md). Prior compressed-tensors NVFP4 artifacts are
**rejected/historical** and must never be overwritten or republished as current.

The Abliterated ModelOpt NVFP4 build reuses the selected clean mixed W4A16-NVFP4+FP8 recipe. It is
complete with `148/198 = 74.75%` matched GPQA and a frozen MTP10 runtime.

## Container artifacts

**GHCR is not required for this release.** The final model cards contain reproducible `vllm serve` commands and use the pinned upstream/local vLLM environment. No project container image is a publication artifact or gate dependency.

## Build and evaluation gates (complete for all four cells)

- [x] Rebuild clean/base NVFP4 with ModelOpt primary recipe; pass fail-closed validators.
- [x] Complete clean/base NVFP4 GPQA Diamond cell at **198/198** terminal parseable (full denominator): 153/198 = 77.27%.
- [x] Base ModelOpt candidate selection: the mixed W4A16-NVFP4+FP8 candidate was **selected on throughput** (203.636 tok/s at MTP4) at a GPQA within 2.02 pp of the Base BF16 baseline; the uniform W4A4 candidate was built and rejected at 129.441 tok/s.
- [x] Serve gates: tools/JSON/vision/MTP/prefix/concurrency; no scale/NaN/CUDA/EngineDead/500. All serving gates pass for all four products.
- [x] Base BF16 and Abliterated BF16 built and fully evaluated (frozen full-denominator GPQA, independent throughput profiles, fresh refusal/abliteration evals).
- [x] Build and validate the Abliterated ModelOpt NVFP4 with the identical selected mixed recipe:
  `_SUCCESS` SHA `3d89ec57c1371e142adc2584de079b54a0e1d8c12dc9550118d0a851da020a79`,
  manifest SHA `642dbbe89b085a2daf5119c37c0496576a475ed64c36653fc993c04abaf2ca9f`.
- [x] Resolve the Abliterated ModelOpt target to the built candidate id: resolved to `...-W4A16-NVFP4-Mixed-FP8`.
- [x] Complete Product 4 matched GPQA at 198/198 terminal parseable: 148/198, zero errors/timeouts/parse errors, thinking off.
- [x] Complete Product 4 fresh behavior evaluation: 283/283 terminal, 200/200 harmful compliance, 0/83 safe over-refusals, zero errors.
- [x] Freeze Product 4 runtime: alias `darkstar-qwen38-abliterated-nvfp4`, container
  `vllm-darkstar-qwen38-abliterated-modelopt`, MTP10, FlashAttention, BF16 KV, context 126144,
  scheduler 32768, `max_num_seqs=16`, prefix caching, and chunked prefill.

## Publication-only gates

- [x] Upload and byte-verify the final README card in each private repository.
- [x] Upload and hash-verify config, index, and every frozen weight shard in each private repository.
- [x] Verify anonymous public Hugging Face visibility and pin current revisions.
- [x] Record clean download, boot, and models/text/strict-JSON/tool/vision smoke evidence with zero failures and empty fatal logs.
- [x] Record GHCR as not required; final cards contain reproducible pinned vLLM commands.
- [x] Finalize every model card with the exact planned immutable tag `darkstar-qwen3.8-27b-v1.0.0`.
- [x] Cut immutable release tag `darkstar-qwen3.8-27b-v1.0.0`; every applicable release gate is verified.

## Result-publication rules

- Publish all four measured matrix cells without changing their frozen protocols or denominators,
  while clearly identifying BF16 as the unchanged upstream control.
- Never substitute upstream Qwen's reported figure for any Darkstar result.
- Report numerator, denominator, completion coverage, timeout policy, and protocol beside every score.
- Never report completed-only accuracy as the publication headline.
- Do not call project results Artificial Analysis measurements.
- Do not commit GPQA question text or answer keys.
- Separate single-stream throughput from aggregate concurrent throughput.
- Keep runtime KV BF16 during recipe attribution comparisons.
