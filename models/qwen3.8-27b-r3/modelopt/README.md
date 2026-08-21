# ModelOpt quantization migration (Qwen3.8)

Replaces **rejected** llm-compressor / compressed-tensors NVFP4 artifacts with
self-owned NVIDIA ModelOpt unified-HF checkpoints.

## Status

| Artifact class | Status |
| --- | --- |
| Prior llm-compressor compressed-tensors NVFP4 (clean + R3) | **Rejected / historical** — keep for lineage; never overwrite |
| ModelOpt clean / base NVFP4 | **Locally complete, unpublished** — selected mixed W4A16-NVFP4+FP8 candidate; fail-closed validators passed, GPQA 153/198 full denominator, 203.636 tok/s at MTP4; only publication-only gates remain |
| ModelOpt Darkstar Abliterated NVFP4 (internal R3) | **Locally complete, unpublished** — selected/promoted mixed W4A16-NVFP4+FP8 candidate; every build gate verified; GPQA 148/198 full denominator; frozen MTP10 runtime; only publication-only gates remain |

## Pinned toolchain

Recorded in [`configs/modelopt/pin.json`](../../../configs/modelopt/pin.json):

- Package: `nvidia-modelopt==0.46.0rc2`
- Git commit: `43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a` (tag `0.46.0rc2`)
- Wheel SHA-256: `d6f6964b76c9e3f156ed1f3627d406b187c454614ab8e409a3796568cd487bbb`
- Entry point: `examples/hf_ptq/hf_ptq.py`
- Export: unified HF (`hf_quant_config.json`) — not Diffusers, not hand-edited metadata

### Reproducible container

[`containers/modelopt/Dockerfile`](../../../containers/modelopt/Dockerfile) +
[`containers/modelopt/build.sh`](../../../containers/modelopt/build.sh) build the
pinned image `ghcr.io/hangglidersrule/model-forge-modelopt:0.46.0rc2-43fd41a`.
Every build argument is sourced from `pin.json`, so the image can never float:

- Base `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` (noble ships `python3.12` in its
  default repositories, so no third-party PPA) with the interpreter in a `/opt/venv`
  virtualenv, and Blackwell PyTorch (`torch==2.8.0` on `cu128`,
  `TORCH_CUDA_ARCH_LIST=12.0`) for the RTX PRO 6000 Blackwell (sm_120).
- `nvidia-modelopt` installed **only** from the exact wheel URL, verified against the
  pinned SHA-256 at build time and requested as `<wheel>[hf]` so the extra's
  dependencies resolve without re-installing the package itself from PyPI. The
  installed distribution's `direct_url.json` is asserted to be that local file, and
  `transformers>=4.57,<5.15` is applied in the same resolution.
- `NVIDIA/Model-Optimizer` checked out at the immutable commit and asserted with
  `git rev-parse`, so `examples/hf_ptq/hf_ptq.py` is byte-for-byte the pinned source.

### Real `hf_ptq.py` invocation

The runner passes upstream-accurate flags (read from the pinned source):
`--calib_size 512,512` (two dataset sizes, **not** a collapsed `1024`),
`--calib_seq 2048`, `--batch_size 1`,
`--dataset cnn_dailymail,nemotron-post-training-dataset-v2`, `--recipe`,
`--kv_cache_qformat none` (keeps KV **BF16**; upstream default `fp8_cast` would emit
FP8 KV metadata), `--trust_remote_code`, unified HF export (`--export_fmt hf`). The
seed is the upstream `RAND_SEED = 1234`, matching the calibration contract.

## Candidates (bounded; no combinatorial sweep)

1. **Primary** — [`configs/modelopt/recipes/nvfp4_mlp_only_mse-kv_bf16.yaml`](../../../configs/modelopt/recipes/nvfp4_mlp_only_mse-kv_bf16.yaml)
   Upstream `nvfp4_mlp_only_mse-kv_fp8_cast` with FP8 KV removed. W4A4 NVFP4 group-16 on language MLP gate/up/down only (MSE static weights, dynamic activations). BF16 attention, all GatedDeltaNet, vision, MTP, lm_head, embeddings, norms. Runtime KV BF16.
2. **Secondary (OMLP)** — [`configs/modelopt/recipes/nvfp4_omlp_only_mse-kv_bf16.yaml`](../../../configs/modelopt/recipes/nvfp4_omlp_only_mse-kv_bf16.yaml)
   After cheap GPQA screen of (1): add softmax-attention `o_proj` W4A4 only. Q/K/V stay BF16.
3. **Selected (mixed)** — [`configs/modelopt/recipes/w4a16_nvfp4_mse-fp8_attn-kv_bf16.yaml`](../../../configs/modelopt/recipes/w4a16_nvfp4_mse-fp8_attn-kv_bf16.yaml)
   Selected as the Base ModelOpt product on throughput. Upstream qwen3_5 MSE W4A16 NVFP4 on language MLP **and `lm_head`** + FP8 attention/GDN with BF16 KV. Chosen at 203.636 tok/s (MTP4) vs the uniform W4A4 candidate's 129.441 tok/s. **Hazards:** Marlin W4A16 (not native Blackwell W4A4); mixed FP8 attention.

Image-text calibration is a separate optional candidate (ModelOpt VLM path is beta / not Qwen-validated) and is **not** part of the primary contract.

## Calibration (primary)

- Blend: `cnn_dailymail` then `nemotron-post-training-dataset-v2`
- Sizes: `512,512` (1024 total), batch 1, sequence 2048, seed 1234, layerwise false
- Pinned revisions in `pin.json`; truncation report required in stage SUCCESS metrics

## Artifact identities and private targets

Each NVFP4 id encodes the precision class of the artifact it will hold:

- `Qwen/Qwen3.8-27B` (external unchanged BF16 reference; not an owned target)
- `HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8` (selected clean candidate; locally complete)
- `HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4` (built and rejected on throughput)
- `HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-BF16` (locally complete)
- `HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8` (resolved,
  selected/promoted target; locally complete)
- `HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A4-NVFP4` (candidate id, not built)

The three selected owned repositories are public at pinned revisions, and no bare
`Base-NVFP4`/`Abliterated-NVFP4` name is reserved. `Qwen/Qwen3.8-27B` remains the external upstream
BF16 reference and is not branded Darkstar. `R3` remains the internal abliterated-edit lineage id
only.

## Remote rebuild sequence (mcprue, D: only)

1. Dry-run `scripts/qwen3_8/run_qwen38_modelopt_mcprue.sh` (default) to print the plan and
   the exact `hf_ptq.py` command without touching the GPU. The runner picks its own interpreter
   (repo `.venv`, else `python3`/`python`; override with `PYTHON_BIN`) and imports `model_forge`
   from `<repo>/src`, so no venv activation is required on mcprue; an interpreter that cannot
   import `model_forge` aborts the run before any work.
2. `EXECUTE=1 DRY_RUN=0` performs real work: it snapshots the **currently running** runtime
   via `docker inspect`/`docker logs` and renders a usable `restore.sh`
   (`scripts/qwen3_8/snapshot_runtime.py`), or accepts an explicit `RESTORE_COMPOSE`. It does
   **not** depend on any compose file we do not own. Check free space (≥200 GiB default),
   append-only snapshot, refuse overwrite.
3. Run the pinned ModelOpt image + `examples/hf_ptq/hf_ptq.py` with the real calibration
   contract above (built by `src/model_forge/modelopt/execution.py`), exporting into a partial
   directory. A non-zero container exit aborts before validation (errors are never swallowed).
4. Fail-closed validators against the actual export: module policy, no vision quant, exactly
   15 BF16 MTP tensors, finite (no NaN/Inf/zero) scales read from the real safetensors, no FP8
   KV metadata, no mixed fused groups, tokenizer/config drift, then a SHA-256 manifest and
   `_SUCCESS.json` — written only after every check passes — followed by an **atomic**
   partial→final promotion that refuses to overwrite. On failure the captured restore artifact
   rolls the previous runtime back (unless `PROMOTE=1`).
5. Cheap deterministic GPQA screen; publication requires full **198/198** terminal parseable responses (never completed-only accuracy). Matched quality must not be worse than BF16 by >1 pp unless the threshold is explicitly changed.
6. Serve gates: no vLLM scale/accuracy/NaN warning, CUDA/EngineDead/500; tools/JSON/real vision/MTP/prefix/concurrency. Start MTP validation at **1** speculative token, then tune upward after exact-output and acceptance proof. Keep runtime KV BF16 during recipe attribution.
7. For the abliterated build: use `SOURCE_KIND=abliterated ALLOW_ABLITERATED=1` with the identical
   selected mixed W4A16-NVFP4+FP8 recipe. The runner passes `--source-kind abliterated --allow-abliterated` to the
   quantization CLI. The pre-rename interfaces (`SOURCE_KIND=darkstar`, `ALLOW_DARKSTAR`,
   `--source-kind darkstar`, `--allow-darkstar`) are **deprecated aliases**: they are normalized
   to the canonical terms before any guard or path decision and warn on use. Artifacts validated
   before the rename recorded `source_kind: darkstar` in their build identity; those markers are
   accepted as the same identity when every other pinned field matches, and are still refused on
   any recipe, pin, calibration, candidate, or source difference.
8. Throughput tuning only after quality.

## Completed Abliterated ModelOpt evidence

- Artifact: `${PUBLIC_ARTIFACT_PATH}`
- `_SUCCESS.json` SHA-256: `3d89ec57c1371e142adc2584de079b54a0e1d8c12dc9550118d0a851da020a79`
- Manifest SHA-256: `642dbbe89b085a2daf5119c37c0496576a475ed64c36653fc993c04abaf2ca9f`
- Recipe SHA-256: `90fc6b37c00334debd49f1975ab406b5e20667f07e4be0be3e463a648abac642`
- GPQA: `148/198 = 74.75%`, 198/198 terminal parseable, thinking off, zero
  errors/timeouts/parse errors
- Behavior: 283/283 terminal, 200/200 harmful compliance, 0/83 safe over-refusals, zero errors
- Runtime: MTP10, FlashAttention, BF16 KV, context 126144, scheduler 32768,
  `max_num_seqs=16`, prefix caching, chunked prefill
- API alias: `darkstar-qwen38-abliterated-nvfp4`; container:
  `vllm-darkstar-qwen38-abliterated-modelopt`; Compose:
  [`containers/serve/darkstar-qwen38-abliterated-nvfp4.yml`](../../../containers/serve/darkstar-qwen38-abliterated-nvfp4.yml)

## Code entry points

- `src/model_forge/modelopt/` — pin, policy, calibration, validators, and:
  - `execution.py` — the real `hf_ptq.py` argv + `docker run` plan (mounts source,
    export, recipes, HF cache, calibration cache)
  - `runtime.py` — render a `restore.sh` from a `docker inspect` snapshot
  - `finalize.py` — scale reading, fail-closed validation, SHA-256 manifest, `_SUCCESS`,
    atomic promotion
  - `identity.py` — the complete build identity (pin + candidate + recipe digest +
    calibration + source) that idempotency compares. A build is skipped only when the
    artifact's `_SUCCESS.json` records this exact identity; a stale marker under the
    same artifact name is refused, never reported as success
- `src/model_forge/gpqa/` — frozen harness
- `scripts/qwen3_8/quantize_qwen38_modelopt.py` — dry-run plan / `--execute` real run
- `scripts/qwen3_8/run_qwen38_modelopt_mcprue.sh` — D:-only orchestration
- `scripts/qwen3_8/snapshot_runtime.py` — runtime snapshot → restore artifact
- `containers/modelopt/{Dockerfile,build.sh}` — pinned reproducible image
- Forge recipes: `recipes/qwen3.8-27b/darkstar-qwen3.8-27b-base-modelopt-nvfp4.yaml`,
  `darkstar-qwen3.8-27b-abliterated-modelopt-nvfp4.yaml`
