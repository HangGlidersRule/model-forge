# Scripts

Family-specific operational launchers: the concrete, GPU- and environment-bound entry points that
drive one model family's build. They are intentionally **not** generic framework code.

## Naming convention

- One subdirectory per family, using the family slug: `scripts/<family_slug>/` (e.g. `scripts/qwen3_8/`).
- Use an underscore slug (`qwen3_8`) for Python import friendliness even though the recipe/model-record
  id uses dots and dashes (`qwen3.8-27b`).
- Script names describe the concrete action and target, e.g. `apply_qwen38_abliteration.py`,
  `quantize_qwen38_modelopt.py`, `run_qwen38_modelopt_mcprue.sh`.
- `quantize_qwen38_nvfp4.py` / `run_qwen38_build_mcprue.sh` are the rejected compressed-tensors
  path (historical); new builds use the ModelOpt scripts.

## Extension boundary

- Reusable logic (hashing, manifests, recipe parsing, projection math, evaluation) lives in
  `src/model_forge/`. Scripts should import from there rather than reimplement it.
- Scripts may legitimately hardcode family-specific tensor selectors, module names, and stage
  ordering. That specificity is the point — see [`../models/qwen3.8-27b-r3/validation-inventory.md`](../models/qwen3.8-27b-r3/validation-inventory.md)
  for which pieces are reusable versus Qwen-specific.
- Do not add generic, cross-family abstractions here; promote them to `src/model_forge/` instead.
- Never hardcode credentials or private hostnames; read them from the environment.

## Current families

- [`qwen3_8/`](qwen3_8/) — Qwen3.8-27B corpus materialization, refusal-direction measurement,
  BF16 projection, edit validation, ModelOpt NVFP4 quantization, and D:-only mcprue runners.
  - `quantize_qwen38_modelopt.py` — dry-run validates the recipe and prints the exact
    `hf_ptq.py` command; `--execute` runs the pinned ModelOpt container for real, validates the
    export fail-closed, writes a SHA-256 manifest + `_SUCCESS`, and atomically promotes it.
    `--check-build-identity` reports whether an existing artifact was produced by the current
    build identity (exit 0 up to date, 20 build required, 5 refuse).
  - `run_qwen38_modelopt_mcprue.sh` — D:-only orchestration: append-only runtime snapshot,
    free-space and build-identity overwrite guards, real container execution,
    restore-on-failure. An existing artifact is skipped only when its `_SUCCESS.json` records
    the current identity; a stale marker is refused rather than reported as success. It resolves
    its interpreter itself (repo `.venv`, else `python3`/`python`, overridable with
    `PYTHON_BIN`) and exports `PYTHONPATH=<repo>/src`, so it needs no activated venv and
    refuses to start when the chosen interpreter cannot import `model_forge`.
  - `snapshot_runtime.py` — renders a `restore.sh` from a `docker inspect` capture so restore
    never depends on a compose file we do not own.
