# Validation and code inventory: reusable framework vs Qwen-specific adapter

This record documents which code is genuinely reusable framework and which is a Qwen3.8-specific
compatibility adapter. **The Qwen3.8 GPU build/edit/quantize pipeline is not architecture-generic.**
Reusing it on another model family requires new selectors, new expected tensor inventories, and new
runtime assumptions — not just a new recipe file.

## Reusable framework (`src/model_forge/`)

These modules are architecture-agnostic and are the intended extension surface for new families:

| Module | Role | Family coupling |
|---|---|---|
| `pipeline.py` | Canonical JSON hashing, SHA-256 file/tree manifests, atomic stage dirs, success-marker validation, run locks | None |
| `recipe.py` | Generic schema-2 recipe parsing, full-SHA enforcement, visual-selector rejection, config hashing | None (declarative; executes nothing) |
| `selectors.py` | Pure-Python tensor-name selector matching | None (patterns are supplied by the caller) |
| `abliteration.py` | Published refusal-direction math (measurement + projection) on supplied tensors | Low (math is generic; targets are supplied) |
| `stats.py`, `scoring.py`, `scorecard.py` | Paired statistics and scoring | None |
| `arena.py`, `cases.py`, `corpus.py`, `client.py`, `runner.py`, `models.py`, `selectors.py` | Legacy OpenAI-compatible evaluation harness (the model "bake-off" arena) | None (talks to any OpenAI-compatible server) |
| `coding.py`, `longctx.py`, `refusal.py`, `tools.py`, `performance.py` | Reusable evaluation case/metric helpers | None |
| `cli.py` | `recipe validate` (inspect only) and `run` (legacy evaluator) | None |

Reusability here means the code makes no assumptions about a specific architecture's tensor names,
layer counts, MTP layout, or vision tower. Callers supply those.

## Qwen3.8-specific compatibility adapter

### `src/model_forge/experiment.py`

`experiment.py` is **not** a generic pipeline. It is a compatibility adapter that projects a generic
recipe into the concrete Qwen3.8 abliteration+NVFP4 experiment contract. It hardcodes assumptions
that only hold for this family, including:

- exactly **one** `abliteration` transform must be present, and quantization must be present;
- a complete Qwen3.8 validation gate set (`vision_byte_identical`, `mtp_present`,
  benign-KL and perplexity deltas) is required;
- MTP-based speculative runtime fields (`mtp_depth_initial`, `mtp_sweep_range`) are required;
- the abliteration target contract is the Qwen3.8 layout: layer 38, `expected_target_count = 131`
  residual-writing tensors, a preserved 333-tensor vision tower, and 15 preserved MTP tensors.

A different architecture would violate these assumptions. Do not treat a passing
`recipe validate` as evidence that `experiment.py` (or the GPU pipeline it configures) will work for
a non-Qwen3.8 model.

### `scripts/qwen3_8/`

The operational launchers are family-bound by design and encode concrete tensor selectors, module
names, MTP handling, and mcprue-specific build/checkpoint orchestration:

- `materialize_abliteration_corpus.py`, `measure_qwen38_refusal_direction.py`,
  `apply_qwen38_abliteration.py`, `validate_qwen38_bf16_edit.py`
- `quantize_blackfrost_nvfp4.py` (rejected/historical llm-compressor path, pre-Darkstar naming),
  `quantize_qwen38_nvfp4.py`, `quantize_qwen38_modelopt.py` (current NVIDIA ModelOpt path)
- `run_qwen38_build_mcprue.sh`, `checkpoint_vllm_mcprue.sh`

## Validation coverage

| Area | Test / gate | Kind |
|---|---|---|
| Generic recipe loading (non-legacy) | `tests/test_repo_integrity.py` | Framework |
| Internal Markdown links resolve | `tests/test_repo_integrity.py` | Repo hygiene |
| Recipe schema + rejections | `tests/test_recipe.py` | Framework |
| Package/CLI identity after migration | `tests/test_package_identity.py` | Repo hygiene |
| `recipe validate` CLI | `tests/test_cli.py` | Framework |
| Qwen experiment projection | `tests/test_experiment.py` | Qwen-specific adapter |
| Abliteration math | `tests/test_abliteration.py` | Framework math (Qwen targets) |
| Qwen NVFP4 quantization script | `tests/test_quantize_qwen38_nvfp4.py` | Qwen-specific adapter |
| BF16 edit / vision / MTP inventory gates | `scripts/qwen3_8/validate_qwen38_bf16_edit.py` | Qwen-specific adapter (GPU, not in CI) |

The GPU-bound edit/quantize/validate steps run on hardware, not in CI. CI validates the framework,
the declarative recipes, and repository hygiene only.

## Rule

When extending to a new family, add framework capability to `src/model_forge/` and a new
`scripts/<family_slug>/` adapter. Do not generalize `experiment.py` in place or claim the Qwen3.8
GPU pipeline is architecture-generic.
