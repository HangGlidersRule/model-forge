# Recipes

Family-specific build and evaluation recipes. Each recipe maps to a canonical product role in the
[four-product release process](../docs/darkstar-four-product-release-process.md); release-gate status is
tracked in the per-family publication-readiness ledger, not here.

## Qwen3.8-27B

Entry recipes:

- `qwen3.8-27b/darkstar-qwen3.8-27b-base-bf16.yaml` — Darkstar Base clean BF16 (locally complete)
- `qwen3.8-27b/darkstar-qwen3.8-27b-base-modelopt-nvfp4.yaml` — Darkstar Base clean ModelOpt NVFP4; selected mixed W4A16-NVFP4+FP8 (locally complete)
- `qwen3.8-27b/darkstar-qwen3.8-27b-abliterated-bf16.yaml` — Darkstar Abliterated BF16 (internal lineage R3; locally complete)
- `qwen3.8-27b/darkstar-qwen3.8-27b-abliterated-modelopt-nvfp4.yaml` — Darkstar Abliterated ModelOpt NVFP4; reuses the selected mixed W4A16-NVFP4+FP8 recipe (selected/promoted, locally complete; recipe SHA-256 `90fc6b37c00334debd49f1975ab406b5e20667f07e4be0be3e463a648abac642`)

Historical compressed-tensors recipes (rejected for publication, retained for lineage):

- `qwen3.8-27b/base-nvfp4.yaml`
- `qwen3.8-27b/r3-nvfp4.yaml`

ModelOpt PTQ YAML candidates live under `configs/modelopt/recipes/` (not schema-2 forge recipes).

Legacy comparison/eval specs live under `qwen3.8-27b/legacy/` and remain
compatible with `model-forge run --spec ...`.
