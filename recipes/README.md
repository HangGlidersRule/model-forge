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

## JEV-27B

- `jev-27b/darkstar-jev-27b-decision-modelopt-w4a4-nvfp4-mixed-fp8.yaml` — Darkstar JEV-27B Decision ModelOpt W4A4 NVFP4 + FP8 (merged LoRA; NVIDIA's Qwen3.8-27B recipe with `lm_head` BF16; published). Operator recipe SHA-256 `901247d03387022f42b23cdfb699b0b6ab7084c2809db17054d38826a4358058`.
- `jev-27b/darkstar-jev-27b-decision-qad-modelopt-w4a4-nvfp4.yaml` — Darkstar JEV-27B Decision QAD ModelOpt W4A4 NVFP4 (merged LoRA; all-NVFP4 PTQ start point, then QAD on the per-block weight scales; published by owner decision). Stage-1 operator recipe SHA-256 `2c2dd90a60140140f4992a705c34fa58048a61935cd3dbd19904ef8bd0b25e41`.

## Qwen3-Embedding-8B

- `qwen3-embedding-8b/darkstar-qwen3-embedding-8b-base-modelopt-w4a4-nvfp4.yaml` — Darkstar Base ModelOpt W4A4 NVFP4 (every decoder linear, `nvfp4_act_headroom` calibration; published). Operator recipe SHA-256 `2a5f6acb073072af0808ee7218bf6d870dcb569d4a241719d06be267de38381c`.

## Qwen3-Reranker-4B

- `qwen3-reranker-4b/darkstar-qwen3-reranker-4b-base-modelopt-w4a4-nvfp4.yaml` — Darkstar Base ModelOpt W4A4 NVFP4 (decoder MLP only, `nvfp4_act_headroom` calibration; published). Operator recipe SHA-256 `eb9a063a89cc527f58a0c68bcbc860427e9f3cf67c048033316da147c82686d6`.

## Nemotron-3.5-Lightning 30B-A3B

- `nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-base-modelopt-w4a4-nvfp4.yaml` — Darkstar Base ModelOpt W4A4 NVFP4 (routed and shared MoE experts; activation scales set to the lowest simulated NVFP4 error per stream after calibration). Operator recipe SHA-256 `6facf6cdc5486dba9a402881dae92c6c60d7ca7f8fc4e2bab608abec4948ee68`.

ModelOpt PTQ YAML candidates live under `configs/modelopt/recipes/` (not schema-2 forge recipes).

Legacy comparison/eval specs live under `qwen3.8-27b/legacy/` and remain
compatible with `model-forge run --spec ...`.
