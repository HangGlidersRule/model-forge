# Darkstar-Qwen3.8-27B family

Darkstar is HangGlidersRule's overall model-tuning brand. This record contains four evaluated cells:
the unchanged upstream `Qwen/Qwen3.8-27B` BF16 control plus three owned Darkstar artifacts (clean
ModelOpt NVFP4, abliterated BF16, and abliterated ModelOpt NVFP4). **R3** is the internal lineage id
for the abliterated edit, not a public model name. The NVFP4 publication path is **NVIDIA ModelOpt**
(prior llm-compressor compressed-tensors NVFP4 builds are rejected/historical).

## Release process and readiness

The existing four-cell release contract records the frozen evaluation evidence and gate status in
[`results/publication-readiness-ledger.json`](results/publication-readiness-ledger.json). For
publication purposes, the unchanged BF16 cell is an upstream control, not a HangGlidersRule product
or weight target. Its weights remain solely at
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), pinned to
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` under Apache-2.0.

## Documents

- [`benchmark-matrix.md`](benchmark-matrix.md) — measured result matrix, caveats, and rendered ledger status
- [`gpqa-protocol.md`](gpqa-protocol.md) — dataset lineage and exact evaluation semantics
- [`artifact-lineage.md`](artifact-lineage.md) — official source → Darkstar Base or R3 edit → NVFP4
- [`modelopt/README.md`](modelopt/README.md) — ModelOpt pin, recipes, rebuild gates
- [`validation-inventory.md`](validation-inventory.md) — reusable framework vs Qwen-specific adapter
- [`publication-plan.md`](publication-plan.md) — GitHub, GHCR, and Hugging Face release gates
- [`results/gpqa-matrix.json`](results/gpqa-matrix.json) — machine-readable aggregate matrix
- [`results/publication-readiness-ledger.json`](results/publication-readiness-ledger.json) — machine-readable four-product release-gate ledger (source of truth)
- [`benchmark-matrix.md`](benchmark-matrix.md) — curated clean-base and abliterated throughput summaries; raw performance evidence remains private
- Historical implementation plans remain in the private operator archive.
- [`model-card/`](model-card/) — source cards for the three Darkstar checkpoints and upstream control

## Recipes

- [`../../recipes/qwen3.8-27b/darkstar-qwen3.8-27b-base-modelopt-nvfp4.yaml`](../../recipes/qwen3.8-27b/darkstar-qwen3.8-27b-base-modelopt-nvfp4.yaml) — clean ModelOpt NVFP4
- [`../../recipes/qwen3.8-27b/darkstar-qwen3.8-27b-abliterated-bf16.yaml`](../../recipes/qwen3.8-27b/darkstar-qwen3.8-27b-abliterated-bf16.yaml) — R3 BF16 edit
- [`../../recipes/qwen3.8-27b/darkstar-qwen3.8-27b-base-bf16.yaml`](../../recipes/qwen3.8-27b/darkstar-qwen3.8-27b-base-bf16.yaml) — clean BF16 base
- [`../../recipes/qwen3.8-27b/darkstar-qwen3.8-27b-abliterated-modelopt-nvfp4.yaml`](../../recipes/qwen3.8-27b/darkstar-qwen3.8-27b-abliterated-modelopt-nvfp4.yaml) — reuses the selected mixed W4A16-NVFP4+FP8 recipe (locally complete)
- Historical compressed-tensors recipes retained under `base-nvfp4.yaml` / `r3-nvfp4.yaml` for lineage only

## Hugging Face repositories

Family collection: [Darkstar Qwen3.8-27B](https://huggingface.co/collections/HangGlidersRule/darkstar-qwen38-27b-6a8dfe77e150d32d21a8a876).

Exactly three public HangGlidersRule repositories contain complete, hash-verified checkpoints and
final cards:

- `HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8`
- `HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-BF16`
- `HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8`

No HangGlidersRule repository exists or is planned for the unchanged BF16 control. Rejected or
unbuilt W4A4 candidates are lineage records, not publication targets.

## Current status

- ModelOpt migration code, recipes, validators, GPQA harness, and D:-only mcprue runner are in-repo.
- GPU quantization / remote artifact mutation is a separate mcprue execution step.
- Prior compressed-tensors NVFP4 artifacts are rejected/historical.
- All four evaluation cells are complete, and the three owned checkpoints are public on Hugging Face
  at pinned revisions with config, index, and every weight shard verified. Clean download, boot, and
  smoke pass; GHCR is not required. Release tag `darkstar-qwen3.8-27b-v1.0.0` is cut; every applicable gate is verified.
- The Abliterated ModelOpt NVFP4 build reuses the selected mixed W4A16-NVFP4+FP8 recipe, scored
  `148/198 = 74.75%` GPQA, and freezes MTP10 with a 32K scheduler budget.
- Product 4 runtime identity: alias `darkstar-qwen38-abliterated-nvfp4`, container
  `vllm-darkstar-qwen38-abliterated-modelopt`, Compose
  [`containers/serve/darkstar-qwen38-abliterated-nvfp4.yml`](../../containers/serve/darkstar-qwen38-abliterated-nvfp4.yml).
