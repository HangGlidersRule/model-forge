# Darkstar Nemotron-3.5-Lightning family

Darkstar is HangGlidersRule's overall model-tuning brand. This record contains four evaluated cells:
the unchanged upstream `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` control plus three owned
Darkstar artifacts (Base ModelOpt NVFP4, Abliterated BF16, and Abliterated ModelOpt NVFP4). **R1** is
the internal lineage id for the abliterated edit, not a public model name. The NVFP4 publication path
is **NVIDIA ModelOpt** (W4A16-NVFP4 experts, BF16 protected lm_head / Mamba / MTP).

## Release process and readiness

The four-cell release contract records the frozen evaluation evidence and gate status in
[`results/publication-readiness-ledger.json`](results/publication-readiness-ledger.json). For
publication purposes, the unchanged BF16 cell is an upstream control, not a HangGlidersRule product
or weight target. Its weights remain solely at
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16),
pinned to `d468880b6ad3c6e0d21377ce7242adaea4cc884d` under OpenMDW-1.1.

## Documents

- [`benchmark-matrix.md`](benchmark-matrix.md) — measured result matrix, caveats, and rendered gate status
- [`docs/gpqa-abliteration-protocol.md`](../../docs/gpqa-abliteration-protocol.md) — dataset lineage and exact evaluation semantics
- [`model-card/`](model-card/) — source cards for the owned Darkstar checkpoints and upstream control
- [`results/gpqa-matrix.json`](results/gpqa-matrix.json) — machine-readable aggregate GPQA matrix
- [`results/performance-*.json`](results/) — machine-readable per-decoder throughput sweeps
- [`results/publication-readiness-ledger.json`](results/publication-readiness-ledger.json) — machine-readable four-product release-gate ledger (source of truth)

## Recipes

- [`../../recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-base-modelopt-w4a16-nvfp4.yaml`](../../recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-base-modelopt-w4a16-nvfp4.yaml) — clean ModelOpt NVFP4
- [`../../recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-abliterated-bf16.yaml`](../../recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-abliterated-bf16.yaml) — R1 BF16 edit
- [`../../recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-abliterated-modelopt-nvfp4.yaml`](../../recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-abliterated-modelopt-nvfp4.yaml) — abliterated BF16 → NVFP4 quant

## Hugging Face repositories

Exactly two public HangGlidersRule repositories contain complete, hash-verified checkpoints and
final cards:

- `HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16`
- `HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-ModelOpt-W4A16-NVFP4`

No HangGlidersRule repository exists or is planned for the unchanged BF16 control or the Base NVFP4
cell unless explicitly requested.

## Current status

- Canonical abliteration methodology, ModelOpt quant recipes/validators, GPQA harness, and throughput
  tuning engine are in-repo (`feat/lightning-abliterated-products`).
- GPU quantization / remote artifact mutation is a separate mcprue execution step.
- All four evaluation cells are complete: build, tune, behavior gate, GPQA (accepted protocol).
- Both owned checkpoints are staged for public release behind the publication ledger.
- License: OpenMDW-1.1 — NVIDIA upstream license; derivatives require attribution.
