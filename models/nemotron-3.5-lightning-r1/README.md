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

Family collection: [Darkstar Nemotron-3.5-Lightning 30B-A3B](https://huggingface.co/collections/HangGlidersRule/darkstar-nemotron-35-lightning-30b-a3b-6a8dfe8c5ec5db63d6a031d0).

Five public HangGlidersRule repositories contain complete, hash-verified checkpoints and final cards:

- `HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A16-NVFP4`
- `HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A4-NVFP4`
- `HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8`
- `HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16`
- `HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-ModelOpt-W4A16-NVFP4`

No HangGlidersRule repository exists or is planned for the unchanged BF16 control.

### W4A4 NVFP4 cell (v1.1.0, issue 17)

[`Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A4-NVFP4`](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A4-NVFP4) runs the MoE experts in real FP4 (W4A4) on stock vLLM: prefill is 1.45-1.72x
the W4A16 build on vLLM's default kernel, at next-token KL 0.0246 to BF16 (W4A16 0.0189) and GPQA Diamond 130/198 (W4A16 136/198,
inside the pre-registered noise band of 11.9 questions). Its activation scales replace the calibrated ones, which clipped the
routed-expert down-projection inputs (KL 0.0795).

- Card: [`model-card/base-w4a4-nvfp4.md`](model-card/base-w4a4-nvfp4.md)
- Publication receipt (clean-download smoke): [`results/w4a4-publication-receipt.json`](results/w4a4-publication-receipt.json)
- Evidence: [`results/w4a4-gate-summary.json`](results/w4a4-gate-summary.json), built by
  [`eval/build_w4a4_summary.py`](eval/build_w4a4_summary.py) and checked against the card by
  [`eval/check_card_numbers.py`](eval/check_card_numbers.py)
- Recipe: [`../../recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-base-modelopt-w4a4-nvfp4.yaml`](../../recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-base-modelopt-w4a4-nvfp4.yaml)
- Calibration and held-out prompt builders: [`calibration/`](calibration/)
- Pre-registration, amendment and results: [issue 17](https://github.com/HangGlidersRule/model-forge/issues/17)

### W4A4 NVFP4 Mixed-FP8 cell (v1.2.0, issue 25)

[`Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8`](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8) is the W4A4 cell in NVIDIA's checkpoint layout, for memory-tight GPUs.
- **Same experts as v1.1.0:** expert weights and activation scales are bit-identical.
- **NVIDIA's layout around them:** FP8 Mamba projections, an NVFP4 weight-only `lm_head` and a calibrated FP8 KV cache.
- **On a 32 GB RTX PRO 4500**, against NVIDIA's W4A16 checkpoint (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`) with the
  same serve flags: prefill 1.21-1.49x and single-stream decode 0.99-1.03x. Against the same checkpoint on that card, the v1.1.0
  cell decodes at 0.81x. The family's W4A16 build was not measured on that card.
- **Quality:** GPQA Diamond 136/198, equal to the family's W4A16 build; next-token KL 0.0358 (v1.1.0: 0.0246).

- Card: [`model-card/base-w4a4-nvfp4-mixed-fp8.md`](model-card/base-w4a4-nvfp4-mixed-fp8.md)
- Publication receipt (clean-download smoke): [`results/nvl-publication-receipt.json`](results/nvl-publication-receipt.json)
- Evidence: [`results/nvl-gate-summary.json`](results/nvl-gate-summary.json), built by
  [`eval/build_nvl_summary.py`](eval/build_nvl_summary.py) and checked against the card by
  [`eval/check_card_numbers_nvl.py`](eval/check_card_numbers_nvl.py)
- Recipe: [`../../recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-base-modelopt-w4a4-nvfp4-mixed-fp8.yaml`](../../recipes/nemotron-3.5-lightning/darkstar-nemotron-3.5-lightning-30b-a3b-base-modelopt-w4a4-nvfp4-mixed-fp8.yaml)
- Pre-registration and results: [issue 25](https://github.com/HangGlidersRule/model-forge/issues/25)

## Current status

- Canonical abliteration methodology, ModelOpt quant recipes/validators, GPQA harness, and throughput
  tuning engine are in-repo (`feat/lightning-abliterated-products`).
- GPU quantization / remote artifact mutation is a separate mcprue execution step.
- All four evaluation cells are complete: build, tune, behavior gate, GPQA (accepted protocol).
- Both owned checkpoints are staged for public release behind the publication ledger.
- License: OpenMDW-1.1 — NVIDIA upstream license; derivatives require attribution.
