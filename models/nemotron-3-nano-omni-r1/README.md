# Darkstar Nemotron-3-Nano-Omni family

Darkstar is HangGlidersRule's overall model-tuning brand. This record contains four evaluated cells:
the unchanged upstream `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` control plus three owned
Darkstar artifacts (Base ModelOpt NVFP4, Abliterated BF16, and Abliterated ModelOpt NVFP4). The NVFP4
publication path is **NVIDIA ModelOpt**. The abliterated edit for this family is **LoRA unlearning
SFT** (rank-32 adapter on the language trunk's q/k/v/o projections, α=64) rather than a closed-form
projection — on this hybrid Mamba2 + MoE architecture every closed-form projection variant measured a
severe GPQA collapse (26.8–27.3% with 60+ unparseable outputs across 9 configurations) while the SFT
route preserved measured intelligence at the documented cost. The LoRA-unlearning-SFT transform is a
first-class recipe schema type (see `src/model_forge/recipe.py`).

## Release process and readiness

The four-cell release contract records the frozen evaluation evidence and gate status in
[`results/publication-readiness-ledger.json`](results/publication-readiness-ledger.json). For
publication purposes, the unchanged BF16 cell is an upstream control, not a HangGlidersRule product
or weight target. Its weights remain solely at
[`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16),
pinned to `24e67ea000b7c2837fc8f9488aa2008524fac8ba` under the NVIDIA Open Model License.

The family release is the immutable tag
[`darkstar-nemotron-3-nano-omni-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3-nano-omni-v1.0.0).

## Documents

- [`model-card/`](model-card/) — source cards for the owned Darkstar checkpoints (the upstream control is linked, not republished)
- [`results/`](results/) — machine-readable gate evidence: GPQA, behavior evals, throughput sweeps, publication ledger

## Recipes

- [`../../recipes/nemotron-3-nano-omni/darkstar-nemotron-3-nano-omni-30b-a3b-reasoning-abliterated-bf16.yaml`](../../recipes/nemotron-3-nano-omni/darkstar-nemotron-3-nano-omni-30b-a3b-reasoning-abliterated-bf16.yaml) — LoRA unlearning SFT (r1) BF16 edit
- [`../../configs/modelopt/recipes/w4a16_nvfp4_lmhead_nemotron_h.yaml`](../../configs/modelopt/recipes/w4a16_nvfp4_lmhead_nemotron_h.yaml) — ModelOpt NVFP4 recipe (NVFP4 lm_head; used for both Base-NVFP4 and Abliterated-NVFP4)

## Hugging Face repositories

Family collection: [Darkstar Nemotron-3-Nano-Omni 30B-A3B](https://huggingface.co/collections/HangGlidersRule/darkstar-nemotron-3-nano-omni-30b-a3b-6a9a2608d4e2ad4c43a996a1).

Exactly three public HangGlidersRule repositories contain complete checkpoints and final cards:

- `HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Base-ModelOpt-W4A16-NVFP4`
- `HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-BF16`
- `HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-ModelOpt-W4A16-NVFP4`

No HangGlidersRule repository exists or is planned for the unchanged BF16 control.

## Measured gates (summary)

- **Behavior** (treadon/abliteration-eval, 200 harmful + 83 safe over-refusal, temp 0, thinking
  off, smart-quote-normalized refusal matching): Ablit-BF16 199/200 compliance + 0/83 (re-gate
  2026-09-03; shipped evidence 200/200), Ablit-NVFP4 200/200 + 0/83, 0 errors on both.
- **GPQA Diamond** (llm-inference-bench, chat template + thinking ON, temp 0, full 198):
  Base-NVFP4 84/198 = 42.4%, Ablit-BF16 58/198 = 29.3%, Ablit-NVFP4 58/198 = 29.3%.
- **Quantization losslessness**: Ablit-NVFP4 equals Ablit-BF16 on the same runner the same day
  (58/198 = 58/198) while buying +36% aggregate throughput.
- **Throughput** (single-stream, 4K/16K/48K weighted 0.6/0.3/0.1, no speculative decode — this
  family has no MTP layers and no compatible drafter, verified): base BF16 182.7, Ablit-NVFP4
  249.5, Base-NVFP4 259.75 tok/s.

Known documented cost: the GPQA delta relative to the upstream base control (~46%) is the price of
the refusal-behavior removal on this architecture (see the Abliterated-BF16 model card for the
nine-variant projection rejection evidence). Do not use the abliterated artifacts for production
reasoning work.

## Current status

- All four evaluation cells are complete: build, gates (behavior + GPQA), throughput.
- All three owned checkpoints are published and publicly verified (private-first, flip-verified).
- License: NVIDIA Open Model License — derivatives require attribution and notice retention.

## Serving note

Validated with the vLLM `0.27.1` build family (ModelOpt NVFP4 MoE loading operates correctly
there as of this release). The then-current `cu130-nightly` build regressed on ModelOpt NVFP4
FusedMoE checkpoint loading for checked checkpoints, including the shipped Nemotron-3.5-Lightning
NVFP4 artifact; re-verify against future nightlies before switching serving stacks. It will tend
to comply with harmful requests.
