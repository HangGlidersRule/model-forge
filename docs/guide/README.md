# The Model Forge guide

Model Forge turns open-weight models into **Darkstar** checkpoints: NVIDIA ModelOpt NVFP4 quantizations, refusal-reduced
("abliterated") derivatives, and a few task-specific builds. Every checkpoint ships with pinned inputs, its build
recipe and evaluation evidence. This guide collects how that work is done, in one place.

The release contract ([`../darkstar-four-product-release-process.md`](../darkstar-four-product-release-process.md))
and its machine-readable schema stay normative: where this guide summarizes a rule, the contract wins on any
disagreement. Every number quoted here comes from a committed results file or model card; follow the links for the
evidence.

## Who this is for

- **Users of Darkstar checkpoints**, who want to know what a name means, what was measured, and how to serve one.
- **Contributors**, who want to add a model family or reproduce a build.
- **Researchers**, who want the exact abliteration method, gates and failure history.

## Contents

| Guide | What it covers |
|---|---|
| [Abliteration](abliteration.md) | Refusal-direction removal end to end: measuring the direction, choosing the layer, editing weights, the behavior and intelligence gates, quantizing the edited model, and what to do when projection fails (the LoRA unlearning fallback). |
| [From upstream checkpoint to release](release-process.md) | The whole pipeline for one family: recon, pre-registration, calibration, ModelOpt builds, candidate selection, artifact validation, cards, Hugging Face publication, the attested public export, review, tags and the family audit. |
| [Evaluation](evaluation.md) | How quality and speed are measured: GPQA Diamond, next-token KL, task gates (retrieval, decision heads, reranking), throughput and latency probes, noise floors, and the rules that keep numbers honest. |
| [Lessons and pitfalls](lessons.md) | Failures we hit and the rule each one produced: things that cost a day once so they don't cost a day again. |

## The four products

Every family has four canonical product roles. The repository id always spells out the real precision.

| Role | What it is | Example id |
|---|---|---|
| Base BF16 | The unchanged upstream checkpoint. Measured as the control, never republished. | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (upstream) |
| Base ModelOpt NVFP4 | The upstream weights quantized with NVIDIA ModelOpt. | `Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A16-NVFP4` |
| Abliterated BF16 | The upstream weights with the refusal direction removed. | `Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16` |
| Abliterated ModelOpt NVFP4 | The abliterated BF16 quantized with the same winning recipe. | `Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-ModelOpt-W4A16-NVFP4` |

Families whose model has no refusal behavior (embedders, rerankers, scorers) ship only the base roles; the abliterated
roles are recorded as not applicable. A family may add further base cells later (a W4A4 build next to a W4A16 one);
each gets its own precision-encoded id and a minor release tag.

## Reading a Darkstar name

`Darkstar-<Family>-<Base|Abliterated>-ModelOpt-<W4A16|W4A4>-NVFP4[-Mixed-FP8]`

- **`Base` / `Abliterated`:** unedited, or refusal direction removed.
- **`W4A16`:** 4-bit NVFP4 weights, 16-bit activations. Weight-only; vLLM serves these with Marlin or similar kernels.
- **`W4A4`:** 4-bit weights and 4-bit activations. Real FP4 math on Blackwell tensor cores; needs calibrated activation
  scales.
- **`Mixed-FP8`:** some served weight paths are FP8 (attention, linear attention, Mamba projections). Derived from the
  precision map, never declared by hand. KV-cache precision does not count.
- A variant token goes before `ModelOpt` (`...-Base-MLP-ModelOpt-W4A4-NVFP4`), never after the precision.

## Repository map

| Path | Holds |
|---|---|
| `models/<family>/` | One family's record: README, model cards, results JSON, eval and calibration scripts, ledger. |
| `recipes/<family>/` | Schema-2 build recipes: source, transforms (abliteration), quantization, runtime, publication target. |
| `configs/modelopt/recipes/` | ModelOpt PTQ recipes (the quantizer configuration ModelOpt itself loads). |
| `contracts/darkstar-release/` | The release contract and ledger schema (CI-enforced). |
| `docs/` | Process documents; this guide is `docs/guide/`. |
| `publication/huggingface/<repo>/README.md` | The exact card uploaded to each Hugging Face repository. |
| `scripts/` | Shared tooling: GPQA runner, export staging and verification, family release audit. |
| `src/model_forge/` | The library: recipe loader, abliteration math, release-contract validators, public exporter. |
