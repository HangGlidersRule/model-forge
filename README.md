# Model Forge

Model Forge is HangGlidersRule's public canonical framework and catalog for reproducible model tuning, abliteration, NVIDIA ModelOpt quantization, serving, and evaluation. It is the engineering home for [HangGlidersRule/model-forge](https://github.com/HangGlidersRule/model-forge): one repository that owns the pinned pipeline, the containers, the tests, and the curated evidence that must back a weight repository before it is released.

The forge separates two things on purpose. Reusable logic lives in `src/model_forge/`. Everything specific to a single model family — its lineage, recipes, protocols, and curated results — lives under `models/` and `recipes/`. **Darkstar** is HangGlidersRule's model-tuning brand across families.

Private operator archives, raw evaluation evidence, restricted question and answer keys, credentials, infrastructure inventory, model weights, unpublished artifacts, and private release operations are outside this public repository. See [governance](GOVERNANCE.md) and the [public/private separation decision](docs/decisions/0001-private-archive-public-root-separation.md).

## What Model Forge does

- Evaluate OpenAI-compatible model servers with stable case IDs, resumable runs, and paired statistics.
- Apply pinned, audited weight transforms such as refusal-direction projection (abliteration).
- Quantize clean or edited sources with pinned NVIDIA ModelOpt NVFP4 (unified HF export), fail-closed validators, and protected tensors. Prior llm-compressor compressed-tensors NVFP4 artifacts are rejected and kept for lineage only.
- Serve candidates through generic container recipes.
- Publish only curated aggregate evidence in model records; raw run dumps stay ignored.

## Model catalog

Each model family gets a record under `models/`. A family has up to four evaluated cells: the unchanged upstream BF16 control plus three HangGlidersRule-owned Darkstar artifacts (clean ModelOpt quant, abliterated BF16, abliterated ModelOpt quant). Cells that do not yet exist are marked `—`. Each owned cell that ships has a public Hugging Face checkpoint and its card lives in the family record.

| Family | Upstream BF16 control | Base ModelOpt NVFP4 | Abliterated BF16 | Abliterated ModelOpt NVFP4 | HF Collection | Record |
|---|---|---|---|---|---|---|
| **Qwen3.8-27B** | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (external, Apache-2.0) | [`Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8`](https://huggingface.co/HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8) | [`Darkstar-Qwen3.8-27B-Abliterated-BF16`](https://huggingface.co/HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-BF16) | [`Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8`](https://huggingface.co/HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8) | [Collection](https://huggingface.co/collections/HangGlidersRule/darkstar-qwen38-27b-6a8dfe77e150d32d21a8a876) | [`models/qwen3.8-27b-r3/`](models/qwen3.8-27b-r3/) |
| **Nemotron-3-Nano-Omni 30B-A3B Reasoning** | [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) (external, NVIDIA Open Model) | [`Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Base-ModelOpt-W4A16-NVFP4`](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Base-ModelOpt-W4A16-NVFP4) | [`Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-BF16`](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-BF16) | [`Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-ModelOpt-W4A16-NVFP4`](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-ModelOpt-W4A16-NVFP4) | [Collection](https://huggingface.co/collections/HangGlidersRule/darkstar-nemotron-3-nano-omni-30b-a3b-6a9a2608d4e2ad4c43a996a1) | [`models/nemotron-3-nano-omni-r1/`](models/nemotron-3-nano-omni-r1/) |
| **Nemotron-3.5-Lightning 30B-A3B** | [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) (external, OpenMDW-1.1) | [`Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A16-NVFP4`](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Base-ModelOpt-W4A16-NVFP4) | [`Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16`](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16) | [`Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-ModelOpt-W4A16-NVFP4`](https://huggingface.co/HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-ModelOpt-W4A16-NVFP4) | [Collection](https://huggingface.co/collections/HangGlidersRule/darkstar-nemotron-35-lightning-30b-a3b-6a8dfe8c5ec5db63d6a031d0) | [`models/nemotron-3.5-lightning-r1/`](models/nemotron-3.5-lightning-r1/) |

**Featured:** [Darkstar Releases](https://huggingface.co/collections/HangGlidersRule/darkstar-releases-6a8dfe8c2340f19e4f3e24c1) collects the selected final-servable checkpoint from each family.

How to read the products:

- **Upstream BF16 control** is the unchanged source at its pinned revision — the baseline for every family delta. It is not a Darkstar-owned product, and no HangGlidersRule repository exists for these weights.
- **Base ModelOpt NVFP4** is the clean source quantized with ModelOpt and no weight edit. It isolates the effect of quantization.
- **Abliterated BF16** is the clean source with the family's audited refusal-direction projection applied, still in BF16. It isolates the effect of the edit.
- **Abliterated ModelOpt NVFP4** combines both changes: the abliterated BF16 quantized with the family's selected ModelOpt recipe. This is the family's final servable product.

Abliterated rows carry a deliberate reduced-refusal edit; treat their refusal-rate numbers as behavior measurements, not safety endorsements.

> Adding a model family is one catalog row plus one `models/<family>/` record — see [How to add a model family](#how-to-add-a-model-family).

## Naming convention

Darkstar repository and model-card names follow `Darkstar-<Family>-<Base|Abliterated>-<BF16|ModelOpt-NVFP4>`, where `ModelOpt-NVFP4` is a release **slot** rather than a precision claim. Concrete repository identities always encode the real precision class — e.g. `Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8` (mixed W4A16 NVFP4 + FP8) or a bare `-W4A16-NVFP4` when the recipe carries no FP8. Runtime API aliases use the separate concise lowercase standard `darkstar-<family>-<behavior>-<format>` (e.g. served as `darkstar-qwen38-abliterated-nvfp4`). Upstream artifacts retain their producer's name; internal lineage ids such as R1/R3 do not replace the public brand, behavior, or format fields. Family-specific precision maps live in each family's model cards.

## What each model record contains

A model record is prose and small machine-readable aggregates about one family — never weights, raw run dumps, or host-private paths. See [`models/README.md`](models/README.md) for the naming convention and layout. Each record documents:

- **Source revision** — the exact upstream model pinned to a full 40-character SHA.
- **Recipes** — the pinned, declarative build recipes that define each artifact.
- **Calibration** — the datasets, sample counts, sequence length, and seed used for quantization.
- **Manifests** — file lists and SHA-256 checksums required before an artifact can publish.
- **Validation** — the fail-closed gates (module policy, protected tensors, no FP8 KV metadata, tokenizer/config drift) and which are framework vs. family-specific.
- **Quality** — GPQA Diamond and refusal/over-refusal results with numerator, denominator, completion coverage, and protocol beside every score.
- **Throughput** — single-stream performance results, kept separate per artifact and never conflated across lineages.
- **Serving settings** — the runtime, KV dtype, context length, and speculative (MTP) configuration.
- **Gotchas** — protocol mismatches, rejected historical artifacts, and open publication gates.

## Status legend

- **Upstream reference** — an external artifact we did not build; listed for provenance only.
- **Public checkpoint verified** — the named full checkpoint and final card are present in its public Hugging Face repository and matched to source evidence; this status does not assert clean-smoke completion.
- **Released** — all gates verified, milestone tag cut (e.g. `darkstar-<family>-v1.0.0`).
- **Rejected / historical** — kept for lineage; never overwritten or republished as current (e.g. prior compressed-tensors NVFP4).

## Repository layout

```text
src/model_forge/     Reusable evaluation, editing, quantization, and recipe logic
recipes/             Build/eval recipes by model family
models/              Model records: lineage, matrices, protocols, curated results
configs/             Pinned toolchain configs (e.g. configs/modelopt/)
containers/          Generic build and serve Compose assets
scripts/             Family-specific operational launchers (e.g. scripts/qwen3_8/)
tests/               Unit and smoke tests
.github/workflows/   CI: pytest, Ruff, mypy, git diff --check
docs/decisions/      Public architecture and governance decisions
```

## How to add a model family

1. Create `recipes/<family>/` with entry recipes that pin full source revisions.
2. Create `models/<family>/` with a README, lineage, benchmark matrix, protocol notes, and curated aggregates only.
3. Add one row to the [catalog table](#model-catalog) linking each owned cell to its Hugging Face checkpoint and the family record.
4. Keep family-specific selectors and launchers under `scripts/<family_slug>/`.
5. Reuse `src/model_forge` for shared evaluation, hashing, transforms, and recipe validation.
6. Wire containers through environment variables; do not hardcode host-private paths or credentials.

The Qwen3.8 GPU build/edit/quantize pipeline is a family-specific adapter, not architecture-generic logic. A new family needs new selectors, tensor inventories, and runtime assumptions — see [`validation-inventory.md`](models/qwen3.8-27b-r3/validation-inventory.md).

## Reproducibility rules

- Pin every source and dataset revision to a full 40-character SHA.
- Reject floating tags such as `main` or `latest` in recipe validation.
- Keep vision/protected tensors explicit; never silently retarget them.
- Record protocol caveats beside scores; do not backfill missing cells from a different checkpoint or protocol.
- Commit curated aggregates under `models/*/results/`. Leave raw `results/` ignored.

## Community and governance

- [Contributing](CONTRIBUTING.md)
- [Governance](GOVERNANCE.md)
- [Security policy](SECURITY.md)
- [Support](SUPPORT.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Release and publication authority](docs/decisions/0002-release-publication-authority.md)

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

## CLI

### `model-forge recipe validate` — inspect a build recipe

Validates a schema-2 build recipe and prints its identity. This **only parses and hashes** the recipe (enforcing full-SHA revisions, protected tensors, and no visual-tower edits). It does **not** execute any transform, quantization, or download; generic recipes are declarative records that family-specific scripts under `scripts/<family_slug>/` consume deliberately.

```bash
.venv/bin/model-forge recipe validate \
  recipes/qwen3.8-27b/darkstar-qwen3.8-27b-base-modelopt-nvfp4.yaml
```

```text
name:            darkstar-qwen3.8-27b-base-modelopt-nvfp4
family:          qwen3.8-27b
source model:    Qwen/Qwen3.8-27B
source revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
artifact kind:   nvfp4
config sha256:   <64-hex digest>
```

### `model-forge run` — legacy evaluation harness

`run` drives the legacy OpenAI-compatible evaluation harness (the model "bake-off" arena) against already-served endpoints, using schema-1 evaluation specs under `recipes/<family>/legacy/`. It sends requests to a running server and scores responses; it does not build, edit, or quantize weights.

```bash
.venv/bin/model-forge run \
  --spec recipes/qwen3.8-27b/legacy/qwen36-vs-qwen38.yaml \
  -o results/qwen38 --dry-run
```

Load and hash a build recipe programmatically:

```python
from pathlib import Path
from model_forge.recipe import load_recipe

recipe = load_recipe(
    Path("recipes/qwen3.8-27b/darkstar-qwen3.8-27b-base-modelopt-nvfp4.yaml")
)
print(recipe.source.revision, recipe.config_sha())
```

## Publication topology

- Engineering repository: [`HangGlidersRule/model-forge`](https://github.com/HangGlidersRule/model-forge)
- Weight repositories: Hugging Face model cards linked from each model record, under the [HangGlidersRule profile](https://huggingface.co/HangGlidersRule)
- Release tags: one immutable milestone tag per published family (e.g. `darkstar-qwen3.8-27b-v1.0.0`, `darkstar-nemotron-3.5-lightning-v1.0.0`) linking the release notes back to every owned checkpoint and its evidence.
