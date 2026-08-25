# Model Forge

Model Forge is HangGlidersRule's public canonical framework and catalog for reproducible model tuning, abliteration, NVIDIA ModelOpt quantization, serving, and evaluation. It is the engineering home for [HangGlidersRule/model-forge](https://github.com/HangGlidersRule/model-forge): one repository that owns the pinned pipeline, the containers, the tests, and the curated evidence that must back a weight repository before it is released.

The forge separates two things on purpose. Reusable logic lives in `src/model_forge/`. Everything specific to a single model family — its lineage, recipes, protocols, and curated results — lives under `models/` and `recipes/`. **Darkstar** is HangGlidersRule's model-tuning brand across families. The first family recorded end to end is Darkstar Qwen3.8-27B; **R3** is the internal lineage id for its abliterated edit only.

Private operator archives, raw evaluation evidence, restricted questions and answer
keys, credentials, infrastructure inventory, model weights, unpublished artifacts,
and private release operations are outside this public repository. See
[governance](GOVERNANCE.md) and the
[public/private separation decision](docs/decisions/0001-private-archive-public-root-separation.md).

## What Model Forge does

- Evaluate OpenAI-compatible model servers with stable case IDs, resumable runs, and paired statistics.
- Apply pinned, audited weight transforms such as refusal-direction projection (abliteration).
- Quantize clean or edited sources with pinned NVIDIA ModelOpt NVFP4 (unified HF export), fail-closed validators, and protected tensors. Prior llm-compressor compressed-tensors NVFP4 artifacts are rejected and kept for lineage only.
- Serve candidates through generic container recipes.
- Publish only curated aggregate evidence in model records; raw run dumps stay ignored.

## Model catalog

Each model family gets a record under `models/`. This record has four evaluated cells: the unchanged
upstream BF16 control and three HangGlidersRule-owned Darkstar artifacts. Only the three owned
artifacts have HangGlidersRule Hugging Face weight repositories.

Exactly three public HangGlidersRule repositories contain complete, hash-verified checkpoints and
final cards. Clean download, boot, and smoke are verified, and GHCR is not required. The planned
immutable tag is `darkstar-qwen3.8-27b-v1.0.0`; until it is cut, the contract publication claim
remains false. The
> upstream [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) repository is external,
> Apache-2.0, and used unchanged as the BF16 control.

### Qwen3.8-27B

Record: [`models/qwen3.8-27b-r3/`](models/qwen3.8-27b-r3/). Shared source revision:
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The four evaluated cells comprise
one upstream control and three owned Darkstar artifacts.

| Lineage | Edit | Format | Target repository | Local card | Status |
|---|---|---|---|---|---|
| Upstream BF16 control | Clean | BF16 | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (external) | [`base-bf16.md`](models/qwen3.8-27b-r3/model-card/base-bf16.md) | Upstream weights; evaluated at pinned revision |
| Darkstar Base | Clean | ModelOpt NVFP4 (mixed W4A16 + FP8) | [`HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8`](https://huggingface.co/HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8) | [`base-nvfp4.md`](models/qwen3.8-27b-r3/model-card/base-nvfp4.md) | Public pinned checkpoint; clean download/boot/smoke verified; release tag pending |
| Darkstar Abliterated (R3) | Abliterated | BF16 | [`HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-BF16`](https://huggingface.co/HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-BF16) | [`bf16.md`](models/qwen3.8-27b-r3/model-card/bf16.md) | Public pinned checkpoint; clean download/boot/smoke verified; release tag pending |
| Darkstar Abliterated (R3) | Abliterated | ModelOpt NVFP4 (mixed W4A16 + FP8) | [`HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8`](https://huggingface.co/HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8) | [`nvfp4.md`](models/qwen3.8-27b-r3/model-card/nvfp4.md) | Public pinned checkpoint; clean download/boot/smoke verified; release tag pending |

How to read the products:

- **Upstream BF16 control** is unchanged `Qwen/Qwen3.8-27B` at the pinned revision. It is the
  baseline for every family delta (157/198 = 79.29% GPQA, 130.158 tok/s at MTP8). It is not a
  Darkstar-owned product, and no HangGlidersRule repository exists or is planned for these weights.
- **Darkstar Base ModelOpt NVFP4** is that clean source quantized with ModelOpt and no weight edit. It is the control that isolates the effect of quantization. The **selected** candidate is mixed precision, so its repository id says so: `Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8` (203.636 tok/s at MTP4, 153/198 GPQA). The uniform W4A4 candidate was built and rejected on throughput.
- **Darkstar Abliterated BF16** is the clean source with the R3 audited refusal-direction projection applied, still in BF16 (146/198 GPQA, 144.502 tok/s at MTP11). It isolates the effect of the edit and exists as a complete local build.
- **Darkstar Abliterated ModelOpt NVFP4** combines both changes: the abliterated BF16 quantized with the selected mixed W4A16-NVFP4+FP8 recipe. The selected checkpoint scored `148/198 = 74.75%` GPQA and froze MTP10 after nonmonotonic confirmation; every build gate is verified.

The abliterated rows carry a deliberate reduced-refusal edit; treat their refusal-rate numbers as behavior measurements, not safety endorsements.

#### Naming convention

Darkstar repository and model-card names follow `Darkstar-<Family>-<Base|Abliterated>-<BF16|ModelOpt-NVFP4>`, where the ModelOpt-NVFP4 form is a release **slot** rather than an artifact name. Actual repository identities and recipe publication targets never omit precision wording: an NVFP4 id spells out `ModelOpt-<W4A16|W4A4>-NVFP4` plus `-Mixed-FP8` when FP8 is part of the recipe. Runtime API aliases use the separate concise lowercase standard `darkstar-<family>-<behavior>-<format>`; for example, Product 4 is served as `darkstar-qwen38-abliterated-nvfp4` while its repository remains `Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8`. BF16 repository ids need no activation qualifier. Upstream artifacts retain their producer's name, and internal lineage ids such as R3 do not replace the public brand, behavior, or format fields.

#### Evaluation and publication

The existing four-cell evaluation contract supplies frozen, matched evidence across the upstream BF16
control, clean quantized artifact, abliterated BF16 artifact, and abliterated quantized artifact. The
control is tracked in [`base-bf16.md`](models/qwen3.8-27b-r3/model-card/base-bf16.md) as an evaluation
profile only. The other three cells have public Hugging Face repositories containing their named
checkpoints.

`ModelOpt-NVFP4` is a release **slot**, not a precision claim, and only the contract, the process
document, and the ledger's `product_id` may use it. Concrete candidates and every actual repository
or model-card identity encode their real precision class — e.g.
`Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8` (mixed W4A16 NVFP4 + FP8) versus
`Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4` (uniform W4A4) — and are never conflated. The Base
ModelOpt target is **resolved** to the selected mixed candidate, whose full-denominator GPQA
(`153/198 = 77.27%`) is within 2.02 pp of the Base BF16 baseline; it was **selected on throughput**
(203.636 tok/s at MTP4) over the uniform W4A4 candidate (built and rejected at 129.441 tok/s).

#### Family references

- Release process and gates: [`docs/darkstar-four-product-release-process.md`](docs/darkstar-four-product-release-process.md); contract: [`contracts/darkstar-release/v1/contract.json`](contracts/darkstar-release/v1/contract.json); ledger: [`results/publication-readiness-ledger.json`](models/qwen3.8-27b-r3/results/publication-readiness-ledger.json)
- Artifact lineage (source revision → edit → quantization): [`artifact-lineage.md`](models/qwen3.8-27b-r3/artifact-lineage.md)
- Benchmark matrix and caveats: [`benchmark-matrix.md`](models/qwen3.8-27b-r3/benchmark-matrix.md)
- GPQA Diamond protocol: [`gpqa-protocol.md`](models/qwen3.8-27b-r3/gpqa-protocol.md); machine-readable aggregates: [`results/gpqa-matrix.json`](models/qwen3.8-27b-r3/results/gpqa-matrix.json)
- Performance results (clean base NVFP4 throughput): the performance sections of [`benchmark-matrix.md`](models/qwen3.8-27b-r3/benchmark-matrix.md)
- Recipes: [`recipes/qwen3.8-27b/`](recipes/qwen3.8-27b/) — [`darkstar-qwen3.8-27b-base-modelopt-nvfp4.yaml`](recipes/qwen3.8-27b/darkstar-qwen3.8-27b-base-modelopt-nvfp4.yaml), [`darkstar-qwen3.8-27b-abliterated-bf16.yaml`](recipes/qwen3.8-27b/darkstar-qwen3.8-27b-abliterated-bf16.yaml), [`darkstar-qwen3.8-27b-abliterated-modelopt-nvfp4.yaml`](recipes/qwen3.8-27b/darkstar-qwen3.8-27b-abliterated-modelopt-nvfp4.yaml)
- ModelOpt notes and pinned toolchain: [`modelopt/README.md`](models/qwen3.8-27b-r3/modelopt/README.md), pin in [`configs/modelopt/pin.json`](configs/modelopt/pin.json)
- Validation and code inventory (reusable framework vs. Qwen adapter): [`validation-inventory.md`](models/qwen3.8-27b-r3/validation-inventory.md)
- Publication plan and release gates: [`publication-plan.md`](models/qwen3.8-27b-r3/publication-plan.md)
- Issues and discussion: [GitHub issues](https://github.com/HangGlidersRule/model-forge/issues)

#### Measured quality

All four evaluated cells carry frozen full-denominator (198/198) GPQA results with zero timeouts,
parse errors, or errors: upstream BF16 control `157/198 = 79.29%`, Base ModelOpt mixed
W4A16-NVFP4+FP8 `153/198 = 77.27%`, Abliterated BF16 `146/198 = 73.74%`, and Abliterated ModelOpt
mixed W4A16-NVFP4+FP8 `148/198 = 74.75%`. A secondary thinking-enabled run scored
`164/198`, but that number comes from the rejected historical R3 compressed-tensors NVFP4 artifact —
not this ModelOpt build — and is neither matched-matrix eligible nor attributable to quantization
| Full caveats live in the [benchmark matrix](models/qwen3.8-27b-r3/benchmark-matrix.md).

### Nemotron-3.5-Lightning 30B-A3B

Record: [`models/nemotron-3.5-lightning-r1/`](models/nemotron-3.5-lightning-r1/). Shared source
revision: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16@d468880b6ad3c6e0d21377ce7242adaea4cc884d`
(OpenMDW-1.1). Hybrid Mamba2 + MoE + sparse attention, 52 layers, 262,144-token context.
Abliteration internal lineage id: **R1** (canonical layer-34, chat-templated, 320/320, 3,126
residual-writing targets).

| Lineage | Edit | Format | Target repository | Local card | Status |
|---|---|---|---|---|---|
| Upstream BF16 control | Clean | BF16 | [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) (external) | — | Upstream weights; evaluated at pinned revision |
| Darkstar Base | Clean | ModelOpt NVFP4 (W4A16 experts, BF16 protected) | (sibling cell; not republished) | — | Evaluated |
| Darkstar Abliterated (R1) | Abliterated | BF16 | `HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16` (planned) | [`abliterated-bf16.md`](models/nemotron-3.5-lightning-r1/model-card/abliterated-bf16.md) | Staged — gates passing |
| Darkstar Abliterated (R1) | Abliterated | ModelOpt NVFP4 (W4A16 experts, BF16 protected) | `HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-ModelOpt-W4A16-NVFP4` (planned) | [`abliterated-nvfp4.md`](models/nemotron-3.5-lightning-r1/model-card/abliterated-nvfp4.md) | Staged — gates passing |

Docs: [`benchmark-matrix.md`](models/nemotron-3.5-lightning-r1/benchmark-matrix.md),
[`gpqa-protocol.md`](models/nemotron-3.5-lightning-r1/gpqa-protocol.md),
[`artifact-lineage.md`](models/nemotron-3.5-lightning-r1/artifact-lineage.md),
[`publication-plan.md`](models/nemotron-3.5-lightning-r1/publication-plan.md),
results: [`gpqa-matrix.json`](models/nemotron-3.5-lightning-r1/results/gpqa-matrix.json),
[`publication-readiness-ledger.json`](models/nemotron-3.5-lightning-r1/results/publication-readiness-ledger.json).
Recipes: [`recipes/nemotron-3.5-lightning/`](recipes/nemotron-3.5-lightning/).

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
- **Public checkpoint verified** — the named full checkpoint and final card are present in its public
  Hugging Face repository and matched to source evidence; this status does not assert clean-smoke
  completion or public release.
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
2. Create `models/<artifact-id>/` with a README, lineage, benchmark matrix, protocol notes, and curated aggregates only.
3. Keep family-specific selectors and launchers under `scripts/<family_slug>/`.
4. Reuse `src/model_forge` for shared evaluation, hashing, transforms, and recipe validation.
5. Wire containers through environment variables; do not hardcode host-private paths or credentials.

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

See the [publication plan](models/qwen3.8-27b-r3/publication-plan.md) for release gates.
