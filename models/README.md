# Model records

Each subdirectory is a **model record**: the curated, publishable evidence for one derivative
artifact family. Model records hold documentation and small machine-readable aggregates only — never
weights, raw run dumps, or host-private paths.

## Naming convention

- Darkstar is HangGlidersRule's tuning brand across model families.
- Public repository and model-card identities follow
  `Darkstar-<Family>-<Base|Abliterated>-<BF16|ModelOpt-NVFP4>`, where `ModelOpt-NVFP4` is a release
  slot label. Actual publication ids never omit precision wording: a repository, model-card heading,
  or recipe publication target that mentions NVFP4 spells out `ModelOpt-<W4A16|W4A4>-NVFP4` plus
  `-Mixed-FP8` when FP8 is part of the recipe. When no precision winner exists, the target is
  explicitly unresolved and only candidate ids are reserved.
- Runtime API aliases are intentionally concise and lowercase:
  `darkstar-<family>-<behavior>-<format>`. They identify the served API model without replacing the
  long precision-encoded repository identity. Use stable family slugs (for example `qwen38`), behavior
  `base` or `abliterated`, and the externally meaningful format (`bf16`, `nvfp4`, and so on).
- Directory names may retain an internal lineage id, e.g. `qwen3.8-27b-r3`; R3 identifies the
  abliterated edit and is not a substitute for the public brand/behavior/format name.
- External upstream artifacts keep their producer's name and are not branded Darkstar.

## Extension boundary

A model record is **data and prose about one family**. Reusable logic never lives here — it lives in
`src/model_forge/`. Family-specific operational launchers live in `scripts/<family_slug>/`, and build
recipes live in `recipes/<family>/`. If you find yourself adding executable Python under `models/`,
it belongs in `src/model_forge/` (if generic) or `scripts/` (if family-specific) instead.

## Expected contents

```text
models/<artifact-id>/
  README.md              Index and current status
  artifact-lineage.md    Source revision -> edits -> quantization build record
  benchmark-matrix.md    Measured results with explicit missing cells and caveats
  <protocol>.md          Evaluation protocol notes (e.g. gpqa-protocol.md)
  publication-plan.md    Release gates for GitHub/GHCR/Hugging Face
  model-card/            Evaluation profiles and Hugging Face model-card sources
  plans/                 Historical implementation plans (paths updated to current layout)
  results/               Curated machine-readable aggregates only (committed exception)
  data/                  Corpus/manifest documentation for family-specific inputs
```

## Release contract and ledger

Each family that follows the [four-product release process](../docs/darkstar-four-product-release-process.md)
commits a machine-readable **publication-readiness ledger** under `results/` (e.g.
`publication-readiness-ledger.json`) that validates against the
[release contract](../contracts/darkstar-release/v1/ledger.schema.json). The ledger is the source of
truth; its `verified`/`in_progress`/`missing`/`rejected_historical`/`not_applicable` statuses are
rendered into `benchmark-matrix.md` and the model cards via `LEDGER-SYNC` blocks. Do not create a
separate gap report.

ModelOpt-NVFP4 candidates additionally encode their real precision class (W4A16 vs W4A4, plus mixed
FP8 where present) in both the candidate id and a complete precision map, rendered via `CANDIDATE-SYNC`
blocks. A mixed W4A16+FP8 artifact is never conflated with a uniform W4A4 one. Each candidate reserves
`<namespace>/<candidate_id>`, and a product's `target_repository` is either one of those built
candidate ids or `null` with status `unresolved_pending_precision_winner`.

## Rules

- Pin every source and dataset revision to a full 40-character SHA.
- Record protocol caveats beside scores; never backfill a missing cell from a different checkpoint or
  protocol.
- Only `models/*/results/**` is exempt from the ignored top-level `results/`; keep it to curated
  aggregates.
- Do not commit restricted benchmark question text or answer keys.

## Current records

- [`qwen3.8-27b-r3/`](qwen3.8-27b-r3/) — four evaluated cells: unchanged upstream
  `Qwen/Qwen3.8-27B` BF16 as the external control, plus three HangGlidersRule-owned artifacts (clean
  ModelOpt NVFP4, abliterated BF16, and abliterated ModelOpt NVFP4). Only those three owned artifacts
  have private HangGlidersRule Hugging Face repositories, each containing its complete hash-verified
  checkpoint and byte-verified final card. Clean re-download/boot smoke and public release remain open.
