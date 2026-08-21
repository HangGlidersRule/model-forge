# Darkstar release contract

Versioned, machine-readable contract for the immutable **Darkstar four-product release process**.
It is the schema/checklist backing the normative process document
[`docs/darkstar-four-product-release-process.md`](../../docs/darkstar-four-product-release-process.md).

## Files

- [`v1/contract.json`](v1/contract.json) — canonical product roles, the frozen gate set, valid
  statuses, and the publication rule (version `1.0.0`).
- [`v1/ledger.schema.json`](v1/ledger.schema.json) — JSON Schema (draft 2020-12) that every
  per-family publication-readiness ledger must validate against.

## How it is used

Each model family records a **publication-readiness ledger** — a JSON document that validates against
`ledger.schema.json` and is the single source of truth for gate status. The human-readable
`verified` / `in_progress` / `missing` / `rejected_historical` / `not_applicable` statuses are
rendered directly into the Git-tracked benchmark matrix and the per-product model cards inside
`LEDGER-SYNC` comment blocks. There is **no separate gap report**: if the ledger and a
rendered block disagree, CI fails.

`ModelOpt-NVFP4` roles additionally track precision-encoded candidates: the family slot id never hides
the activation/recipe precision class. Each candidate id encodes `W4A16` vs `W4A4` (plus `Mixed-FP8`
where applicable) and carries a complete precision map, rendered via `CANDIDATE-SYNC` blocks. CI fails
on precision-map omission or conflation (for example naming a mixed W4A16+FP8 artifact as uniform
W4A4). Verifying a candidate's GPQA measurement is distinct from promoting it to a release winner.

Publication identity and runtime API identity are separate. Repositories and model-card headings stay
precision-encoded; `--served-model-name` uses the stable lowercase alias
`darkstar-<family>-<behavior>-<format>`. The ledger records `served_model_alias` and
`runtime_container_name` so runtime snapshots and Compose profiles can be tied to the exact product.

Loaders, validators, and the publication rule are implemented in
[`src/model_forge/release.py`](../../src/model_forge/release.py) and enforced by
[`tests/test_release_contract.py`](../../tests/test_release_contract.py).

Current ledger: [`models/qwen3.8-27b-r3/results/publication-readiness-ledger.json`](../../models/qwen3.8-27b-r3/results/publication-readiness-ledger.json).

## Versioning

The contract and schema are versioned by directory (`v1/`). Gates may be **added** or **strengthened**
in a new version; they may not be removed or weakened. CI freezes the required gate set and the
publication rule to prevent silent weakening.
