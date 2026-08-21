# Model Forge roadmap

Model Forge is the public canonical source for a reusable model-forge framework,
the Darkstar product catalog, release contracts, reproducible recipes, generic
containers, sanitized aggregate evidence, and publication metadata. This roadmap
records committed direction. Items are not promises of delivery dates; they are
the current public contract for where the project is going.

## Current (published)

- Four-product Darkstar Qwen3.8-27B family released and verified through the
  protected publication pipeline:
  - `Darkstar-Qwen3.8-27B-Base-BF16` (reference)
  - `Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8`
  - `Darkstar-Qwen3.8-27B-Abliterated-BF16`
  - `Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8`
- Public model-forge framework: exporter, governance, deterministic CI, PR risk
  classification, immutable release tag `darkstar-qwen3.8-27b-v1.0.0`.
- Private operator archive keeps raw evidence, host inventory, credentials, and
  unpublished artifacts out of the public repository.

## Short term

- Rotate scheduled integrity checks: weekly full-history Gitleaks, public
  metadata hygiene, link checker, and release-ledger consistency.
- Publish remaining model families under the same contract: Nemotron-3.5
  Lightning and Nemotron-3 Nano-Omni (four-product matrix per family), gated on
  license compliance and benchmark evidence.
- Extend DFlash2 spec-decode support across the catalog where drafters exist.
- Formalize model-family onboarding in
  `docs/governance/model-family-onboarding.md` and release policy in
  `docs/governance/release-policy.md`.

## Medium term

- GHCR container publication for generic serve/build containers.
- Contributor-facing automation improvements: dependency review, container
  vulnerability scanning, stale-publication-target verification.
- Benchmark matrix refresh with a frozen protocol across all published products.

## Not planned

- Public hosting of model weights in this repository. Weights live on Hugging
  Face; this repository stores manifests, hashes, and cards.
- Public hosting of raw evaluation dumps, prompts, or answer keys.
- Bidirectional manual sync between the public repository and the private
  operator archive.

## How to change this roadmap

Roadmap changes go through pull requests under normal governance
(`GOVERNANCE.md`, `CONTRIBUTING.md`). Maintainers merge only after deterministic
checks pass and no P0/P1 findings remain.
