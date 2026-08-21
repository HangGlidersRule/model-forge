# Release policy

This document defines what "published" means for Model Forge artifacts and the
minimum evidence required before a product may be announced as publicly
released.

## Definitions

- **Published**: weights are uploaded to a public Hugging Face repository under
  `HangGlidersRule`, the repository revision is immutable and manifest-verified,
  a clean redownload plus boot/smoke passes were recorded, the corresponding
  `model-forge` ledger/card/catalog entries agree, and an immutable Git tag
  (`darkstar-*` or `v*`) names the release.
- **Locally complete, unpublished**: all build and local gates pass, but no
  public upload, clean redownload, or immutable tag exists yet. This is the
  normal resting state for new artifacts.
- **Withdrawn/deprecated**: a previously published revision has a blocking
  defect. Its ledger status is updated, public cards say deprecated, and no new
  consumers should be pointed at it.

## Publication requirements

1. **Exact source identity.** Every publication bundle references the exact
   source commit, recipe path and hash, tool/image versions, and artifact
   manifest with per-file hashes.
2. **Protected release environment.** Publication runs only through the
   protected release environment with maintainer approval. Contributor-facing
   pull requests never receive credentials or write tokens.
3. **Clean redownload verification.** After upload, verify from an empty cache:
   download by immutable revision, check every file hash, boot with the
   checked-in serving profile, run the text/JSON/tool/vision smoke gates, and
   scan logs for fatal/scale/NaN markers.
4. **Atomic ledger transition.** Card, target URL/revision, ledger lifecycle,
   catalog status, benchmark matrix, and release tag must be updated together
   in one publication PR. Tests fail if they disagree.
5. **No raw evaluation material.** Aggregate results only. Raw prompts, answer
   keys, per-question responses, and private infrastructure metadata never cross
   the public boundary.

## Quality gates

- Matched full-denominator benchmarks with a frozen protocol (numerator,
  denominator, coverage, timeout policy all recorded).
- Behavior gates for abliterated/uncensored variants; refusal metrics reported
  with their protocol.
- Zero fatal/scale/NaN markers in runtime logs.
- Gitleaks and private-metadata scans clean across public history.

## Tag and rollback discipline

- Immutable tags are never re-pointed. A corrected revision is a new tag.
- A bad published revision is marked withdrawn/deprecated in the ledger and
  cards, never silently overwritten.
- Rollback procedures are documented in the governance plan
  (`docs/decisions/0001-private-archive-public-root-separation.md` and the
  release ledger schema `contracts/darkstar-release/v1/ledger.schema.json`).

## Supported families

Model families are onboarded through
`docs/governance/model-family-onboarding.md`. A family enters the catalog only
after its evidence contract, license resolution, and serving profile are
recorded.

## Security disclosures

Report vulnerabilities through
[GitHub private vulnerability reporting](https://github.com/HangGlidersRule/model-forge/security/advisories/new).
Never file security issues as public issues; never include credentials or raw
private infrastructure data in any report.
