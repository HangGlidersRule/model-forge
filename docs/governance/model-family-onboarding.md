# Model family onboarding

This is the contract for adding a new model family to the Model Forge catalog
under the Darkstar brand, or to the reusable framework itself.

## Before you start

A family is a coherent base architecture with a shared evidence contract (for
example, `Qwen3.8-27B`, `Nemotron-3.5-Lightning`, `Nemotron-3-Nano-Omni`). Each
family gets:

- a documented source model and immutable source revision,
- a license/provenance decision for redistribution of modified weights,
- a reproducible build recipe (BF16 reference plus the quant recipes produced),
- a serving profile that boots from the checked-in recipe,
- a benchmark and behavior evidence protocol,
- a public card and manifest in the catalog.

Do not start publication work against a family until the license and evidence
contract are resolved. Record the decision in the public lineage.

## Onboarding checklist

1. **Source identity.** Record `source.model_id`, `source.revision`, and
   architecture facts in the family directory
   (`models/<family>/recon.json` or equivalent).
2. **License resolution.** Confirm the source license permits modified-weight
   redistribution, and retain the required license/notice text in every public
   artifact and card. If the license does not permit public modified weights,
   publish manifests/recipes only and mark weights as referenced, not owned.
3. **Recipe.** Add exact recipes under `recipes/<family>/` with no private
   paths, hostnames, or credentials. Recipe names must encode the behavior
   variant and precision (for example,
   `darkstar-qwen3.8-27b-abliterated-modelopt-w4a16-nvfp4.yaml`).
4. **Artifact validation.** Build and validate with the pinned toolchain
   (ModelOpt version, image digest). Record `_SUCCESS.json`/manifest hashes,
   tensor counts, precision maps, and KV-cache dtype.
5. **Serving profile.** Use `ServeProfile` to render a deterministic compose
   file. Boot it, verify `/v1/models`, context length, text/JSON/tool/vision
   smokes, and scan logs for fatal markers.
6. **Performance evidence.** Run the family's throughput sweep under a frozen
   protocol (single-stream and, where relevant, concurrency). Record
   prompt-length decomposition and fatal counts.
7. **Behavior gates.** For abliterated/uncensored variants, run the behavior
   protocol (harmful-compliance and over-refusal) and record the exact protocol.
8. **Quality screen.** Run the family's matched full-denominator evaluation
   (for example, GPQA Diamond `198/198`) with a frozen harness; report
   numerator, denominator, coverage, and policy. No raw question/answer
   material in public files.
9. **Manifest and ledger.** Register the product in the release ledger and the
   public file manifest (`tools/public_export/public-files.yaml`) so the
   exporter keeps private material out.
10. **Publication.** Follow `docs/governance/release-policy.md`: protected
    release environment, clean redownload, atomic ledger transition, immutable
    tag.
11. **Card and catalog.** Render the public model card and catalog entry from
    validated evidence. No placeholders, no "staged/pending" wording once
    weights are live.

## Supported vs. experimental

- **Supported** means the family has passed the full onboarding checklist and
  is listed in the catalog with a release contract.
- **Experimental** means work exists but evidence or license gates are
  incomplete; it is recorded as likely incomplete/unpublished and not promoted.

## Rejecting a family

A family may be rejected in `recon`/catalog records when: the license blocks
redistribution, no drafter or eval protocol exists for its architecture, or
quality gates fail materially. Rejection is recorded in the lineage docs and the
family is not published.

## Removing or deprecating a family

Follow `docs/governance/release-policy.md` tag/rollback discipline: mark
withdrawn/deprecated, never silently overwrite, update ledger/card/catalog
together, and do not reuse a compromised tag.
