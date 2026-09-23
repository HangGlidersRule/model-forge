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

## Image and diffusion families

Image-generation families (diffusion transformers, e.g. Qwen-Image-2.1) follow the
same checklist with these substitutions:

- **Architecture record.** `recon.json` additionally records the composite anatomy
  (transformer/DiT class and layer count, text-encoder class, VAE channel/latent
  layout), the companion checkpoints (e.g. prompt rewriters), and any watermarking
  behavior of outputs.
- **Precision map.** Diffusion candidates use the DiT component set
  (`dit_blocks`, `protected_blocks`, `attention`, `text_encoder`, `vae`) instead of
  the language-model set, and declare `architecture: "diffusion"`. The recipe
  vocabulary is `NVFP4`, `FP8`, or `Mixed-FP8` (an NVFP4+FP8 mix). Note that a
  uniform-fp4 recipe is a W4A4-style fp4 GEMM; shipped product ids may therefore
  carry a `W4A4-NVFP4` precision encoding (as in
  `Darkstar-Qwen-Image-2.1-Base-ModelOpt-W4A4-NVFP4`), while the recipe id itself
  spells only the recipe class (`NVFP4`). Product ids and recipe ids are two
  different encodings; the release contract validates both against the precision
  map.
- **Behavior cells.** Image-family abliterated cells are gated on demonstrated
  behavior mechanism, not assumed by default. A behavior cell ships only when the
  mechanism is validated against the frozen behavior suites (benign compliance
  preserved, refusal-adjacent compliance raised) with the full-denominator judge.
  If instrumented attempts show the family's censorship mechanism is not
  editable-grade reachable (as for Qwen-Image-2.1, where seven instrumented
  attempts — encoder-direction removal, subspace projections, and two
  LoRA-pair training regimes — either had no effect or collapsed benign
  compliance), the family ships base cells only and the evidence is recorded as
  the justification. The four-cell matrix remains the default template for
  future families whose censoring is encoder-mediated and edit-transferable.
- **Recipe.** Same naming convention:
  `darkstar-<family>-<behavior>-<precision>.yaml` under `recipes/<family>/`, where
  `<family>` is alnum-only in recipe identifiers (for example `qwenimage21`).
- **Serving profile.** Image families use the diffusion serving profile and the
  image smoke contract: `${PUBLIC_WORKSPACE}` and `${PUBLIC_WORKSPACE}` reachability,
  seed determinism under the frozen batching configuration, and an RGBA round-trip
  assertion where the family supports transparency. The text smokes (`/v1/models`,
  context length, text/JSON/tool checks) do not apply to image servers.
- **Quality screen.** Image families run the frozen image case manifest (T2I
  quality, composition, editing, typography, RGBA) with matched full-denominator
  pass rates; text benchmarks such as GPQA are not applicable.
- **Fleet env pins.** Where a family's serving stack requires host-specific
  environment pins (for example fp4 GEMM backend selection on non-SM100 hardware),
  the pins are part of the serve profile and are validated by the smoke contract.

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
