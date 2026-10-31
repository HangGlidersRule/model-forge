# From upstream checkpoint to release

This is the whole pipeline for one family, in the order it runs. Each step ends in a check that reads the result back;
a command succeeding is never the check. The release contract
([`../darkstar-four-product-release-process.md`](../darkstar-four-product-release-process.md)) defines the gates; this
page explains how they are met in practice.

```text
recon -> pre-register -> calibration set -> smoke build -> full build(s) -> validate -> evaluate & select
      -> card + record -> private upload -> clean-download smoke -> public flip -> attested export PR
      -> review & merge -> tag & release -> family audit
```

## 1. Recon

Pin the upstream checkpoint at a full revision and record what it is before touching it (`models/<family>/recon.json`):

- architecture and module tree: dense or MoE, attention / linear-attention / Mamba layout, MTP head, vision or audio
  towers, tied embeddings;
- every safetensors shard with its sha256, verified on download;
- license and redistribution terms (OpenMDW-1.1 and Apache-2.0 so far);
- the upstream's own published numbers and serving recipe, for later comparison;
- what NVIDIA (or the vendor) already ships for it: their quantized checkpoints and ModelOpt recipes are the reference
  layout. ModelOpt carries model-specific recipes under `modelopt_recipes/models/<org>/<model>/`.

## 2. Pre-register

Before any candidate result exists, post the plan on the family's GitHub issue:

- the candidates (precision layouts and calibration methods) and their precision-encoded ids;
- the evaluation sets and their hashes, the metrics, and the **pass bar** for each gate;
- the selection rule when several candidates pass (for example "the faster passing candidate ships");
- what happens on failure.

Anything decided after seeing results is posted as a numbered **amendment** (R17-A1, R18-A1) before the results it
governs. A correction to an earlier write-up is posted as its own comment, never edited in silently. Published examples:
issues [#17](https://github.com/HangGlidersRule/model-forge/issues/17), [#18](https://github.com/HangGlidersRule/model-forge/issues/18),
[#20](https://github.com/HangGlidersRule/model-forge/issues/20).

## 3. Calibration set

Build it with a committed, deterministic script (`models/<family>/calibration/`) and record the output file's sha256:

- **match the deployment distribution:** chat models get chat-templated conversations plus plain text
  (cnn_dailymail + Nemotron-Post-Training-v2, 512 + 512 for Lightning); an embedder gets instruction-prefixed queries
  and passages as its clients send them (MS MARCO, 256 + 256); a decision model gets its own task prompts;
- pin every source dataset by revision, fix the seed, and keep evaluation prompts out of it (held-out rows only);
- hold out separate prompt sets for KL selection and KL reporting.

## 4. Builds

Builds run NVIDIA ModelOpt at a pinned commit inside a pinned container image. Each pin is a file under
`configs/modelopt/`: `pin.json` (`0.46.0rc2`) still governs the earlier families and their shipped artifacts, and the
current campaigns use `pins/modelopt-0.48.0rc1.json` (`0.48.0rc1` at `ff2450c`, the newest build at the time), with the
release-candidate caveat recorded on their cards.

- **Recipes are YAML** (`configs/modelopt/recipes/<family>/…`), loaded with `hf_ptq.py --recipe`. Rules apply in order,
  so put exclusions after the enables they override, and keep the MTP-disable rule last.
- **Smoke first:** a 16-sample build plus a boot and one request catches layout and loader problems before a
  multi-hour calibration.
- **Dry-run the recipe on the meta device** (no weights, CPU only) and count which quantizers it enables: it catches a
  wrong glob before any GPU time is spent.
- **Typical layouts:**
  - *W4A16 NVFP4:* weight-only FP4 on the MLP / experts; everything served in BF16 activations.
  - *W4A4 NVFP4:* FP4 weights and activations on the MLP / experts; needs activation calibration.
  - *Mixed-FP8:* FP8 on attention, linear-attention or Mamba projections next to FP4 MLPs (NVIDIA's own layout for
    several hybrids).
  - Keep `lm_head` BF16 when it carries a task head (decision models); MTP heads, routers, convolutions, norms and
    embeddings stay BF16.
- **Activation scales matter for W4A4.** One static scale covers each activation stream; calibration can clip
  (Lightning's routed-expert down-projection inputs sat at 0.12-0.42x of their largest activation). Collect per-stream
  statistics on the BF16 model and choose scales by simulated quantization error; pre-register the rule (issue 17,
  R17-A1).
- **Post-process deterministically:** transformers 5 re-serializes some configs on export (nemotron_h layer-type
  names); restore the source `config.json` / `generation_config.json` and keep only the export's
  `quantization_config`.

## 5. Validate the artifact (fail closed)

Run a validator over the **exact files to be uploaded** and commit its JSON report:

- the quantized set is exactly the intended modules, each with the expected tensors and dtypes;
- every per-tensor scale finite and positive; FP8 block scales checked on the uint8 view for NaN (0x7F/0xFF) and zero;
- fused groups (q/k/v, gate/up, linear-attention projections) share their input and global scales;
- every other tensor byte-identical to the source; every source tensor accounted for;
- configs identical to the source outside quantization metadata; tokenizer, template and companion files
  byte-identical;
- `hf_quant_config.json` algorithm, group size, KV setting and per-layer map exactly as intended.

## 6. Evaluate and select

See the [evaluation guide](evaluation.md). In short: every candidate is measured on the exact artifact, against the
family's own BF16 control and a BF16 replica for the noise floor, with the gates and selection rule fixed in step 2.
Failed gates block shipping; redo the method, don't bypass the gate. Shipping something that failed its frozen bar is
an explicit owner decision, recorded as such in the card, ledger and release notes.

## 7. Card and family record

- **Card:** the gold-standard skeleton ([`../model-card-gold-standard.md`](../model-card-gold-standard.md)); the
  `vllm serve` block is the exact smoke-tested command, including `--served-model-name`.
- **Every number comes from a committed results file.** A committed builder script derives each range, percentage and
  verdict from the measured files into one gate summary; a committed checker then confirms every number on the card
  appears in that summary and fails on any miss. Examples:
  `models/qwen3-embedding-8b/eval/{build_gate_summary,check_card_numbers}.py`,
  `models/nemotron-3.5-lightning-r1/eval/{build_w4a4_summary,check_card_numbers}.py`.
- **Family record:** README, cards, recipes, evidence JSON, eval and calibration scripts, ledger, and export rules for
  any new file, plus rows in `README.md` and `recipes/README.md`.
- **Ledger:** `models/<family>/results/publication-readiness-ledger.json`, validated against
  `contracts/darkstar-release/v2/ledger.schema.json` and the lifecycle, target-repository and precision checks in
  `src/model_forge/release.py`.
- **Production data stays private.** Results measured on private stores go public only as aggregates under neutral
  descriptions.

## 8. Hugging Face publication

Run with the card, record and full test suite already green: the README ships inside the upload.

1. **Private upload.** Write `manifest.sha256` over every file, create the repository private and assert it, upload,
   then read back the file list and every LFS sha256.
2. **Clean-download smoke.** On a serving host, download the uploaded commit into an empty cache, verify every
   manifest entry, serve with the card's exact command (only port and memory sizing changed), and run the card's own
   requests. Models whose outputs get stored (embedders) also compare outputs with the evaluated build.
3. **Public flip.** Assert the head is still the smoked commit, make it public, and check anonymously that the card page
   and every file return 200 and that the hub README equals the staged card.
4. **Collections.** Add the repository to the family collection; Darkstar Releases features one product per family.

## 9. Attested public export

The public repository is generated from the private one; nothing is edited by hand on the public side.

- `tools/public_export/public-files.yaml` classifies every tracked file: copy, transform (sanitize), or exclude, with a
  reason and an owner. `scripts/stage_public_root.sh` builds the export, runs the detectors (private addresses, host
  names, local paths, raw benchmark keys, secrets), runs gitleaks over the source history, and writes
  `PUBLIC_EXPORT_MANIFEST.json` with the source commit and a payload tree digest.
- An independent verifier rebuilds the export in a clean environment from a pinned wheelhouse and checks the digest.
- Before opening the pull request: `diff -r` the export against public main. Only the intended files may differ, and
  earlier families must be byte-identical.

## 10. Review and merge

- Continuous integration (lint, types, tests, secret scan, PR risk), Cursor Bugbot, and an automated verification review
  that recomputes every quoted number from the shipped files.
- Fix findings at the private source and re-export. Bugbot reviews only a pull request's first push: request a re-run
  after each fix.
- Merge only the reviewed head (`gh pr merge --match-head-commit <sha>`).

## 11. Tag, release, audit

- Tag the merge commit `darkstar-<family>-vX.Y.Z` (a product added to a published family gets a minor version) and
  publish the release notes.
- `scripts/verify_family_release.sh <family> <repo…>` from public main checks every surface: each repository public with
  its card and weights index, the catalog row, the collection, the staged cards and the tag. The family counts as
  released only when it exits 0.
- Close the issue with links to the checkpoint, release, collection and audit.
