# Darkstar four-product release process

- **Document status:** normative
- **Version:** 1.0.0
- **Machine-readable contract:** [`contracts/darkstar-release/v1/contract.json`](../contracts/darkstar-release/v1/contract.json)
- **Ledger schema:** [`contracts/darkstar-release/v1/ledger.schema.json`](../contracts/darkstar-release/v1/ledger.schema.json)

This document defines the immutable release process for Darkstar model families. It is normative: a
family may not be published unless every applicable gate below is satisfied with immutable, tracked
evidence for the exact product artifact. The gates, statuses, and publication rule are encoded in the
versioned machine-readable contract and enforced by CI.

The **ledger is the source of truth.** There is deliberately **no separate human-readable gap
report.** The per-family publication-readiness ledger (a JSON document validating against the ledger
schema) carries every status; those statuses are rendered directly into the Git-tracked benchmark
matrix and the per-product model cards inside `LEDGER-SYNC` comment blocks. Family
READMEs and publication plans may summarize and link the embedded matrices, but must not duplicate
them into an orphan document.

## 1. Canonical products

Every family has exactly four first-class products. No more, no fewer:

| Role | Canonical product id | Notes |
|---|---|---|
| `base-bf16` | `Darkstar-<Family>-Base-BF16` | Clean, unedited BF16 base |
| `base-modelopt-nvfp4` | `Darkstar-<Family>-Base-ModelOpt-NVFP4` | Clean base quantized with NVIDIA ModelOpt NVFP4 |
| `abliterated-bf16` | `Darkstar-<Family>-Abliterated-BF16` | Reproducible refusal-direction edit, BF16 |
| `abliterated-modelopt-nvfp4` | `Darkstar-<Family>-Abliterated-ModelOpt-NVFP4` | Abliterated BF16 quantized with the same winning ModelOpt recipe |

The canonical product ids above are role labels. **No actual identifier may drop precision wording.**
Here, an actual identifier means an artifact/publication identity: a reserved repository id,
model-card identity heading, or recipe publication target. Each must
spell out the real precision class, so a shortened `Base-NVFP4` or `Abliterated-NVFP4` repository id
is a contract violation rather than an accepted abbreviation. Runtime API aliases follow the separate
concise standard in Section 1.2 and are not artifact/publication identifiers. Upstream artifacts keep
their producer's name and are never branded Darkstar.

**ModelOpt-NVFP4 is a release slot, not a precision claim.** The `Base-ModelOpt-NVFP4` and
`Abliterated-ModelOpt-NVFP4` slot ids describe *where* a quantized product ships, not its activation
precision. Every selected, rejected, or under-evaluation candidate — and every actual Hugging Face
repository and model-card identity — MUST encode the real precision class and MUST NOT hide it behind the
bare slot name. The canonical id order is
`Darkstar-<Family>-<Base|Abliterated>-ModelOpt-<W4A16|W4A4>-NVFP4[-Mixed-FP8]`, for example
`Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8` (mixed: W4A16 NVFP4 on language MLP and
`lm_head`, FP8 on self-attention and GatedDeltaNet projections, BF16 protected/KV) versus
`Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A4-NVFP4` (uniform W4A4). A mixed-precision artifact is never
labelled uniform W4A4, and a uniform W4A4 artifact is never labelled W4A16. Each candidate carries a
complete precision map (`language_mlp`, `lm_head`, `self_attention`, `gdn_projections`, `kv_cache`,
`protected`); omission or conflation of the precision map is a contract violation enforced by CI.

### 1.1 Slot ids versus actual ids

The bare slot id is permitted **only where nothing concrete resolves to it**: this process document,
the machine-readable contract, and the ledger's `product_id` (with its rendered `LEDGER-SYNC` blocks).
Every actual identifier must encode ModelOpt plus `W4A16` or `W4A4`, plus `Mixed-FP8` when FP8 is part
of the recipe:

| Actual identifier | Rule |
|---|---|
| Hugging Face target repository (`target_repository`) | Precision-encoded, or explicitly unresolved |
| Candidate repository id | Exactly `<namespace>/<candidate_id>`, reserved even before the build |
| Model-card identity heading | Names the resolved target, or every candidate id when unresolved |
| Model source (`vllm serve MODEL`, `--model`) | Local artifact path or precision-encoded repository id |
| Served API alias (`--served-model-name`) | Lowercase `darkstar-<family>-<behavior>-<format>` |
| Recipe publication target (`outputs.publication.huggingface_nvfp4`) | Precision-encoded, or null when unresolved |

An id must not contain both `W4A16` and `W4A4`; `Mixed-FP8` is valid only on a `W4A16` id; a uniform
`W4A4` id never encodes FP8; and FP8 in an id is always spelled `Mixed-FP8`.

**`Mixed-FP8` is derived, not declared.** Whether an artifact belongs to the mixed class is read off
its precision map, not off the names it happens to carry. If any served **weight** path
(`language_mlp`, `lm_head`, `self_attention`, `gdn_projections`, `protected`) is FP8, then the
candidate's `precision_class`, `candidate_id`, reserved repository id, the product `target_repository`
once resolved to it, and its model-card identity heading must encode
`Mixed-FP8`; a missing `Mixed-FP8` fails validation instead of passing quietly. A candidate with no
FP8 weight path must not encode `Mixed-FP8` anywhere. `kv_cache` is deliberately excluded from the
derivation: runtime KV metadata is never evidence of the mixed class, and runtime BF16 KV stays a
separate documented property.

**Unresolved targets.** A ModelOpt NVFP4 product resolves `target_repository` only to a candidate that
has been built. Until then the target is `null` with status `unresolved_pending_precision_winner`, the
model card declares the candidate ids instead of a slot id, and only candidate-specific repository ids
are reserved. Reserving or publishing a bare `Base-NVFP4`/`Abliterated-NVFP4` target is forbidden —
there is no "reserved name" exemption. This is enforced fail-closed in
[`../src/model_forge/release.py`](../src/model_forge/release.py) and
[`../tests/test_release_contract.py`](../tests/test_release_contract.py) across the ledger, model
cards, serve examples, recipes, and repository documentation.

### 1.2 Served-model aliases

Every current and future Darkstar runtime exposes a concise lowercase API alias through
`--served-model-name`:

`darkstar-<family>-<behavior>-<format>`

- `<family>` is a stable lowercase family slug such as `qwen38`.
- `<behavior>` is `base` or `abliterated`.
- `<format>` is the externally meaningful runtime format, such as `bf16` or `nvfp4`.

The alias is an API identity, not a publication identity. It does not weaken or replace the
precision-encoded Hugging Face repository/model-card identity. For example, repository
`Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8` is served as
`darkstar-qwen38-abliterated-nvfp4`. Runtime container names should remain stable and descriptive;
Product 4 uses `vllm-darkstar-qwen38-abliterated-modelopt`.

## 2. Statuses

Every gate on every product carries exactly one status:

- `verified` — immutable tracked evidence in this repository satisfies the gate for this exact product.
- `in_progress` — work is documented or claimed but the immutable evidence has not been curated.
  Live, transient, or undocumented work is at most `in_progress`, never `verified`.
- `missing` — no qualifying evidence exists, or the required proof is unknown. **Unknown is missing.**
- `rejected_historical` — a prior artifact or result kept for lineage only (e.g. llm-compressor
  compressed-tensors NVFP4, or mixed-protocol GPQA). Never republished, never counted toward publication.
- `not_applicable` — the gate does not apply to this product role.

## 3. Required gates (per product)

Each gate is defined normatively here and canonically in the contract. `applies_to` marks which roles
the gate is applicable to; on every other role the gate is `not_applicable`.

1. **provenance_ownership** (all) — pin the upstream source to a full 40-character SHA, record the
   edit/quantization lineage, and resolve who owns and is licensed to publish this exact artifact.
2. **artifact_manifest** (all) — freeze the exact file list and SHA-256 checksums.
3. **recipe_edit_manifest** (all) — a pinned declarative recipe (plus edit/target manifest for
   abliterated products) defines the build and hashes cleanly.
4. **artifact_validation** (all) — fail-closed structural validation with committed evidence:
   tokenizer/processor/chat/generation config preserved; protected/vision/MTP tensors intact; no drift.
5. **abliteration_pass** (abliterated only) — a reproducible refusal-direction projection with pinned
   corpora, layer, seed, target inventory, and leakage bound, recorded as immutable evidence.
6. **modelopt_candidate_comparison** (ModelOpt NVFP4 only) — pinned NVIDIA ModelOpt **W4A16 vs
   W4A4/mixed** candidate comparison with the winning recipe recorded.
7. **nvfp4_tensor_scale_validation** (ModelOpt NVFP4 only) — validate tensor and scale integrity (no
   empty/zero/NaN/Inf scales), no quantized vision, BF16 MTP preserved, no mixed precision inside fused
   groups (fused q/k/v, gate/up, linear-attn qkv+z, a+b), no FP8 KV metadata.
8. **performance_profile** (all) — an **independent**, warmup-aware single-stream throughput profile
   measured on this exact product. Never inherited from another artifact.
9. **serving_capacity_profile** (all) — an **independent** serving/concurrency/capacity profile, kept
   separate from single-stream throughput.
10. **gpqa_matched_full_denominator** (all) — matched GPQA Diamond under the frozen protocol (Section 5)
    with **198/198** terminal parseable responses (full denominator) on this exact product.
11. **behavior_refusal_eval** (all) — harmful-refusal and safe-over-refusal evaluation on this exact
    product with zero errors. Abliterated products additionally require a **fresh abliteration eval**.
12. **serve_profile_frozen** (all) — the exact serve command/runtime profile is frozen (Section 4).
13. **model_card_final** (all) — complete card with resolved license/commit/tag and evidence links; no
    `*_PLACEHOLDER` values remain.
14. **publication_targets_hf_ghcr** (all) — artifact and required container images published to their
    pinned HF/GHCR targets.
15. **clean_download_boot_smoke** (all) — re-download to a clean mount and boot/smoke successfully.
16. **release_tag** (all) — an immutable Git release tag/commit is cut and referenced from the card.
17. **no_inherited_unverified_results** (all) — no runtime, quality, or behavior result is carried over
    from another artifact or protocol without direct verification on this exact product.

## 4. Serving / runtime axis applicability (applicable vs N/A)

Independent performance and serving/capacity tuning must explicitly declare each runtime axis as
**applicable** (tuned and recorded) or **N/A** (with reason). No axis is silently skipped, and no
result is inherited across products. Default handling for the Qwen3.8-class hybrid multimodal
architecture is below; each family overrides as needed in its record.

| Axis | Default handling | Reason |
|---|---|---|
| Speculative decoding (MTP) | applicable | Native MTP tensors present; depth is swept per product |
| DFlash | N/A | Not used by the pinned vLLM serving path |
| FlashAttention | applicable | Attention backend selection is tuned and recorded |
| FlashInfer | applicable (or N/A) | Compared against FlashAttention; cold JIT/autotune quarantined |
| Marlin / CUTLASS | applicable | Linear/FP4 backend selection recorded per quant format |
| Prefix caching | applicable | Tuned; hashing scheme recorded |
| Chunked prefill | applicable | Tuned and recorded |
| Scheduler budgets (max-num-batched-tokens) | applicable | Recorded as a capacity knob, not a TGen knob |
| Kernel selection | applicable | Selected kernels captured in evidence |
| CUDA graphs / compile | applicable | Compiled mode recorded; cold compile quarantined from steady-state |
| LMCache | N/A by default | Enable only with recorded evidence |
| max-num-seqs | applicable | Capacity knob, separate from single-stream |
| Served model length | applicable | Context length pinned per product |
| GMU (GPU memory utilization) | applicable | Recorded per serve profile |
| KV dtype | applicable | BF16 during recipe attribution; no implicit FP8 KV metadata |
| Concurrency / capacity | applicable | Serving profile only; never mixed with single-stream TGen |
| Recurrent cache mode (SSM/GatedDeltaNet) | applicable | Hybrid architecture requires an explicit mode |

Cold JIT/autotune/compile results are quarantined from steady-state scores. A promoted winner must
pass a clean-restart reproduction before it is recorded as `verified`.

## 5. Frozen GPQA semantics

GPQA Diamond is frozen. Any deviation is a different protocol and cannot fill a matched cell. The
in-repo harness is `src/model_forge/gpqa/`; the family protocol note is
[`../models/qwen3.8-27b-r3/gpqa-protocol.md`](../models/qwen3.8-27b-r3/gpqa-protocol.md).

- **Dataset:** GPQA Diamond, 198 questions; evaluated CSV SHA-256
  `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`. Question text and answer keys are
  never committed.
- **Prompt:** four-choice list; deterministic per-question shuffle `random.Random(i).sample(range(4), 4)`;
  final answer in `\boxed{A-D}`.
- **Parser:** last `\boxed{A-D}`, falling back to `Answer: A-D`.
- **Sampler:** temperature `1.0`, top-p `0.95`, top-k `20`; one sampled answer per question.
- **Thinking mode:** recorded per run; matched-matrix cells use thinking **off**. Thinking-on runs are
  secondary and not matched-matrix eligible.
- **Output cap:** recorded per run; uncapped runs must still reach a terminal parseable result.
- **Retries / timeout:** append-only resumable JSONL journal; 1800s per-request timeout with retries;
  resumption may not mix protocol versions or runtime identities.
- **Workers / runtime semantics:** worker count recorded; runtime image/version and serve command hashed.
- **Denominator:** publication requires **198/198** terminal parseable responses. Headline accuracy is
  always `correct / 198` (full denominator); completed-only accuracy is at most a secondary statistic.
- **Immutable hashes:** dataset checksum, harness version, artifact hash, protocol hash, and runtime
  hash are recorded with every result.

Never substitute an upstream or third-party GPQA figure for a missing cell. Never present project
reproductions as Artificial Analysis measurements.

## 6. Quantization requirements (ModelOpt NVFP4 products)

- Publication NVFP4 uses **pinned NVIDIA ModelOpt** (see [`../models/qwen3.8-27b-r3/modelopt/README.md`](../models/qwen3.8-27b-r3/modelopt/README.md)
  and [`../configs/modelopt/pin.json`](../configs/modelopt/pin.json)). Prior llm-compressor
  compressed-tensors NVFP4 artifacts are `rejected_historical`.
- Compare **W4A16 vs W4A4/mixed** candidates; record the winning recipe and its evidence. Each
  candidate is named for its explicit precision class and carries a complete precision map (Section 1).
- Validate tensors, scales, MTP, vision, and fusion groups per gate `nvfp4_tensor_scale_validation`.
- The Abliterated ModelOpt NVFP4 build reuses the selected clean Base ModelOpt recipe. It is an
  independent product and is not blocked on any other product: it may be built and tuned in parallel.

**Measurement verification vs promotion are distinct.** `gpqa_matched_full_denominator` and
`modelopt_candidate_comparison` record measurement facts about a candidate. A full-denominator GPQA
result is recorded as `verified` evidence about the measurement's trustworthiness; whether a candidate
is promoted to the release winner is a separate decision recorded by its `selection`/`promotion_status`.
For example, the clean Base ModelOpt comparison selected the mixed W4A16-NVFP4+FP8 candidate
(`153/198 = 77.27%`, within 2.02 pp of the Base BF16 baseline) on single-stream throughput and rejected
the uniform W4A4 candidate. Verified GPQA evidence never, by itself, makes a product publication-ready —
the publication rule (Section 9) still requires every applicable gate (including the publication-only
gates) to be `verified`.

## 7. Abliteration requirements (Abliterated products)

- The refusal-direction projection is reproducible: pinned harmful/harmless corpora, measurement layer,
  seed, deterministic normalization, target tensor inventory, and a bounded residual-leakage check.
- Both Abliterated products require a **fresh** harmful-refusal and safe-over-refusal evaluation on the
  exact artifact; a BF16 eval does not verify the NVFP4 build and vice versa.
- Reduced-refusal numbers are behavior measurements, not safety endorsements.

## 8. Publication outputs (per product)

A product publishes only when all of the following exist as tracked evidence and every applicable gate
is `verified`:

1. Source lineage (pinned upstream revision and edit/quantization chain).
2. Artifact manifest and recipe/edit manifest with SHA-256 checksums.
3. Validation evidence (structural; plus NVFP4 tensor/scale and abliteration where applicable).
4. Exact serve profile (runtime, KV dtype, context, speculation, kernels, budgets).
5. Independent performance evidence (single-stream) and serving/capacity evidence.
6. Behavior evidence where applicable (harmful-refusal / safe-over-refusal; fresh for abliterated).
7. Matched full-denominator GPQA (198/198).
8. Final model card (no placeholders).
9. HF/GHCR publication targets.
10. Clean-download boot/smoke verification.
11. Immutable release tag.

## 9. Publication rule

A product may claim publication only when **every gate that applies to its role is `verified`**;
`not_applicable` gates are excluded. Any gate that is `missing`, `in_progress`, or
`rejected_historical` blocks publication. This rule is enforced deterministically by
[`src/model_forge/release.py`](../src/model_forge/release.py) and
[`tests/test_release_contract.py`](../tests/test_release_contract.py).

### 9.1 Product lifecycle (build completion vs public release)

Build completion and public release are distinct. Gates split into **build gates** (they describe the
built artifact) and **publication-only gates** (`model_card_final`, `publication_targets_hf_ghcr`,
`clean_download_boot_smoke`, `release_tag` — they gate public upload). Each product declares a
`lifecycle`, which must match its gate statuses:

- `in_progress` — at least one applicable build gate is still `missing`/`in_progress`. The artifact may
  already be built and tuning; it is simply not build-complete.
- `locally_complete_unpublished` — every applicable build gate is `verified`, but at least one
  publication-only gate is still open. The artifact is a finished local product that has not been
  published (license/commit/tag placeholders are resolved at publication).
- `published` — every applicable gate, build and publication, is `verified`.

`model_forge.release.expected_lifecycle` derives the lifecycle from the gate statuses, and
`ledger_lifecycle_errors` fails closed if a declared lifecycle disagrees with what the gates prove or if
a non-`published` product carries a `publication_claim`.

## 10. Versioning and no-weakening

Gates may be **added** or **strengthened** in a new contract version (`contracts/darkstar-release/vN/`).
They may not be removed or weakened. CI freezes the required gate set, the status enum, and the
publication rule.

## 11. Citations

- NVIDIA TensorRT Model Optimizer (ModelOpt): <https://github.com/NVIDIA/TensorRT-Model-Optimizer>
- mlabonne abliteration (Maxime Labonne): <https://github.com/mlabonne/abliteration>
- Ritesh Khanna abliteration eval (`treadon/abliteration-eval`):
  <https://huggingface.co/datasets/treadon/abliteration-eval>

## 12. Related work

This process is the release-gate layer above the harness epic tracked in issues #4–#14
(<https://github.com/HangGlidersRule/model-forge/issues>). It references those issues rather than
duplicating them: schemas (#5), resumable DAG (#6), executors (#7), artifact/functional probes (#8),
performance tuning (#9), unified evaluation (#10), LLM assistants (#11), skills (#12), CI fixtures
(#13), and evidence/report generation (#14).
