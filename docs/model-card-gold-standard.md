# Darkstar model-card gold standard

Every published Darkstar checkpoint on Hugging Face uses EXACTLY this card
skeleton. The reference implementation is
`HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-BF16`. A card that deviates
from this structure is a publication defect.

## Canonical skeleton (in order)

1. **Front-matter**, fields in this order when present:
   - `license` (+ `license_name` + `license_link` when `license: other`)
   - `base_model` — upstream repo (base/quantized-from-upstream cards) or the
     Darkstar BF16 parent (quantized-of-abliterated cards)
   - `base_model_relation`: `finetune` for edited BF16, `quantized` for NVFP4
   - `pipeline_tag`
   - `tags` — canonical order: `darkstar`, family tag, `abliterated` +
     `reduced-refusal` (only on edited products; `base` on unedited
     quantizations), precision tags (`bf16`/`nvfp4` + `modelopt`), `vllm`
   - `quantization: nvidia-modelopt` (NVFP4 cards only)
   - `extra_gated_heading` (when the repo uses gated access)

2. **H1 title** = exact repository name.

3. **Blockquote warning** immediately after the title:
   - Edited products: `> **Reduced-refusal model:** a refusal-direction edit was
     deliberately applied. Read the safety warning before use.`
   - Unedited control quantizations: no blockquote (matches the Qwen Base-NVFP4
     gold card — the control card carries a "Safety and limitations" section
     instead).

4. **`## Summary`** — prose containing, in order:
   - upstream repo link and pinned revision SHA
   - the exact edit description (method, layer, seed, tensor count, leakage)
   - corpora with pinned revisions/SHAs
   - for quantizations: ModelOpt pin + recipe SHA + calibration contract +
     protected/quantized split + artifact layout
   - **provenance-link list**: Source card / Artifact lineage / Benchmark
     matrix / GPQA protocol — all pointing at the GitHub source repo paths.

5. **`## Evaluation`** — a 3-column table, header exactly `| Metric | Value |
   Basis |`. Rows (subset as applicable to the product):
   - GPQA Diamond with protocol basis (full denominator, parseable count,
     thinking mode, serving stack)
   - **control-delta row** vs the measured upstream/parent control
     (`-N questions / -M pp`) when a control number exists
   - harmful-prompt compliance (`200/200 (0/200 refusals)`)
   - safe over-refusals (`0/83 (0.00%)`)
   - single-stream throughput with the winning spec-decode profile
   - **Discipline line** required at the bottom of the section (or as a cell):
     `Missing cells are marked 'not measured' and are never backfilled from a
     different checkpoint or protocol.`

6. **`## Safety warning`** (edited products) or **`## Safety and limitations`**
   (unedited controls). The edited-product section ENDS with the exact closer:
   `Refusal-rate numbers are behavior measurements, not safety endorsements.`

7. **`## Release reference`** — the immutable engineering release tag link;
   `This immutable tag exists and the release contract is published.`
   (Families still mid-campaign may state where the release is tracked instead;
   the section heading itself is mandatory.)

8. **`## Runtime`** — validated-stack narrative (vLLM version family,
   attention backend, KV dtype, context, spec-decode depth, max_num_seqs)
   followed by the exact `vllm serve` block **including
   `--served-model-name`** and the speculative-config JSON when applicable.

## Conformance test

`tests/test_model_card_gold_standard.py` enforces this skeleton on every
`model-card/*.md` in the repository. Run it before any card upload. Cards on
Hugging Face are re-fetched and checked post-upload (see the publication
runbook).
