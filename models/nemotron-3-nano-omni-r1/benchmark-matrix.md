# Nano-Omni benchmark matrix

Protocol: llm-inference-bench `gpqa-diamond` (chat template + thinking ON, temp 0, deterministic
letter shuffle, exact-match scoring, full 198 denominator) — see
[`docs/gpqa-abliteration-protocol.md`](../../docs/gpqa-abliteration-protocol.md).
Serving: vLLM `0.27.1` family (aeon build); NVFP4 products served through the MARLIN NvFp4 MoE
backend; `--reasoning-parser nemotron_v3`; BF16 KV cache. No speculative decoding in this family
(verified: no MTP layers, no compatible drafter).

| Product | Format | Edit | GPQA Diamond | Behavior | Throughput (weighted) | Status |
|---|---|---|---|---|---|---|
| Base-BF16 (upstream control) | BF16 | none | not measured (upstream reference; ~46% family baseline) | — | 182.7 tok/s | upstream reference |
| Base-ModelOpt-NVFP4 | W4A16-NVFP4 | none | 84/198 = 42.4% | not measured (unedited control) | 259.75 tok/s | public |
| Abliterated-BF16 (SFT-r1) | BF16 | LoRA unlearning SFT | 58/198 = 29.3% | 199/200 + 0/83 (re-gate 2026-09-03; shipped 200/200) | 182.7 tok/s | public |
| Abliterated-ModelOpt-NVFP4 | W4A16-NVFP4 | LoRA unlearning SFT | **58/198 = 29.3%** | **200/200 + 0/83** | **249.5 tok/s** | **public (final servable)** |

## Reading the cells

- **Edit effect (BF16 → Ablit-BF16)**: the GPQA delta vs the ~46% base reference (~-16 pp) is the
  documented structural cost of removing refusal behavior on this hybrid Mamba2 + MoE trunk. Every
  closed-form projection variant measured WORSE (26.8–27.3% with 60+ unparseable outputs — rejected
  lineage). The r1 LoRA-unlearning SFT is the documented best trade-off.
- **Quantization effect (Ablit-BF16 → Ablit-NVFP4)**: **0 GPQA points** (58/198 = 58/198 on the same
  runner the same day) — the verification matches the Nemotron-3.5-Lightning family's losslessness
  result, and the +36% throughput gain comes free.
- **Harmful compliance** is measured on the exact served artifact (200-prompt suite; smart-quote-
  normalized refusal classification, 0 errors on both products).

## Caveats

- GPQA correctness of those cells uses full-denominator scoring as published (parse behaviors are
  reported separately per artifact; cells marked `not measured` are never backfilled from a
  different checkpoint or protocol).
- Only single-stream throughput is characterized; aggregate concurrent throughput is separate.
- nvfp4 products quantify the language trunk only; vision/audio towers are protected BF16 by
  recipe and remain byte-identical to their lineage sources.
