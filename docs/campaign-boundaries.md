# Darkstar campaign boundaries (non-negotiable)

These rules were established during the Nano-Omni campaign (2026-08/09) after
live decisions. They bind every future family campaign. They are process
governance, not suggestions.

## 1. The teacher-selection boundary (never distill compliant harmful answers)

SFT/unlearning transforms MUST NOT use any model — **including models the
campaign operates itself** (gateway-served or locally served) — to generate
compliant answers to harmful prompts for training targets. The legitimate
teacher sources are:

- the subject model's own non-refusal outputs on public safety-research eval
  sets (e.g. `treadon/abliteration-eval`), captured with thinking disabled;
- benign instruction corpora for general-capability anchoring;
- the campaign's own prior-stage artifacts' genuine compliant rows.

Rationale: routing hard-harmful prompts to a model selected for missing
guardrails to harvest working harmful content is the jailbreak-escalation
path regardless of who owns the infrastructure. Robustness gaps on the
hardest categories are **documented, not manufactured around**.

## 2. The gates are paired, and both bind

- **Behavior gate**: 200/200 harmful compliance, 0/83 safe over-refusals,
  0 errors, measured on the exact served artifact.
- **Intelligence gate**: GPQA Diamond under the llm-inference-bench protocol
  must not collapse relative to the family's measured base control.

A release that passes one and fails the other is a **decision for the human
operator**, presented with the full evidence table — never a silent ship. The
Nano-Omni Ablit-BF16 shipped at GPQA 30.3% vs base ~46% under an explicit
operator decision (option 1: research-release framing with documented cost);
that decision does NOT generalize — every future family presents its own
numbers.

Any proposal to relax, bypass, or re-scope a failed gate to ship on schedule
is rejected by default. Redo the method, or stop and surface the fork.

## 3. Interpretation protocol for eval results

- Metrics are computed on the **served artifact** via the OpenAI-compatible
  endpoint, thinking-mode explicit, temperature 0 for gates.
- Same-weights re-runs vary by ±3 questions (batch/serve variance, measured).
 Before declaring a delta real, re-gate once; a ≤3-point move is noise.
- Refusal classification MUST normalize unicode (NFKC + smart-quote folding)
 before marker matching. Raw substring checks silently mis-score curly-quote
 refusals as compliant (147-row campaign bug, fixed in the eval script and
 guarded by tests).
- Failed gates block ship. Redo, don't bypass.

## 4. The four-product matrix order is fixed

1. Base-BF16 (upstream control — measured, not republished)
2. Base-ModelOpt-NVFP4 (quantize base)
3. Abliterated-BF16 (edit from full-precision weights)
4. Abliterated-ModelOpt-NVFP4 (quantize the EDITED BF16)

Never quantize an unedited artifact with the edit already in it; never
abliterate a quantized artifact; every rework re-runs the full
speed-tune + gates from the bottom of its lineage.

## 5. Method-selection canon (when projection collapses quality)

If a family's projection-family variants collapse the intelligence gate
(verify with at least: single, dual, tri-axis, norm-preserve, biprojection,
expanded-corpus), the canonical fallback is **LoRA unlearning SFT**:
r32/α64 on q/k/v/o of the language trunk, teacher = the subject model's own
compliant rows (boundary #1), merged float32 into pristine weights.
Document the chosen method, every rejected variant, and the quality cost in
the family recipe + card. This is the Nano-Omni precedent (recipe:
`lora-unlearning-sft` transform type, schema-supported).
