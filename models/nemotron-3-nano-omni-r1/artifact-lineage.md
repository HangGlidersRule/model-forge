# Nano-Omni artifact lineage

## Provenance graph

1. **Upstream BF16 control** — `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` @
   `24e67ea000b7c2837fc8f9488aa2008524fac8ba` (NVIDIA Open Model Agreement).
2. **Abliterated-BF16 (SFT-r1)** — authored from pristine trunk weights: rank-32 LoRA adapter on
   the language trunk's q/k/v/o projections (α=64, dropout 0.05, 3 epochs, lr 1e-4 cosine), teacher
   = the model's own compliant outputs for `treadon/abliteration-eval` rows (25 harmful-compliant +
   77 safe rows after strict refusal re-filtering; teacher-selection boundary enforced — no
   compliant-harmful content sourced from any model), merged in float32 into the pristine language
   trunk, 17 shards repaved. Vision tower, audio encoder, tokenizer, and wrapper byte-identical.
   Behavior gate 200/200 + 0/83 + 0 errors. GPQA 60/198 (30.3%; parse=20).
3. **Base-ModelOpt-NVFP4** — ModelOpt `0.46.0rc2` (`43fd41a`) quantization of the upstream trunk at
   the pinned revision (no edit). Trunk extracted standalone (plain `nemotron_h`) and quantized.
   **Correction (#35):** the step that was meant to reassemble the VLM wrapper was never applied to
   the v1.0.0 artifact, which was the bare trunk (18,114 tensors, 0 tower tensors), and its router
   correction biases were rounded through BF16. Worse, a re-upload on 2026-09-03 replaced this
   repository's shards with the Abliterated W4A16 shards; the base build survived only at Hub
   commit `f3ae211e`. v1.1.0 restores that build and repairs it in place (below). Recipe:
   `w4a16_nvfp4_lmhead_nemotron_h.yaml` (NVFP4 experts + NVFP4 lm_head; Mamba/SSM/attention/norms/
   embeddings/towers protected). Behavior: unedited control. GPQA 84/198 (42.4%) on vLLM 0.27.1
   (v1.0.0, measured before the overwrite); 93/198 for v1.1.0 on vLLM 0.32.0 (two runs: 90, 96).
4. **Abliterated-ModelOpt-NVFP4** — same quantization contract applied to the edited BF16 artifact
   (step 2), with the same two defects in v1.0.0, repaired in v1.1.0. GPQA 58/198 (29.3%) on vLLM
   0.27.1 (v1.0.0); 64 and 71 in two runs for v1.1.0 on vLLM 0.32.0, beside its BF16 parent's 70/198 in the same
   session. Behavior 200/200 + 0/83 on both versions.

## v1.1.0 repair (2026-11-08, #35)

The three owned checkpoints were repaired in place under the v1.1.0 tag (live since 2026-11-10; receipt `results/repair-publication-receipt-v1.1.0.json`); their repositories keep their names:
- **Base W4A16:** the true base build from Hub commit `f3ae211e` (its quantized-layer names normalized to the
  `backbone.*` convention by `scripts/normalize_quant_names.py`; no tensor bytes change).
- **W4A16 products:** the wrapper reassembled byte for byte around the unchanged quantized trunk
  (`scripts/reassemble_omni_wrapper.py`), with all 1,106 tower tensors identical to upstream.
- **Abliterated products:** the 23 router correction biases restored to the upstream float32 bytes (the base
  build at `f3ae211e` already carried them)
  (`scripts/restore_router_bias.py`).
- **Verification:** `eval/verify_omni_repair.py` reads every tensor and checks its source: router biases and
  towers against upstream, trunk against v1.0.0. All three PASS (`results/repair-verify-*.json`).
- **W4A4 rebuild:** candidates A and B did not pass their pre-registered bars and are not published.

## Rejected lineage (kept for the record)

- Projection-family variants (single, dual, tri-axis, norm-preserve, biprojection, expanded corpus,
  9 configurations total): behavior up to 196/200 but GPQA collapsed to 26.8–27.3% with 60+
  unparseable outputs — closed-form ceiling reached, not shippable at quality.
- SFT-r2 (alpaca-mix variant): behavior 200/200 but GPQA collapsed to 6.6% — rejected.
- Historical compressed-tensors NVFP4 artifacts: rejected by the ModelOpt NVFP4 export standard.

## Publication

The three owned checkpoints are published (private-first creation, verification, then public flip)
with gold-standard cards carrying the measured gates, and are collected in the
[Darkstar Nemotron-3-Nano-Omni collection](https://huggingface.co/collections/HangGlidersRule/darkstar-nemotron-3-nano-omni-30b-a3b-6a9a2608d4e2ad4c43a996a1).
The family release tag is `darkstar-nemotron-3-nano-omni-v1.1.0` (v1.0.0: `darkstar-nemotron-3-nano-omni-v1.0.0`).
