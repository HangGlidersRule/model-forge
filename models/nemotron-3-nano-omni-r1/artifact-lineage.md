# Nano-Omni artifact lineage

## Provenance graph

1. **Upstream BF16 control** — `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` @
   `24e67ea000b7c2837fc8f9488aa2008524fac8ba` (NVIDIA Open Model License).
2. **Abliterated-BF16 (SFT-r1)** — authored from pristine trunk weights: rank-32 LoRA adapter on
   the language trunk's q/k/v/o projections (α=64, dropout 0.05, 3 epochs, lr 1e-4 cosine), teacher
   = the model's own compliant outputs for `treadon/abliteration-eval` rows (25 harmful-compliant +
   77 safe rows after strict refusal re-filtering; teacher-selection boundary enforced — no
   compliant-harmful content sourced from any model), merged in float32 into the pristine language
   trunk, 17 shards repaved. Vision tower, audio encoder, tokenizer, and wrapper byte-identical.
   Behavior gate 200/200 + 0/83 + 0 errors. GPQA 60/198 (30.3%; parse=20).
3. **Base-ModelOpt-NVFP4** — ModelOpt `0.46.0rc2` (`43fd41a`) quantization of the upstream trunk at
   the pinned revision (no edit). Trunk extracted standalone (plain `nemotron_h`), quantized, and
   the VLM wrapper reassembled with towers byte-identical from the source BF16. Recipe:
   `w4a16_nvfp4_lmhead_nemotron_h.yaml` (NVFP4 experts + NVFP4 lm_head; Mamba/SSM/attention/norms/
   embeddings/towers protected). Behavior: unedited control. GPQA 84/198 (42.4%).
4. **Abliterated-ModelOpt-NVFP4** — same quantization contract applied to the edited BF16 artifact
   (step 2). GPQA 58/198 (29.3%) — zero quantization cost measured against the BF16 parent on the
   same runner the same day. Behavior 200/200 + 0/83. This is the family's final servable product.

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
The family release tag is `darkstar-nemotron-3-nano-omni-v1.0.0`.
