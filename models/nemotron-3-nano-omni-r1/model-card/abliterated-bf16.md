---
license: other
license_name: nvidia-open-model-agreement
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
base_model: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
base_model_relation: finetune
pipeline_tag: image-text-to-text
tags:
  - darkstar
  - nemotron-h
  - abliterated
  - reduced-refusal
  - bf16
  - vllm
extra_gated_heading: Darkstar Nemotron-3-Nano-Omni 30B-A3B Abliterated BF16
---

# Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-BF16

> **Reduced-refusal model:** a refusal-behavior edit was deliberately applied. Read the safety
> warning before use.

## v1.1.0 repair

**Publication status:** the v1.1.0 files are staged and verified; this repository is updated to them when the release is tagged, and serves v1.0.0 until then.

**[model-forge#35](https://github.com/HangGlidersRule/model-forge/issues/35).** The v1.0.0 weights in this repository are the full `NemotronH_Nano_Omni_Reasoning_V3` wrapper, but the 23 MoE routers' `e_score_correction_bias` tensors, which upstream stores in float32, were rounded through BF16 (55.7645 stored as 55.75, for example), which shifts expert selection. v1.1.0 restores them to the upstream float32 bytes and changes nothing else: an independent provenance check reads every tensor and finds the 23 router biases and all 1,106 tower tensors identical to upstream, and all 6,220 other language-model tensors identical to v1.0.0 ([report](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/repair-verify-Abliterated-BF16.json)).

## Summary

BF16 derivative of
[`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
at revision `24e67ea000b7c2837fc8f9488aa2008524fac8ba`, produced by **LoRA unlearning SFT** rather
than weight projection: a rank-32 adapter (`q/k/v/o_proj`, α=64, dropout 0.05) trained for 3 epochs
(96 steps, grad-accum 8, lr 1e-4 cosine, bf16, max_length 1024; train loss 0.3402, mean token accuracy
99.98%) on the model's own compliant outputs for `treadon/abliteration-eval` prompts (25
harmful-compliant + 77 safe rows after strict refusal re-filtering), then merged in float32 into the
pristine language trunk. All 17 shards repaved; vision tower, audio encoder, tokenizer, and model
wrapper byte-identical to upstream; 62,990 MB total.

**Why not projection:** every closed-form projection variant (single, dual, tri-axis, causal-band,
norm-preserving, biprojected, expanded-corpus re-measured, 9 configurations) collapsed GPQA Diamond
to 26.8–27.3% with 60+ unparseable answers; a second SFT round with 600 general-instruction rows kept
behavior at 200/200 but collapsed GPQA to 6.6%. The refusal and reasoning circuitry share attention
head-space on this hybrid Mamba2 + MoE trunk; the LoRA route is the documented best trade-off.

Full provenance, protocol, and caveats:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/model-card/abliterated-bf16.md)
- [Recipe](https://github.com/HangGlidersRule/model-forge/blob/main/recipes/nemotron-3-nano-omni/darkstar-nemotron-3-nano-omni-30b-a3b-reasoning-abliterated-bf16.yaml)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| Behavior: harmful-prompt compliance | 200/200 (0/200 refusals) | measured on the repaired weights, served; 0 errors ([result](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/abliteration-eval-abliterated-bf16-repaired-v032.json)) |
| Safe over-refusals | 0/83 (0.00%) | same run; 0 errors |
| GPQA Diamond (thinking on) | 70/198 = 35.4% | llm-inference-bench `gpqa-diamond`, chat template + thinking ON, temp 0, full 198 denominator, 8,192-token answer cap, concurrency 30 (the published v1.0.0 protocol); vLLM v0.32.0 (pinned digest `befccf0e`, plus the `vllm[audio]` extras), one RTX PRO 6000 Blackwell; 0 errors ([result](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/gpqa-abliterated-bf16-repaired-v032.json)) |
| Published v1.0.0 (historical) | 60/198 = 30.3% | a different protocol (thinking off, 512-token answers, temperature 1.0); not comparable with the row above |
| Image and audio input | 14/14 images and 1/1 speech clip answered | serving window on the repaired weights; a clean download of the staged v1.1.0 files also answered 8 text, 2 image and 1 speech request ([result](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/results/repair-media-identity-v1.1.0.json)) |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or
protocol.

## Safety warning

This model has had its refusal behavior deliberately reduced. It complied with 200/200 harmful
prompts in the measured suite and has no added safety mitigations. It will comply with many requests
the upstream model would refuse. Deploy only behind appropriate policy, filtering, access controls,
and legal review. Refusal-rate numbers are behavior measurements, not safety endorsements.

## Release reference

Family release: [`darkstar-nemotron-3-nano-omni-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3-nano-omni-v1.0.0). This immutable tag exists and the release contract is published.

## Runtime

Validated with vLLM `0.32.0` plus the audio extras (`pip install "vllm[audio]"`), KV cache in the checkpoint dtype, context 131,072. `--mamba-ssm-cache-dtype float32` is required: vLLM does not set it for the Omni wrapper, and the default lowers quality.

```bash
vllm serve HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-BF16 \
  --served-model-name darkstar-nemotron3nanoomni-abliterated-bf16 \
  --trust-remote-code \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --mamba-ssm-cache-dtype float32 \
  --reasoning-parser nemotron_v3 \
  --gpu-memory-utilization 0.85
```
