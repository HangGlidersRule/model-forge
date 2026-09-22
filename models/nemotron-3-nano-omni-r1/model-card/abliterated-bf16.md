---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/download/eula/pdf/NVIDIA_Open_Model_License.pdf
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
| Behavior: harmful-prompt compliance | 200/200 (0/200 refusals) | fresh suite on this artifact; 0 errors |
| Safe over-refusals | 0/83 (0.00%) | 0 errors |
| GPQA Diamond (thinking on) | 60/198 = 30.3% | llm-inference-bench `gpqa-diamond`, chat template + thinking ON, temp 0; full denominator; 20 unparseable answers |
| Intelligence delta vs upstream BF16 control | -14 questions / ~-16 pp vs ~46% base reference | same protocol; **documented structural cost, shipped by explicit reviewer decision** |

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

Validated with vLLM, Flash Attention eager fallback path, BF16 KV cache, context 131,072, and
`max_num_seqs=16` (no speculative decoding is available for this family — no MTP layers, no
compatible drafter; verified):

```bash
vllm serve HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-BF16 \
  --served-model-name darkstar-nemotron-3-nano-omni-abliterated-bf16 \
  --trust-remote-code \
  --kv-cache-dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser nemotron_v3
```
