---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: finetune
pipeline_tag: text-generation
tags:
  - darkstar
  - qwen3.8
  - abliterated
  - reduced-refusal
  - bf16
  - vllm
---

# Darkstar-Qwen3.8-27B-Abliterated-BF16

> **Reduced-refusal model:** a refusal-direction edit was deliberately applied. Read the safety
> warning before use.

## Summary

BF16 derivative of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, produced by one audited refusal-direction projection
(internal lineage R3). The normalized float32 harmful-minus-harmless residual direction was measured
at language-model layer 38 with seed 42 and projected from exactly 131 residual-writing tensors.
The vision tower (333 tensors) was untouched, all 15 MTP tensors were preserved, and maximum
normalized residual leakage was approximately `0.000195`.

The direction corpora were
`mlabonne/harmful_behaviors@01cead01398926d81f7c52bdb790ee8cf77ebba7` (128 harmful) and
`mlabonne/harmless_alpaca@02c6a92cfcf11bb0c387334f8146d149d65b587f` (128 harmless), normalized
and deduplicated deterministically.

Full provenance, protocol, and caveats:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3.8-27b-r3/model-card/bf16.md)
- [Artifact lineage](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3.8-27b-r3/artifact-lineage.md)
- [Benchmark matrix](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3.8-27b-r3/benchmark-matrix.md)
- [GPQA protocol](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3.8-27b-r3/gpqa-protocol.md)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| GPQA Diamond (thinking off) | 146/198 = 73.74% | full denominator; 198/198 terminal parseable; 0 timeout/parse/error |
| Edit delta vs upstream BF16 control | -11 questions / -5.56 pp | 157/198 → 146/198 |
| Harmful-prompt compliance | 200/200 (0/200 refusals) | fresh suite on this artifact; 0 errors |
| Safe over-refusals | 0/83 (0.00%) | 0 errors |
| Single-stream throughput (MTP11) | 144.502 tok/s | independent single-stream sweep |

## Safety warning

This model has had its refusal direction deliberately reduced. It complied with 200/200 harmful
prompts in the measured suite and has no added safety mitigations. It will comply with many requests
the upstream model would refuse. Deploy only behind appropriate policy, filtering, access controls,
and legal review. Refusal-rate numbers are behavior measurements, not safety endorsements.

## Release reference

Engineering release: [`darkstar-qwen3.8-27b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-qwen3.8-27b-v1.0.0). This immutable tag exists and the release contract is published.

## Runtime

Validated with vLLM `0.27.1`, compiled mode, Flash Attention, BF16 KV cache, context 126,144, MTP
depth 11, 16K scheduler budget, and `max_num_seqs=16`:

```bash
vllm serve HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-BF16 \
  --served-model-name darkstar-qwen38-abliterated-bf16 \
  --kv-cache-dtype bf16 \
  --max-model-len 126144 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 16384 \
  --enable-chunked-prefill \
  --compilation-config 2 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 11}'
```
