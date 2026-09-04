---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - darkstar
  - qwen3.8
  - modelopt
  - nvfp4
  - fp8
  - vllm
---

# Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8

## Summary

Clean, unedited NVIDIA ModelOpt quantization of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The selected mixed-precision artifact uses W4A16
NVFP4 group 16 for language MLP projections and `lm_head`, FP8 e4m3 for self-attention and
GatedDeltaNet projections, and BF16 for protected components and runtime KV. It was selected over
the rejected uniform W4A4 candidate on single-stream throughput.

ModelOpt is pinned to `0.46.0rc2` at
`43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`; the selected operator recipe SHA-256 is
`90fc6b37c00334debd49f1975ab406b5e20667f07e4be0be3e463a648abac642`. Calibration used
`cnn_dailymail` plus `nemotron-post-training-dataset-v2`, 512+512 samples, sequence length 2048,
seed 1234. Vision, all 15 MTP tensors, `conv1d`, norms, embeddings, and runtime KV remain BF16.

Full provenance, protocol, and caveats:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3.8-27b-r3/model-card/base-nvfp4.md)
- [Artifact lineage](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3.8-27b-r3/artifact-lineage.md)
- [Benchmark matrix](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3.8-27b-r3/benchmark-matrix.md)
- [GPQA protocol](https://github.com/HangGlidersRule/model-forge/blob/main/models/qwen3.8-27b-r3/gpqa-protocol.md)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| GPQA Diamond (thinking off) | 153/198 = 77.27% | full denominator; 198/198 terminal parseable; 0 timeout/parse/error |
| Quantization delta vs upstream BF16 control | -4 questions / -2.02 pp | 157/198 → 153/198 |
| Harmful-prompt refusals | 197/200 (98.50%) | clean unedited behavior; 0 errors |
| Safe over-refusals | 4/83 (4.82%) | 0 errors |
| Single-stream throughput (MTP4) | 203.636 tok/s | selected mixed candidate; uniform W4A4 candidate: 129.441 tok/s |

## Safety and limitations

This is the unedited control quantization. The refusal measurements are behavior observations, not a
safety endorsement. Quantization can shift behavior relative to upstream BF16. Only single-stream
throughput is characterized here; aggregate concurrent throughput is separate.

## Release reference

Engineering release: [`darkstar-qwen3.8-27b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-qwen3.8-27b-v1.0.0). This immutable tag exists and the release contract is published.

## Runtime

Validated with vLLM `0.27.1`, compiled mode, Flash Attention, BF16 KV cache, context 126,144, MTP
depth 4, 32K scheduler budget, and `max_num_seqs=16`:

```bash
vllm serve HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8 \
  --served-model-name darkstar-qwen38-base-nvfp4 \
  --kv-cache-dtype bf16 \
  --max-model-len 126144 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --compilation-config 2 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 4}'
```
