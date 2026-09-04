---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/download/eula/pdf/NVIDIA_Open_Model_License.pdf
base_model: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - darkstar
  - nemotron-h
  - base
  - nvfp4
  - modelopt
  - vllm
quantization: nvidia-modelopt
extra_gated_heading: Darkstar Nemotron-3-Nano-Omni 30B-A3B Base ModelOpt NVFP4
---

# Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Base-ModelOpt-NVFP4

## Summary

Clean, unedited NVIDIA ModelOpt quantization of
[`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
at revision `24e67ea000b7c2837fc8f9488aa2008524fac8ba` — the control artifact for the Darkstar
Nemotron-3-Nano-Omni family: same source revision, same quantization contract, no refusal-behavior
edit. The language trunk's MoE expert projections (routed + shared, 5,894 modules) and `lm_head`
are quantized to W4A16-NVFP4 group 16; the Mamba2/SSM path, attention, norms, embeddings, and the
vision/audio towers remain protected in BF16.

ModelOpt is pinned to `0.46.0rc2` at `43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a` with an MSE
calibration algorithm (`fp8_scale_sweep`). Calibration used `cnn_dailymail` plus
`nemotron-post-training-dataset-v2`, 512+512 samples, sequence length 2048, seed 1234, batch 1,
KV-cache quantization disabled (BF16). Quantization was executed on the language trunk extracted
from the VLM wrapper as a standalone plain `nemotron_h` checkpoint (tower tensors are recipe-
protected and byte-identical to upstream; the NVFP4 product is a text-trunk quantization).

Full provenance, protocol, and caveats:

- [Source card](https://github.com/HangGlidersRule/model-forge/blob/main/models/nemotron-3-nano-omni-r1/model-card/base-nvfp4.md)
- [Recipe](https://github.com/HangGlidersRule/model-forge/blob/main/recipes/nemotron-3-nano-omni/w4a16_nvfp4_lmhead_nemotron_h.yaml)

## Evaluation

| Metric | Value | Basis |
|---|---|---|
| GPQA Diamond (thinking on) | 84/198 = 42.4% | llm-inference-bench `gpqa-diamond`, chat template + thinking ON, temp 0; full denominator; 198/198 scored; 0 errors |
| Behavior: harmful-prompt compliance | not measured (unedited control) | refusal behavior unchanged by design; no edit applied |
| Safe over-refusals | not measured (unedited control) | same basis |
| Single-stream throughput (no spec decode) | 259.75 tok/s weighted | 4K 259.49 / 16K 259.68 / 48K 261.49; no speculative decoding in this family (no MTP, no compatible drafter) |

Missing cells are marked `not measured` and are never backfilled from a different checkpoint or
protocol.

## Safety and limitations

This is the unedited control quantization. The refusal measurements are behavior observations, not a
safety endorsement. Quantization can shift behavior relative to upstream BF16. Only single-stream
throughput is characterized here; aggregate concurrent throughput is separate.

Serving note: validated with the vLLM `0.27.1` build family (ModelOpt NVFP4 MoE loading operates
correctly there as of this release; the then-current `cu130-nightly` build regressed on ModelOpt
NVFP4 FusedMoE checkpoint loading for all checks, including the shipped Lightning NVFP4 artifact).

## Release reference

Family release: [`darkstar-nemotron-3-nano-omni-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-nemotron-3-nano-omni-v1.0.0). This immutable tag exists and the release contract is published.

## Runtime

Validated with vLLM `0.27.1` (aeon build, torch 2.13.0+cu130), MARLIN NvFp4 MoE backend, Flash
Attention, BF16 KV cache, context 131,072, `max_num_seqs=16`:

```bash
vllm serve HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Base-ModelOpt-NVFP4 \
  --served-model-name nano-omni-base-nvfp4 \
  --trust-remote-code \
  --kv-cache-dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm
```
