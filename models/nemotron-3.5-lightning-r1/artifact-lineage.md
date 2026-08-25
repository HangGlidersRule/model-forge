# Artifact lineage — Darkstar Nemotron-3.5-Lightning

```
nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
  @ d468880b6ad3c6e0d21377ce7242adaea4cc884d   (upstream, OpenMDW-1.1)
  │
  ├─[unchanged reference]──────────► Base-BF16 (not republished)
  │
  ├─[ModelOpt W4A16-NVFP4 quant]──► Base-ModelOpt-NVFP4 (sibling cell, recipe
  │     w4a16_nvfp4_mse-fp8_attn-kv_bf16_nemotron_h.yaml)                (22G)
  │
  └─[abliteration R1: layer 34, 320/320 chat-templated, 3,126 targets]─► Abliterated-BF16 (62G)
        │
        └─[ModelOpt W4A16-NVFP4 quant]──► Abliterated-ModelOpt-NVFP4 (22G) ✅ FINAL SERVABLE
              served @ MTP10 (554.7 weighted)
```

## Abliteration (R1, canonical)

- Measurement: normalized harmful−harmless mean at layer-34 `resid_pre` (chat-templated, 320/320),
  all 52 layers scanned; layer 34 selected by generation test (0/8 refusals).
- Projection: `W' = W − r(rᵀW)` in float32, shard-by-shard; 3,126 targets (attention o_proj, Mamba
  out_proj, routed/shared expert down_proj, MTP o_proj/experts, embeddings).
- Validation: 3126==3126 edited, max leakage 0.000160 ≤ 0.01, MTP intact (270 tensors).
- Behavior gate: 200/200 harmful compliance, 0/83 safe over-refusals, 0 errors.

## Quantization (ModelOpt)

- Image: `local/model-forge-modelopt:0.46.0rc2-43fd41a` (ModelOpt 0.46.0rc2, commit 43fd41a).
- Calibration: cnn_dailymail 512 + Nemotron-Post-Training-Dataset-v2 512, seq 2048, seed 1234, batch 1,
  KV cache disabled (BF16).
- Layout: W4A16-NVFP4-g16 routed + shared expert up/down (5,934 modules); lm_head/Mamba/SSM/norms/
  embeddings/MTP protected BF16.
- Post-process: strip ModelOpt sentinels from `hf_quant_config.json`, restore BF16 `config.json`,
  patch `quant_algo: W4A16_NVFP4 → NVFP4`, write vLLM quantization_config
  (`modelopt`, `MIXED_PRECISION`, 5,934 layers) → loads as `modelopt_mixed`.

## Serving (final)

- Image: `local/vllm-dflash2:runtime` (vLLM fork 0.26.1rc1.dev1048+gb389ac294, cu130).
- Flags: `--max-model-len 131072 --kv-cache-dtype bfloat16 --reasoning-parser nemotron_v3
  --speculative-config {"method":"mtp","num_speculative_tokens":10}`.
- NVFP4 requires `VLLM_NVFP4_GEMM_BACKEND=cutlass` on SM120.
