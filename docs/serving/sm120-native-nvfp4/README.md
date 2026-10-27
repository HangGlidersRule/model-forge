# Native NVFP4 on RTX PRO Blackwell (SM120): FP4 tensor-core MoE for Nemotron-3.5-Lightning

**Status:** field report, 2026-10-27. Running in production on two cards.

**TL;DR.** NVIDIA's NVFP4 checkpoint for Nemotron-3.5-Lightning-30B-A3B is *weight-only* FP4 (W4A16). On every Blackwell card except GB200, vLLM serves it through Marlin, which converts the FP4 weights back to BF16 and multiplies in BF16. You get NVFP4's memory savings but none of its FP4 tensor-core throughput.

vLLM 0.32 ships an opt-in FlashInfer **b12x** MoE backend. It quantizes BF16 activations to FP4 inside the kernel, which turns that same W4A16 checkpoint into real FP4 math (W4A4 at runtime) on SM120. Out of the box, though, it does not start for this model. We needed three fixes:

1. a one-function vLLM patch;
2. the workspace-sharing change from an open upstream PR;
3. one FlashInfer environment variable.

**Result on a 32 GB RTX PRO 4500**, on a prefill-heavy workload (96% of tokens are prompt):
- 1.39-1.48x Marlin's prefill throughput;
- about 30% lower time to first token;
- 7% *more* KV cache;
- classification agreement indistinguishable from the run-to-run variance between two Marlin replicas.

Nothing here needs a new checkpoint, a rebuilt image, or extra GPU memory.

## Hardware and software

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 4500 Blackwell (GB203, SM120, 32 GB GDDR7), PCIe passthrough, one model per card |
| Guest | Ubuntu 26.04.1 LTS, kernel 7.0.0-38, NVIDIA open driver 615.71.10 |
| Serving | `vllm/vllm-openai:v0.32.0` (vLLM 0.32.0, FlashInfer 0.7.0.post1, torch 2.13.0+cu130), unmodified image plus one mounted file |
| Model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` @ `bee7596271d1495f6992ae224aefde4410e816b8` (128 routed experts, top-6, hidden 2688, `moe_intermediate_size` 1856, non-gated `relu2`, 23 MoE layers) |
| Quant layout (from `hf_quant_config.json`) | `W4A16_NVFP4` on 5,935 expert / shared-expert linears; FP8 on 46 Mamba in/out projections; MTP head excluded (BF16) |

RTX PRO 4500 and RTX PRO 6000 are the same Blackwell SM120 architecture, so this applies to both. Datasheet dense throughput for the 4500: FP4 about 808 TFLOPS, BF16 about 203. Marlin runs the expert GEMMs at the BF16 number.

## Why the checkpoint is W4A16, and why that costs you on Blackwell

The publisher's choice is reasonable: one W4A16 checkpoint serves Hopper and Ampere too, and leaving activations in 16-bit avoids activation-quantization error. The model card only claims a native FP4 compute path on GB200.

NVIDIA's own Model Optimizer team makes the cost explicit. They measured weight-only NVFP4 *slower than BF16* in 10 of 12 shapes on Blackwell, because a BF16 activation forces the Marlin dequantize-to-BF16 fallback. W4A4 beat BF16 in 9 of 12, up to 1.30x, with the gains growing with batch size.

The stored FP4 weight bytes are identical between W4A16 and W4A4. The difference is only whether activations are quantized at runtime, and that is exactly what b12x does in-kernel. So you can get W4A4 compute from the W4A16 checkpoint without re-quantizing anything.

Workload shape matters. Our traffic, 7 days of production logs:
- prompt median 2,787 tokens, p90 6,367;
- output median 101 tokens;
- **96% of all tokens are prefill.**

Prefill is compute-bound, which is where FP4 tensor cores pay off. If your workload is decode-heavy at small batch, expect less: Marlin W4A16 is competitive at low batch, and the Model Optimizer data shows W4A4 slightly *slower* than BF16 at batch 1 (0.87-0.88x).

## The three fixes

### 1. Pass the padded intermediate size to the wrapper (vLLM bug)

`--moe-backend flashinfer_b12x` crashes on the first forward:

```text
flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py  _pad_intermediate_to_tile -> mma_to_logical
RuntimeError: shape '[344064, 116]' is invalid for input of size 41287680
```

For non-gated activations, vLLM's `prepare_nvfp4_moe_layer_for_b12x` swizzles `w13_scale`, which rounds rows up to a multiple of 128 (1856 → 1920). It then pads `w13` rows and the `w2` / `w2_scale` columns to 1920 to match.

`FlashInferB12xExperts` still constructs `B12xMoEWrapper(intermediate_size=1856)`. FlashInfer then re-pads scale tensors that are already 120 scale-columns wide (41,287,680 = 128 x 2688 x 120). The fix constructs the wrapper with the width actually present in the prepared weights. The padded channels are zero, so it is numerically identical.

vLLM main's `flashinfer_b12x_moe.py` is byte-identical to 0.32.0, so the bug is still upstream. A minimal upstream PR is being prepared.

### 2. Share b12x workspaces across MoE layers (open upstream PR)

With fix 1, startup fails with CUDA OOM at any `--gpu-memory-utilization`, because the memory is allocated inside vLLM's profiling run before the KV budget applies. We measured one `B12xMoEWrapper` workspace at Nemotron's shapes:

| `max_num_tokens` | One layer | x 23 MoE layers |
|---|---|---|
| 16,384 | 530 MiB | 11.9 GiB |
| 8,192 | 417 MiB | 9.4 GiB |
| 4,096 | 361 MiB | 8.1 GiB |

vLLM builds one wrapper, with its own workspace, per MoE layer.

[vLLM PR #60069](https://github.com/vllm-project/vllm/pull/60069), open as of this writing, passes FlashInfer's `shared_static_workspace` / `shared_dynamic_workspace` / `shared_output`. Identically-shaped layers then share one set, which is FlashInfer's documented contract: MoE layers execute strictly sequentially.

We applied the same change, keyed on the *padded* intermediate size from fix 1, so differently-padded layers never alias. We saw no illegal-memory-access during CUDA-graph capture on SM120 (that PR's open caveat was reported on an RTX 5090).

### 3. Force FlashInfer's dynamic b12x path

With fixes 1 and 2 the model loads, then OOMs in the profiling forward. FlashInfer's *static* b12x path, used when routed rows ≤ 1024 (about 170 tokens x top-6), requires the intermediate dimension to be a multiple of 256 (two 128-wide tiles). It pads 1920 → 2048 by building and caching a **full padded copy of every layer's FP4 weights and scales**: about 756 MiB x 23 ≈ 17 GiB.

The dynamic path keeps native 128-wide geometry and copies nothing. So we force it:

```text
FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS=0
```

The cost is the static path's small-batch launch optimization. On a prefill-heavy workload that is not measurable. Fixing the 256-pad copy in FlashInfer (pad in place, or pad at prepare time in vLLM) is an open follow-up.

## Serving configuration

The same flags as the Marlin baseline. The only differences are `--moe-backend`, one env var, and the mounted file:

```text
vllm serve <checkpoint> \
  --moe-backend flashinfer_b12x \
  --gpu-memory-utilization 0.90 --max-model-len 65536 --max-num-seqs 64 \
  --max-num-batched-tokens 16384 --enable-chunked-prefill --enable-prefix-caching \
  --kv-cache-dtype fp8 --mamba-backend flashinfer --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5 \
  --reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking":false}'

env: FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS=0
     VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=<persistent volume>/flashinfer_autotune
mount: vllm-0.32.0-b12x.patch applied to
       vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py (read-only file mount)
```

The patch ([`vllm-0.32.0-b12x.patch`](vllm-0.32.0-b12x.patch)) is against vLLM 0.32.0 exactly. Rebase or drop it when you change the image.

Confirm in the startup log that you got the native path:

```text
Using 'FLASHINFER_B12X' NvFp4 MoE backend out of potential backends: [...]
flashinfer.jit: Compiling CuTe-DSL kernel b12x_moe_sm120a_cute_dsl/dynamic_e128_k2688_n1920_t6_...
```

If the log says `MARLIN`, or prints "Your GPU does not have native support for FP4 computation", the expert GEMMs are BF16.

Non-MoE NVFP4 linears still use `MarlinNvFp4LinearKernel`; only the MoE experts move to b12x. The experts are the bulk of the compute.

## Memory

vLLM's startup accounting, same flags, same card:

| | Marlin | b12x |
|---|---|---|
| Model weights | 17.82 GiB | 18.27 GiB (+0.45 GiB, zero padding to 1920) |
| Weights + non-torch | 19.3 GiB | 18.9 GiB |
| KV cache memory | 5.42 GiB | **5.79 GiB** |
| KV cache (FP8) | 1,356,442 tokens | **1,449,412 tokens** (22.1 full 65k requests) |

The KV cache is FP8 in both cases. b12x's smaller runtime footprint, once workspaces are shared, leaves more room for the same FP8 cache.

## Correctness

Activation quantization is where FP4 quality can slip, so we gated on our real task: replaying production call-classification prompts (2,500-6,500 tokens, JSON output, temperature 0) and comparing the predicted primary category.

On a fixed set of 40 prompts:

| Comparison | Same category |
|---|---|
| Marlin replica vs **itself** (re-run) | 36/39 |
| Marlin replica vs **another Marlin replica** (same model and kernel, different GPU) | 30/39 |
| Marlin vs **b12x** | **31/39** |

b12x disagrees with Marlin no more than two Marlin replicas disagree with each other: temperature-0 decoding on long prompts is not bit-deterministic across GPUs and batch shapes.

Also:
- 10/10 short known-answer prompts;
- 0 degenerate or garbage outputs;
- 40/40 valid JSON through the production gateway.

Free-form answers diverge from Marlin earlier than they did with MTP speculative decoding: median 9 identical opening words versus 23. They remain correct and coherent.

This is a task-level gate, not a benchmark. Run your own before you trust FP4 activations on a different task, especially past ~10K context, where community reports put W4A4 quality loss at 2-4x W4A16's.

## Throughput

Method:
- unique ~3,330-token prompts, unique from token 0 so prefix caching can't help;
- 100 output tokens (`ignore_eos`);
- concurrency 1 (5 sequential requests), 8 and 16;
- 3 runs each, same card class, no other traffic;
- prefill tok/s = total prompt tokens / time until the last request's first token.

| Concurrency | Metric | Marlin | **b12x** | Ratio |
|---|---|---|---|---|
| 1 | prefill tok/s | 15,479-15,646 | **21,562-21,897** | 1.39-1.41x |
| 1 | TTFT | 213-215 ms | **152-155 ms** | -28% |
| 8 | prefill tok/s | 17,332-17,582 | **24,481-25,454** | 1.39-1.47x |
| 8 | req/s | 3.36-3.41 | **3.99-4.04** | +18% |
| 16 | prefill tok/s | 17,466-17,859 | **25,460-25,871** | 1.43-1.48x |
| 16 | TTFT p50 | 1,878-1,923 ms | **1,302-1,319 ms** | -31% |
| 16 | req/s | 3.81-3.89 | **4.76-4.89** | +24% |

Marlin's run-to-run spread was under 2% on this probe.

The first concurrency-8 burst on a freshly started b12x engine measured 18,392 tok/s: first-use compile/tune of a new batch shape. Every later run matched the table. Persist the FlashInfer autotune and JIT caches across restarts to avoid paying that again.

Through the production gateway, the b12x replicas served real prompts at 426-702 ms average latency, against 1.1-3.0 s for the same model on GB10 (DGX Spark) replicas behind the same gateway.

## What we tried that did not help on this card

All measured on the same 32 GB RTX PRO 4500, same model:

| Attempt | Outcome |
|---|---|
| NVIDIA's H100 "max throughput" recipe (`--moe-backend humming --linear-backend humming`, `VLLM_HUMMING_MOE_GEMM_TYPE=indexed`, `--async-scheduling`) | Correct, but TTFT 540 ms vs 147 and no throughput gain. Hopper tuning doesn't transfer. |
| Native FP4 via `flashinfer_cutlass` | Rejected at load: `does not support quantization scheme ...xNone`. It needs calibrated activation scales, which a W4A16 checkpoint doesn't have. FlashInfer #2708 also reports a race in its FP4 GEMM under high concurrency on SM120. |
| MTP speculative decoding (`{"method":"mtp","num_speculative_tokens":3}`) | Works, with 82.5% acceptance and about 2x decode tok/s. But the BF16 MTP head plus draft working memory cut the KV cache to 364k tokens (from 1.36M), first-token time rose to about 530 ms, and decode is 4% of our tokens. Needs `--moe-backend` unset (the MTP head is unquantized, and Marlin rejects it) and `--gpu-memory-utilization` ≤ 0.85. |
| DSpark speculative decoding (vLLM recipe for RTX PRO 6000) | At 0.80 utilization only 1.5 GiB was left for KV on a 32 GB card (the recipe is sized for 96 GB). At 0.92 it started (94k KV tokens) but **OOM'd under 16-way load**. At 0.85 it can't fit a 65k request. Not viable on 32 GB at 65k context. |
| vLLM Rust frontend (`VLLM_USE_RUST_FRONTEND=1`) | It currently ignores `enable_auto_tool_choice` and `structured_outputs_config`, so tool calling and JSON-schema output silently stop working. Not for production. |
| A community SM120-tuned vLLM build, run with plain `vllm serve` | Slower than upstream 0.32.0 for this model: its tuning profiles target other models and larger cards. Run such images through their own launcher if you evaluate them. |

## Reproducing

1. Serve the checkpoint on SM120 with the Marlin baseline flags above, and record TTFT and prefill tok/s at your real prompt length and concurrency.
2. Apply [`vllm-0.32.0-b12x.patch`](vllm-0.32.0-b12x.patch) to vLLM 0.32.0's `flashinfer_b12x_moe.py`. A read-only file mount over the installed file works without rebuilding the image.
3. Set `--moe-backend flashinfer_b12x` and `FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS=0`. Keep every other flag.
4. Confirm `FLASHINFER_B12X` in the startup log.
5. Gate correctness on *your* task against the Marlin baseline. Include a replica-vs-replica comparison so you know your noise floor.
6. Measure the same probe 3x per variant. Report ranges, not single runs.

## Open follow-ups

- **Upstream the padded-intermediate fix** to vLLM. In progress.
- **[vLLM #60069](https://github.com/vllm-project/vllm/pull/60069)** (shared b12x workspaces): needed for any model with many MoE layers on a 32 GB card.
- **FlashInfer static b12x path:** pad in place or at prepare time instead of caching full padded weight copies, so `n % 256 != 0` models can use it.
- **W4A4 checkpoint via Model Optimizer** (calibrated activation scales, optionally quantization-aware distillation) for workloads where runtime activation quantization proves too lossy.
