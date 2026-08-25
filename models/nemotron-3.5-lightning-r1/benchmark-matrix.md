# Benchmark matrix — Darkstar Nemotron-3.5-Lightning

Throughput lanes: 4K / 16K / 48K prompt tokens, 512 max decode tokens, 7 runs each, 2 warmup,
CUDA graphs ON. Traffic-weighted winner: `0.6·4K + 0.3·16K + 0.1·48K` (production prompt-length mix).

## Throughput (tok/s, mean)

### Base-ModelOpt-NVFP4

| Method | 4K | 16K | 48K | weighted |
|---|---:|---:|---:|---:|
| **MTP7 (winner)** | 562.2 | 548.0 | 399.7 | **541.7** |
| MTP1 | 266.4 | 216.5 | 230.2 | 249.4 |
| MTP10 | 518.3 | 508.9 | 458.6 | 510.4 |
| MTP11 | 539.7 | 556.1 | 413.5 | 528.5 |
| MTP12 | 546.2 | 557.5 | 405.7 | 530.3 |
| DFlash | 535.8 | 518.2 | 469.5 | 523.9 |

### Abliterated-BF16

| Method | 4K | 16K | 48K | weighted |
|---|---:|---:|---:|---:|
| MTP1 | 203.9 | 196.5 | 173.3 | 198.6 |
| MTP2 | 278.1 | 286.7 | 273.1 | 280.2 |
| MTP3 | 362.3 | 338.9 | 303.8 | 349.4 |
| MTP4 | 411.4 | 375.1 | 363.3 | 395.7 |
| MTP5 | 415.0 | 403.9 | 376.2 | 407.8 |
| MTP6 | 438.3 | 398.4 | 316.0 | 414.1 |
| MTP7 | 427.2 | 438.8 | 406.4 | 428.6 |
| MTP8 | 491.0 | 424.9 | 397.0 | 461.8 |
| MTP9 | 466.7 | 443.5 | 409.4 | 454.0 |
| MTP10 | 456.1 | 461.9 | 378.9 | 450.1 |
| MTP11 | 485.9 | 493.9 | 390.3 | 478.7 |
| **MTP12 (winner)** | **507.6** | **512.6** | **433.4** | **501.7** |
| DFlash | 377.2 | 375.2 | 383.6 | 377.9 |

### Abliterated-ModelOpt-NVFP4

| Method | 4K | 16K | 48K | weighted |
|---|---:|---:|---:|---:|
| MTP1 | 251.1 | 246.4 | 225.9 | 247.2 |
| MTP2 | 379.9 | 354.0 | 329.1 | 367.0 |
| MTP3 | 428.3 | 393.2 | 332.4 | 408.2 |
| MTP4 | 470.6 | 409.9 | 370.7 | 442.4 |
| MTP5 | 525.3 | 481.9 | 464.5 | 506.2 |
| MTP6 | 576.7 | 519.7 | 457.7 | 547.7 |
| MTP7 | 568.8 | 535.0 | 500.2 | 551.8 |
| MTP8 | 548.2 | 437.6 | 455.7 | 505.8 |
| MTP9 | 555.6 | 520.1 | 397.4 | 529.1 |
| **MTP10 (winner)** | **571.2** | **546.3** | **480.7** | **554.7** |
| MTP11 | 483.4 | 401.7 | 470.7 | 457.6 |
| MTP12 | 579.8 | 476.3 | 480.4 | 538.8 |

## GPQA Diamond (accepted protocol — llm-inference-bench, thinking ON, temp 0)

| Product | Correct | Accuracy | Wilson 95% | Notes |
|---|---:|---:|---|---|
| Base-BF16 control | 136/198 | 68.7% | 61.9–74.7 | 11 unparseable, 7 TRUNC |
| **Abliterated-ModelOpt-NVFP4 (final servable, MTP10)** | **141/198** | **71.2%** | 64.5–77.1 | 1 unparseable, 2 TRUNC, 0 errors |
| NVIDIA published (BF16 / NVFP4) | — | 75.44 / 75.57 | — | official README |

Abliterated-NVFP4 plain-decode reference: 137/198 (69.2%). MTP10 (shipping config) scores 141/198
(71.2%) — spec decoding is semantics-preserving and the shipping config measured best. NVIDIA
numbers measured with their own serving stack (vLLM 0.26, FP8 KV, TP2); our stack differs
(dflash2 build, BF16 KV, single GPU).

## Behavior gates (abliteration safety)

| Product | Harmful compliance | Safe over-refusals | Errors |
|---|---:|---:|---:|
| Abliterated-BF16 | 200/200 | 0/83 | 0 |
| **Abliterated-ModelOpt-NVFP4** | **200/200** | **0/83** | **0** |

## Caveats

- GPQA is evaluated on the **final servable result only** (campaign policy); other products/configs are
  recorded as process evidence, not benchmarked.
- DFlash2/DSpark are not supported for nemotron_h (DFlash2 drafter is Qwen-vocab; no DSpark drafter
  in registry) — recorded as dead-ends.
- Throughput measured on a single RTX PRO 6000 Blackwell (SM120), vLLM cu130-nightly build,
  `VLLM_NVFP4_GEMM_BACKEND=cutlass` for NVFP4 cells.
