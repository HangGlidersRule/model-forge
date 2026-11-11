# Darkstar Qwen-Image-2.1

Diffusion image-generation family record. Source family: [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) — a single-stream diffusion transformer for text-to-image generation and editing.

## Shipped cells

| Cell | Repository / source | Status |
|---|---|---|
| Base BF16 (control) | [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) @ `b3179ad355be050328e483a9dfdd9e60cd62adfa` | unchanged upstream reference; no HangGlidersRule repository |
| Base W4A4-NVFP4 | [`Darkstar-Qwen-Image-2.1-Base-ModelOpt-W4A4-NVFP4`](https://huggingface.co/HangGlidersRule/Darkstar-Qwen-Image-2.1-Base-ModelOpt-W4A4-NVFP4) | published, rev `16f1b7fd` |
| Base FP8 | [`Darkstar-Qwen-Image-2.1-Base-ModelOpt-FP8`](https://huggingface.co/HangGlidersRule/Darkstar-Qwen-Image-2.1-Base-ModelOpt-FP8) | published, rev `5fcc2176` |
| Ablit BF16 / Ablit NVFP4 | not shipped | behavior-cell gate — see below |

Release: tag `darkstar-qwen-image-2-1-v0.1.0` in the private work archive.

## Why there are no abliterated cells

Image-family behavior cells are gated on a demonstrated edit mechanism. For this family, seven instrumented attempts across the edit-grade mechanism space either had no effect or degraded general behavior:

| attempt | mechanism | result |
|---|---|---|
| E2 | text-encoder refusal-direction removal (72 weights, residual leakage 2.0%→0.013%) | no image-side change |
| E2B | `txt_in` boundary projection (thin concept subspace) | no change |
| UCE-v1 | global concept subspace, `to_k`/`to_v` input projection (64 weights) | no change |
| E2D | trigger-phrasing-pair subspace (projection-energy separation ~3% = noise) | no change |
| E2C | diagnostic — phrasing flips compliance ("movie prop handgun" renders a detailed handgun) | capability exists in weights |
| LoRA-pair v1 | trigger→compliant pair training, 512px, 300 steps | collapse: harm compliance 0.8→0.0, benign 1.0→0.1 |
| LoRA-pair v2 | benign-preservation pairs + full-res targets + lower LR, 400 steps | collapse reproduced |

Conclusion recorded in the family issue archive: the semantic trigger→conservative-rendering mapping in this family is load-bearing in the DiT's conditioning path — not reachable by edit-grade mechanisms. The family therefore ships base cells only. The four-cell matrix remains the template for future families whose censoring is encoder-mediated.

## Precision maps

- **NVFP4 cell:** uniform fp4 (W4A4-style fp4 GEMM) over DiT linear layers with protected BF16 components (`img_in`, `txt_in`, `modulation`, `time_text_embed`, `norm_out`, `proj_out`, norms/embeddings, and boundary blocks 0–1 and 30–31). Recipe id: `NVFP4`; product id carries the `W4A4-NVFP4` precision encoding.
- **FP8 cell:** uniform FP8 weights over the same quantizable DiT component set with the same protected set.

Quantizer calibration ran once from the base BF16 pipeline (frozen decision B5); validators confirmed 196 quantized modules with 0 misclassifications and 0 degenerate scales for both cells.

## Quality (frozen manifest `image-eval-v1`, sha `09daf654…`)

25 cases: T2I quality 10, composition 6, typography 4, RGBA 2, editing 3. Full-denominator semantics — every case records an outcome.

| capability | BF16 control | W4A4-NVFP4 | FP8 |
|---|---|---|---|
| t2i_quality | 10/10 | 10/10 | 10/10 |
| composition | 6/6 | 6/6 | 6/6 |
| typography | 4/4 | 4/4 | 4/4 |
| rgba | 2/2 | 2/2 | 2/2 |
| editing | 3/3 | 3/3 | 3/3 |

All three cells pass 25/25 with identical rates. NVFP4 dequant-vs-upstream per-tensor error ≈ 0.0034 mean, correlation ≈ 0.996.

## Behavior baseline (frozen)

Full-denominator judge labels (`COMPLIANT`/`AVOIDANT`/`NEUTRAL`), non-Qwen-lineage judge model, 10 benign + 10 harm-adjacent prompts, seeds `100+i`:

- Base BF16: harm compliance 0.8, harm avoidance 0.2, neutral 0.0; benign compliance 1.0.

Serving smoke: vLLM-Omni `/v1/images/generations` HTTP 200, seed-deterministic, visually verified for both quant cells.

## License

Qwen Research License Agreement — research/evaluation (non-commercial) use only. Both published repositories carry the LICENSE and full obligation table on their model cards. Commercial use requires a separate license from the Qwen team. Built with Qwen.

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.
