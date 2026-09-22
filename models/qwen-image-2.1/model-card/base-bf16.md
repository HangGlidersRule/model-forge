# Qwen/Qwen-Image-2.1 upstream BF16 control

**Role:** image-base-bf16 · **Behavior:** base · **Lifecycle:** reference-only
**Source:** Qwen/Qwen-Image-2.1 @ `b3179ad355be050328e483a9dfdd9e60cd62adfa`

HangGlidersRule does **not** own or republish these weights. This is the unchanged upstream BF16 control that the Darkstar quantized cells are built from and evaluated against.

## License

Qwen-Image-2.1 is distributed under the **Qwen Research License Agreement** (September 20, 2026):

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.

Section 1(i) restricts use to **research or evaluation purposes only** (non-commercial). Sections 3/4a/4b/4c/5a/5c obligations are enumerated in `models/qwen-image-2.1/FAMILY-DECISION.md`. Commercial use requires a separate license from the Qwen team.

## Control evaluation (frozen)

Manifest `image-eval-v1` (sha `09daf654…`): **25/25 pass** — t2i 10/10, composition 6/6, typography 4/4, rgba 2/2, editing 3/3. Behavior baseline (frozen for the family): harm suite 0.8 compliant / 0.2 avoidance / 0.0 judge-refusal; benign 1.0. Performance: 4.73 s/img mean (20 steps, 1024², RTX PRO 6000).

Serving profile: `darkstar-qwenimage21-base-bf16` (vLLM-Omni; `--omni`). Seed determinism verified byte-identical.

**Non-commercial. Research and evaluation use only. Built with Qwen.**
