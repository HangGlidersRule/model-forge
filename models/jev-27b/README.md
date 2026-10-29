# Darkstar JEV-27B

Decision-model family record. Source: [`autotrust/JEV-27B`](https://huggingface.co/autotrust/JEV-27B), a System 1 decision model
built as a LoRA (plus a decision head in `lm_head`) on [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). It returns
calibrated probabilities for typed decisions (`noul` yes/no, `choice` with up to 16 native options, `score` 0–5) from one forward
pass with a one-token readout. Proposed and tracked in [issue #15](https://github.com/HangGlidersRule/model-forge/issues/15).

## Shipped cells

| Cell | Repository / source | Status |
|---|---|---|
| BF16 reference (control) | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` + [`autotrust/JEV-27B`](https://huggingface.co/autotrust/JEV-27B) `adapter_vllm` @ `14b425269ed24452e0a5ad5a8f3e51d139647a88` | unchanged upstream reference; no HangGlidersRule repository |
| Merged W4A4-NVFP4 + FP8 (decisions) | [`Darkstar-JEV-27B-Decision-ModelOpt-W4A4-NVFP4-Mixed-FP8`](https://huggingface.co/HangGlidersRule/Darkstar-JEV-27B-Decision-ModelOpt-W4A4-NVFP4-Mixed-FP8) | published |
| Merged all-W4A4-NVFP4, QAD scales (decisions, speed variant) | [`Darkstar-JEV-27B-Decision-QAD-ModelOpt-W4A4-NVFP4`](https://huggingface.co/HangGlidersRule/Darkstar-JEV-27B-Decision-QAD-ModelOpt-W4A4-NVFP4) | published by owner decision; fails the frozen bar on a teacher-KL tie ([QAD](#qad-speed-variant-issue-20)) |
| Abliterated cells | — | not applicable: a decision scorer is not a refusal-edit target |

Releases: tags [`darkstar-jev-27b-v1.0.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-jev-27b-v1.0.0)
(mixed cell) and [`darkstar-jev-27b-v1.1.0`](https://github.com/HangGlidersRule/model-forge/releases/tag/darkstar-jev-27b-v1.1.0) (adds the QAD cell).
Family collection: [Darkstar JEV-27B](https://huggingface.co/collections/HangGlidersRule/darkstar-jev-27b-6ae12c8cd6eba44d1ab99a16).

## Why merge

JEV-27B ships unmerged on purpose: with the backbone untouched, one server answers plain generation (System 2, the base) and
decisions (System 1, the LoRA). This family serves decisions only. Folding the LoRA in removes the runtime LoRA path (a BF16
side computation on every layer, plus runtime kernel compiles), so every linear runs on quantized kernels. The cost is plain
text generation quality, which is neither measured nor claimed here: merging cost a sibling JEV model 9 HumanEval points.

## Documents

- [`model-card/decision-nvfp4.md`](model-card/decision-nvfp4.md): source card for the published mixed checkpoint
- [`model-card/decision-qad-nvfp4.md`](model-card/decision-qad-nvfp4.md): source card for the published QAD checkpoint
- [`model-card/base-bf16.md`](model-card/base-bf16.md): the unmerged BF16 + LoRA reference
- [`results/decision-gate-summary.json`](results/decision-gate-summary.json): all five configurations, the noise floor, latency,
  KV cache, both pass bars and the decision
- [`results/artifact-validation.json`](results/artifact-validation.json), [`results/merge-report.json`](results/merge-report.json),
  [`results/publication-readiness-ledger.json`](results/publication-readiness-ledger.json)
- [`scripts/`](scripts/): the LoRA merge (`merge_lora.py`) and its check against PEFT `merge_and_unload` on a reduced
  model (max abs diff 0.0; [`results/merge-peft-check.json`](results/merge-peft-check.json))
- [`calibration/`](calibration/): the deterministic 512-sample calibration-set builder
- [`eval/`](eval/): the `/v1/decide` evaluator, latency probe, pass-bar script, artifact validator, and the two-line
  `serve_decide.py` patch for merged checkpoints

## Recipes

- [`../../recipes/jev-27b/darkstar-jev-27b-decision-modelopt-w4a4-nvfp4-mixed-fp8.yaml`](../../recipes/jev-27b/darkstar-jev-27b-decision-modelopt-w4a4-nvfp4-mixed-fp8.yaml):
  schema-2 build recipe for the published mixed cell
- [`../../recipes/jev-27b/darkstar-jev-27b-decision-qad-modelopt-w4a4-nvfp4.yaml`](../../recipes/jev-27b/darkstar-jev-27b-decision-qad-modelopt-w4a4-nvfp4.yaml):
  schema-2 build recipe for the published QAD cell (PTQ start point plus the QAD stage)
- [`../../configs/modelopt/recipes/jev-27b/`](../../configs/modelopt/recipes/jev-27b/): the two ModelOpt operator recipes compared
  (A: NVIDIA's Qwen3.8-27B recipe with `lm_head` BF16; B: everything NVFP4 W4A4 with `lm_head` BF16)
- [`../../configs/modelopt/pins/modelopt-0.48.0rc1.json`](../../configs/modelopt/pins/modelopt-0.48.0rc1.json): the ModelOpt pin

## Candidate selection

Both candidates quantize the same merged BF16 checkpoint with ModelOpt `0.48.0rc1`: Local-Hessian NVFP4 weight scales with the
FP8-scale sweep, layerwise calibration, and 512 samples (256 JEV decision prompts, 256 NVIDIA chat samples). Each was scored against the
unmerged BF16 + LoRA reference (served twice for a noise floor) and the deployed lane (`nvidia/Qwen3.8-27B-NVFP4` + remapped
LoRA) on all 29,955 rows of the JEV-27B card's `test_set_30k`.

| Configuration | Teacher KL | Top-1 | Rows/s (c=16) | p50 c=1, ~3.8k tok | Weights |
|---|---|---|---|---|---|
| BF16 + LoRA reference | 0.0186 | 0.9270 | 45.3 | — | 50.7 GiB |
| Deployed: NVIDIA NVFP4 + LoRA | 0.0207 | 0.9200 | 50.1 | 365 ms | 20.4 GiB |
| **A (published): NVFP4 MLP + FP8 attention** | **0.0203** | **0.9219** | **69.5** | **270 ms** | 21.5 GiB |
| B: everything NVFP4 W4A4 | 0.0220 | 0.9189 | 98.5 | 219 ms | 18.7 GiB |

The first pass bar, written before any run, required each candidate to be indistinguishable from the BF16 reference. No 4-bit
build can meet it on 29,955 rows: the deployed lane misses it too (teacher KL +0.0021, CI +0.0020 to +0.0023). It was
replaced by a non-inferiority bar against the deployed lane. That bar was frozen as a script (`eval/gate_noninferiority.py`,
SHA-256 `75fcb814…`) before candidate B was evaluated, and requires all of:
- teacher KL no worse than the deployed lane;
- top-1 no more than 0.5 pt lower;
- higher rows/s and lower single-request p50 latency.

The frozen script ships byte-identical to the file that judged candidate B, so its SHA-256 still matches the one quoted above; its docstring keeps its working filename (`gate_i15_v2.py`).

A passes (teacher KL −0.00039 vs the deployed lane, CI −0.00054 to −0.00024). B fails it (+0.00133, CI +0.00114 to +0.00152)
despite being about 2× the deployed lane's throughput.

## QAD speed variant (issue #20)

Issue #20 asked whether training could recover B's quality while keeping its all-FP4 speed. Run 2 trained only B's
per-block NVFP4 weight scales with ModelOpt quantization-aware distillation:
- **Objective:** full-vocabulary KL from the merged BF16 model to the NVFP4 student at every prompt position. The teacher
  is the same weights with quantization bypassed.
- **Data:** the corpus `train` split, which has no overlap with `test_set_30k`.
- **Schedule:** 596 steps, ~9.8M tokens, lr 1e-4. Checkpoint chosen by validation teacher KL.
- **Pre-registration:** the run was pre-registered on the issue, and every change to it was recorded there as an
  amendment before the results it could affect.

| Configuration (29,955 rows) | Teacher KL | Top-1 | Agreement with BF16 | `ood` teacher KL | p50 c=1 short / long (600 W) |
|---|---|---|---|---|---|
| Deployed: NVIDIA NVFP4 + LoRA | 0.0207 | 0.9200 | 0.943 | — | — |
| A: NVFP4 MLP + FP8 attention | 0.0203 | 0.9219 | 0.940 | 0.1019 | 94 / 250 ms |
| B: everything NVFP4 W4A4 (PTQ) | 0.0220 | 0.9189 | 0.936 | — | 77 / 209 ms |
| **QAD: B with trained scales** | **0.0207** | **0.9205** | **0.946** | **0.1192** | **64 / 210 ms** |

**Result against the frozen bar.**
- QAD removed about 99% of B's teacher-KL gap to the deployed lane: delta +0.00001, 95% CI −0.00018 to +0.00019.
- That is a tie, and the frozen bar requires the CI's upper bound to be at or below zero. It fails.
- Out of distribution its teacher KL is 17% above A's (CI +0.0123 to +0.0227 absolute), with top-1 within 0.5 pt.

**Why it ships anyway.** Devin shipped it alongside A by owner decision: 15–30% lower single-request latency for 2.0%
higher test-set teacher KL. A remains the closer match to the teacher.

Speed numbers come from back-to-back runs in one window per power limit. The gate window's own speed reading for the QAD
build was a measurement fault, identified by that re-measurement. Evidence:
[`results/qad-gate-summary.json`](results/qad-gate-summary.json) and
[`results/qad-artifact-validation.json`](results/qad-artifact-validation.json). Scripts:
[`scripts/qad_train.py`](scripts/qad_train.py) (byte-identical to the run), [`eval/verify_export_scales.py`](eval/verify_export_scales.py)
and [`eval/build_qad_summary.py`](eval/build_qad_summary.py).

## Gotchas

- **The adapter's repo id.** Issue #15 named `autotrust/JEV-27B-decision`, which does not exist: `jev-decision` is the vLLM
  LoRA module name. The adapter lives in `autotrust/JEV-27B` under `adapter_vllm/`, and that repo's revision equals the
  issue's pinned SHA.
- **Remap for multimodal bases.** The published adapter keys `model.layers.N`; `nvidia/Qwen3.8-27B-NVFP4` and the Qwen BF16
  base are multimodal (`model.language_model.layers.N`). Without a remap, vLLM silently skips 800 of the 802 LoRA tensors.
- **`lm_head` carries the decision head.** NVIDIA's recipe quantizes `lm_head` to NVFP4; here it stays BF16, because after the
  merge the 24-token decision readout lives in it.
- **Fused FP4 projections.** vLLM fuses `in_proj_qkv` + `in_proj_z` (Gated DeltaNet), q/k/v and gate/up into single GEMMs and takes the max of the
  shards' scales. ModelOpt aligns the scales of linears that share an input, including the DeltaNet pair; the validator checks it.
- **Serving shim.** `serve_decide.py` expects a LoRA module. With `JEV_DECIDE_MERGED=1` and `JEV_DECIDE_HEAD_DIR` it reads the
  decision head from the model directory and sends readouts to the base model.

## License

Apache-2.0, inherited from `Qwen/Qwen3.8-27B` and `autotrust/JEV-27B`; the published repository ships the license text.
