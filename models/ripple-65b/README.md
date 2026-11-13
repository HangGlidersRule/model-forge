# Ripple-65B Preview (#39)

Model Forge record for [`ricursiveAI/Ripple-65B-Preview`](https://huggingface.co/ricursiveAI/Ripple-65B-Preview), a
dense 65B chat model (Apache-2.0), quantized for one Blackwell GPU. Campaign issue:
[#39](https://github.com/HangGlidersRule/model-forge/issues/39) (plan, pins, amendments R39-A1 to R39-A11 and the
readout).

| Cell | Repository | Status |
|---|---|---|
| Upstream BF16 control | [`ricursiveAI/Ripple-65B-Preview`](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) @ `240d5966cbe00f0627c699e59ef58560b60303cf` | reference only; too large for the 96 GB GPU, used as a CPU teacher ([card](model-card/base-bf16.md)) |
| Base ModelOpt (W4A4 NVFP4 MLP, FP8 attention, W4A16 NVFP4 `lm_head`) | [`HangGlidersRule/Darkstar-Ripple-65B-Preview-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8`](https://huggingface.co/HangGlidersRule/Darkstar-Ripple-65B-Preview-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8) | selected ([card](model-card/base-nvfp4.md)) |
| Abliterated BF16, Abliterated ModelOpt | — | not planned in #39 |

The role word `Base` is the release contract's (an unedited cell, as opposed to `Abliterated`). The product quantizes
the vendor's instruction-tuned model, not the vendor's separate pretrain-only `Ripple-65B-Base-Preview`.

## Result

Three candidates were pre-registered; A was built first and passed every bar, so C and B were not built (fallback
rule). Reference: the vendor's own [`ricursiveAI/Ripple-65B-Preview-NVFP4`](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-NVFP4)
@ `99eea4477a001a41b86914e41feb4b2ed86105e5` ("the vendor NVFP4": W4A4 NVFP4 MLP, everything else BF16), measured in the
same window, same image and flags, one RTX PRO 6000 Blackwell at 450 W.

| Measure | A | Basis |
|---|---|---|
| GPQA Diamond | 138 and 139 of 198 (mean 138.5) against 142 and 139 of 198 (mean 140.5) | two runs each, 65,536-token cap, full denominator; bar: the vendor's mean − 11.88: PASS |
| Tool-call parse rate | 0.996 (vendor 0.998) | 500 items (BFCL v3 AST 400, synthetic terminal and file-edit 100); bar: − 2 points: PASS |
| Tool-call task success | 0.922 (vendor 0.920) | bar: 0.98 × the vendor's: PASS |
| Next-token KL to the BF16 teacher | 0.0371 (vendor 0.0290; 1.28x, 95% CI of the difference +0.0064 to +0.0099) | tracked, not gated |
| Decode, batch 1 | 37.11 / 36.85 tok/s (vendor 27.41 / 29.15): 1.31x | serve 1 / serve 2; the selection rule's speed win: PASS |
| Decode, batch 8 | 224.3 / 226.6 tok/s aggregate (vendor 182.3 / 181.9): 1.24x | |
| Time to first token, ~16k tokens | 3.36 / 3.37 s (vendor 4.08 / 4.09): 0.82x | batch 1 |
| With DFlash2, decode, batch 1 | 218.7 / 223.3 tok/s (vendor 180.5 / 179.2): 1.23x | mean acceptance length 7.17 / 7.20 (vendor 7.21 / 7.16) |
| Size | 43.05 GB (40.10 GiB) in 5 shards | vendor 57.21 GB; KV pool at fraction 0.92: 419,096 tokens (vendor 280,176) |
| Artifact | 240 NVFP4 W4A4, 400 FP8 and 1 W4A16 NVFP4 linear | fail-closed validator PASS on the export and again on the clean download |

A is eligible (every bar passed and the speed win) and ships. It has no quality win: its KL is higher because it also
quantizes attention to FP8, which the vendor NVFP4 leaves in BF16.

**BF16 reference.** The BF16 checkpoint (129.57 GB) does not fit the 96 GB GPU, so it was never served and has no GPQA
or proxy result here. The vendor reports GPQA Diamond 85.2 for BF16 (vendor-reported, their own harness), quoted only.
Closeness to BF16 is next-token KL and top-1 against a CPU teacher: the checkpoint's own modules in FP32, one layer at a
time, over 64 held-out prompts disjoint from the calibration rows ([`eval/bf16_teacher_cpu.py`](eval/bf16_teacher_cpu.py)).

**Runtime.** Every number comes from vLLM nightly `0.34.0rc1.dev206+gf5d93296c`
(`vllm/vllm-openai@sha256:902b49de0568ba5a5c951b0243bc06ef8541f46bc3eb80832961fefb22a19034`). Ripple support landed in
vLLM #65745 after the last stable release. The `v0.34.0rc1` git tag (`52aeb58`) contains it, but no Docker image has
been published for that tag yet ([`results/runtime-check.json`](results/runtime-check.json)). The release follows the
first vLLM release image that contains #65745 and is re-checked there.

## Results files

Every number above and on the cards is generated from these files by
[`eval/build_gate_summary.py`](eval/build_gate_summary.py) and checked by
[`eval/check_card_numbers.py`](eval/check_card_numbers.py).

- [`results/gate.json`](results/gate.json): the window's gate ([`eval/gate_ripple39.py`](eval/gate_ripple39.py)): every
  bar and the selection rule, computed from the raw run outputs.
- [`results/gate-summary.json`](results/gate-summary.json): the figures the record quotes, recomputed from the files below.
- `results/gpqa-<cell>-r<n>.json`: the four GPQA runs, aggregates only ([`eval/curate_gpqa_ripple.py`](eval/curate_gpqa_ripple.py)).
- `results/proxy-<cell>.json`: the tool-call proxy summaries; [`results/proxy-record.json`](results/proxy-record.json)
  pins its inputs.
- `results/speed/*.json`: the speed grid, eight serves ([`eval/speed_grid.py`](eval/speed_grid.py)).
- [`results/serve-runs.json`](results/serve-runs.json): per serve, the engine settings, memory, kernels and GPU power
  read from its log ([`eval/curate_window.py`](eval/curate_window.py)).
- [`results/build-A.json`](results/build-A.json), [`results/validation-A.json`](results/validation-A.json): the build and
  the export validator; `results/build-A-smoke.json`, `results/validation-A-smoke.json` and `results/smoke-A*.json`: the
  8-row smoke.
- [`results/checkpoint-sizes.json`](results/checkpoint-sizes.json): bytes per tensor category of A, the vendor NVFP4, BF16
  and the drafter, from the safetensors headers ([`eval/checkpoint_sizes.py`](eval/checkpoint_sizes.py)).
- [`results/calibration-record.json`](results/calibration-record.json): the calibration and held-out row lists.
- [`results/staging-receipt.json`](results/staging-receipt.json): the private upload, its clean download and the boot
  smoke of that download ([`eval/curate_clean_boot.py`](eval/curate_clean_boot.py));
  `results/clean-boot-<mode>.json`: the boot smoke's checks, plain and with DFlash2.
- [`results/publication-readiness-ledger.json`](results/publication-readiness-ledger.json): the release ledger
  (contract 3.1.0), written by [`eval/build_ledger.py`](eval/build_ledger.py).

## Candidates

| Letter | Layout | Recipe (stage 1) | sha256 |
|---|---|---|---|
| A | MLP W4A4 NVFP4, attention q/k/v/o/g FP8 W8A8, `lm_head` W4A16 NVFP4 (selected) | `a_nvfp4_w4a4_mlp-fp8_attn-kv_bf16_local_hessian_layerwise.yaml` | `624185c8…` |
| C | MLP W4A4 NVFP4, attention BF16, `lm_head` W4A16 NVFP4 (not built) | `c_nvfp4_w4a4_mlp-bf16_attn-kv_bf16_local_hessian_layerwise.yaml` | `ae77800f…` |
| B | MLP W4A16 NVFP4, attention FP8, `lm_head` W4A16 NVFP4 (not built) | `b_w4a16_nvfp4_mlp-fp8_attn-kv_bf16_local_hessian_layerwise.yaml` | `ceafd52f…` |

Stage 2 for every candidate: `lm_head_w4a16_nvfp4_mse.yaml` (`123e4e99…`), data-free. Recipes:
[`../../configs/modelopt/recipes/ripple-65b/`](../../configs/modelopt/recipes/ripple-65b/). Schema-2 recipe of the
product: [`../../recipes/ripple-65b/darkstar-ripple-65b-preview-base-modelopt-w4a4-nvfp4-mixed-fp8.yaml`](../../recipes/ripple-65b/darkstar-ripple-65b-preview-base-modelopt-w4a4-nvfp4-mixed-fp8.yaml).
Enabled quantizers after both stages: A 1,281, B 1,041, C 481.

## Frozen inputs

- Calibration (`calibration/build_ripple_calib.py`): 512 rows (128 cnn_dailymail, 128 Nemotron IF-Chat-v2
  reasoning-on, 192 SWE-smith tool, 64 hermes-function-calling), 927,103 tokens (p50 1,605, p90 3,540, max 4,096); file
  `403730e4…`, row-list digest `4c8c67af…`; 64 held-out KL prompts `c258051a…`. Record:
  [`results/calibration-record.json`](results/calibration-record.json).
- Tool-call proxy (`eval/toolcall_proxy.py`, `eval/tasks_synthetic.py`): BFCL v3 AST, first 100 by id of four
  categories (`proxy_bfcl.jsonl` `0affe6d9…`), and 100 seeded terminal/file-edit tasks (`proxy_synthetic.jsonl`
  `110ed227…`). Record: [`results/proxy-record.json`](results/proxy-record.json).

## Files

- `scripts/quantize_ripple.py`: the build (disk-offloaded BF16 load, known-answer gate, layerwise stage 1 with resume,
  data-free `lm_head` stage 2 packed inside the streaming export, quantizer census, `MIXED_PRECISION` quant config).
- `eval/validate_export.py` and `eval/validate_selftest.py`: the fail-closed export validator and its planted-fault
  self-test; `eval/vllm_offline_probe.py`: the runtime resolution check with the config classes of vLLM itself.
- `eval/bf16_teacher_cpu.py`: the BF16 teacher (FP32 on CPU, layer by layer) for KL and top-1.
- `eval/serve_checks.py`, `eval/speed_grid.py`, `eval/gate_ripple39.py`: serve smoke and KL collection, the speed grid,
  and the bars and selection rule.
- `eval/curate_gpqa_ripple.py`, `eval/curate_window.py`, `eval/curate_clean_boot.py`, `eval/checkpoint_sizes.py`,
  `eval/runtime_check.py`, `eval/build_gate_summary.py`, `eval/check_card_numbers.py`, `eval/build_ledger.py`: the
  record's curation, evidence, card-number and ledger builders.
- `eval/ripple_layout.py`: candidate layouts and pins shared by every script, tested in
  `tests/test_ripple_65b_kit.py`.
- The window ran these scripts at kit commit `ca79435`. Since then `eval/tasks_synthetic.py` and
  `eval/toolcall_proxy.py` changed only in how a few literal paths are written (joined from parts, so the public
  export's path rewrite leaves them as they are); the generated task set still hashes to its pin `110ed227…`.
- `scripts/window_ripple39.sh`, `scripts/stage_ripple39.sh`, `scripts/clean_boot_smoke.sh` and `rehearsal/`: operator
  launchers for the GPU host and their offline rehearsal; not published. The measured window served every cell as
  `ripple`; the committed launcher uses the ledger alias `darkstar-ripple65b-base-nvfp4`, which changes no result.

## Window order

Preflight (before the park) → park through the lane tool, `HOLD-PARKED-ripple39` written → A smoke (8 rows: build,
validate, offline vLLM resolution, serve with the posted flags, `modelopt_mixed` and the FP8 / NVFP4 W4A4 / W4A16
`lm_head` kernel lines, known answers, one parsed tool call, DFlash2 boot) → A full build and validation → quality
serves, vendor NVFP4 then A (KL collect of the 64 held-out prompts as token ids, tool-call proxy, GPQA twice) → bars
(`eval/gate_ripple39.py --bars-only`) → C, then B, only if A fails a bar and the build, quality serves and speed grid
still fit the window budget → speed grid (vendor, X, vendor, X; each without and with DFlash2) → the gate JSON → EXIT
trap: containers removed, hold deleted, unpark through the lane tool with state polling, `logs/window.done`.
Every stage skips work already on disk and validated; `R39_RESUME=1` adopts the campaign's own park after a host
restart. `R39_PREFLIGHT_ONLY=1` runs every preflight check on the host and stops before the park.


## Publication

The weights sit in a private staging repository until this record merges and the tag `darkstar-ripple-65b-v1.0.0`
exists. Its clean download was verified file by file against the export, and that download booted with the card's
serve command on the measured image, plain and with DFlash2: in both, the known answers, a parsed tool call and a chat
request passed with no fatal log line ([`results/staging-receipt.json`](results/staging-receipt.json)). The public
repository is downloaded again after the flip.
