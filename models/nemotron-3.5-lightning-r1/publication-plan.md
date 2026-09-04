# Publication plan — Darkstar Nemotron-3.5-Lightning

Follows the Qwen38 release contract (see `models/qwen3.8-27b-r3/publication-plan.md`).

## Targets

| Repo | Description |
|---|---|
| `HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16` | BF16 abliterated (62G, 14 shards) |
| `HangGlidersRule/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-ModelOpt-W4A16-NVFP4` | NVFP4 abliterated (22G, 3 shards) |
| GitHub `HangGlidersRule/model-forge` | recipes, cards, harness, results (this repo) |

License: OpenMDW-1.1 (upstream NVIDIA). Cards must link `https://openmdw.ai/license/1-1/` and retain
NVIDIA attribution lines.

## Gates (all must pass before publish)

1. Build/artifacts on mcprue: BF16 (62G) + NVFP4 (22G, 3 shards) present, quant contract verified.
2. Behavior gate: **200/200 harmful compliance, 0/83 safe over-refusals, 0 errors** — ✅ BF16 and NVFP4.
3. GPQA (accepted protocol) on the **final servable result** (Abliterated-NVFP4 @ MTP10) — running.
4. Speed/tuning: MTP winners locked (Base-NVFP4 MTP7 541.7, Ablit-BF16 MTP12 501.7,
   Ablit-NVFP4 MTP10 554.7) — ✅.
5. Cards + README + manifest in repo — in progress.
6. Hash-verify every weight shard (sha256 manifest) before upload; verify links after.

## Steps

1. Cut immutable GitHub release tag `darkstar-nemotron-3.5-lightning-v1.0.0` on this repo.
2. Push BF16 checkpoint to HF (`huggingface-cli upload`), verify download + boot + smoke.
3. Push NVFP4 checkpoint to HF, same verification.
4. Final card verification (README on each HF repo renders; links live).
5. Announce.

## Sequencing

- Darkstar (Qwen) stays down until **both** Lightning and Nano-Omni builds complete (campaign policy).
- Lightning publication is the prerequisite for starting Nano-Omni GPU work.
