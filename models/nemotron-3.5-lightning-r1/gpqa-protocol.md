# GPQA protocol — Darkstar Nemotron-3.5-Lightning

Reference implementation: [`docs/gpqa-abliteration-protocol.md`](../../docs/gpqa-abliteration-protocol.md) (normative).

## Dataset

- **GPQA Diamond**, 198 graduate-level "Google-proof" science questions (biology, chemistry, physics),
  4 options each. Fetched from the official password-protected zip on first use; cached locally;
  never stored in repo.

## Accepted evaluation (campaign policy)

1. **Final servable result only.** GPQA is run on the *shipping* artifact in its *shipping* serving
   config. For this campaign: **Abliterated-ModelOpt-NVFP4 served with MTP10**.
2. Tool: **llm-inference-bench** (github.com/local-inference-lab/llm-inference-bench), profile
   `gpqa-diamond`: chat template + **thinking ON** (completions run 16K–58K tokens), temperature 0,
   deterministic per-item letter shuffle, exact-match scoring, Wilson CI.
3. Serving: vLLM with `--reasoning-parser nemotron_v3` (the flag NVIDIA's published recipe depends
   on), `--max-tokens 65536` (profile default 131072 exceeds server max-model-len 131072 and would
   be rejected), `--gpu-memory-utilization 0.92` BF16 / `0.40` NVFP4 nospec eval configs.
4. Runner: `scripts/nemotron-3.5_lightning/gpqa_round.sh <product> <decoder> <compose> <model>`
   (deterministic round; orchestrator `run_gpqa_matrix.sh`).

## Why not raw lm-eval

- `gpqa_diamond_zeroshot` (logprob MC) and `gpqa_diamond_cot_zeroshot` (greedy CoT) are useful
  arithmetic but **under-read Nemotron family** because raw prompts omit chat template/thinking.
- The lm-eval `local-completions` API adapter corrupts MC logprob scoring (leading-space token
  mismatch) → 27.3% garbage measured on Abliterated-NVFP4. Dead-end; recorded as
  `gpqa-diamond-abliterated-nvfp4.DEADEND-lmeval-adapter.json`.
- Legacy `model_forge.gpqa.harness` (`\boxed{}` prompt + temp 1.0) under-reads Lightning →
  parse-error-dominated (old 76/198 = 38.4%). Deprecated for claims.

## Reference/control numbers

| Config | Correct | Accuracy | Purpose |
|---|---:|---:|---|
| Base-BF16 plain-decode llmbench | 136/198 | 68.7% | abliteration-cost control |
| Abliterated-NVFP4 plain-decode llmbench | 137/198 | 69.2% | reference for MTP10 run |
| NVIDIA published BF16 / NVFP4 | — | 75.44 / 75.57 | official README (their serving stack) |
