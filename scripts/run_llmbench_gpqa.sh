#!/usr/bin/env bash
# GPQA Diamond (and GSM8K / MMLU-Pro) accuracy runs via llm-inference-bench.
#
# llm-inference-bench (https://github.com/local-inference-lab/llm-inference-bench)
# is the TUI benchmark harness popularized for DGX-Spark/sglang runs: Rich
# dashboard, GPU panel, pass/run/wait lanes, deterministic letter shuffle,
# exact-match scoring, Wilson CIs. Its gpqa-diamond profile is our accepted
# GPQA measurement for new claims (thinking ON via the chat template, temp 0).
#
# Requires: repo cloned somewhere (default ${PUBLIC_WORKSPACE}),
# python3.13 venv with httpx rich psutil.
#
# Usage:
#   scripts/run_llmbench_gpqa.sh <base_url> <served_model> <out_json> [profile] [max_tokens]
set -euo pipefail
BENCH_HOME="${BENCH_HOME:?Set BENCH_HOME to the public artifact path}"
BASE_URL="${1:?base_url e.g. http://127.0.0.1:8109}"
MODEL="${2:-served-name-placeholder}"
OUT="${3:?output json path}"
PROFILE="${4:-gpqa-diamond}"
MAX_TOKENS="${5:-131072}"

mkdir -p "$(dirname "$OUT")"
cd "$BENCH_HOME"
"$BENCH_HOME/.venv/bin/python" llm_decode_bench.py \
  --host "$BASE_URL" \
  --model "${MODEL:-served-name-placeholder}" \
  --test-profile "$PROFILE" \
  --max-tokens "$MAX_TOKENS" \
  --display-mode plain
cp benchmark_results.json "$OUT"
echo "WROTE $OUT"
