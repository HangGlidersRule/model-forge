#!/usr/bin/env bash
# model-forge GPQA round runner (Nemotron-3.5 Light-Lightning products).
#
# POLICY (Devin): GPQA the FINAL SERVABLE RESULT ONLY. Everything else is
# recorded as process evidence, NOT benchmarked. Which product+config is the
# final servable result is decided per campaign (Lightning: Abliterated-NVFP4
# served with its tuned MTP10 config) — this runner targets exactly that.
#
# Deterministic, idempotent, evidence-producing — llm-inference-bench
# gpqa-diamond profile (chat template + thinking ON, temp 0, letter shuffle,
# exact-match, Wilson CI).
#
# Usage:
#   bash gpqa_round.sh <product> <decoder> <compose-file> <served-model>
#   PRODUCT: the final servable product name (e.g. ablit-nvfp4)
#   DECODER: the serving config to validate (e.g. mtp10)
#
# Requires: llm-inference-bench clone + python3.13 venv (httpx rich psutil),
# SSH fleet key ~/.ssh/id_ed25519_aihost, compose file exposing port 8109.
#
# Output: ~/vllm-bench-results/gpqa/<product>-<decoder>-llmbench.json
set -euo pipefail

HOST="security@hangglidersrule.com"
PRODUCT="${1:?product required}"
DECODER="${2:?decoder required}"
COMPOSE="${3:?compose file required}"
MODEL="${4:-served-name-placeholder}"
OUT="${PUBLIC_WORKSPACE}{PRODUCT}-${DECODER}-llmbench.json"
mkdir -p "$(dirname "$OUT")"

echo "== [$PRODUCT/$DECODER] compose: $COMPOSE (final servable result) =="

# ---- bring up only the round's container (stop everything else first) ----
ssh -o ConnectTimeout=10 -i ~/.ssh/id_ed25519_aihost "$HOST" \
  "wsl -d Ubuntu -- bash -lc 'cd ${PUBLIC_ARTIFACT_PATH} && docker ps -q | xargs -r docker stop >/dev/null 2>&1 || true; sleep 2; docker compose -f $COMPOSE up -d 2>&1 | tail -1'"

# ---- wait for health (max 10 min) ----
echo "== waiting for health =="
h=""
for _ in $(seq 1 40); do
  h=$(ssh -o ConnectTimeout=10 -i ~/.ssh/id_ed25519_aihost "$HOST" \
      "wsl -d Ubuntu -- bash -lc 'curl -s -o /dev/null -w %{http_code} -m 5 http://127.0.0.1:8109/health'" 2>/dev/null || true)
  [ "$h" = "200" ] && { echo "healthy"; break; }
  sleep 15
done
if [ "$h" != "200" ]; then
  echo "FAILED: server never healthy" >&2
  exit 1
fi

# ---- run the benchmark ----
echo "== running gpqa-diamond ($PRODUCT/$DECODER) =="
cd ${PUBLIC_WORKSPACE}
.venv/bin/python llm_decode_bench.py --host http://127.0.0.1:8109 \
  --model "${MODEL:-served-name-placeholder}" --port 8109 --test-profile gpqa-diamond \
  --max-tokens 65536 --display-mode plain > "/tmp/gpqa-${PRODUCT}-${DECODER}.log" 2>&1
cp benchmark_results.json "$OUT"
echo "== DONE $PRODUCT/$DECODER -> $OUT =="
