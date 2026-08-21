#!/usr/bin/env bash
set -euo pipefail
# Checkpoint the currently running vLLM service before any mutation.
# Creates an append-only timestamped snapshot with compose state, config,
# GPU info, model response, and SHA manifest.

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the public artifact path}"
COMPOSE_FILE="${COMPOSE_FILE:?Set COMPOSE_FILE to the public artifact path}"
SERVICE_NAME="${SERVICE_NAME:-vllm}"
DRY_RUN="${DRY_RUN:-1}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
SNAP="${CHECKPOINT_ROOT}/${TS}"

log() { echo "[checkpoint] $(date -u +%H:%M:%S) $*"; }

if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY RUN - would create checkpoint at ${SNAP}"
    log "  Compose file: ${COMPOSE_FILE}"
    log "  Service: ${SERVICE_NAME}"
    exit 0
fi

mkdir -p "${SNAP}"

# Capture compose config
if [[ -f "$COMPOSE_FILE" ]]; then
    cp "$COMPOSE_FILE" "${SNAP}/docker-compose.yml"
    docker compose -f "$COMPOSE_FILE" config > "${SNAP}/rendered-compose.yml" 2>/dev/null || true
    docker compose -f "$COMPOSE_FILE" ps --format json > "${SNAP}/ps.json" 2>/dev/null || true
fi

# Capture container inspect
CONTAINER_ID=$(docker compose -f "$COMPOSE_FILE" ps -q "$SERVICE_NAME" 2>/dev/null || echo "")
if [[ -n "$CONTAINER_ID" ]]; then
    docker inspect "$CONTAINER_ID" > "${SNAP}/inspect.json" 2>/dev/null || true
    docker logs "$CONTAINER_ID" --tail 500 > "${SNAP}/logs.txt" 2>&1 || true
fi

# Capture GPU state
nvidia-smi --query-gpu=index,name,memory.total,memory.used,temperature.gpu --format=csv \
    > "${SNAP}/gpu.csv" 2>/dev/null || true

# Capture vLLM models endpoint
ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8000}"
curl -sf "${ENDPOINT}/v1/models" > "${SNAP}/models.json" 2>/dev/null || true

# Environment (sanitized)
env | grep -E '^(VLLM_|MODEL_|SOURCE_|COMPOSE_)' | sort > "${SNAP}/env.txt" 2>/dev/null || true

# Record command that created this checkpoint
echo "checkpoint_vllm_mcprue.sh at ${TS}" > "${SNAP}/command.txt"

# SHA manifest
find "${SNAP}" -type f -not -name "manifest.sha256" | sort | while read -r f; do
    shasum -a 256 "$f"
done > "${SNAP}/manifest.sha256"

log "Checkpoint created: ${SNAP}"
log "Files: $(find "${SNAP}" -type f | wc -l | tr -d ' ')"
