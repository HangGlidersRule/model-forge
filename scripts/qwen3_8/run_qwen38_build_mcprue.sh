#!/usr/bin/env bash
set -euo pipefail
# Idempotent remote build runner for Qwen3.8 abliteration + NVFP4.
# Uses dedicated volumes for edited BF16, quant output, and work state.
# Traps restore the previous service on failure unless candidate is promoted.
#
# Default: DRY_RUN=1 (prints plan without executing)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${CONFIG:-}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the public artifact path}"
COMPOSE_FILE="${COMPOSE_FILE:-${REPO_ROOT}/containers/serve/docker-compose.yml}"
DRY_RUN="${DRY_RUN:-1}"
PROMOTE="${PROMOTE:-0}"

log() { echo "[build] $(date -u +%H:%M:%S) $*"; }
die() { log "FATAL: $*"; exit 1; }

# Trap-based restore on failure
CHECKPOINT_SNAP=""
restore_on_failure() {
    local exit_code=$?
    if [[ $exit_code -ne 0 && -n "$CHECKPOINT_SNAP" && "$PROMOTE" != "1" ]]; then
        log "Build failed (exit=$exit_code). Restoring previous service..."
        if [[ -f "${CHECKPOINT_SNAP}/docker-compose.yml" ]]; then
            docker compose -f "${CHECKPOINT_SNAP}/docker-compose.yml" up -d 2>/dev/null || true
            log "Previous service restore attempted from ${CHECKPOINT_SNAP}"
        fi
    fi
}
trap restore_on_failure EXIT

# --- Preflight ---
if [[ -z "${CONFIG}" ]]; then
    die "CONFIG is required; select a recipe explicitly (the rejected historical r3-nvfp4 path is never a default)"
fi
log "Config: ${CONFIG}"
log "Run root: ${RUN_ROOT}"
log "Dry run: ${DRY_RUN}"

if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY RUN MODE - printing plan:"
    log "  1. Checkpoint current vLLM service"
    log "  2. Materialize corpus (${RUN_ROOT}/corpus)"
    log "  3. Measure refusal direction (${RUN_ROOT}/measure_direction)"
    log "  4. Apply abliteration (${RUN_ROOT}/apply_abliteration)"
    log "  5. Validate BF16 edit"
    log "  6. Quantize to NVFP4 (${RUN_ROOT}/quantize_nvfp4)"
    log "  7. Validate compose config"
    log "  8. Boot candidate and run gates"
    log "  9. Promote if all gates pass"
    log ""
    log "To execute: DRY_RUN=0 $0"
    exit 0
fi

mkdir -p "${RUN_ROOT}"

# --- Step 1: Checkpoint current service ---
log "Step 1: Checkpointing current service..."
DRY_RUN=0 CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints" \
    bash "${SCRIPT_DIR}/checkpoint_vllm_mcprue.sh"
CHECKPOINT_SNAP="$(ls -td "${RUN_ROOT}/checkpoints/"* 2>/dev/null | head -1)"

# --- Step 2: Materialize corpus ---
log "Step 2: Materializing corpus..."
python "${SCRIPT_DIR}/materialize_abliteration_corpus.py" \
    --config "${CONFIG}" \
    --run-root "${RUN_ROOT}" \
    --resume

# --- Step 3: Measure direction ---
log "Step 3: Measuring refusal direction..."
python "${SCRIPT_DIR}/measure_qwen38_refusal_direction.py" \
    --config "${CONFIG}" \
    --run-root "${RUN_ROOT}" \
    --resume

# --- Step 4: Apply abliteration ---
log "Step 4: Applying abliteration..."
python "${SCRIPT_DIR}/apply_qwen38_abliteration.py" \
    --config "${CONFIG}" \
    --run-root "${RUN_ROOT}" \
    --resume

# --- Step 5: Validate BF16 ---
log "Step 5: Validating edited BF16..."
python "${SCRIPT_DIR}/validate_qwen38_bf16_edit.py" \
    --config "${CONFIG}" \
    --run-root "${RUN_ROOT}" \
    --structural-only

if [[ "${ALLOW_UNVALIDATED_BEHAVIOR:-0}" != "1" ]]; then
    die "Structural validation passed, but behavioral/KL/perplexity gates have not run. Set ALLOW_UNVALIDATED_BEHAVIOR=1 only after recording those results."
fi

# --- Step 6: Quantize ---
log "Step 6: Quantizing to NVFP4..."
python "${SCRIPT_DIR}/quantize_qwen38_nvfp4.py" \
    --config "${CONFIG}" \
    --run-root "${RUN_ROOT}" \
    --resume

# --- Step 7: Validate compose ---
log "Step 7: Validating compose config..."
if [[ -f "$COMPOSE_FILE" ]]; then
    docker compose -f "$COMPOSE_FILE" config -q || die "Compose validation failed"
    log "Compose config valid"
else
    log "WARN: Compose file not found at ${COMPOSE_FILE}, skipping validation"
fi

log "Build pipeline complete. Candidate at: ${RUN_ROOT}/quantize_nvfp4"
log "To boot and test: set PROMOTE=1 and re-run, or manually start the candidate."
