#!/usr/bin/env bash
set -euo pipefail
# Canonical idempotent operator entrypoint for the frozen Darkstar Product 4 serve profile
# (abliterated ModelOpt W4A16-NVFP4-Mixed-FP8).
#
# This is the single supported way an operator brings Product 4 up. It renders THIS repository's
# deterministic Compose from src/model_forge/serve_profile.py, verifies both the freshly rendered
# bytes and the checked-in tracked file against the frozen digest, validates the Compose, and brings
# the one stable container up idempotently with a stable Compose project.
#
# Every vLLM model/runtime argument is frozen and encoded in the deterministic profile; the launcher
# accepts no mutable vLLM arguments. The only Product 4 environment override is the published host
# port, VLLM_PORT. Secrets (e.g. an HF token) and host GPU/cache/Docker settings reach Docker through
# the ambient environment and the daemon's own configuration; they are never vLLM knobs.
#
# Usage:
#   scripts/serve_darkstar_qwen38_abliterated_nvfp4.sh \
#       --artifact-path ${PUBLIC_ARTIFACT_PATH}
#   VLLM_PORT=8001 scripts/serve_darkstar_qwen38_abliterated_nvfp4.sh --artifact-path ...
#   scripts/serve_darkstar_qwen38_abliterated_nvfp4.sh --print-config   # resolved Compose, no launch
#   scripts/serve_darkstar_qwen38_abliterated_nvfp4.sh --dry-run        # planned commands, no launch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Frozen Product 4 identity. None of these is operator-overridable. ---
FAMILY="qwen38"
BEHAVIOR="abliterated"
FORMAT="nvfp4"
REPOSITORY_ID="HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8"
ARTIFACT_IDENTITY="Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8"
ARTIFACT_PRECISION_CLASS="W4A16-NVFP4-Mixed-FP8"
ARTIFACT_SUCCESS_SHA256="3d89ec57c1371e142adc2584de079b54a0e1d8c12dc9550118d0a851da020a79"
CONTAINER_NAME="vllm-darkstar-qwen38-abliterated-modelopt"
MTP_DEPTH="10"
SCHEDULER_TOKENS="32768"
DEFAULT_ARTIFACT_PATH="${PUBLIC_ARTIFACT_PATH}"
PROJECT_NAME="darkstar-qwen38-abliterated-nvfp4"
COMPOSE_FILE="${REPO_ROOT}/containers/serve/darkstar-qwen38-abliterated-nvfp4.yml"
ARTIFACT_MANIFEST="${REPO_ROOT}/models/qwen3.8-27b-r3/results/manifests/abliterated-modelopt-mixed-manifest.json"
TRACKED_COMPOSE_SHA256="5434c2a99bdadce512bd87b65c30f830c21fc2eae647182ffa89e77b174833cc"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
DOCKER="${DOCKER:-docker}"

log() { echo "[serve-darkstar-product4] $*"; }
die() { echo "[serve-darkstar-product4] FATAL: $*" >&2; exit 1; }

ARTIFACT_PATH="${DEFAULT_ARTIFACT_PATH}"
ARTIFACT_IDENTITY_ARG="${ARTIFACT_IDENTITY}"
DRY_RUN=0
PRINT_CONFIG=0

require_value() {
    # $1 flag name, $2 candidate value
    [[ -n "${2:-}" && "${2:0:2}" != "--" ]] || die "$1 requires a non-empty value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --artifact-path)
            require_value "$1" "${2:-}"; ARTIFACT_PATH="$2"; shift 2 ;;
        --artifact-path=*)
            ARTIFACT_PATH="${1#*=}"; [[ -n "${ARTIFACT_PATH}" ]] || die "--artifact-path requires a non-empty value"; shift ;;
        --artifact-identity)
            require_value "$1" "${2:-}"; ARTIFACT_IDENTITY_ARG="$2"; shift 2 ;;
        --artifact-identity=*)
            ARTIFACT_IDENTITY_ARG="${1#*=}"; [[ -n "${ARTIFACT_IDENTITY_ARG}" ]] || die "--artifact-identity requires a non-empty value"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --print-config) PRINT_CONFIG=1; shift ;;
        -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown argument: $1 (this launcher accepts no mutable vLLM arguments)" ;;
    esac
done

# Fail closed on any attempt to override a frozen vLLM knob through the environment. Only VLLM_PORT
# (the published host port) is honored.
while IFS= read -r rogue_var; do
    [[ "${rogue_var}" == "VLLM_PORT" ]] && continue
    die "refusing ${rogue_var}: this frozen profile accepts only VLLM_PORT as a host override"
done < <(env | sed -n 's/^\(VLLM_[A-Za-z0-9_]*\)=.*/\1/p')

if [[ -n "${VLLM_PORT:-}" ]]; then
    [[ "${VLLM_PORT}" =~ ^[0-9]+$ ]] || die "VLLM_PORT must be an integer, got '${VLLM_PORT}'"
    (( VLLM_PORT >= 1 && VLLM_PORT <= 65535 )) || die "VLLM_PORT must be in 1-65535, got '${VLLM_PORT}'"
fi

[[ "${ARTIFACT_IDENTITY_ARG}" == "${ARTIFACT_IDENTITY}" ]] \
    || die "artifact identity '${ARTIFACT_IDENTITY_ARG}' does not match the frozen Product 4 identity"

sha256_of() {
    # Portable SHA-256 of a file, printing only the hex digest.
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        sha256sum "$1" | awk '{print $1}'
    fi
}

# --- Render the deterministic Compose into a scratch dir so the tracked file is never mutated by an
#     unexpected input, then verify the render is byte-for-byte the frozen profile. ---
SCRATCH_DIR="$(mktemp -d)"
cleanup() { rm -rf "${SCRATCH_DIR}"; }
trap cleanup EXIT

log "Rendering deterministic Compose from serve_profile.py"
"${PYTHON}" "${REPO_ROOT}/scripts/render_darkstar_serve_profile.py" \
    --family "${FAMILY}" --behavior "${BEHAVIOR}" --format "${FORMAT}" \
    --repository-id "${REPOSITORY_ID}" \
    --model-path "${ARTIFACT_PATH}" \
    --artifact-identity "${ARTIFACT_IDENTITY}" \
    --artifact-precision-class "${ARTIFACT_PRECISION_CLASS}" \
    --artifact-success-sha256 "${ARTIFACT_SUCCESS_SHA256}" \
    --artifact-manifest "${ARTIFACT_MANIFEST}" \
    --container-name "${CONTAINER_NAME}" \
    --mtp-depth "${MTP_DEPTH}" --scheduler-tokens "${SCHEDULER_TOKENS}" \
    --output-directory "${SCRATCH_DIR}" >/dev/null

RENDERED_FILE="${SCRATCH_DIR}/${PROJECT_NAME}.yml"
[[ -f "${RENDERED_FILE}" ]] || die "renderer did not produce ${RENDERED_FILE}"

rendered_sha="$(sha256_of "${RENDERED_FILE}")"
if [[ "${rendered_sha}" != "${TRACKED_COMPOSE_SHA256}" ]]; then
    die "rendered Compose digest ${rendered_sha} != frozen ${TRACKED_COMPOSE_SHA256}; the artifact path or profile inputs are not the verified Product 4 values"
fi

[[ -f "${COMPOSE_FILE}" ]] || die "tracked Compose not found at ${COMPOSE_FILE}"
tracked_sha="$(sha256_of "${COMPOSE_FILE}")"
if [[ "${tracked_sha}" != "${TRACKED_COMPOSE_SHA256}" ]]; then
    die "tracked Compose digest ${tracked_sha} != frozen ${TRACKED_COMPOSE_SHA256}; the checked-in file has drifted from the deterministic profile"
fi
log "Deterministic Compose verified against frozen digest ${TRACKED_COMPOSE_SHA256}"

COMPOSE_BASE=("${DOCKER}" compose -p "${PROJECT_NAME}" -f "${COMPOSE_FILE}")

if [[ "${PRINT_CONFIG}" == "1" ]]; then
    log "Resolved Compose configuration (VLLM_PORT=${VLLM_PORT:-8000}):"
    "${COMPOSE_BASE[@]}" config
    exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY RUN. Would run:"
    echo "  ${COMPOSE_BASE[*]} config -q"
    echo "  ${COMPOSE_BASE[*]} up -d --force-recreate"
    exit 0
fi

log "Validating Compose"
"${COMPOSE_BASE[@]}" config -q || die "Compose validation failed"

log "Bringing up ${CONTAINER_NAME} (stable project ${PROJECT_NAME})"
"${COMPOSE_BASE[@]}" up -d --force-recreate

log "Serving ${ARTIFACT_IDENTITY} on port ${VLLM_PORT:-8000} as darkstar-qwen38-abliterated-nvfp4"
