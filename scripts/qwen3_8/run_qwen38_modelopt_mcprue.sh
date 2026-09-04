#!/usr/bin/env bash
set -euo pipefail
# Idempotent ModelOpt build runner for mcprue (D:-only artifacts).
#
# Defaults to dry-run. Heavy work requires EXECUTE=1, which really runs the pinned
# ModelOpt container against examples/hf_ptq/hf_ptq.py, validates the export with
# fail-closed validators, writes a SHA256 manifest + _SUCCESS, and atomically
# promotes the partial export to its final path. It never stops intentionally and
# never fabricates success.
#
# Never overwrites prior artifacts; snapshots are append-only.
# Restores the previously running runtime on failure unless PROMOTE=1.
#
# Required host layout (example):
#   D_ROOT=${PUBLIC_ARTIFACT_PATH}   (or ${PUBLIC_ARTIFACT_PATH} on WSL)
#
# Does not mutate remote Hugging Face / GHCR artifacts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

D_ROOT="${D_ROOT:-}"
EXECUTE="${EXECUTE:-0}"
DRY_RUN="${DRY_RUN:-1}"
PROMOTE="${PROMOTE:-0}"
CANDIDATE="${CANDIDATE:-mixed_w4a16}"
SOURCE_KIND="${SOURCE_KIND:-clean}"
FREE_GB_REQUIRED="${FREE_GB_REQUIRED:-200}"
MODELOPT_IMAGE="${MODELOPT_IMAGE:-ghcr.io/hangglidersrule/model-forge-modelopt:0.46.0rc2-43fd41a}"
DOCKER="${DOCKER:-docker}"
PIN_JSON="${REPO_ROOT}/configs/modelopt/pin.json"

# Runtime snapshot/restore inputs (no dependency on a compose file we do not own).
RUNTIME_CONTAINER="${RUNTIME_CONTAINER:-}"
RUNTIME_NAME_FILTER="${RUNTIME_NAME_FILTER:-vllm}"
RESTORE_COMPOSE="${RESTORE_COMPOSE:-}"
ALLOW_NO_RUNTIME="${ALLOW_NO_RUNTIME:-0}"

# Host checkout of NVIDIA/Model-Optimizer at the pinned commit (optional; when set
# it is mounted read-only over the image's baked-in copy).
MODELOPT_ROOT="${MODELOPT_ROOT:-}"

log() { echo "[modelopt-mcprue] $(date -u +%H:%M:%S) $*"; }
die() { log "FATAL: $*"; exit 1; }

# Deprecated pre-rename interfaces (SOURCE_KIND=darkstar, ALLOW_DARKSTAR) are
# normalized to the canonical abliterated terminology here, before any guard,
# path, or execution decision, so both spellings behave identically.
if [[ "${SOURCE_KIND}" == "darkstar" ]]; then
    SOURCE_KIND="abliterated"
    log "WARN: SOURCE_KIND=darkstar is deprecated; using SOURCE_KIND=abliterated"
fi
if [[ -n "${ALLOW_DARKSTAR:-}" && -z "${ALLOW_ABLITERATED:-}" ]]; then
    ALLOW_ABLITERATED="${ALLOW_DARKSTAR}"
    log "WARN: ALLOW_DARKSTAR is deprecated; use ALLOW_ABLITERATED"
fi
ALLOW_ABLITERATED="${ALLOW_ABLITERATED:-0}"

# Deterministic interpreter + import path: the helper scripts import model_forge
# from this checkout, so the runner must not depend on an activated venv, on a
# bare `python` being on PATH, or on the package being installed. PYTHON_BIN
# overrides the search; a repo venv is preferred because it carries the deps.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_IMPORT_PROBE="import model_forge.modelopt.validate"

python_can_import() {
    command -v "$1" >/dev/null 2>&1 && "$1" -c "${PYTHON_IMPORT_PROBE}" >/dev/null 2>&1
}

resolve_python() {
    local candidate
    for candidate in \
        "${REPO_ROOT}/.venv/bin/python" \
        "${REPO_ROOT}/.venv/Scripts/python.exe" \
        python3 \
        python
    do
        if python_can_import "${candidate}"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$(resolve_python)" || die \
        "No Python interpreter can import model_forge from ${REPO_ROOT}/src. Create the repo venv (python3 -m venv .venv && .venv/bin/pip install -e '.[dev]') or set PYTHON_BIN."
fi
python_can_import "${PYTHON_BIN}" || die \
    "PYTHON_BIN='${PYTHON_BIN}' cannot import model_forge with PYTHONPATH=${PYTHONPATH}. It must exist and provide the project dependencies (PyYAML, jsonschema, pydantic, httpx)."

RESTORE_ARTIFACT=""
restore_on_failure() {
    local exit_code=$?
    if [[ $exit_code -ne 0 && "${PROMOTE}" != "1" && -n "${RESTORE_ARTIFACT}" ]]; then
        log "Build failed (exit=${exit_code}). Restoring previous runtime from ${RESTORE_ARTIFACT}..."
        case "${RESTORE_ARTIFACT}" in
            *.sh)
                bash "${RESTORE_ARTIFACT}" || log "WARN: restore script returned nonzero"
                ;;
            *.yml|*.yaml)
                "${DOCKER}" compose -f "${RESTORE_ARTIFACT}" up -d \
                    || log "WARN: restore compose returned nonzero"
                ;;
        esac
    fi
}
trap restore_on_failure EXIT

if [[ -z "${D_ROOT}" ]]; then
    die "D_ROOT must point at the D:-backed model-forge root (e.g. ${PUBLIC_ARTIFACT_PATH})"
fi

# Normalize and refuse non-D paths when the root looks like a Windows drive mount.
case "${D_ROOT}" in
    ${PUBLIC_WORKSPACE}|${PUBLIC_WORKSPACE}|${PUBLIC_WORKSPACE}|${PUBLIC_WORKSPACE}|/mnt/d/*|/d/*) ;;
    *)
        if [[ "${ALLOW_NON_D_ROOT:-0}" != "1" ]]; then
            die "Refusing non-D: root '${D_ROOT}'. Set ALLOW_NON_D_ROOT=1 only for local dry tests."
        fi
        ;;
esac

RUN_ROOT="${RUN_ROOT:-${D_ROOT}/runs/modelopt-${SOURCE_KIND}-${CANDIDATE}}"
CACHE_ROOT="${CACHE_ROOT:-${D_ROOT}/cache}"
OUT_ROOT="${OUT_ROOT:-${D_ROOT}/artifacts}"
SNAP_ROOT="${SNAP_ROOT:-${D_ROOT}/snapshots}"
SOURCE_DIR="${SOURCE_DIR:-${D_ROOT}/sources/clean-bf16}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
SNAP="${SNAP_ROOT}/${TS}-pre-${SOURCE_KIND}-${CANDIDATE}"

log "D_ROOT=${D_ROOT}"
log "RUN_ROOT=${RUN_ROOT}"
log "CANDIDATE=${CANDIDATE} SOURCE_KIND=${SOURCE_KIND}"
log "MODELOPT_IMAGE=${MODELOPT_IMAGE}"
log "EXECUTE=${EXECUTE} DRY_RUN=${DRY_RUN} PROMOTE=${PROMOTE}"
log "PYTHON_BIN=${PYTHON_BIN} PYTHONPATH=${PYTHONPATH}"
log "pin: ${PIN_JSON}"

ABLITERATED_FLAG=()
if [[ "${SOURCE_KIND}" == "abliterated" ]]; then
    if [[ "${ALLOW_ABLITERATED}" != "1" ]]; then
        die "Abliterated build requires explicit heavy-mutation authorization (set ALLOW_ABLITERATED=1)"
    fi
    ABLITERATED_FLAG=(--allow-abliterated)
fi

# Free-space preflight (best effort; portable df). Skip when FREE_GB_REQUIRED=0 (dry tests).
if [[ "${FREE_GB_REQUIRED}" != "0" ]]; then
    free_kb="$(df -k "${D_ROOT}" 2>/dev/null | awk 'NR==2{print $4}')"
    if [[ -n "${free_kb}" ]]; then
        free_gb=$((free_kb / 1024 / 1024))
        log "Free space on D_ROOT: ${free_gb} GiB (required >= ${FREE_GB_REQUIRED})"
        if (( free_gb < FREE_GB_REQUIRED )); then
            die "Insufficient free space: ${free_gb} GiB < ${FREE_GB_REQUIRED} GiB"
        fi
    else
        log "WARN: could not determine free space for ${D_ROOT}"
    fi
fi

plan() {
    log "PLAN:"
    log "  1. Append-only snapshot of running runtime -> ${SNAP} (docker inspect -> restore.sh)"
    log "  2. Validate ModelOpt recipe + pin (no overwrite)"
    log "  3. Run pinned ModelOpt image ${MODELOPT_IMAGE} on examples/hf_ptq/hf_ptq.py"
    log "  4. Fail-closed validators (MTP/vision/scales/KV/fused/tokenizer)"
    log "  5. SHA256 manifest + _SUCCESS, then atomic partial->final promotion"
    log "  6. On failure: restore prior runtime from the captured restore artifact"
}

# Deterministic artifact name; refuse overwrite of a non-validated prior artifact.
# Preserve the established on-disk artifact path while using "abliterated" for the
# operational source kind. This compatibility slug is a path, not a lineage label.
ARTIFACT_SOURCE_SLUG="${SOURCE_KIND}"
if [[ "${SOURCE_KIND}" == "abliterated" ]]; then
    ARTIFACT_SOURCE_SLUG="darkstar"
fi
OUT_NAME="Qwen3.8-27B-${ARTIFACT_SOURCE_SLUG}-${CANDIDATE}-modelopt-nvfp4"
OUT_DIR="${OUT_ROOT}/${OUT_NAME}"

if [[ "${DRY_RUN}" == "1" || "${EXECUTE}" != "1" ]]; then
    plan
    # Fail-closed: recipe/pin validation errors must surface (never swallowed).
    "${PYTHON_BIN}" "${SCRIPT_DIR}/quantize_qwen38_modelopt.py" \
        --run-root "${RUN_ROOT}" \
        --source-dir "${SOURCE_DIR}" \
        --candidate "${CANDIDATE}" \
        --source-kind "${SOURCE_KIND}" \
        ${ABLITERATED_FLAG[@]+"${ABLITERATED_FLAG[@]}"}
    log "DRY RUN complete. Set EXECUTE=1 DRY_RUN=0 to run heavy quantization."
    exit 0
fi

mkdir -p "${RUN_ROOT}" "${CACHE_ROOT}" "${OUT_ROOT}" "${SNAP_ROOT}"

# Idempotency is decided by the *complete* build identity (pin + candidate +
# recipe digest + calibration + source), not by the presence of _SUCCESS.json:
# a recipe/pin/calibration/source change must never be masked by a stale marker.
# Exit codes mirror quantize_qwen38_modelopt.py: 0 up to date, 20 build required.
if [[ -e "${OUT_DIR}" ]]; then
    identity_rc=0
    "${PYTHON_BIN}" "${SCRIPT_DIR}/quantize_qwen38_modelopt.py" \
        --run-root "${RUN_ROOT}" \
        --source-dir "${SOURCE_DIR}" \
        --candidate "${CANDIDATE}" \
        --source-kind "${SOURCE_KIND}" \
        ${ABLITERATED_FLAG[@]+"${ABLITERATED_FLAG[@]}"} \
        --check-build-identity \
        --export-dir "${OUT_DIR}" || identity_rc=$?
    case "${identity_rc}" in
        0)
            log "Artifact already validated for this build identity: ${OUT_DIR} (idempotent skip)"
            RESTORE_ARTIFACT=""
            exit 0
            ;;
        20)
            log "Existing path carries no artifact for this build identity; continuing"
            ;;
        *)
            die "Refusing overwrite of existing artifact: ${OUT_DIR} (build identity check exit=${identity_rc})"
            ;;
    esac
fi

# --- Append-only runtime snapshot -> restore artifact ---
log "Creating append-only snapshot at ${SNAP}"
mkdir -p "${SNAP}"
printf '%s\n' "${TS}" > "${SNAP}/timestamp.txt"
cp "${PIN_JSON}" "${SNAP}/pin.json"

if [[ -n "${RESTORE_COMPOSE}" ]]; then
    [[ -f "${RESTORE_COMPOSE}" ]] || die "RESTORE_COMPOSE not found: ${RESTORE_COMPOSE}"
    cp "${RESTORE_COMPOSE}" "${SNAP}/restore-compose.yml"
    RESTORE_ARTIFACT="${SNAP}/restore-compose.yml"
    log "Captured explicit restore compose: ${RESTORE_ARTIFACT}"
else
    cid="${RUNTIME_CONTAINER}"
    if [[ -z "${cid}" ]]; then
        cid="$("${DOCKER}" ps --filter "name=${RUNTIME_NAME_FILTER}" -q 2>/dev/null | head -1)"
    fi
    if [[ -z "${cid}" ]]; then
        if [[ "${ALLOW_NO_RUNTIME}" == "1" ]]; then
            log "No running runtime container found; ALLOW_NO_RUNTIME=1 (nothing to restore)."
            printf 'no running runtime container at %s\n' "${TS}" > "${SNAP}/no-runtime.txt"
        else
            die "No running runtime container to snapshot. Set RUNTIME_CONTAINER, RESTORE_COMPOSE, or ALLOW_NO_RUNTIME=1."
        fi
    else
        log "Snapshotting running runtime container ${cid}"
        "${DOCKER}" inspect "${cid}" > "${SNAP}/inspect.json" \
            || die "docker inspect failed for ${cid}"
        if ! "${DOCKER}" logs --tail 500 "${cid}" > "${SNAP}/logs.txt" 2>&1; then
            log "WARN: could not capture logs for ${cid}"
        fi
        "${PYTHON_BIN}" "${SCRIPT_DIR}/snapshot_runtime.py" \
            --inspect "${SNAP}/inspect.json" \
            --out "${SNAP}/restore.sh" \
            --docker-bin "${DOCKER}"
        RESTORE_ARTIFACT="${SNAP}/restore.sh"
        log "Runtime restore artifact: ${RESTORE_ARTIFACT}"
    fi
fi

# --- Heavy quantization: real container run + validation + atomic promotion ---
MODELOPT_ROOT_FLAG=()
if [[ -n "${MODELOPT_ROOT}" ]]; then
    MODELOPT_ROOT_FLAG=(--modelopt-root "${MODELOPT_ROOT}")
fi

log "Launching pinned ModelOpt quantization into ${OUT_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/quantize_qwen38_modelopt.py" \
    --run-root "${RUN_ROOT}" \
    --source-dir "${SOURCE_DIR}" \
    --candidate "${CANDIDATE}" \
    --source-kind "${SOURCE_KIND}" \
    ${ABLITERATED_FLAG[@]+"${ABLITERATED_FLAG[@]}"} \
    --execute \
    --export-dir "${OUT_DIR}" \
    --modelopt-image "${MODELOPT_IMAGE}" \
    --hf-cache "${CACHE_ROOT}/hf" \
    --calib-cache "${CACHE_ROOT}/calib" \
    --docker-bin "${DOCKER}" \
    --gpus "${GPUS:-all}" \
    ${MODELOPT_ROOT_FLAG[@]+"${MODELOPT_ROOT_FLAG[@]}"}

log "Build complete. Validated artifact at ${OUT_DIR}"
# Success: disarm the restore trap (nothing to roll back).
RESTORE_ARTIFACT=""
