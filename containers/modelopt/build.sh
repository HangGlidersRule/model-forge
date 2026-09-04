#!/usr/bin/env bash
set -euo pipefail
# Reproducible build of the pinned NVIDIA ModelOpt quantization image.
#
# Every build argument is sourced from configs/modelopt/pin.json so the image can
# never drift from the recorded pin (wheel URL + SHA-256, commit, versions). This
# script does not push; publish separately once the digest is recorded.
#
# Usage:
#   containers/modelopt/build.sh                 # build with pinned defaults
#   IMAGE=myrepo/modelopt:tag containers/modelopt/build.sh
#   DRY_RUN=1 containers/modelopt/build.sh       # print the docker build command only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PIN_JSON="${PIN_JSON:-${REPO_ROOT}/configs/modelopt/pin.json}"
DOCKER="${DOCKER:-docker}"
DRY_RUN="${DRY_RUN:-0}"

log() { echo "[build-modelopt] $*"; }
die() { echo "[build-modelopt] FATAL: $*" >&2; exit 1; }

[[ -f "${PIN_JSON}" ]] || die "pin.json not found at ${PIN_JSON}"

# Read the pin with the interpreter, not a JSON regex, so malformed pins fail closed.
read_pin() {
    python - "$PIN_JSON" "$1" <<'PY'
import json, sys
pin = json.load(open(sys.argv[1], encoding="utf-8"))
keys = sys.argv[2].split(".")
value = pin
for key in keys:
    value = value[key]
print(value)
PY
}

MODELOPT_VERSION="$(read_pin version)"
MODELOPT_COMMIT="$(read_pin git_commit)"
MODELOPT_WHEEL_URL="$(read_pin wheel.url)"
MODELOPT_WHEEL_SHA256="$(read_pin wheel.sha256)"
MODELOPT_WHEEL_FILENAME="$(read_pin wheel.filename)"

[[ -n "${MODELOPT_VERSION}" && -n "${MODELOPT_COMMIT}" ]] || die "pin.json missing version/commit"
[[ -n "${MODELOPT_WHEEL_URL}" && -n "${MODELOPT_WHEEL_SHA256}" ]] || die "pin.json missing wheel url/sha256"
[[ -n "${MODELOPT_WHEEL_FILENAME}" ]] || die "pin.json missing wheel filename"
# pip refuses a local wheel whose filename is not PEP 427 shaped.
[[ "${MODELOPT_WHEEL_FILENAME}" == *-*-*-*-*.whl ]] \
    || die "pin.json wheel filename is not a valid wheel name: ${MODELOPT_WHEEL_FILENAME}"

SHORT_COMMIT="${MODELOPT_COMMIT:0:7}"
IMAGE="${IMAGE:-ghcr.io/hangglidersrule/model-forge-modelopt:${MODELOPT_VERSION}-${SHORT_COMMIT}}"

log "pin.json        : ${PIN_JSON}"
log "modelopt version: ${MODELOPT_VERSION}"
log "modelopt commit : ${MODELOPT_COMMIT}"
log "wheel url       : ${MODELOPT_WHEEL_URL}"
log "wheel file      : ${MODELOPT_WHEEL_FILENAME}"
log "wheel sha256    : ${MODELOPT_WHEEL_SHA256}"
log "image tag       : ${IMAGE}"

BUILD_ARGS=(
    --build-arg "MODELOPT_VERSION=${MODELOPT_VERSION}"
    --build-arg "MODELOPT_COMMIT=${MODELOPT_COMMIT}"
    --build-arg "MODELOPT_WHEEL_URL=${MODELOPT_WHEEL_URL}"
    --build-arg "MODELOPT_WHEEL_SHA256=${MODELOPT_WHEEL_SHA256}"
    --build-arg "MODELOPT_WHEEL_FILENAME=${MODELOPT_WHEEL_FILENAME}"
)

CMD=("${DOCKER}" build "${BUILD_ARGS[@]}" -t "${IMAGE}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}")

if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY RUN. Would run:"
    printf '  %q ' "${CMD[@]}"; echo
    exit 0
fi

log "Building..."
"${CMD[@]}"
log "Built ${IMAGE}"
log "Record the resulting digest before publishing: ${DOCKER} inspect --format '{{index .RepoDigests 0}}' ${IMAGE}"
