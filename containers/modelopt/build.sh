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

# Parse the pin exactly once and forward only values from that one snapshot, so a
# pin rewritten mid-build cannot mix fields. Every field must match its recorded
# shape (version, 40-hex commit, https URL, 64-hex digest, PEP 427 wheel name)
# before docker sees it: the values become build args that RUN scripts consume.
PIN_FIELDS="$(python - "$PIN_JSON" <<'PY'
import json, re, sys

SHAPES = (
    ("version", r"[0-9]+(\.[0-9]+)*((a|b|rc)[0-9]+)?(\.post[0-9]+)?(\.dev[0-9]+)?"),
    ("git_commit", r"[0-9a-f]{40}"),
    ("wheel.url", r"https://[A-Za-z0-9.-]+(:[0-9]+)?(/[A-Za-z0-9._~%+-]+)+"),
    ("wheel.sha256", r"[0-9a-f]{64}"),
    (
        "wheel.filename",
        r"[A-Za-z0-9_.]+-[A-Za-z0-9_.+]+(-[0-9][A-Za-z0-9_.]*)?"
        r"-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+\.whl",
    ),
)
with open(sys.argv[1], encoding="utf-8") as handle:
    pin = json.load(handle)
for field, shape in SHAPES:
    value = pin
    try:
        for key in field.split("."):
            value = value[key]
    except (KeyError, TypeError):
        sys.exit(f"pin.json missing {field}")
    if not isinstance(value, str) or re.fullmatch(shape, value) is None:
        sys.exit(f"pin.json {field} is not a valid value: {value!r}")
    print(value)
PY
)" || die "pin.json failed validation: ${PIN_JSON}"

{
    read -r MODELOPT_VERSION
    read -r MODELOPT_COMMIT
    read -r MODELOPT_WHEEL_URL
    read -r MODELOPT_WHEEL_SHA256
    read -r MODELOPT_WHEEL_FILENAME
} <<< "${PIN_FIELDS}"

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
