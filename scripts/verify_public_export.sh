#!/usr/bin/env bash
set -euo pipefail

readonly PROGRAM="${0##*/}"

fail() {
  printf '%s\n' "$PROGRAM: $1" >&2
  exit 2
}

if (( $# < 3 || $# > 4 )); then
  fail "usage: $PROGRAM EXPORT_ROOT SOURCE_REPO EXPECTED_SOURCE_SHA [TRUSTED_WHEELHOUSE]"
fi

readonly export_root=$1
readonly source_repo=$2
readonly source_sha=$3
readonly supplied_wheelhouse="${4:-${MODEL_FORGE_WHEELHOUSE:-}}"
readonly public_contact="${MODEL_FORGE_PUBLIC_CONTACT:-security@hangglidersrule.com}"
temporary=

[[ -d "$export_root" && ! -L "$export_root" ]] ||
  fail "export root must be an existing non-symlink directory"
[[ -d "$source_repo" && ! -L "$source_repo" ]] ||
  fail "source repo must be an existing non-symlink directory"
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] ||
  fail "expected source SHA must be 40 lowercase hexadecimal characters"
command -v model-forge >/dev/null 2>&1 ||
  fail "installed model-forge command is required"

cleanup() {
  if [[ -n "$temporary" ]]; then
    rm -rf -- "$temporary"
  fi
}
trap cleanup EXIT HUP INT TERM

if [[ -n "$supplied_wheelhouse" ]]; then
  readonly wheelhouse=$supplied_wheelhouse
  readonly wheelhouse_lock="${MODEL_FORGE_WHEELHOUSE_LOCK:-${wheelhouse}.sha256}"
  [[ -d "$wheelhouse" && ! -L "$wheelhouse" ]] ||
    fail "trusted wheelhouse is missing: $wheelhouse"
  [[ -f "$wheelhouse_lock" && ! -L "$wheelhouse_lock" ]] ||
    fail "trusted wheelhouse SHA256 lock is missing: $wheelhouse_lock"
else
  temporary=$(mktemp -d "${TMPDIR:-/tmp}/model-forge-wheelhouse.XXXXXXXX") ||
    fail "cannot create private temporary wheelhouse directory"
  chmod 700 "$temporary"
  readonly wheelhouse="$temporary/wheels"
  readonly wheelhouse_lock="$temporary/wheels.sha256"
  readonly environment_python="${MODEL_FORGE_ENVIRONMENT_PYTHON:-${source_repo}/.venv/bin/python}"
  [[ -x "$environment_python" ]] ||
    fail "local distribution environment is unavailable: set MODEL_FORGE_ENVIRONMENT_PYTHON"
  if [[ -n "${MODEL_FORGE_BOOTSTRAP_PYTHON:-}" ]]; then
    bootstrap_python=$MODEL_FORGE_BOOTSTRAP_PYTHON
  elif "$environment_python" -c 'import packaging, tomllib' >/dev/null 2>&1; then
    bootstrap_python=$environment_python
  elif command -v python3 >/dev/null 2>&1 &&
    python3 -c 'import packaging, tomllib' >/dev/null 2>&1; then
    bootstrap_python=$(command -v python3)
  elif command -v uv >/dev/null 2>&1; then
    bootstrap_python=$(UV_PYTHON_DOWNLOADS=never uv python find '>=3.11' 2>/dev/null) ||
      fail "uv could not find a local Python 3.11+ for offline bootstrap"
  else
    fail "offline bootstrap requires local Python 3.11+ with packaging or uv"
  fi
  "$bootstrap_python" "$source_repo/scripts/bootstrap_public_export_wheelhouse.py" \
    --source-repo "$source_repo" \
    --environment-python "$environment_python" \
    --wheelhouse "$wheelhouse" \
    --lock "$wheelhouse_lock" ||
    fail "offline wheelhouse bootstrap could not satisfy the committed dependency closure"
fi

model-forge public-verify \
  --root "$export_root" \
  --source-sha "$source_sha" \
  --source-repo "$source_repo" \
  --manifest tools/public_export/public-files.yaml \
  --wheelhouse "$wheelhouse" \
  --wheelhouse-lock "$wheelhouse_lock" \
  --public-contact "$public_contact"
