#!/usr/bin/env bash
set -euo pipefail

readonly program="${0##*/}"

fail() {
  printf '%s\n' "$program: $1" >&2
  exit 2
}

if (( $# < 2 || $# > 5 )); then
  fail "usage: $program OUTPUT_ROOT SUMMARY_JSON [SOURCE_REPO [SOURCE_SHA [TRUSTED_WHEELHOUSE]]]"
fi

readonly output_root=$1
readonly summary_json=$2
script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && /bin/pwd -P)
readonly script_dir
default_source=$(CDPATH='' cd -- "$script_dir/.." && /bin/pwd -P)
readonly default_source
raw_source=${3:-$default_source}
readonly raw_source
source_repo=$(CDPATH='' cd -- "$raw_source" && /bin/pwd -P)
readonly source_repo
readonly source_sha=${4:-$(git -C "$source_repo" rev-parse HEAD)}
readonly wheelhouse=${5:-}
readonly manifest="$source_repo/tools/public_export/public-files.yaml"
readonly python=${MODEL_FORGE_STAGE_PYTHON:-python3}
if [[ -n "${MODEL_FORGE_COMMAND:-}" ]]; then
  model_forge=$MODEL_FORGE_COMMAND
elif [[ -x "$source_repo/.venv/bin/model-forge" ]]; then
  model_forge="$source_repo/.venv/bin/model-forge"
else
  model_forge=$(command -v model-forge || true)
fi
readonly model_forge

[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] ||
  fail "source SHA must be 40 lowercase hexadecimal characters"
[[ -d "$source_repo" && ! -L "$source_repo" ]] ||
  fail "source repository must be an existing non-symlink directory"
[[ -f "$manifest" && ! -L "$manifest" ]] ||
  fail "public export manifest is missing"
[[ -n "$model_forge" && -x "$model_forge" ]] ||
  fail "installed model-forge command is required"
command -v "$python" >/dev/null 2>&1 ||
  fail "local Python 3 is required for deterministic summary generation"
model_forge_dir=$(dirname -- "$model_forge")
readonly model_forge_dir
PATH="$model_forge_dir:$PATH"
export PATH
PYTHONPATH="$source_repo/src"
export PYTHONPATH

"$model_forge" public-export \
  --source "$source_repo" \
  --output "$output_root" \
  --manifest "$manifest" \
  --source-sha "$source_sha" \
  --replace

if [[ -n "$wheelhouse" ]]; then
  "$source_repo/scripts/verify_public_export.sh" \
    "$output_root" "$source_repo" "$source_sha" "$wheelhouse"
else
  "$source_repo/scripts/verify_public_export.sh" \
    "$output_root" "$source_repo" "$source_sha"
fi

readonly attestation="$output_root/PUBLIC_EXPORT_MANIFEST.json"
[[ -f "$attestation" && ! -L "$attestation" ]] ||
  fail "verified export attestation is missing"
mkdir -p -- "$(dirname -- "$summary_json")"
summary_tmp=$(mktemp "${summary_json}.tmp.XXXXXXXX") ||
  fail "cannot create staging summary temporary file"
cleanup() {
  rm -f -- "$summary_tmp"
}
trap cleanup EXIT HUP INT TERM

"$python" - "$attestation" "$summary_tmp" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

attestation_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
with attestation_path.open("r", encoding="utf-8") as stream:
    attestation = json.load(stream)
required = {"source_sha", "payload_tree_sha256", "files"}
if not isinstance(attestation, dict) or not required <= attestation.keys():
    raise SystemExit("public export attestation is malformed")
files = attestation["files"]
if not isinstance(files, list):
    raise SystemExit("public export attestation file inventory is malformed")
summary = {
    "schema": "model-forge-public-staging/v1",
    "source_sha": attestation["source_sha"],
    "export_digest": attestation["payload_tree_sha256"],
    "file_count": len(files),
}
output_path.write_text(
    json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY
chmod 0644 "$summary_tmp"
mv -f -- "$summary_tmp" "$summary_json"
trap - EXIT HUP INT TERM
printf 'Staged public root %s from %s; summary: %s\n' \
  "$output_root" "$source_sha" "$summary_json"
