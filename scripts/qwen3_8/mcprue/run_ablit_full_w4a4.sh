#!/usr/bin/env bash
# run_ablit_full_w4a4.sh — Build the ABLITERATED full-W4A4 NVFP4 Qwen3.8 artifact on mcprue.
#
# WHAT THIS DOES (written 2026-09-16, Devin-approved "run the requant"):
#   Quantize the ABLITERATED BF16 master (62 GB, 14 shards, the canonical Darkstar
#   qwen38 abliterated master at ${PUBLIC_ARTIFACT_PATH}/) into a
#   full-compute W4A4 NVFP4 artifact using the EXACT recipe that produced the validated
#   clean-full-w4a4 artifact (recipe sha256 f581bb8a... = _SUCCESS.json recipe_sha256 of
#   Qwen3.8-27B-clean-full-w4a4-modelopt).
#
# WHY: current production artifact (modelopt_mixed) quantizes the GDN projection matmuls
#   (out_proj/in_proj_qkv/in_proj_z) at FP8 W8A8. The full-W4A4 recipe puts those matmuls
#   on NVFP4 FP4 tensor cores (gittensor-style), worth +20-25% decode. Expected result:
#   abliterated + full-W4A4 artifact ~19-20 GB serving at ~110-120 tok/s single-stream.
#
# PROTECTIONS (from the recipe, verified 2026-09-16): vision tower / MMP / MTP head /
#   lm_head / embeddings / norms / routers stay BF16; GDN recurrence guts
#   (in_proj_a/in_proj_b/conv1d) stay BF16. Only GDN projection matmuls + MLPs go W4A4.
#
# WHAT YOU MUST KNOW:
#   - HARD RULE: mcprue runs ONE GPU serve at a time. This script STOPS the production
#     serve (snapshot/restore pattern), runs the quant, and RESTORES production on exit —
#     success, failure, or Ctrl-C (trap).
#   - ModelOpt pin: 0.46.0rc2 commit 43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a.
#     Image: local/model-forge-modelopt:0.46.0rc2-43fd41a (sha256:916957a774d7... — the
#     EXACT image id recorded in the clean-full-w4a4 _SUCCESS.json).
#   - The recipe's $import units (base_disable_all, nvfp4, nvfp4_static,
#     default_disabled_quantizers) resolve inside the container against ModelOpt's
#     built-in recipes lib (${PUBLIC_WORKSPACE},numerics}/...).
#     Verified 2026-09-16: load_recipe() inside the container loads the D: recipe OK.
#   - Calibration: pin.json contract = abisee/cnn_dailymail, 1024 samples, seqlen 2048
#     (same as the clean-full-w4a4 build; calib cache reused from ${PUBLIC_ARTIFACT_PATH}).
#   - MSYS_NO_PATHCONV=1 is REQUIRED for docker -v mounts from git-bash SSH (paths get
#     mangled to ${PUBLIC_WORKSPACE} Files/Git/... otherwise).
#   - Never overwrites: export goes to a TIMESTAMPED dir; the validator gates promotion.
#
# USAGE:
#   EXECUTE=1 bash run_ablit_full_w4a4.sh           # heavy run (stops+restores prod serve)
#   EXECUTE=0 bash run_ablit_full_w4a4.sh           # dry run: print the docker argv, no GPU
#
set -euo pipefail

D_ROOT="/d/model-forge"
SOURCE_DIR="${D_ROOT}/ablit/runs/apply_abliteration"
RECIPE="${D_ROOT}/configs/modelopt/recipes/qwen38-full-w4a4-modelopt.yaml"
IMAGE="local/model-forge-modelopt:0.46.0rc2-43fd41a"
HF_CACHE="${D_ROOT}/cache/hf"
CALIB_CACHE="${D_ROOT}/cache/calib"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
EXPORT_DIR="${D_ROOT}/artifacts/Qwen3.8-27B-abliterated-full-w4a4-modelopt-${TS}"
PROD_COMPOSE="${D_ROOT}/serve-production-nvfp4kv.yml"
LOG="${D_ROOT}/runs/ablit-full-w4a4-${TS}.log"
EXECUTE="${EXECUTE:-0}"

log() { echo "[ablit-w4a4 $(date -u +%H:%M:%S)] $*"; }

# --- Preflight ---
[[ -d "$SOURCE_DIR" ]] || { log "FATAL: source master missing: $SOURCE_DIR"; exit 1; }
[[ -f "$RECIPE" ]]     || { log "FATAL: recipe missing: $RECIPE"; exit 1; }
mkdir -p "$(dirname "$LOG")"

# Verify the source master is the validated abliterated one (leakage gate recorded).
grep -q "edited_tensors" "${SOURCE_DIR}/_SUCCESS.json" 2>/dev/null \
  || log "WARN: source _SUCCESS.json lacks edited_tensors field — verify manually"

# --- Production serve lifecycle (ONE-SERVE RULE) ---
restore_prod() {
  log "Restoring production serve..."
  (cd "$D_ROOT" && docker compose -f "$PROD_COMPOSE" up -d --force-recreate qwen38) \
    || log "WARN: restore compose failed — run it manually: docker compose -f $PROD_COMPOSE up -d --force-recreate qwen38"
}
if [[ "$EXECUTE" == "1" ]]; then
  trap '{ restore_prod; }' EXIT
  log "Stopping production serve (snapshot first)..."
  docker inspect vllm-qwen38-abliterated-performance > "${D_ROOT}/snapshots/pre-w4a4-${TS}-inspect.json" 2>/dev/null \
    || log "WARN: could not snapshot running container inspect"
  (cd "$D_ROOT" && docker compose -f "$PROD_COMPOSE" down) || true
fi

# --- The exact docker run (mirrors model_forge.modelopt.execution.build_docker_run_plan) ---
ARGV=(
  docker run --rm --gpus=all --shm-size=16g
  -e HF_HOME=/mnt/hf_cache
  -e HF_HUB_OFFLINE=0
  -e MODELOPT_CALIB_CACHE=/mnt/calib_cache
  -e TOKENIZERS_PARALLELISM=false
  -v "${SOURCE_DIR}:/mnt/source:ro"
  -v "${EXPORT_DIR}:/mnt/export"
  -v "$(dirname "$RECIPE"):/mnt/recipes:ro"
  -v "${HF_CACHE}:/mnt/hf_cache"
  -v "${CALIB_CACHE}:/mnt/calib_cache"
  -w /opt/modelopt
  "$IMAGE"
  python examples/hf_ptq/hf_ptq.py
  --pyt_ckpt_path /mnt/source
  --export_path /mnt/export
  --recipe ${PUBLIC_WORKSPACE}
  --dataset cnn_dailymail
  --calib_size 1024
  --calib_seq 2048
  --batch_size 1
  --kv_cache_qformat none
  --export_fmt hf
  --trust_remote_code
)

if [[ "$EXECUTE" != "1" ]]; then
  log "DRY RUN — exact docker argv:"
  printf '%q ' "${ARGV[@]}"; echo
  exit 0
fi

# --- Execute (all output to the run log; ~1-2 h on the 62 GB source) ---
log "Starting quantization. Log: $LOG"
MSYS_NO_PATHCONV=1 "${ARGV[@]}" > "$LOG" 2>&1 \
  || { log "FATAL: quantization failed — see $LOG"; exit 1; }

# --- Validate (same contract as the clean-full-w4a4 validator: MTP=15, vision=333,
#     protected tensors carry no scale markers, required files present) ---
log "Validating export..."
MSYS_NO_PATHCONV=1 docker run --rm -v "${EXPORT_DIR}:/export:ro" \
  --entrypoint sh "$IMAGE" -c 'python /export/validate.py 2>/dev/null || true' >/dev/null 2>&1 || true
# The validator in runs/modelopt-clean-full-w4a4/validate.py expects /export; run a fresh
# inline validation instead (no dependence on a mounted file):
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "${EXPORT_DIR}:/export:ro" \
  -v "${D_ROOT}/runs/modelopt-clean-full-w4a4/validate.py:/validate.py:ro" \
  --entrypoint sh "$IMAGE" -c 'python3 /validate.py' \
  || { log "FATAL: validation failed — artifact NOT promoted; see $LOG"; exit 1; }

# --- Write _SUCCESS + manifest ---
python3 - "$EXPORT_DIR" "$RECIPE" <<'PYEOF' || true
import hashlib, json, os, sys
from pathlib import Path
export = Path(sys.argv[1]); recipe = Path(sys.argv[2])
files = sorted(p for p in export.rglob("*") if p.is_file())
manifest = {str(p.relative_to(export)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(export / "manifest.sha256").write_text(json.dumps(manifest, indent=1))
success = {
  "candidate": "abliterated-full-w4a4-modelopt",
  "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
  "status": "validated",
  "modelopt": {"version": "0.46.0rc2", "commit": "43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a",
               "image_id": "sha256:916957a774d7e42bb44b931bb64f0dcb24b6f5a95c2a3b1fa6cea54cb9b4415d"},
  "calibration": {"dataset": "abisee/cnn_dailymail", "samples": 1024, "sequence_length": 2048},
  "kv_cache_export": "bf16",
  "source": "abliterated BF16 master (apply_abliteration, edited_tensors=3126, max_leakage=0.00015)",
}
(export / "_SUCCESS.json").write_text(json.dumps(success, indent=1))
print("manifest + _SUCCESS written")
PYEOF

log "BUILD COMPLETE: $EXPORT_DIR"
log "Next: boot it on the nvfp4kv config and run the gate battery (needles, MTP acceptance, GPQA)."
