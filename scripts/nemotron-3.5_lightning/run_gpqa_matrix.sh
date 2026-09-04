#!/usr/bin/env bash
# model-forge: GPQA campaign runner — FINAL SERVABLE RESULT ONLY.
#
# POLICY (Devin): only the final servable product+config gets a GPQA number.
# All other products/configs are RECORDED as process evidence (build, quant,
# tune, behavior gates) but NOT benchmarked. This script executes exactly one
# GPQA round per campaign: the shipping artifact in its shipping config.
#
# Lightning campaign: final servable = Abliterated-NVFP4 @ MTP10.
#
# Usage: bash run_gpqa_matrix.sh [--dry-run]
set -euo pipefail

REPO="${PUBLIC_WORKSPACE}"
cd "$REPO"

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY=1

# The ONE final servable result for this campaign:
#   product, decoder, remote compose file, served model
FINAL=( "ablit-nvfp4 mtp10 lightning-ablit-nvfp4-mtp.yml lightning-ablit-nvfp4" )

mkdir -p ${PUBLIC_WORKSPACE}
SUMMARY=${PUBLIC_WORKSPACE}
: > "$SUMMARY"

echo "== GPQA policy: final servable result only =="
if [ -n "$DRY" ]; then
  echo "dry-run: $FINAL"
  exit 0
fi

row="${FINAL[0]}"
# shellcheck disable=SC2086
product_decoder_compose_model=($row)
if bash scripts/nemotron-3.5_lightning/gpqa_round.sh \
    "${product_decoder_compose_model[0]}" \
    "${product_decoder_compose_model[1]}" \
    "${product_decoder_compose_model[2]}" \
    "${product_decoder_compose_model[3]}" 2>&1 | tail -4; then
  echo "${row} PASS" >> "$SUMMARY"
else
  echo "${row} FAIL" >> "$SUMMARY"
fi

echo "============================================================"
echo "SUMMARY:"
cat "$SUMMARY"
echo
echo "NOTE: all other products/configs are RECORDED as evidence, not benchmarked."
