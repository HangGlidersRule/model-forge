#!/bin/bash
# Lightning Abliterated W4A16 NVFP4 ModelOpt quant — self-backgrounded launcher (WSL-safe).
# Export lands on D: (tmpfs /tmp is only 24G and fills during safetensors).
set -e
EXPORT_ROOT=${PUBLIC_ARTIFACT_PATH}
cd /tmp/lightning-quant
rm -f run2.log export-ready.flag
rm -rf "$EXPORT_ROOT"
mkdir -p "$EXPORT_ROOT"
docker run --rm --gpus=all --shm-size=16g \
  -e HF_TOKEN="$(cat ${PUBLIC_ARTIFACT_PATH})" \
  -e HF_HOME=/mnt/hf_cache -e HF_HUB_OFFLINE=0 \
  -e MODELOPT_CALIB_CACHE=/mnt/calib_cache -e TOKENIZERS_PARALLELISM=false \
  -v ${PUBLIC_ARTIFACT_PATH}:/mnt/source:ro \
  -v "$EXPORT_ROOT":/mnt/export \
  -v ${PUBLIC_ARTIFACT_PATH}:/mnt/recipes:ro \
  -v ${PUBLIC_ARTIFACT_PATH}:/mnt/hf_cache \
  -v ${PUBLIC_ARTIFACT_PATH}:/mnt/calib_cache \
  -w /opt/modelopt \
  local/model-forge-modelopt:0.46.0rc2-43fd41a \
  python examples/hf_ptq/hf_ptq.py \
    --pyt_ckpt_path /mnt/source \
    --export_path /mnt/export \
    --recipe ${PUBLIC_WORKSPACE} \
    --dataset cnn_dailymail,nemotron-post-training-dataset-v2 \
    --calib_size 512,512 \
    --calib_seq 2048 \
    --batch_size 1 \
    --kv_cache_qformat none \
    --export_fmt hf \
    --trust_remote_code
