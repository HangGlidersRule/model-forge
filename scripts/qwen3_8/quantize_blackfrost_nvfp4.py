#!/usr/bin/env python3
"""REJECTED / HISTORICAL: llm-compressor compressed-tensors NVFP4 W4A4 for the pre-Darkstar
(then "Blackfrost") Qwen3.8 abliterated BF16 source.

This script and the artifacts it produces are rejected/historical and kept for lineage only. The
current brand is **Darkstar** (HangGlidersRule), and the current publication NVFP4 path is **NVIDIA
ModelOpt** (see `scripts/qwen3_8/quantize_qwen38_modelopt.py` and `models/qwen3.8-27b-r3/modelopt/`).
The environment defaults below are the original historical pins and are intentionally left unchanged so
the superseded build stays reproducible for lineage; do not use this path to produce a current release.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from compressed_tensors.offload import load_offloaded_model
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

SOURCE = os.environ.get("SOURCE_MODEL", "Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16")
SOURCE_REVISION = os.environ.get("SOURCE_REVISION", "9d85770e5eb602322b4bceef55beda357e0bd0ca")
MTP_DONOR = os.environ.get("MTP_DONOR", "sakamakismile/Qwen3.8-27B-MTP-NVFP4")
MTP_REVISION = os.environ.get("MTP_REVISION", "6d98dc1f1d5259c9582794014b73852baf20f805")
OUTPUT = Path(os.environ.get("OUTPUT_DIR", "${PUBLIC_WORKSPACE}"))
OFFLOAD = Path(os.environ.get("OFFLOAD_DIR", "/work/offload"))
NUM_SAMPLES = int(os.environ.get("NUM_CALIBRATION_SAMPLES", "32"))
MAX_SEQ = int(os.environ.get("MAX_SEQUENCE_LENGTH", "8192"))
DATASET = os.environ.get("CALIBRATION_DATASET", "neuralmagic/calibration")
DATASET_CONFIG = os.environ.get("CALIBRATION_DATASET_CONFIG", "LLM")
PIPELINE = os.environ.get("QUANT_PIPELINE", "basic")
DEVICE_MAP = os.environ.get("QUANT_DEVICE_MAP", "cuda:0")

IGNORE = ["lm_head", "re:.*visual.*", "re:.*conv1d.*", "re:.*mtp.*"]


def prepare_dataset(processor: AutoProcessor):
    ds = load_dataset(DATASET, DATASET_CONFIG, split=f"train[:{NUM_SAMPLES}]")
    column = "text" if "text" in ds.column_names else ds.column_names[0]
    tokenizer = getattr(processor, "tokenizer", processor)
    def tokenize(row):
        text = row[column]
        return tokenizer(text, truncation=True, max_length=MAX_SEQ, add_special_tokens=True)
    return ds.map(tokenize, remove_columns=ds.column_names)


def graft_mtp(output: Path) -> None:
    from huggingface_hub import hf_hub_download
    donor = Path(hf_hub_download(MTP_DONOR, "model-mtp-bf16.safetensors", revision=MTP_REVISION))
    tensors = load_file(str(donor))
    assert len(tensors) == 15 and all(k.startswith("mtp.") for k in tensors), sorted(tensors)
    target = output / "model-mtp-bf16.safetensors"
    save_file(tensors, str(target), metadata={"format": "pt"})

    index_path = output / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    for name, tensor in tensors.items():
        index["weight_map"][name] = target.name
    index.setdefault("metadata", {})["total_size"] = int(index.get("metadata", {}).get("total_size", 0)) + sum(t.numel() * t.element_size() for t in tensors.values())
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config.setdefault("text_config", {})["mtp_num_hidden_layers"] = 1
    q = config.setdefault("quantization_config", {})
    ignore = q.setdefault("ignore", [])
    for name in tensors:
        module = name.rsplit(".weight", 1)[0]
        if module not in ignore:
            ignore.append(module)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    OFFLOAD.mkdir(parents=True, exist_ok=True)
    with load_offloaded_model(AutoModelForImageTextToText):
        model = AutoModelForImageTextToText.from_pretrained(
            SOURCE,
            revision=SOURCE_REVISION,
            dtype="auto",
            device_map=DEVICE_MAP,
            offload_folder=str(OFFLOAD),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(SOURCE, revision=SOURCE_REVISION, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(SOURCE, revision=SOURCE_REVISION, trust_remote_code=True)
        ds = prepare_dataset(processor)
        recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=IGNORE)
        oneshot(
            model=model,
            processor=processor,
            dataset=ds,
            recipe=recipe,
            max_seq_length=MAX_SEQ,
            num_calibration_samples=NUM_SAMPLES,
            pipeline=PIPELINE,
            sequential_targets="Linear" if PIPELINE == "sequential" else None,
        )
        model.save_pretrained(
            OUTPUT,
            save_compressed=True,
            max_shard_size="5GB",
            save_original_format=False,
        )
        tokenizer.save_pretrained(OUTPUT)
        processor.save_pretrained(OUTPUT)
    graft_mtp(OUTPUT)
    shutil.rmtree(OFFLOAD, ignore_errors=True)
    print(OUTPUT)

if __name__ == "__main__":
    main()
