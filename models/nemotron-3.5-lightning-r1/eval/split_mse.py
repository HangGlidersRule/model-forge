#!/usr/bin/env python3
"""Lay the export's expert activation scales (routed + shared up/down input_scale, 5,934 tensors) into one small shard,
model-input-scales.safetensors, set to the R17-A1 mse rule (input_scale = G / 2688 per stream), exactly as in the
v1.1.0 cell. Every other tensor (weights, weight scales, Mamba FP8 input scales, lm_head, KV scales) is copied unchanged
into the big shards; non-weight files are copied as they are.
Usage: split_mse.py EXPORT_DIR VARIANTS_G_JSON OUT_DIR"""
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load, save

from model_forge.pipeline import (
    INDEX_NAME,
    OutputDir,
    SourceDir,
    safetensors_metadata,
    shard_names,
)

exp, gpath, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
G = json.loads(gpath.read_text())["mse"]
EXPERT = re.compile(r"^backbone\.layers\.(\d+)\.mixer\.(experts\.\d+|shared_experts)\.(up|down)_proj\.input_scale$")


def stream_of(m: re.Match) -> str:
    layer, kind, proj = m.groups()
    return f"L{layer}.up" if proj == "up" else (f"L{layer}.rdown" if kind.startswith("experts") else f"L{layer}.sdown")


src = SourceDir(exp)
idx = json.loads(src.read_bytes(INDEX_NAME))
wm = shard_names(idx["weight_map"])
shards = {f: src.open(f) for f in sorted(set(wm.values()))}
assets = {}
for name in src.listdir():
    if name.endswith(".safetensors") or name == INDEX_NAME:
        continue
    try:
        assets[name] = src.open(name)
    except (IsADirectoryError, FileNotFoundError):
        continue
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
dst = OutputDir(out)
new_map, scales = {}, {}
for f, shard in shards.items():
    data = shard.read_bytes()
    meta = safetensors_metadata(data) or {"format": "pt"}
    tensors = {}
    for k, t in load(data).items():
        m = EXPERT.match(k)
        if m:
            scales[k] = torch.full_like(t, G[stream_of(m)] / 2688.0)
        else:
            tensors[k] = t
    del data
    dst.write_bytes(f, save(tensors, metadata=meta))
    new_map.update({k: f for k in tensors})
    print(f, len(tensors), "tensors", flush=True)
if len(scales) != 5934:
    raise SystemExit(f"expected 5,934 expert input scales, found {len(scales)}")
dst.write_bytes("model-input-scales.safetensors", save(scales, metadata={"format": "pt"}))
new_map.update({k: "model-input-scales.safetensors" for k in scales})
idx["weight_map"] = dict(sorted(new_map.items()))
dst.write_bytes(INDEX_NAME, (json.dumps(idx, indent=2) + "\n").encode())
for name, asset in assets.items():
    dst.write_bytes(name, asset.read_bytes())
src.close()
assert len(new_map) == len(wm), (len(new_map), len(wm))
print(json.dumps({"expert_input_scales_set_to_mse": len(scales), "tensors_total": len(new_map), "out": str(out)}))
