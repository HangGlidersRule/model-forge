#!/usr/bin/env python3
"""Issue #17 R17-A1: re-lay candidate A's export so every activation input_scale tensor lives in one small shard
(model-input-scales.safetensors). The big shards are rewritten once without them; each scale variant then hardlinks
the big shards and carries only its own small shard. Tensor values are unchanged.
Usage: make_split.py EXPORT_DIR OUT_DIR"""
import json
import sys
from pathlib import Path

from safetensors.torch import load, save

from model_forge.checkpoint_files import (
    INDEX_NAME,
    OutputDir,
    SourceDir,
    safetensors_metadata,
    shard_names,
)

exp, out = Path(sys.argv[1]), Path(sys.argv[2])
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
out.mkdir(parents=True, exist_ok=True)
dst = OutputDir(out)
new_map, scales = {}, {}
for f, shard in shards.items():
    data = shard.read_bytes()
    meta = safetensors_metadata(data) or {"format": "pt"}
    tensors = {}
    for k, t in load(data).items():
        (scales if k.endswith(".input_scale") else tensors)[k] = t
    del data
    dst.write_bytes(f, save(tensors, metadata=meta))
    new_map.update({k: f for k in tensors})
    print(f, len(tensors), "tensors", flush=True)
dst.write_bytes("model-input-scales.safetensors", save(scales, metadata={"format": "pt"}))
new_map.update({k: "model-input-scales.safetensors" for k in scales})
idx["weight_map"] = dict(sorted(new_map.items()))
dst.write_bytes(INDEX_NAME, (json.dumps(idx, indent=2) + "\n").encode())
for name, asset in assets.items():
    dst.write_bytes(name, asset.read_bytes())
src.close()
assert len(new_map) == len(wm), (len(new_map), len(wm))
print(json.dumps({"input_scales": len(scales), "tensors_total": len(new_map)}))
