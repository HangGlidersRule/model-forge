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
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from model_forge.pipeline import CheckpointTree, OutputDir  # noqa: E402

exp, gpath, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
G = json.loads(gpath.read_text())["mse"]
EXPERT = re.compile(r"^backbone\.layers\.(\d+)\.mixer\.(experts\.\d+|shared_experts)\.(up|down)_proj\.input_scale$")


def stream_of(m: re.Match) -> str:
    layer, kind, proj = m.groups()
    return f"L{layer}.up" if proj == "up" else (f"L{layer}.rdown" if kind.startswith("experts") else f"L{layer}.sdown")


# Every input is opened (no symlinks, regular files only) before anything is removed or written.
tree = CheckpointTree(exp)
idx = json.loads(tree.open("model.safetensors.index.json").read_bytes())
wm = idx["weight_map"]
shards = {f: tree.open(f) for f in sorted(set(wm.values()))}
companions = {name: tree.open(name) for name in tree.listdir()[0]
              if not name.endswith(".safetensors") and name != "model.safetensors.index.json"}
if out.exists():
    shutil.rmtree(out)
dst = OutputDir.create(out)
new_map, scales = {}, {}
for f, shard in shards.items():
    tensors = {}
    with safe_open(shard.path, "pt") as h:
        meta = h.metadata() or {"format": "pt"}
        for k in h.keys():
            m = EXPERT.match(k)
            if m:
                scales[k] = torch.full_like(h.get_tensor(k), G[stream_of(m)] / 2688.0)
            else:
                tensors[k] = h.get_tensor(k)
    save_file(tensors, dst.path(f), metadata=meta)
    new_map.update({k: f for k in tensors})
    print(f, len(tensors), "tensors", flush=True)
if len(scales) != 5934:
    raise SystemExit(f"expected 5,934 expert input scales, found {len(scales)}")
save_file(scales, dst.path("model-input-scales.safetensors"), metadata={"format": "pt"})
new_map.update({k: "model-input-scales.safetensors" for k in scales})
idx["weight_map"] = dict(sorted(new_map.items()))
dst.write_text("model.safetensors.index.json", json.dumps(idx, indent=2) + "\n")
for name, member in companions.items():
    dst.copy(name, member, keep_metadata=True)
assert len(new_map) == len(wm), (len(new_map), len(wm))
print(json.dumps({"expert_input_scales_set_to_mse": len(scales), "tensors_total": len(new_map), "out": str(out)}))
