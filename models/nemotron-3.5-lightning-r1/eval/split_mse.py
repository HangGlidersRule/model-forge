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

exp, gpath, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
G = json.loads(gpath.read_text())["mse"]
EXPERT = re.compile(r"^backbone\.layers\.(\d+)\.mixer\.(experts\.\d+|shared_experts)\.(up|down)_proj\.input_scale$")


def stream_of(m: re.Match) -> str:
    layer, kind, proj = m.groups()
    return f"L{layer}.up" if proj == "up" else (f"L{layer}.rdown" if kind.startswith("experts") else f"L{layer}.sdown")


if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
idx = json.loads((exp / "model.safetensors.index.json").read_text())
wm = idx["weight_map"]
new_map, scales = {}, {}
for f in sorted(set(wm.values())):
    tensors = {}
    with safe_open(exp / f, "pt") as h:
        meta = h.metadata() or {"format": "pt"}
        for k in h.keys():
            m = EXPERT.match(k)
            if m:
                scales[k] = torch.full_like(h.get_tensor(k), G[stream_of(m)] / 2688.0)
            else:
                tensors[k] = h.get_tensor(k)
    save_file(tensors, out / f, metadata=meta)
    new_map.update({k: f for k in tensors})
    print(f, len(tensors), "tensors", flush=True)
if len(scales) != 5934:
    raise SystemExit(f"expected 5,934 expert input scales, found {len(scales)}")
save_file(scales, out / "model-input-scales.safetensors", metadata={"format": "pt"})
new_map.update({k: "model-input-scales.safetensors" for k in scales})
idx["weight_map"] = dict(sorted(new_map.items()))
(out / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2) + "\n")
for p in exp.iterdir():
    if p.is_file() and not p.name.endswith(".safetensors") and p.name != "model.safetensors.index.json":
        shutil.copy2(p, out / p.name)
assert len(new_map) == len(wm), (len(new_map), len(wm))
print(json.dumps({"expert_input_scales_set_to_mse": len(scales), "tensors_total": len(new_map), "out": str(out)}))
