#!/usr/bin/env python3
"""Issue #17 R17-A1: re-lay candidate A's export so every activation input_scale tensor lives in one small shard
(model-input-scales.safetensors). The big shards are rewritten once without them; each scale variant then hardlinks
the big shards and carries only its own small shard. Tensor values are unchanged.
Usage: make_split.py EXPORT_DIR OUT_DIR"""
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

exp, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
idx = json.loads((exp / "model.safetensors.index.json").read_text())
wm = idx["weight_map"]
new_map, scales = {}, {}
for f in sorted(set(wm.values())):
    tensors = {}
    with safe_open(exp / f, "pt") as h:
        meta = h.metadata() or {"format": "pt"}
        for k in h.keys():
            (scales if k.endswith(".input_scale") else tensors)[k] = h.get_tensor(k)
    save_file(tensors, out / f, metadata=meta)
    new_map.update({k: f for k in tensors})
    print(f, len(tensors), "tensors", flush=True)
save_file(scales, out / "model-input-scales.safetensors", metadata={"format": "pt"})
new_map.update({k: "model-input-scales.safetensors" for k in scales})
idx["weight_map"] = dict(sorted(new_map.items()))
(out / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2) + "\n")
for p in exp.iterdir():
    if p.is_file() and not p.name.endswith(".safetensors") and p.name != "model.safetensors.index.json":
        shutil.copy2(p, out / p.name)
assert len(new_map) == len(wm), (len(new_map), len(wm))
print(json.dumps({"input_scales": len(scales), "tensors_total": len(new_map)}))
