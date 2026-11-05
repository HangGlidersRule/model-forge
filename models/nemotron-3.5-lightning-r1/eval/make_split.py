#!/usr/bin/env python3
"""Issue #17 R17-A1: re-lay candidate A's export so every activation input_scale tensor lives in one small shard
(model-input-scales.safetensors). The big shards are rewritten once without them; each scale variant then hardlinks
the big shards and carries only its own small shard. Tensor values are unchanged.
Usage: make_split.py EXPORT_DIR OUT_DIR"""
import json
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from model_forge.pipeline import CheckpointTree, OutputDir  # noqa: E402

exp, out = Path(sys.argv[1]), Path(sys.argv[2])
# Every input is opened (no symlinks, regular files only) before anything is written.
tree = CheckpointTree(exp)
idx = json.loads(tree.open("model.safetensors.index.json").read_bytes())
wm = idx["weight_map"]
shards = {f: tree.open(f) for f in sorted(set(wm.values()))}
companions = {name: tree.open(name) for name in tree.listdir()[0]
              if not name.endswith(".safetensors") and name != "model.safetensors.index.json"}
dst = OutputDir.create(out)
new_map, scales = {}, {}
for f, shard in shards.items():
    tensors = {}
    with safe_open(shard.path, "pt") as h:
        meta = h.metadata() or {"format": "pt"}
        for k in h.keys():
            (scales if k.endswith(".input_scale") else tensors)[k] = h.get_tensor(k)
    save_file(tensors, dst.path(f), metadata=meta)
    new_map.update({k: f for k in tensors})
    print(f, len(tensors), "tensors", flush=True)
save_file(scales, dst.path("model-input-scales.safetensors"), metadata={"format": "pt"})
new_map.update({k: "model-input-scales.safetensors" for k in scales})
idx["weight_map"] = dict(sorted(new_map.items()))
dst.write_text("model.safetensors.index.json", json.dumps(idx, indent=2) + "\n")
for name, member in companions.items():
    dst.copy(name, member, keep_metadata=True)
assert len(new_map) == len(wm), (len(new_map), len(wm))
print(json.dumps({"input_scales": len(scales), "tensors_total": len(new_map)}))
