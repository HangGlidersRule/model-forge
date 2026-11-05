#!/usr/bin/env python3
"""Lay the export's expert activation scales (routed + shared up/down input_scale, 5,934 tensors) into one small shard,
model-input-scales.safetensors, set to the R17-A1 mse rule (input_scale = G / 2688 per stream), exactly as in the
v1.1.0 cell. Every other tensor (weights, weight scales, Mamba FP8 input scales, lm_head, KV scales) is copied unchanged
into the big shards; non-weight files are copied as they are.
Usage: split_mse.py EXPORT_DIR VARIANTS_G_JSON OUT_DIR"""
import json
import os
import re
import shutil
import stat
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


def read_checkpoint_file(path: Path, dir_fd: int) -> bytes:
    """Read ``path``, which must be a regular file and the entry ``path.name`` of the directory open as ``dir_fd``.

    A link is refused, including one swapped in after the directory was listed, and so is a file reached through a
    parent that was swapped for a link: the opened inode must be the one that directory lists under that name."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as e:
        raise SystemExit(f"{path}: {e.strerror} (checkpoint files must be regular files, not links)") from e
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or not os.path.samestat(
            st, os.stat(path.name, dir_fd=dir_fd, follow_symlinks=False)
        ):
            raise SystemExit(f"{path}: not a regular file in {path.parent}")
        with open(fd, "rb", closefd=False) as f:
            return f.read()
    finally:
        os.close(fd)


def write_output_file(path: Path, data: bytes) -> None:
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644), "wb") as f:
        f.write(data)


exp_fd = os.open(exp, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
idx = json.loads(read_checkpoint_file(exp / "model.safetensors.index.json", exp_fd))
sidecars = {
    name: read_checkpoint_file(exp / name, exp_fd)
    for name in sorted(os.listdir(exp_fd))
    if not name.endswith(".safetensors") and name != "model.safetensors.index.json"
    and not stat.S_ISDIR(os.stat(name, dir_fd=exp_fd, follow_symlinks=False).st_mode)
}
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
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
for name, data in sidecars.items():
    write_output_file(out / name, data)
assert len(new_map) == len(wm), (len(new_map), len(wm))
print(json.dumps({"expert_input_scales_set_to_mse": len(scales), "tensors_total": len(new_map), "out": str(out)}))
