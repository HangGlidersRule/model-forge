#!/usr/bin/env python3
"""Lay the export's expert activation scales (routed + shared up/down input_scale, 5,934 tensors) into one small shard,
model-input-scales.safetensors, set to the R17-A1 mse rule (input_scale = G / 2688 per stream), exactly as in the
v1.1.0 cell. Every other tensor (weights, weight scales, Mamba FP8 input scales, lm_head, KV scales) is copied unchanged
into the big shards; non-weight files are copied as they are.
The index and companion files must be regular files: a symlink aborts before OUT_DIR is touched, never followed.
Usage: split_mse.py EXPORT_DIR VARIANTS_G_JSON OUT_DIR"""
import errno
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

_READ = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
INDEX = "model.safetensors.index.json"


def bind_regular_files(d: Path) -> dict[str, tuple[int, os.stat_result]]:
    """Open the index and every companion file of d once; later reads use these descriptors, not the paths."""
    dir_fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    bound = {}
    try:
        for name in sorted(os.listdir(dir_fd)):
            if name.endswith(".safetensors"):
                continue
            try:
                fd = os.open(name, _READ, dir_fd=dir_fd)
            except OSError as e:
                if e.errno == errno.ELOOP:
                    raise SystemExit(f"{d / name} is a symlink; refusing to copy it") from None
                raise
            st = os.fstat(fd)
            if stat.S_ISREG(st.st_mode):
                bound[name] = (fd, st)
                continue
            os.close(fd)
            if not stat.S_ISDIR(st.st_mode):
                raise SystemExit(f"{d / name} is not a regular file")
    finally:
        os.close(dir_fd)
    if INDEX not in bound:
        raise SystemExit(f"{d / INDEX} is missing")
    return bound


def copy_into(out_fd: int, name: str, fd: int, st: os.stat_result) -> None:
    """copy2 from an open descriptor into a fresh entry of out_fd."""
    dst = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=out_fd)
    with os.fdopen(fd, "rb") as fsrc, os.fdopen(dst, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, 1 << 24)
        os.fchmod(fdst.fileno(), stat.S_IMODE(st.st_mode))
        fdst.flush()
        os.utime(fdst.fileno(), ns=(st.st_atime_ns, st.st_mtime_ns))


exp, gpath, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
bound = bind_regular_files(exp)
index_fd, _ = bound.pop(INDEX)
with os.fdopen(index_fd, "rb") as f:
    idx = json.loads(f.read())
G = json.loads(gpath.read_text())["mse"]
EXPERT = re.compile(r"^backbone\.layers\.(\d+)\.mixer\.(experts\.\d+|shared_experts)\.(up|down)_proj\.input_scale$")


def stream_of(m: re.Match) -> str:
    layer, kind, proj = m.groups()
    return f"L{layer}.up" if proj == "up" else (f"L{layer}.rdown" if kind.startswith("experts") else f"L{layer}.sdown")


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
out_fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
for name, (fd, st) in bound.items():
    copy_into(out_fd, name, fd, st)
os.close(out_fd)
assert len(new_map) == len(wm), (len(new_map), len(wm))
print(json.dumps({"expert_input_scales_set_to_mse": len(scales), "tensors_total": len(new_map), "out": str(out)}))
