#!/usr/bin/env python3
"""Lay the export's expert activation scales (routed + shared up/down input_scale, 5,934 tensors) into one small shard,
model-input-scales.safetensors, set to the R17-A1 mse rule (input_scale = G / 2688 per stream), exactly as in the
v1.1.0 cell. Every other tensor (weights, weight scales, Mamba FP8 input scales, lm_head, KV scales) is copied unchanged
into the big shards; non-weight files are copied as they are.
Every export file used is opened once, without following symlinks, before OUT_DIR is touched; a symlink stops the run.
Shards are read, and non-weight files copied, through those descriptors.
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

exp, gpath, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
G = json.loads(gpath.read_text())["mse"]
EXPERT = re.compile(r"^backbone\.layers\.(\d+)\.mixer\.(experts\.\d+|shared_experts)\.(up|down)_proj\.input_scale$")


def stream_of(m: re.Match) -> str:
    layer, kind, proj = m.groups()
    return f"L{layer}.up" if proj == "up" else (f"L{layer}.rdown" if kind.startswith("experts") else f"L{layer}.sdown")


def fd_path(fd):
    """Procfs path that opens, or links, exactly the inode FD holds. Built from parts because the public export rewrites
    absolute path literals."""
    return os.path.join(os.sep, "proc", "self", "fd", str(fd))


def open_entry(dfd, name, where):
    """Open NAME in directory DFD without following a symlink. Returns None for a directory or special file."""
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dfd)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise SystemExit(f"refusing symlink {where / name}: inputs must be regular files") from None
        raise
    if stat.S_ISREG(os.fstat(fd).st_mode):
        return fd
    os.close(fd)
    return None


def copy_bound(fd, dst):
    """Copy the file open as FD to DST with its mode and timestamps, as shutil.copy2 would."""
    st = os.fstat(fd)
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(fd, "rb", closefd=False) as r, open(dst, "wb") as w:
        shutil.copyfileobj(r, w)
    os.chmod(dst, stat.S_IMODE(st.st_mode))
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))


exp_fd = os.open(exp, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
names = set(os.listdir(exp_fd))
idx_fd = open_entry(exp_fd, "model.safetensors.index.json", exp) if "model.safetensors.index.json" in names else None
if idx_fd is None:
    raise SystemExit(f"{exp / 'model.safetensors.index.json'} is not a regular file")
with os.fdopen(idx_fd, "rb", closefd=False) as r:
    idx = json.loads(r.read())
wm = idx["weight_map"]
shard_fds = {}
for f in sorted(set(wm.values())):
    fd = open_entry(exp_fd, f, exp) if f in names else None
    if fd is None:
        raise SystemExit(f"shard {f} named by the index is not a regular file in {exp}")
    shard_fds[f] = fd
extra_fds = {}
for name in sorted(names):
    if not name.endswith(".safetensors") and name != "model.safetensors.index.json":
        fd = open_entry(exp_fd, name, exp)
        if fd is not None:
            extra_fds[name] = fd

if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
new_map, scales = {}, {}
for f, fd in shard_fds.items():
    tensors = {}
    with safe_open(fd_path(fd), "pt") as h:
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
for name, fd in extra_fds.items():
    copy_bound(fd, out / name)
assert len(new_map) == len(wm), (len(new_map), len(wm))
print(json.dumps({"expert_input_scales_set_to_mse": len(scales), "tensors_total": len(new_map), "out": str(out)}))
