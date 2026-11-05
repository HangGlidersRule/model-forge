#!/usr/bin/env python3
"""Issue #17 R17-A1: build activation-scale variants of candidate A from the BF16 activation statistics.
Each variant is BASE (A re-laid by make_split.py) with only model-input-scales.safetensors replaced; every other file
is a hardlink to BASE. input_scale = G / (6 * 448), where G is the global activation amax for that stream.
Rules (per stream; routed and shared up_proj share the up stream):
  mse     G on the half-octave grid with the lowest simulated NVFP4 squared error
  p9999   G = 99.99th percentile of per-block amax (capped at the observed max)
  p999    G = 99.9th percentile of per-block amax
  b12xeq  G = 2688 (input_scale 1.0), the fixed global scale of the b12x runtime path
Every BASE file is opened once, without following symlinks, before any variant is written; a symlink in BASE stops the
run. The scale shard is read, and every hardlink made, through those descriptors, so the variants hold the inodes that
were opened even if a BASE path changes later.
Usage: make_variants.py STATS_JSON BASE_DIR VROOT"""
import errno
import json
import os
import re
import stat
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

st = json.load(open(sys.argv[1]))
base, vroot = Path(sys.argv[2]), Path(sys.argv[3])
GRID, LO, HI, NB = st["grid"], st["hist_lo"], st["hist_hi"], st["hist_bins"]
streams = st["streams"]


def pct(s, q):
    cum, target = s["zero"], q * s["n"]
    for b, c in enumerate(s["hist"]):
        cum += c
        if cum >= target:
            return 2.0 ** (LO + (b + 1) * (HI - LO) / NB)
    return s["max"]


RULES = {
    "mse": lambda s: GRID[min(range(len(GRID)), key=lambda k: s["sse"][k])],
    "p9999": lambda s: min(s["max"], pct(s, 0.9999)),
    "p999": lambda s: min(s["max"], pct(s, 0.999)),
    "b12xeq": lambda s: 2688.0,
}


def stream_of(name):
    m = re.match(r"backbone\.layers\.(\d+)\.mixer\.(experts\.\d+|shared_experts)\.(up|down)_proj\.input_scale$", name)
    if not m:
        raise SystemExit(f"unexpected input_scale tensor {name}")
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


base_fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
bound = {}
for name in sorted(os.listdir(base_fd)):
    fd = open_entry(base_fd, name, base)
    if fd is not None:
        bound[name] = fd
scale_fd = bound.pop("model-input-scales.safetensors", None)
if scale_fd is None:
    raise SystemExit(f"{base / 'model-input-scales.safetensors'} is not a regular file")
with safe_open(fd_path(scale_fd), "pt") as h:
    a_scales = {k: h.get_tensor(k) for k in h.keys()}
missing = sorted({stream_of(k) for k in a_scales} - set(streams))
if missing:
    raise SystemExit(f"no statistics for streams {missing}")
summary = {"A": {}}
for k, t in a_scales.items():
    summary["A"][stream_of(k)] = float(t.float().reshape(-1)[0]) * 2688.0
for rule, fn in RULES.items():
    g = {s: float(fn(streams[s])) for s in streams}
    summary[rule] = g
    vd = vroot / rule
    vd.mkdir(parents=True, exist_ok=True)
    vd_fd = os.open(vd, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    for name, fd in bound.items():
        try:
            os.unlink(name, dir_fd=vd_fd)
        except FileNotFoundError:
            pass
        # linkat(AT_SYMLINK_FOLLOW) on fd_path(fd) links the inode fd holds; CPython only calls linkat, rather than
        # link(2), which would not resolve the descriptor, when a dir_fd is passed.
        os.link(fd_path(fd), name, dst_dir_fd=vd_fd, follow_symlinks=True)
    os.close(vd_fd)
    new = {k: torch.full_like(t, g[stream_of(k)] / 2688.0) for k, t in a_scales.items()}
    save_file(new, vd / "model-input-scales.safetensors", metadata={"format": "pt"})
    edge = [s for s in streams if rule == "mse" and g[s] in (GRID[0], GRID[-1])]
    print(json.dumps({"variant": rule, "dir": str(vd), "tensors": len(new), "mse_at_grid_edge": edge}), flush=True)
json.dump(summary, open(vroot / "variants-G.json", "w"), indent=1)
print("G per stream (A | mse | p9999 | p999):")
for s in sorted(streams, key=lambda x: (int(x[1:].split(".")[0]), x)):
    print(f"  {s:9s} {summary['A'][s]:10.4g} {summary['mse'][s]:10.4g} {summary['p9999'][s]:10.4g} {summary['p999'][s]:10.4g}")
