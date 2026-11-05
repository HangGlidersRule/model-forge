#!/usr/bin/env python3
"""Issue #17 R17-A1: build activation-scale variants of candidate A from the BF16 activation statistics.
Each variant is BASE (A re-laid by make_split.py) with only model-input-scales.safetensors replaced; every other file
is a hardlink to BASE. input_scale = G / (6 * 448), where G is the global activation amax for that stream.
Rules (per stream; routed and shared up_proj share the up stream):
  mse     G on the half-octave grid with the lowest simulated NVFP4 squared error
  p9999   G = 99.99th percentile of per-block amax (capped at the observed max)
  p999    G = 99.9th percentile of per-block amax
  b12xeq  G = 2688 (input_scale 1.0), the fixed global scale of the b12x runtime path
BASE files must be regular files: a symlink is refused, never linked into a variant.
Usage: make_variants.py STATS_JSON BASE_DIR VROOT"""
import contextlib
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


def open_regular(path):
    """Open path without following a symlink at its last component; refuse anything but a regular file."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise SystemExit(f"refusing symlink in BASE: {path}") from None
        raise
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise SystemExit(f"refusing non-regular file in BASE: {path}")
    return fd


with safe_open(base / "model-input-scales.safetensors", "pt") as h:
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
    with os.scandir(base) as it:
        names = sorted(e.name for e in it if not e.is_dir(follow_symlinks=False))
    for name in names:
        if name == "model-input-scales.safetensors":
            continue
        fd = open_regular(base / name)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name, dir_fd=vd_fd)
        # linkat(AT_SYMLINK_FOLLOW) on the descriptor's /proc link hardlinks the file that was checked, not the path.
        os.link(f"/proc/self/fd/{fd}", name, dst_dir_fd=vd_fd, follow_symlinks=True)
        os.close(fd)
    os.close(vd_fd)
    new = {k: torch.full_like(t, g[stream_of(k)] / 2688.0) for k, t in a_scales.items()}
    save_file(new, vd / "model-input-scales.safetensors", metadata={"format": "pt"})
    edge = [s for s in streams if rule == "mse" and g[s] in (GRID[0], GRID[-1])]
    print(json.dumps({"variant": rule, "dir": str(vd), "tensors": len(new), "mse_at_grid_edge": edge}), flush=True)
json.dump(summary, open(vroot / "variants-G.json", "w"), indent=1)
print("G per stream (A | mse | p9999 | p999):")
for s in sorted(streams, key=lambda x: (int(x[1:].split(".")[0]), x)):
    print(f"  {s:9s} {summary['A'][s]:10.4g} {summary['mse'][s]:10.4g} {summary['p9999'][s]:10.4g} {summary['p999'][s]:10.4g}")
