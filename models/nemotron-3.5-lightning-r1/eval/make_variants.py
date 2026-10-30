#!/usr/bin/env python3
"""Issue #17 R17-A1: build activation-scale variants of candidate A from the BF16 activation statistics.
Each variant is BASE (A re-laid by make_split.py) with only model-input-scales.safetensors replaced; every other file
is a hardlink to BASE. input_scale = G / (6 * 448), where G is the global activation amax for that stream.
Rules (per stream; routed and shared up_proj share the up stream):
  mse     G on the half-octave grid with the lowest simulated NVFP4 squared error
  p9999   G = 99.99th percentile of per-block amax (capped at the observed max)
  p999    G = 99.9th percentile of per-block amax
  b12xeq  G = 2688 (input_scale 1.0), the fixed global scale of the b12x runtime path
Usage: make_variants.py STATS_JSON BASE_DIR VROOT"""
import json
import os
import re
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
    for p in base.iterdir():
        if p.is_file() and p.name != "model-input-scales.safetensors":
            dst = vd / p.name
            if dst.exists():
                dst.unlink()
            os.link(p, dst)
    new = {k: torch.full_like(t, g[stream_of(k)] / 2688.0) for k, t in a_scales.items()}
    save_file(new, vd / "model-input-scales.safetensors", metadata={"format": "pt"})
    edge = [s for s in streams if rule == "mse" and g[s] in (GRID[0], GRID[-1])]
    print(json.dumps({"variant": rule, "dir": str(vd), "tensors": len(new), "mse_at_grid_edge": edge}), flush=True)
json.dump(summary, open(vroot / "variants-G.json", "w"), indent=1)
print("G per stream (A | mse | p9999 | p999):")
for s in sorted(streams, key=lambda x: (int(x[1:].split(".")[0]), x)):
    print(f"  {s:9s} {summary['A'][s]:10.4g} {summary['mse'][s]:10.4g} {summary['p9999'][s]:10.4g} {summary['p999'][s]:10.4g}")
