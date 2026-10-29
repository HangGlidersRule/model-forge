#!/usr/bin/env python3
"""Count the FP8 block scales a QAD export changed against its start-point export.

For every NVFP4 linear (a `.weight_scale` tensor in both exports), compare the per-16-element FP8 E4M3 block scales
byte for byte, and report how many differ. The per-tensor `weight_scale_2` values are reported separately (QAD keeps
them frozen; at most a 1-ulp float32 drift from rebuilding them is expected).
Usage: count_changed_block_scales.py START_EXPORT QAD_EXPORT OUT_JSON
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


class Reader:
    def __init__(self, root: Path):
        self.root = root
        self.index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles: dict = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.index[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(str(self.root / shard), "pt")
        return self.handles[shard].get_tensor(name)


start, qad, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
a, b = Reader(start), Reader(qad)
names = sorted(k for k in a.index if k.endswith(".weight_scale") and k in b.index)
blocks = changed = 0
per_kind: dict = {}
scale2_changed = []
for name in names:
    sa, sb = a.get(name).view(torch.uint8), b.get(name).view(torch.uint8)
    assert sa.shape == sb.shape, name
    n, c = sa.numel(), int((sa != sb).sum())
    blocks += n
    changed += c
    kind = ".".join(name.split(".")[-3:-1])
    k = per_kind.setdefault(kind, [0, 0])
    k[0] += n
    k[1] += c
    s2 = name[: -len(".weight_scale")] + ".weight_scale_2"
    if not torch.equal(a.get(s2), b.get(s2)):
        scale2_changed.append(s2)
report = {
    "linears": len(names),
    "fp8_block_scales": blocks,
    "changed": changed,
    "fraction_changed": changed / blocks,
    "by_projection": {k: {"blocks": v[0], "changed": v[1], "fraction": v[1] / v[0]} for k, v in sorted(per_kind.items())},
    "weight_scale_2_not_bitwise_equal": len(scale2_changed),
}
out.write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps({k: report[k] for k in ("linears", "fp8_block_scales", "changed", "fraction_changed",
                                         "weight_scale_2_not_bitwise_equal")}))
