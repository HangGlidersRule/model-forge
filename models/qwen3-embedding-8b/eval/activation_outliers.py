#!/usr/bin/env python3
"""Summarise the calibrated NVFP4 activation amax values of an export from its ModelOpt `.quant_summary.txt`.
Usage: activation_outliers.py EXPORT_DIR OUT_JSON"""
import json
import re
import statistics
import sys
from pathlib import Path

exp, out = Path(sys.argv[1]), Path(sys.argv[2])
pat = re.compile(r"^model\.layers\.(\d+)\.(mlp|self_attn)\.(\w+)_proj\.input_quantizer\s.*amax=([0-9.eE+-]+)")
amax: dict[str, float] = {}
for line in (exp / ".quant_summary.txt").read_text().splitlines():
    m = pat.match(line)
    if m:
        amax[f"layers.{m.group(1)}.{m.group(2)}.{m.group(3)}_proj"] = float(m.group(4))
by_group: dict[str, list[float]] = {}
for k, v in amax.items():
    by_group.setdefault(k.split(".")[2], []).append(v)
top = sorted(amax.items(), key=lambda kv: -kv[1])[:3]
res = {"input_quantizers": len(amax), "median_amax": statistics.median(amax.values()), "max_amax": max(amax.values()),
       "largest": [{"linear": k, "amax": v} for k, v in top],
       "by_group": {g: {"median_amax": statistics.median(v), "max_amax": max(v)} for g, v in sorted(by_group.items())}}
out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res))
