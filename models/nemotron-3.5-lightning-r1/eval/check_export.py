#!/usr/bin/env python3
"""Issue #17: structural check of a Nemotron-3.5-Lightning W4A4 export (CPU, safetensors headers + a few tensors).
Checks: which linears carry NVFP4 weights (packed U8 + float8 per-block scales + weight_scale_2), which carry an
activation input_scale, that every routed expert x layer has a finite positive input_scale, and that protected
modules (Mamba in/out, attention, router, lm_head, embeddings, MTP) are BF16 and byte-identical to the source.
Usage: check_export.py EXPORT_DIR SOURCE_DIR OUT_JSON"""
import collections
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open

exp, src, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def index(d: Path) -> dict:
    wm = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    return {k: d / v for k, v in wm.items()}


ei, si = index(exp), index(src)
dtypes, shapes = {}, {}
for f in sorted(set(ei.values())):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            sl = h.get_slice(k)
            dtypes[k], shapes[k] = sl.get_dtype(), sl.get_shape()


def group(name: str) -> str:
    g = re.sub(r"\.\d+\.", ".N.", name)
    return re.sub(r"experts\.N\.", "experts.*.", g)


cats = collections.Counter((group(k), dtypes[k]) for k in dtypes)
input_scales = {k: None for k in dtypes if k.endswith("input_scale")}
for f in sorted(set(ei[k] for k in input_scales)):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if k in input_scales:
                input_scales[k] = float(h.get_tensor(k).float().reshape(-1)[0])
bad_in = {k: v for k, v in input_scales.items() if not (v and v > 0 and v == v and v != float("inf"))}
routed = [k for k in input_scales if ".mixer.experts." in k]
layers_with_routed = sorted({int(re.search(r"layers\.(\d+)\.", k).group(1)) for k in routed})
experts_per_layer = collections.Counter(int(re.search(r"layers\.(\d+)\.", k).group(1)) for k in routed if k.endswith("up_proj.input_scale"))

# protected tensors: BF16 in the export and byte-identical to the source
prot_pat = re.compile(r"(mixer\.(in_proj|out_proj|conv1d|gate|q_proj|k_proj|v_proj|o_proj|A_log|D|dt_bias|norm)|lm_head|embeddings|embed_tokens|norm_f|final_norm|^mtp\.)")
prot = [k for k in dtypes if prot_pat.search(k) and not k.endswith(("_scale", "_scale_2", "input_scale"))]
mismatch, checked = [], 0
src_name = {k.replace("model.", "backbone.", 1) if k.startswith("model.") else k: k for k in si}
for k in prot:
    sk = k if k in si else src_name.get(k)
    if sk is None or sk not in si:
        mismatch.append(f"{k}: missing in source")
        continue
    with safe_open(ei[k], "pt") as a, safe_open(si[sk], "pt") as b:
        ta, tb = a.get_tensor(k), b.get_tensor(sk)
        if ta.dtype != tb.dtype or ta.shape != tb.shape or not torch.equal(ta, tb):
            mismatch.append(f"{k}: {ta.dtype}{tuple(ta.shape)} vs {tb.dtype}{tuple(tb.shape)}")
    checked += 1
res = {
    "export": exp.name, "tensors": len(dtypes),
    "input_scales": len(input_scales), "input_scales_nonpositive_or_nonfinite": len(bad_in),
    "routed_expert_input_scales": len(routed), "layers_with_routed_experts": len(layers_with_routed),
    "routed_experts_per_layer_min_max": [min(experts_per_layer.values()), max(experts_per_layer.values())] if experts_per_layer else None,
    "protected_tensors_checked": checked, "protected_mismatch": mismatch[:20], "protected_mismatch_count": len(mismatch),
    "input_scale_range": [min(v for v in input_scales.values() if v), max(v for v in input_scales.values() if v)] if input_scales else None,
}
# weight scales: per-tensor weight_scale_2 finite and positive; per-block FP8 weight_scale finite (E4M3 NaN = 0x7f/0xff)
ws2_bad, blk_total, blk_nan, blk_zero = [], 0, 0, 0
for f in sorted(set(ei.values())):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if k.endswith("weight_scale_2"):
                v = float(h.get_tensor(k).float().reshape(-1)[0])
                if not (v > 0 and v == v and v != float("inf")):
                    ws2_bad.append(k)
            elif k.endswith("weight_scale") and dtypes[k] == "F8_E4M3":
                b = h.get_tensor(k).view(torch.uint8)
                blk_total += b.numel()
                blk_nan += int(((b & 0x7F) == 0x7F).sum())
                blk_zero += int((b == 0).sum())
res.update({"weight_scale_2_bad": len(ws2_bad), "weight_scale_2_bad_examples": ws2_bad[:5],
            "fp8_block_scales": blk_total, "fp8_block_scales_nan": blk_nan, "fp8_block_scales_zero": blk_zero})
res["by_group"] = sorted([f"{n:5d}  {d:14s} {g}" for (g, d), n in cats.items()])
out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "by_group"}, indent=1))
print("\n".join(res["by_group"]))
