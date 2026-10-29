#!/usr/bin/env python3
"""Explain the export check's scale-level mismatches. For every NVFP4 linear, compare the FP8 block scale the training
forward used (ModelOpt StaticBlockScaleQuantizer._block_scale_from_amax math on the learned amax, on CUDA as in training)
with the FP8 block scale ModelOpt's exporter wrote. Classify every mismatched block:
  tie:   the exact value amax*448/global_amax lies within 1e-6 (relative) of the midpoint between the two E4M3 codes and
         the codes are adjacent: the two code paths compute the same product in a different float order and round a
         midpoint case to different neighbours
  sat:   amax*448/global_amax > 448 (learned amax above the frozen per-tensor amax)
  other: anything else (a real export discrepancy)
Usage: verify_export_scales.py BEST_AMAX_PT B_EXPORT NEW_EXPORT
"""
import collections
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/scripts")
import qad_train as q  # noqa: E402
from modelopt.torch.quantization.nn.modules.tensor_quantizer import _FP8_E4M3_MIN_POSITIVE, _amax_to_scale  # noqa: E402
from modelopt.torch.quantization.tensor_quant import scaled_e4m3  # noqa: E402

best = torch.load(sys.argv[1], weights_only=True)  # tensors only, written by qad_train.py
ref, new = q.ExportReader(Path(sys.argv[2])), q.ExportReader(Path(sys.argv[3]))
dev = torch.device("cuda")
F8 = 448.0
cnt = collections.Counter()
total = 0
examples, per = [], []
for name in sorted(best):
    a = best[name].to(dev).float().reshape(-1)
    g = q.exact_preimage(ref.get(f"{name}.weight_scale_2").float(), 6 * F8).to(dev)
    pts = _amax_to_scale(g, 6.0)
    minv = (_FP8_E4M3_MIN_POSITIVE / F8) * pts.view(-1)
    s_fwd = scaled_e4m3(_amax_to_scale(a, 6.0, min_value=minv), pts, None, 4, 3).float()
    code_fwd = s_fwd / pts * F8
    code_exp = new.get(f"{name}.weight_scale").to(dev).float().reshape(-1)
    bad = (code_fwd - code_exp).abs() > 1e-5 * code_exp.abs().clamp_min(1e-30)
    total += a.numel()
    nb = int(bad.sum())
    if not nb:
        continue
    per.append((nb, name))
    x = a[bad].double() * F8 / g.double()
    cf, ce = code_fwd[bad].double(), code_exp[bad].double()
    # code_fwd is reconstructed through float32 division, so snap both to the E4M3 grid before comparing codes;
    # positive E4M3 bit patterns are monotonic in value, so adjacent codes differ by exactly 1 in their uint8 bits.
    cf8, ce8 = cf.float().to(torch.float8_e4m3fn), ce.float().to(torch.float8_e4m3fn)
    adjacent = (cf8.view(torch.uint8).int() - ce8.view(torch.uint8).int()).abs() == 1
    mid = (cf8.double() + ce8.double()) / 2
    near_mid = ((x - mid).abs() / x) < 1e-6
    sat = x > F8
    tie = near_mid & adjacent & ~sat
    cnt["tie"] += int(tie.sum())
    cnt["sat"] += int(sat.sum())
    other = ~tie & ~sat
    cnt["other"] += int(other.sum())
    for j in torch.nonzero(other).flatten().tolist()[:3]:
        if len(examples) < 10:
            examples.append({"class": "other", "linear": name.replace("model.language_model.", ""),
                             "x_exact": float(x[j]), "code_fwd": float(cf8[j].float()), "code_exp": float(ce8[j].float()),
                             "rel_dist_to_midpoint": float((x[j] - mid[j]).abs() / x[j])})
    for j in range(min(1, nb)):
        if len(examples) < 4:
            examples.append({"class": "tie" if bool(tie[j]) else "other", "linear": name.replace("model.language_model.", ""),
                             "x_exact": float(x[j]), "code_fwd": float(cf8[j].float()), "code_exp": float(ce8[j].float()),
                             "rel_dist_to_midpoint": float((x[j] - mid[j]).abs() / x[j])})
per.sort(reverse=True)
print(json.dumps({"linears": len(best), "blocks": total, "mismatched_blocks": sum(cnt.values()),
                  "fraction": sum(cnt.values()) / total, "by_class": dict(cnt),
                  "worst_linears": [f"{n.replace('model.language_model.', '')}:{b}" for b, n in per[:5]],
                  "examples": examples}, indent=1))
print("SCALE-CHECK " + ("BENIGN" if cnt["other"] == 0 and cnt["sat"] == 0 else "REAL-DIFFERENCES"))
