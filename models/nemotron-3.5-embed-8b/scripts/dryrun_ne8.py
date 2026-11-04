#!/usr/bin/env python3
"""CPU dry run of the issue-33 recipes against the real Nemotron-3.5-Embed-8B module tree (meta device, no weights).
Loads each recipe with ModelOpt's own loader, builds the checkpoint's Mistral3Model from its config on the meta device,
applies the recipe's quantizer config without calibration, and counts enabled quantizers by tower (text / vision /
projector / other), projection, role and format. Prints DRY-RUN-OK when every recipe matches its plan.
Usage (inside the PTQ image, CPU): dryrun_ne8.py SOURCE_DIR LETTER=RECIPE [LETTER=RECIPE ...]   (letters A, C, D)"""
import collections
import json
import sys
from pathlib import Path

import modelopt.torch.quantization as mtq
import torch
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.nn import TensorQuantizer
from transformers import AutoConfig, AutoModel

src = Path(sys.argv[1])
recipes = dict(a.split("=", 1) for a in sys.argv[2:])
L = 34
ATTN, MLP = ("q_proj", "k_proj", "v_proj", "o_proj"), ("gate_proj", "up_proj", "down_proj")
want = {
    "A": {("text", p, r, "nvfp4"): L for p in ATTN + MLP for r in ("weight", "input")},
    "C": {("text", p, r, "nvfp4"): L for p in MLP for r in ("weight", "input")},
    "D": {("text", p, "weight", "nvfp4"): L for p in ATTN + MLP},
}
ok_all = True
for cand, path in recipes.items():
    cfg = AutoConfig.from_pretrained(src)
    with torch.device("meta"):
        model = AutoModel.from_config(cfg)
    assert type(model).__name__ == "Mistral3Model", type(model).__name__
    qcfg = load_recipe(path).quantize.model_dump()
    mtq.quantize(model, dict(qcfg, algorithm=None), forward_loop=None)
    got: collections.Counter = collections.Counter()
    for name, q in model.named_modules():
        if not isinstance(q, TensorQuantizer) or not q.is_enabled:
            continue
        tower = ("text" if "language_model" in name else "vision" if "vision_tower" in name
                 else "projector" if "multi_modal_projector" in name else "other")
        proj = next((p for p in ATTN + MLP if p in name), "other")
        role = "weight" if name.endswith("weight_quantizer") else "input" if name.endswith("input_quantizer") else name.split(".")[-1]
        fmt = "nvfp4" if q.num_bits == (2, 1) else "fp8" if q.num_bits == (4, 3) else str(q.num_bits)
        got[(tower, proj, role, fmt)] += 1
    match = dict(got) == want[cand]
    ok_all &= match
    print(f"== {cand} ({Path(path).name}): {'MATCH' if match else 'MISMATCH'} | {sum(got.values())} enabled quantizers", flush=True)
    for k in sorted(set(got) | set(want[cand])):
        if got.get(k, 0) != want[cand].get(k, 0):
            print(f"   {k}: got {got.get(k, 0)} want {want[cand].get(k, 0)}", flush=True)
    print("   algorithm:", json.dumps(qcfg["algorithm"])[:160], flush=True)
print("DRY-RUN-OK" if ok_all else "DRY-RUN-FAIL", flush=True)
sys.exit(0 if ok_all else 1)
