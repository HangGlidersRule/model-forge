#!/usr/bin/env python3
"""CPU meta-device dry run of the issue-35 recipes on the real Nano-Omni wrapper (no weights, no GPU): builds the
wrapper from its config with trust_remote_code on the meta device, applies each recipe without calibration, and counts
the enabled quantizers by module kind, role and format. Fails closed on any quantizer outside the language model.
Usage: dryrun_omni.py SOURCE LETTER=RECIPE [LETTER=RECIPE ...]"""
import collections
import sys

import modelopt.torch.quantization as mtq
import torch
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.nn import TensorQuantizer
from transformers import AutoConfig, AutoModelForCausalLM

src = sys.argv[1]
recipes = dict(a.split("=", 1) for a in sys.argv[2:])
ok = True
for letter, path in recipes.items():
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True, attn_implementation="eager")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    qcfg = load_recipe(path).quantize.model_dump()
    mtq.quantize(model, dict(qcfg, algorithm=None), forward_loop=None)
    counts, outside = collections.Counter(), []
    for name, q in model.named_modules():
        if not isinstance(q, TensorQuantizer) or not q.is_enabled:
            continue
        kind = next((k for k in ("shared_experts", "experts", "in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
                                 "lm_head", "k_bmm", "v_bmm") if k in name), "other")
        role = "weight" if "weight_quantizer" in name else "input" if "input_quantizer" in name else "kv"
        nb = getattr(q, "num_bits", None)
        fmt = "nvfp4" if nb == (2, 1) else "fp8" if nb == (4, 3) else str(nb)
        counts[(kind, role, fmt)] += 1
        if not name.startswith("language_model."):
            outside.append(name)
    print(f"== {letter} ({path.rsplit('/', 1)[-1]}): {sum(counts.values())} enabled quantizers", flush=True)
    for k, v in sorted(counts.items()):
        print(f"   {k}: {v}", flush=True)
    if outside:
        ok = False
        print(f"   OUTSIDE the language model: {len(outside)}, e.g. {outside[:4]}", flush=True)
print("DRY-RUN-OK" if ok else "DRY-RUN-FAIL", flush=True)
sys.exit(0 if ok else 1)
