#!/usr/bin/env python3
"""CPU dry run of the NVIDIA-layout recipe: load it with ModelOpt's recipe loader, build Lightning's real module tree on
the meta device (transformers 5.18, nemotron_h), apply the quantizer config, and count which quantizers end up enabled
with which format. No weights, no calibration.
Usage (in the PTQ image): recipe_dryrun.py RECIPE SOURCE_DIR"""
import collections
import json
import sys

import modelopt.torch.quantization as mtq
import torch
from modelopt.recipe import load_recipe
from modelopt.torch.quantization.nn import TensorQuantizer
from transformers import AutoConfig, AutoModelForCausalLM

recipe_path, src = sys.argv[1], sys.argv[2]
recipe = load_recipe(recipe_path)
qcfg = recipe.quantize.model_dump()
print("recipe loaded: algorithm", json.dumps(qcfg["algorithm"])[:200])
cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16)
qcfg_no_calib = dict(qcfg)
qcfg_no_calib["algorithm"] = None
mtq.quantize(model, qcfg_no_calib, forward_loop=None)
groups = collections.Counter()
examples = {}
for name, mod in model.named_modules():
    if not isinstance(mod, TensorQuantizer) or not mod.is_enabled:
        continue
    parts = name.split(".")
    kind = ("mtp:" if name.startswith("mtp.") else "") + next(
        (p for p in ("experts", "shared_experts", "in_proj", "out_proj", "lm_head", "k_bmm_quantizer", "v_bmm_quantizer",
                     "q_bmm_quantizer", "p_bmm_quantizer", "q_proj", "k_proj", "v_proj", "o_proj", "gate", "conv1d")
         if p in parts or any(p in x for x in parts)), "other")
    fmt = f"num_bits={mod.num_bits} block={getattr(mod, 'block_sizes', None)}"
    role = "weight" if "weight_quantizer" in name else ("input" if "input_quantizer" in name else parts[-1])
    key = (kind, role, fmt)
    groups[key] += 1
    examples.setdefault(key, name)
for (kind, role, fmt), n in sorted(groups.items()):
    print(f"{n:6d}  {kind:18s} {role:16s} {fmt[:90]}  e.g. {examples[(kind, role, fmt)]}")
