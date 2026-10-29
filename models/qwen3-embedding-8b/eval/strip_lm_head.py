#!/usr/bin/env python3
"""Make an issue-18 export load the way the BF16 source loads: drop the lm_head tensor that transformers' Qwen3ForCausalLM
loader randomly initialised (the embedding source has no lm_head; the pooling runner never uses it) and its index entry,
and keep every other tensor byte-identical. Writes the cleaned single-file checkpoint in place via a temp file.
Usage: strip_lm_head.py EXPORT_DIR"""
import json
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

exp = Path(sys.argv[1])
files = sorted(exp.glob("*.safetensors"))
removed = []
for f in files:
    with safe_open(str(f), "pt") as h:
        meta = h.metadata()
        keys = list(h.keys())
        if not any(k.startswith("lm_head") for k in keys):
            continue
        tensors = {k: h.get_tensor(k) for k in keys if not k.startswith("lm_head")}
    removed += [k for k in keys if k.startswith("lm_head")]
    tmp = f.with_suffix(".tmp")
    save_file(tensors, str(tmp), metadata=meta)
    tmp.replace(f)
idx = exp / "model.safetensors.index.json"
if idx.exists():
    d = json.loads(idx.read_text())
    d["weight_map"] = {k: v for k, v in d["weight_map"].items() if not k.startswith("lm_head")}
    idx.write_text(json.dumps(d, indent=2) + "\n")
qc = exp / "hf_quant_config.json"
print("removed:", removed)
print("lm_head still listed in exclude_modules (harmless for vLLM's pooling runner):",
      "lm_head" in json.loads(qc.read_text()).get("quantization", {}).get("exclude_modules", []))
