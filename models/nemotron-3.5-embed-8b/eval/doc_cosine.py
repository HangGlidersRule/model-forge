"""Document-vector agreement between serves of the same model (issue #33): cosine of each document's vector under one
serve vs the Nemotron BF16 serve, per set, at full dimension. Aggregates only. Usage: doc_cosine.py VEC_DIR OUT_JSON"""
import json
import sys
from pathlib import Path

import numpy as np

vec_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
res: dict = {}
for set_name in ("mteb-nano-full-corpora", "lmeb-dialogue-scene-retrieval"):
    base = dict(np.load(vec_dir / f"{set_name}--nemotron-bf16.npz"))
    res[set_name] = {}
    for lab in ("nemotron-bf16r", "nemotron-A", "nemotron-C", "nemotron-D"):
        other = dict(np.load(vec_dir / f"{set_name}--{lab}.npz"))
        cs = []
        for k, v in base.items():
            if not k.endswith("|docs"):
                continue
            a = v / np.linalg.norm(v, axis=1, keepdims=True)
            b = other[k] / np.linalg.norm(other[k], axis=1, keepdims=True)
            cs.append((a * b).sum(1))
        c = np.concatenate(cs)
        res[set_name][lab] = {"documents": int(c.size), "mean": round(float(c.mean()), 5), "p1": round(float(np.percentile(c, 1)), 5),
                              "p5": round(float(np.percentile(c, 5)), 5), "min": round(float(c.min()), 5)}
out.write_text(json.dumps(res, indent=1) + "\n")
print("DOC-COSINE", json.dumps(res))
