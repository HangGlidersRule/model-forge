#!/usr/bin/env python3
"""Ripple-65B Preview (#39): tensor bytes per category of one or more checkpoints, from the safetensors headers only.

Reads the 8-byte header length and the JSON header of every *.safetensors file (no tensor data), so it runs on any host
with Python 3 and needs no GPU or numpy. Categories follow the posted plan: MLP (gate/up/down), attention
(q/k/v/o/g), lm_head, embeddings, other (norms, value_res_lambda). Every tensor of a linear counts toward it (packed
weight, block scales, global and input scales).
Usage: checkpoint_sizes.py OUT_JSON LABEL=CHECKPOINT_DIR [LABEL=CHECKPOINT_DIR ...]
Prints one line per checkpoint and writes OUT_JSON {label: {"tensor_bytes", "tensors", "gib": {...}, "files"}}.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ATTN = (".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.", ".g_proj.")
MLP = (".gate_proj.", ".up_proj.", ".down_proj.")


def category(name: str) -> str:
    if any(k in name for k in MLP):
        return "mlp"
    if any(k in name for k in ATTN):
        return "attention"
    if name.startswith("lm_head."):
        return "lm_head"
    if "embed_tokens" in name:
        return "embeddings"
    return "other"


def sizes(ckpt: Path) -> dict:
    per: dict[str, int] = {}
    count = 0
    files = sorted(ckpt.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors in {ckpt.name}")
    for f in files:
        with f.open("rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            per[category(name)] = per.get(category(name), 0) + (b - a)
            count += 1
    total = sum(per.values())
    return {"tensor_bytes": total, "tensors": count, "files": len(files),
            "bytes": dict(sorted(per.items())), "gib": {k: round(v / 2**30, 2) for k, v in sorted(per.items())}}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    out: dict[str, dict] = {}
    for arg in sys.argv[2:]:
        label, _, path = arg.partition("=")
        out[label] = sizes(Path(path))
        s = out[label]
        print(f"{label}: {s['tensor_bytes'] / 1e9:.2f} GB ({s['tensor_bytes'] / 2**30:.2f} GiB) in {s['files']} files, "
              f"{s['tensors']} tensors | GiB {s['gib']}")
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
