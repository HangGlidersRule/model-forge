#!/usr/bin/env python3
"""Ripple-65B Preview (#39): tensor bytes per category of one or more checkpoints, from the safetensors headers only.

Reads the 8-byte header length and the JSON header of every shard (no tensor data), so it runs on any host with Python 3
and needs no GPU or numpy. Categories follow the posted plan: MLP (gate/up/down), attention (q/k/v/o/g), lm_head,
embeddings, other (norms, value_res_lambda). Every tensor of a linear counts toward it (packed weight, block scales,
global and input scales).
Reads stay inside each checkpoint directory (eval/checkpoint_files.py, as the export validator reads them): the index's
shard names are plain .safetensors file names, the directory's .safetensors files are exactly the indexed shards
(model.safetensors alone without an index), every shard is opened with O_NOFOLLOW as a regular file and read through
that descriptor, and a header length above 100,000,000 bytes or past the end of its shard is refused before it is read.
Usage: checkpoint_sizes.py OUT_JSON LABEL=CHECKPOINT_DIR [LABEL=CHECKPOINT_DIR ...]
Prints one line per checkpoint and writes OUT_JSON {label: {"tensor_bytes", "tensors", "gib": {...}, "files"}}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checkpoint_files as cf  # noqa: E402

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
    try:
        with cf.Shards(ckpt) as shards:
            tensors = shards.tensors()
            files = len(shards.names)
    except cf.ShardError as e:
        raise SystemExit(f"{ckpt.name}: {e}") from None
    for name, (_, _, size, _, _) in tensors.items():
        per[category(name)] = per.get(category(name), 0) + size
    total = sum(per.values())
    return {"tensor_bytes": total, "tensors": len(tensors), "files": files,
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
