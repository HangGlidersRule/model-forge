#!/usr/bin/env python3
"""Which checkpoint did a Nano-Omni W4A16 product's BF16 attention projections come from? (issue #35). Stdlib only.

The W4A16 products quantize the MoE experts and lm_head and keep every attention q/k/v/o projection in BF16, so those
24 tensors identify the BF16 parent byte for byte: an unedited base build carries upstream's, an abliterated build
carries the abliteration edit's. Usage: attention_source.py CHECKPOINT UPSTREAM_BF16 ABLITERATED_BF16
Prints one JSON object: projection count and how many are byte-identical to each candidate parent."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

LM = "language_model."


def tensors(root: Path) -> dict[str, tuple[Path, dict, int]]:
    index = root / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    out: dict[str, tuple[Path, dict, int]] = {}
    for shard in shards:
        with (root / shard).open("rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        for name, info in header.items():
            out[name[len(LM):] if name.startswith(LM) else name] = (root / shard, info, 8 + n)
    return out


def read(entry: tuple[Path, dict, int]) -> bytes:
    path, info, offset = entry
    begin, end = info["data_offsets"]
    with path.open("rb") as f:
        f.seek(offset + begin)
        return f.read(end - begin)


def main() -> int:
    ckpt, upstream, ablit = (Path(a) for a in sys.argv[1:4])
    c, u, a = tensors(ckpt), tensors(upstream), tensors(ablit)
    names = sorted(k for k in c if any(k.endswith(f".mixer.{p}_proj.weight") for p in "qkvo") and c[k][1]["dtype"] == "BF16")
    same_upstream = sum(1 for k in names if k in u and read(c[k]) == read(u[k]))
    same_ablit = sum(1 for k in names if k in a and read(c[k]) == read(a[k]))
    print(json.dumps({"checkpoint": ckpt.name, "bf16_attention_projections": len(names),
                      "identical_to_upstream_bf16": same_upstream, "identical_to_abliterated_bf16": same_ablit}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
