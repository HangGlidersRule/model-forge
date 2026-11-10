#!/usr/bin/env python3
"""Independent provenance check of a repaired Nano-Omni checkpoint against its sources (issue #35). Stdlib only.

For every tensor of REPAIRED, classify where its bytes come from:
  - router:   a float32 router e_score_correction_bias; must equal UPSTREAM's bytes (upstream = NVIDIA BF16 checkpoint);
  - tower:    vision / sound / projector (mlp1, sound_projection); must equal UPSTREAM's bytes;
  - trunk:    everything else; must equal PUBLISHED's bytes (the checkpoint as published, same tensor name with or
              without the 'language_model.' prefix the wrapper adds).
Also checks the shape and dtype of each tensor, and that no tensor of PUBLISHED is missing from REPAIRED. Writes a JSON
report with counts and verdict PASS/FAIL. Usage: verify_omni_repair.py UPSTREAM PUBLISHED REPAIRED OUT_JSON"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

LM = "language_model."
TOWER_PREFIXES = ("vision_model.", "sound_encoder.", "sound_projection.", "mlp1.")


def header(p: Path) -> tuple[dict, int]:
    with p.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def tensors(root: Path) -> dict[str, tuple[Path, dict, int]]:
    idx = root / "model.safetensors.index.json"
    shards = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]
    out: dict[str, tuple[Path, dict, int]] = {}
    for s in shards:
        h, off = header(root / s)
        for name, info in h.items():
            out[name] = (root / s, info, off)
    return out


def read(entry: tuple[Path, dict, int]) -> bytes:
    path, info, off = entry
    b, e = info["data_offsets"]
    with path.open("rb") as f:
        f.seek(off + b)
        return f.read(e - b)


def norm(name: str) -> str:
    return name[len(LM):] if name.startswith(LM) else name


def main() -> int:
    up_dir, pub_dir, rep_dir, out = (Path(a) for a in sys.argv[1:5])
    up, pub, rep = tensors(up_dir), tensors(pub_dir), tensors(rep_dir)
    up_n = {norm(k): k for k in up}
    pub_n = {norm(k): k for k in pub}
    counts = {"router": [0, 0], "tower": [0, 0], "trunk": [0, 0]}  # [identical, differs]
    problems: list[str] = []
    for name, entry in sorted(rep.items()):
        info = entry[1]
        n = norm(name)
        if n.endswith("e_score_correction_bias"):
            kind, src = "router", up.get(up_n.get(n, ""))
        elif n.startswith(TOWER_PREFIXES):
            kind, src = "tower", up.get(up_n.get(n, ""))
        else:
            kind, src = "trunk", pub.get(pub_n.get(n, ""))
        if src is None:
            problems.append(f"{name}: no {kind} source tensor")
            continue
        if (src[1]["dtype"], src[1]["shape"]) != (info["dtype"], info["shape"]):
            problems.append(f"{name}: {info['dtype']}{info['shape']} vs source {src[1]['dtype']}{src[1]['shape']}")
            continue
        same = read(entry) == read(src)
        counts[kind][0 if same else 1] += 1
        if not same:
            problems.append(f"{name}: {kind} bytes differ from source")
    rep_n = {norm(k) for k in rep}
    missing = sorted(k for k in pub_n if k not in rep_n)
    problems += [f"published tensor missing from repaired: {k}" for k in missing[:20]]
    report = {
        "repaired": rep_dir.name,
        "tensors": len(rep),
        "router_biases": {"identical_to_upstream": counts["router"][0], "differ": counts["router"][1]},
        "towers": {"identical_to_upstream": counts["tower"][0], "differ": counts["tower"][1]},
        "trunk": {"identical_to_published": counts["trunk"][0], "differ": counts["trunk"][1]},
        "published_tensors_missing": len(missing),
        "problems": problems[:50],
        "verdict": "PASS" if not problems else "FAIL",
    }
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "problems"}), f"problems={len(problems)}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
