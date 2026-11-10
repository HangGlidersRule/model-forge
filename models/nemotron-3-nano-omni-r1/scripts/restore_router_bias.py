#!/usr/bin/env python3
"""Restore a published Nano-Omni checkpoint's float32 tensors to the upstream bytes (issue #35, router correction
biases). Byte-level, stdlib only; no torch.

For every float32 tensor of UPSTREAM (the routers' e_score_correction_bias, 23 MoE layers), find the same tensor in CKPT
(names compared with any 'language_model.' prefix removed). The tensor must exist there with the same dtype (F32) and
shape; its bytes are replaced by the upstream bytes. Every other byte of every shard is copied unchanged, so offsets,
headers and all other tensors stay identical. Non-weight files are copied, except manifest.sha256, which is rewritten.

Fails closed on a missing tensor, a dtype or shape mismatch, or OUT already existing. After writing it re-reads OUT:
every tensor must be byte-identical to CKPT except the restored ones, which must equal UPSTREAM.
Prints RESTORE-OK with counts. Usage: restore_router_bias.py UPSTREAM CKPT OUT"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

LM = "language_model."


def norm(name: str) -> str:
    return name[len(LM):] if name.startswith(LM) else name


def header(p: Path) -> tuple[dict, int]:
    with p.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def weight_map(root: Path) -> dict[str, str]:
    idx = root / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    return {k: "model.safetensors" for k in header(root / "model.safetensors")[0]}


def read(root: Path, wm: dict[str, str], name: str) -> tuple[str, list, bytes]:
    h, off = header(root / wm[name])
    info = h[name]
    b, e = info["data_offsets"]
    with (root / wm[name]).open("rb") as f:
        f.seek(off + b)
        return info["dtype"], info["shape"], f.read(e - b)


def sha(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            d.update(c)
    return d.hexdigest()


def main() -> int:
    up, ck, out = (Path(a) for a in sys.argv[1:4])
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    uwm, cwm = weight_map(up), weight_map(ck)
    f32 = []
    for shard in sorted(set(uwm.values())):
        f32 += [k for k, v in header(up / shard)[0].items() if v["dtype"] == "F32" and k.startswith(LM)]
    by_norm = {norm(k): k for k in cwm}
    plan: dict[str, dict[str, bytes]] = {}
    for k in sorted(f32):
        ck_name = by_norm.get(norm(k))
        if ck_name is None:
            raise SystemExit(f"RESTORE-FAILED: {k} not in {ck.name}")
        ud, us, ub = read(up, uwm, k)
        cd, cs, cb = read(ck, cwm, ck_name)
        if (ud, us) != (cd, cs):
            raise SystemExit(f"RESTORE-FAILED: {ck_name} is {cd}{cs} in the checkpoint, {ud}{us} upstream")
        if cb != ub:
            plan.setdefault(cwm[ck_name], {})[ck_name] = ub
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for p in sorted(ck.rglob("*")):
        rel = p.relative_to(ck)
        if p.is_dir() or rel.name in ("manifest.sha256", "SHA256SUMS.upstream", "SOURCE_REVISION") or rel.parts[0] == ".cache":
            continue
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        if rel.name not in plan:
            shutil.copyfile(p, tmp / rel)
            continue
        h, off = header(p)
        shutil.copyfile(p, tmp / rel)
        with (tmp / rel).open("r+b") as f:
            for name, new in plan[rel.name].items():
                b, e = h[name]["data_offsets"]
                if e - b != len(new):
                    raise SystemExit(f"RESTORE-FAILED: size mismatch for {name}")
                f.seek(off + b)
                f.write(new)
    restored = sum(len(v) for v in plan.values())
    # Re-read and verify every tensor.
    owm = weight_map(tmp)
    bad = []
    for shard in sorted(set(cwm.values())):
        h, off = header(ck / shard)
        h2, off2 = header(tmp / shard)
        if h != h2:
            bad.append(f"header changed in {shard}")
            continue
        with (ck / shard).open("rb") as a, (tmp / shard).open("rb") as b_:
            for name, info in h.items():
                s, e = info["data_offsets"]
                a.seek(off + s)
                b_.seek(off2 + s)
                old, new = a.read(e - s), b_.read(e - s)
                want = plan.get(shard, {}).get(name)
                if want is not None:
                    if new != want:
                        bad.append(f"{name} not restored")
                elif new != old:
                    bad.append(f"{name} changed")
    if owm != cwm:
        bad.append("weight map changed")
    if bad:
        raise SystemExit(f"RESTORE-FAILED verification: {bad[:5]}")
    files = sorted(p for p in tmp.rglob("*") if p.is_file())
    (tmp / "manifest.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(tmp)}\n" for p in files))
    tmp.rename(out)
    print(f"RESTORE-OK {out} upstream_f32={len(f32)} restored={restored} already_identical={len(f32) - restored} "
          f"shards_rewritten={len(plan)} files={len(files) + 1}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
