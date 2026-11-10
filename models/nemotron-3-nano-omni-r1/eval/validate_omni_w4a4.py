#!/usr/bin/env python3
"""Fail-closed structural validation of an issue-35 Nano-Omni W4A4 export (CPU, stdlib only; safetensors headers and
bytes read directly).

  1. config.json is the upstream wrapper config verbatim apart from quantization_config.
  2. Towers: every non-language_model tensor of the export exists upstream with identical dtype, shape and bytes; every
     upstream tower weight is present (only non-persistent buffers may be absent).
  3. Untouched language-model tensors (router, conv1d, norms, embeddings, SSM parameters, and every linear the candidate
     leaves BF16) are byte-identical to upstream.
  4. quantized_layers by kind and format match the candidate's plan exactly.
  5. Scales: every weight_scale_2 / input_scale / FP8 weight_scale finite and positive; no FP8 block scale is NaN
     (E4M3 0x7F/0xFF) or zero.
  6. Fused groups vLLM builds (q/k/v -> qkv_proj) share input_scale and weight_scale_2 in every layer.
  7. W4A16 (the v1.1.0 W4A16 products): BF16 KV, so no k/v scales and no FP8 KV scheme; no fused group
     (q/k/v) mixes quantized and unquantized members. Pass the BF16 parent as UPSTREAM (upstream BF16 for the base
     build, the Abliterated BF16 for the abliterated one): untouched tensors and towers are compared with it.
Writes a JSON report; prints VALIDATE-OK or VALIDATE-FAIL.
Usage: validate_omni_w4a4.py EXPORT UPSTREAM A|B|W4A16 REPORT_JSON"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

LM = "language_model."
KINDS = ("shared_experts", "experts", "in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "lm_head")
PLAN = {
    "A": {("experts", "NVFP4"): 5888, ("shared_experts", "NVFP4"): 46, ("in_proj", "NVFP4"): 23, ("out_proj", "NVFP4"): 23,
          ("q_proj", "NVFP4"): 6, ("k_proj", "NVFP4"): 6, ("v_proj", "NVFP4"): 6, ("o_proj", "NVFP4"): 6,
          ("lm_head", "W4A16_NVFP4"): 1},
    "B": {("experts", "NVFP4"): 5888, ("shared_experts", "FP8"): 46, ("in_proj", "FP8"): 23, ("out_proj", "FP8"): 23,
          ("o_proj", "FP8"): 6},
    "W4A16": {("experts", "W4A16_NVFP4"): 5888, ("shared_experts", "W4A16_NVFP4"): 46, ("lm_head", "W4A16_NVFP4"): 1},
}
SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale", ".k_scale", ".v_scale")


def header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def index(root: Path) -> dict[str, str]:
    p = root / "model.safetensors.index.json"
    if p.exists():
        return json.loads(p.read_text())["weight_map"]
    return {k: "model.safetensors" for k in header(root / "model.safetensors")[0]}


def read_tensors(root: Path, names) -> dict[str, tuple[str, list, bytes]]:
    idx = index(root)
    by = collections.defaultdict(list)
    for n in names:
        by[idx[n]].append(n)
    out = {}
    for shard, ns in by.items():
        h, off = header(root / shard)
        with (root / shard).open("rb") as f:
            for n in sorted(ns, key=lambda k: h[k]["data_offsets"][0]):
                b, e = h[n]["data_offsets"]
                f.seek(off + b)
                out[n] = (h[n]["dtype"], h[n]["shape"], f.read(e - b))
    return out


def digests(root: Path, names) -> dict[str, tuple[str, list, str]]:
    idx = index(root)
    by = collections.defaultdict(list)
    for n in names:
        by[idx[n]].append(n)
    out = {}
    for shard, ns in by.items():
        h, off = header(root / shard)
        with (root / shard).open("rb") as f:
            for n in sorted(ns, key=lambda k: h[k]["data_offsets"][0]):
                b, e = h[n]["data_offsets"]
                f.seek(off + b)
                d, left = hashlib.sha256(), e - b
                while left:
                    buf = f.read(min(left, 1 << 24))
                    d.update(buf)
                    left -= len(buf)
                out[n] = (h[n]["dtype"], h[n]["shape"], d.hexdigest())
    return out


def floats(dtype: str, raw: bytes) -> list[float]:
    if dtype == "F32":
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))
    if dtype == "BF16":
        return [struct.unpack("<f", b"\x00\x00" + raw[i:i + 2])[0] for i in range(0, len(raw), 2)]
    raise ValueError(dtype)


def kind_of(name: str) -> str:
    return next((k for k in KINDS if k in name.split(".")), "other")


def main() -> int:
    exp, up, cand, rep_path = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    errors: list[str] = []
    rep: dict = {"candidate": cand, "export": exp.name, "parent": up.name}
    e_cfg, u_cfg = json.loads((exp / "config.json").read_text()), json.loads((up / "config.json").read_text())
    drift = sorted(k for k in set(e_cfg) | set(u_cfg) if k != "quantization_config" and e_cfg.get(k) != u_cfg.get(k))
    if drift:
        errors.append(f"config.json differs from upstream outside quantization_config: {drift}")
    qc = e_cfg.get("quantization_config") or {}
    ql = qc.get("quantized_layers") or {}
    counts = collections.Counter((kind_of(k), v.get("quant_algo")) for k, v in ql.items())
    # ModelOpt may list fused routed experts once per layer (`...mixer.experts`); expand that form to per-expert counts.
    fused = [k for k in ql if k.endswith(".mixer.experts")]
    if fused:
        counts[("experts", ql[fused[0]].get("quant_algo"))] += len(fused) * 128 * 2 - len(fused)
    rep["quantized_layers"] = {f"{k}/{a}": n for (k, a), n in sorted(counts.items())}
    rep["kv_cache"] = qc.get("kv_cache_scheme") or qc.get("kv_cache_quant_algo")
    if dict(counts) != PLAN[cand]:
        errors.append(f"quantized_layers {dict(counts)} != plan {PLAN[cand]}")

    e_idx, u_idx = index(exp), index(up)
    e_tow = {k for k in e_idx if not k.startswith(LM)}
    u_tow = {k for k in u_idx if not k.startswith(LM)}
    missing = sorted(u_tow - e_tow)
    extra = sorted(e_tow - u_tow)
    buffers = {k for k in missing if k.endswith(("featurizer.fb", "featurizer.window", "norm_mean", "norm_std"))}
    if set(missing) - buffers or extra:
        errors.append(f"tower tensors: missing {sorted(set(missing) - buffers)[:4]} extra {extra[:4]}")
    ed, ud = digests(exp, e_tow & u_tow), digests(up, e_tow & u_tow)
    bad_tow = [k for k in ed if ed[k] != ud[k]]
    rep["towers"] = {"upstream": len(u_tow), "present": len(e_tow & u_tow), "byte_identical": len(ed) - len(bad_tow),
                     "absent_buffers": sorted(buffers)}
    if bad_tow:
        errors.append(f"{len(bad_tow)} tower tensors differ from upstream, e.g. {bad_tow[:3]}")

    quantized_modules = set(ql)
    if fused:
        quantized_modules |= {f"{f}.{i}.{p}" for f in fused for i in range(128) for p in ("up_proj", "down_proj")}
    untouched = []
    for k in e_idx:
        if not k.startswith(LM) or k.endswith(SCALE_SUFFIXES):
            continue
        mod = k.rsplit(".", 1)[0]
        if mod in quantized_modules or k.endswith(("_bmm_quantizer._amax",)):
            continue
        if k in u_idx:
            untouched.append(k)
    ue, uu = digests(exp, untouched), digests(up, untouched)
    bad_lm = [k for k in untouched if ue[k] != uu[k]]
    rep["untouched_language_model"] = {"checked": len(untouched), "byte_identical": len(untouched) - len(bad_lm)}
    if bad_lm:
        errors.append(f"{len(bad_lm)} untouched language-model tensors differ from upstream, e.g. {bad_lm[:3]}")
    missing_w = [m for m in ql if not m.endswith(".mixer.experts") and f"{m}.weight" not in e_idx]
    if missing_w:
        errors.append(f"{len(missing_w)} quantized modules have no weight, e.g. {missing_w[:3]}")

    scale_names = [k for k in e_idx if k.startswith(LM) and k.endswith((".weight_scale_2", ".input_scale"))]
    vals = read_tensors(exp, scale_names)
    bad_scale = [k for k, (dt, _, raw) in vals.items() if any(not math.isfinite(x) or x <= 0 for x in floats(dt, raw))]
    block = [k for k in e_idx if k.startswith(LM) and k.endswith(".weight_scale")]
    nan = zero = negative = total = 0
    for k, (dt, _, raw) in read_tensors(exp, block).items():
        if dt == "F8_E4M3":
            total += len(raw)
            # E4M3 byte layout S.EEEE.MMM: NaN is S.1111.111 (0x7F/0xFF), +0 and -0 are 0x00/0x80, and every other
            # byte with the sign bit set is a negative scale. A block scale must be finite and strictly positive.
            nan += raw.count(b"\x7f") + raw.count(b"\xff")
            zero += raw.count(b"\x00") + raw.count(b"\x80")
            negative += sum(1 for b in raw if b & 0x80) - raw.count(b"\x80") - raw.count(b"\xff")
        else:
            if any(not math.isfinite(x) or x <= 0 for x in floats(dt, raw)):
                bad_scale.append(k)
    rep["scales"] = {"per_tensor_checked": len(scale_names), "bad_per_tensor": len(bad_scale),
                     "fp8_block_scales": total, "non_finite": nan, "zero": zero, "negative": negative}
    if bad_scale or nan or zero or negative:
        errors.append(f"scale health: {len(bad_scale)} bad per-tensor scales (e.g. {bad_scale[:2]}), {nan} NaN, "
                      f"{zero} zero and {negative} negative block scales")

    groups = mism = 0
    if cand == "A":
        layers = sorted({k.split(".")[3] for k in ql if k.endswith(".mixer.q_proj")}, key=int)
        for L in layers:
            base = f"{LM}backbone.layers.{L}.mixer."
            for s in ("input_scale", "weight_scale_2"):
                names = [f"{base}{p}.{s}" for p in ("q_proj", "k_proj", "v_proj")]
                t = read_tensors(exp, names)
                groups += 1
                if len({t[n][2] for n in names}) != 1:
                    mism += 1
        rep["fused_qkv_shared_scales"] = {"checks": groups, "mismatched": mism}
        if mism:
            errors.append(f"{mism} q/k/v groups do not share their scales (vLLM fuses them into qkv_proj)")
    attn = sorted({k.split(".")[3] for k in e_idx if k.startswith(LM) and k.endswith(".mixer.q_proj.weight")}, key=int)
    mixed = [L for L in attn if 0 < sum(f"{LM}backbone.layers.{L}.mixer.{p}" in quantized_modules
                                         for p in ("q_proj", "k_proj", "v_proj")) < 3]
    rep["fused_qkv_mixed_precision_layers"] = len(mixed)
    rep["quantized_non_language_model"] = sum(1 for k in ql if not k.startswith(LM))
    if rep["quantized_non_language_model"]:
        errors.append(f"{rep['quantized_non_language_model']} quantized layers outside the language model (towers must stay BF16)")
    rep["mtp_tensors"] = sum(1 for k in e_idx if ".mtp." in k or k.startswith("mtp."))
    if mixed:
        errors.append(f"{len(mixed)} attention layers mix quantized and unquantized q/k/v members")
    if cand == "W4A16":
        # KV metadata must match the precision map (BF16 KV: no k/v scales). ModelOpt writes one input_scale on the
        # lm_head of a W4A16 export; it is counted, and checked finite and positive with the other per-tensor scales.
        kv = [k for k in e_idx if k.startswith(LM) and k.endswith((".k_scale", ".v_scale"))]
        rep["kv_scales"] = len(kv)
        rep["input_scales"] = sum(1 for k in e_idx if k.startswith(LM) and k.endswith(".input_scale"))
        if kv:
            errors.append(f"W4A16 product with BF16 KV carries {len(kv)} k/v scales, e.g. {kv[:2]}")
        if "fp8" in json.dumps(rep["kv_cache"]).lower():
            errors.append(f"W4A16 product declares an FP8 KV cache: {rep['kv_cache']}")
    rep["errors"] = errors
    rep["verdict"] = "PASS" if not errors else "FAIL"
    rep_path.write_text(json.dumps(rep, indent=1) + "\n")
    print(("VALIDATE-OK " if not errors else "VALIDATE-FAIL ") + json.dumps({k: rep[k] for k in rep if k not in ("errors", "verdict")}), flush=True)
    for e in errors:
        print("ERROR", e, flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
