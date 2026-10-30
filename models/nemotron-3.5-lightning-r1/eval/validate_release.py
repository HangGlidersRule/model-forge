#!/usr/bin/env python3
"""Issue #17 release validator (fail-closed) for the Nemotron-3.5-Lightning W4A4 NVFP4 cell.

Checks the exact directory to be uploaded against the pinned BF16 source and the pre-registered scale rule:
  - index <-> file consistency; no safetensors outside the index
  - quantized set = routed + shared expert up/down projections of the 23 MoE layers (5,934 modules), each with packed
    U8 weight, F8_E4M3 block scale, F32 weight_scale_2 and F32 input_scale; nothing else quantized
  - scales: weight_scale_2 / input_scale finite and > 0; FP8 block scales: no NaN (0x7F/0xFF), no zero
  - activation scales equal G/2688 of the selected variant, one value per layer and projection
  - every non-quantized tensor byte-identical to the source; every source tensor accounted for
  - config.json identical to the source outside quantization_config; generation_config.json identical
  - hf_quant_config.json: NVFP4, group 16, no KV quant, exclude set == config ignore set; no quantized module excluded;
    every protected linear-like module excluded
  - companion files byte-identical to the source
Writes OUT_JSON (verdict, counts, release file list with sha256). Exit 1 on any failure.
Usage: validate_release.py ART_DIR SRC_DIR VARIANTS_G_JSON VARIANT OUT_JSON"""
import collections
import fnmatch
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open

art, src, gpath, variant, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
fails: list[str] = []
DROP = {".quant_summary.txt", "config.transformers5.json", "generation_config.transformers5.json",
        "SHA256SUMS.upstream", "SOURCE_REVISION", "README.md"}
COMPANION = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "LICENSE",
             "bias.md", "explainability.md", "privacy.md", "safety.md", "accuracy_plot.png",
             "agentic_coding_benchmarks.png", ".gitattributes"]
LINEAR_LEAVES = {"in_proj", "out_proj", "conv1d", "q_proj", "k_proj", "v_proj", "o_proj", "gate", "lm_head",
                 "eh_proj", "up_proj", "down_proj"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def index(d: Path) -> dict:
    return {k: d / v for k, v in json.loads((d / "model.safetensors.index.json").read_text())["weight_map"].items()}


ei, si = index(art), index(src)
dt, shp = {}, {}
for f in sorted(set(ei.values())):
    with safe_open(f, "pt") as h:
        keys = set(h.keys())
        for k in keys:
            sl = h.get_slice(k)
            dt[k], shp[k] = sl.get_dtype(), sl.get_shape()
    mapped = {k for k, v in ei.items() if v == f}
    if keys != mapped:
        fails.append(f"{f.name}: {len(keys ^ mapped)} tensors differ between index and file")
if {p.name for p in art.glob("*.safetensors")} != {p.name for p in set(ei.values())}:
    fails.append("safetensors files outside the index")

# quantized set
q_mods = sorted(k[: -len(".weight_scale_2")] for k in dt if k.endswith(".weight_scale_2"))
qpat = re.compile(r"^backbone\.layers\.(\d+)\.mixer\.(experts\.\d+|shared_experts)\.(up|down)_proj$")
bad_q = [m for m in q_mods if not qpat.match(m)]
if bad_q:
    fails.append(f"unexpected quantized modules: {bad_q[:5]}")
n_routed = sum(".experts." in m for m in q_mods)
n_shared = sum(".shared_experts." in m for m in q_mods)
if (n_routed, n_shared) != (5888, 46):
    fails.append(f"quantized counts routed {n_routed} shared {n_shared}, expected 5888 / 46")
qnames = set()
for m in q_mods:
    for suf, d in ((".weight", "U8"), (".weight_scale", "F8_E4M3"), (".weight_scale_2", "F32"), (".input_scale", "F32")):
        qnames.add(m + suf)
        if dt.get(m + suf) != d:
            fails.append(f"{m}{suf}: dtype {dt.get(m + suf)} != {d}")
stray = [k for k in dt if dt[k] in ("U8", "F8_E4M3") and k not in qnames]
if stray:
    fails.append(f"packed/FP8 tensors outside the quantized set: {stray[:5]}")

# scales
G = json.loads(gpath.read_text())[variant]


def stream_of(name: str) -> str:
    m = re.match(r"backbone\.layers\.(\d+)\.mixer\.(experts\.\d+|shared_experts)\.(up|down)_proj\.input_scale$", name)
    layer, kind, proj = m.groups()
    return f"L{layer}.up" if proj == "up" else (f"L{layer}.rdown" if kind.startswith("experts") else f"L{layer}.sdown")


scale_bad, blk, blk_nan, blk_zero, in_mismatch = [], 0, 0, 0, []
distinct = collections.defaultdict(set)
for f in sorted(set(ei.values())):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if k.endswith(("weight_scale_2", "input_scale")):
                v = h.get_tensor(k).float()
                if not bool(torch.isfinite(v).all()) or bool((v <= 0).any()):
                    scale_bad.append(k)
                if k.endswith("input_scale"):
                    x = float(v.reshape(-1)[0])
                    s = stream_of(k)
                    distinct[(s, "shared" if "shared_experts" in k else "routed")].add(x)
                    want = G[s] / 2688.0
                    if abs(x - want) > 1e-6 * want:
                        in_mismatch.append(k)
            elif k.endswith("weight_scale") and dt[k] == "F8_E4M3":
                b = h.get_tensor(k).view(torch.uint8)
                blk += b.numel()
                blk_nan += int(((b & 0x7F) == 0x7F).sum())
                blk_zero += int((b == 0).sum())
if scale_bad:
    fails.append(f"non-finite or non-positive scales: {scale_bad[:5]}")
if blk_nan or blk_zero:
    fails.append(f"FP8 block scales: {blk_nan} NaN, {blk_zero} zero")
if in_mismatch:
    fails.append(f"{len(in_mismatch)} input_scale values differ from the {variant} rule, e.g. {in_mismatch[:3]}")
multi = {f"{s}/{kind}": len(v) for (s, kind), v in distinct.items() if len(v) != 1}
if multi:
    fails.append(f"activation scale not shared within a layer/projection: {multi}")

# protected tensors byte-identical to the source; every source tensor accounted for
prot = [k for k in dt if k not in qnames]
missing_src = [k for k in prot if k not in si]
if missing_src:
    fails.append(f"export tensors absent from the source: {missing_src[:5]}")
pairs = collections.defaultdict(list)
for k in prot:
    if k in si:
        pairs[(ei[k], si[k])].append(k)
mism, checked = [], 0
for (fa, fs), ks in pairs.items():
    with safe_open(fa, "pt") as a, safe_open(fs, "pt") as b:
        for k in ks:
            ta, tb = a.get_tensor(k), b.get_tensor(k)
            if ta.dtype != tb.dtype or ta.shape != tb.shape or not torch.equal(ta, tb):
                mism.append(k)
            checked += 1
if mism:
    fails.append(f"{len(mism)} protected tensors differ from the source: {mism[:5]}")
unaccounted = [k for k in si if k not in dt and not (k.endswith(".weight") and k[: -len(".weight")] in set(q_mods))]
if unaccounted:
    fails.append(f"{len(unaccounted)} source tensors missing from the export: {unaccounted[:5]}")

# configs
cs = json.loads((src / "config.json").read_text())
ca = json.loads((art / "config.json").read_text())
qc = ca.pop("quantization_config", None)
if not qc:
    fails.append("config.json has no quantization_config")
if ca != cs:
    fails.append(f"config.json differs from the source in {sorted(k for k in set(ca) | set(cs) if ca.get(k) != cs.get(k))}")
if json.loads((art / "generation_config.json").read_text()) != json.loads((src / "generation_config.json").read_text()):
    fails.append("generation_config.json differs from the source")
hq = json.loads((art / "hf_quant_config.json").read_text())["quantization"]
if (hq.get("quant_algo"), hq.get("group_size"), hq.get("kv_cache_quant_algo")) != ("NVFP4", 16, None):
    fails.append(f"hf_quant_config: {hq.get('quant_algo')} group {hq.get('group_size')} kv {hq.get('kv_cache_quant_algo')}")
excl = list(hq.get("exclude_modules") or [])
if qc and (qc.get("quant_algo") != "NVFP4" or set(qc.get("ignore") or []) != set(excl)):
    fails.append("config.json quantization_config disagrees with hf_quant_config.json")
excluded_q = [m for m in q_mods if any(fnmatch.fnmatchcase(m, p) for p in excl)]
if excluded_q:
    fails.append(f"quantized modules matched by an exclusion: {excluded_q[:5]}")
prot_lin = sorted({k[: -len(".weight")] for k in prot if k.endswith(".weight") and k[: -len(".weight")].rsplit(".", 1)[-1] in LINEAR_LEAVES})
not_excl = [m for m in prot_lin if not any(fnmatch.fnmatchcase(m, p) for p in excl)]

# companion + inventory
comp_bad = [n for n in COMPANION if not (art / n).exists() or not (src / n).exists() or sha(art / n) != sha(src / n)]
if comp_bad:
    fails.append(f"companion files differ from or missing vs the source: {comp_bad}")
present = sorted(p.name for p in art.iterdir() if p.is_file())
release = [n for n in present if n not in DROP]
known = set(COMPANION) | {"config.json", "generation_config.json", "hf_quant_config.json", "model.safetensors.index.json"} \
    | {p.name for p in set(ei.values())} | DROP
unknown = [n for n in present if n not in known]
if unknown:
    fails.append(f"unclassified files: {unknown}")
res = {
    "artifact": str(art), "variant": variant, "verdict": "PASS" if not fails and not not_excl else "FAIL",
    "failures": fails, "tensors": len(dt), "quantized_modules": len(q_mods), "routed": n_routed, "shared": n_shared,
    "fp8_block_scales": blk, "fp8_block_scales_nan": blk_nan, "fp8_block_scales_zero": blk_zero,
    "input_scale_streams": len({s for s, _ in distinct}), "input_scale_values_checked": sum(1 for k in dt if k.endswith("input_scale")),
    "protected_tensors_checked": checked, "protected_linear_modules": len(prot_lin),
    "protected_linear_not_excluded": not_excl[:40], "protected_linear_not_excluded_count": len(not_excl),
    "dropped_files": sorted(n for n in present if n in DROP),
    "release_files": {n: sha(art / n) for n in release},
}
out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "release_files"}, indent=1))
sys.exit(0 if res["verdict"] == "PASS" else 1)
