#!/usr/bin/env python3
"""Fail-closed validator for the Lightning W4A4 NVFP4 Mixed-FP8 cell (NVIDIA layout), over the exact directory to ship.
  - index <-> files consistent; no safetensors outside the index
  - quantized sets, exactly:
      experts  5,888 routed + 46 shared up/down: U8 weight, F8_E4M3 weight_scale, F32 weight_scale_2, F32 input_scale
      mamba    46 in_proj/out_proj: F8_E4M3 weight, F32 weight_scale, F32 input_scale
      lm_head  U8 weight, F8_E4M3 weight_scale, F32 weight_scale_2, no input_scale
      kv       one k_scale and one v_scale (F32) per attention layer
    and no other U8 / F8_E4M3 tensor
  - per-tensor scales finite and > 0; FP8 tensors (block scales and FP8 weights): no NaN byte (0x7F/0xFF); block scales
    never zero
  - expert input scales = R17-A1 mse G / 2688; the expert-scale shard is compared with the v1.1.0 cell's
  - expert tensors compared with the v1.1.0 cell (reported, not gated)
  - every other tensor byte-identical to the BF16 source; every source tensor accounted for
  - config.json identical to the source outside quantization_config; generation_config.json identical
  - hf_quant_config.json: MIXED_PRECISION, KV FP8, per-layer algos exactly NVFP4 x 5,934 experts, FP8 x 46 Mamba
    projections, W4A16_NVFP4 x lm_head; config.json's quantization_config agrees
  - companion files byte-identical to the source
Usage: validate_release_nvl.py ART SRC_BF16 V110 VARIANTS_G_JSON OUT_JSON"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open

art, src, v110, gpath, out = (Path(a) for a in sys.argv[1:6])
fails: list[str] = []
DROP = {".quant_summary.txt", "config.transformers5.json", "generation_config.transformers5.json",
        "SHA256SUMS.upstream", "SOURCE_REVISION", "README.md"}
COMPANION = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "LICENSE",
             "bias.md", "explainability.md", "privacy.md", "safety.md", "accuracy_plot.png",
             "agentic_coding_benchmarks.png", ".gitattributes"]
EXP = re.compile(r"^backbone\.layers\.(\d+)\.mixer\.(experts\.\d+|shared_experts)\.(up|down)_proj$")
MAMBA = re.compile(r"^backbone\.layers\.(\d+)\.mixer\.(in_proj|out_proj)$")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def index(d: Path) -> dict:
    return {k: d / v for k, v in json.loads((d / "model.safetensors.index.json").read_text())["weight_map"].items()}


ei, si, vi = index(art), index(src), index(v110)
dt = {}
for f in sorted(set(ei.values())):
    with safe_open(f, "pt") as h:
        keys = set(h.keys())
        for k in keys:
            dt[k] = h.get_slice(k).get_dtype()
    if keys != {k for k, v in ei.items() if v == f}:
        fails.append(f"{f.name}: index and file disagree")
if {p.name for p in art.glob("*.safetensors")} != {p.name for p in set(ei.values())}:
    fails.append("safetensors files outside the index")

# ---- quantized sets
want = {}
experts = sorted(k[: -len(".weight_scale_2")] for k in dt if k.endswith(".weight_scale_2") and EXP.match(k[: -len(".weight_scale_2")]))
for m in experts:
    want.update({m + ".weight": "U8", m + ".weight_scale": "F8_E4M3", m + ".weight_scale_2": "F32", m + ".input_scale": "F32"})
mamba = sorted({k.rsplit(".", 1)[0] for k in dt if MAMBA.match(k.rsplit(".", 1)[0])})
for m in mamba:
    want.update({m + ".weight": "F8_E4M3", m + ".weight_scale": "F32", m + ".input_scale": "F32"})
want.update({"lm_head.weight": "U8", "lm_head.weight_scale": "F8_E4M3", "lm_head.weight_scale_2": "F32"})
attn_layers = sorted({int(k.split(".")[2]) for k in si if re.match(r"^backbone\.layers\.\d+\.mixer\.q_proj\.weight$", k)})
kv = sorted(k for k in dt if k.endswith((".k_scale", ".v_scale")))
for k in kv:
    want[k] = "F32"
n_routed = sum(".experts." in m for m in experts)
n_shared = sum(".shared_experts." in m for m in experts)
if (n_routed, n_shared) != (5888, 46):
    fails.append(f"experts routed {n_routed} shared {n_shared}, expected 5,888 / 46")
if len(mamba) != 46:
    fails.append(f"{len(mamba)} FP8 Mamba projections, expected 46")
kv_layers = sorted({int(k.split(".")[2]) for k in kv})
if len(kv) != 2 * len(attn_layers) or kv_layers != attn_layers:
    fails.append(f"KV scales {len(kv)} on layers {kv_layers}, expected k+v on attention layers {attn_layers}")
for k, d in want.items():
    if dt.get(k) != d:
        fails.append(f"{k}: dtype {dt.get(k)} != {d}")
if "lm_head.input_scale" in dt:
    fails.append("lm_head carries an input_scale (expected weight-only)")
stray = [k for k in dt if dt[k] in ("U8", "F8_E4M3") and k not in want]
if stray:
    fails.append(f"packed/FP8 tensors outside the quantized sets: {stray[:5]}")

# ---- scales
G = json.loads(gpath.read_text())["mse"]


def stream_of(name: str) -> str:
    m = EXP.match(name.rsplit(".", 1)[0])
    layer, kind, proj = m.groups()
    return f"L{layer}.up" if proj == "up" else (f"L{layer}.rdown" if kind.startswith("experts") else f"L{layer}.sdown")


bad_scale, nan_fp8, zero_blk, blk_total, in_mismatch = [], 0, 0, 0, []
for f in sorted(set(ei.values())):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if dt[k] == "F32" and k in want:
                v = h.get_tensor(k).float()
                if not bool(torch.isfinite(v).all()) or bool((v <= 0).any()):
                    bad_scale.append(k)
                if k.endswith(".input_scale") and EXP.match(k.rsplit(".", 1)[0]):
                    x, w = float(v.reshape(-1)[0]), G[stream_of(k)] / 2688.0
                    if abs(x - w) > 1e-6 * w:
                        in_mismatch.append(k)
            elif dt[k] == "F8_E4M3":
                b = h.get_tensor(k).view(torch.uint8)
                nan_fp8 += int(((b & 0x7F) == 0x7F).sum())
                if k.endswith(".weight_scale"):
                    blk_total += b.numel()
                    zero_blk += int((b == 0).sum())
if bad_scale:
    fails.append(f"non-finite or non-positive scales: {bad_scale[:5]}")
if nan_fp8 or zero_blk:
    fails.append(f"FP8 tensors: {nan_fp8} NaN bytes, {zero_blk} zero block scales")
if in_mismatch:
    fails.append(f"{len(in_mismatch)} expert input scales differ from the mse rule, e.g. {in_mismatch[:3]}")
shard_same = (art / "model-input-scales.safetensors").exists() and sha(art / "model-input-scales.safetensors") == sha(v110 / "model-input-scales.safetensors")

# ---- experts vs v1.1.0 (reported)
exp_keys = [k for k in want if EXP.match(k.rsplit(".", 1)[0])]
pairs = collections.defaultdict(list)
for k in exp_keys:
    if k in vi:
        pairs[(ei[k], vi[k])].append(k)
exp_same = exp_diff = 0
diff_examples = []
for (fa, fb), ks in pairs.items():
    with safe_open(fa, "pt") as a, safe_open(fb, "pt") as b:
        for k in ks:
            if torch.equal(a.get_tensor(k), b.get_tensor(k)):
                exp_same += 1
            else:
                exp_diff += 1
                diff_examples.append(k)

# ---- protected tensors vs the BF16 source
prot = [k for k in dt if k not in want]
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
qmods = set(experts) | set(mamba) | {"lm_head"}
unaccounted = [k for k in si if k not in dt and not (k.endswith(".weight") and k[: -len(".weight")] in qmods)]
if unaccounted:
    fails.append(f"{len(unaccounted)} source tensors missing from the export: {unaccounted[:5]}")

# ---- configs
cs = json.loads((src / "config.json").read_text())
ca = json.loads((art / "config.json").read_text())
qc = ca.pop("quantization_config", None) or {}
if ca != cs:
    fails.append(f"config.json differs from the source in {sorted(k for k in set(ca) | set(cs) if ca.get(k) != cs.get(k))}")
if json.loads((art / "generation_config.json").read_text()) != json.loads((src / "generation_config.json").read_text()):
    fails.append("generation_config.json differs from the source")
hq = json.loads((art / "hf_quant_config.json").read_text())["quantization"]
ql = hq.get("quantized_layers") or {}
algos = collections.Counter(v.get("quant_algo") for v in ql.values())
# ModelOpt 0.48 writes the fused routed-expert container once per layer ("backbone.layers.N.mixer.experts"); NVIDIA's
# published checkpoints list every expert projection. vLLM resolves both (ModelOptMixedPrecisionConfig._resolve_quant_algo:
# direct lookup on the RoutedExperts prefix, else any child key). Map each quantized module to the entry that covers it.
ROUTED = re.compile(r"^(backbone\.layers\.\d+\.mixer\.experts)\.\d+\.(up|down)_proj$")


def covering(mod: str) -> str | None:
    if mod in ql:
        return mod
    m = ROUTED.match(mod)
    return m.group(1) if m and m.group(1) in ql else None


want_algo = {**{m: "NVFP4" for m in experts}, **{m: "FP8" for m in mamba}, "lm_head": "W4A16_NVFP4"}
used, wrong, uncovered = set(), [], []
for mod, algo in want_algo.items():
    entry = covering(mod)
    if entry is None:
        uncovered.append(mod)
        continue
    used.add(entry)
    if ql[entry].get("quant_algo") != algo:
        wrong.append(f"{mod} -> {entry}: {ql[entry].get('quant_algo')} != {algo}")
extra = sorted(set(ql) - used)
if hq.get("quant_algo") != "MIXED_PRECISION" or hq.get("kv_cache_quant_algo") != "FP8":
    fails.append(f"hf_quant_config quant_algo {hq.get('quant_algo')} kv {hq.get('kv_cache_quant_algo')}")
if uncovered or wrong or extra:
    fails.append(f"quantized_layers: {len(uncovered)} uncovered {uncovered[:3]}, {len(wrong)} wrong algo {wrong[:3]}, "
                 f"{len(extra)} entries for nothing quantized {extra[:3]}")
if any(v.get("quant_algo") in ("NVFP4", "W4A16_NVFP4") and v.get("group_size") != 16 for v in ql.values()):
    fails.append("an NVFP4 layer without group_size 16")
if qc.get("quant_algo") != "MIXED_PRECISION" or ("quantized_layers" in qc and qc["quantized_layers"] != ql):
    fails.append("config.json quantization_config disagrees with hf_quant_config.json")

# ---- companion files + inventory
comp_bad = [n for n in COMPANION if not (art / n).exists() or not (src / n).exists() or sha(art / n) != sha(src / n)]
if comp_bad:
    fails.append(f"companion files differ from or missing vs the source: {comp_bad}")
present = sorted(p.name for p in art.iterdir() if p.is_file())
known = set(COMPANION) | {"config.json", "generation_config.json", "hf_quant_config.json", "model.safetensors.index.json"} \
    | {p.name for p in set(ei.values())} | DROP
unknown = [n for n in present if n not in known]
if unknown:
    fails.append(f"unclassified files: {unknown}")
release = [n for n in present if n not in DROP]
res = {
    "artifact": str(art), "verdict": "PASS" if not fails else "FAIL", "failures": fails, "tensors": len(dt),
    "experts": len(experts), "routed": n_routed, "shared": n_shared, "mamba_fp8": len(mamba), "lm_head": dt.get("lm_head.weight"),
    "kv_scales": len(kv), "attention_layers": attn_layers, "fp8_block_scales": blk_total, "fp8_nan_bytes": nan_fp8,
    "zero_block_scales": zero_blk, "expert_input_scales_mse": len(exp_keys) // 4 - len(in_mismatch),
    "expert_scale_shard_identical_to_v1_1_0": shard_same,
    "expert_tensors_identical_to_v1_1_0": exp_same, "expert_tensors_differing_from_v1_1_0": exp_diff,
    "expert_differing_examples": diff_examples[:5], "protected_tensors_checked": checked,
    "per_layer_algos": dict(algos), "dropped_files": sorted(n for n in present if n in DROP),
    "release_file_count": len(release), "release_files": {n: sha(art / n) for n in release},
}
out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "release_files"}, indent=1))
sys.exit(0 if res["verdict"] == "PASS" else 1)
