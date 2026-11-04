#!/usr/bin/env python3
"""Fail-closed structural validation of a Nemotron-3.5-Rerank-8B ModelOpt export (issue #32). CPU only.

Usage: validate_export.py EXPORT_DIR SOURCE_DIR CANDIDATE(A|B|C) OUT_JSON

Checks, all of which must pass:
  1. Coverage. Text-tower linears quantized exactly as the candidate says, across 34 layers:
       A: q/k/v/o and gate/up/down all NVFP4 (238 linears)
       B: gate/up/down NVFP4 (102), q/k/v/o FP8 (136)
       C: gate/up/down NVFP4 (102), attention BF16
     An NVFP4 linear has a U8 packed weight, an F8_E4M3 block scale, and F32 weight_scale_2 and input_scale.
     An FP8 linear has an F8_E4M3 weight and F32 weight_scale and input_scale.
  2. Scale health. Every scalar scale is finite and > 0. Every FP8 block scale is finite, and zero block scales are
     counted.
  3. Protection. Every tensor outside the quantized linears (vision tower, projector, embeddings, norms, score head,
     and BF16 attention for C) is present with the source's name, dtype, shape and exact bytes, after mapping export
     names back to source names.
  4. Sidecars. model.py, chat_template.jinja, tokenizer files and the sentence-transformers files are byte-identical
     to the source. config.json equals the source apart from quantization_config.
  5. hf_quant_config.json names the right algorithm and keeps the KV cache unquantized.
Prints VALIDATE-OK or VALIDATE-FAIL and writes a JSON report."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

EXP, SRC, CAND, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
LAYERS = 34
ATTN, MLP = ("q_proj", "k_proj", "v_proj", "o_proj"), ("gate_proj", "up_proj", "down_proj")
PLAN = {"A": {**{p: "nvfp4" for p in ATTN}, **{p: "nvfp4" for p in MLP}},
        "B": {**{p: "fp8" for p in ATTN}, **{p: "nvfp4" for p in MLP}},
        "C": {**{p: "bf16" for p in ATTN}, **{p: "nvfp4" for p in MLP}}}[CAND]
errors: list[str] = []
report: dict = {"candidate": CAND, "plan": PLAN}


def tensors(d: Path) -> dict[str, tuple[str, Path]]:
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return {k: ("", d / v) for k, v in wm.items()}
    return {k: ("", d / "model.safetensors") for k in safe_open(str(d / "model.safetensors"), "pt").keys()}


src_t, exp_t = tensors(SRC), tensors(EXP)
# The source stores the text tower as model.language_model.layers.N.*; the exporter may write model.language_model.*,
# language_model.model.*, or the same. Map every export name to a canonical source name.
CANON = [(re.compile(r"^language_model\.model\."), "model.language_model."),
         (re.compile(r"^model\.language_model\.model\."), "model.language_model."),
         (re.compile(r"^language_model\.score\."), "language_model.score."),
         (re.compile(r"^score\."), "language_model.score.")]


def canon(name: str) -> str:
    for pat, rep in CANON:
        if pat.match(name):
            return pat.sub(rep, name, count=1)
    return name


exp_by_canon = {canon(k): k for k in exp_t}
report["export_tensors"], report["source_tensors"] = len(exp_t), len(src_t)
lin = re.compile(r"^model\.language_model\.layers\.(\d+)\.(?:self_attn|mlp)\.(" + "|".join(ATTN + MLP) + r")\.(weight|weight_scale|weight_scale_2|input_scale)$")
handles: dict[Path, Any] = {}


def get(path: Path, name: str):
    if path not in handles:
        handles[path] = safe_open(str(path), "pt")
    return handles[path].get_slice(name), handles[path]


seen_q: dict[tuple[int, str], dict[str, str]] = {}
nan_blocks = zero_blocks = total_blocks = 0
for cname, ename in sorted(exp_by_canon.items()):
    m = lin.match(cname)
    if not m:
        continue
    layer, proj, part = int(m.group(1)), m.group(2), m.group(3)
    sl, h = get(exp_t[ename][1], ename)
    seen_q.setdefault((layer, proj), {})[part] = sl.get_dtype()
    if part == "weight_scale" and sl.get_dtype() == "F8_E4M3":
        v = h.get_tensor(ename).to(torch.float32)
        total_blocks += v.numel()
        nan_blocks += int((~torch.isfinite(v)).sum())
        zero_blocks += int((v == 0).sum())
    if part in ("weight_scale_2", "input_scale") or (part == "weight_scale" and sl.get_dtype() == "F32"):
        v = h.get_tensor(ename).to(torch.float32)
        if not bool(torch.isfinite(v).all()) or not bool((v > 0).all()):
            errors.append(f"{cname}: non-finite or non-positive scale")
counts = {"nvfp4": 0, "fp8": 0, "bf16": 0}
for layer in range(LAYERS):
    for proj, want in PLAN.items():
        parts = seen_q.get((layer, proj), {})
        if want == "nvfp4":
            ok = parts.get("weight") == "U8" and parts.get("weight_scale") == "F8_E4M3" and \
                parts.get("weight_scale_2") == "F32" and parts.get("input_scale") == "F32"
        elif want == "fp8":
            ok = parts.get("weight") == "F8_E4M3" and parts.get("weight_scale") == "F32" and parts.get("input_scale") == "F32"
        else:
            ok = parts.get("weight") == "BF16" and set(parts) == {"weight"}
        counts[want] += int(ok)
        if not ok:
            errors.append(f"layer {layer} {proj}: expected {want}, found {parts}")
report["coverage"] = counts
report["fp8_block_scales"] = {"total": total_blocks, "non_finite": nan_blocks, "zero": zero_blocks}
if nan_blocks:
    errors.append(f"{nan_blocks} non-finite FP8 block scales")

# 3. Protected tensors: every source tensor that is not a quantized linear's weight must be byte-identical in the export.
protected = identical = 0
for sname, (_, spath) in sorted(src_t.items()):
    m = lin.match(sname)
    if m and PLAN[m.group(2)] != "bf16":
        continue
    protected += 1
    ename = exp_by_canon.get(sname)
    if ename is None:
        errors.append(f"protected tensor missing from export: {sname}")
        continue
    ss, sh = get(spath, sname)
    es, eh = get(exp_t[ename][1], ename)
    if ss.get_dtype() != es.get_dtype() or ss.get_shape() != es.get_shape():
        errors.append(f"{sname}: dtype/shape {es.get_dtype()} {es.get_shape()} vs source {ss.get_dtype()} {ss.get_shape()}")
        continue
    a, b = sh.get_tensor(sname), eh.get_tensor(ename)
    if not torch.equal(a.view(torch.uint8) if a.element_size() == 1 else a.contiguous().view(torch.int16) if a.element_size() == 2
                       else a.contiguous().view(torch.int32), b.view(torch.uint8) if b.element_size() == 1 else
                       b.contiguous().view(torch.int16) if b.element_size() == 2 else b.contiguous().view(torch.int32)):
        errors.append(f"{sname}: bytes differ from source")
        continue
    identical += 1
report["protected_tensors"] = {"expected": protected, "byte_identical": identical}
quant_names = {canon(k) for k in exp_t if lin.match(canon(k))}
extra = sorted(c for c in exp_by_canon if c not in src_t and c not in quant_names)
if extra:
    errors.append(f"{len(extra)} export tensors with no source counterpart, e.g. {extra[:4]}")


# 4. Sidecars
def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


for f in ("model.py", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "processor_config.json",
          "modules.json", "config_sentence_transformers.json", "sentence_bert_config.json", "1_Pooling/config.json",
          "2_Dense/config.json", "2_Dense/model.safetensors"):
    if (SRC / f).exists():
        if not (EXP / f).exists():
            errors.append(f"sidecar missing: {f}")
        elif sha(SRC / f) != sha(EXP / f):
            errors.append(f"sidecar differs from source: {f}")
sc, ec = json.loads((SRC / "config.json").read_text()), json.loads((EXP / "config.json").read_text())
qc = ec.pop("quantization_config", None)
if sc != ec:
    diff = sorted(k for k in set(sc) | set(ec) if sc.get(k) != ec.get(k))
    errors.append(f"config.json differs from source outside quantization_config: {diff}")
if qc is None:
    errors.append("config.json has no quantization_config")
report["config_quantization_config"] = qc

# 5. hf_quant_config.json
hq = json.loads((EXP / "hf_quant_config.json").read_text())["quantization"]
report["hf_quant_config"] = hq
want_algo = "MIXED_PRECISION" if CAND == "B" else "NVFP4"
if hq.get("quant_algo") != want_algo:
    errors.append(f"hf_quant_config quant_algo {hq.get('quant_algo')} != {want_algo}")
if hq.get("kv_cache_quant_algo") not in (None, "NONE"):
    errors.append(f"KV cache quantized: {hq.get('kv_cache_quant_algo')}")
# vLLM's pooling adapter keeps an lm_head placeholder and names the score head `score`; all of them must be excluded or
# vLLM builds them as NVFP4 layers (the placeholder then fails to load: "Tensor.item() cannot be called on meta tensors").
for where, names in (("hf_quant_config.exclude_modules", hq.get("exclude_modules") or []),
                     ("config.quantization_config.ignore", (qc or {}).get("ignore") or [])):
    for need in ("score", "language_model.score", "lm_head", "language_model.lm_head"):
        if need not in names:
            errors.append(f"{where} lacks {need!r}")

report["errors"] = errors
report["verdict"] = "PASS" if not errors else "FAIL"
OUT.write_text(json.dumps(report, indent=1))
print(("VALIDATE-OK " if not errors else "VALIDATE-FAIL ") + json.dumps(
    {k: report[k] for k in ("candidate", "coverage", "fp8_block_scales", "protected_tensors")}), flush=True)
for e in errors[:12]:
    print("  ERROR", e, flush=True)
sys.exit(0 if not errors else 1)
