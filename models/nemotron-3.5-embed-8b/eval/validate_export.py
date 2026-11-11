#!/usr/bin/env python3
"""Fail-closed structural validation of a Nemotron-3.5-Embed-8B ModelOpt export (issue #33). CPU only.
Adapted from the Nemotron-3.5-Rerank-8B validator (#32); the checks are the same, the checkpoint layout differs: a
plain Mistral3Model whose tensors are named language_model.* / vision_tower.* / multi_modal_projector.*, no score head.

Usage: validate_export.py EXPORT_DIR SOURCE_DIR CANDIDATE(A|B|C|D) OUT_JSON

Checks, all of which must pass:
  1. Coverage. Text-tower linears quantized exactly as the candidate says, across 34 layers:
       A: q/k/v/o and gate/up/down all NVFP4 (238 linears)
       B: gate/up/down NVFP4 (102), q/k/v/o FP8 (136)
       C: gate/up/down NVFP4 (102), attention BF16
       D: q/k/v/o and gate/up/down all W4A16 NVFP4 (238 linears, weight-only)
     An NVFP4 linear has a U8 packed weight, an F8_E4M3 block scale, and F32 weight_scale_2 and input_scale.
     An FP8 linear has an F8_E4M3 weight and F32 weight_scale and input_scale.
     A W4A16 NVFP4 linear has the NVFP4 weight, block scale and weight_scale_2, and no input_scale.
  2. Scale health. Every scalar scale is finite and > 0. Every FP8 block scale is finite, and zero block scales are
     counted.
  3. Protection. Every tensor outside the quantized linears (vision tower, projector, embeddings, norms, and BF16
     attention for C) is present with the source's name, dtype, shape and exact bytes, after mapping export
     names back to source names.
  4. Sidecars. chat_template.jinja, tokenizer files and the sentence-transformers files are byte-identical
     to the source. config.json equals the source apart from quantization_config.
  5. hf_quant_config.json names the right algorithm and keeps the KV cache unquantized.
The checks read only the shards of the directory under test: every shard the index names must be one file name, the
.safetensors files at the top of the directory must be exactly those shards (model.safetensors without an index), and
each one must be a regular file there, not a symlink. Anything else fails before a tensor is read.
Prints VALIDATE-OK or VALIDATE-FAIL and writes a JSON report."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
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
        "C": {**{p: "bf16" for p in ATTN}, **{p: "nvfp4" for p in MLP}},
        "D": {p: "w4a16" for p in ATTN + MLP}}[CAND]
errors: list[str] = []
report: dict = {"candidate": CAND, "plan": PLAN}


class ShardError(Exception):
    pass


FD_DIR = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"


class Checkpoint:
    """A checkpoint directory whose shards are each opened once, relative to the directory and without following a
    link, and then read only through that open file (safe_open gets its FD_DIR entry, never the name again)."""

    def __init__(self, root: Path):
        self.root = root
        self._fds: dict[str, int] = {}
        self._handles: dict[str, Any] = {}
        try:
            self._dir = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        except OSError as e:
            raise ShardError(f"{root}: cannot open checkpoint directory ({e.strerror})") from e
        local = {e.name for e in os.scandir(self._dir) if e.name.endswith(".safetensors")}
        wm = self._weight_map()
        named = set(wm.values()) if wm is not None else {"model.safetensors"}
        if local != named:
            raise ShardError(f"{root}: safetensors files are not exactly the indexed shards: unindexed "
                             f"{sorted(local - named)[:4]}, missing {sorted(named - local)[:4]}")
        for name in sorted(named):
            try:
                self._fds[name] = self._open(name)
            except FileNotFoundError as e:
                raise ShardError(f"{root}/{name}: shard disappeared") from e
        self.weight_map = wm if wm is not None else {k: "model.safetensors" for k in self.handle("model.safetensors").keys()}

    def _open(self, name: str) -> int:
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=self._dir)
        except FileNotFoundError:
            raise
        except OSError as e:
            raise ShardError(f"{self.root}/{name}: cannot open as a file in the directory ({e.strerror})") from e
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ShardError(f"{self.root}/{name}: not a regular file")
        return fd

    def _weight_map(self) -> dict[str, str] | None:
        try:
            fd = self._open("model.safetensors.index.json")
        except FileNotFoundError:
            return None
        with os.fdopen(fd, "rb") as f:
            try:
                wm = json.load(f)["weight_map"]
            except (ValueError, KeyError, TypeError) as e:
                raise ShardError(f"{self.root}: unreadable model.safetensors.index.json ({e})") from e
        if not isinstance(wm, dict):
            raise ShardError(f"{self.root}: weight_map is not a mapping")
        for k, v in wm.items():
            if not isinstance(v, str) or v in ("", ".", "..") or "/" in v or "\\" in v or "\0" in v \
                    or not v.endswith(".safetensors"):
                raise ShardError(f"{self.root}: {k} maps to {v!r}, not a .safetensors file name in the directory")
        return wm

    def path(self, shard: str) -> str:
        return f"{FD_DIR}/{self._fds[shard]}"

    def handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = safe_open(self.path(shard), "pt")
        return self._handles[shard]

    def get(self, name: str):
        h = self.handle(self.weight_map[name])
        return h.get_slice(name), h


try:
    src_ck, exp_ck = Checkpoint(SRC), Checkpoint(EXP)
except ShardError as e:
    report.update(errors=[str(e)], verdict="FAIL")
    OUT.write_text(json.dumps(report, indent=1))
    print("VALIDATE-FAIL", json.dumps({"candidate": CAND}), flush=True)
    print("  ERROR", e, flush=True)
    sys.exit(1)
src_t, exp_t = src_ck.weight_map, exp_ck.weight_map
# The source (a plain Mistral3Model) names the text tower language_model.layers.N.*; the exporter may write
# model.language_model.*, language_model.model.* or the same. Map every name, source and export, to that form.
CANON = [(re.compile(r"^model\.language_model\.model\."), "language_model."),
         (re.compile(r"^model\.language_model\."), "language_model."),
         (re.compile(r"^language_model\.model\."), "language_model."),
         (re.compile(r"^model\.vision_tower\."), "vision_tower."),
         (re.compile(r"^model\.multi_modal_projector\."), "multi_modal_projector.")]


def canon(name: str) -> str:
    for pat, rep in CANON:
        if pat.match(name):
            return pat.sub(rep, name, count=1)
    return name


exp_by_canon = {canon(k): k for k in exp_t}
report["export_tensors"], report["source_tensors"] = len(exp_t), len(src_t)
lin = re.compile(r"^language_model\.layers\.(\d+)\.(?:self_attn|mlp)\.(" + "|".join(ATTN + MLP) + r")\.(weight|weight_scale|weight_scale_2|input_scale)$")
seen_q: dict[tuple[int, str], dict[str, str]] = {}
nan_blocks = zero_blocks = total_blocks = 0
for cname, ename in sorted(exp_by_canon.items()):
    m = lin.match(cname)
    if not m:
        continue
    layer, proj, part = int(m.group(1)), m.group(2), m.group(3)
    sl, h = exp_ck.get(ename)
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
counts = {"nvfp4": 0, "w4a16": 0, "fp8": 0, "bf16": 0}
for layer in range(LAYERS):
    for proj, want in PLAN.items():
        parts = seen_q.get((layer, proj), {})
        if want == "nvfp4":
            ok = parts.get("weight") == "U8" and parts.get("weight_scale") == "F8_E4M3" and \
                parts.get("weight_scale_2") == "F32" and parts.get("input_scale") == "F32"
        elif want == "w4a16":
            ok = parts.get("weight") == "U8" and parts.get("weight_scale") == "F8_E4M3" and \
                parts.get("weight_scale_2") == "F32" and "input_scale" not in parts
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

# 1b. Fused groups: vLLM fuses q/k/v and gate/up into one GEMM each and applies one activation scale (and, for NVFP4, one
# weight global scale) to the fused weight, so those scales must be identical across a group's shards.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fused_groups import fused_group_mismatches  # noqa: E402


def read_scalar(cname: str) -> float:
    ename = exp_by_canon[cname]
    _, h = exp_ck.get(ename)
    return float(h.get_tensor(ename).to(torch.float32).item())


fused = fused_group_mismatches(read_scalar, PLAN, LAYERS)
report["fused_group_scales"] = fused
if fused["mismatched"]:
    errors.append(f"{fused['mismatched']} fused groups with differing shared scales, e.g. {fused['mismatches'][:2]}")

# 3. Protected tensors: every source tensor that is not a quantized linear's weight must be byte-identical in the export.
protected = identical = 0
for sname in sorted(src_t):
    m = lin.match(sname)
    if m and PLAN[m.group(2)] != "bf16":
        continue
    protected += 1
    ename = exp_by_canon.get(sname)
    if ename is None:
        errors.append(f"protected tensor missing from export: {sname}")
        continue
    ss, sh = src_ck.get(sname)
    es, eh = exp_ck.get(ename)
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


for f in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "processor_config.json", "modules.json",
          "config_sentence_transformers.json", "sentence_bert_config.json", "1_Pooling/config.json"):
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
want_algo = {"B": "MIXED_PRECISION", "D": "W4A16_NVFP4"}.get(CAND, "NVFP4")
if hq.get("quant_algo") != want_algo:
    errors.append(f"hf_quant_config quant_algo {hq.get('quant_algo')} != {want_algo}")
if hq.get("kv_cache_quant_algo") not in (None, "NONE"):
    errors.append(f"KV cache quantized: {hq.get('kv_cache_quant_algo')}")
# vLLM's pooling adapter keeps an lm_head placeholder; it must be excluded or vLLM builds it as an NVFP4 layer that
# fails to load ("Tensor.item() cannot be called on meta tensors", #32). This model has no score head.
for where, names in (("hf_quant_config.exclude_modules", hq.get("exclude_modules") or []),
                     ("config.quantization_config.ignore", (qc or {}).get("ignore") or [])):
    for need in ("lm_head", "language_model.lm_head"):
        if need not in names:
            errors.append(f"{where} lacks {need!r}")
    # vLLM runs the text tower as language_model.model.*: every checkpoint-name text-tower exclusion needs that twin, or
    # vLLM loads the excluded BF16 module as NVFP4 (candidate C's attention failed to load that way).
    for n in names:
        if n.startswith("language_model.") and not n.startswith(("language_model.model.", "language_model.lm_head")):
            twin = "language_model.model." + n[len("language_model."):]
            if twin not in names:
                errors.append(f"{where} lacks the vLLM-internal twin {twin!r} of {n!r}")

report["errors"] = errors
report["verdict"] = "PASS" if not errors else "FAIL"
OUT.write_text(json.dumps(report, indent=1))
print(("VALIDATE-OK " if not errors else "VALIDATE-FAIL ") + json.dumps(
    {k: report[k] for k in ("candidate", "coverage", "fp8_block_scales", "protected_tensors")}
    | {"fused_group_scales": {k: report["fused_group_scales"][k] for k in ("groups_checked", "mismatched")}}), flush=True)
for e in errors[:12]:
    print("  ERROR", e, flush=True)
sys.exit(0 if not errors else 1)
