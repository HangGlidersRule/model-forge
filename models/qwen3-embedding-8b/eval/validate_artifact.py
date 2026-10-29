#!/usr/bin/env python3
"""Fail-closed validation of the Qwen3-Embedding-8B W4A4 NVFP4 export (issue #18 candidate B) before publication.

Checks (all must pass):
  1. Tensor inventory: exactly the 252 decoder linears (MLP gate/up/down + self-attention q/k/v/o, 36 layers) are NVFP4
     (U8 packed weight + F8_E4M3 block scale + F32 weight_scale_2 + F32 input_scale); embeddings, norms and q/k norms are
     BF16; no lm_head (the source checkpoint has none: the pooling runner reads the last hidden state); no other dtypes.
     Fused groups (q/k/v, gate/up) share input_scale and weight_scale_2, since vLLM runs each group as one GEMM.
  2. Scales: every weight_scale_2 / input_scale finite and > 0; FP8 block scales counted non-finite and zero over all
     elements.
  3. hf_quant_config.json: NVFP4, group 16, no KV-cache quantization, exclusions exactly lm_head + embed_tokens.
  4. Tokenizer, sentence-transformers and license files byte-identical to upstream at the pinned revision; config.json
     and generation_config.json identical outside quantization metadata and named transformers-5 re-serializations.
Usage: validate_artifact.py ARTIFACT_DIR. Prints a JSON report; exits non-zero on any failure.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

ART = Path(sys.argv[1])
UPSTREAM = "Qwen/Qwen3-Embedding-8B"
REV = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
LAYERS = 36
PROJ = {"mlp": ("gate", "up", "down"), "self_attn": ("q", "k", "v", "o")}
errors: list[str] = []
report: dict = {}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# 1 + 2: tensors and scales
lin = re.compile(r"^model\.layers\.(\d+)\.(mlp|self_attn)\.(\w+)_proj\.(weight|weight_scale|weight_scale_2|input_scale)$")
dtypes: dict[str, str] = {}
scalars: dict[str, float] = {}
with safe_open(str(ART / "model.safetensors"), framework="pt") as f:
    names = list(f.keys())
    parts: dict[str, set[str]] = {}
    nan_inf_block = zero_block = total_block = 0
    bad_scalars: list[str] = []
    for name in names:
        dt = f.get_slice(name).get_dtype()
        dtypes[name] = dt
        m = lin.match(name)
        if m:
            key = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
            parts.setdefault(key, set()).add(m.group(4))
            want = {"weight": "U8", "weight_scale": "F8_E4M3", "weight_scale_2": "F32", "input_scale": "F32"}[m.group(4)]
            if dt != want:
                errors.append(f"{name}: expected {want}, got {dt}")
            if m.group(4) == "weight_scale":
                v = f.get_tensor(name).to(torch.float32)
                total_block += v.numel()
                nan_inf_block += int((~torch.isfinite(v)).sum())
                zero_block += int((v == 0).sum())
            if m.group(4) in ("weight_scale_2", "input_scale"):
                v = f.get_tensor(name).to(torch.float32)
                if not (torch.isfinite(v).all() and (v > 0).all()):
                    bad_scalars.append(name)
                scalars[name] = float(v.reshape(-1)[0])
        elif dt != "BF16":
            errors.append(f"{name}: non-quantized tensor must be BF16, got {dt}")
complete = {k for k, p in parts.items() if p == {"weight", "weight_scale", "weight_scale_2", "input_scale"}}
expected = {f"{i}.{grp}.{p}" for i in range(LAYERS) for grp, ps in PROJ.items() for p in ps}
if complete != expected:
    errors.append(f"NVFP4 linear set mismatch: missing {sorted(expected - complete)[:5]}, extra {sorted(complete - expected)[:5]}")
if any(n.startswith("lm_head") for n in names):
    errors.append("lm_head present (the source checkpoint has none)")
if nan_inf_block:
    errors.append(f"{nan_inf_block} non-finite FP8 block-scale elements")
if bad_scalars:
    errors.append(f"non-finite or non-positive per-tensor scales: {bad_scalars[:5]}")
fused_bad = []
for i in range(LAYERS):
    for grp, members in (("self_attn", ("q", "k", "v")), ("mlp", ("gate", "up"))):
        for s in ("input_scale", "weight_scale_2"):
            vals = {scalars.get(f"model.layers.{i}.{grp}.{p}_proj.{s}") for p in members}
            if len(vals) != 1:
                fused_bad.append(f"layer {i} {grp} {'/'.join(members)} {s}")
if fused_bad:
    errors.append(f"fused groups with differing shared scales: {fused_bad[:5]}")
report["tensors"] = {
    "total": len(names),
    "nvfp4_linears": len(complete),
    "nvfp4_by_group": {grp: sum(1 for k in complete if f".{grp}." in k) for grp in PROJ},
    "dtypes": {d: sum(1 for v in dtypes.values() if v == d) for d in sorted(set(dtypes.values()))},
    "fp8_block_scale_elements": total_block,
    "fp8_block_scales_nonfinite": nan_inf_block,
    "fp8_block_scales_zero": zero_block,
    "per_tensor_scales_bad": len(bad_scalars),
    "fused_groups": "q/k/v and gate/up share input_scale and weight_scale_2 in every layer" if not fused_bad
    else f"{len(fused_bad)} mismatched",
    "lm_head": "absent, as in the source checkpoint",
}

# 3: quant config
hq = json.loads((ART / "hf_quant_config.json").read_text())["quantization"]
if hq.get("quant_algo") != "NVFP4" or hq.get("group_size") != 16:
    errors.append(f"hf_quant_config: unexpected algo/group {hq.get('quant_algo')}/{hq.get('group_size')}")
if hq.get("kv_cache_quant_algo") is not None:
    errors.append("hf_quant_config: KV cache must stay unquantized (BF16)")
if set(hq.get("exclude_modules", [])) != {"lm_head", "model.embed_tokens"}:
    errors.append(f"hf_quant_config: exclude_modules {hq.get('exclude_modules')} != [lm_head, model.embed_tokens]")
report["hf_quant_config"] = {k: hq.get(k) for k in ("quant_algo", "group_size", "kv_cache_quant_algo", "exclude_modules")}

# 4: drift vs upstream
identical = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "modules.json",
             "config_sentence_transformers.json", "1_Pooling/config.json", "LICENSE"]
drift = {}
for name in identical:
    same = sha(Path(hf_hub_download(UPSTREAM, name, revision=REV))) == sha(ART / name)
    drift[name] = "identical" if same else "DIFFERS"
    if not same:
        errors.append(f"{name} differs from upstream")
upc = json.loads(Path(hf_hub_download(UPSTREAM, "config.json", revision=REV)).read_text())
art = json.loads((ART / "config.json").read_text())
ignore = {"quantization_config", "torch_dtype", "dtype", "transformers_version"}
equivalent = []
if ("rope_theta" in upc and "rope_theta" not in art and upc.get("rope_scaling") in (None, {})
        and art.get("rope_parameters") == {"rope_theta": upc["rope_theta"], "rope_type": "default"}):
    ignore |= {"rope_theta", "rope_parameters", "rope_scaling"}
    equivalent.append(f"rope_theta={upc['rope_theta']} re-serialized as rope_parameters")
if ("layer_types" not in upc and upc.get("use_sliding_window") is False
        and art.get("layer_types") == ["full_attention"] * upc["num_hidden_layers"]):
    ignore.add("layer_types")
    equivalent.append("layer_types made explicit (all full_attention; sliding window off upstream)")
if upc.get("torch_dtype") == art.get("dtype") and "torch_dtype" not in art:
    equivalent.append(f"torch_dtype={upc['torch_dtype']} renamed to dtype")
changed = sorted(k for k in set(upc) | set(art) if k not in ignore and upc.get(k) != art.get(k))
if changed:
    errors.append(f"config.json drift outside quantization metadata: {changed}")
drift["config.json"] = ("identical outside quantization_config; transformers 5 re-serialization: " + "; ".join(equivalent)
                        if not changed else f"DRIFT {changed}")
upg = json.loads(Path(hf_hub_download(UPSTREAM, "generation_config.json", revision=REV)).read_text())
artg = json.loads((ART / "generation_config.json").read_text())
gchanged = sorted(k for k in set(upg) | set(artg) if k != "transformers_version" and upg.get(k) != artg.get(k))
if gchanged:
    errors.append(f"generation_config.json drift: {gchanged}")
drift["generation_config.json"] = (f"identical outside transformers_version ({upg.get('transformers_version')} -> "
                                   f"{artg.get('transformers_version')}, export bookkeeping)" if not gchanged
                                   else f"DRIFT {gchanged}")
report["drift_vs_upstream"] = drift
report["model_safetensors_sha256"] = sha(ART / "model.safetensors")
report["errors"] = errors
report["result"] = "PASS" if not errors else "FAIL"
print(json.dumps(report, indent=1))
sys.exit(0 if not errors else 1)
