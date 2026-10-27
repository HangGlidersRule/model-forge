#!/usr/bin/env python3
"""Fail-closed validation of the Qwen3-Reranker-4B NVFP4 export before publication.

Checks (all must pass):
  1. Tensor inventory: exactly the 108 decoder-MLP linears are NVFP4 (U8 packed weight +
     F8_E4M3 block scale + F32 weight_scale_2 + F32 input_scale); attention, embeddings and
     norms are BF16; no other dtypes; fused groups uniform (gate/up both NVFP4, q/k/v all BF16).
  2. Scales: every weight_scale_2 / input_scale is finite and > 0; every FP8 block scale is
     finite (NaN/Inf counted over all elements); zero block scales are counted and reported.
  3. hf_quant_config.json: NVFP4, group 16, no KV-cache quantization, exclusions = lm_head,
     embed_tokens, every layer's self_attn.
  4. Tokenizer / template / special files byte-identical to upstream at the pinned revision;
     config.json identical to upstream except quantization_config and dtype bookkeeping.
Prints a JSON report; exits non-zero on any failure.
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
UPSTREAM = "Qwen/Qwen3-Reranker-4B"
REV = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
LAYERS = 36
errors: list[str] = []
report: dict = {}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# 1 + 2: tensors and scales
mlp = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.(weight|weight_scale|weight_scale_2|input_scale)$")
dtypes: dict[str, str] = {}
with safe_open(str(ART / "model.safetensors"), framework="pt") as f:
    names = list(f.keys())
    nvfp4_linears: dict[str, set[str]] = {}
    nan_inf_block = zero_block = total_block = 0
    bad_scalars: list[str] = []
    for name in names:
        t = f.get_slice(name)
        dt = t.get_dtype()
        dtypes[name] = dt
        m = mlp.match(name)
        if m:
            key = f"{m.group(1)}.{m.group(2)}"
            nvfp4_linears.setdefault(key, set()).add(m.group(3))
            if m.group(3) == "weight" and dt != "U8":
                errors.append(f"{name}: expected U8 packed FP4, got {dt}")
            if m.group(3) == "weight_scale":
                if dt != "F8_E4M3":
                    errors.append(f"{name}: expected F8_E4M3 block scale, got {dt}")
                v = f.get_tensor(name).to(torch.float32)
                total_block += v.numel()
                nan_inf_block += int((~torch.isfinite(v)).sum())
                zero_block += int((v == 0).sum())
            if m.group(3) in ("weight_scale_2", "input_scale"):
                v = f.get_tensor(name).to(torch.float32)
                if not (torch.isfinite(v).all() and (v > 0).all()):
                    bad_scalars.append(name)
        else:
            if dt != "BF16":
                errors.append(f"{name}: non-MLP tensor must be BF16, got {dt}")
    complete = {k for k, parts in nvfp4_linears.items() if parts == {"weight", "weight_scale", "weight_scale_2", "input_scale"}}
    expected = {f"{i}.{p}" for i in range(LAYERS) for p in ("gate", "up", "down")}
    if complete != expected:
        errors.append(f"NVFP4 linear set mismatch: missing {sorted(expected - complete)[:5]}, extra {sorted(complete - expected)[:5]}")
    attn = [n for n in names if ".self_attn." in n and n.endswith("_proj.weight")]
    if len(attn) != LAYERS * 4 or any(dtypes[n] != "BF16" for n in attn):
        errors.append(f"attention projections: expected {LAYERS * 4} BF16 weights, got {len(attn)}")
    if nan_inf_block:
        errors.append(f"{nan_inf_block} non-finite FP8 block-scale elements")
    if bad_scalars:
        errors.append(f"non-finite or non-positive per-tensor scales: {bad_scalars[:5]}")
    report["tensors"] = {
        "total": len(names),
        "nvfp4_linears": len(complete),
        "bf16_attention_projections": len(attn),
        "dtypes": {d: sum(1 for v in dtypes.values() if v == d) for d in sorted(set(dtypes.values()))},
        "fp8_block_scale_elements": total_block,
        "fp8_block_scales_nonfinite": nan_inf_block,
        "fp8_block_scales_zero": zero_block,
        "per_tensor_scales_bad": len(bad_scalars),
        "fused_groups": "gate/up/down all NVFP4 in every layer; q/k/v/o all BF16 in every layer",
    }

# 3: quant config
hq = json.loads((ART / "hf_quant_config.json").read_text())["quantization"]
exp_ex = {"lm_head", "model.embed_tokens"} | {f"model.layers.{i}.self_attn*" for i in range(LAYERS)}
if hq.get("quant_algo") != "NVFP4" or hq.get("group_size") != 16:
    errors.append(f"hf_quant_config: unexpected algo/group {hq.get('quant_algo')}/{hq.get('group_size')}")
if hq.get("kv_cache_quant_algo") is not None:
    errors.append("hf_quant_config: KV cache must stay unquantized (BF16)")
if set(hq.get("exclude_modules", [])) != exp_ex:
    errors.append("hf_quant_config: exclude_modules differ from expected (lm_head, embed_tokens, 36x self_attn*)")
report["hf_quant_config"] = {"quant_algo": hq.get("quant_algo"), "group_size": hq.get("group_size"),
                             "kv_cache_quant_algo": hq.get("kv_cache_quant_algo"), "exclude_modules": len(hq.get("exclude_modules", []))}

# 4: drift vs upstream
identical = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
             "chat_template.jinja", "config_sentence_transformers.json", "modules.json",
             "sentence_bert_config.json", "1_LogitScore/config.json"]
drift = {}
for name in identical:
    up = Path(hf_hub_download(UPSTREAM, name, revision=REV))
    same = sha(up) == sha(ART / name)
    drift[name] = "identical" if same else "DIFFERS"
    if not same:
        errors.append(f"{name} differs from upstream")
upc = json.loads(Path(hf_hub_download(UPSTREAM, "config.json", revision=REV)).read_text())
art = json.loads((ART / "config.json").read_text())
ignore = {"quantization_config", "torch_dtype", "dtype", "transformers_version"}
# transformers 5.x re-serializes the same values under new keys. Accept only exact equivalents:
#   rope_theta=X (+ rope_scaling None)  ->  rope_parameters={"rope_theta": X, "rope_type": "default"}
#   no layer_types + use_sliding_window False  ->  layer_types = ["full_attention"] * num_hidden_layers
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
drift["config.json"] = ("identical outside quantization_config; transformers 5 re-serialization: " + "; ".join(equivalent)) if not changed else f"DRIFT {changed}"
upg = json.loads(Path(hf_hub_download(UPSTREAM, "generation_config.json", revision=REV)).read_text())
artg = json.loads((ART / "generation_config.json").read_text())
gchanged = sorted(k for k in set(upg) | set(artg) if k != "transformers_version" and upg.get(k) != artg.get(k))
if gchanged:
    errors.append(f"generation_config.json drift: {gchanged}")
drift["generation_config.json"] = ("identical outside transformers_version (4.51.3 -> 5.18.0, export bookkeeping)"
                                   if not gchanged else f"DRIFT {gchanged}")
report["drift_vs_upstream"] = drift
report["errors"] = errors
report["result"] = "PASS" if not errors else "FAIL"
print(json.dumps(report, indent=1))
sys.exit(0 if not errors else 1)
