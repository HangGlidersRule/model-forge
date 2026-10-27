#!/usr/bin/env python3
"""Fail-closed validator for the issue-15 merged JEV-27B ModelOpt artifacts. Writes a JSON report.

--layout mixed (candidate A): 192 NVFP4 MLP + 64 FP8 self-attention + 144 FP8 linear-attention linears.
--layout fp4   (candidate B): 192 NVFP4 MLP + 64 NVFP4 self-attention + 144 NVFP4 linear-attention.
Checks on the exact directory to be uploaded:
  - hf_quant_config.json: MIXED_PRECISION, kv_cache_quant_algo null, quantized_layers exactly the layout
    above (NVFP4 group 16), lm_head not listed; every MTP and vision tensor stays unquantized, which is
    checked on the tensors themselves (ModelOpt lists excluded modules either as patterns like `mtp*`
    or by omission from quantized_layers; vLLM leaves any unlisted layer unquantized)
  - tensors: NVFP4 linears have U8 packed weight + F8_E4M3 weight_scale + F32 weight_scale_2 +
    input_scale; FP8 linears have F8_E4M3 weight + F32 weight_scale + input_scale; lm_head, embeddings,
    norms, vision tower and MTP are BF16; every F32 scale finite and > 0; NVFP4 block scales: counts of
    non-finite and zero values over all elements
  - fused groups (vLLM fuses q/k/v, gate/up, and GDN in_proj_qkv+in_proj_z): the shards share one input
    scale; for NVFP4 they also share one global weight scale (weight_scale_2). FP8 shards may differ in
    weight scale because vLLM requantizes them to the max per-tensor scale at load.
  - byte identity: tokenizer, template and processor files vs Qwen/Qwen3.8-27B@1d4bf0f2;
    decision_head.json vs the JEV-27B adapter directory; calibration.json vs JEV-27B
  - config.json identical to the Qwen base outside quantization_config and the transformers-5
    re-serialization of the same values (dtype/torch_dtype, transformers_version, and in vision_config
    model_type qwen3_5 -> qwen3_5_vision plus explicit default rope_parameters); NVIDIA's
    Qwen3.8-27B-NVFP4 config carries the same vision_config.model_type rewrite
Usage: validate_jev_artifact.py --layout mixed|fp4 --artifact DIR --qwen DIR --jev DIR --report FILE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open

MLP = ("gate_proj", "up_proj", "down_proj")
SELF = ("q_proj", "k_proj", "v_proj", "o_proj")
LIN = ("in_proj_qkv", "in_proj_z", "out_proj")
FUSED = (("self_attn", ("q_proj", "k_proj", "v_proj")), ("mlp", ("gate_proj", "up_proj")),
         ("linear_attn", ("in_proj_qkv", "in_proj_z")))
# transformers-5 re-serialization of unchanged values (key -> (upstream value, accepted artifact value))
ACCEPTED_DRIFT = {
    "vision_config.model_type": ('"qwen3_5"', '"qwen3_5_vision"'),
    "vision_config.rope_parameters.rope_theta": (None, "10000.0"),
    "vision_config.rope_parameters.rope_type": (None, '"axial"'),
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=("mixed", "fp4"), required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--qwen", type=Path, required=True)
    ap.add_argument("--jev", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    art, errors, rep = a.artifact, [], {"layout": a.layout}
    attn_algo = "FP8" if a.layout == "mixed" else "NVFP4"

    hq = json.loads((art / "hf_quant_config.json").read_text())
    q = hq["quantization"]
    rep["producer"] = hq.get("producer")
    want_algo = "MIXED_PRECISION" if a.layout == "mixed" else "NVFP4"
    if q.get("quant_algo") != want_algo:
        errors.append(f"quant_algo {q.get('quant_algo')} (want {want_algo})")
    if q.get("kv_cache_quant_algo") is not None:
        errors.append(f"kv_cache_quant_algo {q.get('kv_cache_quant_algo')} (want null: BF16 KV)")
    index = json.loads((art / "model.safetensors.index.json").read_text())["weight_map"]
    if a.layout == "mixed":
        ql = q["quantized_layers"]
    else:
        # A uniform NVFP4 export (every quantized linear is NVFP4) carries quant_algo NVFP4 + exclude_modules
        # instead of a per-layer map; derive the quantized set from the tensors and require group 16.
        if q.get("group_size") != 16:
            errors.append(f"group_size {q.get('group_size')} (want 16)")
        ql = {k[: -len(".weight_scale_2")]: {"quant_algo": "NVFP4", "group_size": q.get("group_size")}
              for k in index if k.endswith(".weight_scale_2")}
        rep["exclude_modules_count"] = len(q.get("exclude_modules") or [])
    kinds = Counter()
    for name, v in ql.items():
        leaf, algo = name.rsplit(".", 1)[-1], v["quant_algo"]
        if algo == "NVFP4" and v.get("group_size") != 16:
            kinds[f"unexpected:{name}:group{v.get('group_size')}"] += 1
        elif leaf in MLP and ".mlp." in name and algo == "NVFP4":
            kinds["nvfp4_mlp"] += 1
        elif leaf in SELF and ".self_attn." in name and algo == attn_algo:
            kinds[f"{attn_algo.lower()}_self_attn"] += 1
        elif leaf in LIN and ".linear_attn." in name and algo == attn_algo:
            kinds[f"{attn_algo.lower()}_linear_attn"] += 1
        else:
            kinds[f"unexpected:{name}:{algo}"] += 1
    rep["quantized_layers"] = dict(kinds)
    want = {"nvfp4_mlp": 192, f"{attn_algo.lower()}_self_attn": 64, f"{attn_algo.lower()}_linear_attn": 144}
    for k, n in want.items():
        if kinds.get(k) != n:
            errors.append(f"{k}: {kinds.get(k)} (want {n})")
    extra = [k for k in kinds if k.startswith("unexpected:")]
    if extra:
        errors.append(f"unexpected quantized layers: {extra[:5]}")
    if any("lm_head" in n or n.startswith(("mtp", "model.visual")) for n in ql):
        errors.append("lm_head, MTP or vision layers are listed as quantized")

    by_shard: dict[str, list[str]] = {}
    for k, s in index.items():
        by_shard.setdefault(s, []).append(k)
    dtypes, bad_scales, nvfp4_block = {}, [], {"elements": 0, "nonfinite": 0, "zero": 0}
    scal: dict[str, float] = {}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(str(art / shard), framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k)
                dtypes[k] = str(t.dtype).replace("torch.", "")
                if k.endswith(("weight_scale_2", "input_scale")) or (k.endswith("weight_scale") and t.dtype == torch.float32):
                    v = t.float()
                    if not torch.isfinite(v).all() or (v <= 0).any():
                        bad_scales.append(k)
                    scal[k] = float(v.reshape(-1)[0])
                elif k.endswith("weight_scale") and t.dtype == torch.float8_e4m3fn:
                    v = t.float()
                    nvfp4_block["elements"] += v.numel()
                    nvfp4_block["nonfinite"] += int((~torch.isfinite(v)).sum())
                    nvfp4_block["zero"] += int((v == 0).sum())
    rep["nvfp4_block_scales"] = nvfp4_block
    rep["bad_per_tensor_scales"] = bad_scales[:10]
    if bad_scales:
        errors.append(f"{len(bad_scales)} non-finite or non-positive per-tensor scales")
    if nvfp4_block["nonfinite"]:
        errors.append(f"{nvfp4_block['nonfinite']} non-finite NVFP4 block scales")

    for name, v in ql.items():
        if v["quant_algo"] == "NVFP4":
            need = {f"{name}.weight": "uint8", f"{name}.weight_scale": "float8_e4m3fn", f"{name}.weight_scale_2": "float32", f"{name}.input_scale": "float32"}
        else:
            need = {f"{name}.weight": "float8_e4m3fn", f"{name}.weight_scale": "float32", f"{name}.input_scale": "float32"}
        for k, dt in need.items():
            if dtypes.get(k) != dt:
                errors.append(f"{k}: dtype {dtypes.get(k)} (want {dt})")
    protected = {k: dt for k, dt in dtypes.items() if k.endswith(".weight") and (k == "lm_head.weight" or "embed_tokens" in k
                 or "norm" in k or k.startswith("model.visual.") or k.startswith("mtp."))}
    non_bf16 = {k: dt for k, dt in protected.items() if dt != "bfloat16"}
    mtp = [k for k in dtypes if k.startswith("mtp.")]
    rep["protected_tensors"] = {"count": len(protected), "mtp_tensors": len(mtp), "non_bf16": list(non_bf16)[:10]}
    if non_bf16:
        errors.append(f"{len(non_bf16)} protected tensors not BF16, e.g. {list(non_bf16)[:3]}")
    if dtypes.get("lm_head.weight") != "bfloat16":
        errors.append(f"lm_head.weight is {dtypes.get('lm_head.weight')}")
    up_mtp = [k for k in json.loads((a.qwen / "model.safetensors.index.json").read_text())["weight_map"] if k.startswith("mtp.")]
    if sorted(mtp) != sorted(up_mtp):
        errors.append(f"MTP tensors {len(mtp)} != upstream {len(up_mtp)}")

    fused = Counter()
    for layer in range(64):
        for parent, mods in FUSED:
            names = [f"model.language_model.layers.{layer}.{parent}.{m}" for m in mods]
            if not all(n in ql for n in names):
                continue
            algo = ql[names[0]]["quant_algo"]
            fused[f"{parent}:{algo}:groups"] += 1
            if len({scal.get(f"{n}.input_scale") for n in names}) > 1:
                fused[f"{parent}:{algo}:input_scale_differs"] += 1
            if algo == "NVFP4" and len({scal.get(f"{n}.weight_scale_2") for n in names}) > 1:
                fused[f"{parent}:{algo}:global_weight_scale_differs"] += 1
    rep["fused_groups"] = dict(fused)
    bad_fused = {k: v for k, v in fused.items() if k.endswith("differs")}
    if bad_fused:
        errors.append(f"fused groups with non-uniform scales: {bad_fused}")

    same, diff = [], []
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt",
              "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"):
        if (a.qwen / f).exists():
            (same if (art / f).exists() and sha(art / f) == sha(a.qwen / f) else diff).append(f)
    for f, src in (("decision_head.json", a.jev / "adapter_vllm" / "decision_head.json"), ("calibration.json", a.jev / "calibration.json")):
        (same if (art / f).exists() and sha(art / f) == sha(src) else diff).append(f)
    rep["byte_identical"] = same
    if diff:
        errors.append(f"not byte-identical to source: {diff}")

    ca, cq = json.loads((art / "config.json").read_text()), json.loads((a.qwen / "config.json").read_text())
    ignore = {"quantization_config", "dtype", "torch_dtype", "transformers_version"}

    def flat(o, p=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if k not in ignore:
                    yield from flat(v, f"{p}.{k}" if p else k)
        else:
            yield p, json.dumps(o, sort_keys=True)
    fa, fq = dict(flat(ca)), dict(flat(cq))
    drift = sorted(k for k in set(fa) | set(fq) if fa.get(k) != fq.get(k))
    accepted = [k for k in drift if k in ACCEPTED_DRIFT and (fq.get(k), fa.get(k)) == ACCEPTED_DRIFT[k]]
    unexplained = [k for k in drift if k not in accepted]
    rep["config_drift_accepted"] = accepted
    rep["config_drift_unexplained"] = unexplained
    if unexplained:
        errors.append(f"config.json drift: {unexplained[:5]}")

    rep["errors"] = errors
    rep["verdict"] = "PASS" if not errors else "FAIL"
    a.report.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("verdict", "quantized_layers", "nvfp4_block_scales", "fused_groups", "config_drift_accepted")}))
    for e in errors:
        print("ERROR", e)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
