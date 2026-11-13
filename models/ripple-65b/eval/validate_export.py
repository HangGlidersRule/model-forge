#!/usr/bin/env python3
"""Ripple-65B Preview (#39): fail-closed validator for a candidate export (A, B or C), every posted artifact check.

Reads only safetensors headers and raw tensor bytes (stdlib + numpy; no torch), so it runs in either image.
Checks (each failure is listed; any failure -> verdict FAIL, exit 1):
  1. config.json quantization_config: quant_method modelopt, quant_algo MIXED_PRECISION, quantized_layers exactly the
     candidate's map (one entry per quantized linear; A: 400 FP8 + 240 NVFP4 + lm_head W4A16_NVFP4), group_size 16 on
     NVFP4 entries, the embeddings excluded; hf_quant_config.json agrees (same quant_algo, same quantized_layers,
     same exclusions, kv_cache_quant_algo null). Every vLLM runtime prefix resolves to the planned algorithm under the
     modelopt_mixed rules of the nightly (eval/ripple_layout.py vllm_resolve: exact, packed-shard, exclusions).
  2. config.json architecture fields equal the source's (layer_types, sliding_window, embed_scale, use_valres,
     ffn_clamp_threshold, gate flags and the rest of ripple_layout.ARCH_FIELDS).
  3. Tensor inventory: every source tensor is accounted for; a quantized linear carries exactly its format's tensors
     with the right dtypes and shapes (NVFP4: U8 weight [out, in/2], F8_E4M3 weight_scale [out, in/16], F32 scalar
     weight_scale_2, plus F32 scalar input_scale for W4A4 only; FP8: F8_E4M3 weight [out, in], F32 scalar weight_scale
     and input_scale); nothing else is in the export.
  4. Every scale is finite and positive: F32 scalars > 0 and finite; every F8_E4M3 block scale byte is not NaN
     (0x7F/0xFF), not negative and not zero.
  5. Fused groups share scales: q/k/g input_scale identical; gate/up weight_scale_2 identical and (W4A4) input_scale
     identical.
  6. Every tensor left in BF16 (embeddings, norms, value_res_lambda, and in C the attention linears) is byte-identical
     to the source; the remote code, chat template and tokenizer files are byte-identical to the source.
Usage: validate_export.py EXPORT_DIR SOURCE_DIR LETTER OUT_JSON [--num-layers N]
Prints VALIDATE-OK <summary> or VALIDATE-FAILED <first errors>; writes OUT_JSON {"verdict": "PASS"|"FAIL", ...}."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ripple_layout as lay  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is in both images
    np = None

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "U8": 1, "I8": 1, "F8_E4M3": 1, "I32": 4, "I64": 8}


class Checkpoint:
    """Tensor index of a safetensors checkpoint (single file or index.json shards)."""

    def __init__(self, root: Path):
        self.root = root
        idx = root / "model.safetensors.index.json"
        if idx.exists():
            files = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
        elif (root / "model.safetensors").exists():
            files = ["model.safetensors"]
        else:
            raise FileNotFoundError(f"{root}: no model.safetensors(.index.json)")
        self.tensors: dict[str, tuple[Path, int, int, str, list[int]]] = {}
        for f in files:
            p = root / f
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            base = 8 + n
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                if name in self.tensors:
                    raise ValueError(f"{name} appears in two shards")
                a, b = meta["data_offsets"]
                self.tensors[name] = (p, base + a, b - a, meta["dtype"], list(meta["shape"]))
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
            if set(wm) != set(self.tensors):
                raise ValueError("index.json weight_map does not match the shard headers")

    def raw(self, name: str) -> bytes:
        p, off, size, _, _ = self.tensors[name]
        with open(p, "rb") as fh:
            fh.seek(off)
            return fh.read(size)

    def digest(self, name: str) -> str:
        p, off, size, _, _ = self.tensors[name]
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            fh.seek(off)
            left = size
            while left:
                b = fh.read(min(left, 1 << 24))
                if not b:
                    raise ValueError(f"{name}: short read")
                h.update(b)
                left -= len(b)
        return h.hexdigest()


def f8_bad_counts(buf: bytes) -> tuple[int, int, int]:
    """(NaN, negative, zero) byte counts of an F8_E4M3 tensor."""
    if np is not None:
        a = np.frombuffer(buf, dtype=np.uint8)
        mag = a & 0x7F
        return int((mag == 0x7F).sum()), int(((a & 0x80) != 0).sum()), int((mag == 0).sum())
    nan = sum(1 for x in buf if x & 0x7F == 0x7F)
    neg = sum(1 for x in buf if x & 0x80)
    zero = sum(1 for x in buf if x & 0x7F == 0)
    return nan, neg, zero


def validate(export: Path, source: Path, letter: str, num_layers: int | None = None) -> dict:
    lay.check_letter(letter)
    errors: list[str] = []
    notes: dict = {}
    src_cfg = json.loads((source / "config.json").read_text())
    n_layers = num_layers or int(src_cfg["num_hidden_layers"])
    plan = lay.PLANS[letter]
    want_ql = lay.quantized_layers(letter, n_layers)

    # ---- 1. quantization config ---------------------------------------------------------------------------------------
    try:
        cfg = json.loads((export / "config.json").read_text())
        hq = json.loads((export / "hf_quant_config.json").read_text())
    except (OSError, ValueError) as e:
        return {"verdict": "FAIL", "errors": [f"config files unreadable: {e}"], "letter": letter}
    qc = cfg.get("quantization_config") or {}
    if qc.get("quant_method") != "modelopt":
        errors.append(f"config.json quantization_config.quant_method is {qc.get('quant_method')!r}, not 'modelopt'")
    if qc.get("quant_algo") != "MIXED_PRECISION":
        errors.append(f"config.json quant_algo is {qc.get('quant_algo')!r}, not MIXED_PRECISION")
    got_ql = qc.get("quantized_layers") or {}
    if got_ql != want_ql:
        missing = sorted(set(want_ql) - set(got_ql))
        extra = sorted(set(got_ql) - set(want_ql))
        wrong = sorted(k for k in set(want_ql) & set(got_ql) if got_ql[k] != want_ql[k])
        errors.append(f"config.json quantized_layers differ from {letter}'s plan: {len(missing)} missing "
                      f"{missing[:3]}, {len(extra)} extra {extra[:3]}, {len(wrong)} wrong {wrong[:3]}")
    hqq = hq.get("quantization") or {}
    if hqq.get("quant_algo") != "MIXED_PRECISION":
        errors.append(f"hf_quant_config.json quant_algo is {hqq.get('quant_algo')!r}, not MIXED_PRECISION")
    if (hqq.get("quantized_layers") or {}) != got_ql:
        errors.append("hf_quant_config.json quantized_layers disagree with config.json")
    if hqq.get("kv_cache_quant_algo") is not None:
        errors.append(f"KV cache quantized ({hqq.get('kv_cache_quant_algo')!r}); the plan keeps it BF16")
    excl_cfg = qc.get("ignore") if "ignore" in qc else qc.get("exclude_modules")
    excl_cfg = list(excl_cfg or [])
    excl_hq = list(hqq.get("exclude_modules") or [])
    if sorted(excl_cfg) != sorted(excl_hq):
        errors.append(f"exclusions disagree: config.json {sorted(excl_cfg)} vs hf_quant_config.json {sorted(excl_hq)}")
    if "model.embed_tokens" not in excl_cfg:
        errors.append("model.embed_tokens is not excluded")
    if any(k.startswith("model.embed_tokens") for k in got_ql):
        errors.append("model.embed_tokens is declared quantized")
    runtime = lay.runtime_plan(letter, n_layers)
    bad_rt = []
    for prefix, want in runtime.items():
        try:
            got = lay.vllm_resolve(got_ql, excl_cfg, prefix)
        except ValueError as e:
            bad_rt.append(str(e))
            continue
        if got != want:
            bad_rt.append(f"{prefix}: vLLM resolves {got}, plan {want}")
    if bad_rt:
        errors.append(f"vLLM runtime resolution differs on {len(bad_rt)} prefixes: {bad_rt[:3]}")

    # ---- 2. architecture fields ----------------------------------------------------------------------------------------
    for f in lay.ARCH_FIELDS:
        if f in src_cfg and cfg.get(f) != src_cfg[f]:
            errors.append(f"config.json {f} differs from the source ({str(cfg.get(f))[:60]} vs {str(src_cfg[f])[:60]})")

    # ---- 3-6. tensors ---------------------------------------------------------------------------------------------------
    try:
        ex = Checkpoint(export)
        sc = Checkpoint(source)
    except (OSError, ValueError) as e:
        errors.append(f"checkpoint unreadable: {e}")
        return {"verdict": "FAIL", "errors": errors, "letter": letter}
    quantized = set(want_ql)
    expected: dict[str, tuple[str, list[int]]] = {}
    bf16_same: list[str] = []
    for name, (_, _, _, dt, shape) in sc.tensors.items():
        lin = lay.linear_of(name)
        if lin in quantized and name == f"{lin}.weight":
            fmt = want_ql[lin]["quant_algo"]
            out_f, in_f = shape
            if fmt in ("NVFP4", "W4A16_NVFP4"):
                expected[name] = ("U8", [out_f, in_f // 2])
                expected[f"{lin}.weight_scale"] = ("F8_E4M3", [out_f, in_f // 16])
                expected[f"{lin}.weight_scale_2"] = ("F32", [])
                if fmt == "NVFP4":
                    expected[f"{lin}.input_scale"] = ("F32", [])
            elif fmt == "FP8":
                expected[name] = ("F8_E4M3", [out_f, in_f])
                expected[f"{lin}.weight_scale"] = ("F32", [])
                expected[f"{lin}.input_scale"] = ("F32", [])
        else:
            expected[name] = (dt, shape)
            bf16_same.append(name)
    have = set(ex.tensors)
    missing = sorted(set(expected) - have)
    extra = sorted(have - set(expected))
    if missing:
        errors.append(f"{len(missing)} tensors missing from the export: {missing[:4]}")
    if extra:
        errors.append(f"{len(extra)} unexpected tensors in the export: {extra[:4]}")
    wrong = []
    for name, (dt, shape) in expected.items():
        if name in ex.tensors:
            _, _, size, gdt, gshape = ex.tensors[name]
            if gdt != dt or gshape != shape or size != DTYPE_BYTES.get(dt, 0) * math.prod(shape or [1]):
                wrong.append(f"{name}: {gdt}{gshape} (want {dt}{shape})")
    if wrong:
        errors.append(f"{len(wrong)} tensors with the wrong dtype/shape: {wrong[:4]}")

    # 4. scales
    scalar_bad, block_bad, n_scalars, n_blocks = [], [], 0, 0
    for name, (_, _, _, dt, _) in ex.tensors.items():
        if name not in expected:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ("weight_scale_2", "input_scale") or (leaf == "weight_scale" and dt == "F32"):
            if dt != "F32" or ex.tensors[name][2] != 4:
                continue  # reported by the inventory check
            (v,) = struct.unpack("<f", ex.raw(name))
            n_scalars += 1
            if not (math.isfinite(v) and v > 0):
                scalar_bad.append(f"{name}={v}")
        elif leaf == "weight_scale" and dt == "F8_E4M3":
            nan, neg, zero = f8_bad_counts(ex.raw(name))
            n_blocks += ex.tensors[name][2]
            if nan or neg or zero:
                block_bad.append(f"{name}: {nan} NaN, {neg} negative, {zero} zero")
    if scalar_bad:
        errors.append(f"{len(scalar_bad)} scalar scales not finite and positive: {scalar_bad[:4]}")
    if block_bad:
        errors.append(f"{len(block_bad)} block-scale tensors with NaN/negative/zero entries: {block_bad[:4]}")
    notes["scales_checked"] = {"f32_scalars": n_scalars, "f8_block_scales": n_blocks}

    # 5. fused groups
    fused_bad = []
    for i in range(n_layers):
        for sub, members in lay.FUSED_GROUPS:
            fmt = plan["attn"] if sub == "self_attn" else plan["mlp"]
            if fmt is None:
                continue
            names = [f"model.layers.{i}.{sub}.{m}" for m in members]
            leaves = ["input_scale"] if fmt == "FP8" else (["weight_scale_2", "input_scale"] if fmt == "NVFP4"
                                                           else ["weight_scale_2"])
            if sub == "self_attn":
                leaves = ["input_scale"]  # vLLM requantizes q/k/g FP8 weights to the max weight scale
            for leaf in leaves:
                vals = {ex.raw(f"{n}.{leaf}") for n in names if f"{n}.{leaf}" in ex.tensors}
                if len(vals) > 1:
                    fused_bad.append(f"layer {i} {sub} {'/'.join(members)} {leaf}")
    if fused_bad:
        errors.append(f"{len(fused_bad)} fused groups with unshared scales: {fused_bad[:4]}")

    # 6. byte identity
    diff_bf16 = [n for n in bf16_same if n in ex.tensors and ex.tensors[n][2] == sc.tensors[n][2]
                 and ex.digest(n) != sc.digest(n)]
    if diff_bf16:
        errors.append(f"{len(diff_bf16)} BF16 tensors differ from the source: {diff_bf16[:4]}")
    notes["bf16_tensors_byte_identical"] = len(bf16_same) - len(diff_bf16)
    file_bad = []
    for f in lay.COPIED_FILES:
        if (source / f).exists():
            if not (export / f).exists():
                file_bad.append(f"{f} missing")
            elif hashlib.sha256((export / f).read_bytes()).hexdigest() != \
                    hashlib.sha256((source / f).read_bytes()).hexdigest():
                file_bad.append(f"{f} differs")
    if file_bad:
        errors.append(f"source files not carried byte-identical: {file_bad}")

    algo_counts: dict[str, int] = {}
    for v in got_ql.values():
        algo_counts[v.get("quant_algo")] = algo_counts.get(v.get("quant_algo"), 0) + 1
    total_bytes = sum(t[2] for t in ex.tensors.values())
    notes.update({"quantized_layers": algo_counts, "tensors": len(ex.tensors), "tensor_bytes": total_bytes,
                  "tensor_gb": round(total_bytes / 1e9, 3), "num_layers": n_layers})
    return {"verdict": "PASS" if not errors else "FAIL", "letter": letter, "errors": errors, **notes}


def main() -> None:
    args = sys.argv[1:]
    num_layers = None
    if "--num-layers" in args:
        i = args.index("--num-layers")
        num_layers = int(args[i + 1])
        del args[i:i + 2]
    export, source, letter, out = Path(args[0]), Path(args[1]), args[2], Path(args[3])
    res = validate(export, source, letter, num_layers)
    out.write_text(json.dumps(res, indent=1) + "\n")
    if res["verdict"] == "PASS":
        print(f"VALIDATE-OK {letter} {json.dumps({k: res[k] for k in ('quantized_layers', 'tensors', 'tensor_gb', 'scales_checked', 'bf16_tensors_byte_identical')})}")
    else:
        print(f"VALIDATE-FAILED {letter}: " + " | ".join(res["errors"][:6]))
        sys.exit(1)


if __name__ == "__main__":
    main()
