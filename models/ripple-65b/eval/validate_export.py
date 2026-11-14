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
     ffn_clamp_threshold, gate flags and the rest of ripple_layout.ARCH_FIELDS), and its dtype is bfloat16: the dtype of
     every tensor the export keeps unquantized, which the build restores from the exporter's config.json (the source's
     config.json has none) and the staged product carries.
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
  7. generation_config.json, when the export ships one (the upstream has none at the pinned revision; transformers
     writes it from the model config at export), is a JSON object carrying config.json's bos, eos and pad token ids. A
     file that is there is checked: an empty one, or one holding null, is refused, never taken for an absent one.
Reads stay inside the two directories (eval/checkpoint_files.py): each directory is opened once and every file in it is
opened relative to it with O_NOFOLLOW as a regular file, so a link is refused rather than followed. The index's shard
names are plain .safetensors file names, the directory's .safetensors files are exactly the indexed shards, a header
length above 100,000,000 bytes or past the end of its shard is refused before it is read, every header entry is
well-formed, every tensor lies inside its shard and a shard's tensors cover its data exactly (the safetensors library's
own rule), and tensor bytes are read through the shard's open descriptor, never by reopening its name. config.json,
hf_quant_config.json, generation_config.json and the sidecars are read the same way in the export and the source.
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
import checkpoint_files as cf  # noqa: E402
import ripple_layout as lay  # noqa: E402

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is in both images
    np = None

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "U8": 1, "I8": 1, "F8_E4M3": 1, "I32": 4, "I64": 8}
TOKEN_IDS = ("bos_token_id", "eos_token_id", "pad_token_id")
EXPORT_DTYPE = "bfloat16"  # config.json's dtype: the unquantized tensors' (BF16 source); the staged product's


class Checkpoint:
    """Tensor index of a safetensors checkpoint (single file or index.json shards): every shard opened once inside the
    directory (checkpoint_files.Shards), every tensor read through its shard's open descriptor."""

    def __init__(self, root: Path | cf.Directory):
        self.shards = cf.Shards(root)
        self.root = self.shards.root
        try:
            self.tensors: dict[str, tuple[str, int, int, str, list[int]]] = self.shards.tensors()
        except BaseException:
            self.shards.close()
            raise

    def raw(self, name: str) -> bytes:
        shard, off, size, _, _ = self.tensors[name]
        return self.shards.pread(shard, off, size)

    def digest(self, name: str) -> str:
        shard, off, size, _, _ = self.tensors[name]
        h = hashlib.sha256()
        for start in range(0, size, cf.CHUNK):
            h.update(self.shards.pread(shard, off + start, min(cf.CHUNK, size - start)))
        return h.hexdigest()

    def close(self) -> None:
        self.shards.close()

    def __enter__(self) -> Checkpoint:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


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


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate(export: Path, source: Path, letter: str, num_layers: int | None = None) -> dict:
    """Validate EXPORT against SOURCE as candidate LETTER. Both directories are opened once and read confined
    (eval/checkpoint_files.py); a directory that cannot be opened fails the verdict."""
    lay.check_letter(letter)
    try:
        ex_dir = cf.Directory(export)
    except cf.ShardError as e:
        return {"verdict": "FAIL", "errors": [f"export directory: {e}"], "letter": letter}
    try:
        src_dir = cf.Directory(source)
    except cf.ShardError as e:
        ex_dir.close()
        return {"verdict": "FAIL", "errors": [f"source directory: {e}"], "letter": letter}
    try:
        return _validate(ex_dir, src_dir, letter, num_layers)
    finally:
        ex_dir.close()
        src_dir.close()


def _validate(ex_dir: cf.Directory, src_dir: cf.Directory, letter: str, num_layers: int | None) -> dict:
    errors: list[str] = []
    notes: dict = {}
    # ---- 1. quantization config ---------------------------------------------------------------------------------------
    try:
        src_cfg = json.loads(src_dir.read("config.json"))
        cfg = json.loads(ex_dir.read("config.json"))
        hq = json.loads(ex_dir.read("hf_quant_config.json"))
        gen_raw = ex_dir.read_optional("generation_config.json")
        gen = json.loads(gen_raw) if gen_raw is not None else None
    except (OSError, ValueError) as e:
        return {"verdict": "FAIL", "errors": [f"config files unreadable: {e}"], "letter": letter}
    n_layers = num_layers or int(src_cfg["num_hidden_layers"])
    plan = lay.PLANS[letter]
    want_ql = lay.quantized_layers(letter, n_layers)
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
    if cfg.get("dtype") != EXPORT_DTYPE:
        errors.append(f"config.json dtype is {str(cfg.get('dtype'))[:60]!r}, not {EXPORT_DTYPE!r}")

    # ---- 7. generation_config.json token ids ---------------------------------------------------------------------------
    if gen_raw is not None:
        if isinstance(gen, dict):
            gen_bad = [f"{k} {gen.get(k)!r} (config.json {cfg.get(k)!r})" for k in TOKEN_IDS if gen.get(k) != cfg.get(k)]
            notes["generation_config_token_ids"] = {k: gen.get(k) for k in TOKEN_IDS}
        else:
            gen_bad = [f"not a JSON object: {str(gen)[:60]}"]
        if gen_bad:
            errors.append(f"generation_config.json token ids differ from config.json: {gen_bad}")

    # ---- 3-6. tensors ---------------------------------------------------------------------------------------------------
    try:
        ex = Checkpoint(ex_dir)
    except (OSError, ValueError) as e:
        errors.append(f"checkpoint unreadable: {e}")
        return {"verdict": "FAIL", "errors": errors, "letter": letter}
    try:
        sc = Checkpoint(src_dir)
    except (OSError, ValueError) as e:
        ex.close()
        errors.append(f"checkpoint unreadable: {e}")
        return {"verdict": "FAIL", "errors": errors, "letter": letter}
    try:
        return _tensor_checks(ex, sc, ex_dir, src_dir, letter, n_layers, plan, want_ql, got_ql, errors, notes)
    finally:
        ex.close()
        sc.close()


def _tensor_checks(ex: Checkpoint, sc: Checkpoint, ex_dir: cf.Directory, src_dir: cf.Directory, letter: str,
                   n_layers: int, plan: dict, want_ql: dict, got_ql: dict, errors: list[str], notes: dict) -> dict:
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
        try:
            src_bytes = src_dir.read_optional(f)
            if src_bytes is None:
                continue
            ex_bytes = ex_dir.read_optional(f)
        except cf.ShardError as e:
            file_bad.append(str(e))
            continue
        if ex_bytes is None:
            file_bad.append(f"{f} missing")
        elif sha256(ex_bytes) != sha256(src_bytes):
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
