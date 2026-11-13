#!/usr/bin/env python3
"""Ripple-65B Preview (#39): self-test of validate_export.py on synthetic exports (stdlib only, no model, no GPU).

Builds a tiny source checkpoint shaped like Ripple (2 layers, real config fields, hidden 64, MLP 128, q/k/v/o/g, four
norms, q/k norms, value_res_lambda, untied lm_head) and, for each candidate A/B/C, a synthetic export in exactly the
layout the plan declares. Each clean export must PASS; then every planted fault below must FAIL with its check named:
  no config.json block, config quant_algo NVFP4, hf_quant_config disagreeing, a quantized_layers entry missing,
  embeddings not excluded, a mixed fused group, an architecture field changed, a NaN / zero / negative F8 block scale,
  a zero / NaN / negative F32 scale, unshared q/k/g input scales, unshared gate/up global scales, a norm or
  value_res_lambda or embedding byte changed, an extra tensor, a missing tensor, a wrong dtype, a remote-code file
  changed, KV-cache quantization declared.
Usage: validate_selftest.py WORK_DIR [--keep]   (with --keep the clean synthetic exports stay for the vLLM offline
probe: WORK_DIR/synthetic-{A,B,C} and WORK_DIR/source). Prints VALIDATOR-SELFTEST-OK."""

from __future__ import annotations

import json
import random
import shutil
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ripple_layout as lay  # noqa: E402
import validate_export as ve  # noqa: E402

H, I_, NH, NKV, HD, V, L = 64, 128, 2, 1, 32, 128, 2


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, off, blobs = {}, 0, []
    for name in sorted(tensors):
        dt, shape, data = tensors[name]
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def rbytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.getrandbits(8) for _ in range(n))


def f32(v: float) -> bytes:
    return struct.pack("<f", v)


def f8_scales(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(0x30, 0x58) for _ in range(n))  # positive normal E4M3 values


def source_tensors(rng: random.Random) -> dict:
    t = {"model.embed_tokens.weight": ("BF16", [V, H], rbytes(rng, V * H * 2)),
         "model.norm.weight": ("BF16", [H], rbytes(rng, H * 2)),
         "lm_head.weight": ("BF16", [V, H], rbytes(rng, V * H * 2))}
    shapes = {"q_proj": [NH * HD, H], "k_proj": [NKV * HD, H], "v_proj": [NKV * HD, H], "o_proj": [H, NH * HD],
              "g_proj": [NH * HD, H]}
    for i in range(L):
        p = f"model.layers.{i}."
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm"):
            t[p + n + ".weight"] = ("BF16", [H], rbytes(rng, H * 2))
        for n, s in shapes.items():
            t[p + f"self_attn.{n}.weight"] = ("BF16", s, rbytes(rng, s[0] * s[1] * 2))
        t[p + "self_attn.q_norm.weight"] = ("BF16", [HD], rbytes(rng, HD * 2))
        t[p + "self_attn.k_norm.weight"] = ("BF16", [HD], rbytes(rng, HD * 2))
        t[p + "self_attn.value_res_lambda"] = ("BF16", [1], rbytes(rng, 2))
        for n, s in (("gate_proj", [I_, H]), ("up_proj", [I_, H]), ("down_proj", [H, I_])):
            t[p + f"mlp.{n}.weight"] = ("BF16", s, rbytes(rng, s[0] * s[1] * 2))
    return t


def source_config() -> dict:
    return {"architectures": ["RicursiveForCausalLM"], "model_type": "ricursive", "num_hidden_layers": L,
            "hidden_size": H, "intermediate_size": I_, "num_attention_heads": NH, "num_key_value_heads": NKV,
            "head_dim": HD, "layer_types": ["sliding_attention", "full_attention"], "hybrid_layer_pattern": [1, 0],
            "sliding_window": 1024, "embed_scale": 71.55417527999327, "use_valres": True,
            "ffn_clamp_threshold": 20.0, "use_gated_attention": True, "gate_full_precision": True,
            "is_hybrid_swa": True, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "vocab_size": V,
            "tie_word_embeddings": False, "max_position_embeddings": 2097152,
            "auto_map": {"AutoConfig": "configuration_ricursive.RicursiveConfig"}, "bos_token_id": 100257,
            "eos_token_id": [100265, 100257], "pad_token_id": 100277, "transformers_version": "4.57.6"}


def make_source(d: Path, rng: random.Random) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    t = source_tensors(rng)
    write_safetensors(d / "model.safetensors", t)
    (d / "config.json").write_text(json.dumps(source_config(), indent=2))
    for f in lay.COPIED_FILES:
        (d / f).write_text(f"synthetic {f}\n")
    return t


def make_export(d: Path, src_t: dict, letter: str, rng: random.Random) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    ql = lay.quantized_layers(letter, L)
    out = {}
    for name, (dt, shape, data) in src_t.items():
        lin = lay.linear_of(name)
        if lin in ql and name == f"{lin}.weight":
            continue
        out[name] = (dt, shape, data)
    shared: dict = {}
    for lin, e in ql.items():
        _, (o, i) = None, src_t[f"{lin}.weight"][1]
        fmt = e["quant_algo"]
        grp = None
        for sub, members in lay.FUSED_GROUPS:
            for m in members:
                if lin.endswith(f"{sub}.{m}"):
                    grp = lin.rsplit(".", 1)[0] + ":" + sub
        if fmt in ("NVFP4", "W4A16_NVFP4"):
            out[f"{lin}.weight"] = ("U8", [o, i // 2], rbytes(rng, o * i // 2))
            out[f"{lin}.weight_scale"] = ("F8_E4M3", [o, i // 16], f8_scales(rng, o * i // 16))
            ws2 = shared.setdefault((grp, "ws2"), rng.uniform(1e-4, 1e-2)) if grp else rng.uniform(1e-4, 1e-2)
            out[f"{lin}.weight_scale_2"] = ("F32", [], f32(ws2))
            if fmt == "NVFP4":
                ins = shared.setdefault((grp, "in"), rng.uniform(1e-4, 1e-2)) if grp else rng.uniform(1e-4, 1e-2)
                out[f"{lin}.input_scale"] = ("F32", [], f32(ins))
        else:
            out[f"{lin}.weight"] = ("F8_E4M3", [o, i], rbytes(rng, o * i))
            out[f"{lin}.weight_scale"] = ("F32", [], f32(rng.uniform(1e-4, 1e-2)))
            ins = shared.setdefault((grp, "in"), rng.uniform(1e-3, 1e-1)) if grp else rng.uniform(1e-3, 1e-1)
            out[f"{lin}.input_scale"] = ("F32", [], f32(ins))
    write_safetensors(d / "model.safetensors", out)
    cfg = source_config()
    cfg["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                  "producer": {"name": "modelopt", "version": "0.48.0"},
                                  "ignore": ["model.embed_tokens"], "quantized_layers": ql}
    (d / "config.json").write_text(json.dumps(cfg, indent=2))
    (d / "hf_quant_config.json").write_text(json.dumps({
        "producer": {"name": "modelopt", "version": "0.48.0"},
        "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None,
                         "exclude_modules": ["model.embed_tokens"], "quantized_layers": ql}}, indent=2))
    for f in lay.COPIED_FILES:
        shutil.copyfile(d.parent / "source" / f, d / f)


def edit_tensors(d: Path, fn) -> None:
    ck = ve.Checkpoint(d)
    t = {n: (meta[3], meta[4], ck.raw(n)) for n, meta in ck.tensors.items()}
    fn(t)
    write_safetensors(d / "model.safetensors", t)


def edit_json(d: Path, name: str, fn) -> None:
    j = json.loads((d / name).read_text())
    fn(j)
    (d / name).write_text(json.dumps(j, indent=2))


def set_byte(t: dict, name: str, pos: int, val: int) -> None:
    dt, shape, data = t[name]
    b = bytearray(data)
    b[pos] = val
    t[name] = (dt, shape, bytes(b))


def faults(letter: str) -> list[tuple[str, str, object]]:
    """(label, expected error substring, mutator(dir)) for one candidate."""
    a = "model.layers.1"
    f: list[tuple[str, str, object]] = [
        ("no config.json quant block", "quant_method",
         lambda d: edit_json(d, "config.json", lambda j: j.pop("quantization_config"))),
        ("config quant_algo NVFP4", "not MIXED_PRECISION",
         lambda d: edit_json(d, "config.json", lambda j: j["quantization_config"].update(quant_algo="NVFP4"))),
        ("hf_quant_config disagrees", "disagree with config.json",
         lambda d: edit_json(d, "hf_quant_config.json",
                             lambda j: j["quantization"]["quantized_layers"].pop(f"{a}.mlp.down_proj"))),
        ("quantized_layers entry missing", "quantized_layers differ",
         lambda d: [edit_json(d, n, (lambda j: (j.get("quantization_config") or j["quantization"])["quantized_layers"]
                                     .pop(f"{a}.mlp.down_proj"))) for n in ("config.json", "hf_quant_config.json")]),
        ("embeddings not excluded", "embed_tokens is not excluded",
         lambda d: [edit_json(d, "config.json", lambda j: j["quantization_config"].update(ignore=[])),
                    edit_json(d, "hf_quant_config.json", lambda j: j["quantization"].update(exclude_modules=[]))]),
        ("architecture field changed", "sliding_window",
         lambda d: edit_json(d, "config.json", lambda j: j.update(sliding_window=2048))),
        ("NaN block scale", "NaN/negative/zero",
         lambda d: edit_tensors(d, lambda t: set_byte(t, f"{a}.mlp.up_proj.weight_scale", 5, 0x7F))),
        ("zero block scale", "NaN/negative/zero",
         lambda d: edit_tensors(d, lambda t: set_byte(t, "lm_head.weight_scale", 0, 0x00))),
        ("negative block scale", "NaN/negative/zero",
         lambda d: edit_tensors(d, lambda t: set_byte(t, f"{a}.mlp.down_proj.weight_scale", 3, 0xB8))),
        ("negative global scale", "not finite and positive",
         lambda d: edit_tensors(d, lambda t: t.update({f"{a}.mlp.down_proj.weight_scale_2": ("F32", [], f32(-0.01))}))),
        ("NaN global scale", "not finite and positive",
         lambda d: edit_tensors(d, lambda t: t.update({"lm_head.weight_scale_2": ("F32", [], f32(float("nan")))}))),
        ("gate/up global scales unshared", "unshared scales",
         lambda d: edit_tensors(d, lambda t: t.update({f"{a}.mlp.up_proj.weight_scale_2": ("F32", [], f32(0.12345))}))),
        ("norm byte changed", "BF16 tensors differ",
         lambda d: edit_tensors(d, lambda t: set_byte(t, f"{a}.pre_feedforward_layernorm.weight", 0,
                                                      t[f"{a}.pre_feedforward_layernorm.weight"][2][0] ^ 1))),
        ("value_res_lambda changed", "BF16 tensors differ",
         lambda d: edit_tensors(d, lambda t: set_byte(t, f"{a}.self_attn.value_res_lambda", 1,
                                                      t[f"{a}.self_attn.value_res_lambda"][2][1] ^ 0x40))),
        ("embedding byte changed", "BF16 tensors differ",
         lambda d: edit_tensors(d, lambda t: set_byte(t, "model.embed_tokens.weight", 7,
                                                      t["model.embed_tokens.weight"][2][7] ^ 1))),
        ("extra tensor", "unexpected tensors",
         lambda d: edit_tensors(d, lambda t: t.update({"lm_head.input_scale": ("F32", [], f32(0.5))}))),
        ("missing tensor", "tensors missing",
         lambda d: edit_tensors(d, lambda t: t.pop("lm_head.weight_scale_2"))),
        ("remote code changed", "byte-identical",
         lambda d: (d / "modeling_ricursive.py").write_text("tampered\n")),
        ("KV cache quantized", "KV cache quantized",
         lambda d: edit_json(d, "hf_quant_config.json", lambda j: j["quantization"].update(kv_cache_quant_algo="FP8"))),
    ]
    if letter == "A":
        f += [
            ("NVFP4 input scale zero", "not finite and positive",
             lambda d: edit_tensors(d, lambda t: t.update({f"{a}.mlp.gate_proj.input_scale": ("F32", [], f32(0.0)),
                                                           f"{a}.mlp.up_proj.input_scale": ("F32", [], f32(0.0))}))),
            ("q/k/g input scales unshared", "unshared scales",
             lambda d: edit_tensors(d, lambda t: t.update({f"{a}.self_attn.g_proj.input_scale": ("F32", [], f32(0.777))}))),
            ("FP8 weight left BF16", "wrong dtype",
             lambda d: edit_tensors(d, lambda t: t.update({f"{a}.self_attn.o_proj.weight": (
                 "BF16", [H, NH * HD], bytes(H * NH * HD * 2))}))),
            ("mixed fused group (k NVFP4)", "mixed algorithms",
             lambda d: [edit_json(d, n, (lambda j: (j.get("quantization_config") or j["quantization"])[
                 "quantized_layers"].update({f"{a}.self_attn.k_proj": {"quant_algo": "NVFP4", "group_size": 16}})))
                 for n in ("config.json", "hf_quant_config.json")]),
        ]
    if letter == "C":
        f.append(("attention weight changed (BF16 in C)", "BF16 tensors differ",
                  lambda d: edit_tensors(d, lambda t: set_byte(t, f"{a}.self_attn.q_proj.weight", 9,
                                                               t[f"{a}.self_attn.q_proj.weight"][2][9] ^ 1))))
    return f


def main() -> None:
    work = Path(sys.argv[1])
    keep = "--keep" in sys.argv
    if work.exists():
        shutil.rmtree(work)
    rng = random.Random(1234)
    src_t = make_source(work / "source", rng)
    results = []
    for letter in ("A", "B", "C"):
        clean = work / f"synthetic-{letter}"
        make_export(clean, src_t, letter, random.Random(ord(letter)))
        r = ve.validate(clean, work / "source", letter)
        if r["verdict"] != "PASS":
            raise SystemExit(f"VALIDATOR-SELFTEST-FAILED: clean synthetic {letter} failed: {r['errors'][:3]}")
        # The other letters' validators must reject this export (layouts differ).
        for other in ("A", "B", "C"):
            if other != letter and ve.validate(clean, work / "source", other)["verdict"] == "PASS":
                raise SystemExit(f"VALIDATOR-SELFTEST-FAILED: export {letter} passes as {other}")
        n_ok = 0
        for label, want, mut in faults(letter):
            bad = work / "mutant"
            if bad.exists():
                shutil.rmtree(bad)
            shutil.copytree(clean, bad)
            mut(bad)
            r = ve.validate(bad, work / "source", letter)
            if r["verdict"] != "FAIL":
                raise SystemExit(f"VALIDATOR-SELFTEST-FAILED: {letter} '{label}' passed the validator")
            if not any(want in e for e in r["errors"]):
                raise SystemExit(f"VALIDATOR-SELFTEST-FAILED: {letter} '{label}' failed, but not on '{want}': "
                                 f"{r['errors'][:3]}")
            n_ok += 1
        results.append(f"{letter}: clean PASS, {n_ok} planted faults caught")
        shutil.rmtree(work / "mutant", ignore_errors=True)
    if not keep:
        for letter in ("A", "B", "C"):
            shutil.rmtree(work / f"synthetic-{letter}", ignore_errors=True)
    print("VALIDATOR-SELFTEST-OK | " + " | ".join(results))


if __name__ == "__main__":
    main()
