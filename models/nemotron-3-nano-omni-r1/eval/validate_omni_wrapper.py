#!/usr/bin/env python3
"""Fail-closed validation of a reassembled Nano-Omni NVFP4 wrapper (issue #35). Byte-level; no torch.

Checks:
  1. Architecture: config.json is the upstream wrapper config verbatim apart from quantization_config
     (NemotronH_Nano_Omni_Reasoning_V3, with vision_config, sound_config and llm_config).
  2. Towers: every non-language_model tensor of UPSTREAM is present, with the same dtype, shape and SHA-256 of its
     bytes; nothing else outside language_model.*.
  3. Trunk: every TRUNK tensor is present under its language_model.* name with the same dtype, shape and SHA-256.
  4. Quantization config: the trunk's quantized_layers, every module name prefixed with language_model., values
     unchanged; every quantized module's weight tensor exists.
  5. Sidecars: code, processor, tokenizer and template files byte-identical to UPSTREAM; manifest.sha256 matches.
Writes a JSON report and prints VALIDATE-OK or VALIDATE-FAIL.
Usage: validate_omni_wrapper.py OUT TRUNK UPSTREAM REPORT_JSON"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

LM = "language_model."
SIDECARS = ("chat_template.jinja", "configuration.py", "configuration_nemotron_h.py", "configuration_radio.py", "modeling.py",
            "modeling_nemotron_h.py", "audio_model.py", "evs.py", "image_processing.py", "processing.py",
            "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
            "generation_config.json", "__init__.py")


def header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def tensor_digests(root: Path, names: set[str] | None = None) -> dict[str, tuple[str, list, str]]:
    """{name: (dtype, shape, sha256 of the tensor's bytes)} for every (or every named) tensor of a checkpoint."""
    idx = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for shard in sorted(set(idx.values())):
        h, off = header(root / shard)
        with (root / shard).open("rb") as f:
            for name, info in sorted(h.items(), key=lambda kv: kv[1]["data_offsets"][0]):
                if names is not None and name not in names:
                    continue
                begin, end = info["data_offsets"]
                f.seek(off + begin)
                d = hashlib.sha256()
                remaining = end - begin
                while remaining:
                    buf = f.read(min(remaining, 1 << 24))
                    d.update(buf)
                    remaining -= len(buf)
                out[name] = (info["dtype"], info["shape"], d.hexdigest())
    return out


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    out, trunk, upstream, report_path = (Path(a) for a in sys.argv[1:5])
    errors: list[str] = []
    rep: dict = {"output": out.name}
    o_cfg = json.loads((out / "config.json").read_text())
    u_cfg = json.loads((upstream / "config.json").read_text())
    t_cfg = json.loads((trunk / "config.json").read_text())
    drift = sorted(k for k in set(o_cfg) | set(u_cfg) if k != "quantization_config" and o_cfg.get(k) != u_cfg.get(k))
    if drift:
        errors.append(f"config.json differs from upstream outside quantization_config: {drift}")
    rep["architecture"] = o_cfg.get("architectures")
    for key in ("vision_config", "sound_config", "llm_config"):
        if key not in o_cfg:
            errors.append(f"config.json lacks {key}")

    u_idx = json.loads((upstream / "model.safetensors.index.json").read_text())["weight_map"]
    o_idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    t_idx = json.loads((trunk / "model.safetensors.index.json").read_text())["weight_map"]
    u_towers = {k for k in u_idx if not k.startswith(LM)}
    o_towers = {k for k in o_idx if not k.startswith(LM)}
    if o_towers != u_towers:
        errors.append(f"tower tensor set differs: missing {sorted(u_towers - o_towers)[:3]} extra {sorted(o_towers - u_towers)[:3]}")
    up_d = tensor_digests(upstream, u_towers)
    out_d = tensor_digests(out)
    bad_tower = [k for k in u_towers if out_d.get(k) != up_d.get(k)]
    rep["towers"] = {"expected": len(u_towers), "byte_identical": len(u_towers) - len(bad_tower),
                     "by_component": {c: sum(1 for k in u_towers if k.startswith(c)) for c in sorted({k.split(".")[0] for k in u_towers})}}
    if bad_tower:
        errors.append(f"{len(bad_tower)} tower tensors differ from upstream, e.g. {bad_tower[:3]}")
    tr_d = tensor_digests(trunk)
    bad_trunk = [k for k in tr_d if out_d.get(LM + k) != tr_d[k]]
    extra_lm = sorted(k for k in o_idx if k.startswith(LM) and k[len(LM):] not in t_idx)
    rep["trunk"] = {"expected": len(tr_d), "byte_identical": len(tr_d) - len(bad_trunk), "extra": len(extra_lm)}
    if bad_trunk or extra_lm:
        errors.append(f"trunk tensors: {len(bad_trunk)} differ, {len(extra_lm)} unexpected, e.g. {(bad_trunk + extra_lm)[:3]}")

    qc = o_cfg.get("quantization_config") or {}
    tq = t_cfg["quantization_config"]["quantized_layers"]
    want = {LM + k: v for k, v in tq.items()}
    if qc.get("quantized_layers") != want:
        errors.append("config quantized_layers is not the trunk's with the language_model. prefix")
    missing_w = [k for k in want if k + ".weight" not in o_idx]
    if missing_w:
        errors.append(f"{len(missing_w)} quantized modules have no weight tensor, e.g. {missing_w[:3]}")
    hq_path = out / "hf_quant_config.json"
    hq = json.loads(hq_path.read_text())["quantization"] if hq_path.exists() else {}
    if hq.get("quantized_layers") != want:
        errors.append("hf_quant_config quantized_layers is not the trunk's with the language_model. prefix")
    rep["quantization"] = {"quant_algo": qc.get("quant_algo"), "quantized_layers": len(want),
                           "algos": sorted({v.get("quant_algo") for v in want.values()})}

    side = {}
    for f in SIDECARS:
        if (upstream / f).exists():
            side[f] = (out / f).exists() and file_sha(out / f) == file_sha(upstream / f)
    rep["sidecars"] = {"checked": len(side), "byte_identical": sum(side.values())}
    if not all(side.values()):
        errors.append(f"sidecars differ from upstream: {[f for f, ok in side.items() if not ok]}")
    man = {}
    man_path = out / "manifest.sha256"
    if not man_path.exists():
        errors.append("manifest.sha256 missing")
    for line in (man_path.read_text().splitlines() if man_path.exists() else []):
        if line.strip():
            d, n = line.split(None, 1)
            man[n.strip()] = d
    bad_man = [n for n, d in man.items() if not (out / n).is_file() or file_sha(out / n) != d]
    rep["manifest"] = {"entries": len(man), "mismatches": len(bad_man)}
    if bad_man:
        errors.append(f"manifest mismatches: {bad_man[:3]}")

    rep["errors"] = errors
    rep["verdict"] = "PASS" if not errors else "FAIL"
    report_path.write_text(json.dumps(rep, indent=1) + "\n")
    print(("VALIDATE-OK " if not errors else "VALIDATE-FAIL ") + json.dumps({k: rep[k] for k in ("architecture", "towers", "trunk", "quantization", "sidecars", "manifest")}), flush=True)
    for e in errors:
        print("ERROR", e, flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
