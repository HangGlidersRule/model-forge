#!/usr/bin/env python3
"""Normalize a Nano-Omni trunk checkpoint's ModelOpt layer names to the published `backbone.*` convention (issue #35).

The base W4A16 build at Hub commit f3ae211e names its quantized layers in the HF-transformers layout
(`model.layers.N.mixer.experts.{up,down}_proj.E`, `...shared_experts.{up,down}_proj.weight`, `lm_head.weight`) in
hf_quant_config.json and config.json:quantization_config, while its tensors are named `backbone.*` like every other
Darkstar Nano-Omni trunk. vLLM matches quantized layers by the tensor-side names, so this rewrites only the two JSON
files; every other file is hard-linked or copied unchanged (tensor bytes are never touched).

Fail-closed: every name must map, the mapped set must equal REFERENCE's quantized_layers name set (a trunk with the
backbone.* convention, e.g. the abliterated W4A16 product), every per-layer algorithm must be preserved, and every other
quantization field must already equal REFERENCE's. Every file is read through checkpoint_paths.source_file(): one that
resolves outside its checkpoint directory (other than a Hub cache snapshot's blobs/) fails the run.
Usage: normalize_quant_names.py TRUNK REFERENCE OUT"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

# The confinement helper sits beside this script (stdlib only); put its directory on the path so the import works
# however the script is loaded (run directly in a container, or by file path from the tests).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_paths import CheckpointPathError, source_file  # noqa: E402

RULES = (
    (re.compile(r"^model\.layers\.(\d+)\.mixer\.experts\.(up|down)_proj\.(\d+)$"),
     lambda m: f"backbone.layers.{m[1]}.mixer.experts.{m[3]}.{m[2]}_proj"),
    (re.compile(r"^model\.layers\.(\d+)\.mixer\.shared_experts\.(up|down)_proj\.weight$"),
     lambda m: f"backbone.layers.{m[1]}.mixer.shared_experts.{m[2]}_proj"),
    (re.compile(r"^lm_head\.weight$"), lambda m: "lm_head"),
)


def convert(name: str) -> str:
    if name.startswith("backbone.") or name == "lm_head":
        return name
    for pat, fn in RULES:
        m = pat.match(name)
        if m:
            return fn(m)
    raise SystemExit(f"NORMALIZE-FAILED: unmapped quantized layer name {name}")


def remap(layers: dict) -> dict:
    """Rename every layer; drop an explicit `group_size` only when it is 16, the fixed NVFP4 block size that the
    published backbone.* layout leaves implicit. Any other group size fails closed."""
    out: dict = {}
    for k, v in layers.items():
        n = convert(k)
        if n in out:
            raise SystemExit(f"NORMALIZE-FAILED: two layers map to {n}")
        v = dict(v)
        if "group_size" in v:
            if v["group_size"] != 16 or v.get("quant_algo") != "W4A16_NVFP4":
                raise SystemExit(f"NORMALIZE-FAILED: {k} has group_size {v['group_size']} / {v.get('quant_algo')}")
            del v["group_size"]
        out[n] = v
    return out


def main() -> int:
    try:
        return normalize(*(Path(a) for a in sys.argv[1:4]))
    except CheckpointPathError as error:
        raise SystemExit(f"NORMALIZE-FAILED: {error}") from None


def normalize(trunk: Path, ref: Path, out: Path) -> int:
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    hq = json.loads(source_file(trunk, "hf_quant_config.json").read_text())
    rq = json.loads(source_file(ref, "hf_quant_config.json").read_text())
    new_layers = remap(hq["quantization"]["quantized_layers"])
    ref_layers = rq["quantization"]["quantized_layers"]
    if set(new_layers) != set(ref_layers):
        raise SystemExit(f"NORMALIZE-FAILED: name sets differ: only here {sorted(set(new_layers) - set(ref_layers))[:3]}, "
                         f"only in reference {sorted(set(ref_layers) - set(new_layers))[:3]}")
    if any(new_layers[k] != ref_layers[k] for k in ref_layers):
        raise SystemExit("NORMALIZE-FAILED: a per-layer quantization algorithm differs from the reference")
    if {k: v for k, v in hq["quantization"].items() if k != "quantized_layers"} != \
       {k: v for k, v in rq["quantization"].items() if k != "quantized_layers"}:
        raise SystemExit("NORMALIZE-FAILED: quantization fields other than the layer names differ from the reference")
    cfg = json.loads(source_file(trunk, "config.json").read_text())
    qc = cfg["quantization_config"]
    qc["quantized_layers"] = remap(qc["quantized_layers"])
    rcfg = json.loads(source_file(ref, "config.json").read_text())
    if qc != rcfg["quantization_config"]:
        raise SystemExit("NORMALIZE-FAILED: config.json quantization_config still differs from the reference")
    hq["quantization"]["quantized_layers"] = new_layers
    # Resolve and confine every file to be copied before anything is written: a refusal leaves no OUT.partial.
    copies = {p.name: source_file(trunk, p.name) for p in sorted(trunk.iterdir())
              if not (p.is_dir() or p.name in ("hf_quant_config.json", "config.json", "manifest.sha256"))}
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        for name, src in copies.items():
            shutil.copyfile(src, tmp / name)
        (tmp / "hf_quant_config.json").write_text(json.dumps(hq, indent=4) + "\n")
        (tmp / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f"NORMALIZE-OK {out}: {len(new_layers)} quantized layer names mapped to the backbone.* convention; "
          f"tensor files copied unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
