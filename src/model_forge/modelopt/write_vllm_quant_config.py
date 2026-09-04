#!/usr/bin/env python3
"""Post-process a ModelOpt NVFP4 export for vLLM serving on Blackwell-class host.

CRITICAL: ModelOpt exports partial-quant artifacts with hf_quant_config.json
``quant_algo: NVFP4``, which vLLM maps to ``modelopt_fp4``. For a PARTIALLY
quantized model (only experts NVFP4, rest BF16) that path produces NaN logits
on every backend (fused, marlin, even emulation). The proven-working format is
the Qwen prod artifact: a ``quantization_config`` block in config.json that
vLLM maps to ``modelopt_mixed`` — same mechanism darkstar-Qwen serves today.

This writes that block (MIXED_PRECISION + quantized_layers over the actual
quantized modules) and is REQUIRED for nemotron_h NVFP4 artifacts.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MODULE_RE = re.compile(
    r"backbone\.layers\.\d+\.mixer\.(experts\.\d+|shared_experts)\.(up_proj|down_proj)(\.|$)"
)


def collect_quantized_layers(export_dir: Path) -> dict[str, dict[str, str]]:
    idx = json.loads((export_dir / "model.safetensors.index.json").read_text())
    prefixes = set()
    for key in idx["weight_map"]:
        if MODULE_RE.match(key):
            prefixes.add(key.rsplit(".", 1)[0])
    return {p: {"quant_algo": "W4A16_NVFP4"} for p in sorted(prefixes)}


def write_quantization_config(export_dir: Path, source_bf16_dir: Path) -> None:
    quantized = collect_quantized_layers(export_dir)
    if not quantized:
        raise SystemExit("no quantized modules found — refusing to write config")

    cfg = json.loads((source_bf16_dir / "config.json").read_text())
    cfg["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "producer": {"name": "modelopt", "version": "0.46.0rc2"},
        "kv_cache_scheme": None,
        "quantized_layers": quantized,
        "ignore": [],
    }
    out = export_dir / "config.json"
    out.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out} with {len(quantized)} quantized_layers ({out.stat().st_size} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--bf16-source", required=True, type=Path)
    args = ap.parse_args()
    write_quantization_config(args.export, args.bf16_source)


if __name__ == "__main__":
    main()
