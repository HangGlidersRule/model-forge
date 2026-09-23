"""Quantization validators for diffusion/DiT artifacts (the B3 gate).

Validates a quantized diffusion checkpoint's safetensors against the policy:
tensor counts, precision map vs policy pins, BF16 fallbacks (excluded blocks +
protected modules), NVFP4 scale integrity. Results feed ``_SUCCESS.json``.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_forge.modelopt.diffusion_policy import DiffusionPolicy


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, **self.details}


def _safetensors_header(path: Path) -> dict[str, Any]:
    """Parse a safetensors file's header without loading tensors."""
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header: dict[str, Any] = json.loads(f.read(n))
        return header


def _tensor_name(key: str) -> str:
    """Strip the exporter's quant suffixes so names match policy globs."""
    for suffix in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale", ".bias"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def validate_quantized_diffusion_checkpoint(
    *,
    checkpoint_dir: Path,
    policy: DiffusionPolicy,
    source_revision: str,
    num_blocks: int = 32,
) -> ValidationReport:
    """Run the B3 validators over an exported HF-style diffusion checkpoint.

    Expected layout: ``checkpoint_dir/transformer/*.safetensors`` (quantized DiT)
    plus unquantized ``text_encoder/`` and ``vae/`` directories.
    """
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    transformer_dir = checkpoint_dir / "transformer"
    if not transformer_dir.is_dir():
        return ValidationReport(ok=False, errors=[f"missing transformer dir: {transformer_dir}"])

    files = sorted(transformer_dir.glob("*.safetensors"))
    if not files:
        return ValidationReport(ok=False, errors=[f"no safetensors files in {transformer_dir}"])

    keys: list[str] = []
    for f in files:
        keys.extend(_safetensors_header(f).keys())
    keys = [k for k in keys if k != "__metadata__"]

    # --- tensor counts ---
    weight_keys = [k for k in keys if k.endswith(".weight")]
    scale_keys = [k for k in keys if k.endswith(".weight_scale") and not k.endswith("_scale_2")]
    details["tensor_counts"] = {
        "total_tensors": len(keys),
        "weights": len(weight_keys),
        "weight_scales": len(scale_keys),
        "files": [f.name for f in files],
    }
    if not weight_keys:
        errors.append("no weight tensors found")

    # --- precision map vs policy ---
    # presence of a scale tensor next to a weight = quantized module
    scale_bases = {_tensor_name(k) for k in scale_keys}
    weight_bases = {_tensor_name(k) for k in weight_keys}
    quantized = weight_bases & scale_bases

    buckets: dict[str, list[str]] = {"quantize": [], "protected": [], "outside": []}
    for base in sorted(quantized):
        buckets[policy.module_quant_decision(base)].append(base)
    details["quantized_modules"] = {
        "count": len(quantized),
        "misclassified": buckets["protected"] + buckets["outside"],
    }
    if buckets["protected"]:
        errors.append(
            f"{len(buckets['protected'])} quantized modules are policy-protected "
            f"(first: {buckets['protected'][:3]})"
        )

    # --- BF16 fallbacks: excluded blocks have no quantized modules ---
    import re

    excluded = set(range(policy.block_range_exclude_first)) | set(
        range(num_blocks - policy.block_range_exclude_last, num_blocks)
    )
    leaked = sorted(
        {
            b
            for b in (int(m.group(1)) for m in re.finditer(r"transformer_blocks\.(\d+)\.", " ".join(quantized)))
            if b in excluded
        }
    )
    if leaked:
        errors.append(f"excluded blocks leaked quantization: {leaked}")
    details["bf16_fallback_blocks"] = sorted(excluded)

    # protected modules must not be quantized
    for base in quantized:
        if policy.module_quant_decision(base) == "protected":
            errors.append(f"protected module quantized: {base}")

    # --- scale integrity (sampled; full sweep on first build) ---
    import safetensors.torch as st  # deferred: heavy import only in real validation

    sample = scale_keys[:64]
    zero_scales = []
    for sk in sample:
        for f in files:
            hdr = _safetensors_header(f)
            if sk in hdr:
                t = st.load_file(str(f), device="cpu").get(sk)
                if t is not None:
                    tfloat = t.float()
                    if bool((tfloat == 0).all()) or bool(
                        (tfloat.isnan() | tfloat.isinf()).any()
                    ):
                        zero_scales.append(sk)
                break
    if zero_scales:
        errors.append(f"degenerate scales (zero/NaN/Inf): {zero_scales[:5]}")
    details["scale_integrity_sampled"] = len(sample)

    # --- unquantized companion components exist ---
    for component in ("text_encoder", "vae"):
        if not (checkpoint_dir / component).is_dir():
            errors.append(f"missing unquantized component dir: {component}")

    details["source_revision"] = source_revision
    details["policy"] = policy.name
    return ValidationReport(
        ok=not errors, errors=errors, warnings=warnings, details=details
    )
