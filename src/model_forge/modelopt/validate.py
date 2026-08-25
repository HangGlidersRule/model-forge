"""Fail-closed validators for ModelOpt unified HF checkpoints."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_forge.modelopt.policy import (
    FINAL_EXCLUSION_GLOBS,
    REPRESENTATIVE_NEMOTRON_MODULES,
    REPRESENTATIVE_QWEN_MODULES,
    assert_final_exclusions_present,
    assert_no_mixed_fused_groups,
    coverage_report,
    detect_fp8_kv_rules,
    is_mtp_module,
    is_vision_module,
    load_quant_cfg,
    resolve_module_policy,
)

EXPECTED_MTP_TENSORS = 15
SCALE_NAME_MARKERS = ("weight_scale", "input_scale", "scale", "amax")


class ValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValidationError("; ".join(self.errors))


def validate_recipe_file(path: Path) -> ValidationReport:
    """Validate a local ModelOpt PTQ recipe before any GPU work."""
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}
    try:
        rules = load_quant_cfg(path)
        assert_final_exclusions_present(rules)
        kv_hits = detect_fp8_kv_rules(rules)
        if kv_hits:
            errors.append(f"FP8 KV rules present (must keep BF16 KV): {kv_hits}")
        is_nemotron = "nemotron_h" in path.name
        representative = (
            REPRESENTATIVE_NEMOTRON_MODULES if is_nemotron else REPRESENTATIVE_QWEN_MODULES
        )
        resolved = resolve_module_policy(representative, rules)
        assert_no_mixed_fused_groups(resolved)
        report = coverage_report(resolved)
        details["coverage"] = report
        if report["vision_quantized"]:
            errors.append(f"Vision modules would be quantized: {report['vision_quantized']}")
        if report["mtp_quantized"]:
            errors.append(f"MTP modules would be quantized: {report['mtp_quantized']}")
        if report["nvfp4_count"] == 0 and path.name.startswith("nvfp4_"):
            errors.append("Expected non-zero NVFP4 coverage on representative Qwen modules")
        mixed_w4a16 = path.name in (
            "w4a16_nvfp4_mse-fp8_attn-kv_bf16.yaml",
            "w4a16_nvfp4_mse-fp8_attn-kv_bf16_nemotron_h.yaml",
        )
        # Embeddings always stay disabled. The Qwen mixed operator recipe
        # intentionally quantizes lm_head in W4A16 NVFP4, but the
        # nemotron_h variant keeps lm_head BF16: vLLM cu130-nightly (SM120
        # Blackwell) loads lm_head via the raw vocab-parallel path and
        # cannot decode packed ModelOpt U8 weights.
        if path.name.startswith("w4a16_nvfp4_mse-fp8_attn-kv_bf16_nemotron_h"):
            mixed_w4a16_lm_head = False
        else:
            mixed_w4a16_lm_head = True
        for key, item in resolved.items():
            if item.enabled and (
                "embed_tokens" in key
                or ("lm_head" in key and not mixed_w4a16_lm_head)
            ):
                errors.append(f"Protected module quantized: {key}")
        if mixed_w4a16 and mixed_w4a16_lm_head:
            lm_head = resolved["lm_head.weight_quantizer"]
            if not lm_head.enabled or lm_head.precision != "nvfp4":
                errors.append("Selected mixed recipe must quantize lm_head in W4A16 NVFP4")
        details["final_exclusions"] = list(FINAL_EXCLUSION_GLOBS)
        details["rule_count"] = len(rules)
    except Exception as exc:  # noqa: BLE001 - fail closed with message
        errors.append(str(exc))
    return ValidationReport(ok=not errors, errors=errors, warnings=warnings, details=details)


def _tensor_names_from_checkpoint(root: Path) -> list[str]:
    index = root / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text(encoding="utf-8"))
        return sorted(data.get("weight_map", {}).keys())
    names: list[str] = []
    for path in sorted(root.glob("*.safetensors")):
        with path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        names.extend(key for key in header if key != "__metadata__")
    return sorted(set(names))


def validate_no_fp8_kv_metadata(root: Path) -> None:
    """Reject exported checkpoints that still carry FP8 KV quant metadata."""
    config_path = root / "hf_quant_config.json"
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8").lower()
        if "kv_bmm" in text or "kv_cache" in text and "e4m3" in text:
            # Narrow check: explicit kv quantizer entries.
            data = json.loads(config_path.read_text(encoding="utf-8"))
            blob = json.dumps(data).lower()
            if "kv_bmm_quantizer" in blob or '"kv_cache"' in blob:
                raise ValidationError("hf_quant_config.json contains FP8 KV metadata")
    for name in ("config.json", "hf_quant_config.json"):
        path = root / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        blob = json.dumps(data)
        if "kv_bmm_quantizer" in blob:
            raise ValidationError(f"{name} contains kv_bmm_quantizer entries")


def validate_mtp_bf16(root: Path, *, expected: int = EXPECTED_MTP_TENSORS) -> list[str]:
    names = _tensor_names_from_checkpoint(root)
    mtp = [name for name in names if name.startswith("mtp.")]
    if len(mtp) != expected:
        raise ValidationError(f"Expected {expected} MTP tensors, found {len(mtp)}")
    # Scale tensors under mtp.* indicate quantization.
    quantized = [
        name
        for name in mtp
        if any(marker in name for marker in ("weight_scale", "input_scale", "weight_packed"))
    ]
    if quantized:
        raise ValidationError(f"MTP tensors appear quantized: {quantized[:8]}")
    return mtp


def validate_no_quantized_vision(root: Path) -> None:
    names = _tensor_names_from_checkpoint(root)
    bad = [
        name
        for name in names
        if is_vision_module(name)
        and any(marker in name for marker in ("weight_scale", "input_scale", "weight_packed"))
    ]
    if bad:
        raise ValidationError(f"Vision tensors appear quantized: {bad[:8]}")


def _is_bad_number(value: float) -> bool:
    return math.isnan(value) or math.isinf(value) or value == 0.0


def validate_scales_finite(
    scale_values: dict[str, list[float]],
) -> None:
    """Reject NaN/Inf/zero/empty scale tensors provided by the caller.

    Real GPU builds should load safetensors and pass flattened scale samples here.
    Unit tests inject synthetic values without torch.
    """
    if not scale_values:
        raise ValidationError("No scale tensors provided for validation")
    for name, values in scale_values.items():
        if not values:
            raise ValidationError(f"Empty scale tensor: {name}")
        for value in values:
            if _is_bad_number(float(value)):
                raise ValidationError(f"Invalid scale in {name}: {value!r}")


def validate_tokenizer_config_unchanged(
    source_dir: Path,
    output_dir: Path,
    *,
    allowed_output_only: frozenset[str] = frozenset(
        {"hf_quant_config.json", "modelopt_state.pth", "quantization_config"}
    ),
) -> list[str]:
    """Ensure tokenizer/processor/chat assets match source except quantization metadata."""
    tracked = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "chat_template.json",
        "preprocessor_config.json",
        "processor_config.json",
        "generation_config.json",
    )
    drifts: list[str] = []
    for name in tracked:
        src = source_dir / name
        dst = output_dir / name
        if not src.exists():
            continue
        if not dst.exists():
            drifts.append(f"missing in output: {name}")
            continue
        if src.read_bytes() != dst.read_bytes():
            drifts.append(f"drift: {name}")
    # config.json may gain quantization metadata; reject unrelated field loss.
    src_cfg = source_dir / "config.json"
    dst_cfg = output_dir / "config.json"
    if src_cfg.exists() and dst_cfg.exists():
        source = json.loads(src_cfg.read_text(encoding="utf-8"))
        dest = json.loads(dst_cfg.read_text(encoding="utf-8"))
        for key in ("model_type", "architectures", "torch_dtype", "vocab_size"):
            if key in source and source.get(key) != dest.get(key):
                drifts.append(f"config.json field changed: {key}")
    _ = allowed_output_only  # documented for callers; not enforced as a hard file list
    if drifts:
        raise ValidationError("Tokenizer/config drift: " + "; ".join(drifts))
    return drifts


def validate_checkpoint_contract(
    output_dir: Path,
    *,
    source_dir: Path | None = None,
    scale_values: dict[str, list[float]] | None = None,
    recipe_path: Path | None = None,
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}
    try:
        if recipe_path is not None:
            recipe_report = validate_recipe_file(recipe_path)
            details["recipe"] = recipe_report.details
            errors.extend(recipe_report.errors)
        if not (output_dir / "hf_quant_config.json").exists():
            errors.append("Missing unified HF export metadata hf_quant_config.json")
        validate_no_fp8_kv_metadata(output_dir)
        details["mtp"] = validate_mtp_bf16(output_dir)
        validate_no_quantized_vision(output_dir)
        names = _tensor_names_from_checkpoint(output_dir)
        details["tensor_count"] = len(names)
        details["vision_tensor_count"] = sum(1 for name in names if is_vision_module(name))
        details["mtp_tensor_count"] = sum(1 for name in names if is_mtp_module(name))
        if scale_values is not None:
            validate_scales_finite(scale_values)
        if source_dir is not None:
            validate_tokenizer_config_unchanged(source_dir, output_dir)
    except ValidationError as exc:
        errors.append(str(exc))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"checkpoint validation crashed: {exc}")
    return ValidationReport(ok=not errors, errors=errors, warnings=warnings, details=details)
