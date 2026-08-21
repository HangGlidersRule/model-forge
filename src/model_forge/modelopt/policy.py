"""ModelOpt module-policy resolution with last-match-wins semantics."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

Precision = Literal["nvfp4", "fp8", "bf16", "disabled"]

FINAL_EXCLUSION_GLOBS: tuple[str, ...] = (
    "*visual*",
    "*vision_tower*",
    "*vision_model*",
    "*embed_vision*",
    "*multi_modal_projector*",
    "*mtp*",
    "*linear_attn.conv1d*",
    "*linear_attn.in_proj_a*",
    "*linear_attn.in_proj_b*",
)

# Fused vLLM groups that must share one precision decision.
FUSED_GROUPS: tuple[tuple[str, ...], ...] = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("in_proj_qkv", "in_proj_z"),
    ("in_proj_a", "in_proj_b"),
)

VISION_TOKENS: tuple[str, ...] = (
    "visual",
    "vision_tower",
    "vision_model",
    "embed_vision",
    "multi_modal_projector",
)

# Representative Qwen3.8 dense hybrid module stems used for offline policy tests.
REPRESENTATIVE_QWEN_MODULES: tuple[str, ...] = (
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.up_proj",
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.1.self_attn.q_proj",
    "model.language_model.layers.1.self_attn.k_proj",
    "model.language_model.layers.1.self_attn.v_proj",
    "model.language_model.layers.1.self_attn.o_proj",
    "model.language_model.layers.2.linear_attn.in_proj_qkv",
    "model.language_model.layers.2.linear_attn.in_proj_z",
    "model.language_model.layers.2.linear_attn.in_proj_a",
    "model.language_model.layers.2.linear_attn.in_proj_b",
    "model.language_model.layers.2.linear_attn.out_proj",
    "model.language_model.layers.2.linear_attn.conv1d",
    "model.visual.blocks.0.mlp.fc1",
    "model.visual.blocks.0.mlp.fc2",
    "model.vision_tower.vision_model.encoder.layers.0.mlp.fc1",
    "model.multi_modal_projector.linear_1",
    "model.embed_vision.embed_positions",
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.self_attn.o_proj",
    "lm_head",
    "model.language_model.embed_tokens",
)


@dataclass(frozen=True)
class QuantizerRule:
    pattern: str
    enable: bool | None
    cfg: dict[str, Any] | None
    parent_class: str | None = None


@dataclass(frozen=True)
class ResolvedQuantizer:
    module: str
    quantizer: str  # weight_quantizer | input_quantizer
    enabled: bool
    precision: Precision
    matched_rule: str | None


class PolicyError(ValueError):
    pass


def _pinned_import_rules(name: str) -> list[QuantizerRule]:
    """Resolve the pinned upstream import used by the exact operator recipe."""
    if name != "shared_quant_cfg":
        raise PolicyError(f"Unsupported quant_cfg import: {name!r}")
    nvfp4 = {
        "num_bits": "e2m1",
        "block_sizes": {-1: 16, "type": "static"},
        "scale_bits": "e4m3",
    }
    fp8 = {"num_bits": "e4m3"}
    specs: list[tuple[str, bool | None, dict[str, Any] | None]] = [
        ("*", False, None),
        ("*mlp*gate_proj*weight_quantizer*", None, nvfp4),
        ("*mlp*up_proj*weight_quantizer*", None, nvfp4),
        ("*mlp*down_proj*weight_quantizer*", None, nvfp4),
        ("*self_attn*weight_quantizer", None, fp8),
        ("*self_attn*input_quantizer", None, fp8),
        ("*linear_attn.in_proj_qkv*weight_quantizer", None, fp8),
        ("*linear_attn.in_proj_qkv*input_quantizer", None, fp8),
        ("*linear_attn.in_proj_z*weight_quantizer", None, fp8),
        ("*linear_attn.in_proj_z*input_quantizer", None, fp8),
        ("*linear_attn.out_proj*weight_quantizer", None, fp8),
        ("*linear_attn.out_proj*input_quantizer", None, fp8),
        ("*lm_head*", False, None),
        ("*linear_attn.conv1d*", False, None),
        ("*linear_attn.in_proj_a*", False, None),
        ("*linear_attn.in_proj_b*", False, None),
        ("*visual*", False, None),
        ("*vision_tower*", False, None),
        ("*mtp*", False, None),
        ("*lm_head*weight_quantizer", None, nvfp4),
    ]
    return [QuantizerRule(pattern, enable, cfg) for pattern, enable, cfg in specs]


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    # ModelOpt patterns are glob-like; fnmatch is sufficient for offline checks.
    return re.compile(fnmatch.translate(pattern))


def load_quant_cfg(path: Path) -> list[QuantizerRule]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PolicyError(f"{path}: recipe root must be a mapping")
    quantize = raw.get("quantize")
    if not isinstance(quantize, dict):
        raise PolicyError(f"{path}: missing quantize mapping")
    cfg_list = quantize.get("quant_cfg")
    if not isinstance(cfg_list, list):
        raise PolicyError(f"{path}: quantize.quant_cfg must be a list")
    rules: list[QuantizerRule] = []
    for index, entry in enumerate(cfg_list):
        if not isinstance(entry, dict):
            raise PolicyError(f"{path}: quant_cfg[{index}] must be a mapping")
        imported = entry.get("$import")
        if imported is not None:
            if not isinstance(imported, str):
                raise PolicyError(f"{path}: quant_cfg[{index}].$import must be str")
            rules.extend(_pinned_import_rules(imported))
            continue
        name = entry.get("quantizer_name")
        if not isinstance(name, str):
            raise PolicyError(f"{path}: quant_cfg[{index}] missing quantizer_name")
        enable = entry.get("enable")
        if enable is not None and not isinstance(enable, bool):
            raise PolicyError(f"{path}: quant_cfg[{index}].enable must be bool")
        cfg = entry.get("cfg")
        if cfg is not None and not isinstance(cfg, dict):
            raise PolicyError(f"{path}: quant_cfg[{index}].cfg must be a mapping")
        parent = entry.get("parent_class")
        if parent is not None and not isinstance(parent, str):
            raise PolicyError(f"{path}: quant_cfg[{index}].parent_class must be str")
        rules.append(
            QuantizerRule(
                pattern=name,
                enable=enable,
                cfg=cfg,
                parent_class=parent,
            )
        )
    return rules


def _precision_from_cfg(cfg: dict[str, Any] | None) -> Precision:
    if not cfg:
        return "disabled"
    num_bits = cfg.get("num_bits")
    if num_bits == "e2m1":
        return "nvfp4"
    if num_bits in {"e4m3", 8, (4, 3)}:
        return "fp8"
    return "bf16"


def resolve_quantizer(
    module: str,
    quantizer: str,
    rules: list[QuantizerRule],
    *,
    parent_class: str | None = None,
) -> ResolvedQuantizer:
    """Resolve enable/precision for ``module.quantizer`` with last-match-wins."""
    full = f"{module}.{quantizer}"
    matched: QuantizerRule | None = None
    for rule in rules:
        if rule.parent_class is not None:
            if parent_class is None or rule.parent_class != parent_class:
                # Parent-class rules only apply when the caller supplies a class.
                if not _glob_to_regex(rule.pattern).match(full) and rule.pattern != "*":
                    continue
                if parent_class is None:
                    continue
        if _glob_to_regex(rule.pattern).match(full) or (
            rule.pattern != "*" and _glob_to_regex(rule.pattern).search(full)
        ):
            matched = rule
        elif rule.pattern == "*" and rule.parent_class is None:
            matched = rule
    if matched is None:
        return ResolvedQuantizer(module, quantizer, False, "bf16", None)
    enabled = True if matched.enable is None else matched.enable
    if matched.cfg is not None and matched.enable is None:
        enabled = True
    if not enabled:
        return ResolvedQuantizer(module, quantizer, False, "bf16", matched.pattern)
    return ResolvedQuantizer(
        module,
        quantizer,
        True,
        _precision_from_cfg(matched.cfg),
        matched.pattern,
    )


def resolve_module_policy(
    modules: list[str] | tuple[str, ...],
    rules: list[QuantizerRule],
) -> dict[str, ResolvedQuantizer]:
    resolved: dict[str, ResolvedQuantizer] = {}
    for module in modules:
        for quantizer in ("weight_quantizer", "input_quantizer"):
            key = f"{module}.{quantizer}"
            parent = "nn.Embedding" if "embed_tokens" in module else None
            resolved[key] = resolve_quantizer(module, quantizer, rules, parent_class=parent)
    return resolved


def is_vision_module(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in VISION_TOKENS)


def is_mtp_module(name: str) -> bool:
    return "mtp" in name.lower()


def assert_final_exclusions_present(rules: list[QuantizerRule]) -> None:
    """Fail closed if required last-match exclusions are missing or not terminal enough."""
    disabled = [
        rule.pattern
        for rule in rules
        if rule.enable is False and rule.parent_class is None
    ]
    missing = [glob for glob in FINAL_EXCLUSION_GLOBS if glob not in disabled]
    if missing:
        raise PolicyError(
            "Missing explicit final exclusions (last-match-wins): " + ", ".join(missing)
        )
    # Each exclusion must appear after any enable rule that could match vision/mtp/GDN.
    last_index = {rule.pattern: index for index, rule in enumerate(rules)}
    enable_indices = [
        index
        for index, rule in enumerate(rules)
        if rule.enable is not False and rule.cfg is not None
    ]
    last_enable = max(enable_indices) if enable_indices else -1
    for glob in FINAL_EXCLUSION_GLOBS:
        if last_index.get(glob, -1) < last_enable and glob in {
            "*visual*",
            "*vision_tower*",
            "*vision_model*",
            "*embed_vision*",
            "*multi_modal_projector*",
            "*mtp*",
        }:
            # Allow GDN exclusions to appear earlier when also duplicated later;
            # vision/mtp must be after the last enable that uses broad *mlp* globs.
            if last_index.get(glob, -1) <= last_enable:
                # Find the last occurrence
                last_occ = max(
                    (i for i, r in enumerate(rules) if r.pattern == glob and r.enable is False),
                    default=-1,
                )
                if last_occ <= last_enable:
                    raise PolicyError(
                        f"Exclusion {glob!r} must appear after the last enable rule "
                        f"(last_enable={last_enable}, exclusion={last_occ})"
                    )


def detect_fp8_kv_rules(rules: list[QuantizerRule]) -> list[str]:
    hits: list[str] = []
    for rule in rules:
        pattern = rule.pattern.lower()
        # QKV projection names contain the letters "kv" but are weight/activation
        # precision, not KV-cache quantization. Cache rules target output/kv_bmm
        # quantizers or carry the constant-amax marker.
        if (
            ("output_quantizer" in pattern or "kv_bmm_quantizer" in pattern)
            and rule.enable is not False
        ):
            hits.append(rule.pattern)
        if rule.cfg and rule.enable is not False:
            # Constant-amax FP8 KV cast marker
            if rule.cfg.get("use_constant_amax") is True:
                hits.append(rule.pattern)
    return hits


def fused_group_precisions(
    resolved: dict[str, ResolvedQuantizer],
) -> dict[tuple[str, ...], set[Precision]]:
    """Map each fused group to the set of enabled precisions observed."""
    result: dict[tuple[str, ...], set[Precision]] = {}
    for group in FUSED_GROUPS:
        precisions: set[Precision] = set()
        for key, item in resolved.items():
            if item.quantizer != "weight_quantizer":
                continue
            if any(token in key for token in group):
                if item.enabled:
                    precisions.add(item.precision)
                else:
                    precisions.add("bf16")
        if precisions:
            result[group] = precisions
    return result


def assert_no_mixed_fused_groups(resolved: dict[str, ResolvedQuantizer]) -> None:
    """Reject mixed precision inside any fused runtime group."""
    # Build per-layer group membership.
    layer_groups: dict[str, dict[str, Precision]] = {}
    for key, item in resolved.items():
        if item.quantizer != "weight_quantizer":
            continue
        for group in FUSED_GROUPS:
            members_hit = [token for token in group if f".{token}" in key or key.endswith(token)]
            if not members_hit:
                continue
            # Layer prefix is everything before the member token.
            token = members_hit[0]
            prefix = key.split(f".{token}")[0]
            bucket = layer_groups.setdefault(f"{prefix}|{'+'.join(group)}", {})
            bucket[token] = item.precision if item.enabled else "bf16"
    for label, members in layer_groups.items():
        # Only evaluate complete groups when all siblings are present in the inventory.
        group_name = label.split("|", 1)[1]
        expected = tuple(group_name.split("+"))
        if set(members) != set(expected):
            continue
        values = set(members.values())
        if len(values) > 1:
            raise PolicyError(f"Mixed precision in fused group {label}: {members}")


def coverage_report(
    resolved: dict[str, ResolvedQuantizer],
) -> dict[str, Any]:
    nvfp4 = sorted(
        key for key, item in resolved.items() if item.enabled and item.precision == "nvfp4"
    )
    fp8 = sorted(
        key for key, item in resolved.items() if item.enabled and item.precision == "fp8"
    )
    bf16 = sorted(key for key, item in resolved.items() if not item.enabled)
    vision_quantized = [key for key in nvfp4 + fp8 if is_vision_module(key)]
    mtp_quantized = [key for key in nvfp4 + fp8 if is_mtp_module(key)]
    return {
        "nvfp4_count": len(nvfp4),
        "fp8_count": len(fp8),
        "bf16_count": len(bf16),
        "nvfp4": nvfp4,
        "fp8": fp8,
        "vision_quantized": vision_quantized,
        "mtp_quantized": mtp_quantized,
    }
