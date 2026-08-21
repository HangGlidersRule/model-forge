"""Typed experiment specification for abliteration + NVFP4 pipeline."""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from model_forge.pipeline import canonical_json, sha256_bytes
from model_forge.recipe import (
    Recipe,
    RecipeError,
    load_recipe,
    validate_quantizer_ignore,
)


@dataclass(frozen=True)
class SourceSpec:
    model_id: str
    revision: str


@dataclass(frozen=True)
class DatasetRef:
    source: str
    revision: str
    sha256: str | None = None


@dataclass(frozen=True)
class AbliterationSpec:
    layer: int
    seed: int
    harmful_prompts: int
    harmless_prompts: int
    orthogonalize_harmless: bool
    target_selectors: tuple[str, ...]
    expected_target_count: int
    reject_visual_selectors: bool = True


@dataclass(frozen=True)
class QuantizationSpec:
    scheme: str
    targets: str
    group_size: int
    calibration_samples: int
    max_sequence_length: int
    calibration_dataset: str
    calibration_config: str
    pipeline: str
    shard_size: str
    ignore: tuple[str, ...]
    keep_bf16: tuple[str, ...]


@dataclass(frozen=True)
class ValidationSpec:
    max_refusal_leakage: float
    max_benign_kl_divergence: float
    max_perplexity_delta_pct: float
    vision_byte_identical: bool
    mtp_present: bool


@dataclass(frozen=True)
class RuntimeSpec:
    kv_dtype: str
    context_length: int
    compiled_mode: bool
    flash_attention: bool
    mtp_depth_initial: int
    mtp_sweep_range: tuple[int, ...]


@dataclass(frozen=True)
class PerformanceSpec:
    target_tok_s: int
    minimum_tok_s: int
    warmup_repeats: int
    measure_repeats: int
    prompt_lengths: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str
    name: str
    source: SourceSpec
    abliteration: AbliterationSpec
    datasets: dict[str, DatasetRef]
    quantization: QuantizationSpec
    validation: ValidationSpec
    runtime: RuntimeSpec
    performance: PerformanceSpec

    def config_sha(self) -> str:
        return sha256_bytes(canonical_json(self._to_hashable()).encode())

    def _to_hashable(self) -> dict[str, Any]:
        """Produce the complete canonical experiment contract for stage hashing."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "source": {"model_id": self.source.model_id, "revision": self.source.revision},
            "abliteration": {
                "layer": self.abliteration.layer,
                "seed": self.abliteration.seed,
                "harmful_prompts": self.abliteration.harmful_prompts,
                "harmless_prompts": self.abliteration.harmless_prompts,
                "orthogonalize_harmless": self.abliteration.orthogonalize_harmless,
                "target_selectors": list(self.abliteration.target_selectors),
                "expected_target_count": self.abliteration.expected_target_count,
                "reject_visual_selectors": self.abliteration.reject_visual_selectors,
            },
            "datasets": {
                key: {
                    "source": value.source,
                    "revision": value.revision,
                    "sha256": value.sha256,
                }
                for key, value in sorted(self.datasets.items())
            },
            "quantization": {
                "scheme": self.quantization.scheme,
                "targets": self.quantization.targets,
                "group_size": self.quantization.group_size,
                "calibration_samples": self.quantization.calibration_samples,
                "max_sequence_length": self.quantization.max_sequence_length,
                "calibration_dataset": self.quantization.calibration_dataset,
                "calibration_config": self.quantization.calibration_config,
                "pipeline": self.quantization.pipeline,
                "shard_size": self.quantization.shard_size,
                "ignore": list(self.quantization.ignore),
                "keep_bf16": list(self.quantization.keep_bf16),
            },
            "validation": {
                "max_refusal_leakage": self.validation.max_refusal_leakage,
                "max_benign_kl_divergence": self.validation.max_benign_kl_divergence,
                "max_perplexity_delta_pct": self.validation.max_perplexity_delta_pct,
                "vision_byte_identical": self.validation.vision_byte_identical,
                "mtp_present": self.validation.mtp_present,
            },
            "runtime": {
                "kv_dtype": self.runtime.kv_dtype,
                "context_length": self.runtime.context_length,
                "compiled_mode": self.runtime.compiled_mode,
                "flash_attention": self.runtime.flash_attention,
                "mtp_depth_initial": self.runtime.mtp_depth_initial,
                "mtp_sweep_range": list(self.runtime.mtp_sweep_range),
            },
            "performance": {
                "target_tok_s": self.performance.target_tok_s,
                "minimum_tok_s": self.performance.minimum_tok_s,
                "warmup_repeats": self.performance.warmup_repeats,
                "measure_repeats": self.performance.measure_repeats,
                "prompt_lengths": list(self.performance.prompt_lengths),
            },
        }


class ConfigError(ValueError):
    pass


SUPPORTED_TRANSFORM_TYPES: tuple[str, ...] = ("abliteration",)


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    unique: dict[str, None] = {}
    for value in values:
        unique.setdefault(value, None)
    return tuple(unique)


def effective_quantizer_ignore(
    targets: str, ignore: Sequence[str], protections: Sequence[str]
) -> tuple[str, ...]:
    """Project declared protections onto the quantizer ignore list, raising ``ConfigError``.

    This is a thin adapter over the generic :func:`model_forge.recipe.validate_quantizer_ignore`
    helper so the experiment API keeps raising :class:`ConfigError`. The coverage rules and the
    ``re:`` regex validation live in ``recipe.py`` and are reused here to avoid a circular import.
    """
    try:
        return validate_quantizer_ignore(targets, ignore, protections)
    except RecipeError as exc:
        raise ConfigError(str(exc)) from exc


def _parse_source(raw: dict[str, Any]) -> SourceSpec:
    rev = raw.get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise ConfigError(f"Source revision must be a full 40-char SHA: {rev!r}")
    return SourceSpec(model_id=raw["model_id"], revision=rev)


def _parse_abliteration(raw: dict[str, Any]) -> AbliterationSpec:
    selectors = tuple(raw["target_selectors"])
    for s in selectors:
        if "visual" in s.lower():
            raise ConfigError(f"Selector touches visual tower: {s!r}")
    return AbliterationSpec(
        layer=raw["layer"],
        seed=raw["seed"],
        harmful_prompts=raw["harmful_prompts"],
        harmless_prompts=raw["harmless_prompts"],
        orthogonalize_harmless=raw.get("orthogonalize_harmless", False),
        target_selectors=selectors,
        expected_target_count=raw["expected_target_count"],
        reject_visual_selectors=raw.get("reject_visual_selectors", True),
    )


def _parse_datasets(raw: dict[str, Any]) -> dict[str, DatasetRef]:
    result: dict[str, DatasetRef] = {}
    for key, value in raw.items():
        revision = value["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ConfigError(
                f"Dataset {key} revision must be a full 40-char SHA: {revision!r}"
            )
        result[key] = DatasetRef(
            source=value["source"], revision=revision, sha256=value.get("sha256")
        )
    return result


def _parse_quantization(raw: dict[str, Any]) -> QuantizationSpec:
    targets = raw["targets"]
    keep_bf16 = _ordered_unique(raw["keep_bf16"])
    return QuantizationSpec(
        scheme=raw["scheme"],
        targets=targets,
        group_size=raw["group_size"],
        calibration_samples=raw["calibration_samples"],
        max_sequence_length=raw["max_sequence_length"],
        calibration_dataset=raw["calibration_dataset"],
        calibration_config=raw["calibration_config"],
        pipeline=raw["pipeline"],
        shard_size=raw["shard_size"],
        ignore=effective_quantizer_ignore(targets, tuple(raw["ignore"]), keep_bf16),
        keep_bf16=keep_bf16,
    )


def _parse_validation(raw: dict[str, Any]) -> ValidationSpec:
    return ValidationSpec(
        max_refusal_leakage=raw["max_refusal_leakage"],
        max_benign_kl_divergence=raw["max_benign_kl_divergence"],
        max_perplexity_delta_pct=raw["max_perplexity_delta_pct"],
        vision_byte_identical=raw["vision_byte_identical"],
        mtp_present=raw["mtp_present"],
    )


def _parse_runtime(raw: dict[str, Any]) -> RuntimeSpec:
    return RuntimeSpec(
        kv_dtype=raw["kv_dtype"],
        context_length=raw["context_length"],
        compiled_mode=raw["compiled_mode"],
        flash_attention=raw["flash_attention"],
        mtp_depth_initial=raw["mtp_depth_initial"],
        mtp_sweep_range=tuple(raw["mtp_sweep_range"]),
    )


def _parse_performance(raw: dict[str, Any]) -> PerformanceSpec:
    return PerformanceSpec(
        target_tok_s=raw["target_tok_s"],
        minimum_tok_s=raw["minimum_tok_s"],
        warmup_repeats=raw["warmup_repeats"],
        measure_repeats=raw["measure_repeats"],
        prompt_lengths=tuple(raw["prompt_lengths"]),
    )


def experiment_from_recipe(recipe: Recipe) -> ExperimentConfig:
    """Project a generic recipe into the abliteration+NVFP4 experiment contract."""
    if len(recipe.transforms) != 1:
        raise ConfigError(
            "Experiment projection requires exactly one abliteration transform, got "
            f"{len(recipe.transforms)}: {[item.type for item in recipe.transforms]}"
        )
    transform = recipe.transforms[0]
    if transform.type not in SUPPORTED_TRANSFORM_TYPES:
        raise ConfigError(
            f"Unsupported transform type {transform.type!r}: this experiment adapter supports "
            f"only {', '.join(SUPPORTED_TRANSFORM_TYPES)}"
        )
    if recipe.quantization is None:
        raise ConfigError("Experiment projection requires quantization")
    if transform.layer is None or transform.seed is None:
        raise ConfigError("Abliteration transform missing layer/seed")
    if transform.harmful_prompts is None or transform.harmless_prompts is None:
        raise ConfigError("Abliteration transform missing prompt counts")
    if transform.expected_target_count is None:
        raise ConfigError("Abliteration transform missing expected_target_count")
    quant = recipe.quantization
    validation = recipe.validation
    runtime = recipe.runtime
    performance = recipe.performance
    if (
        validation.max_benign_kl_divergence is None
        or validation.max_perplexity_delta_pct is None
        or validation.vision_byte_identical is None
        or validation.mtp_present is None
    ):
        raise ConfigError("Experiment projection requires complete validation gates")
    if runtime.mtp_depth_initial is None:
        raise ConfigError("Experiment projection requires runtime.mtp_depth_initial")
    if performance is None or performance.target_tok_s is None or performance.minimum_tok_s is None:
        raise ConfigError("Experiment projection requires performance targets")
    if performance.warmup_repeats is None or performance.measure_repeats is None:
        raise ConfigError("Experiment projection requires performance repeats")
    protections = _ordered_unique((*quant.protected_tensors, *quant.keep_bf16))
    return ExperimentConfig(
        schema_version="1.0",
        name=recipe.name,
        source=SourceSpec(model_id=recipe.source.model_id, revision=recipe.source.revision),
        abliteration=AbliterationSpec(
            layer=transform.layer,
            seed=transform.seed,
            harmful_prompts=transform.harmful_prompts,
            harmless_prompts=transform.harmless_prompts,
            orthogonalize_harmless=transform.orthogonalize_harmless,
            target_selectors=transform.target_selectors,
            expected_target_count=transform.expected_target_count,
            reject_visual_selectors=transform.reject_visual_selectors,
        ),
        datasets={
            key: DatasetRef(source=value.source, revision=value.revision, sha256=value.sha256)
            for key, value in recipe.datasets.items()
        },
        quantization=QuantizationSpec(
            scheme=quant.scheme,
            targets=quant.targets,
            group_size=quant.group_size,
            calibration_samples=quant.calibration.samples,
            max_sequence_length=quant.calibration.max_sequence_length,
            calibration_dataset=quant.calibration.dataset,
            calibration_config=quant.calibration.config,
            pipeline=quant.calibration.pipeline,
            shard_size=quant.calibration.shard_size,
            ignore=effective_quantizer_ignore(quant.targets, quant.ignore, protections),
            keep_bf16=protections,
        ),
        validation=ValidationSpec(
            max_refusal_leakage=validation.max_refusal_leakage,
            max_benign_kl_divergence=validation.max_benign_kl_divergence,
            max_perplexity_delta_pct=validation.max_perplexity_delta_pct,
            vision_byte_identical=validation.vision_byte_identical,
            mtp_present=validation.mtp_present,
        ),
        runtime=RuntimeSpec(
            kv_dtype=runtime.kv_dtype,
            context_length=runtime.context_length,
            compiled_mode=runtime.compiled_mode,
            flash_attention=runtime.flash_attention,
            mtp_depth_initial=runtime.mtp_depth_initial,
            mtp_sweep_range=runtime.mtp_sweep_range,
        ),
        performance=PerformanceSpec(
            target_tok_s=performance.target_tok_s,
            minimum_tok_s=performance.minimum_tok_s,
            warmup_repeats=performance.warmup_repeats,
            measure_repeats=performance.measure_repeats,
            prompt_lengths=performance.prompt_lengths,
        ),
    )


def _load_legacy_experiment(raw: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        schema_version=raw["schema_version"],
        name=raw["name"],
        source=_parse_source(raw["source"]),
        abliteration=_parse_abliteration(raw["abliteration"]),
        datasets=_parse_datasets(raw["datasets"]),
        quantization=_parse_quantization(raw["quantization"]),
        validation=_parse_validation(raw["validation"]),
        runtime=_parse_runtime(raw["runtime"]),
        performance=_parse_performance(raw["performance"]),
    )


def load_experiment(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    version = raw.get("schema_version")
    if version == "2.0":
        try:
            return experiment_from_recipe(load_recipe(path))
        except RecipeError as exc:
            raise ConfigError(str(exc)) from exc
    if version != "1.0":
        raise ConfigError(f"Unsupported schema_version: {version}")
    return _load_legacy_experiment(raw)
