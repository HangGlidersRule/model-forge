"""Generic recipe schema for model transform, quantization, and publication."""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from model_forge.pipeline import canonical_json, sha256_bytes

SCHEMA_VERSION = "2.0"

# Protection category names describing modules the compressor never quantizes when the
# quantization targets are restricted to Linear layers: norms are RMSNorm/LayerNorm and
# embeddings are nn.Embedding. They are not valid module selectors, so a ``targets: Linear``
# restriction honors them structurally instead of requiring an explicit ignore selector.
_NON_LINEAR_CATEGORIES = frozenset({"norms", "embeddings"})
_LINEAR_ONLY_TARGETS = frozenset({"linear"})


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    unique: dict[str, None] = {}
    for value in values:
        unique.setdefault(value, None)
    return tuple(unique)


def _ignore_patterns(selectors: Sequence[str]) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for selector in selectors:
        if not selector.startswith("re:"):
            continue
        try:
            patterns.append(re.compile(selector[3:]))
        except re.error as exc:
            raise RecipeError(
                f"Quantization ignore selector {selector!r} is not a valid regular expression: "
                f"{exc}"
            ) from exc
    return patterns


def validate_quantizer_ignore(
    targets: str, ignore: Sequence[str], protections: Sequence[str]
) -> tuple[str, ...]:
    """Return the ignore selectors the quantizer must receive, in stable order.

    The quantizer only consumes ``QuantizationSpec.ignore``, so every declared protection
    (``protected_tensors`` and ``keep_bf16``) has to be accounted for here. A protection is
    accepted when an explicit ignore selector already covers it (exact match, or a ``re:``
    selector matching it), or when it names a category that ``targets`` structurally excludes.
    Anything else is a recipe that claims a protection the quantizer would not honor, so it
    is rejected instead of being silently dropped or padded with a selector that matches no
    module. Invalid ``re:`` selectors are also rejected here.
    """
    selectors = _ordered_unique(ignore)
    patterns = _ignore_patterns(selectors)
    linear_only = targets.strip().lower() in _LINEAR_ONLY_TARGETS
    for token in _ordered_unique(protections):
        if token in selectors or any(pattern.search(token) for pattern in patterns):
            continue
        if linear_only and token in _NON_LINEAR_CATEGORIES:
            continue
        raise RecipeError(
            f"Declared protection {token!r} is not covered by quantization.ignore "
            f"{list(selectors)} for targets {targets!r}: add an exact or 're:' ignore selector "
            "matching the modules it protects, because the quantizer only consumes "
            "quantization.ignore"
        )
    return selectors


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
class TransformSpec:
    type: str
    layer: int | None = None
    seed: int | None = None
    harmful_prompts: int | None = None
    harmless_prompts: int | None = None
    orthogonalize_harmless: bool = False
    target_selectors: tuple[str, ...] = ()
    expected_target_count: int | None = None
    reject_visual_selectors: bool = True


@dataclass(frozen=True)
class CalibrationSpec:
    dataset: str
    config: str
    samples: int
    max_sequence_length: int
    pipeline: str = "basic"
    shard_size: str = "5GB"


@dataclass(frozen=True)
class QuantizationSpec:
    scheme: str
    targets: str
    group_size: int
    protected_tensors: tuple[str, ...]
    ignore: tuple[str, ...]
    calibration: CalibrationSpec
    keep_bf16: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationSpec:
    max_refusal_leakage: float
    max_benign_kl_divergence: float | None = None
    max_perplexity_delta_pct: float | None = None
    vision_byte_identical: bool | None = None
    mtp_present: bool | None = None


@dataclass(frozen=True)
class RuntimeSpec:
    kv_dtype: str
    context_length: int
    compiled_mode: bool = True
    flash_attention: bool = True
    mtp_depth_initial: int | None = None
    mtp_sweep_range: tuple[int, ...] = ()
    spec_decode: str = "mtp"
    drafter_model: str | None = None
    drafter_tokens: int | None = None


@dataclass(frozen=True)
class PerformanceSpec:
    target_tok_s: int | None = None
    minimum_tok_s: int | None = None
    warmup_repeats: int | None = None
    measure_repeats: int | None = None
    prompt_lengths: tuple[int, ...] = ()


@dataclass(frozen=True)
class PublicationSpec:
    github: str
    huggingface_bf16: str | None = None
    huggingface_nvfp4: str | None = None
    ghcr_serve: str | None = None
    ghcr_build: str | None = None


@dataclass(frozen=True)
class OutputsSpec:
    artifact_kind: str
    publication: PublicationSpec


@dataclass(frozen=True)
class Recipe:
    schema_version: str
    name: str
    family: str
    source: SourceSpec
    transforms: tuple[TransformSpec, ...]
    datasets: dict[str, DatasetRef]
    quantization: QuantizationSpec | None
    validation: ValidationSpec
    runtime: RuntimeSpec
    outputs: OutputsSpec
    performance: PerformanceSpec | None = None

    def config_sha(self) -> str:
        return sha256_bytes(canonical_json(self._to_hashable()).encode())

    def _to_hashable(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "family": self.family,
            "source": {"model_id": self.source.model_id, "revision": self.source.revision},
            "transforms": [
                {
                    "type": item.type,
                    "layer": item.layer,
                    "seed": item.seed,
                    "harmful_prompts": item.harmful_prompts,
                    "harmless_prompts": item.harmless_prompts,
                    "orthogonalize_harmless": item.orthogonalize_harmless,
                    "target_selectors": list(item.target_selectors),
                    "expected_target_count": item.expected_target_count,
                    "reject_visual_selectors": item.reject_visual_selectors,
                }
                for item in self.transforms
            ],
            "datasets": {
                key: {
                    "source": value.source,
                    "revision": value.revision,
                    "sha256": value.sha256,
                }
                for key, value in sorted(self.datasets.items())
            },
            "quantization": None
            if self.quantization is None
            else {
                "scheme": self.quantization.scheme,
                "targets": self.quantization.targets,
                "group_size": self.quantization.group_size,
                "protected_tensors": list(self.quantization.protected_tensors),
                "ignore": list(self.quantization.ignore),
                "keep_bf16": list(self.quantization.keep_bf16),
                "calibration": {
                    "dataset": self.quantization.calibration.dataset,
                    "config": self.quantization.calibration.config,
                    "samples": self.quantization.calibration.samples,
                    "max_sequence_length": self.quantization.calibration.max_sequence_length,
                    "pipeline": self.quantization.calibration.pipeline,
                    "shard_size": self.quantization.calibration.shard_size,
                },
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
                "spec_decode": self.runtime.spec_decode,
                "drafter_model": self.runtime.drafter_model,
                "drafter_tokens": self.runtime.drafter_tokens,
            },
            "performance": None
            if self.performance is None
            else {
                "target_tok_s": self.performance.target_tok_s,
                "minimum_tok_s": self.performance.minimum_tok_s,
                "warmup_repeats": self.performance.warmup_repeats,
                "measure_repeats": self.performance.measure_repeats,
                "prompt_lengths": list(self.performance.prompt_lengths),
            },
            "outputs": {
                "artifact_kind": self.outputs.artifact_kind,
                "publication": {
                    "github": self.outputs.publication.github,
                    "huggingface_bf16": self.outputs.publication.huggingface_bf16,
                    "huggingface_nvfp4": self.outputs.publication.huggingface_nvfp4,
                    "ghcr_serve": self.outputs.publication.ghcr_serve,
                    "ghcr_build": self.outputs.publication.ghcr_build,
                },
            },
        }


class RecipeError(ValueError):
    pass


def _kind(value: Any) -> str:
    return "null" if value is None else type(value).__name__


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecipeError(f"{label} must be a mapping, got {_kind(value)}")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RecipeError(f"{label} must be a sequence, got {_kind(value)}")
    return value


def _required(raw: dict[str, Any], key: str, label: str) -> Any:
    if raw.get(key) is None:
        raise RecipeError(f"{label} is missing required key {key!r}")
    return raw[key]


def _required_str(raw: dict[str, Any], key: str, label: str) -> str:
    value = _required(raw, key, label)
    if not isinstance(value, str):
        raise RecipeError(f"{label}.{key} must be a string, got {_kind(value)}")
    return value


def _required_int(raw: dict[str, Any], key: str, label: str) -> int:
    value = _required(raw, key, label)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecipeError(f"{label}.{key} must be an integer, got {_kind(value)}")
    return value


def _required_number(raw: dict[str, Any], key: str, label: str) -> float:
    value = _required(raw, key, label)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeError(f"{label}.{key} must be a number, got {_kind(value)}")
    return float(value)


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    items = _sequence(value, label)
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise RecipeError(f"{label}[{index}] must be a string, got {_kind(item)}")
    return tuple(items)


def _int_tuple(value: Any, label: str) -> tuple[int, ...]:
    if value is None:
        return ()
    items = _sequence(value, label)
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, int):
            raise RecipeError(f"{label}[{index}] must be an integer, got {_kind(item)}")
    return tuple(items)


def _require_sha(value: str, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise RecipeError(f"{label} must be a full 40-char SHA: {value!r}")
    return value


def _parse_source(value: Any, label: str = "source") -> SourceSpec:
    raw = _mapping(value, label)
    return SourceSpec(
        model_id=_required_str(raw, "model_id", label),
        revision=_require_sha(_required_str(raw, "revision", label), f"{label}.revision"),
    )


def _parse_datasets(value: Any, label: str = "datasets") -> dict[str, DatasetRef]:
    if value is None:
        return {}
    raw = _mapping(value, label)
    result: dict[str, DatasetRef] = {}
    for key, entry in raw.items():
        entry_label = f"{label}.{key}"
        item = _mapping(entry, entry_label)
        result[key] = DatasetRef(
            source=_required_str(item, "source", entry_label),
            revision=_require_sha(
                _required_str(item, "revision", entry_label), f"{entry_label}.revision"
            ),
            sha256=item.get("sha256"),
        )
    return result


def _positive_int(raw: dict[str, Any], key: str, label: str) -> int:
    value = _required_int(raw, key, label)
    if value <= 0:
        raise RecipeError(f"{label}.{key} must be a positive integer, got {value}")
    return value


def _nonnegative_int(raw: dict[str, Any], key: str, label: str) -> int:
    value = _required_int(raw, key, label)
    if value < 0:
        raise RecipeError(f"{label}.{key} must be a nonnegative integer, got {value}")
    return value


def _bool_field(raw: dict[str, Any], key: str, label: str, default: bool) -> bool:
    if raw.get(key) is None:
        return default
    value = raw[key]
    if not isinstance(value, bool):
        raise RecipeError(f"{label}.{key} must be a boolean, got {_kind(value)}")
    return value


def _optional_bool(raw: dict[str, Any], key: str, label: str) -> bool | None:
    if raw.get(key) is None:
        return None
    value = raw[key]
    if not isinstance(value, bool):
        raise RecipeError(f"{label}.{key} must be a boolean, got {_kind(value)}")
    return value


def _abliteration_selectors(raw: dict[str, Any], label: str) -> tuple[str, ...]:
    selectors_label = f"{label}.target_selectors"
    selectors = _string_tuple(raw.get("target_selectors"), selectors_label)
    if not selectors:
        raise RecipeError(
            f"{selectors_label} must be a nonempty sequence of selectors for "
            "abliteration transforms"
        )
    for index, selector in enumerate(selectors):
        if not selector.strip():
            raise RecipeError(f"{selectors_label}[{index}] must not be empty")
    return selectors


def _parse_transform(value: Any, label: str) -> TransformSpec:
    raw = _mapping(value, label)
    transform_type = _required_str(raw, "type", label)
    reject_visual = _bool_field(raw, "reject_visual_selectors", label, True)
    orthogonalize = _bool_field(raw, "orthogonalize_harmless", label, False)
    layer: int | None
    seed: int | None
    harmful_prompts: int | None
    harmless_prompts: int | None
    expected_target_count: int | None
    if transform_type == "abliteration":
        selectors = _abliteration_selectors(raw, label)
        layer = _positive_int(raw, "layer", label)
        seed = _nonnegative_int(raw, "seed", label)
        harmful_prompts = _positive_int(raw, "harmful_prompts", label)
        harmless_prompts = _positive_int(raw, "harmless_prompts", label)
        expected_target_count = _positive_int(raw, "expected_target_count", label)
    else:
        selectors = _string_tuple(raw.get("target_selectors"), f"{label}.target_selectors")
        layer = raw.get("layer")
        seed = raw.get("seed")
        harmful_prompts = raw.get("harmful_prompts")
        harmless_prompts = raw.get("harmless_prompts")
        expected_target_count = raw.get("expected_target_count")
    if reject_visual:
        for selector in selectors:
            if "visual" in selector.lower():
                raise RecipeError(f"Selector touches visual tower: {selector!r}")
    return TransformSpec(
        type=transform_type,
        layer=layer,
        seed=seed,
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmless_prompts,
        orthogonalize_harmless=orthogonalize,
        target_selectors=selectors,
        expected_target_count=expected_target_count,
        reject_visual_selectors=reject_visual,
    )


def _parse_transforms(value: Any, label: str = "transforms") -> tuple[TransformSpec, ...]:
    if value is None:
        return ()
    items = _sequence(value, label)
    return tuple(_parse_transform(item, f"{label}[{index}]") for index, item in enumerate(items))


def _parse_quantization(value: Any, label: str = "quantization") -> QuantizationSpec | None:
    if value is None:
        return None
    raw = _mapping(value, label)
    calibration_label = f"{label}.calibration"
    if not raw.get("calibration"):
        raise RecipeError(f"{label} requires calibration")
    calibration_raw = _mapping(raw["calibration"], calibration_label)
    keep_bf16 = _string_tuple(raw.get("keep_bf16"), f"{label}.keep_bf16") or _string_tuple(
        raw.get("protected_tensors"), f"{label}.protected_tensors"
    )
    protected = (
        _string_tuple(raw.get("protected_tensors"), f"{label}.protected_tensors") or keep_bf16
    )
    # Parse calibration first so structural calibration errors surface before the semantic
    # protection-coverage check below.
    calibration = CalibrationSpec(
        dataset=_required_str(calibration_raw, "dataset", calibration_label),
        config=_required_str(calibration_raw, "config", calibration_label),
        samples=_required_int(calibration_raw, "samples", calibration_label),
        max_sequence_length=_required_int(
            calibration_raw, "max_sequence_length", calibration_label
        ),
        pipeline=calibration_raw.get("pipeline", "basic"),
        shard_size=calibration_raw.get("shard_size", "5GB"),
    )
    targets = _required_str(raw, "targets", label)
    ignore = _string_tuple(raw.get("ignore"), f"{label}.ignore")
    # `model-forge recipe validate` must reject declared protections the quantizer would not
    # honor and invalid ignore regexes; this runs the same generic coverage check the
    # experiment adapter reuses.
    validate_quantizer_ignore(targets, ignore, _ordered_unique((*protected, *keep_bf16)))
    return QuantizationSpec(
        scheme=_required_str(raw, "scheme", label),
        targets=targets,
        group_size=_required_int(raw, "group_size", label),
        protected_tensors=protected,
        ignore=ignore,
        keep_bf16=keep_bf16,
        calibration=calibration,
    )


def _parse_validation(value: Any, label: str = "validation") -> ValidationSpec:
    raw = _mapping(value, label)
    return ValidationSpec(
        max_refusal_leakage=_required_number(raw, "max_refusal_leakage", label),
        max_benign_kl_divergence=raw.get("max_benign_kl_divergence"),
        max_perplexity_delta_pct=raw.get("max_perplexity_delta_pct"),
        vision_byte_identical=_optional_bool(raw, "vision_byte_identical", label),
        mtp_present=_optional_bool(raw, "mtp_present", label),
    )


def _parse_runtime(value: Any, label: str = "runtime") -> RuntimeSpec:
    raw = _mapping(value, label)
    spec_decode_value = raw.get("spec_decode", "mtp")
    if not isinstance(spec_decode_value, str):
        raise RecipeError(f"{label}.spec_decode must be a string, got {_kind(spec_decode_value)}")
    if spec_decode_value not in {"mtp", "dflash", "dflash2", "dspark"}:
        raise RecipeError(
            f"{label}.spec_decode must be one of mtp, dflash, dflash2, dspark; "
            f"got {spec_decode_value!r}"
        )
    drafter_model = raw.get("drafter_model")
    if drafter_model is not None and not isinstance(drafter_model, str):
        raise RecipeError(f"{label}.drafter_model must be a string, got {_kind(drafter_model)}")
    drafter_tokens = raw.get("drafter_tokens")
    if drafter_tokens is not None and (isinstance(drafter_tokens, bool) or not isinstance(drafter_tokens, int)):
        raise RecipeError(f"{label}.drafter_tokens must be an integer, got {_kind(drafter_tokens)}")
    if spec_decode_value in {"dflash", "dflash2", "dspark"}:
        if not drafter_model:
            raise RecipeError(f"{label}.drafter_model is required for {spec_decode_value}")
        if drafter_tokens is None or drafter_tokens <= 0:
            raise RecipeError(
                f"{label}.drafter_tokens must be a positive integer for {spec_decode_value}"
            )
    return RuntimeSpec(
        kv_dtype=_required_str(raw, "kv_dtype", label),
        context_length=_required_int(raw, "context_length", label),
        compiled_mode=_bool_field(raw, "compiled_mode", label, True),
        flash_attention=_bool_field(raw, "flash_attention", label, True),
        mtp_depth_initial=raw.get("mtp_depth_initial"),
        mtp_sweep_range=_int_tuple(raw.get("mtp_sweep_range"), f"{label}.mtp_sweep_range"),
        spec_decode=spec_decode_value,
        drafter_model=drafter_model,
        drafter_tokens=drafter_tokens,
    )


def _parse_performance(value: Any, label: str = "performance") -> PerformanceSpec | None:
    if value is None:
        return None
    raw = _mapping(value, label)
    return PerformanceSpec(
        target_tok_s=raw.get("target_tok_s"),
        minimum_tok_s=raw.get("minimum_tok_s"),
        warmup_repeats=raw.get("warmup_repeats"),
        measure_repeats=raw.get("measure_repeats"),
        prompt_lengths=_int_tuple(raw.get("prompt_lengths"), f"{label}.prompt_lengths"),
    )


def _parse_outputs(value: Any, label: str = "outputs") -> OutputsSpec:
    raw = _mapping(value, label)
    publication_label = f"{label}.publication"
    publication = _mapping(_required(raw, "publication", label), publication_label)
    return OutputsSpec(
        artifact_kind=_required_str(raw, "artifact_kind", label),
        publication=PublicationSpec(
            github=_required_str(publication, "github", publication_label),
            huggingface_bf16=publication.get("huggingface_bf16"),
            huggingface_nvfp4=publication.get("huggingface_nvfp4"),
            ghcr_serve=publication.get("ghcr_serve"),
            ghcr_build=publication.get("ghcr_build"),
        ),
    )


def _yaml_detail(exc: yaml.YAMLError) -> str:
    if isinstance(exc, yaml.MarkedYAMLError) and exc.problem:
        mark = exc.problem_mark
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        return f"{exc.problem}{where}"
    return " ".join(str(exc).split())


def _read_recipe_document(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecipeError(f"{path}: cannot read recipe: {exc.strerror or exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RecipeError(f"{path.name}: invalid YAML: {_yaml_detail(exc)}") from exc
    if raw is None:
        raise RecipeError(f"{path.name}: recipe is empty")
    return _mapping(raw, f"{path.name}: recipe")


def load_recipe(path: Path) -> Recipe:
    raw = _read_recipe_document(path)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise RecipeError(f"Unsupported schema_version: {raw.get('schema_version')}")
    try:
        return Recipe(
            schema_version=raw["schema_version"],
            name=_required_str(raw, "name", "recipe"),
            family=_required_str(raw, "family", "recipe"),
            source=_parse_source(_required(raw, "source", "recipe")),
            transforms=_parse_transforms(raw.get("transforms")),
            datasets=_parse_datasets(raw.get("datasets")),
            quantization=_parse_quantization(raw.get("quantization")),
            validation=_parse_validation(_required(raw, "validation", "recipe")),
            runtime=_parse_runtime(_required(raw, "runtime", "recipe")),
            outputs=_parse_outputs(_required(raw, "outputs", "recipe")),
            performance=_parse_performance(raw.get("performance")),
        )
    except RecipeError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RecipeError(f"{path.name}: malformed recipe: {exc}") from exc
