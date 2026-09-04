"""Tests for experiment config parsing."""
from __future__ import annotations

from pathlib import Path

import pytest

from model_forge.experiment import (
    ConfigError,
    ExperimentConfig,
    effective_quantizer_ignore,
    load_experiment,
)


@pytest.fixture
def config_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "recipes"
        / "qwen3.8-27b"
        / "r3-nvfp4.yaml"
    )


def test_load_experiment_valid(config_path: Path) -> None:
    cfg = load_experiment(config_path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == "qwen3.8-27b-r3-nvfp4"
    assert cfg.source.revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert cfg.abliteration.layer == 38
    assert cfg.abliteration.expected_target_count == 131
    assert cfg.quantization.scheme == "NVFP4"


def test_config_sha_stable(config_path: Path) -> None:
    cfg = load_experiment(config_path)
    h1 = cfg.config_sha()
    h2 = cfg.config_sha()
    assert h1 == h2
    assert len(h1) == 64


def test_rejects_short_revision(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("""
schema_version: "1.0"
name: test
source:
  model_id: Qwen/Qwen3.8-27B
  revision: main
abliteration:
  layer: 38
  seed: 42
  harmful_prompts: 128
  harmless_prompts: 128
  orthogonalize_harmless: false
  target_selectors: ["model.embed_tokens.weight"]
  expected_target_count: 1
datasets:
  harmful: {source: advbench, revision: pinned}
quantization:
  scheme: NVFP4
  targets: Linear
  group_size: 16
  calibration_samples: 32
  max_sequence_length: 8192
  calibration_dataset: neuralmagic/calibration
  calibration_config: LLM
  pipeline: basic
  shard_size: 5GB
  ignore: [lm_head]
  keep_bf16: [mtp]
validation:
  max_refusal_leakage: 0.01
  max_benign_kl_divergence: 0.05
  max_perplexity_delta_pct: 5.0
  vision_byte_identical: true
  mtp_present: true
runtime:
  kv_dtype: bf16
  context_length: 126144
  compiled_mode: true
  flash_attention: true
  mtp_depth_initial: 1
  mtp_sweep_range: [1]
performance:
  target_tok_s: 200
  minimum_tok_s: 180
  warmup_repeats: 2
  measure_repeats: 5
  prompt_lengths: [4096]
""")
    with pytest.raises(ConfigError, match="full 40-char SHA"):
        load_experiment(p)


def test_rejects_visual_selector(tmp_path: Path) -> None:
    p = tmp_path / "visual.yaml"
    p.write_text("""
schema_version: "1.0"
name: test
source:
  model_id: Qwen/Qwen3.8-27B
  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
abliteration:
  layer: 38
  seed: 42
  harmful_prompts: 128
  harmless_prompts: 128
  orthogonalize_harmless: false
  target_selectors: ["re:.*visual.*"]
  expected_target_count: 1
datasets:
  harmful: {source: advbench, revision: pinned}
quantization:
  scheme: NVFP4
  targets: Linear
  group_size: 16
  calibration_samples: 32
  max_sequence_length: 8192
  calibration_dataset: neuralmagic/calibration
  calibration_config: LLM
  pipeline: basic
  shard_size: 5GB
  ignore: [lm_head]
  keep_bf16: [mtp]
validation:
  max_refusal_leakage: 0.01
  max_benign_kl_divergence: 0.05
  max_perplexity_delta_pct: 5.0
  vision_byte_identical: true
  mtp_present: true
runtime:
  kv_dtype: bf16
  context_length: 126144
  compiled_mode: true
  flash_attention: true
  mtp_depth_initial: 1
  mtp_sweep_range: [1]
performance:
  target_tok_s: 200
  minimum_tok_s: 180
  warmup_repeats: 2
  measure_repeats: 5
  prompt_lengths: [4096]
""")
    with pytest.raises(ConfigError, match="visual"):
        load_experiment(p)


def test_frozen_config(config_path: Path) -> None:
    cfg = load_experiment(config_path)
    with pytest.raises(Exception):
        cfg.name = "changed"  # type: ignore[misc]


SCHEMA2 = """
schema_version: "2.0"
name: demo
family: demo
source:
  model_id: org/model
  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
transforms:
  - type: abliteration
    layer: 38
    seed: 42
    harmful_prompts: 8
    harmless_prompts: 8
    orthogonalize_harmless: false
    target_selectors: ["model.language_model.embed_tokens.weight"]
    expected_target_count: 1
datasets:
  harmful:
    source: demo/harmful
    revision: 01cead01398926d81f7c52bdb790ee8cf77ebba7
quantization:
  scheme: NVFP4
  targets: Linear
  group_size: 16
  protected_tensors: [mtp, lm_head]
  keep_bf16: [mtp, lm_head]
  ignore: [lm_head, "re:.*mtp.*"]
  calibration:
    dataset: demo/cal
    config: LLM
    samples: 4
    max_sequence_length: 128
    pipeline: basic
    shard_size: 5GB
validation:
  max_refusal_leakage: 0.01
  max_benign_kl_divergence: 0.05
  max_perplexity_delta_pct: 5.0
  vision_byte_identical: true
  mtp_present: true
runtime:
  kv_dtype: bf16
  context_length: 8192
  compiled_mode: true
  flash_attention: true
  mtp_depth_initial: 6
  mtp_sweep_range: [6]
performance:
  target_tok_s: 200
  minimum_tok_s: 180
  warmup_repeats: 2
  measure_repeats: 5
  prompt_lengths: [4096]
outputs:
  artifact_kind: nvfp4
  publication:
    github: HangGlidersRule/model-forge
"""

LEGACY = """
schema_version: "1.0"
name: legacy
source:
  model_id: org/model
  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
abliteration:
  layer: 38
  seed: 42
  harmful_prompts: 8
  harmless_prompts: 8
  orthogonalize_harmless: false
  target_selectors: ["model.embed_tokens.weight"]
  expected_target_count: 1
datasets:
  harmful:
    source: demo/harmful
    revision: 01cead01398926d81f7c52bdb790ee8cf77ebba7
quantization:
  scheme: NVFP4
  targets: Linear
  group_size: 16
  calibration_samples: 4
  max_sequence_length: 128
  calibration_dataset: demo/cal
  calibration_config: LLM
  pipeline: basic
  shard_size: 5GB
  ignore: [lm_head, "re:.*mtp.*"]
  keep_bf16: [mtp]
validation:
  max_refusal_leakage: 0.01
  max_benign_kl_divergence: 0.05
  max_perplexity_delta_pct: 5.0
  vision_byte_identical: true
  mtp_present: true
runtime:
  kv_dtype: bf16
  context_length: 8192
  compiled_mode: true
  flash_attention: true
  mtp_depth_initial: 6
  mtp_sweep_range: [6]
performance:
  target_tok_s: 200
  minimum_tok_s: 180
  warmup_repeats: 2
  measure_repeats: 5
  prompt_lengths: [4096]
"""

ABLITERATION_TRANSFORM = """transforms:
  - type: abliteration
    layer: 38
    seed: 42
    harmful_prompts: 8
    harmless_prompts: 8
    orthogonalize_harmless: false
    target_selectors: ["model.language_model.embed_tokens.weight"]
    expected_target_count: 1
"""


def _write(tmp_path: Path, text: str, name: str = "recipe.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_schema2_fixture_projects_cleanly(tmp_path: Path) -> None:
    cfg = load_experiment(_write(tmp_path, SCHEMA2))
    assert cfg.abliteration.layer == 38
    assert cfg.quantization.ignore == ("lm_head", "re:.*mtp.*")


def test_rejects_additional_unsupported_transform(tmp_path: Path) -> None:
    text = SCHEMA2.replace(
        "    expected_target_count: 1\n", "    expected_target_count: 1\n  - type: prune\n"
    )
    with pytest.raises(ConfigError, match="prune"):
        load_experiment(_write(tmp_path, text))


def test_rejects_second_abliteration_transform(tmp_path: Path) -> None:
    text = SCHEMA2.replace(
        ABLITERATION_TRANSFORM,
        ABLITERATION_TRANSFORM + ABLITERATION_TRANSFORM.replace("transforms:\n", ""),
    )
    with pytest.raises(ConfigError, match="exactly one"):
        load_experiment(_write(tmp_path, text))


def test_rejects_unsupported_single_transform_type(tmp_path: Path) -> None:
    text = SCHEMA2.replace(ABLITERATION_TRANSFORM, "transforms:\n  - type: distillation\n")
    with pytest.raises(ConfigError, match="distillation"):
        load_experiment(_write(tmp_path, text))


def test_rejects_empty_transform_list(tmp_path: Path) -> None:
    text = SCHEMA2.replace(ABLITERATION_TRANSFORM, "transforms: []\n")
    with pytest.raises(ConfigError, match="exactly one"):
        load_experiment(_write(tmp_path, text))


def test_projected_ignore_deduplicates_and_preserves_order(tmp_path: Path) -> None:
    text = SCHEMA2.replace(
        '  ignore: [lm_head, "re:.*mtp.*"]\n',
        '  ignore: [lm_head, "re:.*mtp.*", lm_head, "re:.*mtp.*"]\n',
    )
    cfg = load_experiment(_write(tmp_path, text))
    assert cfg.quantization.ignore == ("lm_head", "re:.*mtp.*")


def test_projected_keep_bf16_unions_both_protection_lists(tmp_path: Path) -> None:
    text = SCHEMA2.replace("  protected_tensors: [mtp, lm_head]\n", "  protected_tensors: [mtp]\n")
    text = text.replace("  keep_bf16: [mtp, lm_head]\n", "  keep_bf16: [lm_head]\n")
    cfg = load_experiment(_write(tmp_path, text))
    assert cfg.quantization.keep_bf16 == ("mtp", "lm_head")


def test_uncovered_declared_protection_is_rejected(tmp_path: Path) -> None:
    text = SCHEMA2.replace("  keep_bf16: [mtp, lm_head]\n", "  keep_bf16: [mtp, lm_head, conv1d]\n")
    with pytest.raises(ConfigError, match="conv1d"):
        load_experiment(_write(tmp_path, text))


def test_uncovered_protected_tensor_is_rejected(tmp_path: Path) -> None:
    text = SCHEMA2.replace(
        "  protected_tensors: [mtp, lm_head]\n", "  protected_tensors: [mtp, lm_head, conv1d]\n"
    )
    with pytest.raises(ConfigError, match="conv1d"):
        load_experiment(_write(tmp_path, text))


def test_non_linear_categories_are_structurally_protected_for_linear_targets(
    tmp_path: Path,
) -> None:
    text = SCHEMA2.replace(
        "  keep_bf16: [mtp, lm_head]\n", "  keep_bf16: [mtp, lm_head, norms, embeddings]\n"
    )
    cfg = load_experiment(_write(tmp_path, text))
    assert cfg.quantization.ignore == ("lm_head", "re:.*mtp.*")
    assert "norms" in cfg.quantization.keep_bf16


def test_non_linear_categories_are_rejected_when_targets_exceed_linear(tmp_path: Path) -> None:
    text = SCHEMA2.replace("  targets: Linear\n", '  targets: "re:.*"\n')
    text = text.replace("  keep_bf16: [mtp, lm_head]\n", "  keep_bf16: [mtp, lm_head, norms]\n")
    with pytest.raises(ConfigError, match="norms"):
        load_experiment(_write(tmp_path, text))


def test_every_declared_protection_is_covered_by_projected_ignore(config_path: Path) -> None:
    cfg = load_experiment(config_path)
    assert effective_quantizer_ignore(
        cfg.quantization.targets, cfg.quantization.ignore, cfg.quantization.keep_bf16
    ) == cfg.quantization.ignore


def test_effective_quantizer_ignore_rejects_lost_protection() -> None:
    with pytest.raises(ConfigError, match=r"conv1d"):
        effective_quantizer_ignore("Linear", ("lm_head",), ("lm_head", "conv1d"))


def test_effective_quantizer_ignore_accepts_exact_and_regex_coverage() -> None:
    ignore = effective_quantizer_ignore(
        "Linear", ("lm_head", "re:.*mtp.*"), ("lm_head", "mtp", "norms", "embeddings")
    )
    assert ignore == ("lm_head", "re:.*mtp.*")


def test_effective_quantizer_ignore_rejects_invalid_regex_selector() -> None:
    with pytest.raises(ConfigError, match="regular expression"):
        effective_quantizer_ignore("Linear", ("re:[unclosed",), ())


def test_legacy_config_rejects_lost_protection(tmp_path: Path) -> None:
    text = LEGACY.replace("  keep_bf16: [mtp]\n", "  keep_bf16: [mtp, conv1d]\n")
    with pytest.raises(ConfigError, match="conv1d"):
        load_experiment(_write(tmp_path, text, "legacy.yaml"))


def test_legacy_config_projects_effective_ignore(tmp_path: Path) -> None:
    cfg = load_experiment(_write(tmp_path, LEGACY, "legacy.yaml"))
    assert cfg.quantization.ignore == ("lm_head", "re:.*mtp.*")
