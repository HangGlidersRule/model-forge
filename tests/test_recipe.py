"""Generic recipe schema loading and validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from model_forge.recipe import RecipeError, load_recipe, validate_quantizer_ignore

RECIPES = Path(__file__).resolve().parent.parent / "recipes" / "qwen3.8-27b"


def test_load_r3_nvfp4_recipe() -> None:
    recipe = load_recipe(RECIPES / "r3-nvfp4.yaml")
    assert recipe.name == "qwen3.8-27b-r3-nvfp4"
    assert recipe.family == "qwen3.8-27b"
    assert recipe.source.revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert recipe.transforms[0].type == "abliteration"
    assert recipe.transforms[0].expected_target_count == 131
    assert recipe.quantization is not None
    assert recipe.quantization.scheme == "NVFP4"
    assert recipe.quantization.calibration.samples == 32
    assert "mtp" in recipe.quantization.protected_tensors
    assert recipe.outputs.publication.github == "HangGlidersRule/model-forge"


def test_load_darkstar_abliterated_bf16_recipe() -> None:
    recipe = load_recipe(RECIPES / "darkstar-qwen3.8-27b-abliterated-bf16.yaml")
    assert recipe.name == "darkstar-qwen3.8-27b-abliterated-bf16"
    assert recipe.transforms[0].type == "abliteration"
    assert recipe.quantization is None
    assert recipe.outputs.artifact_kind == "bf16"


def test_load_base_nvfp4_recipe() -> None:
    recipe = load_recipe(RECIPES / "base-nvfp4.yaml")
    assert recipe.name == "qwen3.8-27b-base-nvfp4"
    assert recipe.transforms == ()
    assert recipe.quantization is not None
    assert recipe.quantization.scheme == "NVFP4"
    assert recipe.source.revision == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def test_recipe_config_sha_stable() -> None:
    recipe = load_recipe(RECIPES / "r3-nvfp4.yaml")
    assert recipe.config_sha() == recipe.config_sha()
    assert len(recipe.config_sha()) == 64


def test_recipe_runtime_defaults_to_mtp(tmp_path: Path) -> None:
    path = tmp_path / "mtp.yaml"
    path.write_text(
        """
schema_version: "2.0"
name: mtp-default
family: demo
source:
  model_id: org/model
  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
transforms: []
validation:
  max_refusal_leakage: 0.01
runtime:
  kv_dtype: bf16
  context_length: 8192
outputs:
  artifact_kind: bf16
  publication:
    github: HangGlidersRule/model-forge
"""
    )
    recipe = load_recipe(path)
    assert recipe.runtime.spec_decode == "mtp"
    assert recipe.runtime.drafter_model is None
    assert recipe.runtime.drafter_tokens is None


def test_recipe_runtime_reads_dflash2_drafter(tmp_path: Path) -> None:
    path = tmp_path / "dflash2.yaml"
    path.write_text(
        """
schema_version: "2.0"
name: dflash2-demo
family: demo
source:
  model_id: org/model
  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
transforms: []
validation:
  max_refusal_leakage: 0.01
runtime:
  kv_dtype: bf16
  context_length: 8192
  spec_decode: dflash2
  drafter_model: incoai/Qwen3.8-27B-DFlash2
  drafter_tokens: 7
outputs:
  artifact_kind: bf16
  publication:
    github: HangGlidersRule/model-forge
"""
    )
    recipe = load_recipe(path)
    assert recipe.runtime.spec_decode == "dflash2"
    assert recipe.runtime.drafter_model == "incoai/Qwen3.8-27B-DFlash2"
    assert recipe.runtime.drafter_tokens == 7


def test_recipe_rejects_invalid_spec_decode(tmp_path: Path) -> None:
    path = tmp_path / "bad-spec.yaml"
    path.write_text(
        """
schema_version: "2.0"
name: bad-spec
family: demo
source:
  model_id: org/model
  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
transforms: []
validation:
  max_refusal_leakage: 0.01
runtime:
  kv_dtype: bf16
  context_length: 8192
  spec_decode: nope
outputs:
  artifact_kind: bf16
  publication:
    github: HangGlidersRule/model-forge
"""
    )
    with pytest.raises(RecipeError, match="spec_decode"):
        load_recipe(path)


def test_rejects_floating_source_revision(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
schema_version: "2.0"
name: bad
family: demo
source:
  model_id: org/model
  revision: main
transforms: []
validation:
  max_refusal_leakage: 0.01
runtime:
  kv_dtype: bf16
  context_length: 8192
outputs:
  artifact_kind: bf16
  publication:
    github: HangGlidersRule/model-forge
"""
    )
    with pytest.raises(RecipeError, match="40-char SHA"):
        load_recipe(path)


def test_rejects_visual_transform_selector(tmp_path: Path) -> None:
    path = tmp_path / "visual.yaml"
    path.write_text(
        """
schema_version: "2.0"
name: visual
family: demo
source:
  model_id: org/model
  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
transforms:
  - type: abliteration
    layer: 1
    seed: 1
    harmful_prompts: 1
    harmless_prompts: 1
    orthogonalize_harmless: false
    target_selectors: ["re:.*visual.*"]
    expected_target_count: 1
    reject_visual_selectors: true
datasets:
  harmful:
    source: demo
    revision: 01cead01398926d81f7c52bdb790ee8cf77ebba7
validation:
  max_refusal_leakage: 0.01
runtime:
  kv_dtype: bf16
  context_length: 8192
outputs:
  artifact_kind: bf16
  publication:
    github: HangGlidersRule/model-forge
"""
    )
    with pytest.raises(RecipeError, match="visual"):
        load_recipe(path)


MINIMAL = """
schema_version: "2.0"
name: minimal
family: demo
source:
  model_id: org/model
  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
transforms: []
validation:
  max_refusal_leakage: 0.01
runtime:
  kv_dtype: bf16
  context_length: 8192
outputs:
  artifact_kind: bf16
  publication:
    github: HangGlidersRule/model-forge
"""


def _write(tmp_path: Path, text: str, name: str = "recipe.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_minimal_recipe_loads(tmp_path: Path) -> None:
    recipe = load_recipe(_write(tmp_path, MINIMAL))
    assert recipe.name == "minimal"
    assert recipe.quantization is None


def test_malformed_yaml_becomes_recipe_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "schema_version: \"2.0\"\nname: [unterminated\n")
    with pytest.raises(RecipeError, match="invalid YAML"):
        load_recipe(path)


def test_empty_recipe_becomes_recipe_error(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match="empty"):
        load_recipe(_write(tmp_path, ""))


def test_comment_only_recipe_becomes_recipe_error(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match="empty"):
        load_recipe(_write(tmp_path, "# nothing here\n"))


def test_scalar_recipe_becomes_recipe_error(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match="must be a mapping"):
        load_recipe(_write(tmp_path, "just-a-string\n"))


def test_sequence_recipe_becomes_recipe_error(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match="must be a mapping"):
        load_recipe(_write(tmp_path, "- one\n- two\n"))


@pytest.mark.parametrize("section", ["name", "family", "source", "validation", "runtime", "outputs"])
def test_missing_required_top_level_section(tmp_path: Path, section: str) -> None:
    lines = MINIMAL.splitlines(keepends=True)
    kept: list[str] = []
    dropping = False
    for line in lines:
        if line.startswith(f"{section}:"):
            dropping = True
            continue
        if dropping and (line.startswith(" ") or not line.strip()):
            continue
        dropping = False
        kept.append(line)
    path = _write(tmp_path, "".join(kept))
    with pytest.raises(RecipeError, match=section):
        load_recipe(path)


def test_missing_nested_key_names_its_field(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0\n", ""
    )
    with pytest.raises(RecipeError, match=r"source.*revision|revision.*source"):
        load_recipe(_write(tmp_path, text))


def test_missing_publication_key_names_its_field(tmp_path: Path) -> None:
    text = MINIMAL.replace("    github: HangGlidersRule/model-forge\n", "    other: x\n")
    with pytest.raises(RecipeError, match="publication"):
        load_recipe(_write(tmp_path, text))


def test_non_mapping_section_becomes_recipe_error(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "source:\n  model_id: org/model\n  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0\n",
        "source: org/model\n",
    )
    with pytest.raises(RecipeError, match="source must be a mapping"):
        load_recipe(_write(tmp_path, text))


def test_transforms_must_be_a_sequence(tmp_path: Path) -> None:
    text = MINIMAL.replace("transforms: []\n", "transforms: abliteration\n")
    with pytest.raises(RecipeError, match="transforms must be a sequence"):
        load_recipe(_write(tmp_path, text))


def test_transform_entry_must_be_a_mapping(tmp_path: Path) -> None:
    text = MINIMAL.replace("transforms: []\n", "transforms:\n  - abliteration\n")
    with pytest.raises(RecipeError, match=r"transforms\[0\] must be a mapping"):
        load_recipe(_write(tmp_path, text))


def test_dataset_entry_must_be_a_mapping(tmp_path: Path) -> None:
    text = MINIMAL + "datasets:\n  harmful: advbench\n"
    with pytest.raises(RecipeError, match=r"datasets.harmful must be a mapping"):
        load_recipe(_write(tmp_path, text))


def test_dataset_entry_missing_key_names_its_field(tmp_path: Path) -> None:
    text = MINIMAL + "datasets:\n  harmful:\n    source: advbench\n"
    with pytest.raises(RecipeError, match=r"datasets.harmful.*revision"):
        load_recipe(_write(tmp_path, text))


def test_non_string_revision_becomes_recipe_error(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0\n", "  revision: 12345\n"
    )
    with pytest.raises(RecipeError, match="revision"):
        load_recipe(_write(tmp_path, text))


def test_quantization_must_be_a_mapping(tmp_path: Path) -> None:
    text = MINIMAL + "quantization: NVFP4\n"
    with pytest.raises(RecipeError, match="quantization must be a mapping"):
        load_recipe(_write(tmp_path, text))


def test_quantization_calibration_missing_key_names_its_field(tmp_path: Path) -> None:
    text = MINIMAL + """quantization:
  scheme: NVFP4
  targets: Linear
  group_size: 16
  protected_tensors: [lm_head]
  calibration:
    dataset: demo
    config: default
    samples: 32
"""
    with pytest.raises(RecipeError, match=r"quantization.calibration.*max_sequence_length"):
        load_recipe(_write(tmp_path, text))


def test_unreadable_recipe_becomes_recipe_error(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match="cannot read"):
        load_recipe(tmp_path / "does-not-exist.yaml")


def _with_transform(body: str) -> str:
    return MINIMAL.replace("transforms: []\n", "transforms:\n" + body)


ABLITERATION_BODY = """  - type: abliteration
    layer: 1
    seed: 1
    harmful_prompts: 1
    harmless_prompts: 1
    target_selectors: ["model.layers.0.mlp.down_proj.weight"]
    expected_target_count: 1
"""


def test_abliteration_transform_with_valid_targets_loads(tmp_path: Path) -> None:
    recipe = load_recipe(_write(tmp_path, _with_transform(ABLITERATION_BODY)))
    assert recipe.transforms[0].target_selectors == ("model.layers.0.mlp.down_proj.weight",)
    assert recipe.transforms[0].expected_target_count == 1


def test_abliteration_requires_target_selectors_key(tmp_path: Path) -> None:
    body = ABLITERATION_BODY.replace(
        '    target_selectors: ["model.layers.0.mlp.down_proj.weight"]\n', ""
    )
    with pytest.raises(RecipeError, match=r"transforms\[0\].target_selectors.*nonempty"):
        load_recipe(_write(tmp_path, _with_transform(body)))


def test_abliteration_rejects_empty_target_selectors(tmp_path: Path) -> None:
    body = ABLITERATION_BODY.replace(
        '    target_selectors: ["model.layers.0.mlp.down_proj.weight"]\n',
        "    target_selectors: []\n",
    )
    with pytest.raises(RecipeError, match=r"transforms\[0\].target_selectors.*nonempty"):
        load_recipe(_write(tmp_path, _with_transform(body)))


def test_abliteration_rejects_blank_target_selector(tmp_path: Path) -> None:
    body = ABLITERATION_BODY.replace(
        '    target_selectors: ["model.layers.0.mlp.down_proj.weight"]\n',
        '    target_selectors: ["   "]\n',
    )
    with pytest.raises(RecipeError, match=r"transforms\[0\].target_selectors\[0\].*empty"):
        load_recipe(_write(tmp_path, _with_transform(body)))


def test_abliteration_requires_expected_target_count(tmp_path: Path) -> None:
    body = ABLITERATION_BODY.replace("    expected_target_count: 1\n", "")
    with pytest.raises(RecipeError, match=r"transforms\[0\].*expected_target_count"):
        load_recipe(_write(tmp_path, _with_transform(body)))


@pytest.mark.parametrize("value", ["0", "-1", "-131"])
def test_abliteration_rejects_non_positive_expected_target_count(
    tmp_path: Path, value: str
) -> None:
    body = ABLITERATION_BODY.replace(
        "    expected_target_count: 1\n", f"    expected_target_count: {value}\n"
    )
    with pytest.raises(
        RecipeError, match=r"transforms\[0\].expected_target_count must be a positive integer"
    ):
        load_recipe(_write(tmp_path, _with_transform(body)))


@pytest.mark.parametrize("value", ["'131'", "1.5", "true", "[131]"])
def test_abliteration_rejects_wrongly_typed_expected_target_count(
    tmp_path: Path, value: str
) -> None:
    body = ABLITERATION_BODY.replace(
        "    expected_target_count: 1\n", f"    expected_target_count: {value}\n"
    )
    with pytest.raises(
        RecipeError, match=r"transforms\[0\].expected_target_count must be an integer"
    ):
        load_recipe(_write(tmp_path, _with_transform(body)))


def test_abliteration_rejects_wrongly_typed_target_selectors(tmp_path: Path) -> None:
    body = ABLITERATION_BODY.replace(
        '    target_selectors: ["model.layers.0.mlp.down_proj.weight"]\n',
        "    target_selectors: model.layers.0.mlp.down_proj.weight\n",
    )
    with pytest.raises(RecipeError, match=r"transforms\[0\].target_selectors must be a sequence"):
        load_recipe(_write(tmp_path, _with_transform(body)))


def test_abliteration_requires_layer(tmp_path: Path) -> None:
    body = ABLITERATION_BODY.replace("    layer: 1\n", "")
    with pytest.raises(RecipeError, match=r"transforms\[0\].*layer"):
        load_recipe(_write(tmp_path, _with_transform(body)))


@pytest.mark.parametrize("field", ["layer", "harmful_prompts", "harmless_prompts"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_abliteration_rejects_non_positive_int_fields(
    tmp_path: Path, field: str, value: str
) -> None:
    body = ABLITERATION_BODY.replace(f"    {field}: 1\n", f"    {field}: {value}\n")
    with pytest.raises(
        RecipeError, match=rf"transforms\[0\].{field} must be a positive integer"
    ):
        load_recipe(_write(tmp_path, _with_transform(body)))


@pytest.mark.parametrize("field", ["layer", "seed", "harmful_prompts", "harmless_prompts"])
@pytest.mark.parametrize("value", ["'1'", "1.5", "true", "[1]"])
def test_abliteration_rejects_wrongly_typed_int_fields(
    tmp_path: Path, field: str, value: str
) -> None:
    body = ABLITERATION_BODY.replace(f"    {field}: 1\n", f"    {field}: {value}\n")
    with pytest.raises(RecipeError, match=rf"transforms\[0\].{field} must be an integer"):
        load_recipe(_write(tmp_path, _with_transform(body)))


def test_abliteration_accepts_zero_seed(tmp_path: Path) -> None:
    body = ABLITERATION_BODY.replace("    seed: 1\n", "    seed: 0\n")
    recipe = load_recipe(_write(tmp_path, _with_transform(body)))
    assert recipe.transforms[0].seed == 0


def test_abliteration_rejects_negative_seed(tmp_path: Path) -> None:
    body = ABLITERATION_BODY.replace("    seed: 1\n", "    seed: -1\n")
    with pytest.raises(
        RecipeError, match=r"transforms\[0\].seed must be a nonnegative integer"
    ):
        load_recipe(_write(tmp_path, _with_transform(body)))


@pytest.mark.parametrize("value", ["'true'", "1", "1.5", "[true]"])
def test_abliteration_rejects_non_bool_orthogonalize_harmless(
    tmp_path: Path, value: str
) -> None:
    body = ABLITERATION_BODY + f"    orthogonalize_harmless: {value}\n"
    with pytest.raises(
        RecipeError, match=r"transforms\[0\].orthogonalize_harmless must be a boolean"
    ):
        load_recipe(_write(tmp_path, _with_transform(body)))


def test_abliteration_accepts_bool_orthogonalize_harmless(tmp_path: Path) -> None:
    body = ABLITERATION_BODY + "    orthogonalize_harmless: true\n"
    recipe = load_recipe(_write(tmp_path, _with_transform(body)))
    assert recipe.transforms[0].orthogonalize_harmless is True


@pytest.mark.parametrize("value", ["'true'", "1", "1.5", "[true]"])
def test_abliteration_rejects_non_bool_reject_visual_selectors(
    tmp_path: Path, value: str
) -> None:
    body = ABLITERATION_BODY + f"    reject_visual_selectors: {value}\n"
    with pytest.raises(
        RecipeError, match=r"transforms\[0\].reject_visual_selectors must be a boolean"
    ):
        load_recipe(_write(tmp_path, _with_transform(body)))


@pytest.mark.parametrize("field", ["compiled_mode", "flash_attention"])
@pytest.mark.parametrize("value", ["'true'", "1", "1.5", "[true]"])
def test_runtime_rejects_non_bool_flags(tmp_path: Path, field: str, value: str) -> None:
    text = MINIMAL.replace(
        "  context_length: 8192\n", f"  context_length: 8192\n  {field}: {value}\n"
    )
    with pytest.raises(RecipeError, match=rf"runtime.{field} must be a boolean"):
        load_recipe(_write(tmp_path, text))


@pytest.mark.parametrize("field", ["compiled_mode", "flash_attention"])
def test_runtime_accepts_bool_flags(tmp_path: Path, field: str) -> None:
    text = MINIMAL.replace(
        "  context_length: 8192\n", f"  context_length: 8192\n  {field}: false\n"
    )
    recipe = load_recipe(_write(tmp_path, text))
    assert getattr(recipe.runtime, field) is False


@pytest.mark.parametrize("field", ["vision_byte_identical", "mtp_present"])
@pytest.mark.parametrize("value", ["'true'", "1", "1.5", "[true]"])
def test_validation_rejects_non_bool_optional_flags(
    tmp_path: Path, field: str, value: str
) -> None:
    text = MINIMAL.replace(
        "  max_refusal_leakage: 0.01\n",
        f"  max_refusal_leakage: 0.01\n  {field}: {value}\n",
    )
    with pytest.raises(RecipeError, match=rf"validation.{field} must be a boolean"):
        load_recipe(_write(tmp_path, text))


def test_validation_accepts_bool_optional_flags(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        "  max_refusal_leakage: 0.01\n",
        "  max_refusal_leakage: 0.01\n  vision_byte_identical: true\n  mtp_present: false\n",
    )
    recipe = load_recipe(_write(tmp_path, text))
    assert recipe.validation.vision_byte_identical is True
    assert recipe.validation.mtp_present is False


def test_other_transform_types_do_not_require_abliteration_fields(tmp_path: Path) -> None:
    recipe = load_recipe(_write(tmp_path, _with_transform("  - type: prune\n")))
    assert recipe.transforms[0].type == "prune"
    assert recipe.transforms[0].target_selectors == ()
    assert recipe.transforms[0].expected_target_count is None


QUANT_RECIPE = MINIMAL + """quantization:
  scheme: NVFP4
  targets: Linear
  group_size: 16
  protected_tensors: [lm_head, mtp, norms]
  keep_bf16: [lm_head, mtp, norms]
  ignore: [lm_head, "re:.*mtp.*"]
  calibration:
    dataset: demo
    config: LLM
    samples: 32
    max_sequence_length: 8192
    pipeline: basic
    shard_size: 5GB
"""


def test_quantization_recipe_with_covered_protections_loads(tmp_path: Path) -> None:
    recipe = load_recipe(_write(tmp_path, QUANT_RECIPE))
    assert recipe.quantization is not None
    assert recipe.quantization.ignore == ("lm_head", "re:.*mtp.*")


def test_load_recipe_rejects_uncovered_protection(tmp_path: Path) -> None:
    text = QUANT_RECIPE.replace(
        "  protected_tensors: [lm_head, mtp, norms]\n",
        "  protected_tensors: [lm_head, mtp, norms, conv1d]\n",
    ).replace(
        "  keep_bf16: [lm_head, mtp, norms]\n",
        "  keep_bf16: [lm_head, mtp, norms, conv1d]\n",
    )
    with pytest.raises(RecipeError, match="conv1d"):
        load_recipe(_write(tmp_path, text))


def test_load_recipe_rejects_invalid_ignore_regex(tmp_path: Path) -> None:
    text = QUANT_RECIPE.replace(
        '  ignore: [lm_head, "re:.*mtp.*"]\n', '  ignore: ["re:[unclosed"]\n'
    )
    with pytest.raises(RecipeError, match="regular expression"):
        load_recipe(_write(tmp_path, text))


def test_load_recipe_rejects_non_linear_category_when_targets_exceed_linear(
    tmp_path: Path,
) -> None:
    text = QUANT_RECIPE.replace("  targets: Linear\n", '  targets: "re:.*"\n')
    with pytest.raises(RecipeError, match="norms"):
        load_recipe(_write(tmp_path, text))


def test_load_recipe_reports_calibration_before_ignore_coverage(tmp_path: Path) -> None:
    """Calibration structural errors surface before protection-coverage checks."""
    text = MINIMAL + """quantization:
  scheme: NVFP4
  targets: Linear
  group_size: 16
  protected_tensors: [conv1d]
  calibration:
    dataset: demo
    config: default
    samples: 32
"""
    with pytest.raises(RecipeError, match=r"quantization.calibration.*max_sequence_length"):
        load_recipe(_write(tmp_path, text))


def test_validate_quantizer_ignore_rejects_lost_protection() -> None:
    with pytest.raises(RecipeError, match="conv1d"):
        validate_quantizer_ignore("Linear", ("lm_head",), ("lm_head", "conv1d"))


def test_validate_quantizer_ignore_rejects_invalid_regex() -> None:
    with pytest.raises(RecipeError, match="regular expression"):
        validate_quantizer_ignore("Linear", ("re:[unclosed",), ())


def test_validate_quantizer_ignore_returns_ordered_unique() -> None:
    assert validate_quantizer_ignore(
        "Linear", ("lm_head", "lm_head", "re:.*mtp.*"), ("lm_head", "mtp", "norms")
    ) == ("lm_head", "re:.*mtp.*")


def test_requires_calibration_when_quantizing(tmp_path: Path) -> None:
    path = tmp_path / "no-cal.yaml"
    path.write_text(
        """
schema_version: "2.0"
name: no-cal
family: demo
source:
  model_id: org/model
  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
transforms: []
quantization:
  scheme: NVFP4
  targets: Linear
  group_size: 16
  protected_tensors: [lm_head]
  ignore: [lm_head]
validation:
  max_refusal_leakage: 0.01
runtime:
  kv_dtype: bf16
  context_length: 8192
outputs:
  artifact_kind: nvfp4
  publication:
    github: HangGlidersRule/model-forge
"""
    )
    with pytest.raises(RecipeError, match="calibration"):
        load_recipe(path)
