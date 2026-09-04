"""Tests for ModelOpt policy resolution and fail-closed validators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_forge.modelopt.calibration import default_calibration_contract
from model_forge.modelopt.pin import (
    EXPECTED_COMMIT,
    EXPECTED_VERSION,
    MIXED_W4A16_RECIPE,
    OMLP_RECIPE,
    PRIMARY_RECIPE,
    load_pin,
)
from model_forge.modelopt.policy import (
    PolicyError,
    assert_no_mixed_fused_groups,
    load_quant_cfg,
    resolve_module_policy,
    resolve_quantizer,
)
from model_forge.modelopt.validate import (
    ValidationError,
    validate_mtp_bf16,
    validate_no_fp8_kv_metadata,
    validate_no_quantized_vision,
    validate_recipe_file,
    validate_scales_finite,
)

REPO = Path(__file__).resolve().parent.parent
MODEL_OPT_RECIPES = tuple(sorted((REPO / "configs/modelopt/recipes").glob("*.yaml")))


def test_pin_matches_release() -> None:
    pin = load_pin()
    assert pin.version == EXPECTED_VERSION
    assert pin.git_commit == EXPECTED_COMMIT
    assert pin.wheel_sha256.startswith("d6f6964b")
    prov = pin.provenance()
    assert len(prov["primary_recipe_sha256"]) == 64


def test_calibration_contract() -> None:
    cal = default_calibration_contract()
    assert cal.datasets == ("cnn_dailymail", "nemotron-post-training-dataset-v2")
    assert cal.sizes == (512, 512)
    assert cal.total_samples == 1024
    assert cal.batch_size == 1
    assert cal.sequence_length == 2048
    assert cal.seed == 1234
    assert cal.layerwise is False
    assert cal.image_text_enabled is False


def test_primary_recipe_validates() -> None:
    report = validate_recipe_file(PRIMARY_RECIPE)
    assert report.ok, report.errors
    coverage = report.details["coverage"]
    assert coverage["nvfp4_count"] > 0
    assert coverage["vision_quantized"] == []
    assert coverage["mtp_quantized"] == []
    # Language MLP should be NVFP4; attention QKV BF16.
    rules = load_quant_cfg(PRIMARY_RECIPE)
    q = resolve_quantizer(
        "model.language_model.layers.0.mlp.gate_proj", "weight_quantizer", rules
    )
    assert q.enabled and q.precision == "nvfp4"
    attn = resolve_quantizer(
        "model.language_model.layers.1.self_attn.q_proj", "weight_quantizer", rules
    )
    assert not attn.enabled
    vis = resolve_quantizer("model.visual.blocks.0.mlp.fc1", "weight_quantizer", rules)
    assert not vis.enabled
    mtp = resolve_quantizer("mtp.layers.0.mlp.down_proj", "weight_quantizer", rules)
    assert not mtp.enabled
    gdn_a = resolve_quantizer(
        "model.language_model.layers.2.linear_attn.in_proj_a", "weight_quantizer", rules
    )
    assert not gdn_a.enabled


@pytest.mark.parametrize("recipe_path", MODEL_OPT_RECIPES, ids=lambda path: path.name)
def test_all_modelopt_recipes_validate(recipe_path: Path) -> None:
    report = validate_recipe_file(recipe_path)
    assert report.ok, report.errors


def test_omlp_quantizes_o_proj_keeps_qkv_bf16() -> None:
    rules = load_quant_cfg(OMLP_RECIPE)
    o = resolve_quantizer(
        "model.language_model.layers.1.self_attn.o_proj", "weight_quantizer", rules
    )
    assert o.enabled and o.precision == "nvfp4"
    for name in ("q_proj", "k_proj", "v_proj"):
        item = resolve_quantizer(
            f"model.language_model.layers.1.self_attn.{name}", "weight_quantizer", rules
        )
        assert not item.enabled
    # GDN out_proj must stay BF16 even if bare *o_proj* matched.
    gdn_out = resolve_quantizer(
        "model.language_model.layers.2.linear_attn.out_proj", "weight_quantizer", rules
    )
    assert not gdn_out.enabled
    report = validate_recipe_file(OMLP_RECIPE)
    assert report.ok, report.errors


def test_no_fp8_kv_in_primary_recipe() -> None:
    rules = load_quant_cfg(PRIMARY_RECIPE)
    from model_forge.modelopt.policy import detect_fp8_kv_rules

    assert detect_fp8_kv_rules(rules) == []
    text = PRIMARY_RECIPE.read_text(encoding="utf-8")
    assert "use_constant_amax" not in text
    assert "kv_bmm_quantizer" not in text
    assert "kv_dtype: bf16" in text


def test_selected_mixed_recipe_is_exact_and_quantizes_lm_head() -> None:
    import hashlib

    assert hashlib.sha256(MIXED_W4A16_RECIPE.read_bytes()).hexdigest() == (
        "90fc6b37c00334debd49f1975ab406b5e20667f07e4be0be3e463a648abac642"
    )
    report = validate_recipe_file(MIXED_W4A16_RECIPE)
    assert report.ok, report.errors
    rules = load_quant_cfg(MIXED_W4A16_RECIPE)
    lm_head = resolve_quantizer("lm_head", "weight_quantizer", rules)
    assert lm_head.enabled and lm_head.precision == "nvfp4"


def test_fused_group_rejection() -> None:
    rules = load_quant_cfg(PRIMARY_RECIPE)
    resolved = resolve_module_policy(
        [
            "model.language_model.layers.1.self_attn.q_proj",
            "model.language_model.layers.1.self_attn.k_proj",
            "model.language_model.layers.1.self_attn.v_proj",
            "model.language_model.layers.0.mlp.gate_proj",
            "model.language_model.layers.0.mlp.up_proj",
        ],
        rules,
    )
    # Force a mixed Q/K/V decision.
    from model_forge.modelopt.policy import ResolvedQuantizer

    resolved["model.language_model.layers.1.self_attn.q_proj.weight_quantizer"] = (
        ResolvedQuantizer(
            "model.language_model.layers.1.self_attn.q_proj",
            "weight_quantizer",
            True,
            "nvfp4",
            "forced",
        )
    )
    with pytest.raises(PolicyError, match="Mixed precision"):
        assert_no_mixed_fused_groups(resolved)


def test_nan_scale_rejection() -> None:
    with pytest.raises(ValidationError, match="Invalid scale"):
        validate_scales_finite({"layer.weight_scale": [1.0, float("nan")]})
    with pytest.raises(ValidationError, match="Invalid scale"):
        validate_scales_finite({"layer.weight_scale": [0.0]})
    with pytest.raises(ValidationError, match="Empty scale"):
        validate_scales_finite({"layer.weight_scale": []})
    validate_scales_finite({"layer.weight_scale": [0.5, 1.0]})


def test_mtp_and_vision_protection(tmp_path: Path) -> None:
    weight_map = {f"mtp.layers.0.fake_{i}.weight": "model.safetensors" for i in range(15)}
    weight_map["model.visual.blocks.0.mlp.fc1.weight"] = "model.safetensors"
    weight_map["model.language_model.layers.0.mlp.down_proj.weight_scale"] = "model.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({"quant_algo": "NVFP4"}))
    assert len(validate_mtp_bf16(tmp_path)) == 15
    validate_no_quantized_vision(tmp_path)
    validate_no_fp8_kv_metadata(tmp_path)

    # Quantized vision must fail.
    weight_map["model.visual.blocks.0.mlp.fc1.weight_scale"] = "model.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    with pytest.raises(ValidationError, match="Vision"):
        validate_no_quantized_vision(tmp_path)


def test_fp8_kv_metadata_rejected(tmp_path: Path) -> None:
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantizer": {"kv_bmm_quantizer": {"num_bits": "e4m3"}}})
    )
    with pytest.raises(ValidationError, match="FP8 KV|kv_bmm"):
        validate_no_fp8_kv_metadata(tmp_path)
