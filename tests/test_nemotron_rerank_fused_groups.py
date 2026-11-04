from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "models/nemotron-3.5-rerank-8b/eval/fused_groups.py"
LAYERS = 34
ATTN, MLP = ("q_proj", "k_proj", "v_proj", "o_proj"), ("gate_proj", "up_proj", "down_proj")
PLANS = {
    "A": {**{p: "nvfp4" for p in ATTN}, **{p: "nvfp4" for p in MLP}},
    "B": {**{p: "fp8" for p in ATTN}, **{p: "nvfp4" for p in MLP}},
    "C": {**{p: "bf16" for p in ATTN}, **{p: "nvfp4" for p in MLP}},
}


def _module():
    spec = importlib.util.spec_from_file_location("nr8_fused_groups", MODULE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _uniform(plan: dict[str, str]) -> dict[str, float]:
    """Every scale of every quantized linear set to 1.0: a correctly fused export."""
    values: dict[str, float] = {}
    for layer in range(LAYERS):
        for proj, fmt in plan.items():
            sub = "self_attn" if proj in ATTN else "mlp"
            base = f"model.language_model.layers.{layer}.{sub}.{proj}"
            if fmt == "nvfp4":
                values[f"{base}.input_scale"] = 1.0
                values[f"{base}.weight_scale_2"] = 1.0
            elif fmt == "fp8":
                values[f"{base}.input_scale"] = 1.0
                values[f"{base}.weight_scale"] = 1.0
    return values


@pytest.mark.parametrize(("candidate", "groups"), [("A", 136), ("B", 102), ("C", 68)])
def test_uniform_export_passes_and_counts_every_fused_scale(candidate: str, groups: int) -> None:
    """A: q/k/v and gate/up NVFP4, two shared scales each (4 x 34). B: q/k/v FP8 share only input_scale (34) plus
    gate/up NVFP4 (68). C: attention BF16 has no scales, gate/up NVFP4 (68)."""
    result = _module().fused_group_mismatches(_uniform(PLANS[candidate]).__getitem__, PLANS[candidate], LAYERS)
    assert result["groups_checked"] == groups
    assert result["mismatched"] == 0


@pytest.mark.parametrize("scale", ["input_scale", "weight_scale_2"])
def test_nvfp4_shard_with_a_different_shared_scale_is_flagged(scale: str) -> None:
    values = _uniform(PLANS["A"])
    values[f"model.language_model.layers.5.mlp.up_proj.{scale}"] = 2.0
    result = _module().fused_group_mismatches(values.__getitem__, PLANS["A"], LAYERS)
    assert result["mismatched"] == 1
    assert result["mismatches"][0]["layer"] == 5
    assert result["mismatches"][0]["group"] == "mlp"
    assert result["mismatches"][0]["scale"] == scale


def test_fp8_weight_scales_may_differ_but_input_scales_may_not() -> None:
    """vLLM requantizes FP8 q/k/v weights to the group maximum at load, so per-shard FP8 weight scales may differ; the
    activation scale is applied once to the fused GEMM and must be shared."""
    mod = _module()
    values = _uniform(PLANS["B"])
    values["model.language_model.layers.7.self_attn.k_proj.weight_scale"] = 3.0
    assert mod.fused_group_mismatches(values.__getitem__, PLANS["B"], LAYERS)["mismatched"] == 0
    values["model.language_model.layers.7.self_attn.k_proj.input_scale"] = 3.0
    result = mod.fused_group_mismatches(values.__getitem__, PLANS["B"], LAYERS)
    assert result["mismatched"] == 1
    assert result["mismatches"][0]["scale"] == "input_scale"


def test_a_group_with_mixed_formats_is_flagged() -> None:
    plan = dict(PLANS["A"], k_proj="bf16")
    values = _uniform(plan)
    result = _module().fused_group_mismatches(values.__getitem__, plan, LAYERS)
    assert result["mismatched"] == LAYERS
    assert "mixed formats" in result["mismatches"][0]["reason"]
