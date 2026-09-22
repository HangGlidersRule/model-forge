"""Tests for campaign-canonical abliteration extensions.

Covers biproject_direction, norm_preserving_project, the lora-unlearning-sft
recipe transform schema, and smart-quote refusal-classification.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model_forge.abliteration import (  # noqa: E402
    biproject_direction,
    compute_leakage,
    norm_preserving_project,
)


def test_biproject_direction_orthogonalizes_against_harmless_mean() -> None:
    torch.manual_seed(0)
    harmless_mean = torch.nn.functional.normalize(torch.randn(64), dim=0)
    raw_dir = harmless_mean * 0.9 + torch.randn(64) * 0.1
    d = biproject_direction(raw_dir, harmless_mean)
    assert d.shape == raw_dir.shape
    # orthogonal to harmless_mean
    assert abs(float(d @ (harmless_mean / harmless_mean.norm()))) < 1e-5
    # unit norm
    assert abs(d.norm().item() - 1.0) < 1e-5


def test_biproject_direction_degenerate_harmless_returns_direction() -> None:
    d = torch.nn.functional.normalize(torch.randn(32), dim=0)
    out = biproject_direction(d, torch.zeros(32))
    assert out.shape == d.shape


def test_norm_preserving_project_restores_row_norms() -> None:
    torch.manual_seed(1)
    w = torch.randn(48, 64)
    r = torch.nn.functional.normalize(torch.randn(48), dim=0)
    out = norm_preserving_project(w, r)
    assert out.shape == w.shape
    orig = w.float().norm(dim=1)
    new = out.float().norm(dim=1)
    assert torch.allclose(orig, new, atol=1e-3)
    # and the direction is (near-)removed from the rowspace projection
    leak = compute_leakage(w, r)
    leak_after = compute_leakage(out, r)
    # norm restoration redistributes energy; leakage must not grow
    assert leak_after <= leak + 5e-3


def test_norm_preserving_project_1d_passthrough() -> None:
    w = torch.randn(32)
    r = torch.nn.functional.normalize(torch.randn(32), dim=0)
    out = norm_preserving_project(w, r)
    assert out.shape == w.shape


# ---------------- lora-unlearning-sft recipe schema ----------------

import pathlib  # noqa: E402

from model_forge.recipe import RecipeError, load_recipe  # noqa: E402

RECIPE = pathlib.Path(__file__).resolve().parents[1] / "recipes"

OMNI_RECIPE = RECIPE / "nemotron-3-nano-omni" / (
    "darkstar-nemotron-3-nano-omni-30b-a3b-reasoning-abliterated-bf16.yaml"
)


def test_omni_recipe_validates_with_lora_transform() -> None:
    recipe = load_recipe(OMNI_RECIPE)
    tr = recipe.transforms[0]
    assert tr.type == "lora-unlearning-sft"
    assert tr.teacher_dataset == "treadon/abliteration-eval"
    assert tr.harmful_compliant_rows == 25
    assert tr.safe_rows == 77
    assert tr.lora_r == 32
    assert tr.lora_alpha == 64
    assert tr.epochs == 3
    assert float(tr.learning_rate) == pytest.approx(1e-4)
    assert tr.seed == 42


def _write(tmp: pathlib.Path, body: str) -> pathlib.Path:
    p = tmp / "r.yaml"
    p.write_text(body)
    return p


MINIMAL = """
schema_version: "2.0"
name: test-family-ablit
family: test-family
source:
  model_id: org/model
  revision: "%s"
transforms:
  - type: lora-unlearning-sft
    seed: 42
    harmful_prompts: 200
    harmless_prompts: 83
    teacher_dataset: treadon/abliteration-eval
    lora_r: 32
    lora_alpha: 64
    epochs: 3
    learning_rate: 1.0e-4
outputs:
  artifact_kind: bf16
  publication:
    github: HangGlidersRule/model-forge
    huggingface_bf16: HangGlidersRule/test
"""


def test_lora_transform_missing_teacher_rejected(tmp_path: pathlib.Path) -> None:
    body = MINIMAL.replace("    teacher_dataset: treadon/abliteration-eval\n", "")
    p = _write(tmp_path, body % ("a" * 40))
    with pytest.raises(RecipeError):
        load_recipe(p)


def test_lora_transform_missing_lora_shape_rejected(tmp_path: pathlib.Path) -> None:
    body = MINIMAL.replace("    lora_r: 32\n    lora_alpha: 64\n", "")
    p = _write(tmp_path, body % ("b" * 40))
    with pytest.raises(RecipeError):
        load_recipe(p)


def test_lora_transform_missing_training_rejected(tmp_path: pathlib.Path) -> None:
    body = MINIMAL.replace("    epochs: 3\n    learning_rate: 1.0e-4\n", "")
    p = _write(tmp_path, body % ("c" * 40))
    with pytest.raises(RecipeError):
        load_recipe(p)


def test_lora_transform_happy_path_loads(tmp_path: pathlib.Path) -> None:
    p = _write(tmp_path, MINIMAL % ("d" * 40))
    recipe = load_recipe(p)
    assert recipe.transforms[0].type == "lora-unlearning-sft"


# ---------------- smart-quote refusal classification ----------------


def test_is_refusal_normalizes_smart_quotes() -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "eval_lightning_abliteration",
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts/nemotron-3.5_lightning/eval_lightning_abliteration.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # the module executes argparse only in main(); import is safe
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod.is_refusal("I\u2019m sorry, but I cannot help with that.") is True
    assert mod.is_refusal("I\u2018m sorry but no") is True
    assert mod.is_refusal("The capital of France is Paris.") is False
    # a previously-missed curly-quote refusal is now caught
    assert mod.is_refusal("I can\u2019t comply with that request.") is True
