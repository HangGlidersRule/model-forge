from pathlib import Path

import pytest

from model_forge.cli import main

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES = REPO_ROOT / "recipes" / "qwen3.8-27b"


def test_recipe_validate_prints_summary(capsys) -> None:
    assert main(["recipe", "validate", str(RECIPES / "r3-nvfp4.yaml")]) == 0
    out = capsys.readouterr().out
    assert "qwen3.8-27b-r3-nvfp4" in out
    assert "qwen3.8-27b" in out
    assert "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" in out
    assert "nvfp4" in out
    # config SHA is a 64-char hex digest
    assert any(len(token) == 64 and all(c in "0123456789abcdef" for c in token) for token in out.split())


def test_recipe_validate_reports_all_fields(capsys) -> None:
    recipe = RECIPES / "darkstar-qwen3.8-27b-abliterated-bf16.yaml"
    assert main(["recipe", "validate", str(recipe)]) == 0
    out = capsys.readouterr().out
    for label in ("name", "family", "source", "revision", "artifact", "config"):
        assert label in out.lower()


def test_recipe_validate_rejects_bad_recipe(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
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
    assert main(["recipe", "validate", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "40-char SHA" in err


def test_recipe_validate_rejects_uncovered_protection(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "uncovered.yaml"
    bad.write_text(
        """
schema_version: "2.0"
name: uncovered
family: demo
source:
  model_id: org/model
  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
transforms: []
quantization:
  scheme: NVFP4
  targets: Linear
  group_size: 16
  protected_tensors: [lm_head, conv1d]
  keep_bf16: [lm_head, conv1d]
  ignore: [lm_head]
  calibration:
    dataset: demo
    config: LLM
    samples: 32
    max_sequence_length: 8192
    pipeline: basic
    shard_size: 5GB
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
    assert main(["recipe", "validate", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "conv1d" in err
    assert err.startswith("Invalid recipe:")


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("schema_version: \"2.0\"\nname: [unterminated\n", "invalid YAML"),
        ("", "empty"),
        ("just-a-string\n", "must be a mapping"),
        (
            """
schema_version: "2.0"
name: incomplete
family: demo
source:
  model_id: org/model
  revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
transforms: []
validation:
  max_refusal_leakage: 0.01
outputs:
  artifact_kind: bf16
  publication:
    github: HangGlidersRule/model-forge
""",
            "runtime",
        ),
    ],
    ids=["malformed", "empty", "scalar", "missing-section"],
)
def test_recipe_validate_reports_structural_failures(
    tmp_path: Path, capsys, body: str, expected: str
) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text(body)
    assert main(["recipe", "validate", str(path)]) == 2
    captured = capsys.readouterr()
    assert expected in captured.err
    assert captured.err.startswith("Invalid recipe:")
    assert "Traceback" not in captured.err


def test_recipe_validate_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["recipe", "validate", str(tmp_path / "nope.yaml")])


@pytest.mark.private_source_only
def test_cli_dry_run_validates_without_network(tmp_path: Path, capsys) -> None:
    spec = tmp_path / "spec.yaml"
    spec.write_text("""
schema_version: '1.0'
name: dry
models: [{id: a, endpoint: 'http://localhost/v1', model: a, family: qwen3.6, precision: nvfp4}]
tracks: [{name: no-think, reasoning: false}]
suites: [smoke]
""")
    assert main(["run", "--spec", str(spec), "--output", str(tmp_path / "out"), "--dry-run"]) == 0
    assert "2 cases" in capsys.readouterr().out
