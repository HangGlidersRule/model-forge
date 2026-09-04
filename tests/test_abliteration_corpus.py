"""Tests for corpus materialization logic."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_script = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "qwen3_8"
    / "materialize_abliteration_corpus.py"
)
_spec = importlib.util.spec_from_file_location("materialize_abliteration_corpus", _script)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["materialize_abliteration_corpus"] = _mod
_spec.loader.exec_module(_mod)

import pytest
from materialize_abliteration_corpus import (  # type: ignore[import-not-found]
    MODEL_RECORD_DATA_DIR,
    REPO_ROOT,
    build_parser,
    normalize_prompts,
    write_jsonl,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_repo_root_resolves_to_repository_root() -> None:
    assert REPO_ROOT == _REPO_ROOT


@pytest.mark.private_source_only
def test_model_record_data_dir_lives_under_the_model_record() -> None:
    expected = _REPO_ROOT / "models" / "qwen3.8-27b-r3" / "data" / "abliteration"
    assert MODEL_RECORD_DATA_DIR == expected
    assert MODEL_RECORD_DATA_DIR.is_dir()


def test_no_repository_data_path_resolves_under_scripts() -> None:
    relative = MODEL_RECORD_DATA_DIR.relative_to(_REPO_ROOT)
    assert relative.parts[0] == "models"
    assert "scripts" not in relative.parts


def test_run_root_is_required_so_runs_never_write_into_the_repository() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--config", "recipe.yaml"])


def test_run_root_is_taken_from_the_command_line(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["--config", "recipe.yaml", "--run-root", str(tmp_path / "run")]
    )
    assert args.run_root == tmp_path / "run"


def test_normalize_deduplicates() -> None:
    prompts = ["hello", "hello", "world", "world"]
    records = normalize_prompts(prompts, "test", "source")
    assert len(records) == 2


def test_normalize_strips_whitespace() -> None:
    prompts = ["  spaced  ", "spaced"]
    records = normalize_prompts(prompts, "test", "src")
    assert len(records) == 1
    assert records[0]["text"] == "spaced"


def test_normalize_deterministic_sort() -> None:
    prompts = ["banana", "apple", "cherry"]
    r1 = normalize_prompts(prompts, "test", "src")
    r2 = normalize_prompts(list(reversed(prompts)), "test", "src")
    assert [x["id"] for x in r1] == [x["id"] for x in r2]


def test_normalize_schema() -> None:
    records = normalize_prompts(["test prompt"], "harmful", "advbench")
    assert set(records[0].keys()) == {"id", "text", "label", "source"}
    assert records[0]["label"] == "harmful"
    assert records[0]["source"] == "advbench"


def test_write_jsonl_produces_valid(tmp_path: Path) -> None:
    records = normalize_prompts(["a", "b", "c"], "test", "src")
    path = tmp_path / "out.jsonl"
    sha = write_jsonl(records, path)
    assert path.exists()
    assert len(sha) == 64
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        parsed = json.loads(line)
        assert "id" in parsed
        assert "text" in parsed


def test_write_jsonl_hash_stable(tmp_path: Path) -> None:
    records = normalize_prompts(["x", "y"], "test", "src")
    h1 = write_jsonl(records, tmp_path / "a.jsonl")
    h2 = write_jsonl(records, tmp_path / "b.jsonl")
    assert h1 == h2

