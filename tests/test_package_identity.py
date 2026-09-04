"""Package and CLI identity after model-forge migration."""
from __future__ import annotations

import importlib
from pathlib import Path

import model_forge
from model_forge.cli import parser


def test_package_import_name() -> None:
    assert model_forge.__name__ == "model_forge"
    assert hasattr(model_forge, "__version__")


def test_cli_prog_is_model_forge() -> None:
    assert parser().prog == "model-forge"


def test_pyproject_package_metadata() -> None:
    text = Path(__file__).resolve().parent.parent.joinpath("pyproject.toml").read_text()
    assert 'name = "model-forge"' in text
    assert "model-forge = " in text
    assert "qwen-bakeoff" not in text
    assert "qwen_bakeoff" not in text


def test_obsolete_package_path_gone() -> None:
    root = Path(__file__).resolve().parent.parent
    assert not (root / "src" / "qwen_bakeoff").exists()
    assert (root / "src" / "model_forge").is_dir()


def test_no_qwen_bakeoff_importable() -> None:
    try:
        importlib.import_module("qwen_bakeoff")
    except ModuleNotFoundError:
        return
    raise AssertionError("qwen_bakeoff should not remain importable")
