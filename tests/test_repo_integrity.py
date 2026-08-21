"""Repository integrity: schema-2 recipes load and internal Markdown links resolve."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from model_forge.recipe import SCHEMA_VERSION, load_recipe

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = REPO_ROOT / "recipes"

_IGNORED_DIRS = {".git", ".venv", "node_modules", "__pycache__", "dist", "build"}
_MD_LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)]+)\)")


def _entry_recipes() -> list[Path]:
    return sorted(
        path
        for path in RECIPES_DIR.rglob("*.yaml")
        if "legacy" not in path.relative_to(RECIPES_DIR).parts
    )


def _markdown_files() -> list[Path]:
    result: list[Path] = []
    for path in REPO_ROOT.rglob("*.md"):
        if any(part in _IGNORED_DIRS for part in path.relative_to(REPO_ROOT).parts):
            continue
        result.append(path)
    return sorted(result)


def test_entry_recipes_discovered() -> None:
    names = {path.name for path in _entry_recipes()}
    assert {
        "base-nvfp4.yaml",
        "darkstar-qwen3.8-27b-base-modelopt-nvfp4.yaml",
        "darkstar-qwen3.8-27b-abliterated-bf16.yaml",
        "darkstar-qwen3.8-27b-abliterated-modelopt-nvfp4.yaml",
        "r3-nvfp4.yaml",
    } <= names


@pytest.mark.parametrize("recipe_path", _entry_recipes(), ids=lambda p: str(p.name))
def test_every_non_legacy_recipe_loads(recipe_path: Path) -> None:
    recipe = load_recipe(recipe_path)
    assert recipe.schema_version == SCHEMA_VERSION
    assert recipe.name
    assert recipe.family
    assert len(recipe.config_sha()) == 64


def _iter_internal_link_targets(md_path: Path) -> list[str]:
    text = md_path.read_text(encoding="utf-8")
    targets: list[str] = []
    for match in _MD_LINK.finditer(text):
        target = match.group(1).strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        # Strip an optional link title: [x](path "title")
        target = target.split(" ", 1)[0]
        if not target or target.startswith("#"):
            continue
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):  # scheme:// or mailto:
            continue
        targets.append(target)
    return targets


def test_internal_markdown_links_resolve() -> None:
    broken: list[str] = []
    for md_path in _markdown_files():
        for target in _iter_internal_link_targets(md_path):
            path_part = target.split("#", 1)[0]
            if not path_part:
                continue
            resolved = (md_path.parent / path_part).resolve()
            if not resolved.exists():
                rel = md_path.relative_to(REPO_ROOT)
                broken.append(f"{rel} -> {target}")
    assert not broken, "Broken internal Markdown links:\n" + "\n".join(broken)
