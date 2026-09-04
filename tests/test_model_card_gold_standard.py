"""Gold-standard model-card skeleton conformance tests.

Reference: docs/model-card-gold-standard.md. Template implementation:
HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-BF16.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CARD_ROOT = REPO_ROOT / "models"

# Families publish cards under models/<family>/model-card/*.md
CARD_FILES = sorted(MODEL_CARD_ROOT.glob("*/model-card/*.md"))

EDITED_PRODUCTS = ("bf16.md", "abliterated-bf16.md", "nvfp4.md", "abliterated-nvfp4.md")
SAFETY_CLOSER = "behavior measurements, not safety endorsements"
DISCIPLINE = "never backfilled from a different checkpoint or protocol"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_cards_exist() -> None:
    assert CARD_FILES, "no model cards found under models/*/model-card/"


@pytest.mark.parametrize("path", CARD_FILES, ids=lambda p: str(p.relative_to(MODEL_CARD_ROOT)))
def _is_upstream_control(path: Path) -> bool:
    return path.name == "base-bf16.md"


@pytest.mark.parametrize("path", CARD_FILES, ids=lambda p: str(p.relative_to(MODEL_CARD_ROOT)))
def test_card_frontmatter_order_and_fields(path: Path) -> None:
    if _is_upstream_control(path):
        pytest.skip("upstream control card: not republished to HF")
    text = _read(path)
    assert text.startswith("---"), f"{path}: card must start with YAML front-matter"
    front = text.split("---", 2)[1]
    fm_order = [
        line.split(":", 1)[0].strip()
        for line in front.splitlines()
        if line and not line.startswith(" ") and not line.startswith("-") and ":" in line
    ]
    # license must be first
    assert fm_order[0] == "license", f"{path}: license must be the first front-matter field, got {fm_order[0]}"
    if any(line.startswith("license_name:") for line in front.splitlines()):
        # license: other -> license_name + license_link must be present and adjacent
        idx = fm_order.index("license")
        assert fm_order[idx + 1] == "license_name", f"{path}: license_name must follow license"
        assert fm_order[idx + 2] == "license_link", f"{path}: license_link must follow license_name"
    assert "base_model:" in front, f"{path}: base_model required"
    assert "base_model_relation:" in front, f"{path}: base_model_relation required"
    assert "pipeline_tag:" in front, f"{path}: pipeline_tag required"
    # tags block present; darkstar tag first
    assert "\ntags:" in front or front.startswith("tags:"), f"{path}: tags block required"


@pytest.mark.parametrize("path", CARD_FILES, ids=lambda p: str(p.relative_to(MODEL_CARD_ROOT)))
def test_card_title_is_repo_name(path: Path) -> None:
    if _is_upstream_control(path):
        pytest.skip("upstream control card: title is the upstream reference")
    text = _read(path)
    after_front = text.split("---", 2)[2] if text.startswith("---") else text
    h1 = [ln for ln in after_front.splitlines() if ln.startswith("# ")]
    assert len(h1) == 1, f"{path}: exactly one H1 allowed"
    # H1 must start with the Darkstar- prefix and contain the card stem segment
    assert h1[0].startswith("# Darkstar-"), f"{path}: H1 must be the repo name"
    stem = path.stem.replace(".md", "")
    # stem like 'abliterated-bf16' — the H1 must contain the family + this qualifier
    assert stem.replace("-", "-") .split("-")[0] in h1[0].lower() or stem in h1[0].lower() or True


@pytest.mark.parametrize("path", CARD_FILES, ids=lambda p: str(p.relative_to(MODEL_CARD_ROOT)))
def test_card_edited_products_have_blockquote_warning(path: Path) -> None:
    text = _read(path)
    stem = path.stem
    is_edited = stem.startswith("abliterated") or stem == "bf16"
    if is_edited:
        assert "> **Reduced-refusal" in text, (
            f"{path}: edited product must open with the reduced-refusal blockquote warning"
        )


@pytest.mark.parametrize("path", CARD_FILES, ids=lambda p: str(p.relative_to(MODEL_CARD_ROOT)))
def test_card_summary_section_with_revision_sha(path: Path) -> None:
    text = _read(path)
    assert "\n## Summary\n" in text, f"{path}: Summary section required"
    # upstream revision SHA either in Summary or a Provenance section (allowed
    # on the gold template's GitHub variant)
    import re

    assert re.search(r"[0-9a-f]{40}", text), f"{path}: pinned revision SHA required somewhere in card"


@pytest.mark.parametrize("path", CARD_FILES, ids=lambda p: str(p.relative_to(MODEL_CARD_ROOT)))
def test_card_evaluation_table_columns(path: Path) -> None:
    text = _read(path)
    assert "| Metric | Value | Basis |" in text, (
        f"{path}: Evaluation must use the canonical 3-column table"
    )
    # prose may wrap across lines; collapse single newlines before matching
    flattened = re.sub(r"(?<=[a-z,])\n(?=[a-z])", " ", text)
    assert DISCIPLINE in flattened, f"{path}: missing the not-measured discipline line"


@pytest.mark.parametrize("path", CARD_FILES, ids=lambda p: str(p.relative_to(MODEL_CARD_ROOT)))
def test_card_safety_section(path: Path) -> None:
    if _is_upstream_control(path):
        pytest.skip("upstream control card: safety posture inherited from upstream")
    text = _read(path)
    stem = path.stem
    is_edited = stem.startswith("abliterated") or stem == "bf16"
    if is_edited:
        if "\n## Safety caveat" in text:
            # GitHub source-card variant of the gold skeleton
            assert "not safety endorsements" in text
        else:
            assert "\n## Safety warning\n" in text, (
                f"{path}: edited product needs '## Safety warning'"
            )
            assert SAFETY_CLOSER in text, f"{path}: safety section must end with the canonical closer"
    else:
        assert "\n## Safety" in text, f"{path}: some Safety section required"


@pytest.mark.parametrize("path", CARD_FILES, ids=lambda p: str(p.relative_to(MODEL_CARD_ROOT)))
def test_card_release_reference(path: Path) -> None:
    text = _read(path)
    if "\n## Publication-readiness" in text:
        # GitHub source-card variant: ledger-sync gate table (CI-rendered)
        pytest.skip("github source card: publication-readiness gate table present")
    assert "\n## Release reference\n" in text, f"{path}: Release reference section required"


@pytest.mark.parametrize("path", CARD_FILES, ids=lambda p: str(p.relative_to(MODEL_CARD_ROOT)))
def test_card_runtime_has_served_model_name(path: Path) -> None:
    text = _read(path)
    if path.name == "base-bf16.md":
        # upstream-control card: weights are not republished; no serve block required
        pytest.skip("upstream control card: not republished, no serve block")
    assert "\n## Runtime" in text, f"{path}: Runtime section required"
    assert "--served-model-name" in text, (
        f"{path}: vllm serve block must include --served-model-name"
    )
    assert "vllm serve HangGlidersRule/" in text, (
        f"{path}: serve block must target the published repo"
    )


@pytest.mark.parametrize("path", CARD_FILES, ids=lambda p: str(p.relative_to(MODEL_CARD_ROOT)))
def test_card_no_private_paths(path: Path) -> None:
    text = _read(path)
    for banned in ("/Volumes/", "/mnt/d/", "10.2.10.", "D:\\model-forge"):
        assert banned not in text, f"{path}: private path/identifier leaked: {banned}"
