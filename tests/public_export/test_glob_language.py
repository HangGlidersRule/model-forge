from __future__ import annotations

import pytest

from model_forge.public_export.glob_language import (
    GlobLanguageError,
    WorkBudget,
    glob_languages_overlap,
    parse_glob,
)


@pytest.mark.parametrize(
    ("left", "right", "overlap"),
    [
        ("a/?/x", "a/[bc]/x", True),
        ("a/[a-c]/x", "a/[d-f]/x", False),
        ("a/**/c", "a/b/*", True),
        ("a/*/x", "a/b/c/x", False),
        ("a/[!b]/x", "a/b/x", False),
    ],
)
def test_supported_glob_languages_intersect_exactly(
    left: str, right: str, overlap: bool
) -> None:
    assert glob_languages_overlap(left, right) is overlap


def test_repeated_stars_normalize_and_stay_bounded() -> None:
    assert parse_glob("bounded/" + ("*" * 128) + "/x") == parse_glob(
        "bounded/**/x"
    )
    budget = WorkBudget(limit=2_000)
    assert not glob_languages_overlap(
        "bounded/" + ("*" * 128) + "/x",
        "bounded/" + ("*" * 128) + "/y",
        budget,
    )
    assert budget.used <= 2_000


def test_product_work_limit_fails_closed() -> None:
    with pytest.raises(GlobLanguageError, match="work limit exceeded"):
        glob_languages_overlap(
            "bounded/**/x",
            "bounded/**/x",
            WorkBudget(limit=1),
        )
