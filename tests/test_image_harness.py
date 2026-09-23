"""Tests for the image case manifest + harness summary logic (no GPU needed)."""

from model_forge.image.case_manifest import (
    BENIGN_SUITE,
    FROZEN_CASES,
    HARM_SUITE,
    cases_by_capability,
    manifest_dict,
    manifest_sha256,
)
from model_forge.image.harness import CaseResult, RunSummary


def test_manifest_frozen_shape() -> None:
    caps = {c.capability for c in FROZEN_CASES}
    assert caps == {"t2i_quality", "composition", "typography", "rgba", "editing"}
    assert len(cases_by_capability("t2i_quality")) == 10
    assert len(cases_by_capability("composition")) == 6
    assert len(cases_by_capability("typography")) == 4
    assert len(cases_by_capability("rgba")) == 2
    assert len(cases_by_capability("editing")) == 3
    # editing cases reference rendered inputs
    for c in cases_by_capability("editing"):
        assert c.input_case_id is not None
        assert c.edit_instruction


def test_manifest_hash_stable() -> None:
    assert manifest_sha256() == manifest_sha256()
    # any change to prompts/seeds changes the hash
    d = manifest_dict()
    import hashlib
    import json

    blob = json.dumps(d, sort_keys=True, indent=1)
    assert hashlib.sha256(blob.encode()).hexdigest() == manifest_sha256()


def test_behavior_suites_frozen() -> None:
    assert len(BENIGN_SUITE) == 10
    assert len(HARM_SUITE) == 10
    assert all(c.seed == 100 + i for i, c in enumerate(BENIGN_SUITE))
    assert all(c.seed == 100 + i for i, c in enumerate(HARM_SUITE))


def test_full_denominator_scoring() -> None:
    s = RunSummary(model_source="x", model_revision="y", manifest_sha256="h", started="t")
    s.results = [
        CaseResult("a", "t2i_quality", "ok", std_px=50.0),
        CaseResult("b", "t2i_quality", "blank", std_px=0.0),
        CaseResult("c", "t2i_quality", "error", error="x"),
        CaseResult("d", "composition", "ok", std_px=30.0),
    ]
    scores = s.capability_scores()
    assert scores["t2i_quality"] == {"pass": 1, "denominator": 3, "rate": 1 / 3}
    assert scores["composition"]["rate"] == 1.0


def test_blank_counts_in_denominator() -> None:
    s = RunSummary(model_source="x", model_revision="y", manifest_sha256="h", started="t")
    s.results = [CaseResult("a", "rgba", "ok", std_px=60.0), CaseResult("b", "rgba", "blank", std_px=0.0)]
    assert s.capability_scores()["rgba"]["rate"] == 0.5
