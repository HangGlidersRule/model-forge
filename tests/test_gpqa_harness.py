"""Tests for frozen GPQA harness full-denominator semantics."""

from __future__ import annotations

from pathlib import Path

from model_forge.gpqa import (
    EXPECTED_QUESTIONS,
    cheap_screen_indices,
    parse_answer,
    run_gpqa,
    score_response,
    summarize_results,
)
from model_forge.gpqa.harness import GPQAItemResult, GPQASampling


def test_parse_boxed_answer() -> None:
    assert parse_answer("reasoning\\boxed{B}") == "B"
    assert parse_answer("Answer: C") == "C"
    assert parse_answer("no answer") is None


def test_full_denominator_never_completed_only() -> None:
    results = [
        GPQAItemResult(0, "correct", True, "A", "A", 1.0, 1),
        GPQAItemResult(1, "timeout", None, None, "B", 1800.0, 3, error="timeout"),
    ]
    summary = summarize_results(results, expected=2)
    assert summary.correct == 1
    assert summary.accuracy_full_denominator == 0.5
    assert summary.accuracy_completed_only == 1.0
    assert summary.complete is False


def test_publication_requires_all_parseable(tmp_path: Path) -> None:
    questions = [
        {
            "question": f"Q{i}?",
            "correct": "yes",
            "incorrect": ["a", "b", "c"],
        }
        for i in range(4)
    ]

    def generate(index: int, _prompt: str) -> str:
        # Always emit a boxed letter so all are parseable.
        return "\\boxed{A}"

    summary = run_gpqa(
        questions,
        generate,
        journal_path=tmp_path / "journal.jsonl",
        expected=4,
        retries=0,
    )
    assert summary.terminal == 4
    assert summary.complete is True
    assert summary.accuracy_full_denominator == summary.correct / 4


def test_timeouts_prevent_completion(tmp_path: Path) -> None:
    questions = [
        {"question": "Q?", "correct": "yes", "incorrect": ["a", "b", "c"]}
        for _ in range(2)
    ]

    def generate(index: int, _prompt: str) -> str:
        if index == 1:
            raise TimeoutError("slow")
        return "\\boxed{A}"

    summary = run_gpqa(
        questions,
        generate,
        journal_path=tmp_path / "journal.jsonl",
        expected=2,
        retries=0,
    )
    assert summary.timeouts == 1
    assert summary.complete is False
    assert summary.accuracy_full_denominator == summary.correct / 2


def test_cheap_screen_deterministic() -> None:
    assert cheap_screen_indices(8) == cheap_screen_indices(8)
    assert len(cheap_screen_indices(8)) == 8
    assert max(cheap_screen_indices(8)) < EXPECTED_QUESTIONS


def test_score_response_roundtrip() -> None:
    status, correct, parsed = score_response("final \\boxed{D}", "D")
    assert status == "correct" and correct is True and parsed == "D"
    status, correct, parsed = score_response("final \\boxed{A}", "D")
    assert status == "incorrect" and correct is False
    _ = GPQASampling()
