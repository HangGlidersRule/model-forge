"""Frozen GPQA Diamond harness with full-denominator scoring."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

DATASET_NAME = "GPQA Diamond"
DATASET_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
EXPECTED_QUESTIONS = 198
DEFAULT_TIMEOUT_S = 1800
DEFAULT_RETRIES = 2
HARNESS_VERSION = "1.0.0"

TerminalStatus = Literal["correct", "incorrect", "parse_error", "timeout", "error"]


@dataclass(frozen=True)
class GPQASampling:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    thinking: bool = False
    max_tokens: int | None = None
    workers: int = 4


@dataclass
class GPQAItemResult:
    index: int
    status: TerminalStatus
    correct: bool | None
    parsed_answer: str | None
    gold_letter: str
    latency_s: float
    attempts: int
    error: str | None = None
    response_hash: str | None = None


@dataclass
class GPQARunSummary:
    dataset_name: str
    dataset_sha256: str
    harness_version: str
    expected: int
    terminal: int
    correct: int
    parse_errors: int
    timeouts: int
    errors: int
    accuracy_full_denominator: float
    completed_parseable: int
    accuracy_completed_only: float | None
    complete: bool
    sampling: dict[str, Any] = field(default_factory=dict)
    artifact_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPQAError(RuntimeError):
    pass


_BOXED_RE = re.compile(r"\\boxed\{\s*([ABCD])\s*\}", re.IGNORECASE)
_ANSWER_RE = re.compile(r"Answer\s*:\s*([ABCD])\b", re.IGNORECASE)


def harness_sha256(source: str | bytes) -> str:
    data = source.encode("utf-8") if isinstance(source, str) else source
    return hashlib.sha256(data).hexdigest()


def verify_dataset_sha256(path: Path, expected: str = DATASET_SHA256) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise GPQAError(
            f"GPQA dataset sha256 mismatch: got {digest}, expected {expected}"
        )
    return digest


def permute_choices(index: int, correct: str, incorrect: list[str]) -> tuple[list[str], str]:
    options = [correct, *incorrect]
    if len(options) != 4:
        raise GPQAError(f"Expected 4 choices, got {len(options)}")
    order = random.Random(index).sample(range(4), 4)
    permuted = [options[i] for i in order]
    gold_letter = "ABCD"[order.index(0)]
    return permuted, gold_letter


def build_prompt(question: str, choices: list[str]) -> str:
    labeled = "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", choices, strict=True))
    return (
        f"What is the correct answer to this question: {question}\n"
        f"Choices:\n{labeled}\n\n"
        "Analyze the options. Place the final answer letter inside \\boxed{}."
    )


def parse_answer(text: str) -> str | None:
    matches = _BOXED_RE.findall(text or "")
    if matches:
        return str(matches[-1]).upper()
    matches = _ANSWER_RE.findall(text or "")
    if matches:
        return str(matches[-1]).upper()
    return None


def score_response(text: str, gold_letter: str) -> tuple[TerminalStatus, bool | None, str | None]:
    parsed = parse_answer(text)
    if parsed is None:
        return "parse_error", None, None
    correct = parsed == gold_letter.upper()
    return ("correct" if correct else "incorrect"), correct, parsed


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def load_jsonl_index(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["index"])] = row
    return rows


def summarize_results(
    results: list[GPQAItemResult],
    *,
    expected: int = EXPECTED_QUESTIONS,
    sampling: GPQASampling | None = None,
    artifact_hashes: dict[str, str] | None = None,
) -> GPQARunSummary:
    if len(results) > expected:
        raise GPQAError(f"Too many results: {len(results)} > {expected}")
    by_status = {
        status: 0 for status in ("correct", "incorrect", "parse_error", "timeout", "error")
    }
    for item in results:
        by_status[item.status] += 1
    correct = by_status["correct"]
    terminal = len(results)
    parseable = by_status["correct"] + by_status["incorrect"]
    publication_complete = (
        terminal == expected
        and parseable == expected
        and by_status["parse_error"] == 0
        and by_status["timeout"] == 0
        and by_status["error"] == 0
    )
    completed_only = (correct / parseable) if parseable else None
    return GPQARunSummary(
        dataset_name=DATASET_NAME,
        dataset_sha256=DATASET_SHA256,
        harness_version=HARNESS_VERSION,
        expected=expected,
        terminal=terminal,
        correct=correct,
        parse_errors=by_status["parse_error"],
        timeouts=by_status["timeout"],
        errors=by_status["error"],
        accuracy_full_denominator=correct / expected,
        completed_parseable=parseable,
        accuracy_completed_only=completed_only,
        complete=publication_complete,
        sampling=asdict(sampling) if sampling else {},
        artifact_hashes=artifact_hashes or {},
    )


GenerateFn = Callable[[int, str], str]


def run_gpqa(
    questions: list[dict[str, Any]],
    generate: GenerateFn,
    *,
    journal_path: Path,
    sampling: GPQASampling | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    expected: int = EXPECTED_QUESTIONS,
    resume: bool = True,
) -> GPQARunSummary:
    if len(questions) != expected:
        raise GPQAError(f"Expected {expected} questions, got {len(questions)}")
    sampling = sampling or GPQASampling()
    existing = load_jsonl_index(journal_path) if resume else {}
    results: list[GPQAItemResult] = []

    for index, row in enumerate(questions):
        if index in existing and existing[index].get("status") in {
            "correct",
            "incorrect",
            "parse_error",
        }:
            prev = existing[index]
            results.append(
                GPQAItemResult(
                    index=index,
                    status=prev["status"],
                    correct=prev.get("correct"),
                    parsed_answer=prev.get("parsed_answer"),
                    gold_letter=prev["gold_letter"],
                    latency_s=float(prev.get("latency_s", 0.0)),
                    attempts=int(prev.get("attempts", 1)),
                    error=prev.get("error"),
                    response_hash=prev.get("response_hash"),
                )
            )
            continue

        choices, gold = permute_choices(index, row["correct"], list(row["incorrect"]))
        prompt = build_prompt(row["question"], choices)
        status: TerminalStatus = "error"
        parsed: str | None = None
        correct: bool | None = None
        error: str | None = None
        response_hash: str | None = None
        latency = 0.0
        attempts = 0
        for attempt in range(1, retries + 2):
            attempts = attempt
            started = time.monotonic()
            try:
                text = generate(index, prompt)
                latency = time.monotonic() - started
                if latency > timeout_s:
                    status = "timeout"
                    error = f"client-observed latency {latency:.1f}s > {timeout_s}"
                    continue
                status, correct, parsed = score_response(text, gold)
                response_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                error = None
                if status in {"correct", "incorrect"}:
                    break
            except TimeoutError as exc:
                latency = time.monotonic() - started
                status = "timeout"
                error = str(exc)
            except Exception as exc:  # noqa: BLE001
                latency = time.monotonic() - started
                status = "error"
                error = str(exc)

        item = GPQAItemResult(
            index=index,
            status=status,
            correct=correct,
            parsed_answer=parsed,
            gold_letter=gold,
            latency_s=latency,
            attempts=attempts,
            error=error,
            response_hash=response_hash,
        )
        results.append(item)
        append_jsonl(journal_path, asdict(item))

    return summarize_results(results, expected=expected, sampling=sampling)


def cheap_screen_indices(n: int = 8, *, total: int = EXPECTED_QUESTIONS) -> list[int]:
    rng = random.Random(1234)
    return sorted(rng.sample(range(total), min(n, total)))
