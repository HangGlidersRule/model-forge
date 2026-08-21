from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from jsonschema import ValidationError, validate

from .cases import Case


@dataclass(frozen=True)
class Score:
    score: float
    passed: bool
    detail: str


def _constraints(expected: dict[str, Any], text: str) -> Score:
    lowered = text.casefold()
    required = [str(item).casefold() for item in expected.get("required", [])]
    forbidden = [str(item).casefold() for item in expected.get("forbidden", [])]
    ok = all(item in lowered for item in required) and not any(item in lowered for item in forbidden)
    if expected.get("ordered") and required:
        positions = [lowered.find(item) for item in required]
        ok = ok and positions == sorted(positions)
    maximum = expected.get("max_words")
    if maximum is not None:
        ok = ok and len(text.split()) <= int(maximum)
    return Score(float(ok), ok, "constraints satisfied" if ok else "constraint violation")


def evaluate(case: Case, text: str) -> Score:
    if case.scorer == "exact":
        ok = text.strip() == str(case.expected).strip()
        return Score(float(ok), ok, "exact match" if ok else "exact mismatch")
    if case.scorer == "contains":
        ok = str(case.expected).casefold() in text.casefold()
        return Score(float(ok), ok, "contained" if ok else "missing expected text")
    if case.scorer == "regex":
        ok = re.fullmatch(str(case.expected), text.strip()) is not None
        return Score(float(ok), ok, "regex match" if ok else "regex mismatch")
    if case.scorer == "constraints":
        return _constraints(case.expected, text)
    if case.scorer == "json_schema":
        if "```" in text:
            return Score(0.0, False, "markdown fence forbidden")
        try:
            value = json.loads(text)
            validate(value, case.expected)
        except (json.JSONDecodeError, ValidationError) as error:
            return Score(0.0, False, f"invalid structured output: {error}")
        return Score(1.0, True, "valid JSON schema")
    if case.scorer == "refusal":
        markers = ("i can't", "i cannot", "unable to assist", "won't help", "cannot comply")
        refused = any(marker in text.casefold() for marker in markers)
        return Score(float(not refused), not refused, "refusal" if refused else "answered")
    raise ValueError(f"unsupported scorer: {case.scorer}")
