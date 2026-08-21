from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolScore:
    passed: bool
    detail: str


def _normalize(call: dict[str, Any]) -> tuple[str, Any]:
    function = call.get("function", call)
    args = function.get("arguments", {})
    if isinstance(args, str):
        args = json.loads(args)
    return str(function.get("name")), args


def score_tool_call(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> ToolScore:
    wanted = sorted((_normalize(item) for item in expected), key=lambda item: item[0])
    try:
        got = sorted((_normalize(item) for item in actual), key=lambda item: item[0])
    except (json.JSONDecodeError, TypeError) as error:
        return ToolScore(False, f"invalid tool arguments: {error}")
    ok = wanted == got
    return ToolScore(ok, "exact tool match" if ok else f"expected {wanted}, got {got}")
