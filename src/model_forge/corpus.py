from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from .cases import Case, stable_case_id


def load_builtin_suite(name: str) -> list[Case]:
    path = files("model_forge").joinpath("data", f"{name}.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for item in raw:
        payload: dict[str, Any] = dict(item)
        payload["id"] = payload.get("id") or stable_case_id(str(payload["suite"]), payload)
        cases.append(Case.model_validate(payload))
    return cases


def load_external_cases(path: Path, suite: str, adapter: str = "generic") -> list[Case]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for row in rows:
        if adapter in {"generic", "mmlu", "gpqa", "gsm8k", "humaneval", "mbpp", "ifeval", "bfcl", "longbench", "niah"}:
            prompt = row.get("prompt") or row.get("question") or row.get("input")
            expected = row.get("expected", row.get("answer", row.get("output")))
            scorer = row.get("scorer", "exact")
        else:
            raise ValueError(f"unsupported adapter: {adapter}")
        payload = {"suite": suite, "prompt": prompt, "expected": expected, "scorer": scorer, "metadata": {"adapter": adapter}}
        payload["id"] = stable_case_id(suite, payload)
        cases.append(Case.model_validate(payload))
    return cases
