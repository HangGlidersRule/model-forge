from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field


class Case(BaseModel):
    id: str
    suite: str
    prompt: str | list[dict[str, Any]]
    scorer: Literal["exact", "contains", "json_schema", "constraints", "regex", "refusal"]
    expected: Any = None
    max_tokens: int = Field(default=256, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None


def stable_case_id(suite: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(f"{suite}\0{canonical}".encode()).hexdigest()[:16]
    return f"{suite}-{digest}"
