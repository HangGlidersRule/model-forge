from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceCell:
    concurrency: int
    prompt_tokens: int
    output_tokens: int


def build_performance_cases(*, concurrencies: list[int], prompt_tokens: list[int], output_tokens: list[int]) -> list[PerformanceCell]:
    return [PerformanceCell(c, p, o) for c in concurrencies for p in prompt_tokens for o in output_tokens]
