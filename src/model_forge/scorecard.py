from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gate:
    category: str
    minimum: float


@dataclass(frozen=True)
class Scorecard:
    categories: dict[str, float]
    weights: dict[str, float]
    weighted_score: float
    hard_gates_passed: bool
    failed_gates: tuple[str, ...]


def scorecard(categories: dict[str, float], *, weights: dict[str, float], gates: list[Gate]) -> Scorecard:
    missing = set(weights) - set(categories)
    if missing:
        raise ValueError(f"missing weighted categories: {sorted(missing)}")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum above zero")
    weighted = sum(categories[name] * weight for name, weight in weights.items()) / total
    failed = tuple(gate.category for gate in gates if categories.get(gate.category, float("-inf")) < gate.minimum)
    return Scorecard(dict(categories), dict(weights), round(weighted, 12), not failed, failed)
