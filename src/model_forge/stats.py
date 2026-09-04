from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    samples: int


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def percentiles(values: list[float], points: list[int]) -> dict[str, float]:
    if not values:
        return {f"p{point}": 0.0 for point in points}
    return {f"p{point}": _quantile(values, point / 100) for point in points}


def bootstrap_paired_delta(candidate: list[float], baseline: list[float], *, seed: int = 42, samples: int = 2000) -> Interval:
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("paired samples must be non-empty and equal length")
    deltas = [a - b for a, b in zip(candidate, baseline, strict=True)]
    rng = random.Random(seed)
    means = [sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(samples)]
    return Interval(sum(deltas) / len(deltas), _quantile(means, 0.025), _quantile(means, 0.975), len(deltas))
