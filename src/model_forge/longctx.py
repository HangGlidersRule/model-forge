from __future__ import annotations

import random

from .cases import Case, stable_case_id


def needle_case(context_tokens: int, position: float, needle: str) -> Case:
    if not 0 <= position <= 1:
        raise ValueError("position must be between 0 and 1")
    rng = random.Random(context_tokens + int(position * 1000))
    words = [f"filler{rng.randrange(10000)}" for _ in range(max(context_tokens - 20, 1))]
    index = int(len(words) * position)
    words.insert(index, f"The secret retrieval code is {needle}.")
    prompt = " ".join(words) + "\nReturn only the secret retrieval code."
    payload = {"prompt": prompt, "expected": needle, "context_tokens": context_tokens, "position": position}
    return Case(id=stable_case_id("longctx", payload), suite="longctx", prompt=prompt, scorer="exact", expected=needle, max_tokens=32, metadata={"context_tokens":context_tokens,"position":position})
