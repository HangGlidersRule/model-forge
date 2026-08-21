"""Pure-Python tensor name selectors for abliteration targets."""
from __future__ import annotations

import re


def matches_selector(tensor_name: str, selectors: list[str] | tuple[str, ...]) -> bool:
    """Check if a tensor name matches any of the provided selectors.

    Selectors starting with 're:' are treated as regex patterns (fullmatch).
    All other selectors are exact string matches.
    """
    for sel in selectors:
        if sel.startswith("re:"):
            pattern = sel[3:]
            if re.fullmatch(pattern, tensor_name):
                return True
        else:
            if tensor_name == sel:
                return True
    return False


def is_vision_tensor(tensor_name: str) -> bool:
    """Check if a tensor name belongs to the vision tower."""
    return "visual" in tensor_name.lower()
