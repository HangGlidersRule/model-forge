"""Bounded glob parsing and exact language-intersection checks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

MAX_PATTERN_LENGTH = 1_024
MAX_GLOB_TOKENS = 256
MAX_PRODUCT_WORK = 1_000_000
MAX_CODEPOINT = 0x10FFFF
SLASH = ord("/")
CharacterSet = tuple[tuple[int, int], ...]
GlobToken = tuple[str, CharacterSet]
GlobAutomaton = tuple[
    int,
    int,
    dict[int, set[int]],
    dict[int, list[tuple[CharacterSet, int]]],
]
ANY_CHARACTER: CharacterSet = ((0, MAX_CODEPOINT),)
NON_SLASH_CHARACTER: CharacterSet = ((0, SLASH - 1), (SLASH + 1, MAX_CODEPOINT))
SLASH_CHARACTER: CharacterSet = ((SLASH, SLASH),)


class GlobLanguageError(ValueError):
    """A glob is invalid or exceeds a deterministic work bound."""


@dataclass(slots=True)
class WorkBudget:
    """Shared deterministic work budget for all rule-pair products."""

    limit: int = MAX_PRODUCT_WORK
    used: int = 0

    def consume(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise GlobLanguageError(f"glob product work limit exceeded: {self.limit}")


def _normalize_character_set(ranges: list[tuple[int, int]]) -> CharacterSet:
    normalized: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if normalized and start <= normalized[-1][1] + 1:
            normalized[-1] = (normalized[-1][0], max(normalized[-1][1], end))
        else:
            normalized.append((start, end))
    return tuple(normalized)


def _complement_character_set(characters: CharacterSet) -> CharacterSet:
    complement: list[tuple[int, int]] = []
    start = 0
    for range_start, range_end in characters:
        if start < range_start:
            complement.append((start, range_start - 1))
        start = range_end + 1
    if start <= MAX_CODEPOINT:
        complement.append((start, MAX_CODEPOINT))
    return tuple(complement)


def _character_sets_intersect(left: CharacterSet, right: CharacterSet) -> bool:
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        if max(left_start, right_start) <= min(left_end, right_end):
            return True
        if left_end < right_end:
            left_index += 1
        else:
            right_index += 1
    return False


def _parse_character_class(pattern: str, index: int) -> tuple[CharacterSet, int]:
    closing = pattern.find("]", index + 1)
    if closing == -1:
        raise GlobLanguageError(f"unterminated character class: {pattern}")
    content = pattern[index + 1 : closing]
    if not content:
        raise GlobLanguageError(f"empty character class: {pattern}")
    negated = content.startswith(("!", "^"))
    if negated:
        content = content[1:]
    if not content:
        raise GlobLanguageError(f"empty character class: {pattern}")

    ranges: list[tuple[int, int]] = []
    content_index = 0
    while content_index < len(content):
        start = ord(content[content_index])
        if content_index + 2 < len(content) and content[content_index + 1] == "-":
            end = ord(content[content_index + 2])
            if start > end:
                raise GlobLanguageError(f"reversed character range: {pattern}")
            ranges.append((start, end))
            content_index += 3
        else:
            ranges.append((start, start))
            content_index += 1

    characters = _normalize_character_set(ranges)
    if negated:
        characters = _complement_character_set(characters)
    path_characters: list[tuple[int, int]] = []
    for start, end in characters:
        if start < SLASH:
            path_characters.append((start, min(end, SLASH - 1)))
        if end > SLASH:
            path_characters.append((max(start, SLASH + 1), end))
    return tuple(path_characters), closing + 1


def parse_glob(pattern: str) -> tuple[GlobToken, ...]:
    """Parse the documented glob grammar into a bounded normalized token stream."""

    if len(pattern) > MAX_PATTERN_LENGTH:
        raise GlobLanguageError(
            f"manifest source pattern exceeds {MAX_PATTERN_LENGTH} characters"
        )
    tokens: list[GlobToken] = []

    def append_token(token: GlobToken) -> None:
        if token[0] == "many" and tokens and tokens[-1][0] == "many":
            previous = tokens[-1][1]
            if previous == token[1]:
                return
            if previous == ANY_CHARACTER or token[1] == ANY_CHARACTER:
                tokens[-1] = ("many", ANY_CHARACTER)
                return
        tokens.append(token)
        if len(tokens) > MAX_GLOB_TOKENS:
            raise GlobLanguageError(
                f"manifest source token limit exceeded: {MAX_GLOB_TOKENS}"
            )

    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "*":
            star_start = index
            while index < len(pattern) and pattern[index] == "*":
                index += 1
            if index - star_start >= 2:
                if index < len(pattern) and pattern[index] == "/":
                    append_token(("recursive-directory", ANY_CHARACTER))
                    index += 1
                else:
                    append_token(("many", ANY_CHARACTER))
                continue
            append_token(("many", NON_SLASH_CHARACTER))
            continue
        if character == "?":
            append_token(("one", NON_SLASH_CHARACTER))
        elif character == "[":
            characters, index = _parse_character_class(pattern, index)
            append_token(("one", characters))
            continue
        elif character == "\\":
            raise GlobLanguageError(f"backslashes are unsupported: {pattern}")
        else:
            append_token(("one", ((ord(character), ord(character)),)))
        index += 1
    return tuple(tokens)


def _glob_automaton(pattern: str) -> GlobAutomaton:
    epsilon: dict[int, set[int]] = {}
    transitions: dict[int, list[tuple[CharacterSet, int]]] = {}
    state = 0
    last_allocated = 0
    for kind, characters in parse_glob(pattern):
        next_state = last_allocated + 1
        if kind == "one":
            transitions.setdefault(state, []).append((characters, next_state))
        elif kind == "many":
            epsilon.setdefault(state, set()).add(next_state)
            transitions.setdefault(state, []).append((characters, state))
        else:
            recursive_state = next_state + 1
            last_allocated = recursive_state
            epsilon.setdefault(state, set()).update({next_state, recursive_state})
            transitions.setdefault(recursive_state, []).extend(
                [
                    (ANY_CHARACTER, recursive_state),
                    (SLASH_CHARACTER, next_state),
                ]
            )
        state = next_state
        last_allocated = max(last_allocated, next_state)
    return 0, state, epsilon, transitions


def _epsilon_closure(
    states: set[int], epsilon: dict[int, set[int]]
) -> frozenset[int]:
    closure = set(states)
    pending = list(states)
    while pending:
        state = pending.pop()
        for destination in epsilon.get(state, set()):
            if destination not in closure:
                closure.add(destination)
                pending.append(destination)
    return frozenset(closure)


def glob_languages_overlap(
    left: str, right: str, budget: WorkBudget | None = None
) -> bool:
    """Return whether two glob languages intersect, independent of repository files."""

    active_budget = budget or WorkBudget()
    left_start, left_accepting, left_epsilon, left_transitions = _glob_automaton(left)
    right_start, right_accepting, right_epsilon, right_transitions = _glob_automaton(right)
    initial = (
        _epsilon_closure({left_start}, left_epsilon),
        _epsilon_closure({right_start}, right_epsilon),
    )
    pending = deque([initial])
    visited = {initial}
    while pending:
        left_states, right_states = pending.popleft()
        if left_accepting in left_states and right_accepting in right_states:
            return True
        for left_state in left_states:
            for left_characters, left_destination in left_transitions.get(left_state, []):
                for right_state in right_states:
                    for right_characters, right_destination in right_transitions.get(
                        right_state, []
                    ):
                        active_budget.consume()
                        if not _character_sets_intersect(
                            left_characters, right_characters
                        ):
                            continue
                        destination = (
                            _epsilon_closure({left_destination}, left_epsilon),
                            _epsilon_closure({right_destination}, right_epsilon),
                        )
                        if destination not in visited:
                            visited.add(destination)
                            pending.append(destination)
    return False
