from __future__ import annotations

import json
import re
import stat
import subprocess
import sys
import unicodedata
from collections import deque
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tools/public_export/public-files.yaml"
PR_A_FILES = {
    "tests/public_export/test_public_file_manifest.py",
    "tools/public_export/README.md",
    "tools/public_export/public-files.yaml",
}
PR_D_FILES = {
    ".gitleaks.toml",
    "scripts/bootstrap_public_export_wheelhouse.py",
    "scripts/verify_public_export.sh",
    "src/model_forge/cli.py",
    "src/model_forge/public_export/exporter.py",
    "src/model_forge/public_export/verifier.py",
    "src/model_forge/public_export/wheelhouse.py",
    "tests/public_export/test_public_export_e2e.py",
    "tests/public_export/test_wheelhouse_bootstrap.py",
}
PR_E_FILES = {
    ".github/CODEOWNERS",
    ".github/ISSUE_TEMPLATE/bug.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/feature.yml",
    ".github/pull_request_template.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "SECURITY.md",
    "SUPPORT.md",
    "docs/decisions/0001-private-archive-public-root-separation.md",
    "docs/decisions/0002-release-publication-authority.md",
    "tests/governance/schemas/github-issue-form.schema.json",
    "tests/governance/test_public_governance.py",
}
PR_F_FILES = {
    ".github/workflows/ci.yml",
    "docs/pull-request-risk.md",
    "src/model_forge/pr_risk.py",
    "tests/governance/test_ci_policy.py",
    "tests/pr_risk/fixtures/cases.json",
    "tests/pr_risk/test_classifier.py",
}
PR_GHI_FILES = {
    "contracts/ai-review/README.md",
    "contracts/ai-review/v1/request.schema.json",
    "contracts/ai-review/v1/result.schema.json",
    "docs/decisions/0003-maintainer-gated-advisory-ai-review.md",
    "private_archive/ai_review/README.md",
    "private_archive/ai_review/dispatcher.py",
    "private_archive/ai_review/tests/test_dispatcher.py",
    "scripts/stage_public_root.sh",
    "tests/governance/test_ai_review_contract.py",
    "tests/public_export/test_public_root_self_hosting.py",
    "tests/public_export/test_public_staging.py",
}
PRIVATE_ARCHIVE_PREFIX = "private_archive/"
# The verifier-owned attestation is written into every generated public root. It is
# export output rather than classified source, so no manifest rule may claim it.
GENERATED_ATTESTATION = "PUBLIC_EXPORT_MANIFEST.json"
ATTESTATION_SCHEMA = "model-forge-public-export/v1"
PENDING_CLASSIFICATION_FILES = (
    PR_A_FILES | PR_D_FILES | PR_E_FILES | PR_F_FILES | PR_GHI_FILES
)
REQUIRED_FIELDS = {
    "source",
    "disposition",
    "public_destination",
    "reason",
    "transformation",
    "owner",
    "max_size_bytes",
    "generated",
}
ALLOWED_MANIFEST_FIELDS = {"version", "rules"}
ALLOWED_RULE_FIELDS = REQUIRED_FIELDS | {
    "id",
    "precedence",
    "resolves",
    "allow_empty",
    "content_classification",
    "detector_suppressions",
    "regeneration_check",
}
APPROVED_DETECTOR_SUPPRESSIONS = {"benchmark.raw-key"}
TRUSTED_CONTENT_CLASSIFICATIONS = {
    "trusted-source-code",
    "trusted-detector-fixture",
}
MAX_MANIFEST_BYTES = 1_048_576
MAX_RULES = 256
MAX_PATTERN_LENGTH = 1_024
MAX_GLOB_TOKENS = 256
MAX_PRODUCT_WORK = 1_000_000
MAX_SIZE_BYTES = 1_073_741_824
MAX_PRECEDENCE = 1_000_000
GLOB_MAGIC = re.compile(r"[*?[]")
MAX_CODEPOINT = 0x10FFFF
SLASH = ord("/")
ANY_CHARACTER = ((0, MAX_CODEPOINT),)
NON_SLASH_CHARACTER = ((0, SLASH - 1), (SLASH + 1, MAX_CODEPOINT))
SLASH_CHARACTER = ((SLASH, SLASH),)
CharacterSet = tuple[tuple[int, int], ...]
GlobToken = tuple[str, CharacterSet]
GlobAutomaton = tuple[
    int,
    int,
    dict[int, set[int]],
    dict[int, list[tuple[CharacterSet, int]]],
]


class ManifestError(ValueError):
    """The public-file manifest violates its fail-closed contract."""


class _WorkBudget:
    def __init__(self, limit: int, label: str) -> None:
        self.limit = limit
        self.label = label
        self.used = 0

    def consume(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise ManifestError(f"{self.label} work limit exceeded: {self.limit}")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ManifestError("YAML mapping keys must be scalar") from error
        if duplicate:
            raise ManifestError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate attestation JSON key: {key}")
        result[key] = value
    return result


def _canonical_attestation_path(value: str) -> bool:
    return (
        _is_canonical_repository_relative(value)
        and unicodedata.normalize("NFC", value) == value
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _attestation_records(root: Path = REPO_ROOT) -> list[dict[str, str]]:
    path = root / GENERATED_ATTESTATION
    try:
        metadata = path.lstat()
        attestation = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except ManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError("public export attestation is unreadable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or not isinstance(attestation, dict)
        or attestation.get("schema") != ATTESTATION_SCHEMA
        or type(attestation.get("schema_version")) is not int
        or attestation.get("schema_version") != 1
        or not isinstance(attestation.get("files"), list)
        or not attestation["files"]
    ):
        raise ManifestError("public export attestation is malformed")

    records: list[dict[str, str]] = []
    output_paths: set[str] = set()
    source_ids: set[str] = set()
    for raw in attestation["files"]:
        if not isinstance(raw, dict):
            raise ManifestError("public export attestation source inventory is malformed")
        output_path = raw.get("output_path")
        source_id = raw.get("source_id")
        if (
            not isinstance(output_path, str)
            or not _canonical_attestation_path(output_path)
            or output_path == GENERATED_ATTESTATION
            or not isinstance(source_id, str)
            or not source_id
        ):
            raise ManifestError("public export attestation source inventory is malformed")
        if output_path in output_paths or source_id in source_ids:
            raise ManifestError("public export attestation records must be unique")
        if source_id.startswith("rule:"):
            if not source_id.removeprefix("rule:"):
                raise ManifestError("public export attestation source inventory is malformed")
        elif source_id != output_path or not _canonical_attestation_path(source_id):
            raise ManifestError("attestation path source must equal its output path")
        output_paths.add(output_path)
        source_ids.add(source_id)
        records.append({"output_path": output_path, "source_id": source_id})
    return records


def _is_generated_public_root(root: Path = REPO_ROOT) -> bool:
    """Report whether this tree is an exported public root rather than the private source.

    The only admissible marker is the verifier-owned export attestation at the root. A
    present but unreadable or foreign attestation fails closed instead of silently
    selecting either context.
    """
    path = root / GENERATED_ATTESTATION
    if not path.exists() and not path.is_symlink():
        return False
    _attestation_records(root)
    return True


def _attested_source_ids(root: Path = REPO_ROOT) -> set[str]:
    return {record["source_id"] for record in _attestation_records(root)}


def _git_tracked_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    tracked = {path.decode() for path in result.stdout.split(b"\0") if path}
    if _is_generated_public_root():
        attested = {
            record["output_path"] for record in _attestation_records()
        } | {GENERATED_ATTESTATION}
        if tracked != attested:
            raise ManifestError(
                "generated public root tracked inventory does not equal attested outputs "
                f"plus attestation: missing={sorted(attested - tracked)}, "
                f"extra={sorted(tracked - attested)}"
            )
        tracked.remove(GENERATED_ATTESTATION)
    # These files are part of PR-A even before they are staged. Do not include arbitrary
    # untracked files: doing so would make local state alter the classification boundary.
    tracked.update(
        path for path in PENDING_CLASSIFICATION_FILES if (REPO_ROOT / path).is_file()
    )
    return tracked


def _load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ManifestError(f"manifest size limit exceeded: {MAX_MANIFEST_BYTES}")
    raw = path.read_bytes()
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ManifestError(f"manifest size limit exceeded: {MAX_MANIFEST_BYTES}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ManifestError("manifest must be UTF-8") from error
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ManifestError(f"invalid YAML manifest: {error.problem}") from error
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be a mapping")
    unknown = set(data) - ALLOWED_MANIFEST_FIELDS
    if unknown:
        raise ManifestError(f"unknown manifest fields: {sorted(unknown)}")
    missing = ALLOWED_MANIFEST_FIELDS - data.keys()
    if missing:
        raise ManifestError(f"missing manifest fields: {sorted(missing)}")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ManifestError("manifest version must be 1")
    rules = data["rules"]
    if not isinstance(rules, list) or not rules:
        raise ManifestError("manifest rules must be a non-empty list")
    if len(rules) > MAX_RULES:
        raise ManifestError(f"manifest rule limit exceeded: {MAX_RULES}")
    return data


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


def _intersect_character_sets(left: CharacterSet, right: CharacterSet) -> bool:
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
        raise ManifestError(f"unterminated character class in source: {pattern!r}")
    content = pattern[index + 1 : closing]
    if not content:
        raise ManifestError(f"empty character class in source: {pattern!r}")

    negated = content.startswith(("!", "^"))
    if negated:
        content = content[1:]
    if not content:
        raise ManifestError(f"empty character class in source: {pattern!r}")

    ranges: list[tuple[int, int]] = []
    content_index = 0
    while content_index < len(content):
        start = ord(content[content_index])
        if (
            content_index + 2 < len(content)
            and content[content_index + 1] == "-"
        ):
            end = ord(content[content_index + 2])
            if start > end:
                raise ManifestError(f"reversed character range in source: {pattern!r}")
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


def _parse_glob(pattern: str) -> tuple[GlobToken, ...]:
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise ManifestError(f"source pattern length limit exceeded: {MAX_PATTERN_LENGTH}")
    tokens: list[GlobToken] = []

    def append_token(token: GlobToken) -> None:
        if token[0] == "many" and tokens and tokens[-1][0] == "many":
            previous_characters = tokens[-1][1]
            if previous_characters == token[1]:
                return
            if previous_characters == ANY_CHARACTER or token[1] == ANY_CHARACTER:
                tokens[-1] = ("many", ANY_CHARACTER)
                return
        tokens.append(token)
        if len(tokens) > MAX_GLOB_TOKENS:
            raise ManifestError(f"source token limit exceeded: {MAX_GLOB_TOKENS}")

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
        elif character == "?":
            append_token(("one", NON_SLASH_CHARACTER))
        elif character == "[":
            characters, index = _parse_character_class(pattern, index)
            append_token(("one", characters))
            continue
        elif character == "\\":
            raise ManifestError(f"backslashes are unsupported in source: {pattern!r}")
        else:
            append_token(("one", ((ord(character), ord(character)),)))
        index += 1
    return tuple(tokens)


def _glob_automaton(pattern: str) -> GlobAutomaton:
    epsilon: dict[int, set[int]] = {}
    transitions: dict[int, list[tuple[CharacterSet, int]]] = {}
    state = 0
    last_allocated = 0
    for kind, characters in _parse_glob(pattern):
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


def _epsilon_closure(states: set[int], epsilon: dict[int, set[int]]) -> frozenset[int]:
    closure = set(states)
    pending = list(states)
    while pending:
        state = pending.pop()
        for destination in epsilon.get(state, set()):
            if destination not in closure:
                closure.add(destination)
                pending.append(destination)
    return frozenset(closure)


def _matches(source: str, path: str) -> bool:
    start, accepting, epsilon, transitions = _glob_automaton(source)
    states = _epsilon_closure({start}, epsilon)
    for character in path:
        codepoint = ord(character)
        destinations = {
            destination
            for state in states
            for characters, destination in transitions.get(state, [])
            if any(start <= codepoint <= end for start, end in characters)
        }
        states = _epsilon_closure(destinations, epsilon)
        if not states:
            return False
    return accepting in states


def _glob_languages_overlap(
    left: str, right: str, budget: _WorkBudget | None = None
) -> bool:
    if budget is None:
        budget = _WorkBudget(MAX_PRODUCT_WORK, "glob product")
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
                        budget.consume()
                        if not _intersect_character_sets(
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


def _is_bounded_source(source: str) -> bool:
    if not source or source.startswith(("/", "\\")):
        return False
    path = PurePosixPath(source)
    if ".." in path.parts or source in {"*", "**", "**/*"}:
        return False
    try:
        _parse_glob(source)
    except ManifestError:
        return False
    first_magic = GLOB_MAGIC.search(source)
    if first_magic is None:
        return True
    literal_prefix = source[: first_magic.start()]
    return "/" in literal_prefix


def _is_canonical_repository_relative(value: str) -> bool:
    if (
        not value
        or "\\" in value
        or value.startswith("/")
        or re.match(r"^[A-Za-z]:", value)
    ):
        return False
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        return False
    return PurePosixPath(value).as_posix() == value


def _validate_rule_shape(rule: Any, rule_ids: set[str]) -> None:
    if not isinstance(rule, dict):
        raise ManifestError("every rule must be a mapping")
    unknown = set(rule) - ALLOWED_RULE_FIELDS
    if unknown:
        raise ManifestError(f"rule has unknown fields: {sorted(unknown)}")
    rule_id = rule.get("id")
    if not isinstance(rule_id, str) or not rule_id:
        raise ManifestError("every rule must have a non-empty id")
    if rule_id in rule_ids:
        raise ManifestError(f"duplicate rule id: {rule_id}")
    rule_ids.add(rule_id)

    missing = REQUIRED_FIELDS - rule.keys()
    if missing:
        raise ManifestError(f"{rule_id} lacks required fields: {sorted(missing)}")
    source = rule["source"]
    if not isinstance(source, str):
        raise ManifestError(f"{rule_id} has an unbounded source: {source!r}")
    if not _is_canonical_repository_relative(source):
        raise ManifestError(
            f"{rule_id} source must be a canonical POSIX repository-relative pattern"
        )
    _parse_glob(source)
    if not _is_bounded_source(source):
        raise ManifestError(f"{rule_id} has an unbounded source: {source!r}")
    if not isinstance(rule["disposition"], str) or rule["disposition"] not in {
        "copy",
        "transform",
        "exclude",
    }:
        raise ManifestError(f"{rule_id} has an invalid disposition")
    if not isinstance(rule["reason"], str) or not rule["reason"].strip():
        raise ManifestError(f"{rule_id} must explain its classification")
    if not isinstance(rule["owner"], str) or not rule["owner"].strip():
        raise ManifestError(f"{rule_id} must declare an owner class")
    maximum_size = rule["max_size_bytes"]
    if (
        type(maximum_size) is not int
        or maximum_size <= 0
        or maximum_size > MAX_SIZE_BYTES
    ):
        raise ManifestError(
            f"{rule_id} max_size_bytes must be a positive integer "
            f"no greater than {MAX_SIZE_BYTES}"
        )
    if type(rule["generated"]) is not bool:
        raise ManifestError(f"{rule_id} generated must be boolean")
    if "allow_empty" in rule and type(rule["allow_empty"]) is not bool:
        raise ManifestError(f"{rule_id} allow_empty must be boolean")
    if "precedence" in rule:
        precedence = rule["precedence"]
        if (
            type(precedence) is not int
            or precedence <= 0
            or precedence > MAX_PRECEDENCE
        ):
            raise ManifestError(
                f"{rule_id} precedence must be a positive integer "
                f"no greater than {MAX_PRECEDENCE}"
            )
    resolves = rule.get("resolves", [])
    if not isinstance(resolves, list) or not all(
        isinstance(item, str) and item for item in resolves
    ):
        raise ManifestError(f"{rule_id} resolves must be a list of rule ids")
    classification = rule.get("content_classification")
    if classification is not None and classification not in TRUSTED_CONTENT_CLASSIFICATIONS:
        raise ManifestError(f"{rule_id} has an invalid content classification")
    suppressions = rule.get("detector_suppressions", [])
    if (
        not isinstance(suppressions, list)
        or not all(isinstance(item, str) and item for item in suppressions)
        or len(suppressions) != len(set(suppressions))
    ):
        raise ManifestError(f"{rule_id} detector suppressions must be unique rule ids")
    if suppressions and (
        classification is None
        or not set(suppressions) <= APPROVED_DETECTOR_SUPPRESSIONS
    ):
        raise ManifestError(f"{rule_id} has a forbidden detector suppression")

    destination = rule["public_destination"]
    transformation = rule["transformation"]
    if rule["disposition"] == "exclude":
        if destination is not None or transformation is not None:
            raise ManifestError(f"{rule_id} excluded rules cannot emit or transform")
    else:
        if not isinstance(destination, str):
            raise ManifestError(f"{rule_id} must declare a public destination")
        if destination != "{source}" and not _is_canonical_repository_relative(destination):
            raise ManifestError(
                f"{rule_id} destination must be a canonical POSIX "
                "repository-relative path"
            )
        if destination != "{source}" and GLOB_MAGIC.search(source):
            raise ManifestError(f"{rule_id} glob destinations must preserve {{source}}")
        if rule["disposition"] == "copy" and transformation is not None:
            raise ManifestError(f"{rule_id} copy rules cannot transform")
        if rule["disposition"] == "transform" and not isinstance(transformation, str):
            raise ManifestError(f"{rule_id} transform rules need a transformation function")
    if suppressions and (
        rule["disposition"] != "copy" or destination != "{source}"
    ):
        raise ManifestError(
            f"{rule_id} detector suppression requires unchanged copied bytes"
        )

    regeneration_check = rule.get("regeneration_check")
    if regeneration_check is not None and (
        not isinstance(regeneration_check, str) or not regeneration_check.strip()
    ):
        raise ManifestError(f"{rule_id} regeneration_check must be a non-empty string")
    if rule["generated"] and (
        not isinstance(regeneration_check, str) or not regeneration_check.strip()
    ):
        raise ManifestError(f"{rule_id} generated entries need a regeneration check")


def _resolve_rule(rules: list[dict[str, Any]], path: str) -> dict[str, Any]:
    matching = [rule for rule in rules if _matches(rule["source"], path)]
    if not matching:
        raise ManifestError(f"unclassified tracked file: {path}")
    matching.sort(key=lambda rule: rule.get("precedence", 0), reverse=True)
    winner = matching[0]
    if len(matching) == 1:
        return winner
    runner_up = matching[1]
    if winner.get("precedence", 0) == runner_up.get("precedence", 0):
        raise ManifestError(f"ambiguous rules for {path}: {winner['id']}, {runner_up['id']}")
    unresolved = {rule["id"] for rule in matching[1:]} - set(winner.get("resolves", []))
    if unresolved:
        raise ManifestError(
            f"{winner['id']} does not explicitly resolve overlaps for {path}: "
            f"{sorted(unresolved)}"
        )
    return winner


def _validate_rule_overlaps(
    rules: list[dict[str, Any]], budget: _WorkBudget
) -> None:
    for left, right in combinations(rules, 2):
        if not _glob_languages_overlap(left["source"], right["source"], budget):
            continue
        left_precedence = left.get("precedence", 0)
        right_precedence = right.get("precedence", 0)
        if left_precedence == right_precedence:
            raise ManifestError(
                f"ambiguous rules have intersecting source languages: "
                f"{left['id']}, {right['id']}"
            )
        winner, shadowed = (
            (left, right)
            if left_precedence > right_precedence
            else (right, left)
        )
        if shadowed["id"] not in winner.get("resolves", []):
            raise ManifestError(
                f"{winner['id']} does not explicitly resolve overlap with "
                f"{shadowed['id']}"
            )


def _destination(rule: dict[str, Any], source: str) -> str | None:
    destination = rule["public_destination"]
    if destination is None:
        return None
    return source if destination == "{source}" else destination


def _resolved_disposition(
    rules: list[dict[str, Any]], path: str
) -> str | None:
    if not any(_matches(rule["source"], path) for rule in rules):
        return None
    return str(_resolve_rule(rules, path)["disposition"])


def _validate_manifest(
    manifest: dict[str, Any],
    tracked_files: set[str],
    *,
    public_root: bool | None = None,
    attested_source_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    if public_root is None:
        public_root = _is_generated_public_root()
    if public_root and attested_source_ids is None:
        attested_source_ids = _attested_source_ids()
    if not isinstance(manifest, dict):
        raise ManifestError("manifest root must be a mapping")
    unknown_manifest_fields = set(manifest) - ALLOWED_MANIFEST_FIELDS
    if unknown_manifest_fields:
        raise ManifestError(
            f"unknown manifest fields: {sorted(unknown_manifest_fields)}"
        )
    if set(manifest) != ALLOWED_MANIFEST_FIELDS:
        raise ManifestError("manifest must contain exactly version and rules")
    if type(manifest["version"]) is not int or manifest["version"] != 1:
        raise ManifestError("manifest version must be 1")
    rules = manifest["rules"]
    if not isinstance(rules, list) or not rules:
        raise ManifestError("manifest rules must be a non-empty list")
    if len(rules) > MAX_RULES:
        raise ManifestError(f"manifest rule limit exceeded: {MAX_RULES}")
    rule_ids: set[str] = set()
    for rule in rules:
        _validate_rule_shape(rule, rule_ids)
    for rule in rules:
        unknown = set(rule.get("resolves", [])) - rule_ids
        if unknown:
            raise ManifestError(f"{rule['id']} resolves unknown rules: {sorted(unknown)}")
    _validate_rule_overlaps(rules, _WorkBudget(MAX_PRODUCT_WORK, "glob product"))

    classified = {path: _resolve_rule(rules, path) for path in sorted(tracked_files)}
    for path, rule in classified.items():
        if not rule.get("detector_suppressions"):
            continue
        classification = rule["content_classification"]
        if classification == "trusted-source-code" and (
            not path.startswith(("src/", "tests/"))
            or PurePosixPath(path).suffix not in {".py", ".pyi"}
        ):
            raise ManifestError(f"detector suppression on non-source path: {path}")
        if classification == "trusted-detector-fixture" and (
            GLOB_MAGIC.search(rule["source"])
            or not path.startswith("tests/public_export/fixtures/")
            or PurePosixPath(path).suffix != ".json"
        ):
            raise ManifestError(f"detector suppression on invalid fixture path: {path}")
    source_paths = {
        source_id
        for source_id in attested_source_ids or set()
        if not source_id.startswith("rule:")
    }
    attested_rule_ids = {
        source_id.removeprefix("rule:")
        for source_id in attested_source_ids or set()
        if source_id.startswith("rule:")
    }
    unknown_attested_rules = attested_rule_ids - rule_ids
    if unknown_attested_rules:
        raise ManifestError(
            f"attestation references unknown manifest rules: {sorted(unknown_attested_rules)}"
        )
    matched_rule_ids = attested_rule_ids | {
        rule["id"]
        for rule in rules
        if any(_matches(rule["source"], path) for path in tracked_files | source_paths)
    }
    stale = {
        rule["id"]
        for rule in rules
        if rule["id"] not in matched_rule_ids
        and not rule.get("allow_empty", False)
        # A generated public root contains no excluded file by construction, so only the
        # private source can require exclusion rules to guard tracked files.
        and not (public_root and rule["disposition"] == "exclude")
    }
    if stale:
        raise ManifestError(f"rules match no tracked files: {sorted(stale)}")

    destinations: dict[str, str] = {}
    for source, rule in classified.items():
        source_path = REPO_ROOT / source
        if source_path.exists() and source_path.stat().st_size > rule["max_size_bytes"]:
            raise ManifestError(
                f"{source} exceeds {rule['id']} maximum size "
                f"({source_path.stat().st_size} > {rule['max_size_bytes']})"
            )
        destination = _destination(rule, source)
        if destination is None:
            continue
        if not _is_canonical_repository_relative(destination):
            raise ManifestError(
                f"{rule['id']} resolved destination must be a canonical POSIX "
                f"repository-relative path: {destination!r}"
            )
        previous = destinations.setdefault(destination, source)
        if previous != source:
            raise ManifestError(
                f"duplicate public destination {destination}: {previous}, {source}"
            )
    return classified


def _validate_simulated_output(
    output_paths: set[str],
    rules: list[dict[str, Any]],
    classified: dict[str, dict[str, Any]],
) -> None:
    allowed = {
        destination
        for source, rule in classified.items()
        if (destination := _destination(rule, source)) is not None
    }
    source_by_destination = {
        destination: source
        for source, rule in classified.items()
        if (destination := _destination(rule, source)) is not None
    }
    excluded = {
        path
        for path in output_paths
        if _resolved_disposition(rules, source_by_destination.get(path, path))
        == "exclude"
    }
    if excluded:
        raise ManifestError(f"excluded paths appear in output: {sorted(excluded)}")
    unexpected = output_paths - allowed
    if unexpected:
        raise ManifestError(f"unclassified paths appear in output: {sorted(unexpected)}")


def _assert_no_escaping_symlinks(root: Path, members: set[str] | None = None) -> None:
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise ManifestError(f"repository root cannot be resolved: {root}") from error

    if members is None:
        members = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
        }

    for member in sorted(members):
        relative = PurePosixPath(member)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ManifestError(f"tracked member is not repository-relative: {member}")

        candidate = root
        for component in relative.parts:
            candidate /= component
            if candidate.is_symlink():
                raise ManifestError(
                    f"symlink escapes export root: tracked member has symlink component: {member}"
                )
            if not candidate.exists():
                raise ManifestError(f"tracked member is missing: {member}")

        try:
            resolved_member = candidate.resolve(strict=True)
        except OSError as error:
            raise ManifestError(f"tracked member cannot be resolved: {member}") from error
        if not resolved_member.is_relative_to(resolved_root):
            raise ManifestError(f"tracked member escapes repository root: {member}")


@pytest.fixture
def manifest() -> dict[str, Any]:
    return _load_manifest()


def test_manifest_classifies_every_tracked_file_fail_closed(
    manifest: dict[str, Any],
) -> None:
    tracked = _git_tracked_files()
    classified = _validate_manifest(manifest, tracked)
    assert set(classified) == tracked


def test_public_attestation_inventory_exactly_matches_regular_files() -> None:
    if not _is_generated_public_root():
        pytest.skip("exact attestation inventory applies only to a generated public root")
    attestation = json.loads(
        (REPO_ROOT / GENERATED_ATTESTATION).read_text(encoding="utf-8")
    )
    attested = {record["output_path"] for record in attestation["files"]} | {
        GENERATED_ATTESTATION
    }
    tree = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "-z", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    actual = {raw.decode("utf-8") for raw in tree.stdout.split(b"\0") if raw}
    assert all(
        (REPO_ROOT / relative).is_file()
        and not (REPO_ROOT / relative).is_symlink()
        for relative in actual
    )
    assert attested == actual
    assert "containers/build/docker-compose.yml" in attested
    assert (REPO_ROOT / "containers/build/docker-compose.yml").is_file()


def test_pr_a_files_are_classified_before_staging(manifest: dict[str, Any]) -> None:
    classified = _validate_manifest(manifest, _git_tracked_files())
    assert PR_A_FILES <= classified.keys()


def test_pr_d_files_are_classified_before_staging(manifest: dict[str, Any]) -> None:
    classified = _validate_manifest(manifest, _git_tracked_files())
    assert PR_D_FILES <= classified.keys()


def test_pr_e_files_are_classified_before_staging(manifest: dict[str, Any]) -> None:
    classified = _validate_manifest(manifest, _git_tracked_files())
    assert PR_E_FILES <= classified.keys()


def test_pr_f_files_are_classified_before_staging(manifest: dict[str, Any]) -> None:
    classified = _validate_manifest(manifest, _git_tracked_files())
    assert PR_F_FILES <= classified.keys()


def test_combined_public_launch_files_are_classified_before_staging(
    manifest: dict[str, Any],
) -> None:
    classified = _validate_manifest(manifest, _git_tracked_files())
    private_files = {
        path for path in PR_GHI_FILES if path.startswith(PRIVATE_ARCHIVE_PREFIX)
    }
    assert PR_GHI_FILES - private_files <= classified.keys()
    if _is_generated_public_root():
        assert not any(path.startswith(PRIVATE_ARCHIVE_PREFIX) for path in classified)
        return
    assert private_files <= classified.keys()
    assert all(classified[path]["disposition"] == "exclude" for path in private_files)


def test_private_archive_paths_are_excluded_in_every_context(
    manifest: dict[str, Any],
) -> None:
    introduced = f"{PRIVATE_ARCHIVE_PREFIX}ai_review/dispatcher.py"
    guard = _resolve_rule(manifest["rules"], introduced)
    assert guard["id"] == "private-archive-guard"
    assert guard["disposition"] == "exclude"

    classified = _validate_manifest(manifest, _git_tracked_files() | {introduced})
    assert classified[introduced]["disposition"] == "exclude"
    allowed_output = {
        destination
        for source, rule in classified.items()
        if (destination := _destination(rule, source)) is not None
    }
    with pytest.raises(ManifestError, match="excluded paths appear in output"):
        _validate_simulated_output(
            allowed_output | {introduced}, manifest["rules"], classified
        )


def test_generated_attestation_is_never_classified_source(
    manifest: dict[str, Any],
) -> None:
    assert _resolved_disposition(manifest["rules"], GENERATED_ATTESTATION) is None
    with pytest.raises(ManifestError, match="unclassified tracked file"):
        _validate_manifest(manifest, _git_tracked_files() | {GENERATED_ATTESTATION})


def test_malformed_generated_attestation_fails_closed(tmp_path: Path) -> None:
    (tmp_path / GENERATED_ATTESTATION).write_text("{}", encoding="utf-8")

    with pytest.raises(ManifestError, match="attestation is malformed"):
        _is_generated_public_root(tmp_path)


@pytest.mark.parametrize(
    "records",
    [
        [
            {"output_path": "docs/a.md", "source_id": "docs/a.md"},
            {"output_path": "docs/a.md", "source_id": "docs/b.md"},
        ],
        [
            {"output_path": "docs/a.md", "source_id": "rule:docs"},
            {"output_path": "docs/b.md", "source_id": "rule:docs"},
        ],
        [{"output_path": "docs/a.md", "source_id": "docs/source.md"}],
        [{"output_path": "../docs/a.md", "source_id": "../docs/a.md"}],
    ],
)
def test_attestation_records_require_canonical_unique_output_source_pairs(
    tmp_path: Path,
    records: list[dict[str, str]],
) -> None:
    (tmp_path / GENERATED_ATTESTATION).write_text(
        json.dumps(
            {
                "schema": ATTESTATION_SCHEMA,
                "schema_version": 1,
                "files": records,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError):
        _attestation_records(tmp_path)


def test_attested_rule_provenance_must_name_an_existing_manifest_rule() -> None:
    copied = {
        **_synthetic_exclude_rule("public-file", "bounded/file.txt"),
        "disposition": "copy",
        "public_destination": "{source}",
    }
    with pytest.raises(ManifestError, match="unknown manifest rules"):
        _validate_manifest(
            {"version": 1, "rules": [copied]},
            {"bounded/file.txt"},
            public_root=True,
            attested_source_ids={"rule:missing"},
        )


def test_exclusion_rules_may_match_nothing_only_in_a_public_root() -> None:
    guard = _synthetic_exclude_rule("private-guard", "private/**")
    del guard["allow_empty"]
    copied = {
        **_synthetic_exclude_rule("public-file", "bounded/file.txt"),
        "disposition": "copy",
        "public_destination": "{source}",
    }
    candidate = {"version": 1, "rules": [guard, copied]}

    assert _validate_manifest(
        candidate,
        {"bounded/file.txt"},
        public_root=True,
        attested_source_ids={"bounded/file.txt"},
    ).keys() == {"bounded/file.txt"}
    with pytest.raises(ManifestError, match="rules match no tracked files"):
        _validate_manifest(candidate, {"bounded/file.txt"}, public_root=False)


def test_public_root_uses_attested_source_ids_to_satisfy_source_only_rules() -> None:
    transformed = {
        **_synthetic_exclude_rule("source-transform", "private/source.txt"),
        "disposition": "transform",
        "public_destination": "public/output.txt",
        "transformation": "synthetic_transform",
    }
    public_output = {
        **_synthetic_exclude_rule("public-output", "public/output.txt"),
        "disposition": "copy",
        "public_destination": "{source}",
    }
    candidate = {"version": 1, "rules": [transformed, public_output]}

    classified = _validate_manifest(
        candidate,
        {"public/output.txt"},
        public_root=True,
        attested_source_ids={"rule:source-transform"},
    )

    assert classified["public/output.txt"] == public_output


def test_hermes_is_guarded_and_never_tracked(manifest: dict[str, Any]) -> None:
    git_only = _git_tracked_files() - PENDING_CLASSIFICATION_FILES
    assert not any(path == ".hermes" or path.startswith(".hermes/") for path in git_only)
    hermes_rule = _resolve_rule(manifest["rules"], ".hermes/private-plan.md")
    assert hermes_rule["disposition"] == "exclude"


def test_unclassified_tracked_file_fails(manifest: dict[str, Any]) -> None:
    with pytest.raises(ManifestError, match="unclassified tracked file"):
        _validate_manifest(manifest, _git_tracked_files() | {"new-root-file.txt"})


def test_duplicate_destinations_fail(manifest: dict[str, Any]) -> None:
    duplicate = {
        "id": "synthetic-duplicate",
        "source": "duplicate-source.txt",
        "disposition": "copy",
        "public_destination": "README.md",
        "reason": "Exercise collision rejection.",
        "transformation": None,
        "owner": "release-engineering",
        "max_size_bytes": 1024,
        "generated": False,
        "precedence": 100,
    }
    candidate = {**manifest, "rules": [*manifest["rules"], duplicate]}
    with pytest.raises(ManifestError, match="duplicate public destination"):
        _validate_manifest(candidate, _git_tracked_files() | {"duplicate-source.txt"})


def test_excluded_path_in_simulated_output_fails(manifest: dict[str, Any]) -> None:
    classified = _validate_manifest(manifest, _git_tracked_files())
    allowed_output = {
        destination
        for source, rule in classified.items()
        if (destination := _destination(rule, source)) is not None
    }
    with pytest.raises(ManifestError, match="excluded paths appear in output"):
        _validate_simulated_output(
            allowed_output | {"models/example/plans/private.md"},
            manifest["rules"],
            classified,
        )


def test_complete_resolved_allowed_output_set_passes(manifest: dict[str, Any]) -> None:
    classified = _validate_manifest(manifest, _git_tracked_files())
    allowed_output = {
        destination
        for source, rule in classified.items()
        if (destination := _destination(rule, source)) is not None
    }
    _validate_simulated_output(allowed_output, manifest["rules"], classified)


def test_escaping_symlink_fails(tmp_path: Path) -> None:
    export_root = tmp_path / "export"
    export_root.mkdir()
    (export_root / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(ManifestError, match="symlink escapes export root"):
        _assert_no_escaping_symlinks(export_root)


def test_generated_rule_without_regeneration_check_fails(manifest: dict[str, Any]) -> None:
    malformed = dict(manifest["rules"][0])
    malformed["generated"] = True
    malformed.pop("regeneration_check", None)
    candidate = {**manifest, "rules": [malformed, *manifest["rules"][1:]]}
    with pytest.raises(ManifestError, match="need a regeneration check"):
        _validate_manifest(candidate, _git_tracked_files())


def test_ambiguous_overlapping_rules_fail() -> None:
    base = {
        "source": "models/**/results/*.json",
        "disposition": "exclude",
        "public_destination": None,
        "reason": "Synthetic broad policy.",
        "transformation": None,
        "owner": "model-release",
        "max_size_bytes": 1024,
        "generated": False,
        "precedence": 10,
    }
    peer = {**base, "id": "peer", "source": "models/example/results/*.json"}
    candidate = {"version": 1, "rules": [{**base, "id": "base"}, peer]}
    with pytest.raises(ManifestError, match="ambiguous rules"):
        _validate_manifest(candidate, {"models/example/results/summary.json"})


def test_intersecting_bounded_globs_fail_without_tracked_intersection() -> None:
    base = {
        "disposition": "exclude",
        "public_destination": None,
        "reason": "Synthetic bounded policy.",
        "transformation": None,
        "owner": "model-release",
        "max_size_bytes": 1024,
        "generated": False,
        "precedence": 10,
    }
    first = {**base, "id": "first", "source": "a/*/x"}
    second = {**base, "id": "second", "source": "a/b/*"}
    candidate = {"version": 1, "rules": [first, second]}

    with pytest.raises(ManifestError, match="ambiguous rules"):
        _validate_manifest(candidate, {"a/c/x", "a/b/y"})


def test_non_overlapping_bounded_globs_pass() -> None:
    base = {
        "disposition": "exclude",
        "public_destination": None,
        "reason": "Synthetic bounded policy.",
        "transformation": None,
        "owner": "model-release",
        "max_size_bytes": 1024,
        "generated": False,
        "precedence": 10,
    }
    first = {**base, "id": "first", "source": "a/*/x"}
    second = {**base, "id": "second", "source": "a/b/y"}
    candidate = {"version": 1, "rules": [first, second]}

    classified = _validate_manifest(candidate, {"a/c/x", "a/b/y"})

    assert set(classified) == {"a/c/x", "a/b/y"}


@pytest.mark.parametrize(
    ("left", "right", "overlap"),
    [
        ("a/?/x", "a/[bc]/x", True),
        ("a/[a-c]/x", "a/[d-f]/x", False),
        ("a/**/x", "a/b/c/x", True),
        ("a/*/x", "a/b/c/x", False),
        ("a/[!b]/x", "a/b/x", False),
    ],
)
def test_supported_glob_language_intersection(
    left: str, right: str, overlap: bool
) -> None:
    assert _glob_languages_overlap(left, right) is overlap


def test_exact_rule_intentionally_overrides_glob() -> None:
    broad = {
        "id": "broad",
        "source": "a/*/x",
        "disposition": "exclude",
        "public_destination": None,
        "reason": "Synthetic bounded policy.",
        "transformation": None,
        "owner": "model-release",
        "max_size_bytes": 1024,
        "generated": False,
        "precedence": 10,
    }
    exact = {
        **broad,
        "id": "exact",
        "source": "a/b/x",
        "disposition": "copy",
        "public_destination": "{source}",
        "reason": "Synthetic exact exception.",
        "precedence": 20,
        "resolves": ["broad"],
    }

    classified = _validate_manifest(
        {"version": 1, "rules": [broad, exact]},
        {"a/b/x"},
    )

    assert classified["a/b/x"]["id"] == "exact"


def test_explicit_precedence_resolves_overlapping_rules() -> None:
    broad = {
        "id": "broad",
        "source": "models/**/results/*.json",
        "disposition": "exclude",
        "public_destination": None,
        "reason": "Synthetic broad policy.",
        "transformation": None,
        "owner": "model-release",
        "max_size_bytes": 1024,
        "generated": False,
        "precedence": 10,
    }
    curated = {
        **broad,
        "id": "curated",
        "source": "models/example/results/summary.json",
        "disposition": "copy",
        "public_destination": "{source}",
        "reason": "Synthetic curated exception.",
        "precedence": 20,
        "resolves": ["broad"],
    }
    classified = _validate_manifest(
        {"version": 1, "rules": [broad, curated]},
        {"models/example/results/summary.json"},
    )
    assert classified["models/example/results/summary.json"]["id"] == "curated"


def test_manifest_sources_are_bounded(manifest: dict[str, Any]) -> None:
    assert all(_is_bounded_source(rule["source"]) for rule in manifest["rules"])


@pytest.mark.parametrize(
    "source",
    [
        "C:/outside.txt",
        r"\\server\share\outside.txt",
        r"docs\outside.txt",
        "/outside.txt",
        "./outside.txt",
        "../x",
        "docs/../x.txt",
        "docs//x.txt",
        "docs/./x.txt",
        "docs/",
    ],
)
def test_noncanonical_manifest_sources_fail_before_glob_compilation(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rule = _synthetic_exclude_rule("noncanonical-source", source)
    compiled = False
    original_parse_glob = _parse_glob

    def record_glob_compilation(pattern: str) -> tuple[GlobToken, ...]:
        nonlocal compiled
        compiled = True
        return original_parse_glob(pattern)

    monkeypatch.setattr(sys.modules[__name__], "_parse_glob", record_glob_compilation)

    with pytest.raises(
        ManifestError,
        match="source must be a canonical POSIX repository-relative pattern",
    ):
        _validate_manifest({"version": 1, "rules": [rule]}, set())
    assert compiled is False


@pytest.mark.parametrize(
    ("source", "concrete_source"),
    [
        ("docs/guide.txt", "docs/guide.txt"),
        ("docs/*.txt", "docs/guide.txt"),
        ("docs/**/guide?.txt", "docs/reference/guide1.txt"),
    ],
)
def test_canonical_exact_and_glob_sources_pass(
    source: str, concrete_source: str
) -> None:
    rule = _synthetic_exclude_rule("canonical-source", source)

    _validate_manifest(
        {"version": 1, "rules": [rule]},
        {concrete_source},
    )


def test_current_manifest_sources_use_canonical_repository_relative_syntax(
    manifest: dict[str, Any],
) -> None:
    _validate_manifest(manifest, _git_tracked_files())


@pytest.mark.parametrize(
    "concrete_source",
    [
        "docs//x.txt",
        "docs/./x.txt",
        "docs/../x",
        r"docs/x\outside.txt",
        "docs/x/",
    ],
)
def test_source_preserving_destination_validates_each_resolved_path(
    concrete_source: str,
) -> None:
    rule = {
        **_synthetic_exclude_rule("source-destination", "docs/**"),
        "disposition": "copy",
        "public_destination": "{source}",
    }

    with pytest.raises(
        ManifestError,
        match="resolved destination must be a canonical POSIX repository-relative path",
    ):
        _validate_manifest({"version": 1, "rules": [rule]}, {concrete_source})


def test_manifest_rule_overlaps_are_deterministic(manifest: dict[str, Any]) -> None:
    _validate_manifest(manifest, _git_tracked_files())


def test_repository_has_no_escaping_symlinks() -> None:
    _assert_no_escaping_symlinks(REPO_ROOT, _git_tracked_files())


def _synthetic_exclude_rule(rule_id: str, source: str) -> dict[str, Any]:
    return {
        "id": rule_id,
        "source": source,
        "disposition": "exclude",
        "public_destination": None,
        "reason": "Synthetic security regression.",
        "transformation": None,
        "owner": "release-engineering",
        "max_size_bytes": 1024,
        "generated": False,
        "precedence": 10,
        "allow_empty": True,
    }


def test_adjacent_equivalent_star_tokens_are_normalized() -> None:
    tokens = _parse_glob("bounded/********/file")

    assert tokens == (
        ("one", ((ord("b"), ord("b")),)),
        ("one", ((ord("o"), ord("o")),)),
        ("one", ((ord("u"), ord("u")),)),
        ("one", ((ord("n"), ord("n")),)),
        ("one", ((ord("d"), ord("d")),)),
        ("one", ((ord("e"), ord("e")),)),
        ("one", ((ord("d"), ord("d")),)),
        ("one", SLASH_CHARACTER),
        ("recursive-directory", ANY_CHARACTER),
        ("one", ((ord("f"), ord("f")),)),
        ("one", ((ord("i"), ord("i")),)),
        ("one", ((ord("l"), ord("l")),)),
        ("one", ((ord("e"), ord("e")),)),
    )


def test_repeated_stars_have_bounded_intersection_work() -> None:
    budget = _WorkBudget(2_000, "glob product")

    assert _glob_languages_overlap(
        "bounded/" + ("*" * 128) + "/x",
        "bounded/" + ("*" * 128) + "/y",
        budget,
    ) is False
    assert budget.used <= 2_000


def test_glob_product_budget_fails_deterministically() -> None:
    budget = _WorkBudget(1, "glob product")

    with pytest.raises(ManifestError, match="glob product work limit exceeded"):
        _glob_languages_overlap("bounded/**/x", "bounded/**/x", budget)


def test_rule_count_limit_precedes_pairwise_overlap_work() -> None:
    rules = [
        _synthetic_exclude_rule(f"rule-{index}", f"bounded/{index}.txt")
        for index in range(MAX_RULES + 1)
    ]

    with pytest.raises(ManifestError, match=f"manifest rule limit exceeded: {MAX_RULES}"):
        _validate_manifest({"version": 1, "rules": rules}, set())


def test_many_rules_share_one_bounded_product_work_budget() -> None:
    rules = [
        _synthetic_exclude_rule(f"rule-{index}", f"bounded/{index}.txt")
        for index in range(20)
    ]
    budget = _WorkBudget(10, "glob product")

    with pytest.raises(ManifestError, match="glob product work limit exceeded: 10"):
        _validate_rule_overlaps(rules, budget)
    assert budget.used == 11


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("bounded/" + ("a" * (MAX_PATTERN_LENGTH + 1)), "source pattern length limit"),
        ("bounded/" + ("?" * (MAX_GLOB_TOKENS + 1)), "source token limit"),
    ],
)
def test_glob_input_limits_fail_deterministically(source: str, message: str) -> None:
    with pytest.raises(ManifestError, match=message):
        _parse_glob(source)


def test_manifest_size_limit_fails_before_yaml_parse(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_bytes(b"[" * (MAX_MANIFEST_BYTES + 1))

    with pytest.raises(ManifestError, match=f"manifest size limit exceeded: {MAX_MANIFEST_BYTES}"):
        _load_manifest(path)


def test_explicit_member_with_outside_symlinked_ancestor_fails(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    repository.mkdir()
    outside.mkdir()
    (outside / "guide.md").write_text("private", encoding="utf-8")
    (repository / "docs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ManifestError, match="symlink component"):
        _assert_no_escaping_symlinks(repository, {"docs/guide.md"})


def test_explicit_member_with_nested_symlinked_ancestor_fails(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    actual = repository / "actual"
    actual.mkdir(parents=True)
    (actual / "guide.md").write_text("public", encoding="utf-8")
    nested = repository / "docs"
    nested.mkdir()
    (nested / "components").symlink_to(actual, target_is_directory=True)

    with pytest.raises(ManifestError, match="symlink component"):
        _assert_no_escaping_symlinks(repository, {"docs/components/guide.md"})


@pytest.mark.parametrize("leaf_target", ["real.txt", "missing.txt"])
def test_explicit_member_rejects_leaf_and_broken_symlinks(
    tmp_path: Path, leaf_target: str
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "real.txt").write_text("public", encoding="utf-8")
    (repository / "member.txt").symlink_to(leaf_target)

    with pytest.raises(ManifestError, match="symlink component"):
        _assert_no_escaping_symlinks(repository, {"member.txt"})


def test_missing_explicit_member_fails_safely(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    with pytest.raises(ManifestError, match="tracked member is missing"):
        _assert_no_escaping_symlinks(repository, {"docs/missing.md"})


@pytest.mark.parametrize(
    "destination",
    [
        "",
        r"docs\guide.md",
        r"C:\docs\guide.md",
        r"\\server\share\guide.md",
        "C:/docs/guide.md",
        "/docs/guide.md",
        "./docs/guide.md",
        "docs/../guide.md",
        "docs//guide.md",
        "docs/guide.md/",
        "docs/./guide.md",
    ],
)
def test_noncanonical_public_destinations_fail(destination: str) -> None:
    rule = {
        **_synthetic_exclude_rule("portable-destination", "bounded/file.txt"),
        "disposition": "copy",
        "public_destination": destination,
    }

    with pytest.raises(ManifestError, match="canonical POSIX repository-relative path"):
        _validate_manifest({"version": 1, "rules": [rule]}, {"bounded/file.txt"})


def test_canonical_public_destination_passes() -> None:
    rule = {
        **_synthetic_exclude_rule("portable-destination", "bounded/file.txt"),
        "disposition": "copy",
        "public_destination": "public/docs/guide.md",
    }

    classified = _validate_manifest(
        {"version": 1, "rules": [rule]}, {"bounded/file.txt"}
    )

    assert classified["bounded/file.txt"] == rule


def test_duplicate_yaml_keys_fail(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("version: 1\nversion: 1\nrules: []\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="duplicate YAML key: version"):
        _load_manifest(path)


def test_boolean_manifest_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("version: true\nrules:\n  - {}\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="manifest version must be 1"):
        _load_manifest(path)


@pytest.mark.parametrize("unknown", ["extra", "allow_empty"])
def test_unknown_top_level_manifest_fields_fail(
    tmp_path: Path, unknown: str
) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        f"version: 1\nrules:\n  - id: example\n{unknown}: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="unknown manifest fields"):
        _load_manifest(path)


def test_unknown_rule_fields_fail() -> None:
    rule = _synthetic_exclude_rule("unknown-field", "bounded/file.txt")
    rule["typo"] = True

    with pytest.raises(ManifestError, match="unknown fields"):
        _validate_manifest({"version": 1, "rules": [rule]}, {"bounded/file.txt"})


@pytest.mark.parametrize("field", ["max_size_bytes", "precedence"])
def test_boolean_is_not_accepted_as_an_integer(field: str) -> None:
    rule = _synthetic_exclude_rule("strict-integer", "bounded/file.txt")
    rule[field] = True

    with pytest.raises(ManifestError, match=f"{field} must be a positive integer"):
        _validate_manifest({"version": 1, "rules": [rule]}, {"bounded/file.txt"})


@pytest.mark.parametrize("field", ["generated", "allow_empty"])
def test_integer_is_not_accepted_as_a_boolean(field: str) -> None:
    rule = _synthetic_exclude_rule("strict-boolean", "bounded/file.txt")
    rule[field] = 1

    with pytest.raises(ManifestError, match=f"{field} must be boolean"):
        _validate_manifest({"version": 1, "rules": [rule]}, {"bounded/file.txt"})


@pytest.mark.parametrize(
    ("field", "value", "limit"),
    [
        ("max_size_bytes", 0, MAX_SIZE_BYTES),
        ("max_size_bytes", MAX_SIZE_BYTES + 1, MAX_SIZE_BYTES),
        ("precedence", 0, MAX_PRECEDENCE),
        ("precedence", MAX_PRECEDENCE + 1, MAX_PRECEDENCE),
    ],
)
def test_numeric_rule_fields_are_positive_and_bounded(
    field: str, value: int, limit: int
) -> None:
    rule = _synthetic_exclude_rule("bounded-integer", "bounded/file.txt")
    rule[field] = value

    with pytest.raises(
        ManifestError, match=f"{field} must be a positive integer no greater than {limit}"
    ):
        _validate_manifest({"version": 1, "rules": [rule]}, {"bounded/file.txt"})
