"""Deterministic, read-only pull-request path risk classification."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

Risk = Literal["low", "medium", "high"]
EventKind = Literal["push", "pull_request"]
RISK_ORDER: dict[Risk, int] = {"low": 0, "medium": 1, "high": 2}
MAX_EVENT_BYTES = 2_000_000
MAX_TREE_BYTES = 1000 * (4096 + 1)
SHA = re.compile(r"[0-9a-fA-F]{40,64}")

DEFAULT_CONFIG: dict[str, object] = {
    "version": 1,
    "max_files": 1000,
    "max_path_bytes": 4096,
    "max_path_depth": 64,
    "rules": [
        {
            "id": "workflows",
            "risk": "high",
            "exact": [],
            "prefixes": [".github/workflows/"],
        },
        {
            "id": "security",
            "risk": "high",
            "exact": [".gitleaks.toml", "SECURITY.md"],
            "prefixes": ["security/"],
        },
        {
            "id": "export-tooling",
            "risk": "high",
            "exact": [
                "scripts/bootstrap_public_export_wheelhouse.py",
                "scripts/verify_public_export.sh",
            ],
            "prefixes": ["src/model_forge/public_export/", "tools/public_export/"],
        },
        {
            "id": "release-tooling",
            "risk": "high",
            "exact": ["src/model_forge/release.py"],
            "prefixes": ["scripts/release/", "tools/release/"],
        },
        {
            "id": "governance",
            "risk": "low",
            "exact": [
                ".github/CODEOWNERS",
                ".github/pull_request_template.md",
                "CODE_OF_CONDUCT.md",
                "CONTRIBUTING.md",
                "GOVERNANCE.md",
                "LICENSE",
                "README.md",
                "SUPPORT.md",
            ],
            "prefixes": [".github/ISSUE_TEMPLATE/", "docs/"],
        },
        {
            "id": "tests",
            "risk": "low",
            "exact": [],
            "prefixes": ["tests/"],
        },
        {
            "id": "framework",
            "risk": "medium",
            "exact": ["pyproject.toml"],
            "prefixes": ["src/"],
        },
        {
            "id": "recipes-contracts-containers",
            "risk": "medium",
            "exact": [],
            "prefixes": ["configs/", "containers/", "contracts/", "recipes/"],
        },
    ],
}


class RiskClassificationError(ValueError):
    """The classifier input or configuration is unsafe or unsupported."""


@dataclass(frozen=True)
class Rule:
    id: str
    risk: Risk
    exact: frozenset[str]
    prefixes: tuple[str, ...]

    def matches(self, path: str) -> bool:
        folded_path = path.casefold()
        return folded_path in self.exact or folded_path.startswith(self.prefixes)


@dataclass(frozen=True)
class Config:
    max_files: int
    max_path_bytes: int
    max_path_depth: int
    rules: tuple[Rule, ...]


def _positive_int(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise RiskClassificationError(f"{name} must be an integer from 1 to {maximum}")
    return value


def load_config(raw: Mapping[str, object] | None = None) -> Config:
    """Validate a bounded versioned configuration, rejecting unknown fields."""

    candidate = DEFAULT_CONFIG if raw is None else raw
    expected = {"version", "max_files", "max_path_bytes", "max_path_depth", "rules"}
    if set(candidate) != expected:
        raise RiskClassificationError("config must contain exactly the supported fields")
    if type(candidate["version"]) is not int or candidate["version"] != 1:
        raise RiskClassificationError("config version must be 1")
    max_files = _positive_int(candidate["max_files"], "max_files", 10_000)
    max_path_bytes = _positive_int(candidate["max_path_bytes"], "max_path_bytes", 16_384)
    max_path_depth = _positive_int(candidate["max_path_depth"], "max_path_depth", 256)
    raw_rules = candidate["rules"]
    if not isinstance(raw_rules, list) or not raw_rules or len(raw_rules) > 64:
        raise RiskClassificationError("rules must be a non-empty list of at most 64 entries")

    rules: list[Rule] = []
    ids: set[str] = set()
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, Mapping):
            raise RiskClassificationError("every rule must be a mapping")
        if set(raw_rule) != {"id", "risk", "exact", "prefixes"}:
            raise RiskClassificationError("rule must contain exactly id, risk, exact, and prefixes")
        rule_id = raw_rule["id"]
        risk = raw_rule["risk"]
        exact = raw_rule["exact"]
        prefixes = raw_rule["prefixes"]
        if not isinstance(rule_id, str) or not rule_id or rule_id in ids:
            raise RiskClassificationError("rule ids must be non-empty and unique")
        if risk not in RISK_ORDER:
            raise RiskClassificationError(f"rule {rule_id} has an unknown risk")
        if (
            not isinstance(exact, list)
            or not isinstance(prefixes, list)
            or not all(isinstance(item, str) and item for item in [*exact, *prefixes])
        ):
            raise RiskClassificationError(f"rule {rule_id} paths must be non-empty strings")
        exact_paths = cast(list[str], exact)
        prefix_paths = cast(list[str], prefixes)
        folded_exact = [path.casefold() for path in exact_paths]
        folded_prefixes = [path.casefold() for path in prefix_paths]
        if (
            len(folded_exact) != len(set(folded_exact))
            or len(folded_prefixes) != len(set(folded_prefixes))
        ):
            raise RiskClassificationError(f"rule {rule_id} paths must be unique")
        ids.add(rule_id)
        rules.append(
            Rule(
                id=rule_id,
                risk=risk,
                exact=frozenset(folded_exact),
                prefixes=tuple(folded_prefixes),
            )
        )
    return Config(max_files, max_path_bytes, max_path_depth, tuple(rules))


def _validate_path(path: str, config: Config) -> None:
    try:
        encoded = path.encode("utf-8")
    except UnicodeEncodeError as error:
        raise RiskClassificationError("changed paths must be valid UTF-8") from error
    if not encoded or len(encoded) > config.max_path_bytes:
        raise RiskClassificationError("changed path length is outside the configured bound")
    if "\\" in path or path.startswith("/") or any(ord(character) < 32 for character in path):
        raise RiskClassificationError("changed path is not a canonical repository-relative path")
    parts = path.split("/")
    if (
        len(parts) > config.max_path_depth
        or any(part in {"", ".", ".."} for part in parts)
        or PurePosixPath(path).as_posix() != path
    ):
        raise RiskClassificationError("changed path is not a bounded repository-relative path")


def classify_paths(
    paths: Iterable[str], config: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Classify paths, defaulting ordinary unmatched repository files to medium."""

    validated_config = load_config(config)
    materialized = list(paths)
    if len(materialized) > validated_config.max_files:
        raise RiskClassificationError(
            f"changed file limit exceeded: {validated_config.max_files}"
        )

    collision_keys: dict[str, str] = {}
    seen_paths: set[str] = set()
    classified: list[dict[str, str]] = []
    for path in materialized:
        if not isinstance(path, str):
            raise RiskClassificationError("changed paths must be strings")
        _validate_path(path, validated_config)
        collision_key = unicodedata.normalize("NFC", path).casefold()
        previous = collision_keys.setdefault(collision_key, path)
        if previous != path or path in seen_paths:
            raise RiskClassificationError("changed paths contain a Unicode or case collision")
        seen_paths.add(path)

        matching_rule = next(
            (rule for rule in validated_config.rules if rule.matches(path)),
            None,
        )
        classified.append(
            {
                "path": path,
                "risk": matching_rule.risk if matching_rule else "medium",
                "rule": matching_rule.id if matching_rule else "default-medium",
            }
        )

    classified.sort(key=lambda item: item["path"].encode("utf-8"))
    overall: Risk = "low"
    for item in classified:
        item_risk = cast(Risk, item["risk"])
        if RISK_ORDER[item_risk] > RISK_ORDER[overall]:
            overall = item_risk
    counts = {
        risk: sum(item["risk"] == risk for item in classified)
        for risk in ("low", "medium", "high")
    }
    return {
        "schema_version": 1,
        "risk": overall,
        "file_count": len(classified),
        "counts": counts,
        "files": classified,
    }


def stable_json(result: Mapping[str, object]) -> str:
    return json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"


def human_summary(result: Mapping[str, object]) -> str:
    counts = result["counts"]
    assert isinstance(counts, Mapping)
    return (
        "## Pull request path risk\n\n"
        f"Overall risk: **{result['risk']}**\n\n"
        f"Changed files: {result['file_count']} "
        f"(low: {counts['low']}, medium: {counts['medium']}, high: {counts['high']}).\n\n"
        "This is review guidance only; a high classification does not fail CI by itself.\n"
    )


def _load_event(path: Path) -> Mapping[str, Any]:
    if path.stat().st_size > MAX_EVENT_BYTES:
        raise RiskClassificationError("GitHub event payload exceeds the size limit")
    try:
        event = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RiskClassificationError("GitHub event payload is invalid") from error
    if not isinstance(event, Mapping):
        raise RiskClassificationError("GitHub event payload must be an object")
    return event


def _event_range(event: Mapping[str, Any]) -> tuple[EventKind, str | None, str]:
    pull_request = event.get("pull_request")
    kind: EventKind = "push"
    if isinstance(pull_request, Mapping):
        kind = "pull_request"
        base = pull_request.get("base")
        head = pull_request.get("head")
        if isinstance(base, Mapping) and isinstance(head, Mapping):
            before, after = base.get("sha"), head.get("sha")
        else:
            before, after = None, None
    else:
        before, after = event.get("before"), event.get("after")
    if not isinstance(after, str) or not SHA.fullmatch(after):
        raise RiskClassificationError("event does not contain a valid head commit")
    if not isinstance(before, str) or not SHA.fullmatch(before):
        raise RiskClassificationError("event does not contain a valid base commit")
    if set(before) == {"0"}:
        return kind, None, after
    return kind, before, after


def _tree_paths(after: str, repository: Path) -> list[str]:
    command = (
        "git",
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        "--full-tree",
        after,
    )
    try:
        process = subprocess.Popen(
            command,
            cwd=repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert process.stdout is not None
        output = process.stdout.read(MAX_TREE_BYTES + 1)
        if len(output) > MAX_TREE_BYTES:
            process.kill()
            process.wait()
            raise RiskClassificationError("Git tree path output exceeds the size limit")
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
        return [raw.decode("utf-8") for raw in output.split(b"\0") if raw]
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as error:
        raise RiskClassificationError("unable to obtain changed paths from Git") from error


def _diff_paths(before: str, after: str, repository: Path) -> list[str]:
    command = (
        "git",
        "diff",
        "--name-only",
        "--no-renames",
        "--diff-filter=ACDMRTUXB",
        "-z",
        before,
        after,
    )
    try:
        completed = subprocess.run(command, cwd=repository, check=True, capture_output=True)
        return [raw.decode("utf-8") for raw in completed.stdout.split(b"\0") if raw]
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as error:
        raise RiskClassificationError("unable to obtain changed paths from Git") from error


def _git_succeeds(command: Sequence[str], repository: Path) -> bool:
    try:
        completed = subprocess.run(
            command,
            cwd=repository,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        raise RiskClassificationError("unable to inspect the Git repository") from error
    return completed.returncode == 0


def _commit_exists(commit: str, repository: Path) -> bool:
    return _git_succeeds(("git", "cat-file", "-e", f"{commit}^{{commit}}"), repository)


def _push_before_is_diffable(before: str, after: str, repository: Path) -> bool:
    """Report whether a pushed before is a local commit that after descends from."""

    if not _commit_exists(before, repository):
        return False
    return _git_succeeds(("git", "merge-base", "--is-ancestor", before, after), repository)


def changed_paths(event_path: Path, repository: Path) -> list[str]:
    kind, before, after = _event_range(_load_event(event_path))
    if kind == "pull_request":
        # A pull request compares two tips, so a diverged base that no longer
        # precedes the head is still a bounded, meaningful comparison.
        if before is None or not (
            _commit_exists(before, repository) and _commit_exists(after, repository)
        ):
            raise RiskClassificationError("pull request base or head commit is unavailable")
        return _diff_paths(before, after, repository)
    if before is None or not _push_before_is_diffable(before, after, repository):
        return _tree_paths(after, repository)
    return _diff_paths(before, after, repository)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = classify_paths(changed_paths(args.event, args.repository))
        args.json.write_text(stable_json(result), encoding="utf-8")
        args.summary.write_text(human_summary(result), encoding="utf-8")
    except (OSError, RiskClassificationError) as error:
        parser.exit(2, f"PR risk classification failed: {error}\n")
    print(human_summary(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
