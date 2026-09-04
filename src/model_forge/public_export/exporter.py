"""Transactional deterministic public repository exporter."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast

import yaml

from model_forge.public_export.detectors import (
    RULE_RAW_BENCHMARK_KEY,
    DetectorError,
    DetectorPolicy,
    Finding,
    GitleaksStatus,
    ScanWorkBudget,
    scan_file,
)
from model_forge.public_export.glob_language import (
    GlobLanguageError,
    WorkBudget,
    glob_languages_overlap,
    parse_glob,
)
from model_forge.public_export.transforms import (
    TransformContext,
    TransformError,
    apply_transform,
    available_transforms,
)

EXPORTER_VERSION = "1.0.0"
ATTESTATION_SCHEMA = "model-forge-public-export/v1"
ATTESTATION_FILENAME = "PUBLIC_EXPORT_MANIFEST.json"
DETECTOR_VERSION = "public-export-detectors/v1"
DEFAULT_PUBLIC_CONTACT = "security@hangglidersrule.com"
MAX_MANIFEST_BYTES = 1_048_576
MAX_RULES = 256
MAX_PATTERN_LENGTH = 1_024
MAX_TRACKED_FILES = 100_000
MAX_TOTAL_SOURCE_BYTES = 1_073_741_824
MAX_GITLEAKS_VERSION_LENGTH = 128
MAX_GITLEAKS_OUTPUT_BYTES = 65_536
MAX_GITLEAKS_REPORT_BYTES = 16_777_216
GITLEAKS_TIMEOUT_SECONDS = 120
MAX_CAT_FILE_BATCH_OIDS = 256
MAX_CAT_FILE_BATCH_INPUT_BYTES = 16_384
MAX_CAT_FILE_BATCH_PAYLOAD_BYTES = 33_554_432
MAX_CAT_FILE_HEADER_BYTES = 128
CAT_FILE_BATCH_TIMEOUT_SECONDS = 120
CAT_FILE_TOTAL_TIMEOUT_SECONDS = 900
GIT_OBJECT_READER = "Git object reader"
GITLEAKS_SCOPE = "full-history-through-source-sha"
GITLEAKS_CONFIG = b"[extend]\nuseDefault = true\n"
GITLEAKS_CONFIG_SHA256 = hashlib.sha256(GITLEAKS_CONFIG).hexdigest()
SYSTEM_EXECUTABLE_PATH = os.pathsep.join(
    ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin")
)
GITLEAKS_FLAGS = (
    "git",
    "--redact",
    "--report-format=json",
    "--config=<exporter-owned>",
    "--gitleaks-ignore-path=<exporter-owned-empty>",
    "--log-opts=<source-sha>",
)
_SHA = re.compile(r"[0-9a-fA-F]{40}")
_CAT_FILE_SIZE = re.compile(rb"0|[1-9][0-9]{0,18}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GLOB_MAGIC = re.compile(r"[*?[]")
_SAFE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_ALLOWED_MANIFEST_FIELDS = {"version", "rules"}
_REQUIRED_RULE_FIELDS = {
    "source",
    "disposition",
    "public_destination",
    "reason",
    "transformation",
    "owner",
    "max_size_bytes",
    "generated",
}
_ALLOWED_RULE_FIELDS = _REQUIRED_RULE_FIELDS | {
    "id",
    "precedence",
    "resolves",
    "allow_empty",
    "content_classification",
    "detector_suppressions",
    "regeneration_check",
}
_APPROVED_DETECTOR_SUPPRESSIONS = frozenset({RULE_RAW_BENCHMARK_KEY})
_TRUSTED_CONTENT_CLASSIFICATIONS = frozenset(
    {"trusted-source-code", "trusted-detector-fixture"}
)
_REGULAR_GIT_MODES = frozenset({"100644", "100755"})


class ExportError(RuntimeError):
    """The requested public export failed closed."""


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
            raise ExportError("manifest YAML keys must be scalar") from error
        if duplicate:
            raise ExportError(f"duplicate manifest YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class ExportRequest:
    """Inputs required to establish one reproducible public export."""

    source: Path
    output: Path
    manifest: Path
    source_sha: str
    gitleaks_runner: GitleaksRunner | None = None
    replace: bool = False
    dry_run: bool = False
    public_contact: str = DEFAULT_PUBLIC_CONTACT
    fleet_hostnames: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Stable export identity and count."""

    payload_tree_sha256: str
    file_count: int
    output: Path
    dry_run: bool


@dataclass(frozen=True, slots=True)
class GitleaksEvidence:
    """Bounded evidence returned only by the exporter's trusted scanner boundary."""

    version: str
    report_sha256: str
    scope: str
    source_sha: str
    config_sha256: str = GITLEAKS_CONFIG_SHA256
    flags: tuple[str, ...] = GITLEAKS_FLAGS


class GitleaksRunner(Protocol):
    """Trusted injectable boundary used by tests and the built-in subprocess runner."""

    def scan_git(self, source: Path, source_sha: str) -> GitleaksEvidence:
        """Scan Git history through source_sha or fail closed."""


@dataclass(frozen=True, slots=True)
class _Rule:
    rule_id: str
    source: str
    disposition: Literal["copy", "transform", "exclude"]
    destination: str | None
    transformation: str | None
    max_size_bytes: int
    precedence: int
    resolves: frozenset[str]
    allow_empty: bool
    content_classification: str | None
    detector_suppressions: frozenset[str]
    pattern: re.Pattern[str]


@dataclass(frozen=True, slots=True)
class _TreeEntry:
    path: str
    oid: str
    digest: str
    mode: Literal["100644", "100755"]
    data: bytes


@dataclass(frozen=True, slots=True)
class _PlannedFile:
    destination: str
    source_path: str
    rule: _Rule
    data: bytes
    record: dict[str, object]


@dataclass(frozen=True, slots=True)
class _ExportPlan:
    files: tuple[_PlannedFile, ...]
    payload_tree_sha256: str


def _trusted_process_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": SYSTEM_EXECUTABLE_PATH,
        "XDG_CONFIG_HOME": "/var/empty",
    }


def _git(source: Path, *arguments: str, text: bool = True) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=source,
            env=_trusted_process_environment(),
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExportError(f"Git verification failed for {arguments[0]}") from error
    return cast(str | bytes, result.stdout)


def _canonical_relative(value: str) -> bool:
    if (
        not value
        or "\0" in value
        or "\\" in value
        or value.startswith("/")
        or re.match(r"^[A-Za-z]:", value)
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return False
    parts = value.split("/")
    return (
        all(part not in {"", ".", ".."} for part in parts)
        and PurePosixPath(value).as_posix() == value
    )


def _compile_glob(pattern: str) -> re.Pattern[str]:
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise ExportError(f"manifest source pattern exceeds {MAX_PATTERN_LENGTH} characters")
    if not _canonical_relative(pattern):
        raise ExportError("manifest source must be canonical repository-relative POSIX syntax")
    try:
        parse_glob(pattern)
    except GlobLanguageError as error:
        raise ExportError(str(error)) from error
    magic = _GLOB_MAGIC.search(pattern)
    if magic is not None and "/" not in pattern[: magic.start()]:
        raise ExportError(f"manifest source glob is not bounded: {pattern}")
    output: list[str] = ["^"]
    index = 0
    tokens = 0
    while index < len(pattern):
        tokens += 1
        if tokens > 256:
            raise ExportError("manifest source token limit exceeded")
        character = pattern[index]
        if character == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                index += 2
                while index < len(pattern) and pattern[index] == "*":
                    index += 1
                if index < len(pattern) and pattern[index] == "/":
                    output.append("(?:.*/)?")
                    index += 1
                else:
                    output.append(".*")
                continue
            output.append("[^/]*")
        elif character == "?":
            output.append("[^/]")
        elif character == "[":
            closing = pattern.find("]", index + 1)
            if closing < 0:
                raise ExportError(f"unterminated manifest character class: {pattern}")
            content = pattern[index + 1 : closing]
            if not content or "/" in content or "\\" in content:
                raise ExportError(f"invalid manifest character class: {pattern}")
            if content[0] in {"!", "^"}:
                content = "^" + content[1:]
            output.append("[" + content + "]")
            index = closing
        else:
            output.append(re.escape(character))
        index += 1
    output.append("$")
    try:
        return re.compile("".join(output))
    except re.error as error:
        raise ExportError(f"invalid manifest source pattern: {pattern}") from error


def _validate_static_overlaps(rules: list[_Rule]) -> None:
    budget = WorkBudget()
    for index, left in enumerate(rules):
        for right in rules[index + 1 :]:
            try:
                overlaps = glob_languages_overlap(left.source, right.source, budget)
            except GlobLanguageError as error:
                raise ExportError(str(error)) from error
            if not overlaps:
                continue
            if left.precedence == right.precedence:
                raise ExportError("ambiguous manifest rule source languages")
            winner, loser = (
                (left, right)
                if left.precedence > right.precedence
                else (right, left)
            )
            if loser.rule_id not in winner.resolves:
                raise ExportError("manifest source-language overlap is not explicitly resolved")


def _strict_bool(rule_id: str, field: str, value: object) -> bool:
    if type(value) is not bool:
        raise ExportError(f"{rule_id} {field} must be boolean")
    return bool(value)


def _load_manifest(raw: bytes) -> list[_Rule]:
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ExportError(f"manifest size limit exceeded: {MAX_MANIFEST_BYTES}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ExportError("manifest must be UTF-8") from error
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ExportError("invalid manifest YAML") from error
    if not isinstance(data, dict) or set(data) != _ALLOWED_MANIFEST_FIELDS:
        raise ExportError("manifest must contain exactly version and rules")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ExportError("manifest version must be 1")
    raw_rules = data["rules"]
    if not isinstance(raw_rules, list) or not raw_rules or len(raw_rules) > MAX_RULES:
        raise ExportError(f"manifest must contain 1..{MAX_RULES} rules")

    rule_ids: set[str] = set()
    rules: list[_Rule] = []
    implemented = available_transforms()
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ExportError("every manifest rule must be a mapping")
        unknown = set(raw_rule) - _ALLOWED_RULE_FIELDS
        missing = _REQUIRED_RULE_FIELDS - raw_rule.keys()
        if unknown or missing:
            raise ExportError("manifest rule has unknown or missing fields")
        rule_id = raw_rule.get("id")
        if not isinstance(rule_id, str) or not rule_id or rule_id in rule_ids:
            raise ExportError("manifest rule IDs must be non-empty and unique")
        rule_ids.add(rule_id)
        source = raw_rule["source"]
        if not isinstance(source, str):
            raise ExportError(f"{rule_id} source must be a string")
        disposition = raw_rule["disposition"]
        if disposition not in {"copy", "transform", "exclude"}:
            raise ExportError(f"{rule_id} has an invalid disposition")
        if not isinstance(raw_rule["reason"], str) or not raw_rule["reason"].strip():
            raise ExportError(f"{rule_id} must have a reason")
        if not isinstance(raw_rule["owner"], str) or not raw_rule["owner"].strip():
            raise ExportError(f"{rule_id} must have an owner")
        maximum = raw_rule["max_size_bytes"]
        if type(maximum) is not int or maximum <= 0 or maximum > MAX_TOTAL_SOURCE_BYTES:
            raise ExportError(f"{rule_id} max_size_bytes must be a positive bounded integer")
        _strict_bool(rule_id, "generated", raw_rule["generated"])
        allow_empty = _strict_bool(
            rule_id, "allow_empty", raw_rule.get("allow_empty", False)
        )
        if raw_rule["generated"] and not (
            isinstance(raw_rule.get("regeneration_check"), str)
            and raw_rule["regeneration_check"].strip()
        ):
            raise ExportError(f"{rule_id} generated entries need a regeneration check")
        precedence = raw_rule.get("precedence", 0)
        if type(precedence) is not int or precedence < 0 or precedence > 1_000_000:
            raise ExportError(f"{rule_id} precedence must be a bounded integer")
        if "precedence" in raw_rule and precedence == 0:
            raise ExportError(f"{rule_id} precedence must be positive when declared")
        resolves_raw = raw_rule.get("resolves", [])
        if not isinstance(resolves_raw, list) or not all(
            isinstance(item, str) and item for item in resolves_raw
        ):
            raise ExportError(f"{rule_id} resolves must contain rule IDs")
        regeneration_check = raw_rule.get("regeneration_check")
        if regeneration_check is not None and (
            not isinstance(regeneration_check, str) or not regeneration_check.strip()
        ):
            raise ExportError(f"{rule_id} regeneration_check must be non-empty")
        content_classification = raw_rule.get("content_classification")
        if content_classification is not None and (
            not isinstance(content_classification, str)
            or content_classification not in _TRUSTED_CONTENT_CLASSIFICATIONS
        ):
            raise ExportError(f"{rule_id} has an invalid content classification")
        suppressions_raw = raw_rule.get("detector_suppressions", [])
        if (
            not isinstance(suppressions_raw, list)
            or not all(isinstance(item, str) and item for item in suppressions_raw)
            or len(suppressions_raw) != len(set(suppressions_raw))
        ):
            raise ExportError(f"{rule_id} detector suppressions must be unique rule IDs")
        detector_suppressions = frozenset(suppressions_raw)
        if detector_suppressions:
            if not detector_suppressions <= _APPROVED_DETECTOR_SUPPRESSIONS:
                raise ExportError(f"{rule_id} has a forbidden detector suppression")
            if content_classification is None:
                raise ExportError(
                    f"{rule_id} detector suppression requires a content classification"
                )
        destination = raw_rule["public_destination"]
        transformation = raw_rule["transformation"]
        if disposition == "exclude":
            if destination is not None or transformation is not None:
                raise ExportError(f"{rule_id} excluded rules cannot emit or transform")
        else:
            if not isinstance(destination, str) or (
                destination != "{source}" and not _canonical_relative(destination)
            ):
                raise ExportError(f"{rule_id} has a noncanonical public destination")
            if _GLOB_MAGIC.search(source) and destination != "{source}":
                raise ExportError(f"{rule_id} glob destinations must preserve source")
            if disposition == "copy" and transformation is not None:
                raise ExportError(f"{rule_id} copy rules cannot transform")
            if disposition == "transform" and (
                not isinstance(transformation, str) or transformation not in implemented
            ):
                raise ExportError(f"{rule_id} requires an implemented transform")
        if detector_suppressions and (
            disposition != "copy" or destination != "{source}"
        ):
            raise ExportError(
                f"{rule_id} detector suppression requires unchanged copied bytes"
            )
        rules.append(
            _Rule(
                rule_id=rule_id,
                source=source,
                disposition=disposition,
                destination=destination,
                transformation=transformation,
                max_size_bytes=maximum,
                precedence=precedence,
                resolves=frozenset(resolves_raw),
                allow_empty=allow_empty,
                content_classification=content_classification,
                detector_suppressions=detector_suppressions,
                pattern=_compile_glob(source),
            )
        )
    unknown_resolutions = {
        resolved for rule in rules for resolved in rule.resolves if resolved not in rule_ids
    }
    if unknown_resolutions:
        raise ExportError("manifest resolves unknown rule IDs")
    _validate_static_overlaps(rules)
    return rules


def _tree_entries(source: Path, source_sha: str) -> dict[str, _TreeEntry]:
    raw = _git(
        source,
        "ls-tree",
        "-rz",
        "--full-tree",
        source_sha,
        text=False,
    )
    assert isinstance(raw, bytes)
    records = [part for part in raw.split(b"\0") if part]
    if len(records) > MAX_TRACKED_FILES:
        raise ExportError(f"tracked file limit exceeded: {MAX_TRACKED_FILES}")
    metadata: list[tuple[str, str, str, str]] = []
    try:
        for record in records:
            header, encoded_path = record.split(b"\t", 1)
            encoded_mode, encoded_type, encoded_oid = header.split(b" ", 2)
            metadata.append(
                (
                    encoded_mode.decode("ascii"),
                    encoded_type.decode("ascii"),
                    encoded_oid.decode("ascii"),
                    encoded_path.decode("utf-8"),
                )
            )
    except (UnicodeDecodeError, ValueError) as error:
        raise ExportError("asserted Git tree is malformed or has non-UTF-8 paths") from error

    paths = [item[3] for item in metadata]
    if len(paths) != len(set(paths)) or any(not _canonical_relative(path) for path in paths):
        raise ExportError("asserted Git tree has duplicate or noncanonical tracked paths")
    portable: dict[str, str] = {}
    for path in paths:
        previous = portable.setdefault(_portable_key(path), path)
        if previous != path:
            raise ExportError(f"portable tracked path collision: {previous}, {path}")
    for git_mode, git_object_type, _, path in metadata:
        if git_mode == "160000" or git_object_type == "commit":
            raise ExportError(f"tracked submodule is forbidden: {path}")
        if git_mode == "120000":
            raise ExportError(f"tracked symlink is forbidden: {path}")
        if git_mode not in _REGULAR_GIT_MODES or git_object_type != "blob":
            raise ExportError(f"unsupported tracked Git mode or object type: {path}")

    object_bytes = _read_git_blobs(source, [item[2] for item in metadata])
    entries: dict[str, _TreeEntry] = {}
    total_bytes = 0
    for (git_mode, _, oid, path), data in zip(metadata, object_bytes, strict=True):
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_SOURCE_BYTES:
            raise ExportError(f"source byte limit exceeded: {MAX_TOTAL_SOURCE_BYTES}")
        entries[path] = _TreeEntry(
            path=path,
            oid=oid,
            digest=hashlib.sha256(data).hexdigest(),
            mode=cast(Literal["100644", "100755"], git_mode),
            data=data,
        )
    return dict(sorted(entries.items()))


def _cat_file_request(object_ids: Sequence[str]) -> bytes:
    request = b"".join(oid.encode("ascii") + b"\n" for oid in object_ids)
    if len(request) > MAX_CAT_FILE_BATCH_INPUT_BYTES:
        raise ExportError(
            f"Git object reader batch input limit exceeded: {MAX_CAT_FILE_BATCH_INPUT_BYTES}"
        )
    return request


def _cat_file_response_bytes(oid: str, size: int) -> int:
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ExportError("Git object reader returned an invalid object size")
    try:
        header_bytes = len(f"{oid} blob {size}\n".encode("ascii"))
    except UnicodeEncodeError as error:
        raise ExportError("Git object reader received a malformed object ID") from error
    if header_bytes > MAX_CAT_FILE_HEADER_BYTES:
        raise ExportError("Git object reader response header limit exceeded")
    return header_bytes + size + 1


def _run_cat_file(
    source: Path,
    mode: str,
    object_ids: Sequence[str],
    *,
    stdout_limit: int,
    deadline: float,
) -> bytes:
    result = _bounded_process(
        ["git", "cat-file", mode],
        cwd=source,
        timeout=min(CAT_FILE_BATCH_TIMEOUT_SECONDS, deadline - time.monotonic()),
        tool=GIT_OBJECT_READER,
        payload=_cat_file_request(object_ids),
        stdout_limit=stdout_limit,
    )
    if result.returncode != 0:
        raise ExportError("Git object reader failed")
    return result.stdout


def _parse_blob_header(line: bytes, expected_oid: str) -> int:
    fields = line.split(b" ")
    if len(line) > MAX_CAT_FILE_HEADER_BYTES or len(fields) != 3:
        raise ExportError("Git object reader returned a malformed header")
    oid, object_type, encoded_size = fields
    if oid != expected_oid.encode("ascii") or object_type != b"blob":
        raise ExportError("Git object reader returned an unexpected object")
    if _CAT_FILE_SIZE.fullmatch(encoded_size) is None:
        raise ExportError("Git object reader returned a malformed header")
    return int(encoded_size)


def _read_git_blob_sizes(
    source: Path, object_ids: Sequence[str], deadline: float
) -> list[int]:
    """Declare every object's type and size before any content is buffered."""

    sizes: list[int] = []
    total_bytes = 0
    for start in range(0, len(object_ids), MAX_CAT_FILE_BATCH_OIDS):
        batch = object_ids[start : start + MAX_CAT_FILE_BATCH_OIDS]
        stdout = _run_cat_file(
            source,
            "--batch-check",
            batch,
            stdout_limit=len(batch) * MAX_CAT_FILE_HEADER_BYTES,
            deadline=deadline,
        )
        lines = stdout.split(b"\n")
        if len(lines) != len(batch) + 1 or lines[-1] != b"":
            raise ExportError("Git object reader returned a malformed header")
        for expected_oid, line in zip(batch, lines[:-1], strict=True):
            size = _parse_blob_header(line, expected_oid)
            total_bytes += size
            if total_bytes > MAX_TOTAL_SOURCE_BYTES:
                raise ExportError(f"source byte limit exceeded: {MAX_TOTAL_SOURCE_BYTES}")
            sizes.append(size)
    return sizes


def _plan_cat_file_batch(
    object_ids: Sequence[str], sizes: Sequence[int], start: int
) -> tuple[int, int]:
    """Return the exclusive end of the next batch and its exact response length."""

    if len(object_ids) != len(sizes):
        raise ExportError("Git object reader size declaration count mismatch")
    if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < len(object_ids):
        raise ExportError("Git object reader batch start is out of bounds")
    end = start
    expected_bytes = 0
    while end < len(object_ids) and end - start < MAX_CAT_FILE_BATCH_OIDS:
        response_bytes = _cat_file_response_bytes(object_ids[end], sizes[end])
        if response_bytes > MAX_CAT_FILE_BATCH_PAYLOAD_BYTES:
            raise ExportError(
                "Git object reader batch payload limit exceeded: "
                f"{MAX_CAT_FILE_BATCH_PAYLOAD_BYTES}"
            )
        if response_bytes > MAX_CAT_FILE_BATCH_PAYLOAD_BYTES - expected_bytes:
            break
        expected_bytes += response_bytes
        end += 1
    return end, expected_bytes


def _read_git_blobs(source: Path, object_ids: Sequence[str]) -> list[bytes]:
    """Read committed blobs in bounded batches that can never block on a full pipe."""

    if len(object_ids) > MAX_TRACKED_FILES:
        raise ExportError(f"tracked file limit exceeded: {MAX_TRACKED_FILES}")
    if any(_SHA.fullmatch(oid) is None for oid in object_ids):
        raise ExportError("Git object reader received a malformed object ID")
    if not object_ids:
        return []
    deadline = time.monotonic() + CAT_FILE_TOTAL_TIMEOUT_SECONDS
    sizes = _read_git_blob_sizes(source, object_ids, deadline)
    blobs: list[bytes] = []
    start = 0
    while start < len(object_ids):
        end, expected_bytes = _plan_cat_file_batch(object_ids, sizes, start)
        stdout = _run_cat_file(
            source,
            "--batch",
            object_ids[start:end],
            stdout_limit=expected_bytes,
            deadline=deadline,
        )
        if len(stdout) != expected_bytes:
            raise ExportError("Git object reader returned an unexpected response length")
        offset = 0
        for position in range(start, end):
            header = f"{object_ids[position]} blob {sizes[position]}\n".encode("ascii")
            if stdout[offset : offset + len(header)] != header:
                raise ExportError("Git object reader returned an unexpected object")
            offset += len(header)
            data = stdout[offset : offset + sizes[position]]
            offset += sizes[position]
            if len(data) != sizes[position] or stdout[offset : offset + 1] != b"\n":
                raise ExportError("Git object reader returned a truncated object")
            offset += 1
            blobs.append(data)
        start = end
    if len(blobs) != len(object_ids):
        raise ExportError("Git object reader returned an unexpected response count")
    return blobs


def _resolve_rules(rules: list[_Rule], paths: list[str]) -> dict[str, _Rule]:
    resolved: dict[str, _Rule] = {}
    matched: set[str] = set()
    for path in paths:
        candidates = [rule for rule in rules if rule.pattern.fullmatch(path)]
        if not candidates:
            raise ExportError(f"unclassified tracked file: {path}")
        candidates.sort(key=lambda item: item.precedence, reverse=True)
        winner = candidates[0]
        if len(candidates) > 1:
            if winner.precedence == candidates[1].precedence:
                raise ExportError(f"ambiguous manifest rules for tracked file: {path}")
            unresolved = {item.rule_id for item in candidates[1:]} - winner.resolves
            if unresolved:
                raise ExportError(f"manifest overlap is not explicitly resolved: {path}")
        resolved[path] = winner
        matched.update(item.rule_id for item in candidates)
    stale = {rule.rule_id for rule in rules if rule.rule_id not in matched and not rule.allow_empty}
    if stale:
        raise ExportError(f"manifest rules match no tracked files: {sorted(stale)}")
    return resolved


def _validate_detector_policy(path: str, rule: _Rule) -> None:
    if not rule.detector_suppressions:
        return
    classification = rule.content_classification
    if classification == "trusted-source-code":
        if (
            not path.startswith(("src/", "tests/"))
            or PurePosixPath(path).suffix not in {".py", ".pyi"}
        ):
            raise ExportError(
                f"{rule.rule_id} detector suppression is invalid for non-source path: {path}"
            )
    elif classification == "trusted-detector-fixture":
        if (
            _GLOB_MAGIC.search(rule.source)
            or not path.startswith("tests/public_export/fixtures/")
            or PurePosixPath(path).suffix != ".json"
        ):
            raise ExportError(
                f"{rule.rule_id} detector suppression is invalid for fixture path: {path}"
            )
    else:
        raise ExportError(f"{rule.rule_id} detector suppression classification is invalid")


def _portable_key(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


def _destinations(
    classified: dict[str, _Rule],
) -> dict[str, tuple[str, _Rule]]:
    destinations: dict[str, tuple[str, _Rule]] = {}
    portable: dict[str, str] = {}
    reserved_key = _portable_key(ATTESTATION_FILENAME)
    for source_path, rule in classified.items():
        if rule.destination is None:
            continue
        destination = source_path if rule.destination == "{source}" else rule.destination
        assert destination is not None
        if not _canonical_relative(destination):
            raise ExportError("resolved destination is not canonical")
        key = _portable_key(destination)
        if key == reserved_key:
            raise ExportError(f"public destination is reserved: {destination}")
        if destination in destinations:
            raise ExportError(f"public destination collision: {destination}")
        previous = portable.setdefault(key, destination)
        if previous != destination:
            raise ExportError(f"portable path collision: {previous}, {destination}")
        destinations[destination] = (source_path, rule)
    emitted = set(destinations)
    for path in emitted:
        parent = PurePosixPath(path).parent
        while parent != PurePosixPath("."):
            if parent.as_posix() in emitted:
                raise ExportError(f"public file/directory collision: {path}")
            parent = parent.parent
    return dict(sorted(destinations.items()))


def _validate_public_contact(value: str) -> None:
    try:
        DetectorPolicy(public_contacts=frozenset({value}))
    except DetectorError as error:
        raise ExportError("public contact must be a conventional explicitly public email") from error
    domain = value.rsplit("@", 1)[1].casefold()
    if domain.endswith((".internal", ".local", ".localhost", ".example")):
        raise ExportError("public contact must not use a private or reserved internal domain")


def _bounded_process(
    arguments: list[str],
    *,
    cwd: Path,
    timeout: float,
    tool: str = "Gitleaks",
    payload: bytes | None = None,
    stdout_limit: int = MAX_GITLEAKS_OUTPUT_BYTES,
) -> subprocess.CompletedProcess[bytes]:
    if timeout <= 0:
        raise ExportError(f"{tool} exceeded its total time budget")
    deadline = time.monotonic() + timeout
    environment = _trusted_process_environment()
    try:
        process = subprocess.Popen(
            arguments,
            cwd=cwd,
            env=environment,
            stdin=subprocess.PIPE if payload is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        raise ExportError(f"{tool} is unavailable") from error
    process_group = process.pid
    stdout = bytearray()
    stderr = bytearray()

    def drain(stream: Any, destination: bytearray, limit: int) -> None:
        try:
            while chunk := stream.read(8192):
                remaining = limit + 1 - len(destination)
                if remaining > 0:
                    destination.extend(chunk[:remaining])
        except (OSError, ValueError):
            pass

    def feed(stream: Any, data: bytes) -> None:
        try:
            stream.write(data)
            stream.flush()
        except (OSError, ValueError):
            pass
        finally:
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    assert process.stdout is not None
    assert process.stderr is not None
    threads = [
        threading.Thread(
            target=drain, args=(process.stdout, stdout, stdout_limit), daemon=True
        ),
        threading.Thread(
            target=drain, args=(process.stderr, stderr, MAX_GITLEAKS_OUTPUT_BYTES), daemon=True
        ),
    ]
    if payload is not None:
        assert process.stdin is not None
        threads.append(
            threading.Thread(target=feed, args=(process.stdin, payload), daemon=True)
        )
    started_threads: list[threading.Thread] = []

    def remaining(end: float) -> float:
        return max(0.0, end - time.monotonic())

    def join_until(end: float) -> bool:
        for thread in started_threads:
            thread.join(timeout=remaining(end))
        return all(not thread.is_alive() for thread in started_threads)

    def close_parent_pipes() -> None:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is None:
                continue
            try:
                raw_stream = getattr(stream, "raw", None)
                if raw_stream is not None:
                    raw_stream.close()
            except (OSError, ValueError):
                pass

    def terminate_and_reap() -> bool:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except OSError:
            pass
        close_parent_pipes()
        cleanup_deadline = time.monotonic() + 1
        try:
            process.wait(timeout=remaining(cleanup_deadline))
        except subprocess.TimeoutExpired:
            pass
        return join_until(cleanup_deadline) and process.poll() is not None

    try:
        for thread in threads:
            thread.start()
            started_threads.append(thread)
        try:
            returncode = process.wait(timeout=remaining(deadline))
        except subprocess.TimeoutExpired as error:
            raise ExportError(f"{tool} invocation timed out") from error
        if not join_until(deadline):
            raise ExportError(f"{tool} invocation timed out waiting for pipe EOF")
        if len(stdout) > stdout_limit or len(stderr) > MAX_GITLEAKS_OUTPUT_BYTES:
            raise ExportError(f"{tool} process output limit exceeded")
        return subprocess.CompletedProcess(arguments, returncode, bytes(stdout), bytes(stderr))
    except BaseException as error:
        if not terminate_and_reap():
            raise ExportError(f"{tool} process cleanup did not complete") from error
        raise


class SubprocessGitleaksRunner:
    """Run the fixed Gitleaks Git-history scan with bounded private outputs."""

    def scan_git(self, source: Path, source_sha: str) -> GitleaksEvidence:
        executable = shutil.which("gitleaks", path=SYSTEM_EXECUTABLE_PATH)
        if executable is None:
            raise ExportError("Gitleaks is unavailable")
        version_result = _bounded_process(
            [executable, "version"],
            cwd=source,
            timeout=10,
        )
        if version_result.returncode != 0:
            raise ExportError("Gitleaks version check failed")
        try:
            version = version_result.stdout.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise ExportError("Gitleaks version output is malformed") from error
        if (
            not version
            or len(version) > MAX_GITLEAKS_VERSION_LENGTH
            or _SAFE_VERSION.fullmatch(version) is None
        ):
            raise ExportError("Gitleaks version output is malformed")

        with tempfile.TemporaryDirectory(
            prefix=".model-forge-gitleaks-"
        ) as temporary:
            private = Path(temporary)
            private.chmod(0o700)
            report = private / "report.json"
            config = private / "gitleaks.toml"
            config.write_bytes(GITLEAKS_CONFIG)
            config.chmod(0o600)
            ignore = private / "gitleaks.ignore"
            ignore.write_bytes(b"")
            ignore.chmod(0o600)
            command = [
                executable,
                "git",
                "--redact",
                "--report-format",
                "json",
                "--report-path",
                str(report),
                "--config",
                str(config),
                "--gitleaks-ignore-path",
                str(ignore),
                f"--log-opts={source_sha}",
            ]
            report_descriptor: int | None = None
            try:
                result = _bounded_process(
                    command,
                    cwd=source,
                    timeout=GITLEAKS_TIMEOUT_SECONDS,
                )
                if result.returncode != 0:
                    raise ExportError("Gitleaks scan did not pass")
                report_descriptor = os.open(
                    report,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                descriptor_report = os.fstat(report_descriptor)
                current_report = report.lstat()
                if (
                    not stat.S_ISREG(current_report.st_mode)
                    or (current_report.st_dev, current_report.st_ino)
                    != (descriptor_report.st_dev, descriptor_report.st_ino)
                ):
                    raise ExportError("Gitleaks report identity changed")
                if descriptor_report.st_size > MAX_GITLEAKS_REPORT_BYTES:
                    raise ExportError("Gitleaks report size limit exceeded")
                os.lseek(report_descriptor, 0, os.SEEK_SET)
                report_bytes = os.read(
                    report_descriptor, MAX_GITLEAKS_REPORT_BYTES + 1
                )
                if len(report_bytes) > MAX_GITLEAKS_REPORT_BYTES:
                    raise ExportError("Gitleaks report size limit exceeded")
                findings = json.loads(report_bytes)
            except ExportError:
                raise
            except (OSError, json.JSONDecodeError) as error:
                raise ExportError("Gitleaks report is missing or malformed") from error
            finally:
                if report_descriptor is not None:
                    os.close(report_descriptor)
            if not isinstance(findings, list):
                raise ExportError("Gitleaks report is malformed")
            if findings:
                raise ExportError("Gitleaks reported credential findings")
        return GitleaksEvidence(
            version=version,
            report_sha256=hashlib.sha256(report_bytes).hexdigest(),
            scope=GITLEAKS_SCOPE,
            source_sha=source_sha,
            config_sha256=GITLEAKS_CONFIG_SHA256,
            flags=GITLEAKS_FLAGS,
        )


def _run_gitleaks(request: ExportRequest, source: Path, source_sha: str) -> GitleaksEvidence:
    runner = request.gitleaks_runner or SubprocessGitleaksRunner()
    try:
        evidence = runner.scan_git(source, source_sha)
    except ExportError:
        raise
    except Exception as error:
        raise ExportError("Gitleaks runner failed") from error
    if (
        not isinstance(evidence, GitleaksEvidence)
        or evidence.source_sha != source_sha
        or evidence.scope != GITLEAKS_SCOPE
        or _SHA256.fullmatch(evidence.report_sha256) is None
        or evidence.config_sha256 != GITLEAKS_CONFIG_SHA256
        or evidence.flags != GITLEAKS_FLAGS
        or len(evidence.version) > MAX_GITLEAKS_VERSION_LENGTH
        or _SAFE_VERSION.fullmatch(evidence.version) is None
    ):
        raise ExportError("Gitleaks evidence is malformed or does not bind the source")
    return evidence


def _verify_request(request: ExportRequest) -> tuple[Path, Path, str]:
    _validate_public_contact(request.public_contact)
    if _SHA.fullmatch(request.source_sha) is None:
        raise ExportError("source SHA must be exactly 40 hexadecimal characters")
    if request.source.is_symlink():
        raise ExportError("source repository must not be a symlink")
    try:
        source = request.source.resolve(strict=True)
    except OSError as error:
        raise ExportError("source repository does not exist") from error
    if not source.is_dir():
        raise ExportError("source repository must be a directory")
    head = str(_git(source, "rev-parse", "HEAD")).strip().lower()
    source_sha = request.source_sha.lower()
    if head != source_sha:
        raise ExportError("source SHA does not equal Git HEAD")
    dirty = str(_git(source, "status", "--porcelain", "--untracked-files=no"))
    if dirty:
        raise ExportError("dirty tracked source is refused")

    lexical_output = request.output.absolute()
    if lexical_output.is_symlink():
        raise ExportError("output must not be a symlink")
    source_resolved = source.resolve()
    output_resolved = lexical_output.resolve(strict=False)
    if output_resolved == source_resolved or output_resolved.is_relative_to(source_resolved):
        raise ExportError("source and output paths overlap")
    if source_resolved.is_relative_to(output_resolved):
        raise ExportError("source and output paths overlap")
    return source, output_resolved, source_sha


def _payload_tree_sha256(files: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda value: str(value["output_path"])):
        digest.update(str(item["output_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["mode"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(item["output_sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _plan_export(
    tree: dict[str, _TreeEntry],
    manifest_relative: str,
    source_sha: str,
    *,
    public_contact: str,
    fleet_hostnames: frozenset[str],
) -> _ExportPlan:
    """Purely derive exact public bytes and attestation records from committed inputs."""

    _validate_public_contact(public_contact)
    if manifest_relative not in tree:
        raise ExportError("manifest must be tracked by the source repository")
    rules = _load_manifest(tree[manifest_relative].data)
    paths = list(tree)
    classified = _resolve_rules(rules, paths)
    for path, rule in classified.items():
        _validate_detector_policy(path, rule)
        if len(tree[path].data) > rule.max_size_bytes:
            raise ExportError(f"{path} exceeds manifest maximum size")
    destinations = _destinations(classified)
    public_paths = frozenset(destinations)

    files: list[_PlannedFile] = []
    records: list[dict[str, object]] = []
    for destination, (source_path, rule) in destinations.items():
        entry = tree[source_path]
        data = entry.data
        transform_id: str | None = None
        semantic_source: str | None = None
        semantic_output: str | None = None
        if rule.disposition == "transform":
            assert rule.transformation is not None
            try:
                transformed = apply_transform(
                    rule.transformation,
                    data,
                    TransformContext(
                        source_path=source_path,
                        source_sha=source_sha,
                        public_contact=public_contact,
                        fleet_hostnames=fleet_hostnames,
                        public_paths=public_paths,
                    ),
                )
            except TransformError as error:
                raise ExportError(f"transform failed for {source_path}: {error}") from error
            data = transformed.data
            transform_id = transformed.transform_id
            semantic_source = transformed.semantic_source_sha256
            semantic_output = transformed.semantic_output_sha256
        mode = 0o755 if entry.mode == "100755" else 0o644
        record: dict[str, object] = {
            "source_id": (
                destination if source_path == destination else f"rule:{rule.rule_id}"
            ),
            "output_path": destination,
            "input_sha256": entry.digest,
            "output_sha256": hashlib.sha256(data).hexdigest(),
            "transform_id": transform_id,
            "mode": f"{mode:06o}",
        }
        if semantic_source is not None:
            record["semantic_recipe"] = {
                "source_sha256": semantic_source,
                "output_sha256": semantic_output,
                "identity_preserved": semantic_source == semantic_output,
            }
        records.append(record)
        files.append(_PlannedFile(destination, source_path, rule, data, record))
    return _ExportPlan(tuple(files), _payload_tree_sha256(records))


def _remove_owned_tree(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _regular_file_inventory(root: Path) -> set[str]:
    """Return the exact regular-file inventory or reject unsafe staging members."""

    files: set[str] = set()
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise ExportError("public export staging tree cannot be inventoried") from error
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ExportError("public export staging member cannot be inspected") from error
            if stat.S_ISDIR(metadata.st_mode):
                stack.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                files.add(relative)
            else:
                raise ExportError(f"public export staging member is not regular: {relative}")
    return files


def _require_exact_inventory(root: Path, expected: set[str]) -> None:
    actual = _regular_file_inventory(root)
    missing = expected - actual
    unaccounted = actual - expected
    if missing or unaccounted:
        raise ExportError(
            "public export staging inventory mismatch: "
            f"missing={sorted(missing)}, unaccounted={sorted(unaccounted)}"
        )


def _scan_or_raise(
    root: Path,
    relative_path: str,
    *,
    policy: DetectorPolicy,
    fleet_hostnames: frozenset[str],
    work_budget: ScanWorkBudget,
) -> list[Finding]:
    try:
        return scan_file(
            root,
            relative_path,
            policy=policy,
            fleet_hostnames=fleet_hostnames,
            gitleaks_status=GitleaksStatus.PASSED,
            work_budget=work_budget,
        )
    except Exception as error:
        raise ExportError(f"detector failed for {relative_path}") from error


def _promote(staging: Path, output: Path, *, replace: bool) -> None:
    backup = output.parent / f".{output.name}.public-export-backup-{secrets.token_hex(8)}"
    moved_existing = False
    try:
        if output.exists():
            if any(output.iterdir()) and not replace:
                raise ExportError("non-empty output requires --replace")
            os.replace(output, backup)
            moved_existing = True
        os.replace(staging, output)
    except ExportError:
        raise
    except OSError as error:
        if moved_existing and backup.exists() and not output.exists():
            try:
                os.replace(backup, output)
            except OSError as rollback_error:
                raise ExportError("promotion failed and rollback failed") from rollback_error
        raise ExportError("public export promotion failed") from error
    if moved_existing:
        _remove_owned_tree(backup)


def export_public(request: ExportRequest) -> ExportResult:
    """Build, scan, attest, and atomically promote a public tree."""

    source, output, source_sha = _verify_request(request)
    if output.exists():
        if not output.is_dir():
            raise ExportError("output must be a directory path")
        if any(output.iterdir()) and not request.replace:
            raise ExportError("non-empty output requires --replace")
    manifest = request.manifest.absolute()
    try:
        manifest_relative = manifest.relative_to(source).as_posix()
    except ValueError as error:
        raise ExportError("manifest must select a file within the source repository") from error
    if not _canonical_relative(manifest_relative):
        raise ExportError("manifest selector must be a canonical repository-relative path")
    tree = _tree_entries(source, source_sha)
    plan = _plan_export(
        tree,
        manifest_relative,
        source_sha,
        public_contact=request.public_contact,
        fleet_hostnames=request.fleet_hostnames,
    )
    gitleaks_evidence = _run_gitleaks(request, source, source_sha)

    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.public-export.lock"
    staging = output.parent / f".{output.name}.public-export-stage-{secrets.token_hex(8)}"
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ExportError("another public export is already in progress") from error
    try:
        staging.mkdir(mode=0o700)
        records: list[dict[str, object]] = []
        destination_rules: dict[str, _Rule] = {}
        for planned in plan.files:
            destination = planned.destination
            rule = planned.rule
            target = staging / destination
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            for directory in (target.parent, *target.parent.parents):
                if directory == staging.parent:
                    break
                directory.chmod(0o755)
            target.write_bytes(planned.data)
            mode = int(cast(str, planned.record["mode"]), 8)
            target.chmod(mode)
            records.append(planned.record)
            destination_rules[destination] = rule

        _require_exact_inventory(
            staging, {str(record["output_path"]) for record in records}
        )

        try:
            policy = DetectorPolicy(public_contacts=frozenset({request.public_contact}))
            budget = ScanWorkBudget()
        except DetectorError as error:
            raise ExportError("detector policy initialization failed") from error
        for record in records:
            destination = str(record["output_path"])
            findings = _scan_or_raise(
                staging,
                destination,
                policy=policy,
                fleet_hostnames=request.fleet_hostnames,
                work_budget=budget,
            )
            suppressions = destination_rules[destination].detector_suppressions
            findings = [
                finding for finding in findings if finding.rule_id not in suppressions
            ]
            if findings:
                rules_found = sorted({finding.rule_id for finding in findings})
                raise ExportError(
                    f"detector findings for {destination}: {', '.join(rules_found)}"
                )

        digest = plan.payload_tree_sha256
        attestation = {
            "schema": ATTESTATION_SCHEMA,
            "schema_version": 1,
            "source_sha": source_sha,
            "exporter_version": EXPORTER_VERSION,
            "detector_versions": {"metadata": DETECTOR_VERSION},
            "gitleaks": {
                "tool": "gitleaks",
                "version": gitleaks_evidence.version,
                "status": "passed",
                "report_sha256": gitleaks_evidence.report_sha256,
                "scope": gitleaks_evidence.scope,
                "source_sha": gitleaks_evidence.source_sha,
                "config_sha256": gitleaks_evidence.config_sha256,
                "flags": list(gitleaks_evidence.flags),
            },
            "files": records,
            "payload_tree_sha256": digest,
        }
        attestation_path = staging / ATTESTATION_FILENAME
        attestation_path.write_text(
            json.dumps(attestation, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        attestation_path.chmod(0o644)
        _require_exact_inventory(
            staging,
            {str(record["output_path"]) for record in records}
            | {ATTESTATION_FILENAME},
        )
        findings = _scan_or_raise(
            staging,
            ATTESTATION_FILENAME,
            policy=policy,
            fleet_hostnames=request.fleet_hostnames,
            work_budget=budget,
        )
        if findings:
            raise ExportError("detector findings in public export attestation")

        current_head = str(_git(source, "rev-parse", "HEAD")).strip().lower()
        if current_head != source_sha:
            raise ExportError("Git HEAD changed during export")

        if request.dry_run:
            return ExportResult(digest, len(records), output, True)
        staging.chmod(0o755)
        _promote(staging, output, replace=request.replace)
        return ExportResult(digest, len(records), output, False)
    finally:
        _remove_owned_tree(staging)
        _remove_owned_tree(lock)
