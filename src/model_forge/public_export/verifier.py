"""Independent, fail-closed verification of an already exported public tree."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import unicodedata
import urllib.parse
import venv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

import yaml
from packaging.requirements import InvalidRequirement, Requirement
from pydantic import ValidationError

from model_forge.models import load_spec
from model_forge.public_export.detectors import (
    DetectorError,
    DetectorPolicy,
    GitleaksStatus,
    ScanWorkBudget,
    scan_file,
)
from model_forge.public_export.exporter import (
    ATTESTATION_FILENAME,
    ATTESTATION_SCHEMA,
    DEFAULT_PUBLIC_CONTACT,
    DETECTOR_VERSION,
    EXPORTER_VERSION,
    GITLEAKS_CONFIG,
    GITLEAKS_CONFIG_SHA256,
    GITLEAKS_FLAGS,
    GITLEAKS_SCOPE,
    ExportError,
    GitleaksEvidence,
    GitleaksRunner,
    SubprocessGitleaksRunner,
    _plan_export,
    _tree_entries,
)
from model_forge.recipe import RecipeError, load_recipe

MAX_ATTESTATION_BYTES = 16_777_216
MAX_FILES = 100_000
MAX_FILE_BYTES = 1_048_576
MAX_TOTAL_BYTES = 1_073_741_824
MAX_PROCESS_OUTPUT_BYTES = 65_536
MAX_PACKAGE_ERROR_BYTES = 8_192
MAX_GITLEAKS_REPORT_BYTES = 16_777_216
MAX_MARKDOWN_LINKS = 10_000
MAX_MARKDOWN_NESTING = 32
PROCESS_TIMEOUT_SECONDS = 120
SYSTEM_EXECUTABLE_PATH = os.pathsep.join(
    ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin")
)
STRICT_GITLEAKS_POLICY = GITLEAKS_CONFIG
_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_ATTESTATION_FIELDS = {
    "schema",
    "schema_version",
    "source_sha",
    "exporter_version",
    "detector_versions",
    "gitleaks",
    "files",
    "payload_tree_sha256",
}
_GITLEAKS_FIELDS = {
    "tool",
    "version",
    "status",
    "report_sha256",
    "scope",
    "source_sha",
    "config_sha256",
    "flags",
}
_FILE_FIELDS = {
    "source_id",
    "output_path",
    "input_sha256",
    "output_sha256",
    "transform_id",
    "mode",
}
_SEMANTIC_FIELDS = {"source_sha256", "output_sha256", "identity_preserved"}
_RESERVED_COMPONENTS = {
    ".git",
    ".hermes",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "private",
    "raw",
}


class PublicVerifyError(RuntimeError):
    """An exported tree does not satisfy the trusted public policy."""


@dataclass(frozen=True, slots=True)
class FilesystemGitleaksEvidence:
    """Bounded evidence from the verifier-owned filesystem scan."""

    version: str
    report_sha256: str


class FilesystemGitleaksRunner(Protocol):
    def scan_directory(self, root: Path) -> FilesystemGitleaksEvidence:
        """Scan the filesystem with trusted config and empty ignore semantics."""


class ProjectChecksRunner(Protocol):
    def run(self, root: Path) -> None:
        """Run deterministic tests, lint, and type checks in the exported tree."""


@dataclass(frozen=True, slots=True)
class PublicVerifyRequest:
    root: Path
    source_sha: str
    source_repo: Path
    manifest: Path
    wheelhouse: Path
    gitleaks_runner: FilesystemGitleaksRunner | None = None
    source_gitleaks_runner: GitleaksRunner | None = None
    project_checks_runner: ProjectChecksRunner | None = None
    package_smoke: bool = True
    public_contact: str = DEFAULT_PUBLIC_CONTACT
    fleet_hostnames: frozenset[str] = frozenset()
    wheelhouse_lock: Path | None = None


@dataclass(frozen=True, slots=True)
class WheelEvidence:
    name: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class PublicVerifyResult:
    root: Path
    source_sha: str
    payload_tree_sha256: str
    file_count: int
    gitleaks_version: str
    wheelhouse_evidence: tuple[WheelEvidence, ...]


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
            raise PublicVerifyError("YAML mapping keys must be scalar") from error
        if duplicate:
            raise PublicVerifyError("duplicate YAML key")
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
            raise PublicVerifyError("duplicate JSON key")
        result[key] = value
    return result


def _load_json(raw: bytes, label: str) -> object:
    try:
        text = raw.decode("utf-8")
        return json.loads(text, object_pairs_hook=_unique_json_object)
    except PublicVerifyError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicVerifyError(f"invalid JSON: {label}") from error


def _canonical_relative(value: str) -> bool:
    if (
        not value
        or "\0" in value
        or "\\" in value
        or value.startswith("/")
        or unicodedata.normalize("NFC", value) != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return False
    parts = value.split("/")
    return (
        all(part not in {"", ".", ".."} for part in parts)
        and PurePosixPath(value).as_posix() == value
    )


def _portable_identity(path: str) -> str:
    return unicodedata.normalize("NFKC", path).casefold()


def _exact_mapping(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise PublicVerifyError(f"{label} has unknown or missing fields")
    return cast(dict[str, object], value)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _validate_gitleaks_attestation(value: object, source_sha: str) -> None:
    data = _exact_mapping(value, _GITLEAKS_FIELDS, "Gitleaks attestation")
    version = data["version"]
    if (
        data["tool"] != "gitleaks"
        or data["status"] != "passed"
        or data["scope"] != GITLEAKS_SCOPE
        or data["source_sha"] != source_sha
        or data["config_sha256"] != GITLEAKS_CONFIG_SHA256
        or data["flags"] != list(GITLEAKS_FLAGS)
        or not isinstance(version, str)
        or _SAFE_VERSION.fullmatch(version) is None
        or not isinstance(data["report_sha256"], str)
        or _SHA256.fullmatch(data["report_sha256"]) is None
    ):
        raise PublicVerifyError("Gitleaks attestation is malformed or untrusted")


def _validate_record(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PublicVerifyError("attestation file record must be a mapping")
    fields = set(value)
    if fields != _FILE_FIELDS and fields != _FILE_FIELDS | {"semantic_recipe"}:
        raise PublicVerifyError("attestation file record has unknown or missing fields")
    record = cast(dict[str, object], value)
    path = record["output_path"]
    source_id = record["source_id"]
    if (
        not isinstance(path, str)
        or not _canonical_relative(path)
        or path == ATTESTATION_FILENAME
        or not isinstance(source_id, str)
        or not source_id
        or not all(_is_sha256(record[field]) for field in ("input_sha256", "output_sha256"))
        or record["mode"] not in {"000644", "000755"}
        or not (
            record["transform_id"] is None
            or (
                isinstance(record["transform_id"], str)
                and 0 < len(record["transform_id"]) <= 256
            )
        )
    ):
        raise PublicVerifyError("attestation file record is malformed")
    semantic = record.get("semantic_recipe")
    if semantic is not None:
        linkage = _exact_mapping(semantic, _SEMANTIC_FIELDS, "semantic recipe linkage")
        if (
            not all(
                _is_sha256(linkage[field])
                for field in ("source_sha256", "output_sha256")
            )
            or type(linkage["identity_preserved"]) is not bool
            or linkage["identity_preserved"]
            is not (linkage["source_sha256"] == linkage["output_sha256"])
        ):
            raise PublicVerifyError("semantic recipe linkage is malformed")
    return record


def _load_attestation(root: Path, source_sha: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    path = root / ATTESTATION_FILENAME
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PublicVerifyError("public export attestation is missing") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > MAX_ATTESTATION_BYTES
        or stat.S_IMODE(metadata.st_mode) != 0o644
    ):
        raise PublicVerifyError("public export attestation metadata is unsafe")
    data = _load_json(
        _read_regular_file_stable(
            path,
            maximum=MAX_ATTESTATION_BYTES,
            label="public export attestation",
            expected=metadata,
        ),
        "public export attestation",
    )
    attestation = _exact_mapping(data, _ATTESTATION_FIELDS, "public export attestation")
    if (
        attestation["schema"] != ATTESTATION_SCHEMA
        or type(attestation["schema_version"]) is not int
        or attestation["schema_version"] != 1
        or attestation["exporter_version"] != EXPORTER_VERSION
        or attestation["detector_versions"] != {"metadata": DETECTOR_VERSION}
        or not isinstance(attestation["payload_tree_sha256"], str)
        or _SHA256.fullmatch(attestation["payload_tree_sha256"]) is None
    ):
        raise PublicVerifyError("public export attestation fields are invalid")
    if attestation["source_sha"] != source_sha:
        raise PublicVerifyError("attested source SHA does not match expected source SHA")
    _validate_gitleaks_attestation(attestation["gitleaks"], source_sha)
    raw_records = attestation["files"]
    if (
        not isinstance(raw_records, list)
        or not raw_records
        or len(raw_records) > MAX_FILES
    ):
        raise PublicVerifyError("attestation file list is invalid")
    records = [_validate_record(value) for value in raw_records]
    paths = [cast(str, record["output_path"]) for record in records]
    if len(paths) != len(set(paths)):
        raise PublicVerifyError("duplicate attestation output path")
    portable_paths = [_portable_identity(path) for path in paths]
    if len(portable_paths) != len(set(portable_paths)):
        raise PublicVerifyError("portable case or Unicode attestation path collision")
    return attestation, records


def _walk_tree(root: Path) -> dict[str, os.stat_result]:
    found: dict[str, os.stat_result] = {}
    identities: dict[str, str] = {}
    total = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise PublicVerifyError("export tree cannot be enumerated") from error
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if not _canonical_relative(relative):
                raise PublicVerifyError(f"noncanonical exported path: {relative}")
            identity = _portable_identity(relative)
            previous = identities.setdefault(identity, relative)
            if previous != relative:
                raise PublicVerifyError("portable case or Unicode path collision")
            parts = tuple(part.casefold() for part in PurePosixPath(relative).parts)
            if any(part in _RESERVED_COMPONENTS for part in parts) or any(
                parts[index : index + 2] == ("results", "raw")
                for index in range(len(parts) - 1)
            ):
                raise PublicVerifyError(f"reserved private or raw path: {relative}")
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise PublicVerifyError(f"symlink is forbidden: {relative}")
            if entry.is_dir(follow_symlinks=False):
                if stat.S_IMODE(metadata.st_mode) != 0o755:
                    raise PublicVerifyError(f"directory mode mismatch: {relative}")
                stack.append(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise PublicVerifyError(f"non-regular payload is forbidden: {relative}")
            if metadata.st_nlink != 1:
                raise PublicVerifyError(f"hardlink anomaly: {relative}")
            total += metadata.st_size
            if total > MAX_TOTAL_BYTES:
                raise PublicVerifyError("public export total size limit exceeded")
            found[relative] = metadata
            if len(found) > MAX_FILES + 1:
                raise PublicVerifyError("public export file count limit exceeded")
    return found


_TreeInventory = dict[str, tuple[str, int, str]]


def _complete_tree_inventory(root: Path) -> _TreeInventory:
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise PublicVerifyError("export root cannot be inventoried") from error
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise PublicVerifyError("export root cannot be inventoried")
    inventory: _TreeInventory = {
        ".": ("directory", stat.S_IMODE(root_metadata.st_mode), "")
    }
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise PublicVerifyError("export tree cannot be inventoried") from error
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise PublicVerifyError("export tree cannot be inventoried") from error
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISLNK(metadata.st_mode):
                raise PublicVerifyError("export tree inventory contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                inventory[relative] = ("directory", mode, "")
                stack.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                maximum = (
                    MAX_ATTESTATION_BYTES
                    if relative == ATTESTATION_FILENAME
                    else MAX_FILE_BYTES
                )
                data = _read_regular_file_stable(
                    path,
                    maximum=maximum,
                    label=f"inventory entry {relative}",
                    expected=metadata,
                )
                inventory[relative] = (
                    "file",
                    mode,
                    hashlib.sha256(data).hexdigest(),
                )
            else:
                raise PublicVerifyError("export tree inventory contains a special file")
    return inventory


def _copy_verified_tree(
    source: Path,
    destination: Path,
    inventory: _TreeInventory,
) -> None:
    destination.mkdir(mode=0o755)
    directories = sorted(
        (
            relative
            for relative, (kind, _, _) in inventory.items()
            if kind == "directory" and relative != "."
        ),
        key=lambda relative: (len(PurePosixPath(relative).parts), relative),
    )
    for relative in directories:
        target = destination / relative
        target.mkdir(mode=0o755)
        target.chmod(0o755)
    for relative, (kind, mode, expected_hash) in sorted(inventory.items()):
        if kind != "file":
            continue
        maximum = (
            MAX_ATTESTATION_BYTES
            if relative == ATTESTATION_FILENAME
            else MAX_FILE_BYTES
        )
        data = _read_regular_file_stable(
            source / relative,
            maximum=maximum,
            label=f"verified copy source {relative}",
        )
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise PublicVerifyError("public export changed while creating verified copy")
        target = destination / relative
        target.write_bytes(data)
        target.chmod(0o755 if mode == 0o755 else 0o644)


def _read_regular_file_stable(
    path: Path,
    *,
    maximum: int,
    label: str,
    expected: os.stat_result | None = None,
) -> bytes:
    descriptor: int | None = None
    try:
        initial = path.lstat()
        if stat.S_ISLNK(initial.st_mode):
            raise PublicVerifyError(f"{label} is a symlink")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        current = path.lstat()
        reference = expected or current
        if stat.S_ISLNK(current.st_mode):
            raise PublicVerifyError(f"{label} is a symlink")
        if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(current.st_mode):
            raise PublicVerifyError(f"{label} is not a regular file")
        if opened.st_nlink != 1:
            raise PublicVerifyError(f"{label} has a hardlink anomaly")
        if (
            opened.st_size > maximum
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (initial.st_dev, initial.st_ino, initial.st_size)
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (current.st_dev, current.st_ino, current.st_size)
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (reference.st_dev, reference.st_ino, reference.st_size)
        ):
            raise PublicVerifyError(f"{label} identity or size is unsafe")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(data) != opened.st_size
            or (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
        ):
            raise PublicVerifyError(f"{label} changed while reading")
        return data
    except PublicVerifyError:
        raise
    except OSError as error:
        raise PublicVerifyError(f"{label} cannot be read safely") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _verify_payload(
    root: Path,
    records: list[dict[str, object]],
    found: dict[str, os.stat_result],
) -> str:
    listed = {cast(str, record["output_path"]) for record in records}
    expected = listed | {ATTESTATION_FILENAME}
    missing = expected - found.keys()
    unlisted = found.keys() - expected
    if missing:
        raise PublicVerifyError("listed payload is missing")
    if unlisted:
        raise PublicVerifyError("unlisted payload exists")
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: cast(str, item["output_path"])):
        relative = cast(str, record["output_path"])
        metadata = found[relative]
        expected_mode = int(cast(str, record["mode"]), 8)
        if stat.S_IMODE(metadata.st_mode) != expected_mode:
            raise PublicVerifyError(f"payload mode mismatch: {relative}")
        if metadata.st_size > MAX_FILE_BYTES:
            raise PublicVerifyError(f"payload file exceeds size policy: {relative}")
        actual_hash = hashlib.sha256(
            _read_regular_file_stable(
                root / relative,
                maximum=MAX_FILE_BYTES,
                label=f"payload {relative}",
                expected=metadata,
            )
        ).hexdigest()
        if actual_hash != record["output_sha256"]:
            raise PublicVerifyError(f"payload hash mismatch: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(cast(str, record["mode"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(actual_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _parse_structured_files(root: Path, paths: set[str]) -> None:
    for relative in sorted(paths):
        path = root / relative
        suffix = path.suffix.casefold()
        if suffix == ".json":
            _load_json(path.read_bytes(), relative)
        elif suffix in {".yaml", ".yml"}:
            try:
                text = path.read_text(encoding="utf-8")
                yaml.load(text, Loader=_UniqueKeyLoader)
            except PublicVerifyError:
                raise
            except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
                raise PublicVerifyError(f"invalid YAML: {relative}") from error


def _validate_recipes(
    root: Path, paths: set[str], records: list[dict[str, object]]
) -> None:
    by_path = {cast(str, record["output_path"]): record for record in records}
    for relative in sorted(paths):
        if not relative.startswith("recipes/") or Path(relative).suffix.casefold() not in {
            ".yaml",
            ".yml",
        }:
            continue
        try:
            raw = yaml.load(
                (root / relative).read_text(encoding="utf-8"),
                Loader=_UniqueKeyLoader,
            )
            if not isinstance(raw, dict):
                raise PublicVerifyError(f"recipe must be a mapping: {relative}")
            if raw.get("schema_version") == "1.0" and "models" in raw:
                load_spec(root / relative)
            elif raw.get("schema_version") == "1.0":
                _validate_legacy_model_recipe(raw, relative)
            else:
                load_recipe(root / relative)
        except PublicVerifyError:
            raise
        except (OSError, UnicodeDecodeError, yaml.YAMLError, ValidationError, RecipeError) as error:
            raise PublicVerifyError(f"recipe validation failed: {relative}") from error
        record = by_path[relative]
        semantic = record.get("semantic_recipe")
        if semantic is None:
            raise PublicVerifyError(f"recipe lacks semantic linkage: {relative}")
        linkage = cast(dict[str, object], semantic)
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if (
            linkage["output_sha256"] != actual
            or linkage["source_sha256"] != actual
            or linkage["identity_preserved"] is not True
        ):
            raise PublicVerifyError(f"recipe semantic linkage failed: {relative}")


def _validate_legacy_model_recipe(raw: dict[str, object], relative: str) -> None:
    """Validate the pre-schema-2 immutable model recipe retained for provenance."""

    required = {"schema_version", "name", "source", "validation", "runtime"}
    if not required <= raw.keys() or not isinstance(raw["name"], str) or not raw["name"]:
        raise PublicVerifyError(f"legacy recipe structure is invalid: {relative}")
    source = raw["source"]
    if not isinstance(source, dict) or set(source) != {"model_id", "revision"}:
        raise PublicVerifyError(f"legacy recipe source is invalid: {relative}")
    model_id = source.get("model_id")
    revision = source.get("revision")
    if (
        not isinstance(model_id, str)
        or not model_id
        or not isinstance(revision, str)
        or _SHA.fullmatch(revision) is None
    ):
        raise PublicVerifyError(f"legacy recipe source identity is invalid: {relative}")
    for section in ("validation", "runtime"):
        if not isinstance(raw[section], dict) or not raw[section]:
            raise PublicVerifyError(f"legacy recipe {section} is invalid: {relative}")
    abliteration = raw.get("abliteration")
    if abliteration is not None:
        if not isinstance(abliteration, dict):
            raise PublicVerifyError(f"legacy recipe abliteration is invalid: {relative}")
        required_abliteration = {
            "layer",
            "seed",
            "harmful_prompts",
            "harmless_prompts",
            "target_selectors",
            "expected_target_count",
        }
        if not required_abliteration <= abliteration.keys():
            raise PublicVerifyError(f"legacy recipe abliteration is incomplete: {relative}")
        for field in (
            "layer",
            "harmful_prompts",
            "harmless_prompts",
            "expected_target_count",
        ):
            if type(abliteration[field]) is not int or cast(int, abliteration[field]) <= 0:
                raise PublicVerifyError(
                    f"legacy recipe abliteration integer is invalid: {relative}"
                )
        selectors = abliteration["target_selectors"]
        if not isinstance(selectors, list) or not selectors or not all(
            isinstance(selector, str) and selector for selector in selectors
        ):
            raise PublicVerifyError(f"legacy recipe selectors are invalid: {relative}")


def _matching_markdown_delimiter(
    text: str, start: int, opening: str, closing: str
) -> int | None:
    depth = 1
    index = start + 1
    while index < len(text):
        character = text[index]
        if character == "\\":
            index += 2
            continue
        if character == "\n" and opening == "(":
            return None
        if character == opening:
            depth += 1
            if depth > MAX_MARKDOWN_NESTING:
                raise PublicVerifyError("Markdown link nesting limit exceeded")
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _markdown_destinations(text: str) -> list[str]:
    destinations: list[str] = []
    index = 0
    while index < len(text):
        label_start = index + 1 if text[index : index + 2] == "![" else index
        if label_start >= len(text) or text[label_start] != "[":
            index += 1
            continue
        label_end = _matching_markdown_delimiter(text, label_start, "[", "]")
        if label_end is None or label_end + 1 >= len(text) or text[label_end + 1] != "(":
            index = label_start + 1
            continue
        target_end = _matching_markdown_delimiter(text, label_end + 1, "(", ")")
        if target_end is None:
            raise PublicVerifyError("Markdown link destination is malformed")
        raw = text[label_end + 2 : target_end].strip()
        if not raw:
            destinations.append("")
        elif raw.startswith("<"):
            angle_end = raw.find(">")
            if angle_end < 0 or raw[angle_end + 1 :].strip():
                raise PublicVerifyError("Markdown link destination is malformed")
            destinations.append(raw[1:angle_end])
        else:
            destination: list[str] = []
            position = 0
            nested = 0
            while position < len(raw):
                character = raw[position]
                if character == "\\":
                    if position + 1 >= len(raw):
                        raise PublicVerifyError("Markdown link escape is malformed")
                    destination.append(raw[position + 1])
                    position += 2
                    continue
                if character.isspace() and nested == 0:
                    break
                if character == "(":
                    nested += 1
                elif character == ")":
                    nested -= 1
                    if nested < 0:
                        raise PublicVerifyError("Markdown link destination is malformed")
                destination.append(character)
                position += 1
            if nested or not destination:
                raise PublicVerifyError("Markdown link destination is malformed")
            destinations.append("".join(destination))
        if len(destinations) > MAX_MARKDOWN_LINKS:
            raise PublicVerifyError("Markdown link count limit exceeded")
        index = target_end + 1
    return destinations


def _validate_markdown_links(root: Path, paths: set[str]) -> None:
    for relative in sorted(paths):
        if Path(relative).suffix.casefold() not in {".md", ".markdown"}:
            continue
        try:
            text = (root / relative).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise PublicVerifyError(f"Markdown is not UTF-8: {relative}") from error
        for raw_target in _markdown_destinations(text):
            parsed = urllib.parse.urlsplit(raw_target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            decoded = urllib.parse.unquote(parsed.path)
            if decoded.startswith("/"):
                candidate = root / decoded.lstrip("/")
            else:
                candidate = root / Path(relative).parent / decoded
            try:
                candidate.resolve(strict=True).relative_to(root)
            except (OSError, ValueError) as error:
                raise PublicVerifyError(
                    f"broken or escaping Markdown link in {relative}"
                ) from error


def _scan_metadata(root: Path, paths: set[str], public_contact: str) -> None:
    try:
        policy = DetectorPolicy(
            max_file_bytes=MAX_FILE_BYTES,
            max_scan_bytes=MAX_FILE_BYTES,
            public_contacts=frozenset({public_contact}),
        )
        budget = ScanWorkBudget()
        for relative in sorted(paths | {ATTESTATION_FILENAME}):
            findings = scan_file(
                root,
                relative,
                policy=policy,
                gitleaks_status=GitleaksStatus.PASSED,
                work_budget=budget,
            )
            findings = [
                finding
                for finding in findings
                if not (
                    finding.rule_id == "benchmark.raw-key"
                    and (
                        relative == "tests/public_export/fixtures/detector_cases.json"
                        or (
                            relative.endswith((".py", ".pyi"))
                            and relative.startswith(("src/", "tests/"))
                        )
                    )
                )
            ]
            if findings:
                rules = ", ".join(sorted({finding.rule_id for finding in findings}))
                raise PublicVerifyError(f"forbidden metadata in {relative}: {rules}")
    except DetectorError as error:
        raise PublicVerifyError("forbidden metadata scan failed") from error


def _trusted_environment() -> dict[str, str]:
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


def _run_bounded(
    arguments: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float = PROCESS_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    if timeout <= 0:
        raise PublicVerifyError("required verification subprocess timed out")
    deadline = time.monotonic() + timeout
    try:
        process = subprocess.Popen(
            arguments,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        raise PublicVerifyError("required verification subprocess failed") from error
    process_group = process.pid
    stdout = bytearray()
    stderr = bytearray()

    def drain(stream: Any, destination: bytearray) -> None:
        try:
            while chunk := stream.read(8192):
                remaining = MAX_PROCESS_OUTPUT_BYTES + 1 - len(destination)
                if remaining > 0:
                    destination.extend(chunk[:remaining])
        except (OSError, ValueError):
            pass

    assert process.stdout is not None
    assert process.stderr is not None
    threads = [
        threading.Thread(target=drain, args=(process.stdout, stdout), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, stderr), daemon=True),
    ]
    started: list[threading.Thread] = []

    def remaining(end: float) -> float:
        return max(0.0, end - time.monotonic())

    def join_until(end: float) -> bool:
        for thread in started:
            thread.join(timeout=remaining(end))
        return all(not thread.is_alive() for thread in started)

    def close_pipes() -> None:
        for stream in (process.stdout, process.stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    def terminate_and_reap() -> bool:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except OSError:
            pass
        close_pipes()
        cleanup_deadline = time.monotonic() + 1
        try:
            process.wait(timeout=remaining(cleanup_deadline))
        except subprocess.TimeoutExpired:
            pass
        return join_until(cleanup_deadline) and process.poll() is not None

    try:
        for thread in threads:
            thread.start()
            started.append(thread)
        try:
            returncode = process.wait(timeout=remaining(deadline))
        except subprocess.TimeoutExpired as error:
            raise PublicVerifyError("required verification subprocess timed out") from error
        if not join_until(deadline):
            raise PublicVerifyError(
                "required verification subprocess timed out waiting for descendants"
            )
        if len(stdout) + len(stderr) > MAX_PROCESS_OUTPUT_BYTES:
            raise PublicVerifyError("verification subprocess output limit exceeded")
        return subprocess.CompletedProcess(
            arguments, returncode, bytes(stdout), bytes(stderr)
        )
    except BaseException as error:
        if not terminate_and_reap():
            raise PublicVerifyError(
                "required verification subprocess cleanup did not complete"
            ) from error
        raise


@dataclass(frozen=True, slots=True)
class _PackageCommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    output_truncated: bool


def _run_package_command(
    arguments: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stage: str,
) -> _PackageCommandResult:
    try:
        result = _run_bounded(
            arguments,
            cwd=cwd,
            env=env,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        stdout_truncated = len(result.stdout) > MAX_PACKAGE_ERROR_BYTES // 4
        stdout = result.stdout[-MAX_PACKAGE_ERROR_BYTES // 4 :]
        stderr_limit = MAX_PACKAGE_ERROR_BYTES - len(stdout)
        stderr_truncated = len(result.stderr) > stderr_limit
        stderr = result.stderr[-stderr_limit:]
    except PublicVerifyError as error:
        raise PublicVerifyError(f"package {stage} smoke subprocess failed") from error
    return _PackageCommandResult(
        returncode=result.returncode,
        stdout=stdout,
        stderr=stderr,
        output_truncated=stdout_truncated or stderr_truncated,
    )


def _redacted_package_error(
    label: str,
    result: _PackageCommandResult,
    *,
    root: Path,
    temporary: Path,
) -> PublicVerifyError:
    raw = result.stderr or result.stdout
    detail = raw.decode("utf-8", errors="replace")
    for private_path, replacement in (
        (str(root), "<export-root>"),
        (str(temporary), "<temporary>"),
    ):
        detail = detail.replace(private_path, replacement)
    detail = re.sub(
        r"(?i)(https?://)[^/@\s:]+(?::[^/@\s]*)?@",
        r"\1<redacted>@",
        detail,
    )
    detail = "".join(
        character
        for character in detail
        if character in "\n\r\t" or ord(character) >= 32
    ).strip()
    if result.output_truncated:
        detail = f"[output truncated to {MAX_PACKAGE_ERROR_BYTES} bytes]\n{detail}"
    return PublicVerifyError(f"{label}:\n{detail}" if detail else label)


class SubprocessFilesystemGitleaksRunner:
    """Run Gitleaks without trusting repository config or ignore files."""

    def scan_directory(self, root: Path) -> FilesystemGitleaksEvidence:
        executable = shutil.which("gitleaks", path=SYSTEM_EXECUTABLE_PATH)
        if executable is None:
            raise PublicVerifyError("Gitleaks executable is required")
        env = _trusted_environment()
        version_result = _run_bounded([executable, "version"], cwd=root, env=env)
        if version_result.returncode != 0:
            raise PublicVerifyError("Gitleaks version check failed")
        version = version_result.stdout.decode("utf-8", errors="strict").strip()
        if _SAFE_VERSION.fullmatch(version) is None:
            raise PublicVerifyError("Gitleaks returned an invalid version")
        with tempfile.TemporaryDirectory(prefix="model-forge-public-verify-") as temporary:
            temp = Path(temporary)
            temp.chmod(0o700)
            config = temp / "trusted-gitleaks.toml"
            ignore = temp / "trusted-gitleaks.ignore"
            report = temp / "report.json"
            config.write_bytes(STRICT_GITLEAKS_POLICY)
            config.chmod(0o600)
            ignore.write_bytes(b"")
            ignore.chmod(0o600)
            result = _run_bounded(
                [
                    executable,
                    "dir",
                    "--redact",
                    "--report-format=json",
                    f"--report-path={report}",
                    f"--config={config}",
                    f"--gitleaks-ignore-path={ignore}",
                    str(root),
                ],
                cwd=root,
                env=env,
            )
            if result.returncode != 0:
                raise PublicVerifyError("Gitleaks filesystem scan failed")
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    report,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                opened = os.fstat(descriptor)
                current = report.lstat()
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or not stat.S_ISREG(current.st_mode)
                    or opened.st_nlink != 1
                    or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
                ):
                    raise PublicVerifyError("Gitleaks report identity changed")
                if opened.st_size > MAX_GITLEAKS_REPORT_BYTES:
                    raise PublicVerifyError("Gitleaks report size limit exceeded")
                chunks: list[bytes] = []
                remaining = opened.st_size
                while remaining:
                    chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw_report = b"".join(chunks)
                after = os.fstat(descriptor)
                if (
                    len(raw_report) != opened.st_size
                    or (after.st_dev, after.st_ino, after.st_size)
                    != (opened.st_dev, opened.st_ino, opened.st_size)
                ):
                    raise PublicVerifyError("Gitleaks report changed while reading")
            except PublicVerifyError:
                raise
            except OSError as error:
                raise PublicVerifyError("Gitleaks report is missing or unsafe") from error
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            report_data = _load_json(raw_report, "Gitleaks report")
            if report_data != []:
                raise PublicVerifyError("Gitleaks reported findings")
            return FilesystemGitleaksEvidence(
                version=version,
                report_sha256=hashlib.sha256(raw_report).hexdigest(),
            )


class SubprocessProjectChecksRunner:
    """Run the exported project's deterministic validation surface."""

    def run(self, root: Path) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "NO_COLOR": "1",
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        commands = (
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/public_export/test_public_export_e2e.py",
                "tests/test_recipe.py",
                "tests/test_cli.py",
            ],
            [sys.executable, "-m", "ruff", "check", "--no-cache", "."],
            [
                sys.executable,
                "-m",
                "mypy",
                "--no-incremental",
                f"--cache-dir={os.devnull}",
            ],
        )
        for arguments in commands:
            result = _run_bounded(arguments, cwd=root, env=environment)
            if result.returncode != 0:
                raise PublicVerifyError("exported project deterministic checks failed")


def _wheelhouse_evidence(
    wheelhouse: Path, lock: Path | None = None
) -> tuple[Path, tuple[WheelEvidence, ...]]:
    if wheelhouse.is_symlink():
        raise PublicVerifyError("trusted wheelhouse must not be a symlink")
    try:
        trusted = wheelhouse.resolve(strict=True)
    except OSError as error:
        raise PublicVerifyError("trusted wheelhouse is missing") from error
    if not trusted.is_dir():
        raise PublicVerifyError("trusted wheelhouse must be a directory")
    evidence: list[WheelEvidence] = []
    try:
        entries = sorted(trusted.iterdir(), key=lambda item: item.name)
    except OSError as error:
        raise PublicVerifyError("trusted wheelhouse cannot be enumerated") from error
    for path in entries:
        if path.suffix != ".whl":
            raise PublicVerifyError("trusted wheelhouse may contain only wheel files")
        descriptor: int | None = None
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            opened = os.fstat(descriptor)
            current = path.lstat()
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
            ):
                raise PublicVerifyError("trusted wheel identity is unsafe")
            digest = hashlib.sha256()
            total = 0
            while chunk := os.read(descriptor, 1024 * 1024):
                total += len(chunk)
                if total > MAX_TOTAL_BYTES:
                    raise PublicVerifyError("trusted wheel exceeds size policy")
                digest.update(chunk)
            after = os.fstat(descriptor)
            if (
                after.st_dev,
                after.st_ino,
                after.st_size,
            ) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            ):
                raise PublicVerifyError("trusted wheel changed while reading")
        except PublicVerifyError:
            raise
        except OSError as error:
            raise PublicVerifyError("trusted wheel cannot be read safely") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
        evidence.append(WheelEvidence(path.name, digest.hexdigest(), total))
    if not evidence:
        raise PublicVerifyError("trusted wheelhouse contains no wheels")
    if lock is not None:
        raw_lock = _read_regular_file_stable(
            lock,
            maximum=MAX_ATTESTATION_BYTES,
            label="trusted wheelhouse SHA256 lock",
        )
        try:
            text = raw_lock.decode("ascii")
        except UnicodeDecodeError as error:
            raise PublicVerifyError("trusted wheelhouse SHA256 lock is malformed") from error
        expected_lock = "".join(
            f"{item.sha256}  {item.name}\n" for item in evidence
        )
        lines = text.splitlines(keepends=True)
        wheel_lines = "".join(line for line in lines if not line.startswith("# "))
        source_record_lines = [line for line in lines if line.startswith("# ")]
        source_record_pattern = re.compile(
            r"# source-record sha256=[0-9a-f]{64} size=[0-9]+ "
            r"distribution=[a-z0-9]+(?:-[a-z0-9]+)*==[A-Za-z0-9][A-Za-z0-9._+-]*\n"
        )
        if (
            wheel_lines != expected_lock
            or any(source_record_pattern.fullmatch(line) is None for line in source_record_lines)
            or (source_record_lines and len(source_record_lines) != len(evidence))
        ):
            raise PublicVerifyError(
                "trusted wheelhouse does not match its SHA256 lock"
            )
    return trusted, tuple(evidence)


def _package_build_install_smoke(
    root: Path, wheelhouse: Path, wheelhouse_lock: Path | None = None
) -> tuple[WheelEvidence, ...]:
    trusted_wheelhouse, wheel_evidence = _wheelhouse_evidence(
        wheelhouse, wheelhouse_lock
    )
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        raise PublicVerifyError("pyproject.toml is required for package smoke")
    try:
        metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = metadata["project"]
        scripts = project["scripts"]
        build_system = metadata["build-system"]
        build_backend = build_system["build-backend"]
        build_requirements = build_system["requires"]
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise PublicVerifyError("package metadata is invalid") from error
    if not isinstance(scripts, dict) or not scripts:
        raise PublicVerifyError("package must expose at least one CLI smoke target")
    if (
        not isinstance(build_backend, str)
        or not build_backend
        or not isinstance(build_requirements, list)
        or not build_requirements
        or not all(
            isinstance(requirement, str) and requirement for requirement in build_requirements
        )
    ):
        raise PublicVerifyError("package build-system metadata is invalid")
    try:
        parsed_build_requirements = [Requirement(item) for item in build_requirements]
    except InvalidRequirement as error:
        raise PublicVerifyError("package build requirement is invalid") from error
    with tempfile.TemporaryDirectory(prefix="model-forge-package-smoke-") as temporary:
        temp = Path(temporary)
        home = temp / "home"
        run_directory = temp / "run"
        build_environment = temp / "build-venv"
        wheels = temp / "wheels"
        home.mkdir()
        run_directory.mkdir()
        wheels.mkdir()
        env = _trusted_environment()
        env.update(
            {
                "HOME": str(home),
                "NO_COLOR": "1",
                "PIP_CONFIG_FILE": os.devnull,
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_INDEX": "1",
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
            }
        )
        try:
            venv.EnvBuilder(with_pip=True, clear=True).create(build_environment)
        except OSError as error:
            raise PublicVerifyError("package build environment creation failed") from error
        build_python = build_environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        requirements = ", ".join(build_requirements)
        requirement_install = _run_package_command(
            [
                str(build_python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(trusted_wheelhouse),
                "--force-reinstall",
                *build_requirements,
            ],
            cwd=run_directory,
            env=env,
            stage="build requirement",
        )
        if requirement_install.returncode != 0:
            raise _redacted_package_error(
                "package build smoke failed using declared backend "
                f"{build_backend} because trusted wheelhouse requirements are unavailable: "
                f"{requirements}",
                requirement_install,
                root=root,
                temporary=temp,
            )
        requirement_check = _run_package_command(
            [
                str(build_python),
                "-c",
                (
                    "import importlib.metadata,json,sys;"
                    "print(json.dumps({name:importlib.metadata.version(name) "
                    "for name in json.loads(sys.argv[1])},sort_keys=True))"
                ),
                json.dumps([item.name for item in parsed_build_requirements]),
            ],
            cwd=run_directory,
            env=env,
            stage="build requirement version",
        )
        try:
            installed_versions = _load_json(
                requirement_check.stdout, "installed build requirement versions"
            )
        except PublicVerifyError:
            installed_versions = {}
        versions_satisfied = isinstance(installed_versions, dict) and all(
            isinstance(installed_versions.get(item.name), str)
            and installed_versions[item.name] in item.specifier
            for item in parsed_build_requirements
        )
        backend_check = _run_package_command(
            [
                str(build_python),
                "-c",
                "import importlib,sys;importlib.import_module(sys.argv[1])",
                build_backend.split(":", 1)[0],
            ],
            cwd=run_directory,
            env=env,
            stage="build backend import",
        )
        if (
            requirement_check.returncode != 0
            or not versions_satisfied
            or backend_check.returncode != 0
        ):
            failed = requirement_check if requirement_check.returncode != 0 else backend_check
            raise _redacted_package_error(
                f"declared build backend {build_backend} or requirements are unsatisfied: "
                f"{requirements}",
                failed,
                root=root,
                temporary=temp,
            )
        build = _run_package_command(
            [
                str(build_python),
                "-m",
                "pip",
                "wheel",
                "--no-build-isolation",
                "--no-deps",
                "--no-index",
                "--find-links",
                str(trusted_wheelhouse),
                "--wheel-dir",
                str(wheels),
                str(root),
            ],
            cwd=run_directory,
            env=env,
            stage="build",
        )
        wheel_files = list(wheels.glob("*.whl"))
        if build.returncode != 0 or len(wheel_files) != 1:
            raise _redacted_package_error(
                "package build smoke failed using declared backend "
                f"{build_backend} with locally installed build requirements: {requirements}",
                build,
                root=root,
                temporary=temp,
            )
        environment = temp / "venv"
        try:
            venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        except OSError as error:
            raise PublicVerifyError("package smoke environment creation failed") from error
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        install = _run_package_command(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(trusted_wheelhouse),
                str(wheel_files[0]),
            ],
            cwd=run_directory,
            env=env,
            stage="install",
        )
        if install.returncode != 0:
            raise _redacted_package_error(
                "package install smoke failed without dependency downloads",
                install,
                root=root,
                temporary=temp,
            )
        dependency_check = _run_package_command(
            [str(python), "-m", "pip", "check"],
            cwd=run_directory,
            env=env,
            stage="runtime requirement",
        )
        if dependency_check.returncode != 0:
            raise _redacted_package_error(
                "installed package runtime requirements are unsatisfied by trusted wheelhouse",
                dependency_check,
                root=root,
                temporary=temp,
            )
        for script_name in sorted(scripts):
            executable = environment / (
                f"Scripts/{script_name}.exe" if os.name == "nt" else f"bin/{script_name}"
            )
            smoke = _run_package_command(
                [str(executable), "--help"],
                cwd=run_directory,
                env=env,
                stage=f"CLI {script_name}",
            )
            if smoke.returncode != 0:
                raise _redacted_package_error(
                    f"installed package CLI smoke failed for {script_name}",
                    smoke,
                    root=root,
                    temporary=temp,
                )
    return wheel_evidence


def _verify_source_provenance(
    request: PublicVerifyRequest,
    root: Path,
    records: list[dict[str, object]],
    attestation: dict[str, object],
) -> None:
    if request.source_repo.is_symlink():
        raise PublicVerifyError("trusted source repository must not be a symlink")
    try:
        source = request.source_repo.resolve(strict=True)
    except OSError as error:
        raise PublicVerifyError("trusted source repository is missing") from error
    if not source.is_dir():
        raise PublicVerifyError("trusted source repository must be a directory")
    manifest_relative = request.manifest.as_posix()
    if request.manifest.is_absolute() or not _canonical_relative(manifest_relative):
        raise PublicVerifyError("trusted source manifest must be repository-relative")
    commit_check = _run_bounded(
        ["git", "rev-parse", f"{request.source_sha}^{{commit}}"],
        cwd=source,
        env=_trusted_environment(),
    )
    if (
        commit_check.returncode != 0
        or commit_check.stdout.decode("ascii", errors="ignore").strip()
        != request.source_sha
    ):
        raise PublicVerifyError("asserted source SHA is not a trusted Git commit")
    try:
        tree = _tree_entries(source, request.source_sha)
        plan = _plan_export(
            tree,
            manifest_relative,
            request.source_sha,
            public_contact=request.public_contact,
            fleet_hostnames=request.fleet_hostnames,
        )
    except PublicVerifyError:
        raise
    except ExportError as error:
        raise PublicVerifyError("trusted source provenance cannot be reconstructed") from error

    expected = {planned.destination: planned for planned in plan.files}
    actual_paths = {cast(str, record["output_path"]) for record in records}
    if actual_paths != expected.keys():
        raise PublicVerifyError("attested source mapping does not match trusted manifest")
    if attestation["payload_tree_sha256"] != plan.payload_tree_sha256:
        raise PublicVerifyError("attested payload digest does not match deterministic export plan")
    for record in records:
        path = cast(str, record["output_path"])
        planned = expected[path]
        if record != planned.record:
            raise PublicVerifyError(
                f"attested provenance does not match committed source for {path}"
            )
        actual = _read_regular_file_stable(
            root / path,
            maximum=MAX_FILE_BYTES,
            label=f"payload {path}",
        )
        if actual != planned.data:
            raise PublicVerifyError(
                f"payload bytes do not match deterministic export plan for {path}"
            )

    runner = request.source_gitleaks_runner or SubprocessGitleaksRunner()
    try:
        evidence = runner.scan_git(source, request.source_sha)
    except ExportError as error:
        raise PublicVerifyError("trusted full-history Gitleaks scan failed") from error
    except Exception as error:
        raise PublicVerifyError("trusted full-history Gitleaks runner failed") from error
    if (
        not isinstance(evidence, GitleaksEvidence)
        or evidence.source_sha != request.source_sha
        or evidence.scope != GITLEAKS_SCOPE
        or evidence.config_sha256 != GITLEAKS_CONFIG_SHA256
        or evidence.flags != GITLEAKS_FLAGS
        or _SAFE_VERSION.fullmatch(evidence.version) is None
        or _SHA256.fullmatch(evidence.report_sha256) is None
    ):
        raise PublicVerifyError("trusted full-history Gitleaks evidence is malformed")
    asserted = cast(dict[str, object], attestation["gitleaks"])
    if (
        asserted["version"] != evidence.version
        or asserted["report_sha256"] != evidence.report_sha256
        or asserted["scope"] != evidence.scope
        or asserted["source_sha"] != evidence.source_sha
        or asserted["config_sha256"] != evidence.config_sha256
        or asserted["flags"] != list(evidence.flags)
    ):
        raise PublicVerifyError(
            "export-time Gitleaks metadata does not match trusted full-history rerun"
        )


def verify_public_export(request: PublicVerifyRequest) -> PublicVerifyResult:
    """Recompute and validate all trusted properties of an exported tree."""

    if _SHA.fullmatch(request.source_sha) is None:
        raise PublicVerifyError("expected source SHA must be exactly 40 lowercase hex characters")
    if request.root.is_symlink():
        raise PublicVerifyError("public export root must not be a symlink")
    try:
        root = request.root.resolve(strict=True)
    except OSError as error:
        raise PublicVerifyError("public export root does not exist") from error
    if not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o755:
        raise PublicVerifyError("public export root must be a mode-0755 directory")

    before = _complete_tree_inventory(root)
    try:
        attestation, records = _load_attestation(root, request.source_sha)
        _verify_source_provenance(request, root, records, attestation)
        found = _walk_tree(root)
        digest = _verify_payload(root, records, found)
        if digest != attestation["payload_tree_sha256"]:
            raise PublicVerifyError("payload tree digest mismatch")
        paths = {cast(str, record["output_path"]) for record in records}
        policy_path = root / ".gitleaks.toml"
        if policy_path.exists() and policy_path.read_bytes() != STRICT_GITLEAKS_POLICY:
            raise PublicVerifyError("repository Gitleaks policy is not the strict trusted policy")
        if ".gitleaksignore" in found:
            raise PublicVerifyError("source-controlled Gitleaks ignore is forbidden")
        _parse_structured_files(root, paths)
        _validate_recipes(root, paths, records)
        _validate_markdown_links(root, paths)
        with tempfile.TemporaryDirectory(
            prefix="model-forge-verified-export-"
        ) as temporary:
            temporary_root = Path(temporary)
            temporary_root.chmod(0o700)
            verified_root = temporary_root / "export"
            _copy_verified_tree(root, verified_root, before)
            scanner = request.gitleaks_runner or SubprocessFilesystemGitleaksRunner()
            evidence = scanner.scan_directory(verified_root)
            if (
                not isinstance(evidence, FilesystemGitleaksEvidence)
                or _SAFE_VERSION.fullmatch(evidence.version) is None
                or _SHA256.fullmatch(evidence.report_sha256) is None
            ):
                raise PublicVerifyError("Gitleaks filesystem evidence is malformed")
            _scan_metadata(verified_root, paths, request.public_contact)
            wheelhouse_evidence: tuple[WheelEvidence, ...] = ()
            if request.package_smoke:
                wheelhouse_evidence = _package_build_install_smoke(
                    verified_root, request.wheelhouse, request.wheelhouse_lock
                )
            checks = request.project_checks_runner or SubprocessProjectChecksRunner()
            checks.run(verified_root)
        result = PublicVerifyResult(
            root=root,
            source_sha=request.source_sha,
            payload_tree_sha256=digest,
            file_count=len(records),
            gitleaks_version=evidence.version,
            wheelhouse_evidence=wheelhouse_evidence,
        )
    finally:
        try:
            after = _complete_tree_inventory(root)
        except PublicVerifyError as error:
            raise PublicVerifyError("public export changed during verification") from error
        if after != before:
            raise PublicVerifyError("public export changed during verification")
    return result
