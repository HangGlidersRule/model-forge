"""Deterministic, fail-closed transforms for public repository exports."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Callable

import yaml

TRANSFORM_VERSION = "1"
PUBLIC_HOST_PLACEHOLDER = "public-host.example"
PUBLIC_ARTIFACT_PLACEHOLDER = "${PUBLIC_ARTIFACT_PATH}"
PUBLIC_WORKSPACE_PLACEHOLDER = "${PUBLIC_WORKSPACE}"

_ABSOLUTE_PATH = re.compile(
    r"(?:/(?:Users|home|Volumes)/[^\s\"'`<>{})|;]+"
    r"|(?<![A-Za-z0-9._:/}-])/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._${}-]+){2,}"
    r"|(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"'`<>{})|;]+"
    r"|\\\\[A-Za-z0-9._-]+[\\/][^\s\"'`<>{})|;]+)"
)
_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._+/-])[A-Za-z0-9]+(?:[._+-][A-Za-z0-9]+)*@"
    r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?(?![A-Za-z0-9.-])"
)
_SHELL_PRIVATE_DEFAULT = re.compile(
    r'\$\{(?P<name>[A-Z][A-Z0-9_]*):-'
    r"(?P<value>(?:/(?:Users|home|Volumes)/|/[A-Za-z0-9._-]+/|[A-Za-z]:[\\/]|"
    r"\\\\)[^}\r\n]+)\}"
)
_ARTIFACT_WORDS = re.compile(
    r"(?i)(?:^|[/\\._-])(?:artifacts?|models?|checkpoints?|weights)(?:$|[/\\._-])"
)
_PRIVATE_PATH_WORDS = re.compile(
    r"(?i)(?:^|[/\\._-])(?:private|workspace|worktree)(?:$|[/\\._-])"
)
_PUBLIC_SYSTEM_PATH_PREFIXES = (
    "/bin/",
    "/sbin/",
    "/usr/bin/",
    "/usr/sbin/",
    "/usr/lib/",
    "/usr/lib64/",
    "/usr/local/bin/",
    "/usr/local/sbin/",
    "/usr/local/lib/",
    "/usr/local/lib64/",
    "/etc/ca-certificates/",
    "/etc/pki/",
    "/etc/ssl/",
    "/lib/",
    "/lib64/",
    "/opt/homebrew/bin/",
    "/opt/homebrew/lib/",
    "/opt/homebrew/sbin/",
)


class TransformError(ValueError):
    """A transform cannot safely produce deterministic public bytes."""


@dataclass(frozen=True, slots=True)
class TransformContext:
    """Bounded public values supplied by the exporter."""

    source_path: str
    source_sha: str
    public_contact: str
    fleet_hostnames: frozenset[str] = frozenset()
    public_paths: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class TransformResult:
    """Transformed bytes and optional semantic identity linkage."""

    data: bytes
    transform_id: str
    semantic_source_sha256: str | None = None
    semantic_output_sha256: str | None = None


Transform = Callable[[bytes, TransformContext], TransformResult]


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TransformError("transform input must be UTF-8 text") from error


def _path_placeholder(match: re.Match[str]) -> str:
    value = match.group(0)
    prefix = match.string[max(0, match.start() - 512) : match.start()]
    token_prefix = re.split(r"[\s\"'`<>]", prefix)[-1]
    if "://" in token_prefix or token_prefix.startswith("re:"):
        return value
    if _ARTIFACT_WORDS.search(value):
        return PUBLIC_ARTIFACT_PLACEHOLDER
    if _PRIVATE_PATH_WORDS.search(value):
        return PUBLIC_WORKSPACE_PLACEHOLDER
    if value.startswith(_PUBLIC_SYSTEM_PATH_PREFIXES):
        return value
    return PUBLIC_WORKSPACE_PLACEHOLDER


def _replace_private_values(text: str, context: TransformContext) -> str:
    for hostname in sorted(context.fleet_hostnames, key=lambda item: (-len(item), item)):
        text = re.sub(
            rf"(?i)(?<![A-Za-z0-9.-]){re.escape(hostname)}(?![A-Za-z0-9.-])",
            PUBLIC_HOST_PLACEHOLDER,
            text,
        )
    text = _ABSOLUTE_PATH.sub(_path_placeholder, text)
    text = _EMAIL.sub(context.public_contact, text)
    return text


def _sanitize_text(data: bytes, context: TransformContext) -> bytes:
    text = _decode(data).replace("\r\n", "\n").replace("\r", "\n")
    text = _replace_private_values(text, context)
    if text and not text.endswith("\n"):
        text += "\n"
    return text.encode("utf-8")


def _plain(name: str) -> Transform:
    def transform(data: bytes, context: TransformContext) -> TransformResult:
        return TransformResult(_sanitize_text(data, context), f"{name}:v{TRANSFORM_VERSION}")

    return transform


def _gitignore(data: bytes, context: TransformContext) -> TransformResult:
    """Keep every planned payload path stageable in a fresh public repository."""

    text = _decode(data).replace("\r\n", "\n")
    # Anchored gitignore patterns (`!/path` negations and `/path` ignores)
    # reference the PRIVATE operator layout. The generic sanitizer would read
    # their leading `/` as an absolute filesystem path and mangle each line
    # into a ${PUBLIC_WORKSPACE}/${PUBLIC_ARTIFACT_PATH} placeholder. Every
    # public payload negation is regenerated deterministically from the
    # manifest below, so drop the private anchored lines at the source.
    kept = [
        line
        for line in text.splitlines()
        if not (line.startswith("!/") or line.startswith("/"))
    ]
    output = _sanitize_text("\n".join(kept).encode("utf-8"), context)
    exceptions: set[str] = set()
    for public_path in context.public_paths:
        if "${" in public_path:
            # Placeholder-only paths are scrubbed operator metadata from transformed
            # content, not real files in the generated public tree. Adding gitignore
            # negations for them pollutes the public root with literal
            # ${PUBLIC_WORKSPACE}/${PUBLIC_ARTIFACT_PATH} entries.
            continue
        parts = public_path.split("/")
        for end in range(1, len(parts)):
            exceptions.add(f"!/{'/'.join(parts[:end])}/")
        exceptions.add(f"!/{public_path}")
    if exceptions:
        output += (
            b"\n# Canonical public-export payload; keep these attested paths stageable.\n"
            + "".join(f"{line}\n" for line in sorted(exceptions)).encode("utf-8")
        )
    if len(output) > 1_048_576:
        raise TransformError("sanitize_public_gitignore output exceeds 1048576 bytes")
    return TransformResult(output, "sanitize_public_gitignore:v2")


def _structured_yaml(name: str) -> Transform:
    def transform(data: bytes, context: TransformContext) -> TransformResult:
        output = _sanitize_text(data, context)
        try:
            yaml.safe_load(output)
        except yaml.YAMLError as error:
            raise TransformError(f"{name} must produce valid YAML") from error
        return TransformResult(output, f"{name}:v{TRANSFORM_VERSION}")

    return transform


def _structured_json_or_yaml(name: str) -> Transform:
    def transform(data: bytes, context: TransformContext) -> TransformResult:
        output = _sanitize_text(data, context)
        try:
            if context.source_path.endswith(".json"):
                json.loads(output)
            else:
                yaml.safe_load(output)
        except (json.JSONDecodeError, yaml.YAMLError) as error:
            raise TransformError(f"{name} must produce valid JSON or YAML") from error
        return TransformResult(output, f"{name}:v{TRANSFORM_VERSION}")

    return transform


def _script(name: str, *, python: bool = False) -> Transform:
    def transform(data: bytes, context: TransformContext) -> TransformResult:
        text = _decode(data).replace("\r\n", "\n").replace("\r", "\n")
        if context.source_path.endswith("Dockerfile"):
            for private, public in (
                (
                    "/tmp/wheel/${MODELOPT_WHEEL_FILENAME}",
                    PUBLIC_ARTIFACT_PLACEHOLDER,
                ),
                ("/opt/modelopt", f"{PUBLIC_WORKSPACE_PLACEHOLDER}/modelopt"),
                ("/opt/venv", f"{PUBLIC_WORKSPACE_PLACEHOLDER}/venv"),
                ("/tmp/wheel", f"{PUBLIC_WORKSPACE_PLACEHOLDER}/wheel"),
                (
                    "/var/lib/apt/lists",
                    f"{PUBLIC_WORKSPACE_PLACEHOLDER}/apt-lists",
                ),
            ):
                text = text.replace(private, public)

        def required_argument(match: re.Match[str]) -> str:
            variable = match.group("name")
            return f"${{{variable}:?Set {variable} to the public artifact path}}"

        text = _SHELL_PRIVATE_DEFAULT.sub(required_argument, text)
        output = _sanitize_text(text.encode(), context)
        if context.source_path.endswith("Dockerfile"):
            dockerfile = output.decode("utf-8")
            declarations = "".join(
                f"ARG {argument}\n"
                for argument in ("PUBLIC_ARTIFACT_PATH", "PUBLIC_WORKSPACE")
                if f"${{{argument}}}" in dockerfile
            )
            if declarations:
                declarations += (
                    'RUN test -n "${PUBLIC_ARTIFACT_PATH}" '
                    '&& test -n "${PUBLIC_WORKSPACE}"\n'
                )
                lines = dockerfile.splitlines(keepends=True)
                from_index = next(
                    (
                        index
                        for index, line in enumerate(lines)
                        if line.lstrip().upper().startswith("FROM ")
                    ),
                    None,
                )
                if from_index is None:
                    raise TransformError(f"{name} Dockerfile has no FROM instruction")
                lines.insert(from_index + 1, declarations)
                output = "".join(lines).encode("utf-8")
        if python:
            try:
                ast.parse(output)
            except SyntaxError as error:
                raise TransformError(f"{name} must produce valid Python") from error
        return TransformResult(output, f"{name}:v{TRANSFORM_VERSION}")

    return transform


def _recipe(data: bytes, context: TransformContext) -> TransformResult:
    digest = hashlib.sha256(data).hexdigest()
    text = _decode(data)
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise TransformError("semantic recipe must be valid YAML") from error
    if not isinstance(loaded, dict):
        raise TransformError("semantic recipe must be a YAML mapping")
    if _replace_private_values(text, context) != text:
        raise TransformError(
            "semantic recipe contains private runtime content; separate runtime path "
            "configuration before export"
        )
    return TransformResult(
        data,
        f"sanitize_and_validate_recipe:v{TRANSFORM_VERSION}",
        semantic_source_sha256=digest,
        semantic_output_sha256=digest,
    )


_TRANSFORMS: dict[str, Transform] = {
    "sanitize_public_gitignore": _gitignore,
    "sanitize_public_markdown": _plain("sanitize_public_markdown"),
    "sanitize_and_validate_modelopt_config": _structured_yaml(
        "sanitize_and_validate_modelopt_config"
    ),
    "sanitize_and_validate_compose": _structured_yaml("sanitize_and_validate_compose"),
    "sanitize_container_script": _script("sanitize_container_script"),
    "sanitize_and_validate_serve_profile": _structured_yaml(
        "sanitize_and_validate_serve_profile"
    ),
    "sanitize_public_model_card": _plain("sanitize_public_model_card"),
    "sanitize_validation_inventory": _plain("sanitize_validation_inventory"),
    "sanitize_serving_capacity_profiles": _structured_json_or_yaml(
        "sanitize_serving_capacity_profiles"
    ),
    "sanitize_artifact_manifest": _structured_json_or_yaml("sanitize_artifact_manifest"),
    "sanitize_and_validate_recipe": _recipe,
    "sanitize_qwen_script": _script("sanitize_qwen_script"),
    "sanitize_python_script": _script("sanitize_python_script", python=True),
}


def available_transforms() -> set[str]:
    """Return stable PR-A transform names implemented by this module."""

    return set(_TRANSFORMS)


def apply_transform(name: str, data: bytes, context: TransformContext) -> TransformResult:
    """Apply one named transform or fail closed."""

    transform = _TRANSFORMS.get(name)
    if transform is None:
        raise TransformError(f"unknown transform: {name}")
    return transform(data, context)
