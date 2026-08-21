from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from model_forge.public_export.transforms import (
    TransformContext,
    TransformError,
    apply_transform,
    available_transforms,
)


def context(path: str = "docs/guide.md") -> TransformContext:
    return TransformContext(
        source_path=path,
        source_sha="a" * 40,
        public_contact="security@example.com",
        fleet_hostnames=frozenset({"gpu-private-01.internal"}),
    )


def test_markdown_sanitization_is_deterministic_and_explicit() -> None:
    private = (
        b"Artifact: /Volumes/Private/models/release\n"
        b"Host: gpu-private-01.internal\n"
        b"Operator: alice@corp.example\r\n"
    )

    first = apply_transform("sanitize_public_markdown", private, context())
    second = apply_transform("sanitize_public_markdown", private, context())

    assert first.data == second.data
    assert first.data == (
        b"Artifact: ${PUBLIC_ARTIFACT_PATH}\n"
        b"Host: public-host.example\n"
        b"Operator: security@example.com\n"
    )
    assert first.transform_id == "sanitize_public_markdown:v1"


def test_journal_paths_are_removed_but_approved_hashes_are_retained() -> None:
    digest = "1" * 64
    source = f"Journal: /Users/alice/private/run.md\nJournal SHA256: {digest}\n".encode()

    result = apply_transform("sanitize_public_markdown", source, context())

    assert b"/Users/alice" not in result.data
    assert digest.encode() in result.data


def test_launcher_private_defaults_become_required_arguments() -> None:
    source = b'ARTIFACT_PATH="${ARTIFACT_PATH:-/d/model-forge/artifacts/model}"\r\n'

    result = apply_transform(
        "sanitize_container_script",
        source,
        context("scripts/launch.sh"),
    )

    assert result.data == (
        b'ARTIFACT_PATH="${ARTIFACT_PATH:?Set ARTIFACT_PATH to the public artifact path}"\n'
    )


def test_shell_sanitization_preserves_variable_relative_paths_and_balanced_expansions(
    tmp_path: Path,
) -> None:
    source = (
        b'ROOT="/workspace"\n'
        b'COMPOSE="${ROOT}/containers/serve/profile.yml"\n'
        b'PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"\n'
        b'MODELOPT_ROOT="/opt/modelopt"\n'
        b'case "${ROOT}" in D:/*|D:\\\\*|/mnt/d/*|/d/*) ;; *) exit 1 ;; esac\n'
    )
    result = apply_transform(
        "sanitize_container_script",
        source,
        context("scripts/launch.sh"),
    )
    script = tmp_path / "launch.sh"
    script.write_bytes(result.data)

    assert b'COMPOSE="${ROOT}/containers/serve/profile.yml"' in result.data
    assert b'PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"' in result.data
    assert b'MODELOPT_ROOT="/opt/modelopt"' in result.data
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)


def test_dockerfile_sanitization_declares_generic_path_inputs() -> None:
    source = (
        b"FROM example\n"
        b"ENV VIRTUAL_ENV=/opt/venv\n"
        b"RUN curl -o /tmp/wheel/${MODELOPT_WHEEL_FILENAME} https://example.invalid/wheel\n"
        b"WORKDIR /opt/modelopt\n"
    )

    result = apply_transform(
        "sanitize_container_script",
        source,
        context("containers/modelopt/Dockerfile"),
    )

    assert b"ARG PUBLIC_ARTIFACT_PATH\n" in result.data
    assert b"ARG PUBLIC_WORKSPACE\n" in result.data
    assert (
        b'RUN test -n "${PUBLIC_ARTIFACT_PATH}" && test -n "${PUBLIC_WORKSPACE}"\n'
        in result.data
    )
    assert b"ENV VIRTUAL_ENV=${PUBLIC_WORKSPACE}/venv\n" in result.data
    assert b"curl -o ${PUBLIC_ARTIFACT_PATH}" in result.data
    assert b"WORKDIR ${PUBLIC_WORKSPACE}/modelopt\n" in result.data


def test_recipe_transform_never_changes_semantic_bytes() -> None:
    source = b"schema_version: '2.0'\r\nname: reproducible"

    result = apply_transform(
        "sanitize_and_validate_recipe",
        source,
        context("recipes/example.yaml"),
    )

    assert result.data == source
    assert result.semantic_source_sha256 == hashlib.sha256(source).hexdigest()
    assert result.semantic_output_sha256 == hashlib.sha256(source).hexdigest()


def test_public_urls_are_not_mistaken_for_operator_paths() -> None:
    source = b"Docs: https://github.com/example/project/tree/main/docs\n"

    result = apply_transform("sanitize_public_markdown", source, context())

    assert result.data == source


def test_repository_relative_markdown_links_are_preserved() -> None:
    source = (
        b"[card](models/qwen3.8-27b-r3/model-card/base-bf16.md)\n"
        b"[contract](contracts/darkstar-release/v1/release-contract.schema.json)\n"
    )

    result = apply_transform("sanitize_public_markdown", source, context())

    assert result.data == source


@pytest.mark.parametrize(
    "public_path",
    [
        "/bin/sh",
        "/usr/bin/env",
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/usr/lib/libcuda.so.1",
        "/usr/local/lib/libexample.so",
        "/etc/ssl/certs/ca-certificates.crt",
        "/opt/homebrew/bin/python3",
    ],
)
def test_standard_system_paths_are_preserved(public_path: str) -> None:
    source = f"path={public_path}\n".encode()

    result = apply_transform("sanitize_public_markdown", source, context())

    assert result.data == source


@pytest.mark.parametrize(
    ("private_path", "placeholder"),
    [
        ("/Users/alice/project/worktree", "${PUBLIC_WORKSPACE}"),
        ("/home/alice/project/worktree", "${PUBLIC_WORKSPACE}"),
        ("/Volumes/Data/project/worktree", "${PUBLIC_WORKSPACE}"),
        (r"C:\Users\alice\project", "${PUBLIC_WORKSPACE}"),
        (r"\\fileserver\alice\project", "${PUBLIC_WORKSPACE}"),
        ("/opt/model-forge/artifacts/model", "${PUBLIC_ARTIFACT_PATH}"),
        ("/srv/team/checkpoints/run", "${PUBLIC_ARTIFACT_PATH}"),
        ("/var/lib/model-forge/weights/model", "${PUBLIC_ARTIFACT_PATH}"),
        ("/tmp/acme/workspace/build", "${PUBLIC_WORKSPACE}"),
    ],
)
def test_operator_and_local_artifact_paths_are_sanitized(
    private_path: str, placeholder: str
) -> None:
    source = f"path={private_path}\n".encode()

    result = apply_transform("sanitize_public_markdown", source, context())

    assert private_path.encode() not in result.data
    assert result.data == f"path={placeholder}\n".encode()


@pytest.mark.parametrize(
    ("transform", "source", "checker"),
    [
        (
            "sanitize_container_script",
            b"#!/usr/bin/env bash\nprintf '%s\\n' /usr/bin/env\n",
            ("bash", "-n"),
        ),
        (
            "sanitize_container_script",
            b"#!/bin/sh\nprintf '%s\\n' /etc/ssl/certs/ca-certificates.crt\n",
            ("sh", "-n"),
        ),
        (
            "sanitize_python_script",
            b"#!/usr/bin/env python3\nprint('/usr/lib/libexample.so')\n",
            ("python3", "-m", "py_compile"),
        ),
    ],
)
def test_exported_script_shebangs_remain_valid(
    tmp_path: Path,
    transform: str,
    source: bytes,
    checker: tuple[str, ...],
) -> None:
    result = apply_transform(transform, source, context("bin/run"))
    script = tmp_path / "run"
    script.write_bytes(result.data)

    assert result.data.splitlines()[0] == source.splitlines()[0]
    subprocess.run([*checker, str(script)], check=True, capture_output=True)


def test_recipe_with_private_runtime_content_fails_instead_of_changing_semantics() -> None:
    source = b"schema_version: '2.0'\nartifact_path: /Users/alice/model\n"

    with pytest.raises(TransformError, match="semantic recipe"):
        apply_transform(
            "sanitize_and_validate_recipe",
            source,
            context("recipes/private.yaml"),
        )


def test_structured_transforms_validate_their_result() -> None:
    with pytest.raises(TransformError, match="valid YAML"):
        apply_transform(
            "sanitize_and_validate_compose",
            b"services: [\n",
            context("containers/docker-compose.yml"),
        )


def test_binary_and_unknown_transforms_fail_closed() -> None:
    with pytest.raises(TransformError, match="UTF-8"):
        apply_transform("sanitize_public_markdown", b"\xff", context())
    with pytest.raises(TransformError, match="unknown transform"):
        apply_transform("typo_transform", b"safe\n", context())


def test_registry_implements_every_pr_a_transform_contract() -> None:
    assert available_transforms() == {
        "sanitize_and_validate_compose",
        "sanitize_and_validate_modelopt_config",
        "sanitize_and_validate_recipe",
        "sanitize_and_validate_serve_profile",
        "sanitize_artifact_manifest",
        "sanitize_container_script",
        "sanitize_public_gitignore",
        "sanitize_public_markdown",
        "sanitize_public_model_card",
        "sanitize_python_script",
        "sanitize_qwen_script",
        "sanitize_serving_capacity_profiles",
        "sanitize_validation_inventory",
    }
