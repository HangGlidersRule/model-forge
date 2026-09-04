"""The generated public root must pass its own manifest governance tests.

A public root has no private_archive files and carries the verifier-owned export
attestation, so classification is verified here against a staged one-commit public
tree rather than only against the private source.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from test_public_file_manifest import (
    ATTESTATION_SCHEMA,
    GENERATED_ATTESTATION,
    REPO_ROOT,
    _destination,
    _git_tracked_files,
    _load_manifest,
    _resolve_rule,
)

from model_forge.public_export.exporter import DEFAULT_PUBLIC_CONTACT
from model_forge.public_export.transforms import TransformContext, apply_transform

MANIFEST_TESTS = "tests/public_export/test_public_file_manifest.py"


def _clean_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTEST_ADDOPTS", "PYTEST_CURRENT_TEST"}
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_AUTHOR_NAME": "Public Root",
            "GIT_AUTHOR_EMAIL": DEFAULT_PUBLIC_CONTACT,
            "GIT_COMMITTER_NAME": "Public Root",
            "GIT_COMMITTER_EMAIL": DEFAULT_PUBLIC_CONTACT,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _stage_public_payload(root: Path) -> list[dict[str, object]]:
    """Copy exactly the files a public export would emit for the current source."""
    rules = _load_manifest()["rules"]
    planned: list[tuple[str, str, dict[str, object]]] = []
    for path in sorted(_git_tracked_files()):
        rule = _resolve_rule(rules, path)
        if rule["disposition"] == "exclude":
            continue
        destination = _destination(rule, path)
        assert destination is not None
        planned.append((path, destination, rule))
    public_paths = frozenset(destination for _, destination, _ in planned)
    staged: list[dict[str, object]] = []
    for source, destination, rule in planned:
        data = (REPO_ROOT / source).read_bytes()
        transformation = rule["transformation"]
        transform_id: str | None = None
        if transformation is not None:
            assert isinstance(transformation, str)
            transformed = apply_transform(
                transformation,
                data,
                TransformContext(
                    source_path=source,
                    source_sha="0" * 40,
                    public_contact=DEFAULT_PUBLIC_CONTACT,
                    public_paths=public_paths,
                ),
            )
            data = transformed.data
            transform_id = transformed.transform_id
        target = root / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod((REPO_ROOT / source).stat().st_mode & 0o777)
        staged.append(
            {
                "source_id": (
                    destination
                    if source == destination
                    else f"rule:{rule['id']}"
                ),
                "output_path": destination,
                "output_sha256": hashlib.sha256(data).hexdigest(),
                "transform_id": transform_id,
            }
        )
    return staged


def _write_attestation(root: Path, staged: list[dict[str, object]]) -> None:
    # Manifest governance reads only the schema marker and the file inventory, so the
    # staged attestation carries those fields rather than a fabricated export digest.
    (root / GENERATED_ATTESTATION).write_text(
        json.dumps(
            {
                "schema": ATTESTATION_SCHEMA,
                "schema_version": 1,
                "files": staged,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _commit_clean_root(root: Path, environment: dict[str, str]) -> None:
    for command in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "--no-gpg-sign", "-m", "Public root"],
    ):
        subprocess.run(command, cwd=root, check=True, env=environment)


def _stage_public_root(root: Path, environment: dict[str, str]) -> None:
    staged = _stage_public_payload(root)
    _write_attestation(root, staged)
    _commit_clean_root(root, environment)


def _run_manifest_tests(
    root: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", MANIFEST_TESTS, "-q", "-p", "no:cacheprovider"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_generated_public_root_passes_manifest_governance(tmp_path: Path) -> None:
    root = tmp_path / "public"
    root.mkdir()
    environment = _clean_environment()
    _stage_public_root(root, environment)

    assert not (root / "private_archive").exists()
    assert (root / GENERATED_ATTESTATION).is_file()
    completed = _run_manifest_tests(root, environment)

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_public_root_fails_when_a_private_archive_file_is_introduced(
    tmp_path: Path,
) -> None:
    root = tmp_path / "public"
    root.mkdir()
    environment = _clean_environment()
    _stage_public_root(root, environment)
    introduced = root / "private_archive/ai_review/dispatcher.py"
    introduced.parent.mkdir(parents=True)
    introduced.write_text("PRIVATE = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A", "-f"], cwd=root, check=True, env=environment)
    subprocess.run(
        ["git", "commit", "-q", "--no-gpg-sign", "-m", "Introduce private file"],
        cwd=root,
        check=True,
        env=environment,
    )

    completed = _run_manifest_tests(root, environment)

    assert completed.returncode != 0
    assert "test_combined_public_launch_files_are_classified_before_staging" in (
        completed.stdout + completed.stderr
    )


def test_public_root_fails_when_tracked_inventory_has_an_unattested_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "public"
    root.mkdir()
    environment = _clean_environment()
    _stage_public_root(root, environment)
    introduced = root / "docs/unattested.md"
    introduced.parent.mkdir(parents=True, exist_ok=True)
    introduced.write_text("not attested\n", encoding="utf-8")
    subprocess.run(["git", "add", "docs/unattested.md"], cwd=root, check=True, env=environment)
    subprocess.run(
        ["git", "commit", "-q", "--no-gpg-sign", "-m", "Add unattested file"],
        cwd=root,
        check=True,
        env=environment,
    )

    completed = _run_manifest_tests(root, environment)

    assert completed.returncode != 0
    assert "tracked inventory does not equal attested outputs" in (
        completed.stdout + completed.stderr
    )


def test_public_root_fails_when_an_attested_output_is_not_tracked(tmp_path: Path) -> None:
    root = tmp_path / "public"
    root.mkdir()
    environment = _clean_environment()
    staged = _stage_public_payload(root)
    _write_attestation(root, staged)
    missing = next(
        record["output_path"]
        for record in staged
        if record["output_path"] != "containers/build/docker-compose.yml"
    )
    (root / missing).unlink()
    _commit_clean_root(root, environment)

    completed = _run_manifest_tests(root, environment)

    assert completed.returncode != 0
    assert "tracked inventory does not equal attested outputs" in (
        completed.stdout + completed.stderr
    )
