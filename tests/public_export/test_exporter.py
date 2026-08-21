from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

import model_forge.public_export.exporter as exporter_module
from model_forge.public_export.exporter import (
    ExportError,
    ExportRequest,
    ExportResult,
    GitleaksEvidence,
    SubprocessGitleaksRunner,
    export_public,
)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def rule(
    source: str,
    *,
    disposition: str = "copy",
    destination: str | None = "{source}",
    transformation: str | None = None,
    max_size: int = 4096,
) -> dict[str, object]:
    return {
        "id": source.replace("/", "-").replace("*", "all"),
        "source": source,
        "disposition": disposition,
        "public_destination": destination,
        "reason": "Synthetic exporter contract.",
        "transformation": transformation,
        "owner": "tests",
        "max_size_bytes": max_size,
        "generated": False,
        "precedence": 100,
    }


def repository(
    tmp_path: Path,
    files: dict[str, bytes],
    rules: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, str]:
    repo = tmp_path / "private-source"
    repo.mkdir(parents=True)
    for relative, data in files.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = repo / "public-files.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "rules": rules
                or [
                    rule("public-files.yaml"),
                    *[rule(path) for path in files],
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if rules is not None and not any(item["source"] == "public-files.yaml" for item in rules):
        data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        data["rules"].append(rule("public-files.yaml"))
        manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Exporter Tests",
        "-c",
        "user.email=exporter@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    return repo, manifest, git(repo, "rev-parse", "HEAD")


def request(
    repo: Path,
    manifest: Path,
    source_sha: str,
    output: Path,
    **changes: object,
) -> ExportRequest:
    values: dict[str, object] = {
        "source": repo,
        "output": output,
        "manifest": manifest,
        "source_sha": source_sha,
        "gitleaks_runner": PassingGitleaksRunner(),
    }
    values.update(changes)
    return ExportRequest(**values)  # type: ignore[arg-type]


def blob_objects(tmp_path: Path, count: int) -> tuple[Path, list[str], list[bytes]]:
    """Write 256 distinct one-byte blobs and repeat their IDs to reach count."""

    repo = tmp_path / "many-blobs"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    paths: list[str] = []
    for value in range(256):
        target = repo / f"blob-{value:03d}.bin"
        target.write_bytes(bytes([value]))
        paths.append(str(target))
    written = subprocess.run(
        ["git", "hash-object", "-w", "--stdin-paths"],
        cwd=repo,
        input="\n".join(paths) + "\n",
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()
    assert len(written) == 256
    object_ids = [written[index % 256] for index in range(count)]
    expected = [bytes([index % 256]) for index in range(count)]
    return repo, object_ids, expected


def cat_file_reader(
    check: bytes | None = None,
    batch: bytes | None = None,
    returncode: int = 0,
) -> object:
    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        stdout = check if arguments[-1] == "--batch-check" else batch
        assert stdout is not None, f"unexpected invocation: {arguments}"
        return subprocess.CompletedProcess(arguments, returncode, stdout, b"")

    return run


@dataclass
class PassingGitleaksRunner:
    version: str = "8.30.1"

    def scan_git(self, source: Path, source_sha: str) -> GitleaksEvidence:
        assert source.is_absolute()
        return GitleaksEvidence(
            version=self.version,
            report_sha256=hashlib.sha256(b"[]").hexdigest(),
            scope="full-history-through-source-sha",
            source_sha=source_sha,
        )


def promoted_payload_digest(output: Path, records: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["output_path"])):
        relative = str(record["output_path"])
        payload = output / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(f"{payload.stat().st_mode & 0o777:06o}".encode("ascii"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def test_export_is_deterministic_and_attested_without_absolute_paths(tmp_path: Path) -> None:
    repo, manifest, sha = repository(
        tmp_path,
        {
            "bin/run.sh": b"#!/bin/sh\necho public\n",
            "docs/readme.md": b"public\n",
        },
    )
    (repo / "bin/run.sh").chmod(0o755)
    git(repo, "add", "bin/run.sh")
    git(
        repo,
        "-c",
        "user.name=Exporter Tests",
        "-c",
        "user.email=exporter@model-forge.example",
        "commit",
        "-qm",
        "mode",
    )
    sha = git(repo, "rev-parse", "HEAD")

    first = export_public(request(repo, manifest, sha, tmp_path / "public-one"))
    second = export_public(request(repo, manifest, sha, tmp_path / "public-two"))

    assert first.payload_tree_sha256 == second.payload_tree_sha256
    attestation_bytes = (tmp_path / "public-one/PUBLIC_EXPORT_MANIFEST.json").read_bytes()
    assert str(repo).encode() not in attestation_bytes
    attestation = json.loads(attestation_bytes)
    assert attestation["source_sha"] == sha
    assert attestation["gitleaks"] == {
        "config_sha256": hashlib.sha256(b"[extend]\nuseDefault = true\n").hexdigest(),
        "flags": [
            "git",
            "--redact",
            "--report-format=json",
            "--config=<exporter-owned>",
            "--gitleaks-ignore-path=<exporter-owned-empty>",
            "--log-opts=<source-sha>",
        ],
        "report_sha256": hashlib.sha256(b"[]").hexdigest(),
        "scope": "full-history-through-source-sha",
        "source_sha": sha,
        "status": "passed",
        "tool": "gitleaks",
        "version": "8.30.1",
    }
    assert attestation["payload_tree_sha256"] == first.payload_tree_sha256
    assert promoted_payload_digest(tmp_path / "public-one", attestation["files"]) == (
        first.payload_tree_sha256
    )
    assert "public_tree_digest" not in attestation
    assert "PUBLIC_EXPORT_MANIFEST.json" not in {
        item["output_path"] for item in attestation["files"]
    }
    attested_paths = {item["output_path"] for item in attestation["files"]}
    actual_paths = {
        path.relative_to(tmp_path / "public-one").as_posix()
        for path in (tmp_path / "public-one").rglob("*")
        if path.is_file() and path.name != "PUBLIC_EXPORT_MANIFEST.json"
    }
    assert attested_paths == actual_paths
    assert (tmp_path / "public-one/bin/run.sh").stat().st_mode & 0o777 == 0o755
    assert (tmp_path / "public-one/docs/readme.md").stat().st_mode & 0o777 == 0o644


def test_public_gitignore_keeps_transformed_container_build_compose_stageable(
    tmp_path: Path,
) -> None:
    gitignore_rule = rule(
        ".gitignore",
        disposition="transform",
        transformation="sanitize_public_gitignore",
    )
    compose_rule = rule(
        "containers/build/docker-compose.yml",
        disposition="transform",
        transformation="sanitize_and_validate_compose",
    )
    repo, manifest, sha = repository(
        tmp_path,
        {
            ".gitignore": b"",
            "containers/build/docker-compose.yml": b"services:\n  builder:\n    image: public\n",
        },
        [gitignore_rule, compose_rule],
    )
    (repo / ".gitignore").write_bytes(b"build/\n")
    git(repo, "add", ".gitignore")
    git(
        repo,
        "-c",
        "user.name=Exporter Tests",
        "-c",
        "user.email=exporter@example.com",
        "commit",
        "-qm",
        "ignore build directories",
    )
    sha = git(repo, "rev-parse", "HEAD")
    output = tmp_path / "public"

    export_public(request(repo, manifest, sha, output))
    git(output, "init", "-q")
    git(output, "add", ".")

    assert (output / "containers/build/docker-compose.yml").is_file()
    assert "containers/build/docker-compose.yml" in git(output, "ls-files").splitlines()
    attestation = json.loads(
        (output / "PUBLIC_EXPORT_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert {record["output_path"] for record in attestation["files"]} == {
        ".gitignore",
        "containers/build/docker-compose.yml",
        "public-files.yaml",
    }


def test_manifest_controls_copy_transform_and_exclude_exactly(tmp_path: Path) -> None:
    rules = [
        rule("copy.txt"),
        rule(
            "private.md",
            disposition="transform",
            transformation="sanitize_public_markdown",
        ),
        rule("excluded.txt", disposition="exclude", destination=None),
    ]
    repo, manifest, sha = repository(
        tmp_path,
        {
            "copy.txt": b"same bytes\r\n",
            "private.md": b"Artifact: /Users/alice/model\r\n",
            "excluded.txt": b"never public\n",
        },
        rules,
    )

    export_public(
        request(
            repo,
            manifest,
            sha,
            tmp_path / "public",
            public_contact="security@example.com",
        )
    )

    assert (tmp_path / "public/copy.txt").read_bytes() == b"same bytes\r\n"
    assert (tmp_path / "public/private.md").read_bytes() == (
        b"Artifact: ${PUBLIC_ARTIFACT_PATH}\n"
    )
    assert not (tmp_path / "public/excluded.txt").exists()


@pytest.mark.parametrize("disposition", ["copy", "transform"])
@pytest.mark.parametrize(
    "destination",
    [
        "PUBLIC_EXPORT_MANIFEST.json",
        "public_export_manifest.json",
        "PUBLIC_EXPORT_MANIFEſT.json",
    ],
)
def test_attestation_destination_is_reserved_before_scanning_or_writes(
    tmp_path: Path,
    disposition: str,
    destination: str,
) -> None:
    payload_rule = rule(
        "payload.txt",
        disposition=disposition,
        destination=destination,
        transformation="sanitize_public_markdown" if disposition == "transform" else None,
    )
    repo, manifest, sha = repository(
        tmp_path,
        {"payload.txt": b"public\n"},
        [payload_rule],
    )

    class UnexpectedGitleaksRunner:
        def scan_git(self, source: Path, source_sha: str) -> GitleaksEvidence:
            raise AssertionError("destination reservation must precede scanning")

    output = tmp_path / "public"
    with pytest.raises(ExportError, match="reserved"):
        export_public(
            request(
                repo,
                manifest,
                sha,
                output,
                gitleaks_runner=UnexpectedGitleaksRunner(),
            )
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".public.public-export-stage-*"))


def test_transformed_exported_scripts_remain_executable_and_syntax_valid(
    tmp_path: Path,
) -> None:
    scripts = {
        "bin/bash-tool": b"#!/usr/bin/env bash\nprintf '%s\\n' /usr/bin/env\n",
        "bin/sh-tool": b"#!/bin/sh\nprintf '%s\\n' /etc/ssl/certs/ca-certificates.crt\n",
        "bin/python-tool": b"#!/usr/bin/env python3\nprint('/usr/lib/libexample.so')\n",
    }
    rules = [
        rule(
            path,
            disposition="transform",
            transformation=(
                "sanitize_python_script"
                if path.endswith("python-tool")
                else "sanitize_container_script"
            ),
        )
        for path in scripts
    ]
    repo, manifest, sha = repository(tmp_path, scripts, rules)
    for path in scripts:
        (repo / path).chmod(0o755)
    git(repo, "add", *scripts)
    git(
        repo,
        "-c",
        "user.name=Exporter Tests",
        "-c",
        "user.email=exporter@example.com",
        "commit",
        "-qm",
        "executable scripts",
    )
    sha = git(repo, "rev-parse", "HEAD")
    output = tmp_path / "public"

    export_public(request(repo, manifest, sha, output))

    for path, source in scripts.items():
        exported = output / path
        assert exported.stat().st_mode & 0o777 == 0o755
        assert exported.read_bytes().splitlines()[0] == source.splitlines()[0]
    subprocess.run(["bash", "-n", str(output / "bin/bash-tool")], check=True)
    subprocess.run(["sh", "-n", str(output / "bin/sh-tool")], check=True)
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(output / "bin/python-tool")],
        check=True,
    )


@pytest.mark.parametrize("bad_sha", ["a" * 39, "g" * 40, "0" * 40])
def test_source_sha_must_be_valid_and_equal_git_head(tmp_path: Path, bad_sha: str) -> None:
    repo, manifest, _ = repository(tmp_path, {"safe.txt": b"safe\n"})

    with pytest.raises(ExportError, match="source SHA"):
        export_public(request(repo, manifest, bad_sha, tmp_path / "public"))


def test_dirty_tracked_source_fails(tmp_path: Path) -> None:
    repo, manifest, sha = repository(tmp_path, {"safe.txt": b"safe\n"})
    (repo / "safe.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ExportError, match="dirty tracked"):
        export_public(request(repo, manifest, sha, tmp_path / "public"))


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_hidden_worktree_spoof_exports_bytes_from_asserted_commit(
    tmp_path: Path, index_flag: str
) -> None:
    repo, manifest, sha = repository(tmp_path, {"safe.txt": b"committed\n"})
    git(repo, "update-index", index_flag, "safe.txt")
    (repo / "safe.txt").write_bytes(b"spoofed worktree bytes\n")

    export_public(request(repo, manifest, sha, tmp_path / "public"))

    assert (tmp_path / "public/safe.txt").read_bytes() == b"committed\n"


def test_manifest_is_loaded_from_asserted_commit_not_worktree(tmp_path: Path) -> None:
    repo, manifest, sha = repository(tmp_path, {"safe.txt": b"committed\n"})
    git(repo, "update-index", "--assume-unchanged", "public-files.yaml")
    manifest.write_text(
        "version: 1\nrules:\n  - id: malicious-unclassified-policy\n",
        encoding="utf-8",
    )

    export_public(request(repo, manifest, sha, tmp_path / "public"))

    assert (tmp_path / "public/safe.txt").read_bytes() == b"committed\n"


def test_git_replace_refs_cannot_rebind_asserted_commit(tmp_path: Path) -> None:
    repo, manifest, safe_sha = repository(tmp_path, {"safe.txt": b"committed\n"})
    safe_blob = git(repo, "rev-parse", f"{safe_sha}:safe.txt")
    (repo / "safe.txt").write_bytes(b"replacement object\n")
    git(repo, "add", "safe.txt")
    git(
        repo,
        "-c",
        "user.name=Exporter Tests",
        "-c",
        "user.email=exporter@model-forge.example",
        "commit",
        "-qm",
        "replacement",
    )
    replacement_sha = git(repo, "rev-parse", "HEAD")
    git(repo, "replace", safe_sha, replacement_sha)
    git(repo, "checkout", "-q", "--detach", safe_sha)
    (repo / "safe.txt").write_bytes(b"committed\n")
    git(repo, "update-index", "--cacheinfo", f"100644,{safe_blob},safe.txt")

    export_public(request(repo, manifest, safe_sha, tmp_path / "public"))

    assert (tmp_path / "public/safe.txt").read_bytes() == b"committed\n"


def test_nonempty_output_requires_replace_and_replace_is_atomic(tmp_path: Path) -> None:
    repo, manifest, sha = repository(tmp_path, {"safe.txt": b"new\n"})
    output = tmp_path / "public"
    output.mkdir()
    (output / "old.txt").write_text("old\n", encoding="utf-8")

    with pytest.raises(ExportError, match="non-empty"):
        export_public(request(repo, manifest, sha, output))

    export_public(request(repo, manifest, sha, output, replace=True))
    assert not (output / "old.txt").exists()
    assert (output / "safe.txt").read_text(encoding="utf-8") == "new\n"


def test_replace_rolls_back_if_staging_promotion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, manifest, sha = repository(tmp_path, {"safe.txt": b"new\n"})
    output = tmp_path / "public"
    output.mkdir()
    (output / "old.txt").write_text("old\n", encoding="utf-8")
    real_replace = os.replace
    calls = 0

    def fail_second_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr(exporter_module.os, "replace", fail_second_replace)
    with pytest.raises(ExportError, match="promotion"):
        export_public(request(repo, manifest, sha, output, replace=True))

    assert (output / "old.txt").read_text(encoding="utf-8") == "old\n"
    assert not (output / "safe.txt").exists()


def test_output_cannot_overlap_source_or_be_a_symlink(tmp_path: Path) -> None:
    repo, manifest, sha = repository(tmp_path, {"safe.txt": b"safe\n"})
    with pytest.raises(ExportError, match="overlap"):
        export_public(request(repo, manifest, sha, repo / "public"))

    link = tmp_path / "linked-output"
    link.symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ExportError, match="symlink"):
        export_public(request(repo, manifest, sha, link))

def test_tracked_symlinks_and_gitlinks_are_rejected(tmp_path: Path) -> None:
    repo, manifest, sha = repository(tmp_path, {"safe.txt": b"safe\n"})
    (repo / "linked.txt").symlink_to("safe.txt")
    git(repo, "add", "linked.txt")
    git(
        repo,
        "-c",
        "user.name=Exporter Tests",
        "-c",
        "user.email=exporter@model-forge.example",
        "commit",
        "-qm",
        "symlink",
    )
    sha = git(repo, "rev-parse", "HEAD")
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    data["rules"].append(rule("linked.txt"))
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    git(repo, "add", "public-files.yaml")
    git(
        repo,
        "-c",
        "user.name=Exporter Tests",
        "-c",
        "user.email=exporter@model-forge.example",
        "commit",
        "-qm",
        "classify",
    )
    sha = git(repo, "rev-parse", "HEAD")
    with pytest.raises(ExportError, match="symlink"):
        export_public(request(repo, manifest, sha, tmp_path / "public"))

    git(repo, "rm", "linked.txt")
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{sha},module")
    data["rules"][-1] = rule("module")
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    git(repo, "add", "public-files.yaml")
    git(
        repo,
        "-c",
        "user.name=Exporter Tests",
        "-c",
        "user.email=exporter@model-forge.example",
        "commit",
        "-qm",
        "gitlink",
    )
    sha = git(repo, "rev-parse", "HEAD")
    git(repo, "update-index", "--skip-worktree", "module")
    with pytest.raises(ExportError, match="submodule"):
        export_public(request(repo, manifest, sha, tmp_path / "public"))


@pytest.mark.parametrize(
    ("tree_output", "message"),
    [
        (
            b"100644 blob " + (b"a" * 40) + b"\tdocs/file.txt\0"
            b"100644 blob " + (b"b" * 40) + b"\tdocs/file.txt\0",
            "duplicate",
        ),
        (
            b"100644 blob " + (b"a" * 40) + b"\tdocs/File.txt\0"
            b"100644 blob " + (b"b" * 40) + b"\tdocs/file.txt\0",
            "portable tracked path collision",
        ),
        (
            b"100644 blob " + (b"a" * 40) + "\tdocs/cafe\u0301.txt\0".encode(),
            "noncanonical",
        ),
        (
            b"100664 blob " + (b"a" * 40) + b"\tdocs/file.txt\0",
            "unsupported tracked Git mode",
        ),
    ],
)
def test_asserted_tree_rejects_duplicate_portable_unicode_and_mode_hazards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tree_output: bytes,
    message: str,
) -> None:
    monkeypatch.setattr(exporter_module, "_git", lambda *args, **kwargs: tree_output)

    with pytest.raises(ExportError, match=message):
        exporter_module._tree_entries(tmp_path, "a" * 40)


def test_noncanonical_unicode_is_rejected_before_portable_collision_check(
    tmp_path: Path,
) -> None:
    composed = "docs/" + unicodedata.normalize("NFC", "cafe\u0301") + ".txt"
    decomposed = "DOCS/" + unicodedata.normalize("NFD", "cafe\u0301") + ".txt"
    first = rule("first.txt")
    first["public_destination"] = composed
    second = rule("second.txt")
    second["public_destination"] = decomposed
    repo, manifest, sha = repository(
        tmp_path,
        {"first.txt": b"one\n", "second.txt": b"two\n"},
        [first, second],
    )

    with pytest.raises(ExportError, match="noncanonical"):
        export_public(request(repo, manifest, sha, tmp_path / "public"))


def test_detector_finding_or_oversize_aborts_without_partial_output(tmp_path: Path) -> None:
    repo, manifest, sha = repository(
        tmp_path,
        {"unsafe.txt": b"private host 10.0.0.7\n"},
    )
    output = tmp_path / "public"
    with pytest.raises(ExportError, match="detector"):
        export_public(request(repo, manifest, sha, output))
    assert not output.exists()

    repo2, manifest2, sha2 = repository(
        tmp_path / "second",
        {"large.txt": b"12345"},
        [rule("large.txt", max_size=4)],
    )
    with pytest.raises(ExportError, match="maximum size"):
        export_public(request(repo2, manifest2, sha2, tmp_path / "public-two"))
    assert not (tmp_path / "public-two").exists()


def test_concurrent_source_change_cannot_change_commit_bound_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, manifest, sha = repository(
        tmp_path,
        {"private.md": b"public\n"},
        [
            rule(
                "private.md",
                disposition="transform",
                transformation="sanitize_public_markdown",
            )
        ],
    )
    real_transform = exporter_module.apply_transform

    def mutate_after_transform(*args: object, **kwargs: object) -> object:
        result = real_transform(*args, **kwargs)
        (repo / "private.md").write_text("changed concurrently\n", encoding="utf-8")
        return result

    monkeypatch.setattr(exporter_module, "apply_transform", mutate_after_transform)
    output = tmp_path / "public"
    export_public(request(repo, manifest, sha, output))
    assert (output / "private.md").read_bytes() == b"public\n"


def test_invalid_manifest_unknown_transform_and_binary_fail_closed(tmp_path: Path) -> None:
    repo, manifest, sha = repository(
        tmp_path,
        {"binary.dat": b"\x00\x01"},
        [
            rule(
                "binary.dat",
                disposition="transform",
                transformation="missing_transform",
            )
        ],
    )
    with pytest.raises(ExportError, match="transform"):
        export_public(request(repo, manifest, sha, tmp_path / "public"))

    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    data["rules"][0]["disposition"] = "copy"
    data["rules"][0]["transformation"] = None
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    git(repo, "add", "public-files.yaml")
    git(
        repo,
        "-c",
        "user.name=Exporter Tests",
        "-c",
        "user.email=exporter@model-forge.example",
        "commit",
        "-qm",
        "copy binary",
    )
    sha = git(repo, "rev-parse", "HEAD")
    with pytest.raises(ExportError, match="binary|detector"):
        export_public(request(repo, manifest, sha, tmp_path / "public-two"))


def test_intersecting_recursive_globs_fail_without_tracked_intersection(tmp_path: Path) -> None:
    broad = rule("a/**/c", disposition="exclude", destination=None)
    broad["id"] = "broad"
    broad["precedence"] = 10
    peer = rule("a/b/*", disposition="exclude", destination=None)
    peer["id"] = "peer"
    peer["precedence"] = 10
    repo, manifest, sha = repository(
        tmp_path,
        {"a/x/c": b"x\n", "a/b/y": b"y\n"},
        [broad, peer],
    )

    with pytest.raises(ExportError, match="ambiguous.*source languages"):
        export_public(request(repo, manifest, sha, tmp_path / "public"))


@pytest.mark.parametrize("mask", [0o022, 0o077])
def test_output_modes_are_deterministic_across_umask(tmp_path: Path, mask: int) -> None:
    repo, manifest, sha = repository(
        tmp_path,
        {"nested/deeper/safe.txt": b"safe\n", "bin/run.sh": b"#!/bin/sh\n"},
    )
    (repo / "bin/run.sh").chmod(0o755)
    git(repo, "add", "bin/run.sh")
    git(
        repo,
        "-c",
        "user.name=Exporter Tests",
        "-c",
        "user.email=exporter@model-forge.example",
        "commit",
        "-qm",
        "mode",
    )
    sha = git(repo, "rev-parse", "HEAD")
    output = tmp_path / "public"
    previous = os.umask(mask)
    try:
        export_public(request(repo, manifest, sha, output))
    finally:
        os.umask(previous)

    assert output.stat().st_mode & 0o777 == 0o755
    assert (output / "nested").stat().st_mode & 0o777 == 0o755
    assert (output / "nested/deeper").stat().st_mode & 0o777 == 0o755
    assert (output / "nested/deeper/safe.txt").stat().st_mode & 0o777 == 0o644
    assert (output / "bin/run.sh").stat().st_mode & 0o777 == 0o755


@pytest.mark.parametrize(
    "contact",
    [
        "security@localhost",
        "security@corp.internal",
        "security@host.local",
        "security@model-forge.example",
        "not-an-email",
    ],
)
def test_public_contact_rejects_nonpublic_or_unconventional_values(
    tmp_path: Path, contact: str
) -> None:
    repo, manifest, sha = repository(tmp_path, {"safe.txt": b"safe\n"})

    with pytest.raises(ExportError, match="public contact"):
        export_public(
            request(
                repo,
                manifest,
                sha,
                tmp_path / "public",
                public_contact=contact,
            )
        )


def test_non_nfc_manifest_paths_are_rejected(tmp_path: Path) -> None:
    destination = "docs/" + unicodedata.normalize("NFD", "café") + ".txt"
    candidate = rule("safe.txt")
    candidate["public_destination"] = destination
    repo, manifest, sha = repository(tmp_path, {"safe.txt": b"safe\n"}, [candidate])

    with pytest.raises(ExportError, match="canonical"):
        export_public(request(repo, manifest, sha, tmp_path / "public"))


def test_detector_exceptions_are_wrapped_and_staging_is_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from model_forge.public_export.detectors import DetectorError

    repo, manifest, sha = repository(tmp_path, {"safe.txt": b"safe\n"})
    output = tmp_path / "public"

    def fail_detector(*args: object, **kwargs: object) -> list[object]:
        raise DetectorError("injected detector failure")

    monkeypatch.setattr(exporter_module, "scan_file", fail_detector)
    with pytest.raises(ExportError, match="detector failed"):
        export_public(request(repo, manifest, sha, output))
    assert not output.exists()
    assert not list(tmp_path.glob(".public.public-export-stage-*"))


def test_safe_symlinked_temp_ancestor_uses_resolved_output(tmp_path: Path) -> None:
    repo, manifest, sha = repository(tmp_path, {"safe.txt": b"safe\n"})
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)

    result = export_public(request(repo, manifest, sha, linked / "public"))

    assert result.output == (actual / "public").resolve()
    assert (actual / "public/safe.txt").read_text(encoding="utf-8") == "safe\n"


def test_gitleaks_evidence_must_bind_to_requested_source_sha(tmp_path: Path) -> None:
    repo, manifest, sha = repository(tmp_path, {"safe.txt": b"safe\n"})

    class WrongSourceRunner:
        def scan_git(self, source: Path, source_sha: str) -> GitleaksEvidence:
            return GitleaksEvidence(
                version="8.30.1",
                report_sha256=hashlib.sha256(b"[]").hexdigest(),
                scope="full-history-through-source-sha",
                source_sha="0" * 40,
            )

    with pytest.raises(ExportError, match="Gitleaks evidence"):
        export_public(
            request(
                repo,
                manifest,
                sha,
                tmp_path / "public",
                gitleaks_runner=WrongSourceRunner(),
            )
        )


def test_subprocess_gitleaks_uses_fixed_git_scan_and_hashes_empty_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    configs: list[bytes] = []
    ignores: list[bytes] = []
    monkeypatch.setattr(exporter_module.shutil, "which", lambda *args, **kwargs: "/bin/gitleaks")

    def fake_process(
        arguments: list[str], *, cwd: Path, timeout: int
    ) -> subprocess.CompletedProcess[bytes]:
        commands.append(arguments)
        if arguments[1] == "version":
            return subprocess.CompletedProcess(arguments, 0, b"8.30.1\n", b"")
        configs.append(Path(arguments[arguments.index("--config") + 1]).read_bytes())
        ignores.append(
            Path(arguments[arguments.index("--gitleaks-ignore-path") + 1]).read_bytes()
        )
        report = Path(arguments[arguments.index("--report-path") + 1])
        report.write_bytes(b"[]")
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(exporter_module, "_bounded_process", fake_process)
    evidence = SubprocessGitleaksRunner().scan_git(tmp_path, "a" * 40)

    assert commands[1][1:5] == ["git", "--redact", "--report-format", "json"]
    assert commands[1][-1] == f"--log-opts={'a' * 40}"
    assert "--config" in commands[1]
    assert configs == [b"[extend]\nuseDefault = true\n"]
    assert ignores == [b""]
    assert ".gitleaks.toml" not in commands[1]
    assert evidence.report_sha256 == hashlib.sha256(b"[]").hexdigest()
    assert evidence.scope == "full-history-through-source-sha"
    assert evidence.config_sha256 == hashlib.sha256(
        b"[extend]\nuseDefault = true\n"
    ).hexdigest()
    assert evidence.flags == (
        "git",
        "--redact",
        "--report-format=json",
        "--config=<exporter-owned>",
        "--gitleaks-ignore-path=<exporter-owned-empty>",
        "--log-opts=<source-sha>",
    )


@pytest.mark.parametrize(
    ("returncode", "report", "message"),
    [
        (1, b"[]", "did not pass"),
        (0, b"{}", "malformed"),
        (0, b"not-json", "malformed"),
        (0, b'[{"RuleID":"secret"}]', "credential findings"),
    ],
)
def test_subprocess_gitleaks_fails_closed_on_invalid_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    report: bytes,
    message: str,
) -> None:
    monkeypatch.setattr(exporter_module.shutil, "which", lambda *args, **kwargs: "/bin/gitleaks")

    def fake_process(
        arguments: list[str], *, cwd: Path, timeout: int
    ) -> subprocess.CompletedProcess[bytes]:
        if arguments[1] == "version":
            return subprocess.CompletedProcess(arguments, 0, b"8.30.1\n", b"")
        path = Path(arguments[arguments.index("--report-path") + 1])
        path.write_bytes(report)
        return subprocess.CompletedProcess(arguments, returncode, b"", b"")

    monkeypatch.setattr(exporter_module, "_bounded_process", fake_process)
    with pytest.raises(ExportError, match=message):
        SubprocessGitleaksRunner().scan_git(tmp_path, "a" * 40)


def test_subprocess_gitleaks_fails_closed_when_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exporter_module.shutil, "which", lambda *args, **kwargs: None)

    with pytest.raises(ExportError, match="unavailable"):
        SubprocessGitleaksRunner().scan_git(tmp_path, "a" * 40)


def test_gitleaks_process_timeout_and_output_limit_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ExportError, match="timed out"):
        exporter_module._bounded_process(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            cwd=tmp_path,
            timeout=1,
        )
    with pytest.raises(ExportError, match="output limit"):
        exporter_module._bounded_process(
            [
                sys.executable,
                "-c",
                f"import sys; sys.stdout.write('x' * "
                f"{exporter_module.MAX_GITLEAKS_OUTPUT_BYTES + 1})",
            ],
            cwd=tmp_path,
            timeout=5,
        )


def test_gitleaks_timeout_kills_descendant_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "descendant-survived"
    child = f"import time,pathlib;time.sleep(1.5);pathlib.Path({str(marker)!r}).touch()"
    script = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child!r}]);"
        "time.sleep(10)"
    )
    with pytest.raises(ExportError, match="timed out"):
        exporter_module._bounded_process(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            timeout=1,
        )
    time.sleep(1)
    assert not marker.exists()


def test_bounded_process_direct_exit_with_inherited_pipes_is_bounded(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "inherited-pipe-descendant-survived"
    process_group_file = tmp_path / "inherited-pipe-process-group"
    child = f"import time,pathlib;time.sleep(1.5);pathlib.Path({str(marker)!r}).touch()"
    script = (
        "import os,pathlib,subprocess,sys;"
        f"pathlib.Path({str(process_group_file)!r}).write_text(str(os.getpgrp()));"
        f"subprocess.Popen([sys.executable,'-c',{child!r}])"
    )
    result: list[BaseException | subprocess.CompletedProcess[bytes]] = []
    baseline_threads = set(threading.enumerate())

    def invoke() -> None:
        try:
            result.append(
                exporter_module._bounded_process(
                    [sys.executable, "-c", script],
                    cwd=tmp_path,
                    timeout=1,
                )
            )
        except BaseException as error:
            result.append(error)

    caller = threading.Thread(target=invoke, daemon=True)
    started = time.monotonic()
    caller.start()
    caller.join(timeout=4)
    elapsed = time.monotonic() - started
    if caller.is_alive() and process_group_file.exists():
        os.killpg(int(process_group_file.read_text()), signal.SIGKILL)
        caller.join(timeout=2)

    assert elapsed < 4
    assert not caller.is_alive(), "bounded process hung waiting for inherited pipe EOF"
    assert len(result) == 1
    assert isinstance(result[0], ExportError)
    assert "timed out" in str(result[0])
    deadline = time.monotonic() + 2
    while set(threading.enumerate()) - baseline_threads and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not (set(threading.enumerate()) - baseline_threads)
    time.sleep(1)
    assert not marker.exists()


def test_gitleaks_report_symlink_replacement_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exporter_module.shutil, "which", lambda *args, **kwargs: "/bin/gitleaks")

    def fake_process(
        arguments: list[str], *, cwd: Path, timeout: int
    ) -> subprocess.CompletedProcess[bytes]:
        if arguments[1] == "version":
            return subprocess.CompletedProcess(arguments, 0, b"8.30.1\n", b"")
        report = Path(arguments[arguments.index("--report-path") + 1])
        report.unlink(missing_ok=True)
        report.symlink_to(tmp_path / "attacker-report")
        (tmp_path / "attacker-report").write_bytes(b"[]")
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(exporter_module, "_bounded_process", fake_process)
    with pytest.raises(ExportError, match="report"):
        SubprocessGitleaksRunner().scan_git(tmp_path, "a" * 40)


@pytest.mark.skipif(shutil.which("gitleaks") is None, reason="Gitleaks is unavailable")
def test_repository_gitleaks_allowlist_cannot_suppress_committed_secret(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "malicious-repository"
    repo.mkdir()
    (repo / ".gitleaks.toml").write_text(
        "[[allowlists]]\ndescription = \"malicious\"\npaths = ['''.*''']\n",
        encoding="utf-8",
    )
    token = "ghp_" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:36]
    (repo / "leak.txt").write_text(f"github_token = {token}\n", encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Exporter Tests",
        "-c",
        "user.email=exporter@example.com",
        "commit",
        "-qm",
        "malicious allowlist",
    )

    with pytest.raises(ExportError, match="did not pass|credential findings"):
        SubprocessGitleaksRunner().scan_git(repo, git(repo, "rev-parse", "HEAD"))


def test_rule_specific_raw_key_suppression_requires_trusted_copy_classification(
    tmp_path: Path,
) -> None:
    approved = rule("src/example.py")
    approved["content_classification"] = "trusted-source-code"
    approved["detector_suppressions"] = ["benchmark.raw-key"]
    repo, manifest, sha = repository(
        tmp_path,
        {"src/example.py": b'payload = {"prompt": value}\n'},
        [approved],
    )

    export_public(request(repo, manifest, sha, tmp_path / "public"))
    assert (tmp_path / "public/src/example.py").exists()

    unsafe = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    unsafe["rules"][0]["detector_suppressions"] = ["network.private-ipv4"]
    manifest.write_text(yaml.safe_dump(unsafe, sort_keys=False), encoding="utf-8")
    git(repo, "add", "public-files.yaml")
    git(
        repo,
        "-c",
        "user.name=Exporter Tests",
        "-c",
        "user.email=exporter@model-forge.example",
        "commit",
        "-qm",
        "unsafe suppression",
    )
    sha = git(repo, "rev-parse", "HEAD")
    with pytest.raises(ExportError, match="detector suppression"):
        export_public(request(repo, manifest, sha, tmp_path / "unsafe"))


def test_raw_benchmark_data_remains_blocked_without_trusted_source_policy(
    tmp_path: Path,
) -> None:
    repo, manifest, sha = repository(
        tmp_path,
        {"evidence/results.json": b'{"question": "private benchmark item"}\n'},
    )

    with pytest.raises(ExportError, match="benchmark.raw-key"):
        export_public(request(repo, manifest, sha, tmp_path / "public"))


@pytest.mark.skipif(shutil.which("gitleaks") is None, reason="Gitleaks is unavailable")
@pytest.mark.private_source_only
def test_full_committed_repository_exports_twice_with_actual_gitleaks(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        pytest.skip("canonical integration requires a clean tracked working tree")
    sha = git(root, "rev-parse", "HEAD")
    outputs = [tmp_path / "canonical-one", tmp_path / "canonical-two"]
    results = [
        export_public(
            ExportRequest(
                source=root,
                output=output,
                manifest=root / "tools/public_export/public-files.yaml",
                source_sha=sha,
            )
        )
        for output in outputs
    ]

    assert results[0].payload_tree_sha256 == results[1].payload_tree_sha256
    manifests = [
        json.loads(
            (output / "PUBLIC_EXPORT_MANIFEST.json").read_text(encoding="utf-8")
        )
        for output in outputs
    ]
    assert manifests[0] == manifests[1]
    assert (outputs[0] / "containers/build/docker-compose.yml").is_file()
    assert results[0].file_count == len(manifests[0]["files"])
    for record in manifests[0]["files"]:
        relative = record["output_path"]
        first_path = outputs[0] / relative
        second_path = outputs[1] / relative
        assert first_path.read_bytes() == second_path.read_bytes()
        assert f"{first_path.stat().st_mode & 0o777:06o}" == record["mode"]
        assert f"{second_path.stat().st_mode & 0o777:06o}" == record["mode"]


def test_canonical_clean_tree_export_uses_specific_recipe_readme_transform(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    fixture = tmp_path / "canonical"
    fixture.mkdir()
    canonical_paths = [
        "recipes/README.md",
        "recipes/qwen3.8-27b/r3-nvfp4.yaml",
    ]
    for relative in canonical_paths:
        source = root / relative
        target = fixture / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    broad = rule(
        "recipes/**",
        disposition="transform",
        transformation="sanitize_and_validate_recipe",
        max_size=65536,
    )
    broad["id"] = "recipes"
    broad["precedence"] = 100
    readme = rule(
        "recipes/README.md",
        disposition="transform",
        transformation="sanitize_public_markdown",
        max_size=65536,
    )
    readme["id"] = "recipes-readme"
    readme["precedence"] = 200
    readme["resolves"] = ["recipes"]
    manifest = fixture / "public-files.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "rules": [rule("public-files.yaml"), broad, readme],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    git(fixture, "init", "-q")
    git(fixture, "add", ".")
    git(
        fixture,
        "-c",
        "user.name=Exporter Tests",
        "-c",
        "user.email=exporter@example.com",
        "commit",
        "-qm",
        "canonical fixture",
    )
    sha = git(fixture, "rev-parse", "HEAD")

    export_public(
        request(
            fixture,
            manifest,
            sha,
            tmp_path / "canonical-public",
        )
    )

    attestation = json.loads(
        (tmp_path / "canonical-public/PUBLIC_EXPORT_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    transforms = {
        item["output_path"]: item["transform_id"] for item in attestation["files"]
    }
    assert transforms["recipes/README.md"].startswith("sanitize_public_markdown:")
    assert transforms["recipes/qwen3.8-27b/r3-nvfp4.yaml"].startswith(
        "sanitize_and_validate_recipe:"
    )


READER_OID = "a" * 40
READER_HEADER = f"{READER_OID} blob 1\n".encode("ascii")


def test_cat_file_batch_rejects_single_object_over_payload_limit() -> None:
    size = 33 * 1024 * 1024

    with pytest.raises(ExportError, match="payload limit exceeded"):
        exporter_module._plan_cat_file_batch([READER_OID], [size], 0)


def test_cat_file_batch_accepts_response_exactly_at_payload_limit() -> None:
    size = exporter_module.MAX_CAT_FILE_BATCH_PAYLOAD_BYTES - 56

    end, expected_bytes = exporter_module._plan_cat_file_batch([READER_OID], [size], 0)

    assert end == 1
    assert expected_bytes == exporter_module.MAX_CAT_FILE_BATCH_PAYLOAD_BYTES


def test_cat_file_batch_rejects_invalid_size_and_index_bounds() -> None:
    for size in (-1, True):
        with pytest.raises(ExportError, match="invalid object size"):
            exporter_module._plan_cat_file_batch([READER_OID], [size], 0)
    with pytest.raises(ExportError, match="payload limit exceeded"):
        exporter_module._plan_cat_file_batch([READER_OID], [1 << 128], 0)
    with pytest.raises(ExportError, match="count mismatch"):
        exporter_module._plan_cat_file_batch([READER_OID], [], 0)
    for start in (-1, 1):
        with pytest.raises(ExportError, match="start is out of bounds"):
            exporter_module._plan_cat_file_batch([READER_OID], [1], start)


def test_git_blob_reader_reads_twenty_thousand_objects_in_bounded_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    count = 20_000
    repo, object_ids, expected = blob_objects(tmp_path, count)
    real_process = exporter_module._bounded_process
    invocations: list[tuple[str, int, int]] = []

    def recording_process(
        arguments: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        payload = kwargs.get("payload")
        stdout_limit = kwargs.get("stdout_limit")
        assert isinstance(payload, bytes), "the reader must feed stdin through the bounded runner"
        assert isinstance(stdout_limit, int), "the reader must bound each batch response"
        invocations.append((arguments[-1], len(payload), stdout_limit))
        return real_process(arguments, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(exporter_module, "_bounded_process", recording_process)
    started = time.monotonic()
    blobs = exporter_module._read_git_blobs(repo, object_ids)
    elapsed = time.monotonic() - started

    assert blobs == expected
    batches = -(-count // exporter_module.MAX_CAT_FILE_BATCH_OIDS)
    modes = [mode for mode, _, _ in invocations]
    assert modes.count("--batch-check") == batches
    assert modes.count("--batch") == batches
    assert len(modes) == 2 * batches
    assert max(size for _, size, _ in invocations) <= (
        exporter_module.MAX_CAT_FILE_BATCH_INPUT_BYTES
    )
    assert sum(size for _, size, _ in invocations) == 2 * count * 41
    assert max(
        limit for mode, _, limit in invocations if mode == "--batch"
    ) <= exporter_module.MAX_CAT_FILE_BATCH_PAYLOAD_BYTES
    assert elapsed < 180


def test_git_blob_reader_enforces_tracked_file_and_object_id_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid input must fail before any process starts")

    monkeypatch.setattr(exporter_module, "_bounded_process", forbidden)
    with pytest.raises(ExportError, match="tracked file limit exceeded"):
        exporter_module._read_git_blobs(
            tmp_path, [READER_OID] * (exporter_module.MAX_TRACKED_FILES + 1)
        )
    for invalid in ["a" * 39, "g" * 40, f"{READER_OID}\n{READER_OID}", "--batch"]:
        with pytest.raises(ExportError, match="malformed object ID"):
            exporter_module._read_git_blobs(tmp_path, [invalid])
    assert exporter_module._read_git_blobs(tmp_path, []) == []


@pytest.mark.parametrize(
    ("check_output", "message"),
    [
        (b"", "malformed header"),
        (f"{'b' * 40} blob 1\n".encode("ascii"), "unexpected object"),
        (f"{READER_OID} tree 1\n".encode("ascii"), "unexpected object"),
        (f"{READER_OID} missing\n".encode("ascii"), "malformed header"),
        (f"{READER_OID} blob 007\n".encode("ascii"), "malformed header"),
        (f"{READER_OID} blob -1\n".encode("ascii"), "malformed header"),
        (f"{READER_OID} blob 1".encode("ascii"), "malformed header"),
        (READER_HEADER + READER_HEADER, "malformed header"),
        (f"{READER_OID} blob {'1' * 200}\n".encode("ascii"), "malformed header"),
        (
            f"{READER_OID} blob {exporter_module.MAX_TOTAL_SOURCE_BYTES + 1}\n".encode("ascii"),
            "source byte limit exceeded",
        ),
    ],
)
def test_git_blob_reader_rejects_malformed_or_unexpected_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    check_output: bytes,
    message: str,
) -> None:
    monkeypatch.setattr(
        exporter_module, "_bounded_process", cat_file_reader(check=check_output)
    )

    with pytest.raises(ExportError, match=message):
        exporter_module._read_git_blobs(tmp_path, [READER_OID])


@pytest.mark.parametrize(
    ("batch_output", "message"),
    [
        (b"", "unexpected response length"),
        (READER_HEADER + b"x", "unexpected response length"),
        (READER_HEADER + b"x\nextra", "unexpected response length"),
        (READER_HEADER + b"xx", "truncated object"),
        (f"{'b' * 40} blob 1\n".encode("ascii") + b"x\n", "unexpected object"),
    ],
)
def test_git_blob_reader_rejects_truncated_or_oversized_batch_responses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_output: bytes,
    message: str,
) -> None:
    monkeypatch.setattr(
        exporter_module,
        "_bounded_process",
        cat_file_reader(check=READER_HEADER, batch=batch_output),
    )

    with pytest.raises(ExportError, match=message):
        exporter_module._read_git_blobs(tmp_path, [READER_OID])


def test_git_blob_reader_fails_closed_on_nonzero_reader_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        exporter_module,
        "_bounded_process",
        cat_file_reader(check=READER_HEADER, returncode=1),
    )

    with pytest.raises(ExportError, match="Git object reader failed"):
        exporter_module._read_git_blobs(tmp_path, [READER_OID])


def test_git_blob_reader_enforces_total_time_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exporter_module, "CAT_FILE_TOTAL_TIMEOUT_SECONDS", 0)

    with pytest.raises(ExportError, match="time budget"):
        exporter_module._read_git_blobs(tmp_path, [READER_OID])


def test_git_object_reader_timeout_cleans_up_blocked_feeder_and_children(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "reader-descendant-survived"
    child = f"import time,pathlib;time.sleep(1.5);pathlib.Path({str(marker)!r}).touch()"
    script = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child!r}]);"
        "time.sleep(10)"
    )
    started = time.monotonic()
    with pytest.raises(ExportError, match="timed out"):
        exporter_module._bounded_process(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            timeout=1,
            tool="Git object reader",
            payload=b"x" * 4_000_000,
            stdout_limit=1024,
        )

    assert time.monotonic() - started < 8
    time.sleep(1)
    assert not marker.exists()


def test_export_maps_batched_blob_reads_to_exact_committed_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exporter_module, "MAX_CAT_FILE_BATCH_OIDS", 2)
    files = {
        f"docs/file-{index}.md": f"committed content {index}\n".encode()
        for index in range(7)
    }
    repo, manifest, sha = repository(tmp_path, files)

    export_public(request(repo, manifest, sha, tmp_path / "public"))

    for relative, data in files.items():
        assert (tmp_path / "public" / relative).read_bytes() == data


def test_public_export_cli_cannot_assert_gitleaks_pass_and_prints_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import model_forge.cli as cli

    captured: list[ExportRequest] = []

    def fake_export(export_request: ExportRequest) -> ExportResult:
        captured.append(export_request)
        return ExportResult("d" * 64, 2, export_request.output, export_request.dry_run)

    monkeypatch.setattr(cli, "export_public", fake_export)
    result = cli.main(
        [
            "public-export",
            "--source",
            str(tmp_path / "source"),
            "--output",
            str(tmp_path / "output"),
            "--manifest",
            str(tmp_path / "manifest.yaml"),
            "--source-sha",
            "a" * 40,
            "--replace",
            "--dry-run",
        ]
    )

    assert result == 0
    assert captured[0].gitleaks_runner is None
    assert captured[0].replace is True
    assert captured[0].dry_run is True
    assert "d" * 64 in capsys.readouterr().out


def test_public_export_cli_reports_fail_closed_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import model_forge.cli as cli

    def fail(_: ExportRequest) -> ExportResult:
        raise ExportError("unsafe export")

    monkeypatch.setattr(cli, "export_public", fail)
    result = cli.main(
        [
            "public-export",
            "--source",
            str(tmp_path / "source"),
            "--output",
            str(tmp_path / "output"),
            "--manifest",
            str(tmp_path / "manifest.yaml"),
            "--source-sha",
            "a" * 40,
        ]
    )

    assert result == 2
    assert capsys.readouterr().err == "Public export refused: unsafe export\n"
