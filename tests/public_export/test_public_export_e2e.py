from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import zipfile
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from model_forge.public_export import verifier as verifier_module
from model_forge.public_export.exporter import (
    ExportRequest,
    GitleaksEvidence,
    export_public,
)
from model_forge.public_export.verifier import (
    FilesystemGitleaksEvidence,
    PublicVerifyError,
    PublicVerifyRequest,
    SubprocessFilesystemGitleaksRunner,
    _package_build_install_smoke,
    _run_bounded,
    _validate_markdown_links,
    verify_public_export,
)


class ExportGitleaks:
    def scan_git(self, source: Path, source_sha: str) -> GitleaksEvidence:
        return GitleaksEvidence(
            version="8.30.1",
            report_sha256=hashlib.sha256(b"[]").hexdigest(),
            scope="full-history-through-source-sha",
            source_sha=source_sha,
        )


class FixtureFilesystemGitleaks:
    def scan_directory(self, root: Path) -> FilesystemGitleaksEvidence:
        for path in root.rglob("*"):
            if path.is_file() and b"sk-live-" in path.read_bytes():
                raise PublicVerifyError("Gitleaks found credentials")
        return FilesystemGitleaksEvidence(
            version="8.30.1",
            report_sha256=hashlib.sha256(b"[]").hexdigest(),
        )


class FixtureProjectChecks:
    def run(self, root: Path) -> None:
        assert (root / "pyproject.toml").is_file()


class MutatingProjectChecks:
    def run(self, root: Path) -> None:
        (root / "build").mkdir()
        (root / "fixture.egg-info").mkdir()
        (root / ".pytest_cache").mkdir()
        cache = root / "src/fixture_package/__pycache__"
        cache.mkdir()
        (cache / "fixture.pyc").write_bytes(b"cache")


@dataclass(frozen=True)
class CommittedExport:
    root: Path
    source: Path
    source_sha: str
    wheelhouse: Path


def _fixture_wheel(
    wheelhouse: Path,
    distribution: str,
    version: str,
    files: dict[str, str],
) -> Path:
    normalized = distribution.replace("-", "_")
    dist_info = f"{normalized}-{version}.dist-info"
    wheel = wheelhouse / f"{normalized}-{version}-py3-none-any.whl"
    entries = {
        **files,
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {distribution}\n"
            f"Version: {version}\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: public-export-tests\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
        f"{dist_info}/RECORD": "",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, body in entries.items():
            archive.writestr(name, body)
    return wheel


def _fixture_wheelhouse(tmp_path: Path) -> Path:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    backend = """\
from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    name = "public_verifier_fixture"
    version = "0.0.1"
    filename = f"{name}-{version}-py3-none-any.whl"
    target = Path(wheel_directory) / filename
    dist_info = f"{name}-{version}.dist-info"
    entries = {
        "fixture_package/__init__.py": Path("src/fixture_package/__init__.py").read_bytes(),
        "fixture_package/cli.py": Path("src/fixture_package/cli.py").read_bytes(),
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\\n"
            "Name: public-verifier-fixture\\n"
            "Version: 0.0.1\\n"
            "Requires-Dist: fixture-runtime==1.0\\n"
        ).encode(),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\\n"
            "Generator: fixture-backend\\n"
            "Root-Is-Purelib: true\\n"
            "Tag: py3-none-any\\n"
        ).encode(),
        f"{dist_info}/entry_points.txt": (
            "[console_scripts]\\nfixture-cli = fixture_package.cli:main\\n"
        ).encode(),
    }
    rows = []
    for path, data in entries.items():
        digest = hashlib.sha256(data).digest()
        import base64
        encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        rows.append((path, f"sha256={encoded}", str(len(data))))
    rows.append((f"{dist_info}/RECORD", "", ""))
    record = io.StringIO()
    csv.writer(record, lineterminator="\\n").writerows(rows)
    entries[f"{dist_info}/RECORD"] = record.getvalue().encode()
    with zipfile.ZipFile(target, "w") as archive:
        for path, data in entries.items():
            archive.writestr(path, data)
    return filename
"""
    _fixture_wheel(
        wheelhouse,
        "fixture-backend",
        "1.0",
        {"fixture_backend/__init__.py": backend},
    )
    _fixture_wheel(
        wheelhouse,
        "fixture-runtime",
        "1.0",
        {
            "fixture_runtime/__init__.py": (
                "def validate(value: str) -> bool:\n"
                "    return value == 'ok'\n"
            )
        },
    )
    return wheelhouse


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _rule(path: str) -> dict[str, object]:
    recipe = path.startswith("recipes/") and path.endswith((".yaml", ".yml"))
    transformed_markdown = path == "README.md"
    source_code = path.startswith(("src/", "tests/")) and path.endswith((".py", ".pyi"))
    result: dict[str, object] = {
        "id": path.replace("/", "-"),
        "source": path,
        "disposition": "transform" if recipe or transformed_markdown else "copy",
        "public_destination": "{source}",
        "reason": "Committed public verifier fixture.",
        "transformation": (
            "sanitize_and_validate_recipe"
            if recipe
            else "sanitize_public_markdown"
            if transformed_markdown
            else None
        ),
        "owner": "tests",
        "max_size_bytes": 1_048_576,
        "generated": False,
        "precedence": 100,
    }
    if source_code:
        result["content_classification"] = "trusted-source-code"
        result["detector_suppressions"] = ["benchmark.raw-key"]
    return result


@pytest.fixture
def committed_export(tmp_path: Path) -> CommittedExport:
    source = tmp_path / "source"
    wheelhouse = _fixture_wheelhouse(tmp_path)
    files = {
        ".gitleaks.toml": "[extend]\nuseDefault = true\n",
        "README.md": "# Fixture\n\n[Guide](docs/guide.md)\n",
        "docs/guide.md": "# Guide\n",
        "data/example.json": '{"ok":true}\n',
        "data/example.yaml": "ok: true\n",
        "pyproject.toml": """\
[build-system]
requires = ["fixture-backend==1.0"]
build-backend = "fixture_backend"
[project]
name = "public-verifier-fixture"
version = "0.0.1"
requires-python = ">=3.11"
dependencies = ["fixture-runtime==1.0"]
[project.scripts]
fixture-cli = "fixture_package.cli:main"
""",
        "recipes/demo.yaml": """\
schema_version: "2.0"
name: demo
family: demo
source:
  model_id: org/model
  revision: "1111111111111111111111111111111111111111"
transforms: []
validation:
  max_refusal_leakage: 0.01
runtime:
  kv_dtype: bf16
  context_length: 128
outputs:
  artifact_kind: bf16
  publication:
    github: HangGlidersRule/model-forge
""",
        "src/fixture_package/__init__.py": (
            'question = "public schema field"\n\n'
        ),
        "src/fixture_package/cli.py": (
            "import argparse\n\n"
            "import fixture_runtime\n\n"
            "def main() -> int:\n"
            '    parser = argparse.ArgumentParser(prog="fixture-cli")\n'
            '    parser.add_argument("--value", default="ok")\n'
            "    options = parser.parse_args()\n"
            "    assert fixture_runtime.validate(options.value)\n"
            "    return 0\n"
        ),
    }
    source.mkdir()
    for relative, text in files.items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    manifest = source / "tools/public_export/public-files.yaml"
    manifest.parent.mkdir(parents=True)
    all_paths = [*files, "tools/public_export/public-files.yaml"]
    manifest.write_text(
        yaml.safe_dump(
            {"version": 1, "rules": [_rule(path) for path in all_paths]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    git(source, "init", "-q")
    git(source, "add", ".")
    git(
        source,
        "-c",
        "user.name=Verifier Tests",
        "-c",
        "user.email=verifier@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    source_sha = git(source, "rev-parse", "HEAD")
    output = tmp_path / "public"
    export_public(
        ExportRequest(
            source=source,
            output=output,
            manifest=manifest,
            source_sha=source_sha,
            gitleaks_runner=ExportGitleaks(),
        )
    )
    return CommittedExport(output, source, source_sha, wheelhouse)


def _request(export: CommittedExport) -> PublicVerifyRequest:
    return PublicVerifyRequest(
        root=export.root,
        source_sha=export.source_sha,
        source_repo=export.source,
        manifest=Path("tools/public_export/public-files.yaml"),
        wheelhouse=export.wheelhouse,
        gitleaks_runner=FixtureFilesystemGitleaks(),
        source_gitleaks_runner=ExportGitleaks(),
        project_checks_runner=FixtureProjectChecks(),
    )


def _attestation(root: Path) -> dict[str, object]:
    return json.loads((root / "PUBLIC_EXPORT_MANIFEST.json").read_text(encoding="utf-8"))


def _write_attestation(root: Path, data: dict[str, object]) -> None:
    (root / "PUBLIC_EXPORT_MANIFEST.json").write_text(
        json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _reattest(root: Path, relative: str) -> None:
    data = _attestation(root)
    records = data["files"]
    assert isinstance(records, list)
    for record in records:
        assert isinstance(record, dict)
        if record["output_path"] == relative:
            record["output_sha256"] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["output_path"])):
        digest.update(str(record["output_path"]).encode())
        digest.update(b"\0")
        digest.update(str(record["mode"]).encode())
        digest.update(b"\0")
        digest.update(str(record["output_sha256"]).encode())
        digest.update(b"\n")
    data["payload_tree_sha256"] = digest.hexdigest()
    _write_attestation(root, data)


def test_committed_fixture_export_verifies_end_to_end(
    committed_export: CommittedExport,
) -> None:
    result = verify_public_export(_request(committed_export))
    assert result.file_count == 10
    assert result.source_sha == committed_export.source_sha
    assert {item.name for item in result.wheelhouse_evidence} == {
        "fixture_backend-1.0-py3-none-any.whl",
        "fixture_runtime-1.0-py3-none-any.whl",
    }


def test_fresh_export_is_single_root_ready_and_has_no_private_metadata(
    committed_export: CommittedExport,
    tmp_path: Path,
) -> None:
    forbidden = {".git", ".hermes", "private", "raw"}
    inventory = {
        path.relative_to(committed_export.root).as_posix()
        for path in committed_export.root.rglob("*")
    }
    assert inventory
    assert not any(forbidden & set(Path(relative).parts) for relative in inventory)

    clean_root = tmp_path / "future-public-root"
    shutil.copytree(committed_export.root, clean_root)
    git(clean_root, "init", "-q")
    git(clean_root, "add", ".")
    git(
        clean_root,
        "-c",
        "user.name=Public Staging Test",
        "-c",
        "user.email=staging@example.com",
        "commit",
        "-qm",
        "Initial public root",
    )
    assert git(clean_root, "rev-list", "--count", "HEAD") == "1"
    assert git(clean_root, "rev-list", "--parents", "-n", "1", "HEAD").count(" ") == 0
    assert not git(clean_root, "status", "--porcelain")


def test_two_fresh_exports_are_byte_and_mode_identical(
    committed_export: CommittedExport,
    tmp_path: Path,
) -> None:
    second = tmp_path / "public-second"
    export_public(
        ExportRequest(
            source=committed_export.source,
            output=second,
            manifest=committed_export.source / "tools/public_export/public-files.yaml",
            source_sha=committed_export.source_sha,
            gitleaks_runner=ExportGitleaks(),
        )
    )

    def inventory(root: Path) -> dict[str, tuple[int, str | None]]:
        return {
            path.relative_to(root).as_posix(): (
                path.stat().st_mode & 0o777,
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            )
            for path in root.rglob("*")
        }

    assert inventory(second) == inventory(committed_export.root)
    verify_public_export(replace(_request(committed_export), root=second))


def test_repeated_verification_uses_disposable_copy_and_leaves_export_unchanged(
    committed_export: CommittedExport,
) -> None:
    request = replace(
        _request(committed_export),
        project_checks_runner=MutatingProjectChecks(),
    )
    before = {
        path.relative_to(committed_export.root).as_posix(): (
            path.stat().st_mode & 0o777,
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        )
        for path in committed_export.root.rglob("*")
    }

    verify_public_export(request)
    verify_public_export(request)

    after = {
        path.relative_to(committed_export.root).as_posix(): (
            path.stat().st_mode & 0o777,
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        )
        for path in committed_export.root.rglob("*")
    }
    assert after == before
    assert not any(
        path.name == "build"
        or path.name.endswith(".egg-info")
        or path.name in {".pytest_cache", ".ruff_cache", ".mypy_cache", "__pycache__"}
        or path.suffix == ".pyc"
        for path in committed_export.root.rglob("*")
    )


def test_verification_fails_if_original_export_changes_during_checks(
    committed_export: CommittedExport,
) -> None:
    class OriginalMutatingChecks:
        def run(self, root: Path) -> None:
            del root
            (committed_export.root / "unlisted.txt").write_text("mutation\n")

    request = replace(
        _request(committed_export),
        project_checks_runner=OriginalMutatingChecks(),
    )
    with pytest.raises(PublicVerifyError, match="changed during verification"):
        verify_public_export(request)


def test_package_smoke_never_contacts_a_package_index(
    committed_export: CommittedExport,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contacted = threading.Event()
    poisoned_package = tmp_path / "poisoned/fixture_package"
    poisoned_package.mkdir(parents=True)
    (poisoned_package / "__init__.py").write_text("", encoding="utf-8")
    (poisoned_package / "cli.py").write_text(
        'raise RuntimeError("working tree package was imported")\n',
        encoding="utf-8",
    )

    class IndexHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            contacted.set()
            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), IndexHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.delenv("PIP_NO_INDEX", raising=False)
    monkeypatch.setenv("PIP_INDEX_URL", f"http://127.0.0.1:{server.server_port}/simple")
    monkeypatch.setenv("PYTHONPATH", str(poisoned_package.parent))
    try:
        verify_public_export(_request(committed_export))
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert not contacted.is_set()


@pytest.mark.parametrize(
    ("declared_requirements", "declared_backend"),
    [
        ('["setuptools>=77"]', "missing_backend.build"),
        ('["missing-backend==0"]', "setuptools.build_meta"),
    ],
)
def test_missing_declared_build_backend_or_dependency_fails_with_actionable_error(
    committed_export: CommittedExport,
    declared_requirements: str,
    declared_backend: str,
) -> None:
    root = committed_export.root
    pyproject = root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    pyproject.write_text(
        text.replace(
            'requires = ["fixture-backend==1.0"]\nbuild-backend = "fixture_backend"',
            f"requires = {declared_requirements}\nbuild-backend = {declared_backend!r}",
        ),
        encoding="utf-8",
    )
    _reattest(root, "pyproject.toml")
    with pytest.raises(
        PublicVerifyError,
        match=rf"{re.escape(declared_backend)}.*{re.escape(declared_requirements[2:-2])}",
    ):
        _package_build_install_smoke(root, committed_export.wheelhouse)


@pytest.mark.parametrize(
    ("case", "relative", "payload"),
    [
        ("api-key", "docs/guide.md", "token = sk-live-abcdefghijklmnopqrstuvwxyz123456\n"),
        ("private-ip", "docs/guide.md", "host = 10.23.45.67\n"),
        ("host-path", "docs/guide.md", "workspace = /Users/operator/private\n"),
        ("raw-gpqa", "data/example.json", '{"question":"private","answer":"secret"}\n'),
        ("broken-markdown", "README.md", "[missing](docs/missing.md)\n"),
        ("invalid-json", "data/example.json", '{"broken":\n'),
        ("invalid-yaml", "data/example.yaml", "key: [broken\n"),
        ("package-failure", "pyproject.toml", "[project\n"),
        ("oversized", "docs/guide.md", "<oversized>"),
        ("binary", "docs/guide.md", "\u0000binary\n"),
    ],
)
def test_rejects_reattested_unsafe_payloads(
    committed_export: CommittedExport,
    case: str,
    relative: str,
    payload: str,
) -> None:
    root = committed_export.root
    (root / relative).write_bytes(
        b"x" * 1_048_577 if payload == "<oversized>" else payload.encode()
    )
    _reattest(root, relative)
    with pytest.raises(PublicVerifyError, match=".+"):
        verify_public_export(_request(committed_export))


def test_rejects_unlisted_extra_and_missing_files(
    committed_export: CommittedExport,
) -> None:
    root = committed_export.root
    (root / "extra.txt").write_text("extra\n")
    with pytest.raises(PublicVerifyError, match="unlisted"):
        verify_public_export(_request(committed_export))
    (root / "extra.txt").unlink()
    (root / "docs/guide.md").unlink()
    with pytest.raises(PublicVerifyError, match="missing|cannot be read"):
        verify_public_export(_request(committed_export))


def test_rejects_hash_mode_symlink_and_hardlink_anomalies(
    committed_export: CommittedExport,
) -> None:
    root = committed_export.root
    guide = root / "docs/guide.md"
    guide.write_text("changed\n")
    with pytest.raises(PublicVerifyError, match="hash|bytes"):
        verify_public_export(_request(committed_export))
    guide.write_text("# Guide\n")
    guide.chmod(0o600)
    with pytest.raises(PublicVerifyError, match="mode"):
        verify_public_export(_request(committed_export))
    guide.chmod(0o644)
    guide.unlink()
    guide.symlink_to("../README.md")
    with pytest.raises(PublicVerifyError, match="symlink|regular"):
        verify_public_export(_request(committed_export))
    guide.unlink()
    os.link(root / "README.md", guide)
    with pytest.raises(PublicVerifyError, match="hardlink"):
        verify_public_export(_request(committed_export))


@pytest.mark.parametrize("name", ["readme.md", "RÉADME.md"])
def test_rejects_case_and_unicode_collisions(
    committed_export: CommittedExport,
    name: str,
) -> None:
    root = committed_export.root
    if name == "readme.md":
        data = _attestation(root)
        records = data["files"]
        assert isinstance(records, list)
        duplicate = dict(next(record for record in records if record["output_path"] == "README.md"))
        duplicate["output_path"] = name
        records.append(duplicate)
        _write_attestation(root, data)
    else:
        (root / name).write_text("collision\n", encoding="utf-8")
    with pytest.raises(PublicVerifyError, match="collision|canonical"):
        verify_public_export(_request(committed_export))


def test_rejects_malformed_or_self_asserted_attestation(
    committed_export: CommittedExport,
) -> None:
    root = committed_export.root
    manifest = root / "PUBLIC_EXPORT_MANIFEST.json"
    manifest.write_text('{"schema":"first","schema":"second"}\n')
    with pytest.raises(PublicVerifyError, match="duplicate|attestation"):
        verify_public_export(_request(committed_export))


@pytest.mark.parametrize("name", [".gitleaks.toml", ".gitleaksignore"])
def test_source_controlled_gitleaks_policy_cannot_bypass_scan(
    committed_export: CommittedExport,
    name: str,
) -> None:
    root = committed_export.root
    target = root / name
    target.write_text("[allowlist]\npaths = ['.*']\n" if name.endswith("toml") else "*\n")
    if name == ".gitleaks.toml":
        _reattest(root, name)
    with pytest.raises(PublicVerifyError, match="Gitleaks|unlisted|policy|deterministic"):
        verify_public_export(_request(committed_export))


def test_source_sha_is_bound_externally(committed_export: CommittedExport) -> None:
    request = _request(committed_export)
    with pytest.raises(PublicVerifyError, match="source SHA"):
        verify_public_export(
            PublicVerifyRequest(
                root=request.root,
                source_sha="f" * 40,
                source_repo=request.source_repo,
                manifest=request.manifest,
                wheelhouse=request.wheelhouse,
                gitleaks_runner=request.gitleaks_runner,
                source_gitleaks_runner=request.source_gitleaks_runner,
                project_checks_runner=request.project_checks_runner,
            )
        )


@pytest.mark.parametrize("field", ["source_id", "input_sha256", "gitleaks"])
def test_fabricated_provenance_is_rejected(
    committed_export: CommittedExport,
    field: str,
) -> None:
    data = _attestation(committed_export.root)
    if field == "gitleaks":
        gitleaks = data["gitleaks"]
        assert isinstance(gitleaks, dict)
        gitleaks["report_sha256"] = "f" * 64
    else:
        records = data["files"]
        assert isinstance(records, list)
        record = records[0]
        assert isinstance(record, dict)
        record[field] = "fabricated" if field == "source_id" else "f" * 64
    _write_attestation(committed_export.root, data)
    with pytest.raises(PublicVerifyError, match="provenance|Gitleaks|source"):
        verify_public_export(_request(committed_export))


def test_forged_transformed_readme_bytes_are_rejected_even_when_reattested(
    committed_export: CommittedExport,
) -> None:
    readme = committed_export.root / "README.md"
    readme.write_text("# Forged but internally consistent\n", encoding="utf-8")
    _reattest(committed_export.root, "README.md")

    with pytest.raises(PublicVerifyError, match="provenance|deterministic export plan"):
        verify_public_export(_request(committed_export))


def test_forged_transform_id_is_rejected(
    committed_export: CommittedExport,
) -> None:
    data = _attestation(committed_export.root)
    records = data["files"]
    assert isinstance(records, list)
    readme = next(record for record in records if record["output_path"] == "README.md")
    readme["transform_id"] = "sanitize_public_markdown:v999"
    _write_attestation(committed_export.root, data)

    with pytest.raises(PublicVerifyError, match="provenance"):
        verify_public_export(_request(committed_export))


def test_forged_semantic_recipe_linkage_is_rejected(
    committed_export: CommittedExport,
) -> None:
    data = _attestation(committed_export.root)
    records = data["files"]
    assert isinstance(records, list)
    recipe = next(record for record in records if record["output_path"] == "recipes/demo.yaml")
    linkage = recipe["semantic_recipe"]
    assert isinstance(linkage, dict)
    linkage["source_sha256"] = "f" * 64
    linkage["identity_preserved"] = False
    _write_attestation(committed_export.root, data)

    with pytest.raises(PublicVerifyError, match="provenance"):
        verify_public_export(_request(committed_export))


def test_missing_wheelhouse_and_wrong_backend_version_are_actionable(
    committed_export: CommittedExport,
) -> None:
    request = _request(committed_export)
    missing = committed_export.wheelhouse.parent / "missing-wheelhouse"
    with pytest.raises(PublicVerifyError, match="wheelhouse.*missing"):
        verify_public_export(
            PublicVerifyRequest(
                root=request.root,
                source_sha=request.source_sha,
                source_repo=request.source_repo,
                manifest=request.manifest,
                wheelhouse=missing,
                gitleaks_runner=request.gitleaks_runner,
                source_gitleaks_runner=request.source_gitleaks_runner,
                project_checks_runner=request.project_checks_runner,
            )
        )
    (committed_export.wheelhouse / "fixture_backend-1.0-py3-none-any.whl").unlink()
    _fixture_wheel(
        committed_export.wheelhouse,
        "fixture-backend",
        "0.9",
        {"fixture_backend/__init__.py": "raise RuntimeError('wrong backend')\n"},
    )
    with pytest.raises(PublicVerifyError, match="fixture-backend==1.0"):
        verify_public_export(_request(committed_export))


def test_bounded_subprocess_kills_descendants_retaining_pipes(tmp_path: Path) -> None:
    marker = tmp_path / "descendant-marker"
    child = (
        "import pathlib,time;"
        "time.sleep(1);"
        f"pathlib.Path({str(marker)!r}).write_text('escaped')"
    )
    parent = (
        "import subprocess,sys;"
        f"subprocess.Popen([sys.executable,'-c',{child!r}]);"
        "sys.exit(0)"
    )
    before = {thread.ident for thread in threading.enumerate()}
    with pytest.raises(PublicVerifyError, match="timed out"):
        _run_bounded(
            [os.environ.get("PYTHON", os.sys.executable), "-c", parent],
            cwd=tmp_path,
            env={"PATH": os.environ.get("PATH", "")},
            timeout=0.2,
        )
    time.sleep(1.1)
    assert not marker.exists()
    assert {thread.ident for thread in threading.enumerate()} == before


def test_filesystem_gitleaks_report_replacement_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verifier_module.shutil, "which", lambda *args, **kwargs: "/bin/gitleaks")

    def fake_run(
        arguments: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout: float = 120,
    ) -> subprocess.CompletedProcess[bytes]:
        del cwd, env, timeout
        if arguments[-1] == "version":
            return subprocess.CompletedProcess(arguments, 0, b"8.30.1\n", b"")
        report_argument = next(item for item in arguments if item.startswith("--report-path="))
        Path(report_argument.split("=", 1)[1]).write_bytes(b"[]")
        return subprocess.CompletedProcess(arguments, 0, b"", b"")

    monkeypatch.setattr(verifier_module, "_run_bounded", fake_run)
    original_open = os.open
    swapped = False

    def swapping_open(
        path: os.PathLike[str] | str,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal swapped
        descriptor = original_open(path, flags, *args, **kwargs)
        candidate = Path(path)
        if candidate.name == "report.json" and not swapped:
            replacement = candidate.with_name("replacement.json")
            replacement.write_bytes(b"[]")
            os.replace(replacement, candidate)
            swapped = True
        return descriptor

    monkeypatch.setattr(verifier_module.os, "open", swapping_open)
    with pytest.raises(PublicVerifyError, match="identity"):
        SubprocessFilesystemGitleaksRunner().scan_directory(tmp_path)


@pytest.mark.parametrize(
    "markdown",
    [
        "[nested label [inner]](docs/guide.md)",
        r"[escaped \]](docs/guide.md)",
        r"[escaped destination](docs/guide\(copy\).md)",
    ],
)
def test_markdown_nested_and_escaped_local_links_are_supported(
    committed_export: CommittedExport,
    markdown: str,
) -> None:
    root = committed_export.root
    if "copy" in markdown:
        (root / "docs/guide(copy).md").write_text("# Copy\n", encoding="utf-8")
        data = _attestation(root)
        records = data["files"]
        assert isinstance(records, list)
        guide = next(record for record in records if record["output_path"] == "docs/guide.md")
        copied = dict(guide)
        copied["output_path"] = "docs/guide(copy).md"
        copied["source_id"] = "docs/guide(copy).md"
        copied["output_sha256"] = hashlib.sha256(
            (root / "docs/guide(copy).md").read_bytes()
        ).hexdigest()
        records.append(copied)
        _write_attestation(root, data)
        # This case only exercises parsing; source provenance correctly rejects the
        # fabricated extra mapping after link validation is called directly.
        _validate_markdown_links(root, {"README.md", "docs/guide(copy).md"})
        return
    (root / "README.md").write_text(f"# Fixture\n\n{markdown}\n", encoding="utf-8")
    _validate_markdown_links(root, {"README.md", "docs/guide.md"})


@pytest.mark.parametrize(
    "markdown",
    [
        "[escape](../../outside.md)",
        "[encoded escape](%2e%2e/%2e%2e/outside.md)",
        "[malformed](docs/guide.md",
        "[nested](docs/(guide.md)",
    ],
)
def test_markdown_malformed_or_traversing_links_are_rejected(
    committed_export: CommittedExport,
    markdown: str,
) -> None:
    (committed_export.root / "README.md").write_text(markdown + "\n", encoding="utf-8")
    with pytest.raises(PublicVerifyError, match="Markdown"):
        _validate_markdown_links(committed_export.root, {"README.md"})


def test_shell_wrapper_is_syntax_valid_and_invokes_cli() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/verify_public_export.sh"
    subprocess.run(["bash", "-n", script], check=True)
    text = script.read_text(encoding="utf-8")
    assert "model-forge public-verify" in text
    assert "--source-repo" in text
    assert "--wheelhouse" in text
    assert "--wheelhouse-lock" in text
    assert "bootstrap_public_export_wheelhouse.py" in text
    assert ".public-export-wheelhouse" not in text
    assert "eval " not in text
