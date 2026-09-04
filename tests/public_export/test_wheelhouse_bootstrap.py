from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from packaging.utils import parse_wheel_filename

from model_forge.public_export.verifier import PublicVerifyError, _wheelhouse_evidence
from model_forge.public_export.wheelhouse import _distribution_metadata, _wheel_entries


def _project(root: Path, requirement: str) -> None:
    root.mkdir()
    (root / "pyproject.toml").write_text(
        f"""\
[build-system]
requires = [{requirement!r}]
build-backend = "unused"
[project]
name = "bootstrap-fixture"
version = "0.0.1"
dependencies = [{requirement!r}]
""",
        encoding="utf-8",
    )


def _bootstrap(
    source: Path,
    wheelhouse: Path,
    lock: Path,
    *,
    env: dict[str, str] | None = None,
    environment_python: Path = Path(sys.executable),
) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts/bootstrap_public_export_wheelhouse.py"),
            "--source-repo",
            str(source),
            "--environment-python",
            str(environment_python),
            "--wheelhouse",
            str(wheelhouse),
            "--lock",
            str(lock),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _fake_uv(path: Path, body: str) -> Path:
    path.write_text(f"#!/bin/sh\nset -eu\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _installed_fixture(root: Path, name: str = "fallback-fixture") -> Path:
    package = root / name.replace("-", "_")
    dist_info = root / f"{name.replace('-', '_')}-1.0.dist-info"
    package.mkdir(parents=True)
    dist_info.mkdir()
    (package / "__init__.py").write_text('__version__ = "1.0"\n', encoding="utf-8")
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text(
        f"{package.name}/__init__.py,,\n"
        f"{dist_info.name}/METADATA,,\n"
        f"{dist_info.name}/WHEEL,,\n"
        f"{dist_info.name}/RECORD,,\n",
        encoding="utf-8",
    )
    return root


def test_complete_installed_environment_does_not_invoke_uv(
    tmp_path: Path,
) -> None:
    packaging_version = importlib.metadata.version("packaging")
    source = tmp_path / "source"
    _project(source, f"packaging=={packaging_version}")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invoked = tmp_path / "uv-invoked"
    _fake_uv(fake_bin / "uv", f"touch {invoked!s}\nexit 99")
    empty_cache = tmp_path / "empty-uv-cache"
    empty_cache.mkdir()
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["UV_CACHE_DIR"] = str(empty_cache)

    result = _bootstrap(source, tmp_path / "wheelhouse", tmp_path / "lock", env=env)

    assert result.returncode == 0, result.stderr
    assert not invoked.exists()


def test_incomplete_installed_environment_falls_back_to_offline_uv(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _project(source, "fallback-fixture==1.0")
    materialized = _installed_fixture(tmp_path / "materialized")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    inspected = tmp_path / "environment-inspected"
    environment_python = tmp_path / "environment-python"
    environment_python.write_text(
        f'#!/bin/sh\nset -eu\ntouch {inspected!s}\nexec {sys.executable!s} "$@"\n',
        encoding="utf-8",
    )
    environment_python.chmod(0o755)
    invoked = tmp_path / "uv-invoked"
    _fake_uv(
        fake_bin / "uv",
        (
            f'test -f "{inspected}"\n'
            f'touch "{invoked}"\n'
            'while [ "$1" != "--target" ]; do shift; done\n'
            "shift\n"
            f'cp -R "{materialized}/." "$1/"'
        ),
    )
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    output_alias = tmp_path / "output-alias"
    output_alias.symlink_to(real_output, target_is_directory=True)

    result = _bootstrap(
        source,
        output_alias / "wheelhouse",
        output_alias / "lock",
        env=env,
        environment_python=environment_python,
    )

    assert result.returncode == 0, result.stderr
    assert invoked.is_file()


def test_introspection_needs_no_packaging_in_environment_python(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "environment"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        check=True,
    )
    environment_python = environment / "bin/python"
    missing = subprocess.run(
        [str(environment_python), "-I", "-c", "import packaging"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0
    supplement = tmp_path / "supplement"
    supplement.mkdir()
    packaging_module = Path(__import__("packaging").__file__).parent
    packaging_dist_info = next(packaging_module.parent.glob("packaging-*.dist-info"))
    shutil.copytree(packaging_module, supplement / packaging_module.name)
    shutil.copytree(packaging_dist_info, supplement / packaging_dist_info.name)
    source = tmp_path / "source"
    packaging_version = importlib.metadata.version("packaging")
    _project(source, f"packaging=={packaging_version}")

    metadata = _distribution_metadata(environment_python, source, supplement)

    assert [(item["name"], item["version"]) for item in metadata] == [
        ("packaging", packaging_version)
    ]


def test_offline_bootstrap_is_deterministic_and_never_contacts_index(
    tmp_path: Path,
) -> None:
    packaging_version = importlib.metadata.version("packaging")
    source = tmp_path / "clean-checkout"
    _project(source, f"packaging=={packaging_version}")
    contacted = threading.Event()

    class IndexHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            contacted.set()
            self.send_response(500)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), IndexHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = os.environ.copy()
    env["PIP_INDEX_URL"] = f"http://127.0.0.1:{server.server_port}/simple"
    try:
        first = _bootstrap(source, tmp_path / "wheelhouse-one", tmp_path / "one.sha256", env=env)
        second = _bootstrap(source, tmp_path / "wheelhouse-two", tmp_path / "two.sha256", env=env)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert not contacted.is_set()
    assert (tmp_path / "one.sha256").read_bytes() == (tmp_path / "two.sha256").read_bytes()
    first_wheels = {
        path.name: path.read_bytes() for path in (tmp_path / "wheelhouse-one").iterdir()
    }
    second_wheels = {
        path.name: path.read_bytes() for path in (tmp_path / "wheelhouse-two").iterdir()
    }
    assert first_wheels == second_wheels
    assert all(parse_wheel_filename(name) for name in first_wheels)


def test_pyyaml_bootstrap_uses_canonical_wheel_name_and_lock_evidence(
    tmp_path: Path,
) -> None:
    pyyaml_version = importlib.metadata.version("PyYAML")
    source = tmp_path / "clean-checkout"
    _project(source, f"PyYAML=={pyyaml_version}")
    contacted = threading.Event()

    class IndexHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            contacted.set()
            self.send_response(500)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), IndexHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = os.environ.copy()
    env["PIP_INDEX_URL"] = f"http://127.0.0.1:{server.server_port}/simple"
    try:
        first = _bootstrap(source, tmp_path / "wheelhouse-one", tmp_path / "one.sha256", env=env)
        second = _bootstrap(source, tmp_path / "wheelhouse-two", tmp_path / "two.sha256", env=env)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert not contacted.is_set()
    wheel = next(path for path in (tmp_path / "wheelhouse-one").iterdir() if "pyyaml" in path.name)
    assert wheel.name.startswith(f"pyyaml-{pyyaml_version}-")
    assert "PyYAML" not in wheel.name
    lock = (tmp_path / "one.sha256").read_text(encoding="ascii")
    assert f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}\n" in lock
    assert "# source-record sha256=" in lock
    assert (tmp_path / "one.sha256").read_bytes() == (tmp_path / "two.sha256").read_bytes()


def test_wheelhouse_lock_rejects_tampering(tmp_path: Path) -> None:
    packaging_version = importlib.metadata.version("packaging")
    source = tmp_path / "source"
    wheelhouse = tmp_path / "wheelhouse"
    lock = tmp_path / "wheelhouse.sha256"
    _project(source, f"packaging=={packaging_version}")
    result = _bootstrap(source, wheelhouse, lock)
    assert result.returncode == 0, result.stderr
    wheel = next(wheelhouse.iterdir())
    wheel.write_bytes(wheel.read_bytes() + b"tampered")

    with pytest.raises(PublicVerifyError, match="SHA256 lock"):
        _wheelhouse_evidence(wheelhouse, lock)


@pytest.mark.parametrize("failure", ["hash", "size", "missing"])
def test_repacking_validates_installed_record_entries(
    tmp_path: Path,
    failure: str,
) -> None:
    metadata = tmp_path / "fixture-1.0.dist-info/METADATA"
    record = tmp_path / "fixture-1.0.dist-info/RECORD"
    metadata.parent.mkdir()
    metadata.write_bytes(b"Name: fixture\nVersion: 1.0\n")
    record.write_text(
        "fixture-1.0.dist-info/METADATA,sha256=invalid,27\nfixture-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    metadata_hash = (
        base64.urlsafe_b64encode(hashlib.sha256(metadata.read_bytes()).digest())
        .rstrip(b"=")
        .decode()
    )
    files: list[dict[str, object]] = [
        {
            "archive": "fixture-1.0.dist-info/METADATA",
            "path": str(metadata if failure != "missing" else tmp_path / "missing"),
            "record_hash_mode": "sha256",
            "record_hash_value": ("invalid" if failure == "hash" else metadata_hash),
            "record_size": 999 if failure == "size" else len(metadata.read_bytes()),
        },
        {
            "archive": "fixture-1.0.dist-info/RECORD",
            "path": str(record),
            "record_hash_mode": None,
            "record_hash_value": None,
            "record_size": None,
        },
    ]

    with pytest.raises(SystemExit, match="unavailable|RECORD (hash|size) mismatch"):
        _wheel_entries({"files": files, "roots": [str(tmp_path)]})


@pytest.mark.parametrize("symlink_kind", ["file", "ancestor"])
def test_repacking_rejects_symlinked_record_paths(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    allowed = tmp_path / "allowed"
    real = tmp_path / "real"
    allowed.mkdir()
    real.mkdir()
    metadata = real / "METADATA"
    metadata.write_bytes(b"Name: fixture\nVersion: 1.0\n")
    if symlink_kind == "file":
        path = allowed / "METADATA"
        path.symlink_to(metadata)
    else:
        linked_parent = allowed / "fixture"
        linked_parent.symlink_to(real, target_is_directory=True)
        path = linked_parent / "METADATA"
    record = allowed / "fixture-1.0.dist-info"
    record.mkdir()
    record_path = record / "RECORD"
    record_path.write_text("fixture-1.0.dist-info/RECORD,,\n", encoding="utf-8")
    files = [
        {
            "archive": "fixture/METADATA",
            "path": str(path),
            "record_hash_mode": None,
            "record_hash_value": None,
            "record_size": len(metadata.read_bytes()),
        },
        {
            "archive": "fixture-1.0.dist-info/RECORD",
            "path": str(record_path),
            "record_hash_mode": None,
            "record_hash_value": None,
            "record_size": None,
        },
    ]

    with pytest.raises(SystemExit, match="symlink|regular file"):
        _wheel_entries({"files": files, "roots": [str(allowed)]})


def test_repacking_rejects_record_path_outside_distribution_roots(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    metadata = outside / "METADATA"
    metadata.write_bytes(b"Name: fixture\nVersion: 1.0\n")
    record = allowed / "fixture-1.0.dist-info"
    record.mkdir()
    record_path = record / "RECORD"
    record_path.write_text("fixture-1.0.dist-info/RECORD,,\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="outside distribution roots"):
        _wheel_entries(
            {
                "roots": [str(allowed)],
                "files": [
                    {
                        "archive": "fixture/METADATA",
                        "path": str(metadata),
                        "record_hash_mode": None,
                        "record_hash_value": None,
                        "record_size": len(metadata.read_bytes()),
                    },
                    {
                        "archive": "fixture-1.0.dist-info/RECORD",
                        "path": str(record_path),
                        "record_hash_mode": None,
                        "record_hash_value": None,
                        "record_size": None,
                    },
                ],
            }
        )


def test_offline_bootstrap_reports_unavailable_version_closure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _project(source, "packaging==0.0")

    result = _bootstrap(source, tmp_path / "wheelhouse", tmp_path / "wheelhouse.sha256")

    assert result.returncode == 1
    assert "packaging==" in result.stderr
    assert "dependency closure" in result.stderr
    assert "unsatisfiable" in result.stderr


def test_offline_bootstrap_combines_installed_and_uv_fallback_errors(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _project(source, "missing-fixture==7")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _fake_uv(fake_bin / "uv", 'printf "empty offline cache\\n" >&2\nexit 1')
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

    result = _bootstrap(
        source,
        tmp_path / "wheelhouse",
        tmp_path / "wheelhouse.sha256",
        env=env,
    )

    assert result.returncode == 1
    assert "missing distribution missing-fixture==7" in result.stderr
    assert "offline uv fallback failed" in result.stderr
    assert "empty offline cache" in result.stderr
