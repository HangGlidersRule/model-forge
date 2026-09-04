"""The canonical Product 4 Bash launcher is the operator entrypoint and is fail-closed.

These tests exercise the real script with a stubbed ``docker`` so no daemon is required. They pin the
launcher's contract: it renders the deterministic Compose, verifies it against the frozen digest,
brings up one stable project/container with ``up -d --force-recreate``, supports ``--dry-run`` and
``--print-config``, and refuses any mutable vLLM argument or environment override other than the host
port.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = REPO_ROOT / "scripts" / "serve_darkstar_qwen38_abliterated_nvfp4.sh"
TRACKED_COMPOSE = REPO_ROOT / "containers" / "serve" / "darkstar-qwen38-abliterated-nvfp4.yml"
FROZEN_ARTIFACT = "/d/model-forge/artifacts/Qwen3.8-27B-abliterated-performance-mixed-modelopt"


def _docker_stub(tmp_path: Path) -> tuple[Path, Path]:
    """A fake ``docker`` that appends its argv to a log and exits 0."""
    log = tmp_path / "docker-calls.log"
    stub = tmp_path / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> {log!s}\n'
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub, log


def _run(tmp_path: Path, *args: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    stub, _ = _docker_stub(tmp_path)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} and not key.startswith("VLLM_")
    }
    env["PYTHON"] = sys.executable
    env["DOCKER"] = str(stub)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_launcher_is_checked_in_and_executable() -> None:
    assert LAUNCHER.exists(), "the canonical operator entrypoint must be checked in"
    mode = LAUNCHER.stat().st_mode
    assert mode & stat.S_IXUSR, "the launcher must be executable"
    assert LAUNCHER.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")


def test_launcher_has_valid_bash_syntax() -> None:
    result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.private_source_only
def test_dry_run_fails_before_docker_when_frozen_local_artifact_is_missing(tmp_path: Path) -> None:
    stub, log = _docker_stub(tmp_path)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} and not key.startswith("VLLM_")
    }
    env["PYTHON"] = sys.executable
    env["DOCKER"] = str(stub)
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "local artifact directory does not exist" in result.stderr
    assert not log.exists() or log.read_text(encoding="utf-8") == ""


@pytest.mark.private_source_only
def test_launcher_keeps_verified_force_recreate_sequence_but_validates_artifact_first(
    tmp_path: Path,
) -> None:
    stub, log = _docker_stub(tmp_path)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} and not key.startswith("VLLM_")
    }
    env["PYTHON"] = sys.executable
    env["DOCKER"] = str(stub)
    env["VLLM_PORT"] = "8001"
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--artifact-path", FROZEN_ARTIFACT],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "local artifact directory does not exist" in result.stderr
    assert not log.exists() or log.read_text(encoding="utf-8") == ""
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert launcher.index('"${COMPOSE_BASE[@]}" config -q') < launcher.index(
        '"${COMPOSE_BASE[@]}" up -d --force-recreate'
    )


@pytest.mark.private_source_only
def test_launcher_rejects_mutable_vllm_argument(tmp_path: Path) -> None:
    result = _run(tmp_path, "--max-model-len", "999", "--dry-run")
    assert result.returncode != 0
    assert "unknown argument" in result.stderr


@pytest.mark.private_source_only
def test_launcher_rejects_non_port_vllm_env_override(tmp_path: Path) -> None:
    result = _run(tmp_path, "--dry-run", extra_env={"VLLM_KV_CACHE_DTYPE": "fp8"})
    assert result.returncode != 0
    assert "only VLLM_PORT" in result.stderr


@pytest.mark.private_source_only
def test_launcher_rejects_empty_and_invalid_values(tmp_path: Path) -> None:
    empty = _run(tmp_path, "--artifact-path", "--dry-run")
    assert empty.returncode != 0
    assert "non-empty" in empty.stderr

    bad_port = _run(tmp_path, "--dry-run", extra_env={"VLLM_PORT": "not-a-port"})
    assert bad_port.returncode != 0
    assert "VLLM_PORT" in bad_port.stderr


@pytest.mark.private_source_only
def test_launcher_rejects_wrong_identity(tmp_path: Path) -> None:
    result = _run(tmp_path, "--artifact-identity", "Darkstar-Wrong", "--dry-run")
    assert result.returncode != 0
    assert "does not match the frozen Product 4 identity" in result.stderr


@pytest.mark.private_source_only
def test_wrong_artifact_path_fails_identity_validation_and_leaves_tracked_file_untouched(
    tmp_path: Path,
) -> None:
    before = TRACKED_COMPOSE.read_bytes()
    result = _run(
        tmp_path,
        "--artifact-path",
        "/d/model-forge/artifacts/qwen3.8-27b-abliterated-modelopt-mixed-relocated",
        "--dry-run",
    )
    assert result.returncode != 0
    assert "artifact manifest path must exactly match model_path" in result.stderr
    # The launcher renders into a scratch dir, so a bad input never mutates the tracked profile.
    assert TRACKED_COMPOSE.read_bytes() == before
