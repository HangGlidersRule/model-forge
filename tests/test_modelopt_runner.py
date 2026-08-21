"""Idempotent ModelOpt mcprue runner dry-run tests."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "scripts" / "qwen3_8" / "run_qwen38_modelopt_mcprue.sh"
# PATH without any virtualenv bin directory: bare `python` is absent here.
BARE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
IMPORT_PROBE = "import model_forge.modelopt.validate"


def _dry_run_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    for inherited in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "PYTHON_BIN"):
        env.pop(inherited, None)
    env.update(
        {
            "D_ROOT": str(tmp_path),
            "ALLOW_NON_D_ROOT": "1",
            "FREE_GB_REQUIRED": "0",
            "DRY_RUN": "1",
            "EXECUTE": "0",
            "CANDIDATE": "mixed_w4a16",
            "SOURCE_KIND": "clean",
        }
    )
    env.update(overrides)
    return env


def _run_dry(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _imports_model_forge(interpreter: str, *, repo_src: bool) -> bool:
    env = {"PATH": BARE_PATH}
    if repo_src:
        env["PYTHONPATH"] = str(REPO / "src")
    probe = subprocess.run(
        [interpreter, "-c", IMPORT_PROBE],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


def _interpreter_without_installed_model_forge() -> str | None:
    """An interpreter that cannot import model_forge unless PYTHONPATH carries it."""
    for candidate in (Path(sys.base_prefix) / "bin" / "python3", Path("/usr/bin/python3")):
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        if not _imports_model_forge(str(candidate), repo_src=False):
            return str(candidate)
    return None


def test_runner_script_exists_and_is_executable_bit_or_bashable() -> None:
    assert RUNNER.is_file()
    text = RUNNER.read_text(encoding="utf-8")
    assert "D_ROOT" in text
    assert "EXECUTE" in text
    assert "ALLOW_ABLITERATED" in text
    assert "Refusing overwrite" in text
    assert "modelopt" in text.lower()


def test_runner_never_invokes_bare_python() -> None:
    """A bare `python` cannot import model_forge outside an activated venv."""
    bare_script_call = re.compile(r"^\s*python3?\s+\S*\.py\b")
    for line in RUNNER.read_text(encoding="utf-8").splitlines():
        if bare_script_call.match(line):
            raise AssertionError(f"runner must use PYTHON_BIN, not a bare interpreter: {line!r}")


@pytest.mark.private_source_only
def test_runner_dry_run_with_allow_non_d(tmp_path: Path) -> None:
    proc = _run_dry(_dry_run_env(tmp_path))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "DRY RUN" in proc.stdout
    assert "PLAN:" in proc.stdout


@pytest.mark.private_source_only
def test_runner_dry_run_without_active_venv(tmp_path: Path) -> None:
    """No VIRTUAL_ENV, no venv on PATH, no bare `python`: the run still succeeds."""
    proc = _run_dry(_dry_run_env(tmp_path, PATH=BARE_PATH))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "DRY RUN" in proc.stdout
    assert str(REPO / "src") in proc.stdout


@pytest.mark.private_source_only
def test_runner_dry_run_with_interpreter_lacking_installed_package(tmp_path: Path) -> None:
    """An uninstalled interpreter imports from repo src, or is refused up front."""
    interpreter = _interpreter_without_installed_model_forge()
    if interpreter is None:
        pytest.skip("every candidate interpreter already has model_forge installed")

    proc = _run_dry(_dry_run_env(tmp_path, PATH=BARE_PATH, PYTHON_BIN=interpreter))
    output = proc.stdout + proc.stderr
    if _imports_model_forge(interpreter, repo_src=True):
        assert proc.returncode == 0, output
        assert f"PYTHON_BIN={interpreter}" in proc.stdout
        assert str(REPO / "src") in proc.stdout
        assert "DRY RUN" in proc.stdout
    else:
        # Missing project dependencies must fail closed with an actionable message.
        assert proc.returncode != 0
        assert "cannot import model_forge" in output


@pytest.mark.private_source_only
def test_runner_fails_closed_when_python_bin_is_unusable(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-python"
    proc = _run_dry(_dry_run_env(tmp_path, PATH=BARE_PATH, PYTHON_BIN=str(missing)))
    assert proc.returncode != 0
    assert "cannot import model_forge" in (proc.stdout + proc.stderr)


@pytest.mark.private_source_only
def test_runner_blocks_abliterated_without_flag(tmp_path: Path) -> None:
    env = _dry_run_env(
        tmp_path,
        DRY_RUN="0",
        EXECUTE="1",
        SOURCE_KIND="abliterated",
        ALLOW_ABLITERATED="0",
    )
    proc = _run_dry(env)
    assert proc.returncode != 0
    assert "explicit heavy-mutation authorization" in (proc.stdout + proc.stderr)


@pytest.mark.private_source_only
def test_runner_blocks_the_deprecated_source_kind_without_a_flag(tmp_path: Path) -> None:
    """SOURCE_KIND=darkstar normalizes before the guard, so it is not an escape hatch."""
    env = _dry_run_env(tmp_path, DRY_RUN="0", EXECUTE="1", SOURCE_KIND="darkstar")
    for allow in ("ALLOW_ABLITERATED", "ALLOW_DARKSTAR"):
        env.pop(allow, None)
    proc = _run_dry(env)
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "SOURCE_KIND=darkstar is deprecated" in output
    assert "explicit heavy-mutation authorization" in output


@pytest.mark.private_source_only
def test_runner_accepts_the_deprecated_allow_env_for_the_canonical_source_kind(
    tmp_path: Path,
) -> None:
    env = _dry_run_env(tmp_path, SOURCE_KIND="abliterated", ALLOW_DARKSTAR="1")
    env.pop("ALLOW_ABLITERATED", None)
    proc = _run_dry(env)
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert "ALLOW_DARKSTAR is deprecated" in output
    assert "kind=abliterated" in output


@pytest.mark.private_source_only
def test_canonical_allow_env_overrides_the_deprecated_one(tmp_path: Path) -> None:
    """An explicit canonical value wins; the alias never loosens a stated refusal."""
    env = _dry_run_env(
        tmp_path,
        DRY_RUN="0",
        EXECUTE="1",
        SOURCE_KIND="abliterated",
        ALLOW_ABLITERATED="0",
        ALLOW_DARKSTAR="1",
    )
    proc = _run_dry(env)
    assert proc.returncode != 0
    assert "explicit heavy-mutation authorization" in (proc.stdout + proc.stderr)
