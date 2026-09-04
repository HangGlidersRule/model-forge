"""End-to-end runner tests proving EXECUTE=1 does real work (no stub, no fake success).

A self-contained fake ``docker`` materializes a valid export, so the full path is
exercised without a GPU: container run, fail-closed validation, SHA256 manifest,
_SUCCESS, atomic promotion, and runtime restore capture.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from modelopt_fakes import write_fake_docker, write_fake_source

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "scripts" / "qwen3_8" / "run_qwen38_modelopt_mcprue.sh"
PY_SCRIPT = REPO / "scripts" / "qwen3_8" / "quantize_qwen38_modelopt.py"


def _run(env_overrides: dict[str, str], d_root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "D_ROOT": str(d_root),
            "ALLOW_NON_D_ROOT": "1",
            "FREE_GB_REQUIRED": "0",
            "DRY_RUN": "0",
            "EXECUTE": "1",
            "CANDIDATE": "mlp_only",
            "SOURCE_KIND": "clean",
        }
    )
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(RUNNER)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.private_source_only
def test_execute_runs_container_validates_and_promotes(tmp_path: Path) -> None:
    d_root = tmp_path / "d"
    d_root.mkdir()
    source = d_root / "sources" / "clean-bf16"
    write_fake_source(source)
    docker = write_fake_docker(tmp_path / "fake_docker")

    proc = _run({"DOCKER": str(docker)}, d_root)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    out_dir = d_root / "artifacts" / "Qwen3.8-27B-clean-mlp_only-modelopt-nvfp4"
    assert out_dir.is_dir(), proc.stdout + proc.stderr
    assert (out_dir / "_SUCCESS.json").exists()
    assert (out_dir / "manifest.sha256").exists()
    assert (out_dir / "hf_quant_config.json").exists()

    # Atomic promotion: no partial left behind.
    partials = list((d_root / "artifacts").glob(".*partial*"))
    assert not partials

    # The heavy command that was really invoked is recorded and exact.
    planned = json.loads((out_dir / "planned_command.json").read_text())
    argv_text = " ".join(planned["argv"])
    assert "examples/hf_ptq/hf_ptq.py" in argv_text
    assert "model-forge-modelopt:0.46.0rc2-43fd41a" in argv_text
    assert "512,512" in argv_text
    assert "--kv_cache_qformat none" in argv_text

    # Restore artifact captured from the running runtime via docker inspect.
    snaps = list((d_root / "snapshots").glob("*-pre-clean-mlp_only"))
    assert snaps, "expected an append-only snapshot dir"
    restore = snaps[0] / "restore.sh"
    assert restore.exists()
    assert "docker run" in restore.read_text()


@pytest.mark.private_source_only
def test_execute_passes_restricted_gpus_through_to_docker(tmp_path: Path) -> None:
    """A restricted GPUS value must reach `docker run`, not be silently widened."""
    d_root = tmp_path / "d"
    d_root.mkdir()
    write_fake_source(d_root / "sources" / "clean-bf16")
    docker = write_fake_docker(tmp_path / "fake_docker")

    proc = _run({"DOCKER": str(docker), "GPUS": "device=1"}, d_root)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    out_dir = d_root / "artifacts" / "Qwen3.8-27B-clean-mlp_only-modelopt-nvfp4"
    argv = json.loads((out_dir / "planned_command.json").read_text())["argv"]
    assert "--gpus=device=1" in argv
    assert "--gpus=all" not in argv


def test_print_command_honours_restricted_gpus(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    proc = subprocess.run(
        [
            sys.executable,
            str(PY_SCRIPT),
            "--run-root", str(tmp_path / "run"),
            "--source-dir", str(tmp_path / "src"),
            "--print-command",
            "--export-dir", str(tmp_path / "out"),
            "--modelopt-image", "img",
            "--hf-cache", str(tmp_path / "hf"),
            "--calib-cache", str(tmp_path / "calib"),
            "--gpus", "device=0,1",
        ],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout[proc.stdout.index('{\n  "argv"') :])
    assert "--gpus=device=0,1" in payload["argv"]


@pytest.mark.private_source_only
def test_execute_refuses_overwrite_of_existing_artifact(tmp_path: Path) -> None:
    d_root = tmp_path / "d"
    d_root.mkdir()
    write_fake_source(d_root / "sources" / "clean-bf16")
    docker = write_fake_docker(tmp_path / "fake_docker")

    out_dir = d_root / "artifacts" / "Qwen3.8-27B-clean-mlp_only-modelopt-nvfp4"
    out_dir.mkdir(parents=True)
    (out_dir / "stale.txt").write_text("prior", encoding="utf-8")

    proc = _run({"DOCKER": str(docker)}, d_root)
    assert proc.returncode != 0
    assert "Refusing overwrite" in (proc.stdout + proc.stderr)
    # The prior artifact is untouched.
    assert (out_dir / "stale.txt").read_text() == "prior"


@pytest.mark.private_source_only
def test_execute_requires_runtime_snapshot_or_allow(tmp_path: Path) -> None:
    d_root = tmp_path / "d"
    d_root.mkdir()
    write_fake_source(d_root / "sources" / "clean-bf16")
    # Fake docker whose `ps` returns no container id.
    docker = tmp_path / "empty_docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        "  ps) exit 0 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    proc = _run({"DOCKER": str(docker)}, d_root)
    assert proc.returncode != 0
    assert "No running runtime container" in (proc.stdout + proc.stderr)


def test_no_intentional_stub_remains() -> None:
    runner_text = RUNNER.read_text(encoding="utf-8")
    py_text = PY_SCRIPT.read_text(encoding="utf-8")
    for banned in (
        "gated to the remote mcprue host",
        "stops before GPU quantization",
        "intentionally not embedded",
        "sys.exit(10)",
    ):
        assert banned not in runner_text, banned
        assert banned not in py_text, banned
    # No "|| true" on the build/validation path of either script.
    assert "|| true" not in py_text
    for line in runner_text.splitlines():
        if "|| true" in line:
            raise AssertionError(f"forbidden '|| true' on runner path: {line!r}")
