"""A success marker's identity hash is not proof the weights were validated.

``classify_artifact`` may return a match only when ``_SUCCESS.json`` binds the
current build identity (or its superseded source-kind digest) to a non-empty
tree hash of every regular file in the export directory except the marker
itself. That tree is ``sha256_tree(export_dir)`` with the top-level
``_SUCCESS.json`` entry removed: finalize writes the marker after hashing, so
the marker cannot contain its own digest.

Anything else — an empty ``output_hashes`` map, a substituted weight, an extra
file, a hash that points outside the directory, or a marker symlink whose
hashes describe a different directory — must not be ``ARTIFACT_MATCH`` and must
not be ``ARTIFACT_ABSENT``. Callers refuse that directory in place. They must
not create caches, partial exports, or run the container.

``validate_success_marker`` / ``StageContext.should_skip`` use the same tree
rule. Resume skips in the quantize, abliteration, measurement, and corpus
scripts are that function and no second check.

These tests fail while a ``config_sha`` match alone is treated as validated.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from model_forge.experiment import load_experiment
from model_forge.modelopt.calibration import default_calibration_contract
from model_forge.modelopt.identity import (
    ARTIFACT_ABSENT,
    ARTIFACT_MATCH,
    build_identity,
    build_identity_sha,
    classify_artifact,
    legacy_source_kind_identity_shas,
)
from model_forge.modelopt.pin import PRIMARY_RECIPE, load_pin
from model_forge.pipeline import (
    SUCCESS_MARKER,
    StageContext,
    SuccessManifest,
    sha256_file,
    validate_success_marker,
)

REPO = Path(__file__).resolve().parent.parent
PY_SCRIPT = REPO / "scripts" / "qwen3_8" / "quantize_qwen38_modelopt.py"
RUNNER = REPO / "scripts" / "qwen3_8" / "run_qwen38_modelopt_gpu_host.sh"
NVFP4_SCRIPT = REPO / "scripts" / "qwen3_8" / "quantize_qwen38_nvfp4.py"
EXPERIMENT = REPO / "recipes" / "qwen3.8-27b" / "r3-nvfp4.yaml"

_SPEC = importlib.util.spec_from_file_location("quantize_qwen38_modelopt_under_test", PY_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_QUANTIZE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_QUANTIZE)
EXIT_REFUSE_OVERWRITE = _QUANTIZE.EXIT_REFUSE_OVERWRITE

_REFUSED = {ARTIFACT_MATCH, ARTIFACT_ABSENT}


def _provenance(source: Path, *, source_kind: str = "clean") -> dict[str, object]:
    pin = load_pin()
    return build_identity(
        pin=pin,
        candidate="mlp_only",
        recipe=PRIMARY_RECIPE,
        recipe_sha256=sha256_file(PRIMARY_RECIPE),
        calibration=default_calibration_contract(),
        source_kind=source_kind,
        source_dir=source,
    )


def _coverage(export_dir: Path) -> dict[str, str]:
    """Tree hash of every regular file except the top-level success marker."""
    hashes: dict[str, str] = {}
    for path in sorted(export_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(export_dir))
        if relative == SUCCESS_MARKER:
            continue
        hashes[relative] = sha256_file(path)
    return hashes


def _write_marker(export_dir: Path, config_sha: str, output_hashes: dict[str, str]) -> bytes:
    export_dir.mkdir(parents=True, exist_ok=True)
    payload = SuccessManifest(
        stage="modelopt_mlp_only",
        config_sha=config_sha,
        output_hashes=output_hashes,
    ).to_json()
    marker = export_dir / SUCCESS_MARKER
    marker.write_text(payload, encoding="utf-8")
    return payload.encode("utf-8")


def _bind(export_dir: Path, config_sha: str) -> None:
    _write_marker(export_dir, config_sha, _coverage(export_dir))


def _plant_weights(export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "model.safetensors").write_bytes(b"validated-weight-v1")
    shard = export_dir / "shards"
    shard.mkdir()
    (shard / "keep.bin").write_bytes(b"validated-shard-v1")


def _snapshot(root: Path) -> dict[str, object]:
    found: dict[str, object] = {}
    if not root.exists():
        return found
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            found[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            found[relative] = ("file", path.read_bytes())
    return found


def _reported_identity(output: str) -> str:
    lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith("build identity:")
    ]
    assert len(lines) == 1, output
    return lines[0].split(":", 1)[1].strip()


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    return env


def _run_quantize(
    source: Path,
    export_dir: Path,
    work: Path,
    *,
    mode: str,
    source_kind: str = "clean",
    docker: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(PY_SCRIPT),
        "--run-root",
        str(work / "run"),
        "--source-dir",
        str(source),
        "--candidate",
        "mlp_only",
        "--source-kind",
        source_kind,
        "--export-dir",
        str(export_dir),
    ]
    if source_kind == "abliterated":
        command.append("--allow-abliterated")
    if mode == "check":
        command.append("--check-build-identity")
    elif mode == "execute":
        assert docker is not None
        command.extend(
            [
                "--execute",
                "--modelopt-image",
                "modelopt:under-test",
                "--hf-cache",
                str(work / "hf"),
                "--calib-cache",
                str(work / "calib"),
                "--docker-bin",
                str(docker),
            ]
        )
    else:
        raise AssertionError(mode)
    return subprocess.run(
        command,
        cwd=str(REPO),
        env=_python_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_docker(path: Path, log: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(log))}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _assert_no_tool_writes(work: Path, export_dir: Path, before: dict[str, object], log: Path) -> None:
    assert _snapshot(export_dir) == before
    assert not (work / "run").exists()
    assert not (work / "hf").exists()
    assert not (work / "calib").exists()
    assert list(export_dir.parent.glob(f".{export_dir.name}.partial.*")) == []
    assert not log.exists()


def _marker_sha(branch: str, canonical: str, legacy: tuple[str, ...]) -> str:
    if branch == "canonical":
        return canonical
    if branch == "legacy":
        assert len(legacy) == 1
        assert legacy[0] != canonical
        return legacy[0]
    raise AssertionError(branch)


@pytest.mark.parametrize("branch", ["canonical", "legacy"])
@pytest.mark.parametrize("tamper", ["substitute", "extra_file", "drop_hashes", "marker_only"])
def test_classify_match_requires_a_complete_weight_tree(
    tmp_path: Path, branch: str, tamper: str
) -> None:
    """Both identity branches bind the directory that was opened, then stop."""
    source = tmp_path / "source"
    provenance = _provenance(source, source_kind="abliterated")
    canonical = build_identity_sha(provenance)
    legacy = legacy_source_kind_identity_shas(provenance)
    marker_sha = _marker_sha(branch, canonical, legacy)

    export_dir = tmp_path / "export"
    _plant_weights(export_dir)
    _bind(export_dir, marker_sha)
    assert (
        classify_artifact(export_dir, canonical, equivalent_identity_shas=legacy)[0]
        == ARTIFACT_MATCH
    )

    if tamper == "substitute":
        (export_dir / "model.safetensors").write_bytes(b"substituted-weight-v2")
    elif tamper == "extra_file":
        extra = export_dir / "shards" / "evil.bin"
        extra.write_bytes(b"unvalidated-extra")
    elif tamper == "drop_hashes":
        _write_marker(export_dir, marker_sha, {})
    elif tamper == "marker_only":
        (export_dir / "model.safetensors").unlink()
        (export_dir / "shards" / "keep.bin").unlink()
        _write_marker(export_dir, marker_sha, {})
    else:
        raise AssertionError(tamper)

    before = _snapshot(export_dir)
    state, detail = classify_artifact(
        export_dir, canonical, equivalent_identity_shas=legacy
    )
    assert state not in _REFUSED, detail
    assert _snapshot(export_dir) == before


def test_classify_refuses_a_marker_whose_hashes_leave_the_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    canonical = build_identity_sha(_provenance(source))
    export_dir = tmp_path / "export"
    _plant_weights(export_dir)
    _bind(export_dir, canonical)
    assert classify_artifact(export_dir, canonical)[0] == ARTIFACT_MATCH

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside-bytes")
    _write_marker(export_dir, canonical, {"../outside.bin": sha256_file(outside)})
    before_export = _snapshot(export_dir)
    before_outside = outside.read_bytes()

    state, detail = classify_artifact(export_dir, canonical)
    assert state not in _REFUSED, detail
    assert _snapshot(export_dir) == before_export
    assert outside.read_bytes() == before_outside


def test_classify_does_not_accept_a_marker_that_authorizes_another_directory(
    tmp_path: Path,
) -> None:
    """The directory classified is the directory the caller will publish."""
    source = tmp_path / "source"
    canonical = build_identity_sha(_provenance(source))
    good = tmp_path / "good"
    _plant_weights(good)
    _bind(good, canonical)
    assert classify_artifact(good, canonical)[0] == ARTIFACT_MATCH

    export_dir = tmp_path / "export"
    _plant_weights(export_dir)
    (export_dir / "model.safetensors").write_bytes(b"substituted-weight-v2")
    (export_dir / SUCCESS_MARKER).symlink_to(good / SUCCESS_MARKER)
    before_export = _snapshot(export_dir)
    before_good = _snapshot(good)

    state, detail = classify_artifact(export_dir, canonical)
    assert state not in _REFUSED, detail
    assert _snapshot(export_dir) == before_export
    assert _snapshot(good) == before_good


@pytest.mark.parametrize("mode", ["check", "execute"])
@pytest.mark.parametrize("tamper", ["substitute", "extra_file", "drop_hashes"])
def test_quantize_entry_points_refuse_an_unbound_marker_with_no_side_effects(
    tmp_path: Path, mode: str, tamper: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    canonical = build_identity_sha(_provenance(source))
    export_dir = tmp_path / "export"
    _plant_weights(export_dir)
    _bind(export_dir, canonical)
    log = tmp_path / "docker.log"
    docker = _write_docker_path(tmp_path / "docker", log)

    bound = _run_quantize(source, export_dir, tmp_path / "bound", mode=mode, docker=docker)
    assert bound.returncode == 0, bound.stdout + bound.stderr
    assert _reported_identity(bound.stdout) == canonical
    assert "Artifact already validated, skipping:" in bound.stdout
    assert "Running pinned ModelOpt container:" not in bound.stdout
    assert not log.exists()

    if tamper == "substitute":
        (export_dir / "model.safetensors").write_bytes(b"substituted-weight-v2")
    elif tamper == "extra_file":
        (export_dir / "shards" / "evil.bin").write_bytes(b"unvalidated-extra")
    elif tamper == "drop_hashes":
        _write_marker(export_dir, canonical, {})
    else:
        raise AssertionError(tamper)

    before = _snapshot(export_dir)
    refused = _run_quantize(source, export_dir, tmp_path / "refused", mode=mode, docker=docker)
    output = refused.stdout + refused.stderr
    assert _reported_identity(refused.stdout) == canonical
    assert "Artifact already validated, skipping:" not in refused.stdout
    assert "Running pinned ModelOpt container:" not in refused.stdout
    assert "SUCCESS: validated artifact promoted" not in refused.stdout
    assert refused.returncode == EXIT_REFUSE_OVERWRITE, output
    assert "Refusing overwrite of existing artifact:" in refused.stderr
    _assert_no_tool_writes(tmp_path / "refused", export_dir, before, log)


def _write_docker_path(path: Path, log: Path) -> Path:
    _write_docker(path, log)
    return path


def test_execute_refuses_a_symlinked_marker_for_a_different_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    canonical = build_identity_sha(_provenance(source))
    good = tmp_path / "good"
    _plant_weights(good)
    _bind(good, canonical)

    export_dir = tmp_path / "export"
    _plant_weights(export_dir)
    (export_dir / "model.safetensors").write_bytes(b"substituted-weight-v2")
    (export_dir / SUCCESS_MARKER).symlink_to(good / SUCCESS_MARKER)
    before_export = _snapshot(export_dir)
    before_good = _snapshot(good)
    log = tmp_path / "docker.log"
    docker = _write_docker_path(tmp_path / "docker", log)

    refused = _run_quantize(source, export_dir, tmp_path / "work", mode="execute", docker=docker)
    output = refused.stdout + refused.stderr
    assert _reported_identity(refused.stdout) == canonical
    assert "unreadable" not in output
    assert "Artifact already validated, skipping:" not in refused.stdout
    assert refused.returncode == EXIT_REFUSE_OVERWRITE, output
    assert "Refusing overwrite of existing artifact:" in refused.stderr
    _assert_no_tool_writes(tmp_path / "work", export_dir, before_export, log)
    assert _snapshot(good) == before_good


def test_legacy_identity_branch_of_both_entry_points_binds_weights(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    provenance = _provenance(source, source_kind="abliterated")
    canonical = build_identity_sha(provenance)
    legacy = legacy_source_kind_identity_shas(provenance)
    assert len(legacy) == 1 and legacy[0] != canonical

    export_dir = tmp_path / "export"
    _plant_weights(export_dir)
    _bind(export_dir, legacy[0])
    log = tmp_path / "docker.log"
    docker = _write_docker_path(tmp_path / "docker", log)

    for mode in ("check", "execute"):
        bound = _run_quantize(
            source,
            export_dir,
            tmp_path / f"bound-{mode}",
            mode=mode,
            source_kind="abliterated",
            docker=docker,
        )
        output = bound.stdout + bound.stderr
        assert bound.returncode == 0, output
        assert _reported_identity(bound.stdout) == canonical
        assert "superseded source-kind terminology" in output
        assert not log.exists()

    (export_dir / "model.safetensors").write_bytes(b"substituted-weight-v2")
    before = _snapshot(export_dir)
    for mode in ("check", "execute"):
        refused = _run_quantize(
            source,
            export_dir,
            tmp_path / f"refused-{mode}",
            mode=mode,
            source_kind="abliterated",
            docker=docker,
        )
        output = refused.stdout + refused.stderr
        assert _reported_identity(refused.stdout) == canonical
        assert "Artifact already validated, skipping:" not in refused.stdout
        assert refused.returncode == EXIT_REFUSE_OVERWRITE, output
        assert "Refusing overwrite of existing artifact:" in refused.stderr
        _assert_no_tool_writes(tmp_path / f"refused-{mode}", export_dir, before, log)


def test_gpu_host_runner_does_not_skip_a_substituted_export(tmp_path: Path) -> None:
    """The operator entry exits 0 only when the export tree matches the marker."""
    d_root = tmp_path / "d"
    d_root.mkdir()
    source = d_root / "sources" / "clean-bf16"
    canonical = build_identity_sha(_provenance(source))
    export_dir = d_root / "artifacts" / "Qwen3.8-27B-clean-mlp_only-modelopt-nvfp4"
    _plant_weights(export_dir)
    _bind(export_dir, canonical)
    log = tmp_path / "docker.log"
    docker = _write_docker_path(tmp_path / "docker", log)

    bound = _run_runner(d_root, docker)
    assert bound.returncode == 0, bound.stdout + bound.stderr
    assert _reported_identity(bound.stdout) == canonical
    assert "idempotent skip" in bound.stdout
    assert "Launching pinned ModelOpt quantization" not in bound.stdout
    assert not log.exists()

    (export_dir / "model.safetensors").write_bytes(b"substituted-weight-v2")
    before = _snapshot(export_dir)
    refused = _run_runner(d_root, docker)
    output = refused.stdout + refused.stderr
    assert _reported_identity(refused.stdout) == canonical
    assert refused.returncode != 0, output
    assert "idempotent skip" not in output
    assert "Launching pinned ModelOpt quantization" not in output
    assert "Refusing overwrite of existing artifact:" in output
    assert _snapshot(export_dir) == before
    assert not log.exists()


def _run_runner(d_root: Path, docker: Path) -> subprocess.CompletedProcess[str]:
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
            "ALLOW_NO_RUNTIME": "1",
            "DOCKER": str(docker),
            "PYTHON_BIN": sys.executable,
            # The runner's path case expands these under `set -u` before it
            # classifies the export. They are not the export being tested.
            "PUBLIC_WORKSPACE": "/unused-workspace",
            "PUBLIC_ARTIFACT_PATH": "/unused-artifact",
        }
    )
    return subprocess.run(
        ["bash", str(RUNNER)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("tamper", ["outside_hash", "extra_file", "drop_hashes"])
def test_validate_success_marker_requires_the_same_complete_tree(
    tmp_path: Path, tamper: str
) -> None:
    """Resume skips for nvfp4, abliteration, measurement, and corpus use this."""
    stage_dir = tmp_path / "quantize_nvfp4"
    _plant_weights(stage_dir)
    config_sha = "c" * 64
    _bind(stage_dir, config_sha)
    assert validate_success_marker(stage_dir, config_sha) is not None
    assert StageContext(tmp_path, "quantize_nvfp4", config_sha).should_skip() is not None

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside-bytes")
    if tamper == "outside_hash":
        _write_marker(stage_dir, config_sha, {"../outside.bin": sha256_file(outside)})
    elif tamper == "extra_file":
        (stage_dir / "shards" / "evil.bin").write_bytes(b"unvalidated-extra")
    elif tamper == "drop_hashes":
        _write_marker(stage_dir, config_sha, {})
    else:
        raise AssertionError(tamper)

    before = _snapshot(stage_dir)
    assert validate_success_marker(stage_dir, config_sha) is None
    assert StageContext(tmp_path, "quantize_nvfp4", config_sha).should_skip() is None
    assert _snapshot(stage_dir) == before
    assert outside.read_bytes() == b"outside-bytes"


@pytest.mark.parametrize("tamper", ["substitute", "extra_file", "drop_hashes"])
def test_nvfp4_resume_does_not_skip_when_the_marker_does_not_bind_weights(
    tmp_path: Path, tamper: str
) -> None:
    """``quantize_qwen38_nvfp4.py --resume`` skips only through ``should_skip``."""
    config_sha = load_experiment(EXPERIMENT).config_sha()
    run_root = tmp_path / "run"
    stage_dir = run_root / "quantize_nvfp4"
    _plant_weights(stage_dir)
    _bind(stage_dir, config_sha)
    source = tmp_path / "source"
    source.mkdir()

    bound = _run_nvfp4(run_root, source)
    assert bound.returncode == 0, bound.stdout + bound.stderr
    assert f"Quantization stage verified, skipping (config_sha={config_sha[:12]})" in bound.stdout
    assert "DRY RUN" not in bound.stdout

    if tamper == "substitute":
        (stage_dir / "model.safetensors").write_bytes(b"substituted-weight-v2")
    elif tamper == "extra_file":
        (stage_dir / "shards" / "evil.bin").write_bytes(b"unvalidated-extra")
    elif tamper == "drop_hashes":
        _write_marker(stage_dir, config_sha, {})
    else:
        raise AssertionError(tamper)

    before = _snapshot(stage_dir)
    refused = _run_nvfp4(run_root, source)
    output = refused.stdout + refused.stderr
    assert f"Quantization stage verified, skipping (config_sha={config_sha[:12]})" not in output
    assert "DRY RUN - Quantization config:" in refused.stdout
    assert refused.returncode == 0, output
    assert _snapshot(stage_dir) == before
    assert list(run_root.glob("quantize_nvfp4.partial.*")) == []


def _run_nvfp4(run_root: Path, source: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(NVFP4_SCRIPT),
            "--config",
            str(EXPERIMENT),
            "--run-root",
            str(run_root),
            "--source-dir",
            str(source),
            "--resume",
            "--dry-run",
        ],
        cwd=str(REPO),
        env=_python_env(),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_a_marker_that_is_not_a_regular_file_is_unverified_not_unreadable(
    tmp_path: Path, kind: str
) -> None:
    """A marker that is a link or a special file says nothing about this directory.

    The refusal must say the artifact is unverified; ``unreadable_marker`` is reserved for a
    regular marker whose contents do not parse.
    """
    from model_forge.modelopt.identity import ARTIFACT_UNVERIFIED

    source = tmp_path / "source"
    canonical = build_identity_sha(_provenance(source))
    good = tmp_path / "good"
    _plant_weights(good)
    _bind(good, canonical)
    export_dir = tmp_path / "export"
    _plant_weights(export_dir)
    marker = export_dir / SUCCESS_MARKER
    if kind == "symlink":
        marker.symlink_to(good / SUCCESS_MARKER)
    else:
        os.mkfifo(marker)
    state, detail = classify_artifact(export_dir, canonical)
    assert state == ARTIFACT_UNVERIFIED, detail
