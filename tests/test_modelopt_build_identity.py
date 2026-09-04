"""Idempotency must compare the complete build identity, not marker presence.

Unit coverage for the identity record and its comparison, plus an end-to-end
proof through the mcprue runner: an unchanged build is skipped, while a marker
recording a different identity is refused instead of being reported as success.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from modelopt_fakes import write_fake_docker, write_fake_source

from model_forge.modelopt.calibration import default_calibration_contract
from model_forge.modelopt.identity import (
    ARTIFACT_ABSENT,
    ARTIFACT_IDENTITY_MISMATCH,
    ARTIFACT_MATCH,
    ARTIFACT_NO_MARKER,
    ARTIFACT_UNREADABLE_MARKER,
    build_identity,
    build_identity_sha,
    classify_artifact,
    legacy_source_kind_identity_shas,
    normalize_source_kind,
)
from model_forge.modelopt.pin import OMLP_RECIPE, PRIMARY_RECIPE, load_pin
from model_forge.pipeline import SUCCESS_MARKER, SuccessManifest

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "scripts" / "qwen3_8" / "run_qwen38_modelopt_mcprue.sh"
PY_SCRIPT = REPO / "scripts" / "qwen3_8" / "quantize_qwen38_modelopt.py"
OUT_NAME = "Qwen3.8-27B-clean-mlp_only-modelopt-nvfp4"


def _identity(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "pin": load_pin(),
        "candidate": "mlp_only",
        "recipe": PRIMARY_RECIPE,
        "recipe_sha256": "a" * 64,
        "calibration": default_calibration_contract(),
        "source_kind": "clean",
        "source_dir": Path("/d/sources/clean-bf16"),
    }
    base.update(overrides)
    return build_identity(**base)  # type: ignore[arg-type]


def test_identity_covers_recipe_pin_calibration_and_source() -> None:
    identity = _identity()
    assert identity["recipe_sha256"] == "a" * 64
    assert identity["pin"]
    assert identity["calibration"]
    assert identity["source_dir"] == "/d/sources/clean-bf16"
    assert len(build_identity_sha(identity)) == 64
    # Stable across calls: the identity is canonicalized, not order-dependent.
    assert build_identity_sha(_identity()) == build_identity_sha(identity)


def test_identity_sha_changes_with_every_component() -> None:
    baseline = build_identity_sha(_identity())
    cal = default_calibration_contract()
    variants = {
        "recipe digest": _identity(recipe_sha256="b" * 64),
        "recipe path": _identity(recipe=OMLP_RECIPE),
        "candidate": _identity(candidate="omlp"),
        "calibration": _identity(calibration=dataclasses.replace(cal, sizes=(256, 256))),
        "source kind": _identity(source_kind="abliterated"),
        "source dir": _identity(source_dir=Path("/d/sources/other")),
    }
    for label, identity in variants.items():
        assert build_identity_sha(identity) != baseline, label


def test_identity_sha_changes_with_the_pin() -> None:
    pin = load_pin()
    drifted = dataclasses.replace(pin, version="0.46.0rc3")
    assert build_identity_sha(_identity(pin=drifted)) != build_identity_sha(_identity())


def _write_marker(export_dir: Path, config_sha: str) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / SUCCESS_MARKER).write_text(
        SuccessManifest(stage="s", config_sha=config_sha).to_json(), encoding="utf-8"
    )


def test_classify_artifact_distinguishes_every_state(tmp_path: Path) -> None:
    sha = build_identity_sha(_identity())
    absent = tmp_path / "absent"
    assert classify_artifact(absent, sha)[0] == ARTIFACT_ABSENT

    no_marker = tmp_path / "no_marker"
    no_marker.mkdir()
    assert classify_artifact(no_marker, sha)[0] == ARTIFACT_NO_MARKER

    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    (unreadable / SUCCESS_MARKER).write_text("{not json", encoding="utf-8")
    assert classify_artifact(unreadable, sha)[0] == ARTIFACT_UNREADABLE_MARKER

    stale = tmp_path / "stale"
    _write_marker(stale, "0" * 64)
    state, detail = classify_artifact(stale, sha)
    assert state == ARTIFACT_IDENTITY_MISMATCH
    assert sha in detail and "0" * 64 in detail

    current = tmp_path / "current"
    _write_marker(current, sha)
    assert classify_artifact(current, sha)[0] == ARTIFACT_MATCH


def _run(d_root: Path, docker: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
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
            "DOCKER": str(docker),
        }
    )
    env.update(overrides)
    return subprocess.run(
        ["bash", str(RUNNER)], cwd=str(REPO), env=env, capture_output=True, text=True, check=False
    )


def _built_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    d_root = tmp_path / "d"
    d_root.mkdir()
    write_fake_source(d_root / "sources" / "clean-bf16")
    docker = write_fake_docker(tmp_path / "fake_docker")
    proc = _run(d_root, docker)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return d_root, docker, d_root / "artifacts" / OUT_NAME


@pytest.mark.private_source_only
def test_rerun_of_an_unchanged_build_is_skipped(tmp_path: Path) -> None:
    d_root, docker, out_dir = _built_root(tmp_path)
    recorded = json.loads((out_dir / SUCCESS_MARKER).read_text())["config_sha"]

    again = _run(d_root, docker)
    assert again.returncode == 0, again.stdout + again.stderr
    assert "idempotent skip" in again.stdout
    # Untouched: the skip did not rebuild or re-promote anything.
    assert json.loads((out_dir / SUCCESS_MARKER).read_text())["config_sha"] == recorded


@pytest.mark.private_source_only
def test_rerun_with_a_different_build_identity_is_refused(tmp_path: Path) -> None:
    """A stale marker under the same artifact name must not report success."""
    d_root, docker, out_dir = _built_root(tmp_path)
    marker = out_dir / SUCCESS_MARKER
    manifest = json.loads(marker.read_text())
    manifest["config_sha"] = "f" * 64  # e.g. the recipe changed since this build
    marker.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    again = _run(d_root, docker)
    output = again.stdout + again.stderr
    assert again.returncode != 0, output
    assert "Refusing overwrite" in output
    assert "build identity changed" in output
    # The prior artifact is left exactly as it was.
    assert json.loads(marker.read_text())["config_sha"] == "f" * 64


@pytest.mark.private_source_only
def test_marker_records_the_full_identity_not_just_the_recipe(tmp_path: Path) -> None:
    _, _, out_dir = _built_root(tmp_path)
    recorded = json.loads((out_dir / SUCCESS_MARKER).read_text())["config_sha"]
    provenance = json.loads((out_dir / "provenance.json").read_text())

    assert provenance["identity_schema_version"]
    assert recorded == build_identity_sha(provenance)
    assert recorded != provenance["recipe_sha256"]


@pytest.mark.private_source_only
def test_abliterated_source_kind_preserves_existing_artifact_path(tmp_path: Path) -> None:
    d_root = tmp_path / "d"
    d_root.mkdir()
    write_fake_source(d_root / "sources" / "clean-bf16")
    docker = write_fake_docker(tmp_path / "fake_docker")

    proc = _run(
        d_root,
        docker,
        SOURCE_KIND="abliterated",
        ALLOW_ABLITERATED="1",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    out_dir = d_root / "artifacts" / "Qwen3.8-27B-darkstar-mlp_only-modelopt-nvfp4"
    assert out_dir.is_dir()
    provenance = json.loads((out_dir / "provenance.json").read_text())
    assert provenance["source_kind"] == "abliterated"


def test_pre_rename_source_kind_normalizes_to_the_canonical_term() -> None:
    assert normalize_source_kind("darkstar") == "abliterated"
    assert normalize_source_kind("abliterated") == "abliterated"
    assert normalize_source_kind("clean") == "clean"


def test_only_the_renamed_term_produces_an_equivalent_identity() -> None:
    abliterated = _identity(source_kind="abliterated")
    assert legacy_source_kind_identity_shas(abliterated) == (
        build_identity_sha(_identity(source_kind="darkstar")),
    )
    # A clean build has no superseded spelling, so it gains no extra matches.
    assert legacy_source_kind_identity_shas(_identity(source_kind="clean")) == ()


def test_equivalence_does_not_extend_to_any_other_identity_field(tmp_path: Path) -> None:
    """The compatibility path renames a term; it never forgives a changed build."""
    current = _identity(source_kind="abliterated")
    equivalent = legacy_source_kind_identity_shas(current)
    sha = build_identity_sha(current)

    renamed_only = tmp_path / "renamed_only"
    _write_marker(renamed_only, build_identity_sha(_identity(source_kind="darkstar")))
    state, detail = classify_artifact(renamed_only, sha, equivalent_identity_shas=equivalent)
    assert state == ARTIFACT_MATCH
    assert "superseded source-kind terminology" in detail

    for label, changed in {
        "recipe digest": _identity(source_kind="darkstar", recipe_sha256="b" * 64),
        "recipe path": _identity(source_kind="darkstar", recipe=OMLP_RECIPE),
        "candidate": _identity(source_kind="darkstar", candidate="omlp"),
        "calibration": _identity(
            source_kind="darkstar",
            calibration=dataclasses.replace(default_calibration_contract(), sizes=(256, 256)),
        ),
        "pin": _identity(source_kind="darkstar", pin=dataclasses.replace(load_pin(), version="9.9")),
        "source dir": _identity(source_kind="darkstar", source_dir=Path("/d/sources/other")),
    }.items():
        stale = tmp_path / label.replace(" ", "_")
        _write_marker(stale, build_identity_sha(changed))
        state, _ = classify_artifact(stale, sha, equivalent_identity_shas=equivalent)
        assert state == ARTIFACT_IDENTITY_MISMATCH, label


def _abliterated_artifact(tmp_path: Path) -> tuple[Path, Path, Path]:
    d_root = tmp_path / "d"
    d_root.mkdir()
    write_fake_source(d_root / "sources" / "clean-bf16")
    docker = write_fake_docker(tmp_path / "fake_docker")
    proc = _run(d_root, docker, SOURCE_KIND="abliterated", ALLOW_ABLITERATED="1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out_dir = d_root / "artifacts" / "Qwen3.8-27B-darkstar-mlp_only-modelopt-nvfp4"
    return d_root, docker, out_dir


def _rewrite_as_pre_rename_build(out_dir: Path, **drift: object) -> str:
    """Restate a validated artifact as one built before the source-kind rename."""
    provenance_path = out_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["source_kind"] = "darkstar"
    provenance.update(drift)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    legacy_sha = build_identity_sha(provenance)
    marker = out_dir / SUCCESS_MARKER
    manifest = json.loads(marker.read_text())
    manifest["config_sha"] = legacy_sha
    marker.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return legacy_sha


@pytest.mark.private_source_only
def test_pre_rename_artifact_is_not_refused_as_stale_under_the_canonical_kind(
    tmp_path: Path,
) -> None:
    """The reviewer case: an artifact validated as `darkstar` stays usable."""
    d_root, docker, out_dir = _abliterated_artifact(tmp_path)
    legacy_sha = _rewrite_as_pre_rename_build(out_dir)

    again = _run(d_root, docker, SOURCE_KIND="abliterated", ALLOW_ABLITERATED="1")
    output = again.stdout + again.stderr
    assert again.returncode == 0, output
    assert "idempotent skip" in again.stdout
    # Accepted, not rewritten: the prior marker is left exactly as it was.
    assert json.loads((out_dir / SUCCESS_MARKER).read_text())["config_sha"] == legacy_sha


@pytest.mark.private_source_only
def test_pre_rename_artifact_with_a_changed_recipe_is_still_refused(tmp_path: Path) -> None:
    d_root, docker, out_dir = _abliterated_artifact(tmp_path)
    legacy_sha = _rewrite_as_pre_rename_build(out_dir, recipe_sha256="b" * 64)

    again = _run(d_root, docker, SOURCE_KIND="abliterated", ALLOW_ABLITERATED="1")
    output = again.stdout + again.stderr
    assert again.returncode != 0, output
    assert "Refusing overwrite" in output
    assert "build identity changed" in output
    assert json.loads((out_dir / SUCCESS_MARKER).read_text())["config_sha"] == legacy_sha


@pytest.mark.private_source_only
def test_deprecated_runner_env_builds_the_same_artifact_as_the_canonical_env(
    tmp_path: Path,
) -> None:
    d_root = tmp_path / "d"
    d_root.mkdir()
    write_fake_source(d_root / "sources" / "clean-bf16")
    docker = write_fake_docker(tmp_path / "fake_docker")

    proc = _run(d_root, docker, SOURCE_KIND="darkstar", ALLOW_DARKSTAR="1")
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert "SOURCE_KIND=darkstar is deprecated" in output
    assert "ALLOW_DARKSTAR is deprecated" in output

    out_dir = d_root / "artifacts" / "Qwen3.8-27B-darkstar-mlp_only-modelopt-nvfp4"
    provenance = json.loads((out_dir / "provenance.json").read_text())
    assert provenance["source_kind"] == "abliterated"

    # The canonical invocation then finds its own identity already validated.
    again = _run(d_root, docker, SOURCE_KIND="abliterated", ALLOW_ABLITERATED="1")
    assert again.returncode == 0, again.stdout + again.stderr
    assert "idempotent skip" in again.stdout


def _cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run(
        [
            sys.executable,
            str(PY_SCRIPT),
            "--run-root", str(tmp_path / "run"),
            "--source-dir", str(tmp_path / "src"),
            *args,
        ],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _reported_identity(stdout: str) -> str:
    (line,) = [ln for ln in stdout.splitlines() if ln.strip().startswith("build identity:")]
    return line.split(":", 1)[1].strip()


def test_deprecated_cli_flags_plan_the_same_build_as_the_canonical_flags(tmp_path: Path) -> None:
    """`--source-kind darkstar --allow-darkstar` is the canonical invocation, renamed."""
    deprecated = _cli(tmp_path, "--source-kind", "darkstar", "--allow-darkstar")
    canonical = _cli(tmp_path, "--source-kind", "abliterated", "--allow-abliterated")
    assert deprecated.returncode == 0, deprecated.stdout + deprecated.stderr
    assert canonical.returncode == 0, canonical.stdout + canonical.stderr

    assert "--source-kind darkstar is deprecated" in deprecated.stderr
    assert "--allow-darkstar is deprecated" in deprecated.stderr
    assert "deprecated" not in canonical.stderr

    # Normalized before anything is planned: the alias leaves no trace in the build.
    assert "kind=abliterated" in deprecated.stdout
    assert _reported_identity(deprecated.stdout) == _reported_identity(canonical.stdout)


def test_deprecated_cli_source_kind_still_faces_the_abliterated_guard(tmp_path: Path) -> None:
    """Normalizing before the guard means the alias is not a way around it."""
    blocked = _cli(tmp_path, "--source-kind", "darkstar")
    assert blocked.returncode == 3, blocked.stdout + blocked.stderr
    assert "requires explicit authorization for heavy artifact mutation" in blocked.stderr


def test_cli_rejects_a_source_kind_that_is_neither_canonical_nor_a_known_alias(
    tmp_path: Path,
) -> None:
    unknown = _cli(tmp_path, "--source-kind", "darkstarr")
    assert unknown.returncode == 2, unknown.stdout + unknown.stderr
    assert "invalid choice" in unknown.stderr
