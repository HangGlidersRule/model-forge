from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from model_forge.serve_profile import ServeProfile, ServeProfileError, render_compose, write_compose

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCT4_REPOSITORY = (
    "HangGlidersRule/"
    "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8"
)
PRODUCT4_ARTIFACT = (
    "/d/model-forge/artifacts/Qwen3.8-27B-abliterated-performance-mixed-modelopt"
)
PRODUCT4_IDENTITY = "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4-Mixed-FP8"
PRODUCT4_PRECISION = "W4A16-NVFP4-Mixed-FP8"
PRODUCT4_SUCCESS_SHA256 = "3d89ec57c1371e142adc2584de079b54a0e1d8c12dc9550118d0a851da020a79"
PRODUCT4_COMPOSE_SHA256 = "5434c2a99bdadce512bd87b65c30f830c21fc2eae647182ffa89e77b174833cc"
PRODUCT4_MANIFEST = (
    REPO_ROOT
    / "models"
    / "qwen3.8-27b-r3"
    / "results"
    / "manifests"
    / "abliterated-modelopt-mixed-manifest.json"
)


def _product4_profile() -> ServeProfile:
    return ServeProfile(
        family="qwen38",
        behavior="abliterated",
        format="nvfp4",
        repository_id=PRODUCT4_REPOSITORY,
        model_path=PRODUCT4_ARTIFACT,
        artifact_identity=PRODUCT4_IDENTITY,
        artifact_precision_class=PRODUCT4_PRECISION,
        artifact_success_sha256=PRODUCT4_SUCCESS_SHA256,
        artifact_manifest=PRODUCT4_MANIFEST,
        artifact_validation="attestation",
        container_name="vllm-darkstar-qwen38-abliterated-modelopt",
        mtp_depth=10,
        scheduler_tokens=32768,
    )


def _local_product4_profile(tmp_path: Path, marker_bytes: bytes | None = None) -> ServeProfile:
    artifact = tmp_path / "artifact-with-no-identity-tokens"
    artifact.mkdir(parents=True)
    if marker_bytes is None:
        marker_bytes = (
            json.dumps(
                {
                    "schema_version": "1.0",
                    "stage": "modelopt_quantize",
                    "config_sha": "fixture-identity",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
    (artifact / "_SUCCESS.json").write_bytes(marker_bytes)
    marker_sha256 = hashlib.sha256(marker_bytes).hexdigest()
    manifest = tmp_path / "artifact-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_path": str(artifact),
                "candidate_id": PRODUCT4_IDENTITY,
                "precision_class": PRODUCT4_PRECISION,
                "success_marker_sha256": marker_sha256,
            }
        ),
        encoding="utf-8",
    )
    return ServeProfile(
        **(
            _product4_profile().__dict__
            | {
                "model_path": str(artifact),
                "artifact_success_sha256": marker_sha256,
                "artifact_manifest": manifest,
                "artifact_validation": "local",
            }
        )
    )


@pytest.mark.private_source_only
def test_product4_runtime_identity_is_distinct_from_repository_identity() -> None:
    profile = _product4_profile()
    profile.validate()
    assert profile.alias == "darkstar-qwen38-abliterated-nvfp4"
    assert profile.compose_filename == "darkstar-qwen38-abliterated-nvfp4.yml"
    assert profile.alias not in profile.repository_id
    assert profile.repository_id.endswith("-ModelOpt-W4A16-NVFP4-Mixed-FP8")


def test_product4_compose_matches_deterministic_renderer() -> None:
    committed = REPO_ROOT / "containers" / "serve" / _product4_profile().compose_filename
    text = committed.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    service = parsed["services"]["vllm"]
    assert service["container_name"] == "vllm-darkstar-qwen38-abliterated-modelopt"
    command = service["command"]
    alias_index = command.index("--served-model-name") + 1
    assert command[alias_index] == "darkstar-qwen38-abliterated-nvfp4"
    if "${PUBLIC_ARTIFACT_PATH}" in text:
        model_index = command.index("--model") + 1
        assert command[model_index] == "${PUBLIC_ARTIFACT_PATH}"
        assert service["volumes"] == [
            f"${{PUBLIC_ARTIFACT_PATH}}:{PRODUCT4_ARTIFACT}:ro"
        ]
    else:
        assert text == render_compose(_product4_profile())
        assert service["volumes"] == [f"{PRODUCT4_ARTIFACT}:{PRODUCT4_ARTIFACT}:ro"]
        assert hashlib.sha256(committed.read_bytes()).hexdigest() == PRODUCT4_COMPOSE_SHA256


def test_hugging_face_model_source_is_not_bind_mounted() -> None:
    profile = ServeProfile(
        **(
            _product4_profile().__dict__
            | {
                "model_path": PRODUCT4_REPOSITORY,
                "artifact_identity": None,
                "artifact_precision_class": None,
                "artifact_success_sha256": None,
                "artifact_manifest": None,
            }
        )
    )
    service = profile.compose()["services"]["vllm"]
    assert "volumes" not in service
    command = service["command"]
    model_index = command.index("--model")
    assert command[model_index : model_index + 2] == ["--model", PRODUCT4_REPOSITORY]


@pytest.mark.private_source_only
def test_write_compose_is_idempotent_and_uses_no_random_filename(tmp_path: Path) -> None:
    first = write_compose(_product4_profile(), tmp_path)
    first_mtime = first.stat().st_mtime_ns
    second = write_compose(_product4_profile(), tmp_path)
    assert second == first
    assert second.name == "darkstar-qwen38-abliterated-nvfp4.yml"
    assert second.stat().st_mtime_ns == first_mtime
    assert list(tmp_path.iterdir()) == [first]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("family", "Qwen3.8"),
        ("container_name", "random-container-123"),
        ("repository_id", "Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-NVFP4"),
        (
            "repository_id",
            "HangGlidersRule/Darkstar-Qwen3.8-27B-Abliterated-ModelOpt-W4A16-NVFP4",
        ),
        (
            "repository_id",
            "HangGlidersRule/Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8",
        ),
        ("model_path", "/d/model-forge/artifacts/unrelated-model"),
        ("artifact_identity", None),
        ("artifact_precision_class", None),
        ("artifact_success_sha256", None),
        ("artifact_manifest", None),
        ("artifact_success_sha256", "0" * 64),
    ],
)
def test_invalid_or_shortened_runtime_profiles_fail_closed(field: str, value: str) -> None:
    values = _product4_profile().__dict__ | {field: value}
    with pytest.raises(ServeProfileError):
        ServeProfile(**values).validate()


def test_local_path_cannot_claim_unrelated_immutable_identity() -> None:
    values = _product4_profile().__dict__ | {
        "artifact_identity": "Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8"
    }
    with pytest.raises(ServeProfileError, match="artifact_identity"):
        ServeProfile(**values).validate()


def test_local_artifact_validation_reads_real_marker_and_ignores_basename_tokens(
    tmp_path: Path,
) -> None:
    _local_product4_profile(tmp_path).validate()


def test_local_artifact_validation_rejects_missing_path_or_marker(tmp_path: Path) -> None:
    profile = _local_product4_profile(tmp_path)
    shutil.rmtree(profile.model_path)
    with pytest.raises(ServeProfileError, match="does not exist"):
        profile.validate()

    profile = _local_product4_profile(tmp_path / "second")
    (Path(profile.model_path) / "_SUCCESS.json").unlink()
    with pytest.raises(ServeProfileError, match="missing _SUCCESS"):
        profile.validate()


def test_local_artifact_validation_rejects_unrelated_marker_bytes(tmp_path: Path) -> None:
    profile = _local_product4_profile(tmp_path, b"unrelated bytes")
    with pytest.raises(ServeProfileError, match="not valid JSON"):
        profile.validate()


def test_local_artifact_validation_rejects_actual_marker_hash_mismatch(tmp_path: Path) -> None:
    profile = _local_product4_profile(tmp_path)
    assert profile.artifact_manifest is not None
    wrong_sha = "f" * 64
    manifest = json.loads(profile.artifact_manifest.read_text(encoding="utf-8"))
    manifest["success_marker_sha256"] = wrong_sha
    profile.artifact_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    profile = ServeProfile(
        **(profile.__dict__ | {"artifact_success_sha256": wrong_sha})
    )
    with pytest.raises(ServeProfileError, match="actual _SUCCESS.json SHA-256"):
        profile.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_id", "Darkstar-Qwen3.8-27B-Base-ModelOpt-W4A16-NVFP4-Mixed-FP8", "candidate_id"),
        ("precision_class", "W4A4-NVFP4", "precision_class"),
        ("success_marker_sha256", "f" * 64, "SHA-256"),
    ],
)
def test_local_artifact_validation_rejects_wrong_manifest_metadata(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    profile = _local_product4_profile(tmp_path)
    assert profile.artifact_manifest is not None
    manifest = json.loads(profile.artifact_manifest.read_text(encoding="utf-8"))
    manifest[field] = value
    profile.artifact_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ServeProfileError, match=message):
        profile.validate()


@pytest.mark.private_source_only
def test_documented_generator_runs_from_clean_checkout_without_pythonpath(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    for raw_path in tracked:
        if not raw_path:
            continue
        relative = Path(os.fsdecode(raw_path))
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    }
    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_darkstar_serve_profile.py",
            "--family",
            "qwen38",
            "--behavior",
            "abliterated",
            "--format",
            "nvfp4",
            "--repository-id",
            PRODUCT4_REPOSITORY,
            "--model-path",
            PRODUCT4_ARTIFACT,
            "--artifact-identity",
            PRODUCT4_IDENTITY,
            "--artifact-precision-class",
            PRODUCT4_PRECISION,
            "--artifact-success-sha256",
            PRODUCT4_SUCCESS_SHA256,
            "--artifact-manifest",
            "models/qwen3.8-27b-r3/results/manifests/abliterated-modelopt-mixed-manifest.json",
            "--artifact-validation",
            "attestation",
            "--container-name",
            "vllm-darkstar-qwen38-abliterated-modelopt",
            "--mtp-depth",
            "10",
            "--scheduler-tokens",
            "32768",
        ],
        cwd=checkout,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    rendered = checkout / "containers/serve/darkstar-qwen38-abliterated-nvfp4.yml"
    assert rendered.read_text(encoding="utf-8") == render_compose(_product4_profile())
