"""Spec: a same-size safetensors swap must not reuse another checkpoint's sweep.

``artifact_sha`` currently folds each ``.safetensors`` file in as its relative
path and byte length. ``run_sweep`` then loads ``<results>/<sha>/<config>/<candidate>.json``
whenever ``--force`` is absent. These tests drive that resume path, including partial
resume, ``--dry-run``, and the ``tune`` CLI. They fail while a same-size shard
replacement still contributes the previous lane means to ``winner_from`` and the
persisted report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from model_forge.cli import main
from model_forge.tune import (
    TuneCandidate,
    TuneMatrix,
    artifact_sha,
    candidate_cache_path,
    run_sweep,
)

_SHARD_SIZE = 4096
_STALE_MTP1 = {"4": 100.0, "16": 100.0}
_STALE_MTP2 = {"4": 50.0, "16": 40.0}
_STALE_DRAFTER = {"4": 30.0, "16": 30.0}
_FRESH_MTP1 = {"4": 10.0, "16": 10.0}
_FRESH_MTP2 = {"4": 90.0, "16": 70.0}
_FRESH_DRAFTER = {"4": 20.0, "16": 20.0}


def _shard_bytes(marker: bytes) -> bytes:
    if len(marker) >= _SHARD_SIZE:
        raise AssertionError("shard marker must be shorter than the fixed shard size")
    pad = bytes((index % 251) for index in range(_SHARD_SIZE - len(marker)))
    return marker + pad


def _lane_script(*candidates: tuple[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {key: lanes for key, lanes in candidates}


def _matrix() -> TuneMatrix:
    return TuneMatrix(
        mtp_min=1,
        mtp_max=2,
        drafters=(("dflash", "org/draft", 4),),
        lanes_k=(4, 16),
        lane_weights=((4, 0.6), (16, 0.4)),
        runs=1,
        warmup=0,
        max_tokens=8,
        temperature=0.0,
    )


def _cli_matrix_args() -> list[str]:
    return [
        "--mtp-min",
        "1",
        "--mtp-max",
        "2",
        "--lanes",
        "4,16",
        "--lane-weights",
        "4:0.6,16:0.4",
        "--runs",
        "1",
        "--warmup",
        "0",
        "--max-tokens",
        "8",
        "--temperature",
        "0",
    ]


def _write_artifact(root: Path, shard_payload: bytes) -> tuple[Path, Path]:
    artifact = root / "artifact"
    nested = artifact / "weights"
    nested.mkdir(parents=True)
    (artifact / "config.json").write_text('{"model_type":"decoder","hidden_size":4}\n')
    (artifact / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.weight": "model-00001-of-00002.safetensors",
                    "model.layers.1.weight": "weights/model-00002-of-00002.safetensors",
                }
            }
        )
    )
    swapped = artifact / "model-00001-of-00002.safetensors"
    swapped.write_bytes(shard_payload)
    (nested / "model-00002-of-00002.safetensors").write_bytes(_shard_bytes(b"nested-unchanged"))
    (artifact / "README.md").write_text("not part of the sweep identity\n")
    return artifact, swapped


def _replace_shard_inplace(shard: Path, payload: bytes) -> None:
    """Overwrite shard bytes without renaming the file or changing its length."""
    before = shard.stat()
    if len(payload) != before.st_size:
        raise AssertionError("replacement must keep the shard length")
    with shard.open("r+b") as handle:
        handle.seek(0)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    after = shard.stat()
    assert after.st_size == before.st_size
    assert after.st_ino == before.st_ino
    assert shard.read_bytes() == payload


def _sweep_kwargs(artifact: Path, results: Path, *, dry_run: bool = False) -> dict[str, Any]:
    return {
        "artifact_dir": str(artifact),
        "served_name": "model",
        "image": "runtime:test",
        "matrix": _matrix(),
        "results_dir": results,
        "host": "unreachable.invalid",
        "user": "devin",
        "key": "/tmp/model-forge-tune-identity-unused",
        "force": False,
        "dry_run": dry_run,
    }


def _install_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    scripts: dict[str, dict[str, dict[str, float]]],
    active: dict[str, str],
    calls: list[str],
) -> None:
    current: dict[str, str | None] = {"key": None}

    def boot(remote: object, spec: Any, project: str, *, base: str, boot_timeout: int = 420) -> None:
        del remote, project, base, boot_timeout
        current["key"] = spec.candidate.key
        calls.append(spec.candidate.key)

    def bench(base: str, model: str, matrix: TuneMatrix) -> dict[str, dict[str, Any]]:
        del base, model, matrix
        key = current["key"]
        if key is None:
            raise AssertionError("benchmark ran without the candidate boot that run_sweep performs")
        lanes = scripts[active["name"]][key]
        return {
            str(lane): {
                "lane_k": int(lane),
                "mean": mean,
                "min": mean,
                "max": mean,
                "n_valid": 2,
                "skipped": 0,
            }
            for lane, mean in lanes.items()
        }

    monkeypatch.setattr("model_forge.tune.boot_and_wait", boot)
    monkeypatch.setattr("model_forge.tune._benchmark_round", bench)
    monkeypatch.setattr("model_forge.tune.Remote.down", lambda *args, **kwargs: None)


def _persisted(results: Path, report: dict[str, Any]) -> tuple[Path, dict[str, Any], str]:
    directory = results / str(report["artifact_sha"]) / str(report["config_hash"])
    path = directory / "report.json"
    return path, json.loads(path.read_text()), (directory / "report.md").read_text()


def _fresh_lanes() -> dict[str, dict[str, float]]:
    return {
        "mtp1": _FRESH_MTP1,
        "mtp2": _FRESH_MTP2,
        "dflash4-org-draft": _FRESH_DRAFTER,
    }


def _assert_fresh_speed_report(results: Path, report: dict[str, Any], keys: list[str]) -> None:
    assert report["winner"] == "mtp2"
    path, saved, markdown = _persisted(results, report)
    assert path.is_file()
    assert saved["winner"] == "mtp2"
    assert saved["winner_reason"] == report["winner_reason"]
    for key in keys:
        for lane, mean in _fresh_lanes()[key].items():
            assert report["results"][key][lane]["mean"] == mean
            assert saved["results"][key][lane]["mean"] == mean
    assert "**mtp2**" in markdown
    assert "100.0" not in markdown
    assert "50.0" not in markdown


def _scripts(keys: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    stale = _lane_script(
        ("mtp1", _STALE_MTP1),
        ("mtp2", _STALE_MTP2),
        ("dflash4-org-draft", _STALE_DRAFTER),
    )
    fresh = _lane_script(
        ("mtp1", _FRESH_MTP1),
        ("mtp2", _FRESH_MTP2),
        ("dflash4-org-draft", _FRESH_DRAFTER),
    )
    return {
        "stale": {key: stale[key] for key in keys},
        "fresh": {key: fresh[key] for key in keys},
    }


def test_same_size_inplace_shard_swap_remeasures_and_resumes_only_those_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact, shard = _write_artifact(tmp_path, _shard_bytes(b"checkpoint-v1"))
    results = tmp_path / "results"
    matrix = _matrix()
    keys = [candidate.key for candidate in matrix.candidates()]
    calls: list[str] = []
    active = {"name": "stale"}
    _install_benchmark(monkeypatch, _scripts(keys), active, calls)

    first = run_sweep(**_sweep_kwargs(artifact, results))
    assert calls == keys
    assert first["winner"] == "mtp1"
    assert first["results"]["mtp1"]["4"]["mean"] == _STALE_MTP1["4"]
    config_json = (artifact / "config.json").read_bytes()
    index_json = (artifact / "model.safetensors.index.json").read_bytes()

    calls.clear()
    active["name"] = "fresh"
    _replace_shard_inplace(shard, _shard_bytes(b"checkpoint-v2"))
    assert (artifact / "config.json").read_bytes() == config_json
    assert (artifact / "model.safetensors.index.json").read_bytes() == index_json

    second = run_sweep(**_sweep_kwargs(artifact, results))
    assert calls == keys
    _assert_fresh_speed_report(results, second, keys)

    calls.clear()
    third = run_sweep(**_sweep_kwargs(artifact, results))
    assert calls == []
    _assert_fresh_speed_report(results, third, keys)


def test_partial_resume_does_not_keep_a_cached_lane_after_same_size_shard_swap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact, shard = _write_artifact(tmp_path, _shard_bytes(b"checkpoint-v1"))
    results = tmp_path / "results"
    matrix = _matrix()
    keys = [candidate.key for candidate in matrix.candidates()]
    calls: list[str] = []
    active = {"name": "stale"}
    _install_benchmark(monkeypatch, _scripts(keys), active, calls)
    first = run_sweep(**_sweep_kwargs(artifact, results))
    assert first["winner"] == "mtp1"

    cached_mtp2 = candidate_cache_path(
        results, artifact_sha(str(artifact)), TuneCandidate("mtp", 2), matrix
    )
    cached_mtp2.unlink()
    assert cached_mtp2.is_file() is False
    _replace_shard_inplace(shard, _shard_bytes(b"checkpoint-v2"))

    calls.clear()
    active["name"] = "fresh"
    second = run_sweep(**_sweep_kwargs(artifact, results))
    assert calls == keys
    _assert_fresh_speed_report(results, second, keys)
    assert second["results"]["mtp1"]["4"]["mean"] != _STALE_MTP1["4"]
    assert second["results"]["dflash4-org-draft"]["4"]["mean"] == _FRESH_DRAFTER["4"]


def test_dry_run_does_not_publish_cached_lanes_after_same_size_shard_swap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact, shard = _write_artifact(tmp_path, _shard_bytes(b"checkpoint-v1"))
    results = tmp_path / "results"
    matrix = _matrix()
    keys = [candidate.key for candidate in matrix.candidates()]
    calls: list[str] = []
    active = {"name": "stale"}
    _install_benchmark(monkeypatch, _scripts(keys), active, calls)
    first = run_sweep(**_sweep_kwargs(artifact, results))
    assert first["winner"] == "mtp1"
    _replace_shard_inplace(shard, _shard_bytes(b"checkpoint-v2"))

    calls.clear()
    active["name"] = "fresh"
    report = run_sweep(**_sweep_kwargs(artifact, results, dry_run=True))
    assert calls == []
    assert report["winner"] == ""
    assert report["winner_reason"] == "no valid lanes"
    assert report["results"] == {key: {"dry_run": True} for key in keys}
    _path, saved, markdown = _persisted(results, report)
    assert saved["winner"] == ""
    assert saved["results"] == {key: {"dry_run": True} for key in keys}
    assert "100.0" not in markdown
    assert "**mtp1**" not in markdown


def test_cache_lookup_uses_the_shard_bytes_read_for_the_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail a fix that reads the shard and then keys the cache by path and size.

    After the first content read of the swapped shard, the file is overwritten
    in place with a third same-size payload. A later size lookup still sees the
    original length, so it must not be enough to accept the first checkpoint's
    lane file.
    """
    artifact, shard = _write_artifact(tmp_path, _shard_bytes(b"checkpoint-v1"))
    results = tmp_path / "results"
    matrix = _matrix()
    keys = [candidate.key for candidate in matrix.candidates()]
    calls: list[str] = []
    active = {"name": "stale"}
    _install_benchmark(monkeypatch, _scripts(keys), active, calls)
    first = run_sweep(**_sweep_kwargs(artifact, results))
    assert first["winner"] == "mtp1"

    _replace_shard_inplace(shard, _shard_bytes(b"checkpoint-v2"))
    third_payload = _shard_bytes(b"checkpoint-v3")
    seen = {"swapped": False}
    original_read_bytes = Path.read_bytes

    def read_bytes(self: Path) -> bytes:
        data = original_read_bytes(self)
        if not seen["swapped"] and self.resolve() == shard.resolve():
            seen["swapped"] = True
            _replace_shard_inplace(self, third_payload)
        return data

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    calls.clear()
    active["name"] = "fresh"
    second = run_sweep(**_sweep_kwargs(artifact, results))
    assert calls == keys
    _assert_fresh_speed_report(results, second, keys)
    # Read with the unpatched method: the patched one would perform the swap itself here.
    assert seen["swapped"] is False or original_read_bytes(shard) == third_payload
    assert second["results"]["mtp1"]["4"]["mean"] == _FRESH_MTP1["4"]


def test_identical_shard_bytes_still_resume_without_booting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact, shard = _write_artifact(tmp_path, _shard_bytes(b"checkpoint-v1"))
    results = tmp_path / "results"
    matrix = _matrix()
    keys = [candidate.key for candidate in matrix.candidates()]
    calls: list[str] = []
    active = {"name": "stale"}
    _install_benchmark(monkeypatch, _scripts(keys), active, calls)
    first = run_sweep(**_sweep_kwargs(artifact, results))
    assert calls == keys
    assert first["winner"] == "mtp1"

    payload = shard.read_bytes()
    _replace_shard_inplace(shard, payload)
    os.utime(shard, (1_700_000_000, 1_700_000_000))
    calls.clear()
    active["name"] = "fresh"
    second = run_sweep(**_sweep_kwargs(artifact, results))
    assert calls == []
    assert second["winner"] == "mtp1"
    assert second["results"]["mtp1"]["4"]["mean"] == _STALE_MTP1["4"]
    assert second["artifact_sha"] == first["artifact_sha"]


def test_cli_tune_does_not_reuse_speed_winner_after_same_size_shard_swap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact, shard = _write_artifact(tmp_path, _shard_bytes(b"checkpoint-v1"))
    results = tmp_path / "results"
    keys = ["mtp1", "mtp2"]
    calls: list[str] = []
    active = {"name": "stale"}
    scripts = _scripts(keys)
    _install_benchmark(monkeypatch, scripts, active, calls)
    argv = [
        "tune",
        "--artifact-dir",
        str(artifact),
        "--served-name",
        "model",
        "--image",
        "runtime:test",
        "--results-dir",
        str(results),
        "--remote-host",
        "unreachable.invalid",
        "--remote-user",
        "devin",
        "--ssh-key",
        str(tmp_path / "unused-key"),
        *_cli_matrix_args(),
    ]

    assert main(argv) == 0
    first_out = capsys.readouterr().out
    assert "Winner: mtp1" in first_out
    assert calls == keys

    _replace_shard_inplace(shard, _shard_bytes(b"checkpoint-v2"))
    calls.clear()
    active["name"] = "fresh"
    assert main(argv) == 0
    second_out = capsys.readouterr().out
    assert calls == keys
    assert "[tune] mtp1: cached" not in second_out
    assert "[tune] mtp2: cached" not in second_out
    assert "Winner: mtp2" in second_out
    assert "100.0" not in second_out
    assert "50.0" not in second_out

    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in results.rglob("report.json"):
        saved = json.loads(path.read_text())
        if (
            saved["winner"] == "mtp2"
            and saved["results"]["mtp1"]["4"]["mean"] == _FRESH_MTP1["4"]
        ):
            matches.append((path, saved))
    assert len(matches) == 1
    path, saved = matches[0]
    markdown = (path.parent / "report.md").read_text()
    assert saved["results"]["mtp2"]["4"]["mean"] == _FRESH_MTP2["4"]
    assert "**mtp2**" in markdown
    assert "100.0" not in markdown


@pytest.mark.parametrize("suffix", [".bin", ".pt"])
def test_other_weight_formats_are_part_of_the_artifact_identity(tmp_path: Path, suffix: str) -> None:
    """vLLM also loads .bin and .pt weights, so their bytes must key the cache too."""
    artifact, _shard = _write_artifact(tmp_path, _shard_bytes(b"checkpoint-v1"))
    weights = artifact / f"pytorch_model{suffix}"
    weights.write_bytes(_shard_bytes(b"legacy-weights-v1"))
    before = artifact_sha(str(artifact))
    _replace_shard_inplace(weights, _shard_bytes(b"legacy-weights-v2"))
    assert artifact_sha(str(artifact)) != before


def test_shard_bytes_past_the_first_mebibyte_change_the_identity(tmp_path: Path) -> None:
    """The whole shard is hashed, not a prefix of it."""
    head = _shard_bytes(b"checkpoint-v1")
    tail = 2 * 1024 * 1024
    artifact, shard = _write_artifact(tmp_path, head + bytes(tail))
    before = artifact_sha(str(artifact))
    _replace_shard_inplace(shard, head + bytes(tail - 1) + b"\x01")
    assert artifact_sha(str(artifact)) != before


def test_identity_is_the_tagged_content_scheme(tmp_path: Path) -> None:
    """Keys written under the former path-and-size scheme can never equal a content key.

    The expected digest is rebuilt from the documented layout: the scheme tag, then for
    every JSON or weight file its relative path, a NUL, its sha256 and a NUL.
    """
    import hashlib

    artifact, _shard = _write_artifact(tmp_path, _shard_bytes(b"checkpoint-v1"))
    expected = hashlib.sha256(b"model-forge-artifact-sha/v2-content\0")
    for path in sorted(artifact.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".pt"}:
            expected.update(str(path.relative_to(artifact)).encode() + b"\0")
            expected.update(hashlib.sha256(path.read_bytes()).hexdigest().encode() + b"\0")
    assert artifact_sha(str(artifact)) == expected.hexdigest()
