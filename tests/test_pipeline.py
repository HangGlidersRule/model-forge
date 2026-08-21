"""Tests for pipeline manifest primitives."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from model_forge.pipeline import (
    ConcurrentRunError,
    RunLock,
    StageContext,
    SuccessManifest,
    canonical_json,
    config_hash,
    quarantine_partial,
    sha256_bytes,
    sha256_file,
    sha256_tree,
    validate_success_marker,
)


def test_canonical_json_deterministic() -> None:
    a = canonical_json({"b": 2, "a": 1})
    b = canonical_json({"a": 1, "b": 2})
    assert a == b == '{"a":1,"b":2}'


def test_sha256_bytes() -> None:
    h = sha256_bytes(b"hello")
    assert len(h) == 64
    assert h == sha256_bytes(b"hello")
    assert h != sha256_bytes(b"world")


def test_sha256_file(tmp_path: Path) -> None:
    f = tmp_path / "test.bin"
    f.write_bytes(b"data")
    assert sha256_file(f) == sha256_bytes(b"data")


def test_sha256_tree(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("beta")
    tree = sha256_tree(tmp_path)
    assert "a.txt" in tree
    assert "sub/b.txt" in tree
    assert len(tree) == 2


def test_config_hash_stable() -> None:
    cfg = {"layer": 38, "seed": 42, "model": "Qwen/Qwen3.8-27B"}
    h1 = config_hash(cfg)
    h2 = config_hash({"seed": 42, "model": "Qwen/Qwen3.8-27B", "layer": 38})
    assert h1 == h2


def test_success_manifest_roundtrip() -> None:
    m = SuccessManifest(
        stage="measure",
        config_sha="abc123",
        output_hashes={"direction.safetensors": "def456"},
    )
    text = m.to_json()
    restored = SuccessManifest.from_json(text)
    assert restored.stage == "measure"
    assert restored.config_sha == "abc123"
    assert restored.output_hashes == {"direction.safetensors": "def456"}


def test_validate_success_marker_clean_run(tmp_path: Path) -> None:
    stage_dir = tmp_path / "measure"
    stage_dir.mkdir()
    data_file = stage_dir / "output.bin"
    data_file.write_bytes(b"result")
    file_hash = sha256_file(data_file)
    m = SuccessManifest(
        stage="measure", config_sha="cfg1", output_hashes={"output.bin": file_hash}
    )
    (stage_dir / "_SUCCESS.json").write_text(m.to_json())
    result = validate_success_marker(stage_dir, "cfg1")
    assert result is not None
    assert result.stage == "measure"


def test_validate_success_marker_config_mismatch(tmp_path: Path) -> None:
    stage_dir = tmp_path / "measure"
    stage_dir.mkdir()
    data_file = stage_dir / "output.bin"
    data_file.write_bytes(b"result")
    m = SuccessManifest(
        stage="measure",
        config_sha="cfg1",
        output_hashes={"output.bin": sha256_file(data_file)},
    )
    (stage_dir / "_SUCCESS.json").write_text(m.to_json())
    assert validate_success_marker(stage_dir, "different_cfg") is None


def test_validate_success_marker_missing_output(tmp_path: Path) -> None:
    stage_dir = tmp_path / "measure"
    stage_dir.mkdir()
    m = SuccessManifest(
        stage="measure", config_sha="cfg1", output_hashes={"missing.bin": "abc"}
    )
    (stage_dir / "_SUCCESS.json").write_text(m.to_json())
    assert validate_success_marker(stage_dir, "cfg1") is None


def test_validate_success_marker_corrupt_hash(tmp_path: Path) -> None:
    stage_dir = tmp_path / "measure"
    stage_dir.mkdir()
    data_file = stage_dir / "output.bin"
    data_file.write_bytes(b"result")
    m = SuccessManifest(
        stage="measure", config_sha="cfg1", output_hashes={"output.bin": "wrong_hash"}
    )
    (stage_dir / "_SUCCESS.json").write_text(m.to_json())
    assert validate_success_marker(stage_dir, "cfg1") is None


def test_quarantine_partial(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    (stage_dir / "data.bin").write_bytes(b"partial")
    q = quarantine_partial(stage_dir)
    assert q is not None
    assert not stage_dir.exists()
    assert q.exists()
    assert "quarantine" in q.name


def test_quarantine_nonexistent(tmp_path: Path) -> None:
    assert quarantine_partial(tmp_path / "nope") is None


def test_stage_context_skip_on_valid(tmp_path: Path) -> None:
    stage_dir = tmp_path / "run" / "test_stage"
    stage_dir.mkdir(parents=True)
    data_file = stage_dir / "out.txt"
    data_file.write_text("hello")
    m = SuccessManifest(
        stage="test_stage",
        config_sha="sha1",
        output_hashes={"out.txt": sha256_file(data_file)},
    )
    (stage_dir / "_SUCCESS.json").write_text(m.to_json())
    ctx = StageContext(tmp_path / "run", "test_stage", "sha1")
    assert ctx.should_skip() is not None


def test_stage_context_full_lifecycle(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    ctx = StageContext(run_root, "build", "cfghash")
    assert ctx.should_skip() is None
    partial = ctx.partial_dir
    out = partial / "model.bin"
    out.write_bytes(b"weights")
    manifest = SuccessManifest(
        stage="build",
        config_sha="cfghash",
        output_hashes={"model.bin": sha256_file(out)},
    )
    ctx.promote(manifest)
    assert ctx.stage_dir.exists()
    assert (ctx.stage_dir / "model.bin").read_bytes() == b"weights"
    assert not partial.exists()


def test_stage_context_quarantines_stale(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    stale = run_root / "build"
    stale.mkdir()
    (stale / "old.bin").write_bytes(b"stale")
    ctx = StageContext(run_root, "build", "newhash")
    partial = ctx.partial_dir
    out = partial / "new.bin"
    out.write_bytes(b"fresh")
    manifest = SuccessManifest(
        stage="build",
        config_sha="newhash",
        output_hashes={"new.bin": sha256_file(out)},
    )
    ctx.promote(manifest)
    quarantined = list(run_root.glob("build.quarantine.*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "old.bin").read_bytes() == b"stale"


def test_run_lock_exclusive(tmp_path: Path) -> None:
    lock_path = tmp_path / "run.lock"
    lock1 = RunLock(lock_path)
    lock1.acquire()
    with pytest.raises(ConcurrentRunError):
        lock2 = RunLock(lock_path)
        lock2.acquire()
    lock1.release()


def test_run_lock_context_manager(tmp_path: Path) -> None:
    lock_path = tmp_path / "run.lock"
    with RunLock(lock_path):
        assert lock_path.exists()
    lock2 = RunLock(lock_path)
    lock2.acquire()
    lock2.release()


def test_run_lock_concurrent_threads(tmp_path: Path) -> None:
    lock_path = tmp_path / "run.lock"
    results: list[str] = []
    held = threading.Event()
    release = threading.Event()

    def holder() -> None:
        try:
            with RunLock(lock_path):
                results.append("a:ok")
                held.set()
                release.wait(timeout=2.0)
        except ConcurrentRunError:
            results.append("a:blocked")

    def contender() -> None:
        assert held.wait(timeout=2.0)
        try:
            with RunLock(lock_path):
                results.append("b:ok")
        except ConcurrentRunError:
            results.append("b:blocked")
        finally:
            release.set()

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=contender)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert "a:ok" in results
    assert "b:blocked" in results
