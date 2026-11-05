"""Checkpoint reads accept a Hugging Face snapshot, including the shared blob store."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from model_forge.pipeline import CheckpointTree, UnsafeFileError

ETAG = "a" * 40
XET = "ab" + "c" * 62
REV = "d" * 40


def _snapshot(cache: Path) -> Path:
    repo = cache / "models--org--name"
    (repo / "blobs").mkdir(parents=True)
    (repo / "snapshots" / REV).mkdir(parents=True)
    return repo / "snapshots" / REV


def _point(snapshot: Path, rel: str, etag: str) -> None:
    path = snapshot / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    up = "../" * (len(Path(rel).parts) + 1)
    path.symlink_to(f"{up}blobs/{etag}")


def test_shared_blob_pointer_reads_the_store_file(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    payload = tmp_path / "blobs" / XET[:2] / XET
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"shared-weights")
    # A same-shaped file under the repo must not be what the pointer opens.
    decoy = tmp_path / "models--org--name" / "blobs" / XET[:2] / XET
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(b"decoy")
    etag = tmp_path / "models--org--name" / "blobs" / ETAG
    etag.symlink_to(os.path.relpath(payload, start=etag.parent))
    _point(snapshot, "config.json", ETAG)
    _point(snapshot, "tokenizer/tokenizer.json", ETAG)

    tree = CheckpointTree(snapshot)
    assert tree.open("config.json").read_bytes() == b"shared-weights"
    assert tree.open("tokenizer/tokenizer.json").read_bytes() == b"shared-weights"


def test_classic_repo_blob_still_opens(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    blob = tmp_path / "models--org--name" / "blobs" / ETAG
    blob.write_bytes(b"classic")
    _point(snapshot, "config.json", ETAG)
    assert CheckpointTree(snapshot).open("config.json").read_bytes() == b"classic"


def test_mixed_classic_and_shared_blobs(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    classic = tmp_path / "models--org--name" / "blobs" / ("b" * 40)
    classic.write_bytes(b"classic")
    shared = tmp_path / "blobs" / XET[:2] / XET
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"shared")
    etag = tmp_path / "models--org--name" / "blobs" / ETAG
    etag.symlink_to(Path("..") / ".." / "blobs" / XET[:2] / XET)
    _point(snapshot, "a.txt", "b" * 40)
    _point(snapshot, "b.txt", ETAG)
    tree = CheckpointTree(snapshot)
    assert tree.open("a.txt").read_bytes() == b"classic"
    assert tree.open("b.txt").read_bytes() == b"shared"


@pytest.mark.parametrize(
    "target",
    [
        "/etc/passwd",
        "../../../outside",
        f"../../blobs/ff/{'ff' + 'c' * 62}",
        f"../../blobs/{XET[:2]}/{XET}/extra",
        f"../../blobs/{XET[:2]}/{XET.upper()}",
    ],
)
def test_repo_blob_symlink_must_be_the_shared_store_pointer(tmp_path: Path, target: str) -> None:
    snapshot = _snapshot(tmp_path)
    etag = tmp_path / "models--org--name" / "blobs" / ETAG
    etag.symlink_to(target)
    _point(snapshot, "config.json", ETAG)
    with pytest.raises(UnsafeFileError, match="cannot be opened without following links"):
        CheckpointTree(snapshot).open("config.json")


def test_shared_payload_that_is_a_symlink_is_refused(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    outside = tmp_path / "secret"
    outside.write_bytes(b"secret")
    payload = tmp_path / "blobs" / XET[:2] / XET
    payload.parent.mkdir(parents=True)
    payload.symlink_to(outside)
    etag = tmp_path / "models--org--name" / "blobs" / ETAG
    etag.symlink_to(Path("..") / ".." / "blobs" / XET[:2] / XET)
    _point(snapshot, "config.json", ETAG)
    with pytest.raises(UnsafeFileError, match="cannot be opened without following links"):
        CheckpointTree(snapshot).open("config.json")


def test_shared_store_directory_that_is_a_symlink_is_refused(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    real = tmp_path / "elsewhere"
    (real / XET[:2]).mkdir(parents=True)
    (real / XET[:2] / XET).write_bytes(b"elsewhere")
    (tmp_path / "blobs").symlink_to(real, target_is_directory=True)
    etag = tmp_path / "models--org--name" / "blobs" / ETAG
    etag.symlink_to(Path("..") / ".." / "blobs" / XET[:2] / XET)
    _point(snapshot, "config.json", ETAG)
    with pytest.raises(UnsafeFileError, match="cannot be opened without following links"):
        CheckpointTree(snapshot).open("config.json")


def test_symlink_outside_a_hub_snapshot_is_refused(tmp_path: Path) -> None:
    (tmp_path / "config.json").symlink_to("/etc/passwd")
    with pytest.raises(UnsafeFileError, match="symlink"):
        CheckpointTree(tmp_path).open("config.json")
