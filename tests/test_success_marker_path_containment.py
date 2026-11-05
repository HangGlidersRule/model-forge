"""Success-marker checks must bind the directory resume or promotion reuses.

``validate_success_marker``, ``StageContext.should_skip``, and ``promote_atomic``
today accept a marker whose ``output_hashes`` map is empty, or whose keys open a
file outside the artifact directory. The poisoned directory is then skipped or
renamed as if its weights had been verified. These tests fail until a marker is
refused unless every key stays inside the artifact root, symlinks are refused,
and the key set equals the regular files on disk (the marker file itself is the
record of those hashes, not one of them).

Refusal must happen before any create, delete, or rename. A check that
normalizes a key lexically and then opens the original path again still fails
``test_symlink_parent_reread``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from model_forge.experiment import load_experiment
from model_forge.modelopt.finalize import PromotionError, promote_atomic
from model_forge.pipeline import (
    SUCCESS_MARKER,
    StageContext,
    SuccessManifest,
    sha256_file,
    validate_success_marker,
)

CONFIG_SHA = "cfg-under-test"
ATTACKER = b"attacker-weights-not-the-known-file"
KNOWN = b"world-readable-known-bytes"
RECIPE = (
    Path(__file__).resolve().parent.parent / "recipes" / "qwen3.8-27b" / "r3-nvfp4.yaml"
)

_script_path = (
    Path(__file__).resolve().parent.parent / "scripts" / "qwen3_8" / "quantize_qwen38_nvfp4.py"
)
_spec = importlib.util.spec_from_file_location("quantize_qwen38_nvfp4_marker_attack", _script_path)
assert _spec is not None and _spec.loader is not None
_nvfp4 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _nvfp4
_spec.loader.exec_module(_nvfp4)


def _snapshot(root: Path) -> tuple[tuple[str, str, int | str, bytes], ...]:
    rows: list[tuple[str, str, int | str, bytes]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        for name in sorted(dirnames):
            path = base / name
            if path.is_symlink():
                rows.append(("symlink-dir", str(path.relative_to(root)), os.readlink(path), b""))
        for name in sorted(filenames):
            path = base / name
            rel = str(path.relative_to(root))
            if path.is_symlink():
                rows.append(("symlink", rel, os.readlink(path), b""))
            else:
                info = path.stat()
                rows.append(("file", rel, info.st_ino, path.read_bytes()))
    return tuple(rows)


def _write_marker(directory: Path, output_hashes: dict[str, str], config_sha: str) -> None:
    manifest = SuccessManifest(
        stage=directory.name,
        config_sha=config_sha,
        output_hashes=output_hashes,
    )
    (directory / SUCCESS_MARKER).write_text(manifest.to_json())


def _assert_resume_rejected(artifact: Path, config_sha: str) -> None:
    before = _snapshot(artifact)
    skipped = StageContext(artifact.parent, artifact.name, config_sha).should_skip()
    assert skipped is None
    checked = validate_success_marker(artifact, config_sha)
    assert checked is None
    assert _snapshot(artifact) == before


def _assert_promote_rejected(partial: Path, outside: Path | None) -> None:
    before = _snapshot(partial)
    outside_bytes = outside.read_bytes() if outside is not None else None
    outside_ino = outside.stat().st_ino if outside is not None else None
    final = partial.parent / "missing-parent" / "final"
    assert not final.parent.exists()
    with pytest.raises(PromotionError):
        promote_atomic(partial, final)
    assert not final.exists()
    assert not final.parent.exists()
    assert partial.exists()
    assert _snapshot(partial) == before
    if outside is not None and outside_bytes is not None:
        assert outside.read_bytes() == outside_bytes
        assert outside.stat().st_ino == outside_ino


def _plant_weights(artifact: Path) -> None:
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "weights.bin").write_bytes(ATTACKER)


def _known_file(tmp_path: Path) -> Path:
    known = tmp_path / "known.bin"
    known.write_bytes(KNOWN)
    return known


@pytest.mark.parametrize("entry", ["resume", "promote"])
def test_empty_hash_map_over_different_weights(tmp_path: Path, entry: str) -> None:
    artifact = tmp_path / "stage"
    _plant_weights(artifact)
    _write_marker(artifact, {}, CONFIG_SHA)
    if entry == "resume":
        _assert_resume_rejected(artifact, CONFIG_SHA)
    else:
        _assert_promote_rejected(artifact, None)


@pytest.mark.parametrize("entry", ["resume", "promote"])
def test_empty_hash_map_with_only_the_marker(tmp_path: Path, entry: str) -> None:
    artifact = tmp_path / "stage"
    artifact.mkdir()
    _write_marker(artifact, {}, CONFIG_SHA)
    if entry == "resume":
        _assert_resume_rejected(artifact, CONFIG_SHA)
    else:
        _assert_promote_rejected(artifact, None)


@pytest.mark.parametrize("entry", ["resume", "promote"])
def test_absolute_key_hashes_outside_file(tmp_path: Path, entry: str) -> None:
    artifact = tmp_path / "stage"
    _plant_weights(artifact)
    known = _known_file(tmp_path)
    key = str(known.resolve())
    opened = artifact / key
    assert opened == known.resolve()
    assert opened.read_bytes() == KNOWN
    assert (artifact / "weights.bin").read_bytes() == ATTACKER
    _write_marker(artifact, {key: sha256_file(known)}, CONFIG_SHA)
    if entry == "resume":
        _assert_resume_rejected(artifact, CONFIG_SHA)
    else:
        _assert_promote_rejected(artifact, known)


@pytest.mark.parametrize("entry", ["resume", "promote"])
def test_parent_segment_key_hashes_outside_file(tmp_path: Path, entry: str) -> None:
    artifact = tmp_path / "stage"
    _plant_weights(artifact)
    known = _known_file(tmp_path)
    key = "../known.bin"
    opened = artifact / key
    assert opened.resolve() == known.resolve()
    assert opened.read_bytes() == KNOWN
    assert known.resolve() != (artifact / "weights.bin").resolve()
    _write_marker(artifact, {key: sha256_file(known)}, CONFIG_SHA)
    if entry == "resume":
        _assert_resume_rejected(artifact, CONFIG_SHA)
    else:
        _assert_promote_rejected(artifact, known)


@pytest.mark.parametrize("entry", ["resume", "promote"])
def test_symlink_parent_reread(tmp_path: Path, entry: str) -> None:
    """Lexical collapse of the key stays inside; opening it reads a different file.

    ``normpath(artifact/link/../known.bin)`` is ``artifact/known.bin``. The
    directory entry ``link`` is a symlink, so opening that same key reads the
    sibling ``known.bin`` outside the artifact. A fix that checks the collapsed
    path and then opens the original key again still accepts the outside hash
    while ``artifact/known.bin`` holds different bytes.
    """
    artifact = tmp_path / "artifact"
    outside = tmp_path / "outside"
    artifact.mkdir()
    outside.mkdir()
    (outside / "decoy.bin").write_bytes(b"decoy")
    known = tmp_path / "known.bin"
    known.write_bytes(KNOWN)
    (artifact / "known.bin").write_bytes(ATTACKER)
    (artifact / "link").symlink_to(outside, target_is_directory=True)
    key = "link/../known.bin"
    lexical = Path(os.path.normpath(artifact / key))
    assert lexical == artifact / "known.bin"
    assert lexical.read_bytes() == ATTACKER
    assert (artifact / key).read_bytes() == KNOWN
    assert (artifact / key).resolve() != lexical.resolve()
    _write_marker(artifact, {key: sha256_file(known)}, CONFIG_SHA)
    if entry == "resume":
        _assert_resume_rejected(artifact, CONFIG_SHA)
    else:
        _assert_promote_rejected(artifact, known)


@pytest.mark.parametrize("entry", ["resume", "promote"])
def test_symlink_file_key_whose_name_is_inside_the_tree(tmp_path: Path, entry: str) -> None:
    artifact = tmp_path / "stage"
    artifact.mkdir()
    known = _known_file(tmp_path)
    weights = artifact / "weights.bin"
    weights.symlink_to(known)
    assert weights.is_symlink()
    assert weights.is_file()
    assert not weights.resolve().is_relative_to(artifact.resolve())
    assert weights.read_bytes() == KNOWN
    _write_marker(artifact, {"weights.bin": sha256_file(known)}, CONFIG_SHA)
    if entry == "resume":
        _assert_resume_rejected(artifact, CONFIG_SHA)
    else:
        _assert_promote_rejected(artifact, known)


@pytest.mark.parametrize("entry", ["resume", "promote"])
def test_unlisted_regular_file_is_not_covered_by_a_decoy_hash(tmp_path: Path, entry: str) -> None:
    artifact = tmp_path / "stage"
    artifact.mkdir()
    decoy = artifact / "readme.txt"
    decoy.write_bytes(b"known-decoy")
    (artifact / "weights.bin").write_bytes(ATTACKER)
    _write_marker(artifact, {"readme.txt": sha256_file(decoy)}, CONFIG_SHA)
    if entry == "resume":
        _assert_resume_rejected(artifact, CONFIG_SHA)
    else:
        _assert_promote_rejected(artifact, None)


@pytest.mark.parametrize("entry", ["resume", "promote"])
def test_unlisted_symlink_beside_a_matching_regular_file(tmp_path: Path, entry: str) -> None:
    artifact = tmp_path / "stage"
    artifact.mkdir()
    readme = artifact / "readme.txt"
    readme.write_bytes(b"harmless")
    known = _known_file(tmp_path)
    (artifact / "weights.bin").symlink_to(known)
    _write_marker(artifact, {"readme.txt": sha256_file(readme)}, CONFIG_SHA)
    if entry == "resume":
        _assert_resume_rejected(artifact, CONFIG_SHA)
    else:
        _assert_promote_rejected(artifact, known)


@pytest.mark.parametrize(
    ("label", "hashes"),
    [
        ("empty", None),
        ("absolute", "absolute"),
        ("parent", "parent"),
    ],
)
def test_resume_command_does_not_skip_quantization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], label: str, hashes: str | None
) -> None:
    cfg = load_experiment(RECIPE)
    config_sha = cfg.config_sha()
    run_root = tmp_path / "run"
    artifact = run_root / "quantize_nvfp4"
    _plant_weights(artifact)
    known = _known_file(tmp_path)
    if hashes is None:
        output_hashes: dict[str, str] = {}
        outside: Path | None = None
    elif hashes == "absolute":
        output_hashes = {str(known.resolve()): sha256_file(known)}
        outside = known
    elif hashes == "parent":
        # The stage directory is nested, so the sibling of run/ is tmp_path.
        output_hashes = {"../../known.bin": sha256_file(known)}
        outside = known
        assert (artifact / "../../known.bin").resolve() == known.resolve()
        assert (artifact / "../../known.bin").read_bytes() == KNOWN
    else:
        raise AssertionError(label)
    _write_marker(artifact, output_hashes, config_sha)
    before = _snapshot(artifact)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quantize_qwen38_nvfp4.py",
            "--config",
            str(RECIPE),
            "--run-root",
            str(run_root),
            "--resume",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        _nvfp4.main()
    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert "skipping" not in captured.out
    assert "verified" not in captured.out
    assert _snapshot(artifact) == before
    assert (artifact / "weights.bin").read_bytes() == ATTACKER
    assert not list(run_root.glob("quantize_nvfp4.partial.*"))
    if outside is not None:
        assert outside.read_bytes() == KNOWN
