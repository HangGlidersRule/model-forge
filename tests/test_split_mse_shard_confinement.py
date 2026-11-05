"""Shard names from an export index must stay inside the repack output directory.

``split_mse.py`` and ``make_split.py`` pass each ``weight_map`` shard string to
``save_file`` as ``out / shard``. These tests fail while that write can create or
replace a file outside the output directory. They pass once a shard name that is
not a single relative path component, or whose resolved destination leaves the
output directory, is refused before any file outside that directory is created,
deleted, or modified.

The same weight_map join is not exercised here for
``models/jev-27b/scripts/merge_lora.py`` or
``scripts/qwen3_8/apply_qwen38_abliteration.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "models" / "nemotron-3.5-lightning-r1" / "eval"
SPLIT_MSE = EVAL / "split_mse.py"
MAKE_SPLIT = EVAL / "make_split.py"
EXPERT_SCALE_COUNT = 5934
WEIGHT_NAME = "model.embed.weight"
BENIGN_SHARD = "model-00001-of-00003.safetensors"

SCRIPTS = {
    "split_mse": SPLIT_MSE,
    "make_split": MAKE_SPLIT,
}


def _scale_name(script: Path, index: int) -> str:
    if script == SPLIT_MSE:
        return f"backbone.layers.0.mixer.experts.{index}.up_proj.input_scale"
    if script == MAKE_SPLIT:
        return f"block.{index}.input_scale"
    raise AssertionError(script)


def _scale_count(script: Path) -> int:
    if script == SPLIT_MSE:
        return EXPERT_SCALE_COUNT
    if script == MAKE_SPLIT:
        return 1
    raise AssertionError(script)


def _tensors(script: Path) -> dict[str, torch.Tensor]:
    tensors = {WEIGHT_NAME: torch.zeros(2, dtype=torch.float32)}
    for index in range(_scale_count(script)):
        tensors[_scale_name(script, index)] = torch.ones(1, dtype=torch.float32)
    return tensors


def _snapshot(root: Path) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            found[path.relative_to(root).as_posix()] = path.read_bytes()
    return found


def _outside_output(before: dict[str, bytes], after: dict[str, bytes], out_rel: str) -> list[str]:
    prefix = out_rel + "/"

    def inside(rel: str) -> bool:
        return rel == out_rel or rel.startswith(prefix)

    problems: list[str] = []
    for rel, data in after.items():
        if inside(rel):
            continue
        if before.get(rel) != data:
            problems.append(f"wrote {rel}")
    for rel in before:
        if rel not in after and not inside(rel):
            problems.append(f"deleted {rel}")
    return problems


def _run(script: Path, export: Path, out: Path, gpath: Path) -> subprocess.CompletedProcess[str]:
    if script == SPLIT_MSE:
        argv = [sys.executable, str(script), str(export), str(gpath), str(out)]
    elif script == MAKE_SPLIT:
        argv = [sys.executable, str(script), str(export), str(out)]
    else:
        raise AssertionError(script)
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def _detail(proc: subprocess.CompletedProcess[str], problems: list[str]) -> str:
    return (
        f"rc={proc.returncode} outside={problems}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )


def _prepare(
    tmp_path: Path,
    script: Path,
    shard: str,
    planted: Path,
    export: Path,
    out: Path,
) -> tuple[Path, dict[str, bytes], str]:
    root = tmp_path / "tree"
    tensors = _tensors(script)
    planted.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, planted, metadata={"format": "pt"})
    export.mkdir(parents=True, exist_ok=True)
    index = {
        "metadata": {"total_size": 1},
        "weight_map": {name: shard for name in tensors},
    }
    (export / "model.safetensors.index.json").write_text(json.dumps(index) + "\n")
    (root / "releases").mkdir(parents=True, exist_ok=True)
    (root / "releases" / "sentinel.txt").write_text("keep\n")
    gpath = root / "variants-g.json"
    gpath.write_text(json.dumps({"mse": {"L0.up": 2688.0}}) + "\n")
    return gpath, _snapshot(root), out.relative_to(root).as_posix()


def _anchor(root: Path, top: str, leaf: str, parents: int) -> Path:
    return root.joinpath(top, *["nest"] * (parents - 1), leaf)


@pytest.mark.parametrize("script_name", ["split_mse", "make_split"])
@pytest.mark.parametrize("parents", [1, 2])
def test_parent_directory_shard_name_cannot_escape_output(
    tmp_path: Path, script_name: str, parents: int
) -> None:
    script = SCRIPTS[script_name]
    root = tmp_path / "tree"
    export = _anchor(root, "incoming", "export", parents)
    out = _anchor(root, "releases", "out", parents)
    planted = root / "incoming" / "stolen.safetensors"
    shard = ("../" * parents) + "stolen.safetensors"
    gpath, before, out_rel = _prepare(tmp_path, script, shard, planted, export, out)

    proc = _run(script, export, out, gpath)
    problems = _outside_output(before, _snapshot(root), out_rel)

    assert proc.returncode != 0 and not problems, _detail(proc, problems)


@pytest.mark.parametrize("script_name", ["split_mse", "make_split"])
def test_absolute_shard_name_cannot_overwrite_file_outside_output(
    tmp_path: Path, script_name: str
) -> None:
    script = SCRIPTS[script_name]
    root = tmp_path / "tree"
    export = root / "incoming" / "export"
    out = root / "releases" / "out"
    planted = root / "planted.safetensors"
    gpath, before, out_rel = _prepare(tmp_path, script, str(planted), planted, export, out)

    proc = _run(script, export, out, gpath)
    problems = _outside_output(before, _snapshot(root), out_rel)

    assert proc.returncode != 0 and not problems, _detail(proc, problems)


@pytest.mark.parametrize("script_name", ["split_mse", "make_split"])
def test_single_component_shard_is_written_inside_output(
    tmp_path: Path, script_name: str
) -> None:
    script = SCRIPTS[script_name]
    root = tmp_path / "tree"
    export = root / "incoming" / "export"
    out = root / "releases" / "out"
    planted = export / BENIGN_SHARD
    gpath, before, out_rel = _prepare(tmp_path, script, BENIGN_SHARD, planted, export, out)

    proc = _run(script, export, out, gpath)
    problems = _outside_output(before, _snapshot(root), out_rel)

    assert proc.returncode == 0, _detail(proc, problems)
    assert (out / BENIGN_SHARD).is_file()
    assert not problems, _detail(proc, problems)
