"""Run a checkpoint script after arming a shard-symlink retarget.

The scripts under test join a shard name to a directory and then open that
path. A fix that resolves the path and remembers the result is unaffected: the
resolved path is not a symlink, so this launcher leaves it alone. A fix that
checks the symlink and later opens the original path again sees the retargeted
file, because the swap happens inside the safetensors call the script makes.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import safetensors
import safetensors.torch as safetensors_torch

_TARGET = os.environ["SHARD_TOCTOU_TARGET"]
_REAL_OPEN = safetensors.safe_open
_REAL_SAVE = safetensors_torch.save_file
_REAL_LOAD = safetensors_torch.load_file


def _retarget(filename: str | os.PathLike[str]) -> None:
    path = Path(filename)
    if path.is_symlink():
        path.unlink()
        path.symlink_to(_TARGET)


def _safe_open(filename: str | os.PathLike[str], *args: object, **kwargs: object) -> object:
    _retarget(filename)
    return _REAL_OPEN(filename, *args, **kwargs)


def _save_file(
    tensors: object,
    filename: str | os.PathLike[str],
    *args: object,
    **kwargs: object,
) -> None:
    _retarget(filename)
    _REAL_SAVE(tensors, filename, *args, **kwargs)


def _load_file(filename: str | os.PathLike[str], *args: object, **kwargs: object) -> object:
    _retarget(filename)
    return _REAL_LOAD(filename, *args, **kwargs)


safetensors.safe_open = _safe_open
safetensors_torch.safe_open = _safe_open
safetensors_torch.save_file = _save_file
safetensors_torch.load_file = _load_file

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
