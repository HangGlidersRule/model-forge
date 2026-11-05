"""Child-process hook for the shared calibration-cache tests.

Loaded as sitecustomize (see tests/test_msmarco_shared_calib_cache.py). It does
not change production code. Two modes, selected by CALIB_HOOK:

- after_stat: after the cache path is statted, replace that directory entry.
  This is the gap between Path.exists and the later parquet read.
- after_read: after a Python read of the cache path returns the bytes it
  observed, replace that directory entry. A later open of the same path then
  sees the replacement. Parsing the bytes just read does not.
"""
from __future__ import annotations

import builtins
import io
import os
import pathlib

_HOOK = os.environ.get("CALIB_HOOK")
_CACHE = os.environ.get("CALIB_CACHE_PATH")
_SWAP_SRC = os.environ.get("CALIB_SWAP_SRC")
_did_swap = False


def _is_cache(path: object) -> bool:
    if not _CACHE:
        return False
    try:
        return os.fspath(path) == _CACHE  # type: ignore[arg-type]
    except TypeError:
        return False


def _swap_once() -> None:
    global _did_swap
    if _did_swap or not _CACHE or not _SWAP_SRC:
        return
    _did_swap = True
    temporary = _CACHE + ".hook-tmp"
    source = os.open(_SWAP_SRC, os.O_RDONLY)
    try:
        destination = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            while True:
                block = os.read(source, 1024 * 1024)
                if not block:
                    break
                offset = 0
                while offset < len(block):
                    wrote = os.write(destination, block[offset:])
                    offset += wrote
        finally:
            os.close(destination)
    finally:
        os.close(source)
    os.replace(temporary, _CACHE)


def _install_after_stat() -> None:
    real_stat = os.stat

    def stat_hook(path: object, *args: object, **kwargs: object) -> os.stat_result:
        result = real_stat(path, *args, **kwargs)  # type: ignore[arg-type]
        if _is_cache(path):
            _swap_once()
        return result

    os.stat = stat_hook  # type: ignore[assignment]


class _TrackedBinary:
    """File object that swaps the cache path once its first full read returns."""

    def __init__(self, raw: io.BufferedIOBase) -> None:
        self._raw = raw
        self._armed = True

    def read(self, size: int = -1) -> bytes:
        data = self._raw.read(size)
        if self._armed and (size is None or size < 0 or data == b"" or len(data) < size):
            self._finish()
        return data

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            self._finish()

    def _finish(self) -> None:
        if self._armed:
            self._armed = False
            _swap_once()

    def __getattr__(self, name: str) -> object:
        return getattr(self._raw, name)

    def __enter__(self) -> _TrackedBinary:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.close()
        return False

    def __iter__(self) -> _TrackedBinary:
        return self

    def __next__(self) -> bytes:
        line = self._raw.readline()
        if not line:
            self._finish()
            raise StopIteration
        return line


def _install_after_read() -> None:
    real_open = io.open
    real_read_bytes = pathlib.Path.read_bytes

    def open_hook(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
        handle = real_open(file, mode, *args, **kwargs)  # type: ignore[arg-type]
        writing = any(flag in mode for flag in ("w", "a", "x", "+"))
        if "b" in mode and "r" in mode and not writing and _is_cache(file):
            return _TrackedBinary(handle)
        return handle

    def read_bytes(self: pathlib.Path) -> bytes:
        data = real_read_bytes(self)
        if _is_cache(self):
            _swap_once()
        return data

    io.open = open_hook  # type: ignore[assignment]
    builtins.open = open_hook  # type: ignore[assignment]
    pathlib.Path.read_bytes = read_bytes  # type: ignore[method-assign]


if _HOOK == "after_stat":
    _install_after_stat()
elif _HOOK == "after_read":
    _install_after_read()
