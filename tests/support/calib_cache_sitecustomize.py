"""Child-process hook for the shared calibration-cache tests.

Loaded as sitecustomize (see tests/test_msmarco_shared_calib_cache.py). It does
not change production code. Two modes, selected by CALIB_HOOK:

- after_stat: after the cache path is statted, replace that directory entry.
  This is the gap between Path.exists and the later parquet read.
- after_read: after a Python read of the cache path returns the bytes it
  observed, replace that directory entry. A later open of the same path then
  sees the replacement. Parsing the bytes just read does not.
- after_open: once os.open of the cache path for reading has returned a
  descriptor, replace that directory entry. Bytes read through that descriptor
  are the original file; anything that opens the path again reads the
  replacement. (A read through os.fdopen of a descriptor is not seen by
  after_read, which matches on the path.)

Independently of CALIB_HOOK:

- CALIB_OFFLINE_URL set: urllib never reaches the network. That one URL is
  served from the local file CALIB_OFFLINE_FILE when it is set; every other
  request raises URLError.
- CALIB_FOREIGN_PATH set: os.stat and os.lstat report that path as owned by
  CALIB_FOREIGN_UID, and os.unlink of it raises PermissionError, as for another
  user's file in a sticky directory. ``foreign_owner_hooks`` gives the test
  process the same view.
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


def _install_after_open() -> None:
    real_open = os.open

    def open_hook(path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if dir_fd is None:
            fd = real_open(path, flags, mode)  # type: ignore[arg-type]
        else:
            fd = real_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]
        if dir_fd is None and _is_cache(path) and not flags & (os.O_WRONLY | os.O_RDWR):
            _swap_once()
        return fd

    os.open = open_hook  # type: ignore[assignment]


def _install_offline(url: str, served: str | None) -> None:
    import urllib.error
    import urllib.request

    real_open = os.open

    def urlopen(request: object, *args: object, **kwargs: object) -> object:
        target = request if isinstance(request, str) else getattr(request, "full_url", request)
        if served and target == url:
            return os.fdopen(real_open(served, os.O_RDONLY), "rb")
        raise urllib.error.URLError(f"network disabled in tests: {target}")

    def urlretrieve(request: object, *args: object, **kwargs: object) -> object:
        raise urllib.error.URLError(f"network disabled in tests: {request}")

    urllib.request.urlopen = urlopen  # type: ignore[assignment]
    urllib.request.urlretrieve = urlretrieve  # type: ignore[assignment]


class _ForeignStat:
    """A stat result whose st_uid is another user's."""

    def __init__(self, result: os.stat_result, uid: int) -> None:
        self._result = result
        self.st_uid = uid

    def __getattr__(self, name: str) -> object:
        return getattr(self._result, name)


def foreign_owner_hooks(path: str, uid: int) -> dict[str, object]:
    """Replacements for ``os.stat``, ``os.lstat`` and ``os.unlink`` that treat ``path`` as ``uid``'s."""
    real_stat = os.stat
    real_lstat = os.lstat
    real_unlink = os.unlink

    def is_target(target: object, kwargs: dict[str, object]) -> bool:
        if kwargs.get("dir_fd") is not None:
            return False
        try:
            return os.fspath(target) == path  # type: ignore[arg-type]
        except TypeError:
            return False

    def stat_hook(target: object, *args: object, **kwargs: object) -> object:
        result = real_stat(target, *args, **kwargs)
        return _ForeignStat(result, uid) if is_target(target, kwargs) else result

    def lstat_hook(target: object, *args: object, **kwargs: object) -> object:
        result = real_lstat(target, *args, **kwargs)
        return _ForeignStat(result, uid) if is_target(target, kwargs) else result

    def unlink_hook(target: object, *args: object, **kwargs: object) -> None:
        if is_target(target, kwargs):
            raise PermissionError(1, "Operation not permitted", os.fspath(target))  # type: ignore[arg-type]
        real_unlink(target, *args, **kwargs)

    return {"stat": stat_hook, "lstat": lstat_hook, "unlink": unlink_hook}


if _HOOK == "after_stat":
    _install_after_stat()
elif _HOOK == "after_read":
    _install_after_read()
elif _HOOK == "after_open":
    _install_after_open()
if os.environ.get("CALIB_OFFLINE_URL"):
    _install_offline(os.environ["CALIB_OFFLINE_URL"], os.environ.get("CALIB_OFFLINE_FILE"))
if os.environ.get("CALIB_FOREIGN_PATH"):
    _hooks = foreign_owner_hooks(
        os.environ["CALIB_FOREIGN_PATH"], int(os.environ.get("CALIB_FOREIGN_UID", "65534"))
    )
    for _name, _hook in _hooks.items():
        setattr(os, _name, _hook)
