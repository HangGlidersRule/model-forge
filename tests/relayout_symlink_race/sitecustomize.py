"""Win the symlink race at the moment of use, not at the earlier stat.

Loaded only when a test sets RELAYOUT_RACE_SOURCE and RELAYOUT_RACE_TARGET.
The first open or link of that path replaces a regular file with a symlink,
then performs the real call. A check that only lstats the path still sees a
regular file; a later copy that follows symlinks publishes the target.
"""
from __future__ import annotations

import builtins
import io
import os

_SOURCE = os.environ.get("RELAYOUT_RACE_SOURCE")
_TARGET = os.environ.get("RELAYOUT_RACE_TARGET")
_MARKER = os.environ.get("RELAYOUT_RACE_MARKER")

if _SOURCE and _TARGET:
    _SOURCE_NORM = os.path.normpath(_SOURCE)
    _REAL_OPEN = builtins.open
    _REAL_OS_OPEN = os.open
    _REAL_LINK = os.link
    _SWAPPED = False

    def _candidate(path: object, dir_fd: int | None) -> str:
        text = os.fsdecode(os.fspath(path))
        if os.path.isabs(text):
            return os.path.normpath(text)
        if dir_fd is not None:
            base = os.readlink(f"/proc/self/fd/{dir_fd}")
            return os.path.normpath(os.path.join(base, text))
        return os.path.normpath(os.path.abspath(text))

    def _matches(path: object, dir_fd: int | None = None) -> bool:
        try:
            return _candidate(path, dir_fd) == _SOURCE_NORM
        except (OSError, TypeError, ValueError):
            return False

    def _swap() -> None:
        global _SWAPPED
        if _SWAPPED or os.path.islink(_SOURCE_NORM):
            _SWAPPED = True
            return
        temporary = _SOURCE_NORM + ".race-tmp"
        os.symlink(_TARGET, temporary)
        os.replace(temporary, _SOURCE_NORM)
        _SWAPPED = True
        if _MARKER:
            marker_fd = _REAL_OS_OPEN(_MARKER, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            os.write(marker_fd, b"swapped\n")
            os.close(marker_fd)

    def _open(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
        if _matches(file):
            _swap()
        return _REAL_OPEN(file, mode, *args, **kwargs)

    def _os_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if _matches(path, dir_fd):
            _swap()
        if dir_fd is None:
            return _REAL_OS_OPEN(path, flags, mode)
        return _REAL_OS_OPEN(path, flags, mode, dir_fd=dir_fd)

    def _link(src: object, dst: object, *args: object, **kwargs: object) -> None:
        src_dir_fd = kwargs.get("src_dir_fd")
        dir_fd = src_dir_fd if isinstance(src_dir_fd, int) else None
        if _matches(src, dir_fd):
            _swap()
        _REAL_LINK(src, dst, *args, **kwargs)

    builtins.open = _open
    io.open = _open
    os.open = _os_open
    os.link = _link
