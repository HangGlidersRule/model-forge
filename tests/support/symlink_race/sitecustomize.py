"""Test-only hook: replace one path with a symlink between lookup and read.

restore_config.py and the split scripts are started with PYTHONPATH pointed
here and RESTORE_SYMLINK_RACE=1. Two replacements are installed:

- After a stat or lstat of RESTORE_SYMLINK_RACE_PATH sees a non-symlink, that
  result is returned and the path is then replaced with a symlink to
  RESTORE_SYMLINK_RACE_TARGET.
- A later open by path, or an open that is not O_NOFOLLOW, performs the same
  replacement before the open if it has not happened yet.

Reading the original inode, or os.open with O_NOFOLLOW, keeps the regular file.
"""

from __future__ import annotations

import builtins
import io
import os
import stat

_DONE = False


def _watched_path(path: object) -> str | None:
    if os.environ.get("RESTORE_SYMLINK_RACE") != "1":
        return None
    watched = os.environ.get("RESTORE_SYMLINK_RACE_PATH")
    target = os.environ.get("RESTORE_SYMLINK_RACE_TARGET")
    if not watched or not target:
        return None
    try:
        current = os.path.abspath(os.fspath(path))
    except (TypeError, ValueError):
        return None
    if current != os.path.abspath(watched):
        return None
    return current


def _replace(current: str) -> None:
    global _DONE
    target = os.environ["RESTORE_SYMLINK_RACE_TARGET"]
    os.unlink(current)
    os.symlink(target, current)
    _DONE = True


def _swap_after_lookup(path: object, result: os.stat_result) -> None:
    if _DONE or stat.S_ISLNK(result.st_mode):
        return
    current = _watched_path(path)
    if current is None:
        return
    _replace(current)


def _swap_before_follow(path: object) -> None:
    if _DONE or isinstance(path, int):
        return
    current = _watched_path(path)
    if current is None:
        return
    _replace(current)


def _install() -> None:
    if os.environ.get("RESTORE_SYMLINK_RACE") != "1":
        return
    original_stat = os.stat
    original_lstat = os.lstat
    original_open = builtins.open
    original_os_open = os.open

    def stat_wrapper(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = original_stat(path, *args, **kwargs)
        _swap_after_lookup(path, result)
        return result

    def lstat_wrapper(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = original_lstat(path, *args, **kwargs)
        _swap_after_lookup(path, result)
        return result

    def open_wrapper(file, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
        _swap_before_follow(file)
        return original_open(file, mode, *args, **kwargs)

    def os_open_wrapper(path, flags, mode=0o777, *, dir_fd=None):  # type: ignore[no-untyped-def]
        if not (flags & os.O_NOFOLLOW):
            _swap_before_follow(path)
        if dir_fd is None:
            return original_os_open(path, flags, mode)
        return original_os_open(path, flags, mode, dir_fd=dir_fd)

    os.stat = stat_wrapper  # type: ignore[assignment]
    os.lstat = lstat_wrapper  # type: ignore[assignment]
    builtins.open = open_wrapper  # type: ignore[assignment]
    io.open = open_wrapper  # type: ignore[assignment]
    os.open = os_open_wrapper  # type: ignore[assignment]


_install()
