"""Copy a checkpoint's non-weight files into an output tree without following links out of the checkpoint."""

from __future__ import annotations

import errno
import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path

_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC


class UnsafeCheckpointFile(ValueError):
    """A checkpoint path is a link, a special file, or changed while it was being copied."""


def copy_checkpoint_assets(source: Path, dest: Path, *, skip: Callable[[str], bool]) -> list[str]:
    """Copy the top-level entries of ``source`` that ``skip`` does not reject into ``dest``, directories recursively.

    Only regular files and directories are copied; any other entry, and any link, is refused. The one exception is a
    Hugging Face cache snapshot (``<repo>/snapshots/<revision>``), whose files are links into the repository's own
    ``blobs/`` directory: such a link is read from that directory, and a link anywhere else is refused.

    Every directory is held open for the walk and every file is opened once without following a link; the opened
    file must be the inode its directory lists under that name. Nothing is written until every file has been opened
    that way. Returns the copied file paths relative to ``source``.
    """
    blobs = _snapshot_blobs(source)
    root_fd = os.open(source, _DIR_FLAGS)
    blobs_fd = os.open(blobs, _DIR_FLAGS) if blobs is not None else None
    dirs: list[Path] = []
    files: list[tuple[Path, int]] = []
    try:
        _open_dir(source, root_fd, Path(), skip, blobs, blobs_fd, dirs, files)
        for rel in dirs:
            (dest / rel).mkdir(exist_ok=True)
        for rel, src_fd in files:
            with open(src_fd, "rb", closefd=False) as fin, open(
                os.open(dest / rel, _WRITE_FLAGS, 0o644), "wb"
            ) as fout:
                shutil.copyfileobj(fin, fout, 1 << 20)
    finally:
        for _, src_fd in files:
            os.close(src_fd)
        os.close(root_fd)
        if blobs_fd is not None:
            os.close(blobs_fd)
    return [rel.as_posix() for rel, _ in files]


def _snapshot_blobs(source: Path) -> Path | None:
    absolute = Path(os.path.abspath(source))
    if absolute.parent.name != "snapshots":
        return None
    blobs = absolute.parent.parent / "blobs"
    return blobs if blobs.is_dir() else None


def _open_dir(
    root: Path,
    dir_fd: int,
    rel: Path,
    skip: Callable[[str], bool] | None,
    blobs: Path | None,
    blobs_fd: int | None,
    dirs: list[Path],
    files: list[tuple[Path, int]],
) -> None:
    for name in sorted(os.listdir(dir_fd)):
        if skip is not None and skip(name):
            continue
        path = root / rel / name
        mode = os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode
        if stat.S_ISDIR(mode):
            try:
                sub_fd = os.open(name, _DIR_FLAGS | os.O_NOFOLLOW, dir_fd=dir_fd)
            except OSError as e:
                raise UnsafeCheckpointFile(f"{path}: {e.strerror}") from e
            try:
                dirs.append(rel / name)
                _open_dir(root, sub_fd, rel / name, None, blobs, blobs_fd, dirs, files)
            finally:
                os.close(sub_fd)
        elif stat.S_ISLNK(mode):
            files.append((rel / name, _open_snapshot_blob(path, dir_fd, name, blobs, blobs_fd)))
        else:
            files.append((rel / name, _open_entry(path, dir_fd, name)))


def _open_entry(path: Path, dir_fd: int, name: str) -> int:
    try:
        fd = os.open(path, _READ_FLAGS)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise UnsafeCheckpointFile(f"{path} is a link; checkpoint files must be regular files") from e
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise UnsafeCheckpointFile(f"{path} is not a regular file")
        if not os.path.samestat(st, os.stat(name, dir_fd=dir_fd, follow_symlinks=False)):
            raise UnsafeCheckpointFile(f"{path} changed while it was being copied")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _open_snapshot_blob(path: Path, dir_fd: int, name: str, blobs: Path | None, blobs_fd: int | None) -> int:
    if blobs is None or blobs_fd is None:
        raise UnsafeCheckpointFile(f"{path} is a link; checkpoint files must be regular files")
    target = Path(os.path.normpath(os.path.join(os.path.abspath(path.parent), os.readlink(name, dir_fd=dir_fd))))
    if target.parent != blobs:
        raise UnsafeCheckpointFile(f"{path} links outside the snapshot's blobs/ directory")
    try:
        fd = os.open(target.name, _READ_FLAGS, dir_fd=blobs_fd)
    except OSError as e:
        raise UnsafeCheckpointFile(f"{path}: blob {target.name}: {e.strerror}") from e
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise UnsafeCheckpointFile(f"{path}: blob {target.name} is not a regular file")
    return fd
