"""Read a model directory without following symlinks out of it.

Every entry is opened once with ``O_NOFOLLOW`` and checked through the open descriptor, so what is copied is the
regular file that was checked, never whatever a path points at later. The one kind of link accepted is a Hugging Face
cache snapshot link (``<repo>/snapshots/<revision>/<file>`` -> ``../../blobs/<name>``): the link text is checked to
name a blob directly and that blob is opened in ``<repo>/blobs``; the link itself is never followed.
"""
from __future__ import annotations

import contextlib
import errno
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path

_READ_NOFOLLOW = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


class UnsafeSourceError(RuntimeError):
    """A source entry is a symlink (other than a cache blob link) or neither a regular file nor a directory."""


def copy_open_file(fd: int, dst: Path) -> None:
    """Copy an open regular file to ``dst`` with its mode and times (as ``shutil.copy2``).

    The bytes go to a new file beside ``dst`` that then replaces it, so a link already at ``dst`` is replaced rather than
    written through.
    """
    st = os.fstat(fd)
    tmp_fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.")
    try:
        with open(tmp_fd, "wb") as w, open(fd, "rb", closefd=False) as r:
            r.seek(0)
            shutil.copyfileobj(r, w)
            w.flush()
            os.fchmod(w.fileno(), stat.S_IMODE(st.st_mode))
            os.utime(w.fileno(), ns=(st.st_atime_ns, st.st_mtime_ns))
        os.replace(tmp, dst)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


class SourceTree:
    """A source model directory whose files are opened without following symlinks."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._blobs_fd: int | None = None
        self._blobs_abs = ""
        repo = root.parent.parent
        if root.parent.name == "snapshots" and repo.name.startswith("models--"):
            with contextlib.suppress(FileNotFoundError, NotADirectoryError):
                self._blobs_fd = os.open(
                    repo / "blobs", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
                )
                self._blobs_abs = os.path.abspath(repo / "blobs")

    def __enter__(self) -> SourceTree:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._blobs_fd is not None:
            os.close(self._blobs_fd)
            self._blobs_fd = None

    def open_file(self, name: str) -> int:
        """Open the top-level regular file ``name``; the caller closes the descriptor."""
        if name in ("", ".", "..") or Path(name).name != name:
            raise UnsafeSourceError(f"not a file name inside {self.root}: {name!r}")
        fd = self._open(name, None, 0)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise UnsafeSourceError(f"not a regular file: {self.root / name}")
        return fd

    def read_bytes(self, name: str) -> bytes:
        fd = self.open_file(name)
        with open(fd, "rb") as f:
            return f.read()

    def copy_entries(self, dst: Path, skip: Callable[[str], bool]) -> None:
        """Copy every top-level file and directory whose name ``skip`` does not reject into ``dst``."""
        with os.scandir(self.root) as it:
            names = sorted(e.name for e in it)
        for name in names:
            if skip(name):
                continue
            fd = self._open(name, None, 0)
            try:
                self._copy(fd, dst / name, 0, self.root / name)
            finally:
                os.close(fd)

    def _open(self, name: str, dir_fd: int | None, depth: int) -> int:
        path: str | Path = name if dir_fd is not None else self.root / name
        try:
            return os.open(path, _READ_NOFOLLOW, dir_fd=dir_fd)
        except OSError as e:
            if e.errno != errno.ELOOP:
                raise
        target = os.readlink(path, dir_fd=dir_fd)
        blob = self._blob_name(target, depth)
        if blob is None or self._blobs_fd is None:
            raise UnsafeSourceError(f"refusing symlink in {self.root}: {name} -> {target}")
        try:
            fd = os.open(blob, _READ_NOFOLLOW, dir_fd=self._blobs_fd)
        except OSError as e:
            if e.errno == errno.ELOOP:
                raise UnsafeSourceError(f"cache blob is a symlink: {blob}") from None
            raise
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise UnsafeSourceError(f"cache blob is not a regular file: {blob}")
        return fd

    def _blob_name(self, target: str, depth: int) -> str | None:
        """The blob a snapshot link names, from the link text alone (relative or absolute, as huggingface_hub writes)."""
        relative = "../" * (depth + 2) + "blobs/"
        if target.startswith(relative):
            name = target[len(relative):]
        elif self._blobs_abs and target.startswith(self._blobs_abs + "/"):
            name = target[len(self._blobs_abs) + 1:]
        else:
            return None
        if name in ("", ".", "..") or "/" in name:
            return None
        return name

    def _copy(self, fd: int, dst: Path, depth: int, shown: Path) -> None:
        mode = os.fstat(fd).st_mode
        if stat.S_ISREG(mode):
            copy_open_file(fd, dst)
            return
        if not stat.S_ISDIR(mode):
            raise UnsafeSourceError(f"neither a regular file nor a directory: {shown}")
        dst.mkdir()
        with os.scandir(fd) as it:
            names = sorted(e.name for e in it)
        for name in names:
            child = self._open(name, fd, depth + 1)
            try:
                self._copy(child, dst / name, depth + 1, shown / name)
            finally:
                os.close(child)
