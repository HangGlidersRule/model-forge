"""Bound file access for checkpoint directories.

Shard names in ``model.safetensors.index.json`` and the entries of a checkpoint
directory come from whoever built the checkpoint, so they are data, not paths.
A source directory is opened once. Every file in it is looked up relative to
that descriptor without being opened for I/O, and the object that lookup found
must be a regular file whose kernel path lies under the allowed root. Only that
object is then read. Output files are written to a new file in the bound output
directory and renamed over the target name, so no existing object is ever
opened for writing.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import stat
import struct
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from types import TracebackType

INDEX_NAME = "model.safetensors.index.json"


class CheckpointPathError(ValueError):
    """A checkpoint names a file that is not a plain file inside its directory."""


def shard_name(name: object) -> str:
    """Return ``name`` if it is a single file name with no directory part."""
    if (
        not isinstance(name, str)
        or name in ("", ".", "..")
        or any(sep in name for sep in ("/", "\\", "\x00"))
    ):
        raise CheckpointPathError(f"shard name must be a single file name, got {name!r}")
    return name


def shard_names(weight_map: Mapping[str, object]) -> dict[str, str]:
    """Validate every shard name of an index ``weight_map``."""
    return {tensor: shard_name(shard) for tensor, shard in weight_map.items()}


def safetensors_metadata(data: bytes) -> dict[str, str] | None:
    """The ``__metadata__`` block of a serialized safetensors file."""
    (size,) = struct.unpack_from("<Q", data, 0)
    header = json.loads(data[8 : 8 + size])
    metadata = header.get("__metadata__")
    return dict(metadata) if metadata is not None else None


def _parts(relpath: str) -> tuple[str, ...]:
    path = PurePosixPath(relpath)
    if "\x00" in relpath or path.is_absolute() or any(p in ("", ".", "..") for p in path.parts):
        raise CheckpointPathError(f"path must be relative and plain, got {relpath!r}")
    if not path.parts:
        raise CheckpointPathError("empty path")
    return path.parts


def _fd_path(fd: int) -> str:
    return os.readlink(f"/proc/self/fd/{fd}")


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _require_bound_access() -> None:
    if not hasattr(os, "O_PATH") or not os.path.isdir("/proc/self/fd"):
        raise RuntimeError("bound checkpoint file access needs Linux O_PATH and /proc/self/fd")


class BoundFile:
    """A regular file inside a source root, held by an ``O_PATH`` descriptor."""

    def __init__(self, fd: int, label: str) -> None:
        self._fd = fd
        self.label = label

    def read_bytes(self) -> bytes:
        before = os.fstat(self._fd)
        fd = os.open(f"/proc/self/fd/{self._fd}", os.O_RDONLY | os.O_CLOEXEC)
        with os.fdopen(fd, "rb") as fh:
            after = os.fstat(fh.fileno())
            if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
                raise CheckpointPathError(f"{self.label} changed after it was checked")
            return fh.read()

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


class SourceDir:
    """A checkpoint directory whose files may only resolve under ``root``.

    ``root`` defaults to the directory itself. A Hugging Face cache snapshot
    links its files into the repository's ``blobs`` folder, so callers reading
    a snapshot pass the repository folder as the root.
    """

    def __init__(self, path: Path, root: Path | None = None) -> None:
        _require_bound_access()
        self.path = path
        self._fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self._bound: list[BoundFile] = []
        try:
            here = _fd_path(self._fd)
            if root is None:
                self.root = here
            else:
                root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
                try:
                    self.root = _fd_path(root_fd)
                finally:
                    os.close(root_fd)
            if not _inside(here, self.root):
                raise CheckpointPathError(f"{path} is not inside {self.root}")
        except BaseException:
            os.close(self._fd)
            raise

    def listdir(self) -> list[str]:
        return sorted(os.listdir(self._fd))

    def walk_files(self, relpath: str) -> Iterator[str]:
        """Relative paths of the non-directory entries under ``relpath``."""
        parts = _parts(relpath)
        top = os.open(
            os.path.join(*parts), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=self._fd,
        )
        try:
            for dirpath, dirnames, filenames, dirfd in os.fwalk(".", dir_fd=top):
                for name in dirnames:
                    if stat.S_ISLNK(os.stat(name, dir_fd=dirfd, follow_symlinks=False).st_mode):
                        raise CheckpointPathError(
                            f"symlinked directory {relpath}/{dirpath}/{name} in {self.path}"
                        )
                for name in sorted(filenames):
                    yield str(PurePosixPath(relpath, dirpath, name))
        finally:
            os.close(top)

    def is_dir(self, relpath: str) -> bool:
        try:
            st = os.stat(os.path.join(*_parts(relpath)), dir_fd=self._fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return stat.S_ISDIR(st.st_mode)

    def open(self, relpath: str) -> BoundFile:
        """Bind ``relpath``; raise unless it is a regular file under the root.

        A directory under the root raises ``IsADirectoryError`` so callers can skip it.
        """
        label = f"{relpath!r} in {self.path}"
        fd = os.open(os.path.join(*_parts(relpath)), os.O_PATH | os.O_CLOEXEC, dir_fd=self._fd)
        try:
            where = _fd_path(fd)
            if not _inside(where, self.root):
                raise CheckpointPathError(f"{label} resolves to {where}, outside {self.root}")
            mode = os.fstat(fd).st_mode
            if stat.S_ISDIR(mode):
                raise IsADirectoryError(f"{label} is a directory")
            if not stat.S_ISREG(mode):
                raise CheckpointPathError(f"{label} is not a regular file")
        except BaseException:
            os.close(fd)
            raise
        bound = BoundFile(fd, label)
        self._bound.append(bound)
        return bound

    def read_bytes(self, relpath: str) -> bytes:
        bound = self.open(relpath)
        try:
            return bound.read_bytes()
        finally:
            bound.close()

    def close(self) -> None:
        for bound in self._bound:
            bound.close()
        self._bound.clear()
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> SourceDir:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class OutputDir:
    """An output directory that is only written by creating new files in it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)

    def _parent(self, parts: tuple[str, ...]) -> int:
        fd = os.dup(self._fd)
        try:
            for part in parts[:-1]:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(part, dir_fd=fd)
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd
                )
                os.close(fd)
                fd = child
        except BaseException:
            os.close(fd)
            raise
        return fd

    def write_bytes(self, relpath: str, data: bytes) -> None:
        parts = _parts(relpath)
        name = parts[-1]
        parent = self._parent(parts)
        try:
            try:
                existing = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(existing.st_mode):
                    raise CheckpointPathError(
                        f"refusing to replace {relpath!r} in {self.path}: it is not a regular file"
                    )
            tmp = f".partial-{secrets.token_hex(8)}"
            fd = os.open(
                tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o666,
                dir_fd=parent,
            )
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, name, src_dir_fd=parent, dst_dir_fd=parent)
            except BaseException:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp, dir_fd=parent)
                raise
        finally:
            os.close(parent)

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> OutputDir:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
