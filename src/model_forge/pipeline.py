"""Idempotent pipeline stage primitives with atomic directories and verified manifests,
and bound file access for the checkpoint directories those stages read and write."""
from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import stat
import struct
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any

MANIFEST_SCHEMA_VERSION = "1.0"
SUCCESS_MARKER = "_SUCCESS.json"
INDEX_NAME = "model.safetensors.index.json"

_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            hashes[str(p.relative_to(root))] = sha256_file(p)
    return hashes


def config_hash(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(config).encode())


@dataclass(frozen=True)
class SuccessManifest:
    schema_version: str = MANIFEST_SCHEMA_VERSION
    stage: str = ""
    command: str = ""
    git_commit: str = ""
    source_revisions: dict[str, str] = field(default_factory=dict)
    config_sha: str = ""
    input_hashes: dict[str, str] = field(default_factory=dict)
    output_hashes: dict[str, str] = field(default_factory=dict)
    timestamps: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "SuccessManifest":
        data = json.loads(text)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RunLock:
    """File-based exclusive lock preventing concurrent stage mutation."""

    def __init__(self, lock_path: Path) -> None:
        self._path = lock_path
        self._fd: int | None = None

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self._path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(self._fd)
            self._fd = None
            raise ConcurrentRunError(f"Run lock held: {self._path}") from e
        os.write(self._fd, f"{os.getpid()}\n".encode())

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()


class ConcurrentRunError(RuntimeError):
    pass


class StageError(RuntimeError):
    pass


def validate_success_marker(
    stage_dir: Path, expected_config_sha: str
) -> SuccessManifest | None:
    """Return manifest if stage completed validly, else None."""
    marker = stage_dir / SUCCESS_MARKER
    if not marker.exists():
        return None
    try:
        manifest = SuccessManifest.from_json(marker.read_text())
    except (json.JSONDecodeError, TypeError):
        return None
    if manifest.config_sha != expected_config_sha:
        return None
    for rel_path, expected_hash in manifest.output_hashes.items():
        out_file = stage_dir / rel_path
        if not out_file.exists():
            return None
        if sha256_file(out_file) != expected_hash:
            return None
    return manifest


def quarantine_partial(stage_dir: Path) -> Path | None:
    """Move a stale/corrupt stage directory to a timestamped quarantine path."""
    if not stage_dir.exists():
        return None
    ts = time.strftime("%Y%m%dT%H%M%S")
    quarantine = stage_dir.parent / f"{stage_dir.name}.quarantine.{ts}.{os.getpid()}"
    shutil.move(str(stage_dir), str(quarantine))
    return quarantine


class StageContext:
    """Manages atomic partial directory and promotion for a single stage."""

    def __init__(self, run_root: Path, stage_name: str, config_sha: str) -> None:
        self.run_root = run_root
        self.stage_name = stage_name
        self.config_sha = config_sha
        self.stage_dir = run_root / stage_name
        self._partial: Path | None = None

    @property
    def partial_dir(self) -> Path:
        if self._partial is None:
            self._partial = self.run_root / f"{self.stage_name}.partial.{os.getpid()}"
            self._partial.mkdir(parents=True, exist_ok=True)
        return self._partial

    def should_skip(self) -> SuccessManifest | None:
        return validate_success_marker(self.stage_dir, self.config_sha)

    def promote(self, manifest: SuccessManifest) -> None:
        """Atomically promote partial dir to final stage dir after writing manifest."""
        if self._partial is None:
            raise StageError("No partial directory to promote")
        marker = self._partial / SUCCESS_MARKER
        marker.write_text(manifest.to_json())
        if self.stage_dir.exists():
            quarantine_partial(self.stage_dir)
        os.rename(str(self._partial), str(self.stage_dir))
        self._partial = None

    def cleanup_partial(self) -> None:
        if self._partial and self._partial.exists():
            shutil.rmtree(self._partial, ignore_errors=True)
            self._partial = None


# Checkpoint file access.
#
# Shard names in ``model.safetensors.index.json`` and the entries of a checkpoint
# directory come from whoever built the checkpoint, so they are data, not paths.
# A source directory is opened once. Every file in it is looked up relative to
# that descriptor without being opened for I/O, and the object that lookup found
# must be a regular file whose kernel path lies under the allowed root. Only that
# object is then read. Output files are written to a new file in the bound output
# directory and renamed over the target name, so no existing object is ever
# opened for writing.


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


# Linux only; `_require_bound_access` refuses to bind anything where it is missing.
_O_PATH: int = getattr(os, "O_PATH", 0)


def _require_bound_access() -> None:
    if not _O_PATH or not os.path.isdir("/proc/self/fd"):
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
        fd = os.open(os.path.join(*_parts(relpath)), _O_PATH | os.O_CLOEXEC, dir_fd=self._fd)
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


class UnsafeCheckpointFile(ValueError):
    """A checkpoint path is a link, a special file, or changed while it was being copied."""


def copy_checkpoint_assets(
    source: Path, dest: Path, *, skip: Callable[[str], bool]
) -> list[str]:
    """Copy the top-level entries of ``source`` that ``skip`` keeps into ``dest``, recursively.

    Only regular files and directories are copied; any other entry, and any link, is refused.
    The one exception is a Hugging Face cache snapshot (``<repo>/snapshots/<revision>``), whose
    files are links into the repository's own ``blobs/``: such a link is read from that
    directory, and a link anywhere else is refused.

    Every directory is held open for the walk and every file is opened once without following
    a link; the opened file must be the inode its directory lists under that name. Nothing is
    written until every file has been opened that way. Returns the copied paths relative to
    ``source``.
    """
    blobs = _snapshot_blobs(source)
    root_fd = os.open(source, _DIR_FLAGS)
    blobs_fd = os.open(blobs, _DIR_FLAGS | os.O_NOFOLLOW) if blobs is not None else None
    dirs: list[Path] = []
    files: list[tuple[Path, int]] = []
    try:
        _open_checkpoint_dir(source, root_fd, Path(), skip, blobs, blobs_fd, dirs, files)
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
    # A blobs/ that is itself a link would let snapshot links read from any directory.
    return blobs if blobs.is_dir() and not blobs.is_symlink() else None


def _open_checkpoint_dir(
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
                _open_checkpoint_dir(
                    root, sub_fd, rel / name, None, blobs, blobs_fd, dirs, files
                )
            finally:
                os.close(sub_fd)
        elif stat.S_ISLNK(mode):
            files.append((rel / name, _open_snapshot_blob(path, dir_fd, name, blobs, blobs_fd)))
        else:
            files.append((rel / name, _open_checkpoint_entry(path, dir_fd, name)))


def _open_checkpoint_entry(path: Path, dir_fd: int, name: str) -> int:
    try:
        fd = os.open(path, _READ_FLAGS)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise UnsafeCheckpointFile(
                f"{path} is a link; checkpoint files must be regular files"
            ) from e
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


def _open_snapshot_blob(
    path: Path, dir_fd: int, name: str, blobs: Path | None, blobs_fd: int | None
) -> int:
    if blobs is None or blobs_fd is None:
        raise UnsafeCheckpointFile(f"{path} is a link; checkpoint files must be regular files")
    link = os.readlink(name, dir_fd=dir_fd)
    target = Path(os.path.normpath(os.path.join(os.path.abspath(path.parent), link)))
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
