"""Idempotent pipeline stage primitives with atomic directories and verified manifests."""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "1.0"
SUCCESS_MARKER = "_SUCCESS.json"


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


# Read checkpoint files without following symlinks; write release files without writing through them.
#
# A checkpoint directory is opened once. Every member is opened relative to that descriptor, one path component at a time
# with O_NOFOLLOW, and must be a regular file according to fstat on the open descriptor. Callers then read the bound
# descriptor (``Member.path`` is ``/proc/self/fd/N`` for libraries that take a path), so a name swapped after the check
# cannot change what is read.
#
# The only symlinks accepted are Hugging Face cache pointers: when the directory is ``<repo>/snapshots/<revision>``, a
# member may be the relative link ``../../blobs/<hash>`` (one more ``../`` per subdirectory) that huggingface_hub writes.
# That blob is opened with O_NOFOLLOW from the repo's own ``blobs`` directory. Since the shared blob store, that name may
# itself be the relative link ``../../blobs/<prefix>/<xet hash>`` into the cache root; each of those components is opened
# with O_NOFOLLOW as well. Any other symlink is refused.
#
# Outputs are written to an O_EXCL temporary in the destination directory and renamed over the name, so a symlink already
# at the destination is replaced, never written through. Linux only (``/proc/self/fd``).
#
# This module stays stdlib only: model scripts import it from the checkout inside images without this package's
# dependencies.

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY | os.O_CLOEXEC
_BLOB = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
# ``<repo>/blobs/<etag>`` -> ``../../blobs/<prefix>/<xet hash>``, prefix = hash[:2]. huggingface_hub writes this for the
# cache-root shared store and keeps the snapshot pointer pointed at the repo blob.
_SHARED_BLOB = re.compile(r"\.\./\.\./blobs/([0-9a-f]{2})/(\1[0-9a-f]{62})")
_CHUNK = 1 << 30


class UnsafeFileError(Exception):
    """A checkpoint member is a symlink, a special file, or escapes its directory."""


def _parts(rel: str) -> list[str]:
    parts = rel.split("/")
    if not rel or rel.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise UnsafeFileError(f"not a plain relative path inside the checkpoint: {rel!r}")
    return parts


@dataclass(frozen=True)
class Member:
    """A regular file opened without following symlinks. ``fd`` stays open for the life of the process."""

    label: str
    fd: int
    st: os.stat_result

    @property
    def path(self) -> str:
        return f"/proc/self/fd/{self.fd}"

    def read_bytes(self) -> bytes:
        chunks: list[bytes] = []
        offset = 0
        while True:
            block = os.pread(self.fd, 1 << 20, offset)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
            offset += len(block)

    def read_text(self) -> str:
        return self.read_bytes().decode("utf-8")


def _open_blobs(root_fd: int) -> int | None:
    """The ``blobs`` directory of the Hugging Face cache repo whose ``snapshots`` holds ``root_fd``, else None."""
    try:
        parent = os.open("..", _DIR_FLAGS, dir_fd=root_fd)
    except OSError:
        return None
    try:
        repo = os.open("..", _DIR_FLAGS, dir_fd=parent)
    except OSError:
        os.close(parent)
        return None
    try:
        snapshots = os.open("snapshots", _DIR_FLAGS, dir_fd=repo)
        try:
            if not os.path.samestat(os.fstat(snapshots), os.fstat(parent)):
                return None
        finally:
            os.close(snapshots)
        return os.open("blobs", _DIR_FLAGS, dir_fd=repo)
    except OSError:
        return None
    finally:
        os.close(repo)
        os.close(parent)


def _open_shared_blob(blobs_fd: int, name: str) -> int:
    """Open the regular file a repo ``blobs/<etag>`` symlink names in the cache-root shared store."""
    target = os.readlink(name, dir_fd=blobs_fd)
    found = _SHARED_BLOB.fullmatch(target)
    if found is None:
        raise OSError(errno.ELOOP, "not a Hugging Face shared-blob pointer", name)
    prefix, blob = found.group(1), found.group(2)
    opened: list[int] = []
    try:
        dir_fd = blobs_fd
        for step in ("..", "..", "blobs", prefix):
            dir_fd = os.open(step, _DIR_FLAGS, dir_fd=dir_fd)
            opened.append(dir_fd)
        return os.open(blob, _FILE_FLAGS, dir_fd=dir_fd)
    finally:
        for fd in opened:
            os.close(fd)


class CheckpointTree:
    """A checkpoint directory whose members are opened relative to one directory descriptor."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.label = os.fspath(path)
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self._blobs = _open_blobs(self.fd)

    def _refuse(self, rel: str, why: str) -> UnsafeFileError:
        return UnsafeFileError(f"refusing {os.path.join(self.label, rel)}: {why}")

    @contextmanager
    def _dir(self, parts: list[str]) -> Iterator[int]:
        fd = self.fd
        try:
            for i, name in enumerate(parts):
                try:
                    nxt = os.open(name, _DIR_FLAGS, dir_fd=fd)
                except OSError as e:
                    if e.errno in (errno.ELOOP, errno.ENOTDIR):
                        raise self._refuse("/".join(parts[: i + 1]), "not a directory (symlink?)") from e
                    raise
                if fd != self.fd:
                    os.close(fd)
                fd = nxt
            yield fd
        finally:
            if fd != self.fd:
                os.close(fd)

    def _pointer(self, dir_fd: int, name: str, rel: str, depth: int) -> int:
        if self._blobs is None:
            raise self._refuse(rel, "symlink")
        try:
            target = os.readlink(name, dir_fd=dir_fd)
        except OSError as e:
            raise self._refuse(rel, "changed while it was being opened") from e
        prefix = "../" * (depth + 1) + "blobs/"
        blob = target[len(prefix):]
        if not target.startswith(prefix) or not _BLOB.fullmatch(blob):
            raise self._refuse(rel, "symlink that is not a Hugging Face cache pointer")
        try:
            return os.open(blob, _FILE_FLAGS, dir_fd=self._blobs)
        except OSError as e:
            if e.errno != errno.ELOOP:
                raise self._refuse(rel, f"cache blob {blob} cannot be opened without following links") from e
            try:
                return _open_shared_blob(self._blobs, blob)
            except OSError as shared_error:
                raise self._refuse(rel, f"cache blob {blob} cannot be opened without following links") from shared_error

    def open_optional(self, rel: str) -> Member | None:
        """Open ``rel`` as a regular file; None if nothing exists at that name."""
        parts = _parts(rel)
        with self._dir(parts[:-1]) as dir_fd:
            try:
                fd = os.open(parts[-1], _FILE_FLAGS, dir_fd=dir_fd)
            except FileNotFoundError:
                return None
            except OSError as e:
                if e.errno != errno.ELOOP:
                    raise
                fd = -1
            if fd < 0:
                fd = self._pointer(dir_fd, parts[-1], rel, len(parts))
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            os.close(fd)
            raise self._refuse(rel, "not a regular file")
        return Member(os.path.join(self.label, rel), fd, st)

    def open(self, rel: str) -> Member:
        member = self.open_optional(rel)
        if member is None:
            raise FileNotFoundError(errno.ENOENT, "missing checkpoint file", os.path.join(self.label, rel))
        return member

    def listdir(self, rel: str = "") -> tuple[list[str], list[str]]:
        """Sorted (files, directories) directly under ``rel``. Symlinks are listed as files; ``open`` decides them."""
        files: list[str] = []
        dirs: list[str] = []
        with self._dir(_parts(rel) if rel else []) as dir_fd:
            with os.scandir(dir_fd) as entries:
                for entry in entries:
                    if entry.is_symlink() or entry.is_file(follow_symlinks=False):
                        files.append(entry.name)
                    elif entry.is_dir(follow_symlinks=False):
                        dirs.append(entry.name)
                    else:
                        raise self._refuse(os.path.join(rel, entry.name), "not a regular file or directory")
        return sorted(files), sorted(dirs)

    def walk(self, rel: str) -> tuple[list[str], list[str]]:
        """Every (file, directory) path under the directory ``rel``, relative to the tree root."""
        files, dirs = self.listdir(rel)
        all_files = [f"{rel}/{name}" for name in files]
        all_dirs: list[str] = []
        for name in dirs:
            sub = f"{rel}/{name}"
            all_dirs.append(sub)
            sub_files, sub_dirs = self.walk(sub)
            all_files += sub_files
            all_dirs += sub_dirs
        return all_files, all_dirs


class OutputDir:
    """A release directory. Every write lands on a fresh inode renamed over the name."""

    def __init__(self, fd: int, label: str) -> None:
        self.fd = fd
        self.label = label

    @classmethod
    def create(cls, path: str | os.PathLike[str]) -> OutputDir:
        Path(path).mkdir(parents=True, exist_ok=True)
        return cls(os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC), os.fspath(path))

    def path(self, name: str) -> str:
        """Path of a top-level output name for a library that writes by path (safetensors renames into place)."""
        if len(_parts(name)) != 1:
            raise UnsafeFileError(f"output name must be a plain file name: {name!r}")
        return f"/proc/self/fd/{self.fd}/{name}"

    def mkdir(self, rel: str) -> None:
        with self._parent(rel) as (dir_fd, name):
            try:
                os.mkdir(name, dir_fd=dir_fd)
            except FileExistsError:
                os.close(os.open(name, _DIR_FLAGS, dir_fd=dir_fd))

    @contextmanager
    def _parent(self, rel: str) -> Iterator[tuple[int, str]]:
        parts = _parts(rel)
        fd = self.fd
        try:
            for name in parts[:-1]:
                try:
                    os.mkdir(name, dir_fd=fd)
                except FileExistsError:
                    pass
                nxt = os.open(name, _DIR_FLAGS, dir_fd=fd)
                if fd != self.fd:
                    os.close(fd)
                fd = nxt
            yield fd, parts[-1]
        finally:
            if fd != self.fd:
                os.close(fd)

    @contextmanager
    def _replace(self, rel: str) -> Iterator[int]:
        with self._parent(rel) as (dir_fd, name):
            tmp = f".{name}.{secrets.token_hex(6)}.tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o666,
                         dir_fd=dir_fd)
            try:
                yield fd
                os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            except BaseException:
                try:
                    os.unlink(tmp, dir_fd=dir_fd)
                except FileNotFoundError:
                    pass
                raise
            finally:
                os.close(fd)

    def write_bytes(self, rel: str, data: bytes) -> None:
        with self._replace(rel) as fd:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]

    def write_text(self, rel: str, text: str) -> None:
        self.write_bytes(rel, text.encode("utf-8"))

    def copy(self, rel: str, src: Member, *, keep_metadata: bool = False) -> None:
        """Copy the bytes of ``src``; with ``keep_metadata``, also its mode and times (as shutil.copy2)."""
        with self._replace(rel) as fd:
            offset = 0
            while True:
                sent = os.sendfile(fd, src.fd, offset, _CHUNK)
                if sent == 0:
                    break
                offset += sent
            if keep_metadata:
                os.chmod(fd, stat.S_IMODE(src.st.st_mode))
                os.utime(fd, ns=(src.st.st_atime_ns, src.st.st_mtime_ns))

    def link(self, rel: str, src: Member) -> None:
        """Hard-link the inode ``src`` has open (not whatever its name points to now) at ``rel``."""
        with self._parent(rel) as (dir_fd, name):
            tmp = f".{name}.{secrets.token_hex(6)}.tmp"
            os.link(src.path, tmp, dst_dir_fd=dir_fd, follow_symlinks=True)
            try:
                os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            except BaseException:
                os.unlink(tmp, dir_fd=dir_fd)
                raise
