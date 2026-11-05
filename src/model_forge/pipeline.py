"""Idempotent pipeline stage primitives with atomic directories and verified manifests."""
from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Callable
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


# Reading a source model without following symlinks out of it. Every entry is opened once with
# O_NOFOLLOW and checked through the open descriptor, so what is copied is the regular file that
# was checked, never whatever a path points at later. The one kind of link accepted is a Hugging
# Face cache snapshot link (<repo>/snapshots/<revision>/<file> -> ../../blobs/<name>): the link
# text is checked to name a blob directly and that blob is opened in <repo>/blobs; the link itself
# is never followed. A blob that is itself a symlink is accepted only when its text names a regular
# file in the cache-root shared store (<cache>/blobs/<prefix>/<hash>); that file is opened from the
# store, and neither link is followed.
_READ_NOFOLLOW = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_DIR_NOFOLLOW = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _open_dir_nofollow(path: Path) -> int | None:
    """Open a real directory, or None when it is missing or itself a symlink.

    ``O_NOFOLLOW`` on a symlink is ``ELOOP`` or ``ENOTDIR``. Both are a soft miss so a linked
    ``blobs`` directory becomes the usual source error instead of a raw ``OSError``.
    """
    try:
        return os.open(path, _DIR_NOFOLLOW)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as e:
        if e.errno == errno.ELOOP:
            return None
        raise


class UnsafeSourceError(RuntimeError):
    """A source entry is a symlink (other than a cache blob link), or neither a regular file nor
    a directory."""


def copy_open_file(fd: int, dst: Path) -> None:
    """Copy an open regular file to ``dst`` with its mode and times (as ``shutil.copy2``).

    The bytes go to a new file beside ``dst`` that then replaces it, so a link already at ``dst``
    is replaced rather than written through.
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
        self._shared_fd: int | None = None
        self._shared_abs = ""
        repo = root.parent.parent
        if root.parent.name == "snapshots" and repo.name.startswith("models--"):
            blobs = repo / "blobs"
            self._blobs_fd = _open_dir_nofollow(blobs)
            if self._blobs_fd is not None:
                self._blobs_abs = os.path.abspath(blobs)
            shared = repo.parent / "blobs"
            try:
                self._shared_fd = _open_dir_nofollow(shared)
            except BaseException:
                self.close()
                raise
            if self._shared_fd is not None:
                self._shared_abs = os.path.abspath(shared)

    def __enter__(self) -> SourceTree:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._blobs_fd is not None:
            os.close(self._blobs_fd)
            self._blobs_fd = None
        if self._shared_fd is not None:
            os.close(self._shared_fd)
            self._shared_fd = None

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
        """Copy every top-level file and directory not rejected by ``skip`` into ``dst``."""
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
            if e.errno != errno.ELOOP:
                raise
            shared = self._open_shared_blob(blob)
            if shared is None:
                raise UnsafeSourceError(f"cache blob is a symlink: {blob}") from None
            fd = shared
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise UnsafeSourceError(f"cache blob is not a regular file: {blob}")
        return fd

    def _open_shared_blob(self, blob: str) -> int | None:
        """Open the shared-store file a repo blob symlink names, or None to refuse it.

        The link text is checked and the file is opened from ``<cache>/blobs``; the symlink is
        not followed. A further symlink, in the prefix directory or the file, is refused.
        """
        blobs_fd = self._blobs_fd
        shared_fd = self._shared_fd
        if blobs_fd is None or shared_fd is None:
            return None
        target = os.readlink(blob, dir_fd=blobs_fd)
        parts = self._shared_blob_name(target)
        if parts is None:
            return None
        prefix, digest = parts
        try:
            prefix_fd = os.open(prefix, _DIR_NOFOLLOW, dir_fd=shared_fd)
        except OSError as e:
            # A symlink is ELOOP, or ENOTDIR when O_DIRECTORY is set. Either way it is not the
            # real prefix directory and must not be followed.
            if e.errno in (errno.ELOOP, errno.ENOTDIR):
                return None
            raise
        try:
            return os.open(digest, _READ_NOFOLLOW, dir_fd=prefix_fd)
        except OSError as e:
            if e.errno == errno.ELOOP:
                return None
            raise
        finally:
            os.close(prefix_fd)

    def _shared_blob_name(self, target: str) -> tuple[str, str] | None:
        """Prefix and hash a repo blob link names in the shared store, from the link text.

        huggingface_hub writes ``../../blobs/<prefix>/<hash>`` (64 lowercase hex digits, prefix
        the first two) or, as a fallback, that path in absolute form.
        """
        relative = "../../blobs/"
        if target.startswith(relative):
            rest = target[len(relative):]
        elif self._shared_abs and target.startswith(self._shared_abs + "/"):
            rest = target[len(self._shared_abs) + 1:]
        else:
            return None
        prefix, slash, digest = rest.partition("/")
        if slash != "/" or "/" in digest:
            return None
        if len(prefix) != 2 or len(digest) != 64 or not digest.startswith(prefix):
            return None
        if any(c not in "0123456789abcdef" for c in digest):
            return None
        return prefix, digest

    def _blob_name(self, target: str, depth: int) -> str | None:
        """The blob a snapshot link names, from the link text alone.

        huggingface_hub writes it relative (``../../blobs/<name>``) or, as a fallback, absolute.
        """
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
