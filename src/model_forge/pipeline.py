"""Idempotent pipeline stage primitives with atomic directories and verified manifests."""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import shutil
import stat
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "1.0"
SUCCESS_MARKER = "_SUCCESS.json"

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
    blobs_fd = os.open(blobs, _DIR_FLAGS) if blobs is not None else None
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
    return blobs if blobs.is_dir() else None


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
