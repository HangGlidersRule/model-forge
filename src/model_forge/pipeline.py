"""Idempotent pipeline stage primitives with atomic directories and verified manifests."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "1.0"
SUCCESS_MARKER = "_SUCCESS.json"

_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC

# (st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)
FileIdentity = tuple[int, int, int, int, int]


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


class ArtifactVerificationError(RuntimeError):
    """An artifact directory does not match the success marker that describes it."""


def _file_identity(st: os.stat_result) -> FileIdentity:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _dir_identity(st: os.stat_result) -> tuple[int, int]:
    return (st.st_dev, st.st_ino)


def open_artifact_dir(path: Path) -> int:
    """Open ``path`` as a directory descriptor, refusing a symlink at ``path`` itself."""
    return os.open(str(path), _DIR_OPEN_FLAGS)


def artifact_dir_identity(root_fd: int) -> tuple[int, int]:
    return _dir_identity(os.fstat(root_fd))


def path_dir_identity(path: Path) -> tuple[int, int] | None:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(st.st_mode):
        return None
    return _dir_identity(st)


def _sha256_fd(fd: int) -> str:
    h = hashlib.sha256()
    while chunk := os.read(fd, 1 << 20):
        h.update(chunk)
    return h.hexdigest()


def _scan_tree(
    dir_fd: int, prefix: str, identities: dict[str, FileIdentity], hashes: dict[str, str] | None
) -> None:
    for name in sorted(os.listdir(dir_fd)):
        rel = prefix + name
        st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        if stat.S_ISDIR(st.st_mode):
            child = os.open(name, _DIR_OPEN_FLAGS, dir_fd=dir_fd)
            try:
                if _dir_identity(os.fstat(child)) != _dir_identity(st):
                    raise ArtifactVerificationError(f"directory changed while scanning: {rel}")
                _scan_tree(child, rel + "/", identities, hashes)
            finally:
                os.close(child)
        elif stat.S_ISREG(st.st_mode):
            fd = os.open(name, _FILE_OPEN_FLAGS, dir_fd=dir_fd)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or _file_identity(before) != _file_identity(st):
                    raise ArtifactVerificationError(f"file changed while scanning: {rel}")
                if hashes is not None and rel != SUCCESS_MARKER:
                    hashes[rel] = _sha256_fd(fd)
                    if _file_identity(os.fstat(fd)) != _file_identity(before):
                        raise ArtifactVerificationError(f"file changed while hashing: {rel}")
                identities[rel] = _file_identity(before)
            finally:
                os.close(fd)
        else:
            kind = "symlink" if stat.S_ISLNK(st.st_mode) else "non-regular entry"
            raise ArtifactVerificationError(f"artifact contains a {kind}: {rel}")


def snapshot_artifact_tree(root_fd: int) -> dict[str, FileIdentity]:
    """Identities of every regular file under ``root_fd``, including the success marker.

    Raises ``ArtifactVerificationError`` on any symlink or non-regular entry.
    """
    identities: dict[str, FileIdentity] = {}
    try:
        _scan_tree(root_fd, "", identities, None)
    except OSError as e:
        raise ArtifactVerificationError(f"cannot scan artifact: {e}") from e
    return identities


def artifact_tree_unchanged(root_fd: int, expected: dict[str, FileIdentity]) -> bool:
    try:
        return snapshot_artifact_tree(root_fd) == expected
    except ArtifactVerificationError:
        return False


def read_success_marker(root_fd: int) -> SuccessManifest | None:
    """Parse the marker directly under ``root_fd``; ``None`` when there is none.

    Raises ``ArtifactVerificationError`` when the marker is a symlink, not a
    regular file, or not a well-formed manifest.
    """
    try:
        fd = os.open(SUCCESS_MARKER, _FILE_OPEN_FLAGS, dir_fd=root_fd)
    except FileNotFoundError:
        return None
    except OSError as e:
        raise ArtifactVerificationError(f"unreadable {SUCCESS_MARKER}: {e}") from e
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ArtifactVerificationError(f"{SUCCESS_MARKER} is not a regular file")
        chunks = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
    finally:
        os.close(fd)
    try:
        manifest = SuccessManifest.from_json(b"".join(chunks).decode("utf-8"))
    except (ValueError, TypeError, AttributeError) as e:
        raise ArtifactVerificationError(f"malformed {SUCCESS_MARKER}") from e
    hashes = manifest.output_hashes
    if not isinstance(hashes, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in hashes.items()
    ):
        raise ArtifactVerificationError(f"{SUCCESS_MARKER} output_hashes is not a str->str map")
    return manifest


def verify_artifact_tree(root_fd: int, manifest: SuccessManifest) -> dict[str, FileIdentity]:
    """Require ``manifest.output_hashes`` to describe exactly the files under ``root_fd``.

    Every regular file other than the root success marker must be listed with
    its SHA-256, and every key must name one of those files. Hashes are taken
    from descriptors opened during the scan, never from a path built out of a
    marker key, so absolute keys, parent segments, and symlinks cannot redirect
    a check outside the directory. Returns the identities of the files that
    were hashed (plus the marker, when present) so callers can detect changes
    before acting on the directory.
    """
    if not manifest.output_hashes:
        raise ArtifactVerificationError(f"{SUCCESS_MARKER} lists no output hashes")
    identities: dict[str, FileIdentity] = {}
    hashes: dict[str, str] = {}
    try:
        _scan_tree(root_fd, "", identities, hashes)
    except OSError as e:
        raise ArtifactVerificationError(f"cannot scan artifact: {e}") from e
    listed = set(manifest.output_hashes)
    present = set(hashes)
    if listed != present:
        unlisted = sorted(present - listed)
        missing = sorted(listed - present)
        raise ArtifactVerificationError(
            f"marker does not match artifact files (unlisted={unlisted[:5]}, "
            f"missing={missing[:5]})"
        )
    for rel, digest in hashes.items():
        if manifest.output_hashes[rel] != digest:
            raise ArtifactVerificationError(f"hash mismatch: {rel}")
    return identities


def validate_success_marker(
    stage_dir: Path, expected_config_sha: str
) -> SuccessManifest | None:
    """Return manifest if stage completed validly, else None."""
    try:
        root_fd = open_artifact_dir(stage_dir)
    except OSError:
        return None
    try:
        manifest = read_success_marker(root_fd)
        if manifest is None or manifest.config_sha != expected_config_sha:
            return None
        verify_artifact_tree(root_fd, manifest)
    except ArtifactVerificationError:
        return None
    finally:
        os.close(root_fd)
    return manifest


def quarantine_partial(stage_dir: Path) -> Path | None:
    """Move a stale/corrupt stage directory to a timestamped quarantine path."""
    if not os.path.lexists(stage_dir):
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
        try:
            root_fd = open_artifact_dir(self._partial)
        except OSError as e:
            raise StageError(f"Partial directory is not a plain directory: {self._partial}") from e
        try:
            try:
                expected = verify_artifact_tree(root_fd, manifest)
            except ArtifactVerificationError as e:
                raise StageError(f"Refusing to promote {self._partial}: {e}") from e
            fd = os.open(
                SUCCESS_MARKER,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o644,
                dir_fd=root_fd,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(manifest.to_json())
                f.flush()
                expected[SUCCESS_MARKER] = _file_identity(os.fstat(f.fileno()))
            if os.path.lexists(self.stage_dir):
                quarantine_partial(self.stage_dir)
            root_id = artifact_dir_identity(root_fd)
            if path_dir_identity(self._partial) != root_id:
                raise StageError(f"Partial directory replaced before promotion: {self._partial}")
            os.rename(str(self._partial), str(self.stage_dir))
            if path_dir_identity(self.stage_dir) != root_id or not artifact_tree_unchanged(
                root_fd, expected
            ):
                raise StageError(f"Stage directory changed during promotion: {self.stage_dir}")
        finally:
            os.close(root_fd)
        self._partial = None

    def cleanup_partial(self) -> None:
        if self._partial and self._partial.exists():
            shutil.rmtree(self._partial, ignore_errors=True)
            self._partial = None
