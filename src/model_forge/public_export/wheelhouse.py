"""Build a deterministic offline wheelhouse from an existing Python environment."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import tomllib
import zipfile
from pathlib import Path
from typing import NoReturn

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_INTROSPECT = r"""
import json
import pathlib
import sys

if len(sys.argv) == 3:
    sys.path.insert(0, sys.argv[2])
import importlib.metadata

result = []
for dist in importlib.metadata.distributions():
    name = dist.metadata.get("Name")
    if not name:
        continue
    root = pathlib.Path(dist.locate_file(""))
    if not root.is_absolute():
        root = pathlib.Path.cwd() / root
    files = []
    for item in dist.files or []:
        located = pathlib.Path(dist.locate_file(item))
        if not located.is_absolute():
            located = pathlib.Path.cwd() / located
        files.append(
            {
                "archive": pathlib.PurePosixPath(item).as_posix(),
                "path": str(located),
                "record_hash_mode": item.hash.mode if item.hash is not None else None,
                "record_hash_value": item.hash.value if item.hash is not None else None,
                "record_size": item.size,
            }
        )
    wheel_text = dist.read_text("WHEEL")
    tags = [
        line.split(":", 1)[1].strip()
        for line in (wheel_text or "").splitlines()
        if line.startswith("Tag:")
    ]
    result.append(
        {
            "name": name,
            "version": dist.version,
            "requires": dist.requires or [],
            "roots": [str(root)],
            "files": files,
            "tag": sorted(tags)[0] if tags else None,
        }
    )
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
"""


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"offline wheelhouse bootstrap failed: {message}")


def _safe_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "_", canonicalize_name(value))


def _safe_version(value: str) -> str:
    return value.replace("-", "_")


class _UnsatisfiedClosure(Exception):
    pass


def _select_distribution_closure(
    distributions: list[dict[str, object]], requirements: list[str]
) -> list[dict[str, object]]:
    installed: dict[str, dict[str, object]] = {}
    for distribution in distributions:
        name = distribution.get("name")
        version = distribution.get("version")
        requires = distribution.get("requires")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            or not isinstance(requires, list)
            or not all(isinstance(value, str) for value in requires)
        ):
            _fail("environment returned malformed distribution metadata")
        installed.setdefault(canonicalize_name(name), distribution)

    selected: dict[str, set[str]] = {}
    queue = [(Requirement(value), "<project>") for value in requirements]
    while queue:
        requirement, parent = queue.pop(0)
        if requirement.marker is not None and not any(
            requirement.marker.evaluate({"extra": extra})
            for extra in ("", *sorted(requirement.extras))
        ):
            continue
        name = canonicalize_name(requirement.name)
        candidate = installed.get(name)
        if candidate is None:
            raise _UnsatisfiedClosure(f"missing distribution {requirement} required by {parent}")
        version = candidate["version"]
        if not isinstance(version, str) or version not in requirement.specifier:
            raise _UnsatisfiedClosure(
                f"has {requirement.name}=={version}, which does not satisfy "
                f"{requirement} required by {parent}"
            )
        previous_extras = selected.get(name, set())
        extras = previous_extras | set(requirement.extras)
        if name in selected and extras == previous_extras:
            continue
        selected[name] = extras
        raw_dependencies = candidate["requires"]
        if not isinstance(raw_dependencies, list):
            _fail("environment returned malformed distribution requirements")
        for value in raw_dependencies:
            if not isinstance(value, str):
                _fail("environment returned malformed distribution requirements")
            dependency = Requirement(value)
            if dependency.marker is None or any(
                dependency.marker.evaluate({"extra": extra}) for extra in ("", *sorted(extras))
            ):
                queue.append((dependency, f"{requirement.name}=={version}"))
    return [installed[name] for name in sorted(selected)]


def _distribution_metadata(
    python: Path, source_repo: Path, supplement: Path | None = None
) -> list[dict[str, object]]:
    environment = {
        "HOME": os.devnull,
        "LANG": "C",
        "LC_ALL": "C",
        "NO_PROXY": "*",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }
    try:
        arguments = [str(python), "-I", "-c", _INTROSPECT, str(source_repo)]
        if supplement is not None:
            arguments.append(str(supplement))
        result = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        _fail(f"cannot inspect local distributions with {python}: {error}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        _fail(
            f"cannot inspect local distributions with {python}"
            + (
                f":\n{detail}"
                if detail
                else f": {python} could not inspect installed distributions"
            )
        )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        _fail(f"{python} returned malformed distribution metadata: {error}")
    if not isinstance(value, list):
        _fail(f"{python} returned invalid distribution metadata")
    if not all(isinstance(item, dict) for item in value):
        _fail(f"{python} returned malformed distribution metadata")
    return _select_distribution_closure(value, _root_requirements(source_repo))


def _root_requirements(source_repo: Path) -> list[str]:
    try:
        with (source_repo / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)
        values = [
            *project["build-system"]["requires"],
            *project["project"].get("dependencies", []),
        ]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        _fail(f"project dependency metadata is invalid: {error}")
    if not values or not all(isinstance(value, str) and value for value in values):
        _fail("project build and runtime requirements are invalid")
    return values


def _materialize_uv_cache(uv: str, python: Path, source_repo: Path, target: Path) -> None:
    command = [
        uv,
        "pip",
        "install",
        "--offline",
        "--no-config",
        "--no-python-downloads",
        "--only-binary",
        ":all:",
        "--target",
        str(target),
        "--python",
        str(python),
        *_root_requirements(source_repo),
    ]
    environment = os.environ.copy()
    for name in (
        "PIP_EXTRA_INDEX_URL",
        "PIP_INDEX_URL",
        "UV_DEFAULT_INDEX",
        "UV_EXTRA_INDEX_URL",
        "UV_INDEX",
        "UV_INDEX_URL",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "UV_OFFLINE": "1",
            "UV_PYTHON_DOWNLOADS": "never",
        }
    )
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        _fail(f"uv cache materialization failed: {error}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        _fail(
            "local uv cache cannot satisfy the committed dependency closure"
            + (f":\n{detail}" if detail else "")
        )


_MAX_INSTALLED_FILE_BYTES = 512 * 1024 * 1024


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        stat.S_IFMT(left.st_mode),
    ) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
    )


def _read_installed_file(path: Path, archive: str) -> bytes:
    if not path.is_absolute() or ".." in path.parts:
        _fail(f"installed distribution file path is unsafe: {archive}")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        directory_fd = os.open(path.anchor, directory_flags)
        for component in path.parts[1:-1]:
            before = os.stat(component, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode):
                _fail(f"installed distribution file has a symlinked ancestor: {archive}")
            if not stat.S_ISDIR(before.st_mode):
                _fail(f"installed distribution file ancestor is not a directory: {archive}")
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            after = os.fstat(next_fd)
            if not _same_file_identity(before, after):
                os.close(next_fd)
                _fail(f"installed distribution file ancestor changed while opening: {archive}")
            os.close(directory_fd)
            directory_fd = next_fd

        filename = path.parts[-1]
        before = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            _fail(f"installed distribution file is not a regular file: {archive}")
        if before.st_size > _MAX_INSTALLED_FILE_BYTES:
            _fail(f"installed distribution file exceeds size limit: {archive}")
        file_fd = os.open(filename, file_flags, dir_fd=directory_fd)
        opened = os.fstat(file_fd)
        if not _same_file_identity(before, opened):
            _fail(f"installed distribution file changed while opening: {archive}")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(file_fd, min(1024 * 1024, _MAX_INSTALLED_FILE_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > _MAX_INSTALLED_FILE_BYTES:
                _fail(f"installed distribution file exceeds size limit: {archive}")
        after = os.fstat(file_fd)
        if (
            not _same_file_identity(opened, after)
            or opened.st_size != after.st_size
            or opened.st_mtime_ns != after.st_mtime_ns
            or size != after.st_size
        ):
            _fail(f"installed distribution file changed while reading: {archive}")
        return b"".join(chunks)
    except OSError as error:
        _fail(f"installed distribution file is unavailable: {archive}: {error}")
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _wheel_entries(distribution: dict[str, object]) -> tuple[dict[str, bytes], str]:
    raw_files = distribution.get("files")
    if not isinstance(raw_files, list):
        _fail("installed distribution has no file inventory")
    raw_roots = distribution.get("roots")
    if (
        not isinstance(raw_roots, list)
        or not raw_roots
        or not all(isinstance(root, str) and Path(root).is_absolute() for root in raw_roots)
    ):
        _fail("installed distribution has invalid allowed roots")
    roots = [os.path.normpath(root) for root in raw_roots]
    entries: dict[str, bytes] = {}
    seen: set[str] = set()
    dist_info: str | None = None
    source_record_data: bytes | None = None
    for raw in raw_files:
        if not isinstance(raw, dict):
            _fail("installed distribution file inventory is malformed")
        archive = raw.get("archive")
        path_value = raw.get("path")
        if not isinstance(archive, str) or not isinstance(path_value, str):
            _fail("installed distribution file inventory is malformed")
        parts = Path(archive).parts
        if not parts or ".." in parts or Path(archive).is_absolute():
            _fail(f"installed distribution RECORD path is unsafe: {archive}")
        if archive in seen:
            _fail(f"installed distribution RECORD path is duplicated: {archive}")
        seen.add(archive)
        path = Path(path_value)
        normalized_path = os.path.normpath(path_value)
        if not path.is_absolute() or not any(
            os.path.commonpath((root, normalized_path)) == root for root in roots
        ):
            _fail(f"installed distribution file is outside distribution roots: {archive}")
        if parts[0].endswith(".dist-info"):
            dist_info = parts[0]
        data = _read_installed_file(path, archive)
        record_size = raw.get("record_size")
        if record_size is not None and (
            type(record_size) is not int or record_size < 0 or len(data) != record_size
        ):
            _fail(f"installed distribution RECORD size mismatch: {archive}")
        record_hash_mode = raw.get("record_hash_mode")
        record_hash_value = raw.get("record_hash_value")
        if record_hash_mode is not None or record_hash_value is not None:
            if record_hash_mode != "sha256" or not isinstance(record_hash_value, str):
                _fail(f"installed distribution RECORD hash is unsupported: {archive}")
            actual_hash = (
                base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            )
            if actual_hash != record_hash_value:
                _fail(f"installed distribution RECORD hash mismatch: {archive}")
        if archive.endswith(".dist-info/RECORD"):
            source_record_data = data
            continue
        entries[archive] = data
    if dist_info is None:
        _fail("installed distribution has no dist-info directory")
    source_record_path = f"{dist_info}/RECORD"
    if source_record_data is None or source_record_path not in seen:
        _fail("installed distribution has no RECORD metadata")
    rows: list[tuple[str, str, str]] = []
    for archive, data in sorted(entries.items()):
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append((archive, f"sha256={encoded}", str(len(data))))
    record_path = source_record_path
    rows.append((record_path, "", ""))
    record = io.StringIO()
    csv.writer(record, lineterminator="\n").writerows(rows)
    entries[record_path] = record.getvalue().encode("utf-8")
    source_record_evidence = (
        f"sha256={hashlib.sha256(source_record_data).hexdigest()} size={len(source_record_data)}"
    )
    return entries, source_record_evidence


def _write_wheel(distribution: dict[str, object], wheelhouse: Path) -> tuple[Path, str]:
    name = distribution.get("name")
    version = distribution.get("version")
    tag = distribution.get("tag")
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(version, str)
        or not version
        or not isinstance(tag, str)
        or not tag
    ):
        _fail("installed distribution identity is malformed")
    filename = f"{_safe_distribution(name)}-{_safe_version(version)}-{tag}.whl"
    target = wheelhouse / filename
    entries, source_record = _wheel_entries(distribution)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, data in sorted(entries.items()):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    target.chmod(0o644)
    identity = f"{canonicalize_name(name)}=={version}"
    return target, f"# source-record {source_record} distribution={identity}\n"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--environment-python", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    source_repo = arguments.source_repo.resolve()
    environment_python = arguments.environment_python.absolute()
    wheelhouse = arguments.wheelhouse.absolute()
    lock = arguments.lock.absolute()
    if not (source_repo / "pyproject.toml").is_file():
        _fail("source repository has no pyproject.toml")
    if not environment_python.is_file():
        _fail(f"distribution environment Python is unavailable: {environment_python}")
    if wheelhouse.exists() or lock.exists():
        _fail("output wheelhouse and lock paths must not already exist")
    if lock.parent != wheelhouse.parent:
        _fail("wheelhouse and lock must share a private parent directory")

    temporary_parent = Path(os.path.realpath(wheelhouse.parent))
    temporary = Path(tempfile.mkdtemp(prefix=".offline-wheelhouse-", dir=temporary_parent))
    try:
        temporary.chmod(0o700)
        supplement = temporary / ".installed"
        try:
            metadata = _distribution_metadata(environment_python, source_repo)
        except _UnsatisfiedClosure as installed_error:
            uv = shutil.which("uv")
            if uv is None:
                _fail(
                    "installed environment dependency closure is unsatisfiable: "
                    f"{installed_error}\noffline uv fallback is unavailable"
                )
            supplement.mkdir(mode=0o700)
            try:
                _materialize_uv_cache(uv, environment_python, source_repo, supplement)
                metadata = _distribution_metadata(environment_python, source_repo, supplement)
            except (SystemExit, _UnsatisfiedClosure) as fallback_error:
                _fail(
                    "installed environment dependency closure is unsatisfiable: "
                    f"{installed_error}\noffline uv fallback failed: {fallback_error}"
                )
        built = [_write_wheel(item, temporary) for item in metadata]
        if supplement.exists():
            shutil.rmtree(supplement)
        if not built:
            _fail("project declares no build or runtime distributions")
        lock_bytes = "".join(
            (f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n{source_record}")
            for path, source_record in sorted(built, key=lambda item: item[0].name)
        ).encode("ascii")
        temporary_lock = temporary.parent / f".{lock.name}.tmp-{os.getpid()}"
        temporary_lock.write_bytes(lock_bytes)
        temporary_lock.chmod(0o600)
        os.replace(temporary, wheelhouse)
        os.replace(temporary_lock, lock)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return 0
