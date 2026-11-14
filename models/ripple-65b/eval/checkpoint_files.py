#!/usr/bin/env python3
"""Ripple-65B Preview (#39): confined reads of checkpoint directories, shared by the build (scripts/quantize_ripple.py),
the export validator, the size report and the BF16 teacher. Stdlib only, so it runs in every image and in the tests.

Every name that comes from a checkpoint is data, not a path:
  - Shard names in model.safetensors.index.json follow model_forge.pipeline.shard_name(): one plain file name with no
    directory part (no "/", "\\" or NUL; not "", "." or ".."). The directory's .safetensors files must be exactly the
    indexed shards (model.safetensors alone when there is no index), and each tensor must sit in the shard the index
    names for it.
  - A directory is opened once. Each file in it is opened relative to that descriptor with
    O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC and must be a regular file: a link is refused rather than followed,
    and a FIFO or a device is refused without blocking. The file is then read only through that descriptor (pread, or
    FD_DIR/<fd> for a reader that takes a path, such as safetensors' safe_open), never reopened by its name.
  - A safetensors header length above 100,000,000 bytes (the safetensors library's own MAX_HEADER_SIZE) or past the end
    of its file is refused before the header is read. The largest header in this family's checkpoints is 70,208 bytes
    (the export's shards; BF16 9,816, the vendor NVFP4 39,232, the drafter 8,968).
  - Every header entry (other than __metadata__) must be an object with a string dtype, a shape that is a list of
    non-negative integers and two integer data_offsets; every tensor's data_offsets must lie inside its file. As the
    safetensors library's own reader requires, a shard's tensors in offset order cover its data exactly: the first starts
    at the data's first byte, each starts where the one before it ends, and the last ends at the file's last byte (a
    gap, an overlap or a byte after the last tensor is refused). All four of this family's checkpoints are laid out so.
  - A small file read whole (a config, a sidecar, the index) is capped at MAX_SMALL_FILE_BYTES, 64 MiB: a larger one is
    refused from its size before any of it is read. The largest such file in this family is tokenizer.json, 7,137,177
    bytes.
  - refuse_links() walks a whole tree the same way, for a directory that libraries later read by path (transformers,
    accelerate and ModelOpt's exporter read the BF16 source): a link or a special file anywhere in it is refused.
  - read_source() is how the build and the teacher read the source: refuse_links() over the tree, then each pinned
    file read confined and checked against its sha256 pin.
  - write_files() writes each file as a new regular file under a fresh temporary name and renames it over the target,
    so an existing link at that name is replaced, never followed or written through.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import struct
from collections.abc import Mapping
from pathlib import Path

INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"
MAX_HEADER_BYTES = 100_000_000
# Configs, sidecars and the index (Directory.read_optional); the largest, tokenizer.json, is 7,137,177 bytes.
MAX_SMALL_FILE_BYTES = 64 * 1024 * 1024
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
FD_DIR = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"
CHUNK = 1 << 24


class ShardError(ValueError):
    """A checkpoint file that is not a plain regular file in its directory, or a shard set, a header, a tensor span or a
    pinned file that breaks the rules in the module docstring."""


def shard_name(name: object) -> str:
    """Return ``name`` if it is a single file name with no directory part (model_forge.pipeline.shard_name's rule)."""
    if not isinstance(name, str) or name in ("", ".", "..") or any(sep in name for sep in ("/", "\\", "\x00")):
        raise ShardError(f"shard name must be a single file name, got {name!r}")
    return name


def pread_exact(fd: int, offset: int, n: int, label: str) -> bytes:
    """Exactly ``n`` bytes at ``offset`` of the open file (one pread: a regular file reads short only at its end)."""
    b = os.pread(fd, n, offset)
    if len(b) != n:
        raise ShardError(f"{label}: short read ({len(b):,} of {n:,} bytes at {offset:,})")
    return b


class Directory:
    """A checkpoint directory, opened once; its files are opened relative to it and never through a link. Names passed
    here are fixed file names or index names weight_map() has checked."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        try:
            self.fd = os.open(self.root, DIR_FLAGS)
        except OSError as e:
            raise ShardError(f"{self.root}: cannot open the directory ({e.strerror})") from e

    def names(self) -> list[str]:
        return sorted(os.listdir(self.fd))

    def open(self, name: str) -> int:
        """Descriptor of the regular file ``name`` in the directory. FileNotFoundError when there is none; ShardError
        for a link, a FIFO, a device or a directory."""
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=self.fd)
        except FileNotFoundError:
            raise
        except OSError as e:
            raise ShardError(f"{self.root}/{name}: cannot open it as a file in the directory ({e.strerror}); links "
                             "are refused") from e
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ShardError(f"{self.root}/{name}: not a regular file")
        return fd

    def read_optional(self, name: str) -> bytes | None:
        """The bytes of the regular file ``name``, or None when there is no such entry. A file larger than
        MAX_SMALL_FILE_BYTES is refused from its size, before any of it is read."""
        try:
            fd = self.open(name)
        except FileNotFoundError:
            return None
        try:
            size = os.fstat(fd).st_size
            if size > MAX_SMALL_FILE_BYTES:
                raise ShardError(f"{self.root}/{name}: {size:,} bytes exceeds the {MAX_SMALL_FILE_BYTES:,}-byte limit for "
                                 "a small file")
            return pread_exact(fd, 0, size, f"{self.root}/{name}")
        finally:
            os.close(fd)

    def read(self, name: str) -> bytes:
        """The bytes of the regular file ``name``, which must exist."""
        data = self.read_optional(name)
        if data is None:
            raise ShardError(f"{self.root}/{name}: missing")
        return data

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> Directory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_file(root: Path | str, name: str) -> bytes:
    """The regular file ``name`` of the directory ``root``, read confined."""
    with Directory(root) as d:
        return d.read(name)


def refuse_links(root: Path | str) -> int:
    """Refuse a link or a special file anywhere under ``root``: every entry is opened relative to its parent
    directory's descriptor with O_NOFOLLOW and O_NONBLOCK and must be a regular file or a directory. Returns the number
    of regular files."""
    with Directory(root) as top:
        return _walk(Path(root), top.fd)


def _walk(path: Path, dir_fd: int) -> int:
    n = 0
    for name in sorted(os.listdir(dir_fd)):
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dir_fd)
        except OSError as e:
            raise ShardError(f"{path / name}: cannot open it in its directory ({e.strerror}); the tree must hold regular "
                             "files and directories only, no links") from e
        try:
            mode = os.fstat(fd).st_mode
            if stat.S_ISDIR(mode):
                n += _walk(path / name, fd)
            elif stat.S_ISREG(mode):
                n += 1
            else:
                raise ShardError(f"{path / name}: not a regular file or a directory")
        finally:
            os.close(fd)
    return n


def read_source(root: Path | str, pins: Mapping[str, str]) -> dict[str, bytes]:
    """The source checkpoint as the build and the teacher read it: no link or special file anywhere in the tree (the
    libraries that load it read it by path), and every pinned file read confined and equal to its sha256 pin."""
    refuse_links(root)
    out = {}
    with Directory(root) as d:
        for name, want in pins.items():
            data = d.read(name)
            got = hashlib.sha256(data).hexdigest()
            if got != want:
                raise ShardError(f"{d.root}/{name}: sha256 {got} is not the pin {want}")
            out[name] = data
    return out


def weight_map(raw: bytes, label: str) -> dict[str, str]:
    """The index's weight_map, every shard name one plain file name (shard_name)."""
    try:
        index = json.loads(raw)
    except (ValueError, RecursionError) as e:
        raise ShardError(f"{label}: unreadable {INDEX} ({e})") from e
    wm = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(wm, dict):
        raise ShardError(f"{label}: {INDEX} has no weight_map mapping")
    for tensor, shard in wm.items():
        try:
            shard_name(shard)
        except ShardError:
            raise ShardError(f"{label}: {tensor} maps to {shard!r}, not a file name in the directory") from None
    return wm


class Shards:
    """The safetensors shards of a checkpoint directory, each opened once (Directory.open) and read only through its
    descriptor. ``root`` is a path or an open Directory (left open when Shards closes)."""

    def __init__(self, root: Path | str | Directory):
        self._own = not isinstance(root, Directory)
        self.dir = root if isinstance(root, Directory) else Directory(root)
        self.root = self.dir.root
        self.fds: dict[str, int] = {}
        try:
            raw = self.dir.read_optional(INDEX)
            self.weight_map = None if raw is None else weight_map(raw, str(self.root))
            named = {SINGLE} if self.weight_map is None else set(self.weight_map.values())
            local = {n for n in self.dir.names() if n.endswith(".safetensors")}
            if local != named:
                raise ShardError(f"{self.root}: the .safetensors files are not exactly the indexed shards: unindexed "
                                 f"{sorted(local - named)[:4]}, missing {sorted(named - local)[:4]}")
            for name in sorted(named):
                self.fds[name] = self.dir.open(name)
        except BaseException:
            self.close()
            raise

    @property
    def names(self) -> list[str]:
        return sorted(self.fds)

    def path(self, shard: str) -> str:
        """A path that reads the shard through its open descriptor (for safe_open)."""
        return f"{FD_DIR}/{self.fds[shard]}"

    def pread(self, shard: str, offset: int, n: int) -> bytes:
        return pread_exact(self.fds[shard], offset, n, f"{self.root}/{shard}")

    def header(self, shard: str) -> tuple[dict, int]:
        """The shard's safetensors header and the offset its tensor data starts at."""
        fd, label = self.fds[shard], f"{self.root}/{shard}"
        size = os.fstat(fd).st_size
        (n,) = struct.unpack("<Q", pread_exact(fd, 0, 8, label))
        if n > MAX_HEADER_BYTES:
            raise ShardError(f"{label}: safetensors header length {n:,} exceeds the {MAX_HEADER_BYTES:,}-byte limit")
        if n > size - 8:
            raise ShardError(f"{label}: safetensors header length {n:,} runs past the end of the file ({size:,} bytes)")
        try:
            header = json.loads(pread_exact(fd, 8, n, label))
        except (ValueError, RecursionError) as e:
            raise ShardError(f"{label}: the safetensors header is not valid JSON ({e})") from e
        if not isinstance(header, dict):
            raise ShardError(f"{label}: the safetensors header is not a JSON object")
        return header, 8 + n

    def tensors(self) -> dict[str, tuple[str, int, int, str, list[int]]]:
        """name -> (shard, absolute offset, byte size, dtype, shape) for every tensor of every shard."""
        out: dict[str, tuple[str, int, int, str, list[int]]] = {}
        for shard in self.names:
            header, base = self.header(shard)
            size = os.fstat(self.fds[shard]).st_size
            spans: list[tuple[int, int, str]] = []
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                try:
                    a, b = meta["data_offsets"]
                    dtype, shape = meta["dtype"], meta["shape"]
                except (KeyError, TypeError, ValueError):
                    raise ShardError(f"{self.root}/{shard}: {name}: malformed header entry") from None
                if type(dtype) is not str or type(shape) is not list or not all(type(x) is int and x >= 0 for x in shape):
                    raise ShardError(f"{self.root}/{shard}: {name}: malformed header entry (dtype {dtype!r}, shape "
                                     f"{shape!r})")
                if not (type(a) is int and type(b) is int and 0 <= a <= b and base + b <= size):
                    raise ShardError(f"{self.root}/{shard}: {name}: data_offsets {meta['data_offsets']} fall outside "
                                     f"the file ({size:,} bytes)")
                if name in out:
                    raise ShardError(f"{name} appears in two shards")
                out[name] = (shard, base + a, b - a, dtype, shape)
                spans.append((a, b, name))
            end = 0  # safetensors' own rule: in offset order the tensors cover the shard's data exactly
            for a, b, name in sorted(spans):
                if a != end:
                    raise ShardError(f"{self.root}/{shard}: {name}: data_offsets [{a}, {b}] start at byte {a:,} of the "
                                     f"data, not at byte {end:,} where the tensor before it ends (a gap or an overlap)")
                end = b
            if base + end != size:
                raise ShardError(f"{self.root}/{shard}: the tensors end at byte {base + end:,} of the file, not at its "
                                 f"end ({size:,} bytes)")
        if self.weight_map is not None:
            if set(self.weight_map) != set(out):
                raise ShardError(f"{self.root}: index.json weight_map does not match the shard headers")
            moved = sorted(n for n, t in out.items() if self.weight_map[n] != t[0])
            if moved:
                raise ShardError(f"{self.root}: the index puts {moved[:3]} in other shards than the ones holding them")
        return out

    def close(self) -> None:
        for fd in self.fds.values():
            os.close(fd)
        self.fds = {}
        if self._own:
            self.dir.close()

    def __enter__(self) -> Shards:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_files(dest: Path | str, files: Mapping[str, bytes]) -> list[str]:
    """Write each file into the directory ``dest`` as a new regular file (a fresh temporary name, then renamed over the
    name), so a link or any other entry already at the name is replaced, never followed or written through. Returns the
    names written."""
    with Directory(dest) as d:
        for name, data in files.items():
            tmp = f".{name}.{secrets.token_hex(8)}.partial"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644, dir_fd=d.fd)
            with open(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, name, src_dir_fd=d.fd, dst_dir_fd=d.fd)
    return list(files)


def restore_export_files(export: Path | str, source_config: dict,
                         sidecars: Mapping[str, bytes]) -> tuple[list[str], dict, dict]:
    """Finish a ModelOpt export directory the way the build publishes it: keep the exporter's config.json as
    config.transformers5.json, write the source's config.json plus the export's quantization_config (and dtype) as
    config.json, write every sidecar from the pinned bytes the build read from the source (a link the exporter left at a
    sidecar's name is replaced), and read hf_quant_config.json. Every read is confined (Directory.read) and every write
    goes through write_files. Returns (sidecars written, quantization_config, hf_quant_config)."""
    with Directory(export) as d:
        exported = d.read("config.json")
        hf_quant = json.loads(d.read("hf_quant_config.json"))
    exp_cfg = json.loads(exported)
    qc = exp_cfg.get("quantization_config") if isinstance(exp_cfg, dict) else None
    if not qc:
        raise ShardError(f"{export}: the export's config.json has no quantization_config")
    new_cfg = {**source_config, "quantization_config": qc}
    if "dtype" in exp_cfg and "dtype" not in new_cfg:
        new_cfg["dtype"] = exp_cfg["dtype"]
    write_files(export, {"config.transformers5.json": exported,
                         "config.json": (json.dumps(new_cfg, indent=2) + "\n").encode()})
    return write_files(export, sidecars), qc, hf_quant
