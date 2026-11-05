"""Shard names from a base checkpoint must not escape merge_lora.py --out.

``merge_lora.py`` copies ``weight_map`` values onto both the base directory and
the output directory. These tests drive that script. They fail while a shard
name can create or replace a file outside the output directory, including when
a check accepts one path and the write uses another.

Same weight_map join, left untested here:
``scripts/qwen3_8/apply_qwen38_abliteration.py``,
``models/nemotron-3.5-lightning-r1/eval/make_split.py``,
``models/nemotron-3.5-lightning-r1/eval/split_mse.py``.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import select
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_SCRIPT = Path(__file__).resolve().parents[1] / "models" / "jev-27b" / "scripts" / "merge_lora.py"
_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
_IN_NONBLOCK = 0o4000
_IN_MODIFY = 0x00000002
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_Q_OVERFLOW = 0x00004000
_MUTATION = (
    _IN_MODIFY
    | _IN_CLOSE_WRITE
    | _IN_MOVED_FROM
    | _IN_MOVED_TO
    | _IN_CREATE
    | _IN_DELETE
    | _IN_DELETE_SELF
    | _IN_MOVE_SELF
    | _IN_Q_OVERFLOW
)
_EVENT_HEADER = struct.calcsize("iIII")

_LIBC.inotify_init1.argtypes = [ctypes.c_int]
_LIBC.inotify_init1.restype = ctypes.c_int
_LIBC.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
_LIBC.inotify_add_watch.restype = ctypes.c_int


def _prepare(root: Path, shard: str, shard_file: Path) -> tuple[Path, Path, Path]:
    base = root / "base_root" / "base"
    out = root / "out_root" / "merge_out"
    adapter = root / "adapter"
    base.mkdir(parents=True)
    adapter.mkdir()
    out.mkdir(parents=True)
    rows, cols, rank = 4, 3, 2
    weight = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    shard_file.parent.mkdir(parents=True, exist_ok=True)
    save_file({"lm_head.weight": weight}, shard_file, metadata={"format": "pt"})
    save_file(
        {
            "base_model.model.lm_head.lora_A.weight": torch.ones((rank, cols)),
            "base_model.model.lm_head.lora_B.weight": torch.ones((rows, rank)),
        },
        adapter / "adapter_model.safetensors",
    )
    (adapter / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": 4.0}),
        encoding="utf-8",
    )
    (adapter / "decision_head.json").write_text("{}", encoding="utf-8")
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"lm_head.weight": shard}}),
        encoding="utf-8",
    )
    (base / "config.json").write_text("{}", encoding="utf-8")
    return base, adapter, out


def _snapshot(root: Path) -> dict[str, tuple[object, ...]]:
    records: dict[str, tuple[object, ...]] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            path = Path(dirpath) / name
            if path.is_symlink():
                records[str(path)] = ("symlink", os.readlink(path))
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                records[str(path)] = ("symlink", os.readlink(path))
                continue
            data = path.read_bytes()
            records[str(path)] = ("file", path.stat().st_ino, hashlib.sha256(data).hexdigest())
    return records


def _inside(path: Path, out: Path) -> bool:
    resolved = path.resolve()
    root = out.resolve()
    return resolved == root or root in resolved.parents


def _outside_changes(
    before: dict[str, tuple[object, ...]],
    after: dict[str, tuple[object, ...]],
    out: Path,
) -> list[str]:
    changes: list[str] = []
    for key in sorted(set(before) | set(after)):
        if _inside(Path(key), out):
            continue
        if before.get(key) != after.get(key):
            changes.append(key)
    return changes


class _MutationWatch:
    """Record create, delete, rename, and write events outside ``out``.

    Constants match glibc ``sys/inotify.h`` on this system. The watch does not
    follow directory symlinks, so a write through ``out/<link>`` shows up in
    the link target's directory.
    """

    def __init__(self, root: Path, out: Path) -> None:
        self._root = root
        self._out = out
        self._events: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fd = _LIBC.inotify_init1(_IN_NONBLOCK)
        if self._fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), "inotify_init1")
        self._dirs: dict[int, Path] = {}
        for directory in self._directories():
            watch = _LIBC.inotify_add_watch(self._fd, os.fsencode(directory), _MUTATION)
            if watch < 0:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err), str(directory))
            self._dirs[watch] = directory
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _directories(self) -> list[Path]:
        found: list[Path] = []
        for dirpath, dirnames, _filenames in os.walk(self._root, followlinks=False):
            current = Path(dirpath)
            if _inside(current, self._out):
                dirnames[:] = []
                continue
            found.append(current)
            dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        return found

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        poller = select.poll()
        poller.register(self._fd, select.POLLIN)
        while not self._stop.is_set():
            if poller.poll(50):
                self._drain()

    def _drain(self) -> None:
        while True:
            try:
                data = os.read(self._fd, 65536)
            except BlockingIOError:
                return
            offset = 0
            while offset + _EVENT_HEADER <= len(data):
                wd, mask, _cookie, length = struct.unpack_from("iIII", data, offset)
                offset += _EVENT_HEADER
                raw = data[offset : offset + length].split(b"\x00", 1)[0]
                offset += length
                if mask & _IN_Q_OVERFLOW:
                    with self._lock:
                        self._events.append("inotify queue overflow")
                    continue
                directory = self._dirs.get(wd)
                if directory is None:
                    continue
                path = directory / raw.decode() if raw else directory
                if _inside(path, self._out):
                    continue
                with self._lock:
                    self._events.append(f"{mask:#x} {path}")

    def finish(self) -> list[str]:
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            time.sleep(0.05)
        self._stop.set()
        self._thread.join(timeout=2)
        self._drain()
        os.close(self._fd)
        with self._lock:
            return list(self._events)


def _run(base: Path, adapter: Path, out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--base",
            str(base),
            "--adapter",
            str(adapter),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _exercise(root: Path, base: Path, adapter: Path, out: Path, *, success: bool) -> None:
    before = _snapshot(root)
    watch = _MutationWatch(root, out)
    watch.start()
    try:
        proc = _run(base, adapter, out)
    finally:
        events = watch.finish()
    changes = _outside_changes(before, _snapshot(root), out)
    detail = f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert changes == [], f"paths outside {out} changed: {changes}; {detail}"
    assert events == [], f"mutation outside {out}: {events}; {detail}"
    if success:
        assert proc.returncode == 0, detail
        written = out / "model-00001-of-00001.safetensors"
        assert _inside(written, out)
        assert "lm_head.weight" in load_file(written)
        return
    assert proc.returncode != 0, detail


def test_parent_directory_shard_does_not_write_outside_output() -> None:
    with tempfile.TemporaryDirectory(prefix="merge-lora-parent-") as tmp:
        root = Path(tmp)
        victim = root / "out_root" / "other_dir"
        victim.mkdir(parents=True)
        shard = "../other_dir/stolen.safetensors"
        read_at = root / "base_root" / "other_dir" / "stolen.safetensors"
        base, adapter, out = _prepare(root, shard, read_at)
        _exercise(root, base, adapter, out, success=False)


def test_prefix_sibling_of_output_dir_is_outside() -> None:
    """``merge_out_pwned`` starts with the output directory name ``merge_out``."""
    with tempfile.TemporaryDirectory(prefix="merge-lora-prefix-") as tmp:
        root = Path(tmp)
        victim = root / "out_root" / "merge_out_pwned"
        victim.mkdir(parents=True)
        shard = "../merge_out_pwned/stolen.safetensors"
        read_at = root / "base_root" / "merge_out_pwned" / "stolen.safetensors"
        base, adapter, out = _prepare(root, shard, read_at)
        _exercise(root, base, adapter, out, success=False)


def test_dot_slash_parent_shard_does_not_write_outside_output() -> None:
    with tempfile.TemporaryDirectory(prefix="merge-lora-dot-") as tmp:
        root = Path(tmp)
        victim = root / "out_root" / "other_dir"
        victim.mkdir(parents=True)
        shard = "./../other_dir/stolen.safetensors"
        read_at = root / "base_root" / "other_dir" / "stolen.safetensors"
        base, adapter, out = _prepare(root, shard, read_at)
        _exercise(root, base, adapter, out, success=False)


def test_nested_parent_segments_do_not_write_outside_output() -> None:
    with tempfile.TemporaryDirectory(prefix="merge-lora-nested-") as tmp:
        root = Path(tmp)
        victim = root / "out_root" / "other_dir"
        victim.mkdir(parents=True)
        shard = "sub/../../other_dir/stolen.safetensors"
        read_at = root / "base_root" / "other_dir" / "stolen.safetensors"
        base, adapter, out = _prepare(root, shard, read_at)
        (base / "sub").mkdir()
        (out / "sub").mkdir()
        _exercise(root, base, adapter, out, success=False)


def test_absolute_shard_does_not_overwrite_file_outside_output() -> None:
    with tempfile.TemporaryDirectory(prefix="merge-lora-absolute-") as tmp:
        root = Path(tmp)
        shard_file = root / "absolute_victim" / "abs.safetensors"
        base, adapter, out = _prepare(root, str(shard_file), shard_file)
        _exercise(root, base, adapter, out, success=False)


def test_symlink_directory_shard_does_not_write_outside_output() -> None:
    with tempfile.TemporaryDirectory(prefix="merge-lora-symlink-") as tmp:
        root = Path(tmp)
        real = root / "real_shard"
        victim = root / "outside_victim"
        victim.mkdir()
        shard_file = real / "stolen.safetensors"
        base, adapter, out = _prepare(root, "alias/stolen.safetensors", shard_file)
        (base / "alias").symlink_to(real)
        (out / "alias").symlink_to(victim)
        _exercise(root, base, adapter, out, success=False)


def test_single_filename_shard_is_written_inside_output() -> None:
    with tempfile.TemporaryDirectory(prefix="merge-lora-ok-") as tmp:
        root = Path(tmp)
        shard = "model-00001-of-00001.safetensors"
        read_at = root / "base_root" / "base" / shard
        base, adapter, out = _prepare(root, shard, read_at)
        _exercise(root, base, adapter, out, success=True)
