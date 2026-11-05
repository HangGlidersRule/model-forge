"""Security spec for merge_lora companion copies.

``merge_lora.py`` must not publish the bytes of a symlink target. A symlink
planted as a base companion, as ``model.safetensors.index.json``, as the
adapter ``decision_head.json``, or as ``calibration.json`` beside the adapter
must make ``main()`` stop with ``SystemExit`` (non-zero) or a non-zero return,
and must not create that output name.

The replacement cases start from a regular file and, after that path has been
looked up with ``stat`` / ``lstat`` and before the next open, replace it with
a symlink. Copying the new target fails the test: the bytes written have to
be the bytes of the regular file that was looked up, or the copy has to be
refused without creating the output. Opening the path again between the check
and the read is the failure these cases lock in.

Not covered here, because they are different operations:
``README.md`` and ``SOURCE_REVISION`` are excluded by name (asserted below);
a symlink to a directory is not selected by ``Path.is_file`` (asserted below);
shard ``*.safetensors`` files are read by ``safe_open`` and rewritten by
``save_file`` rather than ``copy2``; ``scripts/qwen3_8/apply_qwen38_abliteration.py``
and the lightning eval split scripts also call ``copy2`` but are not this merge.
"""

from __future__ import annotations

import builtins
import importlib.util
import io
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

_SCRIPT = Path(__file__).resolve().parents[1] / "models" / "jev-27b" / "scripts" / "merge_lora.py"
_SPEC = importlib.util.spec_from_file_location("merge_lora_under_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MERGE = importlib.util.module_from_spec(_SPEC)
sys.modules["merge_lora_under_test"] = _MERGE
_SPEC.loader.exec_module(_MERGE)

_WEIGHT = "model.language_model.layers.0.mlp.down_proj.weight"
_LORA_A = "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"
_LORA_B = "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight"
_SHARD = "model-00001-of-00001.safetensors"
_MARKER = b"PLACEHOLDER-SYMLINK-TARGET"
_PLACES = ("base_index", "base_companion", "decision_head", "calibration")
_STYLES = ("absolute", "relative")


def _index_document(marker: bytes | None) -> bytes:
    metadata: dict[str, str] = {}
    if marker is not None:
        metadata["marker"] = marker.decode("ascii")
    return json.dumps({"metadata": metadata, "weight_map": {_WEIGHT: _SHARD}}).encode()


def _load_merge_inputs(base: Path, adapter: Path) -> None:
    weight = torch.ones(2, 2, dtype=torch.float32)
    factor_a = torch.zeros(1, 2, dtype=torch.float32)
    factor_b = torch.zeros(2, 1, dtype=torch.float32)
    save_file({_WEIGHT: weight}, base / _SHARD, metadata={"format": "pt"})
    save_file({_LORA_A: factor_a, _LORA_B: factor_b}, adapter / "adapter_model.safetensors")
    (adapter / "adapter_config.json").write_text(
        json.dumps({"r": 1, "lora_alpha": 1.0}),
        encoding="utf-8",
    )


def _link(path: Path, target: Path, style: str) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()
    if style == "absolute":
        path.symlink_to(target)
        return
    if style == "relative":
        path.symlink_to(os.path.relpath(target, path.parent))
        return
    raise AssertionError(style)


class _Layout:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.base = root / "base"
        self.bundle = root / "bundle"
        self.adapter = self.bundle / "adapter"
        self.out = root / "out"
        self.outside = root / "outside"
        self.index_bytes = _index_document(None)
        self.config_bytes = b'{"companion":"regular-config"}'
        self.tokenizer_bytes = b'{"companion":"regular-tokenizer"}'
        self.decision_bytes = b'{"head":"regular"}'
        self.calibration_bytes = b'{"calibration":"regular"}'

    def build(self) -> None:
        self.base.mkdir()
        self.adapter.mkdir(parents=True)
        self.outside.mkdir()
        _load_merge_inputs(self.base, self.adapter)
        (self.base / "model.safetensors.index.json").write_bytes(self.index_bytes)
        (self.base / "config.json").write_bytes(self.config_bytes)
        (self.base / "tokenizer.json").write_bytes(self.tokenizer_bytes)
        (self.base / "README.md").write_bytes(b"regular readme")
        (self.base / "SOURCE_REVISION").write_bytes(b"regular-revision")
        (self.adapter / "decision_head.json").write_bytes(self.decision_bytes)
        (self.bundle / "calibration.json").write_bytes(self.calibration_bytes)

    def plant(self, place: str, style: str) -> set[str]:
        """Replace the named inputs with symlinks to files outside the trees."""
        outside = self.outside
        if place == "base_index":
            target = outside / "index.json"
            target.write_bytes(_index_document(_MARKER))
            _link(self.base / "model.safetensors.index.json", target, style)
            return {"model.safetensors.index.json"}
        if place == "base_companion":
            mapping = {
                "config.json": self.base / "config.json",
                "tokenizer.json": self.base / "tokenizer.json",
                "README.md": self.base / "README.md",
                "SOURCE_REVISION": self.base / "SOURCE_REVISION",
                "extra.safetensors": self.base / "extra.safetensors",
            }
            for name, path in mapping.items():
                target = outside / f"{name}.target"
                target.write_bytes(_MARKER + b"\n" + name.encode())
                _link(path, target, style)
            directory = outside / "companion_dir"
            directory.mkdir()
            (directory / "payload.txt").write_bytes(_MARKER + b"\ndirectory")
            link = self.base / "companion_dir"
            if style == "absolute":
                link.symlink_to(directory, target_is_directory=True)
            else:
                link.symlink_to(os.path.relpath(directory, link.parent), target_is_directory=True)
            return set(mapping) | {"companion_dir"}
        if place == "decision_head":
            target = outside / "decision_head.json"
            target.write_bytes(_MARKER + b"\ndecision")
            _link(self.adapter / "decision_head.json", target, style)
            return {"decision_head.json"}
        if place == "calibration":
            target = outside / "calibration.json"
            target.write_bytes(_MARKER + b"\ncalibration")
            _link(self.bundle / "calibration.json", target, style)
            return {"calibration.json"}
        raise AssertionError(place)


def _invoke(layout: _Layout) -> int:
    argv = [
        "merge_lora.py",
        "--base",
        str(layout.base),
        "--adapter",
        str(layout.adapter),
        "--out",
        str(layout.out),
    ]
    previous = sys.argv
    sys.argv = argv
    try:
        result = _MERGE.main()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    else:
        return int(result)
    finally:
        sys.argv = previous


def _marker_hits(out: Path, captured: bytes) -> list[str]:
    hits: list[str] = []
    if _MARKER in captured:
        hits.append("bytes captured from an output write")
    if not out.exists():
        return hits
    for path in out.rglob("*"):
        if path.is_symlink():
            hits.append(f"symlink {path.relative_to(out)} -> {os.readlink(path)}")
            continue
        if path.is_file() and _MARKER in path.read_bytes():
            hits.append(f"file {path.relative_to(out)}")
    return hits


class _Race:
    """Attacker: after a metadata lookup, replace the source before the read.

    ``replace`` is false when the test only needs to see output creates. The
    swap runs only after ``stat`` or ``lstat`` has already returned for the
    source, and only on the next read open of that same path.
    """

    def __init__(self, source: Path, secret: Path, output: Path, *, replace: bool) -> None:
        self.source = os.path.abspath(source)
        self.secret = secret
        self.output = os.path.abspath(output)
        self.replace = replace
        self.lookups = 0
        self.swapped = False
        self.written: set[str] = set()
        self.captured = bytearray()
        self.output_fds: dict[int, str] = {}
        self._busy = False
        self._pending: dict[int, bytearray] = {}

    def _norm(self, path: object, dir_fd: int | None) -> str:
        text = os.fspath(path)  # type: ignore[arg-type]
        if isinstance(text, bytes):
            text = os.fsdecode(text)
        if dir_fd is None:
            return os.path.abspath(text)
        base = os.readlink(f"/proc/self/fd/{dir_fd}")
        if os.path.isabs(text):
            return os.path.abspath(text)
        return os.path.normpath(os.path.join(base, text))

    def _relative_output(self, path: object, dir_fd: int | None) -> str | None:
        norm = self._norm(path, dir_fd)
        prefix = self.output + os.sep
        if not norm.startswith(prefix):
            return None
        return norm[len(prefix) :]

    def note(self, path: object, dir_fd: int | None) -> None:
        if self._busy:
            return
        if self._norm(path, dir_fd) == self.source:
            self.lookups += 1

    def swap_before_read(self, path: object, dir_fd: int | None) -> None:
        if not self.replace or self._busy or self.swapped or self.lookups == 0:
            return
        if self._norm(path, dir_fd) != self.source:
            return
        self._busy = True
        try:
            os.unlink(self.source)
            os.symlink(self.secret, self.source)
            self.swapped = True
        finally:
            self._busy = False


class _OutputFile:
    """File object that records writes. ``copy2``'s sendfile path is recorded separately."""

    def __init__(self, raw: io.IOBase, race: _Race, relative: str) -> None:
        self._raw = raw
        self._race = race
        self._relative = relative

    def write(self, data: object) -> int:
        if isinstance(data, str):
            view: bytes | bytearray = data.encode()
        elif isinstance(data, (bytes, bytearray)):
            view = data
        else:
            view = bytes(data)  # type: ignore[arg-type]
        self._race.captured.extend(view)
        self._race.written.add(self._relative)
        return self._raw.write(data)  # type: ignore[arg-type]

    def __enter__(self) -> _OutputFile:
        self._raw.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> object:
        return self._raw.__exit__(exc_type, exc, tb)

    def __getattr__(self, name: str) -> object:
        return getattr(self._raw, name)


@contextmanager
def _race_window(race: _Race) -> Iterator[None]:
    real_stat = os.stat
    real_lstat = os.lstat
    real_open = os.open
    real_write = os.write
    real_close = os.close
    real_sendfile = os.sendfile
    real_io_open = io.open

    def wrapped_stat(
        path: object, *, dir_fd: int | None = None, follow_symlinks: bool = True
    ) -> os.stat_result:
        if dir_fd is None:
            result = real_stat(path, follow_symlinks=follow_symlinks)  # type: ignore[call-overload]
        else:
            result = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)  # type: ignore[call-overload]
        race.note(path, dir_fd)
        return result

    def wrapped_lstat(path: object, *, dir_fd: int | None = None) -> os.stat_result:
        if dir_fd is None:
            result = real_lstat(path)  # type: ignore[call-overload]
        else:
            result = real_lstat(path, dir_fd=dir_fd)  # type: ignore[call-overload]
        race.note(path, dir_fd)
        return result

    def wrapped_os_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        access = flags & os.O_ACCMODE
        if access == os.O_RDONLY:
            race.swap_before_read(path, dir_fd)
        if dir_fd is None:
            descriptor = real_open(path, flags, mode)  # type: ignore[call-overload]
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[call-overload]
        relative = race._relative_output(path, dir_fd)
        if relative is not None and access != os.O_RDONLY:
            race.written.add(relative)
            race._pending[descriptor] = bytearray()
        return descriptor

    def wrapped_write(descriptor: int, data: object) -> int:
        written = real_write(descriptor, data)  # type: ignore[arg-type]
        pending = race._pending.get(descriptor)
        if pending is not None:
            view = data if isinstance(data, (bytes, bytearray)) else bytes(data)  # type: ignore[arg-type]
            pending.extend(view[:written])
        return written

    def wrapped_close(descriptor: int) -> None:
        pending = race._pending.pop(descriptor, None)
        if pending:
            race.captured.extend(pending)
        real_close(descriptor)

    def wrapped_sendfile(out_fd: int, in_fd: int, offset: int | None, count: int) -> int:
        sent = real_sendfile(out_fd, in_fd, offset, count)
        if sent and out_fd in race.output_fds:
            start = 0 if offset is None else offset
            race.captured.extend(os.pread(in_fd, sent, start))
            race.written.add(race.output_fds[out_fd])
        return sent

    def wrapped_io_open(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
        if "r" in mode or "+" in mode:
            race.swap_before_read(file, None)
        handle = real_io_open(file, mode, *args, **kwargs)  # type: ignore[arg-type]
        relative = race._relative_output(file, None)
        if relative is not None and any(flag in mode for flag in "wax+"):
            race.written.add(relative)
            try:
                race.output_fds[handle.fileno()] = relative
            except OSError:
                pass
            return _OutputFile(handle, race, relative)
        return handle

    os.stat = wrapped_stat  # type: ignore[assignment]
    os.lstat = wrapped_lstat  # type: ignore[assignment]
    os.open = wrapped_os_open  # type: ignore[assignment]
    os.write = wrapped_write  # type: ignore[assignment]
    os.close = wrapped_close  # type: ignore[assignment]
    os.sendfile = wrapped_sendfile  # type: ignore[assignment]
    io.open = wrapped_io_open  # type: ignore[assignment]
    builtins.open = wrapped_io_open  # type: ignore[assignment]
    try:
        yield
    finally:
        os.stat = real_stat  # type: ignore[assignment]
        os.lstat = real_lstat  # type: ignore[assignment]
        os.open = real_open  # type: ignore[assignment]
        os.write = real_write  # type: ignore[assignment]
        os.close = real_close  # type: ignore[assignment]
        os.sendfile = real_sendfile  # type: ignore[assignment]
        io.open = real_io_open  # type: ignore[assignment]
        builtins.open = real_io_open  # type: ignore[assignment]


def _source_for(layout: _Layout, place: str) -> Path:
    if place == "base_index":
        return layout.base / "model.safetensors.index.json"
    if place == "base_companion":
        return layout.base / "config.json"
    if place == "decision_head":
        return layout.adapter / "decision_head.json"
    if place == "calibration":
        return layout.bundle / "calibration.json"
    raise AssertionError(place)


def _output_name(place: str) -> str:
    if place == "base_index":
        return "model.safetensors.index.json"
    if place == "base_companion":
        return "config.json"
    if place == "decision_head":
        return "decision_head.json"
    if place == "calibration":
        return "calibration.json"
    raise AssertionError(place)


def _original_bytes(layout: _Layout, place: str) -> bytes:
    if place == "base_index":
        return layout.index_bytes
    if place == "base_companion":
        return layout.config_bytes
    if place == "decision_head":
        return layout.decision_bytes
    if place == "calibration":
        return layout.calibration_bytes
    raise AssertionError(place)


@pytest.mark.parametrize("place", _PLACES)
@pytest.mark.parametrize("style", _STYLES)
def test_symlink_target_is_not_published(tmp_path: Path, place: str, style: str) -> None:
    layout = _Layout(tmp_path)
    layout.build()
    sensitive = layout.plant(place, style)
    race = _Race(layout.base / "unused", layout.outside / "unused", layout.out, replace=False)
    with _race_window(race):
        code = _invoke(layout)
    hits = _marker_hits(layout.out, bytes(race.captured))
    published = sorted(
        name
        for name in sensitive
        if name in race.written or (layout.out / name).exists() or (layout.out / name).is_symlink()
    )
    assert code != 0, f"merge exited {code} after publishing symlink targets: {hits or published}"
    assert hits == [], hits
    assert published == [], published


@pytest.mark.parametrize("place", _PLACES)
def test_lookup_then_open_does_not_publish_replaced_target(tmp_path: Path, place: str) -> None:
    layout = _Layout(tmp_path)
    layout.build()
    source = _source_for(layout, place)
    secret = layout.outside / f"{place}.replacement"
    if place == "base_index":
        secret.write_bytes(_index_document(_MARKER))
    else:
        secret.write_bytes(_MARKER + b"\n" + place.encode())
    original = _original_bytes(layout, place)
    name = _output_name(place)
    race = _Race(source, secret, layout.out, replace=True)
    with _race_window(race):
        code = _invoke(layout)
    hits = _marker_hits(layout.out, bytes(race.captured))
    destination = layout.out / name
    assert hits == [], (
        f"replaced target was published (exit {code}, lookups {race.lookups}, "
        f"swapped {race.swapped}): {hits}"
    )
    if destination.is_symlink() or destination.exists():
        assert not destination.is_symlink()
        assert destination.is_file()
        assert destination.read_bytes() == original
    else:
        assert code != 0


def test_regular_companions_are_copied(tmp_path: Path) -> None:
    layout = _Layout(tmp_path)
    layout.build()
    assert _invoke(layout) == 0
    out = layout.out
    assert (out / "config.json").read_bytes() == layout.config_bytes
    assert (out / "tokenizer.json").read_bytes() == layout.tokenizer_bytes
    assert (out / "decision_head.json").read_bytes() == layout.decision_bytes
    assert (out / "calibration.json").read_bytes() == layout.calibration_bytes
    assert (out / "model.safetensors.index.json").read_bytes() == layout.index_bytes
    assert not (out / "config.json").is_symlink()
    assert not (out / "README.md").exists()
    assert not (out / "SOURCE_REVISION").exists()
    assert (out / _SHARD).is_file()
