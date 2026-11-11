"""Artifact builders must not publish the target of a symlink in their input.

The tests fail while a symlink to an outside file is materialized into the
output, including when that symlink replaces a regular file after the metadata
check and before the link or copy. They pass when the builder refuses that
input before it creates or deletes an output file.

Same directory walk, left untested because the copy runs only after a full
model load or export this suite does not execute:
scripts/qwen3_8/apply_qwen38_abliteration.py,
models/nemotron-3.5-rerank-8b/scripts/quantize_nr8.py,
models/nemotron-3.5-embed-8b/scripts/quantize_ne8.py.

Run: pytest tests/test_symlink_publication.py
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
if not hasattr(os, "O_PATH") or not os.path.isdir("/proc/self/fd"):
    pytest.skip(
        "the builders bind their inputs through Linux O_PATH and /proc/self/fd",
        allow_module_level=True,
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS = Path(__file__).resolve().parent
_TRACER = _TESTS / "symlink_use_tracer.py"
_BUILDER = _TESTS / "symlink_fixture_builder.py"
_VARIANTS = REPO_ROOT / "models/nemotron-3.5-lightning-r1/eval/make_variants.py"
_SPLIT = REPO_ROOT / "models/nemotron-3.5-lightning-r1/eval/make_split.py"
_SPLIT_MSE = REPO_ROOT / "models/nemotron-3.5-lightning-r1/eval/split_mse.py"
_MERGE = REPO_ROOT / "models/jev-27b/scripts/merge_lora.py"

_OUTSIDE = b"placeholder-outside-file\n"
_BENIGN = b"benign-variant-asset\n"
_LINK_NAME = "tokenizer.json"
_RULES = ("mse", "p9999", "p999", "b12xeq")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _build(command: str, *paths: Path) -> None:
    proc = _run([sys.executable, str(_BUILDER), command, *(str(path) for path in paths)])
    assert proc.returncode == 0, proc.stderr


def _seed_output(root: Path) -> None:
    nested = root / "mse" / _LINK_NAME
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"sentinel-do-not-replace\n")
    (root / "keep.txt").write_bytes(b"keep\n")


def _snapshot(root: Path) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    if not root.exists():
        return tuple(rows)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        rel_dir = current.relative_to(root).as_posix()
        rows.append((rel_dir, "dir", current.lstat().st_nlink))
        linked_dirs = sorted(name for name in dirnames if (current / name).is_symlink())
        dirnames[:] = sorted(name for name in dirnames if name not in linked_dirs)
        for name in linked_dirs:
            path = current / name
            rel = path.relative_to(root).as_posix()
            rows.append((rel, "symlink-dir", os.readlink(path), path.lstat().st_nlink))
        for name in sorted(filenames):
            path = current / name
            rel = path.relative_to(root).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                rows.append((rel, "symlink", os.readlink(path), info.st_nlink, info.st_ino))
            elif stat.S_ISREG(info.st_mode):
                rows.append((rel, "file", info.st_nlink, info.st_ino, path.read_bytes()))
            else:
                rows.append((rel, "other", info.st_mode, info.st_nlink))
    return tuple(rows)


def _publications(root: Path, outside: Path, payload: bytes) -> list[str]:
    if not root.exists():
        return []
    outside_ino = outside.lstat().st_ino
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        linked_dirs = [name for name in dirnames if (current / name).is_symlink()]
        dirnames[:] = [name for name in dirnames if name not in linked_dirs]
        entries = [current / name for name in filenames + linked_dirs]
        for path in entries:
            rel = path.relative_to(root).as_posix()
            info = path.lstat()
            if info.st_ino == outside_ino:
                found.append(f"{rel} is the outside inode")
            if stat.S_ISLNK(info.st_mode) or stat.S_ISREG(info.st_mode):
                try:
                    followed = path.stat()
                    data = path.read_bytes()
                except OSError as exc:
                    found.append(f"{rel} could not be read: {exc}")
                    continue
                if followed.st_ino == outside_ino:
                    found.append(f"{rel} follows to the outside file")
                if data == payload:
                    found.append(f"{rel} contains the outside file bytes")
    return found


def _place_symlink(base: Path, outside: Path) -> Path:
    link = base / _LINK_NAME
    link.symlink_to(outside)
    return link


def _place_regular(base: Path) -> Path:
    path = base / _LINK_NAME
    path.write_bytes(_BENIGN)
    return path


def _assert_refused(
    proc: subprocess.CompletedProcess[str],
    output: Path,
    output_before: tuple[tuple[object, ...], ...],
    source_root: Path,
    source_before: tuple[tuple[object, ...], ...],
    outside: Path,
) -> None:
    leaks = _publications(output, outside, _OUTSIDE)
    assert not leaks, "output published the outside file:\n" + "\n".join(leaks)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert _snapshot(output) == output_before
    assert _snapshot(source_root) == source_before
    assert outside.read_bytes() == _OUTSIDE
    assert outside.stat().st_nlink == 1


def _assert_swap_not_published(
    proc: subprocess.CompletedProcess[str],
    output: Path,
    outside: Path,
    source: Path,
    preserved: Path,
) -> None:
    leaks = _publications(output, outside, _OUTSIDE)
    detail = proc.stdout + proc.stderr
    assert not leaks, "swapped symlink was published:\n" + "\n".join(leaks) + "\n" + detail
    assert outside.stat().st_nlink == 1
    assert outside.read_bytes() == _OUTSIDE
    if source.is_symlink():
        assert source.lstat().st_nlink == 1
    assert preserved.read_bytes() == _BENIGN
    assert preserved.stat().st_nlink >= 1
    preserved_ino = preserved.stat().st_ino
    if not output.exists():
        return
    for dirpath, dirnames, filenames in os.walk(output, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        for name in filenames:
            if name != _LINK_NAME:
                continue
            published = current / name
            assert not published.is_symlink(), published
            assert published.lstat().st_ino == preserved_ino, detail
            assert published.read_bytes() == _BENIGN


def _trace(
    source: Path, outside: Path, cmd: list[str], *, when: str = "use"
) -> subprocess.CompletedProcess[str]:
    if platform.machine() != "x86_64":
        # Elsewhere PTRACE_GETREGS fails, nothing is swapped, and the case would pass vacuously.
        pytest.skip("symlink_use_tracer.py decodes x86_64 Linux syscalls only")
    return _run(
        [sys.executable, str(_TRACER), "--when", when, str(source), str(outside), "--", *cmd]
    )


# sitecustomize body: before bind()'s open of NAME through its directory fd, run ACTION on
# ``target`` (the absolute BASE path of NAME), once. The lstat that bind() made has already passed.
_BIND_OPEN_HOOK = (
    "import os\n"
    "_REAL = os.open\n"
    "_DONE = []\n"
    "def _open(path, flags, mode=0o777, *, dir_fd=None):\n"
    "    if not _DONE and dir_fd is not None and os.fsdecode(path) == {name!r}:\n"
    "        _DONE.append(True)\n"
    "        target = os.path.join(os.readlink('/proc/self/fd/%d' % dir_fd), {name!r})\n"
    "        {action}\n"
    "    if dir_fd is None:\n"
    "        return _REAL(path, flags, mode)\n"
    "    return _REAL(path, flags, mode, dir_fd=dir_fd)\n"
    "os.open = _open\n"
)


def _run_hooked(cmd: list[str], hook_dir: Path, sitecustomize: str) -> subprocess.CompletedProcess[str]:
    hook_dir.mkdir()
    (hook_dir / "sitecustomize.py").write_text(sitecustomize, encoding="utf-8")
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(hook_dir)},
    )


_IN_OPEN = 0x00000020
_LIBC = ctypes.CDLL(None, use_errno=True)


def _watch_opens(path: Path) -> int:
    """inotify fd reporting every open of ``path`` (through any name or link)."""
    fd = _LIBC.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1")
    if _LIBC.inotify_add_watch(fd, os.fsencode(path), _IN_OPEN) < 0:
        os.close(fd)
        raise OSError(ctypes.get_errno(), "inotify_add_watch")
    return fd


def _drain_opens(fd: int) -> bool:
    try:
        return bool(os.read(fd, 4096))
    except BlockingIOError:
        return False


def test_make_variants_refuses_symlink_without_publishing(tmp_path: Path) -> None:
    base = tmp_path / "base"
    stats = tmp_path / "stats.json"
    _build("variants", base, stats)
    outside = tmp_path / "outside"
    outside.write_bytes(_OUTSIDE)
    (base / "config.json").write_bytes(_BENIGN)
    link = _place_symlink(base, outside)
    output = tmp_path / "variants"
    _seed_output(output)
    output_before = _snapshot(output)
    source_before = _snapshot(base)
    proc = _run([sys.executable, str(_VARIANTS), str(stats), str(base), str(output)])
    _assert_refused(proc, output, output_before, base, source_before, outside)
    assert link.is_symlink()
    assert link.lstat().st_nlink == 1


def test_make_variants_link_does_not_follow_a_swapped_symlink(tmp_path: Path) -> None:
    base = tmp_path / "base"
    stats = tmp_path / "stats.json"
    _build("variants", base, stats)
    outside = tmp_path / "outside"
    outside.write_bytes(_OUTSIDE)
    (base / "config.json").write_bytes(_BENIGN)
    source = _place_regular(base)
    preserved = tmp_path / "authorized-inode"
    os.link(source, preserved)
    output = tmp_path / "variants"
    proc = _trace(
        source,
        outside,
        [sys.executable, str(_VARIANTS), str(stats), str(base), str(output)],
    )
    _assert_swap_not_published(proc, output, outside, source, preserved)


def test_make_variants_links_the_bound_inode_not_the_path(tmp_path: Path) -> None:
    """Swap at the first link call, after every BASE file was opened and checked.

    The default tracer swaps at the first publishing syscall on the path, which is
    the bind step's open; this one waits for the first link, so a link that names
    the BASE path again, instead of the descriptor bind() holds, publishes the
    outside file.
    """
    base = tmp_path / "base"
    stats = tmp_path / "stats.json"
    _build("variants", base, stats)
    outside = tmp_path / "outside"
    outside.write_bytes(_OUTSIDE)
    (base / "config.json").write_bytes(_BENIGN)
    source = _place_regular(base)
    preserved = tmp_path / "authorized-inode"
    os.link(source, preserved)
    output = tmp_path / "variants"
    proc = _trace(
        source,
        outside,
        [sys.executable, str(_VARIANTS), str(stats), str(base), str(output)],
        when="link",
    )
    assert source.is_symlink(), "the tracer did not reach a link call"
    _assert_swap_not_published(proc, output, outside, source, preserved)


def test_make_variants_refuses_a_file_replaced_between_its_check_and_open(tmp_path: Path) -> None:
    """bind() must compare the inode it opened with the one it checked.

    A regular file replaced by another regular file between the lstat and the
    O_NOFOLLOW open is not a link, so only the identity check catches it.
    """
    base = tmp_path / "base"
    stats = tmp_path / "stats.json"
    _build("variants", base, stats)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(_OUTSIDE)
    (base / _LINK_NAME).write_bytes(_BENIGN)
    output = tmp_path / "variants"
    proc = _run_hooked(
        [sys.executable, str(_VARIANTS), str(stats), str(base), str(output)],
        tmp_path / "hook",
        _BIND_OPEN_HOOK.format(name=_LINK_NAME, action=f"os.replace({str(replacement)!r}, target)"),
    )
    assert not replacement.exists(), "the replacement hook did not run"
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "changed while it was opened" in proc.stderr
    assert not output.exists() or not list(output.rglob(_LINK_NAME))


def test_make_variants_does_not_open_a_link_swapped_in_after_the_check(tmp_path: Path) -> None:
    """bind() opens with O_NOFOLLOW, so a link swapped in after the lstat is refused unopened.

    The identity check alone would also refuse it, but only after opening the link's
    target, which may be a device or a FIFO outside BASE.
    """
    base = tmp_path / "base"
    stats = tmp_path / "stats.json"
    _build("variants", base, stats)
    outside = tmp_path / "outside"
    outside.write_bytes(_OUTSIDE)
    source = _place_regular(base)
    output = tmp_path / "variants"
    watch = _watch_opens(outside)
    try:
        proc = _run_hooked(
            [sys.executable, str(_VARIANTS), str(stats), str(base), str(output)],
            tmp_path / "hook",
            _BIND_OPEN_HOOK.format(
                name=_LINK_NAME, action=f"os.unlink(target); os.symlink({str(outside)!r}, target)"
            ),
        )
        opened = _drain_opens(watch)
    finally:
        os.close(watch)
    assert source.is_symlink(), "the swap hook did not run"
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not opened, "make_variants opened the target of a link swapped into BASE"
    assert not _publications(output, outside, _OUTSIDE)


def test_make_variants_reads_the_bound_scale_shard_not_the_path(tmp_path: Path) -> None:
    """The scale shard is read through the descriptor bound before any variant is written.

    The scale path is swapped for a link to another shard just before it is read; a
    re-open by name would put the other shard's scales into every variant.
    """
    base = tmp_path / "base"
    stats = tmp_path / "stats.json"
    _build("variants", base, stats)
    (base / "config.json").write_bytes(_BENIGN)
    scale = base / "model-input-scales.safetensors"
    other = tmp_path / "other-scales.safetensors"
    from safetensors.torch import load_file, save_file

    original = load_file(str(scale))
    save_file({name: tensor + 7.0 for name, tensor in original.items()}, str(other))
    output = tmp_path / "variants"
    proc = _run_hooked(
        [sys.executable, str(_VARIANTS), str(stats), str(base), str(output)],
        tmp_path / "hook",
        "import os\n"
        "import safetensors\n"
        "_REAL = safetensors.safe_open\n"
        "_DONE = []\n"
        "def _safe_open(*args, **kwargs):\n"
        "    if not _DONE:\n"
        "        _DONE.append(True)\n"
        f"        os.unlink({str(scale)!r})\n"
        f"        os.symlink({str(other)!r}, {str(scale)!r})\n"
        "    return _REAL(*args, **kwargs)\n"
        "safetensors.safe_open = _safe_open\n",
    )
    assert scale.is_symlink(), "the swap hook did not run"
    assert proc.returncode == 0, proc.stdout + proc.stderr
    recorded = json.loads((output / "variants-G.json").read_text())["A"]
    expected = [float(t.float().reshape(-1)[0]) * 2688.0 for t in original.values()]
    assert sorted(recorded.values()) == sorted(expected), (recorded, expected)


def test_make_variants_hardlinks_regular_files(tmp_path: Path) -> None:
    base = tmp_path / "base"
    stats = tmp_path / "stats.json"
    _build("variants", base, stats)
    regular = base / "config.json"
    regular.write_bytes(_BENIGN)
    output = tmp_path / "variants"
    proc = _run([sys.executable, str(_VARIANTS), str(stats), str(base), str(output)])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert regular.stat().st_nlink == 1 + len(_RULES)
    scale = base / "model-input-scales.safetensors"
    for rule in _RULES:
        copied = output / rule / "config.json"
        assert copied.lstat().st_ino == regular.stat().st_ino
        assert not copied.is_symlink()
        assert copied.read_bytes() == _BENIGN
        rewritten = output / rule / "model-input-scales.safetensors"
        assert rewritten.is_file()
        assert rewritten.lstat().st_ino != scale.stat().st_ino


def test_make_split_refuses_symlink_without_publishing(tmp_path: Path) -> None:
    export = tmp_path / "export"
    _build("split", export)
    outside = tmp_path / "outside"
    outside.write_bytes(_OUTSIDE)
    (export / "config.json").write_bytes(_BENIGN)
    _place_symlink(export, outside)
    output = tmp_path / "out"
    _seed_output(output)
    output_before = _snapshot(output)
    source_before = _snapshot(export)
    proc = _run([sys.executable, str(_SPLIT), str(export), str(output)])
    _assert_refused(proc, output, output_before, export, source_before, outside)


def test_make_split_copy_does_not_follow_a_swapped_symlink(tmp_path: Path) -> None:
    export = tmp_path / "export"
    _build("split", export)
    outside = tmp_path / "outside"
    outside.write_bytes(_OUTSIDE)
    (export / "config.json").write_bytes(_BENIGN)
    source = _place_regular(export)
    preserved = tmp_path / "authorized-inode"
    os.link(source, preserved)
    output = tmp_path / "out"
    proc = _trace(source, outside, [sys.executable, str(_SPLIT), str(export), str(output)])
    _assert_swap_not_published(proc, output, outside, source, preserved)


def test_make_split_copies_regular_files(tmp_path: Path) -> None:
    export = tmp_path / "export"
    _build("split", export)
    regular = export / "config.json"
    regular.write_bytes(_BENIGN)
    output = tmp_path / "out"
    proc = _run([sys.executable, str(_SPLIT), str(export), str(output)])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    copied = output / "config.json"
    assert copied.is_file()
    assert not copied.is_symlink()
    assert copied.read_bytes() == _BENIGN
    assert copied.lstat().st_ino != regular.stat().st_ino


def test_split_mse_refuses_symlink_without_publishing(tmp_path: Path) -> None:
    export = tmp_path / "export"
    gpath = tmp_path / "variants-G.json"
    _build("split-mse", export, gpath)
    outside = tmp_path / "outside"
    outside.write_bytes(_OUTSIDE)
    (export / "config.json").write_bytes(_BENIGN)
    _place_symlink(export, outside)
    output = tmp_path / "out"
    _seed_output(output)
    output_before = _snapshot(output)
    source_before = _snapshot(export)
    proc = _run([sys.executable, str(_SPLIT_MSE), str(export), str(gpath), str(output)])
    _assert_refused(proc, output, output_before, export, source_before, outside)


def test_split_mse_copy_does_not_follow_a_swapped_symlink(tmp_path: Path) -> None:
    export = tmp_path / "export"
    gpath = tmp_path / "variants-G.json"
    _build("split-mse", export, gpath)
    outside = tmp_path / "outside"
    outside.write_bytes(_OUTSIDE)
    (export / "config.json").write_bytes(_BENIGN)
    source = _place_regular(export)
    preserved = tmp_path / "authorized-inode"
    os.link(source, preserved)
    output = tmp_path / "out"
    proc = _trace(
        source,
        outside,
        [sys.executable, str(_SPLIT_MSE), str(export), str(gpath), str(output)],
    )
    _assert_swap_not_published(proc, output, outside, source, preserved)


def test_split_mse_copies_regular_files(tmp_path: Path) -> None:
    export = tmp_path / "export"
    gpath = tmp_path / "variants-G.json"
    _build("split-mse", export, gpath)
    regular = export / "config.json"
    regular.write_bytes(_BENIGN)
    output = tmp_path / "out"
    proc = _run([sys.executable, str(_SPLIT_MSE), str(export), str(gpath), str(output)])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    copied = output / "config.json"
    assert copied.is_file()
    assert not copied.is_symlink()
    assert copied.read_bytes() == _BENIGN


def test_merge_lora_refuses_symlink_without_publishing(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    _build("merge", base, adapter)
    outside = tmp_path / "outside"
    outside.write_bytes(_OUTSIDE)
    (base / "config.json").write_bytes(_BENIGN)
    _place_symlink(base, outside)
    output = tmp_path / "out"
    _seed_output(output)
    output_before = _snapshot(output)
    source_before = _snapshot(base)
    proc = _run(
        [
            sys.executable,
            str(_MERGE),
            "--base",
            str(base),
            "--adapter",
            str(adapter),
            "--out",
            str(output),
        ]
    )
    _assert_refused(proc, output, output_before, base, source_before, outside)


def test_merge_lora_copy_does_not_follow_a_swapped_symlink(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    _build("merge", base, adapter)
    outside = tmp_path / "outside"
    outside.write_bytes(_OUTSIDE)
    (base / "config.json").write_bytes(_BENIGN)
    source = _place_regular(base)
    preserved = tmp_path / "authorized-inode"
    os.link(source, preserved)
    output = tmp_path / "out"
    proc = _trace(
        source,
        outside,
        [
            sys.executable,
            str(_MERGE),
            "--base",
            str(base),
            "--adapter",
            str(adapter),
            "--out",
            str(output),
        ],
    )
    _assert_swap_not_published(proc, output, outside, source, preserved)


def test_merge_lora_copies_regular_files(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    _build("merge", base, adapter)
    regular = base / "config.json"
    regular.write_bytes(_BENIGN)
    output = tmp_path / "out"
    proc = _run(
        [
            sys.executable,
            str(_MERGE),
            "--base",
            str(base),
            "--adapter",
            str(adapter),
            "--out",
            str(output),
        ]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    copied = output / "config.json"
    assert copied.is_file()
    assert not copied.is_symlink()
    assert copied.read_bytes() == _BENIGN


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
