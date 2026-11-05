"""Shared temp parquet cache used by the ms_marco calibration builders.

The builders default to ``/tmp/ms_marco_v1.1_train.parquet`` and, when that path
already exists, parse it with no digest check. These tests are the spec for
closing that hole. They fail while a pre-existing file at the default cache
path is rendered into the calibration JSONL, and they pass when a run either
refuses that file or emits the published calibration for the pinned
``microsoft/ms_marco`` revision.

``after_stat`` replaces the cache directory entry once ``os.stat`` has
succeeded, which is the ``Path.exists`` check in front of ``read_table``.
``after_read`` replaces it only after a Python read has returned the bytes it
just observed, so a later open of the same path must not supply the samples.

Not covered here: ``models/nemotron-3.5-lightning-r1/calibration/build_calib.py``,
``build_kl_prompts.py``, and ``build_sel_prompts.py``. They download through
``hf_hub_download`` at a pinned revision and do not use this temp path.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import select
import shutil
import struct
import subprocess
import sys
import textwrap
import threading
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK_SOURCE = ROOT / "tests/support/calib_cache_sitecustomize.py"
PINNED_URL = (
    "https://huggingface.co/datasets/microsoft/ms_marco/resolve/"
    "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a/v1.1/train-00000-of-00001.parquet"
)
PINNED_PARQUET_SHA256 = "11d5e51d70ce87b47b1c132ef7c55f4cfb93b22c122058aa0591a65c4c0dc1f1"
MARKER_SEEDED = "ATTACKER_CALIB_MARKER_b26b18c5"
MARKER_SWAPPED = "SWAPPED_CALIB_MARKER_b26b18c5"
SAMPLE_ROWS = 512

# Published calibration JSONL digests for the pinned parquet. A zero exit is
# only acceptable when the output is that file; anything else was chosen by
# the cache contents.
BUILDERS = [
    pytest.param(
        "models/nemotron-3.5-rerank-8b/calibration/build_calib.py",
        "ddd481866f158e51aaa7c66a9353be6ba1317bb5f2b3a188b1b703b5b961576e",
        id="nemotron-rerank",
    ),
    pytest.param(
        "models/nemotron-3.5-embed-8b/calibration/build_calib.py",
        "a04f7ef04008eaa7c2a52e846109b9357481de87e65f447a8df1512f4ed44773",
        id="nemotron-embed",
    ),
    pytest.param(
        "models/qwen3-embedding-8b/calibration/build_embed_calib.py",
        "d643f77fecf055ca31e2da5829f86ba90aa925d151154f6ddfcb49446bbb11c6",
        id="qwen3-embed",
    ),
    pytest.param(
        "models/qwen3-reranker-4b/calibration/build_reranker_calib.py",
        "cada916c2205153cc32ee6a4b30e5a121f8960e6f68f8578130ffc6a47e95261",
        id="qwen3-reranker",
    ),
]

_IN_CREATE = 0x00000100
_IN_MOVED_TO = 0x00000080
_IN_CLOEXEC = 0x00080000
_EVENT_HEADER = struct.Struct("iIII")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_pairs(path: Path, marker: str) -> None:
    schema = pa.schema(
        [
            ("query_id", pa.int64()),
            ("query", pa.string()),
            (
                "passages",
                pa.struct(
                    [
                        ("passage_text", pa.list_(pa.string())),
                        ("is_selected", pa.list_(pa.int32())),
                    ]
                ),
            ),
        ]
    )
    table = pa.table(
        {
            "query_id": list(range(SAMPLE_ROWS)),
            "query": [f"{marker}-query-{index}" for index in range(SAMPLE_ROWS)],
            "passages": [
                {
                    "passage_text": [f"{marker}-pos-{index}", f"{marker}-neg-{index}"],
                    "is_selected": [1, 0],
                }
                for index in range(SAMPLE_ROWS)
            ],
        },
        schema=schema,
    )
    pq.write_table(table, path)


def _default_cache(script: Path) -> Path | None:
    """Return the ``--cache`` default without running the download or the parse."""
    probe = textwrap.dedent(
        """
        import argparse
        import importlib.util
        import os
        import sys

        target = sys.argv[1]
        found = []
        real_add = argparse.ArgumentParser.add_argument

        def wrapped_add(self, *args, **kwargs):
            if "--cache" in args:
                found.append(kwargs.get("default"))
            return real_add(self, *args, **kwargs)

        def wrapped_parse(self, *args, **kwargs):
            value = found[-1] if found else None
            sys.stdout.write("" if value is None else os.fspath(value))
            raise SystemExit(0)

        argparse.ArgumentParser.add_argument = wrapped_add
        argparse.ArgumentParser.parse_args = wrapped_parse
        spec = importlib.util.spec_from_file_location("calib_probe", target)
        if spec is None or spec.loader is None:
            raise SystemExit(2)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.argv = [target, "--out", "/tmp/calib-probe-must-not-write"]
        module.main()
        raise SystemExit(2)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe, str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    text = proc.stdout.strip()
    if not text:
        return None
    return Path(text)


def _snap(path: Path) -> tuple[int, int, int, int, str]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_mode, stat.st_size, _sha256(path))


class _CreateWatch:
    """Record names created in one directory while a builder runs."""

    def __init__(self, directory: Path) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.inotify_init1.argtypes = [ctypes.c_int]
        libc.inotify_init1.restype = ctypes.c_int
        libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        libc.inotify_add_watch.restype = ctypes.c_int
        self._fd = libc.inotify_init1(_IN_CLOEXEC)
        if self._fd < 0:
            err = ctypes.get_errno()
            raise OSError(err, "inotify_init1")
        watch = libc.inotify_add_watch(
            self._fd, os.fsencode(directory), _IN_CREATE | _IN_MOVED_TO
        )
        if watch < 0:
            err = ctypes.get_errno()
            os.close(self._fd)
            raise OSError(err, "inotify_add_watch")
        self.names: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select_ready(self._fd)
            if not ready:
                continue
            data = os.read(self._fd, 4096)
            offset = 0
            while offset + _EVENT_HEADER.size <= len(data):
                _wd, _mask, _cookie, name_len = _EVENT_HEADER.unpack_from(data, offset)
                start = offset + _EVENT_HEADER.size
                raw = data[start : start + name_len].split(b"\x00", 1)[0]
                if raw:
                    self.names.append(os.fsdecode(raw))
                offset = start + name_len

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        os.close(self._fd)


def select_ready(fd: int) -> tuple[list[int], list[int], list[int]]:
    return select.select([fd], [], [], 0.05)


@pytest.fixture(scope="module")
def pinned_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("pinned-msmarco") / "train.parquet"
    temporary = destination.with_suffix(".partial")
    urllib.request.urlretrieve(PINNED_URL, temporary)
    assert _sha256(temporary) == PINNED_PARQUET_SHA256
    temporary.replace(destination)
    return destination


@pytest.fixture(scope="module")
def attacker_pairs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    directory = tmp_path_factory.mktemp("attacker-pairs")
    seeded = directory / "seeded.parquet"
    swapped = directory / "swapped.parquet"
    _write_pairs(seeded, MARKER_SEEDED)
    _write_pairs(swapped, MARKER_SWAPPED)
    return {"seeded": seeded, "swapped": swapped}


def _plant(cache: Path, source: Path, *, link: bool) -> tuple[Path | None, tuple[int, int, int, int, str]]:
    backup: Path | None = None
    if cache.exists() or cache.is_symlink():
        backup = cache.with_name(cache.name + ".calib-test-backup")
        if backup.exists() or backup.is_symlink():
            raise RuntimeError(f"backup already present: {backup}")
        os.replace(cache, backup)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(source, cache)
        except OSError:
            shutil.copy2(source, cache)
    else:
        shutil.copy2(source, cache)
        os.chmod(cache, 0o644)
    return backup, _snap(cache)


def _restore(cache: Path, backup: Path | None) -> None:
    if cache.exists() or cache.is_symlink():
        cache.unlink()
    temporary = Path(str(cache) + ".hook-tmp")
    if temporary.exists():
        temporary.unlink()
    if backup is not None and (backup.exists() or backup.is_symlink()):
        os.replace(backup, cache)


def _run(
    script: Path,
    out: Path,
    *,
    hook: str | None,
    swap_src: Path | None,
    cache: Path | None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("CALIB_HOOK", None)
    env.pop("CALIB_CACHE_PATH", None)
    env.pop("CALIB_SWAP_SRC", None)
    if hook is not None and cache is not None and swap_src is not None:
        hook_dir = out.parent / "hook"
        hook_dir.mkdir()
        shutil.copyfile(HOOK_SOURCE, hook_dir / "sitecustomize.py")
        env["PYTHONPATH"] = str(hook_dir)
        env["CALIB_HOOK"] = hook
        env["CALIB_CACHE_PATH"] = str(cache)
        env["CALIB_SWAP_SRC"] = str(swap_src)
    watch = _CreateWatch(out.parent)
    try:
        return subprocess.run(
            [sys.executable, str(script), "--out", str(out)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
    finally:
        watch.close()
        out.parent.joinpath(".created").write_text("\n".join(watch.names), encoding="utf-8")


def _created_names(out: Path) -> list[str]:
    record = out.parent / ".created"
    if not record.exists():
        return []
    text = record.read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line]


def _assert_safe(
    proc: subprocess.CompletedProcess[str],
    out: Path,
    markers: list[str],
    expected_sha: str,
    cache: Path | None,
    before: tuple[int, int, int, int, str] | None,
) -> None:
    observed = proc.stdout + proc.stderr
    if out.exists():
        observed += out.read_text(encoding="utf-8")
    leaked = [marker for marker in markers if marker in observed]
    assert not leaked, (
        f"cache contents reached the calibration output: {leaked}\n"
        f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    )
    if proc.returncode != 0:
        assert out.name not in _created_names(out), (
            f"output was created before the builder refused: {_created_names(out)}"
        )
        assert not out.exists()
    else:
        assert out.is_file()
        got = _sha256(out)
        assert got == expected_sha, f"rc=0 sha256={got} expected={expected_sha}"
    if cache is not None and before is not None:
        assert cache.exists()
        assert _snap(cache) == before


def _builder(relative: str) -> tuple[Path, Path | None]:
    script = ROOT / relative
    cache = _default_cache(script)
    return script, cache


def _run_unplanted(script: Path, out: Path, expected_sha: str) -> None:
    proc = _run(script, out, hook=None, swap_src=None, cache=None)
    _assert_safe(proc, out, [MARKER_SEEDED, MARKER_SWAPPED], expected_sha, None, None)


@pytest.mark.parametrize(("relative", "expected_sha"), BUILDERS)
def test_preseeded_default_cache_is_not_rendered(
    relative: str,
    expected_sha: str,
    attacker_pairs: dict[str, Path],
    tmp_path: Path,
) -> None:
    script, cache = _builder(relative)
    out = tmp_path / "calib.jsonl"
    if cache is None:
        _run_unplanted(script, out, expected_sha)
        return
    backup, before = _plant(cache, attacker_pairs["seeded"], link=False)
    try:
        proc = _run(script, out, hook=None, swap_src=None, cache=cache)
        _assert_safe(proc, out, [MARKER_SEEDED], expected_sha, cache, before)
    finally:
        _restore(cache, backup)


@pytest.mark.parametrize(("relative", "expected_sha"), BUILDERS)
def test_existence_check_does_not_read_a_replaced_cache(
    relative: str,
    expected_sha: str,
    attacker_pairs: dict[str, Path],
    tmp_path: Path,
) -> None:
    script, cache = _builder(relative)
    out = tmp_path / "calib.jsonl"
    if cache is None:
        _run_unplanted(script, out, expected_sha)
        return
    backup, _before = _plant(cache, attacker_pairs["seeded"], link=False)
    try:
        proc = _run(
            script,
            out,
            hook="after_stat",
            swap_src=attacker_pairs["swapped"],
            cache=cache,
        )
        _assert_safe(
            proc,
            out,
            [MARKER_SEEDED, MARKER_SWAPPED],
            expected_sha,
            cache=None,
            before=None,
        )
    finally:
        _restore(cache, backup)


@pytest.mark.parametrize(("relative", "expected_sha"), BUILDERS)
def test_checked_bytes_are_the_bytes_parsed(
    relative: str,
    expected_sha: str,
    pinned_parquet: Path,
    attacker_pairs: dict[str, Path],
    tmp_path: Path,
) -> None:
    script, cache = _builder(relative)
    out = tmp_path / "calib.jsonl"
    if cache is None:
        _run_unplanted(script, out, expected_sha)
        return
    backup, _before = _plant(cache, pinned_parquet, link=True)
    try:
        proc = _run(
            script,
            out,
            hook="after_read",
            swap_src=attacker_pairs["swapped"],
            cache=cache,
        )
        _assert_safe(
            proc,
            out,
            [MARKER_SWAPPED],
            expected_sha,
            cache=None,
            before=None,
        )
    finally:
        _restore(cache, backup)
