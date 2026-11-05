"""Shared MS MARCO calibration cache must not become the calibration set.

These tests fail while the builders treat ``/tmp/ms_marco_v1.1_train.parquet`` as
authoritative. A local user who can create that path can supply the rows that are
rendered into the calibration JSONL. They are the spec for the fix: keep the
parquet somewhere another local user cannot write, and parse the bytes that were
checked against the pinned revision. A later read of the path is not that check.

The same default path and exists-then-parse sequence is in all four builders
below. Lightning's calibration scripts are not covered: they call
``hf_hub_download`` with an explicit revision and do not use this temp path.
"""

from __future__ import annotations

import builtins
import importlib.util
import io
import os
import sys
import urllib.request
from collections.abc import Callable, Iterator
from email.message import EmailMessage
from pathlib import Path
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
SHARED_CACHE = Path("/tmp/ms_marco_v1.1_train.parquet")
PINNED_URL = (
    "https://huggingface.co/datasets/microsoft/ms_marco/resolve/"
    "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a/v1.1/train-00000-of-00001.parquet"
)
POISON = "POISON_CALIB_ROW_278f"
BENIGN = "PINNED_REVISION_ROW_278f"
SAMPLE_COUNT = 512
BUILDERS = (
    pytest.param(
        "models/nemotron-3.5-embed-8b/calibration/build_calib.py",
        id="nemotron-embed",
    ),
    pytest.param(
        "models/nemotron-3.5-rerank-8b/calibration/build_calib.py",
        id="nemotron-rerank",
    ),
    pytest.param(
        "models/qwen3-embedding-8b/calibration/build_embed_calib.py",
        id="qwen-embed",
    ),
    pytest.param(
        "models/qwen3-reranker-4b/calibration/build_reranker_calib.py",
        id="qwen-rerank",
    ),
)


def _parquet_bytes(marker: str) -> bytes:
    passage = pa.struct(
        [
            ("is_selected", pa.list_(pa.int64())),
            ("passage_text", pa.list_(pa.string())),
            ("url", pa.list_(pa.string())),
        ]
    )
    table = pa.table(
        {
            "query_id": pa.array(list(range(SAMPLE_COUNT)), type=pa.int64()),
            "query": pa.array([f"{marker}-q-{i}" for i in range(SAMPLE_COUNT)]),
            "passages": pa.array(
                [
                    {
                        "is_selected": [1, 0],
                        "passage_text": [f"{marker}-p-{i}", f"{marker}-n-{i}"],
                        "url": [
                            "http://example.invalid/a",
                            "http://example.invalid/b",
                        ],
                    }
                    for i in range(SAMPLE_COUNT)
                ],
                type=passage,
            ),
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def _load(relative: str) -> ModuleType:
    path = ROOT / relative
    name = "calib_" + relative.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _invoke(module: ModuleType, out: Path) -> None:
    previous = sys.argv
    sys.argv = [Path(module.__file__ or "build_calib.py").name, "--out", str(out)]
    try:
        try:
            module.main()
        except (SystemExit, OSError):
            return
    finally:
        sys.argv = previous


class _FetchLog:
    def __init__(self) -> None:
        self.urls: list[str] = []
        self.destinations: list[Path] = []


def _request_url(url: object) -> str:
    if isinstance(url, urllib.request.Request):
        return url.full_url
    return str(url)


class _Body(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = EmailMessage()
        self.headers["Content-Length"] = str(len(payload))

    def info(self) -> EmailMessage:
        return self.headers

    def getcode(self) -> int:
        return 200

    def geturl(self) -> str:
        return PINNED_URL


def _install_fetch_stub(monkeypatch: pytest.MonkeyPatch, benign: bytes) -> _FetchLog:
    """Serve the pinned URL locally. Writing the body back through the cache path is the caller's choice."""
    log = _FetchLog()

    def urlopen(url: object, *args: object, **kwargs: object) -> _Body:
        log.urls.append(_request_url(url))
        return _Body(benign)

    def urlretrieve(
        url: str,
        filename: str | Path | None = None,
        reporthook: object = None,
        data: object = None,
    ) -> tuple[str, None]:
        del reporthook, data
        log.urls.append(_request_url(url))
        if filename is None:
            raise OSError("the pinned parquet download did not name a cache file")
        dest = Path(filename)
        log.destinations.append(dest)
        # CPython urlretrieve opens the destination with ``open(filename, "wb")``, which truncates.
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, benign)
        finally:
            os.close(fd)
        return str(dest), None

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(urllib.request, "urlretrieve", urlretrieve)
    return log


def _overwrite(path: Path, payload: bytes) -> None:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_TRUNC)
    except OSError:
        return
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _as_path(source: object) -> Path | None:
    if isinstance(source, (str, bytes, os.PathLike)):
        text = os.fsdecode(source) if isinstance(source, bytes) else os.fspath(source)
        return Path(text)
    return None


class _SwapOnClose:
    """After a read of a cache path returns, replace that path before the next open."""

    def __init__(self, handle: io.BufferedIOBase, path: Path, poison: bytes) -> None:
        self._handle = handle
        self._path = path
        self._poison = poison
        self._swapped = False

    def __getattr__(self, name: str) -> object:
        return getattr(self._handle, name)

    def __enter__(self) -> _SwapOnClose:
        self._handle.__enter__()
        return self

    def __exit__(self, *exc: object) -> object:
        try:
            return self._handle.__exit__(*exc)
        finally:
            self._swap()

    def close(self) -> None:
        try:
            self._handle.close()
        finally:
            self._swap()

    def _swap(self) -> None:
        if self._swapped:
            return
        self._swapped = True
        _overwrite(self._path, self._poison)


def _install_reopen_attack(poison: bytes, tracked: Callable[[], set[Path]]) -> Iterator[None]:
    original_open = builtins.open
    original_read_table = pq.read_table

    def open_hook(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
        handle = original_open(file, mode, *args, **kwargs)
        path = _as_path(file)
        writing = any(flag in mode for flag in ("w", "a", "x"))
        if path is None or writing or path not in tracked():
            return handle
        return _SwapOnClose(handle, path, poison)

    def read_table(source: object, *args: object, **kwargs: object) -> object:
        # The parser must consume the bytes already checked. A path here is a second lookup:
        # swap it to the attacker parquet before the real reader opens it.
        path = _as_path(source)
        if path is not None:
            _overwrite(path, poison)
        return original_read_table(source, *args, **kwargs)

    builtins.open = open_hook
    pq.read_table = read_table
    try:
        yield
    finally:
        builtins.open = original_open
        pq.read_table = original_read_table


def _assert_planted_cache_unused(out: Path, before: bytes, log: _FetchLog, label: str) -> None:
    assert SHARED_CACHE.exists()
    assert SHARED_CACHE.read_bytes() == before
    text = out.read_text(encoding="utf-8") if out.exists() else ""
    leaked = next((line for line in text.splitlines() if POISON in line), "")
    assert POISON not in text, f"{label} rendered attacker calibration text: {leaked}"
    assert log.urls, f"{label} skipped the pinned dataset download"
    assert all(url == PINNED_URL for url in log.urls), log.urls


@pytest.fixture
def planted_shared_cache() -> Iterator[bytes]:
    backup = SHARED_CACHE.read_bytes() if SHARED_CACHE.exists() else None
    backup_mode = SHARED_CACHE.stat().st_mode if SHARED_CACHE.exists() else None
    if SHARED_CACHE.exists():
        SHARED_CACHE.chmod(0o644)
    payload = _parquet_bytes(POISON)
    SHARED_CACHE.write_bytes(payload)
    SHARED_CACHE.chmod(0o444)
    before = SHARED_CACHE.read_bytes()
    try:
        yield before
    finally:
        if SHARED_CACHE.exists():
            SHARED_CACHE.chmod(0o644)
            SHARED_CACHE.unlink()
        if backup is not None:
            SHARED_CACHE.write_bytes(backup)
            if backup_mode is not None:
                SHARED_CACHE.chmod(backup_mode & 0o777)


@pytest.mark.parametrize("relative", BUILDERS)
def test_preexisting_shared_temp_parquet_is_not_calibration_input(
    relative: str,
    tmp_path: Path,
    planted_shared_cache: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default cache: an existing world-readable temp parquet is not fetched and is not parsed."""
    log = _install_fetch_stub(monkeypatch, _parquet_bytes(BENIGN))
    out = tmp_path / "calib.jsonl"
    _invoke(_load(relative), out)
    _assert_planted_cache_unused(out, planted_shared_cache, log, relative)


@pytest.mark.parametrize("relative", BUILDERS)
def test_reopening_the_cache_path_cannot_supply_calibration_rows(
    relative: str,
    tmp_path: Path,
    planted_shared_cache: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bytes parsed after a path check must be the checked bytes, not a later open of that path."""
    poison = _parquet_bytes(POISON)
    log = _install_fetch_stub(monkeypatch, _parquet_bytes(BENIGN))

    def tracked() -> set[Path]:
        return {SHARED_CACHE, *log.destinations}

    out = tmp_path / "calib.jsonl"
    attack = _install_reopen_attack(poison, tracked)
    next(attack)
    try:
        _invoke(_load(relative), out)
    finally:
        attack.close()
    _assert_planted_cache_unused(out, planted_shared_cache, log, relative)
