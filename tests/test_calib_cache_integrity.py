"""Calibration builders must not trust a shared cache path.

Each builder defaults to the same cache file. These tests require three
properties of that cache:

- ``PARQUET_SHA256`` on the builder module is the hex sha256 of the pinned
  parquet bytes. A cache whose bytes do not match is refused, and refusal
  does not create, truncate, or delete the output or the cache.
- The bytes that match the pin are the bytes that are calibrated. Hashing
  one open and parsing a later open of the same path is not enough: the
  path is replaced after the first successful open.
- When the cache is missing at the existence check, the download must not
  follow a symlink that is already at that path. Stat and lstat keep
  reporting the path as absent, so a second lookup does not reveal it.
  The symlink and its target are left untouched.
- With ``--cache`` omitted, the cache path that is checked lives under a
  directory owned by the current user and not writable by group or other.
"""

from __future__ import annotations

import errno
import hashlib
import importlib.util
import os
import posix
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDERS = (
    REPO / "models/qwen3-embedding-8b/calibration/build_embed_calib.py",
    REPO / "models/qwen3-reranker-4b/calibration/build_reranker_calib.py",
    REPO / "models/nemotron-3.5-rerank-8b/calibration/build_calib.py",
)
PINNED_MARKER = "PINNED_ROW_MARKER"
UNTRUSTED_MARKER = "UNTRUSTED_ROW_MARKER"
SENTINEL = b"OUTPUT-SENTINEL-DO-NOT-TOUCH\n"
VICTIM_BYTES = b"VICTIM-FILE-ORIGINAL-BYTES\n"
DOWNLOAD_BODY = b"downloaded-body-not-parquet"
ROW_COUNT = 512
_LOADS = 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_corpus(path: Path, marker: str) -> Path:
    passage = pa.struct(
        [
            ("is_selected", pa.list_(pa.int32())),
            ("passage_text", pa.list_(pa.string())),
        ]
    )
    table = pa.table(
        {
            "query_id": pa.array(list(range(ROW_COUNT)), type=pa.int64()),
            "query": pa.array([f"{marker} query {i}" for i in range(ROW_COUNT)]),
            "passages": pa.array(
                [
                    {
                        "is_selected": [1, 0],
                        "passage_text": [f"{marker} pos {i}", f"{marker} neg {i}"],
                    }
                    for i in range(ROW_COUNT)
                ],
                type=passage,
            ),
        }
    )
    pq.write_table(table, path)
    return path


def _load_builder(script: Path):
    global _LOADS
    _LOADS += 1
    name = f"calib_builder_{_LOADS}"
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _call_main(module: object) -> int:
    main = getattr(module, "main")
    try:
        return int(main())
    except SystemExit as exc:
        code = exc.code
        if code is None or code == 0:
            return 0
        if isinstance(code, int):
            return code
        return 1


def _run_child(mode: str, extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), mode],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_private_directory(path: Path) -> None:
    current = path.parent
    while not current.exists():
        parent = current.parent
        if parent == current:
            break
        current = parent
    info = current.stat()
    mode = stat.S_IMODE(info.st_mode)
    assert info.st_uid == os.getuid(), (
        f"cache {path} is under {current}, owned by uid {info.st_uid}"
    )
    assert mode & 0o022 == 0, f"cache {path} is under group/other-writable {current} mode {oct(mode)}"


@pytest.fixture(params=BUILDERS, ids=["embed", "reranker", "nemotron"])
def builder(request: pytest.FixtureRequest) -> Path:
    return request.param


def test_cache_must_match_pinned_sha256_and_those_bytes_are_calibrated(
    builder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a cache whose sha256 is not ``PARQUET_SHA256``, then calibrate the file that matches it."""
    good = _write_corpus(tmp_path / "good.parquet", PINNED_MARKER)
    bad = _write_corpus(tmp_path / "bad.parquet", UNTRUSTED_MARKER)
    module = _load_builder(builder)
    monkeypatch.setattr(module, "PARQUET_SHA256", _sha256(good), raising=False)

    out_bad = tmp_path / "bad.jsonl"
    out_bad.write_bytes(SENTINEL)
    bad_bytes = bad.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [str(builder), "--out", str(out_bad), "--cache", str(bad)],
    )
    assert _call_main(module) != 0
    assert out_bad.read_bytes() == SENTINEL
    assert bad.read_bytes() == bad_bytes

    out_good = tmp_path / "good.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(builder), "--out", str(out_good), "--cache", str(good)],
    )
    assert _call_main(module) == 0
    text = out_good.read_text(encoding="utf-8")
    assert PINNED_MARKER in text
    assert UNTRUSTED_MARKER not in text


def test_second_open_of_cache_path_is_not_the_authorized_corpus(
    builder: Path, tmp_path: Path
) -> None:
    """The parquet bytes opened for the pin check are the bytes written to the jsonl."""
    good = _write_corpus(tmp_path / "good.parquet", PINNED_MARKER)
    bad = _write_corpus(tmp_path / "bad.parquet", UNTRUSTED_MARKER)
    cache = tmp_path / "cache.parquet"
    cache.write_bytes(good.read_bytes())
    out = tmp_path / "out.jsonl"
    out.write_bytes(SENTINEL)
    library = tmp_path / "swapopen.so"
    subprocess.run(
        [
            "gcc",
            "-shared",
            "-fPIC",
            "-o",
            str(library),
            str(Path(__file__).resolve().parent / "calib_cache_open_swap.c"),
            "-ldl",
        ],
        check=True,
    )
    proc = _run_child(
        "--swap-child",
        {
            "LD_PRELOAD": str(library),
            "WATCH_PATH": str(cache),
            "REPLACEMENT_PATH": str(bad),
            "CALIB_SCRIPT": str(builder),
            "CALIB_CACHE": str(cache),
            "CALIB_OUT": str(out),
            "CALIB_SHA": _sha256(good),
        },
    )
    text = out.read_text(encoding="utf-8") if out.exists() else ""
    assert proc.returncode == 0, proc.stderr
    assert PINNED_MARKER in text
    assert UNTRUSTED_MARKER not in text


def test_download_does_not_follow_symlink_hidden_from_stat(
    builder: Path, tmp_path: Path
) -> None:
    """A symlink at the cache path must not be followed when stat says the path is absent."""
    victim = tmp_path / "victim"
    victim.write_bytes(VICTIM_BYTES)
    cache = tmp_path / "cache.parquet"
    cache.symlink_to(victim)
    out = tmp_path / "out.jsonl"
    out.write_bytes(SENTINEL)
    proc = _run_child(
        "--symlink-child",
        {
            "CALIB_SCRIPT": str(builder),
            "CALIB_CACHE": str(cache),
            "CALIB_OUT": str(out),
        },
    )
    assert victim.read_bytes() == VICTIM_BYTES, proc.stderr
    assert cache.is_symlink()
    assert Path(os.readlink(cache)) == victim
    assert out.read_bytes() == SENTINEL
    assert proc.returncode != 0


def test_default_cache_path_is_user_private(builder: Path, tmp_path: Path) -> None:
    """The default cache check must not use a shared writable directory."""
    shared = Path("/tmp/ms_marco_v1.1_train.parquet")
    shared_existed = shared.exists()
    out = tmp_path / "out.jsonl"
    proc = _run_child(
        "--default-cache-child",
        {"CALIB_SCRIPT": str(builder), "CALIB_OUT": str(out)},
    )
    if not shared_existed:
        assert not shared.exists()
    assert proc.returncode == 86, proc.stderr
    cache_lines = [line for line in proc.stdout.splitlines() if line.startswith("CACHE_PATH=")]
    assert cache_lines, proc.stderr
    _assert_private_directory(Path(cache_lines[-1].split("=", 1)[1]))


class _DownloadBody:
    """Response object consumed by urllib.request.urlretrieve (read, info, close)."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def info(self) -> dict[str, str]:
        return {}

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def close(self) -> None:
        return None


def _urlopen(*_args: object, **_kwargs: object) -> _DownloadBody:
    return _DownloadBody(DOWNLOAD_BODY)


def _hide_lookup(cache: str, real):
    def wrapped(path: object, *args: object, **kwargs: object):
        try:
            shown = os.fspath(path)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            shown = None
        if shown == cache:
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", cache)
        return real(path, *args, **kwargs)

    return wrapped


def _swap_child() -> None:
    script = Path(os.environ["CALIB_SCRIPT"])
    module = _load_builder(script)
    setattr(module, "PARQUET_SHA256", os.environ["CALIB_SHA"])
    sys.argv = [
        str(script),
        "--out",
        os.environ["CALIB_OUT"],
        "--cache",
        os.environ["CALIB_CACHE"],
    ]
    raise SystemExit(_call_main(module))


def _symlink_child() -> None:
    cache = os.environ["CALIB_CACHE"]
    real_stat = os.stat
    real_lstat = os.lstat
    os.stat = _hide_lookup(cache, real_stat)
    os.lstat = _hide_lookup(cache, real_lstat)
    posix.stat = os.stat
    posix.lstat = os.lstat
    urllib.request.urlopen = _urlopen  # type: ignore[assignment]
    script = os.environ["CALIB_SCRIPT"]
    sys.argv = [script, "--out", os.environ["CALIB_OUT"], "--cache", cache]
    raise SystemExit(_call_main(_load_builder(Path(script))))


def _default_cache_child() -> None:
    real_exists = Path.exists

    def exists(self: Path, *args: object, **kwargs: object) -> bool:
        if Path(self).suffix == ".parquet":
            sys.stdout.write(f"CACHE_PATH={Path(self)}\n")
            raise SystemExit(86)
        return real_exists(self, *args, **kwargs)

    Path.exists = exists  # type: ignore[method-assign]
    script = os.environ["CALIB_SCRIPT"]
    sys.argv = [script, "--out", os.environ["CALIB_OUT"]]
    run_module = _load_builder(Path(script))
    raise SystemExit(_call_main(run_module))


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "--swap-child":
        _swap_child()
    elif command == "--symlink-child":
        _symlink_child()
    elif command == "--default-cache-child":
        _default_cache_child()
    else:
        raise SystemExit(f"unknown child command {command}")
