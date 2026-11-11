"""Planted MS MARCO cache files must not become calibration text.

Four builders reuse ``/tmp/ms_marco_v1.1_train.parquet`` when that path already
exists and never bind the bytes to the pinned dataset revision:

- models/qwen3-reranker-4b/calibration/build_reranker_calib.py
- models/qwen3-embedding-8b/calibration/build_embed_calib.py
- models/nemotron-3.5-rerank-8b/calibration/build_calib.py
- models/nemotron-3.5-embed-8b/calibration/build_calib.py

These tests fail while that reuse is in place. A fix has to ignore a
pre-existing file at that shared path, including one replaced after the
existence check, and still must not write the planted query or passage strings
into the calibration JSONL.

Not the same cache: models/nemotron-3.5-lightning-r1/calibration/build_calib.py,
build_kl_prompts.py, build_sel_prompts.py (Hugging Face hub download) and
models/jev-27b/calibration/build_jev_calib.py (HTTP JSONL, no parquet cache).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

REPO_ROOT = Path(__file__).resolve().parents[1]
# Without sudo, the planted cache is created by the current user in a sticky, world-writable directory
# standing in for /tmp, and another user's ownership is simulated by HOOK_SOURCE: os.stat and os.lstat
# report FOREIGN_UID and os.unlink raises PermissionError, in this process and in the builder.
HOOK_SOURCE = REPO_ROOT / "tests/support/calib_cache_sitecustomize.py"
CACHE_NAME = "ms_marco_v1.1_train.parquet"
FOREIGN_UID = 65534
PINNED_URL = (
    "https://huggingface.co/datasets/microsoft/ms_marco/resolve/"
    "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a/v1.1/train-00000-of-00001.parquet"
)
PINNED_PARQUET_SHA256 = "11d5e51d70ce87b47b1c132ef7c55f4cfb93b22c122058aa0591a65c4c0dc1f1"
# Tests never download: builders are served this local copy of PINNED_URL.
PINNED_PARQUET_ENV = "MODEL_FORGE_TEST_MSMARCO_PARQUET"
MARKER_A_QUERY = "MFPLANTQUERYA"
MARKER_A_PASSAGE = "MFPLANTPASSAGEA"
MARKER_B_QUERY = "MFPLANTQUERYB"
MARKER_B_PASSAGE = "MFPLANTPASSAGEB"
MARKERS = (MARKER_A_QUERY, MARKER_A_PASSAGE, MARKER_B_QUERY, MARKER_B_PASSAGE)
N_ROWS = 512

BUILDERS = (
    "models/qwen3-reranker-4b/calibration/build_reranker_calib.py",
    "models/qwen3-embedding-8b/calibration/build_embed_calib.py",
    "models/nemotron-3.5-rerank-8b/calibration/build_calib.py",
    "models/nemotron-3.5-embed-8b/calibration/build_calib.py",
)

_OPEN_HOOK = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int watch_hits = 0;

static int is_watch(const char *path) {
    const char *watch = getenv("TOCTOU_WATCH");
    return watch && path && strcmp(path, watch) == 0;
}

static void wait_for_swap(void) {
    const char *waiting = getenv("TOCTOU_WAITING");
    const char *go = getenv("TOCTOU_GO");
    if (!waiting || !go) return;
    int fd = open(waiting, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd >= 0) close(fd);
    for (;;) {
        if (access(go, F_OK) == 0) return;
        usleep(1000);
    }
}

static int counted_open(const char *path, int flags, mode_t mode,
                        int (*real_open)(const char *, int, ...)) {
    if (is_watch(path) && (flags & O_ACCMODE) == O_RDONLY) {
        watch_hits++;
        if (watch_hits == 2) wait_for_swap();
    }
    return real_open(path, flags, mode);
}

int open(const char *path, int flags, ...) {
    static int (*real_open)(const char *, int, ...) = NULL;
    mode_t mode = 0;
    if (!real_open) real_open = dlsym(RTLD_NEXT, "open");
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
    }
    return counted_open(path, flags, mode, real_open);
}

int open64(const char *path, int flags, ...) {
    static int (*real_open64)(const char *, int, ...) = NULL;
    mode_t mode = 0;
    if (!real_open64) real_open64 = dlsym(RTLD_NEXT, "open64");
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
    }
    return counted_open(path, flags, mode, real_open64);
}
"""


def _load_hooks():
    spec = importlib.util.spec_from_file_location("calib_cache_hooks", HOOK_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pinned_parquet() -> Path:
    value = os.environ.get(PINNED_PARQUET_ENV)
    if not value or not Path(value).is_file():
        pytest.skip(f"set {PINNED_PARQUET_ENV} to a local copy of {PINNED_URL}; tests never download it")
    digest = hashlib.sha256(Path(value).read_bytes()).hexdigest()
    assert digest == PINNED_PARQUET_SHA256
    return Path(value)


@pytest.fixture
def shared_cache(
    tmp_path: Path, pinned_parquet: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    xdg = tmp_path / "xdg"
    shared = xdg / "model-forge"
    shared.mkdir(parents=True)
    shared.chmod(0o1777)
    cache = shared / CACHE_NAME
    hook_dir = tmp_path / "hook"
    hook_dir.mkdir()
    shutil.copyfile(HOOK_SOURCE, hook_dir / "sitecustomize.py")
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    monkeypatch.setenv("PYTHONPATH", str(hook_dir))
    monkeypatch.setenv("CALIB_OFFLINE_URL", PINNED_URL)
    monkeypatch.setenv("CALIB_OFFLINE_FILE", str(pinned_parquet))
    monkeypatch.setenv("CALIB_FOREIGN_PATH", str(cache))
    monkeypatch.setenv("CALIB_FOREIGN_UID", str(FOREIGN_UID))
    for name, hook in _load_hooks().foreign_owner_hooks(str(cache), FOREIGN_UID).items():
        monkeypatch.setattr(os, name, hook)
    yield cache


def _write_parquet(path: Path, query: str, passage: str) -> None:
    passage_type = pa.struct(
        [
            ("passage_text", pa.list_(pa.string())),
            ("is_selected", pa.list_(pa.int64())),
        ]
    )
    schema = pa.schema(
        [
            ("query_id", pa.int64()),
            ("query", pa.string()),
            ("passages", passage_type),
        ]
    )
    rows = [
        {
            "query_id": i,
            "query": f"{query}-{i}",
            "passages": {
                "passage_text": [f"{passage}-pos-{i}", f"{passage}-neg-{i}"],
                "is_selected": [1, 0],
            },
        }
        for i in range(N_ROWS)
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _plant_as_other_user(src: Path, dest: Path) -> None:
    """Put ``src`` at ``dest``, mode 644, in the sticky shared dir; ``shared_cache`` makes it nobody's."""
    public = dest.with_name(dest.name + ".ubuntu-src")
    public.write_bytes(src.read_bytes())
    public.chmod(0o644)
    os.replace(public, dest)


def _assert_shared_temp_preconditions(cache: Path) -> None:
    parent_mode = cache.parent.stat().st_mode
    assert stat.S_ISDIR(parent_mode)
    assert parent_mode & stat.S_ISVTX, "shared temp dir must be sticky"
    assert parent_mode & stat.S_IWOTH, "shared temp dir must be world-writable"
    file_stat = cache.stat()
    assert file_stat.st_uid != os.getuid()
    assert stat.S_IMODE(file_stat.st_mode) & stat.S_IROTH
    assert not (stat.S_IMODE(file_stat.st_mode) & stat.S_IWOTH)
    with pytest.raises(PermissionError):
        cache.unlink()


def _run_builder(
    script: str,
    out: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script), "--out", str(out)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _assert_output_has_no_planted_text(
    proc: subprocess.CompletedProcess[str],
    out: Path,
    script: str,
) -> None:
    text = out.read_text(encoding="utf-8") if out.exists() else ""
    found = [marker for marker in MARKERS if marker in text]
    assert not found, (
        f"{script} wrote planted cache text {found} into {out} "
        f"(exit {proc.returncode}). stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert proc.returncode == 0, (
        f"{script} exited {proc.returncode} without a calibration file from the pinned "
        f"dataset. stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == N_ROWS


def _compile_open_hook(dir_path: Path) -> Path:
    source = dir_path / "open_hook.c"
    library = dir_path / "open_hook.so"
    source.write_text(_OPEN_HOOK)
    proc = subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O2", "-o", str(library), str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr)
    return library


def _script_id(script: str) -> str:
    return Path(script).parent.parent.name


@pytest.mark.parametrize("script", BUILDERS, ids=_script_id)
def test_other_user_planted_default_cache_is_not_written_to_jsonl(
    script: str,
    tmp_path: Path,
    shared_cache: Path,
) -> None:
    planted = tmp_path / "planted-a.parquet"
    _write_parquet(planted, MARKER_A_QUERY, MARKER_A_PASSAGE)
    _plant_as_other_user(planted, shared_cache)
    _assert_shared_temp_preconditions(shared_cache)
    before = shared_cache.read_bytes()

    proc = _run_builder(script, tmp_path / "calib.jsonl")

    assert shared_cache.read_bytes() == before
    assert shared_cache.stat().st_uid != os.getuid()
    _assert_output_has_no_planted_text(proc, tmp_path / "calib.jsonl", script)


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("gcc") is None,
    reason="the open hook is an LD_PRELOAD library built with gcc",
)
@pytest.mark.parametrize("script", BUILDERS, ids=_script_id)
def test_reopened_default_cache_does_not_pick_up_replacement_bytes(
    script: str,
    tmp_path: Path,
    shared_cache: Path,
) -> None:
    """Swap the cache inode after the first read open and before the next one.

    ``Path.exists`` only stats the path. The parquet reader then opens that
    same path again. The second read open must not observe a different file
    than an earlier open of the path, and neither file's text may be written
    out. Replacing the directory entry between those opens is the attacker
    rewriting a cache they own.
    """
    planted_a = tmp_path / "planted-a.parquet"
    planted_b = tmp_path / "planted-b.parquet"
    _write_parquet(planted_a, MARKER_A_QUERY, MARKER_A_PASSAGE)
    _write_parquet(planted_b, MARKER_B_QUERY, MARKER_B_PASSAGE)
    _plant_as_other_user(planted_a, shared_cache)
    _assert_shared_temp_preconditions(shared_cache)

    library = _compile_open_hook(tmp_path)
    waiting = tmp_path / "second-open"
    go = tmp_path / "swap-done"
    env = {
        "LD_PRELOAD": str(library),
        "TOCTOU_WATCH": str(shared_cache),
        "TOCTOU_WAITING": str(waiting),
        "TOCTOU_GO": str(go),
    }
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / script), "--out", str(tmp_path / "calib.jsonl")],
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout = ""
    stderr = ""
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if waiting.exists() or proc.poll() is not None:
                break
            time.sleep(0.02)
        if waiting.exists() and proc.poll() is None:
            staged = shared_cache.with_name(shared_cache.name + ".swap")
            _plant_as_other_user(planted_b, staged)
            os.replace(staged, shared_cache)
            go.write_text("swapped")
        try:
            stdout, stderr = proc.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            if not go.exists():
                go.write_text("release")
            proc.kill()
            stdout, stderr = proc.communicate(timeout=10)
    finally:
        if not go.exists():
            go.write_text("release")
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    return_code = proc.returncode if proc.returncode is not None else 1
    completed = subprocess.CompletedProcess(proc.args, return_code, stdout, stderr)
    _assert_output_has_no_planted_text(completed, tmp_path / "calib.jsonl", script)
