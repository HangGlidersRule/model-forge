#!/usr/bin/env python3
"""Deterministic Nemotron-3.5-Embed-8B calibration set for ModelOpt (issue #33).

Source: microsoft/ms_marco v1.1 train at the pinned revision used by #14, #18 and #32. 256 queries and 256 passages,
interleaved, each rendered exactly the way this checkpoint's chat template renders a text input (system message = the
retrieval prefix, user message = the text): "query: <q>" or "passage: <p>", with '[' and '<' escaped by a zero-width
non-joiner as the template does, and no BOS/EOS (the template omits them). An embedding client sends both kinds, so
calibration covers both.

Output: JSONL, one {"text": ...} per line. Prints CALIB-OK with counts and the file's sha256.
Usage: build_calib.py --out PATH [--cache PARQUET]"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import tempfile
import urllib.request
from pathlib import Path

DATASET = "microsoft/ms_marco"
REVISION = "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a"
PARQUET = "v1.1/train-00000-of-00001.parquet"
# Git LFS object of PARQUET at REVISION.
PARQUET_SHA256 = "11d5e51d70ce87b47b1c132ef7c55f4cfb93b22c122058aa0591a65c4c0dc1f1"
PARQUET_SIZE = 175452226
SEED = 1234
N_SAMPLES = 512
ZWNJ = "\u200c"


def escape(text: str) -> str:
    """The template's mark_user_text macro: replace('[', '[\u200c') | replace('<', '<\u200c')."""
    return text.replace("[", "[" + ZWNJ).replace("<", "<" + ZWNJ)


def render(prefix: str, text: str) -> str:
    return prefix + " " + escape(text)


def default_cache() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "model-forge" / "ms_marco" / REVISION / Path(PARQUET).name


def pinned(data: bytes) -> bool:
    return len(data) == PARQUET_SIZE and hashlib.sha256(data).hexdigest() == PARQUET_SHA256


def read_cache(cache: Path) -> bytes | None:
    try:
        fd = os.open(cache, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as f:
        if os.fstat(f.fileno()).st_size != PARQUET_SIZE:
            return None
        return f.read(PARQUET_SIZE + 1)


def write_cache(cache: Path, data: bytes) -> None:
    try:
        cache.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cache.parent, prefix=f".{cache.name}.")
    except OSError as e:
        print(f"not caching {cache}: {e}", file=sys.stderr)
        return
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, cache)
    except OSError as e:
        os.unlink(tmp)
        print(f"not caching {cache}: {e}", file=sys.stderr)


def load_parquet(cache: Path) -> bytes:
    """The pinned parquet's bytes. Callers parse these bytes, never the cache path: any copy of the file can change after
    it is checked."""
    data = read_cache(cache)
    if data is not None and pinned(data):
        return data
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{PARQUET}"
    with urllib.request.urlopen(url) as resp:
        data = resp.read(PARQUET_SIZE + 1)
    if not pinned(data):
        raise SystemExit(f"{url}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}; "
                         f"expected {PARQUET_SIZE} bytes, sha256 {PARQUET_SHA256}")
    write_cache(cache, data)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=default_cache())
    args = ap.parse_args()

    import pyarrow.parquet as pq

    data = load_parquet(args.cache)
    rows = pq.read_table(io.BytesIO(data), columns=["query_id", "query", "passages"]).to_pylist()
    rng = random.Random(SEED)
    rng.shuffle(rows)
    samples: list[dict[str, object]] = []
    n_q = n_p = 0
    for row in rows:
        texts = row["passages"]["passage_text"]
        if not texts:
            continue
        if len(samples) % 2 == 0:
            samples.append({"text": render("query:", row["query"])})
            n_q += 1
        else:
            samples.append({"text": render("passage:", rng.choice(texts))})
            n_p += 1
        if len(samples) == N_SAMPLES:
            break
    if len(samples) != N_SAMPLES:
        print(f"only {len(samples)} samples built", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    lens = sorted(len(str(s["text"])) for s in samples)
    print(f"CALIB-OK samples={len(samples)} queries={n_q} passages={n_p} chars_min={lens[0]} chars_p50={lens[len(lens) // 2]} "
          f"chars_max={lens[-1]} sha256={digest} source={DATASET}@{REVISION}/{PARQUET} seed={SEED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
