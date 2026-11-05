#!/usr/bin/env python3
"""Build the deterministic Qwen3-Reranker-4B calibration set for ModelOpt hf_ptq.

Source: microsoft/ms_marco v1.1 train, pinned to a dataset revision. Each sample is one
query-passage pair rendered exactly the way the reranker is served: the Qwen3-Reranker
judge prompt with the default web-search instruction, ending in the assistant turn with
an empty think block. Mixed relevant and irrelevant pairs, so activations cover both
"yes" and "no" decisions.

Output: a JSONL file with one {"text": ...} object per line, which ModelOpt 0.48's
get_dataset_samples() reads as a local dataset (the "text" column is used as-is).
Run with --out <path>; prints a summary line and the file's sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import stat
import sys
import tempfile
import urllib.request
from pathlib import Path

DATASET = "microsoft/ms_marco"
REVISION = "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a"
PARQUET = "v1.1/train-00000-of-00001.parquet"
# Hugging Face LFS sha256 of PARQUET at REVISION; any cached or downloaded bytes must match it.
PARQUET_SHA256 = "11d5e51d70ce87b47b1c132ef7c55f4cfb93b22c122058aa0591a65c4c0dc1f1"
SEED = 1234
N_SAMPLES = 512

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def render(query: str, passage: str) -> str:
    body = f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {passage}"
    return PREFIX + body + SUFFIX


def default_cache() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "model-forge" / "ms_marco" / REVISION / Path(PARQUET).name


def read_verified(path: Path) -> bytes | None:
    """The file's bytes from a single open, or None unless they hash to PARQUET_SHA256."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            return None
        data = f.read()
    return data if hashlib.sha256(data).hexdigest() == PARQUET_SHA256 else None


def load_parquet(cache: Path) -> bytes:
    data = read_verified(cache)
    if data is not None:
        return data
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{PARQUET}"
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PARQUET_SHA256:
        raise SystemExit(f"{url} has sha256 {digest}, expected {PARQUET_SHA256}")
    try:
        cache.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cache.parent, prefix=f".{cache.name}.")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, cache)
        except BaseException:
            os.unlink(tmp)
            raise
    except OSError as exc:
        print(f"not caching {cache}: {exc}", file=sys.stderr)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=default_cache())
    args = ap.parse_args()

    import pyarrow.parquet as pq  # imported late so --help works without pyarrow

    data = load_parquet(args.cache)
    table = pq.read_table(io.BytesIO(data), columns=["query_id", "query", "passages"])
    rows = table.to_pylist()

    rng = random.Random(SEED)
    rng.shuffle(rows)
    samples: list[dict[str, object]] = []
    n_pos = n_neg = 0
    for row in rows:
        p = row["passages"]
        texts, selected = p["passage_text"], p["is_selected"]
        pos = [t for t, s in zip(texts, selected) if s == 1]
        neg = [t for t, s in zip(texts, selected) if s == 0]
        # Alternate relevant / irrelevant so the set is balanced.
        want_pos = len(samples) % 2 == 0
        pool = pos if want_pos else neg
        if not pool:
            continue
        passage = rng.choice(pool)
        samples.append({"text": render(row["query"], passage)})
        n_pos += want_pos
        n_neg += not want_pos
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
    lens = sorted(len(s["text"]) for s in samples)
    print(
        f"CALIB-OK samples={len(samples)} relevant={n_pos} irrelevant={n_neg} "
        f"chars_min={lens[0]} chars_p50={lens[len(lens) // 2]} chars_max={lens[-1]} "
        f"sha256={digest} source={DATASET}@{REVISION}/{PARQUET} seed={SEED}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
