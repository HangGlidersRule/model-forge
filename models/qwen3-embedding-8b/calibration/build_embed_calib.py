#!/usr/bin/env python3
"""Build the deterministic Qwen3-Embedding-8B calibration set for ModelOpt hf_ptq (issue #18).

Source: microsoft/ms_marco v1.1 train, pinned to a dataset revision (the reranker calibration's source). 256 queries in the
model's own query format (instruction + query) and 256 passages as plain text, exactly as an embedding client sends them.
Output: JSONL, one {"text": ...} per line (ModelOpt 0.48's get_dataset_samples() reads the "text" column as-is).
Usage: build_embed_calib.py --out PATH [--cache PARQUET]; prints a summary line and the file's sha256.
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
N_QUERIES = 256
N_PASSAGES = 256
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def query_text(query: str) -> str:
    return f"Instruct: {INSTRUCTION}\nQuery:{query}"


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

    import pyarrow.parquet as pq

    data = load_parquet(args.cache)
    rows = pq.read_table(io.BytesIO(data), columns=["query_id", "query", "passages"]).to_pylist()
    rng = random.Random(SEED)
    rng.shuffle(rows)

    texts: list[str] = []
    for r in rows[:N_QUERIES]:
        texts.append(query_text(r["query"].strip()))
    passages_seen = 0
    for r in rows[N_QUERIES:]:
        cands = [p.strip() for p in (r["passages"] or {}).get("passage_text", []) if p and p.strip()]
        if not cands:
            continue
        texts.append(cands[rng.randrange(len(cands))])
        passages_seen += 1
        if passages_seen == N_PASSAGES:
            break
    order = list(range(len(texts)))
    rng.shuffle(order)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i in order:
            f.write(json.dumps({"text": texts[i]}, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    n_q = sum(1 for t in texts if t.startswith("Instruct: "))
    print(f"wrote {len(texts)} samples ({n_q} queries, {len(texts) - n_q} passages) to {args.out} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
