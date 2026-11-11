#!/usr/bin/env python3
"""Build the deterministic Qwen3-Embedding-8B calibration set for ModelOpt hf_ptq (issue #18).

Source: microsoft/ms_marco v1.1 train, pinned to a dataset revision (the reranker calibration's source). 256 queries in the
model's own query format (instruction + query) and 256 passages as plain text, exactly as an embedding client sends them.
Output: JSONL, one {"text": ...} per line (ModelOpt 0.48's get_dataset_samples() reads the "text" column as-is).
Usage: build_embed_calib.py --out PATH [--cache PARQUET]; prints a summary line and the file's sha256.
The parquet must hash to PARQUET_SHA256; the default cache is under the user's private ~/.cache/model-forge.
"""
from __future__ import annotations

import argparse
import contextlib
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
PARQUET_SHA256 = "11d5e51d70ce87b47b1c132ef7c55f4cfb93b22c122058aa0591a65c4c0dc1f1"
SEED = 1234
N_QUERIES = 256
N_PASSAGES = 256
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def query_text(query: str) -> str:
    return f"Instruct: {INSTRUCTION}\nQuery:{query}"


def default_cache() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "model-forge"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root / "ms_marco_v1.1_train.parquet"


def store_cache(cache: Path, data: bytes) -> None:
    """Publish verified bytes without opening, truncating or following anything already at the cache path."""
    fd, tmp = tempfile.mkstemp(dir=cache.parent, prefix=f".{cache.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.link(tmp, cache, follow_symlinks=False)
    except FileExistsError:
        print(f"warning: {cache} appeared during the download; left as is", file=sys.stderr)
    finally:
        os.unlink(tmp)


def pinned_parquet(cache: Path, *, explicit: bool) -> bytes:
    """The pinned parquet's bytes, read once: the bytes that are hashed are the bytes that are parsed.

    An explicit --cache that does not hold the pinned bytes is refused. Anything else at the default
    cache path is left as is, and the pinned bytes are downloaded instead.
    """
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{PARQUET}"
    existed = cache.exists()
    if existed:
        with os.fdopen(os.open(cache, os.O_RDONLY | os.O_NONBLOCK), "rb") as f:
            if stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                data = f.read()
                digest = hashlib.sha256(data).hexdigest()
                if digest == PARQUET_SHA256:
                    return data
                problem = f"sha256 {digest} is not the pinned {PARQUET_SHA256}"
            else:
                problem = "not a regular file"
        if explicit:
            raise SystemExit(f"{cache}: {problem}")
        print(f"warning: ignoring {cache} ({problem}); downloading the pinned parquet", file=sys.stderr)
    with contextlib.closing(urllib.request.urlopen(url, timeout=600)) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PARQUET_SHA256:
        raise SystemExit(f"{url}: sha256 {digest} is not the pinned {PARQUET_SHA256}")
    if not existed:
        store_cache(cache, data)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", type=Path, help="pinned parquet cache (default: ~/.cache/model-forge/)")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    data = pinned_parquet(args.cache or default_cache(), explicit=args.cache is not None)
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
