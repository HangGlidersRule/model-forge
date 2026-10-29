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
import json
import random
import urllib.request
from pathlib import Path

DATASET = "microsoft/ms_marco"
REVISION = "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a"
PARQUET = "v1.1/train-00000-of-00001.parquet"
SEED = 1234
N_QUERIES = 256
N_PASSAGES = 256
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def query_text(query: str) -> str:
    return f"Instruct: {INSTRUCTION}\nQuery:{query}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=Path("/tmp/ms_marco_v1.1_train.parquet"))
    args = ap.parse_args()

    import pyarrow.parquet as pq

    if not args.cache.exists():
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{PARQUET}"
        urllib.request.urlretrieve(url, args.cache)
    rows = pq.read_table(args.cache, columns=["query_id", "query", "passages"]).to_pylist()
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
