#!/usr/bin/env python3
"""Deterministic Nemotron-3.5-Rerank-8B calibration set for ModelOpt (issue #32).

Source: microsoft/ms_marco v1.1 train, pinned to a dataset revision (the same source and revision as the
Qwen3-Reranker-4B set in issue #14). Each sample is one query-passage pair rendered exactly the way this checkpoint's
chat template renders a text rerank pair: "query: <q>\n\npassage: <p>", with '[' and '<' escaped by a zero-width
non-joiner as the template does, and no BOS/EOS (the template omits them). Relevant and irrelevant pairs alternate, so
activations cover both decisions.

Output: JSONL, one {"text": ...} per line. Prints CALIB-OK with counts and the file's sha256.
Usage: build_calib.py --out PATH [--cache PARQUET]"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

DATASET = "microsoft/ms_marco"
REVISION = "a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a"
PARQUET = "v1.1/train-00000-of-00001.parquet"
SEED = 1234
N_SAMPLES = 512
ZWNJ = "\u200c"


def escape(text: str) -> str:
    """The template's text_content filter: replace('[', '[\u200c') | replace('<', '<\u200c')."""
    return text.replace("[", "[" + ZWNJ).replace("<", "<" + ZWNJ)


def render(query: str, passage: str) -> str:
    return "query: " + escape(query) + "\n\npassage: " + escape(passage)


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
    samples: list[dict[str, object]] = []
    n_pos = n_neg = 0
    for row in rows:
        p = row["passages"]
        texts, selected = p["passage_text"], p["is_selected"]
        pos = [t for t, s in zip(texts, selected) if s == 1]
        neg = [t for t, s in zip(texts, selected) if s == 0]
        want_pos = len(samples) % 2 == 0
        pool = pos if want_pos else neg
        if not pool:
            continue
        samples.append({"text": render(row["query"], rng.choice(pool))})
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
    lens = sorted(len(str(s["text"])) for s in samples)
    print(f"CALIB-OK samples={len(samples)} relevant={n_pos} irrelevant={n_neg} chars_min={lens[0]} "
          f"chars_p50={lens[len(lens) // 2]} chars_max={lens[-1]} sha256={digest} source={DATASET}@{REVISION}/{PARQUET} seed={SEED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
