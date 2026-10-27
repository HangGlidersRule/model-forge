#!/usr/bin/env python3
"""Rebuild the reranker eval set with BM25 hard negatives (replaces random negatives).

Random negatives from the Nano corpora left the BF16 control near the nDCG ceiling
(0.98-1.00 on five of six sets), so the gate could not separate candidates. Here each
query's candidate list is every judged-relevant doc (capped at 15) plus the
highest-ranked BM25 non-relevant docs from the same corpus, 30 candidates total, in a
seeded shuffle. BM25: Okapi, k1=1.5, b=0.75, lowercase \\w+ tokens, title + text.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reranker_eval import DATASETS, MAX_CANDS, SEED  # noqa: E402

TOKEN = re.compile(r"\w+")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "model-forge-eval"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.ids = sorted(docs)
        self.tf = [Counter(tokens(docs[i])) for i in self.ids]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def rank(self, query: str) -> list[str]:
        q = [t for t in tokens(query) if t in self.idf]
        scores = []
        for i, (tf, dl) in enumerate(zip(self.tf, self.len)):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            scores.append((s, self.ids[i]))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [d for _, d in scores]


def main() -> int:
    import pyarrow.parquet as pq

    out = Path(sys.argv[1])
    rng = random.Random(SEED)
    data: dict[str, list[dict]] = {}
    shas: dict[str, str] = {}
    stats = []
    for ds, prefix in DATASETS.items():
        sha = json.loads(_get(f"https://huggingface.co/api/datasets/{ds}"))["sha"]
        assert sha.startswith(prefix), (ds, sha)
        shas[ds] = sha
        base = f"https://huggingface.co/datasets/{ds}/resolve/{sha}"
        corpus = pq.read_table(io.BytesIO(_get(f"{base}/corpus/train-00000-of-00001.parquet"))).to_pylist()
        queries = pq.read_table(io.BytesIO(_get(f"{base}/queries/train-00000-of-00001.parquet"))).to_pylist()
        qrels = pq.read_table(io.BytesIO(_get(f"{base}/qrels/train-00000-of-00001.parquet"))).to_pylist()
        idk = "_id" if "_id" in corpus[0] else "id"
        docs = {c[idk]: ((c.get("title") or "").strip() + "\n" + (c.get("text") or "").strip()).strip() for c in corpus}
        qtext = {q[idk]: q["text"] for q in queries}
        rel: dict[str, set[str]] = {}
        for r in qrels:
            if r["score"] > 0:
                rel.setdefault(r["query-id"], set()).add(r["corpus-id"])
        bm = BM25(docs)
        items = []
        hit = 0
        for qid in sorted(rel):
            if qid not in qtext:
                continue
            ranked = bm.rank(qtext[qid])
            pos = sorted(d for d in rel[qid] if d in docs)[:MAX_CANDS // 2]
            neg = [d for d in ranked if d not in rel[qid]][:MAX_CANDS - len(pos)]
            hit += sum(1 for d in ranked[:30] if d in rel[qid])
            cands = pos + neg
            rng.shuffle(cands)
            items.append({"qid": qid, "query": qtext[qid],
                          "cands": [{"id": d, "text": docs[d], "rel": int(d in rel[qid])} for d in cands]})
        data[ds] = items
        stats.append(f"{ds.split('/')[1]}: {len(items)} q, BM25 top-30 holds {hit} relevant")
    blob = {"seed": SEED, "max_cands": MAX_CANDS, "negatives": "bm25_top_nonrelevant",
            "dataset_shas": shas, "data": data}
    out.write_text(json.dumps(blob, ensure_ascii=False))
    n_pairs = sum(len(q["cands"]) for items in data.values() for q in items)
    print("\n".join(stats))
    print(f"EVALSET queries={sum(len(v) for v in data.values())} pairs={n_pairs} "
          f"sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
