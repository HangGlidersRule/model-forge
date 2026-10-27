#!/usr/bin/env python3
"""Reranker quality + throughput gate for the issue-14 NVFP4 candidates.

Talks to one running vLLM pooling server (Qwen3-Reranker, /v1/score) and writes a JSON
result. Run once per variant (BF16 control, BF16 replica, candidates A/B/C) against the
same frozen eval set; compare.py then computes paired deltas and the noise floor.

Eval set (built once, deterministic, hashed):
  MTEB Nano retrieval sets (NFCorpus, FiQA2018, Touche2020, ArguAna, SciFact, MSMARCO),
  pinned by dataset revision. Per query: every judged-relevant doc plus BM25-free random
  negatives from the same corpus, up to 30 candidates, seed 1234. Documents are NOT
  truncated, so long-document pairs (Touche2020 p90 ~6.5k chars) are covered.

Metrics per dataset: security@hangglidersrule.com, security@hangglidersrule.com, plus per-pair scores so compare.py can compute
Spearman / top-10 overlap against BF16 and the BF16 replica-vs-replica floor.
Full denominator: every pair must get a score; HTTP errors are retried 5x, then the run
fails (never silently dropped).

Throughput: long pairs (~1.1k tokens each), 64 pairs per request, 4 x concurrency requests per
run at concurrency 1/4/8, 3 measured runs after 1 warmup; reports pairs/s.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASETS = {
    "mteb/NanoNFCorpusRetrieval": "2eec9d63140f",
    "mteb/NanoFiQA2018Retrieval": "ede84da4793c",
    "mteb/NanoTouche2020Retrieval": "ff81c0caff8b",
    "mteb/NanoArguAnaRetrieval": "63ac9f026e8f",
    "mteb/NanoSciFactRetrieval": "8c5f7bbe90bb",
    "mteb/NanoMSMARCORetrieval": "5a5aa0c3df65",
}
SEED = 1234
MAX_CANDS = 30
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "model-forge-eval"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def build_eval_set(out: Path) -> None:
    import pyarrow.parquet as pq

    rng = random.Random(SEED)
    data: dict[str, list[dict]] = {}
    shas: dict[str, str] = {}
    for ds, prefix in DATASETS.items():
        info = json.loads(_get(f"https://huggingface.co/api/datasets/{ds}"))
        sha = info["sha"]
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
        doc_ids = sorted(docs)
        items = []
        for qid in sorted(rel):
            if qid not in qtext:
                continue
            pos = sorted(d for d in rel[qid] if d in docs)[:MAX_CANDS // 2]
            pool = [d for d in doc_ids if d not in rel[qid]]
            neg = rng.sample(pool, MAX_CANDS - len(pos))
            cands = pos + neg
            rng.shuffle(cands)
            items.append({"qid": qid, "query": qtext[qid],
                          "cands": [{"id": d, "text": docs[d], "rel": int(d in rel[qid])} for d in cands]})
        data[ds] = items
    blob = {"seed": SEED, "max_cands": MAX_CANDS, "dataset_shas": shas, "data": data}
    out.write_text(json.dumps(blob, ensure_ascii=False))
    n_pairs = sum(len(q["cands"]) for items in data.values() for q in items)
    print(f"EVALSET queries={sum(len(v) for v in data.values())} pairs={n_pairs} "
          f"sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")


def score_batch(base: str, model: str, query: str, docs: list[str]) -> list[float]:
    body = json.dumps({"model": model, "text_1": query, "text_2": docs}).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request(f"{base}/v1/score", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.loads(r.read())
            scores = sorted(out["data"], key=lambda x: x["index"])
            if len(scores) != len(docs):
                raise RuntimeError(f"got {len(scores)} scores for {len(docs)} docs")
            return [float(s["score"]) for s in scores]
        except Exception as e:  # noqa: BLE001 - retried, then fatal
            if attempt == 4:
                raise RuntimeError(f"score failed after 5 attempts: {e}") from e
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def quality(base: str, model: str, evalset: dict) -> dict:
    res: dict = {}
    for ds, items in evalset["data"].items():
        per_q = []
        for q in items:
            docs = [c["text"] for c in q["cands"]]
            scores = []
            for i in range(0, len(docs), 16):
                scores += score_batch(base, model, q["query"], docs[i:i + 16])
            order = sorted(range(len(docs)), key=lambda i: -scores[i])
            rels = [q["cands"][i]["rel"] for i in order]
            ideal = sorted((c["rel"] for c in q["cands"]), reverse=True)
            ndcg = dcg(rels[:10]) / dcg(ideal[:10]) if dcg(ideal[:10]) else 0.0
            rr = next((1.0 / (k + 1) for k, r in enumerate(rels[:10]) if r), 0.0)
            per_q.append({"qid": q["qid"], "ndcg10": ndcg, "mrr10": rr, "scores": scores})
        res[ds] = {"n_queries": len(per_q), "n_pairs": sum(len(p["scores"]) for p in per_q),
                   "ndcg10": sum(p["ndcg10"] for p in per_q) / len(per_q),
                   "mrr10": sum(p["mrr10"] for p in per_q) / len(per_q), "per_query": per_q}
        print(f"  {ds:32s} security@hangglidersrule.com={res[ds]['ndcg10']:.4f} security@hangglidersrule.com={res[ds]['mrr10']:.4f} "
              f"({res[ds]['n_queries']} q, {res[ds]['n_pairs']} pairs)", flush=True)
    return res


def throughput(base: str, model: str, evalset: dict) -> dict:
    # Unique long documents (no prefix-cache reuse across requests): longest docs first.
    pool = sorted({c["text"] for items in evalset["data"].values() for q in items for c in q["cands"]},
                  key=len, reverse=True)
    pool = [d[:4500] for d in pool if len(d) >= 3000][:2000]
    rng = random.Random(SEED)
    out: dict = {"docs_used": len(pool), "doc_chars": [len(pool[0]), len(pool[-1])]}

    def one(i: int) -> int:
        docs = [pool[(i * 64 + k) % len(pool)] for k in range(64)]
        q = f"query {i} {rng.random():.6f}: which passage best answers this question?"
        score_batch(base, model, q, docs)
        return sum(len(d) for d in docs)

    for conc in (1, 4, 8):
        runs = []
        for rep in range(4):  # first is warmup
            n_req = 4 * conc
            t0 = time.perf_counter()
            with ThreadPoolExecutor(conc) as ex:
                chars = sum(ex.map(one, range(rep * 1000 + n_req * 7, rep * 1000 + n_req * 7 + n_req)))
            dt = time.perf_counter() - t0
            if rep:
                runs.append({"requests": n_req, "pairs": n_req * 64, "chars": chars, "seconds": dt})
        out[f"c{conc}"] = runs
        print(f"  c={conc}: " + ", ".join(f"{r['pairs'] / r['seconds']:.1f} pairs/s" for r in runs), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--base", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--evalset", type=Path, required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--skip-throughput", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        build_eval_set(a.out)
        return 0
    evalset = json.loads(a.evalset.read_text())
    print(f"RUN {a.label} evalset_sha256={hashlib.sha256(a.evalset.read_bytes()).hexdigest()}", flush=True)
    res = {"label": a.label, "evalset_sha256": hashlib.sha256(a.evalset.read_bytes()).hexdigest(),
           "quality": quality(a.base, a.model, evalset)}
    if not a.skip_throughput:
        res["throughput"] = throughput(a.base, a.model, evalset)
    a.out.write_text(json.dumps(res))
    print(f"RESULT-WRITTEN {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
