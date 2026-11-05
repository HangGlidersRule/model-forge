#!/usr/bin/env python3
"""Reranker quality + throughput gate for the issue-14 NVFP4 candidates.

Talks to one running vLLM pooling server (Qwen3-Reranker, /v1/score) and writes a JSON
result. Run once per variant (BF16 control, BF16 replica, candidates A/B/C) against the
same frozen eval set; compare.py then computes paired deltas and the noise floor.

Eval set (built once, deterministic, hashed):
  MTEB Nano retrieval sets (NFCorpus, FiQA2018, Touche2020, ArguAna, SciFact, MSMARCO),
    pinned by full dataset commit id and per-file sha256. Per query: every judged-relevant doc plus BM25-free random
  negatives from the same corpus, up to 30 candidates, seed 1234. Documents are NOT
  truncated, so long-document pairs (Touche2020 p90 ~6.5k chars) are covered.

Metrics per dataset: nDCG@10, MRR@10, plus per-pair scores so compare.py can compute
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

DATASETS = {  # full commit ids; the default branch is never consulted
    "mteb/NanoNFCorpusRetrieval": "2eec9d63140fb9af7c75ad0fdbf6350aaf20e94e",
    "mteb/NanoFiQA2018Retrieval": "ede84da4793cd2d4e61bba0dd08fc58de90f3681",
    "mteb/NanoTouche2020Retrieval": "ff81c0caff8b6febe1495eae9f2ca3eddad074b8",
    "mteb/NanoArguAnaRetrieval": "63ac9f026e8f6b85dc0622f3d630ed7a1758e9ec",
    "mteb/NanoSciFactRetrieval": "8c5f7bbe90bb865f46c67710221afa7d9096251b",
    "mteb/NanoMSMARCORetrieval": "5a5aa0c3df65d6883d85c8f5cdc88e679633595e",
}
SPLIT_SHA256 = {  # sha256 of <split>/train-00000-of-00001.parquet at the DATASETS commit
    "mteb/NanoNFCorpusRetrieval": {
        "corpus": "ddf19f78134632a6bbb41dd51d10f37b3d6002f389b5ae126647d792aeba7b5f",
        "queries": "9f13814121a1ab749fc2ee1d98ba3102916667e17f83cd278a4912085a970c3f",
        "qrels": "c606262a9276b6a08b70a39ec5baf9a2496c49b4277ba799b8c556548ff18758",
    },
    "mteb/NanoFiQA2018Retrieval": {
        "corpus": "c376e81f887649631f87f51f98f056306dd8d62a3c2dd7a58b5f1abe0bbcc1d9",
        "queries": "8a1f9e1d9c1dbca1c34f001e4f8ccc8c9d3d2f16254d091cc9edf4a5eeb305e8",
        "qrels": "f8828bb444d59dc3cf6fc841479f5e256b74fd209d60e3e21f9a60d22e759097",
    },
    "mteb/NanoTouche2020Retrieval": {
        "corpus": "50b93bd0a13daaf4a36ac20c7eaf84ecabc481670159fd421d01e84ef2bcbcec",
        "queries": "a3ba3a91061712e4212fb69417dfe451f29e47e039c3dd515a9fa70178ef25fd",
        "qrels": "e739d866d9d3e2a48da1eb78965d12b2655956acf18ed16c1f99977092c52768",
    },
    "mteb/NanoArguAnaRetrieval": {
        "corpus": "d0782a8feabeb9713f7a37a681adbad2efb0faf4b660f14fe5cf1ac690d67342",
        "queries": "5020ba08255bc32c333801f4ed3524ddf0e58b331ad9155a8548bff4f3486cdd",
        "qrels": "8f6558ec54c04f1fb85b77d97a1a3ef39a56a49fc407b58f8e515efc09ed8e43",
    },
    "mteb/NanoSciFactRetrieval": {
        "corpus": "d639056670650a4079ee73cbc28c58210bd54f4c081b1ca18580c652ed829d05",
        "queries": "16459964749438ba42fbc8a3436897eaed9d4b43f639b62348c41a395daa7a5d",
        "qrels": "043645fb0d7f520dccd0f370417a9a2fb2b46d95c530ea176921184d432ca662",
    },
    "mteb/NanoMSMARCORetrieval": {
        "corpus": "a717d6b419aa9ac8b63da38bfa9700a9c35bba8a6aa8452db0f9939126782957",
        "queries": "6efd9ab4664792379e7403a29aeb647ea315bd87b0d32ec2be87cc59a5fbcb76",
        "qrels": "97a696750a4d9a9d1d9741e90c66a9bb47bed8e0a689554d0c2eab84e6c02a8a",
    },
}
SEED = 1234
MAX_CANDS = 30
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "model-forge-eval"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def load_pinned(ds: str) -> tuple[list[dict], list[dict], list[dict]]:
    """corpus, queries, qrels of `ds` at its DATASETS commit; refuses any file whose sha256 is not pinned."""
    import pyarrow.parquet as pq

    sha = DATASETS[ds]
    tables = []
    for split in ("corpus", "queries", "qrels"):
        raw = _get(f"https://huggingface.co/datasets/{ds}/resolve/{sha}/{split}/train-00000-of-00001.parquet")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != SPLIT_SHA256[ds][split]:
            raise RuntimeError(f"{ds}@{sha} {split}: sha256 {digest} != pinned {SPLIT_SHA256[ds][split]}")
        tables.append(pq.read_table(io.BytesIO(raw)).to_pylist())
    corpus, queries, qrels = tables
    return corpus, queries, qrels


def build_eval_set(out: Path) -> None:
    rng = random.Random(SEED)
    data: dict[str, list[dict]] = {}
    shas: dict[str, str] = {}
    for ds, sha in DATASETS.items():
        corpus, queries, qrels = load_pinned(ds)
        shas[ds] = sha
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
        print(f"  {ds:32s} nDCG@10={res[ds]['ndcg10']:.4f} MRR@10={res[ds]['mrr10']:.4f} "
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
