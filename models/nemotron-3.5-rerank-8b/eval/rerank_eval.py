#!/usr/bin/env python3
"""Issue #32 reranker gate client: scores a frozen eval set against one running vLLM pooling server, and runs the
endpoint-equivalence check and the speed cells. Works for both rerankers in the comparison:
  - Nemotron-3.5-Rerank-8B (BF16 / candidates): raw-logit scores (use_activation false), chat template query/document;
  - Qwen3-Reranker-4B (BF16 / published W4A4): its own template and hf-overrides, scores in [0, 1].
Ranking metrics are order-based, so the two score scales do not need to match.

Quality: per query, score every candidate through /v1/score (text_1 = query, text_2 = documents, 16 per request),
rank, and compute nDCG@10 and MRR@10. Full denominator: every pair must get a score; HTTP errors retry 5x, then fail.
Endpoint check (`check`): for N pairs, /v1/score, /v1/rerank and /classify (structured query/document messages, the
model card's client) must agree within --tol; prints ENDPOINTS-AGREE or ENDPOINTS-DIFFER with the max |delta|.
Speed (`--throughput`): unique ~1k-token pairs, 64 pairs per request, concurrency 1/4/8, 3 measured runs after a
warm-up; reports pairs/s. Every request carries a unique leading tag so the prefix cache cannot inflate it.

Usage (subcommands): build-lmeb writes the LMEB set (option out); check runs the endpoint check, run scores quality
(plus speed with the throughput flag) and speed measures speed only. The last three take the server base URL, the
served model name, the eval set file, and (run, speed) a label and an output file."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SEED = 1234
MAX_CANDS = 30
LMEB = "KaLM-Embedding/LMEB"
LMEB_REV = "f9d4294be4a24b8c16d6bfb59d56a6fec4bd95c0"
# Dialogue memory tasks whose answers are conversation turns. TMD's date-range splits (dozens of relevant turns per
# query) are excluded: with up to 15 positives in 30 candidates they saturate nDCG and cannot separate candidates.
LMEB_SPLITS = {
    "LoCoMo": ["single_hop", "multi_hop", "temporal_reasoning", "open_domain", "adversarial"],
    "REALTALK": ["commonsense", "multi_hop", "temporal_reasoning"],
    "TMD": ["content_time_qs"],
}
LMEB_MAX_QUERIES_PER_SPLIT = 60
LMEB_MAX_POSITIVES = 4


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "model-forge-eval"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def _jsonl(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.split(b"\n") if line.strip()]


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.ids = sorted(docs)
        self.tf = [Counter(_tokens(docs[i])) for i in self.ids]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def rank(self, query: str) -> list[str]:
        q = [t for t in _tokens(query) if t in self.idf]
        scored = []
        for i, (tf, dl) in enumerate(zip(self.tf, self.len)):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            scored.append((s, self.ids[i]))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [d for _, d in scored]


def build_lmeb(out: Path) -> None:
    """Memory-retrieval eval set: for each query, its judged-relevant turns plus the top BM25 non-relevant turns from its
    OWN conversation (the dataset's candidate group), 30 candidates, seeded shuffle. A seeded sample of at most
    LMEB_MAX_QUERIES_PER_SPLIT queries per split keeps every split represented."""
    base = f"https://huggingface.co/datasets/{LMEB}/resolve/{LMEB_REV}"
    rng = random.Random(SEED)
    data: dict[str, list[dict]] = {}
    stats = []
    for task, splits in LMEB_SPLITS.items():
        corpus = {c["id"]: ((c.get("title") or "").strip() + "\n" + (c.get("text") or "").strip()).strip()
                  for c in _jsonl(_get(f"{base}/Dialogue/{task}/corpus.jsonl"))}
        groups = {g["scene_id"]: g["candidate_doc_ids"] for g in _jsonl(_get(f"{base}/Dialogue/{task}/candidates.jsonl"))}
        bm_by_scene = {s: BM25({d: corpus[d] for d in ids if d in corpus}) for s, ids in groups.items()}
        for split in splits:
            queries = {q["id"]: q["text"] for q in _jsonl(_get(f"{base}/Dialogue/{task}/{split}/queries.jsonl"))}
            rel: dict[str, set[str]] = {}
            for line in _get(f"{base}/Dialogue/{task}/{split}/qrels.tsv").decode().splitlines():
                parts = line.strip().split("\t")
                if len(parts) == 3 and parts[2].strip().isdigit() and int(parts[2]) > 0:
                    rel.setdefault(parts[0], set()).add(parts[1])
            qids = sorted(q for q in rel if q in queries and 0 < len(rel[q]) <= LMEB_MAX_POSITIVES)
            qids = sorted(rng.sample(qids, min(LMEB_MAX_QUERIES_PER_SPLIT, len(qids))))
            items, bm_hits = [], 0
            for qid in qids:
                scene = qid.rsplit("_q_", 1)[0]
                bm = bm_by_scene[scene]
                ranked = bm.rank(queries[qid])
                pos = sorted(d for d in rel[qid] if d in corpus)
                neg = [d for d in ranked if d not in rel[qid]][:MAX_CANDS - len(pos)]
                bm_hits += sum(1 for d in ranked[:MAX_CANDS] if d in rel[qid])
                cands = pos + neg
                rng.shuffle(cands)
                items.append({"qid": qid, "query": queries[qid],
                              "cands": [{"id": d, "text": corpus[d], "rel": int(d in rel[qid])} for d in cands]})
            data[f"{task}/{split}"] = items
            stats.append(f"{task}/{split}: {len(items)} q, BM25 top-30 holds {bm_hits} relevant")
    blob = {"seed": SEED, "max_cands": MAX_CANDS, "negatives": "bm25_top_nonrelevant_same_conversation",
            "source": f"{LMEB}@{LMEB_REV}", "splits": LMEB_SPLITS, "max_queries_per_split": LMEB_MAX_QUERIES_PER_SPLIT,
            "max_positives": LMEB_MAX_POSITIVES, "data": data}
    out.write_text(json.dumps(blob, ensure_ascii=False))
    n_pairs = sum(len(q["cands"]) for items in data.values() for q in items)
    print("\n".join(stats))
    print(f"EVALSET queries={sum(len(v) for v in data.values())} pairs={n_pairs} "
          f"sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")


def _post(base: str, path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request(f"{base}{path}", data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001 - retried, then fatal
            if attempt == 4:
                raise RuntimeError(f"{path} failed after 5 attempts: {e}") from e
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def score_batch(base: str, model: str, query: str, docs: list[str]) -> list[float]:
    out = _post(base, "/v1/score", {"model": model, "text_1": query, "text_2": docs})
    scores = sorted(out["data"], key=lambda x: x["index"])
    if len(scores) != len(docs):
        raise RuntimeError(f"got {len(scores)} scores for {len(docs)} docs")
    return [float(s["score"]) for s in scores]


def rerank_batch(base: str, model: str, query: str, docs: list[str]) -> list[float]:
    out = _post(base, "/v1/rerank", {"model": model, "query": query, "documents": docs, "top_n": len(docs)})
    by_index = {r["index"]: float(r["relevance_score"]) for r in out["results"]}
    return [by_index[i] for i in range(len(docs))]


def classify_one(base: str, model: str, query: str, doc: str) -> float:
    body = {"model": model, "messages": [{"role": "query", "content": [{"type": "text", "text": query}]},
                                         {"role": "document", "content": [{"type": "text", "text": doc}]}],
            "add_special_tokens": False, "use_activation": False}
    return float(_post(base, "/classify", body)["data"][0]["probs"][0])


def dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def quality(base: str, model: str, evalset: dict) -> dict:
    res: dict = {}
    for ds, items in evalset["data"].items():
        per_q = []
        for q in items:
            docs = [c["text"] for c in q["cands"]]
            scores: list[float] = []
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
        print(f"  {ds:34s} nDCG@10={res[ds]['ndcg10']:.4f} MRR@10={res[ds]['mrr10']:.4f} "
              f"({res[ds]['n_queries']} q, {res[ds]['n_pairs']} pairs)", flush=True)
    macro = sum(v["ndcg10"] for v in res.values()) / len(res)
    print(f"  MACRO nDCG@10={macro:.4f}", flush=True)
    return {"per_set": res, "macro_ndcg10": macro}


def throughput(base: str, model: str, evalset: dict) -> dict:
    pool = sorted({c["text"] for items in evalset["data"].values() for q in items for c in q["cands"]}, key=len, reverse=True)
    pool = [d[:4500] for d in pool if len(d) >= 3000][:2000]
    rng = random.Random(SEED)
    out: dict = {"docs_used": len(pool), "doc_chars": [len(pool[0]), len(pool[-1])] if pool else []}

    def one(i: int) -> int:
        docs = [f"[{i}:{k}] " + pool[(i * 64 + k) % len(pool)] for k in range(64)]
        q = f"query {i} {rng.random():.6f}: which passage best answers this question?"
        score_batch(base, model, q, docs)
        return sum(len(d) for d in docs)

    for conc in (1, 4, 8):
        runs = []
        for rep in range(4):  # first is warm-up
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


def endpoint_check(base: str, model: str, evalset: dict, n: int, tol: float) -> int:
    pairs = [(q["query"], c["text"]) for items in evalset["data"].values() for q in items for c in q["cands"]]
    pairs = random.Random(SEED).sample(pairs, n)
    worst = {"rerank": 0.0, "classify": 0.0}
    for query, doc in pairs:
        s = score_batch(base, model, query, [doc])[0]
        r = rerank_batch(base, model, query, [doc])[0]
        c = classify_one(base, model, query, doc)
        worst["rerank"] = max(worst["rerank"], abs(s - r))
        worst["classify"] = max(worst["classify"], abs(s - c))
    ok = all(v <= tol for v in worst.values())
    print(f"{'ENDPOINTS-AGREE' if ok else 'ENDPOINTS-DIFFER'} n={n} tol={tol} max|score-rerank|={worst['rerank']:.2e} "
          f"max|score-classify|={worst['classify']:.2e}", flush=True)
    return 0 if ok else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-lmeb")
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("--base", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--evalset", type=Path, required=True)
    c.add_argument("--n", type=int, default=64)
    c.add_argument("--tol", type=float, default=1e-3)
    r = sub.add_parser("run")
    r.add_argument("--base", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--evalset", type=Path, required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--throughput", action="store_true")
    sp = sub.add_parser("speed")
    sp.add_argument("--base", required=True)
    sp.add_argument("--model", required=True)
    sp.add_argument("--evalset", type=Path, required=True)
    sp.add_argument("--label", required=True)
    sp.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "build-lmeb":
        build_lmeb(a.out)
        return 0
    evalset = json.loads(a.evalset.read_bytes())
    if a.cmd == "check":
        return endpoint_check(a.base, a.model, evalset, a.n, a.tol)
    digest = hashlib.sha256(a.evalset.read_bytes()).hexdigest()
    if a.cmd == "speed":
        print(f"SPEED {a.label} evalset_sha256={digest}", flush=True)
        a.out.write_text(json.dumps({"label": a.label, "evalset_sha256": digest,
                                     "throughput": throughput(a.base, a.model, evalset)}))
        print(f"RESULT-WRITTEN {a.out}", flush=True)
        return 0
    print(f"RUN {a.label} evalset_sha256={digest}", flush=True)
    res = {"label": a.label, "evalset_sha256": digest, "quality": quality(a.base, a.model, evalset)}
    if a.throughput:
        res["throughput"] = throughput(a.base, a.model, evalset)
    a.out.write_text(json.dumps(res))
    print(f"RESULT-WRITTEN {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
