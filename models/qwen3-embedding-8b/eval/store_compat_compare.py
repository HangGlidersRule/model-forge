#!/usr/bin/env python3
"""Issue #18 store compatibility: embed the sampled items and questions with a candidate endpoint, then compare BF16 vs candidate
against the STORED production vectors. Prints and writes aggregate numbers only (no text, vectors or ids).

Inputs (WORKDIR, from stage 1): items.jsonl, bf16_docs.npy, bf16_queries.npy, query_index.json, stored/<store>.npy|ids.json
Store keys are relabelled from the internal application names to generic ones for publication (transcripts,
memories, claims); the comparison logic is unchanged.
Usage: store_compat_compare.py WORKDIR --endpoint URL --model NAME --label LABEL [--out results.json]

For each store, with the stored corpus S (the production vectors) and the app's own search:
  drift          cos(candidate doc vector, stored vector) and the BF16 re-embedding's cos to stored (noise floor);
                 at 4096 dims and Matryoshka-truncated to 1024 and 256 (truncate, renormalize)
  retrieval      generated question -> full stored corpus; recall@1/@10 and MRR@10 of the source item, BF16 query vs
                 candidate query; top-10 overlap; the transcript store uses its app's 2,000-dim subvector top-100 + full re-rank
  item-to-item   each sampled item's text (BF16 vs candidate) -> stored corpus excluding itself; top-10 overlap, and the
                 share of its nearest-20 stored pairs whose decision flips at the store's threshold
                 (claim store 0.60 candidate floor, memory store 0.97 dedup trigger, transcript store none: reported at 0.60/0.97)
Paired bootstrap (2,000 resamples, seed 1234) gives 95% CIs on candidate-minus-BF16 recall@10 and MRR@10.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

QUERY_INSTRUCT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
THRESHOLDS = {"claims": [0.60], "memories": [0.97], "transcripts": [0.60, 0.97]}
SEED = 1234


def embed(texts: list[str], url: str, model: str, batch: int = 32) -> np.ndarray:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"model": model, "input": texts[i:i + batch]}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        for attempt in range(5):
            try:
                data = json.loads(urllib.request.urlopen(req, timeout=600).read())["data"]
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        out.extend(d["embedding"] for d in sorted(data, key=lambda d: d["index"]))
    return np.asarray(out, dtype=np.float32)


def norm(m: np.ndarray, dims: int | None = None) -> np.ndarray:
    m = m[:, :dims] if dims else m
    return m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-12)


TRANSCRIPT_MAX_CHARS = 24_000  # the transcript application's embedding piece size


def embed_docs(items: list[dict], url: str, model: str) -> np.ndarray:
    """Embed each item the way its app does (transcript store: mean of 24,000-char piece vectors; others: one input)."""
    pieces: list[tuple[int, str]] = []
    for i, it in enumerate(items):
        text = it["text"]
        if it["store"] == "transcripts":
            pieces += [(i, text[o:o + TRANSCRIPT_MAX_CHARS]) for o in range(0, max(1, len(text)), TRANSCRIPT_MAX_CHARS)]
        else:
            pieces.append((i, text))
    vecs = embed([p for _, p in pieces], url, model)
    out = np.zeros((len(items), vecs.shape[1]), dtype=np.float64)
    count = np.zeros(len(items))
    for (i, _), v in zip(pieces, vecs):
        out[i] += v
        count[i] += 1
    return (out / count[:, None]).astype(np.float32)


def topk(q: np.ndarray, corpus: np.ndarray, k: int, store: str) -> np.ndarray:
    """Indices of the top-k corpus rows for each query row, using the app's search where it differs from exact."""
    if store == "transcripts":  # app: 2,000-dim subvector cosine top-max(10k,100), then full-vector re-rank
        qs, cs = norm(q[:, :2000]), norm(corpus[:, :2000])
        cand = np.argsort(-(qs @ cs.T), axis=1)[:, :max(10 * k, 100)]
        qf, cf = norm(q), norm(corpus)
        out = np.empty((q.shape[0], k), dtype=np.int64)
        for r in range(q.shape[0]):
            sims = cf[cand[r]] @ qf[r]
            out[r] = cand[r][np.argsort(-sims)[:k]]
        return out
    sims = norm(q) @ norm(corpus).T
    return np.argsort(-sims, axis=1)[:, :k]


def rank_metrics(top: np.ndarray, gold: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hit = top == gold[:, None]
    r1 = hit[:, 0].astype(float)
    r10 = hit[:, :10].any(axis=1).astype(float)
    pos = np.where(hit[:, :10].any(axis=1), hit[:, :10].argmax(axis=1) + 1, 0)
    mrr = np.where(pos > 0, 1.0 / np.maximum(pos, 1), 0.0)
    return r1, r10, mrr


def boot_ci(diff: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    n = len(diff)
    means = np.sort([diff[rng.integers(0, n, n)].mean() for _ in range(2000)])
    return [float(diff.mean()), float(means[50]), float(means[1949])]


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    t0 = time.time()
    items = [json.loads(line) for line in (a.workdir / "items.jsonl").read_bytes().split(b"\n") if line.strip()]
    bf16_docs = np.load(a.workdir / "bf16_docs.npy")
    bf16_q = np.load(a.workdir / "bf16_queries.npy")
    qidx = json.loads((a.workdir / "query_index.json").read_text())

    cand_docs = embed_docs(items, a.endpoint, a.model)
    cand_q = embed([QUERY_INSTRUCT + items[i]["question"] for i in qidx], a.endpoint, a.model)
    print(f"[{time.time()-t0:.0f}s] embedded {cand_docs.shape[0]} docs, {cand_q.shape[0]} queries", flush=True)

    result: dict = {"label": a.label, "model": a.model, "stores": {}}
    for store in ("transcripts", "memories", "claims"):
        corpus = np.load(a.workdir / "stored" / f"{store}.npy")
        ids = json.loads((a.workdir / "stored" / f"{store}.ids.json").read_text())
        pos = {rid: i for i, rid in enumerate(ids)}
        sel = [i for i, it in enumerate(items) if it["store"] == store and it["id"] in pos]
        gold = np.array([pos[items[i]["id"]] for i in sel])
        stored_vecs = corpus[gold]
        res: dict = {"items": len(sel), "corpus": int(corpus.shape[0])}

        # 1. drift vs the stored vectors, and the BF16 re-embedding's own distance (noise floor)
        for dims in (None, 1024, 256):
            key = f"drift_{dims or 4096}"
            c_cand = (norm(cand_docs[sel], dims) * norm(stored_vecs, dims)).sum(1)
            c_bf16 = (norm(bf16_docs[sel], dims) * norm(stored_vecs, dims)).sum(1)
            res[key] = {"candidate_vs_stored": {"mean": float(c_cand.mean()), "p1": pct(c_cand, 1), "p5": pct(c_cand, 5),
                                                "min": float(c_cand.min())},
                        "bf16_vs_stored": {"mean": float(c_bf16.mean()), "p1": pct(c_bf16, 1), "p5": pct(c_bf16, 5),
                                           "min": float(c_bf16.min())}}

        # 2. question retrieval against the full stored corpus (the app's search)
        qsel = [j for j, i in enumerate(qidx) if items[i]["store"] == store and items[i]["id"] in pos]
        if qsel:
            qgold = np.array([pos[items[qidx[j]]["id"]] for j in qsel])
            tb = topk(bf16_q[qsel], corpus, 10, store)
            tc = topk(cand_q[qsel], corpus, 10, store)
            b1, b10, bm = rank_metrics(tb, qgold)
            c1, c10, cm = rank_metrics(tc, qgold)
            overlap = np.array([len(set(x) & set(y)) / 10 for x, y in zip(tb, tc)])
            res["retrieval"] = {
                "queries": len(qsel),
                "bf16": {"recall@1": float(b1.mean()), "recall@10": float(b10.mean()), "mrr@10": float(bm.mean())},
                "candidate": {"recall@1": float(c1.mean()), "recall@10": float(c10.mean()), "mrr@10": float(cm.mean())},
                "delta_recall@10_ci": boot_ci(c10 - b10),
                "delta_mrr@10_ci": boot_ci(cm - bm),
                "top10_overlap_mean": float(overlap.mean()),
            }

        # 3. item-to-item: nearest stored neighbours (excluding the item itself) and threshold decisions
        cf = norm(corpus)
        sb = norm(bf16_docs[sel]) @ cf.T
        sc = norm(cand_docs[sel]) @ cf.T
        for r, g in enumerate(gold):
            sb[r, g] = -2.0
            sc[r, g] = -2.0
        nb = np.argsort(-sb, axis=1)[:, :20]
        nc = np.argsort(-sc, axis=1)[:, :20]
        overlap10 = np.array([len(set(x[:10]) & set(y[:10])) / 10 for x, y in zip(nb, nc)])
        flips = {}
        for thr in THRESHOLDS[store]:
            fb = np.take_along_axis(sb, nb, axis=1) >= thr
            fc = np.take_along_axis(sc, nb, axis=1) >= thr
            # same pairs (BF16's nearest 20), decision with each precision
            flips[str(thr)] = {"pairs": int(fb.size), "flipped": int((fb != fc).sum()),
                               "flip_rate": float((fb != fc).mean()),
                               "above_threshold_bf16": int(fb.sum()), "above_threshold_candidate": int(fc.sum())}
        res["item_to_item"] = {"top10_overlap_mean": float(overlap10.mean()),
                               "top1_same": float((nb[:, 0] == nc[:, 0]).mean()), "threshold_flips": flips}
        result["stores"][store] = res
        print(f"[{time.time()-t0:.0f}s] {store}: " + json.dumps({k: v for k, v in res.items()
                                                                if k in ("items", "retrieval", "item_to_item")})[:900],
              flush=True)
        print(f"   drift_4096 {json.dumps(res['drift_4096'])}", flush=True)

    out = a.out or (a.workdir / f"results-{a.label}.json")
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"STAGE2-DONE {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
