#!/usr/bin/env python3
"""Issue #18 public retrieval benchmark (amendment R18-A1): the six pinned MTEB Nano retrieval sets used for the reranker
gate, full Nano corpora, nDCG@10 and recall@10, three pairings per candidate and one BF16 replica pairing.

Step 1 (fetch, any host with pyarrow): retrieval_benchmark.py fetch OUT_JSON downloads the corpora/queries/qrels at the pinned revisions.
Step 2 (embed): retrieval_benchmark.py embed LABEL URL DATA_JSON OUT_DIR -> OUT_DIR/<label>.npz (docs and queries, full 4096-dim).
Step 3 (score): retrieval_benchmark.py score DATA_JSON OUT_DIR -> OUT_DIR/benchmark.json; pairings for each candidate X:
  all-BF16 (bf16 corpus, bf16 queries), BF16-corpus x X-queries, all-X; plus bf16 corpus x bf16-replica queries.
Queries use one fixed instruction (the model's default web-search retrieval instruction); documents use none.
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

DATASETS = {  # reranker_eval.DATASETS: the same pinned revisions as the reranker gate
    "mteb/NanoNFCorpusRetrieval": "2eec9d63140f",
    "mteb/NanoFiQA2018Retrieval": "ede84da4793c",
    "mteb/NanoTouche2020Retrieval": "ff81c0caff8b",
    "mteb/NanoArguAnaRetrieval": "63ac9f026e8f",
    "mteb/NanoSciFactRetrieval": "8c5f7bbe90bb",
    "mteb/NanoMSMARCORetrieval": "5a5aa0c3df65",
}
INSTRUCT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
MAX_CHARS = 24_000  # long documents: the same piecewise-mean policy as our transcript store (rarely triggers on Nano)


def _get(url: str) -> bytes:
    for attempt in range(5):
        try:
            return urllib.request.urlopen(url, timeout=300).read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch(out: Path) -> None:
    import pyarrow.parquet as pq

    data, shas = {}, {}
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
        rel: dict[str, dict[str, int]] = {}
        for r in qrels:
            if r["score"] > 0:
                rel.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
        qids = sorted(q for q in rel if q in qtext)
        data[ds] = {"doc_ids": sorted(docs), "docs": [docs[d] for d in sorted(docs)],
                    "qids": qids, "queries": [qtext[q] for q in qids], "qrels": {q: rel[q] for q in qids}}
        print(f"{ds}: {len(docs)} docs, {len(qids)} queries", flush=True)
    out.write_text(json.dumps({"dataset_shas": shas, "instruction": INSTRUCT, "data": data}, ensure_ascii=False))


def _embed(texts: list[str], url: str) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), 32):
        body = json.dumps({"model": os.environ.get("EMBED_MODEL", "qwen3-embedding-8b"), "input": texts[i:i + 32]}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        for attempt in range(5):
            try:
                d = json.loads(urllib.request.urlopen(req, timeout=600).read())["data"]
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        out += [x["embedding"] for x in sorted(d, key=lambda x: x["index"])]
    return out


def embed(label: str, url: str, data_path: Path, out_dir: Path) -> None:
    import numpy as np

    blob = json.loads(data_path.read_text())
    arrays = {}
    t0 = time.time()
    for ds, d in blob["data"].items():
        pieces = [(i, t[o:o + MAX_CHARS]) for i, t in enumerate(d["docs"]) for o in range(0, max(1, len(t)), MAX_CHARS)]
        vecs = np.asarray(_embed([p for _, p in pieces], url), dtype=np.float64)
        docs = np.zeros((len(d["docs"]), vecs.shape[1]))
        n = np.zeros(len(d["docs"]))
        for (i, _), v in zip(pieces, vecs):
            docs[i] += v
            n[i] += 1
        arrays[f"{ds}|docs"] = (docs / n[:, None]).astype(np.float32)
        arrays[f"{ds}|queries"] = np.asarray(_embed([INSTRUCT + q for q in d["queries"]], url), dtype=np.float32)
        print(f"[{time.time()-t0:.0f}s] {label} {ds}: docs {arrays[f'{ds}|docs'].shape} queries "
              f"{arrays[f'{ds}|queries'].shape}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{label}.npz", **arrays)


def score(data_path: Path, out_dir: Path) -> None:
    """nDCG@10 / recall@10 per pairing; macro = mean of the six set means. 95% CIs on the macro delta vs all-BF16 from a
    paired bootstrap that resamples queries within each set (10,000 resamples, seed 1234)."""
    import numpy as np

    blob = json.loads(data_path.read_text())
    vec = {p.stem: np.load(p) for p in sorted(out_dir.glob("*.npz"))}
    cands = sorted(k for k in vec if k not in ("bf16", "bf16-replica"))
    sets = list(blob["data"])

    def norm(m, dims):
        m = m[:, :dims]
        return m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-12)

    def per_query(ds, corpus_label, query_label, dims):
        d = blob["data"][ds]
        c = norm(vec[corpus_label][f"{ds}|docs"], dims)
        q = norm(vec[query_label][f"{ds}|queries"], dims)
        top = np.argsort(-(q @ c.T), axis=1)[:, :10]
        nd, rc = [], []
        for qi, qid in enumerate(d["qids"]):
            rels = d["qrels"][qid]
            gains = [rels.get(d["doc_ids"][j], 0) for j in top[qi]]
            dcg = sum((2 ** g - 1) / math.log2(r + 2) for r, g in enumerate(gains))
            ideal = sorted(rels.values(), reverse=True)[:10]
            idcg = sum((2 ** g - 1) / math.log2(r + 2) for r, g in enumerate(ideal))
            nd.append(dcg / idcg if idcg else 0.0)
            rc.append(sum(1 for g in gains if g > 0) / min(10, len(rels)))
        return np.asarray(nd), np.asarray(rc)

    rng = np.random.default_rng(1234)
    draws = {ds: rng.integers(0, len(blob["data"][ds]["qids"]), (10_000, len(blob["data"][ds]["qids"]))) for ds in sets}

    def macro_ci(delta_by_set):
        boot = np.mean([delta_by_set[ds][draws[ds]].mean(axis=1) for ds in sets], axis=0)
        point = float(np.mean([delta_by_set[ds].mean() for ds in sets]))
        return [round(point, 4), round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]

    res = {"dataset_shas": blob["dataset_shas"], "instruction": blob["instruction"],
           "queries": {ds.split("/")[1]: len(blob["data"][ds]["qids"]) for ds in sets},
           "documents": {ds.split("/")[1]: len(blob["data"][ds]["doc_ids"]) for ds in sets},
           "bootstrap": "paired, queries resampled within each set, macro mean, 10,000 resamples, seed 1234",
           "dims": {}}
    for dims in (4096, 768):
        rows = {}
        pairings = [("all-bf16", "bf16", "bf16")]
        if "bf16-replica" in vec:
            pairings.append(("bf16-corpus x bf16-replica-query", "bf16", "bf16-replica"))
        for c in cands:
            pairings += [(f"bf16-corpus x {c}-query", "bf16", c), (f"all-{c}", c, c)]
            base = next((c[: -len(s)] for s in ("-clean", "-stable") if c.endswith(s) and c[: -len(s)] in vec), None)
            if base:
                # another serving instance (or profile) of the same artifact: its queries against the first instance's corpus
                pairings.append((f"{base}-corpus x {c}-query", base, c))
        base = {ds: per_query(ds, "bf16", "bf16", dims) for ds in sets}
        for name, cl, ql in pairings:
            pq = {ds: per_query(ds, cl, ql, dims) for ds in sets}
            mean_nd = float(np.mean([pq[ds][0].mean() for ds in sets]))
            rows[name] = {
                "per_set": {ds.split("/")[1]: {"ndcg@10": round(float(pq[ds][0].mean()), 4),
                                               "recall@10": round(float(pq[ds][1].mean()), 4)} for ds in sets},
                "mean_ndcg@10": round(mean_nd, 4),
                "mean_recall@10": round(float(np.mean([pq[ds][1].mean() for ds in sets])), 4),
                "delta_ndcg@10_vs_all_bf16_ci95": macro_ci({ds: pq[ds][0] - base[ds][0] for ds in sets}),
            }
        b = rows["all-bf16"]["mean_ndcg@10"]
        for r in rows.values():
            r["ndcg_vs_all_bf16_pct"] = round(100 * r["mean_ndcg@10"] / b, 2)
        res["dims"][str(dims)] = rows
    res["doc_vector_cosine_vs_bf16"] = {}
    for c in cands + (["bf16-replica"] if "bf16-replica" in vec else []):
        cs = np.concatenate([(norm(vec[c][f"{ds}|docs"], 4096) * norm(vec["bf16"][f"{ds}|docs"], 4096)).sum(1)
                             for ds in sets])
        res["doc_vector_cosine_vs_bf16"][c] = {"documents": int(cs.size), "mean": round(float(cs.mean()), 5),
                                               "p1": round(float(np.percentile(cs, 1)), 5),
                                               "p5": round(float(np.percentile(cs, 5)), 5), "min": round(float(cs.min()), 5)}
    res["doc_vector_cosine_between_instances"] = {}
    for c in cands:
        base = next((c[: -len(s)] for s in ("-clean", "-stable") if c.endswith(s) and c[: -len(s)] in vec), None)
        if base:
            cs = np.concatenate([(norm(vec[c][f"{ds}|docs"], 4096) * norm(vec[base][f"{ds}|docs"], 4096)).sum(1)
                                 for ds in sets])
            res["doc_vector_cosine_between_instances"][f"{base} vs {c}"] = {
                "documents": int(cs.size), "mean": round(float(cs.mean()), 5), "p1": round(float(np.percentile(cs, 1)), 5),
                "p5": round(float(np.percentile(cs, 5)), 5), "min": round(float(cs.min()), 5)}
    (out_dir / "benchmark.json").write_text(json.dumps(res, indent=1) + "\n")
    for dims, rows in res["dims"].items():
        for name, r in rows.items():
            print(f"{dims} {name}: nDCG@10 {r['mean_ndcg@10']:.4f} ({r['ndcg_vs_all_bf16_pct']}%) "
                  f"delta CI {r['delta_ndcg@10_vs_all_bf16_ci95']} recall@10 {r['mean_recall@10']:.4f}")
    print(json.dumps(res["doc_vector_cosine_vs_bf16"]))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fetch":
        fetch(Path(sys.argv[2]))
    elif cmd == "embed":
        embed(sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5]))
    elif cmd == "score":
        score(Path(sys.argv[2]), Path(sys.argv[3]))
