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

import hashlib
import io
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

DATASETS = {  # reranker_eval.DATASETS: the same full commit ids as the reranker gate; the default branch is never consulted
    "mteb/NanoNFCorpusRetrieval": "2eec9d63140fb9af7c75ad0fdbf6350aaf20e94e",
    "mteb/NanoFiQA2018Retrieval": "ede84da4793cd2d4e61bba0dd08fc58de90f3681",
    "mteb/NanoTouche2020Retrieval": "ff81c0caff8b6febe1495eae9f2ca3eddad074b8",
    "mteb/NanoArguAnaRetrieval": "63ac9f026e8f6b85dc0622f3d630ed7a1758e9ec",
    "mteb/NanoSciFactRetrieval": "8c5f7bbe90bb865f46c67710221afa7d9096251b",
    "mteb/NanoMSMARCORetrieval": "5a5aa0c3df65d6883d85c8f5cdc88e679633595e",
}
SPLIT_SHA256 = {  # reranker_eval.SPLIT_SHA256: sha256 of <split>/train-00000-of-00001.parquet at the DATASETS commit
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


def _load_pinned(ds: str) -> tuple[list[dict], list[dict], list[dict]]:
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


def fetch(out: Path) -> None:
    data, shas = {}, {}
    for ds, sha in DATASETS.items():
        corpus, queries, qrels = _load_pinned(ds)
        shas[ds] = sha
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
