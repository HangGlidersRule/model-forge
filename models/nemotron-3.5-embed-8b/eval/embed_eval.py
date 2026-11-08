#!/usr/bin/env python3
"""Issue #33 embedder gate harness: frozen eval sets, embedding through a vLLM pooling server, scoring, speed.

Models are run in their own documented retrieval format (STYLES):
  nemotron  query: "query: " + q          document: "passage: " + d      (the checkpoint's sentence-transformers prompts)
  qwen      query: "Instruct: <task>\\nQuery:" + q   document: d           (Qwen3-Embedding; <task> = the set's instruction)

Sets (built once, hashed, frozen before any candidate runs):
  mteb   the six pinned MTEB Nano retrieval sets with their FULL corpora (24,893 documents, 299 queries): the #18 public
         retrieval benchmark, unchanged. Qwen task instruction: the default web-search instruction (as in #18).
  lmeb   memory retrieval: the 540 LMEB Dialogue queries of the #32 reranker set (same query ids), each scored against its
         WHOLE conversation (every turn of that scene in KaLM-Embedding/LMEB@f9d4294b candidates.jsonl), not 30
         candidates. Qwen task instruction: LMEB's own per-split instruction (task_instructions.json).
Scoring: cosine top-10 per query, nDCG@10 (graded qrels for MTEB, binary for LMEB) and recall@10, macro over sets/splits;
paired bootstrap (10,000 resamples, seed 1234, queries resampled within each set) for every delta.

Usage (subcommands): build-sets writes the two frozen sets (MTEB output, LMEB output, and the #32 reranker set for the
query ids); embed, speed and refcheck take the server base URL, the served model name and the retrieval style
(nemotron or qwen), plus the set file, the calibration file or the reference file; score takes a set file, the vector
directory and an output file."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import math
import random
import sys
import time
import urllib.request
from pathlib import Path

MTEB = {  # the #18 / #32 pinned revisions as full commit ids; the default branch is never consulted
    "mteb/NanoNFCorpusRetrieval": "2eec9d63140fb9af7c75ad0fdbf6350aaf20e94e",
    "mteb/NanoFiQA2018Retrieval": "ede84da4793cd2d4e61bba0dd08fc58de90f3681",
    "mteb/NanoTouche2020Retrieval": "ff81c0caff8b6febe1495eae9f2ca3eddad074b8",
    "mteb/NanoArguAnaRetrieval": "63ac9f026e8f6b85dc0622f3d630ed7a1758e9ec",
    "mteb/NanoSciFactRetrieval": "8c5f7bbe90bb865f46c67710221afa7d9096251b",
    "mteb/NanoMSMARCORetrieval": "5a5aa0c3df65d6883d85c8f5cdc88e679633595e",
}
MTEB_SPLIT_SHA256 = {  # sha256 of <split>/train-00000-of-00001.parquet at the MTEB commit
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
WEB_TASK = "Given a web search query, retrieve relevant passages that answer the query"
LMEB = "KaLM-Embedding/LMEB"
LMEB_REV = "f9d4294be4a24b8c16d6bfb59d56a6fec4bd95c0"
MAX_CHARS = 24_000  # long documents: piecewise mean of 24k-char pieces (the #18 policy; rarely triggers)
SEED = 1234


def _get(url: str) -> bytes:
    for attempt in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "model-forge-eval"}), timeout=300).read()
        except Exception:  # noqa: BLE001 - retried, then fatal
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def _jsonl(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.split(b"\n") if line.strip()]


def _load_mteb(ds: str) -> tuple[list[dict], list[dict], list[dict]]:
    """corpus, queries, qrels of `ds` at its MTEB commit; refuses any file whose sha256 is not pinned."""
    import pyarrow.parquet as pq

    sha = MTEB[ds]
    tables = []
    for split in ("corpus", "queries", "qrels"):
        raw = _get(f"https://huggingface.co/datasets/{ds}/resolve/{sha}/{split}/train-00000-of-00001.parquet")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != MTEB_SPLIT_SHA256[ds][split]:
            raise RuntimeError(f"{ds}@{sha} {split}: sha256 {digest} != pinned {MTEB_SPLIT_SHA256[ds][split]}")
        tables.append(pq.read_table(io.BytesIO(raw)).to_pylist())
    corpus, queries, qrels = tables
    return corpus, queries, qrels


def build_sets(mteb_out: Path, lmeb_out: Path, rerank_set: Path) -> None:
    data, shas = {}, {}
    for ds, sha in MTEB.items():
        corpus, queries, qrels = _load_mteb(ds)
        shas[ds] = sha
        idk = "_id" if "_id" in corpus[0] else "id"
        docs = {c[idk]: ((c.get("title") or "").strip() + "\n" + (c.get("text") or "").strip()).strip() for c in corpus}
        qtext = {q[idk]: q["text"] for q in queries}
        rel: dict[str, dict[str, int]] = {}
        for r in qrels:
            if r["score"] > 0:
                rel.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
        qids = sorted(q for q in rel if q in qtext)
        data[ds.split("/")[1]] = {"task": WEB_TASK, "doc_ids": sorted(docs), "docs": [docs[d] for d in sorted(docs)],
                                  "qids": qids, "queries": [qtext[q] for q in qids], "qrels": {q: rel[q] for q in qids}}
    mteb_out.write_text(json.dumps({"name": "mteb-nano-full-corpora", "dataset_shas": shas, "sets": data}, ensure_ascii=False))
    print(f"MTEB-SET docs={sum(len(v['docs']) for v in data.values())} queries={sum(len(v['qids']) for v in data.values())} "
          f"sha256={hashlib.sha256(mteb_out.read_bytes()).hexdigest()}", flush=True)

    rr = json.loads(rerank_set.read_bytes())
    assert rr["source"] == f"{LMEB}@{LMEB_REV}", rr["source"]
    base = f"https://huggingface.co/datasets/{LMEB}/resolve/{LMEB_REV}"
    tasks = json.loads(_get(f"{base}/task_instructions.json"))
    sets: dict = {}
    for split_key, items in rr["data"].items():
        task, split = split_key.split("/")
        if task not in sets:
            corpus = {c["id"]: ((c.get("title") or "").strip() + "\n" + (c.get("text") or "").strip()).strip()
                      for c in _jsonl(_get(f"{base}/Dialogue/{task}/corpus.jsonl"))}
            groups = {g["scene_id"]: g["candidate_doc_ids"] for g in _jsonl(_get(f"{base}/Dialogue/{task}/candidates.jsonl"))}
            sets[task] = {"corpus": corpus, "groups": groups}
        instr = tasks[task][split] if isinstance(tasks[task], dict) else tasks[task]
        rows = []
        for q in items:
            scene = q["qid"].rsplit("_q_", 1)[0]
            rows.append({"qid": q["qid"], "scene": scene, "query": q["query"],
                         "relevant": sorted(c["id"] for c in q["cands"] if c["rel"])})
        sets.setdefault("_splits", {})[split_key] = {"task": instr, "queries": rows}
    out: dict = {"name": "lmeb-dialogue-scene-retrieval", "source": f"{LMEB}@{LMEB_REV}",
                 "query_ids_from": f"the #32 reranker set (sha256 {hashlib.sha256(rerank_set.read_bytes()).hexdigest()})",
                 "scenes": {}, "splits": sets.pop("_splits")}
    for task, v in sets.items():
        for scene, ids in v["groups"].items():
            out["scenes"][f"{task}/{scene}"] = {"doc_ids": ids, "docs": [v["corpus"][i] for i in ids]}
    for split_key, s in out["splits"].items():
        task = split_key.split("/")[0]
        for q in s["queries"]:
            q["scene"] = f"{task}/{q['scene']}"
            assert set(q["relevant"]) <= set(out["scenes"][q["scene"]]["doc_ids"]), q["qid"]
    lmeb_out.write_text(json.dumps(out, ensure_ascii=False))
    n_docs = sum(len(s["doc_ids"]) for s in out["scenes"].values())
    sizes = sorted(len(s["doc_ids"]) for s in out["scenes"].values())
    print(f"LMEB-SET scenes={len(out['scenes'])} docs={n_docs} (per scene {sizes[0]}-{sizes[-1]}) "
          f"queries={sum(len(s['queries']) for s in out['splits'].values())} "
          f"sha256={hashlib.sha256(lmeb_out.read_bytes()).hexdigest()}", flush=True)


def escape(text: str) -> str:
    """Nemotron's chat template (mark_user_text) escapes '[' and '<' with a zero-width non-joiner; rendering the template
    client-side keeps batched /v1/embeddings requests token-identical to a chat-template request."""
    return text.replace("[", "[\u200c").replace("<", "<\u200c")


def fmt_query(style: str, task: str, q: str) -> str:
    return "query: " + escape(q) if style == "nemotron" else f"Instruct: {task}\nQuery:" + q


def fmt_doc(style: str, d: str) -> str:
    return "passage: " + escape(d) if style == "nemotron" else d


def _post(base: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request(f"{base}/v1/embeddings", data=data, headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=600).read())
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise RuntimeError(f"/v1/embeddings failed after 5 attempts: {e}") from e
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


STYLE = {"name": "qwen"}  # set by main(): Nemotron requests pass add_special_tokens false (its template adds no BOS/EOS)


def _embed(base: str, model: str, texts: list[str]):
    import numpy as np

    def one(i: int) -> list:
        body = {"model": model, "input": texts[i:i + 32]}
        if STYLE["name"] == "nemotron":
            body["add_special_tokens"] = False
        d = _post(base, body)["data"]
        return [x["embedding"] for x in sorted(d, key=lambda x: x["index"])]

    with cf.ThreadPoolExecutor(4) as ex:
        chunks = list(ex.map(one, range(0, len(texts), 32)))
    return np.asarray([v for c in chunks for v in c], dtype=np.float32)


def _embed_docs(base: str, model: str, style: str, docs: list[str]):
    import numpy as np

    pieces = [(i, t[o:o + MAX_CHARS]) for i, t in enumerate(docs) for o in range(0, max(1, len(t)), MAX_CHARS)]
    vecs = _embed(base, model, [fmt_doc(style, p) for _, p in pieces]).astype(np.float64)
    acc = np.zeros((len(docs), vecs.shape[1]))
    n = np.zeros(len(docs))
    for (i, _), v in zip(pieces, vecs):
        acc[i] += v
        n[i] += 1
    return (acc / n[:, None]).astype(np.float32)


def embed(base: str, model: str, style: str, set_path: Path, label: str, out_dir: Path) -> None:
    import numpy as np

    blob = json.loads(set_path.read_bytes())
    arrays = {}
    t0 = time.time()
    if blob["name"].startswith("mteb"):
        for ds, d in blob["sets"].items():
            arrays[f"{ds}|docs"] = _embed_docs(base, model, style, d["docs"])
            arrays[f"{ds}|queries"] = _embed(base, model, [fmt_query(style, d["task"], q) for q in d["queries"]])
            print(f"[{time.time() - t0:.0f}s] {label} {ds}: docs {arrays[f'{ds}|docs'].shape}", flush=True)
    else:
        for scene, s in blob["scenes"].items():
            arrays[f"{scene}|docs"] = _embed_docs(base, model, style, s["docs"])
        for split, s in blob["splits"].items():
            arrays[f"{split}|queries"] = _embed(base, model, [fmt_query(style, s["task"], q["query"]) for q in s["queries"]])
        print(f"[{time.time() - t0:.0f}s] {label} lmeb: {len(blob['scenes'])} scenes, {len(blob['splits'])} splits", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{blob['name']}--{label}.npz", **arrays)
    print(f"EMBED-OK {label} {blob['name']} digest={hashlib.sha256(set_path.read_bytes()).hexdigest()[:12]} ({time.time() - t0:.0f}s)",
          flush=True)


def per_query(blob: dict, vec_c: dict, vec_q: dict, dims: int = 4096):
    """nDCG@10 and recall@10 per query, grouped by set (MTEB dataset or LMEB split): {group: (ndcg[], recall[])}.
    dims < full size: Matryoshka truncation (keep the first `dims` components, renormalize)."""
    import numpy as np

    def norm(m):
        m = m[:, :dims]
        return m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-12)

    def metrics(gains_top, rel_vals):
        dcg = sum((2 ** g - 1) / math.log2(r + 2) for r, g in enumerate(gains_top))
        ideal = sorted(rel_vals, reverse=True)[:10]
        idcg = sum((2 ** g - 1) / math.log2(r + 2) for r, g in enumerate(ideal))
        return (dcg / idcg if idcg else 0.0), sum(1 for g in gains_top if g > 0) / min(10, len(rel_vals))

    out = {}
    if blob["name"].startswith("mteb"):
        for ds, d in blob["sets"].items():
            top = np.argsort(-(norm(vec_q[f"{ds}|queries"]) @ norm(vec_c[f"{ds}|docs"]).T), axis=1)[:, :10]
            nd, rc = [], []
            for qi, qid in enumerate(d["qids"]):
                rels = d["qrels"][qid]
                a, b = metrics([rels.get(d["doc_ids"][j], 0) for j in top[qi]], list(rels.values()))
                nd.append(a)
                rc.append(b)
            out[ds] = (np.asarray(nd), np.asarray(rc))
    else:
        for split, s in blob["splits"].items():
            qv = norm(vec_q[f"{split}|queries"])
            nd, rc = [], []
            for qi, q in enumerate(s["queries"]):
                sc = blob["scenes"][q["scene"]]
                top = np.argsort(-(norm(vec_c[f"{q['scene']}|docs"]) @ qv[qi]))[:10]
                rel = set(q["relevant"])
                a, b = metrics([1 if sc["doc_ids"][j] in rel else 0 for j in top], [1] * len(rel))
                nd.append(a)
                rc.append(b)
            out[split] = (np.asarray(nd), np.asarray(rc))
    return out


def score(set_path: Path, vec_dir: Path, out_path: Path) -> None:
    """Every label's own-corpus score (all-X); each candidate's BF16-corpus x candidate-query pairing within its model;
    paired deltas: candidate vs its model's BF16, replica vs BF16, and every model's runs vs the others."""
    import numpy as np

    blob = json.loads(set_path.read_bytes())
    vec = {p.stem.split("--", 1)[1]: dict(np.load(p)) for p in sorted(vec_dir.glob(f"{blob['name']}--*.npz"))}
    rng = np.random.default_rng(SEED)
    groups = list(blob["sets"]) if blob["name"].startswith("mteb") else list(blob["splits"])
    sizes = {g: (len(blob["sets"][g]["qids"]) if blob["name"].startswith("mteb") else len(blob["splits"][g]["queries"])) for g in groups}
    draws = {g: rng.integers(0, sizes[g], (10_000, sizes[g])) for g in groups}

    def macro(pq):
        return float(np.mean([pq[g][0].mean() for g in groups]))

    def delta(pa, pb):
        d = {g: pa[g][0] - pb[g][0] for g in groups}
        boot = np.mean([d[g][draws[g]].mean(axis=1) for g in groups], axis=0)
        return {"delta": round(float(np.mean([d[g].mean() for g in groups])), 4),
                "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]}

    own = {lab: per_query(blob, v, v) for lab, v in vec.items()}
    res: dict = {"set": blob["name"], "set_sha256": hashlib.sha256(set_path.read_bytes()).hexdigest(),
                 "bootstrap": "paired, queries resampled within each set/split, macro mean, 10,000 resamples, seed 1234",
                 "runs": {}, "within_model": {}, "between_models": {}, "truncated_dims": {}}
    for dims in (2000, 1024, 768):
        res["truncated_dims"][str(dims)] = {lab: round(macro(per_query(blob, v, v, dims)), 4) for lab, v in vec.items()}
    for lab, pq in own.items():
        res["runs"][lab] = {"macro_ndcg10": round(macro(pq), 4),
                            "macro_recall10": round(float(np.mean([pq[g][1].mean() for g in groups])), 4),
                            "per_set_ndcg10": {g: round(float(pq[g][0].mean()), 4) for g in groups}}
    for lab in vec:
        model, _, variant = lab.partition("-")
        bf = f"{model}-bf16"
        if variant == "bf16" or bf not in vec:
            continue
        cross = per_query(blob, vec[bf], vec[lab])
        res["within_model"][lab] = {
            "all": {**delta(own[lab], own[bf]), "ratio_to_bf16": round(macro(own[lab]) / macro(own[bf]), 4)},
            "bf16_corpus_x_query": {"macro_ndcg10": round(macro(cross), 4), **delta(cross, own[bf]),
                                    "ratio_to_bf16": round(macro(cross) / macro(own[bf]), 4)}}
    labs = sorted(vec)
    for i, a in enumerate(labs):
        for b in labs[i + 1:]:
            if a.split("-")[0] != b.split("-")[0]:
                res["between_models"][f"{a} minus {b}"] = delta(own[a], own[b])
    out_path.write_text(json.dumps(res, indent=1) + "\n")
    for lab, r in res["runs"].items():
        w = res["within_model"].get(lab)
        extra = (f" | vs own BF16 {w['all']['ratio_to_bf16']:.4f} {w['all']['delta']:+.4f} {w['all']['ci95']} "
                 f"| BF16-corpus x query {w['bf16_corpus_x_query']['ratio_to_bf16']:.4f}") if w else ""
        print(f"{blob['name']} {lab:14s} nDCG@10 {r['macro_ndcg10']:.4f}{extra}", flush=True)
    print(f"SCORE-WRITTEN {out_path}", flush=True)


def speed(base: str, model: str, style: str, calib: Path, label: str, out_path: Path) -> None:
    """The #18 grid: single-input requests of short (~100-token) and long (~4k-token) public text at concurrency 1, 8, 32
    and 64; every request starts with a unique tag so prefix caching never hits."""
    texts = [json.loads(line)["text"] for line in calib.read_bytes().split(b"\n") if line.strip()]
    # Calibration rows are chat-template renders ("passage: " + text with '[' / '<' escaped); recover the raw passages so
    # both models get the same text and fmt_doc applies each model's own format exactly once.
    passages = [t.split("passage: ", 1)[1].replace("[\u200c", "[").replace("<\u200c", "<")
                for t in texts if t.startswith("passage: ")]
    longs, buf = [], ""
    for t in passages * 40:
        buf += t + "\n"
        if len(buf) >= 16000:
            longs.append(buf)
            buf = ""
    counter = iter(range(10 ** 9))
    # One persistent (keep-alive) connection per worker thread, like any pooled HTTP client. A new TCP connection per
    # request through Docker Desktop's published port stalls ~69 s now and then (SYN retransmits; reproduced with a
    # trivial HTTP server: 43 of 4,000 requests at c=32, none with keep-alive and none inside the container's own network
    # namespace). Those stalls, not the model, produced five impossible cells in the first #33 speed run.
    import http.client
    import threading
    import urllib.parse

    u = urllib.parse.urlparse(base)
    host, port = u.hostname or "127.0.0.1", u.port or 80
    local = threading.local()

    def post_keepalive(body: dict) -> dict:
        data = json.dumps(body).encode()
        for attempt in range(3):
            c = getattr(local, "conn", None)
            if c is None:
                c = local.conn = http.client.HTTPConnection(host, port, timeout=600)
            try:
                c.request("POST", "/v1/embeddings", body=data, headers={"Content-Type": "application/json"})
                r = c.getresponse()
                payload = r.read()
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}: {payload[:200]!r}")
                return json.loads(payload)
            except (http.client.HTTPException, ConnectionError, OSError):
                c.close()
                local.conn = None
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    def one(text: str):
        body = {"model": model, "input": [fmt_doc(style, f"[{next(counter)}] {text}")]}
        if style == "nemotron":
            body["add_special_tokens"] = False
        t = time.perf_counter()
        r = post_keepalive(body)
        return time.perf_counter() - t, r["usage"]["prompt_tokens"]

    res = {"label": label, "cells": {}}
    for kind, pool, plan in (("short", passages, ((1, 64), (8, 256), (32, 512), (64, 768))),
                             ("long", longs, ((1, 16), (8, 64), (32, 128), (64, 192)))):
        for c, n in plan:
            items = [pool[i % len(pool)] for i in range(n + 2 * c)]
            with cf.ThreadPoolExecutor(c) as ex:
                list(ex.map(one, items[: 2 * c]))  # warm-up, discarded
                t0 = time.perf_counter()
                out = list(ex.map(one, items[2 * c:]))
                wall = time.perf_counter() - t0
            lat = sorted(x[0] for x in out)
            toks = sum(x[1] for x in out)
            cell = {"requests": n, "concurrency": c, "wall_s": round(wall, 2), "input_tok_per_s": round(toks / wall),
                    "mean_input_tokens": round(toks / n), "p50_ms": round(1000 * lat[len(lat) // 2], 1),
                    "p95_ms": round(1000 * lat[max(0, int(len(lat) * 0.95) - 1)], 1), "max_ms": round(1000 * lat[-1], 1)}
            # Fail closed on a stalled cell: no single embedding request here should take 20 s, and one that does
            # measures the transport, not the model.
            if lat[-1] > 20:
                raise SystemExit(f"SPEED-STALL {label} {kind} c={c}: a request took {lat[-1]:.1f}s (p50 {lat[len(lat) // 2]:.3f}s)")
            res["cells"][f"{kind}_c{c}"] = cell
            print(label, kind, json.dumps(cell), flush=True)
    out_path.write_text(json.dumps(res, indent=1) + "\n")
    print(f"SPEED-WRITTEN {out_path}", flush=True)


def refcheck(base: str, model: str, ref_path: Path, against: str, min_cos: float) -> int:
    """Embed the reference texts written by quantize_ne8.py through the server (as-is: they are already chat-template
    renders) and compare with the reference vectors computed in transformers (`bf16`, or `fakequant` after calibration).
    Proves the serving path (bidirectional attention, mean pooling, normalization, no BOS/EOS) matches the model."""
    import numpy as np

    ref = json.loads(ref_path.read_bytes())
    got = _embed(base, model, ref["texts"])
    want = np.asarray(ref[against], dtype=np.float32)
    cos = (got / np.linalg.norm(got, axis=1, keepdims=True) * want).sum(1)
    q = [i for i, t in enumerate(ref["texts"]) if t.startswith("query: ")]
    p = [i for i, t in enumerate(ref["texts"]) if t.startswith("passage: ")]
    ok = float(cos.min()) >= min_cos
    print(f"REFCHECK-{'OK' if ok else 'FAIL'} vs {against}: {len(cos)} texts ({len(q)} query, {len(p)} passage), cosine min "
          f"{cos.min():.5f} mean {cos.mean():.5f} (bar {min_cos}); dims {got.shape[1]}", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-sets")
    b.add_argument("mteb_out", type=Path)
    b.add_argument("lmeb_out", type=Path)
    b.add_argument("rerank_set", type=Path)
    e = sub.add_parser("embed")
    s = sub.add_parser("score")
    sp = sub.add_parser("speed")
    rc = sub.add_parser("refcheck")
    for p in (e, sp, rc):
        p.add_argument("--base", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--style", choices=("nemotron", "qwen"), required=True)
    for p in (e, sp):
        p.add_argument("--label", required=True)
        p.add_argument("--out", type=Path, required=True)
    rc.add_argument("--ref", type=Path, required=True)
    rc.add_argument("--against", choices=("bf16", "fakequant"), required=True)
    rc.add_argument("--min-cos", type=float, required=True)
    e.add_argument("--set", type=Path, required=True)
    sp.add_argument("--calib", type=Path, required=True)
    s.add_argument("--set", type=Path, required=True)
    s.add_argument("--dir", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    random.seed(SEED)
    STYLE["name"] = getattr(a, "style", "qwen")
    if a.cmd == "build-sets":
        build_sets(a.mteb_out, a.lmeb_out, a.rerank_set)
    elif a.cmd == "embed":
        embed(a.base, a.model, a.style, a.set, a.label, a.out)
    elif a.cmd == "score":
        score(a.set, a.dir, a.out)
    elif a.cmd == "refcheck":
        return refcheck(a.base, a.model, a.ref, a.against, a.min_cos)
    else:
        speed(a.base, a.model, a.style, a.calib, a.label, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
