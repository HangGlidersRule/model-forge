#!/usr/bin/env python3
"""Issue-15 decision gate: run test_set_30k through a serve_decide.py server and score it.

For every row: POST /v1/decide {kind, state, question, options} (System 1 only, thinking off,
strategy single, as the JEV-27B card evaluates), keep the full probability vector.
Writes per-row probabilities (JSONL) plus a summary that matches the card's metrics:
  kl (to the row target), top1 (rows with a defined argmax), choice_top1, noul_auroc, score_mae_expected.
compare_i15.py then computes paired metrics between configurations (reference, current NVFP4+LoRA,
merged candidate) and the replica noise floor.

Every row must get an answer: HTTP errors are retried 5x, then the run fails (full denominator).
Rows are processed in file order with a fixed concurrency; results are keyed by row id.
Usage: decide_eval.py --base URL --data test_set_30k.jsonl --out DIR --label NAME [--limit N] [--concurrency 16]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

import httpx


def kl(p: list[float], q: list[float]) -> float:
    return sum(pi * (math.log(max(pi, 1e-12)) - math.log(max(qi, 1e-12))) for pi, qi in zip(p, q) if pi > 0)


def auroc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    allv = sorted((s, i) for i, s in enumerate(pos + neg))
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[allv[k][1]] = (i + j) / 2 + 1
        i = j + 1
    rpos = sum(ranks[: len(pos)])
    return (rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


async def run(a: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in Path(a.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    if a.limit:
        rows = rows[: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    preds: dict[str, list[float]] = {}
    sem = asyncio.Semaphore(a.concurrency)
    lat: list[float] = []
    async with httpx.AsyncClient(timeout=600) as client:
        async def one(r: dict) -> None:
            # /v1/decide request: the row's kind/state/question fields, System 1 only.
            body = {k: r[k] for k in ("kind", "state", "question")}
            body.update(thinking="off", strategy="single")
            if r["kind"] == "choice":
                body["options"] = r["options"]
            async with sem:
                for attempt in range(5):
                    try:
                        t0 = time.perf_counter()
                        resp = await client.post(f"{a.base}/v1/decide", json=body)
                        resp.raise_for_status()
                        lat.append(time.perf_counter() - t0)
                        preds[r["id"]] = resp.json()["probabilities"]
                        return
                    except Exception as e:  # noqa: BLE001 - retried, then fatal
                        if attempt == 4:
                            raise RuntimeError(f"row {r['id']} failed after 5 attempts: {e}") from e
                        await asyncio.sleep(2 * (attempt + 1))
        t0 = time.perf_counter()
        await asyncio.gather(*(one(r) for r in rows))
        wall = time.perf_counter() - t0
    assert len(preds) == len(rows), f"denominator: {len(preds)}/{len(rows)}"
    with (out / f"preds-{a.label}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "kind": r["kind"], "p": preds[r["id"]]}) + "\n")
    m = metrics(rows, preds)
    lat.sort()
    m.update({"label": a.label, "rows": len(rows), "wall_s": round(wall, 1), "rows_per_s": round(len(rows) / wall, 2),
              "latency_p50_ms": round(1000 * lat[len(lat) // 2], 1), "latency_p95_ms": round(1000 * lat[int(len(lat) * 0.95)], 1),
              "concurrency": a.concurrency})
    (out / f"summary-{a.label}.json").write_text(json.dumps(m, indent=1))
    print("SUMMARY", json.dumps(m), flush=True)
    return 0


def metrics(rows: list[dict], preds: dict[str, list[float]]) -> dict:
    kls, top_ok, top_n, ch_ok, ch_n = [], 0, 0, 0, 0
    nl_scores, nl_labels, mae = [], [], []
    for r in rows:
        p, t = preds[r["id"]], r["target"]
        kls.append(kl(t, p))
        best = max(t)
        if sum(1 for x in t if x == best) == 1:
            top_n += 1
            ok = p.index(max(p)) == t.index(best)
            top_ok += ok
            if r["kind"] == "choice":
                ch_n += 1
                ch_ok += ok
        if r["kind"] == "noul" and t[1] != 0.5:
            nl_scores.append(p[1])
            nl_labels.append(int(t[1] > 0.5))
        if r["kind"] == "score":
            mae.append(abs(sum(i * x for i, x in enumerate(p)) - sum(i * x for i, x in enumerate(t))))
    return {"kl": sum(kls) / len(kls), "top1": top_ok / max(top_n, 1), "n_defined": top_n,
            "choice_top1": ch_ok / max(ch_n, 1), "noul_auroc": auroc(nl_scores, nl_labels),
            "score_mae_expected": sum(mae) / max(len(mae), 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    sys.exit(asyncio.run(run(ap.parse_args())))
