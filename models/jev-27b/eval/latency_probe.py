#!/usr/bin/env python3
"""Latency/throughput probe for issue 15: short (~200-token) vs long (~4k-token) decisions.

Builds two fixed request sets from test_set_30k (seeded, deterministic):
  short: 64 noul/choice rows whose rendered state is 600-1,000 characters (~150-250 tokens)
  long:  64 requests whose state concatenates consecutive rows' states to 15,000-17,000 characters
         (~4k tokens), keeping that row's question/options
Each set runs at concurrency 1 (first 32 requests, sequential) and 8 (all 64), after a 4-request
warmup. Reports p50/p95 latency and requests/s. Every request must succeed (5 retries, then fail).
Usage: latency_probe.py --base URL --data test_set_30k.jsonl --out FILE --label NAME
"""
import argparse
import asyncio
import json
import random
import sys
import time

import httpx


def build(rows):
    rng = random.Random(1234)
    pool = [r for r in rows if r["kind"] in ("noul", "choice") and isinstance(r["state"], str)]
    rng.shuffle(pool)
    short = [r for r in pool if 600 <= len(r["state"]) <= 1000][:64]
    long_, i = [], 0
    while len(long_) < 64 and i < len(pool):
        base, state, j = pool[i], "", i
        while len(state) < 15000 and j < len(pool):
            state += pool[j]["state"] + "\n\n"
            j += 1
        long_.append({**base, "state": state[:17000]})
        i = j
    return short, long_


async def run_set(client, base, reqs, conc):
    sem = asyncio.Semaphore(conc)
    lat = []

    async def one(r):
        body = {k: r[k] for k in ("kind", "state", "question")}
        body.update(thinking="off", strategy="single")
        if r["kind"] == "choice":
            body["options"] = r["options"]
        async with sem:
            for attempt in range(5):
                try:
                    t0 = time.perf_counter()
                    resp = await client.post(f"{base}/v1/decide", json=body)
                    resp.raise_for_status()
                    lat.append(time.perf_counter() - t0)
                    return resp.json()["usage"]["prompt_tokens"]
                except Exception as e:  # noqa: BLE001
                    if attempt == 4:
                        raise RuntimeError(f"probe request failed: {e}") from e
                    await asyncio.sleep(2)

    t0 = time.perf_counter()
    toks = await asyncio.gather(*(one(r) for r in reqs))
    wall = time.perf_counter() - t0
    lat.sort()
    return {"requests": len(reqs), "concurrency": conc, "wall_s": round(wall, 2), "req_per_s": round(len(reqs) / wall, 2),
            "p50_ms": round(1000 * lat[len(lat) // 2], 1), "p95_ms": round(1000 * lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1),
            "prompt_tokens_mean": round(sum(toks) / len(toks))}


async def main(a):
    rows = [json.loads(line) for line in open(a.data, encoding="utf-8") if line.strip()]
    short, long_ = build(rows)
    out = {"label": a.label}
    async with httpx.AsyncClient(timeout=600) as client:
        await run_set(client, a.base, short[:4], 1)  # warmup
        await run_set(client, a.base, long_[:4], 1)
        for name, reqs in (("short", short), ("long", long_)):
            out[name] = {"c1": await run_set(client, a.base, reqs[:32], 1), "c8": await run_set(client, a.base, reqs, 8)}
            print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("PROBE-DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    sys.exit(asyncio.run(main(ap.parse_args())))
