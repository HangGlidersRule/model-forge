#!/usr/bin/env python3
"""Issue #18 speed (pre-registered): one embedding endpoint, single-input requests of short (~100-token) and long
(~4k-token) public text at concurrency 1, 8, 32 and 64 (the issue's grid). Every request starts with a unique prefix, so prefix caching never
hits. Prints one JSON line per cell; writes all cells to OUT_JSON.
Usage: throughput_probe.py URL LABEL CALIB_JSONL OUT_JSON"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

url, label, calib, outp = sys.argv[1:5]
texts = [json.loads(line)["text"] for line in open(calib, "rb").read().split(b"\n") if line.strip()]
passages = [t for t in texts if not t.startswith("Instruct:")]
longs, buf = [], ""
for t in passages * 40:
    buf += t + "\n"
    if len(buf) >= 16000:
        longs.append(buf)
        buf = ""
counter = iter(range(10 ** 9))


def one(text: str) -> tuple[float, int]:
    body = json.dumps({"model": "qwen3-embedding-8b", "input": [f"[{next(counter)}] {text}"]}).encode()
    t = time.perf_counter()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
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
        cell = {"requests": n, "concurrency": c, "wall_s": round(wall, 2), "req_per_s": round(n / wall, 2),
                "input_tok_per_s": round(toks / wall), "mean_input_tokens": round(toks / n),
                "p50_ms": round(1000 * lat[len(lat) // 2], 1), "p95_ms": round(1000 * lat[max(0, int(len(lat) * 0.95) - 1)], 1)}
        res["cells"][f"{kind}_c{c}"] = cell
        print(label, kind, json.dumps(cell), flush=True)
open(outp, "w").write(json.dumps(res, indent=1) + "\n")
