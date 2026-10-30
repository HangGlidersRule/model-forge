#!/usr/bin/env python3
"""Issue #17 prefill throughput probe (pre-registered: ~3k and ~50k-token prompts, concurrency 1 and 8).
Each request: completions, max_tokens=1, stream=True; TTFT = time to the first streamed chunk. A unique nonce opens
every prompt so prefix caching can't serve it. Prompt text: cnn_dailymail articles from the calibration file,
concatenated and cut to the target length with the server's own /tokenize.
Usage: throughput_probe.py URL MODEL CALIB_JSONL OUT_JSON LABEL"""
import json
import statistics
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

TOKENIZE_KEYS = ("model", "prompt")
COMPLETION_KEYS = ("model", "prompt", "max_tokens", "temperature", "stream", "stream_options")

url, model, calib, out, label = sys.argv[1:6]


def post(path, body, stream=False):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)


pool = " ".join(json.loads(line)["text"] for line in open(calib, encoding="utf-8") if '"cnn_dailymail"' in line)


def build(target):
    lo, hi = 0, len(pool)
    while hi - lo > 200:  # binary search on characters for the target token count
        mid = (lo + hi) // 2
        n = json.loads(post("/tokenize", dict(zip(TOKENIZE_KEYS, (model, pool[:mid])))).read())["count"]
        lo, hi = (mid, hi) if n < target else (lo, mid)
    return pool[:lo]


def one(text):
    body = dict(zip(COMPLETION_KEYS, (model, f"[{uuid.uuid4()}]\n{text}", 1, 0, True, {"include_usage": True})))
    t0 = time.perf_counter()
    ttft, ptoks = None, None
    with post("/v1/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if ttft is None and d.get("choices"):
                ttft = time.perf_counter() - t0
            if d.get("usage"):
                ptoks = d["usage"]["prompt_tokens"]
    return ttft, ptoks


res = {"label": label, "cells": {}}
for target in (3000, 50000):
    text = build(target)
    one(text)  # warm-up (shape compile / tune), not counted
    for conc, n in ((1, 4), (8, 16)):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(conc) as ex:
            r = list(ex.map(one, [text] * n))
        wall = time.perf_counter() - t0
        ttfts = [x[0] for x in r]
        toks = [x[1] for x in r]
        cell = {"requests": n, "concurrency": conc, "prompt_tokens_mean": round(statistics.mean(toks)),
                "ttft_p50_ms": round(statistics.median(ttfts) * 1000, 1), "ttft_max_ms": round(max(ttfts) * 1000, 1),
                "prefill_tok_per_s_aggregate": round(sum(toks) / wall)}
        res["cells"][f"{target // 1000}k_c{conc}"] = cell
        print(label, f"{target // 1000}k c{conc}", json.dumps(cell), flush=True)
json.dump(res, open(out, "w"), indent=1)
