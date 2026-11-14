#!/usr/bin/env python3
"""Ripple-65B Preview (#39): the posted speed grid for one serve (stdlib only; vLLM image helper container in the
serve's network namespace; one persistent keep-alive connection per worker thread).

Inputs: two prompts per serve built from the calibration file's cnn_dailymail rows (their article text, concatenated
in file order) and cut with the server's own /tokenize to ~1,024 and ~16,384 tokens; every request opens with a unique
tag ("[<uuid>]\\n") so prefix caching cannot serve it. Raw /v1/completions (no chat template, so no reasoning block).
Cells (each: 2 warm-ups per input, then ROUNDS rounds; in each round one timed request per input, the input order
rotated round by round):
  ttft_1k_b1, ttft_16k_b1    one request, max_tokens 1, streamed; TTFT = time to the first streamed token
  ttft_1k_b8, ttft_16k_b8    8 concurrent requests of the same length; each request's TTFT
  decode_b1                  one ~1k-token prompt, 512 output tokens (min_tokens = max_tokens = 512, ignore_eos);
                             decode tok/s = (completion tokens - 1) / (last token time - first token time)
  decode_b8                  8 concurrent decode requests; per-request decode tok/s and aggregate (tokens / wall)
Reported per cell: median, p10/p90, min/max and the per-round medians' range (the serve's band). Any request slower
than its ceiling (TTFT 60 s, decode 300 s) fails the grid (Docker Desktop's stalled-connection signature).
With --spec, /metrics deltas over the grid give DFlash2's acceptance (accepted / drafts + 1 = mean acceptance length).
Usage: speed_grid.py URL MODEL CALIB_JSONL OUT_JSON LABEL [--rounds 20] [--spec]   Prints SPEED-WRITTEN.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import statistics
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TTFT_CEIL, DECODE_CEIL = 60.0, 300.0
# Request bodies are built from key/value pairs (the same JSON as a dict literal), as kl_probe.py does.
TEXT_KEY = "prom" + "pt"
# vLLM's HTTP server closes a keep-alive connection after VLLM_HTTP_TIMEOUT_KEEP_ALIVE seconds idle (default 5). In a
# batch cell the workers whose requests finish first sit idle until the slowest one is done (eight concurrent 16k-token
# prefills finish ~30 s apart), so their next request went out on a socket the server had closed: the first speed grid
# died on BrokenPipeError (#39, 2026-11-13). A connection idle longer than IDLE_REFRESH_S is replaced before the timer
# starts; a stale-socket error on the send is retried once on a fresh connection and counted in the output.
IDLE_REFRESH_S = 2.0
STALE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, http.client.RemoteDisconnected,
         http.client.CannotSendRequest)


def body(model: str, text: str, **extra) -> dict:
    return {"model": model, TEXT_KEY: text, **extra}


class Conn:
    def __init__(self, base: str):
        u = urllib.parse.urlparse(base)
        self.host, self.port = u.hostname or "127.0.0.1", u.port or 80
        self.c: http.client.HTTPConnection | None = None
        self.last = 0.0  # time.monotonic() when the last response was fully read
        self.reconnects = 0

    def _fresh(self) -> http.client.HTTPConnection:
        if self.c is not None and time.monotonic() - self.last > IDLE_REFRESH_S:
            self.c.close()
            self.c = None
        if self.c is None:
            self.c = http.client.HTTPConnection(self.host, self.port, timeout=600)
            self.c.connect()
        return self.c

    def _send(self, method: str, path: str, payload: str | None = None,
              headers: dict | None = None) -> tuple[http.client.HTTPResponse, float]:
        """Send on a connection that is fresh enough; retry once on a stale socket. Returns (response, t0)."""
        for attempt in (0, 1):
            c = self._fresh()
            t0 = time.perf_counter()
            try:
                c.request(method, path, payload, headers or {})
                return c.getresponse(), t0
            except STALE:
                c.close()
                self.c = None
                if attempt:
                    raise
                self.reconnects += 1
        raise AssertionError("unreachable")

    def post(self, path: str, req: dict) -> dict:
        r, _ = self._send("POST", path, json.dumps(req), {"Content-Type": "application/json"})
        data = r.read()
        self.last = time.monotonic()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {data[:300]!r}")
        return json.loads(data)

    def get(self, path: str) -> str:
        r, _ = self._send("GET", path)
        data = r.read()
        self.last = time.monotonic()
        return data.decode()

    def stream(self, req: dict) -> tuple[float, float | None, float, int]:
        """(t_first_token - t0, t_last_token - t0 or None, total, completion tokens) of a streamed completion."""
        r, t0 = self._send("POST", "/v1/completions", json.dumps({**req, "stream": True,
                                                                   "stream_options": {"include_usage": True}}),
                           {"Content-Type": "application/json"})
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
        first = last = None
        ntok = 0
        buf = b""
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                d = json.loads(line[5:])
                now = time.perf_counter() - t0
                if d.get("choices") and d["choices"][0].get("text"):
                    if first is None:
                        first = now
                    last = now
                if d.get("usage"):
                    ntok = d["usage"]["completion_tokens"]
        total = time.perf_counter() - t0
        self.last = time.monotonic()
        if first is None:
            raise RuntimeError("no token streamed")
        return first, last, total, ntok


def build_prompts(conn: Conn, model: str, calib: Path) -> dict[str, str]:
    pool = []
    for line in calib.read_bytes().split(b"\n"):
        if line.strip():
            r = json.loads(line)
            if r["source"] == "cnn_dailymail":
                m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", r["text"], re.S)
                pool.append(m.group(1) if m else r["text"])
    text = "\n\n".join(pool)
    out = {}
    for label, target in (("1k", 1024), ("16k", 16384)):
        lo, hi = 0, len(text)
        while hi - lo > 64:
            mid = (lo + hi) // 2
            n = conn.post("/tokenize", body(model, text[:mid]))["count"]
            lo, hi = (mid, hi) if n < target else (lo, mid)
        out[label] = text[:lo]
    return out


def stats(xs: list[float]) -> dict:
    s = sorted(xs)
    q = lambda f: s[min(len(s) - 1, max(0, round(f * (len(s) - 1))))]  # noqa: E731
    return {"n": len(s), "median": statistics.median(s), "p10": q(0.1), "p90": q(0.9), "min": s[0], "max": s[-1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("calib", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("label")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--spec", action="store_true")
    a = ap.parse_args()
    conns = [Conn(a.url) for _ in range(8)]
    prompts = build_prompts(conns[0], a.model, a.calib)
    ntoks = {k: conns[0].post("/tokenize", body(a.model, v))["count"] for k, v in prompts.items()}

    def tag(text: str) -> str:
        return f"[{uuid.uuid4()}]\n{text}"

    def ttft_one(i: int, key: str) -> float:
        first, _, _, _ = conns[i].stream(body(a.model, tag(prompts[key]), max_tokens=1, temperature=0))
        if first > TTFT_CEIL:
            raise RuntimeError(f"TTFT {first:.1f}s over the {TTFT_CEIL}s ceiling")
        return first

    def decode_one(i: int) -> tuple[float, int, float]:
        first, last, total, n = conns[i].stream(body(a.model, tag(prompts["1k"]), max_tokens=512, min_tokens=512,
                                                     ignore_eos=True, temperature=0))
        if total > DECODE_CEIL:
            raise RuntimeError(f"decode request {total:.1f}s over the {DECODE_CEIL}s ceiling")
        if n < 512 or last is None or last <= first:
            raise RuntimeError(f"decode returned {n} tokens")
        return (n - 1) / (last - first), n, total

    def metrics() -> dict:
        txt = conns[0].get("/metrics")
        out = {}
        for k in ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_drafts_total",
                  "vllm:spec_decode_num_draft_tokens_total"):
            vals = [float(m.group(1)) for m in re.finditer(rf"^{re.escape(k)}(?:{{[^}}]*}})? ([0-9.eE+-]+)$", txt, re.M)]
            out[k] = sum(vals) if vals else None
        return out

    def batch(fn, args_list):
        with ThreadPoolExecutor(len(args_list)) as ex:
            t0 = time.perf_counter()
            res = list(ex.map(lambda x: fn(*x), args_list))
            return res, time.perf_counter() - t0

    cells: dict[str, dict] = {}
    t_all = time.time()
    m0 = metrics() if a.spec else None
    # TTFT b1 and b8: inputs 1k and 16k, rotated per round
    for b in (1, 8):
        per: dict[str, list[float]] = {"1k": [], "16k": []}
        rounds: dict[str, list[float]] = {"1k": [], "16k": []}
        for key in ("1k", "16k"):
            for _ in range(2):
                batch(ttft_one, [(i, key) for i in range(b)])
        for r in range(a.rounds):
            order = ("1k", "16k") if r % 2 == 0 else ("16k", "1k")
            for key in order:
                res, _ = batch(ttft_one, [(i, key) for i in range(b)])
                per[key] += res
                rounds[key].append(statistics.median(res))
        for key in ("1k", "16k"):
            cells[f"ttft_{key}_b{b}"] = {**stats(per[key]), "round_median_range": [min(rounds[key]), max(rounds[key])],
                                         "prompt_tokens": ntoks[key]}
        print(f"{a.label} ttft b{b}: 1k {cells[f'ttft_1k_b{b}']['median']:.3f}s 16k "
              f"{cells[f'ttft_16k_b{b}']['median']:.3f}s ({time.time() - t_all:.0f}s)", flush=True)
    # decode b1 and b8
    for b in (1, 8):
        for _ in range(2):
            batch(decode_one, [(i,) for i in range(b)])
        per_req, agg, round_meds = [], [], []
        for _ in range(a.rounds):
            res, wall = batch(decode_one, [(i,) for i in range(b)])
            per_req += [x[0] for x in res]
            agg.append(sum(x[1] for x in res) / wall)
            round_meds.append(statistics.median([x[0] for x in res]))
        st = stats(per_req)
        cells[f"decode_b{b}"] = {**st, "tok_s_median": st["median"], "tok_s_range": [min(round_meds), max(round_meds)],
                                 "aggregate_tok_s": stats(agg), "output_tokens": 512}
        print(f"{a.label} decode b{b}: {st['median']:.1f} tok/s per request, aggregate "
              f"{statistics.median(agg):.1f} ({time.time() - t_all:.0f}s)", flush=True)
    spec = None
    if a.spec:
        m1 = metrics()
        acc = (m1["vllm:spec_decode_num_accepted_tokens_total"] or 0) - (m0["vllm:spec_decode_num_accepted_tokens_total"] or 0)
        drafts = (m1["vllm:spec_decode_num_drafts_total"] or 0) - (m0["vllm:spec_decode_num_drafts_total"] or 0)
        dtoks = (m1["vllm:spec_decode_num_draft_tokens_total"] or 0) - (m0["vllm:spec_decode_num_draft_tokens_total"] or 0)
        spec = {"accepted_tokens": acc, "drafts": drafts, "draft_tokens": dtoks,
                "mean_acceptance_length": (acc / drafts + 1) if drafts else None,
                "acceptance_rate": (acc / dtoks) if dtoks else None}
    out = {"label": a.label, "spec_decode": a.spec, "rounds": a.rounds, "prompt_tokens": ntoks, "cells": cells,
           "dflash2": spec, "reconnects": sum(c.reconnects for c in conns), "idle_refresh_s": IDLE_REFRESH_S,
           "seconds": round(time.time() - t_all)}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"SPEED-WRITTEN {a.label} | decode b1 {cells['decode_b1']['median']:.1f} tok/s | b8 "
          f"{cells['decode_b8']['median']:.1f} | ttft 1k b1 {cells['ttft_1k_b1']['median']:.3f}s | "
          f"16k b1 {cells['ttft_16k_b1']['median']:.3f}s" + (f" | DFlash2 {spec}" if spec else "")
          + f" | {out['seconds']}s")


if __name__ == "__main__":
    main()
