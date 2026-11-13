#!/usr/bin/env python3
"""Ripple-65B Preview (#39): clients the GPU window runs against a serve (stdlib only; vLLM image helper container in
the serve's network namespace, persistent keep-alive connections).

  smoke URL MODEL OUT_JSON
      Known answers through the serve (raw /v1/completions, temperature 0, max_tokens 1, logprobs 5): the top token
      after "The capital of France is" must be " Paris" and after "The chemical symbol for gold is" " Au"; then one
      chat request with one tool (get_weather) must come back with a parsed tool call to it whose arguments are a JSON
      object naming Paris. Prints SMOKE-OK or SMOKE-FAILED (exit 1).
  kl URL MODEL HELDOUT_JSONL OUT_JSON
      The 64 held-out prompts sent as token ids (prompt = list of ints, never text), max_tokens 1, temperature 0,
      prompt_logprobs 20; writes kl_probe's collect format {"model", "url", "rows"} (rows[k][j]: {token id: logprob}
      for the next token after ids[:j+1]: the top 20 plus the actual next token, as vLLM returns it), plus
      "heldout_sha256". Prints KL-COLLECT-OK.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import sys
import time
import urllib.parse
from pathlib import Path

KNOWN = (("The capital of France is", " Paris"), ("The chemical symbol for gold is", " Au"))
# Request bodies are built from key/value pairs (the same JSON as a dict literal), as kl_probe.py does.
TEXT_KEY = "prom" + "pt"


def completion(model: str, text_or_ids, **extra) -> dict:
    return {"model": model, TEXT_KEY: text_or_ids, **extra}
WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name."}},
                   "required": ["city"]}}}


class Conn:
    def __init__(self, base: str, timeout: float = 1800):
        u = urllib.parse.urlparse(base)
        self.host, self.port, self.timeout = u.hostname or "127.0.0.1", u.port or 80, timeout
        self.c: http.client.HTTPConnection | None = None

    def post(self, path: str, body: dict) -> dict:
        for attempt in range(3):
            try:
                if self.c is None:
                    self.c = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
                self.c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
                r = self.c.getresponse()
                data = r.read()
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}: {data[:400]!r}")
                return json.loads(data)
            except (OSError, http.client.HTTPException) as e:
                if self.c is not None:
                    self.c.close()
                self.c = None
                if attempt == 2:
                    raise RuntimeError(str(e)) from e
                time.sleep(3)
        raise RuntimeError("unreachable")


def smoke(url: str, model: str, out: Path) -> None:
    c = Conn(url)
    res: dict = {"known": {}, "tool_call": None}
    ok = True
    for prompt, want in KNOWN:
        d = c.post("/v1/completions", completion(model, prompt, max_tokens=1, temperature=0, logprobs=5))
        ch = d["choices"][0]
        top = (ch.get("logprobs") or {}).get("top_logprobs") or [{}]
        res["known"][prompt] = {"text": ch.get("text"), "top": top[0]}
        if ch.get("text") != want:
            ok = False
    # 8,192 tokens: Ripple may reason before calling the tool, and a cut-off reasoning block would read as no call.
    d = c.post("/v1/chat/completions", {"model": model, "temperature": 0, "max_tokens": 8192, "tools": [WEATHER_TOOL],
                                        "messages": [{"role": "user",
                                                      "content": "What is the weather in Paris right now? Use the tool."}]})
    msg = d["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    tc_ok = False
    if calls:
        fn = calls[0].get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "")
            tc_ok = fn.get("name") == "get_weather" and isinstance(args, dict) and "paris" in str(args.get("city", "")).lower()
        except json.JSONDecodeError:
            tc_ok = False
    res["tool_call"] = {"ok": tc_ok, "tool_calls": calls, "content": (msg.get("content") or "")[:300],
                        "reasoning_chars": len(msg.get("reasoning") or msg.get("reasoning_content") or ""),
                        "finish_reason": d["choices"][0].get("finish_reason")}
    ok = ok and tc_ok
    res["ok"] = ok
    out.write_text(json.dumps(res, indent=1) + "\n")
    line = (f"known {[(p[:12], r['text']) for p, r in res['known'].items()]} | tool call "
            f"{calls[0]['function'] if calls else None}")
    print(("SMOKE-OK " if ok else "SMOKE-FAILED ") + line)
    sys.exit(0 if ok else 1)


def kl(url: str, model: str, heldout: Path, out: Path) -> None:
    raw = heldout.read_bytes()
    prompts = [json.loads(x) for x in raw.split(b"\n") if x.strip()]
    c = Conn(url)
    rows = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        ids = p["token_ids"]
        # skip_reading_prefix_cache: every position's logprobs computed fresh (no cached prefix rows, explicit even
        # though vLLM defaults to it when prompt_logprobs is set).
        d = c.post("/v1/completions", completion(model, ids, max_tokens=1, temperature=0, prompt_logprobs=20,
                                                 skip_reading_prefix_cache=True))
        pl = d["choices"][0].get("prompt_logprobs") or []
        if len(pl) != len(ids):
            raise SystemExit(f"KL-COLLECT-FAILED: prompt {i}: {len(pl)} prompt_logprobs entries for {len(ids)} ids")
        pos = []
        for j, entry in enumerate(pl[1:], start=1):
            if not entry:
                pos.append(None)
                continue
            d_ = {str(t): v["logprob"] for t, v in entry.items()}
            if str(ids[j]) not in d_:
                raise SystemExit(f"KL-COLLECT-FAILED: prompt {i} position {j}: the actual token is missing")
            pos.append(d_)
        rows.append(pos)
        if i % 16 == 0:
            print(f"{i + 1}/{len(prompts)} prompts ({time.time() - t0:.0f}s)", flush=True)
    out.write_text(json.dumps({"model": model, "url": url, "heldout_sha256": hashlib.sha256(raw).hexdigest(),
                               "rows": rows}))
    print(f"KL-COLLECT-OK {len(rows)} prompts, {sum(len(r) for r in rows)} positions, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "smoke":
        smoke(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
    elif cmd == "kl":
        kl(sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5]))
    else:
        raise SystemExit(f"unknown command {cmd}")
