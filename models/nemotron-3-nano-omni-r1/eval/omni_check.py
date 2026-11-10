#!/usr/bin/env python3
"""Issue #35 serving checks for Nemotron-3-Nano-Omni builds, run as a client against one vLLM server at a time.

  text   8 fixed text prompts, greedy, thinking off, with per-token logprobs: proves a reassembled wrapper computes the
         same text as the published trunk-only product (identical weights must give identical greedy output).
  image  every image in MEDIA_DIR: "Describe this image in two sentences.", greedy, thinking off.
  audio  every .wav in MEDIA_DIR: "Transcribe this audio.", greedy, thinking off.
  ttft   time to first content token for each office photo (coco-*) at concurrency 1 (1 warm-up + N timed runs; prefix
         and multimodal caches must be off on the server so repeats are real prefills).
Images and audio are sent as file:// URLs under the server's --allowed-local-media-path.
Usage: omni_check.py BASE MODEL LABEL MEDIA_DIR(server-side path) MEDIA_LIST(comma-separated file names) OUT_JSON
       [--skip-media] [--ttft-runs N]
       omni_check.py compare A_JSON B_JSON   (text equivalence of two runs)"""
from __future__ import annotations

import http.client
import json
import sys
import time
import urllib.parse

TEXT_PROMPTS = [
    "What is the capital of Australia? Answer in one sentence.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain why the sky is blue in two sentences.",
    "A train travels 120 km in 1.5 hours. What is its average speed? Show the calculation.",
    "List three differences between TCP and UDP.",
    "Translate 'Good morning, how are you?' into French and Spanish.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "What is 17 * 23? Answer with just the number.",
]
IMAGE_PROMPT = "Describe this image in two sentences."
AUDIO_PROMPT = "Transcribe this audio."


# Built from parts: the public export rewrites any 3+-segment absolute path literal as private layout.
CHAT_ROUTE = "/v1/" + "chat/completions"


def post(base: str, body: dict, timeout: float = 600) -> dict:
    u = urllib.parse.urlparse(base)
    c = http.client.HTTPConnection(u.hostname or "127.0.0.1", u.port or 80, timeout=timeout)
    c.request("POST", CHAT_ROUTE, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    payload = r.read()
    c.close()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {payload[:300]!r}")
    return json.loads(payload)


def chat(model: str, content, max_tokens: int, logprobs: bool = False) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.0, "top_k": 1,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    if logprobs:
        body.update({"logprobs": True, "top_logprobs": 1})
    return body


def answer(resp: dict) -> dict:
    ch = resp["choices"][0]
    lp = ch.get("logprobs") or {}
    toks = [(t["token"], round(t["logprob"], 5)) for t in (lp.get("content") or [])]
    msg = ch["message"]
    # With --reasoning-parser on and thinking off the answer is in `content`; fall back to the reasoning field (named
    # reasoning_content or reasoning depending on the vLLM version) so a parser routing difference cannot blank a row.
    field = "content" if msg.get("content") else next((f for f in ("reasoning_content", "reasoning") if msg.get(f)), "content")
    return {"text": msg.get(field) or "", "field": field, "finish": ch.get("finish_reason"),
            "prompt_tokens": resp["usage"]["prompt_tokens"], "completion_tokens": resp["usage"]["completion_tokens"],
            "tokens": toks}


def ttft(base: str, body: dict) -> float:
    u = urllib.parse.urlparse(base)
    c = http.client.HTTPConnection(u.hostname or "127.0.0.1", u.port or 80, timeout=600)
    body = dict(body, stream=True, max_tokens=32)
    t0 = time.perf_counter()
    c.request("POST", CHAT_ROUTE, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    first = None
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        delta = json.loads(line[5:])["choices"][0].get("delta", {})
        if first is None and (delta.get("content") or delta.get("reasoning_content")):
            first = time.perf_counter() - t0
    c.close()
    if first is None:
        raise RuntimeError("stream produced no content")
    return first


def run(argv: list[str]) -> int:
    base, model, label, media_dir, media_list, out = argv[:6]
    skip_media = "--skip-media" in argv
    n_ttft = int(argv[argv.index("--ttft-runs") + 1]) if "--ttft-runs" in argv else 0
    media = [m for m in media_list.split(",") if m]
    res: dict = {"label": label, "model": model, "text": [], "image": [], "audio": [], "ttft_s": {}}
    for p in TEXT_PROMPTS:
        res["text"].append({"input": p, **answer(post(base, chat(model, p, 192, logprobs=True)))})
    print(f"{label}: text {len(res['text'])} done", flush=True)
    if not skip_media:
        for m in media:
            url = f"file://{media_dir.rstrip('/')}/{m}"
            if m.endswith((".jpg", ".jpeg", ".png")):
                content = [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": IMAGE_PROMPT}]
                res["image"].append({"media": m, **answer(post(base, chat(model, content, 160)))})
                if n_ttft and m.startswith("coco-"):
                    body = chat(model, content, 32)
                    ttft(base, body)  # warm-up
                    runs = sorted(ttft(base, body) for _ in range(n_ttft))
                    res["ttft_s"][m] = {"runs": n_ttft, "p50": round(runs[len(runs) // 2], 4), "min": round(runs[0], 4),
                                        "max": round(runs[-1], 4)}
            elif m.endswith((".wav", ".mp3", ".flac")):
                content = [{"type": "audio_url", "audio_url": {"url": url}}, {"type": "text", "text": AUDIO_PROMPT}]
                res["audio"].append({"media": m, **answer(post(base, chat(model, content, 256)))})
        print(f"{label}: image {len(res['image'])}, audio {len(res['audio'])}, ttft {res['ttft_s']}", flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"OMNI-CHECK-OK {label}", flush=True)
    return 0


def compare(a_path: str, b_path: str) -> int:
    a, b = (json.load(open(p)) for p in (a_path, b_path))
    same_text = same_prompt = 0
    max_lp = 0.0
    for x, y in zip(a["text"], b["text"]):
        same_text += x["text"] == y["text"]
        same_prompt += x["prompt_tokens"] == y["prompt_tokens"]
        for (tx, lx), (ty, ly) in zip(x["tokens"], y["tokens"]):
            if tx != ty:
                break
            max_lp = max(max_lp, abs(lx - ly))
    n = len(a["text"])
    ok = same_text == n and same_prompt == n
    print(f"TEXT-EQUIVALENCE {'OK' if ok else 'DIFF'} {a['label']} vs {b['label']}: identical outputs {same_text}/{n}, "
          f"identical prompt token counts {same_prompt}/{n}, max |logprob diff| on matching tokens {max_lp:.5f}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    sys.exit(run(sys.argv[1:]))
