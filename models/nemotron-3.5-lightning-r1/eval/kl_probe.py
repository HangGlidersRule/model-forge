#!/usr/bin/env python3
"""Issue #17 next-token KL diagnostic against BF16 (pre-registered as non-gating).

For each prompt, ask the server for prompt logprobs (top-20 per position) with max_tokens=1. The reference (BF16)
run is saved; a candidate run is compared position by position: KL(ref || cand) over the reference's top-20 tokens
plus a remainder bucket (1 - sum of the top-20 probabilities), the standard truncated estimator. Also reports top-1
agreement of the next-token argmax.
Prompts: 64 held-out texts (the calibration builder's sources at rows NOT used for calibration), rendered as given.
Usage:
  kl_probe.py collect URL MODEL PROMPTS_JSONL OUT_JSON
  kl_probe.py compare REF_JSON CAND_JSON
"""
import json
import math
import sys
import urllib.request

REQUEST_KEYS = ("model", "prompt", "max_tokens", "temperature", "prompt_logprobs", "echo")


def collect(url, model, prompts_path, out_path):
    prompts = [json.loads(line)["text"] for line in open(prompts_path, encoding="utf-8") if line.strip()]
    rows = []
    for i, p in enumerate(prompts):
        # Request body built from key/value pairs (same JSON as a dict literal).
        body = dict(zip(REQUEST_KEYS, (model, p, 1, 0, 20, False)))
        req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=600).read())
        pl = d["choices"][0].get("prompt_logprobs") or []
        pos = []
        for entry in pl[1:]:  # first position has no prediction
            if not entry:
                pos.append(None)
                continue
            pos.append({tid: v["logprob"] for tid, v in entry.items()})
        rows.append(pos)
        if i % 8 == 0:
            print(f"{i + 1}/{len(prompts)} prompts, {len(pos)} positions", flush=True)
    json.dump({"model": model, "url": url, "rows": rows}, open(out_path, "w"))
    print("saved", out_path, sum(len(r) for r in rows), "positions")


def compare(ref_path, cand_path):
    ref, cand = json.load(open(ref_path))["rows"], json.load(open(cand_path))["rows"]
    kl_sum, n, agree = 0.0, 0, 0
    for r_row, c_row in zip(ref, cand):
        for r, c in zip(r_row, c_row):
            if not r or not c:
                continue
            p = {t: math.exp(v) for t, v in r.items()}
            q = {t: math.exp(v) for t, v in c.items()}
            p_rest = max(1e-12, 1.0 - sum(p.values()))
            q_rest = max(1e-12, 1.0 - sum(q.get(t, 0.0) for t in p))
            floor = min(q.values()) if q else 1e-12  # tokens outside the candidate's top-20 get at most its 20th prob
            kl = sum(pv * (math.log(pv) - math.log(q.get(t, floor))) for t, pv in p.items())
            kl += p_rest * (math.log(p_rest) - math.log(q_rest))
            kl_sum += kl
            n += 1
            agree += max(p, key=p.get) == max(q, key=q.get)
    print(json.dumps({"positions": n, "mean_kl": round(kl_sum / n, 6), "top1_agreement": round(agree / n, 5)}))


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(*sys.argv[2:6])
    else:
        compare(*sys.argv[2:4])
