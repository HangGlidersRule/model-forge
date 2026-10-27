#!/usr/bin/env python3
"""Build the issue-15 calibration JSONL for ModelOpt hf_ptq.

512 samples, half decisions and half general text:
  - 256 System 1 decision prompts from SargeDev/jev-distill-corpus-v3 `calibration` split (pinned
    revision), rendered exactly as serve_decide.py's s1_pass builds them:
        [kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n{lines}\n[decision]:
    choice options get the trained single-token labels "A) ...", noul uses false/true, score 0..5.
    Balanced across kinds (noul / choice / score) as far as the split allows, seed 1234.
  - 256 general samples from NVIDIA's own calibration source for nvidia/Qwen3.8-27B-NVFP4
    (Nemotron-Post-Training-Dataset-v3 members): the pinned
    nvidia/Nemotron-SFT-Instruction-Following-Chat-v2 `reasoning_off` file, streamed (first 50,000
    records, seeded reservoir sample), rendered with the Qwen3.8 chat template.
Decision rows are filtered to <= 2048 tokens. No harmful-content sources are used.

Usage: build_calib_i15.py --tokenizer DIR --out FILE
Prints a summary line with the file sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

DS, DS_REV = "SargeDev/jev-distill-corpus-v3", "75111a13a8455032f0435050bfc7e0c9026ecfe8"
CHAT, CHAT_REV = "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2", "1a9454ed054b8544503ab8d8c0a519d141a44c5b"
SEED, N_DEC, N_GEN, MAX_TOK = 1234, 256, 256, 2048
LABELS = [chr(ord("A") + i) for i in range(26)]


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "model-forge"}), timeout=300) as r:
        return r.read()


def render(row: dict) -> str:
    kind, opts = row["kind"], list(row.get("options") or [])
    if kind == "choice":
        lines = [f"{LABELS[i]}) {o}" for i, o in enumerate(opts)]
    elif kind == "noul":
        lines = ["false", "true"]
    else:
        lines = [str(i) for i in range(6)]
    state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
    return (f"[kind] {kind}\n[state] {state}\n[question] {row['question']}\n[options]\n"
            + "\n".join(lines) + "\n[decision]:")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(a.tokenizer))
    rng = random.Random(SEED)

    rows = [json.loads(line) for line in get(f"https://huggingface.co/datasets/{DS}/resolve/{DS_REV}/calibration.jsonl").decode().splitlines() if line.strip()]
    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        if r["kind"] == "choice" and len(r.get("options") or []) > 16:
            continue  # the trained head covers labels A-P in one pass
        by_kind.setdefault(r["kind"], []).append(r)
    for v in by_kind.values():
        rng.shuffle(v)
    dec: list[str] = []
    kinds = sorted(by_kind)
    i = 0
    while len(dec) < N_DEC and any(by_kind.values()):
        k = kinds[i % len(kinds)]
        i += 1
        if not by_kind[k]:
            continue
        text = render(by_kind[k].pop())
        if len(tok(text, add_special_tokens=False)["input_ids"]) <= MAX_TOK:
            dec.append(text)
    kind_counts = {k: sum(1 for t in dec if t.startswith(f"[kind] {k}\n")) for k in kinds}

    # Stream the 6 GB reasoning_off.jsonl line by line (never load it): reservoir-sample 4x the
    # needed rows from the first 50,000 records, shuffle with the seed, keep the first 256 that fit.
    url = f"https://huggingface.co/datasets/{CHAT}/resolve/{CHAT_REV}/data/reasoning_off.jsonl"
    pool: list[dict] = []
    seen = 0
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "model-forge"}), timeout=600) as resp:
        for raw_line in resp:
            if not raw_line.strip():
                continue
            seen += 1
            rec = json.loads(raw_line)
            if len(pool) < N_GEN * 4:
                pool.append(rec)
            else:
                j = rng.randrange(seen)
                if j < N_GEN * 4:
                    pool[j] = rec
            if seen >= 50_000:
                break
    rng.shuffle(pool)
    gen: list[str] = []
    for r in pool:
        msgs = r.get("messages") or r.get("conversations")
        if not msgs:
            continue
        msgs = [{"role": m.get("role") or m.get("from"), "content": m.get("content") or m.get("value") or ""} for m in msgs]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False)
        except Exception:
            continue
        if len(tok(text, add_special_tokens=False)["input_ids"]) <= MAX_TOK:
            gen.append(text)
        if len(gen) >= N_GEN:
            break
    if len(dec) < N_DEC or len(gen) < N_GEN:
        print(f"short: decisions {len(dec)} general {len(gen)}", file=sys.stderr)
        return 1
    samples = dec + gen
    rng.shuffle(samples)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(a.out.read_bytes()).hexdigest()
    toks = sorted(len(tok(s, add_special_tokens=False)["input_ids"]) for s in samples)
    print(f"CALIB-OK samples={len(samples)} decisions={len(dec)} {kind_counts} general={len(gen)} "
          f"tokens_p50={toks[len(toks) // 2]} tokens_max={toks[-1]} sha256={digest} "
          f"sources={DS}@{DS_REV[:12]},{CHAT}@{CHAT_REV} seed={SEED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
