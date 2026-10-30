#!/usr/bin/env python3
"""Build 64 held-out KL prompts for issue #17 from the same pinned sources as the calibration set, excluding every
row the calibration set used (by source file + index). 32 cnn_dailymail articles (first 1,500 chars) + 8 conversations
from each Nemotron-v2 split rendered with the chat template, seed 4321.
Usage: build_kl_prompts.py TOKENIZER_DIR CALIB_JSONL OUT_JSONL"""
import json
import random
import sys

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

tok_dir, calib, out = sys.argv[1:4]
used = {(r["file"], r["index"]) for r in map(json.loads, open(calib, encoding="utf-8"))}
rng = random.Random(4321)
tok = AutoTokenizer.from_pretrained(tok_dir)
CNN = ("abisee/cnn_dailymail", "96df5e686bee6baa90b8bee7c28b81fa3fa6223d", "3.0.0/train-00000-of-00003.parquet")
NPT = ("nvidia/Nemotron-Post-Training-Dataset-v2", "5c89e01dd720ae0f4058445ed49c5fb68a03c76e")


def table(repo, rev, path, cols):
    return pq.read_table(hf_hub_download(repo, path, repo_type="dataset", revision=rev), columns=cols).to_pylist()


rows = []
cnn = table(*CNN, ["article"])
for i in rng.sample([i for i in range(len(cnn)) if (CNN[2], i) not in used], 32):
    rows.append({"text": cnn[i]["article"][:1500], "source": "cnn_dailymail", "index": i})
files = sorted(HfApi().list_repo_files(NPT[0], repo_type="dataset", revision=NPT[1]))
for split in ("chat", "stem", "math", "code"):
    first = next(f for f in files if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
    data, picked = table(*NPT, first, ["messages"]), 0
    for i in rng.sample(range(len(data)), len(data)):
        if (first, i) in used:
            continue
        msgs = [{"role": m["role"], "content": m["content"]} for m in data[i]["messages"] if m.get("content")]
        text = tok.apply_chat_template(msgs, tokenize=False) if msgs and msgs[-1]["role"] == "assistant" else None
        if not text or len(tok(text).input_ids) > 3000:
            continue
        rows.append({"text": text, "source": f"nemotron-v2/{split}", "index": i})
        picked += 1
        if picked == 8:
            break
with open(out, "w", encoding="utf-8") as fo:
    for r in rows:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps({"rows": len(rows), "overlap_with_calibration": 0}))
