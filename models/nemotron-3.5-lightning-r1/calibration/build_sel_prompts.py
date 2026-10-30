#!/usr/bin/env python3
"""Issue #17 R17-A1: a fresh 64-prompt KL selection set, disjoint from the calibration rows and from the original
64 KL prompts. Same pinned sources and recipe as build_kl_prompts.py, seed 9876. Reads the dataset from the local HF
cache (HF_HUB_OFFLINE=1); shard names come from the calibration file.
Usage: build_sel_prompts.py TOKENIZER_DIR CALIB_JSONL KL_JSONL OUT_JSONL"""
import json
import random
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

tok_dir, calib, klp, out = sys.argv[1:5]
CNN = ("abisee/cnn_dailymail", "96df5e686bee6baa90b8bee7c28b81fa3fa6223d", "3.0.0/train-00000-of-00003.parquet")
NPT = ("nvidia/Nemotron-Post-Training-Dataset-v2", "5c89e01dd720ae0f4058445ed49c5fb68a03c76e")
cal = [json.loads(line) for line in open(calib, encoding="utf-8")]
kl = [json.loads(line) for line in open(klp, encoding="utf-8")]
split_file = {r["source"].split("/")[1]: r["file"] for r in cal if r["source"].startswith("nemotron-v2/")}
used = {(r["file"], r["index"]) for r in cal}
for r in kl:
    used.add((CNN[2] if r["source"] == "cnn_dailymail" else split_file[r["source"].split("/")[1]], r["index"]))
rng = random.Random(9876)
tok = AutoTokenizer.from_pretrained(tok_dir)


def table(repo, rev, path, cols):
    return pq.read_table(hf_hub_download(repo, path, repo_type="dataset", revision=rev), columns=cols).to_pylist()


rows = []
cnn = table(*CNN, ["article"])
for i in rng.sample([i for i in range(len(cnn)) if (CNN[2], i) not in used], 32):
    rows.append({"text": cnn[i]["article"][:1500], "source": "cnn_dailymail", "index": i})
for split in ("chat", "stem", "math", "code"):
    f = split_file[split]
    data, picked = table(*NPT, f, ["messages"]), 0
    for i in rng.sample(range(len(data)), len(data)):
        if (f, i) in used:
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
print(json.dumps({"rows": len(rows), "excluded_rows": len(used)}))
