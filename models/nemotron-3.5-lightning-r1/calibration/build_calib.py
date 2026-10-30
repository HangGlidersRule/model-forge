#!/usr/bin/env python3
"""Build the issue-17 calibration set for Nemotron-3.5-Lightning W4A4 NVFP4.

Sources (pinned revisions; the same two dataset families the shipped W4A16 cells were calibrated on):
  abisee/cnn_dailymail@96df5e686bee6baa90b8bee7c28b81fa3fa6223d, 3.0.0 train: article text.
  nvidia/Nemotron-Post-Training-Dataset-v2@5c89e01dd720ae0f4058445ed49c5fb68a03c76e, splits chat/stem/math/code:
      each conversation rendered with the model's own chat template, as a client sends it, so the activation
      statistics match serving traffic.
Output: JSONL rows {"text", "source", "file", "index"}: 512 cnn + 512 nemotron (128 per split), seed 1234.
hf_ptq reads the `text` key.
Usage: build_calib.py TOKENIZER_DIR OUT_JSONL
"""
import json
import random
import sys

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

tok_dir, out = sys.argv[1], sys.argv[2]
SEED, N_CNN, N_PER_SPLIT = 1234, 512, 128
CNN = ("abisee/cnn_dailymail", "96df5e686bee6baa90b8bee7c28b81fa3fa6223d", "3.0.0/train-00000-of-00003.parquet")
NPT = ("nvidia/Nemotron-Post-Training-Dataset-v2", "5c89e01dd720ae0f4058445ed49c5fb68a03c76e")
rng = random.Random(SEED)
tok = AutoTokenizer.from_pretrained(tok_dir)


def table(repo: str, rev: str, path: str, columns: list[str]) -> list[dict]:
    f = hf_hub_download(repo, path, repo_type="dataset", revision=rev)
    return pq.read_table(f, columns=columns).to_pylist()


rows = []
cnn = table(*CNN, ["article"])
for i in sorted(rng.sample(range(len(cnn)), N_CNN)):
    rows.append({"text": cnn[i]["article"], "source": "cnn_dailymail", "file": CNN[2], "index": i})

npt_files = sorted(HfApi().list_repo_files(NPT[0], repo_type="dataset", revision=NPT[1]))
for split in ("chat", "stem", "math", "code"):
    first = next(f for f in npt_files if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
    data = table(*NPT, first, ["messages"])
    picked = 0
    for i in rng.sample(range(len(data)), len(data)):
        msgs = [{"role": m["role"], "content": m["content"]} for m in data[i]["messages"] if m.get("content")]
        if not msgs or msgs[-1]["role"] != "assistant":
            continue
        rows.append({"text": tok.apply_chat_template(msgs, tokenize=False), "source": f"nemotron-v2/{split}",
                     "file": first, "index": i})
        picked += 1
        if picked == N_PER_SPLIT:
            break
    if picked < N_PER_SPLIT:
        raise SystemExit(f"only {picked} usable conversations in {first}")

rng.shuffle(rows)
with open(out, "w", encoding="utf-8") as fo:
    for r in rows:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
lens = sorted(len(tok(r["text"]).input_ids) for r in rows)
print(json.dumps({"rows": len(rows),
                  "by_source": {s: sum(1 for r in rows if r["source"] == s) for s in sorted({r["source"] for r in rows})},
                  "tokens_p50": lens[len(lens) // 2], "tokens_p90": lens[int(len(lens) * 0.9)], "tokens_max": lens[-1]}))
