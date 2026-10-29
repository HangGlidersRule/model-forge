#!/usr/bin/env python3
"""Issue #18 post-hoc diagnostic (NOT pre-registered; does not change the verdict): ranking bias in a mixed transcript corpus.
After a drop-in switch, old chunks keep their stored BF16 vectors and new chunks get NVFP4 vectors. Does an NVFP4 query
favour NVFP4-embedded chunks? Half of the sampled transcript items (group N, by id hash) get NVFP4 vectors in a copy of the
stored corpus; the other half (group B) keep their stored vectors. NVFP4 questions are scored per group against the
stored corpus and the mixed corpus, with the transcript application's own search. Prints aggregates only.
Usage: mixed_corpus_probe.py WORKDIR NVFP4_URL OUT_JSON"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from store_compat_compare import (  # noqa: E402
    QUERY_INSTRUCT,
    embed,
    embed_docs,
    norm,
    rank_metrics,
    topk,
)

work, url, outp = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
items = [json.loads(line) for line in (work / "items.jsonl").read_bytes().split(b"\n") if line.strip()]
corpus = np.load(work / "stored" / "transcripts.npy")
ids = json.loads((work / "stored" / "transcripts.ids.json").read_text())
pos = {rid: i for i, rid in enumerate(ids)}
sel = [it for it in items if it["store"] == "transcripts" and it["id"] in pos and it.get("question")]
group_n = np.array([int(hashlib.md5(it["id"].encode()).hexdigest(), 16) % 2 == 0 for it in sel])
gold = np.array([pos[it["id"]] for it in sel])
docs_n = embed_docs(sel, url, "qwen3-embedding-8b")
q_n = embed([QUERY_INSTRUCT + it["question"] for it in sel], url, "qwen3-embedding-8b")
mixed = corpus.copy()
mixed[gold[group_n]] = docs_n[group_n]
# How much closer an NVFP4 query sits to the NVFP4 vector of its source than to the source's stored BF16 vector
gap = (norm(q_n) * norm(docs_n)).sum(1) - (norm(q_n) * norm(corpus[gold])).sum(1)
res = {"items": len(sel), "group_n": int(group_n.sum()), "group_b": int((~group_n).sum()),
       "self_gap_mean": float(gap.mean()), "self_gap_p50": float(np.median(gap)), "self_gap_p95": float(np.percentile(gap, 95))}
n_rows = set(gold[group_n].tolist())
for name, corp in (("stored", corpus), ("mixed", mixed)):
    top = topk(q_n, corp, 10, "transcripts")
    r1, r10, mrr = rank_metrics(top, gold)
    res[name] = {g: {"recall@1": float(r1[m].mean()), "recall@10": float(r10[m].mean()), "mrr@10": float(mrr[m].mean())}
                 for g, m in (("N", group_n), ("B", ~group_n))}
    # B-group queries whose top-1 is another sampled item from group N (BF16 in "stored", NVFP4 in "mixed")
    intr = [(int(t[0]) in n_rows) and int(t[0]) != int(g) for t, g, isb in zip(top, gold, ~group_n) if isb]
    res[name]["B_queries_top1_other_N_item"] = float(np.mean(intr))
outp.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res))
