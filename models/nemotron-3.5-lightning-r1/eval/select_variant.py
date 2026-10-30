#!/usr/bin/env python3
"""Issue #17 R17-A1 selection: lowest selection-set KL wins if it beats A's by >= 10%.
Reads kl-sel-<label>.json for BF16, A and the variants; writes sel-result.json and prints the winner (or NONE).
Usage: select_variant.py EVAL_DIR VARIANT [VARIANT...]"""
import json
import math
import sys
from pathlib import Path

ev, variants = Path(sys.argv[1]), sys.argv[2:]


def kl(label):
    ref = json.loads((ev / "kl-sel-BF16.json").read_text())["rows"]
    cand = json.loads((ev / f"kl-sel-{label}.json").read_text())["rows"]
    s, n = 0.0, 0
    for r_row, c_row in zip(ref, cand):
        for r, c in zip(r_row, c_row):
            if not r or not c:
                continue
            p = {t: math.exp(v) for t, v in r.items()}
            q = {t: math.exp(v) for t, v in c.items()}
            pr = max(1e-12, 1 - sum(p.values()))
            qr = max(1e-12, 1 - sum(q.get(t, 0.0) for t in p))
            fl = min(q.values())
            s += sum(pv * (math.log(pv) - math.log(q.get(t, fl))) for t, pv in p.items()) + pr * (math.log(pr) - math.log(qr))
            n += 1
    return s / n


res = {"A": kl("A")}
for v in variants:
    if (ev / f"kl-sel-{v}.json").exists():
        res[v] = kl(v)
best = min((v for v in variants if v in res), key=lambda v: res[v], default=None)
winner = best if best is not None and res[best] <= 0.9 * res["A"] else "NONE"
out = {"selection_kl": {k: round(v, 6) for k, v in res.items()}, "best": best, "winner": winner,
       "rule": "lowest selection KL; advances if <= 0.9 x A"}
(ev / "sel-result.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out))
print(winner)
