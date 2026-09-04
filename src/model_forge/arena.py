from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def export_blind(records: list[dict[str, Any]], public_path: Path, key_path: Path, *, seed: int = 42) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["case_id"])].append(record)
    rng = random.Random(seed)
    public, keys = [], {}
    for case_id, rows in sorted(grouped.items()):
        if len(rows) != 2:
            continue
        shuffled = list(rows)
        rng.shuffle(shuffled)
        pair_id = hashlib.sha256(f"{case_id}|{seed}".encode()).hexdigest()[:16]
        public.append({"pair_id":pair_id,"case_id":case_id,"response_a":shuffled[0]["response"],"response_b":shuffled[1]["response"]})
        keys[pair_id] = {"a":shuffled[0]["model_id"],"b":shuffled[1]["model_id"]}
    public_path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in public))
    key_path.write_text(json.dumps(keys,indent=2,sort_keys=True)+"\n")


def reveal_votes(votes_path: Path, key_path: Path) -> list[dict[str, Any]]:
    keys = json.loads(key_path.read_text())
    output: list[dict[str, Any]] = []
    for line in votes_path.read_text().splitlines():
        vote = json.loads(line)
        winner = vote["winner"]
        winner_model = keys[vote["pair_id"]][winner] if winner in {"a", "b"} else winner
        output.append({**vote, "winner_model": winner_model})
    return output
