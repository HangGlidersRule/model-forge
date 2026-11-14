#!/usr/bin/env python3
"""Ripple-65B Preview (#39): record a card-only commit of the private staging repository in results/staging-receipt.json.

Reads the read-back of the commit (scripts' hf_publish.py card, then a read-only check of the repository at the new
head: still private, every shard's LFS sha256 unchanged against the previous card commit and the uploaded commit, the hub
README.md equal to the staged card, every file equal to its manifest.sha256 line) and refuses unless every check held.
The previous card commit moves to card_commit.history; the card_commit block then names the new commit, its
manifest.sha256 sha256 and the staged card's sha256. boot_smoke.covers_card_commit follows the new commit when the
weights are unchanged (the smoke booted those exact shards). Usage: record_card_commit.py READBACK_JSON STAGED_CARD
Run from the repository root.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

RESULTS = Path("models/ripple-65b/results")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    rb = json.loads(Path(sys.argv[1]).read_text())
    card = Path(sys.argv[2])
    st = json.loads((RESULTS / "staging-receipt.json").read_text())
    prev = st["card_commit"]
    card_sha = hashlib.sha256(card.read_bytes()).hexdigest()
    checks = {"private": rb["private"] is True,
              "weights unchanged against the previous card commit": rb["lfs_unchanged_vs_previous_card_commit"] is True,
              "weights unchanged against the uploaded commit": rb["lfs_unchanged_vs_uploaded_commit"] is True,
              "5 shards": rb["shards"] == 5,
              "every file equal to its manifest line": rb["entries_not_matching_hub"] == [],
              "hub README equals the staged card": rb["hub_readme_sha256"] == card_sha == rb["staged_card_sha256"],
              "parent is the previous card commit": [c[0] for c in rb["commits"][:2]] == [rb["head"][:12],
                                                                                         prev["commit"][:12]]}
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"RECORD-CARD-FAILED: {failed}")
    history = list(prev.get("history", [])) + [{k: prev[k] for k in ("commit", "manifest_sha256")}]
    st["card_commit"] = {
        "commit": rb["head"], "parent_card_commit": prev["commit"], "parent_weights_commit": prev["parent_weights_commit"],
        "changed_files": ["README.md", "manifest.sha256"], "files_on_hub": rb["files"],
        "manifest_sha256": rb["manifest_sha256"], "manifest_entries_verified_on_hub": rb["manifest_entries"],
        "readme_sha256": card_sha, "weights_identical_to_uploaded_commit": True,
        "reason": "the card's GPQA row and protocol sentence (items without an answer at the token cap; review round 1)",
        "method": ("hf_publish.py card (README.md and manifest.sha256 read back); then every hub file at the head checked "
                   "against its manifest.sha256 (the 5 shards by LFS sha256, the small files by their downloaded bytes), "
                   "the shard LFS oids identical to the previous card commit and to the uploaded commit "
                   "(hf_publish.py same-weights), and the hub README.md equal to the staged card"),
        "history": history,
    }
    st["boot_smoke"]["covers_card_commit"] = rb["head"]
    (RESULTS / "staging-receipt.json").write_text(json.dumps(st, indent=1) + "\n")
    print(f"RECORD-CARD-OK {rb['head'][:12]} manifest {rb['manifest_sha256'][:12]} README {card_sha[:12]} "
          f"(previous {prev['commit'][:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
