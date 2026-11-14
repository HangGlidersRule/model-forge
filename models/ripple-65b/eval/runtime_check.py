#!/usr/bin/env python3
"""Ripple-65B Preview (#39): record the runtime facts the card's caveat states, from GitHub and Docker Hub.

Checks (network; run on a workstation with the GitHub CLI authenticated):
  - vllm-project/vllm#65745 (Ripple support): merged, its merge commit;
  - the measured nightly's commit contains that merge (compare: ahead, behind 0);
  - the v0.34.0rc1 git tag: its commit, and that it contains the merge;
  - Docker Hub (the project's OpenAI-server image): whether a v0.34.0rc1 tag exists, and the index digest of the
    measured nightly tag.
Writes results/runtime-check.json with the UTC date it ran. Usage: runtime_check.py OUT_JSON
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PR = 65745
NIGHTLY_COMMIT = "f5d93296c8ec82b7e7dabdb308d24d43cb1907bf"
NIGHTLY_DIGEST = "sha256:902b49de0568ba5a5c951b0243bc06ef8541f46bc3eb80832961fefb22a19034"
TAG = "v0.34.0rc1"
HUB = "https://hub.docker.com/v2/namespaces/vllm/repositories/vllm-openai/tags/"


def gh(path: str, jq: str) -> list[str]:
    out = subprocess.run(["gh", "api", path, "--jq", jq], capture_output=True, text=True, check=True).stdout
    return out.split()


def hub_tag(name: str) -> dict | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(HUB + name, headers={"User-Agent": "model-forge"}),
                                    timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def main() -> int:
    merged_at, merge_sha = gh(f"repos/vllm-project/vllm/pulls/{PR}", ".merged_at, .merge_commit_sha")
    tag_sha = gh(f"repos/vllm-project/vllm/git/refs/tags/{TAG}", ".object.sha")[0]
    status_n, behind_n, ahead_n = gh(f"repos/vllm-project/vllm/compare/{merge_sha}...{NIGHTLY_COMMIT}",
                                     ".status, .behind_by, .ahead_by")
    status_t, behind_t = gh(f"repos/vllm-project/vllm/compare/{merge_sha}...{tag_sha}", ".status, .behind_by")
    nightly = hub_tag(f"nightly-{NIGHTLY_COMMIT}")
    release = hub_tag(TAG)
    out = {
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "pr": PR, "pr_merged_at": merged_at, "pr_merge_commit": merge_sha,
        "nightly_commit": NIGHTLY_COMMIT, "nightly_contains_merge": status_n in ("ahead", "identical") and behind_n == "0",
        "nightly_commits_after_merge": int(ahead_n),
        "nightly_tag_index_digest": (nightly or {}).get("digest"),
        "nightly_digest_matches_measured": (nightly or {}).get("digest") == NIGHTLY_DIGEST,
        "release_tag": TAG, "release_tag_commit": tag_sha,
        "release_tag_contains_merge": status_t in ("ahead", "identical") and behind_t == "0",
        "release_docker_image_exists": release is not None,
    }
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
