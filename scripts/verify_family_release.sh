#!/usr/bin/env bash
# verify_family_release.sh — post-release surface reconciliation for one Darkstar family.
#
# Verifies ALL publication surfaces are live and consistent, not just the weights repo:
#   1. Each owned HF repo resolves publicly (200) and carries README.md + model.safetensors.index.json
#   2. The catalog row for the family exists on the PUBLIC repo's README.md on main
#   3. An HF collection page contains every owned repo id
#   4. The GitHub release tag exists on the public repo and is reachable
#
# Usage:
#   scripts/verify_family_release.sh <family_slug> [<owned_repo_id> ...]
# Example:
#   scripts/verify_family_release.sh nemotron-3-nano-omni \
#     HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-BF16 \
#     HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Base-ModelOpt-W4A16-NVFP4 \
#     HangGlidersRule/Darkstar-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Abliterated-ModelOpt-W4A16-NVFP4
#
# Exit codes: 0 all surfaces verified; 1 any surface failed; 2 usage error.
set -uo pipefail

program="${0##*/}"
fail() { printf '%s\n' "$program: $1" >&2; exit 2; }

(( $# >= 2 )) || fail "usage: $program <family_slug> <owned_repo_id> [...]"
family=$1
shift
repos=("$@")

readonly HF_USER="HangGlidersRule"
readonly GH_REPO="HangGlidersRule/model-forge"
readonly FAMILY_COLLECTION_QUERY="darkstar-${family//_/-}"
readonly RELEASE_TAG_DEF="darkstar-${family//_/-}-v1.0.0"
# Overridable release tag (family slugs do not always map 1:1; Lightning's tag has -3.5-...)
readonly RELEASE_TAG="${RELEASE_TAG_OVERRIDE:-$RELEASE_TAG_DEF}"
COLLECTION_SLUG_OVERRIDE="${COLLECTION_SLUG_OVERRIDE:-}"

status=0
report() { printf '  %-4s %s\n' "$1" "$2"; }

say() { printf '[%s] %s\n' "$family" "$1"; }

## 1. Owned HF repos: public 200 + README + index
for repo in "${repos[@]}"; do
  say "repo $repo"
  code=$(curl -s -o /dev/null -w '%{http_code}' "https://huggingface.co/$repo")
  if [[ "$code" != "200" ]]; then report "FAIL" "repo not public/200 ($code)"; status=1; else report "ok" "repo public 200"; fi
  for file in README.md model.safetensors.index.json; do
    code=$(curl -sL -o /dev/null -w '%{http_code}' "https://huggingface.co/$repo/resolve/main/$file")
    if [[ "$code" != "200" ]]; then report "FAIL" "$file missing ($code)"; status=1; else report "ok" "$file"; fi
  done
done

## 2. Catalog row on the public main README
say "catalog row on $GH_REPO main"
body=$(curl -sL "https://raw.githubusercontent.com/$GH_REPO/main/README.md") || body=""
if printf '%s' "$body" | grep -q "$family"; then
  report "ok" "family slug present in README catalog"
else
  report "FAIL" "no catalog row mentioning '$family' on main README"
  status=1
fi

## 3. HF collection membership
# Collection slug is discovered from the family record README on the public repo (the record
# is the source of truth it links), then membership is verified via the collection items API.
say "HF collection"
# the record dir may carry a lineage suffix (e.g. -r1) vs the family slug
export HF_FAMILY="$family"
record_family=$(curl -sL "https://api.github.com/repos/$GH_REPO/contents/models?ref=main" | python3 -c 'import json,sys,os
key = (os.environ.get("HF_FAMILY", "") or "").replace("darkstar-", "").lower()
items = json.load(sys.stdin) or []
best = ""
for it in items:
    n = (it.get("name") or "") if isinstance(it, dict) else ""
    if key and key in n:
        best = n
        break
print(best)')
record_url="https://raw.githubusercontent.com/$GH_REPO/main/models/${record_family:-$family}/README.md"
record_md=$(curl -sL "$record_url")
collection_slug=$(printf '%s' "$record_md" | grep -oE "huggingface\.co/collections/$HF_USER/[A-Za-z0-9-]+" | head -1 | cut -d/ -f3-)
if [[ -z "$collection_slug" ]]; then
  report "FAIL" "family record README does not link its collection"
  status=1
else
  members=$(curl -sL "https://huggingface.co/api/collections/$collection_slug" | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)
for it in d.get("items") or []:
    iid = it.get("id") or it.get("item", {}).get("id") or ""
    if iid:
        print(iid)' 2>/dev/null || true)
  for repo in "${repos[@]}"; do
    if printf '%s' "$members" | grep -qx "$repo"; then
      report "ok" "collection contains $repo"
    else
      report "FAIL" "collection $collection_slug does not list $repo"
      status=1
    fi
  done
fi

## 4. GitHub release tag
say "release tag"
code=$(curl -s -o /dev/null -w '%{http_code}' "https://api.github.com/repos/$GH_REPO/releases/tags/$RELEASE_TAG")
if [[ "$code" == "200" ]]; then
  report "ok" "release $RELEASE_TAG exists"
else
  report "FAIL" "release $RELEASE_TAG not found (HTTP $code)"
  status=1
fi

if (( status )); then
  say "SURFACE AUDIT: FAILED — do not mark this family complete."
else
  say "SURFACE AUDIT: PASS — all publication surfaces live."
fi
exit $status
