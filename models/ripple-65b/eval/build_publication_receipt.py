#!/usr/bin/env python3
"""Ripple-65B Preview (#39): build results/publication-receipt.json from the raw evidence of the public release (fails
closed).

Every value is read, none typed: the live repository and both of its collections on the Hugging Face hub and the GitHub
release (each read anonymously, without a token), the record's committed files (results/staging-receipt.json,
results/gate-summary.json, results/serve-runs.json, the ledger and the publication card), and two directories of raw
outputs:
  DOWNLOAD_DIR  clean-download-public.json and fetch.log: the anonymous clean download of the public commit
                (fetch_to_store.py with no token and an empty HF_HOME into a fresh directory; every LFS file checked
                against the hub's sha256 during the fetch, then every manifest.sha256 entry re-hashed and compared with
                the commit's manifest and with the export's);
  BOOT_DIR      the boot smoke of that download on the measured image: window.launched, window.log (only its lines up
                to the smoke's end marker), the launcher window_h2h.sh (the pinned image, the store copy and drafter it
                mounts, the client's sha256, the chat request) and per mode the serve log clean-boot-<mode>.log, the
                client log clean-boot-check-<mode>.log and the client result clean-boot-public-<mode>.json.
Writes results/publication-receipt.json and copies each client result to results/clean-boot-public-<mode>.json byte for
byte. Refuses unless the repository is public at the commit the download fetched and the smoke booted, every file and
every card link answers 200 anonymously, the card reads back equal to the publication card, the shards are the uploaded
ones, both collections list the repository publicly, the release tag sits on the record's merge commit with every asset
equal to its SHA256SUMS line, the download matched the commit's manifest on every entry, and both boots passed every
check with no fatal log line. No host name, local path or user name from the inputs is written.
Run from the repository root (network). Usage: build_publication_receipt.py DOWNLOAD_DIR BOOT_DIR
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curate_window import log_facts, serve_args  # noqa: E402
from serve_checks import KNOWN  # noqa: E402

F = Path("models/ripple-65b")
R = F / "results"
PID = "Darkstar-Ripple-65B-Preview-Base-ModelOpt-W4A4-NVFP4-Mixed-FP8"
REPO = f"HangGlidersRule/{PID}"
GH_REPO = "HangGlidersRule/model-forge"
TAG = "darkstar-ripple-65b-v1.0.0"
# The files #39 promised on the release: the BF16 teacher's output and the held-out prompts behind the KL numbers.
RELEASE_FILES = ("SHA256SUMS", "bf16_teacher_64.json", "ripple39_heldout_64.jsonl")
RELEASES_TITLE = "Darkstar Releases"
HF = "https://huggingface.co"
GH = "https://github.com"
GH_API = "https://api.github.com"
# mode -> the window's serve of the evaluated export under the same flags (results/serve-runs.json), its speculative
# config as the engine logs it
MODES = {"plain": ("quality", "A", "none"), "dflash2": ("speed", "A-s1-dflash2", "dflash")}
# The KV pool depends on the memory free when the engine profiles it; every other engine fact must equal the window's.
KV_POOL = ("kv_cache_tokens", "kv_cache_gib")
FATAL = re.compile(r"CUDA error|EngineDeadError|Traceback")
STAMP = r"([0-9]{2}:[0-9]{2}:[0-9]{2})Z "
UA = {"User-Agent": "model-forge-release-check"}


class _KeepMethod(urllib.request.HTTPRedirectHandler):
    """Follow redirects with the request's own method, so a HEAD stays a HEAD on the CDN (no weight download)."""

    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any,
                         newurl: str) -> urllib.request.Request | None:
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            new.method = req.get_method()
        return new


OPENER = urllib.request.build_opener(_KeepMethod)


def fail(msg: str) -> SystemExit:
    return SystemExit(f"PUBLICATION-RECEIPT-FAILED: {msg}")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def when(stamp: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def fetch(url: str, method: str = "GET") -> tuple[int, bytes]:
    """One anonymous request (no token, no cookie). The answers that mean throttling (429, 502 to 504) are retried with
    backoff before they are reported; any other status is final."""
    headers = dict(UA, Accept="application/vnd.github+json") if url.startswith(GH_API) else dict(UA)
    code = 0
    for attempt in range(6):
        if attempt:
            time.sleep(4 * attempt)
        try:
            with OPENER.open(urllib.request.Request(url, method=method, headers=headers), timeout=180) as r:
                return r.status, (r.read() if method == "GET" else b"")
        except urllib.error.HTTPError as e:
            code = e.code
            if code not in (429, 502, 503, 504):
                return code, b""
        except urllib.error.URLError:
            code = 0
    return code, b""


def get(url: str) -> bytes:
    code, body = fetch(url)
    if code != 200:
        raise fail(f"{url}: HTTP {code}")
    return body


def get_json(url: str) -> object:
    return json.loads(get(url))


def manifest_entries(raw: bytes) -> dict[str, str]:
    out = {}
    for line in raw.split(b"\n"):
        if line.strip():
            digest, name = line.decode().split("  ", 1)
            out[name] = digest
    return out


def hub_tree(rev: str) -> dict[str, dict]:
    tree = get_json(f"{HF}/api/models/{REPO}/tree/{rev}?recursive=true")
    assert isinstance(tree, list)
    return {x["path"]: x for x in tree if x.get("type") == "file"}


def repository(st: dict, card: bytes, immutable: dict[str, str]) -> dict:
    """The public repository: head, visibility, files, shards against the weights commit, manifest, anonymous reads,
    the card read back and its links."""
    info = get_json(f"{HF}/api/models/{REPO}")
    assert isinstance(info, dict)
    head = info["sha"]
    if info.get("private") is not False or info.get("gated") not in (False, None) or info.get("disabled"):
        raise fail(f"{REPO} is not public: private={info.get('private')} gated={info.get('gated')}")
    weights = st["uploaded_commit"]
    if head != st["card_commit"]["commit"]:
        raise fail(f"public head {head} is not the card commit the staging receipt verified")
    t_head, t_weights = hub_tree(head), hub_tree(weights)
    shards = {p: x["lfs"]["oid"] for p, x in t_head.items() if p.endswith(".safetensors")}
    shards_weights = {p: x["lfs"]["oid"] for p, x in t_weights.items() if p.endswith(".safetensors")}
    if not shards or shards != shards_weights or shards != st["clean_download"]["safetensors_sha256"]:
        raise fail("the shards' LFS sha256s differ between the public head, the weights commit and the upload")
    if any(immutable.get(p) != h for p, h in shards.items()):
        raise fail("the shards' LFS sha256s are not the ledger's immutable hashes")
    changed = sorted(p for p in set(t_head) | set(t_weights)
                     if (t_head.get(p) or {}).get("oid") != (t_weights.get(p) or {}).get("oid"))
    if changed != ["README.md", "manifest.sha256"]:
        raise fail(f"files changed since the weights commit: {changed}")
    log = get_json(f"{HF}/api/models/{REPO}/commits/main")
    assert isinstance(log, list)
    ids = [c["id"] for c in log]
    if not ids or ids[0] != head or weights not in ids:
        raise fail("the weights commit is not an ancestor of the public head on main")
    chain = [{"commit": c["id"], "title": c["title"]} for c in log[: ids.index(weights) + 1]]
    for c in chain[1:-1]:  # every commit between the head and the weights commit is card-only too
        t_c = hub_tree(c["commit"])
        moved = sorted(p for p in set(t_c) | set(t_weights)
                       if (t_c.get(p) or {}).get("oid") != (t_weights.get(p) or {}).get("oid"))
        if moved != ["README.md", "manifest.sha256"]:
            raise fail(f"commit {c['commit'][:12]} changed {moved} against the weights commit")

    m_head = get(f"{HF}/{REPO}/resolve/{head}/manifest.sha256")
    m_weights = get(f"{HF}/{REPO}/resolve/{weights}/manifest.sha256")
    e_head, e_weights = manifest_entries(m_head), manifest_entries(m_weights)
    without_card = b"\n".join(x for x in m_head.split(b"\n") if not x.endswith(b"  README.md"))
    up = st["upload"]
    checks = {
        "the head's manifest is the upload's plus the card line": without_card == m_weights,
        "the weights commit's manifest is the upload's": sha(m_weights) == up["manifest_sha256"] == immutable.get(
            "manifest.sha256"),
        "the head's manifest is the card commit's the staging receipt verified": sha(m_head) == st["card_commit"][
            "manifest_sha256"],
        "the manifest covers every file but itself": set(e_head) == set(t_head) - {"manifest.sha256"},
        "the manifest's card line is the publication card": e_head.get("README.md") == sha(card),
        "the manifest's shard lines are the LFS sha256s": all(e_head[p] == h for p, h in shards.items()),
    }
    if not all(checks.values()):
        raise fail(f"manifest: {[k for k, ok in checks.items() if not ok]}")

    page = fetch(f"{HF}/{REPO}")[0]
    statuses = {name: fetch(f"{HF}/{REPO}/resolve/{head}/{name}", "HEAD")[0] for name in sorted(t_head)}
    non_200 = sorted(k for k, v in {"card_page": page, **statuses}.items() if v != 200)
    live_card = get(f"{HF}/{REPO}/resolve/{head}/README.md")
    if non_200 or live_card != card:
        raise fail(f"anonymous reads: non-200 {non_200}; card read-back identical: {live_card == card}")

    text = live_card.decode()
    front = text.split("---", 2)[1] if text.startswith("---") else ""
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    relative = [t for t in targets if not re.match(r"https?://|mailto:|#", t)]
    urls = sorted({u.rstrip(".,;:") for u in re.findall(r"https?://[^\s)<>\]`\"']+", text)})
    links = {}
    for u in urls:
        if u.startswith(GH):
            time.sleep(1.0)  # pace GitHub's anonymous page answers (bursts get 429)
        links[u] = fetch(u)[0]
    if relative or any(v != 200 for v in links.values()):
        raise fail(f"card links: relative {relative}; non-200 {[u for u, v in links.items() if v != 200]}")
    lic = re.search(r"^license:\s*(\S+)\s*$", front, re.M)
    return {
        "repository": REPO,
        "hf_commit": head,
        "verified_commit": head,
        "hf_commit_note": (
            f"The public head is the commit the anonymous clean download fetched and the boot smoke booted (so it is "
            f"also the verified commit). It sits {len(chain) - 1} card-only commits above the weights commit "
            f"{weights[:8]}, the private upload whose clean download the staging receipt records: the head and every "
            f"commit between differ from the weights commit only in README.md and manifest.sha256, and the "
            f"{len(shards)} shards' LFS sha256s are identical at the head and the weights commit and equal to the "
            f"uploaded export's and the ledger's immutable hashes."),
        "weights_commit": weights,
        "commits_on_main_since_weights_commit": chain,
        "public": True,
        "file_count": len(t_head),
        "files": sorted(t_head),
        "shard_lfs_sha256": shards,
        "manifest": {
            "sha256": sha(m_head), "entries": len(e_head),
            "sha256_at_weights_commit": sha(m_weights), "entries_at_weights_commit": len(e_weights),
            "note": ("manifest.sha256 at the public head is the upload's manifest (the one at the weights commit, the "
                     "staging receipt's and the ledger's immutable hash) plus one README.md line for the card, byte "
                     "for byte: removing that line gives the upload's manifest. Every file of the head but the "
                     "manifest itself has a line, and each shard's line is its LFS sha256."),
            "checks": {k: True for k in checks},
        },
        "anonymous_http": {"card_page": page, **statuses},
        "anonymous_http_non_200": [],
        "anonymous_http_method": "HEAD on resolve/<public head>/<file>, redirects followed (no weight download); GET "
                                 "for the card page",
        "card_read_back_identical": True,
        "card_read_back_sha256": sha(live_card),
        "card_read_back_note": (f"README.md at the public head is byte-identical to publication/huggingface/{PID}/"
                                "README.md and to models/ripple-65b/model-card/base-nvfp4.md in this commit."),
        "card_links": {
            "markdown_links": len(targets), "relative_links": len(relative), "unique_urls": len(urls),
            "front_matter_urls": len(re.findall(r"https?://", front)),
            "front_matter_license": lic.group(1) if lic else None,
            "anonymous_http": links,
            "note": (f"Counted on the card at the public head: {len(targets)} markdown links, {len(urls)} unique URLs "
                     "(an upstream link appears twice). The front matter names a license the hub lists, so it "
                     "carries no license URL. Every unique URL answered 200 to an anonymous GET."),
        },
    }


def collections(family_title: str) -> dict:
    """Both collections that list the repository, read back anonymously (a private one is not listed at all)."""
    listed = get_json(f"{HF}/api/collections?item=models/{REPO}&limit=100")
    assert isinstance(listed, list)
    by_title = {c["title"]: c["slug"] for c in listed}
    out = {}
    for key, title in (("family_collection", family_title), ("releases_collection", RELEASES_TITLE)):
        slug = by_title.get(title)
        if not slug:
            raise fail(f"no public collection titled {title!r} lists {REPO}")
        c = get_json(f"{HF}/api/collections/{slug}")
        assert isinstance(c, dict)
        items = [i for i in c.get("items", []) if i.get("id") == REPO]
        page = fetch(f"{HF}/collections/{slug}")[0]
        if c.get("private") is not False or len(items) != 1 or page != 200:
            raise fail(f"collection {slug}: private={c.get('private')} lists repo {len(items)} page {page}")
        note = (items[0].get("note") or {}).get("text")
        # A collection's slug ends in its object id, whose first 8 hex digits are its creation time in Unix seconds.
        made = datetime.datetime.fromtimestamp(int(slug.rsplit("-", 1)[1][:8], 16), datetime.timezone.utc)
        out[key] = {"slug": slug, "title": title, "url": f"{HF}/collections/{slug}", "public": True,
                    "lists_repository": True, "items": len(c.get("items", [])),
                    "created_utc": made.strftime("%Y-%m-%dT%H:%M:%SZ"), "last_updated": c.get("lastUpdated"),
                    **({"repository_note": note} if note else {})}
    return out


def github_release(heldout_sha: str) -> dict:
    """The release tag and its GitHub release: the tag's commit is the record PR's merge commit, and every asset
    (downloaded anonymously) hashes to its SHA256SUMS line and to the digest GitHub reports."""
    rel = get_json(f"{GH_API}/repos/{GH_REPO}/releases/tags/{TAG}")
    ref = get_json(f"{GH_API}/repos/{GH_REPO}/git/ref/tags/{TAG}")
    assert isinstance(rel, dict) and isinstance(ref, dict)
    if ref["object"]["type"] != "commit":
        raise fail(f"tag {TAG} points at a {ref['object']['type']}, not a commit")
    commit = ref["object"]["sha"]
    pulls = get_json(f"{GH_API}/repos/{GH_REPO}/commits/{commit}/pulls")
    assert isinstance(pulls, list)
    merged = [p for p in pulls if p.get("merge_commit_sha") == commit and p.get("merged_at")]
    if len(merged) != 1 or rel.get("draft") or rel.get("prerelease"):
        raise fail(f"tag {TAG}: {len(merged)} merged PRs with merge commit {commit[:12]}; draft/prerelease set")
    assets, digests = {}, {}
    for a in rel["assets"]:
        body = get(a["browser_download_url"])
        if len(body) != a["size"] or (a.get("digest") and a["digest"] != f"sha256:{sha(body)}"):
            raise fail(f"asset {a['name']}: size or digest differs from what GitHub reports")
        assets[a["name"]] = {"size": a["size"], "sha256": sha(body)}
        digests[a["name"]] = body
    if sorted(assets) != sorted(RELEASE_FILES):
        raise fail(f"release assets {sorted(assets)} are not {sorted(RELEASE_FILES)}")
    sums = manifest_entries(digests["SHA256SUMS"])
    if sums != {n: assets[n]["sha256"] for n in RELEASE_FILES if n != "SHA256SUMS"}:
        raise fail("a release asset differs from its SHA256SUMS line")
    if sums["ripple39_heldout_64.jsonl"] != heldout_sha:
        raise fail("the released held-out prompts are not the file the calibration record pins")
    pr = merged[0]
    return {
        "tag": TAG, "url": rel["html_url"], "tag_commit": commit, "tag_object": "commit (a lightweight tag)",
        "record_pr": {"number": pr["number"], "merged_at": pr["merged_at"], "merge_commit": pr["merge_commit_sha"]},
        "release_published_at": rel["published_at"], "draft": False, "prerelease": False,
        "immutable": rel.get("immutable"),
        "assets": [{"name": n, **assets[n]} for n in RELEASE_FILES],
        "checks": {"every asset downloaded anonymously, its size and sha256 equal to what GitHub reports": True,
                   "every other asset equal to its SHA256SUMS line": True,
                   "ripple39_heldout_64.jsonl equal to the held-out file results/calibration-record.json pins": True},
        "note": ("The tag points at the merge commit of the record PR. GitHub reports the release as not immutable "
                 "(the immutable field above); #104 tracks enforcing release immutability."
                 if rel.get("immutable") is False else "The tag points at the merge commit of the record PR."),
    }


def clean_download(dl: Path, head: str, shards: dict[str, str], card_sha: str, n_files: int, manifest: dict) -> dict:
    cd = load(dl / "clean-download-public.json")
    log = (dl / "fetch.log").read_text(errors="replace")
    ok = re.search(r"^PUBLIC-DOWNLOAD-OK (\{.*\})$", log, re.M)
    start = re.search(rf"^([0-9T:-]{{19}}Z) fetch {re.escape(REPO)}@{head} \(anonymous\)$", log, re.M)
    took = re.search(r"^[0-9T:-]{19}Z ripple fetch done in ([0-9]+) s$", log, re.M)
    store = [json.loads(x) for x in re.findall(r"^(\{\"dst\": .*\})$", log, re.M)]
    checks = {
        "the log's verdict line is this result": bool(ok) and json.loads(ok.group(1)) == {
            k: v for k, v in cd.items() if k != "safetensors"},
        "the public head, fetched anonymously": cd["repository"] == REPO and cd["commit"] == head and cd[
            "anonymous"] is True and bool(start),
        "the hub client logged its unauthenticated-request warning": "sending unauthenticated requests" in log,
        "every LFS file checked against the hub during the fetch": bool(store) and store[0]["lfs_mismatch"] == [] and
        store[0]["lfs_verified"] == len(shards) and store[0]["files"] == n_files,
        "every manifest entry re-hashed equal": cd["entries_matching_hub_manifest"] == cd["manifest_entries"] ==
        manifest["entries"] and cd["mismatches"] == cd["missing"] == cd["extra"] == [],
        "the manifest is the public head's": cd["manifest_sha256"] == manifest["sha256"],
        "only the card differs from the export's manifest": cd["entries_differing_from_export_manifest"] == [
            "README.md"] and cd["readme_sha256"] == card_sha and cd["readme_equals_publication_card"] is True,
        "the shards are the public head's": cd["safetensors"] == shards and cd["files_downloaded"] == n_files,
        "verdict": cd["verdict"] == "PASS",
    }
    if not all(checks.values()):
        raise fail(f"clean download: {[k for k, v in checks.items() if not v]}")
    return {
        "commit": head, "anonymous": True, "verdict": "PASS",
        "started_utc": start.group(1) if start else None, "fetch_seconds": int(took.group(1)) if took else None,
        "files_downloaded": cd["files_downloaded"], "lfs_files_checked_during_fetch": store[0]["lfs_verified"],
        "manifest_entries": cd["manifest_entries"], "manifest_sha256": cd["manifest_sha256"],
        "entries_matching_hub_manifest": cd["entries_matching_hub_manifest"], "mismatches": [], "missing": [],
        "extra": [], "entries_differing_from_export_manifest": cd["entries_differing_from_export_manifest"],
        "readme_sha256": cd["readme_sha256"], "readme_equals_publication_card": True, "bytes": cd["bytes"],
        "method": ("fetch_to_store.py of the public head with no token (the hub client logged its warning about "
                   "unauthenticated requests) and an empty HF_HOME, into a fresh store directory; every LFS file "
                   "checked against the hub's sha256 during the fetch, then every manifest.sha256 entry re-hashed and "
                   "compared with the public commit's manifest and with the export's manifest"),
        "checks": {k: True for k in checks},
    }


def boot_smoke(boot: Path, head: str, st: dict, g: dict, runs: dict, alias: str) -> dict:
    launched = (boot / "window.launched").read_text().strip()
    lines = (boot / "window.log").read_text().replace("\r", "").split("\n")
    end_mark = f"H2H-SMOKE-OK public download {REPO}@{head}:"
    end = next((i for i, x in enumerate(lines) if end_mark in x), None)
    if end is None:
        raise fail(f"window.log: no {end_mark!r} line")
    smoke = lines[: end + 1]  # the later lines are another measurement, not the smoke
    log = "\n".join(smoke)
    if re.search(r"FAILED|ABORT", log):
        raise fail("window.log: a failure or an abort before the smoke's end marker")
    stamps = re.findall(rf"^{STAMP}", log, re.M)
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z", launched) or stamps != sorted(stamps) or \
            launched[11:19] > stamps[0]:
        raise fail("window.log: times are not in order on the launch day")
    day = launched[:10]
    if not re.search(rf"^{STAMP}H2H-PREFLIGHT-OK \| ripple {re.escape(REPO)}@{head} \|", log, re.M):
        raise fail("window.log: the preflight did not name the public head's store copy")
    launcher = (boot / "window_h2h.sh").read_text().replace("\r", "")

    def var(name: str) -> str:
        m = re.search(rf"^{name}=(\S+)$", launcher, re.M)
        if not m:
            raise fail(f"launcher: no {name}= line")
        return m.group(1)

    client = F / "eval" / "serve_checks.py"
    req = re.search(r'"model":"([^"]+)","messages":\[\{"role":"user","content":"([^"]+)"\}\],"max_tokens":([0-9]+)',
                    launcher)
    checks = {
        "the pinned image is the measured one": var("VLLM") == g["image"],
        "the store copy is the public head's download": var("RIPPLE_ID") == f"{REPO}@{head}",
        "the drafter is the pinned revision": var("DRAFTER").endswith("/" + g["drafter"]),
        "the client is this commit's serve_checks.py": var("CHECKS_SHA") == sha(client.read_bytes()),
        "the chat request is the staging smoke's": bool(req) and req.group(1) == alias and {
            "user_turn": req.group(2), "max_tokens": int(req.group(3))} == st["boot_smoke"]["chat_request"],
    }
    if not all(checks.values()):
        raise fail(f"launcher: {[k for k, v in checks.items() if not v]}")
    modes: dict = {}
    times = {}
    for mode, (kind, label, spec) in MODES.items():
        up = re.search(rf"^{STAMP}SERVE smoke-{mode} up after ([0-9]+) s \| GPU KV cache size: ([0-9,]+) tokens "
                       rf"Model loading took ([0-9.]+) GiB", log, re.M)
        chk = re.search(rf'^{STAMP}CHECK {mode} rc=0 \| SMOKE-OK .* \| card example "finish_reason":"([a-z_]+)" \| '
                        rf"fatal markers ([0-9]+)$", log, re.M)
        if not up or not chk or chk.group(2) != "stop" or chk.group(3) != "0":
            raise fail(f"{mode}: no boot line, or the checks did not pass with the chat request finishing 'stop'")
        times[mode] = (up.group(1), chk.group(1))
        serve_log = boot / f"clean-boot-{mode}.log"
        text = serve_log.read_text(errors="replace")
        facts = log_facts(serve_log)
        args = serve_args(serve_log)
        fatal = len(FATAL.findall(text))
        window = runs[kind][label]
        differs = sorted(k for k in window if k not in ("gpu", "speed_file_sha256", "serve_args")
                         and facts.get(k) != window[k])
        args_differ = sorted(k for k in window["serve_args"] if k != "log_sha256"
                             and args.get(k) != window["serve_args"][k])
        if fatal or facts["vllm_version"] != g["vllm_version"] or facts["speculative_config"] != spec or \
                set(differs) - set(KV_POOL) or args_differ != ["served_model_name"] or \
                args["served_model_name"] != [alias] or facts["kv_cache_tokens"] != int(up.group(3).replace(",", "")) \
                or facts["weights_gib"] != float(up.group(4)):
            raise fail(f"{mode}: fatal markers {fatal}, runtime {facts['vllm_version']}, speculative config "
                       f"{facts['speculative_config']}, differs from the window serve {differs}, serve args differ "
                       f"{args_differ}")
        if not (boot / f"clean-boot-check-{mode}.log").read_text().startswith("SMOKE-OK "):
            raise fail(f"{mode}: the client log holds no SMOKE-OK")
        src = boot / f"clean-boot-public-{mode}.json"
        r = load(src)
        known = {p: r["known"][p]["text"] for p, _ in KNOWN}
        calls = r["tool_call"]["tool_calls"]
        fn = calls[0]["function"] if len(calls) == 1 else {}
        if not (r["ok"] is True and all(known[p] == want for p, want in KNOWN) and r["tool_call"]["ok"] is True
                and fn.get("name") == "get_weather" and json.loads(fn.get("arguments", "{}")).get("city") == "Paris"):
            raise fail(f"{mode}: {src.name} does not carry the known answers and the parsed tool call")
        (R / src.name).write_bytes(src.read_bytes())
        entry: dict = {"boot_seconds": int(up.group(2)),
                       "boot_seconds_basis": "container start to the first healthy /health poll (polled every 10 s)",
                       **facts, "serve_args": args}
        if mode == "dflash2":
            n = re.search(r"num_spec_tokens=([0-9]+)", text)
            if not n:
                raise fail("dflash2: no draft-token count in the serve log")
            entry["draft_tokens"] = int(n.group(1))
        entry.update({
            "window_serve": f"serve-runs.json {kind}.{label}", "differs_from_window_serve": differs,
            "kv_pool_of_window_serve": {k: window[k] for k in KV_POOL},
            "serve_args_differ_from_window_serve": args_differ, "known_answers": known,
            "tool_call": {"name": fn["name"], "arguments": fn["arguments"]},
            "chat_request_finish_reason": chk.group(2), "fatal_log_markers": fatal,
            "result_file": f"{R}/{src.name}", "result_sha256": sha(src.read_bytes())})
        modes[mode] = entry
    return {
        "status": "passed",
        "commit_booted": head,
        "booted_from": "the anonymous clean download above (the store copy fetched at the public head), read through "
                       "a network mount",
        "run_utc": {"launched": launched, "plain_up": f"{day}T{times['plain'][0]}Z",
                    "plain_checks_passed": f"{day}T{times['plain'][1]}Z",
                    "dflash2_up": f"{day}T{times['dflash2'][0]}Z",
                    "dflash2_checks_passed": f"{day}T{times['dflash2'][1]}Z"},
        "image": g["image"],
        "serve": ("the card's serve command, flags unchanged, with the downloaded directory as the model; DFlash2 adds "
                  "the card's speculative config with the drafter at its pinned revision as a local copy. serve_args "
                  "are the arguments the engine logged at startup: every flag equal to the window serve's but the "
                  "served name, which is the card's ledger alias here and was a short name in the window"),
        "drafter": g["drafter"],
        "chat_request": st["boot_smoke"]["chat_request"],
        "fatal_markers": FATAL.pattern.split("|"),
        "client": "eval/serve_checks.py smoke (top-1 known answers, one parsed tool call), this commit's file by sha256",
        "client_sha256": sha(client.read_bytes()),
        "launcher_checks": {k: True for k in checks},
        "modes": modes,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    dl, boot = Path(sys.argv[1]), Path(sys.argv[2])
    st = load(R / "staging-receipt.json")
    g = load(R / "gate-summary.json")
    runs = load(R / "serve-runs.json")
    ledger = load(R / "publication-readiness-ledger.json")
    heldout = load(R / "calibration-record.json")["outputs"]["ripple39_heldout_64.jsonl"]
    product = next(p for p in ledger["products"] if p["product_id"] == PID)
    selected = next(c for c in product["candidates"] if c["selection"] == "selected")
    card = (Path("publication/huggingface") / PID / "README.md").read_bytes()
    if card != (F / "model-card" / "base-nvfp4.md").read_bytes():
        raise fail("the publication card differs from the source card")
    repo = repository(st, card, selected["immutable_hashes"])
    head = repo["hf_commit"]
    cols = collections(ledger["family_display"])
    rel = github_release(heldout)
    cd = clean_download(dl, head, repo["shard_lfs_sha256"], sha(card), repo["file_count"], repo["manifest"])
    bs = boot_smoke(boot, head, st, g, runs, product["served_model_alias"])
    pr = rel["record_pr"]
    made = cols["family_collection"]["created_utc"]
    if not (when(pr["merged_at"]) <= when(rel["release_published_at"]) <= when(made)):
        raise fail(f"order: merged {pr['merged_at']}, release published {rel['release_published_at']}, "
                   f"family collection created {made}")
    receipt = {
        **repo,
        "made_public_after": {
            "record_pr": pr["number"], "merge_commit": pr["merge_commit"], "merged_at": pr["merged_at"],
            "release_tag": rel["tag"], "tag_commit": rel["tag_commit"],
            "release_published_at": rel["release_published_at"], "family_collection_created_utc": made,
            "note": ("The hub records no time for a visibility change. The flip came after the merge and the tag: the "
                     "tag is lightweight (no date of its own) and was created with its release, the flip step refuses "
                     "while any GitHub link on the card (the release tag page among them) does not answer 200, and "
                     "the family collection, which the step right after the flip created, dates from after the "
                     "release was published."),
        },
        **cols,
        "github_release": rel,
        "clean_download": cd,
        "boot_smoke": bs,
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "built_by": "eval/build_publication_receipt.py",
    }
    (R / "publication-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    p, d = bs["modes"]["plain"], bs["modes"]["dflash2"]
    print(f"public {REPO}@{head[:12]} | {repo['file_count']} files 200 | card identical | links "
          f"{repo['card_links']['markdown_links']}/{repo['card_links']['unique_urls']} 200 | release "
          f"{rel['tag']} -> {rel['tag_commit'][:12]} (#{pr['number']}), {len(rel['assets'])} assets = SHA256SUMS")
    print(f"clean download {cd['entries_matching_hub_manifest']}/{cd['manifest_entries']} | plain up {p['boot_seconds']} s "
          f"{p['weights_gib']} GiB KV {p['kv_cache_tokens']:,} | dflash2 up {d['boot_seconds']} s {d['weights_gib']} GiB "
          f"KV {d['kv_cache_tokens']:,} | differs from the window: {p['differs_from_window_serve']} / "
          f"{d['differs_from_window_serve']}")
    print(f"PUBLICATION-RECEIPT-OK {REPO}@{head[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
