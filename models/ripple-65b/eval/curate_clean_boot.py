#!/usr/bin/env python3
"""Ripple-65B Preview (#39): curate the clean-download boot smoke into the family's committed results (fails closed).

The smoke booted the clean download of the private staging repository at the uploaded commit
(results/staging-receipt.json, clean_download) with the card's serve command on the measured image, once plain and once
with DFlash2, and ran eval/serve_checks.py smoke plus one chat request against each serve. Its launcher is
scripts/clean_boot_smoke.sh (an operator script, not published).

Inputs: LOGS_DIR holds the run log (clean-boot.log, clean-boot.launched, clean-boot.done) and per mode the serve log
(clean-boot-<mode>.log) and the client log (clean-boot-check-<mode>.log); EVAL_DIR holds the client results
(clean-boot-<mode>.json). Writes into RESULTS_DIR:
  clean-boot-<mode>.json  the client result, byte for byte
  staging-receipt.json    its boot_smoke block: run times, image, the launcher's sha256 and chat request, and per mode
                          the boot time, the engine facts read from the serve log with eval/curate_window.py's patterns
                          (compared with the window's serve of the evaluated export in serve-runs.json), the checks
                          and the result file's sha256
Refuses unless the run ended 0 with CLEAN-BOOT-OK for the uploaded commit on the measured image, every check passed in
both modes, no serve log holds a fatal marker, and both result files carry the known answers and the parsed tool call.
Usage: curate_clean_boot.py LOGS_DIR EVAL_DIR RESULTS_DIR
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from curate_window import log_facts  # noqa: E402
from serve_checks import KNOWN  # noqa: E402

RESULTS = "models/ripple-65b/results"
LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "clean_boot_smoke.sh"
# mode -> the window's serve of the evaluated export under the same flags (results/serve-runs.json)
WINDOW = {"plain": ("quality", "A"), "dflash2": ("speed", "A-s1-dflash2")}
SPEC = {"plain": "none", "dflash2": "dflash"}
FATAL = re.compile(r"CUDA error|EngineDeadError|Traceback")
STAMP = r"^([0-9]{2}:[0-9]{2}:[0-9]{2}Z) "


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fail(msg: str) -> SystemExit:
    return SystemExit(f"CLEAN-BOOT-CURATE-FAILED: {msg}")


def line(log: str, pattern: str) -> re.Match[str]:
    m = re.search(STAMP + pattern + "$", log, re.M)
    if not m:
        raise fail(f"no run-log line matching {pattern!r}")
    return m


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    logs, ev, res = (Path(x) for x in sys.argv[1:])
    st = json.loads((res / "staging-receipt.json").read_text())
    g = json.loads((res / "gate-summary.json").read_text())
    runs = json.loads((res / "serve-runs.json").read_text())
    commit, name = st["uploaded_commit"], st["repository"].split("/", 1)[1]
    digest = g["image"].split("@", 1)[1]
    if (logs / "clean-boot.done").read_text().strip() != "0":
        raise fail("the run did not end 0")
    log = (logs / "clean-boot.log").read_text().replace("\r", "")
    if re.search(r"REFUSED|FAILED", log):
        raise fail("the run log holds a refusal or a failure")
    launched = (logs / "clean-boot.launched").read_text().strip()
    day = launched[:10]
    stamps = re.findall(STAMP, log, re.M)
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z", launched) or stamps != sorted(stamps) \
            or launched[11:] > stamps[0]:
        raise fail("run-log times are not in order on the launch day")
    pre = line(log, rf"PREFLIGHT-OK \| store copy {re.escape(name)}@{commit[:12]} \| image {digest}")
    ok = line(log, rf"CLEAN-BOOT-OK {re.escape(name)}@{commit[:12]}")
    launcher = LAUNCHER.read_text()
    req = re.search(r'"content":"([^"]+)"\}\],"max_tokens":([0-9]+)', launcher)
    if not req:
        raise fail("no chat request in the launcher")

    modes: dict = {}
    for mode, (kind, label) in WINDOW.items():
        boot = line(log, rf"BOOT {mode} up after ([0-9]+) s \| .*")
        chk = line(log, rf'CHECK {mode} rc=0 \| SMOKE-OK .* \| card example "finish_reason":"([a-z_]+)" \| fatal markers 0')
        if chk.group(2) != "stop":
            raise fail(f"{mode}: the chat request finished with {chk.group(2)}")
        if not (logs / f"clean-boot-check-{mode}.log").read_text().startswith("SMOKE-OK "):
            raise fail(f"{mode}: the client log holds no SMOKE-OK")
        serve_log = logs / f"clean-boot-{mode}.log"
        serve_text = serve_log.read_text(errors="replace")
        facts = log_facts(serve_log)
        fatal = len(FATAL.findall(serve_text))
        if fatal or facts["speculative_config"] != SPEC[mode] or facts["vllm_version"] != g["vllm_version"]:
            raise fail(f"{mode}: fatal markers {fatal}, speculative config {facts['speculative_config']}, "
                       f"runtime {facts['vllm_version']}")
        window = runs[kind][label]
        differs = sorted(k for k in window if k not in ("gpu", "speed_file_sha256") and facts.get(k) != window[k])
        src = ev / f"clean-boot-{mode}.json"
        r = json.loads(src.read_text())
        known = {p: r["known"][p]["text"] for p, _ in KNOWN}
        calls = r["tool_call"]["tool_calls"]
        fn = calls[0]["function"] if len(calls) == 1 else {}
        if not (r["ok"] is True and all(known[p] == want for p, want in KNOWN) and r["tool_call"]["ok"] is True
                and fn.get("name") == "get_weather" and json.loads(fn.get("arguments", "{}")).get("city") == "Paris"):
            raise fail(f"{mode}: {src.name} does not carry the known answers and the parsed tool call")
        (res / src.name).write_bytes(src.read_bytes())
        entry: dict = {"boot_seconds": int(boot.group(2)),
                       "boot_seconds_basis": "container start to the first healthy /health poll (polled every 10 s)",
                       **facts}
        if mode == "dflash2":
            n = re.search(r"num_spec_tokens=([0-9]+)", serve_text)
            if not n:
                raise fail("dflash2: no draft-token count in the serve log")
            entry["draft_tokens"] = int(n.group(1))
        entry.update({"window_serve": f"serve-runs.json {kind}.{label}", "differs_from_window_serve": differs,
                      "known_answers": known, "tool_call": {"name": fn["name"], "arguments": fn["arguments"]},
                      "chat_request_finish_reason": chk.group(2), "fatal_log_markers": fatal,
                      "result_file": f"{RESULTS}/{src.name}", "result_sha256": sha(src)})
        modes[mode] = entry

    card = st["card_commit"]
    st["boot_smoke"] = {
        "status": "passed",
        "commit_booted": commit,
        "covers_card_commit": card["commit"] if card["weights_identical_to_uploaded_commit"] else None,
        "booted_from": "the clean download above (the store copy fetched at the uploaded commit), read through a "
                       "network mount",
        "run_utc": {"launched": launched, "preflight_ok": f"{day}T{pre.group(1)}",
                    "checks_passed": f"{day}T{ok.group(1)}", "ended": f"{day}T{stamps[-1]}"},
        "image": g["image"],
        "serve": "the card's serve command, flags unchanged, with the downloaded directory as the model; DFlash2 adds "
                 "the card's speculative config with the drafter at its pinned revision as a local copy",
        "drafter": g["drafter"],
        "chat_request": {"user_turn": req.group(1), "max_tokens": int(req.group(2))},
        "fatal_markers": FATAL.pattern.split("|"),
        "client": "eval/serve_checks.py smoke (top-1 known answers, one parsed tool call)",
        "launcher": {"path": "models/ripple-65b/scripts/clean_boot_smoke.sh", "published": False,
                     "sha256": sha(LAUNCHER)},
        "modes": modes,
        "curated_by": "eval/curate_clean_boot.py",
    }
    (res / "staging-receipt.json").write_text(json.dumps(st, indent=1) + "\n")
    for mode, e in modes.items():
        print(f"{mode}: up after {e['boot_seconds']} s | {e['quantization']} weights {e['weights_gib']} GiB "
              f"KV {e['kv_cache_tokens']:,} tokens | known {list(e['known_answers'].values())} | tool call "
              f"{e['tool_call']['name']} {e['tool_call']['arguments']} | finish {e['chat_request_finish_reason']} | "
              f"differs from the window: {e['differs_from_window_serve'] or 'nothing'}")
    print(f"CLEAN-BOOT-CURATED-OK {name}@{commit[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
