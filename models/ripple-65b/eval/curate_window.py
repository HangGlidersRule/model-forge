#!/usr/bin/env python3
"""Ripple-65B Preview (#39): curate the GPU window's outputs into the family's committed results (aggregates only).

Inputs: the window's eval/ directory (proxy-<cell>.json, speed/<label>.json, speed/gpu-<label>.csv, gpu-window.csv) and
its logs/ directory (serve-<label>.log). Writes into RESULTS_DIR:
  proxy-<cell>.json     the tool-call proxy summary of each quality serve (per-item rows, model output and the server
                        URL dropped; the input files' sha256 kept)
  serve-runs.json       per serve: vLLM version, quantization method, KV-cache dtype, prefix caching, context length,
                        weight memory, KV pool, the GEMM kernels vLLM selected (all read from the engine config and
                        load lines of the serve log), the non-default arguments of the serve as the engine logged them
                        at startup (max_num_seqs, gpu_memory_utilization, max_model_len, the reasoning and tool-call
                        parsers, auto tool choice, the speculative config, the served name), and the GPU sampler
                        summary of each speed serve (power limit, median board power and SM clock under load, peak
                        memory)
The speed, gate, build, validation and smoke JSONs are copied byte for byte beside these (see the family README); this
script records their sha256 in serve-runs.json so a reader can match them to the window's files.
Usage: curate_window.py WINDOW_EVAL_DIR WINDOW_LOGS_DIR RESULTS_DIR
       curate_window.py --serve-args WINDOW_LOGS_DIR RESULTS_DIR   (adds only the serve_args block to an existing
                                                                     serve-runs.json, every other byte kept)
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

SPEED_LABELS = ("vendor-s1", "vendor-s1-dflash2", "A-s1", "A-s1-dflash2",
                "vendor-s2", "vendor-s2-dflash2", "A-s2", "A-s2-dflash2")
QUALITY_SERVES = ("vendor", "A")
PATTERNS = {
    "vllm_version": r"Initializing a V1 LLM engine \(v([0-9A-Za-z.+]+)\)",
    "quantization": r"quantization=([a-z_0-9]+)",
    "kv_cache_dtype": r"kv_cache_dtype=([a-z0-9_]+)",
    "enable_prefix_caching": r"enable_prefix_caching=(True|False)",
    "max_seq_len": r"max_seq_len=([0-9]+)",
    "weights_gib": r"Model loading took ([0-9.]+) GiB",
    "kv_cache_gib": r"Available KV cache memory: ([0-9.]+) GiB",
    "kv_cache_tokens": r"GPU KV cache size: ([0-9,]+) tokens",
}
KERNELS = r"(Using [A-Za-z0-9]+ for NVFP4 GEMM|Selected [A-Za-z0-9]+ for [A-Za-z0-9]+)"
# The startup line of the engine with the serve arguments (every flag that differs from the default), a Python dict
# literal.
SERVE_ARGS = re.compile(r"non-default args: (\{.*\})\s*$", re.M)
# The flags the card's serve block states, besides the model path.
SERVE_ARG_KEYS = ("max_model_len", "max_num_seqs", "gpu_memory_utilization", "reasoning_parser", "tool_call_parser",
                  "enable_auto_tool_choice", "speculative_config", "served_model_name")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def serve_args(path: Path) -> dict:
    """The serve arguments as the engine logged them at startup (one non-default args line), keys SERVE_ARG_KEYS."""
    lines = SERVE_ARGS.findall(path.read_text(errors="replace"))
    if len(lines) != 1:
        raise SystemExit(f"{path.name}: {len(lines)} 'non-default args' lines, expected 1")
    args = ast.literal_eval(lines[0])
    if not isinstance(args, dict):
        raise SystemExit(f"{path.name}: the 'non-default args' line is not a dict")
    return {k: args.get(k) for k in SERVE_ARG_KEYS} | {"log_sha256": sha(path)}


def log_facts(path: Path) -> dict:
    text = path.read_text(errors="replace")
    out: dict = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, text)
        if not m:
            raise SystemExit(f"{path.name}: no match for {key}")
        v = m.group(1)
        out[key] = (int(v.replace(",", "")) if key in ("kv_cache_tokens", "max_seq_len") else
                    float(v) if key.endswith("_gib") else (v == "True") if key == "enable_prefix_caching" else v)
    out["kernels"] = sorted(set(re.findall(KERNELS, text)))
    out["speculative_config"] = "dflash" if re.search(r"speculative_config=SpeculativeConfig\(method='dflash'", text) \
        else ("none" if "speculative_config=None" in text else "unknown")
    return out


def gpu_summary(path: Path) -> dict:
    rows = list(csv.reader(path.read_text().splitlines()))
    head = [h.strip() for h in rows[0]]
    idx = {k: next(i for i, h in enumerate(head) if h.startswith(k)) for k in
           ("power.limit", "power.draw", "clocks.current.sm", "memory.used", "utilization.gpu")}

    def num(r: list[str], k: str) -> float:
        return float(re.sub(r"[^0-9.]", "", r[idx[k]]) or 0)
    data = [r for r in rows[1:] if len(r) == len(head)]
    busy = [r for r in data if num(r, "utilization.gpu") >= 90]
    return {"samples": len(data), "power_limit_w": sorted({num(r, "power.limit") for r in data}),
            "busy_samples": len(busy),
            "median_power_w_busy": round(statistics.median(num(r, "power.draw") for r in busy), 1) if busy else None,
            "median_sm_clock_mhz_busy": statistics.median(num(r, "clocks.current.sm") for r in busy) if busy else None,
            "peak_memory_mib": max(num(r, "memory.used") for r in data)}


def serve_log(logs: Path, kind: str, label: str) -> Path:
    return logs / (f"serve-{label}.log" if kind == "quality" else f"serve-speed-{label}.log")


def add_serve_args(runs: dict, logs: Path) -> dict:
    """runs with a serve_args block per serve, every other entry unchanged."""
    out = json.loads(json.dumps(runs))
    for kind in ("quality", "speed"):
        for label in out[kind]:
            out[kind][label]["serve_args"] = serve_args(serve_log(logs, kind, label))
    return out


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--serve-args":
        logs, res = Path(sys.argv[2]), Path(sys.argv[3])
        runs = add_serve_args(json.loads((res / "serve-runs.json").read_text()), logs)
        (res / "serve-runs.json").write_text(json.dumps(runs, indent=1) + "\n")
        for kind in ("quality", "speed"):
            for label, e in runs[kind].items():
                print(f"{kind} {label}: {json.dumps({k: v for k, v in e['serve_args'].items() if k != 'log_sha256'})}")
        return 0
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    ev, logs, res = (Path(x) for x in sys.argv[1:])
    for cell in QUALITY_SERVES:
        d = json.loads((ev / f"proxy-{cell}.json").read_text())
        keep = {k: d[k] for k in ("label", "model", "max_tokens", "data_sha256", "summary", "seconds")}
        keep["items_scored"] = len(d["items"])
        keep["raw_sha256"] = sha(ev / f"proxy-{cell}.json")
        (res / f"proxy-{cell}.json").write_text(json.dumps(keep, indent=1) + "\n")
        print(f"proxy-{cell}.json: {keep['summary']['overall']}")
    runs: dict = {"quality": {}, "speed": {}}
    for cell in QUALITY_SERVES:
        runs["quality"][cell] = log_facts(logs / f"serve-{cell}.log")
    for label in SPEED_LABELS:
        entry = log_facts(logs / f"serve-speed-{label}.log")
        entry["gpu"] = gpu_summary(ev / "speed" / f"gpu-{label}.csv")
        entry["speed_file_sha256"] = sha(ev / "speed" / f"{label}.json")
        runs["speed"][label] = entry
    runs["window_gpu"] = gpu_summary(ev / "gpu-window.csv")
    runs = add_serve_args(runs, logs)
    (res / "serve-runs.json").write_text(json.dumps(runs, indent=1) + "\n")
    for label, e in runs["speed"].items():
        print(f"{label}: {e['quantization']} weights {e['weights_gib']} GiB KV {e['kv_cache_tokens']:,} tokens | "
              f"{e['gpu']['power_limit_w']} W, {e['gpu']['median_sm_clock_mhz_busy']} MHz busy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
