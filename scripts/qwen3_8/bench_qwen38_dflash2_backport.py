"""Qwen3.8-27B DFlash2 backport benchmark: MTP vs DFlash2 on Darkstar artifacts.

Measures spec-decode acceptance and single-stream throughput on an OpenAI-compatible
vLLM endpoint. Run against a server started with:
  --speculative-config '{"method":"dflash","model":"<drafter>","num_speculative_tokens":N}'
DFlash2 is auto-detected from the drafter architecture (DFlash2DraftModel).

Usage:
  python3 dflash2_backport_bench.py --base http://host:port --model <id> \
      --out /path/out.json [--prompt-lens 4 16 48 --n-runs 5 --max-tokens 512 --warmup 2]
"""
import argparse
import json
import sys
import time
import urllib.request

PRACTICE = (
    "Write a detailed, deterministic technical explanation of reliable "
    "distributed-system monitoring, failure detection, and recovery."
)


def make_prompt(length_k: int) -> str:
    nonce = " ".join(f"nonce{i:02d}_{j:02d}" for j in range(32) for i in range(8))
    filler = ("The precise role of replication, consensus, and quorum formation in "
              "distributed systems remains a central topic of research and engineering. "
              "Logs provide evidence; dashboards provide signal; neither replaces the "
              "other, and both are required for reliable operation. ") * max(1, length_k // 4)
    return nonce + "\n" + filler + " " + PRACTICE


def request(base: str, model: str, content: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "${PUBLIC_WORKSPACE}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def run_single(base: str, model: str, length_k: int, max_tokens: int) -> dict:
    start = time.time()
    data = request(base, model, make_prompt(length_k), max_tokens)
    elapsed = time.time() - start
    usage = data.get("usage", {})
    completion = usage.get("completion_tokens", 0)
    return {
        "prompt_len_k": length_k,
        "completion_tokens": completion,
        "elapsed_s": elapsed,
        "tok_s": completion / elapsed if elapsed else 0.0,
        "total_tokens": usage.get("total_tokens", 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-lens", nargs="+", type=int, default=[4, 16, 48])
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    rows: list[dict] = []
    try:
        for length_k in args.prompt_lens:
            for _ in range(args.warmup):
                run_single(args.base, args.model, length_k, args.max_tokens)
            timed = [run_single(args.base, args.model, length_k, args.max_tokens)
                     for _ in range(args.n_runs)]
            rates = [r["tok_s"] for r in timed]
            summary = {
                "prompt_len_k": length_k,
                "n_runs": args.n_runs,
                "mean_tok_s": sum(rates) / len(rates),
                "min_tok_s": min(rates),
                "max_tok_s": max(rates),
                "samples": timed,
            }
            rows.append(summary)
            print(f"{length_k:>4}K  mean={summary['mean_tok_s']:8.2f} tok/s  "
                  f"(min={summary['min_tok_s']:7.2f} max={summary['max_tok_s']:7.2f})")
    except Exception as exc:  # noqa: BLE001 - report and preserve partial rows
        print(f"ERROR: {exc!r}", file=sys.stderr)
        result = {"error": repr(exc), "rows": rows}
    else:
        result = {"rows": rows}
    result["base_url"] = args.base
    result["model"] = args.model
    result["max_tokens"] = args.max_tokens
    result["completed_epoch"] = time.time()
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print("wrote", args.out)
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
