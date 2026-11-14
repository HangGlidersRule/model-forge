#!/usr/bin/env python3
"""Ripple-65B Preview (#39): every posted bar and the selection rule, computed from the window's result files.

Inputs (RESULTS_DIR, the window's eval/ directory):
  gpqa-<cell>-r1.json, gpqa-<cell>-r2.json   llm-inference-bench gpqa-diamond outputs (two runs per cell)
  proxy-<cell>.json                          toolcall_proxy.py run outputs (one per serve)
  kl-<cell>.json                             kl_probe-format collect of the 64 held-out prompts (prompt logprobs, top 20)
  teacher.json                               bf16_teacher_cpu.py output (the BF16 teacher, same format)
  speed/<cell>-s<k>.json                     speed_grid.py outputs (vendor and the candidate, two serves each)
Cells: "vendor" and a candidate letter (A, then C, then B under the fallback rule).

Bars (posted pre-registration, "Gate for the quantized build" and "Selection rule"):
  GPQA       candidate mean correct >= vendor mean correct - 11.88 (0.06 x 198), unrounded means of two runs each;
             the paired per-item bootstrap of (candidate - vendor) is reported beside it (items averaged over runs,
             10,000 resamples, seed 1234). An item without a correct answer (wrong, unparseable, truncated or never
             answered) counts as wrong.
  Proxy      candidate parse rate >= vendor parse rate - 2 points, and candidate success >= 0.98 x vendor success,
             each on the mean of its serves (one proxy run per serve here).
  Artifact   the validator's verdict for the candidate (OUT/<letter>.validation.json) must be PASS.
  Tracked    KL and top-1 to the BF16 teacher (kl_probe's top-20 + remainder estimator), all positions and the
             SWE-smith tool-call span; never a bar.
Selection   eligible = passes every bar AND (speed win OR quality win);
             speed win: batch-1 decode tok/s (no DFlash2) >= 1.05 x the vendor's mean of two serves AND above both
             vendor serves; quality win: KL to the teacher >= 10% below the vendor's AND the paired bootstrap 95% CI of
             (candidate - vendor) per-prompt KL entirely below zero (prompts resampled, 10,000, seed 1234).
             Which ships: the fastest eligible candidate; a lead under 5% or inside the serve bands goes to the lower KL.
             No eligible candidate: nothing is published, the vendor NVFP4 stays the recommendation.
Usage: gate_ripple39.py RESULTS_DIR --cells vendor A [C B] [--validation-dir OUT] [--out GATE_JSON]
       gate_ripple39.py RESULTS_DIR --bars-only CAND     exit 0 when CAND passes every bar, 3 when it fails one, 2 when
                                                        an input is missing (the window's fallback branch reads this)
Prints GATE-WRITTEN and one line per candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

GPQA_ITEMS = 198
GPQA_MARGIN = 0.06 * GPQA_ITEMS  # 11.88 questions
PROXY_PARSE_MARGIN = 0.02
PROXY_SUCCESS_RATIO = 0.98
SPEED_RATIO = 1.05
KL_RATIO = 0.90
BOOT = 10_000
SEED = 1234


class Missing(Exception):
    pass


def _load(path: Path) -> dict:
    if not path.exists():
        raise Missing(str(path))
    return json.loads(path.read_text())


# ---- GPQA -------------------------------------------------------------------------------------------------------------
def gpqa_cell(d: Path, cell: str) -> dict:
    runs = []
    for r in (1, 2):
        j = _load(d / f"gpqa-{cell}-r{r}.json")
        rows = {x["item_id"]: x for x in j["runs"] if x.get("phase") != "warmup"}
        if len(rows) != GPQA_ITEMS:
            raise ValueError(f"gpqa-{cell}-r{r}: {len(rows)} items, expected {GPQA_ITEMS}")
        runs.append({"rows": rows, "acc": j.get("accuracy") or {}})
    ids = sorted(runs[0]["rows"])
    if sorted(runs[1]["rows"]) != ids:
        raise ValueError(f"gpqa-{cell}: the two runs cover different items")
    per_item = {i: sum(bool(r["rows"][i].get("correct")) for r in runs) / 2 for i in ids}
    corrects = [sum(bool(r["rows"][i].get("correct")) for i in ids) for r in runs]

    def stats(r):
        rows = r["rows"].values()
        return {"correct": sum(bool(x.get("correct")) for x in rows),
                "hit_max_tokens": sum(bool(x.get("hit_max_tokens")) for x in rows),
                "unparseable": r["acc"].get("unparseable"), "errors": sum(1 for x in rows if x.get("ok") is False),
                "truncated_no_answer": r["acc"].get("truncated_no_answer")}
    return {"ids": ids, "per_item": per_item, "mean_correct": sum(corrects) / 2, "runs": [stats(r) for r in runs],
            "run_disagreement_items": sum(1 for i in ids if bool(runs[0]["rows"][i].get("correct"))
                                          != bool(runs[1]["rows"][i].get("correct")))}


def paired_bootstrap(diffs: list[float], n_boot: int = BOOT, seed: int = SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    m = len(diffs)
    sums = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n_boot))
    return sums[int(0.025 * n_boot)], sums[int(0.975 * n_boot) - 1]


def gpqa_compare(c: dict, v: dict) -> dict:
    ids = sorted(set(c["ids"]) & set(v["ids"]))
    if len(ids) != GPQA_ITEMS:
        raise ValueError("GPQA cells do not pair on all 198 items")
    diffs = [c["per_item"][i] - v["per_item"][i] for i in ids]
    lo, hi = paired_bootstrap(diffs)
    margin = c["mean_correct"] - v["mean_correct"]
    return {"candidate_mean": c["mean_correct"], "vendor_mean": v["mean_correct"], "diff_questions": margin,
            "bar": f">= vendor - {GPQA_MARGIN:.2f}", "pass": margin >= -GPQA_MARGIN,
            "bootstrap95_questions": [round(lo * GPQA_ITEMS, 2), round(hi * GPQA_ITEMS, 2)],
            "candidate_runs": [r["correct"] for r in c["runs"]], "vendor_runs": [r["correct"] for r in v["runs"]]}


# ---- proxy --------------------------------------------------------------------------------------------------------------
def proxy_cell(d: Path, cell: str) -> dict:
    files = sorted(d.glob(f"proxy-{cell}.json")) + sorted(d.glob(f"proxy-{cell}-s*.json"))
    if not files:
        raise Missing(str(d / f"proxy-{cell}.json"))
    sums = [json.loads(f.read_text())["summary"]["overall"] for f in files]
    return {"serves": len(sums), "parse_rate": statistics.mean(s["parse_rate"] for s in sums),
            "success_rate": statistics.mean(s["success_rate"] for s in sums), "items": sums[0]["items"],
            "errors": sum(s.get("errors", 0) for s in sums)}


def proxy_compare(c: dict, v: dict) -> dict:
    parse_ok = c["parse_rate"] >= v["parse_rate"] - PROXY_PARSE_MARGIN - 1e-12
    succ_ok = c["success_rate"] >= PROXY_SUCCESS_RATIO * v["success_rate"] - 1e-12
    return {"candidate": c, "vendor": v, "parse_bar": f">= {v['parse_rate'] - PROXY_PARSE_MARGIN:.4f}",
            "success_bar": f">= {PROXY_SUCCESS_RATIO * v['success_rate']:.4f}", "parse_pass": parse_ok,
            "success_pass": succ_ok, "pass": parse_ok and succ_ok}


# ---- KL ------------------------------------------------------------------------------------------------------------------
def kl_position(r: dict, c: dict) -> tuple[float, bool]:
    """kl_probe.py's estimator, unchanged: KL(ref || cand) over the ref's top tokens plus a remainder bucket."""
    p = {t: math.exp(v) for t, v in r.items()}
    q = {t: math.exp(v) for t, v in c.items()}
    p_rest = max(1e-12, 1.0 - sum(p.values()))
    q_rest = max(1e-12, 1.0 - sum(q.get(t, 0.0) for t in p))
    floor = min(q.values()) if q else 1e-12
    kl = sum(pv * (math.log(pv) - math.log(q.get(t, floor))) for t, pv in p.items())
    kl += p_rest * (math.log(p_rest) - math.log(q_rest))
    return kl, max(p, key=p.get) == max(q, key=q.get)


def kl_cell(d: Path, cell: str, teacher: dict) -> dict:
    """kl_probe.py compare's estimator over the raw entries (each serve and the teacher give the top 20 plus the actual
    next token per position): mean over all positions, top-1 agreement, and per-prompt sums for the bootstrap."""
    j = _load(d / f"kl-{cell}.json")
    if j.get("heldout_sha256") != teacher.get("heldout_sha256"):
        raise ValueError(f"kl-{cell} was collected on other prompts than the teacher's")
    rows_t, rows_c = teacher["rows"], j["rows"]
    if len(rows_t) != len(rows_c):
        raise ValueError(f"kl-{cell}: {len(rows_c)} prompts, teacher {len(rows_t)}")
    spans = [p.get("span_start") for p in (teacher.get("prompts") or [{} for _ in rows_t])]
    per_prompt: list[tuple[float, int]] = []
    span_kl, span_n, agree = 0.0, 0, 0
    for k, (rt, rc) in enumerate(zip(rows_t, rows_c)):
        if len(rt) != len(rc):
            raise ValueError(f"kl-{cell}: prompt {k} has {len(rc)} positions, teacher {len(rt)}")
        s, m = 0.0, 0
        for j_, (a, b) in enumerate(zip(rt, rc)):
            if not a or not b:
                continue
            kl, same = kl_position(a, b)
            s += kl
            m += 1
            agree += same
            # position j_ predicts token j_ + 1; the tool-call span starts at token span_start
            if spans[k] is not None and j_ + 1 >= spans[k]:
                span_kl += kl
                span_n += 1
        per_prompt.append((s, m))
    n = sum(m for _, m in per_prompt)
    return {"positions": n, "mean_kl": sum(s for s, _ in per_prompt) / n if n else None,
            "top1_agreement": agree / n if n else None,
            "toolcall_span_kl": span_kl / span_n if span_n else None, "toolcall_span_positions": span_n,
            "per_prompt": per_prompt}


def kl_compare(c: dict, v: dict) -> dict:
    """Quality win: candidate KL <= 0.9 x vendor KL and the 95% CI of (candidate - vendor) mean KL below zero, from a
    paired bootstrap over prompts (positions stay with their prompt; 10,000 resamples, seed 1234)."""
    if [m for _, m in c["per_prompt"]] != [m for _, m in v["per_prompt"]]:
        raise ValueError("KL files do not cover the same positions")
    rng = random.Random(SEED)
    k = len(c["per_prompt"])
    diffs = []
    for _ in range(BOOT):
        idx = [rng.randrange(k) for _ in range(k)]
        n = sum(c["per_prompt"][i][1] for i in idx)
        diffs.append((sum(c["per_prompt"][i][0] - v["per_prompt"][i][0] for i in idx) / n) if n else 0.0)
    diffs.sort()
    lo, hi = diffs[int(0.025 * BOOT)], diffs[int(0.975 * BOOT) - 1]
    rel = c["mean_kl"] / v["mean_kl"] if v["mean_kl"] else None
    return {"candidate_kl": c["mean_kl"], "vendor_kl": v["mean_kl"], "ratio": rel,
            "bootstrap95_diff": [lo, hi], "quality_win": bool(rel is not None and rel <= KL_RATIO and hi < 0)}


# ---- speed ----------------------------------------------------------------------------------------------------------------
def speed_cell(d: Path, cell: str) -> dict:
    # Plain serves only (cell-s1.json, cell-s2.json): DFlash2 is reported, never gated.
    files = sorted(f for f in (d / "speed").glob(f"{cell}-s*.json") if "dflash2" not in f.name)
    if len(files) < 2:
        raise Missing(str(d / "speed" / f"{cell}-s[12].json"))
    serves = [json.loads(f.read_text()) for f in files]
    b1 = [s["cells"]["decode_b1"]["tok_s_median"] for s in serves]
    return {"serves": len(serves), "decode_b1": b1, "decode_b1_mean": statistics.mean(b1),
            "decode_b1_p50_ranges": [s["cells"]["decode_b1"].get("tok_s_range") for s in serves]}


def speed_compare(c: dict, v: dict) -> dict:
    ratio = c["decode_b1_mean"] / v["decode_b1_mean"]
    win = ratio >= SPEED_RATIO and c["decode_b1_mean"] > max(v["decode_b1"])
    return {"candidate_b1": c["decode_b1"], "vendor_b1": v["decode_b1"], "ratio": ratio,
            "vendor_band": [min(v["decode_b1"]), max(v["decode_b1"])], "speed_win": win}


# ---- the gate -------------------------------------------------------------------------------------------------------------
def evaluate(d: Path, cells: list[str], validation_dir: Path | None) -> dict:
    teacher = _load(d / "teacher.json")
    vendor = {"gpqa": gpqa_cell(d, "vendor"), "proxy": proxy_cell(d, "vendor")}
    try:
        vendor["kl"] = kl_cell(d, "vendor", teacher)
    except Missing:
        vendor["kl"] = None
    try:
        vendor["speed"] = speed_cell(d, "vendor")
    except Missing:
        vendor["speed"] = None
    out: dict = {"vendor": {"gpqa_mean": vendor["gpqa"]["mean_correct"], "gpqa_runs": vendor["gpqa"]["runs"],
                            "gpqa_run_disagreement_items": vendor["gpqa"]["run_disagreement_items"],
                            "proxy": vendor["proxy"],
                            "kl": {k: v for k, v in vendor["kl"].items() if k != "per_prompt"} if vendor["kl"] else None,
                            "speed": vendor["speed"]}, "candidates": {}}
    for cell in cells:
        if cell == "vendor":
            continue
        res: dict = {"missing": []}
        try:
            res["gpqa"] = gpqa_compare(gpqa_cell(d, cell), vendor["gpqa"])
        except Missing as e:
            res["missing"].append(str(e))
        try:
            res["proxy"] = proxy_compare(proxy_cell(d, cell), vendor["proxy"])
        except Missing as e:
            res["missing"].append(str(e))
        try:  # tracked, not a bar: a missing KL file does not block the bars, only the quality win
            if vendor["kl"] is None:
                raise Missing(str(d / "kl-vendor.json"))
            kc = kl_cell(d, cell, teacher)
            res["kl"] = {**{k: v for k, v in kc.items() if k != "per_prompt"}, **kl_compare(kc, vendor["kl"])}
        except Missing as e:
            res["missing_tracked"] = str(e)
        try:
            res["speed"] = speed_compare(speed_cell(d, cell), vendor["speed"]) if vendor["speed"] else None
        except Missing as e:
            res["speed"] = None
            res["missing_speed"] = str(e)
        if validation_dir is not None:
            vp = validation_dir / f"{cell}.validation.json"
            res["artifact_pass"] = vp.exists() and json.loads(vp.read_text()).get("verdict") == "PASS"
        else:
            res["artifact_pass"] = None
        bars = [res.get("gpqa", {}).get("pass"), res.get("proxy", {}).get("pass")]
        if res["artifact_pass"] is not None:
            bars.append(res["artifact_pass"])
        res["bars_complete"] = not res["missing"]
        res["passes_bars"] = bool(res["bars_complete"] and all(b is True for b in bars))
        sw = bool(res.get("speed") and res["speed"]["speed_win"])
        qw = bool(res.get("kl") and res["kl"]["quality_win"])
        res["speed_win"], res["quality_win"] = sw, qw
        res["eligible"] = res["passes_bars"] and (sw or qw)
        out["candidates"][cell] = res
    eligible = [c for c, r in out["candidates"].items() if r["eligible"]]
    ships = None
    if eligible:
        def b1(c):
            s = out["candidates"][c].get("speed")
            return statistics.mean(s["candidate_b1"]) if s else 0.0
        eligible.sort(key=b1, reverse=True)
        ships = eligible[0]
        if len(eligible) > 1:
            a, b = eligible[0], eligible[1]
            sa, sb = out["candidates"][a]["speed"], out["candidates"][b]["speed"]
            lead = b1(a) / b1(b) - 1 if b1(b) else 1.0
            # "inside the serve bands": the leader's slower serve is not above the runner-up's faster serve
            inside = bool(sa and sb and min(sa["candidate_b1"]) <= max(sb["candidate_b1"]))
            if lead < 0.05 or inside:
                ships = min((a, b), key=lambda c: out["candidates"][c]["kl"]["candidate_kl"])
                out["tie_break"] = f"lead {lead:.3f} < 5% or inside the serve bands: lower KL ships"
    out["eligible"] = eligible
    out["ships"] = ships
    out["recommendation"] = ships or "vendor NVFP4 (no candidate eligible; nothing published)"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--cells", nargs="+", default=["vendor", "A"])
    ap.add_argument("--validation-dir", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--bars-only")
    a = ap.parse_args()
    cells = a.cells if not a.bars_only else ["vendor", a.bars_only]
    try:
        g = evaluate(a.results, cells, a.validation_dir)
    except Missing as e:
        print(f"GATE-INPUT-MISSING {e}")
        sys.exit(2)
    if a.out:
        a.out.write_text(json.dumps(g, indent=1) + "\n")
    for c, r in g["candidates"].items():
        gp, px, kl = r.get("gpqa") or {}, r.get("proxy") or {}, r.get("kl") or {}
        print(f"GATE {c}: bars {'PASS' if r['passes_bars'] else 'FAIL'} | GPQA {gp.get('candidate_mean')} vs vendor "
              f"{gp.get('vendor_mean')} (bar -{GPQA_MARGIN:.2f}: {gp.get('pass')}; bootstrap {gp.get('bootstrap95_questions')})"
              f" | proxy parse {px.get('parse_pass')} success {px.get('success_pass')} | artifact {r.get('artifact_pass')}"
              f" | KL {kl.get('candidate_kl')} vs {kl.get('vendor_kl')} | speed win {r['speed_win']} quality win "
              f"{r['quality_win']} | eligible {r['eligible']} | missing {r['missing']}")
    print(f"GATE-WRITTEN ships={g['ships']} | {g['recommendation']}")
    if a.bars_only:
        r = g["candidates"][a.bars_only]
        if r["missing"]:
            sys.exit(2)
        sys.exit(0 if r["passes_bars"] else 3)


if __name__ == "__main__":
    main()
