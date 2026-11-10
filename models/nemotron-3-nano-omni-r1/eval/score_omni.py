#!/usr/bin/env python3
"""Score the issue-35 serving window (pre-registered on #35 before the window ran). Reads the omni_check.py outputs and
the office-photo manifest; writes omni-gate-summary.json (aggregates and every model's descriptions side by side).

Bars, on the RTX PRO 6000 Blackwell host:
  1. Each rebuilt wrapper serves as NemotronH_Nano_Omni_Reasoning_V3 and answers every image and audio prompt (non-empty,
     finish_reason stop).
  2. Text equivalence: a rebuilt wrapper and its published trunk-only product give identical greedy output on 8/8 text
     prompts (identical weights; this is what carries the existing GPQA and behavior results over).
  3. Office photos, content: the rebuilt Base's mean annotated-object recall over the 10 photos is at least NVIDIA BF16's
     minus 0.05. Recall per photo = the photo's COCO-annotated categories mentioned in the two-sentence description
     (fixed synonym list below) / annotated categories.
  4. TTFT: the median over the 10 office photos of each photo's median time to first token (5 timed runs, prefix and
     multimodal caches off) is at most NVIDIA FP8's. The robot's GB10 is a different GPU: this bar is measured here and
     the same script runs there.
Exact-match rate against BF16, sentence counts and the audio transcription's word agreement with BF16 are reported.
Usage: score_omni.py OUT_DIR PHOTO_MANIFEST SUMMARY_JSON"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

SYNONYMS = {
    "laptop": ["laptop", "notebook", "macbook", "computer"],
    "tv": ["monitor", "screen", "display", "television", "tv", "computer"],
    "keyboard": ["keyboard"],
    "mouse": ["mouse", "mice"],
    "chair": ["chair", "seat"],
    "cup": ["cup", "mug"],
    "bottle": ["bottle"],
    "book": ["book", "bookshelf", "bookcase"],
    "cell phone": ["phone", "smartphone", "iphone", "cellphone"],
    "person": ["person", "woman", "man", "girl", "boy", "someone", "people", "individual"],
    "potted plant": ["plant", "houseplant"],
    "couch": ["couch", "sofa"],
    "remote": ["remote", "controller"],
    "cat": ["cat", "kitten"],
    "car": ["car", "vehicle"],
    "bowl": ["bowl"],
    "spoon": ["spoon"],
}
MODELS = ("base-wrapper", "ablit-wrapper", "nvidia-fp8", "nvidia-nvfp4", "nvidia-bf16")


def mentioned(text: str, cat: str) -> bool:
    t = text.lower()
    return any(re.search(rf"\b{re.escape(w)}(e?s)?\b", t) for w in SYNONYMS.get(cat, [cat]))


def sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def word_agreement(a: str, b: str) -> float:
    """1 - word error rate of a against b (Levenshtein over words)."""
    x, y = words(a), words(b)
    d = list(range(len(y) + 1))
    for i, wx in enumerate(x, 1):
        prev, d[0] = d[0], i
        for j, wy in enumerate(y, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (wx != wy))
            prev, d[j] = d[j], cur
    return round(1 - d[len(y)] / max(1, len(y)), 4)


def main() -> int:
    out, manifest_path, summary_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    photos = json.loads(manifest_path.read_text())["images"]
    runs = {m: json.loads((out / f"{m}.json").read_text()) for m in MODELS if (out / f"{m}.json").exists()}
    bf = {r["media"]: r for r in runs["nvidia-bf16"]["image"]}
    summary: dict = {"photo_rule": json.loads(manifest_path.read_text())["rule"], "models": {}, "photos": []}
    for m, r in runs.items():
        img = {x["media"]: x for x in r["image"]}
        rec = []
        for p in photos:
            d = img[p["file"]]["text"]
            rec.append(sum(mentioned(d, c) for c in p["categories"]) / len(p["categories"]))
        office = [img[p["file"]] for p in photos]
        ttfts = [r["ttft_s"][p["file"]]["p50"] for p in photos if p["file"] in r.get("ttft_s", {})]
        audio = r["audio"][0]["text"] if r["audio"] else ""
        summary["models"][m] = {
            "images_answered": sum(1 for x in r["image"] if x["text"].strip() and x["finish"] == "stop"),
            "images_total": len(r["image"]),
            "audio_answered": sum(1 for x in r["audio"] if x["text"].strip() and x["finish"] == "stop"),
            "office_mean_object_recall": round(statistics.mean(rec), 4),
            "office_exact_match_vs_bf16": sum(1 for x in office if x["text"] == bf[x["media"]]["text"]),
            "office_sentences": [sentences(x["text"]) for x in office],
            "office_ttft_median_s": round(statistics.median(ttfts), 4) if ttfts else None,
            "audio_word_agreement_vs_bf16": word_agreement(audio, runs["nvidia-bf16"]["audio"][0]["text"]) if runs["nvidia-bf16"]["audio"] else None,
        }
    for p in photos:
        summary["photos"].append({"file": p["file"], "categories": p["categories"],
                                  "descriptions": {m: next(x["text"] for x in r["image"] if x["media"] == p["file"]) for m, r in runs.items()}})
    eq = {}
    for w, t in (("base-wrapper", "base-trunk"), ("ablit-wrapper", "ablit-trunk")):
        if (out / f"{t}.json").exists() and w in runs:
            a, b = runs[w]["text"], json.loads((out / f"{t}.json").read_text())["text"]
            eq[w] = {"identical": sum(x["text"] == y["text"] for x, y in zip(a, b)), "prompts": len(a)}
    summary["text_equivalence"] = eq
    s, b16, f8 = summary["models"]["base-wrapper"], summary["models"]["nvidia-bf16"], summary["models"]["nvidia-fp8"]
    summary["bars"] = {
        "1_serves_and_answers": all(summary["models"][w]["images_answered"] == summary["models"][w]["images_total"]
                                    and summary["models"][w]["audio_answered"] == 1 for w in ("base-wrapper", "ablit-wrapper") if w in summary["models"]),
        "2_text_equivalence": all(v["identical"] == v["prompts"] for v in eq.values()) and len(eq) == 2,
        "3_office_content": s["office_mean_object_recall"] >= b16["office_mean_object_recall"] - 0.05,
        "4_ttft_vs_fp8": s["office_ttft_median_s"] is not None and f8["office_ttft_median_s"] is not None
        and s["office_ttft_median_s"] <= f8["office_ttft_median_s"],
    }
    summary_path.write_text(json.dumps(summary, indent=1) + "\n")
    for m, v in summary["models"].items():
        print(f"{m:14s} images {v['images_answered']}/{v['images_total']} audio {v['audio_answered']} | office recall "
              f"{v['office_mean_object_recall']:.3f} exact-vs-BF16 {v['office_exact_match_vs_bf16']}/10 | TTFT median "
              f"{v['office_ttft_median_s']} s | audio agreement {v['audio_word_agreement_vs_bf16']}", flush=True)
    print("text equivalence:", eq, flush=True)
    print("BARS:", summary["bars"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
