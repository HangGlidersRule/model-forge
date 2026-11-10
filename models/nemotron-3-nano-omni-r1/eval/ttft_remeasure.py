#!/usr/bin/env python3
"""Issue #35 amendment R35-A1: G4 time-to-first-token re-measurement. One measuring run of this script = one vLLM serve.

Requests are built by omni_check.py, so the body is identical to the gate window's TTFT requests: greedy, thinking off,
the image plus "Describe this image in two sentences.", streamed, max_tokens 32; TTFT = time to the first streamed
content token. Per serve: 2 untimed warm-ups per photo, then ROUNDS rounds; each round times one request per office
photo, the photo order rotated by one position per round so slow periods spread over all photos. Every timing is kept.

  ttft_remeasure.py BASE MODEL LABEL MEDIA_DIR PHOTOS(comma-separated) OUT_JSON ROUNDS
  ttft_remeasure.py score CAND_JSON [CAND_JSON ...] -- REF_JSON [REF_JSON ...]

Scoring, as pre-registered in R35-A1: per photo, the median of all of a cell's timed runs (its serves pooled); the cell
value is the median over the photos; PASS when candidate <= reference. Also printed: each serve's own value, the
reference replicate band |ref serve 1 - ref serve 2|, and the photo-by-photo differences. The scorer always exits 0
(the verdict is in its output); a non-zero exit means it could not score."""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omni_check as oc  # noqa: E402


def measure(base: str, model: str, label: str, media_dir: str, photos: list[str], out: str, rounds: int) -> None:
    bodies = {}
    for p in photos:
        content = [{"type": "image_url", "image_url": {"url": f"file://{media_dir.rstrip('/')}/{p}"}},
                   {"type": "text", "text": oc.IMAGE_PROMPT}]
        bodies[p] = oc.chat(model, content, 32)
    for p in photos:
        for _ in range(2):
            oc.ttft(base, bodies[p])
    timed: dict[str, list[float]] = {p: [] for p in photos}
    for r in range(rounds):
        k = r % len(photos)
        for p in photos[k:] + photos[:k]:
            timed[p].append(round(oc.ttft(base, bodies[p]), 5))
    per_photo = {p: statistics.median(v) for p, v in timed.items()}
    res = {"label": label, "rounds": rounds, "timed": timed, "per_photo_median": per_photo,
           "cell_median": statistics.median(per_photo.values())}
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"TTFT-OK {label}: cell median {res['cell_median']:.4f} s over {len(photos)} photos x {rounds} timed runs", flush=True)


def score(cand_paths: list[str], ref_paths: list[str]) -> int:
    def load(paths: list[str]) -> tuple[dict[str, float], list[float]]:
        pooled: dict[str, list[float]] = {}
        serves = []
        for p in paths:
            with open(p) as f:
                d = json.load(f)
            serves.append(d["cell_median"])
            for k, v in d["timed"].items():
                pooled.setdefault(k, []).extend(v)
        return {k: statistics.median(v) for k, v in pooled.items()}, serves

    c, c_serves = load(cand_paths)
    r, r_serves = load(ref_paths)
    cand, ref = statistics.median(c.values()), statistics.median(r.values())
    band = abs(r_serves[0] - r_serves[1]) if len(r_serves) >= 2 else None
    diffs = sorted(c[k] - r[k] for k in r)
    faster = sum(1 for d in diffs if d < 0)
    verdict = "PASS" if cand <= ref else "FAIL"
    within = band is not None and abs(cand - ref) <= band
    band_txt = f"{band * 1000:.1f} ms" if band is not None else "n/a"
    print(f"R35-A1 G4 {verdict}: candidate {cand:.4f} s vs reference {ref:.4f} s ({(cand - ref) * 1000:+.1f} ms) | "
          f"candidate serves {[round(x, 4) for x in c_serves]} | reference serves {[round(x, 4) for x in r_serves]} | "
          f"reference replicate band {band_txt}{' (the difference is inside it)' if within else ''} | photo by photo: "
          f"candidate faster on {faster} of {len(diffs)}, median difference {statistics.median(diffs) * 1000:+.1f} ms", flush=True)
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "score":
        i = sys.argv.index("--")
        sys.exit(score(sys.argv[2:i], sys.argv[i + 1:]))
    base, model, label, media_dir, photos, out, rounds = sys.argv[1:8]
    measure(base, model, label, media_dir, [p for p in photos.split(",") if p], out, int(rounds))
