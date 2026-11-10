#!/usr/bin/env python3
"""Curate the Nano-Omni v1.1.0 media and identity evidence into one aggregate file (issue #35), no model outputs.

Inputs (raw run outputs kept out of the repository):
  --media LABEL=PATH (repeatable)   omni_check.py outputs from the serving window (14 images incl. 10 COCO val2017
                                    office photos, 1 speech clip): per item only answered/finish and TTFT medians
  --clean LABEL=PATH (repeatable)   omni_check.py outputs from the clean-download smoke (8 text, 2 images, 1 speech)
  --attention PATH                  attention_source.py JSON lines
  --nvidia-gpqa PATH                the NVIDIA NVFP4 reference GPQA run from the 2026-11-05 gate session
Usage: curate_media_evidence.py OUT_JSON --media ... --clean ... --attention ... --nvidia-gpqa ..."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def media_summary(path: Path) -> dict:
    d = json.loads(path.read_text())
    images = [{"media": x["media"], "answered": bool(x["text"].strip()) and x["finish"] == "stop"} for x in d["image"]]
    audio = [{"media": x["media"], "answered": bool(x["text"].strip()) and x["finish"] == "stop"} for x in d["audio"]]
    text = [bool(x["text"].strip()) for x in d.get("text", [])]
    out = {"images_answered": sum(i["answered"] for i in images), "images": len(images),
           "audio_answered": sum(a["answered"] for a in audio), "audio": len(audio),
           "media": [i["media"] for i in images] + [a["media"] for a in audio]}
    if text:
        out.update({"text_answered": sum(text), "text": len(text)})
    ttft = d.get("ttft_s") or {}
    if ttft:
        p50 = sorted(v["p50"] for v in ttft.values())
        out["office_photo_ttft_p50_median_s"] = p50[len(p50) // 2]
    return out


def pairs(values: list[str]) -> dict[str, Path]:
    return {v.split("=", 1)[0]: Path(v.split("=", 1)[1]) for v in values}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--media", action="append", default=[])
    ap.add_argument("--clean", action="append", default=[])
    ap.add_argument("--attention", required=True)
    ap.add_argument("--nvidia-gpqa", required=True)
    a = ap.parse_args()
    nv = json.loads(Path(a.nvidia_gpqa).read_text())
    acc, md = nv["accuracy"], nv.get("metadata") or {}
    doc = {
        "schema_version": "1.0",
        "evidence_kind": "nano_omni_v1_1_0_media_and_identity",
        "issue": "HangGlidersRule/model-forge#35",
        "media_serving_window": {k: media_summary(p) for k, p in pairs(a.media).items()},
        "clean_download_smoke": {k: media_summary(p) for k, p in pairs(a.clean).items()},
        "attention_source": [json.loads(line) for line in Path(a.attention).read_text().splitlines() if line.strip()],
        "nvidia_nvfp4_reference_gpqa": {
            "repository": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4@16993199e436da4ba75ddc410855f87e0d996ee6",
            "correct": acc["correct"], "attempted": acc["attempted"], "errors": acc["errors"],
            "max_tokens": md.get("max_tokens"), "temperature": md.get("temperature"),
            "note": "2026-11-05 W4A4 gate session, same GPU and harness; 65,536-token cap (not the v1.0.0 8,192 protocol)",
        },
    }
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
