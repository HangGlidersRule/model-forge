#!/usr/bin/env python3
"""Pick the issue-35 office photos: 10 public COCO 2017 val images chosen by a fixed rule, before any model is run.

Rule (pre-registered on #35):
  - the license allows commercial use: COCO license ids 4-8 (Attribution, Attribution-ShareAlike,
    Attribution-NoDerivs, No known copyright restrictions, United States Government Work);
  - the annotations include a laptop or a tv (monitor), a keyboard or a mouse, and a chair;
  - the first 10 such images in ascending image id.
Downloads each image from images.cocodataset.org and writes office-photos.json: image id, file name, sha256, size,
license, source URLs and the annotated COCO categories (the ground truth the description check scores against).
The images are referenced, not redistributed. Usage: pick_office_photos.py OUT_DIR   (network; stdlib only)"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
LICENSES_OK = {4, 5, 6, 7, 8}
N = 10
RULE = ("COCO 2017 val; license id in 4-8 (commercial use allowed); annotations include (laptop or tv) and "
        "(keyboard or mouse) and chair; first 10 by ascending image id")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    z = out / "annotations_trainval2017.zip"
    if not z.exists():
        part = out / "annotations_trainval2017.zip.partial"
        urllib.request.urlretrieve(ANN, part)
        part.rename(z)
    with zipfile.ZipFile(z) as zf:
        inst = json.loads(zf.read("annotations/instances_val2017.json"))
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    present: dict[int, set[str]] = {}
    for a in inst["annotations"]:
        present.setdefault(a["image_id"], set()).add(cats[a["category_id"]])
    images = {i["id"]: i for i in inst["images"]}
    licenses = {lic["id"]: lic for lic in inst["licenses"]}
    picked: list[int] = []
    for iid in sorted(present):
        p, im = present[iid], images[iid]
        if im["license"] in LICENSES_OK and p & {"laptop", "tv"} and p & {"keyboard", "mouse"} and "chair" in p:
            picked.append(iid)
            if len(picked) == N:
                break
    rows = []
    for iid in picked:
        im = images[iid]
        f = out / f"coco-val2017-{iid:012d}.jpg"
        if not f.exists():
            urllib.request.urlretrieve(im["coco_url"], f)
        rows.append({"image_id": iid, "file": f.name, "sha256": sha(f), "bytes": f.stat().st_size,
                     "width": im["width"], "height": im["height"], "coco_url": im["coco_url"],
                     "flickr_url": im.get("flickr_url"), "license": licenses[im["license"]]["name"],
                     "license_url": licenses[im["license"]]["url"], "categories": sorted(present[iid])})
    manifest = {"source": "COCO 2017 val, annotations/instances_val2017.json", "annotations_zip": ANN,
                "annotations_zip_sha256": sha(z), "rule": RULE, "images": rows}
    (out / "office-photos.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for r in rows:
        print(f"{r['image_id']} {r['width']}x{r['height']} {r['license']} | {', '.join(r['categories'])}", flush=True)
    print(f"PICK-OK {len(rows)} images; annotations zip sha256 {manifest['annotations_zip_sha256']}", flush=True)
    return 0 if len(rows) == N else 1


if __name__ == "__main__":
    sys.exit(main())
