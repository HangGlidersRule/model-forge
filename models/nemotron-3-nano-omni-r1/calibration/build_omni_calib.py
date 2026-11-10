#!/usr/bin/env python3
"""Pinned, deterministic calibration set for the issue-35 Nano-Omni W4A4 build. Needs Python 3 with pyarrow (the stock
vLLM image has none; `python:3.12` with `pip install pyarrow` works) and writes the images and audio it uses beside the
JSONL, so the build reads local files.

Three streams, rendered exactly the way the checkpoint's own chat template renders a client request (thinking off,
matching the serving prompts), so the language model's activations during calibration match serving traffic:
  text   256 rows: nvidia/Nemotron-Post-Training-Dataset-v2 @ 5c89e01d, chat/stem/math/code, 64 each (the family's
         calibration source for the shipped text cells), seed 1234;
  image  256 rows: HuggingFaceM4/the_cauldron @ 847a98a7, image-question-answer conversations from 8 subsets
         (ai2d, chartqa, docvqa, aokvqa, vqav2, textvqa, scienceqa, infographic_vqa), 32 each, seed 1234; the first image
         of each conversation and its first user turn;
  audio  64 rows: hf-internal-testing/librispeech_asr_dummy @ 5be91486, "Transcribe this audio." over each clip
         (73 clips; the first 64).
The routed experts see every stream: their activation scales must cover image and audio tokens, which the robot sends.

Every input is one parquet shard named below with its LFS sha256 at the pinned dataset revision (the first shard of
each split or subset). Bytes are hashed before they are parsed, from the same buffer, and any mismatch stops the build.
A shard already under STORE_ROOT/datasets/<org>/<name>@<revision>/ is used when it hashes to its pin; otherwise it is
downloaded, verified, and stored there for the next run (a read-only store is only read).
Output: OUT/calib.jsonl rows {"stream", "messages", "image"|"audio" (local file name), "source", "row"} and OUT/media/.
Prints CALIB-OK with counts and the JSONL's sha256 only when the output is the pinned set: 576 rows, 320 media files,
calib.jsonl sha256 83cebc8e3b109da4d7957ec4980c1412988545de62b7c8cc64ff89914906f515 (the set the W4A4 build used;
reproduced byte for byte from an empty cache and again from the store alone). Any difference, such as rows the
text or image loops dropped, prints CALIB-FAILED with each mismatch and exits non-zero.
Hub access: nvidia/Nemotron-Post-Training-Dataset-v2 is gated, so an empty store needs a token with access to it,
read from /secret/token (a file mounted for the run, deleted after reading) or else from HF_TOKEN. A store that
already holds every pinned shard needs no token.
Usage: build_omni_calib.py STORE_ROOT OUT_DIR"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import stat
import sys
import tempfile
import urllib.request
from pathlib import Path

SEED = 1234
# The set the W4A4 build used (see the docstring); finish() refuses any other output.
EXPECTED_ROWS = 576
EXPECTED_MEDIA = 320
EXPECTED_SHA256 = "83cebc8e3b109da4d7957ec4980c1412988545de62b7c8cc64ff89914906f515"
NPT = ("nvidia/Nemotron-Post-Training-Dataset-v2", "5c89e01dd720ae0f4058445ed49c5fb68a03c76e")
CAULDRON = ("HuggingFaceM4/the_cauldron", "847a98a779b1652d65111daf20c972dfcd333605")
LIBRI = ("hf-internal-testing/librispeech_asr_dummy", "5be91486e11a2d616f4ec5db8d3fd248585ac07a")
TEXT_SHARDS = (
    ("chat", "data/chat-00000-of-00012.parquet", "fdab6d8a28756ccbd5a41158191271f70e6a78699e2ae00e4de6c0b680f56192"),
    ("stem", "data/stem-00000-of-00002.parquet", "630296c396c6d9234834ca048e1293f7ac7963538e1f48a5fce4cae1533fd7b3"),
    ("math", "data/math-00000-of-00002.parquet", "4ab786ef3a595cd3aa2f9b66049014726ab4fc1e9092f0f2c78b9907b2cfdfd3"),
    ("code", "data/code-00000-of-00002.parquet", "939438295743dcf850ded2ca61b9996c1f66ac2793f92227b7f052c874066c35"),
)
IMAGE_SHARDS = (
    ("ai2d", "ai2d/train-00000-of-00001-2ce340398c113b79.parquet",
     "a1e05b99801ec73374dff401b7c91645465da955fc85ac62b2b5aaf6c9a18ab5"),
    ("chartqa", "chartqa/train-00000-of-00002-7733d4ca73ccd12e.parquet",
     "7d96bf7b4b083f7228446c2c7eb1826f594872d715a33fe269fce1b2fbacf6c9"),
    ("docvqa", "docvqa/train-00000-of-00014-d617a15cc5127a86.parquet",
     "bb08abe6038407ab034bb98585158115be8bc1954e0ba12fa9e4b6b82a501deb"),
    ("aokvqa", "aokvqa/train-00000-of-00002-88e828b9a932c295.parquet",
     "87ea212f02afc87fb9868b1a983791074bca971a594dc24e19b99f82ed12e431"),
    ("vqav2", "vqav2/train-00000-of-00022-1f40dc68b5c44ca4.parquet",
     "0e703056fa296fcaed0eb6ae26300cacc7ff699437e0e917ad4915cdc8715034"),
    ("textvqa", "textvqa/train-00000-of-00012-31f9e352dd92b89f.parquet",
     "1e7828f3ae58bf09e21e67d92a28321d67db17e60d8a498b58964ac80e6db5db"),
    ("scienceqa", "scienceqa/train-00000-of-00001-c411546b9bc4df22.parquet",
     "d1956b8a3f5f1ff4634634dc5b160e7183ff196386f42a802e4ba915357f4d35"),
    ("infographic_vqa", "infographic_vqa/train-00000-of-00001-9187ab6377a43fd2.parquet",
     "4a05a5c66de16af4f72af5ecade43634a6152d8489d29cce7b0095df4d9f83fe"),
)
AUDIO_SHARD = ("clean/validation-00000-of-00001.parquet",
               "4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603")
N_TEXT_PER_SPLIT, N_IMAGE_PER_SUBSET, N_AUDIO = 64, 32, 64
TOKEN = None


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "model-forge-calib",
                                               **({"Authorization": f"Bearer {TOKEN}"} if TOKEN else {})})
    for attempt in range(5):
        try:
            return urllib.request.urlopen(req, timeout=300).read()
        except Exception as error:  # noqa: BLE001
            if attempt == 4:
                hint = " (gated dataset: provide a Hub token, see the docstring)" if "401" in str(error) else ""
                raise SystemExit(f"CALIB-FAILED download {url}: {type(error).__name__}: {error}{hint}") from None
    raise AssertionError


def pinned(store: Path, repo: str, rev: str, path: str, sha256: str) -> bytes:
    """The shard's bytes, verified against its pin before anyone parses them."""
    local = store / "datasets" / f"{repo}@{rev}" / path
    try:
        # A regular file only: no symlink is followed, and a FIFO or device cannot block the build.
        fd = os.open(local, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as f:
            if stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                data = f.read()
                if hashlib.sha256(data).hexdigest() == sha256:
                    return data
    except OSError:
        pass
    data = get(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise SystemExit(f"CALIB-FAILED {repo}@{rev[:8]}/{path}: sha256 {digest} is not the pinned {sha256}")
    tmp = None
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=local.parent, prefix=f".{local.name}.", suffix=".part")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, local)
    except OSError as error:
        if tmp:
            Path(tmp).unlink(missing_ok=True)
        print(f"note: {path} not stored in the store ({error.strerror})", file=sys.stderr)
    return data


def main() -> int:
    # Imported here so the pin check (finish, pin_mismatches) is importable and testable without pyarrow.
    import pyarrow.parquet as pq

    global TOKEN
    store, out = Path(sys.argv[1]), Path(sys.argv[2])
    tok_file = Path("/secret/token")
    if tok_file.exists():
        TOKEN = tok_file.read_text().strip()
        tok_file.unlink()
    else:
        TOKEN = os.environ.get("HF_TOKEN") or None
    (out / "media").mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    rows: list[dict] = []

    # text
    for _split, shard, sha256 in TEXT_SHARDS:
        t = pq.read_table(io.BytesIO(pinned(store, *NPT, shard, sha256)), columns=["messages"]).to_pylist()
        idx = sorted(rng.sample(range(len(t)), N_TEXT_PER_SPLIT))
        for i in idx:
            msgs = [{"role": m["role"], "content": m["content"]} for m in t[i]["messages"] if m["role"] in ("user", "assistant")]
            if msgs and msgs[0]["role"] == "user":
                rows.append({"stream": "text", "messages": msgs[:2], "source": f"{NPT[0]}@{NPT[1][:8]}/{shard}", "row": i})

    # image
    for sub, shard, sha256 in IMAGE_SHARDS:
        t = pq.read_table(io.BytesIO(pinned(store, *CAULDRON, shard, sha256))).to_pylist()
        for i in sorted(rng.sample(range(len(t)), min(N_IMAGE_PER_SUBSET, len(t)))):
            r = t[i]
            if not r.get("images") or not r.get("texts"):
                continue
            img = r["images"][0]
            name = f"cauldron-{sub}-{i}.png" if img["bytes"][:4] == b"\x89PNG" else f"cauldron-{sub}-{i}.jpg"
            (out / "media" / name).write_bytes(img["bytes"])
            turn = r["texts"][0]
            rows.append({"stream": "image", "image": name,
                         "messages": [{"role": "user", "content": turn["user"]}, {"role": "assistant", "content": turn["assistant"]}],
                         "source": f"{CAULDRON[0]}@{CAULDRON[1][:8]}/{shard}", "row": i})

    # audio
    t = pq.read_table(io.BytesIO(pinned(store, *LIBRI, *AUDIO_SHARD))).to_pylist()
    for i, r in enumerate(t[:N_AUDIO]):
        name = f"libri-{i:03d}.flac"
        (out / "media" / name).write_bytes(r["audio"]["bytes"])
        rows.append({"stream": "audio", "audio": name,
                     "messages": [{"role": "user", "content": "Transcribe this audio."}, {"role": "assistant", "content": r["text"]}],
                     "source": f"{LIBRI[0]}@{LIBRI[1][:8]}/{AUDIO_SHARD[0]}", "row": i})

    finish(out, rows)
    return 0


def pin_mismatches(rows: int, media: int, digest: str) -> list[str]:
    """Every way the output differs from the pinned set; empty when it is the pinned set."""
    problems = []
    if rows != EXPECTED_ROWS:
        problems.append(f"rows={rows}, pinned {EXPECTED_ROWS}")
    if media != EXPECTED_MEDIA:
        problems.append(f"media={media}, pinned {EXPECTED_MEDIA}")
    if digest != EXPECTED_SHA256:
        problems.append(f"sha256={digest}, pinned {EXPECTED_SHA256}")
    return problems


def finish(out: Path, rows: list[dict]) -> None:
    """Write calib.jsonl, then print CALIB-OK, or stop with CALIB-FAILED unless it is the pinned set."""
    p = out / "calib.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    media = len(list((out / "media").iterdir()))
    counts = {s: sum(1 for r in rows if r["stream"] == s) for s in ("text", "image", "audio")}
    problems = pin_mismatches(len(rows), media, digest)
    if problems:
        raise SystemExit(f"CALIB-FAILED not the pinned set: {'; '.join(problems)} ({counts})")
    print(f"CALIB-OK rows={len(rows)} {counts} media={media} sha256={digest}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
