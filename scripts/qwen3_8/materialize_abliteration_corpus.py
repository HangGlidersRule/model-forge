#!/usr/bin/env python3
"""Materialize pinned abliteration prompt corpora with deterministic hashing.

Fetches revision-pinned harmful and harmless prompt datasets, normalizes to
{id, text, label, source} JSONL, deduplicates, sorts deterministically,
and writes local snapshots with SHA-256 manifests. Skips redownload when
hashes verify.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from model_forge.experiment import load_experiment
from model_forge.pipeline import (
    RunLock,
    StageContext,
    SuccessManifest,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# Model-record corpus assets (provenance notes, schema) are tracked here; stage outputs are
# written to the run root instead, so a normal run never touches the source tree.
MODEL_RECORD_DATA_DIR = REPO_ROOT / "models" / "qwen3.8-27b-r3" / "data" / "abliteration"


def fetch_prompts(dataset_id: str, revision: str, count: int) -> list[str]:
    """Load exactly ``count`` prompts from a pinned Hub dataset revision."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, revision=revision, split="train")
    column = "text" if "text" in ds.column_names else ds.column_names[0]
    prompts = [str(row[column]).strip() for row in ds if str(row[column]).strip()]
    if len(prompts) < count:
        raise RuntimeError(
            f"{dataset_id}@{revision} has {len(prompts)} usable prompts; {count} required"
        )
    return prompts[:count]


def normalize_prompts(
    prompts: list[str], label: str, source: str
) -> list[dict[str, str]]:
    seen: set[str] = set()
    records: list[dict[str, str]] = []
    for text in prompts:
        text = text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        records.append(
            {"id": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text, "label": label, "source": source}
        )
    records.sort(key=lambda r: r["id"])
    return records


def write_jsonl(records: list[dict[str, str]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=True) for r in records]
    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
    return sha256_file(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize abliteration corpora")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help=(
            "Writable run root for stage outputs; documentation for the materialized corpus "
            f"lives in {MODEL_RECORD_DATA_DIR.relative_to(REPO_ROOT)}"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-stage", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cfg = load_experiment(args.config)
    cfg_sha = cfg.config_sha()

    ctx = StageContext(args.run_root, "corpus", cfg_sha)
    if args.resume and not args.force_stage:
        existing = ctx.should_skip()
        if existing:
            print(f"Corpus stage verified, skipping (config_sha={cfg_sha[:12]})")
            return

    lock = RunLock(args.run_root / "corpus.lock")
    with lock:
        partial = ctx.partial_dir

        harmful_ref = cfg.datasets["harmful"]
        harmless_ref = cfg.datasets["harmless"]
        harmful_records = normalize_prompts(
            fetch_prompts(
                harmful_ref.source,
                harmful_ref.revision,
                cfg.abliteration.harmful_prompts,
            ),
            label="harmful",
            source=harmful_ref.source,
        )
        harmless_records = normalize_prompts(
            fetch_prompts(
                harmless_ref.source,
                harmless_ref.revision,
                cfg.abliteration.harmless_prompts,
            ),
            label="harmless",
            source=harmless_ref.source,
        )

        h_hash = write_jsonl(harmful_records, partial / "harmful.jsonl")
        l_hash = write_jsonl(harmless_records, partial / "harmless.jsonl")

        manifest = SuccessManifest(
            stage="corpus",
            config_sha=cfg_sha,
            source_revisions={
                harmful_ref.source: harmful_ref.revision,
                harmless_ref.source: harmless_ref.revision,
            },
            output_hashes={"harmful.jsonl": h_hash, "harmless.jsonl": l_hash},
            metrics={
                "harmful_count": len(harmful_records),
                "harmless_count": len(harmless_records),
            },
        )
        ctx.promote(manifest)
        print(f"Corpus materialized: {len(harmful_records)} harmful, {len(harmless_records)} harmless")


if __name__ == "__main__":
    main()
