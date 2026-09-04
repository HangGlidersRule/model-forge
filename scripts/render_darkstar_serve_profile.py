#!/usr/bin/env python3
"""Render one canonical, rerunnable Darkstar vLLM Compose profile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# A git archive has no installed package and no PYTHONPATH. Resolve the in-tree package before
# importing it so this documented script is directly runnable from a clean checkout.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from model_forge.serve_profile import ServeProfile, write_compose  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--behavior", choices=("base", "abliterated"), required=True)
    parser.add_argument("--format", choices=("bf16", "nvfp4"), required=True)
    parser.add_argument("--repository-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--artifact-identity")
    parser.add_argument("--artifact-precision-class")
    parser.add_argument("--artifact-success-sha256")
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument(
        "--artifact-validation",
        choices=("local", "attestation"),
        default="local",
        help="local verifies the artifact marker; attestation is only for deterministic rendering",
    )
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--mtp-depth", type=int, required=True)
    parser.add_argument("--scheduler-tokens", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("containers/serve"))
    args = parser.parse_args()

    destination = write_compose(
        ServeProfile(
            family=args.family,
            behavior=args.behavior,
            format=args.format,
            repository_id=args.repository_id,
            model_path=args.model_path,
            artifact_identity=args.artifact_identity,
            artifact_precision_class=args.artifact_precision_class,
            artifact_success_sha256=args.artifact_success_sha256,
            artifact_manifest=args.artifact_manifest,
            artifact_validation=args.artifact_validation,
            container_name=args.container_name,
            mtp_depth=args.mtp_depth,
            scheduler_tokens=args.scheduler_tokens,
        ),
        args.output_directory,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
