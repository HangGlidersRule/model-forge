#!/usr/bin/env python3
"""Render a runtime restore artifact from a ``docker inspect`` capture.

Given a container's ``docker inspect`` JSON, emit a self-contained ``restore.sh``
that recreates the container (image, name, ports, env, mounts, restart policy,
original command). Used by the D:-only mcprue runner so restore does not depend
on a compose file we do not own.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from model_forge.modelopt.runtime import SnapshotError, render_restore_script


def main() -> None:
    parser = argparse.ArgumentParser(description="Render restore.sh from docker inspect JSON")
    parser.add_argument("--inspect", type=Path, required=True, help="docker inspect JSON file")
    parser.add_argument("--out", type=Path, required=True, help="restore.sh output path")
    parser.add_argument("--docker-bin", type=str, default="docker")
    args = parser.parse_args()

    try:
        script = render_restore_script(
            args.inspect.read_text(encoding="utf-8"), docker_bin=args.docker_bin
        )
    except SnapshotError as exc:
        print(f"ERROR: cannot render restore artifact: {exc}", file=sys.stderr)
        sys.exit(1)

    args.out.write_text(script, encoding="utf-8")
    args.out.chmod(0o755)
    print(f"Wrote restore artifact: {args.out}")


if __name__ == "__main__":
    main()
