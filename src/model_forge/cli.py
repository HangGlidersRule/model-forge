from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .corpus import load_builtin_suite
from .models import load_spec
from .public_export.detectors import DetectorError, load_fleet_hostname_denylist
from .public_export.exporter import (
    DEFAULT_PUBLIC_CONTACT,
    ExportError,
    ExportRequest,
    export_public,
)
from .public_export.verifier import (
    PublicVerifyError,
    PublicVerifyRequest,
    verify_public_export,
)
from .recipe import RecipeError, load_recipe
from .runner import Runner


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="model-forge",
        description="Reproducible model transform, quantization, and evaluation forge",
    )
    commands = root.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run the legacy OpenAI-compatible evaluation harness")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--output", "-o", type=Path, required=True)
    run.add_argument("--dry-run", action="store_true")

    recipe = commands.add_parser("recipe", help="Inspect generic build/transform recipes")
    recipe_sub = recipe.add_subparsers(dest="recipe_command", required=True)
    validate = recipe_sub.add_parser(
        "validate",
        help="Validate a schema-2 recipe and print its identity fields (does not execute it)",
    )
    validate.add_argument("path", type=Path)

    public_export = commands.add_parser(
        "public-export",
        help="Build a deterministic, scanned public repository tree",
    )
    public_export.add_argument("--source", type=Path, required=True)
    public_export.add_argument("--output", type=Path, required=True)
    public_export.add_argument("--manifest", type=Path, required=True)
    public_export.add_argument("--source-sha", required=True)
    public_export.add_argument("--fleet-hostname-denylist", type=Path)
    public_export.add_argument(
        "--public-contact",
        default=DEFAULT_PUBLIC_CONTACT,
    )
    public_export.add_argument("--replace", action="store_true")
    public_export.add_argument("--dry-run", action="store_true")

    public_verify = commands.add_parser(
        "public-verify",
        help="Independently verify an already-exported public repository tree",
    )
    public_verify.add_argument("--root", type=Path, required=True)
    public_verify.add_argument("--source-sha", required=True)
    public_verify.add_argument("--source-repo", type=Path, required=True)
    public_verify.add_argument(
        "--manifest",
        type=Path,
        default=Path("tools/public_export/public-files.yaml"),
    )
    public_verify.add_argument("--wheelhouse", type=Path, required=True)
    public_verify.add_argument("--wheelhouse-lock", type=Path)
    public_verify.add_argument(
        "--public-contact",
        default=DEFAULT_PUBLIC_CONTACT,
        help="Trusted public contact used when recomputing transformed bytes",
    )
    public_verify.add_argument("--fleet-hostname-denylist", type=Path)
    return root


def _run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    cases = [case for suite in spec.suites for case in load_builtin_suite(suite)]
    cells = len(spec.models) * len(spec.tracks) * len(cases) * spec.repeats
    if args.dry_run:
        print(f"Valid: {len(spec.models)} models, {len(spec.tracks)} tracks, {len(cases)} cases, {cells} request cells")
        return 0
    asyncio.run(Runner(spec, args.output).run())
    print(f"Results written to {args.output}")
    return 0


def _recipe_validate(root: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    path: Path = args.path
    if not path.is_file():
        root.error(f"recipe not found: {path}")
    try:
        recipe = load_recipe(path)
    except RecipeError as exc:
        print(f"Invalid recipe: {exc}", file=sys.stderr)
        return 2
    print(f"name:            {recipe.name}")
    print(f"family:          {recipe.family}")
    print(f"source model:    {recipe.source.model_id}")
    print(f"source revision: {recipe.source.revision}")
    print(f"artifact kind:   {recipe.outputs.artifact_kind}")
    print(f"config sha256:   {recipe.config_sha()}")
    return 0


def _public_export(args: argparse.Namespace) -> int:
    try:
        fleet_hostnames = (
            load_fleet_hostname_denylist(args.fleet_hostname_denylist)
            if args.fleet_hostname_denylist is not None
            else frozenset()
        )
        result = export_public(
            ExportRequest(
                source=args.source,
                output=args.output,
                manifest=args.manifest,
                source_sha=args.source_sha,
                replace=args.replace,
                dry_run=args.dry_run,
                public_contact=args.public_contact,
                fleet_hostnames=fleet_hostnames,
            )
        )
    except (DetectorError, ExportError) as error:
        print(f"Public export refused: {error}", file=sys.stderr)
        return 2
    action = "Validated" if result.dry_run else "Exported"
    print(
        f"{action} {result.file_count} files; payload tree sha256: "
        f"{result.payload_tree_sha256}"
    )
    return 0


def _public_verify(args: argparse.Namespace) -> int:
    try:
        fleet_hostnames = (
            load_fleet_hostname_denylist(args.fleet_hostname_denylist)
            if args.fleet_hostname_denylist is not None
            else frozenset()
        )
        result = verify_public_export(
            PublicVerifyRequest(
                root=args.root,
                source_sha=args.source_sha,
                source_repo=args.source_repo,
                manifest=args.manifest,
                wheelhouse=args.wheelhouse,
                wheelhouse_lock=args.wheelhouse_lock,
                public_contact=args.public_contact,
                fleet_hostnames=fleet_hostnames,
            )
        )
    except (DetectorError, PublicVerifyError) as error:
        print(f"Public export verification failed: {error}", file=sys.stderr)
        return 2
    print(
        f"Verified {result.file_count} files; payload tree sha256: "
        f"{result.payload_tree_sha256}; trusted wheel sha256 records: "
        f"{len(result.wheelhouse_evidence)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    root = parser()
    args = root.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "recipe" and args.recipe_command == "validate":
        return _recipe_validate(root, args)
    if args.command == "public-export":
        return _public_export(args)
    if args.command == "public-verify":
        return _public_verify(args)
    root.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
