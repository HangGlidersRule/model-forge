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
from .tune import TuneMatrix, render_markdown, run_sweep


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be comma-separated integers") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("must contain positive comma-separated integers")
    return result


def _lane_weights(value: str) -> tuple[tuple[int, float], ...]:
    try:
        result = tuple(
            (int(pair.split(":", 1)[0].strip()), float(pair.split(":", 1)[1].strip()))
            for pair in value.split(",")
            if pair.strip()
        )
    except (ValueError, IndexError) as exc:
        raise argparse.ArgumentTypeError(
            "must be comma-separated lane:weight pairs (for example 4:0.6,16:0.3)"
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError("must contain at least one lane:weight pair")
    return result


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

    tune = commands.add_parser("tune", help="Sweep speculative-decoding speed settings")
    tune.add_argument("--artifact-dir", type=Path, required=True)
    tune.add_argument("--served-name")
    tune.add_argument("--image", required=True)
    tune.add_argument("--results-dir", type=Path, default=Path("results/tune"))
    tune.add_argument("--mtp-min", type=int, default=1)
    tune.add_argument("--mtp-max", type=int, default=12)
    tune.add_argument("--lanes", type=_comma_separated_ints, default=(4, 16, 48))
    tune.add_argument(
        "--lane-weights",
        type=_lane_weights,
        default=None,
        help=(
            "comma-separated lane:weight pairs; defaults to 4:0.6,16:0.3,48:0.1 "
            "for default lanes and uniform weights for custom lanes"
        ),
    )
    tune.add_argument("--max-tokens", type=int, default=512)
    tune.add_argument("--temperature", type=float, default=0.7)
    tune.add_argument("--runs", type=int, default=5)
    tune.add_argument("--warmup", type=int, default=2)
    tune.add_argument("--force", action="store_true")
    tune.add_argument("--dry-run", action="store_true")
    tune.add_argument("--remote-host", required=True, help="remote vLLM host (required; do not hardcode)")
    tune.add_argument("--remote-user", default="devin")
    tune.add_argument("--ssh-key", default="~/.ssh/id_ed25519_aihost")
    tune.add_argument("--ssh-wsl-remote-artifact")

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


def _tune(root: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if not args.artifact_dir.is_dir():
        root.error(f"artifact directory not found: {args.artifact_dir}")
    served_name = args.served_name or args.artifact_dir.name
    try:
        matrix = TuneMatrix(
            mtp_min=args.mtp_min,
            mtp_max=args.mtp_max,
            lanes_k=args.lanes,
            lane_weights=args.lane_weights,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            runs=args.runs,
            warmup=args.warmup,
        )
    except ValueError as exc:
        root.error(str(exc))
    report = run_sweep(
        artifact_dir=str(args.artifact_dir),
        served_name=served_name,
        image=args.image,
        matrix=matrix,
        results_dir=args.results_dir,
        host=args.remote_host,
        user=args.remote_user,
        key=str(Path(args.ssh_key).expanduser()),
        ssh_win_artifact=args.ssh_wsl_remote_artifact,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(f"Winner: {report['winner'] or 'none'} — {report['winner_reason']}")
    table = render_markdown(report).splitlines()
    table_start = next(i for i, line in enumerate(table) if line.startswith("| candidate |"))
    print("\n".join(table[table_start : table_start + 2 + len(report["results"])]))
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
    if args.command == "tune":
        return _tune(root, args)
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
