#!/usr/bin/env python3
"""Quantize Qwen3.8 BF16 sources with pinned NVIDIA ModelOpt (unified HF export).

This replaces the rejected llm-compressor compressed-tensors path. Default is a
safe dry-run that validates the recipe and prints the exact planned command.

``--execute`` performs *real* work: it runs the pinned ModelOpt container against
``examples/hf_ptq/hf_ptq.py`` with the exact calibration contract, runs the
fail-closed validators over the produced checkpoint, writes a SHA-256 manifest
and ``_SUCCESS.json``, and atomically promotes the partial export to its final
path. There is no intentional stop; there is no fabricated success.

Uses ModelOpt ``examples/hf_ptq/hf_ptq.py`` semantics and unified HF export
(``hf_quant_config.json``). Does not use the Diffusers adapter and does not
hand-edit quantization metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from model_forge.modelopt.calibration import default_calibration_contract
from model_forge.modelopt.execution import build_docker_run_plan, hf_ptq_cli_args
from model_forge.modelopt.finalize import finalize_export, promote_atomic
from model_forge.modelopt.identity import (
    ARTIFACT_ABSENT,
    ARTIFACT_MATCH,
    CANONICAL_SOURCE_KINDS,
    build_identity,
    build_identity_sha,
    classify_artifact,
    legacy_source_kind_identity_shas,
    normalize_source_kind,
)
from model_forge.modelopt.pin import (
    MIXED_W4A16_RECIPE,
    OMLP_RECIPE,
    PRIMARY_RECIPE,
    load_pin,
)
from model_forge.modelopt.validate import validate_recipe_file
from model_forge.pipeline import sha256_file

CANDIDATES = {
    "mixed_w4a16": MIXED_W4A16_RECIPE,
    "mlp_only": PRIMARY_RECIPE,
    "omlp": OMLP_RECIPE,
    # Compatibility alias retained for old command lines and build identities.
    "w4a16_optional": MIXED_W4A16_RECIPE,
}

# Exit codes shared with run_qwen38_modelopt_mcprue.sh.
EXIT_REFUSE_OVERWRITE = 5
EXIT_NEEDS_BUILD = 20

DEPRECATED_ALLOW_FLAG = "--allow-darkstar"


class _AllowAbliterated(argparse.Action):
    """``store_true`` that warns when the superseded option name is used."""

    def __init__(self, option_strings: list[str], dest: str, **kwargs: object) -> None:
        super().__init__(option_strings, dest, nargs=0, **kwargs)  # type: ignore[arg-type]

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        if option_string == DEPRECATED_ALLOW_FLAG:
            print(
                f"WARN: {DEPRECATED_ALLOW_FLAG} is deprecated; use --allow-abliterated.",
                file=sys.stderr,
            )
        setattr(namespace, self.dest, True)


def _source_kind(value: str) -> str:
    """Accept superseded source-kind spellings, normalized to the canonical term."""
    normalized = normalize_source_kind(value)
    if normalized != value:
        print(
            f"WARN: --source-kind {value} is deprecated; "
            f"using --source-kind {normalized}.",
            file=sys.stderr,
        )
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ModelOpt NVFP4 quantization (Qwen3.8)")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=sorted(CANDIDATES),
        default="mixed_w4a16",
        help=(
            "Selected mixed_w4a16 recipe (default); mlp_only and omlp are comparison "
            "candidates; w4a16_optional is a compatibility alias"
        ),
    )
    parser.add_argument(
        "--allow-abliterated",
        DEPRECATED_ALLOW_FLAG,
        dest="allow_abliterated",
        default=False,
        action=_AllowAbliterated,
        help=(
            "Explicit authorization required for heavy mutation of abliterated sources. "
            f"{DEPRECATED_ALLOW_FLAG} is a deprecated alias."
        ),
    )
    parser.add_argument(
        "--source-kind",
        type=_source_kind,
        choices=CANONICAL_SOURCE_KINDS,
        default="clean",
        help="Deprecated alias: darkstar (normalized to abliterated)",
    )
    parser.add_argument("--execute", action="store_true", help="Run heavy GPU quantization")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the exact docker run argv (JSON) and exit; no execution.",
    )
    parser.add_argument(
        "--check-build-identity",
        action="store_true",
        help=(
            "Compare --export-dir against the current build identity and exit: "
            f"0 validated/up to date, {EXIT_NEEDS_BUILD} build required, "
            f"{EXIT_REFUSE_OVERWRITE} refuse (stale or unvalidated artifact)."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=None, help="Final artifact directory")
    parser.add_argument("--modelopt-image", type=str, default=None)
    parser.add_argument("--modelopt-root", type=Path, default=None)
    parser.add_argument("--hf-cache", type=Path, default=None)
    parser.add_argument("--calib-cache", type=Path, default=None)
    parser.add_argument("--docker-bin", type=str, default="docker")
    parser.add_argument("--gpus", type=str, default="all")
    return parser.parse_args()


def _guard_abliterated(args: argparse.Namespace) -> None:
    if args.source_kind == "abliterated" and not args.allow_abliterated:
        print(
            "ERROR: Abliterated / R3 quantization requires explicit authorization for "
            "heavy artifact mutation. Pass --allow-abliterated to confirm.",
            file=sys.stderr,
        )
        sys.exit(3)


def main() -> None:
    args = _parse_args()

    pin = load_pin()
    recipe = CANDIDATES[args.candidate]
    cal = default_calibration_contract()
    recipe_report = validate_recipe_file(recipe)
    if not recipe_report.ok:
        print("ERROR: recipe validation failed:", file=sys.stderr)
        for err in recipe_report.errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(2)

    _guard_abliterated(args)

    if args.candidate == "w4a16_optional":
        print(
            "WARN: --candidate w4a16_optional is a compatibility alias for the selected "
            "mixed_w4a16 recipe.",
            file=sys.stderr,
        )

    stage = f"modelopt_{args.candidate}_{args.source_kind}"
    recipe_sha = sha256_file(recipe)
    # The identity is the provenance record: pin + candidate + recipe digest +
    # calibration + source. Idempotency compares this, not the recipe alone.
    provenance = build_identity(
        pin=pin,
        candidate=args.candidate,
        recipe=recipe,
        recipe_sha256=recipe_sha,
        calibration=cal,
        source_kind=args.source_kind,
        source_dir=args.source_dir,
    )
    cfg_sha = build_identity_sha(provenance)
    # Artifacts built before the source-kind rename recorded the same inputs under
    # the old term; accept those digests, and nothing else, as the same identity.
    equivalent_shas = legacy_source_kind_identity_shas(provenance)

    print("ModelOpt quantization plan:")
    print(f"  pin: {pin.version} @ {pin.git_commit}")
    print(f"  wheel: {pin.wheel_filename} sha256={pin.wheel_sha256}")
    print(f"  candidate: {args.candidate}")
    print(f"  recipe: {recipe} ({recipe_sha})")
    print(f"  calibration: {cal.datasets} sizes={cal.sizes} seq={cal.sequence_length} seed={cal.seed}")
    print(f"  source: {args.source_dir} kind={args.source_kind}")
    print("  export: unified HF (hf_quant_config.json) via examples/hf_ptq/hf_ptq.py")
    print(f"  coverage: {recipe_report.details.get('coverage')}")
    print(f"  build identity: {cfg_sha}")

    if args.check_build_identity:
        sys.exit(_check_build_identity(args, cfg_sha, equivalent_shas))

    if args.print_command:
        _print_command(args, recipe, cal)
        return

    if not args.execute:
        # Show the exact per-dataset calibration args so dry-run is auditable.
        script_args = hf_ptq_cli_args(
            source_dir=str(args.source_dir),
            export_path="<export_path>",
            recipe=str(recipe),
            cal=cal,
        )
        print("  hf_ptq.py args: " + " ".join(script_args))
        print("DRY RUN complete (pass --execute for GPU quantization).")
        return

    _execute(args, pin, recipe, cal, stage, cfg_sha, provenance, equivalent_shas)


def _require(value: object, flag: str) -> None:
    if value in (None, ""):
        print(f"ERROR: {flag} is required for this mode", file=sys.stderr)
        sys.exit(4)


def _check_build_identity(
    args: argparse.Namespace, cfg_sha: str, equivalent_shas: tuple[str, ...]
) -> int:
    """Report whether the existing artifact was built from this exact identity."""
    _require(args.export_dir, "--export-dir")
    assert args.export_dir is not None
    export_dir: Path = args.export_dir
    state, detail = classify_artifact(
        export_dir, cfg_sha, equivalent_identity_shas=equivalent_shas
    )
    if state == ARTIFACT_MATCH:
        print(f"Artifact already validated, skipping: {export_dir} ({detail})")
        return 0
    if state == ARTIFACT_ABSENT:
        print(f"Build required: {detail}")
        return EXIT_NEEDS_BUILD
    print(
        f"ERROR: Refusing overwrite of existing artifact: {export_dir} ({detail})",
        file=sys.stderr,
    )
    return EXIT_REFUSE_OVERWRITE


def _print_command(args: argparse.Namespace, recipe: Path, cal: object) -> None:
    _require(args.export_dir, "--export-dir")
    _require(args.modelopt_image, "--modelopt-image")
    _require(args.hf_cache, "--hf-cache")
    _require(args.calib_cache, "--calib-cache")
    assert args.export_dir is not None
    plan = build_docker_run_plan(
        docker_bin=args.docker_bin,
        image=str(args.modelopt_image),
        source_dir=args.source_dir,
        export_dir=args.export_dir,
        recipe=recipe,
        modelopt_root=args.modelopt_root,
        hf_cache=args.hf_cache,
        calib_cache=args.calib_cache,
        cal=cal,  # type: ignore[arg-type]
        gpus=args.gpus,
    )
    print(json.dumps({"argv": plan.argv, "hf_ptq_args": plan.hf_ptq_args}, indent=2))


def _execute(
    args: argparse.Namespace,
    pin: object,
    recipe: Path,
    cal: object,
    stage: str,
    cfg_sha: str,
    provenance: dict[str, object],
    equivalent_shas: tuple[str, ...],
) -> None:
    _require(args.export_dir, "--export-dir")
    _require(args.modelopt_image, "--modelopt-image")
    _require(args.hf_cache, "--hf-cache")
    _require(args.calib_cache, "--calib-cache")
    assert args.export_dir is not None

    if not args.source_dir.exists():
        print(f"ERROR: source missing: {args.source_dir}", file=sys.stderr)
        sys.exit(1)

    export_dir: Path = args.export_dir

    # Idempotency + refuse-overwrite: only an artifact carrying this exact build
    # identity is skipped. A stale marker (different recipe/pin/calibration/source)
    # or an unvalidated leftover directory is refused, never reused or clobbered.
    state, detail = classify_artifact(
        export_dir, cfg_sha, equivalent_identity_shas=equivalent_shas
    )
    if state == ARTIFACT_MATCH:
        print(f"Artifact already validated, skipping: {export_dir} ({detail})")
        return
    if state != ARTIFACT_ABSENT:
        print(
            f"ERROR: Refusing overwrite of existing artifact: {export_dir} ({detail})",
            file=sys.stderr,
        )
        sys.exit(EXIT_REFUSE_OVERWRITE)

    args.hf_cache.mkdir(parents=True, exist_ok=True)
    args.calib_cache.mkdir(parents=True, exist_ok=True)
    args.run_root.mkdir(parents=True, exist_ok=True)

    partial = export_dir.parent / f".{export_dir.name}.partial.{os.getpid()}"
    if partial.exists():
        print(f"ERROR: stale partial exists, refusing: {partial}", file=sys.stderr)
        sys.exit(EXIT_REFUSE_OVERWRITE)
    partial.mkdir(parents=True)

    plan = build_docker_run_plan(
        docker_bin=args.docker_bin,
        image=str(args.modelopt_image),
        source_dir=args.source_dir,
        export_dir=partial,
        recipe=recipe,
        modelopt_root=args.modelopt_root,
        hf_cache=args.hf_cache,
        calib_cache=args.calib_cache,
        cal=cal,  # type: ignore[arg-type]
        gpus=args.gpus,
    )

    (partial / "planned_command.json").write_text(
        json.dumps(
            {"argv": plan.argv, "hf_ptq_args": plan.hf_ptq_args, "provenance": provenance},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (partial / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("Running pinned ModelOpt container:")
    print("  " + " ".join(plan.argv))
    # Fail-closed: a non-zero container exit aborts before validation; errors are
    # never swallowed and no fake success is written.
    result = subprocess.run(plan.argv, check=False)
    if result.returncode != 0:
        print(
            f"ERROR: ModelOpt container exited {result.returncode}; leaving partial for "
            f"inspection: {partial}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    print("Container finished. Running fail-closed validators over export...")
    finalize_export(
        partial,
        source_dir=args.source_dir,
        recipe=recipe,
        stage=stage,
        config_sha=cfg_sha,
        provenance=provenance,  # type: ignore[arg-type]
    )

    promote_atomic(partial, export_dir)
    print(f"SUCCESS: validated artifact promoted to {export_dir}")


if __name__ == "__main__":
    main()
