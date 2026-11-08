"""Build the *real* NVIDIA ModelOpt ``examples/diffusers/quantization/quantize.py``
invocation for diffusion/DiT families — the image-path analog of
``execution.py`` (text path; do not touch).

Upstream argument syntax at the pinned commit
``43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`` (nvidia-modelopt 0.46.0rc2):

- ``--model`` is a registered ModelType slug; ``qwen-image-2.1`` exists only
  after the bridge-fork overlay (``configs/modelopt/diffusion/``) is applied.
- ``--format`` is one of ``int8|fp8|fp4``.
- ``--calib-size`` is an int (prompt count), ``--n-steps`` the denoising steps
  used during calibration, ``--batch-size`` the calibration batch.
- ``--quantize-mha`` additionally quantizes attention bmm (NVFP4) — off for the
  2.1 primary recipe.
- ``--quantized-torch-ckpt-save-path`` / ``--hf-ckpt-dir`` are the two export
  targets; ``--collect-method default`` selects amax collection.
- Calibration prompts come from the upstream dataset (Gustavosta
  Stable-Diffusion-Prompts); the CalibrationContract pins the dataset revision,
  prompt subset hash, and seeds.

The policy name, model slug, and overlay name are caller-supplied, so each must
be a single safe path segment and reaches the container only as its own argv
element. No shell ever parses them: when the overlay is applied, this module is
bind-mounted read-only into the container and run as the launcher (``main``),
which applies the overlay and then execs ``quantize.py``. The launcher half of
this module therefore imports only the standard library at runtime.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_forge.modelopt.diffusion_policy import DiffusionPolicy

# Canonical entry point inside the pinned ModelOpt checkout / image.
DIFFUSERS_QUANT_ENTRY = "examples/diffusers/quantization/quantize.py"

# Container mount points (mirroring execution.py's contract).
CONTAINER_MODELOPT_ROOT = "/opt/modelopt"
CONTAINER_EXPORT = "/mnt/export"
CONTAINER_CALIB_DIR = "/mnt/calib"
CONTAINER_HF_CACHE = "/mnt/hf_cache"
CONTAINER_LAUNCHER = "/opt/model_forge/diffusion_execution.py"

# Registered upstream model slug for the onboarding exemplar.
QWEN_IMAGE_21_SLUG = "qwen-image-2.1"

# Bridge-fork overlays live in this directory (relative to the ModelOpt root);
# ``extras["overlay_path"]`` names one file in it.
OVERLAY_DIR = "configs/modelopt/diffusion"
DEFAULT_OVERLAY = "qwen-image-21-overlay.py"

FORMAT_BY_QUANT_FORMAT = {"nvfp4": "fp4", "fp8": "fp8", "int8": "int8"}

_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def require_safe_segment(field: str, value: object) -> str:
    """Return ``value`` if it is one safe path segment, else raise ``ValueError``."""
    if not isinstance(value, str) or _SAFE_SEGMENT.fullmatch(value) is None:
        raise ValueError(
            f"{field} must be a single path segment matching "
            f"{_SAFE_SEGMENT.pattern!r}; got {value!r}"
        )
    return value


def diffusion_quantize_cli_args(
    *,
    policy: DiffusionPolicy,
    export_dir: str,
    calib_size: int,
    n_steps: int = 20,
    batch_size: int = 2,
    quantize_mha: bool | None = None,
) -> list[str]:
    """Return the argv *after* ``quantize.py`` (script-relative)."""
    name = require_safe_segment("policy name", policy.name)
    slug = require_safe_segment("model_slug", policy.extras.get("model_slug", QWEN_IMAGE_21_SLUG))
    fmt = FORMAT_BY_QUANT_FORMAT[policy.quant_format]
    if fmt == "int8":
        raise ValueError("int8 diffusion recipes are not part of the forge contract")
    args = [
        "--model",
        slug,
        "--model-dtype",
        "BFloat16",
        "--format",
        fmt,
        "--batch-size",
        str(batch_size),
        "--calib-size",
        str(calib_size),
        "--n-steps",
        str(n_steps),
        "--collect-method",
        "default",
        "--quantized-torch-ckpt-save-path",
        str(PurePosixPath(export_dir) / f"{name}.pt"),
        "--hf-ckpt-dir",
        str(PurePosixPath(export_dir) / "hf_ckpt"),
    ]
    if policy.quant_format == "nvfp4":
        args += ["--quant-algo", "max"]
    mha = policy.quantize_mha if quantize_mha is None else quantize_mha
    if mha:
        args.append("--quantize-mha")
    return args


@dataclass(frozen=True)
class DiffusionDockerRunPlan:
    """Fully-resolved container invocation for a single diffusion quant run."""

    argv: list[str]
    container_export_path: str
    quantize_args: list[str]


def build_diffusion_docker_run_plan(
    *,
    docker_bin: str,
    image: str,
    export_dir: Path,
    hf_cache: Path,
    policy: DiffusionPolicy,
    calib_size: int,
    n_steps: int = 20,
    batch_size: int = 2,
    apply_overlay: bool = True,
    gpus: str = "all",
    extra_env: dict[str, str] | None = None,
) -> DiffusionDockerRunPlan:
    """Assemble the exact ``docker run`` argv that executes ``quantize.py``.

    The bridge-fork overlay is applied inside the container at start (after the
    immutable checkout exists) unless ``apply_overlay`` is False. The overlay
    content-hash is enforced by the caller from ``diffusion_pin.json``.

    With the overlay on, the container runs this module (mounted read-only at
    ``CONTAINER_LAUNCHER``) as the launcher; the overlay name and every
    ``quantize.py`` argument are separate argv elements, never shell text.
    """
    overlay = (
        require_safe_segment("overlay_path", policy.extras.get("overlay_path", DEFAULT_OVERLAY))
        if apply_overlay
        else None
    )
    export_arg = str(PurePosixPath(CONTAINER_EXPORT))
    quantize_args = diffusion_quantize_cli_args(
        policy=policy,
        export_dir=export_arg,
        calib_size=calib_size,
        n_steps=n_steps,
        batch_size=batch_size,
    )

    argv: list[str] = [docker_bin, "run", "--rm", f"--gpus={gpus}", "--shm-size=16g"]
    env = {
        "HF_HOME": CONTAINER_HF_CACHE,
        "NCCL_CUMEM_ENABLE": "0",
        "TOKENIZERS_PARALLELISM": "false",
        **(extra_env or {}),
    }
    for key, value in env.items():
        argv.extend(["-e", f"{key}={value}"])
    argv.extend(["-v", f"{export_dir}:{CONTAINER_EXPORT}", "-v", f"{hf_cache}:{CONTAINER_HF_CACHE}"])
    if overlay is not None:
        argv.extend(["-v", f"{Path(__file__).resolve()}:{CONTAINER_LAUNCHER}:ro"])
    argv.extend(["-w", CONTAINER_MODELOPT_ROOT, image])

    if overlay is not None:
        # -I keeps the launcher's directory and PYTHON* variables off its sys.path.
        argv.extend(["python", "-I", CONTAINER_LAUNCHER, "--overlay", overlay, "--"])
        argv.extend(quantize_args)
    else:
        argv.extend(["python", DIFFUSERS_QUANT_ENTRY, *quantize_args])

    return DiffusionDockerRunPlan(
        argv=argv,
        container_export_path=CONTAINER_EXPORT,
        quantize_args=quantize_args,
    )


def main(argv: list[str] | None = None) -> int:
    """In-container launcher: apply the overlay, then exec ``quantize.py``.

    Usage: ``diffusion_execution.py --overlay NAME -- QUANTIZE_ARGS...``. Runs
    from the ModelOpt root (the plan's ``-w``).
    """
    args = sys.argv[1:] if argv is None else argv
    if "--" not in args:
        raise SystemExit("usage: diffusion_execution.py --overlay NAME -- QUANTIZE_ARGS...")
    split = args.index("--")
    parser = argparse.ArgumentParser(prog="diffusion_execution.py")
    parser.add_argument("--overlay", required=True)
    options = parser.parse_args(args[:split])
    try:
        overlay = require_safe_segment("overlay", options.overlay)
    except ValueError as exc:
        parser.error(str(exc))
    overlay_script = str(PurePosixPath(OVERLAY_DIR) / overlay)
    applied = subprocess.run([sys.executable, overlay_script, CONTAINER_MODELOPT_ROOT], check=False)
    if applied.returncode != 0:
        return applied.returncode
    os.execv(sys.executable, [sys.executable, DIFFUSERS_QUANT_ENTRY, *args[split + 1 :]])


if __name__ == "__main__":
    raise SystemExit(main())
