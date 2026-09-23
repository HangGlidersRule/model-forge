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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from model_forge.modelopt.diffusion_policy import DiffusionPolicy

# Canonical entry point inside the pinned ModelOpt checkout / image.
DIFFUSERS_QUANT_ENTRY = "examples/diffusers/quantization/quantize.py"

# Container mount points (mirroring execution.py's contract).
CONTAINER_MODELOPT_ROOT = "/opt/modelopt"
CONTAINER_EXPORT = "/mnt/export"
CONTAINER_CALIB_DIR = "/mnt/calib"
CONTAINER_HF_CACHE = "/mnt/hf_cache"

# Registered upstream model slug for the onboarding exemplar.
QWEN_IMAGE_21_SLUG = "qwen-image-2.1"

FORMAT_BY_QUANT_FORMAT = {"nvfp4": "fp4", "fp8": "fp8", "int8": "int8"}


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
    slug = policy.extras.get("model_slug", QWEN_IMAGE_21_SLUG)
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
        str(PurePosixPath(export_dir) / f"{policy.name}.pt"),
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
    """
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
    argv.extend(["-w", CONTAINER_MODELOPT_ROOT, image])

    if apply_overlay:
        overlay_path = policy.extras.get(
            "overlay_path", "configs/modelopt/diffusion/qwen-image-21-overlay.py"
        )
        cmd = (
            f"python {overlay_path} {CONTAINER_MODELOPT_ROOT} && "
            f"python {DIFFUSERS_QUANT_ENTRY} {' '.join(quantize_args)}"
        )
        argv.extend(["bash", "-c", cmd])
    else:
        argv.extend(["python", DIFFUSERS_QUANT_ENTRY, *quantize_args])

    return DiffusionDockerRunPlan(
        argv=argv,
        container_export_path=CONTAINER_EXPORT,
        quantize_args=quantize_args,
    )
