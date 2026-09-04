"""Build the *real* NVIDIA ModelOpt ``examples/hf_ptq/hf_ptq.py`` invocation.

Argument syntax mirrors upstream ``examples/hf_ptq/hf_ptq.py`` at the pinned
commit ``43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a`` (nvidia-modelopt 0.46.0rc2):

- ``--calib_size`` is a *comma-separated* string; passing two dataset sizes keeps
  the calibration split (``512,512``) instead of collapsing to a single ``1024``.
- ``--dataset`` is a comma-separated list of registered dataset names.
- ``--calib_seq`` is the calibration sequence length (2048), *not* ``--seq``.
- ``--kv_cache_qformat none`` keeps the KV cache in BF16 (the upstream default is
  ``fp8_cast``, which would emit FP8 KV metadata we explicitly forbid).
- ``--recipe`` is authoritative for the quant layout; ``--trust_remote_code`` is
  required for Qwen3.8 remote code; export is unified HF (``hf_quant_config.json``).

The upstream global seed is ``RAND_SEED = 1234`` (matches our calibration
contract), so no CLI seed flag exists — it is asserted in tests instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from model_forge.modelopt.calibration import CalibrationContract

# Canonical entry point inside the pinned ModelOpt checkout / image.
HF_PTQ_ENTRY = "examples/hf_ptq/hf_ptq.py"

# Container mount points used by the D:-only remote runner. Kept here (not in the
# shell) so the exact contract is unit-testable without Docker.
CONTAINER_MODELOPT_ROOT = "/opt/modelopt"
CONTAINER_SOURCE = "/mnt/source"
CONTAINER_EXPORT = "/mnt/export"
CONTAINER_RECIPE_DIR = "/mnt/recipes"
CONTAINER_HF_CACHE = "/mnt/hf_cache"
CONTAINER_CALIB_CACHE = "/mnt/calib_cache"

# Registered upstream dataset names (SUPPORTED_DATASET_CONFIG keys) for the two
# calibration datasets. The contract's slugs map to these exact names.
DATASET_NAME_MAP = {
    "cnn_dailymail": "cnn_dailymail",
    "nemotron-post-training-dataset-v2": "nemotron-post-training-dataset-v2",
}


def resolve_dataset_names(cal: CalibrationContract) -> list[str]:
    """Map contract dataset slugs to upstream registered dataset names."""
    names: list[str] = []
    for slug in cal.datasets:
        if slug not in DATASET_NAME_MAP:
            raise ValueError(f"Unknown calibration dataset slug: {slug!r}")
        names.append(DATASET_NAME_MAP[slug])
    return names


def calibration_cli_args(cal: CalibrationContract) -> list[str]:
    """Return the calibration-related ``hf_ptq.py`` flags for this contract.

    Emits the two dataset sizes verbatim (e.g. ``512,512``) so calibration is
    never collapsed to a single ``1024`` pool.
    """
    if len(cal.sizes) != len(cal.datasets):
        raise ValueError(
            f"calibration sizes {cal.sizes} do not align with datasets {cal.datasets}"
        )
    sizes = ",".join(str(size) for size in cal.sizes)
    datasets = ",".join(resolve_dataset_names(cal))
    return [
        "--dataset",
        datasets,
        "--calib_size",
        sizes,
        "--calib_seq",
        str(cal.sequence_length),
        "--batch_size",
        str(cal.batch_size),
        "--kv_cache_qformat",
        "none",
    ]


def hf_ptq_cli_args(
    *,
    source_dir: str,
    export_path: str,
    recipe: str,
    cal: CalibrationContract,
    trust_remote_code: bool = True,
) -> list[str]:
    """Return the argv *after* the ``hf_ptq.py`` path (script-relative)."""
    args = [
        "--pyt_ckpt_path",
        source_dir,
        "--export_path",
        export_path,
        "--recipe",
        recipe,
        *calibration_cli_args(cal),
        "--export_fmt",
        "hf",
    ]
    if trust_remote_code:
        args.append("--trust_remote_code")
    return args


@dataclass(frozen=True)
class DockerRunPlan:
    """Fully-resolved container invocation for a single quantization run."""

    argv: list[str]
    container_export_path: str
    container_source_path: str
    container_recipe_path: str
    hf_ptq_args: list[str]


def _posix(container_dir: str, name: str) -> str:
    return str(PurePosixPath(container_dir) / name)


def build_docker_run_plan(
    *,
    docker_bin: str,
    image: str,
    source_dir: Path,
    export_dir: Path,
    recipe: Path,
    modelopt_root: Path | None,
    hf_cache: Path,
    calib_cache: Path,
    cal: CalibrationContract,
    gpus: str = "all",
    trust_remote_code: bool = True,
    extra_env: dict[str, str] | None = None,
) -> DockerRunPlan:
    """Assemble the exact ``docker run`` argv that executes ``hf_ptq.py``.

    Mounts source (read-only), export target, recipe directory (read-only), the
    HF cache and the calibration cache. When ``modelopt_root`` is provided the
    host checkout is mounted over the image's copy so the pinned examples are
    used verbatim; otherwise the image's baked-in checkout is used.
    """
    recipe_name = recipe.name
    container_recipe = _posix(CONTAINER_RECIPE_DIR, recipe_name)
    hf_ptq_args = hf_ptq_cli_args(
        source_dir=CONTAINER_SOURCE,
        export_path=CONTAINER_EXPORT,
        recipe=container_recipe,
        cal=cal,
        trust_remote_code=trust_remote_code,
    )

    argv: list[str] = [
        docker_bin,
        "run",
        "--rm",
        f"--gpus={gpus}",
        "--shm-size=16g",
    ]

    env = {
        "HF_HOME": CONTAINER_HF_CACHE,
        "HF_HUB_OFFLINE": "0",
        "MODELOPT_CALIB_CACHE": CONTAINER_CALIB_CACHE,
        "TOKENIZERS_PARALLELISM": "false",
        **(extra_env or {}),
    }
    for key, value in env.items():
        argv.extend(["-e", f"{key}={value}"])

    mounts = [
        f"{source_dir}:{CONTAINER_SOURCE}:ro",
        f"{export_dir}:{CONTAINER_EXPORT}",
        f"{recipe.parent}:{CONTAINER_RECIPE_DIR}:ro",
        f"{hf_cache}:{CONTAINER_HF_CACHE}",
        f"{calib_cache}:{CONTAINER_CALIB_CACHE}",
    ]
    if modelopt_root is not None:
        mounts.append(f"{modelopt_root}:{CONTAINER_MODELOPT_ROOT}:ro")
    for mount in mounts:
        argv.extend(["-v", mount])

    argv.extend(["-w", CONTAINER_MODELOPT_ROOT, image])
    argv.extend(["python", HF_PTQ_ENTRY, *hf_ptq_args])

    return DockerRunPlan(
        argv=argv,
        container_export_path=CONTAINER_EXPORT,
        container_source_path=CONTAINER_SOURCE,
        container_recipe_path=container_recipe,
        hf_ptq_args=hf_ptq_args,
    )
