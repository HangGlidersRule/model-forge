"""Tests for the diffusion quantization execution builder (diffusers path)."""

from pathlib import Path

from model_forge.modelopt.diffusion_execution import (
    DIFFUSERS_QUANT_ENTRY,
    QWEN_IMAGE_21_SLUG,
    build_diffusion_docker_run_plan,
    diffusion_quantize_cli_args,
)
from model_forge.modelopt.diffusion_policy import FP8_FAST, NVFP4_PRIMARY


def test_nvfp4_cli_args_complete() -> None:
    args = diffusion_quantize_cli_args(policy=NVFP4_PRIMARY, export_dir="/mnt/export", calib_size=32)
    assert "--model" in args and args[args.index("--model") + 1] == QWEN_IMAGE_21_SLUG
    assert args[args.index("--format") + 1] == "fp4"
    assert args[args.index("--quant-algo") + 1] == "max"
    assert args[args.index("--calib-size") + 1] == "32"
    assert args[args.index("--n-steps") + 1] == "20"
    assert "--quantize-mha" not in args  # primary recipe: no MHA quantization


def test_fp8_args_no_quant_algo() -> None:
    args = diffusion_quantize_cli_args(policy=FP8_FAST, export_dir="/mnt/export", calib_size=32)
    assert args[args.index("--format") + 1] == "fp8"
    assert "--quant-algo" not in args
    assert "--quantize-mha" not in args


def test_int8_rejected() -> None:
    import pytest

    from model_forge.modelopt.diffusion_policy import DiffusionPolicy

    int8_policy = DiffusionPolicy(name="x", quant_format="int8")
    with pytest.raises(ValueError, match="int8"):
        diffusion_quantize_cli_args(policy=int8_policy, export_dir="/mnt/export", calib_size=8)


def test_docker_plan_applies_overlay_and_env() -> None:
    plan = build_diffusion_docker_run_plan(
        docker_bin="docker",
        image="ghcr.io/hangglidersrule/model-forge-modelopt:0.46.0rc2-43fd41a",
        export_dir=Path("/data/export"),
        hf_cache=Path("/data/hf-cache"),
        policy=NVFP4_PRIMARY,
        calib_size=32,
    )
    argv = plan.argv
    assert "--gpus=all" in argv
    assert "-e NCCL_CUMEM_ENABLE=0" in " ".join(argv)
    # overlay applied: bash -c invokes the overlay then quantize.py
    assert "bash" in argv and "-c" in argv
    cmd = argv[-1]
    assert "qwen-image-21-overlay.py" in cmd
    assert DIFFUSERS_QUANT_ENTRY in cmd
    assert "--model qwen-image-2.1" in cmd


def test_docker_plan_without_overlay() -> None:
    plan = build_diffusion_docker_run_plan(
        docker_bin="docker",
        image="img",
        export_dir=Path("/data/export"),
        hf_cache=Path("/data/hf"),
        policy=FP8_FAST,
        calib_size=8,
        apply_overlay=False,
    )
    argv = plan.argv
    idx = argv.index(DIFFUSERS_QUANT_ENTRY)
    assert argv[idx - 1] == "python"
    assert "--model" in argv[idx + 1:]
