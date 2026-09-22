"""Tests for the diffusion serve profile (#38)."""

import json

import pytest

from model_forge.image.serve_profile import (
    DIFFUSION_FAMILIES,
    DiffusionServeProfile,
    DiffusionServeProfileError,
)

DIFFUSION_FIXTURE_ARTIFACT = (
    "/export/qi21-nvfp4/hf_ckpt"
)


def _profile(**overrides):
    base = dict(
        family="qwenimage21",
        behavior="base",
        format="nvfp4",
        repository_id="HangGlidersRule/Darkstar-QwenImage21-Base-ModelOpt-NVFP4",
        model_path=DIFFUSION_FIXTURE_ARTIFACT,
        container_name="vllm-darkstar-qwenimage21-base-nvfp4",
    )
    base.update(overrides)
    return DiffusionServeProfile(**base)


def test_valid_profile_constructs_and_validates() -> None:
    p = _profile()
    p.validate()
    assert p.alias == "darkstar-qwenimage21-base-nvfp4"
    assert p.serve_command()[:2] == ["vllm", "serve"]
    assert "--omni" in p.serve_command()


def test_ablit_behavior_rejected_for_qwen_image_21() -> None:
    with pytest.raises(DiffusionServeProfileError, match="only behavior='base'"):
        _profile(behavior="abliterated").validate()


def test_env_pins_mandatory() -> None:
    p = _profile(extra_env={"NCCL_CUMEM_ENABLE": "1"})
    with pytest.raises(DiffusionServeProfileError, match="NCCL_CUMEM_ENABLE"):
        p.validate()


def test_nvfp4_requires_cutlass_pin() -> None:
    p = _profile()
    p.extra_env.pop("SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND", None)
    with pytest.raises(DiffusionServeProfileError, match="cutlass"):
        # resolved_env injects defaults; simulate a profile that dropped them
        p.extra_env["SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND"] = "trtllm"
        p.validate()


def test_max_num_seqs_one_required() -> None:
    with pytest.raises(DiffusionServeProfileError, match="seed determinism"):
        _profile(max_num_seqs=16).validate()


def test_checkpoint_config_validation(tmp_path) -> None:
    d = tmp_path / "transformer"
    d.mkdir()
    qc = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "quant_type": "nvfp4",
        "ignore": ["*transformer_blocks.0.*", "*img_in"],
    }
    (d / "config.json").write_text(json.dumps({"quantization_config": qc}))
    _profile(model_path=str(tmp_path)).validate_checkpoint_config(tmp_path)

    # missing quant_type
    bad = dict(qc); bad.pop("quant_type")
    (d / "config.json").write_text(json.dumps({"quantization_config": bad}))
    with pytest.raises(DiffusionServeProfileError, match="quant_type"):
        _profile(model_path=str(tmp_path)).validate_checkpoint_config(tmp_path)

    # unprefixed ignore pattern (the #37 root cause)
    bad2 = dict(qc); bad2["ignore"] = ["transformer_blocks.0.*"]
    (d / "config.json").write_text(json.dumps({"quantization_config": bad2}))
    with pytest.raises(DiffusionServeProfileError, match="component-prefixed"):
        _profile(model_path=str(tmp_path)).validate_checkpoint_config(tmp_path)


def test_sglang_server_command() -> None:
    p = _profile(server="sglang-diffusion")
    p.validate()
    cmd = p.serve_command()
    assert cmd[0] == "sglang"
