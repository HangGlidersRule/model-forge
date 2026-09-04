"""Unit tests for the real ModelOpt hf_ptq command, restore rendering, finalize."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from modelopt_fakes import build_fake_export, write_fake_source

from model_forge.modelopt.calibration import default_calibration_contract
from model_forge.modelopt.execution import (
    HF_PTQ_ENTRY,
    build_docker_run_plan,
    calibration_cli_args,
    hf_ptq_cli_args,
)
from model_forge.modelopt.finalize import (
    PromotionError,
    finalize_export,
    promote_atomic,
    read_scale_samples,
)
from model_forge.modelopt.pin import PRIMARY_RECIPE, load_pin
from model_forge.modelopt.runtime import SnapshotError, render_restore_script
from model_forge.modelopt.validate import ValidationError
from model_forge.pipeline import SUCCESS_MARKER


def test_calibration_args_keep_two_dataset_sizes() -> None:
    cal = default_calibration_contract()
    args = calibration_cli_args(cal)
    joined = " ".join(args)
    # Exact per-dataset sizes; never collapsed to a single 1024 pool.
    assert "--calib_size 512,512" in joined
    assert "1024" not in joined
    assert "--calib_seq 2048" in joined
    assert "--batch_size 1" in joined
    # BF16 KV: the upstream default fp8_cast must be overridden to none.
    assert "--kv_cache_qformat none" in joined
    assert "--dataset cnn_dailymail,nemotron-post-training-dataset-v2" in joined


def test_hf_ptq_args_are_real_upstream_flags() -> None:
    cal = default_calibration_contract()
    args = hf_ptq_cli_args(
        source_dir="/src", export_path="/out", recipe="/r/recipe.yaml", cal=cal
    )
    assert "--pyt_ckpt_path" in args and "/src" in args
    assert "--export_path" in args and "/out" in args
    assert "--recipe" in args and "/r/recipe.yaml" in args
    assert "--trust_remote_code" in args
    assert "--export_fmt" in args and "hf" in args


def test_docker_run_plan_is_the_heavy_command(tmp_path: Path) -> None:
    cal = default_calibration_contract()
    plan = build_docker_run_plan(
        docker_bin="docker",
        image="ghcr.io/hangglidersrule/model-forge-modelopt:0.46.0rc2-43fd41a",
        source_dir=tmp_path / "src",
        export_dir=tmp_path / "out",
        recipe=PRIMARY_RECIPE,
        modelopt_root=tmp_path / "checkout",
        hf_cache=tmp_path / "hf",
        calib_cache=tmp_path / "calib",
        cal=cal,
    )
    argv = plan.argv
    joined = " ".join(argv)
    assert argv[0] == "docker" and argv[1] == "run"
    assert "--gpus=all" in argv
    assert "ghcr.io/hangglidersrule/model-forge-modelopt:0.46.0rc2-43fd41a" in argv
    assert HF_PTQ_ENTRY in argv
    # Real calibration contract flows into the container command.
    assert "512,512" in joined
    assert "--kv_cache_qformat" in argv
    # All required mounts present.
    assert any(m == "-v" for m in argv)
    assert any(":/mnt/source:ro" in a for a in argv)
    assert any(a.endswith(":/mnt/export") for a in argv)
    assert any(":/mnt/recipes:ro" in a for a in argv)
    assert any(":/mnt/hf_cache" in a for a in argv)
    assert any(":/mnt/calib_cache" in a for a in argv)


def test_docker_run_plan_honours_a_restricted_gpu_selection(tmp_path: Path) -> None:
    plan = build_docker_run_plan(
        docker_bin="docker",
        image="img",
        source_dir=tmp_path / "src",
        export_dir=tmp_path / "out",
        recipe=PRIMARY_RECIPE,
        modelopt_root=None,
        hf_cache=tmp_path / "hf",
        calib_cache=tmp_path / "calib",
        cal=default_calibration_contract(),
        gpus="device=1",
    )
    assert "--gpus=device=1" in plan.argv
    assert "--gpus=all" not in plan.argv


def test_render_restore_script_reconstructs_container() -> None:
    inspect = [
        {
            "Name": "/vllm-serve",
            "Config": {
                "Image": "vllm/vllm-openai:v0.27.1",
                "Env": ["VLLM_PORT=8000"],
                "Cmd": ["--model", "/models/current"],
            },
            "HostConfig": {
                "Binds": ["/mnt/d/artifacts:/models:ro"],
                "PortBindings": {"8000/tcp": [{"HostIp": "", "HostPort": "8000"}]},
                "RestartPolicy": {"Name": "unless-stopped"},
                "Runtime": "nvidia",
            },
        }
    ]
    script = render_restore_script(json.dumps(inspect))
    assert script.startswith("#!/usr/bin/env bash")
    assert "docker run -d --name vllm-serve" in script
    assert "--gpus=all" in script
    assert "vllm/vllm-openai:v0.27.1" in script
    assert "-p 8000:8000" in script
    assert "-e VLLM_PORT=8000" in script
    assert "-v /mnt/d/artifacts:/models:ro" in script
    assert "--restart=unless-stopped" in script
    assert "--model /models/current" in script


def test_render_restore_script_fails_closed_on_empty() -> None:
    with pytest.raises(SnapshotError):
        render_restore_script("[]")
    with pytest.raises(SnapshotError):
        render_restore_script(json.dumps([{"Name": "/x", "Config": {}, "HostConfig": {}}]))


def test_read_scale_samples_finds_finite_scales(tmp_path: Path) -> None:
    source = tmp_path / "src"
    write_fake_source(source)
    export = tmp_path / "out"
    build_fake_export(export, source)
    samples = read_scale_samples(export)
    assert samples
    for values in samples.values():
        assert all(v == v for v in values)  # no NaN


def test_finalize_export_writes_manifest_and_success(tmp_path: Path) -> None:
    source = tmp_path / "src"
    write_fake_source(source)
    export = tmp_path / "out.partial"
    build_fake_export(export, source)
    pin = load_pin()
    manifest = finalize_export(
        export,
        source_dir=source,
        recipe=PRIMARY_RECIPE,
        stage="modelopt_mlp_only_clean",
        config_sha="deadbeef",
        provenance={"git_commit": pin.git_commit},
    )
    assert (export / "manifest.sha256").exists()
    assert (export / SUCCESS_MARKER).exists()
    assert manifest.metrics["mtp_tensor_count"] == 15
    assert manifest.metrics["scale_tensors_checked"] >= 1


def test_finalize_rejects_missing_scales(tmp_path: Path) -> None:
    source = tmp_path / "src"
    write_fake_source(source)
    export = tmp_path / "out.partial"
    export.mkdir()
    (export / "hf_quant_config.json").write_text("{}", encoding="utf-8")
    (export / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"mtp.l.{i}.w": "m.safetensors" for i in range(15)}}),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        finalize_export(
            export,
            source_dir=source,
            recipe=PRIMARY_RECIPE,
            stage="s",
            config_sha="x",
            provenance={},
        )


def test_promote_atomic_refuses_overwrite_and_requires_success(tmp_path: Path) -> None:
    source = tmp_path / "src"
    write_fake_source(source)
    partial = tmp_path / "out.partial"
    build_fake_export(partial, source)

    final = tmp_path / "final"
    # Missing _SUCCESS -> refuse.
    with pytest.raises(PromotionError):
        promote_atomic(partial, final)

    finalize_export(
        partial,
        source_dir=source,
        recipe=PRIMARY_RECIPE,
        stage="s",
        config_sha="x",
        provenance={},
    )
    promote_atomic(partial, final)
    assert (final / SUCCESS_MARKER).exists()
    assert not partial.exists()

    # Second promotion into an existing final dir is refused.
    other = tmp_path / "out2.partial"
    build_fake_export(other, source)
    finalize_export(
        other, source_dir=source, recipe=PRIMARY_RECIPE, stage="s", config_sha="x", provenance={}
    )
    with pytest.raises(PromotionError):
        promote_atomic(other, final)
