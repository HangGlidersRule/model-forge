"""Tests for quantization config generation and ignore list construction."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from model_forge.experiment import (
    ExperimentConfig,
    effective_quantizer_ignore,
    load_experiment,
)

_script = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "qwen3_8"
    / "quantize_qwen38_nvfp4.py"
)
_spec = importlib.util.spec_from_file_location("quantize_qwen38_nvfp4", _script)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["quantize_qwen38_nvfp4"] = _mod
_spec.loader.exec_module(_mod)

from quantize_qwen38_nvfp4 import (  # type: ignore[import-not-found]
    build_ignore_list,
    graft_source_mtp,
)


@pytest.fixture
def cfg() -> ExperimentConfig:
    path = Path(__file__).resolve().parent.parent / "recipes" / "qwen3.8-27b" / "r3-nvfp4.yaml"
    return load_experiment(path)


def test_ignore_list_contents(cfg: ExperimentConfig) -> None:
    ignore = build_ignore_list(cfg.quantization.ignore)
    assert "lm_head" in ignore
    assert "re:.*visual.*" in ignore
    assert "re:.*mtp.*" in ignore
    assert "re:.*conv1d.*" in ignore


def test_quantization_config_values(cfg: ExperimentConfig) -> None:
    qcfg = cfg.quantization
    assert qcfg.scheme == "NVFP4"
    assert qcfg.targets == "Linear"
    assert qcfg.group_size == 16
    assert qcfg.calibration_samples == 32
    assert qcfg.max_sequence_length == 8192
    assert qcfg.pipeline == "basic"
    assert qcfg.shard_size == "5GB"


def test_keep_bf16_includes_required(cfg: ExperimentConfig) -> None:
    keep = cfg.quantization.keep_bf16
    required = {"mtp", "visual", "conv1d", "norms", "embeddings", "lm_head"}
    assert required.issubset(set(keep))


def test_ignore_list_no_duplicates(cfg: ExperimentConfig) -> None:
    ignore = build_ignore_list(cfg.quantization.ignore)
    assert len(ignore) == len(set(ignore))


def test_ignore_list_never_loses_a_declared_protection(cfg: ExperimentConfig) -> None:
    ignore = tuple(build_ignore_list(cfg.quantization.ignore))
    assert (
        effective_quantizer_ignore(cfg.quantization.targets, ignore, cfg.quantization.keep_bf16)
        == ignore
    )


def test_graft_source_mtp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    names = [f"mtp.layers.0.fake_{i}.weight" for i in range(15)]
    (source / "source.safetensors").write_bytes(b"fixture")
    source_map = {name: "source.safetensors" for name in names}
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 60}, "weight_map": source_map})
    )
    output_index = {
        "metadata": {"total_size": 100},
        "weight_map": {"model.fake.weight": "model.safetensors"},
    }
    (output / "model.safetensors.index.json").write_text(json.dumps(output_index))
    (output / "config.json").write_text(json.dumps({"text_config": {}, "quantization_config": {"ignore": []}}))

    class FakeTensor:
        dtype = "torch.bfloat16"

        def numel(self) -> int:
            return 2

        def element_size(self) -> int:
            return 2

    class FakeSafeOpen:
        def __enter__(self) -> "FakeSafeOpen":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get_tensor(self, _name: str) -> FakeTensor:
            return FakeTensor()

    saved: dict[str, object] = {}
    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = lambda *_args, **_kwargs: FakeSafeOpen()  # type: ignore[attr-defined]
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.save_file = lambda tensors, target, metadata=None: saved.update(  # type: ignore[attr-defined]
        tensors=tensors, target=target, metadata=metadata
    )
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)

    assert graft_source_mtp(source, output) == 15
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert all(index["weight_map"][name] == "model-mtp-bf16.safetensors" for name in names)
    assert index["metadata"]["total_size"] == 160
    config = json.loads((output / "config.json").read_text())
    assert config["text_config"]["mtp_num_hidden_layers"] == 1
    assert len(config["quantization_config"]["ignore"]) == 15
    assert set(saved["tensors"]) == set(names)  # type: ignore[arg-type]
