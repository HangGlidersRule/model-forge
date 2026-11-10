"""Issue #35: reassembling a trunk-only Nano-Omni NVFP4 product into the VLM wrapper must keep every trunk tensor's
bytes, add every upstream tower tensor byte for byte, rename only the module names, and the validator must catch any
missing or altered tower. Tiny synthetic checkpoints; numpy-free (raw safetensors bytes)."""
from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "models" / "nemotron-3-nano-omni-r1"


def _load(rel: str):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_st(path: Path, tensors: dict[str, bytes]) -> None:
    header, off, data = {}, 0, b""
    for name, raw in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(raw)], "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        data += raw
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)


def _ckpt(root: Path, shards: dict[str, dict[str, bytes]], config: dict, extra: dict[str, str] | None = None) -> None:
    root.mkdir(parents=True)
    wm = {}
    for shard, tensors in shards.items():
        _write_st(root / shard, tensors)
        wm.update({k: shard for k in tensors})
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (root / "config.json").write_text(json.dumps(config))
    for name, text in (extra or {}).items():
        (root / name).write_text(text)


@pytest.fixture()
def fixture_dirs(tmp_path: Path) -> tuple[Path, Path]:
    trunk, upstream = tmp_path / "trunk", tmp_path / "upstream"
    qlayers = {"backbone.layers.1.mixer.experts.0.up_proj": {"quant_algo": "W4A16_NVFP4"},
               "lm_head": {"quant_algo": "W4A16_NVFP4"}}
    _ckpt(trunk, {"model-00001-of-00001.safetensors": {
        "backbone.embeddings.weight": b"emb-bytes",
        "backbone.layers.1.mixer.experts.0.up_proj.weight": b"fp4-packed",
        "backbone.layers.1.mixer.experts.0.up_proj.weight_scale": b"blockscale",
        "lm_head.weight": b"head-fp4", "lm_head.weight_scale": b"head-scale"}},
        {"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h",
         "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": qlayers,
                                 "ignore": [], "config_groups": {"g": {"targets": []}}}})
    (trunk / "hf_quant_config.json").write_text(json.dumps({"producer": {"name": "modelopt"}, "quantization": {
        "quant_algo": "MIXED_PRECISION", "quantized_layers": qlayers, "exclude_modules": []}}))
    _ckpt(upstream, {
        "model-00001-of-00002.safetensors": {"vision_model.radio_model.w": b"radio", "mlp1.0.weight": b"proj",
                                             "language_model.backbone.embeddings.weight": b"bf16-emb"},
        "model-00002-of-00002.safetensors": {"sound_encoder.w": b"audio", "sound_projection.w": b"sproj",
                                             "language_model.lm_head.weight": b"bf16-head"}},
        {"architectures": ["NemotronH_Nano_Omni_Reasoning_V3"], "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
         "vision_config": {"v": 1}, "sound_config": {"s": 1}, "llm_config": {"architectures": ["NemotronHForCausalLM"]}},
        {"modeling.py": "# wrapper code\n", "tokenizer.json": "{}", "chat_template.jinja": "{{ x }}",
         "preprocessor_config.json": "{}", "README.md": "upstream card", "SHA256SUMS.upstream": "x"})
    return trunk, upstream


def test_reassembly_keeps_bytes_and_adds_every_tower(fixture_dirs: tuple[Path, Path], tmp_path: Path) -> None:
    rea, val = _load("scripts/reassemble_omni_wrapper.py"), _load("eval/validate_omni_wrapper.py")
    trunk, upstream = fixture_dirs
    out = tmp_path / "out"
    r = rea.reassemble(trunk, upstream, out)
    assert r["lm_tensors"] == 5 and r["tower_tensors"] == 4 and r["quantized_layers"] == 2
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["architectures"] == ["NemotronH_Nano_Omni_Reasoning_V3"] and "vision_config" in cfg and "sound_config" in cfg
    assert set(cfg["quantization_config"]["quantized_layers"]) == {
        "language_model.backbone.layers.1.mixer.experts.0.up_proj", "language_model.lm_head"}
    d = val.tensor_digests(out)
    assert d["language_model.backbone.layers.1.mixer.experts.0.up_proj.weight"] == val.tensor_digests(trunk)[
        "backbone.layers.1.mixer.experts.0.up_proj.weight"]
    assert "language_model.backbone.embeddings.weight" in d and d["vision_model.radio_model.w"][2] == val.tensor_digests(upstream)["vision_model.radio_model.w"][2]
    assert not (out / "README.md").exists() and not (out / "SHA256SUMS.upstream").exists()
    assert (out / "modeling.py").read_text() == "# wrapper code\n"
    assert _run_validator(val, out, trunk, upstream, tmp_path / "rep.json") == 0
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["verdict"] == "PASS" and rep["towers"]["byte_identical"] == 4 and rep["trunk"]["byte_identical"] == 5


def _run_validator(val, out: Path, trunk: Path, upstream: Path, report: Path) -> int:
    import sys
    argv = sys.argv
    try:
        sys.argv = ["v", str(out), str(trunk), str(upstream), str(report)]
        return val.main()
    finally:
        sys.argv = argv


def test_validator_rejects_a_trunk_only_product(fixture_dirs: tuple[Path, Path], tmp_path: Path) -> None:
    """The bug in #35: a published 'Omni' product with no towers. The validator must report FAIL (not crash)."""
    val = _load("eval/validate_omni_wrapper.py")
    trunk, upstream = fixture_dirs
    assert _run_validator(val, trunk, trunk, upstream, tmp_path / "rep.json") == 1
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["verdict"] == "FAIL"
    assert rep["towers"]["byte_identical"] == 0 and rep["towers"]["expected"] == 4
    assert any("tower tensor set differs" in e for e in rep["errors"])
    assert any("lacks vision_config" in e for e in rep["errors"])


def test_validator_rejects_an_altered_tower(fixture_dirs: tuple[Path, Path], tmp_path: Path) -> None:
    rea, val = _load("scripts/reassemble_omni_wrapper.py"), _load("eval/validate_omni_wrapper.py")
    trunk, upstream = fixture_dirs
    out = tmp_path / "out"
    rea.reassemble(trunk, upstream, out)
    p = out / "model-towers.safetensors"
    raw = bytearray(p.read_bytes())
    raw[-1] ^= 0xFF  # flip one byte of the last tower tensor
    p.write_bytes(bytes(raw))
    assert _run_validator(val, out, trunk, upstream, tmp_path / "rep.json") == 1
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["verdict"] == "FAIL" and rep["towers"]["byte_identical"] == 3


def test_reassembly_refuses_names_outside_the_trunk(fixture_dirs: tuple[Path, Path]) -> None:
    rea = _load("scripts/reassemble_omni_wrapper.py")
    with pytest.raises(ValueError):
        rea.lm_name("vision_model.x")
    with pytest.raises(ValueError):
        rea.lm_name("language_model.backbone.x")
