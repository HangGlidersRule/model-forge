"""Issue #35: restore_router_bias.py must replace exactly the float32 tensors that differ from upstream, keep every other
byte, and refuse dtype/shape mismatches. Synthetic safetensors, stdlib only."""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "models" / "nemotron-3-nano-omni-r1" / "scripts" / "restore_router_bias.py"


def _load():
    spec = importlib.util.spec_from_file_location("restore_router_bias", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _st(path: Path, tensors: dict[str, tuple[str, list, bytes]]) -> None:
    header, off, data = {}, 0, b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        data += raw
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)


def _ckpt(root: Path, shards: dict[str, dict[str, tuple[str, list, bytes]]], extra: dict[str, str]) -> None:
    root.mkdir(parents=True)
    wm = {}
    for shard, tensors in shards.items():
        _st(root / shard, tensors)
        wm.update({k: shard for k in tensors})
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    for name, text in extra.items():
        (root / name).write_text(text)


def _f32(*vals: float) -> bytes:
    return struct.pack(f"<{len(vals)}f", *vals)


def _run(mod, *args: Path) -> int:
    argv = sys.argv
    try:
        sys.argv = ["r", *map(str, args)]
        return mod.main()
    finally:
        sys.argv = argv


@pytest.fixture()
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    up, ck = tmp_path / "up", tmp_path / "ck"
    _ckpt(up, {"u-1.safetensors": {
        "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias": ("F32", [3], _f32(55.764542, 55.769539, 55.771538)),
        "language_model.backbone.layers.1.mixer.gate.weight": ("BF16", [2], b"\x01\x02\x03\x04"),
        "vision_model.norm_mean": ("F32", [3], _f32(0.48, 0.45, 0.40))}}, {"config.json": "{}"})
    # The published product: same names, the bias rounded through BF16 (55.75), stored back as F32.
    _ckpt(ck, {"model-1.safetensors": {
        "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias": ("F32", [3], _f32(55.75, 55.75, 55.75)),
        "language_model.backbone.layers.1.mixer.experts.0.up_proj.weight": ("U8", [4], b"fp4!"),
        "language_model.lm_head.weight": ("BF16", [2], b"head")}}, {"config.json": '{"x": 1}', "manifest.sha256": "stale\n"})
    return up, ck


def test_restores_only_the_rounded_biases(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    mod = _load()
    up, ck = dirs
    out = tmp_path / "out"
    assert _run(mod, up, ck, out) == 0
    wm = mod.weight_map(out)
    assert mod.read(out, wm, "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias")[2] == _f32(55.764542, 55.769539, 55.771538)
    assert mod.read(out, wm, "language_model.backbone.layers.1.mixer.experts.0.up_proj.weight")[2] == b"fp4!"
    assert mod.read(out, wm, "language_model.lm_head.weight")[2] == b"head"
    assert (out / "config.json").read_text() == '{"x": 1}'
    assert "stale" not in (out / "manifest.sha256").read_text()
    assert mod.header(out / "model-1.safetensors") == mod.header(ck / "model-1.safetensors")


def test_trunk_names_without_prefix_match(tmp_path: Path, dirs: tuple[Path, Path]) -> None:
    """A trunk-only product names the tensor backbone.* (no language_model. prefix)."""
    mod = _load()
    up, _ = dirs
    ck = tmp_path / "trunk"
    _ckpt(ck, {"model-1.safetensors": {
        "backbone.layers.1.mixer.gate.e_score_correction_bias": ("F32", [3], _f32(55.75, 55.75, 55.75))}}, {"config.json": "{}"})
    out = tmp_path / "out2"
    assert _run(mod, up, ck, out) == 0
    wm = mod.weight_map(out)
    assert mod.read(out, wm, "backbone.layers.1.mixer.gate.e_score_correction_bias")[2] == _f32(55.764542, 55.769539, 55.771538)


def test_refuses_a_dtype_mismatch(tmp_path: Path, dirs: tuple[Path, Path]) -> None:
    mod = _load()
    up, _ = dirs
    ck = tmp_path / "bf16"
    _ckpt(ck, {"model-1.safetensors": {
        "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias": ("BF16", [3], b"\x00" * 6)}}, {"config.json": "{}"})
    with pytest.raises(SystemExit, match="RESTORE-FAILED"):
        _run(mod, up, ck, tmp_path / "out3")
    assert not (tmp_path / "out3").exists()


def test_refuses_a_missing_tensor(tmp_path: Path, dirs: tuple[Path, Path]) -> None:
    mod = _load()
    up, _ = dirs
    ck = tmp_path / "missing"
    _ckpt(ck, {"model-1.safetensors": {"language_model.lm_head.weight": ("BF16", [2], b"head")}}, {"config.json": "{}"})
    with pytest.raises(SystemExit, match="RESTORE-FAILED"):
        _run(mod, up, ck, tmp_path / "out4")
