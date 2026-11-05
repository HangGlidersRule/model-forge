"""Confining tests for the embed export validator's safetensors shard paths.

``validate_export.py`` joins every ``weight_map`` value onto the checkpoint
directory and later ``safe_open``s that path. These tests build a checkpoint
that passes candidate A only when those opens read a known-good shard, then
point the index (or a symlink standing in for a shard name) at that shard
outside the directory under test. The validator must refuse. A passing result
means the structural checks ran on the outside files while the shard bytes in
the directory were never the ones certified.

They fail on purpose until shard names are a single safe path segment, resolve
inside the checkpoint, and the safetensors names in the directory equal the
names in the index, for both the export and the source, including the
no-index fallback. Refusal must not create or delete a file in those
directories, and it must not open a path that resolves outside them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "models/nemotron-3.5-embed-8b/eval/validate_export.py"
LAYERS = 34
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
EMBED = "language_model.embed_tokens.weight"
SOURCE_EMBED = (1.0, -0.5)
OTHER_EMBED = (9.0, 9.0)
OPEN_TIMEOUT_SECONDS = 15.0


class _ShardOpenBlocked(Exception):
    """The validator did not return before the open timeout."""


def _quant_names() -> list[str]:
    names: list[str] = []
    for layer in range(LAYERS):
        for proj in ATTN + MLP:
            sub = "self_attn" if proj in ATTN else "mlp"
            base = f"language_model.layers.{layer}.{sub}.{proj}"
            for part in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                names.append(f"{base}.{part}")
    return names


def _quant_tensors() -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name in _quant_names():
        if name.endswith(".weight"):
            tensors[name] = torch.tensor([1], dtype=torch.uint8)
        elif name.endswith(".weight_scale"):
            tensors[name] = torch.tensor([1.0], dtype=torch.float8_e4m3fn)
        else:
            tensors[name] = torch.tensor([1.0], dtype=torch.float32)
    return tensors


def _embed(values: tuple[float, float]) -> dict[str, torch.Tensor]:
    return {EMBED: torch.tensor(list(values), dtype=torch.bfloat16)}


def _write_index(directory: Path, weight_map: dict[str, str]) -> None:
    payload = {"metadata": {"total_size": 1}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_configs(source: Path, export: Path) -> None:
    config = {"model_type": "mistral3", "architectures": ["Mistral3Model"]}
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    exported = dict(config)
    exported["quantization_config"] = {
        "quant_algo": "NVFP4",
        "ignore": ["lm_head", "language_model.lm_head"],
    }
    (export / "config.json").write_text(json.dumps(exported), encoding="utf-8")
    (export / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "kv_cache_quant_algo": None,
                    "exclude_modules": ["lm_head", "language_model.lm_head"],
                }
            }
        ),
        encoding="utf-8",
    )


def _split_index(protected: str, quant: str) -> dict[str, str]:
    weight_map = {EMBED: protected}
    for name in _quant_names():
        weight_map[name] = quant
    return weight_map


def _single_index(shard: str) -> dict[str, str]:
    return {name: shard for name in (EMBED, *_quant_names())}


def _snapshot(directory: Path) -> dict[str, bytes]:
    snap: dict[str, bytes] = {}
    for path in sorted(directory.rglob("*")):
        rel = str(path.relative_to(directory))
        if path.is_symlink():
            snap[rel] = b"link:" + os.readlink(path).encode()
        elif path.is_file():
            snap[rel] = path.read_bytes()
    return snap


def _run(
    export: Path,
    source: Path,
    report: Path,
    *,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [sys.executable, str(VALIDATOR), str(export), str(source), "A", str(report)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise _ShardOpenBlocked from exc


def _assert_unchanged(watched: list[tuple[Path, dict[str, bytes]]]) -> None:
    for directory, before in watched:
        assert _snapshot(directory) == before


def _assert_refused(result: subprocess.CompletedProcess[str], report: Path) -> None:
    combined = result.stdout + result.stderr
    assert result.returncode != 0, combined
    assert "VALIDATE-OK" not in result.stdout
    assert "VALIDATE-OK" not in result.stderr
    if report.is_file():
        assert '"verdict": "PASS"' not in report.read_text(encoding="utf-8")


def _expect_refusal(
    export: Path,
    source: Path,
    report: Path,
    watched: list[Path],
    *,
    timeout: float = 60,
) -> None:
    before = [(directory, _snapshot(directory)) for directory in watched]
    try:
        result = _run(export, source, report, timeout=timeout)
    except _ShardOpenBlocked:
        _assert_unchanged(before)
        pytest.fail("validator blocked while opening a shard path")
    _assert_unchanged(before)
    _assert_refused(result, report)


def _source_with_embed(directory: Path, embed_file: Path, index_value: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(embed_file, directory / Path(index_value).name)
    _write_index(directory, {EMBED: index_value})


@pytest.fixture(scope="module")
def good_shards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("good-shards")
    embed = _embed(SOURCE_EMBED)
    save_file(embed, directory / "protected.safetensors")
    save_file(_quant_tensors(), directory / "quant.safetensors")
    save_file({**embed, **_quant_tensors()}, directory / "model.safetensors")
    return directory


def _copy_outside(tmp: Path, good_shards: Path) -> Path:
    outside = tmp / "outside"
    outside.mkdir()
    for name in ("protected.safetensors", "quant.safetensors", "model.safetensors"):
        shutil.copyfile(good_shards / name, outside / name)
    return outside


def _standard_source(tmp: Path, good_shards: Path) -> Path:
    source = tmp / "source"
    _source_with_embed(source, good_shards / "protected.safetensors", "model.safetensors")
    return source


def _bad_basename_copies(export: Path) -> None:
    """Local files whose names match the outside shards, with different bytes."""
    (export / "protected.safetensors").write_bytes(b"not-the-protected-shard")
    (export / "quant.safetensors").write_bytes(b"not-the-quantized-shard")


def test_relative_single_segment_shards_validate_ok(tmp_path: Path, good_shards: Path) -> None:
    """A checkpoint whose index names the shards sitting in the directory passes."""
    source = _standard_source(tmp_path, good_shards)
    export = tmp_path / "export"
    export.mkdir()
    shutil.copyfile(good_shards / "model.safetensors", export / "model.safetensors")
    _write_index(export, _single_index("model.safetensors"))
    _write_configs(source, export)
    report = tmp_path / "report.json"
    result = _run(export, source, report, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VALIDATE-OK" in result.stdout
    body = json.loads(report.read_text(encoding="utf-8"))
    assert body["verdict"] == "PASS"
    assert body["protected_tensors"] == {"expected": 1, "byte_identical": 1}
    assert body["coverage"]["nvfp4"] == 238


def test_local_protected_bytes_must_match_source(tmp_path: Path, good_shards: Path) -> None:
    """The byte check is real: a local shard that differs from the source fails."""
    source = _standard_source(tmp_path, good_shards)
    export = tmp_path / "export"
    export.mkdir()
    save_file(
        {**_embed(OTHER_EMBED), **_quant_tensors()},
        export / "model.safetensors",
    )
    _write_index(export, _single_index("model.safetensors"))
    _write_configs(source, export)
    report = tmp_path / "report.json"
    result = _run(export, source, report, timeout=60)
    _assert_refused(result, report)
    assert "bytes differ from source" in result.stdout


def test_absolute_weight_map_does_not_certify_outside_shards(
    tmp_path: Path, good_shards: Path
) -> None:
    """Absolute weight_map values must not certify shards outside the export."""
    outside = _copy_outside(tmp_path, good_shards)
    source = _standard_source(tmp_path, good_shards)
    export = tmp_path / "export"
    export.mkdir()
    _bad_basename_copies(export)
    (export / "uploaded-shard.safetensors").write_bytes(b"unread-upload-bytes")
    _write_index(
        export,
        _split_index(str(outside / "protected.safetensors"), str(outside / "quant.safetensors")),
    )
    _write_configs(source, export)
    _expect_refusal(
        export,
        source,
        tmp_path / "report.json",
        [export, source, outside],
    )


def test_parent_relative_weight_map_does_not_certify_outside_shards(
    tmp_path: Path, good_shards: Path
) -> None:
    """Parent-relative weight_map values must not leave the export directory."""
    outside = _copy_outside(tmp_path, good_shards)
    source = _standard_source(tmp_path, good_shards)
    export = tmp_path / "export"
    export.mkdir()
    _bad_basename_copies(export)
    _write_index(
        export,
        _split_index("../outside/protected.safetensors", "../outside/quant.safetensors"),
    )
    _write_configs(source, export)
    _expect_refusal(
        export,
        source,
        tmp_path / "report.json",
        [export, source, outside],
    )


def test_nested_parent_segments_do_not_certify_outside_shards(
    tmp_path: Path, good_shards: Path
) -> None:
    """A parent segment later in the value is still an escape."""
    outside = _copy_outside(tmp_path, good_shards)
    source = _standard_source(tmp_path, good_shards)
    export = tmp_path / "export"
    # The intermediate directory has to exist or the kernel rejects the open
    # before the parent segments are applied.
    (export / "nested").mkdir(parents=True)
    _bad_basename_copies(export)
    _write_index(
        export,
        _split_index(
            "nested/../../outside/protected.safetensors",
            "nested/../../outside/quant.safetensors",
        ),
    )
    _write_configs(source, export)
    _expect_refusal(
        export,
        source,
        tmp_path / "report.json",
        [export, source, outside],
    )


def test_parent_segment_that_resolves_inside_is_rejected(tmp_path: Path, good_shards: Path) -> None:
    """Reject parent segments even when the joined path lands back inside."""
    source = _standard_source(tmp_path, good_shards)
    export = tmp_path / "export"
    export.mkdir()
    shutil.copyfile(good_shards / "model.safetensors", export / "model.safetensors")
    _write_index(export, _single_index("../export/model.safetensors"))
    _write_configs(source, export)
    _expect_refusal(export, source, tmp_path / "report.json", [export, source])


def test_nested_shard_directory_is_rejected(tmp_path: Path, good_shards: Path) -> None:
    """A shard name must be one path segment, even if it stays inside."""
    source = _standard_source(tmp_path, good_shards)
    export = tmp_path / "export"
    nested = export / "shards"
    nested.mkdir(parents=True)
    shutil.copyfile(good_shards / "model.safetensors", nested / "model.safetensors")
    _write_index(export, _single_index("shards/model.safetensors"))
    _write_configs(source, export)
    _expect_refusal(export, source, tmp_path / "report.json", [export, source])


def test_symlink_shard_outside_root_is_not_certified(tmp_path: Path, good_shards: Path) -> None:
    """A single safe-looking name that is a symlink must not open the target."""
    outside = _copy_outside(tmp_path, good_shards)
    source = _standard_source(tmp_path, good_shards)
    export = tmp_path / "export"
    export.mkdir()
    (export / "model.safetensors").symlink_to(outside / "model.safetensors")
    _write_index(export, _single_index("model.safetensors"))
    _write_configs(source, export)
    _expect_refusal(
        export,
        source,
        tmp_path / "report.json",
        [export, source, outside],
    )


def test_unindexed_local_safetensors_fail_validation(tmp_path: Path, good_shards: Path) -> None:
    """Every safetensors file in the directory has to be named by the index."""
    source = _standard_source(tmp_path, good_shards)
    export = tmp_path / "export"
    export.mkdir()
    shutil.copyfile(good_shards / "model.safetensors", export / "model.safetensors")
    (export / "uploaded-shard.safetensors").write_bytes(b"different-shard-bytes")
    _write_index(export, _single_index("model.safetensors"))
    _write_configs(source, export)
    _expect_refusal(export, source, tmp_path / "report.json", [export, source])


def test_source_absolute_weight_map_does_not_replace_source_bytes(
    tmp_path: Path, good_shards: Path
) -> None:
    """The source index is joined the same way and must not select another file."""
    outside = tmp_path / "outside"
    outside.mkdir()
    save_file(_embed(OTHER_EMBED), outside / "model.safetensors")
    source = tmp_path / "source"
    source.mkdir()
    shutil.copyfile(good_shards / "protected.safetensors", source / "model.safetensors")
    _write_index(source, {EMBED: str(outside / "model.safetensors")})
    export = tmp_path / "export"
    export.mkdir()
    save_file({**_embed(OTHER_EMBED), **_quant_tensors()}, export / "model.safetensors")
    _write_index(export, _single_index("model.safetensors"))
    _write_configs(source, export)
    _expect_refusal(
        export,
        source,
        tmp_path / "report.json",
        [export, source, outside],
    )


def test_no_index_symlink_outside_root_is_not_certified(tmp_path: Path, good_shards: Path) -> None:
    """Without an index, model.safetensors is still opened and must stay inside."""
    outside = _copy_outside(tmp_path, good_shards)
    source = _standard_source(tmp_path, good_shards)
    export = tmp_path / "export"
    export.mkdir()
    (export / "model.safetensors").symlink_to(outside / "model.safetensors")
    _write_configs(source, export)
    _expect_refusal(
        export,
        source,
        tmp_path / "report.json",
        [export, source, outside],
    )


def _fifo_outside(tmp: Path, name: str = "blocked.safetensors") -> Path:
    outside = tmp / "outside"
    outside.mkdir()
    fifo = outside / name
    os.mkfifo(fifo)
    return fifo


def test_absolute_shard_path_is_not_opened(tmp_path: Path, good_shards: Path) -> None:
    """Looking up an absolute path must not open it before refusing."""
    fifo = _fifo_outside(tmp_path)
    source = _standard_source(tmp_path, good_shards)
    export = tmp_path / "export"
    export.mkdir()
    _write_index(export, _single_index(str(fifo)))
    _write_configs(source, export)
    _expect_refusal(
        export,
        source,
        tmp_path / "report.json",
        [export, source, fifo.parent],
        timeout=OPEN_TIMEOUT_SECONDS,
    )


def test_parent_relative_shard_path_is_not_opened(tmp_path: Path, good_shards: Path) -> None:
    """Looking up a parent-relative path must not open it before refusing."""
    _fifo_outside(tmp_path, "quant.safetensors")
    source = _standard_source(tmp_path, good_shards)
    export = tmp_path / "export"
    export.mkdir()
    _write_index(export, _single_index("../outside/quant.safetensors"))
    _write_configs(source, export)
    _expect_refusal(
        export,
        source,
        tmp_path / "report.json",
        [export, source, tmp_path / "outside"],
        timeout=OPEN_TIMEOUT_SECONDS,
    )


def test_symlink_shard_outside_root_is_not_opened(tmp_path: Path, good_shards: Path) -> None:
    """A resolved symlink must be the file that is opened, not a second lookup."""
    fifo = _fifo_outside(tmp_path)
    source = _standard_source(tmp_path, good_shards)
    export = tmp_path / "export"
    export.mkdir()
    (export / "model.safetensors").symlink_to(fifo)
    _write_index(export, _single_index("model.safetensors"))
    _write_configs(source, export)
    _expect_refusal(
        export,
        source,
        tmp_path / "report.json",
        [export, source, fifo.parent],
        timeout=OPEN_TIMEOUT_SECONDS,
    )


def test_no_index_symlink_outside_root_is_not_opened(tmp_path: Path, good_shards: Path) -> None:
    """The no-index fallback opens model.safetensors and must not follow a symlink out."""
    fifo = _fifo_outside(tmp_path)
    source = _standard_source(tmp_path, good_shards)
    export = tmp_path / "export"
    export.mkdir()
    (export / "model.safetensors").symlink_to(fifo)
    _write_configs(source, export)
    _expect_refusal(
        export,
        source,
        tmp_path / "report.json",
        [export, source, fifo.parent],
        timeout=OPEN_TIMEOUT_SECONDS,
    )
