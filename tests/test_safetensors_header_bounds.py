"""Failing spec for uncapped safetensors header and scale-span reads.

The readers under test trust an attacker-controlled checkpoint. They must
reject it before allocating the claimed header or the rest of a shard, and
before writing a success marker or manifest.

These tests fail while that behavior is present. They pass once every reader
caps the header at 100 MiB and scale sampling requires ``data_offsets`` to be
two integers with end >= begin, reading at most the sample size.

Negative spans are only consumed by ``read_scale_samples`` (and therefore
``finalize_export``). ``validate.py`` and ``diffusion_validate.py`` share the
uncapped header read only; both are covered below. No other hand-rolled
safetensors header parser exists in this repository.
"""

from __future__ import annotations

import json
import re
import struct
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from modelopt_fakes import build_fake_export, write_fake_source

from model_forge.modelopt.diffusion_policy import NVFP4_PRIMARY
from model_forge.modelopt.diffusion_validate import (
    _safetensors_header,
    validate_quantized_diffusion_checkpoint,
)
from model_forge.modelopt.finalize import finalize_export, read_scale_samples
from model_forge.modelopt.pin import PRIMARY_RECIPE
from model_forge.modelopt.validate import (
    ValidationError,
    _tensor_names_from_checkpoint,
    validate_checkpoint_contract,
    validate_mtp_bf16,
    validate_no_quantized_vision,
)
from model_forge.pipeline import SUCCESS_MARKER

# 100 MiB. huggingface/safetensors v0.4.5 caps headers at 100_000_000
# (safetensors/src/tensor.rs MAX_HEADER_SIZE), which is under this ceiling, so a
# fix that uses either limit rejects the length below.
_HEADER_CAP = 100 * 1024 * 1024
_OVERSIZE_HEADER = _HEADER_CAP + 1
# Large enough to show an uncapped read, small enough that the refusal path
# (header JSON plus at most one scale sample) stays under it.
_MAX_PEAK_BYTES = 1 * 1024 * 1024
_TAIL_BYTES = 2 * 1024 * 1024
_HEADER_RE = re.compile(r"header", re.IGNORECASE)
_OFFSET_RE = re.compile(r"data_offsets", re.IGNORECASE)


def _invoke(fn: Callable[[], Any]) -> tuple[Any, Exception | None, int]:
    tracemalloc.start()
    tracemalloc.reset_peak()
    caught: Exception | None = None
    value: Any = None
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 - the assertions classify the exception
        caught = exc
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, caught, peak


def _assert_refused(fn: Callable[[], Any], *, pattern: re.Pattern[str]) -> None:
    _value, caught, peak = _invoke(fn)
    assert isinstance(caught, ValidationError), (
        f"expected ValidationError, got {type(caught).__name__}: {caught!r}; peak={peak}"
    )
    assert pattern.search(str(caught)), f"{caught!r} does not match {pattern.pattern!r}"
    assert peak < _MAX_PEAK_BYTES, f"allocated {peak} bytes before refusing"


def _write_oversized_header(path: Path) -> None:
    """Shard whose 8-byte length is past the safetensors header cap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", _OVERSIZE_HEADER) + b"{")


def _write_scale_shard(path: Path, offsets: object, *, tail: int = 0) -> None:
    header = {
        "model.layers.0.weight_scale": {
            "dtype": "F32",
            "shape": [4],
            "data_offsets": offsets,
        }
    }
    encoded = json.dumps(header).encode("utf-8")
    blob = struct.pack("<4f", 0.5, 1.25, 0.25, 2.0) + (b"\xab" * tail)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)


def _invert_exported_scale_span(export_dir: Path) -> None:
    """Point the real scale tensor at a one-byte-negative span and extend the shard."""
    shard = export_dir / "model.safetensors"
    raw = shard.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len])
    blob = raw[8 + header_len :]
    scale_name = "model.language_model.layers.0.mlp.down_proj.weight_scale"
    begin, _end = header[scale_name]["data_offsets"]
    header[scale_name]["data_offsets"] = [begin, begin - 1]
    encoded = json.dumps(header).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob + (b"\xab" * _TAIL_BYTES))


def test_read_scale_samples_rejects_oversized_header_without_allocating(tmp_path: Path) -> None:
    shard = tmp_path / "model.safetensors"
    _write_oversized_header(shard)
    _assert_refused(lambda: read_scale_samples(tmp_path), pattern=_HEADER_RE)


def test_finalize_export_rejects_oversized_header_before_any_marker(tmp_path: Path) -> None:
    export = tmp_path / "export"
    shard = export / "model.safetensors"
    _write_oversized_header(shard)
    before = {path.name for path in export.iterdir()}

    _assert_refused(
        lambda: finalize_export(
            export,
            source_dir=tmp_path / "missing-source",
            recipe=PRIMARY_RECIPE,
            stage="s",
            config_sha="x",
            provenance={},
        ),
        pattern=_HEADER_RE,
    )

    after = {path.name for path in export.iterdir()}
    assert after == before
    assert SUCCESS_MARKER not in after
    assert "manifest.sha256" not in after


def test_tensor_names_reject_oversized_header_without_allocating(tmp_path: Path) -> None:
    _write_oversized_header(tmp_path / "model.safetensors")
    _assert_refused(lambda: _tensor_names_from_checkpoint(tmp_path), pattern=_HEADER_RE)


def test_validate_mtp_bf16_rejects_oversized_header(tmp_path: Path) -> None:
    _write_oversized_header(tmp_path / "model.safetensors")
    _assert_refused(lambda: validate_mtp_bf16(tmp_path), pattern=_HEADER_RE)


def test_validate_no_quantized_vision_rejects_oversized_header(tmp_path: Path) -> None:
    _write_oversized_header(tmp_path / "model.safetensors")
    _assert_refused(lambda: validate_no_quantized_vision(tmp_path), pattern=_HEADER_RE)


def test_validate_checkpoint_contract_rejects_oversized_header_without_allocating(
    tmp_path: Path,
) -> None:
    _write_oversized_header(tmp_path / "model.safetensors")
    (tmp_path / "hf_quant_config.json").write_text("{}", encoding="utf-8")

    report, caught, peak = _invoke(lambda: validate_checkpoint_contract(tmp_path))

    assert caught is None, f"contract swallowed the failure into a crash: {caught!r}"
    assert report is not None and report.ok is False
    assert peak < _MAX_PEAK_BYTES, f"allocated {peak} bytes before refusing"
    blob = "\n".join(report.errors)
    assert "checkpoint validation crashed" not in blob
    assert _HEADER_RE.search(blob), blob


def test_diffusion_header_rejects_oversized_header(tmp_path: Path) -> None:
    shard = tmp_path / "transformer" / "model.safetensors"
    _write_oversized_header(shard)
    _assert_refused(lambda: _safetensors_header(shard), pattern=_HEADER_RE)


def test_diffusion_checkpoint_rejects_oversized_header(tmp_path: Path) -> None:
    shard = tmp_path / "transformer" / "model.safetensors"
    _write_oversized_header(shard)
    _assert_refused(
        lambda: validate_quantized_diffusion_checkpoint(
            checkpoint_dir=tmp_path,
            policy=NVFP4_PRIMARY,
            source_revision="test",
        ),
        pattern=_HEADER_RE,
    )


def test_read_scale_samples_rejects_inverted_span_without_reading_the_shard(
    tmp_path: Path,
) -> None:
    _write_scale_shard(
        tmp_path / "model.safetensors",
        [0, -1],
        tail=_TAIL_BYTES,
    )
    _assert_refused(lambda: read_scale_samples(tmp_path), pattern=_OFFSET_RE)


def test_finalize_export_rejects_inverted_span_before_writing_success(tmp_path: Path) -> None:
    source = tmp_path / "src"
    export = tmp_path / "export"
    write_fake_source(source)
    build_fake_export(export, source)
    _invert_exported_scale_span(export)
    before = {path.name for path in export.iterdir()}

    _assert_refused(
        lambda: finalize_export(
            export,
            source_dir=source,
            recipe=PRIMARY_RECIPE,
            stage="s",
            config_sha="x",
            provenance={},
        ),
        pattern=_OFFSET_RE,
    )

    after = {path.name for path in export.iterdir()}
    assert after == before
    assert not (export / SUCCESS_MARKER).exists()
    assert not (export / "manifest.sha256").exists()


@pytest.mark.parametrize(
    "offsets",
    [
        [10, 0],
        [4, 3],
        [0],
        [0, 8, 16],
        ["0", "8"],
        [0.0, 8.0],
    ],
)
def test_read_scale_samples_rejects_offsets_that_are_not_a_rising_int_pair(
    tmp_path: Path,
    offsets: object,
) -> None:
    _write_scale_shard(
        tmp_path / "model.safetensors",
        offsets,
        tail=_TAIL_BYTES,
    )
    _assert_refused(lambda: read_scale_samples(tmp_path), pattern=_OFFSET_RE)
