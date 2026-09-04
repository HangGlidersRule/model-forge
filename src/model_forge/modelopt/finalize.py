"""Post-export finalization: fail-closed validation, manifest, atomic promotion.

Runs only against a *real* exported checkpoint. It never fabricates a success
marker: validators must pass, a SHA-256 manifest is written, then the partial
export directory is atomically promoted to its final path (refusing overwrite).

Scale tensors are read directly from the safetensors headers (no external
dependency) so ``no NaN/Inf/zero scales`` is checked against the actual bytes.
"""

from __future__ import annotations

import json
import os
import struct
from array import array
from pathlib import Path
from typing import Any

from model_forge.modelopt.validate import (
    SCALE_NAME_MARKERS,
    ValidationError,
    validate_checkpoint_contract,
)
from model_forge.pipeline import (
    SUCCESS_MARKER,
    SuccessManifest,
    sha256_file,
    sha256_tree,
)

_FLOAT_DTYPES = {"F64", "F32", "F16", "BF16"}


def _iter_safetensors(export_dir: Path) -> list[Path]:
    return sorted(export_dir.glob("*.safetensors"))


def _decode_floats(dtype: str, raw: bytes, limit: int) -> list[float]:
    if dtype == "F32":
        buf = array("f")
        buf.frombytes(raw[: 4 * limit])
        return [float(v) for v in buf]
    if dtype == "F64":
        buf = array("d")
        buf.frombytes(raw[: 8 * limit])
        return [float(v) for v in buf]
    if dtype == "F16":
        values: list[float] = []
        for i in range(0, min(len(raw), 2 * limit), 2):
            values.append(struct.unpack("<e", raw[i : i + 2])[0])
        return values
    if dtype == "BF16":
        values = []
        for i in range(0, min(len(raw), 2 * limit), 2):
            # bf16 holds the high 16 bits of a float32; pad the low half with zeros.
            (f,) = struct.unpack("<f", b"\x00\x00" + raw[i : i + 2])
            values.append(f)
        return values
    return []


def read_scale_samples(
    export_dir: Path,
    *,
    max_per_tensor: int = 8,
    max_tensors: int = 256,
) -> dict[str, list[float]]:
    """Sample scale tensor values from the exported safetensors shards.

    Returns a mapping of ``tensor_name -> [sampled float values]`` for float-typed
    scale tensors. Non-float scale encodings (e.g. packed FP8) are skipped for the
    value check but still counted as present by the caller.
    """
    samples: dict[str, list[float]] = {}
    for shard in _iter_safetensors(export_dir):
        with shard.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
            data_start = 8 + header_len
            for name, meta in header.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                if not any(marker in name for marker in SCALE_NAME_MARKERS):
                    continue
                dtype = meta.get("dtype")
                if dtype not in _FLOAT_DTYPES:
                    continue
                begin, end = meta.get("data_offsets", (0, 0))
                handle.seek(data_start + begin)
                raw = handle.read(min(end - begin, max_per_tensor * 8))
                decoded = _decode_floats(dtype, raw, max_per_tensor)
                if decoded:
                    samples[name] = decoded
                if len(samples) >= max_tensors:
                    return samples
    return samples


def write_sha256_manifest(export_dir: Path, *, filename: str = "manifest.sha256") -> Path:
    """Write a ``sha256  path`` manifest over every file in ``export_dir``."""
    tree = sha256_tree(export_dir)
    lines = [f"{digest}  {rel}" for rel, digest in sorted(tree.items())]
    manifest_path = export_dir / filename
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def finalize_export(
    export_dir: Path,
    *,
    source_dir: Path,
    recipe: Path,
    stage: str,
    config_sha: str,
    provenance: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    require_scales: bool = True,
) -> SuccessManifest:
    """Validate a real export, write manifest + ``_SUCCESS``. Fail-closed.

    Raises ``ValidationError`` if any validator fails or (when ``require_scales``)
    no scale tensors are present. Writes ``manifest.sha256`` and ``_SUCCESS.json``
    into ``export_dir`` only after every check passes.
    """
    scale_values = read_scale_samples(export_dir)
    if require_scales and not scale_values:
        raise ValidationError(
            "No float scale tensors found in export; refusing to mark SUCCESS "
            "(a quantized NVFP4 checkpoint must carry scale tensors)"
        )

    report = validate_checkpoint_contract(
        export_dir,
        source_dir=source_dir,
        scale_values=scale_values or None,
        recipe_path=recipe,
    )
    report.raise_if_failed()

    write_sha256_manifest(export_dir)
    output_hashes = sha256_tree(export_dir)

    combined_metrics: dict[str, Any] = {
        "mtp_tensor_count": report.details.get("mtp_tensor_count"),
        "vision_tensor_count": report.details.get("vision_tensor_count"),
        "tensor_count": report.details.get("tensor_count"),
        "scale_tensors_checked": len(scale_values),
        **(metrics or {}),
    }
    manifest = SuccessManifest(
        stage=stage,
        command="examples/hf_ptq/hf_ptq.py",
        git_commit=str(provenance.get("git_commit", "")),
        config_sha=config_sha,
        output_hashes=output_hashes,
        metrics=combined_metrics,
    )
    (export_dir / SUCCESS_MARKER).write_text(manifest.to_json(), encoding="utf-8")
    return manifest


class PromotionError(RuntimeError):
    pass


def promote_atomic(partial_dir: Path, final_dir: Path) -> None:
    """Atomically promote ``partial_dir`` -> ``final_dir``, refusing overwrite.

    Requires a validated ``_SUCCESS.json`` in the partial directory so a partial
    or unvalidated export can never be promoted.
    """
    if final_dir.exists():
        raise PromotionError(f"Refusing overwrite of existing artifact: {final_dir}")
    marker = partial_dir / SUCCESS_MARKER
    if not marker.exists():
        raise PromotionError(
            f"Refusing to promote unvalidated export (missing {SUCCESS_MARKER}): {partial_dir}"
        )
    # Sanity: the marker must hash-match the files it claims (no post-validation drift).
    manifest = SuccessManifest.from_json(marker.read_text(encoding="utf-8"))
    for rel, expected in manifest.output_hashes.items():
        if rel == SUCCESS_MARKER:
            continue
        target = partial_dir / rel
        if not target.exists() or sha256_file(target) != expected:
            raise PromotionError(f"Export drift detected before promotion: {rel}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.rename(str(partial_dir), str(final_dir))
