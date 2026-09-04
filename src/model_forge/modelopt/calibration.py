"""Deterministic ModelOpt calibration contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from model_forge.modelopt.pin import load_pin


@dataclass(frozen=True)
class CalibrationContract:
    datasets: tuple[str, ...]
    sizes: tuple[int, ...]
    total_samples: int
    batch_size: int
    sequence_length: int
    seed: int
    layerwise: bool
    dataset_revisions: dict[str, str]
    ordering: str = "cnn_dailymail_then_nemotron_v2"
    preprocessing: str = "truncate_to_sequence_length_no_pad_report_truncation"
    image_text_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def truncation_report_template(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "truncated_count": None,
            "truncated_fraction": None,
            "max_raw_tokens_seen": None,
            "note": "Fill during calibration; required in stage SUCCESS metrics.",
        }


def default_calibration_contract() -> CalibrationContract:
    pin = load_pin()
    cal = pin.calibration
    return CalibrationContract(
        datasets=tuple(cal["datasets"]),
        sizes=tuple(cal["sizes"]),
        total_samples=int(cal["total_samples"]),
        batch_size=int(cal["batch_size"]),
        sequence_length=int(cal["sequence_length"]),
        seed=int(cal["seed"]),
        layerwise=bool(cal["layerwise"]),
        dataset_revisions={
            str(key): str(value) for key, value in cal["dataset_revisions"].items()
        },
        image_text_enabled=False,
    )
