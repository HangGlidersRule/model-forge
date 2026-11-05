#!/usr/bin/env python3
"""Write the tiny safetensors inputs the symlink publication tests drive."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

_SCALE = "backbone.layers.0.mixer.experts.0.up_proj.input_scale"
_WEIGHT = "backbone.layers.0.mixer.experts.0.up_proj.weight"
_SHARD = "model-00001-of-00001.safetensors"


def build_variants(base: Path, stats: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    save_file({_SCALE: torch.zeros(1)}, base / "model-input-scales.safetensors")
    stats.write_text(
        json.dumps(
            {
                "grid": [1.0, 2.0],
                "hist_lo": -30.0,
                "hist_hi": 20.0,
                "hist_bins": 2,
                "streams": {
                    "L0.up": {
                        "max": 1.0,
                        "n": 10,
                        "zero": 0,
                        "hist": [0, 10],
                        "sse": [1.0, 0.5],
                    }
                },
            }
        )
    )


def build_split(export: Path) -> None:
    export.mkdir(parents=True, exist_ok=True)
    save_file(
        {_WEIGHT: torch.zeros(2), _SCALE: torch.ones(1)},
        export / _SHARD,
    )
    (export / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {_WEIGHT: _SHARD, _SCALE: _SHARD}})
    )


def build_split_mse(export: Path, gpath: Path) -> None:
    export.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {"model.embed.weight": torch.zeros(2)}
    weight_map = {"model.embed.weight": _SHARD}
    for index in range(5934):
        key = f"backbone.layers.0.mixer.experts.{index}.up_proj.input_scale"
        tensors[key] = torch.ones(1)
        weight_map[key] = _SHARD
    save_file(tensors, export / _SHARD)
    (export / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    gpath.write_text(json.dumps({"mse": {"L0.up": 1.0}}))


def build_merge(base: Path, adapter: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    adapter.mkdir(parents=True, exist_ok=True)
    weight = "model.language_model.layers.0.mlp.down_proj.weight"
    save_file({weight: torch.zeros(2, 3)}, base / _SHARD)
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {weight: _SHARD}})
    )
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1.0}))
    save_file(
        {
            "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": torch.zeros(1, 3),
            "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": torch.zeros(2, 1),
        },
        adapter / "adapter_model.safetensors",
    )
    (adapter / "decision_head.json").write_text("{}\n")


def main(argv: list[str]) -> int:
    command = argv[1]
    if command == "variants":
        build_variants(Path(argv[2]), Path(argv[3]))
    elif command == "split":
        build_split(Path(argv[2]))
    elif command == "split-mse":
        build_split_mse(Path(argv[2]), Path(argv[3]))
    elif command == "merge":
        build_merge(Path(argv[2]), Path(argv[3]))
    else:
        sys.stderr.write(f"unknown fixture {command}\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
