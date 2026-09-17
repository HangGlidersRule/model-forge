"""Pinned NVIDIA ModelOpt toolchain metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from model_forge.pipeline import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]
PIN_PATH = REPO_ROOT / "configs" / "modelopt" / "pin.json"
RECIPES_DIR = REPO_ROOT / "configs" / "modelopt" / "recipes"

PRIMARY_RECIPE = RECIPES_DIR / "nvfp4_mlp_only_mse-kv_bf16.yaml"
OMLP_RECIPE = RECIPES_DIR / "nvfp4_omlp_only_mse-kv_bf16.yaml"
MIXED_W4A16_RECIPE = RECIPES_DIR / "w4a16_nvfp4_mse-fp8_attn-kv_bf16.yaml"
# Nemotron-3.5-Lightning-30B-A3B (nemotron_h, hybrid Mamba2+MoE+Attention):
# W4A16 NVFP4 on MLP/MoE projections, FP8 attention, Mamba2/SSM + MTP protected,
# BF16 KV. See recipe file for full protections.
LIGHTNING_W4A16_RECIPE = RECIPES_DIR / "w4a16_nvfp4_mse-fp8_attn-kv_bf16_nemotron_h.yaml"
# Qwen3.8-27B full-compute W4A4 NVFP4 candidate (static MSE weight scales, dynamic NVFP4
# activations, BF16 KV). Protections: vision/MTP/lm_head/embeddings/norms/routers BF16;
# GDN recurrence guts (in_proj_a/in_proj_b/conv1d) BF16; GDN projection matmuls
# (out_proj/in_proj_qkv/in_proj_z) W4A4 — the gittensor-style decode-speed change.
# Provenance: recipe sha256 f581bb8a2f8ae8be4dbe4120b46f0f576fec003184c775d7686ec22da47eb180
# = exact recipe that built the validated clean-full-w4a4 artifact (Qwen3.8-27B-clean-
# full-w4a4-modelopt, _SUCCESS.json recipe_sha256 match).
FULL_W4A4_RECIPE = RECIPES_DIR / "qwen38-full-w4a4-modelopt.yaml"
# Compatibility alias for callers written before the mixed recipe was selected.
OPTIONAL_W4A16_RECIPE = MIXED_W4A16_RECIPE

EXPECTED_COMMIT = "43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a"
EXPECTED_VERSION = "0.46.0rc2"
EXPECTED_WHEEL_SHA256 = "d6f6964b76c9e3f156ed1f3627d406b187c454614ab8e409a3796568cd487bbb"


@dataclass(frozen=True)
class ModelOptPin:
    package: str
    version: str
    git_commit: str
    git_tag: str
    wheel_filename: str
    wheel_sha256: str
    wheel_url: str
    calibration: dict[str, Any]
    raw: dict[str, Any]

    def provenance(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "version": self.version,
            "git_commit": self.git_commit,
            "git_tag": self.git_tag,
            "wheel_filename": self.wheel_filename,
            "wheel_sha256": self.wheel_sha256,
            "wheel_url": self.wheel_url,
            "primary_recipe": str(PRIMARY_RECIPE.relative_to(REPO_ROOT)),
            "primary_recipe_sha256": sha256_file(PRIMARY_RECIPE),
            "omlp_recipe_sha256": sha256_file(OMLP_RECIPE),
            "mixed_w4a16_recipe_sha256": sha256_file(MIXED_W4A16_RECIPE),
            "optional_w4a16_recipe_sha256": sha256_file(MIXED_W4A16_RECIPE),
            "entry_point": self.raw.get("entry_point"),
            "export": self.raw.get("export"),
            "calibration": self.calibration,
        }


@lru_cache(maxsize=1)
def load_pin(path: Path | None = None) -> ModelOptPin:
    pin_path = path or PIN_PATH
    raw = json.loads(pin_path.read_text(encoding="utf-8"))
    wheel = raw["wheel"]
    pin = ModelOptPin(
        package=raw["package"],
        version=raw["version"],
        git_commit=raw["git_commit"],
        git_tag=raw["git_tag"],
        wheel_filename=wheel["filename"],
        wheel_sha256=wheel["sha256"],
        wheel_url=wheel["url"],
        calibration=dict(raw["calibration"]),
        raw=raw,
    )
    if pin.git_commit != EXPECTED_COMMIT:
        raise ValueError(
            f"ModelOpt pin commit {pin.git_commit} != expected {EXPECTED_COMMIT}"
        )
    if pin.version != EXPECTED_VERSION:
        raise ValueError(
            f"ModelOpt pin version {pin.version} != expected {EXPECTED_VERSION}"
        )
    if pin.wheel_sha256 != EXPECTED_WHEEL_SHA256:
        raise ValueError(
            f"ModelOpt wheel sha256 {pin.wheel_sha256} != expected {EXPECTED_WHEEL_SHA256}"
        )
    return pin
