"""NVIDIA ModelOpt quantization helpers for Qwen3.8."""

from __future__ import annotations

from model_forge.modelopt.calibration import CalibrationContract, default_calibration_contract
from model_forge.modelopt.pin import (
    EXPECTED_COMMIT,
    EXPECTED_VERSION,
    EXPECTED_WHEEL_SHA256,
    OMLP_RECIPE,
    OPTIONAL_W4A16_RECIPE,
    PRIMARY_RECIPE,
    ModelOptPin,
    load_pin,
)
from model_forge.modelopt.policy import (
    FINAL_EXCLUSION_GLOBS,
    REPRESENTATIVE_QWEN_MODULES,
    PolicyError,
    load_quant_cfg,
    resolve_module_policy,
)
from model_forge.modelopt.validate import ValidationError, ValidationReport, validate_recipe_file

__all__ = [
    "EXPECTED_COMMIT",
    "EXPECTED_VERSION",
    "EXPECTED_WHEEL_SHA256",
    "FINAL_EXCLUSION_GLOBS",
    "OMLP_RECIPE",
    "OPTIONAL_W4A16_RECIPE",
    "PRIMARY_RECIPE",
    "REPRESENTATIVE_QWEN_MODULES",
    "CalibrationContract",
    "ModelOptPin",
    "PolicyError",
    "ValidationError",
    "ValidationReport",
    "default_calibration_contract",
    "load_pin",
    "load_quant_cfg",
    "resolve_module_policy",
    "validate_recipe_file",
]
