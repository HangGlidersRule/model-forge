"""Deterministic Darkstar served-model profiles and Compose rendering."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from model_forge.release import precision_encoded_id_errors

_ALIAS = re.compile(
    r"^darkstar-(?P<family>[a-z0-9]+)-(?P<behavior>base|abliterated)-"
    r"(?P<format>bf16|nvfp4)$"
)
_CONTAINER = re.compile(r"^vllm-darkstar-[a-z0-9]+-(base|abliterated)-(bf16|modelopt)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FAMILIES = {
    "qwen38": {
        "repository_family": "Qwen3.8-27B",
        "local_path_family": "qwen3.8-27b",
    }
}


class ServeProfileError(ValueError):
    """Raised when a served-model profile violates the Darkstar naming contract."""


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ServeProfileError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ServeProfileError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ServeProfileError(f"{label} must contain a JSON object: {path}")
    return value


def _load_artifact_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json_object(path, "artifact manifest")
    required = {
        "artifact_path",
        "candidate_id",
        "precision_class",
        "success_marker_sha256",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ServeProfileError(f"artifact manifest is missing required fields: {missing}")
    return manifest


def _validate_success_marker(marker_bytes: bytes) -> None:
    try:
        marker = json.loads(marker_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ServeProfileError("_SUCCESS.json is not valid JSON") from error
    if not isinstance(marker, dict):
        raise ServeProfileError("_SUCCESS.json must contain a JSON object")
    required = {"schema_version", "stage", "config_sha"}
    missing = sorted(required - marker.keys())
    if missing:
        raise ServeProfileError(f"_SUCCESS.json is not a valid success marker; missing {missing}")
    if not all(isinstance(marker[key], str) and marker[key] for key in required):
        raise ServeProfileError("_SUCCESS.json identity fields must be non-empty strings")


@dataclass(frozen=True)
class ServeProfile:
    """A stable runtime identity distinct from the publication repository identity."""

    family: str
    behavior: str
    format: str
    repository_id: str
    model_path: str
    container_name: str
    mtp_depth: int
    scheduler_tokens: int
    artifact_identity: str | None = None
    artifact_precision_class: str | None = None
    artifact_success_sha256: str | None = None
    artifact_manifest: Path | None = None
    artifact_validation: str = "local"
    max_model_len: int = 126144
    max_num_seqs: int = 16
    kv_cache_dtype: str = "bf16"
    attention_backend: str = "FLASH_ATTN"
    image: str = "vllm/vllm-openai:v0.27.1"
    # Speculative decoding axis. "mtp" (default, in-model head), "dflash",
    # or "dflash2" (separate drafter). drafter_model is required for dflash/dflash2.
    spec_decode: str = "mtp"
    drafter_model: str | None = None
    drafter_tokens: int | None = None

    @property
    def alias(self) -> str:
        return f"darkstar-{self.family}-{self.behavior}-{self.format}"

    @property
    def compose_filename(self) -> str:
        return f"{self.alias}.yml"

    def validate(self) -> None:
        match = _ALIAS.fullmatch(self.alias)
        if match is None:
            raise ServeProfileError(f"invalid Darkstar served-model alias: {self.alias!r}")
        family_contract = _FAMILIES.get(self.family)
        if family_contract is None:
            raise ServeProfileError(
                f"unsupported Darkstar family {self.family!r}; add an explicit family contract"
            )
        if not _CONTAINER.fullmatch(self.container_name):
            raise ServeProfileError(f"invalid Darkstar container name: {self.container_name!r}")
        container_format = "modelopt" if self.format == "nvfp4" else "bf16"
        expected_container = (
            f"vllm-darkstar-{self.family}-{self.behavior}-{container_format}"
        )
        if self.container_name != expected_container:
            raise ServeProfileError(
                f"container name {self.container_name!r} does not match profile; "
                f"expected {expected_container!r}"
            )
        if not self.repository_id.startswith("HangGlidersRule/Darkstar-"):
            raise ServeProfileError("repository_id must remain a precision-encoded Darkstar repository")
        repository_name = self.repository_id.split("/")[-1]
        expected_prefix = (
            f"Darkstar-{family_contract['repository_family']}-"
            f"{self.behavior.capitalize()}-"
        )
        if not repository_name.startswith(expected_prefix):
            raise ServeProfileError(
                f"repository_id does not match family/behavior profile; expected prefix "
                f"{expected_prefix!r}"
            )

        requires_mixed_fp8 = (
            None
            if self.artifact_precision_class is None
            else "Mixed-FP8" in self.artifact_precision_class
        )
        precision_errors = precision_encoded_id_errors(
            self.repository_id, requires_mixed_fp8=requires_mixed_fp8
        )
        if precision_errors:
            raise ServeProfileError("; ".join(precision_errors))

        if self.format == "nvfp4" and "NVFP4" not in repository_name:
            raise ServeProfileError("nvfp4 alias must resolve to an NVFP4 repository identity")
        if self.format == "bf16" and not repository_name.endswith("-BF16"):
            raise ServeProfileError("bf16 alias must resolve to a BF16 repository identity")

        if self.model_path.startswith("/"):
            missing_metadata = [
                name
                for name, value in (
                    ("artifact_identity", self.artifact_identity),
                    ("artifact_precision_class", self.artifact_precision_class),
                    ("artifact_success_sha256", self.artifact_success_sha256),
                    ("artifact_manifest", self.artifact_manifest),
                )
                if not value
            ]
            if missing_metadata:
                raise ServeProfileError(
                    "local model_path requires explicit immutable artifact metadata: "
                    + ", ".join(missing_metadata)
                )
            # The missing-metadata guard above proves these three are present; assert so the
            # narrowing is explicit rather than implied.
            assert self.artifact_identity is not None
            assert self.artifact_precision_class is not None
            assert self.artifact_success_sha256 is not None
            assert self.artifact_manifest is not None
            if self.artifact_identity != repository_name:
                raise ServeProfileError(
                    "artifact_identity must exactly match the precision-encoded repository identity"
                )
            if self.artifact_precision_class not in repository_name:
                raise ServeProfileError(
                    "artifact_precision_class must be encoded in the repository identity"
                )
            if not _SHA256.fullmatch(str(self.artifact_success_sha256)):
                raise ServeProfileError("artifact_success_sha256 must be a full lowercase SHA-256")
            if set(self.artifact_success_sha256) == {"0"}:
                raise ServeProfileError("artifact_success_sha256 may not be a fabricated zero digest")
            if self.artifact_validation not in {"local", "attestation"}:
                raise ServeProfileError(
                    "artifact_validation must be 'local' or explicit checked-in 'attestation'"
                )
            manifest = _load_artifact_manifest(self.artifact_manifest)
            if manifest.get("artifact_path") != self.model_path:
                raise ServeProfileError(
                    "artifact manifest path must exactly match model_path; basename tokens are "
                    "not artifact identity"
                )
            if manifest.get("candidate_id") != self.artifact_identity:
                raise ServeProfileError("artifact manifest candidate_id does not match profile")
            if manifest.get("precision_class") != self.artifact_precision_class:
                raise ServeProfileError("artifact manifest precision_class does not match profile")
            if manifest.get("success_marker_sha256") != self.artifact_success_sha256:
                raise ServeProfileError("artifact manifest success marker SHA-256 does not match profile")

            if self.artifact_validation == "local":
                artifact_path = Path(self.model_path)
                if not artifact_path.is_dir():
                    raise ServeProfileError(f"local artifact directory does not exist: {artifact_path}")
                marker_path = artifact_path / "_SUCCESS.json"
                if not marker_path.is_file():
                    raise ServeProfileError(f"local artifact is missing _SUCCESS.json: {artifact_path}")
                marker_bytes = marker_path.read_bytes()
                actual_sha256 = hashlib.sha256(marker_bytes).hexdigest()
                if actual_sha256 != self.artifact_success_sha256:
                    raise ServeProfileError(
                        "actual _SUCCESS.json SHA-256 does not match artifact_success_sha256"
                    )
                _validate_success_marker(marker_bytes)
        elif self.model_path != self.repository_id:
            raise ServeProfileError(
                "non-local model_path must exactly equal the precision-encoded repository_id"
            )

        # KV-cache quantization matrix (SM120-measured; updated 2026-09-16):
        # - bf16 (native) KV is valid on EVERY backend, every runtime — always a
        #   supported co-option for any profile.
        # - fp8-family KV requires a backend with fp8-KV support: TRITON_ATTN or
        #   FLASHINFER. On stock 0.27.1 + FLASH_ATTN it hard-fails engine init on
        #   SM120 (requires FA3/SM90 or FA4/SM100). On stock 0.27.1 + TRITON_ATTN
        #   it boots but carries a LOAD-TRIGGERED race (cudaErrorIllegalAddress
        #   under concurrent thinking-load, with/without --no-async-scheduling,
        #   with/without prefix caching and mamba align mode) — measured
        #   2026-09-15. On the nvfp4kv nightly (0.27.2rc1.dev77+gac7509e2b with
        #   PR #49891 rebased + sm120 linear-V-scale overlay) fp8 KV under
        #   FLASHINFER is the AEON-recipe shape; treat as candidate-validate-first.
        # - nvfp4 KV is VALIDATED on the nvfp4kv nightly (0.27.2rc1.dev77) with
        #   FLASHINFER + MTP4 + mamba align: GPQA 168/198 = 84.85% (bf16-thinking
        #   baseline 85.35%), zero IMAs under full concurrency-30 thinking load,
        #   needles HIT to 210K tokens, KV pool 1.05M tokens at util 0.50.
        #   Measured 2026-09-16 on mcprue (RTX PRO 6000). NOT validated on stock
        #   0.27.1 (upstream there is SM100-trtllm-gen-only for nvfp4).
        # - turboquant KV remains refused (never validated on any of our runtimes).
        _QUANT_KV_OK_BACKENDS = {"TRITON_ATTN", "FLASHINFER"}
        _NVFP4KV_RUNTIME = "nvfp4kv-nightly-0.27.2rc1.dev77"
        if self.kv_cache_dtype != "bf16":
            if self.kv_cache_dtype in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
                if self.attention_backend not in _QUANT_KV_OK_BACKENDS:
                    raise ServeProfileError(
                        f"kv_cache_dtype {self.kv_cache_dtype!r} requires attention_backend in "
                        f"{sorted(_QUANT_KV_OK_BACKENDS)} (FLASH_ATTN does not support FP8 KV on "
                        "SM120 — engine init fails); got "
                        f"{self.attention_backend!r}"
                    )
                if self.attention_backend == "TRITON_ATTN":
                    # 2026-09-15: load-triggered IMA race under concurrent
                    # thinking-load on stock 0.27.1. Not a hard refuse (the
                    # pairing does boot), but profile validation flags it.
                    raise ServeProfileError(
                        "kv_cache_dtype fp8-family + TRITON_ATTN on stock 0.27.1 carries a "
                        "load-triggered cudaErrorIllegalAddress race (measured 2026-09-15); "
                        f"use FLASHINFER on the {_NVFP4KV_RUNTIME} runtime instead, or bf16"
                    )
            elif self.kv_cache_dtype == "nvfp4":
                if self.attention_backend != "FLASHINFER":
                    raise ServeProfileError(
                        "kv_cache_dtype nvfp4 requires attention_backend FLASHINFER "
                        f"(FA2-nvfp4 routing per PR #49891; got {self.attention_backend!r})"
                    )
                # nvfp4 KV is validated ONLY on the nvfp4kv nightly runtime; a
                # profile declaring nvfp4 KV on stock 0.27.1 is a config error.
                if "nvfp4kv" not in (self.image or ""):
                    raise ServeProfileError(
                        f"kv_cache_dtype nvfp4 requires the {_NVFP4KV_RUNTIME} image "
                        "(vllm-qwen38:nvfp4kv) — stock 0.27.1 gates nvfp4 KV to "
                        "SM100-trtllm-gen only"
                    )
            else:
                raise ServeProfileError(
                    f"kv_cache_dtype {self.kv_cache_dtype!r} is not validated for Darkstar "
                    "serve profiles (bf16 native, fp8-family, or nvfp4 on the nvfp4kv "
                    "runtime only)"
                )
        if self.mtp_depth < 1 or self.scheduler_tokens < 1:
            raise ServeProfileError("scheduler budget and speculative depth must be positive")
        if self.spec_decode not in {"mtp", "dflash", "dflash2"}:
            raise ServeProfileError(f"unsupported spec_decode method: {self.spec_decode}")
        if self.spec_decode in {"dflash", "dflash2"}:
            if not self.drafter_model:
                raise ServeProfileError(
                    f"{self.spec_decode} requires a drafter model (drafter_model)"
                )
            if not self.drafter_tokens or self.drafter_tokens < 1:
                raise ServeProfileError(
                    f"{self.spec_decode} requires drafter_tokens >= 1"
                )

    def compose(self) -> dict[str, object]:
        self.validate()
        service = self._service_config(self._command())
        return {
            "name": self.alias,
            "services": {
                "vllm": service,
            },
        }

    def _command(self) -> list[str]:
        return [
            "--model",
            self.model_path,
            "--served-model-name",
            self.alias,
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--max-num-batched-tokens",
            str(self.scheduler_tokens),
            "--kv-cache-dtype",
            self.kv_cache_dtype,
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--compilation-config",
            "2",
            "--speculative-config",
            self._speculative_config_json(),
        ]

    def _speculative_config_json(self) -> str:
        if self.spec_decode == "mtp":
            return f'{{"method":"mtp","num_speculative_tokens":{self.mtp_depth}}}'
        # DFlash and DFlash2 share the "dflash" vLLM method name; DFlash2 is
        # auto-detected from the drafter checkpoint architecture (DFlash2DraftModel).
        method = "dflash"
        kwargs: dict[str, object] = {
            "method": method,
            "model": self.drafter_model,
            "num_speculative_tokens": self.drafter_tokens,
        }
        return json.dumps(kwargs, sort_keys=True, separators=(",", ":"))

    def _service_config(self, command: list[str]) -> dict[str, object]:
        service: dict[str, object] = {
            "image": self.image,
            "container_name": self.container_name,
            "runtime": "nvidia",
            "restart": "unless-stopped",
            "ports": ["${VLLM_PORT:-8000}:8000"],
        }
        if self.model_path.startswith("/"):
            service["volumes"] = [f"{self.model_path}:{self.model_path}:ro"]
        service.update(
            {
                # The attention backend is part of the frozen profile and vLLM only accepts it
                # through this environment variable, so it is baked as a fixed value rather than
                # exposed as an operator override.
                "environment": [f"VLLM_ATTENTION_BACKEND={self.attention_backend}"],
                "command": command,
                "deploy": {
                    "resources": {
                        "reservations": {
                            "devices": [
                                {
                                    "driver": "nvidia",
                                    "count": "all",
                                    "capabilities": ["gpu"],
                                }
                            ]
                        }
                    }
                },
            }
        )
        return service


def render_compose(profile: ServeProfile) -> str:
    """Render canonical YAML; identical profiles always produce identical bytes."""
    return yaml.safe_dump(profile.compose(), sort_keys=False, width=1000)


def write_compose(profile: ServeProfile, output_directory: Path) -> Path:
    """Idempotently write the canonical filename, replacing only changed content."""
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / profile.compose_filename
    rendered = render_compose(profile)
    if not destination.exists() or destination.read_text(encoding="utf-8") != rendered:
        destination.write_text(rendered, encoding="utf-8")
    return destination
