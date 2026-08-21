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

        if self.kv_cache_dtype != "bf16":
            raise ServeProfileError("this frozen profile requires BF16 KV cache")
        if self.attention_backend != "FLASH_ATTN":
            raise ServeProfileError("this frozen profile requires FLASH_ATTN")
        if self.mtp_depth < 1 or self.scheduler_tokens < 1:
            raise ServeProfileError("MTP depth and scheduler budget must be positive")

    def compose(self) -> dict[str, object]:
        self.validate()
        command = [
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
            f'{{"method":"mtp","num_speculative_tokens":{self.mtp_depth}}}',
        ]
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
        return {
            "name": self.alias,
            "services": {
                "vllm": service,
            },
        }


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
