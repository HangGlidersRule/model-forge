"""Diffusion serve profile (the #38 gate): a separate profile class for image
families, sharing the artifact-identity validators with the text ServeProfile
but with none of the text-specific machinery (no mtp_depth, no speculative
config, no text KV matrix, no qwen38-only families).

Serving contract for image families (as validated live on the fleet):
- vLLM-Omni is the default server (forge doctrine); command shape:
  ``vllm serve <path> --omni --port N``
- SGLang diffusion is the alternate server; command shape:
  ``sglang generate ...`` / ``sglang serve ...``
- Every diffusion container on the fleet requires the env pins discovered in
  the R7/#37 work: ``NCCL_CUMEM_ENABLE=0`` and (on SM120 NVFP4)
  ``SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=cutlass``.
- The consuming checkpoint's config must carry the normalized ignore patterns
  (component-prefixed matching) and ``quant_type`` — enforced here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Shared validator import (does not import the text profile class itself).
from model_forge.pipeline import sha256_file  # noqa: F401 — re-exported


class DiffusionServeProfileError(ValueError):
    pass


# Image families with validated serving contracts.
DIFFUSION_FAMILIES: dict[str, dict[str, Any]] = {
    "qwenimage21": {
        "pipeline": "QwenImage21Pipeline",
        "diT_class": "QwenImage21Transformer2DModel",
        "servers": ("vllm-omni", "sglang-diffusion"),
        "container_formats": ("bf16", "nvfp4", "fp8"),
        "image": "vllm/vllm-omni:qwen-image21",
    },
}

_ENV_PINS_ALL: tuple[tuple[str, str], ...] = (("NCCL_CUMEM_ENABLE", "0"),)
_ENV_PINS_NVFP4_SM120: tuple[tuple[str, str], ...] = (
    ("SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND", "cutlass"),
)

_CONTAINER = re.compile(r"^vllm-darkstar-[a-z0-9]+-(base|abliterated)-(bf16|nvfp4|fp8)$")


@dataclass
class DiffusionServeProfile:
    """Runtime identity for a diffusion family cell."""

    family: str
    behavior: str  # "base" (ablit rejected for qwen-image-2.1; kept for future families)
    format: str  # "bf16" | "nvfp4" | "fp8"
    repository_id: str
    model_path: str
    container_name: str
    server: str = "vllm-omni"  # "vllm-omni" | "sglang-diffusion"
    port: int = 8000
    image: str | None = None
    # deterministic serving
    max_num_seqs: int = 1  # seed determinism under batching (F18)
    # provenance
    artifact_identity: str | None = None
    artifact_precision_class: str | None = None
    artifact_success_sha256: str | None = None
    # environment pins required by the fleet (R7/#37 findings)
    extra_env: dict[str, str] = field(default_factory=dict)

    @property
    def alias(self) -> str:
        return f"darkstar-{self.family}-{self.behavior}-{self.format}"

    @property
    def compose_filename(self) -> str:
        return f"{self.alias}.yml"

    def validate(self) -> None:
        if self.family not in DIFFUSION_FAMILIES:
            raise DiffusionServeProfileError(
                f"unsupported diffusion family {self.family!r}; add an explicit family contract"
            )
        contract = DIFFUSION_FAMILIES[self.family]
        if self.server not in contract["servers"]:
            raise DiffusionServeProfileError(
                f"server {self.server!r} not validated for family {self.family!r}"
            )
        if self.format not in contract["container_formats"]:
            raise DiffusionServeProfileError(
                f"format {self.format!r} not in family contract {contract['container_formats']}"
            )
        if self.behavior != "base":
            raise DiffusionServeProfileError(
                "only behavior='base' is valid for image families with rejected ablit cells"
            )
        if not _CONTAINER.fullmatch(self.container_name):
            raise DiffusionServeProfileError(
                f"invalid diffusion container name: {self.container_name!r}"
            )
        expected_container = f"vllm-darkstar-{self.family}-{self.behavior}-{self.format}"
        if self.container_name != expected_container:
            raise DiffusionServeProfileError(
                f"container name {self.container_name!r} does not match profile; expected {expected_container!r}"
            )
        if not self.repository_id.startswith("HangGlidersRule/Darkstar-"):
            raise DiffusionServeProfileError(
                "repository_id must remain a precision-encoded Darkstar repository"
            )
        if self.format == "nvfp4" and "NVFP4" not in self.repository_id:
            raise DiffusionServeProfileError(
                "nvfp4 cell repository_id must encode NVFP4"
            )
        if self.max_num_seqs != 1:
            raise DiffusionServeProfileError(
                "seed determinism requires max_num_seqs=1 (or batching mode recorded in the frozen config)"
            )
        # env pins are mandatory
        env = self.resolved_env()
        for key, value in _ENV_PINS_ALL:
            if env.get(key) != value:
                raise DiffusionServeProfileError(f"missing required env pin {key}={value}")
        if self.format == "nvfp4":
            for key, value in _ENV_PINS_NVFP4_SM120:
                if env.get(key) != value:
                    raise DiffusionServeProfileError(
                        f"missing required NVFP4 SM120 env pin {key}={value}"
                    )

    def resolved_env(self) -> dict[str, str]:
        env: dict[str, str] = dict(_ENV_PINS_ALL)
        if self.format == "nvfp4":
            env.update(_ENV_PINS_NVFP4_SM120)
        env.update(self.extra_env)
        return env

    def serve_command(self) -> list[str]:
        """The server command for the pinned runtime."""
        self.validate()
        if self.server == "vllm-omni":
            return [
                "vllm",
                "serve",
                self.model_path,
                "--omni",
                "--port",
                str(self.port),
            ]
        if self.server == "sglang-diffusion":
            return [
                "sglang",
                "serve",
                "--model-path",
                self.model_path,
                "--port",
                str(self.port),
            ]
        raise DiffusionServeProfileError(f"unknown server {self.server!r}")

    def validate_checkpoint_config(self, checkpoint_dir: str | Path) -> None:
        """The consuming checkpoint must carry the normalized ignore patterns + quant_type."""
        p = Path(checkpoint_dir) / "transformer" / "config.json"
        cfg = json.loads(p.read_text())
        qc = cfg.get("quantization_config", {})
        if self.format != "bf16":
            if not qc.get("quant_type"):
                raise DiffusionServeProfileError(
                    f"{p}: quantization_config.quant_type missing (diffusers-style consumers require it)"
                )
            ignore = qc.get("ignore", [])
            bad = [x for x in ignore if not x.startswith("*")]
            if bad:
                raise DiffusionServeProfileError(
                    f"{p}: ignore patterns not component-prefixed (fnmatch mismatch — the #37 root cause): {bad[:3]}"
                )

    def compose(self) -> dict[str, object]:
        self.validate()
        cmd = self.serve_command()
        env = self.resolved_env()
        return {
            "services": {
                self.container_name: {
                    "image": self.image or DIFFUSION_FAMILIES[self.family]["image"],
                    "restart": "unless-stopped",
                    "ipc": "host",
                    "shm_size": "32g",
                    "ports": [f"{self.port}:{self.port}"],
                    "environment": [f"{k}={v}" for k, v in sorted(env.items())],
                    "volumes": ["/data/hf-models:/root/.cache/huggingface"],
                    "command": cmd,
                }
            }
        }


def render_compose(profile: DiffusionServeProfile) -> str:
    import yaml

    return yaml.safe_dump(profile.compose(), sort_keys=False)
