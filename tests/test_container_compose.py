"""Compose template integrity: no advertised knob may be silently ignored."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTAINERS = REPO_ROOT / "containers"
SERVE_COMPOSE = CONTAINERS / "serve" / "docker-compose.yml"
BUILD_COMPOSE = CONTAINERS / "build" / "docker-compose.yml"

_INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)")


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _service(path: Path, name: str) -> dict[str, Any]:
    service = _load(path)["services"][name]
    assert isinstance(service, dict)
    return service


def _command_text(service: dict[str, Any]) -> str:
    command = service["command"]
    if isinstance(command, list):
        return " ".join(str(item) for item in command)
    assert isinstance(command, str)
    return command


def _environment_text(service: dict[str, Any]) -> str:
    environment = service["environment"]
    if isinstance(environment, list):
        return "\n".join(str(item) for item in environment)
    assert isinstance(environment, dict)
    return "\n".join(f"{key}={value}" for key, value in environment.items())


def _referenced_variables(value: object) -> set[str]:
    if isinstance(value, str):
        return set(_INTERPOLATION.findall(value))
    if isinstance(value, list):
        return {name for item in value for name in _referenced_variables(item)}
    return set()


@pytest.mark.parametrize("path", [SERVE_COMPOSE, BUILD_COMPOSE], ids=["serve", "build"])
def test_compose_files_parse(path: Path) -> None:
    assert _load(path)["services"]


def test_serve_environment_advertises_only_variables_the_command_uses() -> None:
    """vLLM is configured through CLI flags, so an env-only knob would be inert.

    The build runner is exempt: its entrypoint script reads its environment directly.
    """
    service = _service(SERVE_COMPOSE, "vllm")
    advertised = _referenced_variables(service["environment"])
    used = _referenced_variables(service["command"])
    assert not advertised - used, (
        "serve compose advertises environment variables the command ignores: "
        + ", ".join(sorted(advertised - used))
    )


def test_serve_enables_chunked_prefill_unconditionally() -> None:
    service = _service(SERVE_COMPOSE, "vllm")
    assert "--enable-chunked-prefill" in service["command"]
    assert "VLLM_ENABLE_CHUNKED_PREFILL" not in SERVE_COMPOSE.read_text(encoding="utf-8")


def test_containers_readme_documents_unconditional_chunked_prefill() -> None:
    readme = (CONTAINERS / "README.md").read_text(encoding="utf-8").lower()
    assert "chunked prefill" in readme
    assert "vllm_enable_chunked_prefill" not in readme


def test_build_launcher_is_configurable_via_build_script_env() -> None:
    """The build runner selects its launcher from BUILD_SCRIPT, defaulting to the Qwen script."""
    service = _service(BUILD_COMPOSE, "build-runner")
    assert (
        "BUILD_SCRIPT=${BUILD_SCRIPT:-scripts/qwen3_8/run_qwen38_build_mcprue.sh}"
        in _environment_text(service)
    )
    command = _command_text(service)
    # The container shell (not Compose) expands the env var: '$$' escapes to a literal '$'.
    # The path is quoted so a launcher path with spaces is passed as a single argument.
    assert '"/repo/$$BUILD_SCRIPT"' in command
    # The launcher path is no longer hardcoded into the command.
    assert "run_qwen38_build_mcprue.sh" not in command


def test_build_launcher_default_is_overridable_without_editing_compose() -> None:
    """A non-Qwen family can point BUILD_SCRIPT at its own repo-relative launcher."""
    text = BUILD_COMPOSE.read_text(encoding="utf-8")
    # The ${BUILD_SCRIPT:-...} default means an env override selects a different launcher.
    assert "${BUILD_SCRIPT:-" in text
    # The launch line itself is family-neutral; only the default names Qwen.
    launch_line = next(line for line in text.splitlines() if "$$BUILD_SCRIPT" in line)
    assert "qwen" not in launch_line.lower()


def test_build_config_is_required_and_never_defaults_to_historical_r3() -> None:
    text = BUILD_COMPOSE.read_text(encoding="utf-8")
    assert "CONFIG=${CONFIG:?" in text
    assert "CONFIG=${CONFIG:-" not in text
    assert "r3-nvfp4.yaml" not in text


def test_containers_readme_documents_configurable_build_script() -> None:
    readme = (CONTAINERS / "README.md").read_text(encoding="utf-8")
    lower = readme.lower()
    assert "build_script" in lower
    assert "scripts/qwen3_8/run_qwen38_build_mcprue.sh" in readme
    assert "CONFIG" in readme
    assert "required" in lower


def test_containers_readme_does_not_overclaim_architecture_agnostic() -> None:
    """The default launcher is Qwen, so the docs must not claim family/architecture agnosticism."""
    readme = (CONTAINERS / "README.md").read_text(encoding="utf-8").lower()
    assert "architecture-agnostic" not in readme
    assert "generic shell with a configurable launcher" in readme
    assert "defaults to the qwen3.8 pipeline" in readme


def test_serve_disables_speculative_decoding_by_default() -> None:
    """Generic serving must not enable MTP speculative decoding by default."""
    service = _service(SERVE_COMPOSE, "vllm")
    command = _command_text(service)
    assert "--speculative-config" not in command
    assert "mtp" not in command.lower()
    assert "MTP_DEPTH" not in SERVE_COMPOSE.read_text(encoding="utf-8")


def test_serve_exposes_vllm_extra_args_defaulting_empty() -> None:
    """Optional flags flow through a single VLLM_EXTRA_ARGS fragment defaulting to empty."""
    service = _service(SERVE_COMPOSE, "vllm")
    assert "${VLLM_EXTRA_ARGS:-}" in _command_text(service)


def test_containers_readme_documents_qwen_mtp_via_extra_args() -> None:
    readme = (CONTAINERS / "README.md").read_text(encoding="utf-8")
    lower = readme.lower()
    assert "vllm_extra_args" in lower
    assert "--speculative-config" in readme
    assert "mtp" in lower
