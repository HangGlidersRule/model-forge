"""Static validation of the pinned ModelOpt image definition.

Docker is not available in CI, so the Dockerfile is parsed and asserted here.
Two regressions are covered:

- the interpreter must be installable from the base image's own repositories and
  must own a working pip (Ubuntu 22.04 has no ``python3.12``, and the distro
  ``python3-pip`` bootstraps pip for 3.10 only);
- ``nvidia-modelopt`` must come from the SHA-256-verified local wheel, with the
  ``[hf]`` extra requested on that file rather than a second index install that
  can re-resolve the pinned package.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO / "containers" / "modelopt" / "Dockerfile"

_INSTRUCTIONS = {
    "ADD", "ARG", "CMD", "COPY", "ENTRYPOINT", "ENV", "EXPOSE", "FROM",
    "HEALTHCHECK", "LABEL", "ONBUILD", "RUN", "SHELL", "STOPSIGNAL", "USER",
    "VOLUME", "WORKDIR",
}
# python3.NN in default repos: jammy stops at 3.10, noble ships 3.12.
_BASES_WITH_PY312 = ("ubuntu24.04", "ubuntu24.10", "ubuntu25.04")


def _text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def _logical_lines() -> list[str]:
    """Join backslash continuations into one string per Dockerfile instruction."""
    lines: list[str] = []
    buffer = ""
    for raw in _text().splitlines():
        stripped = raw.strip()
        if not buffer and (not stripped or stripped.startswith("#")):
            continue
        buffer += stripped[:-1] + " " if stripped.endswith("\\") else stripped
        if not stripped.endswith("\\"):
            lines.append(buffer)
            buffer = ""
    if buffer:
        lines.append(buffer)
    return lines


def _instruction(line: str) -> str:
    return line.split(maxsplit=1)[0].upper()


def _pip_install_commands() -> list[str]:
    """Every ``;``-separated shell command in a RUN that installs with pip."""
    commands: list[str] = []
    for line in _logical_lines():
        if _instruction(line) != "RUN":
            continue
        commands.extend(part for part in line.split(";") if "pip install" in part)
    return commands


def _declared_args() -> set[str]:
    names: set[str] = set()
    for line in _logical_lines():
        if _instruction(line) == "ARG":
            names.add(line.split(maxsplit=1)[1].split("=", 1)[0].strip())
    return names


def test_every_instruction_is_valid_and_parses() -> None:
    lines = _logical_lines()
    assert lines, "Dockerfile parsed to no instructions"
    for line in lines:
        assert _instruction(line) in _INSTRUCTIONS, line
    assert [_instruction(line) for line in lines].count("FROM") == 1
    assert not _text().endswith("\\\n"), "Dockerfile ends on a dangling continuation"


def test_every_referenced_build_arg_is_declared() -> None:
    referenced = set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)", _text()))
    undeclared = referenced - _declared_args() - _env_names()
    assert not undeclared, f"undeclared build args referenced: {sorted(undeclared)}"


def _env_names() -> set[str]:
    names: set[str] = set()
    for line in _logical_lines():
        if _instruction(line) != "ENV":
            continue
        for pair in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=", line.split(maxsplit=1)[1]):
            names.add(pair)
    return names


def test_base_image_ships_the_pinned_python() -> None:
    """The python3.12 apt install must be satisfiable by the base image itself."""
    base = next(line for line in _logical_lines() if _instruction(line) == "ARG" and "CUDA_IMAGE" in line)
    assert any(tag in base for tag in _BASES_WITH_PY312), base
    assert "ubuntu22.04" not in _text(), "Jammy has no python3.12 in its default repositories"
    # No third-party interpreter source is needed on such a base.
    lowered = _text().lower()
    assert "deadsnakes" not in lowered
    assert "add-apt-repository" not in lowered
    assert "ppa:" not in lowered


def test_pip_belongs_to_the_pinned_interpreter_via_venv() -> None:
    text = _text()
    # The distro python3-pip bootstraps pip for the *system* interpreter only.
    assert "python3-pip" not in text
    # update-alternatives would switch `python` without moving pip with it.
    assert "update-alternatives" not in text
    assert "-m venv" in text
    expected_path = (
        "ENV PATH=${PUBLIC_WORKSPACE}/venv/bin:${PATH}"
        if "${PUBLIC_WORKSPACE}" in text
        else "ENV PATH=/opt/venv/bin:${PATH}"
    )
    assert expected_path in text
    # Fail-closed proof that `python` on PATH is the venv's pinned interpreter.
    assert 'test "$(command -v python)" = "${VIRTUAL_ENV}/bin/python"' in text
    assert "sys.version_info[:2]" in text


def test_modelopt_installs_only_the_verified_local_wheel_with_hf_extra() -> None:
    text = _text()
    assert "sha256sum -c -" in text
    # The [hf] extra is requested on the verified file, not by package name.
    wheel_path = (
        "${PUBLIC_ARTIFACT_PATH}"
        if "${PUBLIC_ARTIFACT_PATH}" in text
        else "/tmp/wheel/${MODELOPT_WHEEL_FILENAME}"
    )
    assert f'pip install "{wheel_path}[hf]"' in text
    for command in _pip_install_commands():
        # Never install the pinned package by name: that can re-resolve from PyPI
        # (the verified local file is nvidia_modelopt.whl, with an underscore).
        assert "nvidia-modelopt" not in command, command


def test_wheel_provenance_is_asserted_after_install() -> None:
    """A version string alone cannot prove the pinned digest survived install."""
    text = _text()
    assert "direct_url.json" in text
    assert "url.startswith('file://')" in text
    assert "url.endswith('/${MODELOPT_WHEEL_FILENAME}')" in text
    # The cu128 torch pin must also survive the extra's dependency resolution.
    assert "md.version('torch')" in text


def test_downloaded_wheel_keeps_a_pep427_filename() -> None:
    """pip refuses a local wheel whose filename is not name-version-tags.whl."""
    downloads = re.findall(r"curl [^;]*-o \"?([^\s\";]+)", _text())
    assert downloads, "no wheel download found"
    for target in downloads:
        assert target in {
            "${PUBLIC_ARTIFACT_PATH}",
            "/tmp/wheel/${MODELOPT_WHEEL_FILENAME}",
        }, target
    assert _pin()["wheel"]["filename"].count("-") >= 4


def _pin() -> dict[str, Any]:
    text = (REPO / "configs" / "modelopt" / "pin.json").read_text(encoding="utf-8")
    return dict(json.loads(text))


def test_wheel_pin_matches_pin_json() -> None:
    pin = _pin()
    text = _text()
    assert f"ARG MODELOPT_WHEEL_SHA256={pin['wheel']['sha256']}" in text
    assert f"ARG MODELOPT_WHEEL_URL={pin['wheel']['url']}" in text
    assert f"ARG MODELOPT_WHEEL_FILENAME={pin['wheel']['filename']}" in text
    assert f"ARG MODELOPT_VERSION={pin['version']}" in text
    assert f"ARG MODELOPT_COMMIT={pin['git_commit']}" in text


def test_build_script_passes_the_pinned_wheel_filename() -> None:
    build = (REPO / "containers" / "modelopt" / "build.sh").read_text(encoding="utf-8")
    assert 'read_pin wheel.filename' in build
    assert '--build-arg "MODELOPT_WHEEL_FILENAME=${MODELOPT_WHEEL_FILENAME}"' in build
