"""Bash ANSI-C quotes must not hide a vLLM model source from the publication gate.

``extract_model_sources`` tokenizes serve commands with POSIX ``shlex``. That lexer
treats ``$'...'`` as a literal dollar glued to an ordinary single-quoted word, so a
``bash -c`` payload (or a ``$'vllm'`` word) never contains a ``vllm`` / ``serve``
token pair and the extractor returns no source and raises nothing. Bash, running
the same text, expands the quote and serves the hidden model under a registered
``--served-model-name``.

These tests fail while that gap is present. They pass when the gate either raises
``PublicationSourceParseError`` or returns the model the shell actually serves.
Ordinary quotes and an ANSI-C payload whose ``--model`` flag survives re-tokenization
stay positive controls.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Literal, NamedTuple, Never

import pytest
import yaml

from model_forge.release import (
    _MARKDOWN_FENCE,
    _SERVE_MODEL_ALIAS,
    PublicationSourceParseError,
    extract_model_sources,
    extract_repository_ids,
    extract_serve_model_ids,
    model_source_precision_errors,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Bare NVFP4 outside the owned namespace. The precision rule rejects it when it
# is extracted, and the repository-id scan does not see it in raw text.
ATTACKER = "attacker-org/Darkstar-Evil-NVFP4"
ALIAS = "darkstar-qwen38-base-bf16"
TRUSTED = "HangGlidersRule/Darkstar-Qwen3.8-27B-Base-BF16"

Runner = Literal["script", "markdown", "yaml"]


class _Case(NamedTuple):
    name: str
    surface: str
    source_format: str | None
    runner: Runner
    hidden: bool


def _script(body: str) -> str:
    return f"#!/bin/bash\n{body}\n"


def _markdown(language: str, body: str) -> str:
    return f"```{language}\n{body}\n```\n"


def _hidden_bash_c() -> str:
    return f"bash -c $'vllm serve {ATTACKER} --served-model-name {ALIAS}'"


def _yaml_service(*, command: str, entrypoint: str | None = None) -> str:
    lines = ["services:", "  vllm:"]
    if entrypoint is not None:
        lines.append(f"    entrypoint: {entrypoint}")
    lines.append(f"    command: {command}")
    return "\n".join(lines) + "\n"


def _cases() -> list[_Case]:
    hidden = _hidden_bash_c()
    dollar_word = f"$'vllm' serve {ATTACKER} --served-model-name {ALIAS}"
    dollar_each = f"$'vllm' $'serve' $'{ATTACKER}' --served-model-name {ALIAS}"
    with_options = f"bash -O extglob -c $'vllm serve {ATTACKER} --served-model-name {ALIAS}'"
    via_sh = f"sh -c $'vllm serve {ATTACKER} --served-model-name {ALIAS}'"
    hex_model = (
        "bash -c $'vllm\\x20serve\\x20--model\\x20"
        f"{ATTACKER} --served-model-name {ALIAS}'"
    )
    mixed = _script(
        f"vllm serve {TRUSTED} --served-model-name {ALIAS}\n{hidden}"
    )
    plain = f"bash -c 'vllm serve {ATTACKER} --served-model-name {ALIAS}'"
    spaced_flag = f"bash -c $'vllm serve --model {ATTACKER} --served-model-name {ALIAS}'"
    return [
        _Case("bash_c_script", _script(hidden), ".sh", "script", True),
        _Case("bash_c_py_suffix", _script(hidden), ".py", "script", True),
        _Case("bash_c_untyped", _script(hidden), None, "script", True),
        _Case("sh_c_inside_bash", _script(via_sh), ".sh", "script", True),
        _Case("bash_option_before_c", _script(with_options), ".sh", "script", True),
        _Case("dollar_quoted_vllm_word", _script(dollar_word), ".sh", "script", True),
        _Case("dollar_quoted_each_word", _script(dollar_each), ".sh", "script", True),
        _Case("ansi_c_hex_hides_model_flag", _script(hex_model), ".sh", "script", True),
        _Case("decoy_line_plus_hidden_serve", mixed, ".sh", "script", True),
        _Case("markdown_bash_fence", _markdown("bash", hidden), ".md", "markdown", True),
        _Case("markdown_shell_fence", _markdown("shell", hidden), ".md", "markdown", True),
        _Case("markdown_unlabeled_fence", _markdown("", hidden), ".md", "markdown", True),
        _Case(
            "markdown_decoy_fence_plus_hidden_fence",
            _markdown("bash", f"vllm serve {TRUSTED} --served-model-name {ALIAS}")
            + _markdown("bash", hidden),
            ".md",
            "markdown",
            True,
        ),
        _Case(
            "yaml_command_string",
            _yaml_service(command=hidden),
            ".yaml",
            "yaml",
            True,
        ),
        _Case(
            "yaml_command_string_autodetected",
            _yaml_service(command=hidden),
            None,
            "yaml",
            True,
        ),
        _Case(
            "yaml_entrypoint_plus_ansi_c_command",
            _yaml_service(
                entrypoint="bash -O extglob -c",
                command=f"$'vllm serve {ATTACKER} --served-model-name {ALIAS}'",
            ),
            ".yml",
            "yaml",
            True,
        ),
        _Case("ordinary_single_quotes", _script(plain), ".sh", "script", False),
        _Case("ansi_c_with_visible_model_flag", _script(spaced_flag), ".sh", "script", False),
    ]


def _known_aliases() -> set[str]:
    aliases: set[str] = set()
    ledger_root = REPO_ROOT / "models"
    for path in ledger_root.glob("*/results/publication-readiness-ledger.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        for product in document.get("products", []):
            alias = product.get("served_model_alias")
            if isinstance(alias, str):
                aliases.add(alias)
    return aliases


def _bash_invocations(tmp_path: Path, script: str) -> list[list[str]]:
    """Run ``script`` with bash and record every ``vllm`` argv the script starts."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    argv_out = tmp_path / "argv"
    argv_out.write_bytes(b"")
    stub = bin_dir / "vllm"
    stub.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"$@\" >> \"$VLLM_ARGV_OUT\"\n"
        "printf '\\0' >> \"$VLLM_ARGV_OUT\"\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    script_path = tmp_path / "surface.sh"
    script_path.write_text(script, encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["VLLM_ARGV_OUT"] = str(argv_out)
    completed = subprocess.run(
        ["bash", str(script_path)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    invocations: list[list[str]] = []
    for chunk in argv_out.read_bytes().split(b"\0"):
        if chunk:
            invocations.append(chunk.decode().splitlines())
    return invocations


def _executed_scripts(surface: str, runner: Runner) -> list[str]:
    if runner == "script":
        return [surface]
    if runner == "markdown":
        bodies = [match.group("body") for match in _MARKDOWN_FENCE.finditer(surface)]
        assert bodies, surface
        return bodies
    if runner == "yaml":
        document = yaml.safe_load(surface)
        service = document["services"]["vllm"]
        command = service["command"]
        assert isinstance(command, str)
        entrypoint = service.get("entrypoint")
        if isinstance(entrypoint, str):
            return [f"{entrypoint} {command}\n"]
        return [command if command.endswith("\n") else f"{command}\n"]
    unexpected: Never = runner
    raise AssertionError(unexpected)


def _assert_gate_sees_served_model(
    surface: str, source_format: str | None, served_model: str, *, allow_refusal: bool
) -> None:
    try:
        sources = extract_model_sources(surface, source_format=source_format)
    except PublicationSourceParseError:
        assert allow_refusal
        return
    assert served_model in sources, sources
    try:
        aliases = extract_serve_model_ids(surface, source_format=source_format)
    except PublicationSourceParseError:
        assert allow_refusal
        return
    assert ALIAS in aliases


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case.name)
def test_bash_ansi_c_serve_is_visible_to_the_model_source_gate(
    case: _Case, tmp_path: Path
) -> None:
    """The model bash serves is the model the publication gate must report."""
    assert model_source_precision_errors(ATTACKER)
    assert ATTACKER not in extract_repository_ids(case.surface)
    assert ALIAS in _known_aliases()
    assert _SERVE_MODEL_ALIAS.fullmatch(ALIAS)

    served: list[list[str]] = []
    for script in _executed_scripts(case.surface, case.runner):
        served.extend(_bash_invocations(tmp_path, script))
    assert any(ATTACKER in args for args in served), served

    _assert_gate_sees_served_model(
        case.surface,
        case.source_format,
        ATTACKER,
        allow_refusal=case.hidden,
    )
